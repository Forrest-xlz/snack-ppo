use ppo_snake::{
    config::Config,
    evaluation::evaluate_policy,
    model::ActorCritic,
    ppo::PpoTrainer,
    rollout::{collect_rollout, VectorEnv},
};
use tch::{nn, no_grad, Device, Kind, Tensor};

fn config(version: &str) -> Config {
    let mut value = serde_json::to_value(Config::load("config/smoke.yaml").unwrap()).unwrap();
    value["model"]["version"] = serde_json::json!(version);
    serde_json::from_value(value).expect("compact entity version must be supported")
}

pub fn exercise(version: &str, device: Device) {
    let cfg = config(version);
    let store = nn::VarStore::new(device);
    let model = ActorCritic::from_config(&store.root(), &cfg);
    let mut envs =
        VectorEnv::with_features(&cfg.environment, 42, cfg.model.version.feature_encoding())
            .unwrap();
    let rollout = collect_rollout(&model, &mut envs, 4, device).unwrap();
    let mut trainer = PpoTrainer::new(&store, &cfg.ppo, 42).unwrap();
    let update = trainer.update(&model, &rollout, &[0.0; 2], device).unwrap();
    assert!(update.metrics.total_loss.is_finite());
    let variables = store.variables();
    assert_eq!(variables["cell_input.fc1.weight"].size()[0], 32);
    assert_eq!(variables["cell_input.fc2.weight"].size(), [16, 32]);
    assert_eq!(variables["encoder.qkv.weight"].size(), [48, 16]);
    assert_eq!(variables["encoder2.ff_in.weight"].size(), [32, 16]);
    assert!(!variables.contains_key("encoder3.qkv.weight"));
    for (name, weight) in &variables {
        let grad = weight.grad();
        assert!(grad.defined(), "{version}: {name}");
        assert_eq!(grad.isfinite().all().int64_value(&[]), 1, "{name}");
    }
    let inputs = Tensor::from_slice(&envs.observations())
        .view([2, envs.observation_size() as i64])
        .to_device(device);
    let before = no_grad(|| model.forward(&inputs));
    assert_eq!(before.0.size(), [2, 4]);
    assert_eq!(before.1.size(), [2]);
    let dir = tempfile::tempdir().unwrap();
    store.save(dir.path().join("compact.ot")).unwrap();
    let mut restored_store = nn::VarStore::new(device);
    let restored = ActorCritic::from_config(&restored_store.root(), &cfg);
    restored_store.load(dir.path().join("compact.ot")).unwrap();
    let after = no_grad(|| restored.forward(&inputs));
    assert!(before.0.equal(&after.0));
    assert!(before.1.equal(&after.1));
    if device.is_cuda() {
        let backend = model.attention_backend(device, 2, 25, true).unwrap();
        assert_eq!(backend.matches("layer").count(), 2);
        assert_eq!(backend.matches("queries=28,").count(), 1, "{backend}");
        assert_eq!(backend.matches("queries=2,").count(), 1, "{backend}");
    }
    let evaluation = evaluate_policy(&model, &cfg, device, 1000045).unwrap();
    assert!(evaluation.mean_score.is_finite());
    assert_eq!(variables["cell_input.fc1.weight"].kind(), Kind::Float);
}
