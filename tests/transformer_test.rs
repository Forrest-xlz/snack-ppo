use std::{fs, process::Command};

use ppo_snake::{config::Config, env::SnakeEnv, model::ActorCritic};
use serde_json::json;
use tch::{nn, Device, IndexOp, Kind, Tensor};

fn config() -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["model"]["version"] = json!("transformer_v1");
    value["environment"]["width"] = json!(7);
    serde_json::from_value(value).expect("transformer_v1 must be supported")
}

#[test]
fn transformer_uses_head_and_separate_critic_readouts() {
    let config = config();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &config);
    // Disable cross-token mixing: changing only the head selection must change
    // actor output, while the separately selected critic token stays identical.
    tch::no_grad(|| {
        for (name, mut tensor) in store.variables() {
            if name.contains("qkv") || name.contains("ff_out") || name == "token_embedding.weight" {
                let _ = tensor.zero_();
            }
        }
    });
    let observations = Tensor::zeros([2, 105], (Kind::Float, Device::Cpu));
    let _ = observations.i((0, 0)).fill_(1.0);
    let _ = observations.i((1, 34)).fill_(1.0);
    let (logits, values) = tch::no_grad(|| model.forward(&observations));
    assert_eq!(logits.size(), [2, 4]);
    assert_eq!(values.size(), [2]);
    assert!(!logits.get(0).allclose(&logits.get(1), 1e-5, 1e-6, false));
    assert!(
        values.get(0).allclose(&values.get(1), 1e-5, 1e-6, false),
        "critic batch rows differ: {} vs {}",
        values.double_value(&[0]),
        values.double_value(&[1])
    );
    for index in 0..2 {
        let (single_logits, single_value) =
            tch::no_grad(|| model.forward(&observations.narrow(0, index, 1)));
        assert!(single_logits
            .squeeze_dim(0)
            .allclose(&logits.get(index), 1e-5, 1e-6, false));
        assert!(single_value
            .squeeze_dim(0)
            .allclose(&values.get(index), 1e-5, 1e-6, false));
    }
}

#[test]
fn transformer_backpropagates_globally_and_reloads_exactly() {
    let config = config();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &config);
    let env = SnakeEnv::new(7, 5, 42).unwrap();
    let observations = Tensor::from_slice(&env.observation())
        .view([1, 105])
        .set_requires_grad(true);
    let (logits, values) = model.forward(&observations);
    assert_eq!(logits.size(), [1, 4]);
    assert_eq!(values.size(), [1]);
    assert_eq!(logits.isfinite().all().int64_value(&[]), 1);
    assert_eq!(values.isfinite().all().int64_value(&[]), 1);
    (logits.square().sum(Kind::Float) + values.square().sum(Kind::Float)).backward();
    assert!(observations.grad().i((0, 104)).abs().double_value(&[]) > 0.0);
    for name in [
        "critic_token",
        "positions",
        "encoder.qkv.weight",
        "policy.weight",
        "value.weight",
    ] {
        let grad = store.variables()[name].grad();
        assert_eq!(grad.isfinite().all().int64_value(&[]), 1, "{name}");
        assert!(
            grad.abs().sum(Kind::Float).double_value(&[]) > 0.0,
            "{name}"
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("transformer.ot");
    store.save(&checkpoint).unwrap();
    let mut restored_store = nn::VarStore::new(Device::Cpu);
    let restored = ActorCritic::from_config(&restored_store.root(), &config);
    restored_store.load(checkpoint).unwrap();
    let (restored_logits, restored_values) = tch::no_grad(|| restored.forward(&observations));
    assert!(restored_logits.equal(&logits));
    assert!(restored_values.equal(&values));
}

#[test]
fn transformer_cli_trains_evaluates_and_saves_on_rectangular_board() {
    let mut config = config();
    let directory = tempfile::tempdir().unwrap();
    config.training.checkpoint_dir = directory.path().to_owned();
    config.wandb.run_name = Some("transformer-smoke".into());
    let path = directory.path().join("config.yaml");
    fs::write(&path, serde_yaml::to_string(&config).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("eval"));
    let weights = Tensor::load_multi(directory.path().join("transformer-smoke/latest.ot")).unwrap();
    assert!(weights
        .iter()
        .any(|(name, tensor)| name == "positions" && tensor.size() == [1, 35, 32]));
    assert!(weights
        .iter()
        .any(|(name, tensor)| name == "critic_token" && tensor.size() == [1, 1, 32]));
}

#[test]
fn experiment_b_keeps_experiment_a_training_protocol() {
    let a = Config::load("config/experiments/feature_architecture/A.yaml").unwrap();
    let b = Config::load("config/experiments/feature_architecture/B.yaml").unwrap();
    assert_eq!(b.device, ppo_snake::config::DeviceConfig::Cuda);
    assert!(b.kickstarting.is_none());
    assert!(b.training.init_checkpoint.is_none());
    assert_eq!(b.training.pretrained_env_samples, 0);
    let store = nn::VarStore::new(Device::Cpu);
    let _model = ActorCritic::from_config(&store.root(), &b);
    assert_eq!(
        serde_json::to_value(&b).unwrap()["model"]["version"],
        "transformer_v1_small"
    );
    assert_eq!(b.model.version.feature_encoding().channels(), 66);
    assert_eq!(store.variables()["cell_input.fc1.weight"].size(), [32, 66]);
    assert_eq!(store.variables()["cell_input.fc2.weight"].size(), [16, 32]);
    assert!(store.variables().contains_key("encoder2.qkv.weight"));
    assert!(!store.variables().contains_key("encoder3.qkv.weight"));
    assert!(!store.variables().contains_key("encoder4.qkv.weight"));
    assert_eq!(
        b.wandb.run_name.as_deref(),
        Some("E004 feature architecture exp-B")
    );
    assert_eq!(
        b.inference.model_path.to_string_lossy(),
        "checkpoints/E004_feature_architecture_exp-B/latest.ot"
    );
    let expected = serde_json::to_value(a).unwrap();
    let mut actual = serde_json::to_value(b).unwrap();
    actual["device"] = expected["device"].clone();
    actual["model"]["version"] = expected["model"]["version"].clone();
    actual["wandb"]["run_name"] = expected["wandb"]["run_name"].clone();
    actual["inference"]["model_path"] = expected["inference"]["model_path"].clone();
    assert_eq!(actual, expected);
}
