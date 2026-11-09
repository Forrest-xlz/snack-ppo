use ppo_snake::{config::Config, env::SnakeEnv, model::ActorCritic};
use serde_json::json;
use std::{fs, process::Command};
use tch::{nn, Device, Kind, Tensor};

#[test]
fn two_layer_cli_updates_both_blocks_and_reloads() {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["model"]["version"] = json!("transformer_v3");
    value["environment"]["width"] = json!(7);
    value["training"]["checkpoint_dir"] = json!(directory.path());
    value["wandb"]["run_name"] = json!("two-layer-smoke");
    let config: Config =
        serde_json::from_value(value).expect("two-layer version must be supported");
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
    assert!(String::from_utf8_lossy(&result.stdout).contains("evaluation"));
    let mut store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &config);
    let checkpoint = directory.path().join("two-layer-smoke/latest.ot");
    store.load(&checkpoint).unwrap();
    let env = SnakeEnv::new(7, 5, 42).unwrap();
    let observations =
        Tensor::from_slice(&config.model.version.feature_encoding().encode(&env)).view([1, 245]);
    let (logits, values) = model.forward(&observations);
    (logits.square().sum(Kind::Float) + values.square().sum(Kind::Float)).backward();
    for name in [
        "encoder.qkv.weight",
        "encoder2.qkv.weight",
        "encoder.ff_out.weight",
        "encoder2.ff_out.weight",
    ] {
        let grad = store.variables()[name].grad();
        assert_eq!(grad.isfinite().all().int64_value(&[]), 1, "{name}");
        assert!(
            grad.abs().sum(Kind::Float).double_value(&[]) > 0.0,
            "{name}"
        );
    }
    let mut restored_store = nn::VarStore::new(Device::Cpu);
    let restored = ActorCritic::from_config(&restored_store.root(), &config);
    restored_store.load(checkpoint).unwrap();
    let (restored_logits, restored_values) = tch::no_grad(|| restored.forward(&observations));
    assert!(logits.equal(&restored_logits));
    assert!(values.equal(&restored_values));
}
