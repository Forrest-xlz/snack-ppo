use std::{fs, path::Path, process::Command};

use ppo_snake::config::Config;
use ppo_snake::{env::SnakeEnv, model::ActorCritic};
use serde_json::json;
use tch::{nn, Device, Kind, Tensor};

fn cnn_config(output: &Path) -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["features"] = json!({"version": "board_v1"});
    value["model"]["version"] = json!("cnn_v1");
    value["environment"]["width"] = json!(7);
    value["training"]["checkpoint_dir"] = json!(output);
    value["wandb"]["run_name"] = json!("cnn-smoke");
    serde_json::from_value(value).expect("CNN and feature versions must parse")
}

#[test]
fn old_configs_default_to_original_features_and_mlp_and_reject_unknown_versions() {
    let config: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(config).unwrap();
    assert_eq!(value["features"]["version"], "board_v1");
    assert_eq!(value["model"]["version"], "mlp_v1");
    value["model"]["version"] = json!("unknown_model");
    assert!(serde_json::from_value::<Config>(value.clone()).is_err());
    value["model"]["version"] = json!("cnn_v1");
    value["features"]["version"] = json!("unknown_features");
    assert!(serde_json::from_value::<Config>(value).is_err());
}

#[test]
fn cnn_cli_trains_on_a_rectangular_board_and_writes_convolution_weights() {
    let directory = tempfile::tempdir().unwrap();
    let config = cnn_config(directory.path());
    config.validate().unwrap();
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
    let weights = tch::Tensor::load_multi(directory.path().join("cnn-smoke/latest.ot")).unwrap();
    let shape = |name: &str| {
        weights
            .iter()
            .find(|(key, _)| key == name)
            .unwrap()
            .1
            .size()
    };
    assert_eq!(shape("conv1.weight"), [16, 3, 3, 3]);
    assert_eq!(shape("conv2.weight"), [32, 16, 3, 3]);
    assert_eq!(shape("conv3.weight"), [64, 32, 3, 3]);
    assert_eq!(shape("projection.weight"), [16, 64 * 2 * 2]);
    assert_eq!(shape("policy.weight"), [4, 16]);
    assert_eq!(shape("value.weight"), [1, 16]);
}

#[test]
fn experiment_a_changes_only_architecture_and_run_identity_from_large_baseline() {
    let config = Config::load("config/experiments/feature_architecture/A.yaml").unwrap();
    let baseline = Config::load("config/experiments/batch_rollout/C.yaml").unwrap();
    let mut actual = serde_json::to_value(&config).unwrap();
    let expected = serde_json::to_value(&baseline).unwrap();
    assert_eq!(actual["features"]["version"], "board_v1");
    assert_eq!(actual["model"]["version"], "cnn_v1");
    assert_eq!(
        config.wandb.group.as_deref(),
        Some("E004 feature architecture")
    );
    assert_eq!(
        config.wandb.run_name.as_deref(),
        Some("E004 feature architecture exp-A")
    );
    assert_eq!(
        config.inference.model_path.to_string_lossy(),
        "checkpoints/E004_feature_architecture_exp-A/latest.ot"
    );
    actual["model"]["version"] = expected["model"]["version"].clone();
    actual["wandb"] = expected["wandb"].clone();
    actual["inference"] = expected["inference"].clone();
    assert_eq!(actual, expected);
}

#[test]
fn board_v1_keeps_the_original_head_body_food_channel_order() {
    let directory = tempfile::tempdir().unwrap();
    let config = cnn_config(directory.path());
    let env = SnakeEnv::new(7, 5, 42).unwrap();
    let encoded = config.features.version.encode(&env);
    let mut expected = vec![0.0; 105];
    expected[17] = 1.0;
    expected[35 + 16] = 1.0;
    expected[35 + 15] = 1.0;
    let food = env.food().unwrap();
    expected[70 + food.y * 7 + food.x] = 1.0;
    assert_eq!(encoded, expected);
}

#[test]
fn configured_cnn_checkpoint_round_trip_preserves_logits_and_values() {
    let directory = tempfile::tempdir().unwrap();
    let config = cnn_config(directory.path());
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &config);
    let observations = Tensor::rand([3, 105], (Kind::Float, Device::Cpu));
    let (logits, values) = tch::no_grad(|| model.forward(&observations));
    assert_eq!(logits.size(), [3, 4]);
    assert_eq!(values.size(), [3]);
    assert_eq!(logits.isfinite().all().int64_value(&[]), 1);
    assert_eq!(values.isfinite().all().int64_value(&[]), 1);
    let path = directory.path().join("cnn.ot");
    store.save(&path).unwrap();
    let mut restored_store = nn::VarStore::new(Device::Cpu);
    let restored = ActorCritic::from_config(&restored_store.root(), &config);
    restored_store.load(&path).unwrap();
    let (restored_logits, restored_values) = tch::no_grad(|| restored.forward(&observations));
    assert!(restored_logits.equal(&logits));
    assert!(restored_values.equal(&values));
}
