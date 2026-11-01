use std::{fs, path::Path, process::Command};

use ppo_snake::{config::Config, model::ActorCritic};
use serde_json::json;
use tch::{nn, Device};

fn branch_config(source: &Path, output: &Path, pretrained: usize) -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["training"]["init_checkpoint"] = json!(source);
    value["training"]["pretrained_env_samples"] = json!(pretrained);
    value["training"]["checkpoint_dir"] = json!(output);
    value["training"]["max_env_samples"] = json!(6);
    value["ppo"]["learning_rate"] = json!(1e-30);
    value["wandb"]["run_name"] = json!("branch");
    serde_json::from_value(value).expect("weights-only initialization must parse")
}

fn save_source(path: &Path) {
    let store = nn::VarStore::new(Device::Cpu);
    let _model = ActorCritic::new(&store.root(), 75, 16);
    tch::no_grad(|| {
        for (_, mut tensor) in store.variables() {
            let _ = tensor.fill_(0.125);
        }
    });
    store.save(path).unwrap();
}

fn run_cli(config: &Config, path: &Path) -> std::process::Output {
    fs::write(path, serde_yaml::to_string(config).unwrap()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn weights_only_branch_loads_source_and_counts_only_new_samples_toward_budget() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("baseline.ot");
    save_source(&source);
    let original = fs::read(&source).unwrap();
    let config = branch_config(&source, &directory.path().join("outputs"), 123_400_000);
    config.validate().unwrap();
    let result = run_cli(&config, &directory.path().join("branch.yaml"));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(
        stdout.contains("total_samples=123400000 finetune_samples=0"),
        "{stdout}"
    );
    assert!(stdout.find("initial_evaluation").unwrap() < stdout.find("iteration=1").unwrap());
    assert!(
        stdout.contains("total_samples=123400008 finetune_samples=8"),
        "{stdout}"
    );
    let final_path = directory.path().join("outputs/branch/samples-123400008.ot");
    assert!(final_path.is_file());
    assert!(!directory
        .path()
        .join("outputs/branch/samples-8.ot")
        .exists());
    assert_eq!(fs::read(&source).unwrap(), original);
    // The tiny learning rate leaves nonzero f32 weights unchanged. A random
    // initialization, or loading only the policy head, cannot pass this check.
    let source_tensors = tch::Tensor::load_multi(&source).unwrap();
    let final_tensors = tch::Tensor::load_multi(final_path).unwrap();
    assert_eq!(source_tensors.len(), final_tensors.len());
    for (name, expected) in source_tensors {
        let actual = &final_tensors
            .iter()
            .find(|(key, _)| key == &name)
            .unwrap()
            .1;
        assert!(actual.equal(&expected), "loaded weights differ for {name}");
    }
}

#[test]
fn branch_rejects_output_directory_containing_the_source_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let source_dir = directory.path().join("branch");
    fs::create_dir(&source_dir).unwrap();
    let source = source_dir.join("baseline.ot");
    save_source(&source);
    let original = fs::read(&source).unwrap();
    let config = branch_config(&source, directory.path(), 100);
    let result = run_cli(&config, &directory.path().join("branch.yaml"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("initialization checkpoint"));
    assert_eq!(fs::read(source).unwrap(), original);
    assert!(!source_dir.join("latest.ot").exists());
}

#[test]
fn missing_source_fails_before_training() {
    let directory = tempfile::tempdir().unwrap();
    let config = branch_config(&directory.path().join("missing.ot"), directory.path(), 100);
    let result = run_cli(&config, &directory.path().join("branch.yaml"));
    assert!(!result.status.success());
    assert!(!String::from_utf8_lossy(&result.stdout).contains("iteration=1"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("initialization checkpoint"));
}

#[test]
fn pretrained_samples_require_a_source_and_cannot_overflow_the_cumulative_budget() {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["training"]["pretrained_env_samples"] = json!(100);
    let config: Config = serde_json::from_value(value.clone()).unwrap();
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("training.init_checkpoint"));
    value["training"]["init_checkpoint"] = json!("source.ot");
    value["training"]["pretrained_env_samples"] = json!(usize::MAX);
    let config: Config = serde_json::from_value(value).unwrap();
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("training.pretrained_env_samples"));
}
