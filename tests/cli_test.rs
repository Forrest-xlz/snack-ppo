use std::process::Command;

#[test]
fn running_without_an_explicit_config_shows_usage() {
    let output = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .output()
        .expect("binary starts");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("usage: ppo-snake --config PATH"),
        "{stderr}"
    );
}

#[test]
fn checkpoint_directory_inspection_is_read_only_and_respects_configured_identity() {
    let directory = tempfile::tempdir().unwrap();
    let mut config: ppo_snake::config::Config =
        serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    config.training.checkpoint_dir = directory.path().join("versioned-output");
    config.wandb.run_name = Some("E004 feature architecture exp-B-v4".into());
    let path = directory.path().join("config.yaml");
    std::fs::write(&path, serde_yaml::to_string(&config).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--checkpoint-directory", "--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let reported = std::path::PathBuf::from(String::from_utf8(result.stdout).unwrap().trim());
    assert_eq!(
        reported,
        config
            .training
            .checkpoint_dir
            .join("E004_feature_architecture_exp-B-v4")
    );
    assert!(!config.training.checkpoint_dir.exists());
}
