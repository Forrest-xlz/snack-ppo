use ppo_snake::{config::Config, trainer::run_training};

#[test]
fn one_iteration_cpu_training_writes_checkpoint() {
    let yaml = include_str!("../config/smoke.yaml");
    let config: Config = serde_yaml::from_str(yaml).unwrap();
    config.validate().unwrap();
    run_training(&config).unwrap();
    assert!(config
        .training
        .checkpoint_dir
        .join("smoke")
        .join("latest.ot")
        .is_file());
}

#[test]
fn sample_cap_stops_after_the_rollout_that_crosses_it() {
    let yaml = include_str!("../config/smoke.yaml");
    let mut config: Config = serde_yaml::from_str(yaml).unwrap();
    let directory = tempfile::tempdir().unwrap();
    config.training.checkpoint_dir = directory.path().to_path_buf();
    config.training.max_env_samples = 6;
    config.validate().unwrap();
    run_training(&config).unwrap();
    assert!(directory
        .path()
        .join("smoke")
        .join("samples-8.ot")
        .is_file());
    assert!(!directory.path().join("smoke").join("best.ot").exists());
}
