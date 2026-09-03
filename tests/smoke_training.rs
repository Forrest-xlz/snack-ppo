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
