use ppo_snake::config::Config;

#[test]
fn sample_intervals_are_required_and_validate_positive_values() {
    let config: Config = serde_yaml::from_str(VALID).unwrap();
    assert_eq!(config.evaluation.interval_samples, 32_768_000);
    assert_eq!(config.metrics.log_interval_samples, 500_000);
    let invalid = VALID.replace("interval_samples: 32768000", "interval_samples: 0");
    let invalid: Config = serde_yaml::from_str(&invalid).unwrap();
    assert!(invalid.validate().is_err());
}

const VALID: &str = r#"
mode: train
device: cpu
seed: 42
environment:
  width: 12
  height: 12
  num_envs: 64
  rollout_steps: 256
model:
  hidden_size: 256
ppo:
  learning_rate: 0.0003
  batch_size: 1024
  epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
training:
  max_env_samples: 800000000
  checkpoint_interval_samples: 50000000
  checkpoint_dir: checkpoints
inference:
  model_path: checkpoints/latest.pt
  fps: 10
  deterministic: true
  cell_size: 36
evaluation:
  interval_samples: 32768000
  episodes: 20
  deterministic: true
  max_steps_per_episode: 5000
metrics:
  score_window_episodes: 1000
  log_interval_samples: 500000
early_stopping:
  enabled: true
  window_samples: 50000000
  patience_windows: 3
  min_episodes: 256
wandb:
  enabled: true
  project: ppo-snake
  entity: null
  group: batch-size-study
  run_name: null
  mode: offline
  python_executable: python
"#;

fn parse(yaml: &str) -> Config {
    serde_yaml::from_str(yaml).expect("valid test yaml")
}

#[test]
fn complete_yaml_deserializes_and_validates() {
    let config = parse(VALID);
    config.validate().expect("configuration should validate");
    assert_eq!(config.environment.num_envs, 64);
    assert_eq!(config.environment.rollout_steps, 256);
    assert_eq!(config.ppo.batch_size, 1024);
    assert_eq!(config.ppo.epochs, 4);
    assert_eq!(config.metrics.score_window_episodes, 1000);
    assert_eq!(config.metrics.log_interval_samples, 500000);
    assert_eq!(config.wandb.group.as_deref(), Some("batch-size-study"));
    assert!((config.ppo.learning_rate - 0.0003).abs() < f64::EPSILON);
}

#[test]
fn zero_metric_window_is_rejected() {
    let config = parse(&VALID.replace("log_interval_samples: 500000", "log_interval_samples: 0"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("metrics.log_interval_samples"), "{error}");
}

#[test]
fn zero_environment_count_is_rejected() {
    let config = parse(&VALID.replace("num_envs: 64", "num_envs: 0"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("environment.num_envs"), "{error}");
}

#[test]
fn zero_rollout_is_rejected() {
    let config = parse(&VALID.replace("rollout_steps: 256", "rollout_steps: 0"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("environment.rollout_steps"), "{error}");
}

#[test]
fn non_positive_learning_rate_is_rejected() {
    let config = parse(&VALID.replace("learning_rate: 0.0003", "learning_rate: 0.0"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("ppo.learning_rate"), "{error}");
}

#[test]
fn invalid_gamma_is_rejected() {
    let config = parse(&VALID.replace("gamma: 0.99", "gamma: 1.1"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("ppo.gamma"), "{error}");
}

#[test]
fn board_too_small_for_initial_snake_is_rejected() {
    let config = parse(&VALID.replace("width: 12", "width: 2"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("environment.width"), "{error}");
}

#[test]
fn zero_evaluation_interval_is_rejected() {
    let config = parse(&VALID.replace("interval_samples: 32768000", "interval_samples: 0"));
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("evaluation.interval_samples"), "{error}");
}
