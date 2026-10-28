use ppo_snake::config::Config;

#[test]
fn group_c_config_has_sample_based_limits() {
    let config = Config::load("config/experiments/batch_rollout/C.yaml").unwrap();
    assert_eq!(config.environment.num_envs, 1280);
    assert_eq!(config.environment.rollout_steps, 64);
    assert_eq!(config.ppo.batch_size, 10_240);
    assert_eq!(config.training.max_env_samples, 800_000_000);
    assert_eq!(config.training.checkpoint_interval_samples, 50_000_000);
    assert_eq!(config.evaluation.interval_samples, 50_000_000);
    assert_eq!(config.metrics.score_window_episodes, 1024);
    assert!(config.early_stopping.enabled);
    assert_eq!(config.early_stopping.window_samples, 30_000_000);
    assert_eq!(config.early_stopping.patience_windows, 3);
}

#[test]
fn six_configs_hold_baseline_fixed_and_change_only_the_study_axes() {
    let baseline = Config::load("config/experiments/batch_rollout/C.yaml").unwrap();
    let cases = [
        ("A", 32, 256, 2048),
        ("B", 128, 1024, 8192),
        ("C", 1280, 10_240, 81_920),
        ("D", 128, 256, 8192),
        ("E", 1280, 1024, 81_920),
        ("F", 320, 10_240, 20_480),
    ];
    for (name, envs, batch, rollout) in cases {
        let path = format!("config/experiments/batch_rollout/{name}.yaml");
        let config = Config::load(&path).unwrap();
        assert_eq!(config.environment.num_envs, envs, "{name}");
        assert_eq!(
            config.environment.num_envs * config.environment.rollout_steps,
            rollout,
            "{name}"
        );
        assert_eq!(config.ppo.batch_size, batch, "{name}");
        assert_eq!(config.early_stopping.window_samples, 30_000_000, "{name}");
        assert_eq!(
            config.wandb.group.as_deref(),
            Some("E001 batch rollout scale"),
            "{name}"
        );
        assert_eq!(
            config.wandb.run_name.as_deref(),
            Some(format!("E001 batch rollout scale exp-{name}").as_str()),
            "{name}"
        );
        assert_eq!(
            config.inference.model_path.to_string_lossy(),
            format!("checkpoints/E001_batch_rollout_scale_exp-{name}/best.ot"),
            "{name}"
        );
        assert_eq!(config.ppo.learning_rate, 0.0003, "{name}");
        assert_eq!(config.seed, 42, "{name}");
        assert_eq!(config.training.max_env_samples, 800_000_000, "{name}");
        assert_eq!(
            config.model.hidden_size, baseline.model.hidden_size,
            "{name}"
        );
        assert_eq!(config.ppo.epochs, baseline.ppo.epochs, "{name}");
        assert_eq!(config.ppo.gamma, baseline.ppo.gamma, "{name}");
        assert_eq!(config.ppo.gae_lambda, baseline.ppo.gae_lambda, "{name}");
        assert_eq!(config.ppo.clip_range, baseline.ppo.clip_range, "{name}");
        assert_eq!(config.ppo.entropy_coef, baseline.ppo.entropy_coef, "{name}");
        assert_eq!(
            config.evaluation.interval_samples, baseline.evaluation.interval_samples,
            "{name}"
        );
        assert_eq!(
            config.evaluation.episodes, baseline.evaluation.episodes,
            "{name}"
        );
        assert_eq!(
            config.metrics.score_window_episodes, baseline.metrics.score_window_episodes,
            "{name}"
        );
        assert_eq!(
            config.early_stopping.window_samples, baseline.early_stopping.window_samples,
            "{name}"
        );
        assert_eq!(
            config.early_stopping.patience_windows, baseline.early_stopping.patience_windows,
            "{name}"
        );
    }
}

#[test]
fn iteration_based_fields_are_rejected_and_early_stop_limits_validate() {
    let yaml = include_str!("../config/experiments/batch_rollout/C.yaml");
    assert!(serde_yaml::from_str::<Config>(
        &yaml.replace("max_env_samples: 800000000", "total_iterations: 8000")
    )
    .is_err());
    let invalid = yaml.replace("window_samples: 30000000", "window_samples: 0");
    let config: Config = serde_yaml::from_str(&invalid).unwrap();
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("early_stopping.window_samples"));
}
