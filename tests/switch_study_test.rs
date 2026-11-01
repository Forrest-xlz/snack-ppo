use ppo_snake::config::Config;

#[test]
fn switches_share_sources_and_change_only_target_scale() {
    let sources = [
        ("A", 100, 100_001_792, vec!["B", "C"]),
        ("A", 300, 300_001_280, vec!["B", "C"]),
        ("A", 550, 550_000_640, vec!["B", "C"]),
        ("B", 200, 200_007_680, vec!["C"]),
        ("B", 500, 500_006_912, vec!["C"]),
        ("B", 850, 850_001_920, vec!["C"]),
        ("C", 150, 150_077_440, vec!["B"]),
        ("C", 450, 450_068_480, vec!["B"]),
        ("C", 850, 850_001_920, vec!["B"]),
    ];
    let mut controls = 0;
    let mut switches = 0;
    for (source, nominal_millions, source_samples, targets) in sources {
        for target in targets {
            let name = format!("{source}-{nominal_millions}M-to-{target}");
            let config = Config::load(format!(
                "config/experiments/batch_rollout_switch/{name}.yaml"
            ))
            .unwrap();
            let mut baseline =
                Config::load(format!("config/experiments/batch_rollout/{target}.yaml")).unwrap();
            assert_eq!(config.training.pretrained_env_samples, source_samples);
            assert_eq!(
                config
                    .training
                    .init_checkpoint
                    .as_ref()
                    .unwrap()
                    .to_string_lossy(),
                format!(
                    "checkpoints/E001_batch_rollout_scale_exp-{source}/samples-{source_samples}.ot"
                )
            );
            assert_eq!(config.training.max_env_samples, 300_000_000);
            assert_eq!(
                config.wandb.group.as_deref(),
                Some("E003 batch rollout switch")
            );
            assert_eq!(
                config.wandb.run_name.as_deref(),
                Some(format!("E003 batch rollout switch {name}").as_str())
            );
            assert_eq!(
                config.inference.model_path.to_string_lossy(),
                format!("checkpoints/E003_batch_rollout_switch_{name}/latest.ot")
            );
            assert_eq!(
                config.environment.num_envs * config.environment.rollout_steps
                    / config.ppo.batch_size,
                8
            );
            assert_eq!(config.ppo.epochs, 1);
            baseline.training.max_env_samples = 300_000_000;
            let mut normalized = config.clone();
            normalized.training.init_checkpoint = None;
            normalized.training.pretrained_env_samples = 0;
            normalized.inference.model_path = baseline.inference.model_path.clone();
            normalized.wandb.group = baseline.wandb.group.clone();
            normalized.wandb.run_name = baseline.wandb.run_name.clone();
            assert_eq!(
                serde_json::to_value(normalized).unwrap(),
                serde_json::to_value(baseline).unwrap(),
                "unexpected change in {name}"
            );
            if source == target {
                controls += 1;
            } else {
                switches += 1;
            }
        }
    }
    assert_eq!(controls, 0);
    assert_eq!(
        std::fs::read_dir("config/experiments/batch_rollout_switch")
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "yaml"))
            .count(),
        12
    );
    assert_eq!(switches, 12);
}
