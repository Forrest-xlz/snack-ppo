use ppo_snake::{
    config::Config,
    model::ActorCritic,
    teacher_kl::{policy_kl, EvalTeacher},
};
use serde_json::json;
use tch::{nn, no_grad, Device, Kind, Tensor};

fn configured(coef: f64) -> Config {
    let mut value = serde_json::to_value(Config::load("config/smoke.yaml").unwrap()).unwrap();
    value["teacher_kl"] = json!({"enabled": true, "coef": coef});
    serde_json::from_value(value).unwrap()
}

#[test]
fn teacher_config_rejects_invalid_coefficients_and_conflicting_objectives() {
    let cfg = configured(0.001);
    cfg.validate().unwrap();
    for coef in [-0.1, f64::NAN, f64::INFINITY] {
        let mut invalid = cfg.clone();
        invalid.teacher_kl.as_mut().unwrap().coef = coef;
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("teacher_kl.coef"));
    }
    let mut conflict = serde_json::to_value(&cfg).unwrap();
    conflict["kickstarting"] = json!({"teacher_checkpoint":"teacher.ot", "teacher_model":{"version":"mlp_v1","hidden_size":16}, "initial_weight":1.0,"decay_samples":100});
    let conflict: Config = serde_json::from_value(conflict).unwrap();
    assert!(conflict
        .validate()
        .unwrap_err()
        .to_string()
        .contains("teacher_kl"));
}

#[test]
fn forward_kl_has_correct_value_and_only_student_gradients() {
    let teacher = Tensor::from_slice(&[0.75f32, 0.25, 0.0, 0.0])
        .view([1, 4])
        .set_requires_grad(true);
    let logits = Tensor::from_slice(&[0.0f32; 4])
        .view([1, 4])
        .set_requires_grad(true);
    let kl = policy_kl(&teacher, &logits.log_softmax(-1, Kind::Float)).mean(Kind::Float);
    let expected = 0.75 * 3.0f64.ln();
    assert!((kl.double_value(&[]) - expected).abs() < 1e-6);
    kl.backward();
    assert!(!teacher.grad().defined());
    assert!(logits.grad().allclose(
        &Tensor::from_slice(&[-0.5f32, 0.0, 0.25, 0.25]).view([1, 4]),
        1e-6,
        1e-6,
        false
    ));
    let probs = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4]).view([1, 4]);
    assert!(
        policy_kl(&probs, &probs.log())
            .abs()
            .max()
            .double_value(&[])
            < 1e-7
    );
}

#[test]
fn teacher_is_an_independent_frozen_snapshot_replaced_only_at_new_eval_highs() {
    let cfg = configured(0.001);
    let mut teacher = EvalTeacher::new(&cfg, Device::Cpu).unwrap();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &cfg);
    let input = Tensor::rand([3, 75], (Kind::Float, Device::Cpu));
    assert!(teacher.policy().is_none());
    assert!(teacher.observe(0.0, 8, &store).unwrap());
    let first = teacher.policy().unwrap().probabilities(&input, 2);
    let dir = tempfile::tempdir().unwrap();
    teacher.save(dir.path()).unwrap();
    assert!(!first.requires_grad());
    assert!(first.allclose(
        &no_grad(|| model.forward(&input)).0.softmax(-1, Kind::Float),
        1e-6,
        1e-6,
        false
    ));
    no_grad(|| {
        for (name, mut weight) in store.variables() {
            if name.ends_with("policy.bias") {
                weight.copy_(&Tensor::from_slice(&[8.0f32, 0.0, 0.0, 0.0]));
            }
        }
    });
    let changed = no_grad(|| model.forward(&input)).0.softmax(-1, Kind::Float);
    assert!(!changed.allclose(&first, 1e-4, 1e-4, false));
    for score in [0.0, -1.0] {
        assert!(!teacher.observe(score, 16, &store).unwrap());
        assert!(first.equal(&teacher.policy().unwrap().probabilities(&input, 2)));
    }
    assert!(teacher.observe(f64::NAN, 16, &store).is_err());
    assert!(teacher.observe(1.0, 24, &store).unwrap());
    assert!(changed.allclose(
        &teacher.policy().unwrap().probabilities(&input, 2),
        1e-6,
        1e-6,
        false
    ));
    assert_eq!(teacher.best_mean_score(), Some(1.0));
    assert_eq!(teacher.source_samples(), Some(24));
    teacher.save(dir.path()).unwrap();
    let mut restored = nn::VarStore::new(Device::Cpu);
    let restored_model = ActorCritic::from_config(&restored.root(), &cfg);
    restored.load(dir.path().join("teacher.ot")).unwrap();
    assert!(changed.allclose(
        &no_grad(|| restored_model.forward(&input))
            .0
            .softmax(-1, Kind::Float),
        1e-6,
        1e-6,
        false
    ));
}

fn run_cli(cfg: &Config, dir: &std::path::Path, name: &str) -> std::process::Output {
    let path = dir.join(format!("{name}.yaml"));
    std::fs::write(&path, serde_yaml::to_string(cfg).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn zero_and_disabled_teacher_match_baseline_checkpoints_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = configured(0.0);
    cfg.training.checkpoint_dir = dir.path().join("checkpoints");
    cfg.training.max_env_samples = 24;
    cfg.wandb.run_name = Some("zero".into());
    run_cli(&cfg, dir.path(), "zero");
    cfg.teacher_kl.as_mut().unwrap().enabled = false;
    cfg.teacher_kl.as_mut().unwrap().coef = 0.001;
    cfg.wandb.run_name = Some("disabled".into());
    run_cli(&cfg, dir.path(), "disabled");
    cfg.teacher_kl = None;
    cfg.wandb.run_name = Some("plain".into());
    run_cli(&cfg, dir.path(), "plain");
    let mut reference = nn::VarStore::new(Device::Cpu);
    let _model = ActorCritic::from_config(&reference.root(), &cfg);
    reference
        .load(dir.path().join("checkpoints/plain/latest.ot"))
        .unwrap();
    for run in ["zero", "disabled"] {
        let mut actual = nn::VarStore::new(Device::Cpu);
        let _model = ActorCritic::from_config(&actual.root(), &cfg);
        actual
            .load(dir.path().join(format!("checkpoints/{run}/latest.ot")))
            .unwrap();
        for (name, weight) in reference.variables() {
            assert!(weight.equal(&actual.variables()[&name]), "{run} {name}");
        }
        assert!(!dir
            .path()
            .join(format!("checkpoints/{run}/teacher.ot"))
            .exists());
    }
}

#[test]
fn cli_initializes_teacher_after_first_eval_and_preserves_early_stop() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = configured(0.001);
    cfg.model.version = ppo_snake::config::ModelVersion::TransformerV1Small;
    cfg.training.checkpoint_dir = dir.path().join("checkpoints");
    cfg.training.max_env_samples = 80;
    cfg.evaluation.max_steps_per_episode = 1; // All eval scores are zero: first establishes teacher, ties keep it.
    cfg.early_stopping.enabled = true;
    cfg.early_stopping.patience_evaluations = 2;
    let output = run_cli(&cfg, dir.path(), "active");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("teacher_kl_coef=0.000000"), "{stdout}");
    assert!(stdout.contains("teacher_kl_coef=0.001000"), "{stdout}");
    assert_eq!(
        stdout.matches("teacher_kl_updated=true").count(),
        1,
        "{stdout}"
    );
    assert_eq!(
        stdout.matches("teacher_kl_updated=false").count(),
        2,
        "{stdout}"
    );
    assert!(
        stdout.contains("training_end=early_stop total_samples=24"),
        "{stdout}"
    );
    assert!(dir.path().join("checkpoints/smoke/teacher.ot").is_file());
    // Allocating the teacher must not alter the student's initial training
    // trajectory. All scores tie, so teacher.ot is exactly the first eval.
    cfg.teacher_kl = None;
    cfg.training.max_env_samples = 8;
    cfg.early_stopping.enabled = false;
    cfg.wandb.run_name = Some("baseline-prefix".into());
    run_cli(&cfg, dir.path(), "baseline-prefix");
    let mut first_teacher = nn::VarStore::new(Device::Cpu);
    let _teacher_model = ActorCritic::from_config(&first_teacher.root(), &cfg);
    first_teacher
        .load(dir.path().join("checkpoints/smoke/teacher.ot"))
        .unwrap();
    let mut prefix = nn::VarStore::new(Device::Cpu);
    let _prefix_model = ActorCritic::from_config(&prefix.root(), &cfg);
    prefix
        .load(dir.path().join("checkpoints/baseline-prefix/latest.ot"))
        .unwrap();
    for (name, weight) in prefix.variables() {
        assert!(weight.equal(&first_teacher.variables()[&name]), "{name}");
    }
}

#[test]
fn ppo_adds_configured_forward_kl_and_keeps_teacher_frozen() {
    use ppo_snake::{ppo::PpoTrainer, rollout::RolloutBuffer};
    let cfg = configured(0.5);
    let mut teacher = EvalTeacher::new(&cfg, Device::Cpu).unwrap();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &cfg);
    teacher.observe(1.0, 0, &store).unwrap();
    no_grad(|| {
        let mut bias = store.variables().remove("policy.bias").unwrap();
        bias.copy_(&Tensor::from_slice(&[8.0f32, 0.0, 0.0, 0.0]));
    });
    let mut reference_store = nn::VarStore::new(Device::Cpu);
    let reference = ActorCritic::from_config(&reference_store.root(), &cfg);
    reference_store.copy(&store).unwrap();
    let inputs = Tensor::rand([2, 75], (Kind::Float, Device::Cpu));
    let probs = teacher.policy().unwrap().probabilities(&inputs, 1);
    let log_probs = no_grad(|| model.forward(&inputs))
        .0
        .log_softmax(-1, Kind::Float);
    let expected_kl = policy_kl(&probs, &log_probs)
        .mean(Kind::Float)
        .double_value(&[]);
    assert!(expected_kl > 0.1);
    let rollout = RolloutBuffer {
        observations: Vec::<f32>::try_from(inputs.view([-1])).unwrap(),
        actions: vec![0, 1],
        rewards: vec![0.0, 1.0],
        dones: vec![false, true],
        values: vec![0.0; 2],
        log_probs: vec![-1.3862944; 2],
        episode_scores: vec![1],
        reverse_action_count: 0,
        steps: 2,
        num_envs: 1,
        observation_size: 75,
    };
    let mut optimizer = PpoTrainer::new(&store, &cfg.ppo, 42).unwrap();
    let mut baseline = PpoTrainer::new(&reference_store, &cfg.ppo, 42).unwrap();
    let update = optimizer
        .update_with_teacher_kl(
            &model,
            &rollout,
            &[0.0],
            Device::Cpu,
            Some((teacher.policy().unwrap(), 0.5)),
        )
        .unwrap();
    let plain = baseline
        .update(&reference, &rollout, &[0.0], Device::Cpu)
        .unwrap();
    assert!((update.metrics.teacher_kl_loss - expected_kl).abs() < 1e-6);
    assert!(
        (update.metrics.total_loss - plain.metrics.total_loss - 0.5 * expected_kl).abs() < 2e-6
    );
    assert!((update.metrics.teacher_kl_penalty - 0.5 * expected_kl).abs() < 1e-6);
    assert_eq!(update.metrics.kickstarting_loss, 0.0);
    assert!(probs.equal(&teacher.policy().unwrap().probabilities(&inputs, 1)));
    assert!(store
        .variables()
        .iter()
        .any(|(name, weight)| !weight.equal(&reference_store.variables()[name])));
}

#[test]
fn uploaded_penalty_is_sample_weighted_across_teacher_activation() {
    use ppo_snake::{metrics::SampleWindow, ppo::PpoMetrics, trainer::training_metric_groups};
    let mut window = SampleWindow::default();
    window.push(8, &[], 0, PpoMetrics::default(), 1);
    window.push(
        8,
        &[],
        0,
        PpoMetrics {
            teacher_kl_loss: 2.0,
            teacher_kl_coef: 0.001,
            teacher_kl_penalty: 0.002,
            ..Default::default()
        },
        1,
    );
    let logged = training_metric_groups(&window.snapshot(), None, 16, 2, 1.0, 16.0);
    assert!((logged["train_samples/teacher_kl_penalty"] - 0.001).abs() < 1e-12);
}

#[test]
fn experiment_d_adds_teacher_kl_without_changing_b_protocol() {
    let b = Config::load("config/experiments/feature_architecture/B.yaml").unwrap();
    let d = Config::load("config/experiments/feature_architecture/D.yaml").unwrap();
    assert!(d.teacher_kl.as_ref().unwrap().enabled);
    assert_eq!(d.teacher_kl.as_ref().unwrap().coef, 0.001);
    assert_eq!(
        d.wandb.run_name.as_deref(),
        Some("E004 feature architecture exp-D")
    );
    assert_eq!(
        d.inference.model_path.to_string_lossy(),
        "checkpoints/E004_feature_architecture_exp-D/latest.ot"
    );
    let expected = serde_json::to_value(b).unwrap();
    let mut actual = serde_json::to_value(d).unwrap();
    actual.as_object_mut().unwrap().remove("teacher_kl");
    actual["wandb"]["run_name"] = expected["wandb"]["run_name"].clone();
    actual["inference"]["model_path"] = expected["inference"]["model_path"].clone();
    assert_eq!(actual, expected);
}
