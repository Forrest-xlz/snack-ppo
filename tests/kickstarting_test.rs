use std::{fs, process::Command};

use ppo_snake::{
    config::Config, kickstarting::TeacherPolicy, model::ActorCritic, ppo::PpoTrainer,
    rollout::RolloutBuffer,
};
use serde_json::json;
use tch::{nn, Cuda, Device, Kind, Tensor};

fn configured(source: &std::path::Path) -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["model"]["version"] = json!("mlp_v1");
    value["kickstarting"] = json!({
        "teacher_checkpoint": source,
        "teacher_model": {"version": "mlp_v1", "hidden_size": 16},
        "initial_weight": 1.0,
        "decay_samples": 200000000
    });
    serde_json::from_value(value).expect("optional kickstarting configuration must be supported")
}

#[test]
fn kickstarting_config_is_optional_and_rejects_invalid_settings() {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    assert!(serde_json::to_value(base)
        .unwrap()
        .get("kickstarting")
        .is_none());
    let valid = configured(std::path::Path::new("teacher.ot"));
    valid.validate().unwrap();
    for (field, invalid) in [
        ("initial_weight", json!(-1.0)),
        ("decay_samples", json!(0)),
        ("teacher_checkpoint", json!("")),
    ] {
        let mut value = serde_json::to_value(&valid).unwrap();
        value["kickstarting"][field] = invalid;
        assert!(
            serde_json::from_value::<Config>(value)
                .unwrap()
                .validate()
                .is_err(),
            "{field}"
        );
    }
}

#[test]
fn student_cli_uses_teacher_then_turns_auxiliary_loss_off() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("teacher.ot");
    let store = nn::VarStore::new(Device::Cpu);
    let _teacher = ActorCritic::new(&store.root(), 75, 16);
    store.save(&source).unwrap();
    let source_bytes = fs::read(&source).unwrap();
    let mut value = serde_json::to_value(configured(&source)).unwrap();
    value["training"]["checkpoint_dir"] = json!(directory.path().join("students"));
    value["training"]["max_env_samples"] = json!(16);
    value["kickstarting"]["decay_samples"] = json!(8);
    value["wandb"]["run_name"] = json!("kickstarting-smoke");
    let config: Config = serde_json::from_value(value).unwrap();
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
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("teacher_weight=1.000000"), "{stdout}");
    assert!(stdout.contains("teacher_weight=0.000000"), "{stdout}");
    assert!(stdout.contains("evaluation"));
    assert_eq!(fs::read(source).unwrap(), source_bytes);
    let mut student_store = nn::VarStore::new(Device::Cpu);
    let student = ActorCritic::from_config(&student_store.root(), &config);
    student_store
        .load(
            directory
                .path()
                .join("students/kickstarting-smoke/latest.ot"),
        )
        .unwrap();
    let observations = Tensor::zeros([1, 75], (Kind::Float, Device::Cpu));
    let (_, values) = tch::no_grad(|| student.forward(&observations));
    assert_eq!(values.isfinite().all().int64_value(&[]), 1);
}

#[test]
fn missing_teacher_fails_before_collecting_samples() {
    let directory = tempfile::tempdir().unwrap();
    let config = configured(&directory.path().join("missing.ot"));
    let path = directory.path().join("config.yaml");
    fs::write(&path, serde_yaml::to_string(&config).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("teacher"), "{stderr}");
    assert!(!String::from_utf8_lossy(&result.stdout).contains("total_samples="));
}

#[test]
fn incompatible_teacher_and_output_overlap_fail_without_touching_expert() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("same-run");
    fs::create_dir(&output).unwrap();
    let source = output.join("teacher.ot");
    let store = nn::VarStore::new(Device::Cpu);
    let _model = ActorCritic::new(&store.root(), 75, 16);
    store.save(&source).unwrap();
    let original = fs::read(&source).unwrap();
    let mut config = configured(&source);
    config.training.checkpoint_dir = directory.path().to_path_buf();
    let overlap = TeacherPolicy::load(&config, Device::Cpu, "same-run");
    assert!(overlap
        .err()
        .unwrap()
        .to_string()
        .contains("must not contain"));
    config
        .kickstarting
        .as_mut()
        .unwrap()
        .teacher_model
        .hidden_size = 32;
    let incompatible = TeacherPolicy::load(&config, Device::Cpu, "different-run");
    assert!(incompatible
        .err()
        .unwrap()
        .to_string()
        .contains("teacher weights"));
    assert_eq!(fs::read(&source).unwrap(), original);
}

#[test]
fn teacher_uses_only_its_original_feature_planes_and_is_frozen() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("teacher.ot");
    let store = nn::VarStore::new(Device::Cpu);
    let original = ActorCritic::new(&store.root(), 75, 16);
    store.save(&source).unwrap();
    let mut config = configured(&source);
    config.training.checkpoint_dir = directory.path().join("student");
    let teacher = TeacherPolicy::load(&config, Device::Cpu, "test")
        .unwrap()
        .unwrap();
    let input = Tensor::rand([3, 175], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let actual = teacher.probabilities(&input, 2);
    let (logits, _) = tch::no_grad(|| original.forward(&input.narrow(1, 0, 75)));
    assert!(actual.allclose(&logits.softmax(-1, Kind::Float), 1e-5, 1e-6, false));
    assert!(!actual.requires_grad());
    let alternate = Tensor::cat(
        &[
            input.narrow(1, 0, 75),
            Tensor::ones([3, 100], (Kind::Float, Device::Cpu)),
        ],
        1,
    );
    assert!(actual.equal(&teacher.probabilities(&alternate, 2)));
}

#[test]
fn zero_weight_skips_teacher_inference_and_matches_plain_ppo_exactly() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("teacher.ot");
    let teacher_store = nn::VarStore::new(Device::Cpu);
    let _original = ActorCritic::new(&teacher_store.root(), 75, 16);
    teacher_store.save(&source).unwrap();
    let mut config = configured(&source);
    config.training.checkpoint_dir = directory.path().join("students");
    let teacher = TeacherPolicy::load(&config, Device::Cpu, "zero-test")
        .unwrap()
        .unwrap();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::new(&store.root(), 3, 8);
    let mut reference_store = nn::VarStore::new(Device::Cpu);
    let reference = ActorCritic::new(&reference_store.root(), 3, 8);
    reference_store.copy(&store).unwrap();
    let mut optimizer = PpoTrainer::new(&store, &config.ppo, 42).unwrap();
    let mut reference_optimizer = PpoTrainer::new(&reference_store, &config.ppo, 42).unwrap();
    let rollout = RolloutBuffer {
        // Intentionally too narrow for the teacher: any unwanted teacher
        // inference at zero weight would fail rather than silently cost time.
        observations: vec![0.0; 12],
        actions: vec![0, 1, 2, 3],
        rewards: vec![0.0, 1.0, 0.0, 0.0],
        dones: vec![false, false, false, true],
        values: vec![0.0; 4],
        log_probs: vec![-1.3862944; 4],
        episode_scores: vec![1],
        reverse_action_count: 0,
        steps: 2,
        num_envs: 2,
        observation_size: 3,
    };
    let update = optimizer
        .update_with_kickstarting(
            &model,
            &rollout,
            &[0.0, 0.0],
            Device::Cpu,
            Some((&teacher, 0.0)),
        )
        .unwrap();
    let expected = reference_optimizer
        .update(&reference, &rollout, &[0.0, 0.0], Device::Cpu)
        .unwrap();
    assert_eq!(update.metrics.total_loss, expected.metrics.total_loss);
    assert_eq!(update.metrics.kickstarting_loss, 0.0);
    for (name, weight) in store.variables() {
        assert!(weight.equal(&reference_store.variables()[&name]), "{name}");
    }
}

#[test]
fn cuda_student_has_finite_outputs_gradients_and_reports_its_actual_backend() {
    if let Err(error) = ppo_snake::trainer::resolve_device(ppo_snake::config::DeviceConfig::Cuda) {
        assert!(
            std::env::var_os("PPO_SNAKE_REQUIRE_CUDA_TESTS").is_none(),
            "CUDA validation required but unavailable: {error:#}"
        );
        eprintln!("Skipping optional CUDA check: {error:#}");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let mut config = configured(&directory.path().join("unused.ot"));
    config.model.version = ppo_snake::config::ModelVersion::TransformerV1Small;
    config.kickstarting = None;
    let store = nn::VarStore::new(Device::Cuda(0));
    let model = ActorCritic::from_config(&store.root(), &config);
    let backend = model
        .attention_backend(Device::Cuda(0), 2, 25, true)
        .unwrap();
    assert!(backend.contains("BFloat16"), "{backend}");
    let observations = Tensor::rand([2, 1703], (Kind::Float, Device::Cuda(0)));
    let output = model.evaluate_actions(
        &observations,
        &Tensor::zeros([2], (Kind::Int64, Device::Cuda(0))),
    );
    let teacher_probs = Tensor::full([2, 4], 0.25, (Kind::Float, Device::Cuda(0)));
    let ce = ppo_snake::kickstarting::policy_cross_entropy(&teacher_probs, &output.log_probs_all)
        .mean(Kind::Float);
    let loss = ce + output.values.square().mean(Kind::Float);
    assert!(loss.double_value(&[]).is_finite());
    loss.backward();
    for (name, weight) in store.variables() {
        let grad = weight.grad();
        assert!(grad.defined(), "{name}");
        assert_eq!(grad.isfinite().all().int64_value(&[]), 1, "{name}");
    }
    Cuda::synchronize(0);
}
