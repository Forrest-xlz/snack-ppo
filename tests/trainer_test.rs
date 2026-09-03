use ppo_snake::{
    config::DeviceConfig,
    model::ActorCritic,
    trainer::{
        checkpoint_path_with_legacy_fallback, evaluation_metric_groups, load_checkpoint,
        resolve_device, resolve_inference_checkpoint, run_checkpoint_directory, sanitize_run_name,
        save_checkpoint_atomic, training_metric_groups,
    },
};
use ppo_snake::{evaluation::EvaluationMetrics, ppo::PpoMetrics};
use tch::{nn, Device};

#[test]
fn cpu_device_always_resolves() {
    assert_eq!(resolve_device(DeviceConfig::Cpu).unwrap(), Device::Cpu);
}

#[test]
fn configured_run_name_creates_a_checkpoint_subdirectory() {
    let path = run_checkpoint_directory(std::path::Path::new("checkpoints"), "snake-lr3e4");
    assert_eq!(
        path,
        std::path::Path::new("checkpoints").join("snake-lr3e4")
    );
}

#[test]
fn run_name_cannot_escape_checkpoint_directory() {
    assert_eq!(sanitize_run_name("../team\\run:1"), "team_run_1");
}

#[test]
fn cuda_never_silently_falls_back_to_cpu() {
    match resolve_device(DeviceConfig::Cuda) {
        Ok(device) => assert_eq!(device, Device::Cuda(0)),
        Err(error) => assert!(error.to_string().contains("CUDA")),
    }
}

#[test]
fn checkpoint_saved_by_training_can_be_loaded_by_var_store() {
    let directory = tempfile::tempdir().unwrap();
    let source = nn::VarStore::new(Device::Cpu);
    let _source_model = ActorCritic::new(&source.root(), 75, 16);
    let checkpoint = save_checkpoint_atomic(&source, directory.path(), 1).unwrap();

    let mut destination = nn::VarStore::new(Device::Cpu);
    let _destination_model = ActorCritic::new(&destination.root(), 75, 16);
    destination.load(checkpoint).unwrap();
}

#[test]
fn legacy_pt_checkpoint_from_this_project_still_loads() {
    let directory = tempfile::tempdir().unwrap();
    let legacy = directory.path().join("latest.pt");
    let source = nn::VarStore::new(Device::Cpu);
    let _source_model = ActorCritic::new(&source.root(), 75, 16);
    source.save(&legacy).unwrap();

    let mut destination = nn::VarStore::new(Device::Cpu);
    let _destination_model = ActorCritic::new(&destination.root(), 75, 16);
    load_checkpoint(&mut destination, &legacy).unwrap();
}

#[test]
fn missing_new_checkpoint_path_falls_back_to_existing_legacy_pt() {
    let directory = tempfile::tempdir().unwrap();
    let requested = directory.path().join("latest.ot");
    let legacy = directory.path().join("latest.pt");
    std::fs::write(&legacy, b"legacy").unwrap();
    assert_eq!(checkpoint_path_with_legacy_fallback(&requested), legacy);
}

#[test]
fn training_metrics_are_duplicated_for_sample_and_iteration_axes() {
    let metrics = training_metric_groups(
        PpoMetrics {
            policy_loss: 1.0,
            value_loss: 2.0,
            entropy: 3.0,
            total_loss: 4.0,
            critic_explained_variance: 0.5,
            policy_approx_kl: 0.01,
        },
        1.5,
        4,
        0.0003,
        7,
        8192,
    );
    assert_eq!(metrics["train_samples/policy_loss"], 1.0);
    assert_eq!(metrics["train_iteration/policy_loss"], 1.0);
    assert_eq!(metrics["train_samples/critic_explained_variance"], 0.5);
    assert_eq!(metrics["train_iteration/policy_approx_kl"], 0.01);
    assert_eq!(metrics["total_samples"], 8192.0);
    assert_eq!(metrics["iteration"], 7.0);
    assert!(!metrics.keys().any(|key| key.starts_with("train/")));
}

#[test]
fn evaluation_metrics_are_duplicated_for_sample_and_iteration_axes() {
    let metrics = evaluation_metric_groups(
        EvaluationMetrics {
            mean_score: 3.0,
            min_score: 1,
            max_score: 6,
        },
        7,
        8192,
    );
    assert_eq!(metrics["eval_samples/mean_score"], 3.0);
    assert_eq!(metrics["eval_iteration/max_score"], 6.0);
    assert!(!metrics.keys().any(|key| key.starts_with("evaluation/")));
}

#[test]
fn inference_finds_latest_checkpoint_inside_configured_run_folder() {
    let directory = tempfile::tempdir().unwrap();
    let run_directory = directory.path().join("snake-run");
    std::fs::create_dir(&run_directory).unwrap();
    let expected = run_directory.join("latest.ot");
    std::fs::write(&expected, b"checkpoint").unwrap();
    let requested = directory.path().join("latest.ot");
    assert_eq!(
        resolve_inference_checkpoint(&requested, directory.path(), Some("snake-run")),
        expected
    );
}
