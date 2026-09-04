use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub fn sanitize_run_name(run_name: &str) -> String {
    let mut sanitized = String::with_capacity(run_name.len());
    let mut previous_separator = false;
    for character in run_name.chars() {
        let valid = character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.');
        let replacement = if valid { character } else { '_' };
        if replacement == '_' && previous_separator {
            continue;
        }
        sanitized.push(replacement);
        previous_separator = replacement == '_';
    }
    let sanitized = sanitized.trim_matches(['.', '_', '-']);
    if sanitized.is_empty() {
        "run".to_owned()
    } else {
        sanitized.to_owned()
    }
}

pub fn generated_run_name() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("run-{seconds}")
}

pub fn run_checkpoint_directory(base: &Path, run_name: &str) -> PathBuf {
    base.join(sanitize_run_name(run_name))
}

pub fn training_metric_groups(
    metrics: crate::ppo::PpoMetrics,
    average_score: f64,
    max_score: u32,
    learning_rate: f64,
    iteration: usize,
    total_samples: usize,
) -> BTreeMap<String, f64> {
    let values = [
        ("policy_loss", metrics.policy_loss),
        ("value_loss", metrics.value_loss),
        ("entropy", metrics.entropy),
        ("total_loss", metrics.total_loss),
        ("mean_episode_score", average_score),
        ("max_episode_score", max_score as f64),
        ("learning_rate", learning_rate),
        (
            "critic_explained_variance",
            metrics.critic_explained_variance,
        ),
        ("policy_approx_kl", metrics.policy_approx_kl),
    ];
    let mut grouped = BTreeMap::from([
        ("iteration".to_owned(), iteration as f64),
        ("total_samples".to_owned(), total_samples as f64),
    ]);
    for (name, value) in values {
        grouped.insert(format!("train_samples/{name}"), value);
        grouped.insert(format!("train_iteration/{name}"), value);
    }
    grouped
}

pub fn evaluation_metric_groups(
    metrics: crate::evaluation::EvaluationMetrics,
    iteration: usize,
    total_samples: usize,
) -> BTreeMap<String, f64> {
    let values = [
        ("mean_score", metrics.mean_score),
        ("min_score", metrics.min_score as f64),
        ("max_score", metrics.max_score as f64),
    ];
    let mut grouped = BTreeMap::from([
        ("iteration".to_owned(), iteration as f64),
        ("total_samples".to_owned(), total_samples as f64),
    ]);
    for (name, value) in values {
        grouped.insert(format!("eval_samples/{name}"), value);
        grouped.insert(format!("eval_iteration/{name}"), value);
    }
    grouped
}

pub fn recent_metric_groups(
    snapshot: &crate::metrics::MetricSnapshot,
    iteration: usize,
    total_samples: usize,
) -> BTreeMap<String, f64> {
    let mut values = Vec::new();
    if let Some(scores) = snapshot.scores {
        values.extend([
            ("mean_episode_score", scores.mean),
            ("min_episode_score", scores.min as f64),
            ("max_episode_score", scores.max as f64),
        ]);
    }
    if let Some(transitions) = snapshot.transitions {
        values.extend([
            ("policy_loss", transitions.policy_loss),
            ("value_loss", transitions.value_loss),
            ("entropy", transitions.entropy),
            ("total_loss", transitions.total_loss),
            ("policy_approx_kl", transitions.policy_approx_kl),
            (
                "critic_explained_variance",
                transitions.critic_explained_variance,
            ),
        ]);
    }
    let mut grouped = BTreeMap::from([
        ("iteration".to_owned(), iteration as f64),
        ("total_samples".to_owned(), total_samples as f64),
    ]);
    for (name, value) in values {
        grouped.insert(format!("recent_samples/{name}"), value);
        grouped.insert(format!("recent_iteration/{name}"), value);
    }
    grouped
}

use anyhow::{bail, Context, Result};
use tch::{nn, no_grad, Cuda, Device, Tensor};

use crate::{
    config::{Config, DeviceConfig},
    evaluation::evaluate_policy,
    metrics::MetricWindows,
    model::ActorCritic,
    ppo::PpoTrainer,
    rollout::{collect_rollout, VectorEnv},
    wandb::WandbLogger,
};

pub fn resolve_device(config: DeviceConfig) -> Result<Device> {
    match config {
        DeviceConfig::Cpu => Ok(Device::Cpu),
        DeviceConfig::Cuda if Cuda::is_available() => Ok(Device::Cuda(0)),
        DeviceConfig::Cuda => bail!(
            "CUDA was requested but is unavailable; install a CUDA-enabled LibTorch and check the NVIDIA runtime"
        ),
    }
}

pub fn save_checkpoint_atomic(
    store: &nn::VarStore,
    directory: &Path,
    iteration: usize,
) -> Result<PathBuf> {
    fs::create_dir_all(directory).with_context(|| {
        format!(
            "failed to create checkpoint directory {}",
            directory.display()
        )
    })?;
    let numbered = directory.join(format!("iteration-{iteration}.ot"));
    let numbered_temp = directory.join(format!(".iteration-{iteration}.tmp.ot"));
    store
        .save(&numbered_temp)
        .context("failed to write checkpoint temporary file")?;
    fs::rename(&numbered_temp, &numbered)
        .with_context(|| format!("failed to install checkpoint {}", numbered.display()))?;

    let latest = directory.join("latest.ot");
    let latest_temp = directory.join(".latest.tmp.ot");
    fs::copy(&numbered, &latest_temp).context("failed to stage latest checkpoint")?;
    if latest.exists() {
        fs::remove_file(&latest).context("failed to replace previous latest checkpoint")?;
    }
    fs::rename(&latest_temp, &latest).context("failed to install latest checkpoint")?;
    Ok(numbered)
}

pub fn load_checkpoint(store: &mut nn::VarStore, path: &Path) -> Result<()> {
    let extension = path.extension().and_then(|value| value.to_str());
    if matches!(extension, Some("pt" | "bin")) {
        let temporary = path.with_file_name(format!(".legacy-load-{}.ot", std::process::id()));
        fs::copy(path, &temporary)
            .with_context(|| format!("failed to stage legacy checkpoint {}", path.display()))?;
        let result = store
            .load(&temporary)
            .with_context(|| format!("failed to load legacy checkpoint {}", path.display()));
        let cleanup = fs::remove_file(&temporary).with_context(|| {
            format!(
                "failed to remove temporary checkpoint {}",
                temporary.display()
            )
        });
        result.and(cleanup)
    } else {
        store
            .load(path)
            .with_context(|| format!("failed to load checkpoint {}", path.display()))
    }
}

pub fn checkpoint_path_with_legacy_fallback(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    if path.extension().and_then(|value| value.to_str()) == Some("ot") {
        let legacy = path.with_extension("pt");
        if legacy.exists() {
            return legacy;
        }
    }
    path.to_path_buf()
}

pub fn resolve_inference_checkpoint(
    requested: &Path,
    checkpoint_base: &Path,
    run_name: Option<&str>,
) -> PathBuf {
    let requested = checkpoint_path_with_legacy_fallback(requested);
    if requested.exists() {
        return requested;
    }
    if let Some(run_name) = run_name {
        let run_latest = run_checkpoint_directory(checkpoint_base, run_name).join("latest.ot");
        let run_latest = checkpoint_path_with_legacy_fallback(&run_latest);
        if run_latest.exists() {
            return run_latest;
        }
    }
    requested
}

pub fn run_training(config: &Config) -> Result<()> {
    let device = resolve_device(config.device)?;
    tch::manual_seed(config.seed as i64);
    let store = nn::VarStore::new(device);
    let input_size = (3 * config.environment.width * config.environment.height) as i64;
    let model = ActorCritic::new(&store.root(), input_size, config.model.hidden_size);
    let mut optimizer = PpoTrainer::new(&store, &config.ppo, config.seed)?;
    let mut envs = VectorEnv::new(&config.environment, config.seed)?;
    let run_name = config
        .wandb
        .run_name
        .as_deref()
        .map(sanitize_run_name)
        .unwrap_or_else(generated_run_name);
    let checkpoint_directory = run_checkpoint_directory(&config.training.checkpoint_dir, &run_name);
    let mut wandb = WandbLogger::start(config, &run_name)?;
    let samples_per_iteration = config.environment.num_envs * config.environment.rollout_steps;
    let mut metric_windows = MetricWindows::new(
        config.metrics.score_window_episodes,
        config.metrics.transition_window_samples,
    );
    println!("device={device:?} samples_per_iteration={samples_per_iteration}");

    for iteration in 1..=config.training.total_iterations {
        let rollout = collect_rollout(&model, &mut envs, config.environment.rollout_steps, device)?;
        let final_observations = envs.observations();
        let final_tensor = Tensor::from_slice(&final_observations)
            .view([envs.len() as i64, envs.observation_size() as i64])
            .to_device(device);
        let (_, final_values) = no_grad(|| model.forward(&final_tensor));
        let final_values = Vec::<f32>::try_from(&final_values.to_device(Device::Cpu))
            .context("failed to copy bootstrap values to CPU")?;
        let update = optimizer.update(&model, &rollout, &final_values, device)?;
        let metrics = update.metrics;
        metric_windows.extend_scores(rollout.episode_scores.iter().copied());
        metric_windows.extend_transitions(update.transition_metrics);
        let average_score = if rollout.episode_scores.is_empty() {
            0.0
        } else {
            rollout
                .episode_scores
                .iter()
                .map(|&score| score as f64)
                .sum::<f64>()
                / rollout.episode_scores.len() as f64
        };
        let max_score = rollout.episode_scores.iter().copied().max().unwrap_or(0);
        println!(
            "iteration={iteration} total_samples={} batch_samples={} avg_score={average_score:.3} max_score={max_score} policy_loss={:.5} value_loss={:.5} entropy={:.5} explained_variance={:.5} approx_kl={:.6}",
            iteration * samples_per_iteration,
            rollout.len(),
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.critic_explained_variance,
            metrics.policy_approx_kl,
        );
        let total_samples = iteration * samples_per_iteration;
        let mut logged = training_metric_groups(
            metrics,
            average_score,
            max_score,
            config.ppo.learning_rate,
            iteration,
            total_samples,
        );
        logged.extend(recent_metric_groups(
            &metric_windows.snapshot(),
            iteration,
            total_samples,
        ));
        if iteration % config.evaluation.interval == 0 {
            let evaluation = evaluate_policy(
                &model,
                config,
                device,
                config.seed.wrapping_add(iteration as u64 * 1_000_003),
            )?;
            println!(
                "evaluation iteration={iteration} episodes={} mean_score={:.3} min_score={} max_score={}",
                config.evaluation.episodes,
                evaluation.mean_score,
                evaluation.min_score,
                evaluation.max_score,
            );
            logged.extend(evaluation_metric_groups(
                evaluation,
                iteration,
                total_samples,
            ));
        }
        wandb.log(&logged)?;
        if iteration % config.training.checkpoint_interval == 0
            || iteration == config.training.total_iterations
        {
            let path = save_checkpoint_atomic(&store, &checkpoint_directory, iteration)?;
            println!("checkpoint={}", path.display());
        }
    }
    wandb.finish()?;
    Ok(())
}
