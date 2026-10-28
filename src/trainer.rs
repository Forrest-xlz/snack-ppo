use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
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

pub fn advance_sample_threshold(next: usize, interval: usize, total_samples: usize) -> usize {
    if total_samples < next {
        return next;
    }
    let intervals_crossed = (total_samples - next) / interval + 1;
    next.saturating_add(intervals_crossed.saturating_mul(interval))
}

pub fn run_checkpoint_directory(base: &Path, run_name: &str) -> PathBuf {
    base.join(sanitize_run_name(run_name))
}

pub fn training_metric_groups(
    snapshot: &crate::metrics::SampleWindowSnapshot,
    scores: Option<crate::metrics::ScoreWindowMetrics>,
    total_samples: usize,
    optimizer_steps_total: usize,
    wall_time_seconds: f64,
    samples_per_second: f64,
) -> BTreeMap<String, f64> {
    let metrics = snapshot.ppo;
    let values = [
        ("policy_loss", metrics.policy_loss),
        ("value_loss", metrics.value_loss),
        ("entropy", metrics.entropy),
        ("total_loss", metrics.total_loss),
        ("completed_episode_count", snapshot.episode_count as f64),
        ("window_samples", snapshot.samples as f64),
        ("reverse_action_rate", snapshot.reverse_action_rate),
        ("optimizer_steps_total", optimizer_steps_total as f64),
        ("wall_time_seconds", wall_time_seconds),
        ("samples_per_second", samples_per_second),
        ("raw_adv_mean", metrics.raw_adv_mean),
        ("raw_adv_std", metrics.raw_adv_std),
        ("return_mean", metrics.return_mean),
        ("return_std", metrics.return_std),
        ("clip_fraction", metrics.clip_fraction),
        (
            "critic_explained_variance",
            metrics.critic_explained_variance,
        ),
        ("policy_approx_kl", metrics.policy_approx_kl),
    ];
    let mut grouped = BTreeMap::from([("total_samples".to_owned(), total_samples as f64)]);
    for (name, value) in values {
        grouped.insert(format!("train_samples/{name}"), value);
    }
    if let Some(scores) = scores {
        grouped.insert("train_samples/mean_episode_score".to_owned(), scores.mean);
        grouped.insert("train_samples/std_episode_score".to_owned(), scores.std);
        grouped.insert(
            "train_samples/min_episode_score".to_owned(),
            scores.min as f64,
        );
        grouped.insert(
            "train_samples/max_episode_score".to_owned(),
            scores.max as f64,
        );
        grouped.insert(
            "train_samples/score_window_episode_count".to_owned(),
            scores.count as f64,
        );
    }
    grouped
}

pub fn evaluation_metric_groups(
    metrics: crate::evaluation::EvaluationMetrics,
    total_samples: usize,
) -> BTreeMap<String, f64> {
    let values = [
        ("mean_score", metrics.mean_score),
        ("std_score", metrics.std_score),
        ("min_score", metrics.min_score as f64),
        ("max_score", metrics.max_score as f64),
    ];
    let mut grouped = BTreeMap::from([("total_samples".to_owned(), total_samples as f64)]);
    for (name, value) in values {
        grouped.insert(format!("eval_samples/{name}"), value);
    }
    grouped
}

pub fn eval_stop_metric_groups(
    update: crate::metrics::EarlyStopUpdate,
    total_samples: usize,
) -> BTreeMap<String, f64> {
    let mut grouped = BTreeMap::from([("total_samples".to_owned(), total_samples as f64)]);
    for (name, value) in [
        ("best_mean_score", update.best_mean_score),
        ("stale_evaluations", update.stale_evaluations as f64),
        ("early_stop_triggered", f64::from(update.should_stop)),
    ] {
        grouped.insert(format!("eval_samples/{name}"), value);
    }
    grouped
}

use anyhow::{bail, Context, Result};
use tch::{nn, no_grad, Cuda, Device, Tensor};

use crate::{
    config::{Config, DeviceConfig},
    evaluation::evaluate_policy,
    metrics::{EarlyStopper, SampleWindow, ScoreWindow},
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
    total_samples: usize,
) -> Result<PathBuf> {
    fs::create_dir_all(directory).with_context(|| {
        format!(
            "failed to create checkpoint directory {}",
            directory.display()
        )
    })?;
    let numbered = directory.join(format!("samples-{total_samples}.ot"));
    let numbered_temp = directory.join(format!(".samples-{total_samples}.tmp.ot"));
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
        .clone()
        .unwrap_or_else(generated_run_name);
    let checkpoint_directory = run_checkpoint_directory(&config.training.checkpoint_dir, &run_name);
    let mut wandb = WandbLogger::start(config, &run_name)?;
    let samples_per_iteration = config.environment.num_envs * config.environment.rollout_steps;
    let mut score_window = ScoreWindow::new(config.metrics.score_window_episodes);
    let mut early_stopper = EarlyStopper::new(config.early_stopping.patience_evaluations);
    let mut sample_window = SampleWindow::default();
    let mut optimizer_steps_total = 0usize;
    let mut next_log_samples = config.metrics.log_interval_samples;
    let mut next_eval_samples = config.evaluation.interval_samples;
    let mut next_checkpoint_samples = config.training.checkpoint_interval_samples;
    let mut total_samples = 0usize;
    let mut iteration = 0usize;
    let started = Instant::now();
    let mut last_log_time = started;
    println!("device={device:?} samples_per_iteration={samples_per_iteration}");

    while total_samples < config.training.max_env_samples {
        iteration += 1;
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
        optimizer_steps_total += metrics.optimizer_steps;
        sample_window.push(
            rollout.len(),
            &rollout.episode_scores,
            rollout.reverse_action_count,
            metrics,
            metrics.optimizer_steps,
        );
        score_window.extend(rollout.episode_scores.iter().copied());
        total_samples += rollout.len();
        let reached_budget = total_samples >= config.training.max_env_samples;
        let should_log = total_samples >= next_log_samples || reached_budget;
        let should_evaluate = total_samples >= next_eval_samples || reached_budget;
        let mut should_stop = false;
        let mut logged = BTreeMap::from([("total_samples".to_owned(), total_samples as f64)]);
        if should_log {
            let now = Instant::now();
            let elapsed = (now - last_log_time).as_secs_f64().max(f64::EPSILON);
            let snapshot = sample_window.snapshot();
            logged.extend(training_metric_groups(
                &snapshot,
                score_window.snapshot(),
                total_samples,
                optimizer_steps_total,
                (now - started).as_secs_f64(),
                snapshot.samples as f64 / elapsed,
            ));
            logged.insert(
                "train_samples/learning_rate".to_owned(),
                config.ppo.learning_rate,
            );
            println!(
                "iteration={iteration} total_samples={total_samples} window_samples={} episodes={} rolling_score={:?} value_loss={:.5} entropy={:.5} approx_kl={:.6}",
                snapshot.samples,
                snapshot.episode_count,
                score_window.snapshot().map(|scores| scores.mean),
                snapshot.ppo.value_loss,
                snapshot.ppo.entropy,
                snapshot.ppo.policy_approx_kl,
            );
            sample_window.clear();
            last_log_time = now;
            next_log_samples = advance_sample_threshold(
                next_log_samples,
                config.metrics.log_interval_samples,
                total_samples,
            );
        }
        if should_evaluate {
            let evaluation =
                evaluate_policy(&model, config, device, config.seed.wrapping_add(1_000_003))?;
            println!(
                "evaluation iteration={iteration} episodes={} mean_score={:.3} min_score={} max_score={}",
                config.evaluation.episodes,
                evaluation.mean_score,
                evaluation.min_score,
                evaluation.max_score,
            );
            logged.extend(evaluation_metric_groups(evaluation, total_samples));
            if config.early_stopping.enabled {
                let update = early_stopper.observe(evaluation.mean_score);
                logged.extend(eval_stop_metric_groups(update, total_samples));
                should_stop = update.should_stop;
                println!(
                    "eval_early_stop total_samples={total_samples} best_score={:.3} stale_evaluations={} triggered={}",
                    update.best_mean_score, update.stale_evaluations, should_stop,
                );
            }
            next_eval_samples = advance_sample_threshold(
                next_eval_samples,
                config.evaluation.interval_samples,
                total_samples,
            );
        }
        if should_log || should_evaluate {
            wandb.log(&logged)?;
        }
        if total_samples >= next_checkpoint_samples || should_stop || reached_budget {
            let path = save_checkpoint_atomic(&store, &checkpoint_directory, total_samples)?;
            println!("checkpoint={}", path.display());
            next_checkpoint_samples = advance_sample_threshold(
                next_checkpoint_samples,
                config.training.checkpoint_interval_samples,
                total_samples,
            );
        }
        if should_stop || reached_budget {
            println!(
                "training_end={} total_samples={total_samples}",
                if should_stop {
                    "early_stop"
                } else {
                    "sample_budget"
                }
            );
            break;
        }
    }
    wandb.finish()?;
    Ok(())
}
