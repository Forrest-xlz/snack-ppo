use std::time::Instant;

use anyhow::{bail, Context, Result};
use ppo_snake::{
    config::Config,
    kickstarting::TeacherPolicy,
    model::ActorCritic,
    ppo::PpoTrainer,
    rollout::{collect_rollout, VectorEnv},
    teacher_kl::EvalTeacher,
    trainer::resolve_device,
};
use tch::{nn, no_grad, Cuda, Device, Tensor};

fn sync(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut path = None;
    let mut device_override = None;
    let mut warmup = 1usize;
    let mut rollouts = 2usize;
    let mut no_teacher = false;
    let mut output_path = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--no-teacher" {
            no_teacher = true;
            index += 1;
            continue;
        }
        let value = args
            .get(index + 1)
            .context("missing benchmark argument value")?;
        match flag {
            "--config" => path = Some(value.clone()),
            "--device" => device_override = Some(value.clone()),
            "--warmup" => warmup = value.parse()?,
            "--rollouts" => rollouts = value.parse()?,
            "--output" => output_path = Some(std::path::PathBuf::from(value)),
            _ => bail!("unknown benchmark option {flag}"),
        }
        index += 2;
    }
    anyhow::ensure!(rollouts > 0, "rollouts must be positive");
    let mut config = Config::load(path.context(
        "use --config PATH [--device cpu|cuda] [--warmup 1] [--rollouts 2] [--no-teacher]",
    )?)?;
    if let Some(device) = device_override {
        config.device = serde_yaml::from_str(&device)?;
    }
    if no_teacher {
        config.kickstarting = None;
        config.teacher_kl = None;
    }
    let device = resolve_device(config.device)?;
    let mut eval_teacher = EvalTeacher::new(&config, device)?;
    tch::manual_seed(config.seed as i64);
    let store = nn::VarStore::new(device);
    let model = ActorCritic::from_config(&store.root(), &config);
    if eval_teacher.enabled() {
        // Benchmark steady-state teacher overhead immediately, without running
        // a formal evaluation or writing teacher/checkpoint files.
        eval_teacher.observe(0.0, 0, &store)?;
    }
    let backend = model.attention_backend(
        device,
        config.environment.num_envs as i64,
        (config.environment.width * config.environment.height) as i64,
        false,
    );
    let training_backend = model.attention_backend(
        device,
        config.ppo.batch_size as i64,
        (config.environment.width * config.environment.height) as i64,
        true,
    );
    // A benchmark must never write formal run checkpoints or initialize W&B.
    config.training.checkpoint_dir = std::path::PathBuf::from("target/benchmark-teacher-guard");
    let teacher = TeacherPolicy::load(&config, device, "benchmark")?;
    let mut optimizer = PpoTrainer::new(&store, &config.ppo, config.seed)?;
    let mut envs = VectorEnv::with_features(
        &config.environment,
        config.seed,
        config.model.version.feature_encoding(),
    )?;
    let samples_per_rollout = config.environment.num_envs * config.environment.rollout_steps;
    let mut collection_seconds = 0.0;
    let mut update_seconds = 0.0;
    let mut bootstrap_seconds = 0.0;
    for iteration in 0..warmup + rollouts {
        sync(device);
        let collection_start = Instant::now();
        let rollout = collect_rollout(&model, &mut envs, config.environment.rollout_steps, device)?;
        sync(device);
        let collection_elapsed = collection_start.elapsed().as_secs_f64();
        let bootstrap_start = Instant::now();
        let input = Tensor::from_slice(&envs.observations())
            .view([envs.len() as i64, envs.observation_size() as i64])
            .to_device(device);
        let (_, values) = no_grad(|| model.forward(&input));
        let values = Vec::<f32>::try_from(&values.to_device(Device::Cpu))?;
        sync(device);
        let bootstrap_elapsed = bootstrap_start.elapsed().as_secs_f64();
        let update_start = Instant::now();
        let weight = config
            .kickstarting
            .as_ref()
            .map_or(0.0, |kick| kick.weight_at(iteration * samples_per_rollout));
        let update = if eval_teacher.enabled() {
            optimizer.update_with_teacher_kl(
                &model,
                &rollout,
                &values,
                device,
                eval_teacher.policy().map(|p| (p, eval_teacher.coef())),
            )?
        } else {
            optimizer.update_with_kickstarting(
                &model,
                &rollout,
                &values,
                device,
                teacher.as_ref().map(|p| (p, weight)),
            )?
        };
        sync(device);
        let update_elapsed = update_start.elapsed().as_secs_f64();
        anyhow::ensure!(
            update.metrics.total_loss.is_finite(),
            "non-finite benchmark training loss"
        );
        eprintln!("benchmark rollout={} warmup={} collection_seconds={collection_elapsed:.4} update_seconds={update_elapsed:.4}",iteration+1,iteration<warmup);
        if iteration >= warmup {
            collection_seconds += collection_elapsed;
            bootstrap_seconds += bootstrap_elapsed;
            update_seconds += update_elapsed;
        }
    }
    let samples = rollouts * samples_per_rollout;
    let report = serde_json::json!({"device":format!("{device:?}"),"model_version":config.model.version,
        "collection_attention_backend":backend,"update_attention_backend":training_backend,
        "teacher_enabled":teacher.is_some() || eval_teacher.enabled(),
        "teacher_kl_coef":eval_teacher.coef(),
        "teacher_kl_benchmark_snapshot":if eval_teacher.enabled() {Some("initial_student_without_evaluation")} else {None},
        "num_envs":config.environment.num_envs,
        "rollout_steps":config.environment.rollout_steps,"batch_size":config.ppo.batch_size,
        "epochs":config.ppo.epochs,"warmup_rollouts":warmup,"timed_rollouts":rollouts,
        "samples":samples,"collection_seconds":collection_seconds,"update_seconds":update_seconds,
        "bootstrap_seconds":bootstrap_seconds,"collection_samples_per_second":samples as f64/collection_seconds,
        "training_samples_per_second":samples as f64/(collection_seconds+update_seconds+bootstrap_seconds)});
    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    println!("{report}");
    Ok(())
}
