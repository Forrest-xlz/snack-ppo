use anyhow::Result;
use ppo_snake::{
    config::{Config, Mode},
    renderer::run_inference,
    trainer::{run_checkpoint_directory, run_training},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (config, inspect_directory) = match args.as_slice() {
        [flag, path] if flag == "--config" => (Config::load(path)?, false),
        [inspect, flag, path] if inspect == "--checkpoint-directory" && flag == "--config" => {
            (Config::load(path)?, true)
        }
        _ => anyhow::bail!("usage: ppo-snake --config PATH"),
    };
    if inspect_directory {
        let name = config.wandb.run_name.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--checkpoint-directory requires an explicit wandb.run_name")
        })?;
        println!(
            "{}",
            run_checkpoint_directory(&config.training.checkpoint_dir, name).display()
        );
        return Ok(());
    }
    match config.mode {
        Mode::Train => run_training(&config),
        Mode::Infer => run_inference(config),
    }
}
