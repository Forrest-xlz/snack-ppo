use anyhow::Result;
use ppo_snake::{
    config::{Config, Mode},
    renderer::run_inference,
    trainer::run_training,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let config = match args.as_slice() {
        [flag, path] if flag == "--config" => Config::load(path)?,
        _ => anyhow::bail!("usage: ppo-snake --config PATH"),
    };
    match config.mode {
        Mode::Train => run_training(&config),
        Mode::Infer => run_inference(config),
    }
}
