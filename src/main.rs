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
    let config = Config::load_default()?;
    match config.mode {
        Mode::Train => run_training(&config),
        Mode::Infer => run_inference(config),
    }
}
