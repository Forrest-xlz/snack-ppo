//! CPU-only feature benchmark; never starts training or W&B.
use anyhow::{Context, Result};
use ppo_snake::{
    config::Config,
    env::{Direction, Point, SnakeEnv},
};
use std::{collections::VecDeque, hint::black_box, time::Instant};

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let output = args
        .first()
        .context("usage: benchmark_features OUTPUT.json [ITERATIONS]")?;
    let iterations = args.get(1).map_or(Ok(10000usize), |s| s.parse())?;
    anyhow::ensure!(iterations > 0, "iterations must be positive");
    let config = Config::load("config/experiments/feature_architecture/B.yaml")?;
    let encoding = config.model.version.feature_encoding();
    let mut results = Vec::new();
    for length in [3usize, 73, 113] {
        let points = (0..12)
            .flat_map(|y| (0..12).map(move |i| Point::new(if y % 2 == 0 { i } else { 11 - i }, y)))
            .take(length)
            .collect::<VecDeque<_>>();
        let env = SnakeEnv::from_state(
            12,
            12,
            points,
            Direction::Left,
            Some(Point::new(11, 11)),
            42,
        )?;
        let features = encoding.encode(&env);
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(encoding.encode(black_box(&env)));
        }
        let seconds = started.elapsed().as_secs_f64();
        results.push(serde_json::json!({"length":length,"iterations":iterations,
            "seconds":seconds,"microseconds_per_state":seconds*1e6/iterations as f64,
            "features":features}));
        println!(
            "length={length} microseconds_per_state={:.3}",
            seconds * 1e6 / iterations as f64
        );
    }
    let path = std::path::Path::new(output);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string(&results)?)?;
    Ok(())
}
