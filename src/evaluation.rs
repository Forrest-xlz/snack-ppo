use anyhow::Result;
use tch::{no_grad, Device, Tensor};

use crate::{
    config::{Config, EvaluationConfig},
    env::{Action, SnakeEnv},
    model::ActorCritic,
};

#[derive(Debug, Clone, Copy)]
pub struct EvaluationMetrics {
    pub mean_score: f64,
    pub std_score: f64,
    pub min_score: u32,
    pub max_score: u32,
}

pub fn evaluate_policy(
    model: &ActorCritic,
    config: &Config,
    device: Device,
    seed: u64,
) -> Result<EvaluationMetrics> {
    evaluate_with(
        model,
        &config.evaluation,
        config.environment.width,
        config.environment.height,
        device,
        seed,
    )
}

fn evaluate_with(
    model: &ActorCritic,
    evaluation: &EvaluationConfig,
    width: usize,
    height: usize,
    device: Device,
    seed: u64,
) -> Result<EvaluationMetrics> {
    let mut scores = Vec::with_capacity(evaluation.episodes);
    for episode in 0..evaluation.episodes {
        let mut env = SnakeEnv::new(width, height, seed + episode as u64)?;
        for _ in 0..evaluation.max_steps_per_episode {
            let observation = env.observation();
            let input = Tensor::from_slice(&observation)
                .view([1, observation.len() as i64])
                .to_device(device);
            let output = no_grad(|| model.actions_and_values(&input, evaluation.deterministic));
            let result = env.step(Action::from_index(output.actions.int64_value(&[0]))?);
            if result.done {
                break;
            }
        }
        scores.push(env.score());
    }
    let sum = scores.iter().map(|&score| score as f64).sum::<f64>();
    let mean_score = sum / scores.len() as f64;
    Ok(EvaluationMetrics {
        mean_score,
        std_score: (scores
            .iter()
            .map(|&score| (score as f64).powi(2))
            .sum::<f64>()
            / scores.len() as f64
            - mean_score.powi(2))
        .max(0.0)
        .sqrt(),
        min_score: scores.iter().copied().min().unwrap_or(0),
        max_score: scores.iter().copied().max().unwrap_or(0),
    })
}
