use anyhow::{Context, Result};
use tch::{no_grad, Device, Tensor};

use crate::{
    config::{Config, EvaluationConfig},
    env::{Action, SnakeEnv},
    features::FeatureEncoding,
    model::ActorCritic,
};

#[derive(Debug, Clone, Copy)]
pub struct EvaluationMetrics {
    pub mean_score: f64,
    pub std_score: f64,
    pub min_score: u32,
    pub max_score: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripted_actions(
        observations: &[f32],
        batch: usize,
        size: usize,
        area: usize,
        width: usize,
    ) -> Vec<i64> {
        (0..batch)
            .map(|row| {
                let food = observations[row * size + 2 * area..row * size + 3 * area]
                    .iter()
                    .position(|v| *v > 0.5)
                    .unwrap_or(0);
                if (food % width) % 2 == 0 {
                    3
                } else {
                    0
                } // right or up
            })
            .collect()
    }

    #[test]
    fn active_batches_match_serial_scores_seeds_and_step_counts() {
        let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
        for encoding in [
            FeatureEncoding::HeadBodyFood,
            FeatureEncoding::HeadBodyFoodGeometryOrder,
        ] {
            let mut evaluation = base.evaluation.clone();
            evaluation.episodes = 23;
            evaluation.max_steps_per_episode = 30;
            let mut expected = Vec::new();
            let mut expected_steps = 0;
            for episode in 0..evaluation.episodes {
                let mut env = SnakeEnv::new(7, 5, 42 + episode as u64).unwrap();
                for _ in 0..evaluation.max_steps_per_episode {
                    let obs = encoding.encode(&env);
                    let action = scripted_actions(&obs, 1, obs.len(), 35, 7)[0];
                    let result = env.step(Action::from_index(action).unwrap());
                    expected_steps += 1;
                    if result.done {
                        break;
                    }
                }
                expected.push(env.score());
            }
            for cap in [1, 5, 1024] {
                evaluation.batch_size = cap;
                let mut batch_sizes = Vec::new();
                let scores = evaluate_episodes(&evaluation, 7, 5, 42, encoding, |obs, batch| {
                    batch_sizes.push(batch);
                    Ok(scripted_actions(
                        obs,
                        batch,
                        encoding.observation_size(7, 5),
                        35,
                        7,
                    ))
                })
                .unwrap();
                assert_eq!(scores, expected);
                assert_eq!(batch_sizes.iter().sum::<usize>(), expected_steps);
                assert_eq!(batch_sizes[0], cap.min(23));
                assert!(batch_sizes.iter().all(|n| *n > 0 && *n <= cap));
            }
        }
    }

    #[test]
    fn truncation_limits_each_episode_and_handles_partial_final_cohort() {
        let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
        let mut evaluation = base.evaluation;
        evaluation.episodes = 7;
        evaluation.batch_size = 4;
        evaluation.max_steps_per_episode = 1;
        let mut batches = Vec::new();
        let scores = evaluate_episodes(
            &evaluation,
            5,
            5,
            123,
            FeatureEncoding::HeadBodyFood,
            |_, batch| {
                batches.push(batch);
                Ok(vec![3; batch])
            },
        )
        .unwrap();
        assert_eq!(scores.len(), 7);
        assert_eq!(batches, vec![4, 3]);
    }

    #[test]
    fn batch_1024_is_a_single_model_call_for_1024_one_step_episodes() {
        let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
        let mut evaluation = base.evaluation;
        evaluation.episodes = 1024;
        evaluation.batch_size = 1024;
        evaluation.max_steps_per_episode = 1;
        let mut calls = Vec::new();
        let scores = evaluate_episodes(
            &evaluation,
            5,
            5,
            123,
            FeatureEncoding::HeadBodyFood,
            |_, batch| {
                calls.push(batch);
                Ok(vec![3; batch])
            },
        )
        .unwrap();
        assert_eq!(scores.len(), 1024);
        assert_eq!(calls, vec![1024]);
    }
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
        config.model.version.feature_encoding(),
    )
}

fn evaluate_with(
    model: &ActorCritic,
    evaluation: &EvaluationConfig,
    width: usize,
    height: usize,
    device: Device,
    seed: u64,
    feature_encoding: FeatureEncoding,
) -> Result<EvaluationMetrics> {
    let observation_size = feature_encoding.observation_size(width, height);
    let scores = evaluate_episodes(
        evaluation,
        width,
        height,
        seed,
        feature_encoding,
        |observations, batch| {
            let input = Tensor::from_slice(observations)
                .view([batch as i64, observation_size as i64])
                .to_device(device);
            let output = no_grad(|| model.actions_and_values(&input, evaluation.deterministic));
            Vec::<i64>::try_from(&output.actions.to_device(Device::Cpu))
                .context("failed to copy evaluation action batch to CPU")
        },
    )?;
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

/// Each episode keeps its original ID/seed. Completed games leave the active
/// cohort without resetting; truncation is applied separately to every game.
fn evaluate_episodes<F>(
    evaluation: &EvaluationConfig,
    width: usize,
    height: usize,
    seed: u64,
    encoding: FeatureEncoding,
    mut predict: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&[f32], usize) -> Result<Vec<i64>>,
{
    anyhow::ensure!(
        evaluation.episodes > 0,
        "evaluation.episodes must be positive"
    );
    anyhow::ensure!(
        evaluation.batch_size > 0,
        "evaluation.batch_size must be positive"
    );
    anyhow::ensure!(
        evaluation.max_steps_per_episode > 0,
        "evaluation.max_steps_per_episode must be positive"
    );
    let mut scores = vec![0; evaluation.episodes];
    for start in (0..evaluation.episodes).step_by(evaluation.batch_size) {
        let end = start
            .saturating_add(evaluation.batch_size)
            .min(evaluation.episodes);
        let mut active = (start..end)
            .map(|episode| {
                Ok((
                    episode,
                    SnakeEnv::new(width, height, seed.wrapping_add(episode as u64))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for _ in 0..evaluation.max_steps_per_episode {
            if active.is_empty() {
                break;
            }
            let observations = active
                .iter()
                .flat_map(|(_, env)| encoding.encode(env))
                .collect::<Vec<_>>();
            let actions = predict(&observations, active.len())?;
            anyhow::ensure!(
                actions.len() == active.len(),
                "evaluation policy returned an incorrect action count"
            );
            for ((episode, env), action) in active.iter_mut().zip(actions) {
                let result = env.step(Action::from_index(action)?);
                if result.done {
                    scores[*episode] = result.score;
                }
            }
            active.retain(|(_, env)| !env.is_done());
        }
        // Survivors reached the per-episode step cap, without being reset.
        for (episode, env) in active {
            scores[episode] = env.score();
        }
    }
    Ok(scores)
}
