use anyhow::Result;
use rand::{seq::SliceRandom, Rng};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::{config::PpoConfig, model::ActorCritic, rollout::RolloutBuffer};

#[derive(Debug, Clone, Copy)]
pub struct GaeParams {
    pub steps: usize,
    pub envs: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
}

pub fn compute_gae(
    rewards: &[f32],
    dones: &[bool],
    values: &[f32],
    last_values: &[f32],
    params: GaeParams,
) -> (Vec<f32>, Vec<f32>) {
    let GaeParams {
        steps,
        envs,
        gamma,
        gae_lambda,
    } = params;
    assert_eq!(rewards.len(), steps * envs);
    assert_eq!(dones.len(), rewards.len());
    assert_eq!(values.len(), rewards.len());
    assert_eq!(last_values.len(), envs);
    let mut advantages = vec![0.0; rewards.len()];
    let mut running = vec![0.0; envs];
    for step in (0..steps).rev() {
        for env in 0..envs {
            let index = step * envs + env;
            let next_value = if step + 1 == steps {
                last_values[env]
            } else {
                values[(step + 1) * envs + env]
            };
            let non_terminal = if dones[index] { 0.0 } else { 1.0 };
            let delta = rewards[index] + gamma * next_value * non_terminal - values[index];
            running[env] = delta + gamma * gae_lambda * non_terminal * running[env];
            advantages[index] = running[env];
        }
    }
    let returns = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

pub fn shuffled_minibatches<R: Rng + ?Sized>(
    sample_count: usize,
    batch_size: usize,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    let mut indices = (0..sample_count).collect::<Vec<_>>();
    indices.shuffle(rng);
    indices.chunks(batch_size).map(<[usize]>::to_vec).collect()
}

pub fn explained_variance(returns: &[f32], predictions: &[f32]) -> f64 {
    assert_eq!(returns.len(), predictions.len());
    if returns.is_empty() {
        return 0.0;
    }
    let variance_returns = population_variance(returns.iter().map(|&value| value as f64));
    if variance_returns <= f64::EPSILON {
        return 0.0;
    }
    let errors = returns
        .iter()
        .zip(predictions)
        .map(|(&target, &prediction)| (target - prediction) as f64);
    1.0 - population_variance(errors) / variance_returns
}

pub fn approximate_kl(old_log_probs: &[f64], new_log_probs: &[f64]) -> f64 {
    assert_eq!(old_log_probs.len(), new_log_probs.len());
    if old_log_probs.is_empty() {
        return 0.0;
    }
    old_log_probs
        .iter()
        .zip(new_log_probs)
        .map(|(&old, &new)| {
            let log_ratio = new - old;
            (-log_ratio).exp() - 1.0 + log_ratio
        })
        .sum::<f64>()
        / old_log_probs.len() as f64
}

fn population_variance(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PpoMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
    pub critic_explained_variance: f64,
    pub policy_approx_kl: f64,
}

pub struct PpoTrainer {
    optimizer: nn::Optimizer,
    config: PpoConfig,
    rng: ChaCha8Rng,
}

impl PpoTrainer {
    pub fn new(store: &nn::VarStore, config: &PpoConfig, seed: u64) -> Result<Self> {
        let optimizer = nn::Adam::default().build(store, config.learning_rate)?;
        Ok(Self {
            optimizer,
            config: config.clone(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    pub fn update(
        &mut self,
        model: &ActorCritic,
        rollout: &RolloutBuffer,
        last_values: &[f32],
        device: Device,
    ) -> Result<PpoMetrics> {
        let (advantages, returns) = compute_gae(
            &rollout.rewards,
            &rollout.dones,
            &rollout.values,
            last_values,
            GaeParams {
                steps: rollout.steps,
                envs: rollout.num_envs,
                gamma: self.config.gamma,
                gae_lambda: self.config.gae_lambda,
            },
        );
        let critic_explained_variance = explained_variance(&returns, &rollout.values);
        let sample_count = rollout.len();
        let observations = Tensor::from_slice(&rollout.observations)
            .view([sample_count as i64, rollout.observation_size as i64])
            .to_device(device);
        let actions = Tensor::from_slice(&rollout.actions).to_device(device);
        let old_log_probs = Tensor::from_slice(&rollout.log_probs).to_device(device);
        let returns = Tensor::from_slice(&returns).to_device(device);
        let advantages = Tensor::from_slice(&advantages).to_device(device);
        let advantages =
            (&advantages - advantages.mean(Kind::Float)) / (advantages.std(false) + 1e-8);

        let mut metrics = PpoMetrics::default();
        let mut updates = 0usize;
        for _ in 0..self.config.epochs {
            for batch in shuffled_minibatches(sample_count, self.config.batch_size, &mut self.rng) {
                let indices = Tensor::from_slice(
                    &batch
                        .into_iter()
                        .map(|value| value as i64)
                        .collect::<Vec<_>>(),
                )
                .to_device(device);
                let batch_observations = observations.index_select(0, &indices);
                let batch_actions = actions.index_select(0, &indices);
                let evaluation = model.evaluate_actions(&batch_observations, &batch_actions);
                let batch_old_log_probs = old_log_probs.index_select(0, &indices);
                let batch_advantages = advantages.index_select(0, &indices);
                let batch_returns = returns.index_select(0, &indices);

                let log_ratio = &evaluation.log_probs - batch_old_log_probs;
                let ratio = log_ratio.exp();
                let approximate_kl = ((-&log_ratio).exp() - 1.0 + &log_ratio).mean(Kind::Float);
                let surrogate = &ratio * &batch_advantages;
                let clipped = ratio
                    .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                    * &batch_advantages;
                let policy_loss = -surrogate.minimum(&clipped).mean(Kind::Float);
                let value_loss = (&batch_returns - &evaluation.values)
                    .pow_tensor_scalar(2.0)
                    .mean(Kind::Float);
                let entropy = evaluation.entropy.mean(Kind::Float);
                let total_loss = &policy_loss + self.config.value_coef * &value_loss
                    - self.config.entropy_coef * &entropy;
                self.optimizer
                    .backward_step_clip(&total_loss, self.config.max_grad_norm);

                metrics.policy_loss += policy_loss.double_value(&[]);
                metrics.value_loss += value_loss.double_value(&[]);
                metrics.entropy += entropy.double_value(&[]);
                metrics.total_loss += total_loss.double_value(&[]);
                metrics.policy_approx_kl += approximate_kl.double_value(&[]);
                updates += 1;
            }
        }
        let divisor = updates as f64;
        metrics.policy_loss /= divisor;
        metrics.value_loss /= divisor;
        metrics.entropy /= divisor;
        metrics.total_loss /= divisor;
        metrics.policy_approx_kl /= divisor;
        metrics.critic_explained_variance = critic_explained_variance;
        Ok(metrics)
    }
}
