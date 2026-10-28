use anyhow::{Context, Result};
use rayon::prelude::*;
use tch::{no_grad, Device, Tensor};

use crate::{
    config::EnvironmentConfig,
    env::{Action, SnakeEnv},
    model::ActorCritic,
};

pub struct EnvTransition {
    pub reward: f32,
    pub done: bool,
    pub terminal_score: Option<u32>,
    pub reversed_action: bool,
}

pub struct VectorEnv {
    envs: Vec<SnakeEnv>,
    observation_size: usize,
}

impl VectorEnv {
    pub fn new(config: &EnvironmentConfig, base_seed: u64) -> Result<Self> {
        let envs = (0..config.num_envs)
            .map(|index| SnakeEnv::new(config.width, config.height, base_seed + index as u64))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            envs,
            observation_size: 3 * config.width * config.height,
        })
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }
    pub fn observation_size(&self) -> usize {
        self.observation_size
    }

    pub fn observations(&self) -> Vec<f32> {
        self.envs.iter().flat_map(SnakeEnv::observation).collect()
    }

    pub fn step(&mut self, actions: &[i64]) -> Result<Vec<EnvTransition>> {
        if actions.len() != self.envs.len() {
            anyhow::bail!(
                "expected {} actions, got {}",
                self.envs.len(),
                actions.len()
            );
        }
        self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, &action)| {
                let action = Action::from_index(action)?;
                let reversed_action = env.would_reverse(action);
                let result = env.step(action);
                let terminal_score = result.done.then_some(result.score);
                if result.done {
                    env.reset();
                }
                Ok(EnvTransition {
                    reward: result.reward,
                    done: result.done,
                    terminal_score,
                    reversed_action,
                })
            })
            .collect()
    }
}

pub struct RolloutBuffer {
    pub observations: Vec<f32>,
    pub actions: Vec<i64>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub values: Vec<f32>,
    pub log_probs: Vec<f32>,
    pub episode_scores: Vec<u32>,
    pub reverse_action_count: usize,
    pub steps: usize,
    pub num_envs: usize,
    pub observation_size: usize,
}

impl RolloutBuffer {
    pub fn len(&self) -> usize {
        self.actions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

pub fn collect_rollout(
    model: &ActorCritic,
    envs: &mut VectorEnv,
    rollout_steps: usize,
    device: Device,
) -> Result<RolloutBuffer> {
    let count = envs.len() * rollout_steps;
    let mut buffer = RolloutBuffer {
        observations: Vec::with_capacity(count * envs.observation_size()),
        actions: Vec::with_capacity(count),
        rewards: Vec::with_capacity(count),
        dones: Vec::with_capacity(count),
        values: Vec::with_capacity(count),
        log_probs: Vec::with_capacity(count),
        episode_scores: Vec::new(),
        reverse_action_count: 0,
        steps: rollout_steps,
        num_envs: envs.len(),
        observation_size: envs.observation_size(),
    };

    for _ in 0..rollout_steps {
        let observations = envs.observations();
        let tensor = Tensor::from_slice(&observations)
            .view([envs.len() as i64, envs.observation_size() as i64])
            .to_device(device);
        let output = no_grad(|| model.actions_and_values(&tensor, false));
        let actions = Vec::<i64>::try_from(&output.actions.to_device(Device::Cpu))
            .context("failed to copy sampled actions to CPU")?;
        let values = Vec::<f32>::try_from(&output.values.to_device(Device::Cpu))
            .context("failed to copy values to CPU")?;
        let log_probs = Vec::<f32>::try_from(&output.log_probs.to_device(Device::Cpu))
            .context("failed to copy log probabilities to CPU")?;
        let transitions = envs.step(&actions)?;

        buffer.observations.extend(observations);
        buffer.actions.extend(actions);
        buffer.values.extend(values);
        buffer.log_probs.extend(log_probs);
        for transition in transitions {
            buffer.reverse_action_count += usize::from(transition.reversed_action);
            buffer.rewards.push(transition.reward);
            buffer.dones.push(transition.done);
            if let Some(score) = transition.terminal_score {
                buffer.episode_scores.push(score);
            }
        }
    }
    Ok(buffer)
}
