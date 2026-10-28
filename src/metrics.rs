use std::collections::VecDeque;

use crate::ppo::PpoMetrics;

#[derive(Debug, Clone, Copy)]
pub struct ScoreWindowMetrics {
    pub mean: f64,
    pub std: f64,
    pub count: usize,
    pub min: u32,
    pub max: u32,
}

pub struct ScoreWindow {
    scores: VecDeque<u32>,
    capacity: usize,
}

impl ScoreWindow {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            scores: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn extend(&mut self, scores: impl IntoIterator<Item = u32>) {
        for score in scores {
            self.scores.push_back(score);
            if self.scores.len() > self.capacity {
                self.scores.pop_front();
            }
        }
    }

    pub fn snapshot(&self) -> Option<ScoreWindowMetrics> {
        if self.scores.is_empty() {
            return None;
        }
        let count = self.scores.len();
        let mean = self.scores.iter().map(|&score| score as f64).sum::<f64>() / count as f64;
        let std = (self
            .scores
            .iter()
            .map(|&score| (score as f64).powi(2))
            .sum::<f64>()
            / count as f64
            - mean.powi(2))
        .max(0.0)
        .sqrt();
        Some(ScoreWindowMetrics {
            mean,
            std,
            count,
            min: *self.scores.iter().min().unwrap(),
            max: *self.scores.iter().max().unwrap(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EarlyStopUpdate {
    pub best_mean_score: f64,
    pub stale_evaluations: usize,
    pub improved: bool,
    pub should_stop: bool,
}

pub struct EarlyStopper {
    patience_evaluations: usize,
    best_mean_score: Option<f64>,
    stale_evaluations: usize,
}

impl EarlyStopper {
    pub fn new(patience_evaluations: usize) -> Self {
        assert!(patience_evaluations > 0);
        Self {
            patience_evaluations,
            best_mean_score: None,
            stale_evaluations: 0,
        }
    }

    pub fn observe(&mut self, mean_score: f64) -> EarlyStopUpdate {
        let improved = self.best_mean_score.is_none_or(|best| mean_score > best);
        if improved {
            self.best_mean_score = Some(mean_score);
            self.stale_evaluations = 0;
        } else {
            self.stale_evaluations += 1;
        }
        EarlyStopUpdate {
            best_mean_score: self.best_mean_score.unwrap(),
            stale_evaluations: self.stale_evaluations,
            improved,
            should_stop: self.stale_evaluations >= self.patience_evaluations,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SampleWindowSnapshot {
    pub samples: usize,
    pub episode_count: usize,
    pub reverse_action_rate: f64,
    pub optimizer_steps: usize,
    pub ppo: PpoMetrics,
}

#[derive(Default)]
pub struct SampleWindow {
    samples: usize,
    episode_count: usize,
    reverse_actions: usize,
    optimizer_steps: usize,
    updates: Vec<(usize, PpoMetrics)>,
}

impl SampleWindow {
    pub fn push(
        &mut self,
        samples: usize,
        scores: &[u32],
        reverse_actions: usize,
        ppo: PpoMetrics,
        optimizer_steps: usize,
    ) {
        self.samples += samples;
        self.episode_count += scores.len();
        self.reverse_actions += reverse_actions;
        self.optimizer_steps += optimizer_steps;
        self.updates.push((samples, ppo));
    }

    pub fn snapshot(&self) -> SampleWindowSnapshot {
        let weighted = |get: fn(&PpoMetrics) -> f64| {
            if self.samples == 0 {
                0.0
            } else {
                self.updates
                    .iter()
                    .map(|(samples, metrics)| *samples as f64 * get(metrics))
                    .sum::<f64>()
                    / self.samples as f64
            }
        };
        let pooled_std =
            |mean: f64, get_mean: fn(&PpoMetrics) -> f64, get_std: fn(&PpoMetrics) -> f64| {
                (self
                    .updates
                    .iter()
                    .map(|(samples, metrics)| {
                        *samples as f64 * (get_std(metrics).powi(2) + get_mean(metrics).powi(2))
                    })
                    .sum::<f64>()
                    / self.samples.max(1) as f64
                    - mean.powi(2))
                .max(0.0)
                .sqrt()
            };
        let raw_adv_mean = weighted(|m| m.raw_adv_mean);
        let return_mean = weighted(|m| m.return_mean);
        SampleWindowSnapshot {
            samples: self.samples,
            episode_count: self.episode_count,
            reverse_action_rate: if self.samples == 0 {
                0.0
            } else {
                self.reverse_actions as f64 / self.samples as f64
            },
            optimizer_steps: self.optimizer_steps,
            ppo: PpoMetrics {
                policy_loss: weighted(|m| m.policy_loss),
                value_loss: weighted(|m| m.value_loss),
                entropy: weighted(|m| m.entropy),
                total_loss: weighted(|m| m.total_loss),
                critic_explained_variance: weighted(|m| m.critic_explained_variance),
                policy_approx_kl: weighted(|m| m.policy_approx_kl),
                raw_adv_mean,
                raw_adv_std: pooled_std(raw_adv_mean, |m| m.raw_adv_mean, |m| m.raw_adv_std),
                return_mean,
                return_std: pooled_std(return_mean, |m| m.return_mean, |m| m.return_std),
                clip_fraction: weighted(|m| m.clip_fraction),
                optimizer_steps: self.optimizer_steps,
            },
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}
