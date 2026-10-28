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
    pub window_samples: usize,
    pub episode_count: usize,
    pub mean_score: Option<f64>,
    pub best_mean_score: Option<f64>,
    pub stale_windows: usize,
    pub improved: bool,
    pub should_stop: bool,
}

pub struct EarlyStopper {
    window_samples: usize,
    patience_windows: usize,
    min_episodes: usize,
    next_threshold: usize,
    collected_samples: usize,
    episode_count: usize,
    score_sum: f64,
    best_mean_score: Option<f64>,
    stale_windows: usize,
}

impl EarlyStopper {
    pub fn new(window_samples: usize, patience_windows: usize, min_episodes: usize) -> Self {
        assert!(window_samples > 0 && patience_windows > 0 && min_episodes > 0);
        Self {
            window_samples,
            patience_windows,
            min_episodes,
            next_threshold: window_samples,
            collected_samples: 0,
            episode_count: 0,
            score_sum: 0.0,
            best_mean_score: None,
            stale_windows: 0,
        }
    }

    pub fn observe(
        &mut self,
        total_samples: usize,
        rollout_samples: usize,
        scores: &[u32],
    ) -> Option<EarlyStopUpdate> {
        self.collected_samples += rollout_samples;
        self.episode_count += scores.len();
        self.score_sum += scores.iter().map(|&score| score as f64).sum::<f64>();
        if total_samples < self.next_threshold {
            return None;
        }
        let mean_score = (self.episode_count >= self.min_episodes)
            .then(|| self.score_sum / self.episode_count as f64);
        let improved =
            mean_score.is_some_and(|mean| self.best_mean_score.is_none_or(|best| mean > best));
        if improved {
            self.best_mean_score = mean_score;
            self.stale_windows = 0;
        } else if mean_score.is_some() && self.best_mean_score.is_some() {
            self.stale_windows += 1;
        }
        let update = EarlyStopUpdate {
            window_samples: self.collected_samples,
            episode_count: self.episode_count,
            mean_score,
            best_mean_score: self.best_mean_score,
            stale_windows: self.stale_windows,
            improved,
            should_stop: self.stale_windows >= self.patience_windows,
        };
        self.collected_samples = 0;
        self.episode_count = 0;
        self.score_sum = 0.0;
        while self.next_threshold <= total_samples {
            self.next_threshold = self.next_threshold.saturating_add(self.window_samples);
            if self.next_threshold == usize::MAX {
                break;
            }
        }
        Some(update)
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
