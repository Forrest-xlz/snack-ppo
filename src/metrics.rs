use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, Default)]
pub struct TransitionMetric {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
    pub policy_approx_kl: f64,
    pub value_target: f64,
    pub value_prediction: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreWindowMetrics {
    pub mean: f64,
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct TransitionWindowMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
    pub policy_approx_kl: f64,
    pub critic_explained_variance: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MetricSnapshot {
    pub scores: Option<ScoreWindowMetrics>,
    pub transitions: Option<TransitionWindowMetrics>,
}

pub struct MetricWindows {
    scores: VecDeque<u32>,
    transitions: VecDeque<TransitionMetric>,
    score_capacity: usize,
    transition_capacity: usize,
    transition_sums: TransitionSums,
}

#[derive(Default)]
struct TransitionSums {
    policy_loss: f64,
    value_loss: f64,
    entropy: f64,
    total_loss: f64,
    policy_approx_kl: f64,
    target: f64,
    target_squared: f64,
    error: f64,
    error_squared: f64,
}

impl TransitionSums {
    fn add(&mut self, metric: TransitionMetric, sign: f64) {
        let error = metric.value_target - metric.value_prediction;
        self.policy_loss += sign * metric.policy_loss;
        self.value_loss += sign * metric.value_loss;
        self.entropy += sign * metric.entropy;
        self.total_loss += sign * metric.total_loss;
        self.policy_approx_kl += sign * metric.policy_approx_kl;
        self.target += sign * metric.value_target;
        self.target_squared += sign * metric.value_target.powi(2);
        self.error += sign * error;
        self.error_squared += sign * error.powi(2);
    }
}

impl MetricWindows {
    pub fn new(score_capacity: usize, transition_capacity: usize) -> Self {
        assert!(score_capacity > 0);
        assert!(transition_capacity > 0);
        Self {
            scores: VecDeque::with_capacity(score_capacity),
            transitions: VecDeque::with_capacity(transition_capacity),
            score_capacity,
            transition_capacity,
            transition_sums: TransitionSums::default(),
        }
    }

    pub fn extend_scores(&mut self, scores: impl IntoIterator<Item = u32>) {
        for score in scores {
            self.scores.push_back(score);
            if self.scores.len() > self.score_capacity {
                self.scores.pop_front();
            }
        }
    }

    pub fn extend_transitions(&mut self, transitions: impl IntoIterator<Item = TransitionMetric>) {
        for transition in transitions {
            self.transition_sums.add(transition, 1.0);
            self.transitions.push_back(transition);
            if self.transitions.len() > self.transition_capacity {
                if let Some(removed) = self.transitions.pop_front() {
                    self.transition_sums.add(removed, -1.0);
                }
            }
        }
    }

    pub fn snapshot(&self) -> MetricSnapshot {
        let scores = (!self.scores.is_empty()).then(|| {
            let total = self.scores.iter().map(|&score| score as f64).sum::<f64>();
            ScoreWindowMetrics {
                mean: total / self.scores.len() as f64,
                min: self.scores.iter().copied().min().unwrap_or(0),
                max: self.scores.iter().copied().max().unwrap_or(0),
            }
        });
        let transitions = (!self.transitions.is_empty()).then(|| {
            let count = self.transitions.len() as f64;
            let target_variance = (self.transition_sums.target_squared / count
                - (self.transition_sums.target / count).powi(2))
            .max(0.0);
            let error_variance = (self.transition_sums.error_squared / count
                - (self.transition_sums.error / count).powi(2))
            .max(0.0);
            let critic_explained_variance = if target_variance <= f64::EPSILON {
                0.0
            } else {
                1.0 - error_variance / target_variance
            };
            TransitionWindowMetrics {
                policy_loss: self.transition_sums.policy_loss / count,
                value_loss: self.transition_sums.value_loss / count,
                entropy: self.transition_sums.entropy / count,
                total_loss: self.transition_sums.total_loss / count,
                policy_approx_kl: self.transition_sums.policy_approx_kl / count,
                critic_explained_variance,
            }
        });
        MetricSnapshot {
            scores,
            transitions,
        }
    }
}
