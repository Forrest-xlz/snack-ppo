use anyhow::{Context, Result};
use tch::{nn, no_grad, Device, Kind, Tensor};

use crate::{
    config::Config,
    model::ActorCritic,
    trainer::{load_checkpoint, run_checkpoint_directory},
};

/// Full teacher policy, on student states. Detach is defensive: neither the
/// labels nor teacher weights receive gradients from the student's loss.
pub fn policy_cross_entropy(teacher_probabilities: &Tensor, student_log_probs: &Tensor) -> Tensor {
    -(teacher_probabilities.detach() * student_log_probs).sum_dim_intlist(
        [-1i64].as_slice(),
        false,
        Kind::Float,
    )
}

pub struct TeacherPolicy {
    model: ActorCritic,
    store: nn::VarStore,
    input_size: i64,
}

impl TeacherPolicy {
    pub fn load(config: &Config, device: Device, run_name: &str) -> Result<Option<Self>> {
        let Some(kick) = &config.kickstarting else {
            return Ok(None);
        };
        let source = std::fs::canonicalize(&kick.teacher_checkpoint).with_context(|| {
            format!(
                "failed to find teacher checkpoint {}",
                kick.teacher_checkpoint.display()
            )
        })?;
        let output = run_checkpoint_directory(&config.training.checkpoint_dir, run_name);
        std::fs::create_dir_all(&output)?;
        let output = std::fs::canonicalize(output)?;
        anyhow::ensure!(
            !source.starts_with(&output),
            "student output directory must not contain the teacher checkpoint"
        );
        let mut teacher_config = config.clone();
        teacher_config.model = kick.teacher_model.clone();
        teacher_config.kickstarting = None;
        let mut store = nn::VarStore::new(device);
        let model = ActorCritic::from_config(&store.root(), &teacher_config);
        load_checkpoint(&mut store, &source)
            .context("failed to load teacher weights; check teacher model and board dimensions")?;
        store.freeze();
        let input_size = kick
            .teacher_model
            .version
            .feature_encoding()
            .observation_size(config.environment.width, config.environment.height)
            as i64;
        println!("kickstarting_teacher={} frozen=true", source.display());
        Ok(Some(Self {
            model,
            store,
            input_size,
        }))
    }

    /// Allocate before seeding the student to leave its RNG stream unchanged.
    pub(crate) fn empty_snapshot(config: &Config, device: Device) -> Self {
        let mut store = nn::VarStore::new(device);
        let model = ActorCritic::from_config(&store.root(), config);
        store.freeze();
        Self {
            model,
            store,
            input_size: config
                .model
                .version
                .feature_encoding()
                .observation_size(config.environment.width, config.environment.height)
                as i64,
        }
    }

    pub(crate) fn copy_weights(&mut self, source: &nn::VarStore) -> Result<()> {
        self.store.copy(source)?;
        self.store.freeze();
        Ok(())
    }

    pub(crate) fn save(&self, path: &std::path::Path) -> Result<()> {
        self.store.save(path)?;
        Ok(())
    }

    /// Cache only four probabilities per state once per rollout, chunking
    /// teacher inference so its hidden activations do not grow with rollout R.
    pub fn probabilities(&self, student_observations: &Tensor, batch_size: usize) -> Tensor {
        no_grad(|| {
            let inputs = student_observations.narrow(1, 0, self.input_size);
            let probabilities = inputs
                .split(batch_size as i64, 0)
                .iter()
                .map(|batch| {
                    let (logits, _) = self.model.forward(batch);
                    logits.softmax(-1, Kind::Float)
                })
                .collect::<Vec<_>>();
            Tensor::cat(&probabilities, 0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_uses_soft_teacher_distribution_and_only_student_gradients() {
        let teacher = Tensor::from_slice(&[0.75f32, 0.25])
            .view([1, 2])
            .set_requires_grad(true);
        let student = Tensor::from_slice(&[0.2f32.ln(), 0.8f32.ln()])
            .view([1, 2])
            .set_requires_grad(true);
        let loss = policy_cross_entropy(&teacher, &student).mean(Kind::Float);
        let expected = -0.75 * 0.2f64.ln() - 0.25 * 0.8f64.ln();
        assert!((loss.double_value(&[]) - expected).abs() < 1e-6);
        loss.backward();
        assert!(!teacher.grad().defined());
        assert!(student.grad().allclose(
            &Tensor::from_slice(&[-0.75f32, -0.25]).view([1, 2]),
            1e-6,
            1e-6,
            false
        ));
    }

    #[test]
    fn linear_schedule_is_sample_based_and_remains_zero_after_decay() {
        let config = crate::config::KickstartingConfig {
            teacher_checkpoint: "teacher.ot".into(),
            teacher_model: crate::config::ModelConfig {
                version: crate::config::ModelVersion::MlpV1,
                hidden_size: 256,
            },
            initial_weight: 1.0,
            decay_samples: 200_000_000,
        };
        assert_eq!(config.weight_at(0), 1.0);
        assert_eq!(config.weight_at(100_000_000), 0.5);
        assert_eq!(config.weight_at(200_000_000), 0.0);
        assert_eq!(config.weight_at(1_200_000_000), 0.0);
    }
}
