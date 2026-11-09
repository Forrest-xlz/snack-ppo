use std::path::Path;

use anyhow::Result;
use tch::{nn, Device, Kind, Tensor};

use crate::{config::Config, kickstarting::TeacherPolicy};

/// Forward KL of full action distributions; only the student receives gradients.
pub fn policy_kl(teacher_probabilities: &Tensor, student_log_probs: &Tensor) -> Tensor {
    let teacher = teacher_probabilities.detach();
    (&teacher * (teacher.clamp_min(1e-30).log() - student_log_probs)).sum_dim_intlist(
        [-1i64].as_slice(),
        false,
        Kind::Float,
    )
}

pub struct EvalTeacher {
    policy: Option<TeacherPolicy>,
    coef: f64,
    best_mean_score: Option<f64>,
    source_samples: Option<usize>,
}

impl EvalTeacher {
    /// Construct before manual_seed / student initialization. The random
    /// placeholder weights are never used; the first eval copies the student.
    pub fn new(config: &Config, device: Device) -> Result<Self> {
        let coef = config
            .teacher_kl
            .as_ref()
            .filter(|cfg| cfg.enabled)
            .map_or(0.0, |cfg| cfg.coef);
        Ok(Self {
            policy: (coef > 0.0).then(|| TeacherPolicy::empty_snapshot(config, device)),
            coef,
            best_mean_score: None,
            source_samples: None,
        })
    }

    pub fn policy(&self) -> Option<&TeacherPolicy> {
        self.best_mean_score.and(self.policy.as_ref())
    }

    pub fn coef(&self) -> f64 {
        if self.policy().is_some() {
            self.coef
        } else {
            0.0
        }
    }

    pub fn enabled(&self) -> bool {
        self.policy.is_some()
    }

    pub fn best_mean_score(&self) -> Option<f64> {
        self.best_mean_score
    }

    pub fn source_samples(&self) -> Option<usize> {
        self.source_samples
    }

    pub fn observe(
        &mut self,
        mean_score: f64,
        samples: usize,
        student: &nn::VarStore,
    ) -> Result<bool> {
        anyhow::ensure!(
            mean_score.is_finite(),
            "teacher eval mean score must be finite"
        );
        let Some(policy) = &mut self.policy else {
            return Ok(false);
        };
        if self.best_mean_score.is_some_and(|best| mean_score <= best) {
            return Ok(false);
        }
        policy.copy_weights(student)?;
        self.best_mean_score = Some(mean_score);
        self.source_samples = Some(samples);
        Ok(true)
    }

    pub fn save(&self, directory: &Path) -> Result<()> {
        let Some(policy) = self.policy() else {
            return Ok(());
        };
        std::fs::create_dir_all(directory)?;
        let temporary = directory.join(".teacher.tmp.ot");
        let target = directory.join("teacher.ot");
        policy.save(&temporary)?;
        if target.exists() {
            std::fs::remove_file(&target)?;
        }
        std::fs::rename(temporary, target)?;
        Ok(())
    }
}
