use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::features::FeatureEncoding;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Train,
    Infer,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceConfig {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub mode: Mode,
    pub device: DeviceConfig,
    pub seed: u64,
    pub environment: EnvironmentConfig,
    pub model: ModelConfig,
    pub ppo: PpoConfig,
    pub training: TrainingConfig,
    pub inference: InferenceConfig,
    pub evaluation: EvaluationConfig,
    pub metrics: MetricsConfig,
    pub early_stopping: EarlyStoppingConfig,
    pub wandb: WandbConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kickstarting: Option<KickstartingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_kl: Option<TeacherKlConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherKlConfig {
    pub enabled: bool,
    #[serde(default = "default_teacher_kl_coef")]
    pub coef: f64,
}

fn default_teacher_kl_coef() -> f64 {
    0.001
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KickstartingConfig {
    pub teacher_checkpoint: PathBuf,
    pub teacher_model: ModelConfig,
    pub initial_weight: f64,
    pub decay_samples: usize,
}

impl KickstartingConfig {
    pub fn weight_at(&self, student_samples: usize) -> f64 {
        self.initial_weight
            * (1.0 - student_samples as f64 / self.decay_samples.max(1) as f64).max(0.0)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentConfig {
    pub width: usize,
    pub height: usize,
    pub num_envs: usize,
    pub rollout_steps: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[serde(default)]
    pub version: ModelVersion,
    pub hidden_size: i64,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
pub enum ModelVersion {
    #[default]
    #[serde(rename = "mlp_v1")]
    MlpV1,
    #[serde(rename = "cnn_v1")]
    CnnV1,
    #[serde(rename = "transformer_v1_base")]
    TransformerV1Base,
    #[serde(rename = "transformer_v1_small")]
    TransformerV1Small,
}

impl ModelVersion {
    pub fn feature_encoding(self) -> FeatureEncoding {
        match self {
            Self::MlpV1 | Self::CnnV1 => FeatureEncoding::HeadBodyFood,
            Self::TransformerV1Base | Self::TransformerV1Small => FeatureEncoding::EntityRegions,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PpoConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_range: f64,
    pub value_coef: f64,
    pub entropy_coef: f64,
    pub max_grad_norm: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub max_env_samples: usize,
    pub checkpoint_interval_samples: usize,
    pub checkpoint_dir: PathBuf,
    /// Optional weights-only initialization; Adam is always created fresh.
    #[serde(default)]
    pub init_checkpoint: Option<PathBuf>,
    /// Samples already used to train the initialization checkpoint.
    #[serde(default)]
    pub pretrained_env_samples: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceConfig {
    pub model_path: PathBuf,
    pub fps: u32,
    pub deterministic: bool,
    pub cell_size: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationConfig {
    pub interval_samples: usize,
    pub episodes: usize,
    #[serde(default = "default_evaluation_batch_size")]
    pub batch_size: usize,
    pub deterministic: bool,
    pub max_steps_per_episode: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    pub score_window_episodes: usize,
    #[serde(default = "default_log_interval_samples")]
    pub log_interval_samples: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience_evaluations: usize,
}

fn default_log_interval_samples() -> usize {
    1_000_000
}

fn default_evaluation_batch_size() -> usize {
    1024
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WandbMode {
    Online,
    Offline,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WandbConfig {
    pub enabled: bool,
    pub project: String,
    pub entity: Option<String>,
    pub group: Option<String>,
    pub run_name: Option<String>,
    pub mode: WandbMode,
    pub python_executable: PathBuf,
}

#[derive(Debug, Error)]
#[error("invalid configuration field {field}: {message}")]
pub struct ConfigError {
    field: &'static str,
    message: &'static str,
}

impl ConfigError {
    fn new(field: &'static str, message: &'static str) -> Self {
        Self { field, message }
    }
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let yaml = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = serde_yaml::from_str(&yaml)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> std::result::Result<(), ConfigError> {
        positive_usize("environment.width", self.environment.width)?;
        if self.environment.width < 3 {
            return Err(ConfigError::new(
                "environment.width",
                "must be at least 3 for the initial snake",
            ));
        }
        positive_usize("environment.height", self.environment.height)?;
        positive_usize("environment.num_envs", self.environment.num_envs)?;
        positive_usize("environment.rollout_steps", self.environment.rollout_steps)?;
        if self.model.hidden_size <= 0 {
            return Err(ConfigError::new("model.hidden_size", "must be positive"));
        }
        positive_f64("ppo.learning_rate", self.ppo.learning_rate)?;
        positive_usize("ppo.batch_size", self.ppo.batch_size)?;
        positive_usize("ppo.epochs", self.ppo.epochs)?;
        unit_interval_f32("ppo.gamma", self.ppo.gamma)?;
        unit_interval_f32("ppo.gae_lambda", self.ppo.gae_lambda)?;
        if !(0.0..=1.0).contains(&self.ppo.clip_range) || self.ppo.clip_range == 0.0 {
            return Err(ConfigError::new(
                "ppo.clip_range",
                "must be in the interval (0, 1]",
            ));
        }
        non_negative("ppo.value_coef", self.ppo.value_coef)?;
        non_negative("ppo.entropy_coef", self.ppo.entropy_coef)?;
        positive_f64("ppo.max_grad_norm", self.ppo.max_grad_norm)?;
        if let Some(teacher) = &self.teacher_kl {
            non_negative("teacher_kl.coef", teacher.coef)?;
            if teacher.enabled && teacher.coef > 0.0 && self.kickstarting.is_some() {
                return Err(ConfigError::new(
                    "teacher_kl",
                    "cannot enable evaluation-best teacher KL together with kickstarting",
                ));
            }
        }
        if let Some(kick) = &self.kickstarting {
            non_negative("kickstarting.initial_weight", kick.initial_weight)?;
            positive_usize("kickstarting.decay_samples", kick.decay_samples)?;
            if kick.teacher_checkpoint.as_os_str().is_empty() {
                return Err(ConfigError::new(
                    "kickstarting.teacher_checkpoint",
                    "must not be empty",
                ));
            }
            if kick.teacher_model.hidden_size <= 0 {
                return Err(ConfigError::new(
                    "kickstarting.teacher_model.hidden_size",
                    "must be positive",
                ));
            }
            if !kick
                .teacher_model
                .version
                .feature_encoding()
                .is_prefix_of(self.model.version.feature_encoding())
            {
                return Err(ConfigError::new(
                    "kickstarting.teacher_model.version",
                    "teacher features must be a prefix of the student features",
                ));
            }
        }
        positive_usize("training.max_env_samples", self.training.max_env_samples)?;
        if self.training.pretrained_env_samples > 0 && self.training.init_checkpoint.is_none() {
            return Err(ConfigError::new(
                "training.init_checkpoint",
                "is required when pretrained_env_samples is nonzero",
            ));
        }
        if self
            .training
            .init_checkpoint
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(ConfigError::new(
                "training.init_checkpoint",
                "must not be empty",
            ));
        }
        if self
            .training
            .pretrained_env_samples
            .checked_add(self.training.max_env_samples)
            .is_none()
        {
            return Err(ConfigError::new(
                "training.pretrained_env_samples",
                "plus max_env_samples must fit in the sample counter",
            ));
        }
        positive_usize(
            "training.checkpoint_interval_samples",
            self.training.checkpoint_interval_samples,
        )?;
        if self.training.checkpoint_dir.as_os_str().is_empty() {
            return Err(ConfigError::new(
                "training.checkpoint_dir",
                "must not be empty",
            ));
        }
        if self.inference.model_path.as_os_str().is_empty() {
            return Err(ConfigError::new(
                "inference.model_path",
                "must not be empty",
            ));
        }
        if self.inference.fps == 0 {
            return Err(ConfigError::new("inference.fps", "must be positive"));
        }
        if self.inference.cell_size == 0 {
            return Err(ConfigError::new("inference.cell_size", "must be positive"));
        }
        positive_usize(
            "evaluation.interval_samples",
            self.evaluation.interval_samples,
        )?;
        positive_usize("evaluation.episodes", self.evaluation.episodes)?;
        positive_usize("evaluation.batch_size", self.evaluation.batch_size)?;
        positive_usize(
            "evaluation.max_steps_per_episode",
            self.evaluation.max_steps_per_episode,
        )?;
        positive_usize(
            "metrics.score_window_episodes",
            self.metrics.score_window_episodes,
        )?;
        positive_usize(
            "metrics.log_interval_samples",
            self.metrics.log_interval_samples,
        )?;
        positive_usize(
            "early_stopping.patience_evaluations",
            self.early_stopping.patience_evaluations,
        )?;
        if self.wandb.enabled && self.wandb.project.trim().is_empty() {
            return Err(ConfigError::new(
                "wandb.project",
                "must not be empty when enabled",
            ));
        }
        if self.wandb.enabled && self.wandb.python_executable.as_os_str().is_empty() {
            return Err(ConfigError::new(
                "wandb.python_executable",
                "must not be empty when enabled",
            ));
        }
        Ok(())
    }
}

fn positive_usize(field: &'static str, value: usize) -> Result<(), ConfigError> {
    if value == 0 {
        Err(ConfigError::new(field, "must be positive"))
    } else {
        Ok(())
    }
}

fn positive_f64(field: &'static str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        Err(ConfigError::new(field, "must be finite and positive"))
    } else {
        Ok(())
    }
}

fn non_negative(field: &'static str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value < 0.0 {
        Err(ConfigError::new(field, "must be finite and non-negative"))
    } else {
        Ok(())
    }
}

fn unit_interval_f32(field: &'static str, value: f32) -> Result<(), ConfigError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(ConfigError::new(field, "must be in the interval [0, 1]"))
    } else {
        Ok(())
    }
}
