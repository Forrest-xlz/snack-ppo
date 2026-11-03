use tch::{nn, nn::Module, Kind, Tensor};

use crate::config::{Config, ModelVersion};

pub struct ActorCritic {
    shared: nn::Sequential,
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

pub struct PolicyOutput {
    pub actions: Tensor,
    pub log_probs: Tensor,
    pub values: Tensor,
}

pub struct EvaluationOutput {
    pub log_probs: Tensor,
    pub entropy: Tensor,
    pub values: Tensor,
}

impl ActorCritic {
    pub fn from_config(path: &nn::Path<'_>, config: &Config) -> Self {
        let channels = config.features.version.channels() as i64;
        let height = config.environment.height as i64;
        let width = config.environment.width as i64;
        match config.model.version {
            ModelVersion::MlpV1 => {
                Self::new(path, channels * height * width, config.model.hidden_size)
            }
            ModelVersion::CnnV1 => {
                Self::cnn(path, channels, height, width, config.model.hidden_size)
            }
        }
    }

    fn cnn(path: &nn::Path<'_>, channels: i64, height: i64, width: i64, hidden_size: i64) -> Self {
        let downsample = nn::ConvConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let preserve = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let flattened = 64 * ((height + 3) / 4) * ((width + 3) / 4);
        let shared = nn::seq()
            .add_fn(move |input| input.view([input.size()[0], channels, height, width]))
            .add(nn::conv2d(path / "conv1", channels, 16, 3, downsample))
            .add_fn(Tensor::relu)
            .add(nn::conv2d(path / "conv2", 16, 32, 3, downsample))
            .add_fn(Tensor::relu)
            .add(nn::conv2d(path / "conv3", 32, 64, 3, preserve))
            .add_fn(Tensor::relu)
            .add_fn(|input| input.flatten(1, -1))
            .add(nn::linear(
                path / "projection",
                flattened,
                hidden_size,
                Default::default(),
            ))
            .add_fn(Tensor::relu);
        Self {
            shared,
            policy_head: nn::linear(path / "policy", hidden_size, 4, Default::default()),
            value_head: nn::linear(path / "value", hidden_size, 1, Default::default()),
        }
    }

    pub fn new(path: &nn::Path<'_>, input_size: i64, hidden_size: i64) -> Self {
        let shared = nn::seq()
            .add(nn::linear(
                path / "fc1",
                input_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(Tensor::relu)
            .add(nn::linear(
                path / "fc2",
                hidden_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(Tensor::relu);
        let policy_head = nn::linear(path / "policy", hidden_size, 4, Default::default());
        let value_head = nn::linear(path / "value", hidden_size, 1, Default::default());
        Self {
            shared,
            policy_head,
            value_head,
        }
    }

    pub fn forward(&self, observations: &Tensor) -> (Tensor, Tensor) {
        let features = self.shared.forward(observations);
        let logits = features.apply(&self.policy_head);
        let values = features.apply(&self.value_head).squeeze_dim(-1);
        (logits, values)
    }

    pub fn actions_and_values(&self, observations: &Tensor, deterministic: bool) -> PolicyOutput {
        let (logits, values) = self.forward(observations);
        let log_probs_all = logits.log_softmax(-1, Kind::Float);
        let actions = if deterministic {
            logits.argmax(-1, false)
        } else {
            logits
                .softmax(-1, Kind::Float)
                .multinomial(1, true)
                .squeeze_dim(-1)
        };
        let log_probs = log_probs_all
            .gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1);
        PolicyOutput {
            actions,
            log_probs,
            values,
        }
    }

    pub fn evaluate_actions(&self, observations: &Tensor, actions: &Tensor) -> EvaluationOutput {
        let (logits, values) = self.forward(observations);
        let log_probs_all = logits.log_softmax(-1, Kind::Float);
        let probabilities = logits.softmax(-1, Kind::Float);
        let log_probs = log_probs_all
            .gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1);
        let entropy = -(probabilities * log_probs_all).sum_dim_intlist(
            [-1i64].as_slice(),
            false,
            Kind::Float,
        );
        EvaluationOutput {
            log_probs,
            entropy,
            values,
        }
    }
}
