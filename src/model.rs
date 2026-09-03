use tch::{nn, nn::Module, Kind, Tensor};

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
