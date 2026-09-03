use ppo_snake::model::ActorCritic;
use tch::{nn, Device, Kind, Tensor};

#[test]
fn policy_outputs_have_batch_shapes_and_are_finite() {
    tch::manual_seed(7);
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::new(&store.root(), 75, 32);
    let observations = Tensor::zeros([3, 75], (Kind::Float, Device::Cpu));
    let output = model.actions_and_values(&observations, false);
    assert_eq!(output.actions.size(), [3]);
    assert_eq!(output.log_probs.size(), [3]);
    assert_eq!(output.values.size(), [3]);
    assert_eq!(output.log_probs.isfinite().all().int64_value(&[]), 1);
}

#[test]
fn deterministic_actions_can_be_evaluated_again() {
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::new(&store.root(), 75, 16);
    let observations = Tensor::randn([2, 75], (Kind::Float, Device::Cpu));
    let output = model.actions_and_values(&observations, true);
    let evaluation = model.evaluate_actions(&observations, &output.actions);
    assert_eq!(evaluation.log_probs.size(), [2]);
    assert_eq!(evaluation.entropy.size(), [2]);
    assert_eq!(evaluation.values.size(), [2]);
}
