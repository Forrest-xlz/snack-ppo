use ppo_snake::{
    config::EnvironmentConfig,
    model::ActorCritic,
    rollout::{collect_rollout, VectorEnv},
};
use tch::{nn, Device};

#[test]
fn vector_env_reports_reverse_requests_before_applying_action() {
    let config = EnvironmentConfig {
        width: 5,
        height: 5,
        num_envs: 1,
        rollout_steps: 1,
    };
    let mut envs = VectorEnv::new(&config, 1).unwrap();
    let transition = envs.step(&[2]).unwrap();
    assert!(transition[0].reversed_action);
}

#[test]
fn rollout_collects_environment_count_times_steps() {
    let config = EnvironmentConfig {
        width: 5,
        height: 5,
        num_envs: 3,
        rollout_steps: 7,
    };
    let mut envs = VectorEnv::new(&config, 11).unwrap();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::new(&store.root(), 75, 16);
    let rollout = collect_rollout(&model, &mut envs, 7, Device::Cpu).unwrap();
    assert_eq!(rollout.len(), 21);
    assert_eq!(rollout.actions.len(), 21);
    assert_eq!(rollout.rewards.len(), 21);
    assert_eq!(rollout.dones.len(), 21);
    assert_eq!(rollout.values.len(), 21);
    assert_eq!(rollout.log_probs.len(), 21);
}
