use ppo_snake::{config::Config, evaluation::evaluate_policy, model::ActorCritic};
use tch::{nn, Device};

#[test]
fn fixed_seed_evaluation_is_reproducible_and_reports_score_dispersion() {
    let config: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::new(&store.root(), 3 * 5 * 5, 16);
    let first = evaluate_policy(&model, &config, Device::Cpu, 123).unwrap();
    let second = evaluate_policy(&model, &config, Device::Cpu, 123).unwrap();
    assert_eq!(first.mean_score, second.mean_score);
    assert_eq!(first.std_score, second.std_score);
    assert_eq!(first.min_score, second.min_score);
    assert_eq!(first.max_score, second.max_score);
    assert!(first.std_score.is_finite());
}
