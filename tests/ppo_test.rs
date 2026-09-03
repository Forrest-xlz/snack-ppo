use ppo_snake::ppo::{
    approximate_kl, compute_gae, explained_variance, shuffled_minibatches, GaeParams,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn compute_gae_masks_terminal_bootstrap() {
    let (advantages, returns) = compute_gae(
        &[1.0],
        &[true],
        &[0.25],
        &[10.0],
        GaeParams {
            steps: 1,
            envs: 1,
            gamma: 0.99,
            gae_lambda: 0.95,
        },
    );
    assert!((advantages[0] - 0.75).abs() < 1e-5);
    assert!((returns[0] - 1.0).abs() < 1e-5);
}

#[test]
fn compute_gae_uses_non_terminal_bootstrap() {
    let (advantages, returns) = compute_gae(
        &[0.0],
        &[false],
        &[0.5],
        &[2.0],
        GaeParams {
            steps: 1,
            envs: 1,
            gamma: 0.9,
            gae_lambda: 0.95,
        },
    );
    assert!((advantages[0] - 1.3).abs() < 1e-5);
    assert!((returns[0] - 1.8).abs() < 1e-5);
}

#[test]
fn minibatches_cover_tail_without_duplicates() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let batches = shuffled_minibatches(10, 4, &mut rng);
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [4, 4, 2]);
    let mut indices = batches.into_iter().flatten().collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(indices, (0..10).collect::<Vec<_>>());
}

#[test]
fn critic_explained_variance_matches_hand_calculation() {
    let returns = [1.0, 2.0, 3.0];
    let predictions = [1.0, 1.0, 2.0];
    assert!((explained_variance(&returns, &predictions) - (2.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn action_head_approximate_kl_is_zero_for_unchanged_policy() {
    assert!(approximate_kl(&[-0.7, -1.2], &[-0.7, -1.2]).abs() < 1e-8);
}

#[test]
fn action_head_approximate_kl_matches_ppo_formula() {
    let old = [-1.0];
    let new = [-0.8];
    let expected = (-0.2f64).exp() - 1.0 + 0.2;
    assert!((approximate_kl(&old, &new) - expected).abs() < 1e-8);
}
