use ppo_snake::metrics::{EarlyStopper, ScoreWindow};

#[test]
fn rolling_train_score_uses_latest_completed_episodes() {
    let mut scores = ScoreWindow::new(3);
    assert!(scores.snapshot().is_none());
    scores.extend([1, 3, 5]);
    scores.extend([7]);
    let snapshot = scores.snapshot().unwrap();
    assert_eq!(snapshot.count, 3);
    assert_eq!(snapshot.mean, 5.0);
    assert_eq!(snapshot.min, 3);
    assert_eq!(snapshot.max, 7);
    scores.extend([]);
    assert_eq!(scores.snapshot().unwrap().mean, 5.0);
}

#[test]
fn early_stop_counts_only_evaluations_and_resets_after_a_new_high() {
    let mut stop = EarlyStopper::new(3);
    let first = stop.observe(10.0);
    assert!(first.improved);
    assert!(!first.should_stop);
    assert_eq!(stop.observe(9.0).stale_evaluations, 1);
    assert_eq!(stop.observe(10.0).stale_evaluations, 2);
    let rebound = stop.observe(11.0);
    assert!(rebound.improved);
    assert_eq!(rebound.stale_evaluations, 0);
    assert_eq!(rebound.best_mean_score, 11.0);
    assert_eq!(stop.observe(11.0).stale_evaluations, 1);
    assert_eq!(stop.observe(10.0).stale_evaluations, 2);
    let third_stale_eval = stop.observe(9.0);
    assert!(third_stale_eval.should_stop);
    assert_eq!(third_stale_eval.stale_evaluations, 3);
}
