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
fn early_stop_requires_patience_after_last_window_record() {
    let mut stop = EarlyStopper::new(10, 2, 2);
    assert!(stop.observe(5, 5, &[2]).is_none());
    let first = stop.observe(10, 5, &[2]).unwrap();
    assert_eq!(first.mean_score, Some(2.0));
    assert!(first.improved);
    assert!(!first.should_stop);
    assert_eq!(stop.observe(20, 10, &[1, 3]).unwrap().stale_windows, 1);
    let rebound = stop.observe(30, 10, &[4, 4]).unwrap();
    assert!(rebound.improved);
    assert_eq!(rebound.stale_windows, 0);
    assert_eq!(stop.observe(40, 10, &[4, 4]).unwrap().stale_windows, 1);
    let final_window = stop.observe(50, 10, &[3, 3]).unwrap();
    assert!(final_window.should_stop);
    assert_eq!(final_window.stale_windows, 2);
}

#[test]
fn early_stop_does_not_count_windows_with_too_few_episodes() {
    let mut stop = EarlyStopper::new(10, 2, 2);
    let empty = stop.observe(10, 10, &[]).unwrap();
    assert_eq!(empty.mean_score, None);
    assert_eq!(empty.stale_windows, 0);
    assert_eq!(
        stop.observe(20, 10, &[3, 5]).unwrap().best_mean_score,
        Some(4.0)
    );
    let thin = stop.observe(30, 10, &[0]).unwrap();
    assert_eq!(thin.mean_score, None);
    assert_eq!(thin.stale_windows, 0);
}
