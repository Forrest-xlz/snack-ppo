use ppo_snake::evaluation::EvaluationMetrics;
use ppo_snake::metrics::{EarlyStopper, SampleWindow, ScoreWindow};
use ppo_snake::ppo::PpoMetrics;
use ppo_snake::trainer::{
    advance_sample_threshold, eval_stop_metric_groups, evaluation_metric_groups,
    training_metric_groups,
};

#[test]
fn train_score_is_rolling_and_empty_uploads_keep_it() {
    let mut window = SampleWindow::default();
    window.push(8, &[2, 6], 1, PpoMetrics::default(), 2);
    window.push(8, &[], 0, PpoMetrics::default(), 2);
    let snapshot = window.snapshot();
    assert_eq!(snapshot.samples, 16);
    assert_eq!(snapshot.episode_count, 2);
    assert_eq!(snapshot.reverse_action_rate, 1.0 / 16.0);
    assert_eq!(snapshot.optimizer_steps, 4);
    let mut scores = ScoreWindow::new(1024);
    scores.extend([2, 6]);
    let logged = training_metric_groups(&snapshot, scores.snapshot(), 16, 4, 2.0, 8.0);
    assert_eq!(logged["train_samples/mean_episode_score"], 4.0);
    assert_eq!(logged["train_samples/completed_episode_count"], 2.0);
    assert!(!logged.keys().any(|key| key.contains("_iteration/")));
    window.clear();
    window.push(8, &[], 0, PpoMetrics::default(), 2);
    let logged = training_metric_groups(&window.snapshot(), scores.snapshot(), 24, 6, 3.0, 8.0);
    assert_eq!(logged["train_samples/mean_episode_score"], 4.0);
    assert_eq!(logged["train_samples/completed_episode_count"], 0.0);
}

#[test]
fn sample_window_weights_ppo_diagnostics_by_transition_count() {
    let mut window = SampleWindow::default();
    window.push(
        2,
        &[],
        0,
        PpoMetrics {
            raw_adv_mean: 1.0,
            ..Default::default()
        },
        1,
    );
    window.push(
        6,
        &[],
        0,
        PpoMetrics {
            raw_adv_mean: 3.0,
            ..Default::default()
        },
        3,
    );
    assert_eq!(window.snapshot().ppo.raw_adv_mean, 2.5);
    assert!((window.snapshot().ppo.raw_adv_std - 0.75f64.sqrt()).abs() < 1e-12);
}

#[test]
fn evaluation_metrics_only_use_sample_axis() {
    let eval = evaluation_metric_groups(
        EvaluationMetrics {
            mean_score: 3.0,
            std_score: 2.0,
            min_score: 1,
            max_score: 6,
        },
        8192,
    );
    assert_eq!(eval["eval_samples/std_score"], 2.0);
    assert!(!eval.keys().any(|key| key.contains("_iteration/")));
}

#[test]
fn sample_schedule_advances_past_large_rollouts_without_drifting() {
    assert_eq!(
        advance_sample_threshold(1_000_000, 1_000_000, 819_200),
        1_000_000
    );
    assert_eq!(
        advance_sample_threshold(1_000_000, 1_000_000, 1_064_960),
        2_000_000
    );
    assert_eq!(
        advance_sample_threshold(1_000_000, 1_000_000, 3_112_960),
        4_000_000
    );
}

#[test]
fn eval_stop_metrics_live_with_eval_and_have_no_window_group() {
    let mut stopper = EarlyStopper::new(2);
    stopper.observe(4.0);
    let update = stopper.observe(3.0);
    let logged = eval_stop_metric_groups(update, 10);
    assert_eq!(logged["eval_samples/best_mean_score"], 4.0);
    assert_eq!(logged["eval_samples/stale_evaluations"], 1.0);
    assert_eq!(logged["eval_samples/early_stop_triggered"], 0.0);
    assert!(!logged
        .keys()
        .any(|key| key.starts_with("early_stop_samples/")));
    let triggered = eval_stop_metric_groups(stopper.observe(2.0), 20);
    assert_eq!(triggered["eval_samples/stale_evaluations"], 2.0);
    assert_eq!(triggered["eval_samples/early_stop_triggered"], 1.0);
}
