# Sources, calculations and validation

## Coverage and source

The read-only export covers all **32 runs** returned by `xlz1014361885-tokyo/ppo-snake`, with **26,717 history rows**. Export completion: **2026-11-09T03:58:21.216403+00:00**. There are no failed metric exports. This is a time-bounded snapshot, not a live dashboard. Run configs/summary/history were retrieved together; interrupted runs are not completed by extrapolation.

The [inventory](data/inventory.json) retains run URLs, states, row counts, per-run export timestamps and history SHA-256 hashes. [run_configs.json](data/run_configs.json) contains the actual W&B configurations. Local full archives and summaries remain in `analysis/wandb/raw/<run_id>/`; they are Git-ignored, while the report's portable derived evidence is retained here. No model checkpoints were downloaded. Attempts to obtain `output.log` for the two crashed runs returned 404, so no crash cause is established.

## Definitions

- Primary score: `eval_samples/mean_score`, deterministic actions, 1,024 fixed held-out environment seeds, 5,000 steps per episode. All archived configs share seed 42, a 12×12 board, eval interval 25M and these eval settings.
- Stochastic training score: `train_samples/mean_episode_score`, the latest 1,024 completed training games. It is not an on-the-fly estimate of the deterministic policy's held-out score.
- Last eval: last nonmissing evaluation row by sample count. Its clock can precede the last training row.
- Best eval: maximum historical eval mean. Ties use the earliest observed maximum. This is an optimistically selected trajectory statistic, not an independent test score.
- Last-four mean: arithmetic mean of the final four eval means. These share training history and evaluation seeds; they are not independent seed repetitions or a confidence interval.
- Common-budget score: recorded eval nearest the nominal target, accepted only within **one actual rollout R** of it. No interpolation, extrapolation, zero-filling or terminal-value carry-forward is used. Missing common-budget points remain missing.
- E003 total clock: `total_samples = pretrained_env_samples + finetune_samples`. The +300M comparison targets the exact prefix count plus 300M; the displayed nominal switch names round actual checkpoint counts.
- R: `num_envs × rollout_steps`; optimizer steps: `(total_samples - prefix) × epochs / b` for these divisible rollouts. Epochs change training compute per environment sample.
- Fixed coefficient: D uses forward KL of full teacher/student action distributions with coefficient 0.001. Its logged `teacher_kl_penalty` already includes that coefficient. `teacher_student_kl` belongs to the separate fixed-expert kickstarting path and is zero here; it is not D's teacher loss.
- Finished early stop is identified from the recorded `eval_samples/early_stop_triggered` flag. Sample-cap termination is identified against the run's own configured new-sample cap. Other finished runs are labeled ended-before-cap. W&B `crashed` is retained as partial even when the final observed score is high.

## Smoothing and plots

Eval curves are never smoothed. For training diagnostics, the causal sample-distance EMA uses

`alpha_i = 1 - 2^(-(S_i-S_(i-1))/H)`

`EMA_i = EMA_(i-1) + alpha_i * (value_i - EMA_(i-1))`.

The first available value initializes the EMA. H=25M for group diagnostics, H=5M for the A-epoch-2 collapse window and H=10M for per-run training scores. This uses environment-sample distance rather than number of W&B uploads; a larger rollout does not accidentally change the smoothing scale. Raw upload values remain faintly visible. E001 entropy/value-loss/approx-KL diagnostics use log axes. The collapse plot's diagnostic y ranges use the displayed 600–850M window. Performance plots share the 0–145 display range, with theoretical score maximum 141.

No seed confidence intervals are drawn. Logged `eval_samples/std_score` is dispersion across games, not uncertainty across training seeds. Per-episode score arrays were not logged, preventing paired episode-difference intervals, win rates, score quantiles or robust aggregate estimates requiring those arrays. The same eval seeds also guide early stopping and teacher replacement, so they form a validation set for selection decisions rather than a pristine final test set.

## Audit checks and independent spot checks

The [validation record](data/validation.json) checks every archive's hash/row count, unique history steps, monotonic sample clock, score bounds, final history coverage versus W&B summary, E003 clock arithmetic and optimizer-step accounting. D's recorded teacher scores and replacement flags agree with strict cumulative eval highs. All 32 histories passed. These checks establish coverage/arithmetic, not statistical certainty or causal identification.

The following values were independently recomputed directly from the compressed raw histories rather than copied from the report table:

- E001 C final score **103.2373046875** at **1,200,046,080** samples; best **110.6455078125** at **1,050,050,560**.
- E002 A-ep2 eval **48.0283203125** at **700,000,256**, falling to **13.994140625** at **750,000,128** and recovering partly to **41.91015625** at **800,000,000**.
- E004 B-v7 best **125.5458984375** at **275,005,440**, last **71.6181640625** at **575,078,400**; D at that same clock **120.8212890625**, a **49.203125**-point difference.
- E004 C last eval **132.935546875** at **400,015,360**; its last training upload is **419,020,800**. Neither number is substituted for the other.
- D's first eval **20.1943359375** differs from B-v7's **14.9384765625** before the teacher penalty starts. This materially limits attribution of the full observed difference to KL.

Diagnostic-window prose uses means of the raw logged upload aggregates, not smoothed figure values. These windows have roughly equal 1M sample uploads. Their estimates describe the respective policies' visited-state distributions; value losses do not measure identical prediction problems across different policies. [training_diagnostics.csv.gz](data/training_diagnostics.csv.gz) preserves the values used to draw these diagnostics.

Implementation detail: despite the config name `max_grad_norm`, the code calls tch's `backward_step_clip`, which applies **per-element gradient-value clipping** at 0.5, not global L2-norm clipping. This was checked in the installed tch 0.24 source (`nn/optimizer.rs`, `backward_step_clip` calls `clip_grad_value`). The report preserves this implementation and does not silently reinterpret the results as a different PPO variant. No training implementation was changed while producing this report. Feature/model transitions, CUDA vs CPU, fixed learning rate, number of optimizer steps, adaptive stopping and missing reset-only E003 controls all limit broader causal claims.

## Reproduction

1. Run [export_wandb_runs.py](../../scripts/export_wandb_runs.py) with existing W&B credentials to retrieve unsampled histories.
2. Run [analyze_wandb_runs.py](../../scripts/analyze_wandb_runs.py) using the `orbit_wars` Python environment to validate data and regenerate portable tables and PNG figures.
3. Run [build_experiment_readme.py](../../scripts/build_experiment_readme.py) in that environment to rebuild the English README, catalog and this methods file from reviewed tables and narrative. The narrative is specific to this snapshot and must be reviewed before using a later export.

No network configuration, W&B run, training configuration, checkpoint, or model weight was changed by this analysis. The only external operations were authenticated reads from W&B and reads of the methodological references.

## Evaluation references

[Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315) discusses experimental design, variation and comparison pitfalls. [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) emphasizes implementation/reproducibility effects. [Rliable's visual explanation](https://araffin.github.io/post/rliable/) illustrates uncertainty across runs/tasks; this one-task, one-training-seed dataset cannot supply those independent units by treating evaluation episodes as training repetitions.
