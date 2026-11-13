"""Build the English experiment report from the reviewed analysis tables."""

import json
from pathlib import Path

import pandas as pd

from readme_sections import blocks as readme_blocks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/analysis/data"
results = pd.read_csv(DATA / "run_results.csv")
configs = json.loads((DATA / "run_configs.json").read_text(encoding="utf-8"))
inventory = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))
evaluations = pd.read_csv(DATA / "eval_history.csv")
lookup = {(r.study, r.variant): r for r in results.itertuples(index=False)}


def number(value, decimals=2):
    return "—" if value is None or pd.isna(value) else f"{value:.{decimals}f}"


def millions(value):
    return number(value / 1e6, 0)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def at(row, nominal):
    frame = evaluations[evaluations.run_id == row.run_id]
    nearest = frame.iloc[(frame.total_samples - nominal * 1e6).abs().argmin()]
    assert abs(nearest.total_samples - nominal * 1e6) <= row.rollout_samples + 1
    return float(nearest["eval_samples/mean_score"])


base = {key: lookup[("E001", key)] for key in "ABCDEF"}

root_report = (ROOT / "docs/analysis/readme-template.md").read_text(encoding="utf-8")
for key, value in readme_blocks(results, lookup, configs, at, number, millions, table).items():
    root_report = root_report.replace("{{" + key + "}}", value)
assert "{{" not in root_report, "Unresolved README template block"

def run_interpretation(row):
    if row.study == "E001":
        notes = {
            "A":"The small-scale baseline stays well below B/C's late score range. It stops after 12 evals without a new high; this does not establish an infinite-sample ceiling.",
            "B":"At fixed R=8,192, this larger mini-batch outperforms D over much of training. It also shows transient eval drops, so its best point should not stand in for the full trajectory.",
            "C":"The largest joint scale has the highest observed E001 peak and late-four mean. It reaches the sample cap, with some regression after its peak; no asymptotic ceiling is measured.",
            "D":"At fixed R=8,192 this b=256 run is much weaker than B. Increasing rollout relative to A does not produce a higher observed peak.",
            "E":"At fixed R=81,920 it trails C through much of the curve. Its last single eval rebounds to 98.21, but the last-four mean is only 84.71, illustrating endpoint sensitivity.",
            "F":"Compared with C at the same b=10,240 and 1,200M budget, the smaller rollout has lower last and last-four eval scores. This rollout benefit does not generalize to the b=256/1,024 comparisons.",
        }
        return notes[row.variant]
    if row.study == "E002":
        parent = base[row.variant[0]]
        delta = row.eval_at_300M - parent.eval_at_300M
        text = f"At the common 300M budget this is {abs(delta):.2f} points {'above' if delta>0 else 'below'} the epoch-1 {row.variant[0]} baseline. "
        if row.variant in ["A-ep8","B-ep8"]:
            text += "The last eval is near 1: a severe regression under this unchanged-LR, high-reuse configuration."
        elif row.variant == "A-ep2":
            text += "The early advantage does not survive: both eval and training score fall sharply around 700–750M, followed by partial recovery. See the diagnostic case study in the main README."
        elif row.variant in ["C-ep2","C-ep4"]:
            text += "Moderate reuse remains competitive at the large scale, but last, tail-average and best-point rankings differ; no clear general superiority over epoch 1 follows."
        elif row.variant == "C-ep8":
            text += "The large scale delays rather than eliminates degradation at eight epochs; the trajectory ends substantially below its peak."
        else:
            text += "It stops before the corresponding epoch-1 run and does not demonstrate a sustained improvement. More epochs also incur more optimizer work per sample."
        return text
    if row.study == "E003":
        text = f"After +300M the score is {row.eval_300M_after_switch:.2f}, versus {row.initial_eval:.2f} at the switch. "
        if row.variant.startswith("A-"):
            text += "The branch improves, but the observed continuation does not reach the destination-from-scratch late range. This is finite-budget path dependence, with Adam reset and no same-scale reset control."
        elif row.variant.startswith("B-"):
            text += "The branch reaches the large-scale score range. Its pretraining cost must be included before comparing sample efficiency against C from scratch."
        elif row.variant == "C-450M-to-B":
            text += "No later eval exceeds the initial 100.13; the branch stops after the patience window. Lowering the scale does not improve this particular strong starting policy."
        else:
            text += "A strong policy can retain or improve high scores after lowering the scale, contradicting a simple rule that the current batch alone fixes attainable performance."
        return text
    return {
        "A":"This CNN does not improve on the MLP at the matched 1,200M endpoint. The result concerns this architecture and fixed training protocol, not all convolutional policies.",
        "B":"This is the historical transformer_v3 seven-feature grid experiment, not today's small B. Its score remains low and volatile. It ends before its configured cap without a recorded early-stop trigger, so it should not be called converged.",
        "B-v7":"The entity small model exceeds the MLP's observed best score, then regresses by 53.93 points from its peak to the last eval. Its entropy is not monotonically vanishing during that late regression. The current name for this architecture is transformer_v1_small.",
        "C":"The base entity model exceeds the MLP's range early and has the highest observed mean in the export. W&B marks the run crashed at 419M, with last eval at 400M. After that eval the training rolling score falls to 18.49 at 409M and recovers to 129.32 by 419M; no later eval assesses that event. Interruption cause and later stability are unknown.",
        "D":"At the common 575M point it scores 120.82 versus 71.62 without the teacher, then continues to high scores. It still has large downward excursions, and the first pre-KL evaluation differs between runs. The evidence supports a promising direction, not clean attribution or a completed-run stability claim.",
    }[row.variant]


catalog = ["# Per-run experiment catalog", "", "Snapshot downloaded 2026-11-09. Each section identifies the actual logged config, shows both evaluation and training scores, and states the supported interpretation. All runs use seed 42. Raw eval means are the primary performance measure; training scores are stochastic rolling-game metrics. The common protocol, single-seed limitations and diagnostic definitions are in the [main report](../../README.md) and [methods](methods.md).", "", "The 32 histories are all included. C/D's `crashed` status is retained; no failure cause is invented. Best evals are retrospective diagnostics. Last-four means describe each run's own tail, not a common training budget.", ""]
for row in results.itertuples(index=False):
    config = configs[row.run_id]["config"]
    catalog += [f"## {row.study} {row.variant} — `{row.run_id}`", "", f"[Open W&B run]({row.url}) · actual version **`{row.model_version}`** · **{row.end_reason}**.", "", f"**Experiment:** b={row.batch_size:,}, R={row.rollout_samples:,}, {row.num_envs:,} environments × {row.rollout_steps} steps, {row.epochs} PPO epoch(s), LR={row.learning_rate:g}, device `{row.device}`, hidden readout {config['model']['hidden_size']}.", "", f"Observed logged training time: **{row.logged_wall_time_seconds/3600:.2f} h**; optimizer steps: **{row.optimizer_steps_total:,}**. Job times are not matched hardware benchmarks.", ""]
    if row.study == "E003":
        catalog += [f"Initialized from `{config['training']['init_checkpoint']}` after {millions(row.pretrained_samples)}M prefix samples; weights-only loading resets Adam. Training can add up to 1,200M samples, subject to eval early stopping.", ""]
    if row.study == "E004" and row.variant == "D":
        catalog += ["Frozen self-teacher initialized at first eval; strictly higher eval means replace it. Forward KL teacher→student, coefficient 0.001, no critic distillation or coefficient decay.", ""]
    catalog += [table(["Last training M", "Last eval (at M)", "Best eval (at M)", "Last-four mean", "Eval count"],[[millions(row.last_logged_samples), f"{row.last_eval:.2f} ({millions(row.last_eval_samples)})", f"{row.best_eval:.2f} ({millions(row.best_eval_samples)})",number(row.last4_eval_mean),row.eval_count]]), "", f"![{row.name}: evaluation and training scores](figures/runs/{row.run_id}.png)", "", "**Interpretation:** " + run_interpretation(row), ""]

methods = f"""# Sources, calculations and validation

## Coverage and source

The read-only export covers all **{len(results)} runs** returned by `xlz1014361885-tokyo/ppo-snake`, with **{sum(r['history_rows'] for r in inventory['runs']):,} history rows**. Export completion: **{inventory['export_finished_at_utc']}**. There are no failed metric exports. This is a time-bounded snapshot, not a live dashboard. Run configs/summary/history were retrieved together; interrupted runs are not completed by extrapolation.

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
"""

(ROOT / "README.md").write_text(root_report, encoding="utf-8")
(ROOT / "docs/analysis/run-catalog.md").write_text("\n".join(catalog), encoding="utf-8")
(ROOT / "docs/analysis/methods.md").write_text(methods, encoding="utf-8")
print(f"Wrote README, methods and catalog for {len(results)} runs")
