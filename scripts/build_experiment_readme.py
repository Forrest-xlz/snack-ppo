"""Build the English experiment report from the reviewed analysis tables."""

import json
from pathlib import Path

import pandas as pd


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


def link(row):
    return f"[{row.variant}]({row.url})"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def at(row, nominal):
    frame = evaluations[evaluations.run_id == row.run_id]
    nearest = frame.iloc[(frame.total_samples - nominal * 1e6).abs().argmin()]
    assert abs(nearest.total_samples - nominal * 1e6) <= row.rollout_samples + 1
    return float(nearest["eval_samples/mean_score"])


base = {key: lookup[("E001", key)] for key in "ABCDEF"}
small = lookup[("E004", "B-v7")]
large = lookup[("E004", "C")]
teacher = lookup[("E004", "D")]
teacher_cfg = configs[teacher.run_id]["config"]["teacher_kl"]
e001_table = table(
    ["Run", "Mini-batch b", "Rollout R", "Envs", "Eval @300M", "Eval @600M", "Last eval", "Last 4 evals", "Best eval (at M)", "End M / reason"],
    [[link(r), f"{r.batch_size:,}", f"{r.rollout_samples:,}", f"{r.num_envs:,}", number(r.eval_at_300M), number(r.eval_at_600M), number(r.last_eval), number(r.last4_eval_mean), f"{r.best_eval:.2f} ({millions(r.best_eval_samples)})", f"{millions(r.last_logged_samples)} / {r.end_reason}"] for r in [base[key] for key in "ABCDEF"]],
)

epoch_rows = []
for key in "ABC":
    for epoch in [1, 2, 4, 8]:
        row = base[key] if epoch == 1 else lookup[("E002", f"{key}-ep{epoch}")]
        epoch_rows.append([f"[{key}, epoch {epoch}]({row.url})", number(row.eval_at_300M), number(row.last_eval), number(row.last4_eval_mean), number(row.best_eval), millions(row.last_logged_samples), row.end_reason])
e002_table = table(["Scale / epochs", "Eval @300M", "Last eval", "Last 4 evals", "Best eval", "End M", "Reason"], epoch_rows)

switch_rows = []
for row in results[results.study == "E003"].itertuples(index=False):
    switch_rows.append([link(row), number(row.initial_eval), number(row.eval_300M_after_switch), number(row.last_eval), number(row.last4_eval_mean), number(row.best_eval), f"{millions(row.last_logged_samples)} / {millions(row.last_logged_samples-row.pretrained_samples)}"])
e003_table = table(["Schedule", "At switch", "After +300M", "Last eval", "Last 4 evals", "Best eval", "End M: total / additional"], switch_rows)

architecture_rows = []
labels = {"A":"CNN, original 3 planes", "B":"Historical grid Transformer, 7 features", "B-v7":"Entity Transformer small, no teacher", "C":"Entity Transformer base, no teacher", "D":"Entity Transformer small + teacher KL"}
for key in ["A", "B", "B-v7", "C", "D"]:
    row = lookup[("E004", key)]
    architecture_rows.append([link(row), f"`{row.model_version}`", labels[key], number(row.eval_at_300M), number(row.last_eval), number(row.last4_eval_mean), number(row.best_eval), f"{millions(row.last_logged_samples)} / {millions(row.last_eval_samples)}", row.end_reason])
e004_table = table(["Run", "Logged model version", "Experiment", "Eval @300M", "Last eval", "Last 4 evals", "Best eval", "Logged / eval M", "Status"], architecture_rows)

shared = table(["Run", "Eval @300M", "Eval @400M", "Eval @575M"], [
    ["E001 C MLP", number(at(base["C"],300)), number(at(base["C"],400)),number(at(base["C"],575))],
    ["E004 B-v7 small",number(at(small,300)),number(at(small,400)),number(at(small,575))],
    ["E004 C base",number(at(large,300)),number(at(large,400)),"— (interrupted)"],
    ["E004 D small + KL",number(at(teacher,300)),number(at(teacher,400)),number(at(teacher,575))],
])

root_report = f"""# PPO Snake: experiments in reinforcement learning scaling

This project uses a simple **12×12 Snake game to explore reinforcement learning scaling behavior on accessible hardware**. CPU PPO baselines and small models on a laptop GPU make it possible to study batch size, rollout size, data reuse, training history, features and model architecture without a large compute cluster.

The initial snake length is 3, so a full board corresponds to **141 food items / score**, not 144. The goal is to understand what changes learning speed, achieved score and stability. These experiments explore scaling behavior; they do not establish a universal scaling law.

## Results at a glance

This report covers **{len(results)} W&B runs and {sum(r['history_rows'] for r in inventory['runs']):,} complete metric-history rows**, downloaded on **2026-11-09** from the [ppo-snake project](https://wandb.ai/xlz1014361885-tokyo/ppo-snake). All runs use training seed **42**. Results describe these observed trajectories, with no estimate of variation across training seeds.

- **Batch size matters under this PPO configuration.** E001 A/B/C achieved best observed eval scores of **{base['A'].best_eval:.2f} / {base['B'].best_eval:.2f} / {base['C'].best_eval:.2f}**. Comparisons at fixed rollout support a mini-batch effect; increasing rollout by itself does not consistently help.
- **More PPO epochs are not an automatic improvement.** At 300M samples, scale A scored **{base['A'].eval_at_300M:.2f}** with one epoch but **{lookup[('E002','A-ep8')].eval_at_300M:.2f}** with eight. Large-batch C tolerated two or four epochs much better than A/B, with little change in the late score range.
- **Training history matters.** After switching the complete scale, B→C can reach roughly 110, while the A→C schedules observed here end around 76–94. A late C→B switch can retain scores above 100. A larger batch does not erase every training-history effect within the observed budget.
- **The earlier MLP score range is not the game's model ceiling.** E004 base reached **{large.last_eval:.2f} at {millions(large.last_eval_samples)}M**, and small + teacher KL reached **{teacher.last_eval:.2f} at {millions(teacher.last_eval_samples)}M**. Both runs are marked **`crashed` / incomplete** in W&B; these are last observed evaluations, not completed-run guarantees.

The [per-run catalog](docs/analysis/run-catalog.md) includes the design, plot, result and interpretation for **every run**. Exact numbers and logged configurations are available in [run_results.csv](docs/analysis/data/run_results.csv), [eval_history.csv](docs/analysis/data/eval_history.csv), and [run_configs.json](docs/analysis/data/run_configs.json). Implementation details and feature layouts are in the [technical guide](docs/technical-guide.md).

## How to read the comparisons

The primary strength metric is **held-out deterministic `eval_samples/mean_score`**, measured on the same **1,024 evaluation seeds**, with a 5,000-step cap per game. Stochastic training scores use the latest 1,024 completed training games; they help diagnose learning but are not interchangeable with eval.

Evaluation curves below are **raw and unsmoothed**. Training diagnostic plots show faint raw values plus a causal EMA with a **25M-sample half-life**, except where a figure explicitly states 5M or 10M. No curve is extended after a run ends. `M` means one million environment transitions, and displayed sample counts are rounded; the CSV retains exact counts.

**Last eval** is the last recorded evaluation, which can precede the last training upload. **Last 4 evals** is the arithmetic mean of the last four evaluation means; it describes a short tail of the trajectory, not four independent training repetitions. **Best eval** is the maximum observed evaluation mean, shown to reveal peaks and regressions. It is an optimistic, retrospectively selected diagnostic, not an independently verified model-selection score. Different stopping times make endpoint-only rankings incomplete; common-sample comparisons and the full curves are included for that reason.

Shared PPO settings are LR **3e-4**, gamma **0.99**, GAE lambda **0.95**, clipping range **0.2**, value coefficient **1**, entropy coefficient **0.001**, and rollout horizon **64**. The MLP has two 256-unit hidden layers and the original head/body/food planes. Evaluation is scheduled every **25M** samples; **12 consecutive evaluations without a strict new high** trigger early stopping. The cap is **1,200M new samples**, with a possible overshoot below one rollout. Early stopping is a budget heuristic, not evidence of an irreversible ceiling.

## E001 — Mini-batch and rollout size

**Question.** Does the benefit of increasing the joint training scale come from mini-batch size `b`, rollout size `R`, or both? `R = num_envs × 64`. A/B/C keep `R/b=8`. D/E/F provide comparisons that hold one factor fixed. All use one PPO epoch and the same MLP.

{e001_table}

### Joint scale: A, B and C

![E001 joint scaling](docs/analysis/figures/e001_joint_scale.png)

At the common **300M** budget, A/B/C score **{base['A'].eval_at_300M:.2f} / {base['B'].eval_at_300M:.2f} / {base['C'].eval_at_300M:.2f}**. At **600M**, B and C are much closer (**{base['B'].eval_at_600M:.2f} vs {base['C'].eval_at_600M:.2f}**), while A remains lower. Larger scale is associated with a higher late score range, but its advantage changes with training time. The data do not show that every point on the larger-scale curve is better.

### Holding rollout fixed

![E001 fixed rollout](docs/analysis/figures/e001_fixed_rollout.png)

- At **R=8,192**, B (`b=1,024`) scores **{base['B'].eval_at_600M:.2f}** at 600M versus D (`b=256`) at **{base['D'].eval_at_600M:.2f}**.
- At **R=81,920**, C (`b=10,240`) scores **{base['C'].eval_at_600M:.2f}** at 600M versus E (`b=1,024`) at **{base['E'].eval_at_600M:.2f}**.

These comparisons support a mini-batch contribution under the fixed learning rate and optimizer used here. They do **not** isolate gradient variance as the mechanism: at fixed samples, a smaller mini-batch also performs more Adam steps (`steps ≈ samples × epochs / b`).

### Holding mini-batch fixed

![E001 fixed mini-batch](docs/analysis/figures/e001_fixed_batch.png)

Increasing R is not uniformly helpful. A→D at `b=256` does not raise the observed best score, and B→E at `b=1,024` learns more slowly through much of the curve. At `b=10,240`, C's larger rollout finishes at **{base['C'].last_eval:.2f}** versus F's **{base['F'].last_eval:.2f}** at the same 1,200M budget. Rollout effects depend on mini-batch size. Changing R here also changes the number of parallel environments, so it combines policy-refresh frequency and trajectory-diversity effects.

### Diagnostics and conclusion

![E001 training diagnostics](docs/analysis/figures/e001_diagnostics.png)

In the common **500–600M** window, mean value loss is approximately **0.154 / 0.109 / 0.090** for A/B/C; entropy is **0.0186 / 0.0229 / 0.0345**, and raw advantage std is **0.398 / 0.332 / 0.300**. This is consistent with lower value-target error and more stochastic policies at larger scale. These measurements use different learned state distributions and are not a causal explanation. Approximate KL is also not uniformly lower for larger batches.

**Conclusion:** with this fixed-LR, epoch-1 protocol, increasing mini-batch size is a stronger and more consistent direction than increasing rollout alone. The factorial comparisons support that distinction. The available sample budget and early stopping do not prove an absolute, irrecoverable performance ceiling.

## E002 — PPO epochs

**Question.** How much can each rollout be reused before more optimizer updates become harmful? Each A/B/C baseline from E001 is compared with **2, 4 and 8 epochs**, with its mini-batch, rollout, learning rate and model unchanged. More epochs multiply optimizer work per environment sample, so this is a sample-budget comparison, not equal compute.

{e002_table}

![E002 epoch curves](docs/analysis/figures/e002_epochs.png)

**A:** two epochs slightly improve the 300M score (**61.98 vs 58.03**) but later regress; four and eight epochs perform poorly. **B:** all tested epoch increases are worse than epoch 1 at the common 300M point; eight epochs ends near score 1. **C:** two and four epochs achieve roughly the same late score range as epoch 1. Epoch 2 has a higher last eval (**107.87 vs 103.24**), while its last-four mean (**105.63**) is slightly below epoch 1 (**106.19**). The ranking depends on which point is selected. Eight epochs eventually regresses substantially.

### What the A-epoch-2 collapse does and does not show

![A epoch 2 collapse diagnostics](docs/analysis/figures/e002_collapse.png)

The eval mean drops from **48.03 at 700M** to **13.99 at 750M**, then partly recovers to **41.91 at 800M**. Training scores also decline, so this is not solely a deterministic-evaluation artifact. In the **650–700M** window, epoch-2 approximate KL averages about **0.0175**; during **750–775M**, raw advantage std averages **0.519** and critic explained variance averages **0.814**, versus roughly **0.380 / 0.957** before the sharp regression. Entropy is low, but later rises during partial recovery; its trajectory alone does not identify the cause.

The joint pattern is **consistent with excessive policy updating and a deteriorating critic interacting**, but it does not establish which process initiated the regression. PPO's clipping objective is not a hard KL bound, and these logged KL estimates are pre-mini-batch diagnostics, not full-rollout post-update distances. Proving a mechanism would require additional measurements or a controlled change.

**Conclusion:** retain epoch 1 as the conservative baseline for A/B. C tolerates moderate reuse, but these runs do not establish a robust late-score gain from epochs 2–4. An epoch increase without controlling learning rate or policy movement is not a free scaling improvement.

## E003 — Switching the complete batch/rollout scale

**Question.** Can a policy trained at one scale recover or retain performance after the scale changes? Each branch loads an existing weights-only checkpoint, **resets Adam**, and switches the complete A/B/C scale while retaining `R/b=8`. A→B and A→C use nominal switch points **100/300/550M**; B→C uses **200/500/850M**; C→B uses **150/450/850M**. These points were chosen separately for the source curves, not as universal early/middle/late fractions.

Every branch has up to **1,200M additional** samples, with the same eval-based early stopping. There is no 300M hard cap. The initial evaluation at the switch is included. Actual source counts can differ from nominal names by less than one source rollout.

{e003_table}

![E003 total sample clock](docs/analysis/figures/e003_total.png)

![E003 additional sample clock](docs/analysis/figures/e003_additional.png)

The first plot charges pretraining to the total sample clock. The second shows adaptation after switching; the from-scratch reference there excludes the branch's prefix cost and must not be read as an equal-total-budget comparison.

- **A→B:** the branches improve on their initial score, but remain around 67–72 in their last-four means rather than reaching B-from-scratch's late range near 98.
- **A→C:** recovery is incomplete and non-monotonic in switching time. The 300M switch ends at **93.87**, the 100M switch at **83.95**, and the 550M switch at **75.60**. Earlier switching is not always better in these trajectories.
- **B→C:** the 500M switch reaches **110.61**, close to C-from-scratch's best **110.65**, and ends at **108.12**. The 850M switch reaches **107.42** after a shorter adaptation. Medium-scale pretraining can therefore be followed by strong large-scale performance in this sample.
- **C→B:** lowering the scale does not immediately erase high performance. The 850M branch ends at **105.31**, whereas the 450M branch falls from **100.13** initially to **93.82**, stops after +300M, and never exceeds its initial evaluation.

**Conclusion:** the attained policy depends on its training path as well as its current scale. Recovery is observed, but it is not guaranteed within a fixed continuation budget. Because same-scale Adam-reset controls were deliberately not run, these comparisons do not isolate the effect of changing scale from optimizer reset. Their endpoints also use different total budgets; this is evidence about schedules, not proof of an optimal switch time or an irreversible small-batch trap.

## E004 — Features, architecture and evaluation-best teacher KL

**Question.** Can a better representation/model exceed the MLP score range, and can a frozen best-eval teacher reduce the small model's regression? All logged E004 runs use `b=10,240`, `R=81,920`, epoch 1 and the shared PPO/eval protocol.

**Historical names matter.** The W&B run called `exp-B` is **`transformer_v3`**, an earlier seven-feature grid model. `exp-B-v7` is the entity **small** architecture now named `transformer_v1_small`. The current B YAML selects small; it does not reproduce the historical `exp-B` run. C is the base entity model. D is small plus teacher KL. The report uses the configurations stored with the runs, not today's YAML as a substitute for history.

{e004_table}

![E004 architecture comparison](docs/analysis/figures/e004_architectures.png)

The CNN uses only the original three planes and ends at **74.42**, below the MLP's **103.24** at 1,200M. The early seven-feature Transformer is unstable and ends early at **30.98**; it is not evidence that the later entity Transformers fail. The entity models use **144 Cell + Snake + Food + Global tokens**, with **66/21/24/8** respective raw feature dimensions. Small uses input→32→16 MLPs and 2 encoders; base uses input→64→32 and 3 encoders, retaining a 256-unit actor/critic readout. These models change both representation and architecture relative to the MLP, so this study does not identify which individual feature supplies the gain.

For a common sample-clock comparison:

{shared}

The base model reaches **132.94** at 400M, while the MLP is **92.71** at that point. This is evidence that the earlier MLP range was not a game-level performance ceiling. C's history ends at 419M training samples, with its last eval at 400M; D ends at 925M. Both have W&B state `crashed`. A training exception log was unavailable for these runs, so the interruption causes remain unknown. C also has a sharp training-score drop after its last eval: the rolling mean falls to **18.49 at 409M**, then recovers to **129.32 at 419M**. No later held-out eval confirms the state after that event. A high last eval does not prove future stability or convergence. The 5,000-step evaluation cap is also part of what these scores measure.

The stronger representations also cost more in these recorded jobs. Logged training wall time is about **{base['C'].logged_wall_time_seconds/3600:.2f} h** for the MLP's 1,200M samples, **{large.logged_wall_time_seconds/3600:.2f} h** for base's 419M, and **{teacher.logged_wall_time_seconds/3600:.2f} h** for D's 925M. These are observed job times across CPU/CUDA and evolving implementations, not a controlled hardware-speed comparison. Higher score is not synonymous with lower computational cost.

### D: frozen evaluation-best self-teacher

Starting from B's small-model configuration, D adds

```yaml
teacher_kl:
  enabled: true
  coef: {teacher_cfg['coef']}
```

The objective is **`L_PPO + coef × mean_s KL(teacher(.|s) || student(.|s))`**, evaluated on student rollout states. The first scheduled eval initializes the teacher. Each later strict new eval high copies the student into the frozen teacher; ties/regressions keep it. There is no critic distillation, coefficient decay, or student/Adam reset. `teacher.ot` stores this teacher; `latest.ot` remains the current student.

![E004 teacher comparison](docs/analysis/figures/e004_teacher_comparison.png)

The no-teacher small run peaks at **125.55 at 275M**, then ends at **71.62 at 575M**. D scores **{at(teacher,575):.2f} at the same 575M point**, a **{at(teacher,575)-small.last_eval:.2f}-point** difference, and later reaches **131.90**, with last eval **131.55**. Its teacher is replaced **{int(teacher.teacher_updates)} times**, including initialization. D still has regressions, notably **93.19 at 650M**; the penalty is not a monotonic-performance guarantee.

**Attribution remains limited.** The first eval already differs before KL has been applied (**{at(small,25):.2f} without teacher vs {at(teacher,25):.2f} in D**). The runs share nominal configuration and seed, but are not identical pre-intervention trajectories. Consequently, the higher score and better observed retention in D support further investigation of teacher KL, rather than proving that KL alone caused the full improvement.

![E004 teacher diagnostics](docs/analysis/figures/e004_teacher_diagnostics.png)

The small no-teacher run's entropy is about **0.0200** at its 275M peak and **0.0232** at 575M. Thus its late regression cannot be summarized as entropy simply decreasing to zero. At 575M, D has entropy about **0.0356**, value loss **0.120**, and critic explained variance **0.972**, compared with **0.0232 / 0.215 / 0.959** without a teacher. These are correlated diagnostic differences, not an identified causal chain. Teacher KL is separate from PPO approximate KL; loss-term magnitudes do not measure their gradient influence.

**Conclusion:** the entity base and teacher-regularized small trajectories exceed the MLP range in held-out eval. Base is the strongest observed architecture candidate at 400M; teacher KL is a promising retention direction for small. Complete-run stability and clean causal attribution remain unresolved.

## What to take forward

1. Use **large-batch, epoch-1** as the main MLP reference under the current LR; avoid assuming that rollout growth or more epochs will improve it.
2. Keep the **entity base** as a high-score candidate, but obtain a full-budget trajectory before calling its late performance stable. Its current history is incomplete.
3. If the next question is the isolated effect of teacher KL, compare matched initial training states and retain pre-KL trajectory checks. D is encouraging, but the current two runs cannot attribute all of the improvement to the regularizer.
4. At high average scores, log **per-episode eval scores and termination/truncation outcomes** if tail behavior matters. Current mean/std/min/max cannot recover a win rate or paired episode differences. D's last eval still includes a minimum score of 2 despite its 131.55 mean.

## Evidence, limits and reproduction

The export uses W&B **`scan_history(keys=None, use_cache=False)`**, rather than the sampled history used by charts. Configs, summaries, compressed full metric histories and a SHA-256 inventory are downloaded under **`analysis/wandb/raw/`** locally; raw archives are Git-ignored. This is a metric/config export, not a download of model checkpoints or all W&B artifacts. Portable reviewed data and figures are under [docs/analysis](docs/analysis/run-catalog.md). The [methods and validation record](docs/analysis/methods.md) defines calculations, scope, source links and checks.

The one-seed design is deliberate. The 1,024 evaluation games do not substitute for independent training repetitions, and the same eval seeds are reused for stopping and teacher selection. Evaluation is held out from PPO transitions, but it also acts as a validation set for those decisions; there is no fresh test-set confirmation of selected peaks. No confidence bands, significance claims or rliable training-seed uncertainty estimates are invented from these trajectories. These reporting choices follow the concerns discussed in [Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315), [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), and [Rliable's evaluation explanation](https://araffin.github.io/post/rliable/).

Reproduce the download and analysis using existing W&B credentials (do not place API keys in source files):

```powershell
# Export: the base Python has the W&B SDK.
& 'C:/Users/Liuluotu/Anaconda/anaconda3/python.exe' scripts/export_wandb_runs.py

# Analyze/render: orbit_wars has a compatible NumPy/pandas/Matplotlib stack.
& 'C:/Users/Liuluotu/Anaconda/anaconda3/envs/orbit_wars/python.exe' scripts/analyze_wandb_runs.py
& 'C:/Users/Liuluotu/Anaconda/anaconda3/envs/orbit_wars/python.exe' scripts/build_experiment_readme.py
```

`export_wandb_runs.py --refresh` fetches all histories again; by default unchanged, fully exported runs are cached. New runs require reviewing the conclusions as well as regenerating tables. The analysis verifies archive hashes, row counts, unique history steps, sample clocks, score bounds, summary coverage, optimizer-step accounting and D's teacher-update rule. An interrupted export does not create fabricated observations. If W&B requires a proxy, use the numeric proxy port in that command's environment; the scripts do not change system network settings.

## Run the current experiments

The active E004 configs are [A–D](config/experiments/feature_architecture). These are configs for new runs; historical W&B identities remain as documented above. For example, start **D only**:

```powershell
$env:LIBTORCH = 'C:\\Users\\Liuluotu\\Anaconda\\anaconda3\\envs\\orbit_wars\\Lib\\site-packages\\torch'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sweep_feature_architecture.ps1 -StartAt D -Only
```

Change `D` to `B` or `C` to select small or base. The sweep protects existing `latest.ot` files; `-AllowExistingCheckpoints` permits a fresh rerun, not a resume. CUDA uses the installed **memory-efficient BF16 SDPA** backend; it is not compiled CUDA FlashAttention. Full feature layouts, architecture diagrams, inference instructions and Windows setup are in the [technical guide](docs/technical-guide.md).

```powershell
cargo fmt --all -- --check
cargo test --all-targets
```
"""


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
