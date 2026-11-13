# PPO Snake: Reinforcement Learning Scaling Experiments

This project uses the classic **Snake game to study scaling laws in reinforcement learning**, including how **data, compute and model size** affect learning speed, score and training stability.

The board is **12×12**, the initial snake length is **3**, and the maximum score is **141**. Eating food gives reward 1; other transitions give reward 0.

## Main results

- **Larger batches achieve higher scores.** A/B/C use batches **256 / 1,024 / 10,240** and reach best scores of **67.94 / 102.48 / 110.65**. Rollout size and environment count grow proportionally with batch size.
- **Increasing rollout/batch at a fixed batch size does not consistently improve scores.** It helps batch 10,240 when the ratio grows from 2 to 8, but not batch 256 or 1,024 in the tested comparisons.
- **More PPO epochs can learn faster initially, but hurt later scores.** The early speedup is clearest at batch 10,240. Small batches suffer much more from repeated updates; eight epochs causes severe degradation at batch 256 and 1,024.
- **Switching to a larger batch does not generally recover the score of starting with that batch.** A→B and A→C stay lower, while B-500M→C almost catches up. Early, middle and late switches produce different outcomes.
- **Transformer models with richer features score much higher than the baseline MLP.** The small entity Transformer reaches **125.55**; increasing model size raises this to **132.94** and speeds up learning.
- **Teacher KL improves score retention.** Small + teacher KL avoids the sustained late collapse of the no-teacher small run and finishes its recorded trajectory at **131.55**, compared with **71.62** without a teacher.

| Experiment | Content | Model |
| --- | --- | --- |
| [E001.1](#e001-batch) | Batch size, keeping rollout/batch = 8 | Baseline MLP |
| [E001.2](#e001-rollout) | Rollout/batch ratio, keeping batch size fixed | Baseline MLP |
| [E002](#e002) | PPO epochs | Baseline MLP |
| [E003](#e003) | Switch the complete batch/rollout scale during training | Baseline MLP |
| [E004](#e004) | Features, architecture, model size and teacher KL | CNN and Transformer variants |

## Shared configuration

**Both E001 experiments, E002 and E003 use the same baseline MLP and the same three input planes.** E004 fixes the large batch/rollout configuration and changes the features, architecture, model size or teacher loss. E001.1 and E001.2 reuse the original six E001 runs; the A–F run labels are unchanged.

| Parameter | E001.1 / E001.2 | E002: epochs | E003: switching | E004: features / architecture |
| --- | --- | --- | --- | --- |
| Model | MLP 432→256→256 | Same MLP | Same MLP | CNN / Transformer variants |
| Input features | Head, body, food | Same 3 planes | Same 3 planes | 3 planes or 66/21/24/8 entity features |
| Mini-batch `b` | 256 / 1,024 / 10,240 | A/B/C: 256 / 1,024 / 10,240 | Switch between A/B/C | **10,240** |
| Rollout `R` | 2,048 / 8,192 / 20,480 / 81,920 | A/B/C: 2,048 / 8,192 / 81,920 | Switch together with `b`, keeping `R/b=8` | **81,920** |
| Parallel environments | 32 / 128 / 320 / 1,280 | 32 / 128 / 1,280 | Switch together with `b` | **1,280** |
| Rollout steps per environment | 64 | 64 | 64 | 64 |
| PPO epochs | **1** | **1 / 2 / 4 / 8** | **1** | **1** |
| Initialization | From scratch | From scratch | Existing checkpoint; Adam reset | From scratch |
| Device | CPU | CPU | CPU | CNN: CPU; Transformers: CUDA |
| Teacher KL coefficient | 0 | 0 | 0 | 0 except **D: 0.001** |
| Maximum new samples | 1,200M | 1,200M | 1,200M **after the switch** | 1,200M |

| Shared parameter | Value |
| --- | --- |
| Board / initial snake length | 12×12 / 3 |
| Training seed | 42 |
| Optimizer / learning rate | Adam / **0.0003**, fixed |
| Discount `gamma` | 0.99 |
| GAE `lambda` | 0.95 |
| PPO clip range | 0.2 |
| Value loss coefficient | 1.0 |
| Entropy coefficient | 0.001 |
| Gradient clipping threshold | 0.5 |
| Evaluation | Every **25M** samples; **1,024** fixed-seed games; deterministic actions |
| Maximum evaluation episode length | 5,000 steps |
| Early stopping | **12** evaluations without a new best mean score |
| Training score window | Latest 1,024 completed games |
| Checkpoint interval | 50M samples |

**Scores in the result tables are evaluation mean scores.** `M` means one million environment transitions. “Best” is the highest recorded evaluation; “last” is the last recorded evaluation; “last 4 evals” averages the final four points. Evaluation curves are unsmoothed. Training-diagnostic plots use the EMA half-life stated in each figure.

## Baseline model and input features

The batch, rollout, epoch and switching experiments use **`mlp_v1`**, with **177,925 parameters**. Its two shared hidden layers feed an actor with four absolute actions—up, down, left and right—and a scalar value head.

### Baseline features

Each feature is a 12×12 binary plane. The three planes are flattened into **432 values**.

{{ORIGINAL_FEATURES}}

### Baseline architecture

{{MLP_DIAGRAM}}

`N` is the number of states processed together. It is 1 for a single game and larger during parallel sampling, mini-batch training and batched evaluation.

<a id="e001"></a>
<a id="e001-batch"></a>

## E001.1 — Batch size at fixed rollout/batch

### Experiment contents

Compare **A, B and C** with mini-batch sizes **256, 1,024 and 10,240**, keeping **rollout/batch = 8**. Rollout size and the number of parallel environments increase proportionally; the per-environment rollout horizon stays at 64 steps.

### Key configuration

All three runs use the baseline MLP, original three-plane features, epoch 1 and learning rate 0.0003. Each rollout supplies **8 mini-batch optimizer steps**. This is the project's batch-scaling experiment: the fixed quantity is `R/b`, rather than the absolute rollout size.

{{E001_BATCH_CONFIG}}

### Model architecture

{{MLP_DIAGRAM}}

### Results

{{E001_BATCH_RESULTS}}

![Batch experiment: A/B/C with rollout/batch = 8](docs/analysis/figures/e001_batch_scale.png)

![Batch experiment: training diagnostics at rollout/batch = 8](docs/analysis/figures/e001_diagnostics.png)

### Analysis

1. **Batch 256 has a much lower score range.** At 600M samples, A scores **66.26**, compared with **95.23** for B and **95.87** for C.
2. **The largest batch achieves the highest later scores.** B and C are close at 600M, but their best scores are **102.48 and 110.65**. The means of their last four evals are **97.82 and 106.19**, compared with **61.69** for A. These tail windows occur at different training endpoints, as the table shows.
3. **Larger batches have lower value loss and slower entropy decay.** During 500–600M, A/B/C have average value losses of approximately **0.154 / 0.109 / 0.090**, entropy of **0.0186 / 0.0229 / 0.0345**, and raw advantage std of **0.398 / 0.332 / 0.300**.

The comparison measures scaling batch and rollout together at a fixed ratio. It does not separate their individual contributions. At epoch 1, all runs still use each collected transition once, while larger batches reduce the number of optimizer steps per environment sample.

### Author's hypothesis

I suspect that a larger batch raises the model's attainable score plateau, and that simply spending more compute on small-batch training cannot fully make up the gap. In this experiment, the absolute batch and rollout sizes increase together. This may reduce noise in the estimated learning signal and make useful update directions easier to distinguish.

Early in training, a roughly correct decision is enough to improve the score. Near the performance ceiling, the difference between a good action and a slightly better action becomes much smaller, like the tiny margins separating expert players. My interpretation is that late learning needs a finer signal-to-noise resolution. Larger batches and rollouts may reveal these small, consistent advantages that noisy small-batch updates struggle to resolve. The curves motivate this explanation; they do not establish an absolute ceiling or rule out every longer small-batch run.

### Conclusion

**Increasing batch size raises the achieved score range, with batch 10,240 producing the highest observed scores.** The improvement from 1,024 to 10,240 is most visible later in training.

<a id="e001-rollout"></a>

## E001.2 — Rollout/batch ratio at fixed batch size

### Experiment contents

Keep PPO mini-batch size fixed and change **rollout/batch**: compare **A vs D** at batch 256 (**8 vs 32**), **B vs E** at batch 1,024 (**8 vs 80**), and **F vs C** at batch 10,240 (**2 vs 8**). Rollout size changes through the number of environments; the per-environment horizon remains 64.

### Key configuration

All runs use the baseline MLP, original three-plane features, epoch 1 and learning rate 0.0003. `R = number of environments × 64`. At epoch 1, `R/b` is also the number of mini-batch optimizer steps per rollout; increasing it does not increase how often each transition is reused.

{{E001_ROLLOUT_CONFIG}}

### Model architecture

{{MLP_DIAGRAM}}

### Results

{{E001_ROLLOUT_RESULTS}}

![Rollout experiment: varying rollout/batch at fixed batch size](docs/analysis/figures/e001_fixed_batch.png)

### Analysis

1. **At batch 256, increasing R/b from 8 to 32 does not improve the score.** Increasing R from 2,048 to 8,192 changes the best score from **67.94 to 65.83**. At 600M, ratio 8 scores **66.26**, versus **48.16** for ratio 32.
2. **At batch 1,024, increasing R/b from 8 to 80 slows learning.** At 600M, ratio 8 scores **95.23**, versus **77.06** for ratio 80. Their best scores are similar, **102.48 vs 101.03**.
3. **At batch 10,240, increasing R/b from 2 to 8 helps.** At the shared 1,200M endpoint, ratio 8 scores **103.24**, versus **96.64** for ratio 2. Their best scores are **110.65 vs 99.58**.

Rollout size determines how often a new policy collects fresh data. Here it also changes the number of parallel environments. Its effect depends on the mini-batch size rather than following a simple larger-is-better rule.

### Author's hypothesis

At a fixed batch size, a larger rollout means more optimizer steps within one PPO iteration before collecting fresh data. I suspect that these successive updates can move the current policy increasingly far from the policy that collected the rollout. PPO clipping discourages some large probability-ratio changes, but it does not impose a hard limit on the total policy change or guarantee that each update improves performance.

This could explain why collecting a larger rollout does not necessarily help: later updates within the iteration may be less reliable as the policy changes. Each transition is still used once at epoch 1, and the total number of optimizer steps at a fixed sample budget is approximately unchanged. What changes is how many updates occur before the data-collecting policy is refreshed.

### Conclusion

**At a fixed batch size, a larger rollout/batch ratio is not always better.** Ratio 8 outperforms 32 at batch 256 and learns faster than 80 at batch 1,024, while increasing the ratio from 2 to 8 helps batch 10,240. Together with E001.1, this shows that simply increasing rollout at a fixed batch does not reproduce the gains from scaling batch and rollout proportionally.

<a id="e002"></a>

## E002 — PPO epochs

### Experiment contents

Reuse each rollout for **1, 2, 4 or 8 PPO epochs**. Compare how repeated updates affect initial learning speed, later score and collapse at different batch sizes.

### Key configuration

| Scale | Mini-batch b | Rollout R | Environments | Epochs | Learning rate | Model / features |
| --- | --- | --- | --- | --- | --- | --- |
| A | 256 | 2,048 | 32 | 1 / 2 / 4 / 8 | 0.0003 | Baseline MLP / 3 planes |
| B | 1,024 | 8,192 | 128 | 1 / 2 / 4 / 8 | 0.0003 | Baseline MLP / 3 planes |
| C | 10,240 | 81,920 | 1,280 | 1 / 2 / 4 / 8 | 0.0003 | Baseline MLP / 3 planes |

The epoch-1 runs are the A/B/C baselines from E001. All other settings stay the same.

### Model architecture

{{MLP_DIAGRAM}}

The network is unchanged. More epochs repeat optimization over the same rollout.

### Results

![E002: early learning](docs/analysis/figures/e002_early_learning.png)

{{E002_EARLY}}

![E002: complete learning curves](docs/analysis/figures/e002_epochs.png)

{{E002_RESULTS}}

### Analysis

**Initial learning can be faster with more epochs.** At batch 10,240 and 50M samples, epoch 1/2/4/8 scores are **31.80 / 40.91 / 60.41 / 60.99**. At 100M, they are **51.75 / 66.05 / 74.90 / 70.47**. The effect also appears with selected moderate increases at smaller scales: A with epoch 2 scores **48.02** at 25M versus **36.76** with epoch 1; B with epoch 4 scores **54.49** at 50M versus **50.23** with epoch 1. Eight epochs is already too much for A/B early in training.

**More epochs hurt later scores, especially at small batch sizes.** A's last-four scores for epoch 1/2/4/8 are **61.69 / 31.49 / 11.96 / 1.74**. B's are **97.82 / 68.99 / 64.01 / 2.96**. C is much less affected by epochs 2–4: **106.19 / 105.63 / 104.35** for epochs 1/2/4, although epoch 8 still drops to **51.22**.

![E002: A epoch-2 collapse](docs/analysis/figures/e002_collapse.png)

A's epoch-2 run illustrates the later instability: eval falls from **48.03 at 700M** to **13.99 at 750M**, before partially recovering. Training scores fall too; raw advantages fluctuate and critic explained variance deteriorates. More epochs apply more updates before fresh data arrives, and the fixed learning rate makes the cumulative policy change larger. These are plausible contributors to the degradation visible in the curves.

### Author's hypothesis

Increasing epochs repeatedly reuses the same rollout, whereas increasing rollout at epoch 1 supplies more distinct transitions. Both increase the number of updates before fresh data arrives, which may move the policy farther from the policy that collected the data. Repeated reuse may also amplify noise specific to that rollout.

I suspect that extra updates help early learning because the policy mainly needs to learn broad decision patterns. Later, when improvements depend on smaller distinctions, the same repeated updates can damage an already strong policy. The larger-batch configurations also collect larger rollouts, potentially providing more useful information and less noisy update directions. Even after several epochs, there may still be meaningful signal to learn from that dataset. This is my explanation for why the negative effect of extra epochs is weaker at the larger batch sizes.

### Conclusion

**Additional epochs trade faster initial learning for lower later scores and more instability. The smaller the batch, the stronger the damage.** Batch 10,240 tolerates 2–4 epochs relatively well, but still does not improve the late average over epoch 1. Eight epochs degrades all three scales. Epoch 1 remains the best overall choice in this experiment.

<a id="e003"></a>

## E003 — Switching batch and rollout during training

### Experiment contents

Start from an A/B/C checkpoint and switch the complete training scale. Test whether a policy trained with a smaller batch can catch up after switching upward, and whether a policy trained with a larger batch keeps its strength after switching downward. Then compare early, middle and late switches for each source curve.

### Key configuration

| Direction | Source b / R | Target b / R | Switch points | Model | Optimizer on switch |
| --- | --- | --- | --- | --- | --- |
| A→B | 256 / 2,048 | 1,024 / 8,192 | 100 / 300 / 550M | Baseline MLP | Reset Adam |
| A→C | 256 / 2,048 | 10,240 / 81,920 | 100 / 300 / 550M | Baseline MLP | Reset Adam |
| B→C | 1,024 / 8,192 | 10,240 / 81,920 | 200 / 500 / 850M | Baseline MLP | Reset Adam |
| C→B | 10,240 / 81,920 | 1,024 / 8,192 | 150 / 450 / 850M | Baseline MLP | Reset Adam |

Rollout horizon stays at 64 and R/b stays at 8. The model weights are retained; features, learning rate and epoch 1 are unchanged. Each branch can train for up to **1,200M additional samples**, with the same eval-based early stopping.

### Model architecture

{{MLP_DIAGRAM}}

Only the batch/rollout scale changes. The MLP and its input features stay the same throughout the switch.

### Results

{{E003_RESULTS}}

![E003: total sample clock](docs/analysis/figures/e003_total.png)

![E003: additional samples after switching](docs/analysis/figures/e003_additional.png)

The total-sample plot includes pretraining. The additional-sample plot starts at each switch and shows adaptation speed.

### Analysis

**First, compare switching with training from scratch.** The upward switches do not exceed the best scores of starting with the target batch: A→B peaks around **70–72**, below B's **102.48**; A→C peaks around **78–98**, below C's **110.65**. B-500M→C comes very close at **110.61**. The downward C-850M→B branch reaches **108.28**, retaining more strength than B-from-scratch's peak of 102.48, although remaining below the original C peak. Thus the weak A-trained policies do not fully catch up, while a strong large-batch policy can retain much of its strength at a smaller batch.

| Direction | Early switch | Middle switch | Late switch | Timing result |
| --- | --- | --- | --- | --- |
| A→B | Last score **67.91** | **69.10** | **72.07** | Later switches finish slightly higher, but all remain far below B from scratch |
| A→C | **83.95** | **93.87** | **75.60** | Middle is best; the late switch recovers least |
| B→C | **97.71** | **108.12** | **106.01** | Middle and late are stronger than early; middle almost matches C's best score |
| C→B | **97.53** | **93.82** | **105.31** | Late best preserves the stronger source policy |

Switching earlier is therefore not always better. For A→C, the 550M switch stays near the original small-batch score range even after a long continuation. For B→C, more developed source policies transfer better. For C→B, the late checkpoint preserves high scores, while the 450M checkpoint never beats its initial **100.13** and finishes at **93.82**. Branches have different training lengths, shown in the result table, so the timing comparison describes their recorded trajectories.

### Author's hypothesis

I view switching as a change in the parameter-training path, or equivalently, the path along which the policy learns its decisions. Small-batch training may lead the model toward a less effective strategy. By the time the batch is increased, the policy may already have low entropy and rarely visit the alternative behaviors needed to discover a better strategy.

A larger batch can estimate updates more precisely on the states the policy visits, but it does not automatically restore exploration or undo the earlier learning path. This may explain why some small-batch policies fail to catch up after switching, while stronger source policies transfer better. Every switching branch also resets Adam, so the observed path dependence includes that optimizer reset.

### Conclusion

**Increasing batch size partway through training does not automatically recover the score of using the larger batch from the beginning.** Recovery depends on the source policy and switch time. Very small-batch pretraining leaves a larger gap; medium-batch pretraining can nearly catch up. A strong large-batch policy can also keep a high score after the batch is reduced. There is no single early-or-late switching rule across all four directions.

<a id="e004"></a>

## E004 — Features, model architecture and teacher KL

### Experiment contents

Keep the strongest E001 training scale fixed and change the representation and network. Compare the original-feature CNN, the small entity Transformer, the larger entity Transformer, and the small entity Transformer with a best-eval teacher KL loss.

### Key configuration

All E004 runs use **batch 10,240**, **rollout 81,920**, **1,280 environments**, **64 rollout steps**, **epoch 1**, **learning rate 0.0003**, **entropy coefficient 0.001**, and a **256-unit output projection**. Only D adds teacher KL with coefficient 0.001.

{{E004_CONFIG}}

### Model architectures

#### A — CNN with original features

Input features are the same head/body/food planes used by the baseline MLP. All convolutions use zero padding of width 1.

{{CNN_DIAGRAM}}

#### B — Small entity Transformer

The board contributes **144 Cell tokens**, followed by **1 Snake, 1 Food and 1 Global token**, for **147 tokens**. Four independent input MLPs embed the token types. The actor reads Snake and the critic reads Global.

{{SMALL_DIAGRAM}}

#### C — Larger entity Transformer

C uses the same features as small, but expands the input MLP from **input→32→16** to **input→64→32**, the token dimension from **16 to 32**, and the encoder depth from **2 to 3**.

{{BASE_DIAGRAM}}

Both entity models use pre-LayerNorm encoders, 2 attention heads, residual connections, GELU FFNs with width `2 × token_dim`, no dropout and no attention mask. Intermediate layers update all tokens; the final layer produces only the Snake and Global query rows while attending to all 147 tokens.

#### D — Small entity Transformer with a teacher

The student backbone is exactly the small model above. A second, frozen small model supplies the teacher action distribution on student rollout states.

{{TEACHER_DIAGRAM}}

`Total loss = PPO loss + 0.001 × KL(teacher || student)`.

The first evaluation initializes the teacher. Whenever a later evaluation gets a higher mean score, the student weights replace the teacher; otherwise the teacher stays unchanged. Only the student is trained. Teacher updates do not reset the student or Adam. The coefficient is configurable through `teacher_kl.coef`.

### Entity input features

Small, base and D share the following features. The flat observation has **9,557 values**: `144 × 66 + 21 + 24 + 8`. Each context vector occurs once, rather than being repeated across the grid.

| Token | Number of tokens | Features per token | Role |
| --- | --- | --- | --- |
| Cell | 144 | 66 | Occupancy, body order, position, chain links and connected region |
| Snake | 1 | 21 | Snake length, heading, head position and available region |
| Food | 1 | 24 | Food position and its connected region |
| Global | 1 | 8 | Board occupancy and number of connected regions; critic readout |

For the tables below, `A=144`, `L` is snake length, and the body is indexed from the head. Release time `t=L-i` gives tail=1 and head=L, assuming no further growth. Ego coordinates `(dx′,dy′)` are relative to the head and rotated so the current heading points up. A region is a four-neighbor connected component of free cells, food and the current tail cell; the head and other body cells are blocked. `s` is the cell's region size, `h` the largest head-adjacent region, and `K` the number of regions.

#### Cell token — all 66 features

{{CELL_FEATURES}}

The release-time fields are zero on non-snake cells. Missing chain links and region fields on blocked cells are zero. Region adjacency means that the component contains a passable cell next to the head or tail. The region features describe the current board, not a future path through a moving snake.

#### Snake token — all 21 features

{{SNAKE_FEATURES}}

#### Food token — all 24 features

{{FOOD_FEATURES}}

All Food features are zero when there is no food.

#### Global token — all 8 features

{{GLOBAL_FEATURES}}

### Results

{{E004_RESULTS}}

![E004: features and model architectures](docs/analysis/figures/e004_architectures.png)

At matching sample counts:

{{E004_SHARED}}

![E004: small model with and without teacher KL](docs/analysis/figures/e004_teacher_comparison.png)

![E004: teacher diagnostics](docs/analysis/figures/e004_teacher_diagnostics.png)

### Analysis

**Replacing the MLP with the tested CNN does not help.** With the original three input planes, CNN A reaches **75.23** and ends at **74.42**, below the MLP's best **110.65**.

**The richer entity representation and Transformer work much better.** Small reaches **125.55 at 275M**, exceeding the MLP's best score with far fewer samples. Its body-order, geometry, chain-link and connected-region features give the model explicit information about more than occupied cells.

**Increasing model size improves both speed and score.** Base reaches **125.12 at 125M**, while small is at **39.65** at the same point. At 400M, base scores **132.94** versus small's **100.17**. The larger token dimension, wider input MLP and extra encoder layer produce a much stronger result with the same input features.

**The no-teacher small model loses its high score later.** It falls from a peak of **125.55** to a last score of **71.62** at 575M. This is a sustained late regression rather than a single poor evaluation.

**Teacher KL substantially improves stability and the ending score.** At the same 575M point, D scores **120.82**, about **49.20** points higher than small without a teacher. D continues to 925M, reaches a best score of **131.90** and ends at **131.55**. Its curve has temporary drops, including **93.19 at 650M**, but recovers instead of showing the sustained late collapse seen without a teacher. The teacher acts as an anchor to previously successful behavior while the student continues learning.

The recorded base run has its last eval at 400M and training record at 419M. D's last eval is at 925M. Those are the endpoints shown here. Base also has sharp fluctuations in its late training-game scores, so high score and score retention remain separate properties.

### Author's hypothesis

I interpret these results as evidence that the earlier score plateau was partly a limitation of the representation and model's expressive capacity. Richer features expose body order and available space, while entity attention can combine information about distant parts of the snake and board according to the current state. Increasing model size may then make more of that information usable, raising the attainable score further.

My intuition is that the Transformer's flexible interactions support stronger adaptation to the current board, which I associate with a possible meta-learning advantage over the baseline MLP. That is a proposed explanation, rather than a capability measured directly here. The MLP-to-Transformer comparison changes both features and architecture; the small-to-base comparison keeps features fixed and provides the clearer evidence for a model-size benefit.

For teacher KL, I suspect that retaining a successful policy as an anchor limits destructive changes while allowing the student to improve. This offers an explanation for the better score retention, alongside the gains from representation and model size.

### Conclusion

**Transformer + richer basic features improves the score; increasing the model size improves it further. Adding teacher KL makes the small model much more stable, prevents the sustained late collapse in this run, and produces a higher ending score.** These results motivate testing teacher KL on the larger entity model next.

## Running and reproducing the experiments

The current E004 configs are in [config/experiments/feature_architecture](config/experiments/feature_architecture). To run D only:

```powershell
$env:LIBTORCH = 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\Lib\site-packages\torch'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sweep_feature_architecture.ps1 -StartAt D -Only
```

Change `D` to `B` or `C` for small or base. Existing checkpoints are protected by the sweep. `-AllowExistingCheckpoints` starts a fresh rerun rather than resuming it.

To regenerate the figures and README from the downloaded metric histories:

```powershell
& 'C:/Users/Liuluotu/Anaconda/anaconda3/envs/orbit_wars/python.exe' scripts/analyze_wandb_runs.py
& 'C:/Users/Liuluotu/Anaconda/anaconda3/envs/orbit_wars/python.exe' scripts/build_experiment_readme.py
```

Additional details: [per-run plots and results](docs/analysis/run-catalog.md), [result data](docs/analysis/data/run_results.csv), [technical guide](docs/technical-guide.md), and [metric definitions and calculation details](docs/analysis/methods.md).
