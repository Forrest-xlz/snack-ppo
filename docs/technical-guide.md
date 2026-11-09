# Model and training reference

This document describes the current implementation and runnable configs. Historical runs can use older versions: consult the [experiment results](../README.md) and each run's logged config.

### A: Original features with CNN

CNN v1 uses three 3x3 convolutions with padding 1, ReLU after each, followed by a 256-unit projection and the existing linear policy/value heads:

```mermaid
flowchart TD
    X["Original flat features: N x 432"] --> R["Reshape: N x 3 x 12 x 12"]
    R --> C1["Conv 3 to 16, stride 2 + ReLU: N x 16 x 6 x 6"]
    C1 --> C2["Conv 16 to 32, stride 2 + ReLU: N x 32 x 3 x 3"]
    C2 --> C3["Conv 32 to 64, stride 1 + ReLU: N x 64 x 3 x 3"]
    C3 --> F["Flatten: N x 576"]
    F --> H["Linear 576 to 256 + ReLU"]
    H --> P["Policy: Linear 256 to 4 logits"]
    H --> V["Value: Linear 256 to 1"]
```

On the 12x12 board, CNN v1 has **172,581 parameters** versus **177,925** for the original MLP. This keeps parameter counts close, although compute costs and spatial structure differ. The final convolution's nominal receptive field is 15x15, and the dense projection combines all spatial positions. Dimensions are derived from the configured board size. There is no batch normalization, dropout, or change to the PPO optimizer/loss implementation.

The first configuration is [config/experiments/feature_architecture/A.yaml](../config/experiments/feature_architecture/A.yaml). It logs to W&B group `E004 feature architecture`, run `E004 feature architecture exp-A`, with checkpoints in `checkpoints/E004_feature_architecture_exp-A/`. Load a CNN checkpoint with the same model version, board dimensions, and hidden size; existing MLP weights cannot initialize this different architecture.

### B and C: Cell, Snake, Food and Global tokens

Both Transformers use the same **147-token** sequence:

| Type | Count | Raw input dimensions | Role |
| --- | ---: | ---: | --- |
| Cell | 144 | 66 | Occupancy, geometry, body order and static topology |
| Snake | 1 | 21 | State-conditioned actor readout |
| Food | 1 | 24 | Explicit target and target-region information |
| Global | 1 | 8 | State-conditioned critic readout |

The flat observation contains **9,557 values**: 66 channel-major cell planes, followed by the three context vectors (21+24+8). Context vectors are not duplicated across cells or padded to 66 dimensions. Four independent ReLU input MLPs project the token types to the selected encoder width.

#### Cell feature layout

Let A=W*H and L be snake length. For head-first segment index i, release time t=L-i: tail=1, head=L. This is the earliest release time under **no future growth**, not a prediction of a cell staying free forever. Non-snake cells have all release features zero.

| Channels | Features | Encoding |
| --- | --- | --- |
| 0-4 | Head, ordinary body, food, empty, tail | Exclusive one-hot; ordinary body excludes both head and tail |
| 5 | Release time | t/A |
| 6-22 | Exact near-term release | t=1..16 one-hot, plus t>16 overflow |
| 23-31 | Logarithmic release two-hot | Interpolation in log-space between [1,2,4,8,16,32,64,128,144] on 12x12 |
| 32-33 | Absolute x, y | Divide by max(W-1,1), max(H-1,1) |
| 34-35 | Ego x, y | Head-centered coordinates, rotated so heading is up; divide by max(max(W,H)-1,1) |
| 36-51 | Ego Fourier position | For f=1,2,4,8: sin/cos(pi*f*dx'/max(W,H)), then sin/cos(pi*f*dy'/max(W,H)) |
| 52-55 | Link toward the head | Ego up/down/left/right one-hot |
| 56-59 | Link toward the tail | Ego up/down/left/right one-hot |
| 60-61 (B and C) | Region size | size/A, log(1+size)/log(1+A) |
| 62-63 (B and C) | Region size versus length | min(size/L,2), size>=L |
| 64-65 (B and C) | Tail/head adjacency | Region contains a passable cell adjacent to tail/head |

The ego frame rotates geometry and chain directions consistently. The action space remains four absolute directions; the Snake token's absolute heading provides the mapping. Learned absolute cell positions are retained, so the full network is not rotation-invariant. Missing links at the head/tail are zero.

For other board sizes, release nodes are the powers 1..128 strictly below A followed by A, with unused slots zero. This retains nine input slots and avoids duplicated or unsorted endpoints. Length nodes follow the analogous filtering rule below. A dimension of one has absolute coordinate zero; rectangular ego coordinates use the longer side for normalization.

Regions use a four-neighbor graph containing unoccupied cells (**including food**) and the **current tail cell**, blocking the head and remaining body. A single flood fill labels components. Blocked cells have zero region features. Tail/head adjacency means a region contains a cell one step from the respective endpoint; it does not mean the endpoint itself is an ordinary empty cell. These features are a static, optimistic tail-open snapshot: region size, connection to the tail and size>=L are **not safety guarantees**. Moving/growing bodies and newly formed loops can invalidate static reasoning. No time-aware BFS, articulation points, action masks or reward changes are included.

#### Context feature layouts

| Token | Indices | Features |
| --- | --- | --- |
| Snake | 0-1 | L/A, log(L)/log(A) |
| Snake | 2-10 | Linear two-hot on [3,5,10,20,40,60,90,120,144]; on other boards filter nodes below A, append A, pad unused slots |
| Snake | 11-12 | (A-L)/A, log(1+A-L)/log(1+A) |
| Snake | 13-16 | Absolute heading: up/down/left/right one-hot |
| Snake | 17-18 | Absolute head x/y |
| Snake | 19-20 | Largest head-adjacent component: size/A, min(size/L,2); zero if none |
| Food | 0 | Food exists; all Food features are zero if absent |
| Food | 1-2 | Absolute food x/y |
| Food | 3-20 | Food ego position and Fourier encoding, same 18 values as Cell channels 34-51 |
| Food | 21-23 | Food component size/A, min(size/L,2), head-adjacent flag |
| Global | 0-2 | L/A, (A-L)/A, log(1+A-L)/log(1+A) |
| Global | 3-7 | Component count 1..4 one-hot plus >=5; zero components gives all zero |

The environment has no starvation timer, training episode time limit or per-step penalty, so there are no hunger, food-age or remaining-time inputs. The evaluation-only 5,000-step cap is unchanged. The original three binary planes are **not** a compatible prefix: body/tail occupancy changed, and heterogeneous context follows the planes. Legacy teachers are rejected by config validation for these versions; The active B/C experiments do not use a teacher.

#### Network and execution

The small input MLP is Linear(input,32), ReLU, Linear(32,16), ReLU; the base MLP is Linear(input,64), ReLU, Linear(64,32), ReLU. Each model has four learned type embeddings and learned absolute cell positions. Both use two attention heads, pre-norm encoder blocks, GELU FFNs with width twice the embedding dimension, residuals, no dropout and no attention mask.

Small has two width-16 blocks (head dimension 8; FFN16->32->16); base has three width-32 blocks (head dimension 16; FFN32->64->32). Intermediate layers update all 147 tokens. Only the final layer evaluates the Snake and Global query rows against all 147 updated keys/values. Final LayerNorm and a shared embedding-width->256 ReLU projection precede the actor's four logits and critic's scalar value. Actor and critic read Snake and Global respectively.

```mermaid
flowchart TD
    X["Shared Cell / Snake / Food / Global features"] --> B["B: four input to 32 to 16 MLPs + type / position embeddings"]
    X --> C["C: four input to 64 to 32 MLPs + type / position embeddings"]
    B --> B1["Encoder1: all147 tokens, width16"]
    B1 --> B2["Encoder2: Snake / Global queries, all147 keys and values"]
    C --> C1["Encoder1 and Encoder2: all147 tokens, width32"]
    C1 --> C2["Encoder3: Snake / Global queries, all147 keys and values"]
    B2 --> H["Final norm + projection to256"]
    C2 --> H
    H --> P["Snake actor: four logits"]
    H --> V["Global critic: one value"]
```

CUDA parameters, projections and PPO remain FP32; attention Q/K/V use BF16 SDPA. The installed Windows runtime selects **memory-efficient SDPA**, not CUDA FlashAttention. Startup reports each layer's actual backend and query count: small147/2, base147/147/2. The code loads `torch_cuda.dll` before selecting CUDA. The project uses tch0.24 with the installed PyTorch/LibTorch2.11 CUDA runtime; the older 2.7 CPU libraries are incompatible.

#### Memory and feature computation

The FP32 rollout observations occupy about **2.92 GiB of host RAM** at R=81,920. PPO uploads only the current mini-batch (about **0.365 GiB of input observations** at batch10,240), avoiding a complete rollout copy on GPU. Activations, gradients, workspaces and allocator reservations require additional GPU memory.

For board area A, length L<=A and cell channel count C=66, body indexing is O(A+L), four-neighbor connected components are O(A), and filling the complete features is O(A*C). Flood fill runs once per state. Board-size-keyed thread-local tables cache coordinates, Fourier values, release/length expansions and logarithms. Components compute their attributes once; scratch buffers are reused, and independent environments encode into disjoint output slices in parallel. These optimizations preserve feature values and PPO update rules.

Reference short benchmarks on the RTX5070Ti Laptop used one warmup and two measured full rollouts at the actual batch/rollout size. The small architecture measured **37,826 collection samples/s** and **16,847 samples/s including bootstrap/PPO**, with sampled whole-device peak6,794MiB from baseline1,624MiB. The base architecture previously measured **31,765 / 13,598 samples/s**, with peak10,834MiB from baseline1,142MiB. Background usage differed, so these are not strictly paired measurements or a guaranteed speedup ratio. They exclude periodic evaluation/checkpointing and do not establish score performance.

The whole 66-channel CPU encoding measured approximately4.6-5.2 microseconds/state on fixed 12x12 fixtures after caching. To repeat the CPU-only benchmark:

```powershell
cargo run --release --example benchmark_features -- target/benchmarks/features.json 10000
```

BF16 parameter storage alone saves little for these small models. Full BF16 mixed precision could reduce activations but is not enabled; only attention already uses BF16. See the [PyTorch mixed precision documentation](https://docs.pytorch.org/docs/2.11/amp.html).



## Sample-based training and stopping

The YAML uses environment samples for the training cap, checkpoint interval, evaluation interval, and W&B upload interval. Early stopping counts evaluations. The outer PPO iteration remains an internal loop counter. A run can exceed its configured sample cap by less than one rollout `R`, because an update consumes a complete rollout.

All six ready-to-run configurations are in [config/experiments/batch_rollout](../config/experiments/batch_rollout). Pass the desired experiment file explicitly. To launch group C:

```powershell
cargo run --release -- --config config/experiments/batch_rollout/C.yaml
```

Early stopping follows the deterministic evaluation mean score. Evaluation runs every **25M environment samples** on the same **1,024 held-out seeds**, starting at `seed + 1,000,003`; the training environments start at `seed` and use at most 1,280 distinct initial seeds in this study. Each new strict high resets the counter; **12 consecutive evaluations without a new high** stop training (approximately 300M further samples). The 1,200M sample cap remains the maximum. This rule is an experimental heuristic and may miss a later rebound; an early stop does not establish a performance ceiling.

Periodic and final checkpoints save `checkpoints/<safe_run_name>/samples-<total_samples>.ot` and `latest.ot` (for example, `checkpoints/E001_batch_rollout_scale_exp-C/latest.ot`). Periodic/final student checkpoints are not selected as best models. E004 D separately saves its best-eval self-teacher to `teacher.ot`. Evaluation uses 1,024 deterministic episodes with the same evaluation seeds at every checkpoint and across groups, capped at 5,000 steps per episode. Checkpoints are scheduled every 50M environment samples; evaluation is scheduled every 25M.

Evaluation uses the **same model and device as training** (CUDA for B/C/D). `evaluation.batch_size` defaults to **1,024** for old configurations and is explicit in B:

```yaml
evaluation:
  interval_samples: 25000000
  episodes: 1024
  batch_size: 1024
  deterministic: true
  max_steps_per_episode: 5000
```

All **1,024** held-out games start together. Each step encodes the active states into one observation batch, runs one GPU forward pass and copies the action vector to CPU once. Finished games leave the active batch and are never reset or counted twice; remaining games stop at their individual step limit. Thus **1,024** is the maximum inference batch size, which shrinks as episodes finish. If episodes exceed the configured batch size, subsequent cohorts cover the remaining episode IDs, including a partial final cohort. Every episode retains seed `seed + 1,000,003 + episode_index` and the same score/statistical weight. Training rollout/mini-batch sizes and early-stop criteria are unchanged. Periodic evaluation logs report the device and effective initial batch size. Batched floating-point kernels may change near-tied deterministic decisions; stochastic evaluation also changes random-draw ordering compared with serial evaluation.

To run groups A–F sequentially on Windows, preview the sweep, check the runtime dependencies without starting training, and then launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sweep_batch_rollout.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sweep_batch_rollout.ps1 -CheckRuntime
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sweep_batch_rollout.ps1
```

The script builds once and adds the bundled LibTorch DLL directory to `PATH` before starting the release binary. The `usage` line printed during `-CheckRuntime` is expected. The sweep stops if a group fails. Use `-StartAt D` to start from D through F. It refuses to run when a selected group's `latest.ot` already exists; use `-AllowExistingCheckpoints` only when intentionally rerunning that group, because its checkpoint files will be overwritten.

## W&B metrics

The six runs share W&B group `E001 batch rollout scale`, with run names `E001 batch rollout scale exp-A` through `exp-F`. All W&B curves use `total_samples` as the x-axis. There is no `recent_samples/*` group or iteration-axis duplicate.

- `train_samples/mean_episode_score`, `std_episode_score`, `min_episode_score`, and `max_episode_score` describe the **most recent 1,024 completed games**. This is a rolling episode window. Before the first completed game, these fields are omitted. `score_window_episode_count` shows how many games currently fill the window.
- Other `train_samples/*` diagnostics are pooled or sample-weighted over the roughly **1M environment samples since the previous upload**. `completed_episode_count` counts games completed during that upload interval, while `window_samples` reports its actual size. These counts do not define the rolling score window.
- `eval_samples/mean_score`, `std_score`, `min_score`, and `max_score` describe the fixed-seed evaluation episodes. `best_mean_score`, `stale_evaluations`, and `early_stop_triggered` show the evaluation-based stopping state. There is no separate `early_stop_samples/*` window group.

PPO diagnostics include positive policy `entropy` (not entropy loss), `policy_loss`, `value_loss`, `total_loss`, `critic_explained_variance`, `policy_approx_kl`, and `clip_fraction`. The `raw_adv_mean/std` values are GAE advantages **before** normalization; `return_mean/std` describe the value targets. `reverse_action_rate` counts requested reverse actions, which the environment ignores. Cumulative optimizer steps, wall time, and sample throughput help compare compute costs. Approximate KL and clip fraction are measured before each mini-batch optimizer step; they are not full-rollout post-update diagnostics.

## Setup (Windows)

1. Install the stable Rust toolchain with the MSVC target.
2. For GPU training, install an NVIDIA driver and a CUDA runtime compatible with LibTorch.
3. Set `LIBTORCH` to a PyTorch/LibTorch **2.11.0** installation compatible with `tch = 0.24`, and add its `lib` directory to `PATH`. For this machine, reuse `C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\Lib\site-packages\torch`. The project's `download-libtorch` feature remains available for a matching CPU runtime; the old 2.7 runtime is incompatible with the upgraded bindings.
4. Install the W&B Python SDK with `C:\Users\Liuluotu\Anaconda\anaconda3\python.exe -m pip install "wandb>=0.22.3"`. Configure `WANDB_API_KEY` or run `wandb login` for online logging.

If online W&B access needs an HTTP proxy, set `HTTP_PROXY` and `HTTPS_PROXY` in the same PowerShell window used to launch the sweep. Use the numeric HTTP or mixed port configured in your proxy application. The child PowerShell, Rust trainer, and Python bridge inherit these environment variables. A proxy URL must contain a real numeric port, not placeholder text. Alternatively, set the experiment YAML's `wandb.mode` to `offline` to keep local metrics for later synchronization.

Eating one food item gives reward `1.0`; all other outcomes, including death, give `0.0`.


## Visual inference

Copy an existing matching experiment YAML to a local inference YAML. Set `mode: infer`, `wandb.enabled: false`, and `inference.model_path` to the checkpoint to watch. Keep the checkpoint's model version, board dimensions and hidden size. Then run:

```powershell
$env:LIBTORCH = 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\Lib\site-packages\torch'
$env:PATH = "$env:LIBTORCH\lib;$env:PATH"
cargo run --release -- --config config/infer.yaml
```

Create `config/infer.yaml` as described above before using this command. Historical `transformer_v7` weights use the current `transformer_v1_small` architecture. The legacy name is not accepted by the current parser. Playback reads weights and never updates them.
