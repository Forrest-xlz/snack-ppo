# PPO Snake (Rust)

This project uses a small Snake game to study scaling behavior in reinforcement learning without requiring high-end hardware. A 12×12 environment and compact PPO actor–critic make controlled CPU experiments practical. The current study varies PPO mini-batch size and the amount of environment experience collected before each policy refresh.

## Planned six-run study

All runs use training seed 42, 64 rollout steps per environment, a hidden size of 256, one PPO epoch, and a fixed learning rate of `3e-4`. Shared settings are `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `value_coef=1`, `entropy_coef=0.001`, and `max_grad_norm=0.5`. Each run has a **1,200M environment-sample cap** and the same early-stopping rule. Results will be added after the new runs finish.

`R = num_envs × rollout_steps` is the amount collected for one PPO update; `b = ppo.batch_size` is the optimization mini-batch size.

| Group | Mini-batch `b` | Rollout `R` | Environments | `R/b` | Comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 256 | 2,048 | 32 | 8 | Small joint-scaling point; compare with D at fixed `b` |
| B | 1,024 | 8,192 | 128 | 8 | Middle joint-scaling point; compare with D at fixed `R` and E at fixed `b` |
| C | 10,240 | 81,920 | 1,280 | 8 | Large joint-scaling point; compare with E at fixed `R` and F at fixed `b` |
| D | 256 | 8,192 | 128 | 32 | Mini-batch effect versus B at `R=8,192` |
| E | 1,024 | 81,920 | 1,280 | 80 | Mini-batch effect versus C at `R=81,920` |
| F | 10,240 | 20,480 | 320 | 2 | Rollout effect versus C at `b=10,240` |

A–B–C repeat joint scaling at constant `R/b=8`. D–B and E–C vary mini-batch size while holding `R` fixed. A–D, B–E, and F–C vary `R` while holding mini-batch size fixed. With one PPO epoch, changing `b` also changes optimizer steps per environment sample; these comparisons describe the combined PPO mini-batch setting. Keep the learning rate and all other parameters fixed for this study.

## Sample-based training and stopping

The YAML uses environment samples for the training cap, checkpoint interval, evaluation interval, W&B upload interval, and early-stopping window. The outer PPO iteration remains an internal loop counter. A run can exceed its configured sample cap by less than one rollout `R`, because an update consumes a complete rollout.

All six ready-to-run configurations are in [config/experiments/batch_rollout](config/experiments/batch_rollout). Pass the desired experiment file explicitly. To launch group C:

```powershell
cargo run --release -- --config config/experiments/batch_rollout/C.yaml
```

The configured early-stopping rule uses non-overlapping windows of **30M environment samples**. At each completed window, it computes the mean score over games that ended during that window. A window with fewer than **256** completed games is logged but does not change the stopping state. A new highest valid window mean resets the counter; **three consecutive valid windows without a new high** stop training. There is no score ceiling or minimum-improvement threshold. At most, training ends at the 1,200M sample cap; reaching that cap does not establish a performance plateau. The rule is an experimental heuristic, and its response to late rebounds still needs validation with the new runs.

The best early-stopping window saves `checkpoints/<safe_run_name>/best.ot` (for example, `checkpoints/E001_batch_rollout_scale_exp-C/best.ot`). Periodic and final checkpoints save `samples-<total_samples>.ot` and `latest.ot`. Evaluation uses 512 deterministic episodes with the same evaluation seeds at every checkpoint and across groups, capped at 5,000 steps per episode. Both evaluation and regular checkpoints are scheduled every 50M environment samples.

## W&B metrics

The six runs share W&B group `E001 batch rollout scale`, with run names `E001 batch rollout scale exp-A` through `exp-F`. All W&B curves use `total_samples` as the x-axis. There is no `recent_samples/*` group or iteration-axis duplicate.

- `train_samples/mean_episode_score`, `std_episode_score`, `min_episode_score`, and `max_episode_score` describe the **most recent 1,024 completed games**. This is a rolling episode window. Before the first completed game, these fields are omitted. `score_window_episode_count` shows how many games currently fill the window.
- Other `train_samples/*` diagnostics are pooled or sample-weighted over the roughly **1M environment samples since the previous upload**. `completed_episode_count` counts games completed during that upload interval, while `window_samples` reports its actual size. These counts do not define the rolling score window.
- `early_stop_samples/window_mean_score`, `best_mean_score`, `completed_episode_count`, and `stale_windows` show the separate 30M-sample stopping windows. `improved` and `triggered` are 0/1 flags.
- `eval_samples/mean_score`, `std_score`, `min_score`, and `max_score` describe the fixed-seed evaluation episodes.

PPO diagnostics include positive policy `entropy` (not entropy loss), `policy_loss`, `value_loss`, `total_loss`, `critic_explained_variance`, `policy_approx_kl`, and `clip_fraction`. The `raw_adv_mean/std` values are GAE advantages **before** normalization; `return_mean/std` describe the value targets. `reverse_action_rate` counts requested reverse actions, which the environment ignores. Cumulative optimizer steps, wall time, and sample throughput help compare compute costs. Approximate KL and clip fraction are measured before each mini-batch optimizer step; they are not full-rollout post-update diagnostics.

## Setup (Windows)

1. Install the stable Rust toolchain with the MSVC target.
2. For GPU training, install an NVIDIA driver and a CUDA runtime compatible with LibTorch.
3. Set `LIBTORCH` to a LibTorch 2.7 installation compatible with `tch = 0.20`, and add its `lib` directory to `PATH`. The project's `download-libtorch` feature is useful for CPU compilation and tests.
4. Install the W&B Python SDK with `C:\Users\Liuluotu\Anaconda\anaconda3\python.exe -m pip install "wandb>=0.22.3"`. Configure `WANDB_API_KEY` or run `wandb login` for online logging.

Eating one food item gives reward `1.0`; all other outcomes, including death, give `0.0`.

## Visual inference

Set `mode: infer` and choose `inference.model_path` in a YAML config, then run the program with that config. The window displays the snake, food, current score, and best score; close it to exit.

## Verification

```powershell
cargo fmt --all -- --check
cargo test --all-targets
```
