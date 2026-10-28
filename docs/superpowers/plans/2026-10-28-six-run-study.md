# Six-Run Sample-Budget Study Implementation Plan

> **For agentic workers:** Implement each task with a failing test first.

**Goal:** Prepare six PPO Snake configurations with sample-based budgets, a rolling 1,024-episode training score, and configurable sample-window early stopping.

**Architecture:** Keep outer PPO iterations internal. Use total environment samples for the training cap, evaluation, checkpoints, and logging. Keep a rolling episode-score window for train charts and a separate non-overlapping sample window for early stopping. Save the best early-stop checkpoint and final checkpoint.

**Tech Stack:** Rust, tch-rs, serde YAML, W&B bridge, Cargo tests.

## Constraints

- Use one training seed, fixed learning rate, rollout length 64, and the same total sample cap across A–F.
- Remove iteration-based config keys and the `recent_samples/*` W&B group.
- The explicitly selected C config uses mini-batch 10,240, rollout 81,920, and 1,280 environments. All six configs live in `config/experiments/batch_rollout/` and use 30M-sample early-stopping windows.
- Remove historical experiment figures and their derived report artifacts.

## Tasks

- [x] Test rolling score, early-stop plateau/rebound, insufficient episode count, and sample-budget scheduling.
- [x] Implement the metric windows and early-stop state machine.
- [x] Replace iteration-based config fields with sample-based fields and validate the six configurations.
- [x] Update training loop, evaluation/checkpoint scheduling, W&B groups, and sample-based checkpoint names.
- [x] Add A–F runnable configs, require an explicit config path, and update README to describe the new study without historical results.
- [x] Remove historical figure/report artifacts; format, test, and review the diff.
