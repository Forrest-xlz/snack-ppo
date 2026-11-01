# PPO Epoch Scaling Implementation Plan

**Goal:** Run epochs 2, 4, and 8 at the completed E001 A/B/C scales, reusing epoch 1 results.

**Architecture:** Nine standalone YAML configurations derive from the corresponding E001 baseline. A sequential PowerShell sweep follows the existing batch-rollout launcher, including LibTorch discovery, checkpoint protection, dry runs, runtime checks, and restart selection.

**Tech Stack:** Rust configuration loader, YAML, Windows PowerShell.

**Spec:** The experiment matrix and shared settings agreed in the preceding conversation; implement on `exp/E002-epoch-explore`, created from `exp/E001-batch-rollout-explore`.

## Tasks

- [x] Rename E001 locally and create the E002 branch from the same commit.
- [x] Create `config/experiments/epoch_scale/{A,B,C}-ep{2,4,8}.yaml`. Change only epochs and experiment identity relative to each baseline. Use W&B group `E002 epoch scale`, run names `E002 epoch scale A-ep2`, and matching inference checkpoint paths.
- [x] Add `scripts/sweep_epoch_scale.ps1`. Order runs A/B/C at epoch 2, then 4, then 8. Support `-StartAt`, `-DryRun`, `-CheckRuntime`, and `-AllowExistingCheckpoints`. Never rerun E001 baselines by default.
- [x] Update the English README with the completed E001 status, E002 matrix, interpretation, and launch commands.
- [x] Validate configuration equality outside the study axis and identity, all nine launch paths, restart selection, checkpoint refusal, PowerShell syntax, and LibTorch runtime loading. Run existing configuration tests.

## Constraints and review

Keep seed 42, learning rate 3e-4, 1,200M sample cap, 25M evaluation interval, 1,024 held-out evaluation episodes, and patience 12. Keep the training implementation unchanged so the existing baselines remain comparable. Verify that no E002 checkpoint path overlaps E001 or another E002 run. Do not launch long training during verification.
