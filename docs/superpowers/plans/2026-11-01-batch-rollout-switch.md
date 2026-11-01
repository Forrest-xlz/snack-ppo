# E003 Batch/Rollout Switching Plan

**Goal:** Reuse nine E001 epoch-1 checkpoints for 12 scale-switch branches. Every branch loads weights only and creates a fresh Adam optimizer, environment set, RNG state, and early-stopping state.

**Spec:** The experiment design approved in this conversation: A at 100/300/550M branches to B/C; B at 200/500/850M branches to C; C at 150/450/850M branches to B. Target settings are A=(256,2048,32), B=(1024,8192,128), C=(10240,81920,1280), with 64 rollout steps, epochs=1, seed=42, LR=3e-4. Existing PPO update mathematics stay unchanged.

**Architecture:** Add optional weights-only initialization to the training configuration. `max_env_samples` remains the budget consumed by this invocation (300M for E003). `pretrained_env_samples` is the actual source checkpoint count; `total_samples` includes that count while `finetune_samples` starts at zero. Evaluation and checkpoint intervals count new samples. Save new checkpoints under isolated E003 run names, labeled with cumulative samples. Evaluate the loaded model before any training.

**Tech stack:** Rust/tch, YAML, W&B Python bridge, PowerShell.

## Tasks

- [x] Verify and commit E002, then create `exp/E003-batch-rollout-switch`.
- [x] Write failing tests for initialization validation, loaded weights, new/cumulative budgets, initial evaluation, and source checkpoint protection.
- [x] Implement weights-only initialization, fresh-optimizer construction, initial evaluation, and sample-axis logging. Keep old configs valid and baseline behavior unchanged.
- [x] Generate 12 standalone E003 configs referencing actual existing source checkpoints. Add a guarded sequential sweep with preview, runtime check, restart selection, and overwrite protection.
- [x] Document the matrix, reset semantics, 300M additional budget, evaluation/early stopping, metrics, proxy setup, and launch commands in the English README.
- [x] Run the full Rust suite, validate all configs/sweep selections, and perform an offline bridge check. Review the branch and commit verified E003 work.

## Constraints and decisions

- Retain early stopping enabled, evaluation every 25M, 1024 held-out episodes, patience=12. Initialize its best score with the initial evaluation. A 300M hard additional-sample cap applies even if the score is still improving; no extra disabling flag is needed.
- All switching branches reset Adam. Source weights are read-only; output directories must not contain the initialization checkpoint.
- Existing `optimizer_steps_total` and wall time describe only the new branch, without invented historical costs.
- Keep W&B group `E003 batch rollout switch`; names identify source, nominal source time, and target, e.g. `E003 batch rollout switch A-100M-to-B`.
- Baseline source paths contain the actual full-rollout sample count, not the nominal rounded time.

## Progress

E002 commit: `776195a`. Baseline suite: 52 tests passed; E002 sweep preview passed.

Implementation choice: work in the explicitly requested new branch in the existing checkout. Do not launch the long experiment sweep during verification.

Verification: four warm-start tests failed on unsupported initialization fields, then passed after implementation. The matrix test failed before configs existed and caught a source-path substitution mistake during generation; all 21 sources were corrected before training. Full Rust suite: 57 passed. All 21 sweep restart positions, nine source/stage subsets, invalid selections, checkpoint guard, format check, and LibTorch runtime check passed. Real offline W&B history preserved both sample axes and the initial evaluation. A one-rollout release smoke loaded the actual A-100M checkpoint into B settings, wrote the expected cumulative checkpoint, and left the source SHA256 unchanged. Independent read-only review found no important issues. No long training or online W&B runs were launched.

Scope correction: user requested removal of all A-to-A, B-to-B and C-to-C restart controls. Keep only 12 switches, with a combined additional-sample cap of approximately 3.6B. Prior 21-branch verification above records the original implementation; the reduced matrix is verified separately. Baseline comparisons now include both restart and scale effects.

Reduced-matrix verification: full Rust suite 57 passed; all 12 sweep restart positions, nine source/stage subsets, and rejection of removed control names passed.
