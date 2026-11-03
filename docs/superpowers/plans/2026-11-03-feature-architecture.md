# E004 Feature and Architecture Experiments

**Goal:** Configure feature and architecture combinations with one version, and implement experiment A using the existing board features with a CNN instead of the baseline MLP.

**Spec:** User requests a new branch from E003, batch 10240, epoch 1, lettered experiments beginning with A, and otherwise unchanged features/training settings.

**Architecture:** `model.version` selects `mlp_v1` or `cnn_v1`, binding both the head/body/food encoder and the corresponding architecture. An omitted version defaults to the existing encoding and MLP so old configurations and checkpoints remain compatible. Every training, evaluation, and inference path uses the same selectors.

**CNN v1 (implementation choice):** Reshape flat observations to NCHW; Conv3x3 3->16 stride2 padding1, ReLU; Conv3x3 16->32 stride2 padding1, ReLU; Conv3x3 32->64 stride1 padding1, ReLU; flatten; Linear to hidden_size=256, ReLU; existing policy/value heads. On12x12 this has172581 parameters versus177925 for baseline MLP. Spatial dimensions are derived from configured board dimensions, not hardcoded12.

## Tasks

- [x] Create `exp/E004-feature-model-explore` from the clean E003 branch.
- [x] Add failing tests for version parsing/defaults, real CNN training/checkpoint round trip, and unchanged experiment settings.
- [x] Implement feature/model version dispatch and route collection, training, evaluation, and inference through it. Preserve existing MLP tensor names and public constructor.
- [x] Add `config/experiments/feature_architecture/A.yaml` derived from E001 C. Start from scratch with hidden_size256, batch10240, epochs1, rollout81920,1280 environments, LR3e-4, seed42, 1200M max samples, eval25M/1024, patience12.
- [x] Add a runtime-safe launcher/sweep that currently selects A and can discover later lettered configs. Document selectors, architecture dimensions, parameter counts, checkpoint compatibility, and launch commands in the English README.
- [x] Run the full suite, CLI/runtime smoke and launch preview, review the completed diff, and commit verified work.

## Constraints

No new body-order/direction/tail features, reward changes, PPO mathematics changes, learning-rate schedules, gradient-clipping changes, or evaluation/stopping changes. Keep current MLP checkpoint loading functional. Unknown version strings must fail instead of silently falling back. CNN checkpoints require the matching model/feature versions and dimensions.

Progress: new branch created; no long training or online W&B runs should be launched during implementation checks.

Verification: initial tests failed for missing config/version support, then passed. Full suite62 passed, including original MLP/checkpoint tests, rectangular CNN training, exact board_v1 encoding, checkpoint round-trip logits/values, selector defaults/unknown-version rejection, and A-versus-C settings equality. Launcher discovery, lowercase input, rejection of nonexistent B, checkpoint refusal, format and LibTorch runtime checks passed. A release smoke kept the actual1280 environments,64 rollout steps,batch10240,epoch1 and trained one81920-sample rollout, confirming172581 parameters and a saved CNN checkpoint; only the test budget/evaluation count/output/online logging were reduced. Independent read-only review found no important issues. No GUI validation or long training was performed.

User correction: use only model.version; remove the independent features.version configuration. Internal FeatureEncoding is selected by ModelVersion, and the CNN structure remains unchanged.

Single-version verification: updated tests first failed because a separate feature selector was still serialized. After consolidating selection, all62 Rust tests, the A launch preview, and the release LibTorch runtime check passed. Separate feature configuration is rejected; model.version selects encoding in every execution path.
