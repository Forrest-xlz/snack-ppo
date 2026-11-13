# Remove retired Transformer versions

> **For agentic workers:** Use superpowers:executing-plans to implement this plan.

**Goal:** Keep only the current MLP, CNN, entity-small and entity-base models, and finish separating batch and rollout documentation.

**Architecture:** Remove numeric Transformer variants v1–v5/v8 and their 7/31/60-channel encodings, retaining the shared attention block used by small/base. Preserve current checkpoint parameter names, shapes, feature values, CUDA SDPA and teacher KL. Keep recorded experiment evidence as an archive, but remove the retired model from the current README and current architecture comparison.

**Spec:** User requests of 2026-11-13: split batch/rollout discussion; remove the historical transformer_v3 and other unused versions from code.

**Tech stack:** Rust/tch, YAML, Python reporting, Markdown/Mermaid.

## Constraints and review focus
- Supported versions: `mlp_v1`, `cnn_v1`, `transformer_v1_small`, `transformer_v1_base`; no numeric-name aliases.
- E001/E002/E003 and active E004 A–D configurations must continue to load.
- Current small/base checkpoints and 66/21/24/8 features must remain compatible.
- Fixed-expert kickstarting may use matching active feature encodings; self-teacher KL stays unchanged.
- No deletion of checkpoints, raw W&B histories, or historical experiment branches.
- README documents E001.1 batch and E001.2 rollout independently and lists only active architectures.

## Tasks
- [x] Extend the retired-version config test; observe its failure before deleting implementation.
- [x] Save active feature fixtures; delete retired dispatch, encodings and legacy encoder while keeping the current attention block.
- [x] Remove retired-only tests and adapt shared runtime tests to active models; retain active B/C protocol tests.
- [x] Remove retired architecture sections from generated README and comparison figures; retain archived evidence with explicit scope.
- [x] Run full CPU/CUDA suite, compare feature fixtures, release build, report/link checks, and final read-only review.

No training algorithm, optimizer behavior, current model sizes or user training artifacts are to change.

## Verification evidence

- Retired-version rejection test failed before removal, then passed for all v1-v8 numeric names.
- Four pre-cleanup checkpoints load with identical serialized features, actor logits and critic outputs. Comparison applies the same JSON serialization roundtrip to both sides.
- Release feature fixtures match exactly for snake lengths 3, 73 and 113 (9,557 floats per state).
- CUDA-required all-target suite: 104 tests passed; cargo fmt --check and release build passed.
- Report scripts rebuilt; all Markdown file links and 119 entity-feature rows checked; all 44 archive PNGs decoded. E004 comparison visually inspected; nine current Mermaid diagrams retained.
- Fresh read-only reviewer found no actionable issues.

Ruling: Preserve all 32 original run archives and derived evidence unchanged; show 31 runs in current README/catalog after omitting the retired architecture. This keeps historical results inspectable without advertising obsolete runtime support.
