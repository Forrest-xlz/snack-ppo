# Entity token inputs implementation plan

> Execute inline using the approved token design and input -> 64 -> 32 MLPs.

**Goal:** implement Cell/Snake/Food/Global tokens for PPO without changing
the optimizer, reward, evaluation seeds, sample budget or stopping protocol.

**Approved design:** 144 cells + Snake + Food + Global on 12x12; independent
two-linear ReLU input MLPs; three width-32, two-head pre-norm encoders;
Snake actor and Global critic; final-layer selected-query optimization.

## Interfaces and constraints

- `transformer_v5` binds 60 cell channels and 21/24/8 context features.
- `transformer_v6` binds 66 cell channels and the same context features.
- Observations are channel-major cell planes followed by Snake, Food, Global.
  `observation_size` must include the 53 context values everywhere.
- Nine release two-hot slots use powers 1..128 below the board area and a
  final area endpoint; unused slots are zero on small boards. Length two-hot
  uses [3,5,10,20,40,60,90,120,A], filtering endpoints above A and padding.
- Ego coordinates rotate heading to up; square-board scale is W-1, with
  max(W,H)-1 for rectangular boards. Fourier denominator is max(W,H).
- Region graph includes unoccupied cells (food included) and current tail;
  blocks head and other body. Six cell features: size/A, log-size,
  min(size/L,2), size>=L, tail adjacency, head adjacency.
- Both versions share Snake/Food/Global region summaries; their comparison
  isolates providing region features at every cell, not all topology input.
- Occupancy is exclusive: head, ordinary body, food, empty, tail. This is
  not the original three-plane teacher prefix. Reject incompatible teachers.
- Keep all legacy selectors and checkpoint tensor names unchanged.
- B selects v6 with new run/output suffix; E selects v5. No checkpoint is
  overwritten and no full training or W&B connection is started for checks.

## Task 1: features and serialization

- [ ] Add failing tests for feature encoding/version selection, hand-derived
  ego coordinates, exclusive occupancy, release bins/two-hot, link rotation,
  region separation, head/tail adjacency, context offsets and terminal states.
- [ ] Run tests and observe missing-version failures.
- [ ] Add `src/entity_features.rs`; extend FeatureEncoding and Config mappings.
- [ ] Verify rectangular/single-row/full-board finiteness and legacy tests.

## Task 2: heterogeneous transformer

- [ ] Add failing actual forward/PPO/checkpoint tests for both versions.
- [ ] Add `src/entity_transformer.rs`, sharing encoder-block implementation.
  Four MLPs input->64->32, four type embeddings, learned cell positions;
  all 147 tokens in first two layers, Snake/Global final queries.
- [ ] Integrate ActorCritic and accurate per-layer backend reporting.
- [ ] Verify gradients through all input MLPs/blocks and selected-query
  equivalence against full-token Float64 reference.

## Task 3: experiment and end-to-end verification

- [ ] Update B (v6) and add E (v5); maintain protocol and isolated outputs.
- [ ] Update English README with exact layouts, approximate topology meaning,
  network, commands and B/E interpretation; preserve legacy documentation.
- [ ] Run complete cargo test with required CUDA tests, formatting and build.
- [ ] Run an isolated short CUDA train/eval smoke and actual-size CUDA
  benchmark if resources allow. Never change proxy/network/W&B settings.
- [ ] Review complete diff, record evidence and report runnable commands.

## Review focus

The five main risks are context offsets accidentally using channels*area,
incorrect head-frame link rotation, topology leaking into the wrong region,
unsafe reuse of legacy teacher prefixes, and pruning intermediate outputs.
The feature tests, PPO/CLI smoke and full-token reference cover these risks.

## Progress

- Approved in conversation; user specified input->64->32 for all input MLPs.
- Existing dirty feature-experiment branch retained; no commit requested.
- Task 1 complete: seven new tests first failed on missing version selection;
  feature and topology expectations now pass, including all four headings.
- Task 2 complete: four input->64->32 MLPs, context readouts and final-query
  equivalence pass. Both entity versions complete CPU and CUDA PPO/evaluation.
- Task 3 in progress: B/E configs and English README updated. Full cargo suite
  passed with CUDA checks required; release build and actual-size smoke pending.
- Release build passed. B-v6 actual-size CUDA benchmark completed 1 warmup +
  2 measured rollouts, batch10240/R81920: collection15630.74 samples/s,
  overall2752.69 samples/s, BF16 memory-efficient SDPA on all three layers.
  Observed whole-device memory11759MiB/12227MiB includes other applications.
- Fresh read-only reviewer found no concrete defects in features, offsets,
  rotations, regions, readouts, teacher compatibility, configs or documentation.
- Added release t16/t17 overflow and full12x12 endpoint coverage; focused test
  passes. Re-run full suite once after final smoke to include this boundary test.
- Complete: `cargo test --all-targets -- --test-threads=1` passed104 tests with
  `PPO_SNAKE_REQUIRE_CUDA_TESTS=1`; formatting and git diff whitespace checks pass.
- Actual-size main-trainer CUDA smoke passed: 81920 training samples, b10240,
  eval1024/batch1024 (temporary20-step cap), saved isolated target checkpoint.
  B-v6 reports56357 parameters. Formal B/E evaluation remains5000 steps.
- B/E launcher dry runs and read-only actual checkpoint-directory inspection
  passed. README contains measured speed and memory limits. No formal run,
  network setting change, commit or push was performed.
