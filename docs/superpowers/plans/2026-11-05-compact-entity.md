# Compact entity Transformer

User approved two encoders and input MLP input->32->16. Implement inline with
test-driven development, retaining all existing features and PPO settings.

- B selects transformer_v7: EntityRegions66, MLP input->32->16, two width16
  encoders, two heads (head dimension8), FFN16->32->16, Snake/Global readouts.
- E selects transformer_v8: same compact architecture, EntityBase60.
- Existing v5/v6 remain width32/depth3 and load their existing checkpoints.
- Shared readout uses configured hidden_size256; CUDA attention stays BF16,
  parameters/PPO FP32, streamed observations and cached encoding remain active.
- New B-v7/E-v8 run names/output paths protect previous experiment files.

## Work

- [ ] Add compact-model forward/PPO/gradient/reload tests and observe missing
  selector failure. Update config protocol expectations to the new identities.
- [ ] Parameterize shared EncoderBlock dimension without changing legacy
  defaults/init order. Report actual dimension in backend selection probes.
- [ ] Bind v7/v8 to compact EntityTransformer; use blocks.len for final-query
  selection. Extend Float64 full-reference test to both architectures.
- [ ] Update B/E configs and English README, run full tests with CUDA required,
  formatting, release build, actual-size CUDA benchmark and isolated smoke.

No formal training, W&B/network settings, checkpoints outside target, commits
or pushes are requested. New parameter count must be read from the actual model.

## Completion evidence

- Missing compact version first failed the CPU integration test as expected.
- Shared-dimension encoder and old/compact Float64 pruning equivalence pass.
- CPU/CUDA compact integration run in separate processes, as actual launches
  do; initializing CPU serialization before loading CUDA hooks can otherwise
  cache CUDA-unavailable in the same Windows process. Production loader unchanged.
- Full cargo test --all-targets passed108 tests with CUDA required. All legacy
  feature/model selectors remain covered, as do old checkpoint readouts/shapes.
- Release build and B-only launcher preview passed. Fresh reviewer found no
  defects in dynamic QKV dimensions, FFNs, readouts or legacy initialization.
- Actual B-v7 CUDA benchmark: batch10240/R81920, 1 warmup + 2 timed rollouts;
  collection37826.11/s, overall16846.92/s, two memory-efficient BF16 layers.
- Whole-device sampled peak6794MiB from baseline1624MiB (other apps included).
- Main trainer completed one full update, eval1024/batch1024 (temporary20-step
  cap) and isolated target checkpoint; actual model_parameters=18533.
- English README updated; configs remain online and no formal run was started.
