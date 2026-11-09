# B: Seven Features, Two Encoder Layers

**User request:** Change B to seven raw cell features and two encoder layers, keeping CUDA scratch training.

**Design:** New single selector `model.version: transformer_v3` binds the existing seven-plane geometry/order encoding to two pre-norm encoder blocks. Preserve embedding32, heads2, FFN64, final norm/readout projection256, head-token actor, critic-token value, learned positions, CUDA BF16 attention and all PPO/eval settings. Preserve transformer_v1/v2 and their checkpoint names/shapes for old runs and C/D.

Block1 updates every cell and critic token using full self-attention; block2 uses all block1 outputs as K/V, and computes only the consumed head/critic query rows plus their token-wise FFN. Do not prune block1: that would change the network. Preserve first-block tensor names `encoder.*`; use `encoder2.*` for block2. Report both attention backend shapes. B remains no-teacher and no weight initialization.

## Tasks

- [x] Update B expectation and add two-layer forward/gradient/checkpoint/CLI tests; observe failures before implementation.
- [x] Factor token-wise encoder blocks without changing single-layer versions; add v3/depth dispatch and B config.
- [x] Compare selected readouts and all gradients with a literal full-token two-layer reference, then run full suite and full-batch CUDA benchmark for memory/runtime validation.
- [x] Update English README and provide the B-only launch command. No long training or online W&B run during checks.

## Validation

The updated B expectation first failed at old23109 versus expected31781 parameters; the new CLI test failed because transformer_v3 was missing. After implementation, the full83-test suite passed with CUDA tests required. Full-token reference comparisons pass for three-feature/one-layer, seven-feature/one-layer and seven-feature/two-layer configurations, including input and every parameter gradient. Rectangular-board CLI train/eval/reload tests confirm gradients reach both blocks. Old v1/v2 tests remain green, with initialization order and tensor names retained.

The release B CUDA benchmark completed its actual1280-env/64-step rollout, batch10240/epoch1 configuration without memory failure, using1warmup+2timed rollouts. Sampling105716samples/s; collection+bootstrap+PPO38615samples/s. Both layers use memory-efficient BF16 SDPA, with145first-layer queries and2final-layer queries. Runtime reports transformer_v3 and no teacher. The current report is target/benchmarks/E004-B-cuda.json. This short-run throughput is not an end-to-end training-time guarantee.

Format/diff checks and B-only launch preview passed. Independent read-only review found no defects in multi-layer attention, gradients, pruning, version selection, legacy compatibility or docs. No formal training, online logging, commit or push was performed.
