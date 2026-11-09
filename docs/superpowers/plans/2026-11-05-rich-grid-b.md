# B: 31 Features, Input MLP, Three Encoders

**User-approved scope:** B uses the proposed19 raw grid features plus4 wall distances and8 chain direction indicators, a31→32→32 per-token MLP, and3 Transformer encoder blocks. Retain CUDA scratch training, embedding32, heads2, FFN64, existing actor/critic readouts and PPO/eval settings (evalbatch1024). Interpret the user's MLP request as the token input module, preserving the explicitly requested Transformer stack. ReLU follows each MLP linear layer.

**Single selector:** transformer_v4 binds31 features, the input MLP, anddepth3. Earlier versions and C/D remain unchanged. The first two blocks update all145tokens; only the final block prunes unused output rows.

**Feature layout:** channel-major planes:0head,1body excluding head,2food,3empty;4x/(W−1),5y/(H−1);6relative-head x/(W−1),7relative-head y/(H−1);8tail;9t/N,10log(1+t)/log(1+N);11–18 indicators t=1..8;19–22left/right/up/down boundary distances divided byW−1/H−1;23–26direction to previous segment toward head,27–30direction to next segment toward tail, each ordered Up/Down/Left/Right. Snake headindexi0 gives t=L−i, so tailt1; non-snake t0. Missing chain links are allzero. Singleton-coordinate denominators clamp to1; all non-relative features remain[0,1]. Wall distance measures cells to the boundary row/column (boundary=0). No BFS, extra tokens, action mask, or reward changes.

**Existing results:** An old B checkpoint exists. Preserve its files and original run identity. New B uses run `E004 feature architecture exp-B-v4`, output `checkpoints/E004_feature_architecture_exp-B-v4/`. Make launcher guards read the actual configured output directory through a read-only CLI metadata option, avoiding hardcoded per-letter paths. Old3/7-feature model versions remain loadable. The new31-plane layout preserves only the original3-plane prefix; reject seven-plane teacher prefix reuse rather than silently misencoding teacher inputs.

## Tasks

- [x] Add failing tests for31 features, release times/growth, direction endpoints, coordinate/wall edge cases, 3-layer/inputMLP architecture, CLI/checkpoint roundtrip, teacher prefix compatibility, and configured output-path inspection.
- [x] Implement rich encoding and version dispatch, inputMLP and3-block binding without changing older versions.
- [x] Preserve old B outputs with versioned identity and dynamic launcher checkpoint guards; document exact semantics and commands in English.
- [x] Run full tests, real CUDA full-batch benchmark/eval smoke, read-only review and verification; no formal training, online logging or old-checkpoint mutation.

## Verification

New tests first failed for unavailable v4/metadata CLI, and the old B architecture failed the new42149-parameter expectation. All93 tests passed after implementation with CUDA validation required. Rich-feature tests cover original3-plane prefix, relative geometry, release linear/log/exact bins, wall directions, missing chain endpoints and spatial-vs-chain neighbors, single-row finite inputs, movement and growth holding the tail, three-block/inputMLP gradients and checkpoint reload. A7-plane teacher is rejected for this new layout, while the original3-plane expert remains compatible.

Expanded randomized full-token reference checks exposed pre-existing FP32 reduction-order sensitivity, including1.22e-4 gradient difference at a tensor maximum226.34. Algebraic pruning equivalence is now tested in Float64 with strict1e-8relative/1e-10absolute tolerance, for3/7/31 channels and1/2/3blocks, including the new MLP. Twelve random trials passed. CPU attention preserves the input dtype (normal productionFP32 unchanged); CUDA remainsBF16 attention. Actual FP32/BF16 training paths are separately exercised by CLI and GPU checks.

Actual B CUDA benchmark retained1280envs, rollout64, batch10240, epoch1, no teacher: warmup1+timed2rollouts yielded35048.54collection samples/s and9668.30overall samples/s. Both intermediate layers used145queries and final layer2queries, with memory-efficient BF16 SDPA. It completed without allocation failure; each81920-sample31-plane rawbuffer is1.36GiB before extra copies/activations. Report: target/benchmarks/E004-B-cuda.json.

The actual main trainer also completed one full81920-sample CUDA rollout and1,024-episode CUDA evaluation (batch1024, cap20for the short smoke), printedTransformerV4/42149parameters and saved a checkpoint only under target/rich-grid-cuda-smoke. No formal B run/checkpoint or online W&B logging was started. Metadata inspection resolves checkpoints/E004_feature_architecture_exp-B-v4 and does not create directories. B-only launch preview, release build and format/diff checks passed. Independent read-only review found no defects in features, version/MLP dispatch, compatibility, pruning, metadata guard or docs. No commit/push was performed.
