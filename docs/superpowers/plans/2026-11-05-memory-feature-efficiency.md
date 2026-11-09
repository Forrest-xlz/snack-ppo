# PPO observation memory and feature efficiency

Implement inline under writing-plans / test-driven-development. The user reports
insufficient VRAM and asks about BF16 and feature-computation complexity.

## Design

Keep model versions, FP32 parameters/losses, BF16 SDPA, feature values and all
experiment settings unchanged. Reduce memory before changing numeric precision.
The complete rollout stays in CPU memory; gather only a shuffled mini-batch for
each GPU forward. Cache teacher targets in bounded chunks when enabled.

Feature tables cache deterministic geometry/Fourier/release/length expansions
per board size and thread. Region flood fill remains O(A), with component
attributes computed once per region. Reuse scratch allocations, and encode
independent environments in parallel into disjoint preallocated output slices.

## Steps

- [x] Save pre-change feature outputs and CPU timings for L3/73/113.
- [x] Write and observe failing tests for observation batch row order/tail,
  destination-buffer encoding equivalence, and parallel-versus-serial encoding.
- [x] Stream PPO observations and teacher inputs in mini-batches. Preserve
  shuffle/normalization, optimizer step count and losses. Cover teacher path
  and zero weight using existing tests; compare update against prior reference.
- [x] Cache feature expansions without changing formulas; move region attributes
  out of the cell loop; parallelize environment encoding without changing RNG.
- [x] Compare saved feature values and meaningful PPO results; full test suite
  with CUDA required, format/release build, actual-size CUDA benchmark and
  isolated train/eval smoke. Document timings and VRAM observations honestly.

## Main risks

Ordering, tail mini-batches, teacher label alignment, cache board-size keys,
thread-local lifetime, exact formula changes, and GPU benchmark contention.
No network settings, online W&B run, full training, or automatic commit.

## Verification evidence

- Missing batch/destination APIs first failed in new tests, then passed.
- Cached feature outputs for L3/73/113 exactly match pre-change arrays.
- Feature CPU timing: 11.26/11.03/11.01us before, 4.61/4.96/5.18us after.
- Pre/post CPU B/C/D short training checkpoints have max parameter error0,
  including active teacher loss. Legacy and CUDA integration tests pass.
- Actual B-v6 CUDA benchmark: 1 warmup + 2 timed rollouts, b10240/R81920;
  collection31764.55/s, overall13597.79/s, memory-efficient BF16 attention.
- Whole-device memory sampled every0.5s: baseline1142MiB, peak10834MiB;
  includes other applications/caching, not exact live tensor allocation.
- Main CUDA trainer completed full81920-sample update, eval1024/batch1024
  with temporary20-step cap, and isolated checkpoint save. Formal cfg unchanged.
- Fresh read-only reviewer found no defects in row alignment, teacher chunks,
  caches, scratch reset, region semantics or parallel writes.

- Complete: 106 tests passed with CUDA required; formatting, release build and whitespace checks passed.
