# Batched Evaluation

**User request:** Set evaluation inference batch size to1024 rather than one state at a time.

**Design:** Add evaluation.batch_size with default1024 and an explicit setting in B. Evaluate up to batch_size held-out episodes simultaneously, each using the original seed+episode_index and per-episode step limit. One model forward/CPU action-vector transfer per step for all active games; remove completed games without resetting them or counting additional episodes. Never change the model device: evaluation uses the training device (CUDA for B). A final partial cohort and shrinking active batch are supported. Report identical score definitions and do not change training rollouts or teacher use.

**Checks:** Compare batched deterministic trajectories/scores with an independent serial reference using controlled policies; cover mixed termination, truncation, non-divisible episode counts, feature encodings, no accidental resets, repeatability and invalidbatch0. Preserve default loading of old configs. Validate a real1024-episode CUDA eval with current B without online logging or a long training run. Floating-point batch/device effects can affect near-tied actions; stochastic evaluation's sampling order changes under batching.

## Tasks

- [x] Write failing config/default/validation and batching-semantics tests.
- [x] Implement active-episode cohorts and vectorized model inference/transfer; keep episode IDs and held-out seeds stable.
- [x] Update B config and English README; run focused/full tests, release build and actual CUDA1024-episode smoke.
- [x] Read-only review, record checks and provide concise user instructions. No formal training or online logging.

## Verification

The default/zero-batch test first failed because the setting was absent. Batching tests failed before the new field/helper existed, then passed after implementation. The full87-test suite passed with CUDA tests required. Independent serial-reference tests cover both three/seven-plane inputs, stable seeds/episode IDs, mixed endings, exact total step counts, batches1/5/1024, truncation and partial cohorts. A1024-game one-step case verifies exactly one predict call rather than1024single-state calls. Config validation rejects batch0 and old configs load default1024.

The release trainer completed an actual B CUDA smoke: one81920-sample training rollout, followed by1024held-out episodes with initialbatch1024, evaluationcap20steps for the short check, and isolated checkpoint output under target/batched-eval-cuda-smoke. Console confirmed `evaluation iteration=1 device=Cuda(0) batch_size=1024 episodes=1024`. No online logging or formal output was used. Read-only review found no correctness defects; its minor log/documentation discrepancy was fixed by specifying periodic evaluation logs in README. Release compilation and format/diff checks passed.
