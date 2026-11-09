# Lightweight Transformer Encoder

**Goal:** Implement the user's selected encoder-only Transformer, with the snake-head token driving the actor and a learned critic token driving the value head, so training speed can be assessed.

**Architecture:** `model.version: transformer_v1` binds the existing three-plane features to one pre-norm encoder block: cell embedding 3→32, learned position embeddings, one appended critic token, two-head full self-attention, FFN 32→64→32 with GELU, residuals and final LayerNorm. Gather the head token using the original head plane. A shared 32→hidden_size ReLU projection processes the head and critic representations independently before the existing linear actor/value heads. No dropout, masks, downsampling, body-order features, or PPO changes.

**Spec:** User selected full encoder self-attention rather than the previously discussed query cross-attention, explicitly requesting a simple working version for speed testing. Existing MLP/CNN versions and checkpoints must remain compatible.

**Tech stack:** Rust, tch 0.20, existing LibTorch scaled-dot-product attention (dropout zero, noncausal). No new dependencies.

## Tasks

- [x] Add tests for version/config consistency, head-versus-critic readout, global attention gradients, checkpoint round trip, and real CLI PPO training on a rectangular board.
- [x] Implement Transformer module and model dispatch; preserve all existing tensor names for MLP/CNN.
- [x] Add experiment B with the same training/evaluation protocol as A, changing only model version and run identity. Existing alphabetical sweep discovers B automatically.
- [x] Document architecture, limitations, and launch command in English.
- [x] Run targeted and full tests, release build, launcher preview, and one real 81,920-sample rollout with batch 10,240. Report measured local speed with its scope; do not launch a full experiment or online logging.

## Validation focus

Readouts must select the actual head location independently for each batch member, critic must use the appended token, gradients must flow through both heads and attention, positional tensors must follow VarStore device/checkpoint behavior, and non-square boards and N=1 must work. Full attention's cost grows quadratically with board area; the short benchmark cannot establish eventual score.

## Verification record

The four new integration tests initially failed because the version and B config did not exist, then passed after implementation. The full suite passed (66 tests), including existing MLP/CNN and warm-start coverage. The strengthened batch-versus-single readout and measured parameter-count assertions also passed. Release compilation and the B-only launcher preview passed.

The actual release binary completed one 81,920-sample rollout using 1,280 environments, rollout length 64, mini-batch 10,240 and epoch 1. This short check used an isolated output directory under ignored `target/transformer-benchmark`, with W&B disabled and a one-rollout budget. End-to-end wall time was 34.146 seconds (about 2,399 samples/s), including process startup, the mandatory final 1,024-episode evaluation, and checkpoint saving. This is not a steady-state training-only throughput measurement. The runtime reported 23,109 parameters. No full training run or online W&B run was started.

Independent read-only review found no critical or important issues. A stale README sentence about the sweep launching only A was corrected because it now discovers A and B. No commits or pushes were requested for this implementation turn.

## Follow-up: equivalent single-layer readout pruning

The user found the speed too slow. Inspecting the forward pass showed that the one-layer encoder discarded 143 of 145 output tokens. Prune unused query rows and token-wise FFN outputs, retaining all original keys/values, parameters and readout heads. A literal full-token reference verifies both selected outputs and all parameter/input gradients (tolerating floating-point rounding); this characterization passed before and after pruning. This optimization is specific to the final encoder layer: do not prune intermediate representations if depth increases.

The identical short release benchmark improved from 34.146 to 22.645 seconds with default thread settings (about 3,618 samples/s). This includes final evaluation/startup/checkpoint overhead. Forcing OMP/MKL to four threads gave 24.996 seconds, so thread settings were not changed. A repeat with timestamped stdout measured 20.482 seconds for rollout+PPO and 1.742 seconds for the final evaluation (22.5 seconds total), approximately 4,000 training samples/s. Extrapolating training alone gives roughly 83 hours for 1.2B samples, before periodic evaluation overhead and throughput drift. Artifacts are under ignored target/transformer-benchmark; no formal checkpoints were touched. The read-only follow-up review found no issues and confirmed the single-layer equivalence and checkpoint compatibility.

## User update: B uses CUDA from scratch

B.yaml now selects CUDA while retaining the original three features, Transformer v1, batch10240/epoch1 and the existing schedules. It has no kickstarting block, initialization checkpoint or pretrained sample count. The previously accepted available-CUDA fallback remains applicable: this Windows runtime selects memory-efficient SDPA, not CUDA FlashAttention. Added launcher `-Only` to run or benchmark B without also scheduling C/D. Existing-checkpoint protection remains. Config tests explicitly verify CUDA and scratch/no-teacher semantics, normalizing the device difference in protocol comparisons with A/C. No full training or online W&B run is launched during validation.

Validation: the new CUDA assertion first failed on the old CPU config, then all17 targeted Transformer/feature/kickstarting tests passed after the config change. B-only launch preview selected exactly B, and its actual CUDA benchmark completed one warmup plus two timed rollouts with no teacher: collection291931samples/s, overall109079samples/s, memory_efficient BF16 attention for collection and update. The benchmark wrote only target/benchmarks/E004-B-cuda.json. No formal B checkpoint existed at validation time.
