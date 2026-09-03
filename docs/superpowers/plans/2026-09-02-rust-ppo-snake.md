# Rust PPO Snake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust PPO Snake project that trains with batched NVIDIA GPU computation and parallel CPU environments, and renders model inference in a graphical window using YAML-only configuration.

**Architecture:** A pure-Rust Snake environment feeds time-major rollout buffers collected by a vector environment. `tch-rs` performs batched actor-critic inference and PPO optimization, Rayon advances environments on CPU cores, and macroquad owns the inference window.

**Tech Stack:** Rust 2021, tch/LibTorch, rayon, macroquad, serde_yaml, rand_chacha, anyhow, thiserror.

**Spec:** `docs/superpowers/specs/2026-09-02-rust-ppo-snake-design.md`

## Global Constraints

- The executable reads only `config/config.yaml`; no CLI configuration is accepted.
- `device: cuda` must fail clearly when CUDA is unavailable and must not fall back to CPU.
- Every PPO iteration collects exactly `num_envs * rollout_steps` transitions.
- Reward is exactly `1.0` on eating food and `0.0` otherwise, including death.
- Neural-network work runs on the selected tch device; environment step/reset uses Rayon CPU parallelism.
- Inference renders a native graphical window and loads the configured checkpoint.
- New production behavior follows red-green-refactor TDD.

---

### Task 1: Project scaffold and validated YAML configuration

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/config.rs`
- Create: `config/config.yaml`
- Test: `tests/config_test.rs`

**Interfaces:**
- Produces: `Config::load_default() -> anyhow::Result<Config>` and `Config::validate() -> Result<(), ConfigError>`.
- Produces: public `Mode`, `DeviceConfig`, `EnvironmentConfig`, `ModelConfig`, `PpoConfig`, `TrainingConfig`, and `InferenceConfig` serde types.

- [ ] **Step 1: Write failing config tests**

Create tests that deserialize the complete YAML shown in the spec, assert every required field, reject zero `num_envs`, reject zero `rollout_steps`, reject non-positive learning rate, reject invalid probability/range values, and reject a board smaller than the initial snake.

- [ ] **Step 2: Verify RED**

Run: `cargo test --test config_test`

Expected: compilation fails because `ppo_snake::config` does not exist.

- [ ] **Step 3: Implement minimal configuration module**

Add serde data types, exact validation errors containing field names, and fixed-path loading from `config/config.yaml`. Add dependencies with compatible pinned major/minor versions and expose modules from `src/lib.rs`.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test --test config_test`

Expected: all configuration tests pass.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/lib.rs src/config.rs config/config.yaml tests/config_test.rs
git commit -m "feat: add validated yaml configuration"
```

### Task 2: Deterministic Snake environment

**Files:**
- Create: `src/env.rs`
- Test: `tests/env_test.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `Point { x: usize, y: usize }`, `Direction`, `Action`, `StepResult { reward: f32, done: bool, score: u32 }`.
- Produces: `SnakeEnv::new(width, height, seed)`, `reset()`, `step(Action)`, `observation() -> Vec<f32>`, and read-only accessors needed by the renderer.
- Produces: a test-only deterministic state constructor under `#[cfg(test)]` or a public documented `from_state` validator usable by integration tests.

- [ ] **Step 1: Write failing movement and reward tests**

Test normal motion, opposite-action rejection, eating yields exactly `1.0`, non-eating yields `0.0`, growth after food, wall collision, self collision, and death reward `0.0`.

- [ ] **Step 2: Verify RED**

Run: `cargo test --test env_test`

Expected: compilation fails because environment types are missing.

- [ ] **Step 3: Implement movement and collision rules**

Use `VecDeque<Point>` for the snake, a per-environment `ChaCha8Rng`, and explicit direction-opposite checks. Ensure moving into the old tail is legal when the tail moves away.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test --test env_test`

Expected: movement and reward tests pass.

- [ ] **Step 5: Add failing food and observation tests**

Test that food never overlaps the snake, a full board ends the episode, and observation has length `3 * width * height` with exactly one head cell, body cells, and one food cell when food exists.

- [ ] **Step 6: Verify RED, implement, and verify GREEN**

Run before and after implementation: `cargo test --test env_test`

Expected before: new assertions fail; expected after: all environment tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs src/env.rs tests/env_test.rs
git commit -m "feat: implement snake environment"
```

### Task 3: Actor-critic network and action distribution

**Files:**
- Create: `src/model.rs`
- Test: `tests/model_test.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `ActorCritic::new(path, input_size, hidden_size)`.
- Produces: `forward(&Tensor) -> (Tensor, Tensor)`, `actions_and_values(&Tensor, deterministic) -> PolicyOutput`, and `evaluate_actions(&Tensor, &Tensor) -> EvaluationOutput`.
- `PolicyOutput` contains actions, log probabilities, values; `EvaluationOutput` contains log probabilities, entropy, values.

- [ ] **Step 1: Write failing shape and deterministic-action tests**

Use CPU tensors. Assert logits-derived actions have `[batch]`, values have `[batch]`, log probabilities are finite, deterministic actions match logits argmax, and evaluation returns `[batch]` values.

- [ ] **Step 2: Verify RED**

Run: `cargo test --test model_test`

Expected: compilation fails because model types are missing.

- [ ] **Step 3: Implement minimal actor-critic**

Create two shared linear/ReLU layers and separate policy/value heads. Compute categorical sampling with multinomial, selected log-probabilities with gather, and entropy from log-softmax/softmax.

- [ ] **Step 4: Verify GREEN and commit**

Run: `cargo test --test model_test`

Expected: all model tests pass.

```bash
git add src/lib.rs src/model.rs tests/model_test.rs
git commit -m "feat: add actor critic model"
```

### Task 4: Parallel fixed-size rollout collection

**Files:**
- Create: `src/rollout.rs`
- Test: `tests/rollout_test.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `VectorEnv::new(config, base_seed)`, `observations()`, and `step(&[i64]) -> Vec<EnvTransition>`.
- Produces: `RolloutBuffer` with time-major observations, actions, rewards, dones, values, log probabilities, episode scores, plus `len()` and `flatten(device)`.
- Produces: `collect_rollout(model, vector_env, rollout_steps, device) -> anyhow::Result<RolloutBuffer>`.

- [ ] **Step 1: Write failing exact-count and reset-boundary tests**

With CPU and tiny environments, assert buffer length equals `num_envs * rollout_steps`, all tensor-compatible vectors share that length, and environments continue producing transitions after terminal reset.

- [ ] **Step 2: Verify RED**

Run: `cargo test --test rollout_test`

Expected: compilation fails because rollout interfaces are missing.

- [ ] **Step 3: Implement vector environment and collection**

Build observations in environment index order, run one batched model forward per time step, transfer only actions/value/log-prob outputs to CPU, and call `par_iter_mut().zip(actions)` for environment steps. Reset terminal environments after preserving their `done` flag and terminal score.

- [ ] **Step 4: Verify GREEN and commit**

Run: `cargo test --test rollout_test`

Expected: exact-count and reset tests pass.

```bash
git add src/lib.rs src/rollout.rs tests/rollout_test.rs
git commit -m "feat: collect parallel fixed-size rollouts"
```

### Task 5: GAE, mini-batches, and PPO update

**Files:**
- Create: `src/ppo.rs`
- Test: `tests/ppo_test.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `compute_gae(rewards, dones, values, last_values, steps, envs, gamma, lambda) -> (Vec<f32>, Vec<f32>)`.
- Produces: `shuffled_minibatches(sample_count, batch_size, rng) -> Vec<Vec<usize>>`.
- Produces: `PpoTrainer::new(var_store, config)`, `update(model, rollout, last_values) -> anyhow::Result<PpoMetrics>`.

- [ ] **Step 1: Write failing GAE tests**

Use hand-calculated one- and two-step arrays to prove terminal masking and non-terminal bootstrap behavior.

- [ ] **Step 2: Verify RED, implement GAE, verify GREEN**

Run before and after implementation: `cargo test --test ppo_test compute_gae`

Expected before: missing function; expected after: exact numeric assertions pass within `1e-5`.

- [ ] **Step 3: Write failing mini-batch coverage test**

For 10 samples and batch size 4, assert batch lengths `[4, 4, 2]`, every index appears exactly once, and every PPO epoch requests a fresh shuffle.

- [ ] **Step 4: Verify RED, implement batching, verify GREEN**

Run before and after implementation: `cargo test --test ppo_test minibatch`

- [ ] **Step 5: Write failing PPO clipping and parameter-update tests**

Test the clipped surrogate with fixed tensors and assert a one-epoch CPU update changes at least one trainable tensor while returning finite policy loss, value loss, entropy, and total loss.

- [ ] **Step 6: Implement PPO update**

Normalize advantages with epsilon, shuffle each epoch, index-select mini-batches, evaluate actions, compute the exact spec losses, call Adam backward step with gradient clipping, and average metrics over updates.

- [ ] **Step 7: Verify GREEN and commit**

Run: `cargo test --test ppo_test`

Expected: all algorithm tests pass.

```bash
git add src/lib.rs src/ppo.rs tests/ppo_test.rs
git commit -m "feat: implement ppo optimization"
```

### Task 6: Training orchestration and safe checkpoints

**Files:**
- Create: `src/trainer.rs`
- Test: `tests/trainer_test.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `resolve_device(DeviceConfig) -> anyhow::Result<Device>`.
- Produces: `save_checkpoint_atomic(var_store, directory, iteration) -> anyhow::Result<PathBuf>`.
- Produces: `run_training(&Config) -> anyhow::Result<()>`.

- [ ] **Step 1: Write failing device and checkpoint tests**

Assert CPU resolves on all machines, CUDA either resolves to `Device::Cuda(0)` or returns an error containing `CUDA`, and a saved checkpoint creates both `iteration-N.pt` and a loadable `latest.pt` without leaving the temporary filename.

- [ ] **Step 2: Verify RED**

Run: `cargo test --test trainer_test`

- [ ] **Step 3: Implement device resolution and atomic checkpoints**

Create the directory, save to a unique same-directory temporary file, rename to the numbered checkpoint, then safely refresh `latest.pt` through another temporary file. Preserve an existing final file until its replacement is complete.

- [ ] **Step 4: Add failing one-iteration smoke test**

Build an in-memory CPU config with two environments, four rollout steps, batch four, one epoch, and one iteration. Assert training returns successfully and emits a loadable final checkpoint.

- [ ] **Step 5: Implement training loop and verify GREEN**

Run: `cargo test --test trainer_test`

Expected: smoke training and checkpoint round-trip pass.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/trainer.rs tests/trainer_test.rs
git commit -m "feat: orchestrate ppo training"
```

### Task 7: Graphical inference and fixed executable entrypoint

**Files:**
- Create: `src/renderer.rs`
- Create: `src/main.rs`
- Test: `tests/entrypoint_test.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: `run_inference(&Config) -> anyhow::Result<()>` as an async macroquad loop.
- Produces: `dispatch(config) -> anyhow::Result<()>`, separating mode selection from window startup enough to test configuration dispatch.

- [ ] **Step 1: Write failing dispatch tests**

Test that parsed `Mode::Train` and `Mode::Infer` map to distinct dispatch targets without opening a window. Test missing inference checkpoint produces an error containing its configured path.

- [ ] **Step 2: Verify RED**

Run: `cargo test --test entrypoint_test`

- [ ] **Step 3: Implement renderer and entrypoint**

Configure macroquad window dimensions from board and cell size, load the model, pace actions at configured FPS, draw grid/head/body/food plus current and best scores, pause briefly after terminal states, reset, and exit when the window closes. `main` must read only `Config::load_default()`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `cargo test --test entrypoint_test`

Expected: non-window dispatch tests pass.

```bash
git add src/lib.rs src/main.rs src/renderer.rs tests/entrypoint_test.rs
git commit -m "feat: add graphical model inference"
```

### Task 8: Documentation and end-to-end verification

**Files:**
- Create: `README.md`
- Create: `config/smoke.yaml`
- Modify: `config/config.yaml`

**Interfaces:**
- Consumes all preceding public behavior; produces user installation, training, inference, and troubleshooting instructions.

- [ ] **Step 1: Document Windows prerequisites**

Specify Rust stable installation, NVIDIA driver/CUDA compatibility, CUDA LibTorch download, `LIBTORCH`, `PATH`, and `LIBTORCH_USE_PYTORCH` alternatives. Explain that tch and LibTorch versions must match and show how to verify `torch_cuda.dll` is discoverable.

- [ ] **Step 2: Document YAML-only workflows**

Show edits for `mode: train` and `mode: infer`, describe every YAML field, state the exact per-iteration sample formula, and include expected log lines and checkpoint locations. Do not document CLI overrides.

- [ ] **Step 3: Run formatting and static checks**

Run: `cargo fmt --all -- --check`

Run: `cargo clippy --all-targets --all-features -- -D warnings`

Expected: both exit successfully without warnings.

- [ ] **Step 4: Run the full test suite**

Run: `cargo test --all-targets --all-features`

Expected: all unit, algorithm, and CPU integration tests pass.

- [ ] **Step 5: Run CPU smoke training**

Temporarily copy the documented smoke values into the fixed `config/config.yaml`, run `cargo run --release`, and verify the log reports `num_envs * rollout_steps` samples and writes a loadable checkpoint. Restore the default CUDA training configuration afterward.

- [ ] **Step 6: Record manual GPU/window verification steps**

Document `device: cuda` training validation and `mode: infer` window validation for the user's NVIDIA machine, including expected failure messages for missing CUDA runtime or checkpoint.

- [ ] **Step 7: Commit**

```bash
git add README.md config/config.yaml config/smoke.yaml
git commit -m "docs: add setup and verification guide"
```

### Task 9: Periodic evaluation and W&B logging

**Files:**
- Create: `src/evaluation.rs`, `src/wandb.rs`, `scripts/wandb_bridge.py`
- Test: `tests/evaluation_test.rs`, `tests/wandb_test.rs`
- Modify: `src/config.rs`, `src/trainer.rs`, `config/config.yaml`

**Interfaces:**
- Produces: `evaluate_policy(model, config, device, seed) -> Result<EvaluationMetrics>`.
- Produces: `WandbLogger::start(config, run_config)`, `log(step, metrics)`, and `finish()`.

- [ ] Extend config tests and reject zero interval, episodes, or max steps.
- [ ] Implement config GREEN and run `cargo test --test config_test`.
- [ ] Test bounded evaluation with a one-step cap, then implement and run `cargo test --test evaluation_test`.
- [ ] Test newline-delimited JSON logging and early child exit errors.
- [ ] Implement the official Python SDK bridge and integrate training/evaluation metrics.
- [ ] Run `cargo test --all-targets` and document W&B setup and metric names.
