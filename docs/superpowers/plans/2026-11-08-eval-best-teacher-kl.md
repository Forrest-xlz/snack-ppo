# Evaluation-best teacher KL implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Add E004 D, identical to B except a configurable KL penalty toward its best evaluated policy.

**Architecture:** A frozen, independent copy of the student becomes active at the first scheduled evaluation. A strictly higher held-out eval mean score replaces the copy; ties and regressions retain it. Compute forward KL (teacher || student) on student rollout states in bounded inference chunks, with a fixed independent coefficient.

**Tech stack:** Rust, tch 0.24 / LibTorch 2.11, YAML, PowerShell.

**Spec:** User-approved design in this conversation (2026-11-08): first evaluation initializes the teacher, coefficient configurable with default 0.001, no critic distillation or student/Adam resets.

## Global constraints
- Preserve B/C, all existing dirty work, existing checkpoints, PPO clipping, evaluation seeds and early stopping.
- `teacher_kl: { enabled: true, coef: 0.001 }`; optional/disabled/zero must retain ordinary PPO behavior and skip teacher inference.
- D starts from scratch; batch 10240, epoch 1, CUDA, small Transformer, 1200M cap, 25M eval interval, 1024 episodes, patience 12.
- Keep fixed-expert kickstarting behavior; reject enabling both teacher objectives together.
- Allocate the independent teacher before seeding/constructing the student so it does not perturb the student's initialization or sampling RNG stream.
- Save the current teacher to `teacher.ot` on replacement; existing latest/numbered checkpoints remain the student.

## Review focus
- Strict higher-than comparison, ties, nonfinite scores and initial score zero.
- No shared tensor storage or teacher gradients; no optimizer reset at replacement.
- Actual forward KL (including teacher entropy), shuffled target row alignment, bounded GPU memory.
- Baseline equivalence with coefficient zero and no teacher before first eval.
- Actual CPU/CUDA CLI path, sweep discovery, checkpoints and unchanged early stopping.

## Tasks
- [x] Add failing tests for config, forward KL gradients, snapshot replacement, zero-coefficient baseline equivalence, and real CLI training/evaluation/checkpoint behavior.
- [x] Add optional config and evaluation-best teacher state; reuse frozen teacher inference with independent copied weights.
- [x] Add exact forward KL PPO objective and sample-weighted metrics, preserving the existing kickstarting API/objective.
- [x] Integrate eval updates, logs, teacher checkpoint; add D and document English protocol/launch command.
- [x] Verify focused tests, full suite with CUDA required, release build, sweep dry run, formatting and diff checks; review the change.

## Execution notes
- Work in the current workspace as requested; no commit/push or formal training launch.
- RED: new teacher module/config absent; D config absent; weighted penalty metric absent. GREEN: all covered by real tensor, CPU/CUDA CLI and config tests.
- `PPO_SNAKE_REQUIRE_CUDA_TESTS=1 cargo test --all-targets -- --test-threads=1`: 118 passed, zero failed; `target/teacher-kl-all-tests.log`.
- Additional first-eval snapshot comparison against an ordinary-PPO prefix passed after extending the CLI test.
- `cargo build --release --bins --example benchmark_training` passed. CUDA release benchmark smoke activated the teacher and reported coefficient 0.001; `target/benchmarks/teacher-kl-smoke.json` (tiny correctness run, not a throughput estimate).
- D-only sweep dry run and release checkpoint-directory inspection passed; destination `checkpoints/E004_feature_architecture_exp-D`.
- Read-only final review by entity_token_review: no concrete defects; no deferred findings.
