use std::{collections::VecDeque, fs, process::Command};

use ppo_snake::{
    config::Config,
    env::{Action, Direction, Point, SnakeEnv},
    evaluation::evaluate_policy,
    model::ActorCritic,
    ppo::PpoTrainer,
    rollout::{collect_rollout, VectorEnv},
};
use serde_json::json;
use tch::{nn, no_grad, Device, IndexOp, Kind, Tensor};

fn config(version: &str) -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["model"]["version"] = json!(version);
    serde_json::from_value(value).expect("entity token model version must be supported")
}

fn divided_board() -> SnakeEnv {
    // x=2 is a solid wall; the tail at (1,4) belongs to the left region.
    SnakeEnv::from_state(
        6,
        5,
        [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (1, 4)]
            .into_iter()
            .map(Point::from)
            .collect::<VecDeque<_>>(),
        Direction::Up,
        Some(Point::new(4, 0)),
        42,
    )
    .unwrap()
}

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
}

#[test]
fn entity_layout_encodes_exclusive_occupancy_release_and_context_without_padding() {
    let cfg = config("transformer_v1_base");
    let encoding = cfg.model.version.feature_encoding();
    let env = divided_board();
    let f = encoding.encode(&env);
    assert_eq!(f.len(), 2033); // 66*30 + 21 + 24 + 8
    assert_eq!(encoding.observation_size(6, 5), 2033);
    let at = |c: usize, x: usize, y: usize| f[c * 30 + y * 6 + x];
    for (x, y, want) in [(2, 0, 0), (2, 2, 1), (4, 0, 2), (0, 0, 3), (1, 4, 4)] {
        for c in 0..5 {
            assert_eq!(at(c, x, y), f32::from(c == want));
        }
    }
    close(at(5, 2, 0), 6.0 / 30.0);
    close(at(5, 1, 4), 1.0 / 30.0);
    assert_eq!(at(11, 2, 0), 1.0); // release t=6
    assert_eq!(at(6, 1, 4), 1.0); // release t=1
    assert!((5..32).all(|c| at(c, 0, 0) == 0.0));
    // t=3 interpolates between log(2) and log(4).
    close(at(24, 2, 3), 0.4150375);
    close(at(25, 2, 3), 0.5849625);
    close((23..32).map(|c| at(c, 2, 3)).sum(), 1.0);
    close(at(32, 4, 0), 0.8);
    close(at(34, 0, 0), -0.4);
    close(at(35, 0, 0), 0.0);
    close(at(36, 0, 0), -0.8660254); // sin(pi * dx' / max(W,H))
    assert_eq!(at(57, 2, 0), 1.0); // head -> neck, ego-down
    assert_eq!(at(55, 1, 4), 1.0); // tail -> previous, ego-right
    assert!((52..56).all(|c| at(c, 2, 0) == 0.0));
    assert!((56..60).all(|c| at(c, 1, 4) == 0.0));
    let snake = &f[1980..2001];
    close(snake[0], 0.2);
    close(snake[11], 0.8);
    assert_eq!(&snake[13..17], &[1.0, 0.0, 0.0, 0.0]);
    close(snake[17], 0.4);
    let food = &f[2001..2025];
    assert_eq!(food[0], 1.0);
    close(food[1], 0.8);
    close(food[3], 0.4);
    let global = &f[2025..];
    close(global[0], 0.2);
    close(global[1], 0.8);
}

#[test]
fn topology_keeps_separate_regions_and_shares_defined_summaries_with_base_version() {
    let env = divided_board();
    let rich = config("transformer_v1_base")
        .model
        .version
        .feature_encoding()
        .encode(&env);
    let base = config("transformer_v5")
        .model
        .version
        .feature_encoding()
        .encode(&env);
    assert_eq!(base.len(), 1853); // 60*30+53
    assert_eq!(&base[..1800], &rich[..1800]);
    assert_eq!(&base[1800..], &rich[1980..]);
    let at = |c: usize, x: usize, y: usize| rich[c * 30 + y * 6 + x];
    close(at(60, 0, 0), 10.0 / 30.0);
    close(at(60, 5, 0), 15.0 / 30.0);
    close(at(61, 0, 0), 11.0f32.ln() / 31.0f32.ln());
    close(at(62, 0, 0), 10.0 / 6.0);
    assert_eq!(at(63, 0, 0), 1.0);
    assert_eq!(at(64, 0, 0), 1.0); // tail adjacent region
    assert_eq!(at(64, 5, 0), 0.0);
    assert_eq!(at(65, 0, 0), 1.0);
    assert_eq!(at(65, 5, 0), 1.0);
    assert!((60..66).all(|c| at(c, 2, 2) == 0.0)); // blocked body
    close(rich[1980 + 19], 0.5); // max adjacent area: 15
    assert_eq!(rich[1980 + 20], 2.0); // size/L clipped
    close(rich[2001 + 21], 0.5); // food lies in right region
    assert_eq!(rich[2001 + 22], 2.0);
    assert_eq!(rich[2001 + 23], 1.0);
    assert_eq!(&rich[2025 + 3..], &[0.0, 1.0, 0.0, 0.0, 0.0]); // 2 regions
}

#[test]
fn ego_coordinates_and_chain_directions_rotate_consistently_for_every_heading() {
    let cfg = config("transformer_v1_base");
    for (direction, hx, hy) in [
        (Direction::Up, 0, -1),
        (Direction::Down, 0, 1),
        (Direction::Left, -1, 0),
        (Direction::Right, 1, 0),
    ] {
        let p = |d: i32| Point::new((2 + d * hx) as usize, (2 + d * hy) as usize);
        let env = SnakeEnv::from_state(
            5,
            5,
            VecDeque::from([p(0), p(-1), p(-2)]),
            direction,
            Some(p(1)),
            42,
        )
        .unwrap();
        let f = cfg.model.version.feature_encoding().encode(&env);
        let ahead = p(1).y * 5 + p(1).x;
        close(f[34 * 25 + ahead], 0.0);
        close(f[35 * 25 + ahead], -0.25);
        assert_eq!(f[57 * 25 + 12], 1.0); // head -> neck is always ego-down
        assert_eq!(f[52 * 25 + p(-2).y * 5 + p(-2).x], 1.0); // tail toward head = ego-up
    }
}

#[test]
fn terminal_single_row_features_are_finite_and_growth_holds_tail() {
    let encoding = config("transformer_v1_base")
        .model
        .version
        .feature_encoding();
    let env = SnakeEnv::from_state(
        3,
        1,
        VecDeque::from([Point::new(2, 0), Point::new(1, 0), Point::new(0, 0)]),
        Direction::Right,
        None,
        42,
    )
    .unwrap();
    let f = encoding.encode(&env);
    assert!(f.iter().all(|v| v.is_finite()));
    assert_eq!(&f[66 * 3 + 21..66 * 3 + 21 + 24], &[0.0; 24]);
    assert_eq!(f[66 * 3 + 21 + 24 + 1], 0.0);
    let mut env = SnakeEnv::from_state(
        5,
        5,
        VecDeque::from([Point::new(2, 2), Point::new(1, 2), Point::new(0, 2)]),
        Direction::Right,
        Some(Point::new(3, 2)),
        42,
    )
    .unwrap();
    assert_eq!(env.step(Action::Right).reward, 1.0);
    let f = encoding.encode(&env);
    assert_eq!(f[4 * 25 + 10], 1.0); // original tail held
    close(f[5 * 25 + 13], 4.0 / 25.0); // new head length
}

#[test]
fn entity_models_train_all_input_mlps_evaluate_and_reload() {
    tch::set_num_threads(1);
    for version in ["transformer_v5", "transformer_v1_base"] {
        let cfg = config(version);
        let store = nn::VarStore::new(Device::Cpu);
        let model = ActorCritic::from_config(&store.root(), &cfg);
        let mut envs =
            VectorEnv::with_features(&cfg.environment, 42, cfg.model.version.feature_encoding())
                .unwrap();
        let rollout = collect_rollout(&model, &mut envs, 4, Device::Cpu).unwrap();
        assert_eq!(rollout.observations.len(), 8 * envs.observation_size());
        let mut trainer = PpoTrainer::new(&store, &cfg.ppo, 42).unwrap();
        let update = trainer
            .update(&model, &rollout, &[0.0, 0.0], Device::Cpu)
            .unwrap();
        assert!(update.metrics.total_loss.is_finite());
        for name in [
            "cell_input.fc1.weight",
            "cell_input.fc2.weight",
            "snake_input.fc1.weight",
            "food_input.fc1.weight",
            "global_input.fc1.weight",
            "encoder.qkv.weight",
            "encoder2.qkv.weight",
            "encoder3.qkv.weight",
        ] {
            let g = store.variables()[name].grad();
            assert_eq!(g.isfinite().all().int64_value(&[]), 1, "{name}");
            assert!(
                g.abs().sum(Kind::Float).double_value(&[]) > 0.0,
                "{version}: {name}"
            );
        }
        let evaluation = evaluate_policy(&model, &cfg, Device::Cpu, 1000045).unwrap();
        assert!(evaluation.mean_score.is_finite());
        let input =
            Tensor::from_slice(&envs.observations()).view([2, envs.observation_size() as i64]);
        let expected = no_grad(|| model.forward(&input));
        let dir = tempfile::tempdir().unwrap();
        store.save(dir.path().join("entity.ot")).unwrap();
        let mut restored_store = nn::VarStore::new(Device::Cpu);
        let restored = ActorCritic::from_config(&restored_store.root(), &cfg);
        restored_store.load(dir.path().join("entity.ot")).unwrap();
        let actual = no_grad(|| restored.forward(&input));
        assert!(expected.0.equal(&actual.0));
        assert!(expected.1.equal(&actual.1));
        let backend = model.attention_backend(Device::Cpu, 2, 25, false).unwrap();
        assert_eq!(backend.matches("queries=28,").count(), 2);
        assert_eq!(backend.matches("queries=2,").count(), 1);
    }
}

#[test]
fn release_bins_cover_sixteen_overflow_and_full_board_endpoint() {
    let encoding = config("transformer_v1_base")
        .model
        .version
        .feature_encoding();
    let row = SnakeEnv::from_state(
        20,
        1,
        (0..18).rev().map(|x| Point::new(x, 0)).collect(),
        Direction::Right,
        Some(Point::new(19, 0)),
        42,
    )
    .unwrap();
    let f = encoding.encode(&row);
    assert_eq!(f[21 * 20 + 15], 1.0); // t=16, last precise slot
    assert_eq!(f[22 * 20 + 15], 0.0);
    assert_eq!(f[22 * 20 + 16], 1.0); // t=17, overflow
    assert!((6..22).all(|c| f[c * 20 + 16] == 0.0));
    let mut chain = VecDeque::new();
    for y in 0..12 {
        for i in 0..12 {
            chain.push_back(Point::new(if y % 2 == 0 { i } else { 11 - i }, y));
        }
    }
    let full = SnakeEnv::from_state(12, 12, chain, Direction::Left, None, 42).unwrap();
    let f = encoding.encode(&full);
    assert_eq!(f.len(), 9557);
    assert!(f.iter().all(|v| v.is_finite()));
    assert_eq!(f[5 * 144], 1.0); // head release t=144
    assert_eq!(f[31 * 144], 1.0); // final log-two-hot endpoint
    assert!((23..31).all(|c| f[c * 144] == 0.0));
    assert_eq!(f[9504 + 10], 1.0); // L=144 final length node
    assert_eq!(&f[9525..9549], &[0.0; 24]); // no food
    assert_eq!(&f[9550..9552], &[0.0; 2]); // no remaining space
}

#[test]
fn actor_and_critic_use_snake_and_global_instead_of_head_cell_readouts() {
    let cfg = config("transformer_v1_base");
    let store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &cfg);
    no_grad(|| {
        for (name, mut t) in store.variables() {
            if name.contains("qkv") || name.contains("ff_out") {
                let _ = t.zero_();
            }
        }
    });
    let env = SnakeEnv::new(5, 5, 42).unwrap();
    let f = cfg.model.version.feature_encoding().encode(&env);
    let input = Tensor::from_slice(&f).view([1, f.len() as i64]);
    let original = no_grad(|| model.forward(&input));
    let changed = input.copy();
    let _ = changed.i((0, 1650)).fill_(10.0); // Snake length, not a cell
    let snake_changed = no_grad(|| model.forward(&changed));
    assert!(!original.0.allclose(&snake_changed.0, 1e-6, 1e-7, false));
    assert!(original.1.allclose(&snake_changed.1, 1e-6, 1e-7, false));
    let changed = input.copy();
    let _ = changed.i((0, 1695)).fill_(10.0); // Global fill rate
    let global_changed = no_grad(|| model.forward(&changed));
    assert!(original.0.allclose(&global_changed.0, 1e-6, 1e-7, false));
    assert!(!original.1.allclose(&global_changed.1, 1e-6, 1e-7, false));
}

#[test]
fn entity_cli_saves_and_rejects_legacy_teacher_layout() {
    let mut cfg = config("transformer_v1_base");
    let dir = tempfile::tempdir().unwrap();
    cfg.training.checkpoint_dir = dir.path().join("checkpoints");
    cfg.wandb.run_name = Some("entity-smoke".into());
    let path = dir.path().join("config.yaml");
    fs::write(&path, serde_yaml::to_string(&cfg).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(cfg
        .training
        .checkpoint_dir
        .join("entity-smoke/latest.ot")
        .exists());
    let mut value = serde_json::to_value(cfg).unwrap();
    value["kickstarting"] = json!({"teacher_checkpoint":"teacher.ot","teacher_model":{"version":"mlp_v1","hidden_size":16},"initial_weight":1.0,"decay_samples":100});
    let invalid: Config = serde_json::from_value(value).unwrap();
    assert!(
        invalid.validate().is_err(),
        "exclusive body/tail planes are not the legacy prefix"
    );
}

#[test]
fn cuda_entity_models_have_finite_ppo_gradients_and_bf16_attention() {
    use ppo_snake::{config::DeviceConfig, trainer::resolve_device};
    let device = match resolve_device(DeviceConfig::Cuda) {
        Ok(device) => device,
        Err(error) => {
            assert!(
                std::env::var_os("PPO_SNAKE_REQUIRE_CUDA_TESTS").is_none(),
                "{error:#}"
            );
            eprintln!("Skipping optional CUDA check: {error:#}");
            return;
        }
    };
    for version in ["transformer_v5", "transformer_v1_base"] {
        let cfg = config(version);
        let store = nn::VarStore::new(device);
        let model = ActorCritic::from_config(&store.root(), &cfg);
        let mut envs =
            VectorEnv::with_features(&cfg.environment, 42, cfg.model.version.feature_encoding())
                .unwrap();
        let rollout = collect_rollout(&model, &mut envs, 4, device).unwrap();
        let mut trainer = PpoTrainer::new(&store, &cfg.ppo, 42).unwrap();
        let update = trainer.update(&model, &rollout, &[0.0; 2], device).unwrap();
        assert!(update.metrics.total_loss.is_finite());
        for (name, weight) in store.variables() {
            let grad = weight.grad();
            assert!(grad.defined(), "{version}: {name}");
            assert_eq!(grad.isfinite().all().int64_value(&[]), 1, "{name}");
        }
        let backend = model.attention_backend(device, 2, 25, true).unwrap();
        assert_eq!(backend.matches("BFloat16").count(), 3, "{backend}");
        assert_eq!(backend.matches("queries=28,").count(), 2, "{backend}");
        let scores = evaluate_policy(&model, &cfg, device, 1000045).unwrap();
        assert!(scores.mean_score.is_finite());
    }
    tch::Cuda::synchronize(0);
}
