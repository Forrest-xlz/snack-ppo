use std::{collections::VecDeque, fs, process::Command};

use ppo_snake::{
    config::Config,
    env::{Direction, Point, SnakeEnv},
    model::ActorCritic,
    rollout::VectorEnv,
};
use serde_json::json;
use tch::{nn, Device, Tensor};

fn config() -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["model"]["version"] = json!("transformer_v2");
    value["environment"]["width"] = json!(7);
    serde_json::from_value(value).expect("transformer_v2 must select the enriched features")
}

#[test]
fn seven_planes_encode_coordinates_empty_and_chain_distance() {
    let config = config();
    let env = SnakeEnv::new(7, 5, 42).unwrap();
    let encoding = config.model.version.feature_encoding();
    let features = encoding.encode(&env);
    assert_eq!(encoding.channels(), 7);
    assert_eq!(encoding.observation_size(7, 5), 245);
    assert_eq!(features.len(), 245);
    assert_eq!(&features[..105], env.observation());
    for y in 0..5 {
        for x in 0..7 {
            let cell = y * 7 + x;
            assert_eq!(features[105 + cell], x as f32 / 6.0);
            assert_eq!(features[140 + cell], y as f32 / 4.0);
            let occupied = features[cell] + features[35 + cell] + features[70 + cell];
            assert_eq!(
                features[175 + cell],
                if occupied == 0.0 { 1.0 } else { 0.0 }
            );
            let expected = env
                .snake()
                .iter()
                .rev()
                .position(|p| *p == Point::new(x, y))
                .map_or(0.0, |distance| distance as f32 / 34.0);
            assert_eq!(features[210 + cell], expected);
        }
    }
    let food = env.food().unwrap();
    assert_eq!(features[175 + food.y * 7 + food.x], 0.0);
    let envs = VectorEnv::with_features(&config.environment, config.seed, encoding).unwrap();
    assert_eq!(envs.observation_size(), 245);
    assert_eq!(envs.observations().len(), 2 * 245);
}

#[test]
fn chain_order_distinguishes_identical_occupancy_with_different_legal_moves() {
    let config = config();
    let make_env = |points: &[(usize, usize)]| {
        SnakeEnv::from_state(
            3,
            3,
            points
                .iter()
                .copied()
                .map(Point::from)
                .collect::<VecDeque<_>>(),
            Direction::Left,
            Some(Point::new(2, 2)),
            42,
        )
        .unwrap()
    };
    let a = make_env(&[
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (1, 1),
        (1, 2),
        (0, 2),
        (0, 1),
    ]);
    let b = make_env(&[
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (1, 1),
        (0, 1),
        (0, 2),
        (1, 2),
    ]);
    assert_eq!(a.observation(), b.observation());
    let encoding = config.model.version.feature_encoding();
    let a_features = encoding.encode(&a);
    let b_features = encoding.encode(&b);
    assert_ne!(a_features, b_features);
    assert_eq!(a_features[6 * 9 + 3], 0.0); // (0,1) is A's tail
    assert_eq!(b_features[6 * 9 + 3], 2.0 / 8.0);
}

#[test]
fn single_row_full_board_has_finite_features_and_no_empty_cells() {
    let config = config();
    let env = SnakeEnv::from_state(
        3,
        1,
        VecDeque::from([Point::new(2, 0), Point::new(1, 0), Point::new(0, 0)]),
        Direction::Right,
        None,
        42,
    )
    .unwrap();
    let features = config.model.version.feature_encoding().encode(&env);
    assert!(features
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    assert_eq!(&features[12..15], &[0.0, 0.0, 0.0]); // y coordinates
    assert_eq!(&features[15..18], &[0.0, 0.0, 0.0]); // empty
    assert_eq!(&features[18..21], &[0.0, 0.5, 1.0]); // tail to head
}

#[test]
fn enriched_transformer_cli_trains_evaluates_and_reloads() {
    let mut config = config();
    let directory = tempfile::tempdir().unwrap();
    config.training.checkpoint_dir = directory.path().to_owned();
    config.wandb.run_name = Some("transformer-v2-smoke".into());
    let path = directory.path().join("config.yaml");
    fs::write(&path, serde_yaml::to_string(&config).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("evaluation"));
    let mut store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &config);
    let checkpoint = directory.path().join("transformer-v2-smoke/latest.ot");
    store.load(&checkpoint).unwrap();
    assert_eq!(store.variables()["token_embedding.weight"].size(), [32, 7]);
    let env = SnakeEnv::new(7, 5, 42).unwrap();
    let input =
        Tensor::from_slice(&config.model.version.feature_encoding().encode(&env)).view([1, 245]);
    let (logits, values) = tch::no_grad(|| model.forward(&input));
    let mut restored_store = nn::VarStore::new(Device::Cpu);
    let restored = ActorCritic::from_config(&restored_store.root(), &config);
    restored_store.load(checkpoint).unwrap();
    let (restored_logits, restored_values) = tch::no_grad(|| restored.forward(&input));
    assert!(logits.equal(&restored_logits));
    assert!(values.equal(&restored_values));
}

#[test]
fn experiment_c_changes_only_model_architecture_and_identity_from_b() {
    let b = Config::load("config/experiments/feature_architecture/B.yaml").unwrap();
    let c = Config::load("config/experiments/feature_architecture/C.yaml").unwrap();
    assert_eq!(
        serde_json::to_value(&c).unwrap()["model"]["version"],
        "transformer_v1_base"
    );
    assert_eq!(
        c.wandb.run_name.as_deref(),
        Some("E004 feature architecture exp-C")
    );
    assert_eq!(
        c.inference.model_path.to_string_lossy(),
        "checkpoints/E004_feature_architecture_exp-C/latest.ot"
    );
    let expected = serde_json::to_value(b).unwrap();
    let mut actual = serde_json::to_value(c).unwrap();
    actual["device"] = expected["device"].clone();
    actual["model"]["version"] = expected["model"]["version"].clone();
    actual["wandb"]["run_name"] = expected["wandb"]["run_name"].clone();
    actual["inference"]["model_path"] = expected["inference"]["model_path"].clone();
    assert_eq!(actual, expected);
}
