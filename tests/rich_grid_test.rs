use ppo_snake::{
    config::Config,
    env::{Action, Direction, Point, SnakeEnv},
    model::ActorCritic,
};
use serde_json::json;
use std::{collections::VecDeque, fs, process::Command};
use tch::{nn, Device, Kind, Tensor};

fn config() -> Config {
    let base: Config = serde_yaml::from_str(include_str!("../config/smoke.yaml")).unwrap();
    let mut value = serde_json::to_value(base).unwrap();
    value["model"]["version"] = json!("transformer_v4");
    serde_json::from_value(value).expect("rich grid version must be available")
}

fn bent_snake() -> SnakeEnv {
    SnakeEnv::from_state(
        5,
        5,
        [
            (3, 1),
            (3, 2),
            (2, 2),
            (2, 1),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (3, 4),
        ]
        .into_iter()
        .map(Point::from)
        .collect::<VecDeque<_>>(),
        Direction::Up,
        Some(Point::new(4, 0)),
        42,
    )
    .unwrap()
}

#[test]
fn rich_grid_encodes_relative_geometry_release_bins_and_all_chain_directions() {
    let config = config();
    let env = bent_snake();
    let encoding = config.model.version.feature_encoding();
    assert_eq!(encoding.channels(), 31);
    let f = encoding.encode(&env);
    assert_eq!(f.len(), 775);
    assert_eq!(&f[..75], env.observation());
    let at = |channel: usize, x: usize, y: usize| f[channel * 25 + y * 5 + x];
    assert_eq!(at(3, 0, 0), 1.0);
    assert_eq!(at(3, 4, 0), 0.0); // food is not empty
    assert_eq!(at(4, 3, 1), 0.75);
    assert_eq!(at(5, 3, 1), 0.25);
    assert_eq!(at(6, 0, 0), -0.75);
    assert_eq!(at(7, 0, 0), -0.25);
    assert_eq!(at(6, 3, 1), 0.0);
    assert_eq!(at(8, 3, 4), 1.0);
    assert_eq!(at(9, 3, 4), 1.0 / 25.0);
    assert_eq!(at(9, 3, 1), 10.0 / 25.0);
    assert!((at(10, 3, 1) - 11.0f32.ln() / 26.0f32.ln()).abs() < 1e-6);
    assert_eq!(at(11, 3, 4), 1.0); // t=1
    assert_eq!(at(18, 2, 2), 1.0); // t=8
    assert!((11..19).all(|channel| at(channel, 3, 1) == 0.0)); // t=10
    assert!((8..19).all(|channel| at(channel, 0, 0) == 0.0));
    assert_eq!(
        (at(19, 3, 1), at(20, 3, 1), at(21, 3, 1), at(22, 3, 1)),
        (0.75, 0.25, 0.25, 0.75)
    );
    assert!((23..27).all(|channel| at(channel, 3, 1) == 0.0)); // head has no headward link
    assert_eq!(at(28, 3, 1), 1.0); // head -> neck is down
    assert_eq!(at(23, 3, 4), 1.0); // tail -> previous segment is up
    assert!((27..31).all(|channel| at(channel, 3, 4) == 0.0));
    assert_eq!(at(26, 2, 2), 1.0); // toward head is right
    assert_eq!(at(27, 2, 2), 1.0); // toward tail is up
    assert_eq!(at(25, 2, 3), 1.0); // toward head is left
    assert_eq!(at(30, 2, 3), 1.0); // toward tail is right
    assert_eq!(at(24, 3, 1), 0.0);
    assert!((23..31).all(|channel| at(channel, 0, 0) == 0.0));
    // A spatially adjacent cell is not necessarily the next chain segment.
    assert_eq!(at(29, 3, 1), 0.0); // occupied (2,1) is not the neck
}

#[test]
fn release_features_update_after_motion_and_tail_stays_when_eating() {
    let config = config();
    let encoding = config.model.version.feature_encoding();
    let mut env = bent_snake();
    assert!(!env.step(Action::Right).done);
    let moved = encoding.encode(&env);
    assert_eq!(moved[3 * 25 + 4 * 5 + 3], 1.0); // old tail became empty
    assert_eq!(moved[9 * 25 + 4 * 5 + 3], 0.0);
    assert_eq!(moved[8 * 25 + 3 * 5 + 3], 1.0); // previous segment is the new tail
    assert_eq!(env.step(Action::Up).reward, 1.0);
    let eaten = encoding.encode(&env);
    assert_eq!(eaten[8 * 25 + 3 * 5 + 3], 1.0); // tail held during growth
    assert_eq!(eaten[9 * 25 + 3 * 5 + 3], 1.0 / 25.0);
    assert_eq!(eaten[9 * 25 + 4], 11.0 / 25.0); // length increased
}

#[test]
fn full_single_row_has_finite_wall_and_order_features() {
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
    let f = config.model.version.feature_encoding().encode(&env);
    assert!(f.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
    for channel in [3, 5, 7, 21, 22] {
        assert_eq!(&f[channel * 3..channel * 3 + 3], &[0.0; 3]);
    }
    assert_eq!(&f[9 * 3..9 * 3 + 3], &[1.0 / 3.0, 2.0 / 3.0, 1.0]);
}

#[test]
fn rich_grid_cli_trains_three_blocks_input_mlp_and_reloads() {
    let mut config = config();
    let directory = tempfile::tempdir().unwrap();
    config.training.checkpoint_dir = directory.path().to_owned();
    config.wandb.run_name = Some("rich-grid-smoke".into());
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
    let mut store = nn::VarStore::new(Device::Cpu);
    let model = ActorCritic::from_config(&store.root(), &config);
    let checkpoint = directory.path().join("rich-grid-smoke/latest.ot");
    store.load(&checkpoint).unwrap();
    let env = bent_snake();
    let input =
        Tensor::from_slice(&config.model.version.feature_encoding().encode(&env)).view([1, 775]);
    let (logits, values) = model.forward(&input);
    (logits.square().sum(Kind::Float) + values.square().sum(Kind::Float)).backward();
    for name in [
        "token_embedding.weight",
        "token_embedding2.weight",
        "encoder.qkv.weight",
        "encoder2.qkv.weight",
        "encoder3.qkv.weight",
    ] {
        let grad = store.variables()[name].grad();
        assert_eq!(grad.isfinite().all().int64_value(&[]), 1, "{name}");
        assert!(
            grad.abs().sum(Kind::Float).double_value(&[]) > 0.0,
            "{name}"
        );
    }
    let mut restored_store = nn::VarStore::new(Device::Cpu);
    let restored = ActorCritic::from_config(&restored_store.root(), &config);
    restored_store.load(checkpoint).unwrap();
    let (r_logits, r_values) = tch::no_grad(|| restored.forward(&input));
    assert!(logits.equal(&r_logits));
    assert!(values.equal(&r_values));
}

#[test]
fn seven_plane_teacher_is_rejected_for_the_new_layout_but_original_teacher_is_valid() {
    let mut value = serde_json::to_value(config()).unwrap();
    value["kickstarting"] = json!({"teacher_checkpoint":"teacher.ot","teacher_model":{"version":"transformer_v2","hidden_size":16},"initial_weight":1.0,"decay_samples":100});
    assert!(serde_json::from_value::<Config>(value.clone())
        .unwrap()
        .validate()
        .is_err());
    value["kickstarting"]["teacher_model"]["version"] = json!("mlp_v1");
    serde_json::from_value::<Config>(value)
        .unwrap()
        .validate()
        .unwrap();
}
