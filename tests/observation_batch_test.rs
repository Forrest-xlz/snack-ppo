use ppo_snake::{
    config::Config,
    env::SnakeEnv,
    rollout::{RolloutBuffer, VectorEnv},
};
use tch::{Device, Tensor};

#[test]
fn observation_batch_preserves_shuffled_rows_duplicates_and_partial_batch() {
    let rollout = RolloutBuffer {
        observations: vec![10., 11., 12., 20., 21., 22., 30., 31., 32., 40., 41., 42.],
        actions: vec![0; 4],
        rewards: vec![0.; 4],
        dones: vec![false; 4],
        values: vec![0.; 4],
        log_probs: vec![0.; 4],
        episode_scores: vec![],
        reverse_action_count: 0,
        steps: 2,
        num_envs: 2,
        observation_size: 3,
    };
    let actual = rollout.observation_batch(&[3, 1, 3], Device::Cpu);
    assert!(actual.equal(
        &Tensor::from_slice(&[40.0f32, 41., 42., 20., 21., 22., 40., 41., 42.]).view([3, 3])
    ));
    let tail = rollout.observation_batch(&[2], Device::Cpu);
    assert!(tail.equal(&Tensor::from_slice(&[30.0f32, 31., 32.]).view([1, 3])));
    assert_eq!(rollout.observation_batch(&[], Device::Cpu).size(), [0, 3]);
}

#[test]
fn encode_into_overwrites_reused_buffers_and_parallel_env_rows_match_serial_encoding() {
    for version in [
        "mlp_v1",
        "transformer_v2",
        "transformer_v4",
        "transformer_v5",
        "transformer_v1_base",
    ] {
        let mut value = serde_json::to_value(Config::load("config/smoke.yaml").unwrap()).unwrap();
        value["model"]["version"] = serde_json::json!(version);
        value["environment"]["num_envs"] = serde_json::json!(7);
        let cfg: Config = serde_json::from_value(value).unwrap();
        let encoding = cfg.model.version.feature_encoding();
        let count = encoding.observation_size(5, 5);
        let envs = VectorEnv::with_features(&cfg.environment, 42, encoding).unwrap();
        let parallel = envs.observations();
        assert_eq!(parallel.len(), 7 * count);
        let mut output = vec![f32::NAN; count];
        for row in 0..7 {
            output.fill(99.);
            let env = SnakeEnv::new(5, 5, 42 + row as u64).unwrap();
            encoding.encode_into(&env, &mut output);
            assert_eq!(output, encoding.encode(&env), "{version}");
            assert_eq!(
                &parallel[row * count..(row + 1) * count],
                output,
                "{version}"
            );
        }
    }
}
