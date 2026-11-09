#[path = "support/compact_entity.rs"]
mod compact;

#[test]
fn compact_entities_complete_cuda_ppo_with_two_actual_attention_layers() {
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
    for version in ["transformer_v1_small", "transformer_v8"] {
        compact::exercise(version, device);
    }
    tch::Cuda::synchronize(0);
}
