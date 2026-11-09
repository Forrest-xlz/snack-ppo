#[path = "support/compact_entity.rs"]
mod compact;

#[test]
fn compact_entities_complete_cpu_ppo_evaluation_and_checkpoint_reload() {
    tch::set_num_threads(1);
    for version in ["transformer_v1_small", "transformer_v8"] {
        compact::exercise(version, tch::Device::Cpu);
    }
}
