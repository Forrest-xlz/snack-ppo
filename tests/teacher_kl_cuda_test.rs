#[test]
fn small_transformer_uses_cuda_for_student_teacher_and_evaluation() {
    use ppo_snake::config::Config;
    let mut value = serde_json::to_value(Config::load("config/smoke.yaml").unwrap()).unwrap();
    value["device"] = serde_json::json!("cuda");
    value["model"]["version"] = serde_json::json!("transformer_v1_small");
    value["teacher_kl"] = serde_json::json!({"enabled":true,"coef":0.001});
    value["training"]["max_env_samples"] = serde_json::json!(24);
    let dir = tempfile::tempdir().unwrap();
    value["training"]["checkpoint_dir"] = serde_json::json!(dir.path().join("checkpoints"));
    let path = dir.path().join("cuda.yaml");
    std::fs::write(&path, serde_yaml::to_string(&value).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .args(["--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    if !output.status.success() && std::env::var_os("PPO_SNAKE_REQUIRE_CUDA_TESTS").is_none() {
        let err = String::from_utf8_lossy(&output.stderr);
        if err.contains("CUDA was requested but is unavailable")
            || err.contains("CUDA runtime DLL not found")
        {
            return;
        }
    }
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("evaluation iteration=2 device=Cuda(0)"),
        "{stdout}"
    );
    assert!(stdout.contains("teacher_kl_coef=0.001000"), "{stdout}");
    assert!(!stdout.to_lowercase().contains("nan"), "{stdout}");
    assert!(dir.path().join("checkpoints/smoke/teacher.ot").is_file());
}
