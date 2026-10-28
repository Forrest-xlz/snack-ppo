use std::process::Command;

#[test]
fn running_without_an_explicit_config_shows_usage() {
    let output = Command::new(env!("CARGO_BIN_EXE_ppo-snake"))
        .output()
        .expect("binary starts");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("usage: ppo-snake --config PATH"),
        "{stderr}"
    );
}
