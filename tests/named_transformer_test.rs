use ppo_snake::{config::Config, env::SnakeEnv, model::ActorCritic};
use tch::{nn, no_grad, Device, Tensor};

fn config(name: &str) -> Config {
    let mut value = serde_json::to_value(Config::load("config/smoke.yaml").unwrap()).unwrap();
    value["model"]["version"] = serde_json::json!(name);
    serde_json::from_value(value).expect("named transformer must be supported")
}

#[test]
fn named_models_reload_checkpoints_with_their_expected_architectures() {
    for (name, width, dim, depth) in [
        ("transformer_v1_small", 32, 16, 2),
        ("transformer_v1_base", 64, 32, 3),
    ] {
        let cfg = config(name);
        assert_eq!(
            serde_json::to_value(&cfg).unwrap()["model"]["version"],
            name
        );
        let old_store = nn::VarStore::new(Device::Cpu);
        let old_model = ActorCritic::from_config(&old_store.root(), &cfg);
        let env = SnakeEnv::new(5, 5, 42).unwrap();
        let f = cfg.model.version.feature_encoding().encode(&env);
        let input = Tensor::from_slice(&f).view([1, f.len() as i64]);
        let before = no_grad(|| old_model.forward(&input));
        let dir = tempfile::tempdir().unwrap();
        old_store.save(dir.path().join("old.ot")).unwrap();
        let mut store = nn::VarStore::new(Device::Cpu);
        let model = ActorCritic::from_config(&store.root(), &cfg);
        store.load(dir.path().join("old.ot")).unwrap();
        let after = no_grad(|| model.forward(&input));
        assert!(before.0.equal(&after.0));
        assert!(before.1.equal(&after.1));
        assert_eq!(
            store.variables()["cell_input.fc1.weight"].size(),
            [width, 66]
        );
        assert_eq!(
            store.variables()["cell_input.fc2.weight"].size(),
            [dim, width]
        );
        assert_eq!(
            store
                .variables()
                .keys()
                .filter(|k| k.ends_with("qkv.weight"))
                .count(),
            depth
        );
    }
}

#[test]
fn retired_numeric_names_are_rejected() {
    for name in ["transformer_v6", "transformer_v7"] {
        let mut value = serde_json::to_value(config("transformer_v1_small")).unwrap();
        value["model"]["version"] = serde_json::json!(name);
        assert!(
            serde_json::from_value::<Config>(value).is_err(),
            "{name} must not be an alias"
        );
    }
}

#[cfg(windows)]
#[test]
fn feature_sweep_discovers_active_a_b_c_d() {
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/sweep_feature_architecture.ps1",
            "-DryRun",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let configs = stdout
        .lines()
        .filter_map(|line| {
            line.strip_prefix(
                "target/release/ppo-snake.exe --config config/experiments/feature_architecture/",
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(configs, ["A.yaml", "B.yaml", "C.yaml", "D.yaml"]);
}
