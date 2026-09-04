use std::{
    collections::BTreeMap,
    io::Write,
    process::{Child, ChildStdin, Command, Stdio},
};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::config::{Config, WandbMode};

pub struct WandbLogger {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
}

#[derive(Serialize)]
struct Message<'a, T: Serialize> {
    kind: &'a str,
    #[serde(flatten)]
    payload: T,
}

#[derive(Serialize)]
struct Init<'a> {
    project: &'a str,
    entity: &'a Option<String>,
    group: &'a Option<String>,
    run_name: &'a str,
    mode: &'a str,
    config: &'a Config,
}

#[derive(Serialize)]
struct Log<'a> {
    metrics: &'a BTreeMap<String, f64>,
}

impl WandbLogger {
    pub fn start(config: &Config, run_name: &str) -> Result<Self> {
        if !config.wandb.enabled {
            return Ok(Self {
                child: None,
                stdin: None,
            });
        }
        let mut child = Command::new(&config.wandb.python_executable)
            .arg("scripts/wandb_bridge.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start W&B Python bridge with {}",
                    config.wandb.python_executable.display()
                )
            })?;
        let mut stdin = child
            .stdin
            .take()
            .context("W&B bridge stdin was unavailable")?;
        write_json(
            &mut stdin,
            &Message {
                kind: "init",
                payload: Init {
                    project: &config.wandb.project,
                    entity: &config.wandb.entity,
                    group: &config.wandb.group,
                    run_name,
                    mode: match config.wandb.mode {
                        WandbMode::Online => "online",
                        WandbMode::Offline => "offline",
                    },
                    config,
                },
            },
        )?;
        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
        })
    }

    pub fn log(&mut self, metrics: &BTreeMap<String, f64>) -> Result<()> {
        if let Some(stdin) = &mut self.stdin {
            write_json(
                stdin,
                &Message {
                    kind: "log",
                    payload: Log { metrics },
                },
            )?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if let Some(mut stdin) = self.stdin.take() {
            write_json(&mut stdin, &serde_json::json!({"kind": "finish"}))?;
        }
        if let Some(mut child) = self.child.take() {
            let status = child.wait().context("failed waiting for W&B bridge")?;
            if !status.success() {
                bail!("W&B bridge exited with {status}");
            }
        }
        Ok(())
    }
}

fn write_json<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Init, Message};
    use crate::config::{Config, WandbMode};

    #[test]
    fn init_message_includes_wandb_group() {
        let yaml = include_str!("../config/smoke.yaml").replace("group: null", "group: study");
        let config: Config = serde_yaml::from_str(&yaml).unwrap();
        let message = Message {
            kind: "init",
            payload: Init {
                project: &config.wandb.project,
                entity: &config.wandb.entity,
                group: &config.wandb.group,
                run_name: "run-1",
                mode: match config.wandb.mode {
                    WandbMode::Online => "online",
                    WandbMode::Offline => "offline",
                },
                config: &config,
            },
        };

        let json = serde_json::to_value(message).unwrap();
        assert_eq!(json["group"], "study");
    }
}
