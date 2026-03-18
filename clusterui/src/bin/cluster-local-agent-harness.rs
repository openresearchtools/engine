#[path = "../agent.rs"]
mod agent;
#[path = "../catalog.rs"]
mod catalog;
#[path = "../cluster_api.rs"]
mod cluster_api;
#[path = "../model_metadata.rs"]
mod model_metadata;
#[path = "../model_store.rs"]
mod model_store;
#[path = "../protocol.rs"]
mod protocol;
#[path = "../public_server.rs"]
mod public_server;
#[path = "../settings.rs"]
mod settings;

use agent::{default_local_agent_addr, ensure_local_agent, run_agent};
use anyhow::{Context, Result};
use std::env;
use std::path::PathBuf;

fn env_optional(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn runtime_dir_from_settings() -> Option<String> {
    let settings = settings::load_controller_settings_or_default();
    let runtime_dir = settings.runtime_dir.trim();
    if runtime_dir.is_empty() {
        None
    } else {
        Some(runtime_dir.to_string())
    }
}

fn main() -> Result<()> {
    let action = env_optional("CLUSTER_ACTION").unwrap_or_else(|| "serve".to_string());
    let runtime_dir = env_optional("CLUSTER_RUNTIME_DIR")
        .or_else(runtime_dir_from_settings)
        .context("missing CLUSTER_RUNTIME_DIR and no runtime_dir saved in settings.json")?;
    let bind_addr = env_optional("CLUSTER_BIND_ADDR").unwrap_or_else(default_local_agent_addr);

    match action.as_str() {
        "ensure" => {
            ensure_local_agent(&PathBuf::from(&runtime_dir), &bind_addr)?;
            println!("local agent ready at {bind_addr}");
        }
        _ => {
            println!("serving local agent at {bind_addr}");
            run_agent(PathBuf::from(&runtime_dir), bind_addr)?;
        }
    }

    Ok(())
}
