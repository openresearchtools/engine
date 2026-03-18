use crate::cluster_api::default_runtime_dir;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PairedPeerSettings {
    #[serde(default)]
    pub node_id: String,
    #[serde(default)]
    pub display_name: String,
    pub control_addr: String,
    #[serde(default)]
    pub shared_token_obfuscated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControllerSettings {
    pub runtime_dir: String,
    pub local_control_addr: String,
    pub server_bind_addr: String,
    pub server_allow_cors: bool,
    pub server_allowed_origins: String,
    pub server_allowed_client_ips: String,
    pub show_cpu_devices: bool,
    pub show_integrated_gpus: bool,
    pub auto_refresh_enabled: bool,
    #[serde(default)]
    pub multi_node_rpc_enabled: bool,
    #[serde(default)]
    pub theme_preference: ControllerThemePreference,
    #[serde(default)]
    pub pairing_identity: String,
    #[serde(default)]
    pub paired_peers: Vec<PairedPeerSettings>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ControllerThemePreference {
    Dark,
    Light,
    #[default]
    System,
}

impl ControllerThemePreference {
    pub fn as_egui(self) -> egui::ThemePreference {
        match self {
            Self::Dark => egui::ThemePreference::Dark,
            Self::Light => egui::ThemePreference::Light,
            Self::System => egui::ThemePreference::System,
        }
    }
}

impl Default for ControllerSettings {
    fn default() -> Self {
        Self {
            runtime_dir: default_runtime_dir()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            local_control_addr: "127.0.0.1:46211".to_string(),
            server_bind_addr: "127.0.0.1:46310".to_string(),
            server_allow_cors: false,
            server_allowed_origins: String::new(),
            server_allowed_client_ips: String::new(),
            show_cpu_devices: false,
            show_integrated_gpus: false,
            auto_refresh_enabled: false,
            multi_node_rpc_enabled: false,
            theme_preference: ControllerThemePreference::System,
            pairing_identity: String::new(),
            paired_peers: Vec::new(),
        }
    }
}

pub fn controller_settings_path() -> Option<PathBuf> {
    let dir = default_runtime_dir().ok()?;
    fs::create_dir_all(&dir).ok()?;
    Some(dir.join("settings.json"))
}

pub fn default_controller_settings() -> ControllerSettings {
    ControllerSettings::default()
}

pub fn load_controller_settings() -> Option<ControllerSettings> {
    let path = controller_settings_path()?;
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str::<ControllerSettings>(&text).ok()
}

pub fn load_controller_settings_or_default() -> ControllerSettings {
    load_controller_settings().unwrap_or_default()
}

pub fn save_controller_settings(settings: &ControllerSettings) -> Result<(), String> {
    let path = controller_settings_path()
        .ok_or_else(|| "failed to resolve controller settings path".to_string())?;
    let payload = serde_json::to_vec_pretty(settings).map_err(|err| err.to_string())?;
    fs::write(path, payload).map_err(|err| err.to_string())
}

pub fn update_controller_settings<F>(update: F) -> Result<ControllerSettings, String>
where
    F: FnOnce(&mut ControllerSettings),
{
    let mut settings = load_controller_settings_or_default();
    update(&mut settings);
    save_controller_settings(&settings)?;
    Ok(settings)
}

pub fn multi_node_rpc_enabled() -> bool {
    load_controller_settings()
        .map(|settings| settings.multi_node_rpc_enabled)
        .unwrap_or(false)
}
