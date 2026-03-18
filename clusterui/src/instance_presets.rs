use crate::cluster_api::{
    default_load_on_demand_grace_seconds, default_runtime_dir, ManualDeviceAllocation,
    RetentionMode,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstancePreset {
    pub name: String,
    pub model_kind: String,
    pub model_package_folder: String,
    pub model_file_path: String,
    pub mmproj_file_path: Option<String>,
    pub diarization_package_folder: Option<String>,
    pub diarization_file_path: Option<String>,
    pub instance_name: String,
    pub retention_mode: RetentionMode,
    #[serde(default = "default_load_on_demand_grace_seconds")]
    pub load_on_demand_grace_seconds: i32,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
    pub max_predict: i32,
    pub allow_cpu: bool,
    pub allow_integrated_gpu: bool,
    pub preferred_owner_control_addr: Option<String>,
    pub execution_group_id: String,
    pub rpc_servers: Option<String>,
    #[serde(default)]
    pub manual_device_allocations: Vec<ManualDeviceAllocation>,
}

pub fn instance_presets_path() -> Option<PathBuf> {
    let dir = default_runtime_dir().ok()?;
    fs::create_dir_all(&dir).ok()?;
    Some(dir.join("instance-presets.json"))
}

pub fn load_instance_presets() -> Vec<InstancePreset> {
    let Some(path) = instance_presets_path() else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<InstancePreset>>(&text).unwrap_or_default()
}

pub fn save_instance_presets(presets: &[InstancePreset]) -> Result<(), String> {
    let path = instance_presets_path()
        .ok_or_else(|| "failed to resolve instance presets path".to_string())?;
    let payload = serde_json::to_vec_pretty(presets).map_err(|err| err.to_string())?;
    fs::write(path, payload).map_err(|err| err.to_string())
}
