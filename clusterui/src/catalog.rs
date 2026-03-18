use crate::cluster_api::{
    default_load_on_demand_grace_seconds, default_runtime_dir, CreateInstanceParams,
    InstanceModelKind, RetentionMode,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedModelTask {
    Responses,
    Embeddings,
    Rerank,
    Transcription,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedModelEntry {
    pub id: String,
    pub display_name: String,
    pub family: String,
    pub task: ManagedModelTask,
    pub single_device_only: bool,
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub diarization_model_path: Option<String>,
    pub execution_group_id: String,
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
    pub allowed_control_addrs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedModelManifest {
    id: Option<String>,
    display_name: Option<String>,
    task: Option<ManagedModelTask>,
    single_device_only: Option<bool>,
    model_path: Option<String>,
    mmproj_path: Option<String>,
    diarization_model_path: Option<String>,
    execution_group_id: Option<String>,
    retention_mode: Option<String>,
    load_on_demand_grace_seconds: Option<i32>,
    n_ctx: Option<i32>,
    n_batch: Option<i32>,
    n_ubatch: Option<i32>,
    n_parallel: Option<i32>,
    n_threads: Option<i32>,
    n_threads_batch: Option<i32>,
    n_gpu_layers: Option<i32>,
    allowed_control_addrs: Option<Vec<String>>,
}

impl ManagedModelEntry {
    pub fn instance_name(&self) -> &str {
        &self.id
    }

    pub fn supports_vision(&self) -> bool {
        self.task == ManagedModelTask::Responses && self.mmproj_path.is_some()
    }

    pub fn supports_diarization(&self) -> bool {
        self.task == ManagedModelTask::Transcription && self.diarization_model_path.is_some()
    }

    pub fn supports_split_placement(&self) -> bool {
        !self.single_device_only
    }

    pub fn instance_model_kind(&self) -> InstanceModelKind {
        match self.task {
            ManagedModelTask::Responses => {
                if self.supports_vision() {
                    InstanceModelKind::Vision
                } else {
                    InstanceModelKind::Text
                }
            }
            ManagedModelTask::Embeddings => InstanceModelKind::Embeddings,
            ManagedModelTask::Rerank => InstanceModelKind::Rerank,
            ManagedModelTask::Transcription => InstanceModelKind::Whisper,
        }
    }

    pub fn create_instance_params(&self) -> CreateInstanceParams {
        CreateInstanceParams {
            name: self.id.clone(),
            managed_model_id: Some(self.id.clone()),
            model_path: self.model_path.clone(),
            mmproj_path: self.mmproj_path.clone(),
            diarization_model_path: None,
            execution_group_id: self.execution_group_id.clone(),
            rpc_servers: None,
            manual_device_allocations: Vec::new(),
            manual_devices_csv: None,
            manual_tensor_split: None,
            preferred_owner_control_addr: None,
            retention_mode: self.retention_mode,
            load_on_demand_grace_seconds: self.load_on_demand_grace_seconds.max(0),
            embedding: self.task == ManagedModelTask::Embeddings,
            reranking: self.task == ManagedModelTask::Rerank,
            model_kind: self.instance_model_kind(),
            single_device_only: self.single_device_only,
            allow_cpu: false,
            allow_integrated_gpu: false,
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_parallel: self.n_parallel,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            n_gpu_layers: normalize_gpu_layers(self.n_gpu_layers),
        }
    }
}

fn normalize_gpu_layers(value: i32) -> i32 {
    if value == 999 {
        -1
    } else {
        value
    }
}

pub fn default_models_dir() -> Result<PathBuf> {
    if let Some(value) = env::var_os("ENGINE_MODELS_DIR") {
        let path = PathBuf::from(value);
        if !path.as_os_str().is_empty() {
            return Ok(path);
        }
    }

    let runtime_dir = default_runtime_dir()?;
    let base_dir = runtime_dir
        .parent()
        .context("default runtime directory has no parent")?;
    Ok(base_dir.join("models"))
}

pub fn discover_models(models_dir: &Path) -> Result<Vec<ManagedModelEntry>> {
    if !models_dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for family_entry in fs::read_dir(models_dir)
        .with_context(|| format!("failed to read '{}'", models_dir.display()))?
    {
        let family_entry = family_entry?;
        let family_path = family_entry.path();
        if !family_path.is_dir() {
            continue;
        }
        let family = family_entry.file_name().to_string_lossy().into_owned();
        for model_entry in fs::read_dir(&family_path)
            .with_context(|| format!("failed to read '{}'", family_path.display()))?
        {
            let model_entry = model_entry?;
            let model_path = model_entry.path();
            if !model_path.is_dir() {
                continue;
            }
            if let Some(entry) = discover_model_dir(&family, &model_path)? {
                entries.push(entry);
            }
        }
        entries.extend(discover_legacy_family_entries(
            models_dir,
            &family,
            &family_path,
        )?);
    }

    entries.sort_by(|a, b| a.id.cmp(&b.id));
    entries.dedup_by(|lhs, rhs| lhs.id == rhs.id);
    Ok(entries)
}

pub fn find_model_entry(models_dir: &Path, model_id: &str) -> Result<Option<ManagedModelEntry>> {
    let normalized = model_id.trim();
    if normalized.is_empty() {
        return Ok(None);
    }
    let entries = discover_models(models_dir)?;
    Ok(entries.into_iter().find(|entry| entry.id == normalized))
}

fn discover_model_dir(family: &str, model_dir: &Path) -> Result<Option<ManagedModelEntry>> {
    let dir_name = model_dir
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let manifest_path = model_dir.join("manifest.json");
    let manifest = if manifest_path.exists() {
        let text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("failed to read '{}'", manifest_path.display()))?;
        Some(
            serde_json::from_str::<ManagedModelManifest>(&text)
                .with_context(|| format!("failed to parse '{}'", manifest_path.display()))?,
        )
    } else {
        None
    };

    let hinted_task = manifest
        .as_ref()
        .and_then(|value| value.task)
        .unwrap_or_else(|| {
            infer_task_from_hints(
                family,
                Some(&dir_name),
                None,
                manifest
                    .as_ref()
                    .and_then(|value| value.display_name.as_deref()),
            )
        });

    let model_file = match manifest
        .as_ref()
        .and_then(|value| value.model_path.as_ref())
        .map(|value| resolve_model_path(model_dir, value))
    {
        Some(path) => path,
        None => match largest_supported_model_file(model_dir, hinted_task)? {
            Some(path) => path,
            None => return Ok(None),
        },
    };

    let mmproj_path = manifest
        .as_ref()
        .and_then(|value| value.mmproj_path.as_ref())
        .map(|value| resolve_model_path(model_dir, value))
        .or_else(|| discover_mmproj(model_dir));

    let task = manifest
        .as_ref()
        .and_then(|value| value.task)
        .unwrap_or_else(|| {
            infer_task_from_hints(
                family,
                Some(&dir_name),
                Some(&model_file),
                manifest
                    .as_ref()
                    .and_then(|value| value.display_name.as_deref()),
            )
        });

    let id = manifest
        .as_ref()
        .and_then(|value| value.id.clone())
        .unwrap_or_else(|| dir_name.clone());
    let display_name = manifest
        .as_ref()
        .and_then(|value| value.display_name.clone())
        .unwrap_or_else(|| dir_name.replace("__", "/"));

    Ok(Some(ManagedModelEntry {
        id,
        display_name,
        family: family.to_string(),
        task,
        single_device_only: manifest
            .as_ref()
            .and_then(|value| value.single_device_only)
            .unwrap_or_else(|| {
                infer_single_device_only(
                    task,
                    family,
                    Some(&dir_name),
                    Some(&model_file),
                    manifest
                        .as_ref()
                        .and_then(|value| value.display_name.as_deref()),
                )
            }),
        model_path: model_file.display().to_string(),
        mmproj_path: mmproj_path
            .as_ref()
            .map(|value| value.display().to_string()),
        diarization_model_path: manifest
            .as_ref()
            .and_then(|value| value.diarization_model_path.as_ref())
            .map(|value| resolve_model_path(model_dir, value))
            .map(|value| value.display().to_string()),
        execution_group_id: manifest
            .as_ref()
            .and_then(|value| value.execution_group_id.clone())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "cluster:auto".to_string()),
        retention_mode: manifest
            .as_ref()
            .and_then(|value| value.retention_mode.as_deref())
            .and_then(parse_retention_mode)
            .unwrap_or(RetentionMode::LoadOnDemand),
        load_on_demand_grace_seconds: manifest
            .as_ref()
            .and_then(|value| value.load_on_demand_grace_seconds)
            .unwrap_or_else(|| {
                match task {
                    ManagedModelTask::Responses => {
                        if mmproj_path.is_some() {
                            InstanceModelKind::Vision
                        } else {
                            InstanceModelKind::Text
                        }
                    }
                    ManagedModelTask::Embeddings => InstanceModelKind::Embeddings,
                    ManagedModelTask::Rerank => InstanceModelKind::Rerank,
                    ManagedModelTask::Transcription => InstanceModelKind::Whisper,
                }
                .default_load_on_demand_grace_seconds()
            }),
        n_ctx: manifest
            .as_ref()
            .and_then(|value| value.n_ctx)
            .unwrap_or(8192),
        n_batch: manifest
            .as_ref()
            .and_then(|value| value.n_batch)
            .unwrap_or(512),
        n_ubatch: manifest
            .as_ref()
            .and_then(|value| value.n_ubatch)
            .unwrap_or(512),
        n_parallel: manifest
            .as_ref()
            .and_then(|value| value.n_parallel)
            .unwrap_or(1),
        n_threads: manifest
            .as_ref()
            .and_then(|value| value.n_threads)
            .unwrap_or(8),
        n_threads_batch: manifest
            .as_ref()
            .and_then(|value| value.n_threads_batch)
            .unwrap_or(8),
        n_gpu_layers: normalize_gpu_layers(
            manifest
                .as_ref()
                .and_then(|value| value.n_gpu_layers)
                .unwrap_or(-1),
        ),
        allowed_control_addrs: manifest
            .as_ref()
            .and_then(|value| value.allowed_control_addrs.clone())
            .filter(|items| !items.is_empty()),
    }))
}

fn discover_legacy_family_entries(
    models_dir: &Path,
    family: &str,
    family_path: &Path,
) -> Result<Vec<ManagedModelEntry>> {
    let task = infer_task_from_hints(family, None, None, None);
    if family_is_dependency_only(family) {
        return Ok(Vec::new());
    }

    let mmproj_path = discover_mmproj(family_path);
    let diarization_model_path = if task == ManagedModelTask::Transcription {
        discover_diarization_model(models_dir)
    } else {
        None
    };

    let mut entries = Vec::new();
    for path in discover_primary_supported_model_files(family_path, task)? {
        entries.push(ManagedModelEntry {
            id: legacy_model_id(&path),
            display_name: legacy_model_display_name(&path),
            family: family.to_string(),
            task,
            single_device_only: infer_single_device_only(
                task,
                family,
                path.file_stem().and_then(|value| value.to_str()),
                Some(&path),
                None,
            ),
            model_path: path.display().to_string(),
            mmproj_path: if task == ManagedModelTask::Responses {
                mmproj_path
                    .as_ref()
                    .map(|value| value.display().to_string())
            } else {
                None
            },
            diarization_model_path: diarization_model_path
                .as_ref()
                .map(|value| value.display().to_string()),
            execution_group_id: "cluster:auto".to_string(),
            retention_mode: RetentionMode::LoadOnDemand,
            load_on_demand_grace_seconds: match task {
                ManagedModelTask::Responses => {
                    if mmproj_path.is_some() {
                        InstanceModelKind::Vision
                    } else {
                        InstanceModelKind::Text
                    }
                }
                ManagedModelTask::Embeddings => InstanceModelKind::Embeddings,
                ManagedModelTask::Rerank => InstanceModelKind::Rerank,
                ManagedModelTask::Transcription => InstanceModelKind::Whisper,
            }
            .default_load_on_demand_grace_seconds(),
            n_ctx: 8192,
            n_batch: 512,
            n_ubatch: 512,
            n_parallel: 1,
            n_threads: 8,
            n_threads_batch: 8,
            n_gpu_layers: -1,
            allowed_control_addrs: None,
        });
    }
    Ok(entries)
}

fn largest_supported_model_file(
    model_dir: &Path,
    task: ManagedModelTask,
) -> Result<Option<PathBuf>> {
    let groups = supported_model_file_groups(model_dir, task)?;
    Ok(groups
        .into_iter()
        .max_by_key(|group| group.total_bytes)
        .map(|group| group.primary_path))
}

fn discover_mmproj(model_dir: &Path) -> Option<PathBuf> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(model_dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension()?.to_str()?;
        if !ext.eq_ignore_ascii_case("gguf") {
            continue;
        }
        let file_name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
        if file_name.contains("mmproj") {
            matches.push(path);
        }
    }
    matches.sort();
    matches.into_iter().next()
}

fn discover_diarization_model(models_dir: &Path) -> Option<PathBuf> {
    fn push_diarization_candidates(candidates: &mut Vec<PathBuf>, path: &Path, hint_name: &str) {
        let task = infer_task_from_hints(hint_name, None, None, None);
        if let Ok(Some(model_path)) = largest_supported_model_file(path, task) {
            candidates.push(model_path);
        }
    }

    let mut candidates = Vec::new();
    let entries = fs::read_dir(models_dir).ok()?;
    for entry in entries {
        let entry = entry.ok()?;
        let family_path = entry.path();
        if !family_path.is_dir() {
            continue;
        }
        let family_name = entry.file_name().to_string_lossy().into_owned();
        let lowered = family_name.to_ascii_lowercase();
        if lowered.contains("diarization") || lowered.contains("sortformer") {
            push_diarization_candidates(&mut candidates, &family_path, &family_name);
        }

        let child_entries = match fs::read_dir(&family_path) {
            Ok(value) => value,
            Err(_) => continue,
        };
        for child in child_entries {
            let child = child.ok()?;
            let child_path = child.path();
            if !child_path.is_dir() {
                continue;
            }
            let child_name = child.file_name().to_string_lossy().into_owned();
            let child_lowered = child_name.to_ascii_lowercase();
            if child_lowered.contains("diarization") || child_lowered.contains("sortformer") {
                push_diarization_candidates(&mut candidates, &child_path, &child_name);
            }
        }
    }

    candidates.sort_by(|lhs, rhs| {
        fs::metadata(rhs)
            .map(|value| value.len())
            .unwrap_or(0)
            .cmp(&fs::metadata(lhs).map(|value| value.len()).unwrap_or(0))
            .then_with(|| lhs.cmp(rhs))
    });
    candidates.into_iter().next()
}

fn resolve_model_path(model_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        model_dir.join(path)
    }
}

fn model_hint_blob(
    family: &str,
    dir_name: Option<&str>,
    model_path: Option<&Path>,
    display_name: Option<&str>,
) -> String {
    let mut parts = vec![family.to_ascii_lowercase()];
    if let Some(dir_name) = dir_name {
        parts.push(dir_name.to_ascii_lowercase());
    }
    if let Some(display_name) = display_name {
        parts.push(display_name.to_ascii_lowercase());
    }
    if let Some(model_path) = model_path {
        if let Some(file_name) = model_path.file_name().and_then(|value| value.to_str()) {
            parts.push(file_name.to_ascii_lowercase());
        }
    }
    parts.join(" ")
}

fn infer_task_from_hints(
    family: &str,
    dir_name: Option<&str>,
    model_path: Option<&Path>,
    display_name: Option<&str>,
) -> ManagedModelTask {
    let hints = model_hint_blob(family, dir_name, model_path, display_name);
    if hints.contains("embedding") {
        ManagedModelTask::Embeddings
    } else if hints.contains("rerank") {
        ManagedModelTask::Rerank
    } else if hints.contains("audio")
        || hints.contains("transcription")
        || hints.contains("transcribe")
        || hints.contains("speech")
        || hints.contains("realtime")
        || hints.contains("voxtral")
        || hints.contains("whisper")
        || hints.contains("sortformer")
    {
        ManagedModelTask::Transcription
    } else {
        ManagedModelTask::Responses
    }
}

fn infer_single_device_only(
    task: ManagedModelTask,
    family: &str,
    dir_name: Option<&str>,
    model_path: Option<&Path>,
    display_name: Option<&str>,
) -> bool {
    if task == ManagedModelTask::Transcription {
        return true;
    }
    let hints = model_hint_blob(family, dir_name, model_path, display_name);
    hints.contains("voxtral")
        || hints.contains("whisper")
        || hints.contains("sortformer")
        || hints.contains("realtime")
}

fn family_is_dependency_only(family: &str) -> bool {
    family.to_ascii_lowercase().contains("diarization")
}

fn is_supported_legacy_model_file(path: &Path, task: ManagedModelTask) -> bool {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
    {
        Some(value) if value == "gguf" => true,
        Some(value) if value == "bin" && task == ManagedModelTask::Transcription => true,
        _ => false,
    }
}

fn is_mmproj_file(path: &Path) -> bool {
    path.file_name()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .map(|value| value.contains("mmproj"))
        .unwrap_or(false)
}

fn legacy_model_id(path: &Path) -> String {
    sanitize_model_name(
        normalized_model_stem_for_catalog(path)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "model".to_string()),
    )
}

fn legacy_model_display_name(path: &Path) -> String {
    normalized_model_stem_for_catalog(path)
        .map(|value| value.replace('_', " "))
        .unwrap_or_else(|| "Model".to_string())
}

#[derive(Debug, Clone)]
struct SupportedModelFileGroup {
    primary_path: PathBuf,
    total_bytes: u64,
}

fn discover_primary_supported_model_files(
    model_dir: &Path,
    task: ManagedModelTask,
) -> Result<Vec<PathBuf>> {
    Ok(supported_model_file_groups(model_dir, task)?
        .into_iter()
        .map(|group| group.primary_path)
        .collect())
}

fn supported_model_file_groups(
    model_dir: &Path,
    task: ManagedModelTask,
) -> Result<Vec<SupportedModelFileGroup>> {
    let mut grouped: BTreeMap<String, Vec<(PathBuf, u64)>> = BTreeMap::new();
    for entry in fs::read_dir(model_dir)
        .with_context(|| format!("failed to read '{}'", model_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || !is_supported_legacy_model_file(&path, task) || is_mmproj_file(&path)
        {
            continue;
        }
        let key = logical_model_group_key(&path);
        grouped
            .entry(key)
            .or_default()
            .push((path, entry.metadata()?.len()));
    }

    let mut groups = Vec::new();
    for mut files in grouped.into_values() {
        files.sort_by(|lhs, rhs| {
            shard_sort_rank(&lhs.0)
                .cmp(&shard_sort_rank(&rhs.0))
                .then(lhs.0.cmp(&rhs.0))
        });
        let total_bytes = files.iter().map(|(_, size)| *size).sum();
        let primary_path = files
            .iter()
            .find(|(path, _)| is_primary_shard_path(path))
            .map(|(path, _)| path.clone())
            .unwrap_or_else(|| files[0].0.clone());
        groups.push(SupportedModelFileGroup {
            primary_path,
            total_bytes,
        });
    }
    Ok(groups)
}

fn normalized_model_stem_for_catalog(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy().into_owned();
    Some(strip_gguf_shard_suffix(&stem).unwrap_or(stem))
}

fn logical_model_group_key(path: &Path) -> String {
    normalized_model_stem_for_catalog(path)
        .unwrap_or_else(|| path.display().to_string())
        .to_ascii_lowercase()
}

fn shard_sort_rank(path: &Path) -> (i32, String) {
    let stem = path
        .file_stem()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lowered = stem.to_ascii_lowercase();
    if let Some((base, index, total)) = parse_gguf_shard_suffix(&lowered) {
        return (index, format!("{base}:{total}"));
    }
    (i32::MAX, lowered)
}

fn is_primary_shard_path(path: &Path) -> bool {
    let stem = path
        .file_stem()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    match parse_gguf_shard_suffix(&stem.to_ascii_lowercase()) {
        Some((_, index, _)) => index == 1,
        None => true,
    }
}

fn strip_gguf_shard_suffix(value: &str) -> Option<String> {
    parse_gguf_shard_suffix(&value.to_ascii_lowercase())
        .map(|(base, _, _)| value[..base.len()].to_string())
}

fn parse_gguf_shard_suffix(value: &str) -> Option<(String, i32, i32)> {
    let of_pos = value.rfind("-of-")?;
    let total_text = value.get(of_pos + 4..)?;
    if total_text.len() != 5 || !total_text.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let before = value.get(..of_pos)?;
    let dash_pos = before.rfind('-')?;
    let index_text = before.get(dash_pos + 1..)?;
    if index_text.len() != 5 || !index_text.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let base = before.get(..dash_pos)?.to_string();
    let index = index_text.parse::<i32>().ok()?;
    let total = total_text.parse::<i32>().ok()?;
    Some((base, index, total))
}

fn sanitize_model_name(value: String) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_dash = false;
    for ch in value.chars() {
        let keep = if ch.is_ascii_alphanumeric() {
            last_dash = false;
            Some(ch.to_ascii_lowercase())
        } else if matches!(ch, '-' | '_' | ' ' | '.') {
            if last_dash {
                None
            } else {
                last_dash = true;
                Some('-')
            }
        } else {
            None
        };
        if let Some(ch) = keep {
            out.push(ch);
        }
    }
    out.trim_matches('-').to_string()
}

fn parse_retention_mode(value: &str) -> Option<RetentionMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "keep_loaded" | "keeploaded" => Some(RetentionMode::KeepLoaded),
        "load_on_demand" | "loadondemand" => Some(RetentionMode::LoadOnDemand),
        _ => None,
    }
}
