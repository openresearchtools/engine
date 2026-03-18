use anyhow::{bail, Context, Result};
use reqwest::blocking::Client;
use reqwest::header::CONTENT_LENGTH;
use serde::Deserialize;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::catalog::default_models_dir;
use crate::model_metadata::{inspect_model_file, ModelFileMetadata};

const HUGGING_FACE_BASE: &str = "https://huggingface.co";
const USER_AGENT: &str = "ENGINE Cluster Controller/0.1";
const DOWNLOAD_BUFFER_SIZE: usize = 512 * 1024;
const DOWNLOAD_PROGRESS_INTERVAL_MS: u128 = 125;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub relative_path: String,
    pub file_name: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub metadata: Option<ModelFileMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackage {
    pub folder_name: String,
    pub display_name: String,
    pub path: PathBuf,
    pub model_files: Vec<ModelArtifact>,
    pub mmproj_files: Vec<ModelArtifact>,
    pub readme_path: Option<String>,
    pub guessed_repo_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RepoRemoteFile {
    pub path: String,
    pub size: Option<u64>,
    pub selected: bool,
}

#[derive(Debug, Clone)]
pub struct RepoPreview {
    pub repo_id: String,
    pub repo_url: String,
    pub revision: String,
    pub files: Vec<RepoRemoteFile>,
    pub readme_markdown: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DownloadProgress {
    pub current_file: Option<String>,
    pub completed_files: usize,
    pub total_files: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_second: u64,
    pub eta_seconds: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedAudioRepo {
    Whisper,
    RealtimeAudio,
    Diarization,
}

impl SupportedAudioRepo {
    pub fn title(self) -> &'static str {
        match self {
            Self::Whisper => "Transcription",
            Self::RealtimeAudio => "Realtime transcription",
            Self::Diarization => "Diarization",
        }
    }

    pub fn repo_id(self) -> &'static str {
        match self {
            Self::Whisper => "ggerganov/whisper.cpp",
            Self::RealtimeAudio => "openresearchtools/Voxtral-Mini-4B-Realtime-2602",
            Self::Diarization => "openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf",
        }
    }

    pub fn repo_url(self) -> &'static str {
        match self {
            Self::Whisper => "https://huggingface.co/ggerganov/whisper.cpp",
            Self::RealtimeAudio => {
                "https://huggingface.co/openresearchtools/Voxtral-Mini-4B-Realtime-2602"
            }
            Self::Diarization => {
                "https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf"
            }
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Whisper => {
                "Recommended native transcription family. Prefer ggml-large-v3-turbo.bin."
            }
            Self::RealtimeAudio => "Supported realtime audio family for the native bridge path.",
            Self::Diarization => {
                "Supported Sortformer diarization companion for transcription with speakers."
            }
        }
    }

    pub fn matches_recommended_file(self, path: &str) -> bool {
        let lowered = path.to_ascii_lowercase();
        match self {
            Self::Whisper => lowered.ends_with("ggml-large-v3-turbo.bin"),
            Self::RealtimeAudio | Self::Diarization => {
                lowered.ends_with(".gguf") && !lowered.contains("mmproj")
            }
        }
    }
}

pub fn supported_audio_repos() -> [SupportedAudioRepo; 3] {
    [
        SupportedAudioRepo::Whisper,
        SupportedAudioRepo::RealtimeAudio,
        SupportedAudioRepo::Diarization,
    ]
}

pub fn models_root_dir() -> Result<PathBuf> {
    default_models_dir()
}

pub fn model_store_change_marker_path(models_dir: &Path) -> PathBuf {
    models_dir.join(".clusterui-model-store.stamp")
}

pub fn touch_model_store_change_marker(models_dir: &Path) -> Result<()> {
    fs::create_dir_all(models_dir)
        .with_context(|| format!("failed to create '{}'", models_dir.display()))?;
    let marker_path = model_store_change_marker_path(models_dir);
    let mut file = File::create(&marker_path)
        .with_context(|| format!("failed to update '{}'", marker_path.display()))?;
    file.write_all(
        format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
        .as_bytes(),
    )
    .with_context(|| format!("failed to write '{}'", marker_path.display()))?;
    file.flush().ok();
    Ok(())
}

pub fn discover_model_packages(models_dir: &Path) -> Result<Vec<ModelPackage>> {
    if !models_dir.exists() {
        return Ok(Vec::new());
    }

    let mut packages = Vec::new();
    for entry in fs::read_dir(models_dir)
        .with_context(|| format!("failed to read '{}'", models_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(package) = discover_model_package(&path)? {
            packages.push(package);
        }
    }

    packages.sort_by(|lhs, rhs| lhs.display_name.cmp(&rhs.display_name));
    Ok(packages)
}

pub fn load_local_package_readme(package: &ModelPackage) -> Option<String> {
    let relative = package.readme_path.as_ref()?;
    let path = package.path.join(path_from_repo_path(relative));
    fs::read_to_string(path).ok()
}

pub fn normalize_repo_id(input: &str) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("Enter a repo like owner/name.");
    }

    let without_base = trimmed
        .strip_prefix("https://huggingface.co/")
        .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
        .unwrap_or(trimmed);
    let repo = without_base
        .trim_matches('/')
        .strip_prefix("models/")
        .unwrap_or(without_base.trim_matches('/'));
    let mut segments = repo
        .split('/')
        .filter(|segment| !segment.trim().is_empty())
        .collect::<Vec<_>>();
    if segments.len() < 2 {
        bail!("Enter a repo like owner/name.");
    }
    segments.truncate(2);
    Ok(format!("{}/{}", segments[0].trim(), segments[1].trim()))
}

pub fn suggested_folder_name_for_repo(repo_id: &str) -> String {
    sanitize_folder_name(&repo_id.replace('/', "__"))
}

pub fn sanitize_folder_name(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_sep = false;
    for ch in value.trim().chars() {
        let mapped = if ch.is_ascii_alphanumeric() {
            last_sep = false;
            Some(ch)
        } else if matches!(ch, '-' | '_' | ' ' | '.') {
            if last_sep {
                None
            } else {
                last_sep = true;
                Some('_')
            }
        } else {
            None
        };
        if let Some(ch) = mapped {
            out.push(ch);
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn fetch_repo_preview(
    repo_id: &str,
    recommended: Option<SupportedAudioRepo>,
) -> Result<RepoPreview> {
    let repo_id = normalize_repo_id(repo_id)?;
    let client = build_http_client()?;
    let url = format!(
        "{HUGGING_FACE_BASE}/api/models/{}",
        encode_repo_id(&repo_id)
    );
    let response = client
        .get(url)
        .send()
        .context("failed to reach Hugging Face")?
        .error_for_status()
        .context("repo lookup failed")?;
    let info = response
        .text()
        .context("failed to read Hugging Face response body")
        .and_then(|text| {
            serde_json::from_str::<HuggingFaceModelInfo>(&text)
                .context("failed to parse Hugging Face response")
        })?;
    let revision = match info.sha {
        Some(value) if !value.trim().is_empty() => value,
        _ => "main".to_string(),
    };
    let mut files = info
        .siblings
        .into_iter()
        .map(|sibling| {
            let default_selected = recommended
                .map(|kind| kind.matches_recommended_file(&sibling.rfilename))
                .unwrap_or_else(|| default_selected_repo_file(&sibling.rfilename));
            RepoRemoteFile {
                path: sibling.rfilename,
                size: sibling.size,
                selected: default_selected,
            }
        })
        .collect::<Vec<_>>();
    files.sort_by(|lhs, rhs| {
        lhs.path
            .to_ascii_lowercase()
            .cmp(&rhs.path.to_ascii_lowercase())
    });

    let readme_markdown = files
        .iter()
        .find(|file| file.path.eq_ignore_ascii_case("README.md"))
        .and_then(|file| fetch_repo_text_file(&client, &repo_id, &revision, &file.path).ok());

    Ok(RepoPreview {
        repo_id: repo_id.clone(),
        repo_url: format!("{HUGGING_FACE_BASE}/{}", encode_repo_id(&repo_id)),
        revision,
        files,
        readme_markdown,
    })
}

pub fn download_repo_files<F>(
    repo_id: &str,
    revision: &str,
    folder_name: &str,
    files: &[RepoRemoteFile],
    mut on_progress: F,
) -> Result<PathBuf>
where
    F: FnMut(DownloadProgress) + Send + 'static,
{
    let repo_id = normalize_repo_id(repo_id)?;
    let selected_files = files
        .iter()
        .filter(|file| file.selected)
        .cloned()
        .collect::<Vec<_>>();
    if selected_files.is_empty() {
        bail!("Select at least one file before downloading.");
    }

    let root = models_root_dir()?;
    fs::create_dir_all(&root).with_context(|| format!("failed to create '{}'", root.display()))?;
    let target_dir = root.join(sanitize_folder_name(folder_name));
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create '{}'", target_dir.display()))?;

    let client = build_http_client()?;
    let total_bytes = selected_files
        .iter()
        .map(|file| file.size.unwrap_or(0))
        .sum::<u64>();
    let total_files = selected_files.len();
    let download_started = Instant::now();
    let mut downloaded_bytes = 0u64;
    let mut completed_files = 0usize;
    let mut effective_total_bytes = total_bytes;

    on_progress(DownloadProgress {
        current_file: None,
        completed_files,
        total_files,
        downloaded_bytes,
        total_bytes: effective_total_bytes,
        bytes_per_second: 0,
        eta_seconds: None,
    });

    for file in selected_files {
        let destination = target_dir.join(path_from_repo_path(&file.path));
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create '{}'", parent.display()))?;
        }

        if destination.exists() {
            let existing_size = fs::metadata(&destination)
                .with_context(|| format!("failed to stat '{}'", destination.display()))?
                .len();
            if file.size.is_some() && file.size == Some(existing_size) {
                downloaded_bytes = downloaded_bytes.saturating_add(existing_size);
                completed_files += 1;
                on_progress(DownloadProgress {
                    current_file: Some(file.path.clone()),
                    completed_files,
                    total_files,
                    downloaded_bytes,
                    total_bytes: effective_total_bytes,
                    bytes_per_second: download_speed_bytes_per_second(
                        downloaded_bytes,
                        download_started,
                    ),
                    eta_seconds: download_eta_seconds(
                        downloaded_bytes,
                        effective_total_bytes,
                        download_started,
                    ),
                });
                continue;
            }
        }

        let url = resolve_file_url(&repo_id, revision, &file.path);
        let mut response = client
            .get(url)
            .send()
            .with_context(|| format!("failed downloading '{}'", file.path))?
            .error_for_status()
            .with_context(|| format!("failed downloading '{}'", file.path))?;

        let resolved_size = file
            .size
            .or_else(|| content_length_from_headers(response.headers()))
            .unwrap_or(0);
        if file.size.is_none() && resolved_size > 0 {
            effective_total_bytes = effective_total_bytes.saturating_add(resolved_size);
        }
        let part_path = destination.with_extension("part");
        let mut output = File::create(&part_path)
            .with_context(|| format!("failed to create '{}'", part_path.display()))?;
        let mut buffer = vec![0u8; DOWNLOAD_BUFFER_SIZE];
        let mut file_downloaded = 0u64;
        let mut last_emit = Instant::now();

        loop {
            let read = response
                .read(&mut buffer)
                .with_context(|| format!("failed reading '{}'", file.path))?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .with_context(|| format!("failed writing '{}'", part_path.display()))?;

            let read = read as u64;
            file_downloaded = file_downloaded.saturating_add(read);
            downloaded_bytes = downloaded_bytes.saturating_add(read);

            if last_emit.elapsed().as_millis() >= DOWNLOAD_PROGRESS_INTERVAL_MS {
                on_progress(DownloadProgress {
                    current_file: Some(file.path.clone()),
                    completed_files,
                    total_files,
                    downloaded_bytes,
                    total_bytes: effective_total_bytes.max(downloaded_bytes.max(resolved_size)),
                    bytes_per_second: download_speed_bytes_per_second(
                        downloaded_bytes,
                        download_started,
                    ),
                    eta_seconds: download_eta_seconds(
                        downloaded_bytes,
                        effective_total_bytes.max(downloaded_bytes.max(resolved_size)),
                        download_started,
                    ),
                });
                last_emit = Instant::now();
            }
        }

        output
            .flush()
            .with_context(|| format!("failed to flush '{}'", part_path.display()))?;
        fs::rename(&part_path, &destination)
            .or_else(|_| {
                fs::copy(&part_path, &destination)?;
                fs::remove_file(&part_path)
            })
            .with_context(|| format!("failed to finalize '{}'", destination.display()))?;

        completed_files += 1;
        on_progress(DownloadProgress {
            current_file: Some(file.path.clone()),
            completed_files,
            total_files,
            downloaded_bytes,
            total_bytes: effective_total_bytes.max(downloaded_bytes),
            bytes_per_second: download_speed_bytes_per_second(downloaded_bytes, download_started),
            eta_seconds: download_eta_seconds(
                downloaded_bytes,
                effective_total_bytes.max(downloaded_bytes),
                download_started,
            ),
        });
    }

    Ok(target_dir)
}

pub fn import_local_model_files(folder_name: &str, files: &[PathBuf]) -> Result<PathBuf> {
    if files.is_empty() {
        bail!("Pick one or more local files first.");
    }

    let root = models_root_dir()?;
    fs::create_dir_all(&root).with_context(|| format!("failed to create '{}'", root.display()))?;
    let target_dir = root.join(sanitize_folder_name(folder_name));
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create '{}'", target_dir.display()))?;

    for file in files {
        if !file.is_file() {
            continue;
        }
        let Some(file_name) = file.file_name() else {
            continue;
        };
        let destination = target_dir.join(file_name);
        fs::copy(file, &destination).with_context(|| {
            format!(
                "failed to copy '{}' to '{}'",
                file.display(),
                destination.display()
            )
        })?;
    }

    Ok(target_dir)
}

fn discover_model_package(path: &Path) -> Result<Option<ModelPackage>> {
    let folder_name = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let mut model_files = Vec::new();
    let mut mmproj_files = Vec::new();
    let mut readme_path = None::<String>;

    let mut pending = vec![path.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in
            fs::read_dir(&dir).with_context(|| format!("failed to read '{}'", dir.display()))?
        {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                pending.push(entry_path);
                continue;
            }

            let relative = entry_path
                .strip_prefix(path)
                .unwrap_or(&entry_path)
                .to_string_lossy()
                .replace('\\', "/");
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let lowered_name = file_name.to_ascii_lowercase();
            if readme_path.is_none() && lowered_name == "readme.md" {
                readme_path = Some(relative.clone());
            }

            let artifact = ModelArtifact {
                relative_path: relative,
                file_name,
                size_bytes: entry.metadata()?.len(),
                metadata: inspect_model_file(&entry_path),
            };

            let ext = entry_path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase());
            if matches!(ext.as_deref(), Some("gguf")) && lowered_name.contains("mmproj") {
                mmproj_files.push(artifact);
            } else if matches!(ext.as_deref(), Some("gguf") | Some("bin")) {
                model_files.push(artifact);
            }
        }
    }

    if model_files.is_empty() && mmproj_files.is_empty() {
        return Ok(None);
    }

    model_files.sort_by(|lhs, rhs| lhs.relative_path.cmp(&rhs.relative_path));
    mmproj_files.sort_by(|lhs, rhs| lhs.relative_path.cmp(&rhs.relative_path));

    Ok(Some(ModelPackage {
        guessed_repo_id: folder_name
            .split_once("__")
            .map(|(owner, repo)| format!("{owner}/{repo}")),
        display_name: folder_name.replace('_', " "),
        folder_name,
        path: path.to_path_buf(),
        model_files,
        mmproj_files,
        readme_path,
    }))
}

fn default_selected_repo_file(path: &str) -> bool {
    let lowered = path.to_ascii_lowercase();
    lowered.ends_with(".gguf")
        || lowered.ends_with(".bin")
        || lowered.ends_with("readme.md")
        || lowered.contains("mmproj")
}

fn fetch_repo_text_file(
    client: &Client,
    repo_id: &str,
    revision: &str,
    path: &str,
) -> Result<String> {
    let url = resolve_file_url(repo_id, revision, path);
    client
        .get(url)
        .send()
        .with_context(|| format!("failed fetching '{}'", path))?
        .error_for_status()
        .with_context(|| format!("failed fetching '{}'", path))?
        .text()
        .with_context(|| format!("failed reading '{}'", path))
}

fn build_http_client() -> Result<Client> {
    Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .build()
        .context("failed to create HTTP client")
}

fn encode_repo_id(repo_id: &str) -> String {
    repo_id
        .split('/')
        .map(encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn resolve_file_url(repo_id: &str, revision: &str, path: &str) -> String {
    format!(
        "{HUGGING_FACE_BASE}/{}/resolve/{}/{}?download=true",
        encode_repo_id(repo_id),
        encode_path_segment(revision),
        encode_repo_path(path)
    )
}

fn encode_repo_path(path: &str) -> String {
    path.split('/')
        .map(encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_path_segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn path_from_repo_path(path: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for segment in path.split('/').filter(|segment| !segment.is_empty()) {
        out.push(segment);
    }
    out
}

fn content_length_from_headers(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
}

fn download_speed_bytes_per_second(downloaded_bytes: u64, started_at: Instant) -> u64 {
    let elapsed = started_at.elapsed().as_secs_f64();
    if elapsed <= f64::EPSILON {
        0
    } else {
        (downloaded_bytes as f64 / elapsed).round() as u64
    }
}

fn download_eta_seconds(
    downloaded_bytes: u64,
    total_bytes: u64,
    started_at: Instant,
) -> Option<u64> {
    if total_bytes == 0 || downloaded_bytes >= total_bytes {
        return None;
    }

    let speed = download_speed_bytes_per_second(downloaded_bytes, started_at);
    if speed == 0 {
        None
    } else {
        Some((total_bytes.saturating_sub(downloaded_bytes) + speed - 1) / speed)
    }
}

#[derive(Deserialize)]
struct HuggingFaceModelInfo {
    #[serde(default)]
    sha: Option<String>,
    #[serde(default)]
    siblings: Vec<HuggingFaceModelSibling>,
}

#[derive(Deserialize)]
struct HuggingFaceModelSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}
