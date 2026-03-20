use anyhow::{anyhow, bail, Context, Result};
use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tar::Archive;
use zip::ZipArchive;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

const DEFAULT_MANIFEST_URL: &str =
    "https://github.com/openresearchtools/engine/releases/latest/download/engine-manifest.json";
const BUNDLED_ENGINE_MANIFEST_JSON: &str =
    include_str!("../../runtime-manifests/engine-manifest.json");
const BUNDLED_ENGINE_MANIFEST_SOURCES_JSON: &str =
    include_str!("../../runtime-manifests/engine-manifest-sources.json");
const APP_UA: &str = "ENGINE-ClusterUI/1.0";
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
const EMBEDDED_RUNTIME_UNBLOCK_SCRIPT: &str =
    include_str!("../scripts/unblock-unsigned-runtime.ps1");
#[cfg(target_os = "macos")]
const EMBEDDED_RUNTIME_UNBLOCK_SCRIPT: &str =
    include_str!("../scripts/unblock-unsigned-runtime.sh");

#[derive(Debug, Clone, Deserialize, Serialize)]
struct EngineManifest {
    #[serde(default)]
    schema_version: i32,
    #[serde(default)]
    project: String,
    #[serde(default)]
    repository: String,
    #[serde(default)]
    tag: String,
    #[serde(default)]
    generated_at: String,
    #[serde(default)]
    release_url: String,
    #[serde(default)]
    assets: Vec<ManifestAsset>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ManifestAsset {
    #[serde(default)]
    id: String,
    #[serde(default)]
    platform: String,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    archive: String,
    #[serde(default)]
    file_name: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestSources {
    #[serde(default)]
    sources: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeInstallRecommendation {
    pub recommended_backend: String,
    pub recommended_reason: String,
    pub detected_gpu_label: Option<String>,
    pub cuda_candidate_notice: Option<String>,
    pub installed_backend: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct RuntimeInstallGpu {
    label: String,
    vendor: String,
    dedicated: bool,
    vram_gb: Option<f64>,
}

pub fn default_runtime_backends_for_platform() -> Vec<String> {
    let out = if cfg!(target_os = "windows") {
        vec!["vulkan".to_string(), "cuda".to_string()]
    } else if cfg!(target_os = "macos") {
        vec!["metal".to_string()]
    } else {
        vec!["vulkan".to_string()]
    };
    dedupe_preserve_order(out)
}

pub fn available_runtime_backends() -> Vec<String> {
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|path| path.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let manifest = load_engine_manifest_local_or_cached(&exe_dir);
    let out = manifest
        .ok()
        .map(|manifest| {
            filtered_assets_for_platform(&manifest)
                .into_iter()
                .map(|asset| asset.backend.trim().to_ascii_lowercase())
                .filter(|backend| !backend.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(default_runtime_backends_for_platform);
    let mut out = dedupe_preserve_order(out);
    if out.is_empty() {
        out = default_runtime_backends_for_platform();
    }
    out
}

pub fn bundled_controller_version_label() -> String {
    bundled_manifest_release_tag().unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn bundled_manifest_release_tag() -> Option<String> {
    serde_json::from_str::<EngineManifest>(BUNDLED_ENGINE_MANIFEST_JSON)
        .ok()
        .map(|manifest| manifest.tag.trim().to_string())
        .filter(|tag| !tag.is_empty())
}

pub fn runtime_install_recommendation(
    runtime_dir: &Path,
    available_backends: &[String],
) -> RuntimeInstallRecommendation {
    let installed_backend = detect_installed_runtime_backend(runtime_dir);
    let preferred_gpu = preferred_runtime_install_gpu();
    let detected_gpu_label = preferred_gpu.as_ref().map(|gpu| gpu.label.clone());

    if cfg!(target_os = "windows") {
        let fallback_backend = select_available_backend(available_backends, &["vulkan", "cuda"]);
        let mut recommendation = RuntimeInstallRecommendation {
            recommended_backend: fallback_backend.unwrap_or_else(|| "vulkan".to_string()),
            recommended_reason:
                "Vulkan is the default Windows runtime. CPU mode is still available inside it."
                    .to_string(),
            detected_gpu_label,
            cuda_candidate_notice: None,
            installed_backend,
        };

        if let Some(gpu) = preferred_gpu.as_ref() {
            if gpu.vendor.eq_ignore_ascii_case("nvidia") {
                if available_backends
                    .iter()
                    .any(|backend| backend.eq_ignore_ascii_case("cuda"))
                    && supports_conservative_cuda13_family(gpu.label.as_str())
                {
                    recommendation.cuda_candidate_notice = Some(format!(
                        "Detected {}. You might benefit from the CUDA engine if this GPU belongs to one of these NVIDIA CUDA 13 families: RTX 20/30/40/50, RTX A, Ada, or Blackwell. Vulkan remains the safe default.",
                        gpu.label
                    ));
                    recommendation.recommended_reason =
                        "Vulkan is the default Windows runtime. CUDA is offered as an optional faster path only for conservatively recognized NVIDIA CUDA 13 families."
                            .to_string();
                } else {
                    recommendation.recommended_reason =
                        "Detected an NVIDIA GPU, but it was not recognized as one of the conservatively supported CUDA 13 families, so Vulkan stays the default Windows runtime."
                            .to_string();
                }
            } else if gpu.dedicated {
                recommendation.recommended_reason =
                    "Detected a dedicated non-NVIDIA GPU, so Vulkan is the preferred Windows runtime."
                        .to_string();
            } else {
                recommendation.recommended_reason =
                    "Only integrated graphics were detected, so Vulkan is the safest Windows runtime and CPU mode stays available inside it."
                        .to_string();
            }
        } else {
            recommendation.recommended_reason =
                "No dedicated GPU was detected, so Vulkan is the safest Windows runtime and CPU mode stays available inside it."
                    .to_string();
        }

        return recommendation;
    }

    if cfg!(target_os = "macos") {
        return RuntimeInstallRecommendation {
            recommended_backend: select_available_backend(available_backends, &["metal"])
                .unwrap_or_else(|| "metal".to_string()),
            recommended_reason:
                "macOS uses the Metal runtime only, so the controller installs the Metal engine."
                    .to_string(),
            detected_gpu_label,
            cuda_candidate_notice: None,
            installed_backend,
        };
    }

    RuntimeInstallRecommendation {
        recommended_backend: select_available_backend(available_backends, &["vulkan"])
            .unwrap_or_else(|| "vulkan".to_string()),
        recommended_reason:
            "Linux uses the Vulkan runtime by default. CPU mode stays available inside that runtime too."
                .to_string(),
        detected_gpu_label,
        cuda_candidate_notice: None,
        installed_backend,
    }
}

pub fn install_or_repair_runtime_with_backend(
    runtime_dir: &Path,
    preferred_backend: Option<&str>,
    mut on_status: impl FnMut(String),
) -> Result<PathBuf> {
    if runtime_dir.as_os_str().is_empty() {
        bail!("runtime directory is empty");
    }

    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|path| path.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest = load_engine_manifest(&exe_dir)?;
    let mut assets = filtered_assets_for_platform(&manifest);
    if assets.is_empty() {
        bail!(
            "no runtime assets available for platform '{}'",
            current_platform_key()
        );
    }

    let preferred_backend = preferred_backend
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());
    let asset = if let Some(preferred) = preferred_backend {
        if let Some(index) = assets
            .iter()
            .position(|asset| asset.backend.trim().eq_ignore_ascii_case(&preferred))
        {
            assets.remove(index)
        } else {
            bail!(
                "backend '{}' is not available for platform '{}'",
                preferred,
                current_platform_key()
            );
        }
    } else {
        assets.remove(0)
    };

    on_status(format!(
        "Installing runtime asset: {}",
        describe_asset(&asset)
    ));
    let installed = install_runtime_asset(runtime_dir, &asset, &mut on_status)?;
    on_status("Runtime install finished.".to_string());
    Ok(installed)
}

pub fn runtime_missing_messages(runtime_dir: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    if !runtime_dir.exists() {
        missing.push(format!(
            "Runtime directory does not exist: {}",
            runtime_dir.display()
        ));
        return missing;
    }

    let managed_runtime_lib = managed_runtime_library_path(runtime_dir);
    if !managed_runtime_lib.exists() {
        let compat_runtime_lib = runtime_library_path(runtime_dir);
        if compat_runtime_lib.exists() && compat_runtime_lib != managed_runtime_lib {
            missing.push(format!(
                "Managed multi-node runtime is missing: {}. Found direct bridge runtime '{}' instead.",
                managed_runtime_lib
                    .file_name()
                    .map(|value| value.to_string_lossy().into_owned())
                    .unwrap_or_else(|| managed_runtime_lib.display().to_string()),
                compat_runtime_lib
                    .file_name()
                    .map(|value| value.to_string_lossy().into_owned())
                    .unwrap_or_else(|| compat_runtime_lib.display().to_string())
            ));
        } else {
            missing.push(format!(
                "Missing managed runtime library: {}",
                managed_runtime_lib
                    .file_name()
                    .map(|value| value.to_string_lossy().into_owned())
                    .unwrap_or_else(|| managed_runtime_lib.display().to_string())
            ));
        }
    }

    let ffmpeg_dir = ffmpeg_runtime_dir(runtime_dir);
    if !ffmpeg_dir.exists() {
        missing.push(format!(
            "Missing FFmpeg runtime directory: {}",
            ffmpeg_dir.display()
        ));
    }

    let pdfium_dir = runtime_dir.join("vendor").join("pdfium");
    if !pdfium_dir.exists() {
        missing.push(format!(
            "Missing PDFium runtime directory: {}",
            pdfium_dir.display()
        ));
    }

    missing
}

pub fn runtime_unblock_supported() -> bool {
    cfg!(any(target_os = "windows", target_os = "macos"))
}

pub fn unblock_installed_runtime(runtime_dir: &Path) -> Result<String> {
    if !runtime_unblock_supported() {
        return Ok("Runtime unblock is not needed on this platform.".to_string());
    }
    if runtime_dir.as_os_str().is_empty() {
        bail!("runtime directory is empty");
    }
    if !runtime_dir.is_dir() {
        bail!(
            "runtime directory does not exist: '{}'",
            runtime_dir.display()
        );
    }

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    {
        let (temp_root, script_path) = write_embedded_runtime_unblock_script()?;
        let run_result = run_embedded_runtime_unblock_script(&script_path, runtime_dir);
        let _ = fs::remove_dir_all(&temp_root);
        return run_result;
    }

    #[allow(unreachable_code)]
    Ok("Runtime unblock is not needed on this platform.".to_string())
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn embedded_runtime_unblock_script_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "unblock-unsigned-runtime.ps1"
    } else {
        "unblock-unsigned-runtime.sh"
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn write_embedded_runtime_unblock_script() -> Result<(PathBuf, PathBuf)> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_root = env::temp_dir().join(format!("engine-runtime-unblock-{stamp}"));
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("failed creating '{}'", temp_root.display()))?;

    let script_path = temp_root.join(embedded_runtime_unblock_script_file_name());
    fs::write(&script_path, EMBEDDED_RUNTIME_UNBLOCK_SCRIPT)
        .with_context(|| format!("failed writing '{}'", script_path.display()))?;

    #[cfg(unix)]
    {
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755))
            .with_context(|| format!("failed chmod '{}'", script_path.display()))?;
    }

    Ok((temp_root, script_path))
}

#[cfg(target_os = "windows")]
fn run_embedded_runtime_unblock_script(script_path: &Path, runtime_dir: &Path) -> Result<String> {
    let mut command = Command::new("powershell");
    command
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(script_path)
        .arg("-RuntimeDir")
        .arg(runtime_dir)
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    capture_runtime_unblock_output(command, script_path)
}

#[cfg(target_os = "macos")]
fn run_embedded_runtime_unblock_script(script_path: &Path, runtime_dir: &Path) -> Result<String> {
    let mut command = Command::new("sh");
    command
        .arg(script_path)
        .arg(runtime_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    capture_runtime_unblock_output(command, script_path)
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn capture_runtime_unblock_output(mut command: Command, script_path: &Path) -> Result<String> {
    let output = command.output().with_context(|| {
        format!(
            "failed to execute runtime unblock script '{}'",
            script_path.display()
        )
    })?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        let details = if stderr.is_empty() {
            stdout.clone()
        } else {
            stderr
        };
        bail!(
            "runtime unblock script failed ({}): {}",
            script_path.display(),
            details
        );
    }
    if stdout.is_empty() {
        Ok("Runtime unblock complete.".to_string())
    } else {
        Ok(stdout)
    }
}

#[cfg(target_os = "windows")]
pub fn detect_installed_windows_runtime_backend(runtime_dir: &Path) -> Option<String> {
    if !runtime_dir.exists() {
        return None;
    }

    let cuda_dirs = [
        runtime_dir.join("vendor").join("cuda"),
        runtime_dir.join("vendor"),
        runtime_dir.to_path_buf(),
    ];
    let has_cuda_marker = cuda_dirs.iter().any(|dir| {
        has_prefixed_file_case_insensitive(dir, "cublas", ".dll")
            || has_prefixed_file_case_insensitive(dir, "cudart", ".dll")
    });
    if has_cuda_marker {
        return Some("cuda".to_string());
    }

    if runtime_library_path(runtime_dir).exists() {
        return Some("vulkan".to_string());
    }
    None
}

fn default_app_root() -> Result<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let base = env::var_os("APPDATA").ok_or_else(|| anyhow!("APPDATA is not set"))?;
        Ok(PathBuf::from(base).join("OpenResearchTools"))
    }
    #[cfg(target_os = "macos")]
    {
        let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("OpenResearchTools"))
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("OpenResearchTools"))
    }
}

fn current_platform_key() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows-x64"
    } else if cfg!(target_os = "macos") {
        "macos-arm64"
    } else {
        "ubuntu-x64"
    }
}

fn filtered_assets_for_platform(manifest: &EngineManifest) -> Vec<ManifestAsset> {
    let mut assets = manifest
        .assets
        .iter()
        .filter(|asset| asset.platform.eq_ignore_ascii_case(current_platform_key()))
        .cloned()
        .collect::<Vec<_>>();
    assets.sort_by_key(|asset| backend_priority_for_platform(&asset.backend));
    assets
}

fn backend_priority_for_platform(backend: &str) -> usize {
    let backend = backend.to_ascii_lowercase();
    if cfg!(target_os = "windows") {
        return match backend.as_str() {
            "vulkan" => 0,
            "cuda" => 1,
            _ => 9,
        };
    }
    if cfg!(target_os = "macos") {
        return match backend.as_str() {
            "metal" => 0,
            _ => 9,
        };
    }
    match backend.as_str() {
        "vulkan" => 0,
        _ => 9,
    }
}

fn load_engine_manifest_local_or_cached(exe_dir: &Path) -> Result<EngineManifest> {
    let cache = cached_manifest_file()?;
    for candidate in local_manifest_file_candidates(exe_dir)
        .into_iter()
        .chain([cache])
    {
        if !candidate.exists() {
            continue;
        }
        let raw = match fs::read_to_string(&candidate) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if let Ok(manifest) = parse_manifest(&raw) {
            return Ok(manifest);
        }
    }
    if let Ok(manifest) = parse_manifest(BUNDLED_ENGINE_MANIFEST_JSON) {
        return Ok(manifest);
    }
    bail!("no local or cached runtime manifest found")
}

fn load_engine_manifest(exe_dir: &Path) -> Result<EngineManifest> {
    if let Ok(manifest) = load_engine_manifest_local_or_cached(exe_dir) {
        return Ok(manifest);
    }

    let client = Client::builder()
        .user_agent(APP_UA)
        .timeout(Duration::from_secs(25))
        .build()
        .context("failed to build HTTP client")?;
    let mut errors = Vec::new();
    for url in load_manifest_sources(exe_dir) {
        if url.trim().is_empty() {
            continue;
        }
        match client.get(&url).send() {
            Ok(response) => {
                let response = match response.error_for_status() {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(format!("{url}: {err}"));
                        continue;
                    }
                };
                let text = match response.text() {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(format!("{url}: {err}"));
                        continue;
                    }
                };
                match parse_manifest(&text) {
                    Ok(manifest) => {
                        if let Ok(cache_path) = cached_manifest_file() {
                            if let Some(parent) = cache_path.parent() {
                                fs::create_dir_all(parent).ok();
                            }
                            fs::write(
                                &cache_path,
                                serde_json::to_string_pretty(&manifest).unwrap_or_default(),
                            )
                            .ok();
                        }
                        return Ok(manifest);
                    }
                    Err(err) => errors.push(format!("{url}: {err}")),
                }
            }
            Err(err) => errors.push(format!("{url}: {err}")),
        }
    }
    bail!(
        "failed to load engine manifest from local or remote sources:\n{}",
        errors.join("\n")
    )
}

fn parse_manifest(raw: &str) -> Result<EngineManifest> {
    let manifest =
        serde_json::from_str::<EngineManifest>(raw).context("invalid engine manifest json")?;
    if manifest.assets.is_empty() {
        bail!("engine manifest has no assets");
    }
    Ok(manifest)
}

fn local_manifest_file_candidates(exe_dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![exe_dir.join("runtime-manifests").join("engine-manifest.json")];
    if should_probe_source_manifest_files(exe_dir) {
        if let Ok(cwd) = env::current_dir() {
            out.push(cwd.join("runtime-manifests").join("engine-manifest.json"));
        }
        out.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("runtime-manifests")
                .join("engine-manifest.json"),
        );
    }
    out
}

fn manifest_sources_candidates(exe_dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![exe_dir
        .join("runtime-manifests")
        .join("engine-manifest-sources.json")];
    if should_probe_source_manifest_files(exe_dir) {
        if let Ok(cwd) = env::current_dir() {
            out.push(
                cwd.join("runtime-manifests")
                    .join("engine-manifest-sources.json"),
            );
        }
        out.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("runtime-manifests")
                .join("engine-manifest-sources.json"),
        );
    }
    if let Ok(cache_path) = cached_manifest_file() {
        if let Some(parent) = cache_path.parent() {
            out.push(parent.join("engine-manifest-sources.json"));
        }
    }
    dedupe_paths(out)
}

fn should_probe_source_manifest_files(exe_dir: &Path) -> bool {
    !looks_like_installed_engine_runtime(exe_dir)
}

fn looks_like_installed_engine_runtime(exe_dir: &Path) -> bool {
    if exe_dir
        .file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("MacOS"))
        .unwrap_or(false)
    {
        if exe_dir
            .parent()
            .and_then(|path| path.parent())
            .and_then(|path| path.extension())
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case("app"))
            .unwrap_or(false)
        {
            return true;
        }
    }

    let normalized = exe_dir.to_string_lossy().replace('\\', "/");
    let normalized = normalized.to_ascii_lowercase();
    normalized.contains("/openresearchtools/engine")
}

fn cached_manifest_file() -> Result<PathBuf> {
    Ok(default_app_root()?
        .join("runtime-manifests")
        .join("engine-manifest.json"))
}

fn load_manifest_sources(exe_dir: &Path) -> Vec<String> {
    let mut out = vec![DEFAULT_MANIFEST_URL.to_string()];
    if let Ok(parsed) =
        serde_json::from_str::<ManifestSources>(BUNDLED_ENGINE_MANIFEST_SOURCES_JSON)
    {
        for source in parsed.sources {
            let source = source.trim();
            if source.is_empty() {
                continue;
            }
            if !out.iter().any(|known| known.eq_ignore_ascii_case(source)) {
                out.push(source.to_string());
            }
        }
    }
    for candidate in manifest_sources_candidates(exe_dir) {
        if !candidate.exists() {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&candidate) else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<ManifestSources>(&raw) else {
            continue;
        };
        for source in parsed.sources {
            let source = source.trim();
            if source.is_empty() {
                continue;
            }
            if !out.iter().any(|known| known.eq_ignore_ascii_case(source)) {
                out.push(source.to_string());
            }
        }
    }
    out
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        let key = path.to_string_lossy().to_ascii_lowercase();
        if out
            .iter()
            .any(|existing: &PathBuf| existing.to_string_lossy().to_ascii_lowercase() == key)
        {
            continue;
        }
        out.push(path);
    }
    out
}

fn dedupe_preserve_order(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for value in values {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() || !seen.insert(normalized) {
            continue;
        }
        out.push(value);
    }
    out
}

fn select_available_backend(available_backends: &[String], preferred: &[&str]) -> Option<String> {
    for desired in preferred {
        if let Some(found) = available_backends
            .iter()
            .find(|backend| backend.eq_ignore_ascii_case(desired))
        {
            return Some(found.clone());
        }
    }
    available_backends.first().cloned()
}

fn detect_installed_runtime_backend(runtime_dir: &Path) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        detect_installed_windows_runtime_backend(runtime_dir)
    }
    #[cfg(not(target_os = "windows"))]
    {
        if runtime_library_path(runtime_dir).exists() {
            if cfg!(target_os = "macos") {
                Some("metal".to_string())
            } else {
                Some("vulkan".to_string())
            }
        } else {
            None
        }
    }
}

fn describe_asset(asset: &ManifestAsset) -> String {
    let id = if asset.id.trim().is_empty() {
        asset.file_name.trim()
    } else {
        asset.id.trim()
    };
    if asset.backend.trim().is_empty() {
        id.to_string()
    } else {
        format!("{id} ({})", asset.backend.trim())
    }
}

fn install_runtime_asset(
    runtime_dir: &Path,
    asset: &ManifestAsset,
    on_status: &mut impl FnMut(String),
) -> Result<PathBuf> {
    let client = Client::builder()
        .user_agent(APP_UA)
        .timeout(Duration::from_secs(1800))
        .build()
        .context("failed building HTTP client")?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_root = env::temp_dir().join(format!("engine-cluster-runtime-{stamp}"));
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("failed creating '{}'", temp_root.display()))?;

    let install_result = (|| -> Result<PathBuf> {
        if asset.url.trim().is_empty() {
            bail!("runtime asset URL is empty");
        }
        let archive_name = if asset.file_name.trim().is_empty() {
            if asset.archive.eq_ignore_ascii_case("tar.gz") {
                "engine.tar.gz".to_string()
            } else {
                "engine.zip".to_string()
            }
        } else {
            asset.file_name.clone()
        };
        let archive_path = temp_root.join(&archive_name);
        on_status(format!("Downloading runtime: {}", asset.url.trim()));
        download_file_with_progress(&client, &asset.url, &archive_path, |done, total, speed| {
            let status = if let Some(total) = total {
                format!(
                    "Downloading runtime: {} / {} at {}/s",
                    human_bytes(done),
                    human_bytes(total),
                    human_bytes(speed as u64)
                )
            } else {
                format!(
                    "Downloading runtime: {} at {}/s",
                    human_bytes(done),
                    human_bytes(speed as u64)
                )
            };
            on_status(status);
        })?;

        if !asset.sha256.trim().is_empty() {
            let got = sha256_file(&archive_path)?;
            if !got.eq_ignore_ascii_case(asset.sha256.trim()) {
                bail!(
                    "runtime archive SHA256 mismatch: expected {}, got {}",
                    asset.sha256.trim(),
                    got
                );
            }
        }

        if runtime_dir.exists() {
            fs::remove_dir_all(runtime_dir)
                .with_context(|| format!("failed clearing '{}'", runtime_dir.display()))?;
        }
        fs::create_dir_all(runtime_dir)
            .with_context(|| format!("failed creating '{}'", runtime_dir.display()))?;

        let archive_kind = if asset.archive.eq_ignore_ascii_case("tar.gz")
            || archive_name.to_ascii_lowercase().ends_with(".tar.gz")
        {
            "tar.gz"
        } else {
            "zip"
        };
        on_status(format!("Extracting runtime archive ({archive_kind})..."));
        match archive_kind {
            "tar.gz" => extract_tar_gz_file(&archive_path, runtime_dir)?,
            _ => extract_zip_file(&archive_path, runtime_dir)?,
        }
        flatten_single_nested_root(runtime_dir)?;
        Ok(runtime_dir.to_path_buf())
    })();

    fs::remove_dir_all(&temp_root).ok();
    install_result
}

fn download_file_with_progress(
    client: &Client,
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>, f64),
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating '{}'", parent.display()))?;
    }
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("download request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("download request returned error status: {url}"))?;
    let total = response.content_length();
    let tmp = dest.with_extension("download");
    let mut file =
        File::create(&tmp).with_context(|| format!("failed creating '{}'", tmp.display()))?;

    let mut buf = vec![0_u8; 64 * 1024];
    let mut downloaded = 0_u64;
    let started = Instant::now();
    let mut last_emit = Instant::now();
    loop {
        let n = response
            .read(&mut buf)
            .with_context(|| format!("failed reading response body: {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("failed writing '{}'", tmp.display()))?;
        downloaded += n as u64;
        if last_emit.elapsed() >= Duration::from_millis(300) {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            on_progress(downloaded, total, downloaded as f64 / elapsed);
            last_emit = Instant::now();
        }
    }
    file.flush()
        .with_context(|| format!("failed flushing '{}'", tmp.display()))?;
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    on_progress(downloaded, total, downloaded as f64 / elapsed);

    if dest.exists() {
        fs::remove_file(dest).ok();
    }
    fs::rename(&tmp, dest).with_context(|| {
        format!(
            "failed moving downloaded file '{}' to '{}'",
            tmp.display(),
            dest.display()
        )
    })?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed opening '{}'", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed reading '{}'", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_zip_file(zip_path: &Path, out_dir: &Path) -> Result<()> {
    let file =
        File::open(zip_path).with_context(|| format!("failed opening '{}'", zip_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("failed to parse '{}'", zip_path.display()))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("failed reading zip entry #{i}"))?;
        let Some(enclosed) = entry.enclosed_name() else {
            continue;
        };
        let out_path = out_dir.join(enclosed);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)
                .with_context(|| format!("failed creating '{}'", out_path.display()))?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating '{}'", parent.display()))?;
        }
        let mut out = File::create(&out_path)
            .with_context(|| format!("failed creating '{}'", out_path.display()))?;
        std::io::copy(&mut entry, &mut out)
            .with_context(|| format!("failed extracting '{}'", out_path.display()))?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            let _ = fs::set_permissions(&out_path, fs::Permissions::from_mode(mode));
        }
    }
    Ok(())
}

fn extract_tar_gz_file(tgz_path: &Path, out_dir: &Path) -> Result<()> {
    let file =
        File::open(tgz_path).with_context(|| format!("failed opening '{}'", tgz_path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive
        .unpack(out_dir)
        .with_context(|| format!("failed extracting '{}'", tgz_path.display()))?;
    Ok(())
}

fn flatten_single_nested_root(out_dir: &Path) -> Result<()> {
    let mut entries = fs::read_dir(out_dir)
        .with_context(|| format!("failed listing '{}'", out_dir.display()))?
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy() != ".DS_Store")
        .collect::<Vec<_>>();
    if entries.len() != 1 {
        return Ok(());
    }
    let root = entries.remove(0).path();
    if !root.is_dir() {
        return Ok(());
    }
    for entry in
        fs::read_dir(&root).with_context(|| format!("failed listing '{}'", root.display()))?
    {
        let entry = entry.with_context(|| format!("failed reading '{}'", root.display()))?;
        let from = entry.path();
        let to = out_dir.join(entry.file_name());
        fs::rename(&from, &to)
            .with_context(|| format!("failed moving '{}' to '{}'", from.display(), to.display()))?;
    }
    fs::remove_dir_all(&root).with_context(|| format!("failed removing '{}'", root.display()))?;
    Ok(())
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut value = bytes as f64;
    let mut idx = 0usize;
    while value >= 1024.0 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{bytes} {}", UNITS[idx])
    } else {
        format!("{value:.1} {}", UNITS[idx])
    }
}

fn runtime_library_path(runtime_dir: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let preferred = managed_runtime_library_path(runtime_dir);
        if preferred.exists() {
            return preferred;
        }
        runtime_dir.join("llama-server-bridge.dll")
    }
    #[cfg(target_os = "macos")]
    {
        let preferred = managed_runtime_library_path(runtime_dir);
        if preferred.exists() {
            return preferred;
        }
        runtime_dir.join("libllama-server-bridge.dylib")
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let preferred = managed_runtime_library_path(runtime_dir);
        if preferred.exists() {
            return preferred;
        }
        runtime_dir.join("libllama-server-bridge.so")
    }
}

fn managed_runtime_library_path(runtime_dir: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        runtime_dir.join("multi-node-server.dll")
    }
    #[cfg(target_os = "macos")]
    {
        runtime_dir.join("libmulti-node-server.dylib")
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        runtime_dir.join("libmulti-node-server.so")
    }
}

fn ffmpeg_runtime_dir(runtime_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        runtime_dir.join("vendor").join("ffmpeg").join("bin")
    } else {
        runtime_dir.join("vendor").join("ffmpeg").join("lib")
    }
}

#[cfg(target_os = "windows")]
fn has_prefixed_file_case_insensitive(dir: &Path, prefix: &str, suffix: &str) -> bool {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return false;
    };
    let prefix_lc = prefix.to_ascii_lowercase();
    let suffix_lc = suffix.to_ascii_lowercase();
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.starts_with(&prefix_lc) && name.ends_with(&suffix_lc) {
            return true;
        }
    }
    false
}

fn preferred_runtime_install_gpu() -> Option<RuntimeInstallGpu> {
    #[cfg(target_os = "windows")]
    {
        let mut gpus = probe_windows_gpus();
        gpus.sort_by(|lhs, rhs| {
            let lhs_vram = lhs.vram_gb.unwrap_or(0.0);
            let rhs_vram = rhs.vram_gb.unwrap_or(0.0);
            rhs.dedicated
                .cmp(&lhs.dedicated)
                .then_with(|| {
                    rhs_vram
                        .partial_cmp(&lhs_vram)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then(lhs.label.cmp(&rhs.label))
        });
        gpus.into_iter().next()
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

#[cfg(target_os = "windows")]
fn probe_windows_gpus() -> Vec<RuntimeInstallGpu> {
    let dxdiag = probe_windows_gpus_dxdiag();
    if !dxdiag.is_empty() {
        return dxdiag;
    }
    probe_windows_gpus_wmi()
}

#[cfg(target_os = "windows")]
fn probe_windows_gpus_wmi() -> Vec<RuntimeInstallGpu> {
    let stdout = run_command_capture(
        "powershell.exe",
        &[
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "$ErrorActionPreference='Stop'; Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,VideoProcessor,PNPDeviceID | ConvertTo-Json -Compress",
        ],
    );
    let Some(stdout) = stdout else {
        return Vec::new();
    };
    let Some(items) = json_array(stdout.as_str()) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|item| {
            let name = json_field_string(&item, "Name");
            if name.trim().is_empty() || name.eq_ignore_ascii_case("Microsoft Basic Render Driver")
            {
                return None;
            }
            let integrated = is_integrated_name(name.as_str());
            let adapter_vram_gb = json_field_u64(&item, "AdapterRAM")
                .filter(|value| *value > 0)
                .map(|value| value as f64 / 1024_f64.powi(3));
            let inferred_vram_gb = infer_gpu_vram_from_name(name.as_str());
            let vram_gb = match (adapter_vram_gb, inferred_vram_gb) {
                (Some(adapter), Some(inferred)) => Some(adapter.max(inferred)),
                (Some(adapter), None) => Some(adapter),
                (None, Some(inferred)) => Some(inferred),
                (None, None) => None,
            };
            Some(RuntimeInstallGpu {
                label: name.clone(),
                vendor: vendor_from_gpu_name(name.as_str()),
                dedicated: !integrated,
                vram_gb,
            })
        })
        .collect()
}

#[cfg(target_os = "windows")]
fn probe_windows_gpus_dxdiag() -> Vec<RuntimeInstallGpu> {
    let temp_path = dxdiag_temp_path();
    let mut command = Command::new("dxdiag");
    command.args(["/whql:off", "/t"]).arg(temp_path.as_os_str());
    command.creation_flags(CREATE_NO_WINDOW);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(_) => {
            let _ = fs::remove_file(&temp_path);
            return Vec::new();
        }
    };

    let mut completed = false;
    for _ in 0..16 {
        match child.try_wait() {
            Ok(Some(status)) => {
                completed = status.success();
                break;
            }
            Ok(None) => thread::sleep(Duration::from_millis(250)),
            Err(_) => break,
        }
    }
    if !completed {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(&temp_path);
        return Vec::new();
    }

    let mut text = None::<String>;
    for _ in 0..12 {
        if let Ok(candidate) = fs::read_to_string(&temp_path) {
            if !candidate.trim().is_empty() {
                text = Some(candidate);
                break;
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    let _ = fs::remove_file(&temp_path);
    let Some(text) = text else {
        return Vec::new();
    };

    let mut out = Vec::<RuntimeInstallGpu>::new();
    let mut current_name = String::new();
    let mut current_vram_gb = None::<f64>;

    let flush_current = |out: &mut Vec<RuntimeInstallGpu>,
                         current_name: &mut String,
                         current_vram_gb: &mut Option<f64>| {
        let name = current_name.trim().to_string();
        if name.is_empty() || name.eq_ignore_ascii_case("Microsoft Basic Render Driver") {
            current_name.clear();
            *current_vram_gb = None;
            return;
        }
        let integrated = is_integrated_name(name.as_str());
        let vram_gb = current_vram_gb
            .take()
            .or_else(|| infer_gpu_vram_from_name(name.as_str()));
        out.push(RuntimeInstallGpu {
            label: name.clone(),
            vendor: vendor_from_gpu_name(name.as_str()),
            dedicated: !integrated,
            vram_gb,
        });
        current_name.clear();
        *current_vram_gb = None;
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if let Some(rest) = line.strip_prefix("Card name:") {
            if !current_name.trim().is_empty() {
                flush_current(&mut out, &mut current_name, &mut current_vram_gb);
            }
            current_name = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = line.strip_prefix("Dedicated Memory:") {
            if !current_name.trim().is_empty() {
                current_vram_gb = parse_memory_text_gb(rest.trim());
            }
            continue;
        }
        if current_vram_gb.is_none() {
            if let Some(rest) = line.strip_prefix("Display Memory:") {
                if !current_name.trim().is_empty() {
                    current_vram_gb = parse_memory_text_gb(rest.trim());
                }
            }
        }
    }
    if !current_name.trim().is_empty() {
        flush_current(&mut out, &mut current_name, &mut current_vram_gb);
    }

    let mut deduped = Vec::<RuntimeInstallGpu>::new();
    let mut seen = BTreeSet::<String>::new();
    for gpu in out {
        let key = format!(
            "{}::{}",
            gpu.label.to_ascii_lowercase(),
            gpu.vram_gb
                .map(|value| format!("{value:.2}"))
                .unwrap_or_default()
        );
        if seen.insert(key) {
            deduped.push(gpu);
        }
    }
    deduped
}

#[cfg(target_os = "windows")]
fn dxdiag_temp_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0);
    env::temp_dir().join(format!("engine-cluster-dxdiag-{stamp}.txt"))
}

#[cfg(target_os = "windows")]
fn run_command_capture(command: &str, args: &[&str]) -> Option<String> {
    let mut child = Command::new(command);
    child
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let mut child = child.spawn().ok()?;
    let deadline = Instant::now() + Duration::from_secs(4);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    if !status.success() {
        return None;
    }
    let mut stdout = String::new();
    child.stdout.take()?.read_to_string(&mut stdout).ok()?;
    let stdout = stdout.trim();
    if stdout.is_empty() {
        None
    } else {
        Some(stdout.to_string())
    }
}

#[cfg(target_os = "windows")]
fn json_array(raw: &str) -> Option<Vec<Value>> {
    let parsed = serde_json::from_str::<Value>(raw).ok()?;
    match parsed {
        Value::Array(items) => Some(items),
        Value::Null => Some(Vec::new()),
        value => Some(vec![value]),
    }
}

#[cfg(target_os = "windows")]
fn json_field_string(value: &Value, field: &str) -> String {
    value
        .get(field)
        .and_then(|item| {
            item.as_str()
                .map(|text| text.to_string())
                .or_else(|| item.as_i64().map(|number| number.to_string()))
                .or_else(|| item.as_u64().map(|number| number.to_string()))
        })
        .unwrap_or_default()
}

#[cfg(target_os = "windows")]
fn json_field_u64(value: &Value, field: &str) -> Option<u64> {
    value
        .get(field)
        .and_then(|item| {
            item.as_u64().or_else(|| {
                item.as_i64()
                    .and_then(|number| u64::try_from(number.max(0)).ok())
            })
        })
        .or_else(|| {
            value.get(field).and_then(|item| {
                item.as_str()
                    .and_then(|text| text.trim().parse::<u64>().ok())
            })
        })
}

fn vendor_from_gpu_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower.contains("nvidia")
        || lower.contains("geforce")
        || lower.contains("rtx")
        || lower.contains("gtx")
        || lower.contains("quadro")
        || lower.contains("tesla")
        || lower.contains("titan")
        || lower.contains("ada")
    {
        "nvidia".to_string()
    } else if lower.contains("amd")
        || lower.contains("radeon")
        || lower.contains("firepro")
        || lower.contains("instinct")
    {
        "amd".to_string()
    } else if lower.contains("intel")
        || lower.contains("iris")
        || lower.contains("uhd")
        || lower.contains("hd graphics")
    {
        "intel".to_string()
    } else if lower.contains("apple") {
        "apple".to_string()
    } else {
        "unknown".to_string()
    }
}

fn is_integrated_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("intel")
        || lower.contains("iris")
        || lower.contains("uhd")
        || lower.contains("hd graphics")
        || lower.contains("integrated")
        || lower.contains("radeon(tm) graphics")
}

fn infer_gpu_vram_from_name(name: &str) -> Option<f64> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("rtx 4090 laptop")
        || (lower.contains("rtx 4090") && lower.contains("laptop gpu"))
    {
        Some(16.0)
    } else if lower.contains("rtx 4080 laptop")
        || (lower.contains("rtx 4080") && lower.contains("laptop gpu"))
    {
        Some(12.0)
    } else if lower.contains("rtx 5090")
        || lower.contains("rtx 4090")
        || lower.contains("a6000")
        || lower.contains("6000 ada")
        || lower.contains("l40")
    {
        Some(24.0)
    } else if lower.contains("rtx 5080")
        || lower.contains("rtx 4080")
        || lower.contains("rtx 3090")
        || lower.contains("rtx 3080")
        || lower.contains("rtx 5070")
    {
        Some(16.0)
    } else if lower.contains("rtx 4070 ti")
        || lower.contains("rtx 4070")
        || lower.contains("rtx 3070")
        || lower.contains("rtx 3060 12")
        || lower.contains("rx 7900")
        || lower.contains("rx 7800")
    {
        Some(12.0)
    } else if lower.contains("rtx 4060")
        || lower.contains("rtx 3060")
        || lower.contains("rtx 2080")
        || lower.contains("rtx 2070")
        || lower.contains("rx 7700")
        || lower.contains("rx 6800")
    {
        Some(8.0)
    } else if lower.contains("gtx 1660")
        || lower.contains("gtx 1080")
        || lower.contains("gtx 1070")
        || lower.contains("rx 6700")
    {
        Some(6.0)
    } else {
        None
    }
}

fn supports_conservative_cuda13_family(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if !vendor_from_gpu_name(name).eq_ignore_ascii_case("nvidia") {
        return false;
    }
    [
        "rtx 20",
        "rtx 30",
        "rtx 40",
        "rtx 50",
        "rtx a",
        "ada",
        "blackwell",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn parse_memory_text_gb(text: &str) -> Option<f64> {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }
    let value = lower
        .split_whitespace()
        .find_map(|part| part.trim_end_matches("gb").parse::<f64>().ok())?;
    if lower.contains("tb") {
        Some(value * 1024.0)
    } else if lower.contains("mb") {
        Some(value / 1024.0)
    } else {
        Some(value)
    }
}
