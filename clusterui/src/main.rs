#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app_icon;
mod agent;
mod catalog;
mod cluster_api;
mod controller_ui;
mod instance_presets;
mod model_metadata;
mod model_store;
mod node_host;
mod protocol;
mod public_server;
mod runtime_installer;
mod settings;
mod tray;

use agent::{
    ensure_local_agent, preferred_local_control_addr, stop_local_support_processes,
    transfer_model_artifact_between_agents, AgentClient,
};
use app_icon::build_engine_window_icon;
use catalog::{default_models_dir, ManagedModelEntry, ManagedModelTask};
use cluster_api::{
    default_runtime_dir, ChatRequest, CreateInstanceParams, InferenceMetrics, InstanceModelKind,
    RetentionMode,
};
use eframe::{egui, App, CreationContext, NativeOptions};
use egui_commonmark::CommonMarkCache;
use instance_presets::{load_instance_presets, save_instance_presets, InstancePreset};
use model_metadata::{
    estimate_runtime_vram, expanded_model_dependency_relative_paths, inspect_model_file,
    ModelFileMetadata, RuntimeVramEstimate,
};
use model_store::{
    discover_model_packages, download_repo_files, fetch_repo_preview, import_local_model_files,
    load_local_package_readme, model_store_change_marker_path, models_root_dir,
    sanitize_folder_name, suggested_folder_name_for_repo,
    DownloadProgress as ModelDownloadProgress, ModelPackage, RepoPreview, SupportedAudioRepo,
};
use node_host::{
    add_agent_peer, dump_agent_state, remove_agent_peer, run_host_services, NodeHost, StartupArgs,
};
use protocol::{
    ClusterModelArtifactInfo, ClusterModelPackageInfo, DiscoveryMode, DiscoveryStatus, LinkMetrics,
    ModelFileNodeAvailability, ModelPackageNodeAvailability, NodeSnapshot, PairingRequestInfo,
    PeerInfo, PlacementPlan, PlacementStrategy, PublicApiConfigUpdate, PublicApiStatus,
    TelemetrySnapshot, CLUSTER_AGENT_RPC_PORT,
};
use settings::{
    default_controller_settings, load_controller_settings, save_controller_settings,
    ControllerSettings, ControllerThemePreference,
};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tray::{ControllerTray, TrayAction};

fn main() -> eframe::Result<()> {
    let args = StartupArgs::from_env();
    if args.agent_mode {
        if let Err(err) = run_host_services(args.runtime_dir, args.bind_addr) {
            eprintln!("cluster agent failed: {err}");
        }
        return Ok(());
    }

    if args.dump_state {
        if let Err(err) = dump_agent_state(&args.bind_addr) {
            eprintln!("cluster controller probe failed: {err}");
        }
        return Ok(());
    }

    if let Some(control_addr) = args.add_peer.clone() {
        if let Err(err) = add_agent_peer(&args.bind_addr, &control_addr) {
            eprintln!("cluster controller add-peer failed: {err}");
        }
        return Ok(());
    }

    if let Some(control_addr) = args.remove_peer.clone() {
        if let Err(err) = remove_agent_peer(&args.bind_addr, &control_addr) {
            eprintln!("cluster controller remove-peer failed: {err}");
        }
        return Ok(());
    }

    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Engine")
            .with_app_id("openresearchtools.engine")
            .with_icon(build_engine_window_icon())
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([820.0, 280.0]),
        persist_window: false,
        ..Default::default()
    };
    eframe::run_native(
        "Engine",
        options,
        Box::new(move |cc| Ok(Box::new(ClusterControllerApp::new(cc, args)))),
    )
}

struct ClusterControllerApp {
    host: NodeHost,
    runtime_dir_edit: String,
    local_control_addr_edit: String,
    status: String,
    tray: Option<ControllerTray>,
    peers: Vec<PeerInfo>,
    pairing_requests: Vec<PairingRequestInfo>,
    discovery_status: DiscoveryStatus,
    nodes: Vec<NodeSnapshot>,
    preview_node: Option<NodeSnapshot>,
    telemetry: Vec<TelemetrySnapshot>,
    selected_control_addr: Option<String>,
    selected_rpc_peer_addrs: BTreeSet<String>,
    selected_instance_id: Option<i64>,
    create_params: CreateInstanceParams,
    last_plan: Option<PlacementPlan>,
    placement_candidates: Vec<PlacementPlan>,
    chat_request: ChatRequest,
    chat_response: String,
    last_chat_metrics: Option<InferenceMetrics>,
    selected_page: controller_ui::ControllerPage,
    selected_about_document: controller_ui::AboutDocument,
    controller_version_label: String,
    instance_creation_open: bool,
    server_status: Option<PublicApiStatus>,
    server_enabled: bool,
    server_bind_addr_edit: String,
    server_allow_cors: bool,
    server_allowed_origins_edit: String,
    server_allowed_client_ips_edit: String,
    server_api_key_edit: String,
    server_generated_api_key: Option<String>,
    managed_models: Vec<ManagedModelEntry>,
    selected_managed_model_id: Option<String>,
    model_packages: Vec<ModelPackage>,
    available_model_packages: Vec<ModelPackage>,
    available_model_package_details: HashMap<String, ClusterModelPackageInfo>,
    selected_model_package_folder: Option<String>,
    selected_model_file_path: Option<String>,
    selected_mmproj_file_path: Option<String>,
    selected_diarization_package_folder: Option<String>,
    selected_diarization_file_path: Option<String>,
    instance_model_kind: String,
    last_suggested_instance_name: Option<String>,
    instance_name_customized: bool,
    selected_instance_preset_name: Option<String>,
    instance_preset_name_edit: String,
    instance_presets: Vec<InstancePreset>,
    model_search: String,
    model_family_filter: String,
    model_store_mode: ModelStoreMode,
    model_store_repo_input: String,
    model_store_repo_folder_name: String,
    model_store_repo_preview: Option<RepoPreview>,
    model_store_worker_rx: Option<mpsc::Receiver<ModelStoreEvent>>,
    model_store_busy: ModelStoreBusyState,
    model_store_error: Option<String>,
    model_store_progress: ModelDownloadProgress,
    model_store_import_name: String,
    model_store_import_files: Vec<PathBuf>,
    model_transfer_worker_rx: Option<mpsc::Receiver<ModelTransferEvent>>,
    model_transfer_in_progress: bool,
    model_transfer_progress: Option<ModelTransferProgress>,
    readme_markdown_cache: CommonMarkCache,
    controller_worker_tx: mpsc::Sender<ControllerEvent>,
    controller_worker_rx: mpsc::Receiver<ControllerEvent>,
    local_connect_in_progress: bool,
    local_connect_pending_pair_discovery_seconds: Option<u64>,
    cluster_refresh_in_progress: bool,
    cluster_refresh_pending: bool,
    telemetry_refresh_in_progress: bool,
    telemetry_refresh_pending: bool,
    placement_refresh_in_progress: bool,
    placement_refresh_pending: bool,
    managed_models_refresh_in_progress: bool,
    managed_models_refresh_pending: bool,
    available_model_packages_refresh_in_progress: bool,
    available_model_packages_refresh_pending: bool,
    server_status_refresh_in_progress: bool,
    server_status_refresh_pending: bool,
    pairing_poll_in_progress: bool,
    link_benchmark_in_progress: bool,
    manual_refresh_in_progress: bool,
    last_manual_refresh_completed_at: Option<Instant>,
    selected_supported_audio_repo: SupportedAudioRepo,
    allowed_control_addrs: BTreeSet<String>,
    runtime_missing: Vec<String>,
    runtime_install_backends: Vec<String>,
    runtime_install_recommendation: runtime_installer::RuntimeInstallRecommendation,
    selected_runtime_install_backend: usize,
    runtime_install_in_progress: bool,
    runtime_install_status: Option<String>,
    runtime_install_rx: Option<mpsc::Receiver<RuntimeInstallEvent>>,
    show_advanced_instance_editor: bool,
    show_cpu_devices: bool,
    show_integrated_gpus: bool,
    multi_node_rpc_enabled: bool,
    last_auto_refresh: Instant,
    last_telemetry_refresh: Instant,
    last_model_store_marker_poll: Instant,
    last_model_store_marker_modified: Option<SystemTime>,
    auto_refresh_enabled: bool,
    window_hidden_to_tray: bool,
    allow_exit: bool,
    shutdown_requested_at: Option<Instant>,
    last_saved_settings: Option<ControllerSettings>,
    theme_preference: ControllerThemePreference,
    last_pairing_poll: Instant,
    seen_pairing_request_ids: BTreeSet<String>,
    pairing_modal_request_id: Option<String>,
    pairing_request_attention_pending: bool,
    startup_initialized: bool,
    startup_prepare_in_progress: bool,
    startup_connect_due_at: Option<Instant>,
}

enum RuntimeInstallEvent {
    Status(String),
    Finished(Result<PathBuf, String>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelStoreMode {
    LocalInstalled,
    RepoBrowser,
    ImportLocal,
    SupportedAudio,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelStoreBusyState {
    Idle,
    LoadingRepo,
    Downloading,
    Importing,
}

enum ModelStoreEvent {
    RepoLoaded(Result<RepoPreview, String>),
    DownloadProgress(ModelDownloadProgress),
    DownloadFinished(Result<PathBuf, String>),
    ImportFinished(Result<PathBuf, String>),
}

#[derive(Clone)]
struct ModelTransferFilePlan {
    folder_name: String,
    relative_path: String,
    source_control_addr: String,
    source_display_name: String,
    dest_control_addr: String,
    dest_display_name: String,
    size_bytes: u64,
}

#[derive(Clone, Default)]
struct ModelTransferProgress {
    current_file: Option<String>,
    source_display_name: String,
    dest_display_name: String,
    completed_files: usize,
    total_files: usize,
    transferred_bytes: u64,
    total_bytes: u64,
    bytes_per_second: u64,
}

enum ModelTransferEvent {
    Progress(ModelTransferProgress),
    Finished(Result<ModelTransferSummary, String>),
}

struct ModelTransferSummary {
    source_display_names: Vec<String>,
    dest_display_name: String,
    completed_files: usize,
    skipped_files: usize,
}

enum ControllerEvent {
    StartupPrepared(Result<StartupPreparation, String>),
    LocalAgentConnected(Result<String, String>),
    ClusterRefresh(Result<ClusterRefreshPayload, String>),
    TelemetryRefresh(Result<Vec<TelemetrySnapshot>, String>),
    PairingPoll(Result<PairingPollPayload, String>),
    LinkBenchmarks(Result<String, String>),
    PlacementCandidates(Result<Vec<PlacementPlan>, String>),
    ManagedModels(Result<Vec<ManagedModelEntry>, String>),
    AvailableModelPackages(Result<Vec<ClusterModelPackageInfo>, String>),
    ServerStatus(Result<PublicApiStatus, String>),
}

struct ClusterRefreshPayload {
    peers: Vec<PeerInfo>,
    pairing_requests: Vec<PairingRequestInfo>,
    discovery_status: DiscoveryStatus,
    nodes: Vec<NodeSnapshot>,
    telemetry: Vec<TelemetrySnapshot>,
    preview_node: Option<NodeSnapshot>,
    selected_control_addr: Option<String>,
    warnings: Vec<String>,
}

struct PairingPollPayload {
    peers: Vec<PeerInfo>,
    pairing_requests: Vec<PairingRequestInfo>,
    discovery_status: DiscoveryStatus,
    topology_changed: bool,
}

struct StartupPreparation {
    models_dir: PathBuf,
    model_packages: Result<Vec<ModelPackage>, String>,
    model_store_marker_modified: Option<SystemTime>,
    runtime_missing: Vec<String>,
    runtime_install_backends: Vec<String>,
    runtime_install_recommendation: runtime_installer::RuntimeInstallRecommendation,
}

impl ClusterControllerApp {
    fn new(cc: &CreationContext<'_>, args: StartupArgs) -> Self {
        let saved_settings = load_controller_settings();
        let saved_theme_preference = saved_settings
            .as_ref()
            .map(|settings| settings.theme_preference)
            .unwrap_or_default();
        controller_ui::configure_controller_visuals(&cc.egui_ctx, saved_theme_preference.as_egui());
        let (controller_worker_tx, controller_worker_rx) = mpsc::channel();
        let default_runtime_dir_value = default_runtime_dir().ok();
        let default_control_addr = agent::default_local_agent_addr();
        let use_saved_runtime_dir = default_runtime_dir_value
            .as_ref()
            .is_some_and(|value| args.runtime_dir == *value)
            && saved_settings.is_some();
        let use_saved_control_addr =
            args.bind_addr == default_control_addr && saved_settings.is_some();
        let runtime_dir = if use_saved_runtime_dir {
            saved_settings
                .as_ref()
                .map(|settings| PathBuf::from(settings.runtime_dir.clone()))
                .unwrap_or_else(|| args.runtime_dir.clone())
        } else {
            args.runtime_dir.clone()
        };
        let bind_addr = if use_saved_control_addr {
            let configured = saved_settings
                .as_ref()
                .map(|settings| settings.local_control_addr.clone())
                .unwrap_or_else(|| args.bind_addr.clone());
            preferred_local_control_addr(&configured)
        } else {
            preferred_local_control_addr(&args.bind_addr)
        };
        let promote_legacy_cluster_defaults = saved_settings
            .as_ref()
            .map(|settings| {
                settings.local_control_addr.trim() == default_control_addr
                    && !settings.multi_node_rpc_enabled
                    && settings.paired_peers.is_empty()
            })
            .unwrap_or(false);
        let runtime_dir_edit = runtime_dir.display().to_string();
        let local_control_addr_edit = bind_addr.clone();
        let host = NodeHost::new(runtime_dir, bind_addr);
        let runtime_install_backends = runtime_installer::available_runtime_backends();
        let mut app = Self {
            host,
            runtime_dir_edit,
            local_control_addr_edit,
            status: String::new(),
            tray: None,
            peers: Vec::new(),
            pairing_requests: Vec::new(),
            discovery_status: DiscoveryStatus {
                mode: DiscoveryMode::Off,
                active: false,
                expires_unix_ms: 0,
            },
            nodes: Vec::new(),
            preview_node: None,
            telemetry: Vec::new(),
            selected_control_addr: None,
            selected_rpc_peer_addrs: BTreeSet::new(),
            selected_instance_id: None,
            create_params: CreateInstanceParams {
                name: String::new(),
                managed_model_id: None,
                model_path: String::new(),
                mmproj_path: None,
                diarization_model_path: None,
                execution_group_id: String::new(),
                rpc_servers: None,
                manual_device_allocations: Vec::new(),
                manual_devices_csv: None,
                manual_tensor_split: None,
                preferred_owner_control_addr: None,
                retention_mode: RetentionMode::KeepLoaded,
                load_on_demand_grace_seconds: InstanceModelKind::Text
                    .default_load_on_demand_grace_seconds(),
                embedding: false,
                reranking: false,
                model_kind: InstanceModelKind::Text,
                single_device_only: false,
                allow_cpu: false,
                allow_integrated_gpu: false,
                n_ctx: 32768,
                n_batch: 2048,
                n_ubatch: 2048,
                n_parallel: 1,
                n_threads: 0,
                n_threads_batch: 0,
                n_gpu_layers: -1,
            },
            last_plan: None,
            placement_candidates: Vec::new(),
            chat_request: ChatRequest {
                instance_id: 0,
                prompt: String::new(),
                n_predict: 10000,
                temperature: 0.7,
                top_p: 0.95,
                top_k: 40,
                min_p: 0.05,
                repeat_last_n: 64,
                repeat_penalty: 1.05,
                reasoning: None,
                reasoning_budget: -2147483648,
                reasoning_format: None,
            },
            chat_response: String::new(),
            last_chat_metrics: None,
            selected_page: controller_ui::ControllerPage::Instances,
            selected_about_document: controller_ui::AboutDocument::EngineLicense,
            controller_version_label: runtime_installer::bundled_controller_version_label(),
            instance_creation_open: false,
            server_status: None,
            server_enabled: false,
            server_bind_addr_edit: "127.0.0.1:46310".to_string(),
            server_allow_cors: false,
            server_allowed_origins_edit: String::new(),
            server_allowed_client_ips_edit: String::new(),
            server_api_key_edit: String::new(),
            server_generated_api_key: None,
            managed_models: Vec::new(),
            selected_managed_model_id: None,
            model_packages: Vec::new(),
            available_model_packages: Vec::new(),
            available_model_package_details: HashMap::new(),
            selected_model_package_folder: None,
            selected_model_file_path: None,
            selected_mmproj_file_path: None,
            selected_diarization_package_folder: None,
            selected_diarization_file_path: None,
            instance_model_kind: "text".to_string(),
            last_suggested_instance_name: None,
            instance_name_customized: false,
            selected_instance_preset_name: None,
            instance_preset_name_edit: String::new(),
            instance_presets: load_instance_presets(),
            model_search: String::new(),
            model_family_filter: String::new(),
            model_store_mode: ModelStoreMode::LocalInstalled,
            model_store_repo_input: String::new(),
            model_store_repo_folder_name: String::new(),
            model_store_repo_preview: None,
            model_store_worker_rx: None,
            model_store_busy: ModelStoreBusyState::Idle,
            model_store_error: None,
            model_store_progress: ModelDownloadProgress::default(),
            model_store_import_name: String::new(),
            model_store_import_files: Vec::new(),
            model_transfer_worker_rx: None,
            model_transfer_in_progress: false,
            model_transfer_progress: None,
            readme_markdown_cache: CommonMarkCache::default(),
            controller_worker_tx,
            controller_worker_rx,
            local_connect_in_progress: false,
            local_connect_pending_pair_discovery_seconds: None,
            cluster_refresh_in_progress: false,
            cluster_refresh_pending: false,
            telemetry_refresh_in_progress: false,
            telemetry_refresh_pending: false,
            placement_refresh_in_progress: false,
            placement_refresh_pending: false,
            managed_models_refresh_in_progress: false,
            managed_models_refresh_pending: false,
            available_model_packages_refresh_in_progress: false,
            available_model_packages_refresh_pending: false,
            server_status_refresh_in_progress: false,
            server_status_refresh_pending: false,
            pairing_poll_in_progress: false,
            link_benchmark_in_progress: false,
            manual_refresh_in_progress: false,
            last_manual_refresh_completed_at: None,
            selected_supported_audio_repo: SupportedAudioRepo::Whisper,
            allowed_control_addrs: BTreeSet::new(),
            runtime_missing: Vec::new(),
            runtime_install_backends,
            runtime_install_recommendation:
                runtime_installer::RuntimeInstallRecommendation::default(),
            selected_runtime_install_backend: 0,
            runtime_install_in_progress: false,
            runtime_install_status: None,
            runtime_install_rx: None,
            show_advanced_instance_editor: false,
            show_cpu_devices: false,
            show_integrated_gpus: false,
            multi_node_rpc_enabled: false,
            last_auto_refresh: Instant::now(),
            last_telemetry_refresh: Instant::now() - Duration::from_secs(5),
            last_model_store_marker_poll: Instant::now() - Duration::from_secs(2),
            last_model_store_marker_modified: None,
            auto_refresh_enabled: false,
            window_hidden_to_tray: false,
            allow_exit: false,
            shutdown_requested_at: None,
            last_saved_settings: None,
            theme_preference: saved_theme_preference,
            last_pairing_poll: Instant::now() - Duration::from_secs(5),
            seen_pairing_request_ids: BTreeSet::new(),
            pairing_modal_request_id: None,
            pairing_request_attention_pending: false,
            startup_initialized: false,
            startup_prepare_in_progress: false,
            startup_connect_due_at: None,
        };
        if let Some(settings) = saved_settings {
            app.server_bind_addr_edit = settings.server_bind_addr;
            app.server_allow_cors = settings.server_allow_cors;
            app.server_allowed_origins_edit = settings.server_allowed_origins;
            app.server_allowed_client_ips_edit = settings.server_allowed_client_ips;
            app.show_cpu_devices = settings.show_cpu_devices;
            app.show_integrated_gpus = settings.show_integrated_gpus;
            app.auto_refresh_enabled = settings.auto_refresh_enabled;
            app.multi_node_rpc_enabled = if promote_legacy_cluster_defaults {
                true
            } else {
                settings.multi_node_rpc_enabled
            };
            app.theme_preference = settings.theme_preference;
        }
        if let Ok(tray) = ControllerTray::new(&cc.egui_ctx) {
            app.tray = Some(tray);
        }
        app.persist_controller_settings_if_changed();
        app
    }

    fn run_startup_initialization(&mut self) {
        if self.startup_initialized {
            return;
        }
        self.startup_initialized = true;
        self.enqueue_startup_preparation();
        self.status = "Loading Engine controller...".to_string();
    }

    fn enqueue_startup_preparation(&mut self) {
        if self.startup_prepare_in_progress {
            return;
        }
        self.startup_prepare_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        let runtime_dir = PathBuf::from(self.runtime_dir_edit.trim());
        let models_dir = self.local_models_dir();
        std::thread::spawn(move || {
            let model_packages = discover_model_packages(&models_dir).map_err(|err| {
                format!(
                    "failed to scan model folders in '{}': {err:#}",
                    models_dir.display()
                )
            });
            let model_store_marker_modified =
                fs::metadata(model_store_change_marker_path(&models_dir))
                    .ok()
                    .and_then(|meta| meta.modified().ok());
            let runtime_install_backends = runtime_installer::available_runtime_backends();
            let runtime_missing = runtime_installer::runtime_missing_messages(&runtime_dir);
            let runtime_install_recommendation = runtime_installer::runtime_install_recommendation(
                &runtime_dir,
                &runtime_install_backends,
            );
            let _ = tx.send(ControllerEvent::StartupPrepared(Ok(StartupPreparation {
                models_dir,
                model_packages,
                model_store_marker_modified,
                runtime_missing,
                runtime_install_backends,
                runtime_install_recommendation,
            })));
        });
    }

    fn apply_runtime_state_snapshot(
        &mut self,
        runtime_missing: Vec<String>,
        runtime_install_backends: Vec<String>,
        runtime_install_recommendation: runtime_installer::RuntimeInstallRecommendation,
    ) {
        self.runtime_missing = runtime_missing;
        self.runtime_install_backends = runtime_install_backends;
        self.runtime_install_recommendation = runtime_install_recommendation;
        if let Some(installed_backend) = self
            .runtime_install_recommendation
            .installed_backend
            .as_deref()
        {
            if let Some(index) = self
                .runtime_install_backends
                .iter()
                .position(|backend| backend.eq_ignore_ascii_case(installed_backend))
            {
                self.selected_runtime_install_backend = index;
            }
        } else if self.selected_runtime_install_backend >= self.runtime_install_backends.len() {
            if let Some(index) = self.runtime_install_backends.iter().position(|backend| {
                backend
                    .eq_ignore_ascii_case(&self.runtime_install_recommendation.recommended_backend)
            }) {
                self.selected_runtime_install_backend = index;
            } else {
                self.selected_runtime_install_backend = 0;
            }
        }
        if self.runtime_install_backends.is_empty() {
            self.selected_runtime_install_backend = 0;
        } else if self.selected_runtime_install_backend >= self.runtime_install_backends.len() {
            self.selected_runtime_install_backend = self.runtime_install_backends.len() - 1;
        }
    }

    fn apply_local_model_packages_scan_result(
        &mut self,
        models_dir: &Path,
        result: Result<Vec<ModelPackage>, String>,
        marker_modified: Option<SystemTime>,
    ) {
        match result {
            Ok(packages) => {
                self.model_packages = packages;
                self.model_store_error = None;
                self.rebuild_available_model_packages_local_only();
            }
            Err(err) => {
                self.model_packages.clear();
                self.available_model_packages.clear();
                self.available_model_package_details.clear();
                self.selected_model_package_folder = None;
                self.selected_model_file_path = None;
                self.selected_mmproj_file_path = None;
                self.selected_diarization_package_folder = None;
                self.selected_diarization_file_path = None;
                self.model_store_error = Some(if err.is_empty() {
                    format!("failed to scan model folders in '{}'", models_dir.display())
                } else {
                    err
                });
            }
        }
        self.last_model_store_marker_modified = marker_modified;
    }

    fn handle_startup_prepared(&mut self, result: Result<StartupPreparation, String>) {
        self.startup_prepare_in_progress = false;
        match result {
            Ok(payload) => {
                self.apply_local_model_packages_scan_result(
                    &payload.models_dir,
                    payload.model_packages,
                    payload.model_store_marker_modified,
                );
                self.apply_runtime_state_snapshot(
                    payload.runtime_missing,
                    payload.runtime_install_backends,
                    payload.runtime_install_recommendation,
                );
                if self.runtime_missing.is_empty() {
                    self.startup_connect_due_at = Some(Instant::now() + Duration::from_secs(2));
                    self.status = "Engine loaded. Connecting to local node...".to_string();
                } else {
                    self.selected_page = controller_ui::ControllerPage::Settings;
                    self.status =
                        "Engine runtime missing or incomplete. Open Settings and install it."
                            .to_string();
                }
            }
            Err(err) => {
                self.status = err;
            }
        }
    }

    fn rebuild_host_from_inputs(&mut self) {
        self.host = NodeHost::new(
            PathBuf::from(self.runtime_dir_edit.trim()),
            self.local_control_addr_edit.trim().to_string(),
        );
    }

    fn current_controller_settings(&self) -> ControllerSettings {
        let mut settings = load_controller_settings().unwrap_or_else(default_controller_settings);
        settings.runtime_dir = self.runtime_dir_edit.trim().to_string();
        settings.local_control_addr = self.local_control_addr_edit.trim().to_string();
        settings.server_bind_addr = self.server_bind_addr_edit.trim().to_string();
        settings.server_allow_cors = self.server_allow_cors;
        settings.server_allowed_origins = self.server_allowed_origins_edit.clone();
        settings.server_allowed_client_ips = self.server_allowed_client_ips_edit.clone();
        settings.show_cpu_devices = self.show_cpu_devices;
        settings.show_integrated_gpus = self.show_integrated_gpus;
        settings.auto_refresh_enabled = self.auto_refresh_enabled;
        settings.multi_node_rpc_enabled = self.multi_node_rpc_enabled;
        settings.theme_preference = self.theme_preference;
        settings
    }

    fn apply_theme_preference(&self, ctx: &egui::Context) {
        ctx.set_theme(self.theme_preference.as_egui());
    }

    fn persist_controller_settings_if_changed(&mut self) {
        let next = self.current_controller_settings();
        if self.last_saved_settings.as_ref() == Some(&next) {
            return;
        }
        if save_controller_settings(&next).is_ok() {
            self.last_saved_settings = Some(next);
        }
    }

    fn selected_instance_preset(&self) -> Option<&InstancePreset> {
        let selected = self.selected_instance_preset_name.as_ref()?;
        self.instance_presets
            .iter()
            .find(|preset| &preset.name == selected)
    }

    fn default_instance_preset_name(&self) -> String {
        if !self.instance_preset_name_edit.trim().is_empty() {
            return self.instance_preset_name_edit.trim().to_string();
        }
        if !self.create_params.name.trim().is_empty() {
            return self.create_params.name.trim().to_string();
        }
        self.selected_model_package()
            .map(|package| package.display_name.clone())
            .unwrap_or_else(|| "preset".to_string())
    }

    fn open_instance_creation(&mut self, reset_draft_name: bool) {
        self.selected_page = controller_ui::ControllerPage::Instances;
        self.instance_creation_open = true;
        self.selected_instance_id = None;
        if reset_draft_name {
            self.create_params.name.clear();
            self.last_suggested_instance_name = None;
            self.instance_name_customized = false;
            self.selected_instance_preset_name = None;
            self.instance_preset_name_edit.clear();
        }
        self.sync_selected_model_package();
    }

    fn build_current_instance_preset(&self) -> Option<InstancePreset> {
        let package_folder = self.selected_model_package_folder.clone()?;
        let model_file_path = self.selected_model_file_path.clone()?;
        Some(InstancePreset {
            name: self.default_instance_preset_name(),
            model_kind: self.instance_model_kind.clone(),
            model_package_folder: package_folder,
            model_file_path,
            mmproj_file_path: self.selected_mmproj_file_path.clone(),
            diarization_package_folder: self.selected_diarization_package_folder.clone(),
            diarization_file_path: self.selected_diarization_file_path.clone(),
            instance_name: self.create_params.name.clone(),
            retention_mode: self.create_params.retention_mode,
            load_on_demand_grace_seconds: self.create_params.load_on_demand_grace_seconds,
            n_ctx: self.create_params.n_ctx,
            n_batch: self.create_params.n_batch,
            n_ubatch: self.create_params.n_ubatch,
            n_parallel: self.create_params.n_parallel,
            n_threads: self.create_params.n_threads,
            n_threads_batch: self.create_params.n_threads_batch,
            n_gpu_layers: self.create_params.n_gpu_layers,
            max_predict: self.chat_request.n_predict,
            allow_cpu: self.show_cpu_devices,
            allow_integrated_gpu: self.show_integrated_gpus,
            preferred_owner_control_addr: self.create_params.preferred_owner_control_addr.clone(),
            execution_group_id: self.create_params.execution_group_id.clone(),
            rpc_servers: self.create_params.rpc_servers.clone(),
            manual_device_allocations: self.create_params.manual_device_allocations.clone(),
        })
    }

    fn save_current_instance_preset(&mut self) {
        let Some(mut preset) = self.build_current_instance_preset() else {
            self.status =
                "Pick a model folder and primary file before saving a preset.".to_string();
            return;
        };
        preset.name = self.default_instance_preset_name();
        if preset.name.trim().is_empty() {
            self.status = "Preset name is required.".to_string();
            return;
        }
        if let Some(existing) = self
            .instance_presets
            .iter_mut()
            .find(|entry| entry.name.eq_ignore_ascii_case(&preset.name))
        {
            *existing = preset.clone();
        } else {
            self.instance_presets.push(preset.clone());
            self.instance_presets.sort_by(|lhs, rhs| {
                lhs.name
                    .to_ascii_lowercase()
                    .cmp(&rhs.name.to_ascii_lowercase())
            });
        }
        match save_instance_presets(&self.instance_presets) {
            Ok(()) => {
                self.selected_instance_preset_name = Some(preset.name.clone());
                self.instance_preset_name_edit = preset.name.clone();
                self.status = format!("Saved preset '{}'.", preset.name);
            }
            Err(err) => {
                self.status = format!("failed to save presets: {err}");
            }
        }
    }

    fn apply_instance_preset_by_name(&mut self, preset_name: &str) {
        let Some(preset) = self
            .instance_presets
            .iter()
            .find(|entry| entry.name == preset_name)
            .cloned()
        else {
            self.status = format!("Preset '{preset_name}' was not found.");
            return;
        };
        self.selected_instance_preset_name = Some(preset.name.clone());
        self.instance_preset_name_edit = preset.name.clone();
        self.instance_model_kind = preset.model_kind.clone();
        self.selected_model_package_folder = Some(preset.model_package_folder.clone());
        self.selected_model_file_path = Some(preset.model_file_path.clone());
        self.selected_mmproj_file_path = preset.mmproj_file_path.clone();
        self.selected_diarization_package_folder = preset.diarization_package_folder.clone();
        self.selected_diarization_file_path = preset.diarization_file_path.clone();
        self.show_cpu_devices = preset.allow_cpu;
        self.show_integrated_gpus = preset.allow_integrated_gpu;
        self.chat_request.n_predict = preset.max_predict;
        self.open_instance_creation(false);
        self.create_params.name = preset.instance_name.clone();
        self.last_suggested_instance_name =
            self.current_suggested_instance_name()
                .and_then(|suggested| {
                    if suggested == preset.instance_name {
                        Some(suggested)
                    } else {
                        None
                    }
                });
        self.instance_name_customized = self.last_suggested_instance_name.is_none();
        self.create_params.retention_mode = preset.retention_mode;
        self.create_params.load_on_demand_grace_seconds =
            preset.load_on_demand_grace_seconds.max(0);
        self.create_params.n_ctx = preset.n_ctx;
        self.create_params.n_batch = preset.n_batch;
        self.create_params.n_ubatch = preset.n_ubatch;
        self.create_params.n_parallel = preset.n_parallel.max(1);
        self.create_params.n_threads = preset.n_threads;
        self.create_params.n_threads_batch = preset.n_threads_batch;
        self.create_params.n_gpu_layers = preset.n_gpu_layers;
        self.create_params.preferred_owner_control_addr =
            preset.preferred_owner_control_addr.clone();
        self.create_params.execution_group_id = preset.execution_group_id.clone();
        self.create_params.rpc_servers = preset.rpc_servers.clone();
        self.create_params.manual_device_allocations = preset.manual_device_allocations.clone();
        self.create_params.manual_devices_csv = None;
        self.create_params.manual_tensor_split = None;
        self.create_params.allow_cpu = preset.allow_cpu;
        self.create_params.allow_integrated_gpu = preset.allow_integrated_gpu;
        if let Some(owner_control_addr) = self.create_params.preferred_owner_control_addr.clone() {
            self.selected_control_addr = Some(owner_control_addr);
            self.selected_rpc_peer_addrs = self.manual_remote_peer_control_addrs_for_owner();
            let _ = self.refresh_selected_preview();
        }
        self.allowed_control_addrs.clear();
        self.refresh_placement_candidates();
        self.status = format!("Loaded preset '{}'.", preset.name);
    }

    fn delete_selected_instance_preset(&mut self) {
        let Some(selected) = self.selected_instance_preset_name.clone() else {
            self.status = "Choose a preset to delete.".to_string();
            return;
        };
        let before_len = self.instance_presets.len();
        self.instance_presets
            .retain(|preset| preset.name != selected);
        if self.instance_presets.len() == before_len {
            self.status = format!("Preset '{selected}' was not found.");
            return;
        }
        match save_instance_presets(&self.instance_presets) {
            Ok(()) => {
                self.selected_instance_preset_name = None;
                self.instance_preset_name_edit.clear();
                self.status = format!("Deleted preset '{selected}'.");
            }
            Err(err) => {
                self.status = format!("failed to save presets: {err}");
            }
        }
    }

    fn selected_model_metadata(&self) -> Option<ModelFileMetadata> {
        self.selected_model_artifact()
            .and_then(|artifact| artifact.metadata)
            .or_else(|| {
                let model_path = self.create_params.model_path.trim();
                if model_path.is_empty() {
                    None
                } else {
                    inspect_model_file(Path::new(model_path))
                }
            })
    }

    fn selected_model_artifact(&self) -> Option<model_store::ModelArtifact> {
        let selected_relative_path = self.selected_model_file_path.as_deref()?;
        self.selected_model_package_detail()
            .and_then(|details| {
                details
                    .model_file_availability
                    .iter()
                    .find(|entry| entry.artifact.relative_path == selected_relative_path)
                    .map(|entry| entry.artifact.clone())
            })
            .or_else(|| {
                self.selected_model_package()?
                    .model_files
                    .iter()
                    .find_map(|artifact| {
                        (artifact.relative_path == selected_relative_path).then(|| artifact.clone())
                    })
            })
    }

    fn selected_mmproj_artifact(&self) -> Option<model_store::ModelArtifact> {
        let selected_relative_path = self.selected_mmproj_file_path.as_deref()?;
        self.selected_model_package_detail()
            .and_then(|details| {
                details
                    .mmproj_file_availability
                    .iter()
                    .find(|entry| entry.artifact.relative_path == selected_relative_path)
                    .map(|entry| entry.artifact.clone())
            })
            .or_else(|| {
                self.selected_model_package()?
                    .mmproj_files
                    .iter()
                    .find_map(|artifact| {
                        (artifact.relative_path == selected_relative_path).then(|| artifact.clone())
                    })
            })
    }

    fn selected_model_dependency_artifacts(&self) -> Vec<model_store::ModelArtifact> {
        let Some(package) = self.selected_model_package() else {
            return Vec::new();
        };
        let Some(selected_relative_path) = self.selected_model_file_path.as_deref() else {
            return Vec::new();
        };
        let dependency_paths = expanded_model_dependency_relative_paths(
            selected_relative_path,
            package
                .model_files
                .iter()
                .map(|artifact| artifact.relative_path.as_str()),
        );
        dependency_paths
            .into_iter()
            .filter_map(|relative_path| {
                self.selected_model_package_detail()
                    .and_then(|details| {
                        details
                            .model_file_availability
                            .iter()
                            .find(|entry| entry.artifact.relative_path == relative_path)
                            .map(|entry| entry.artifact.clone())
                    })
                    .or_else(|| {
                        package.model_files.iter().find_map(|artifact| {
                            (artifact.relative_path == relative_path).then(|| artifact.clone())
                        })
                    })
            })
            .collect()
    }

    fn effective_single_device_only(&self) -> bool {
        self.create_params.single_device_only
    }

    fn runtime_safe_create_params(&self) -> CreateInstanceParams {
        let mut params = self.create_params.clone();
        params.model_kind = params.effective_model_kind();
        params.embedding = params.model_kind.supports_embeddings();
        params.reranking = params.model_kind.supports_rerank();
        if params.manual_device_allocations.is_empty() {
            params.manual_devices_csv = None;
            params.manual_tensor_split = None;
        } else {
            let detected_layers = self
                .selected_model_metadata()
                .as_ref()
                .and_then(|metadata| metadata.block_count);
            params.execution_group_id = "cluster:manual".to_string();
            params.manual_devices_csv = self.manual_devices_csv_from_allocations();
            params.manual_tensor_split = self.manual_tensor_split_from_allocations(detected_layers);
            params.n_gpu_layers = self.manual_selected_gpu_layers(detected_layers);
            params.rpc_servers = if params
                .manual_device_allocations
                .iter()
                .any(|device| device.rpc_device)
            {
                self.manual_rpc_servers_for_owner()
            } else {
                None
            };
        }
        params
    }

    fn manual_owner_control_addr(&self) -> Option<&str> {
        self.create_params
            .preferred_owner_control_addr
            .as_deref()
            .filter(|value| !value.trim().is_empty())
    }

    fn manual_remote_peer_control_addrs_for_owner(&self) -> BTreeSet<String> {
        let Some(owner_control_addr) = self.manual_owner_control_addr() else {
            return BTreeSet::new();
        };
        self.nodes
            .iter()
            .filter(|node| node.control_addr != owner_control_addr)
            .filter(|node| node.rpc_running)
            .map(|node| node.control_addr.clone())
            .collect()
    }

    fn manual_rpc_servers_for_owner(&self) -> Option<String> {
        let endpoints = self
            .manual_remote_peer_control_addrs_for_owner()
            .into_iter()
            .filter_map(|control_addr| {
                self.nodes
                    .iter()
                    .find(|node| node.control_addr == control_addr)
                    .and_then(rpc_endpoint_for_node)
            })
            .collect::<Vec<_>>();
        if endpoints.is_empty() {
            None
        } else {
            Some(endpoints.join(","))
        }
    }

    fn manual_devices_csv_from_allocations(&self) -> Option<String> {
        let value = self
            .create_params
            .manual_device_allocations
            .iter()
            .map(|device| device.bridge_device_index.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if value.trim().is_empty() {
            None
        } else {
            Some(value)
        }
    }

    fn manual_selected_gpu_layers(&self, detected_layer_count: Option<u32>) -> i32 {
        if self.create_params.manual_device_allocations.is_empty() {
            return self.create_params.n_gpu_layers;
        }
        let selected_layers = self
            .create_params
            .manual_device_allocations
            .iter()
            .map(|device| i64::from(device.layer_count))
            .sum::<i64>()
            .max(0);
        if self.create_params.manual_device_allocations.len() == 1 && selected_layers == 0 {
            return -1;
        }
        match detected_layer_count {
            Some(total_layers) => selected_layers.min(i64::from(total_layers)) as i32,
            None => selected_layers.min(i64::from(i32::MAX)) as i32,
        }
    }

    fn manual_tensor_split_from_allocations(
        &self,
        detected_layer_count: Option<u32>,
    ) -> Option<String> {
        if self.create_params.manual_device_allocations.len() < 2 {
            return None;
        }
        let total_layers = match detected_layer_count {
            Some(total_layers) => self
                .create_params
                .manual_device_allocations
                .iter()
                .map(|device| device.layer_count.min(total_layers))
                .sum::<u32>(),
            None => self
                .create_params
                .manual_device_allocations
                .iter()
                .map(|device| device.layer_count)
                .sum::<u32>(),
        };
        if total_layers == 0 {
            return None;
        }

        let weights = self
            .create_params
            .manual_device_allocations
            .iter()
            .map(|device| {
                let layers = detected_layer_count
                    .map(|total_layers| device.layer_count.min(total_layers))
                    .unwrap_or(device.layer_count);
                let raw = (layers as f64) / (total_layers as f64);
                let mut text = format!("{raw:.6}");
                while text.contains('.') && text.ends_with('0') {
                    text.pop();
                }
                if text.ends_with('.') {
                    text.pop();
                }
                if text.is_empty() {
                    "0".to_string()
                } else {
                    text
                }
            })
            .collect::<Vec<_>>();
        Some(weights.join(","))
    }

    fn selected_model_total_bytes(&self) -> u64 {
        let dependency_bytes = self
            .selected_model_dependency_artifacts()
            .into_iter()
            .map(|artifact| artifact.size_bytes)
            .sum::<u64>();
        if dependency_bytes > 0 {
            return dependency_bytes;
        }
        let model_path = self.create_params.model_path.trim();
        if model_path.is_empty() {
            return 0;
        }
        fs::metadata(model_path).map(|meta| meta.len()).unwrap_or(0)
    }

    fn selected_mmproj_total_bytes(&self) -> u64 {
        self.selected_mmproj_artifact()
            .map(|artifact| artifact.size_bytes)
            .filter(|value| *value > 0)
            .or_else(|| {
                self.create_params
                    .mmproj_path
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .and_then(|value| fs::metadata(value).ok())
                    .map(|meta| meta.len())
            })
            .unwrap_or(0)
    }

    fn selected_runtime_vram_estimate(&self) -> Option<RuntimeVramEstimate> {
        let model_bytes = self.selected_model_total_bytes();
        if model_bytes == 0 {
            return None;
        }
        let metadata = self.selected_model_metadata();
        Some(estimate_runtime_vram(
            model_bytes,
            self.selected_mmproj_total_bytes(),
            metadata.as_ref(),
            self.create_params.n_ctx,
            self.create_params.n_batch,
            self.create_params.n_parallel,
            self.manual_selected_gpu_layers(metadata.as_ref().and_then(|value| value.block_count)),
        ))
    }

    fn placement_candidate_request_params(&self) -> CreateInstanceParams {
        let mut params = self.runtime_safe_create_params();
        params.preferred_owner_control_addr = None;
        params.execution_group_id = "cluster:auto".to_string();
        params.rpc_servers = None;
        params.manual_device_allocations.clear();
        params.manual_devices_csv = None;
        params.manual_tensor_split = None;
        params
    }

    fn apply_runtime_estimate_to_placement_candidates(&mut self) {
        let Some(estimate) = self.selected_runtime_vram_estimate() else {
            return;
        };
        for plan in &mut self.placement_candidates {
            plan.estimated_required_bytes = estimate.required_gpu_bytes;
            if plan.reusable_instance_id.is_some() {
                plan.ready_now = true;
                continue;
            }
            let fits = plan.estimated_group_free_bytes >= estimate.required_gpu_bytes;
            plan.ready_now = fits;
            if fits {
                plan.requires_eviction = false;
            }
        }
        self.placement_candidates.sort_by(|lhs, rhs| {
            let lhs_reuse_rank = usize::from(lhs.reusable_instance_id.is_none());
            let rhs_reuse_rank = usize::from(rhs.reusable_instance_id.is_none());
            let lhs_fits = lhs.estimated_group_free_bytes >= lhs.estimated_required_bytes;
            let rhs_fits = rhs.estimated_group_free_bytes >= rhs.estimated_required_bytes;
            lhs_reuse_rank
                .cmp(&rhs_reuse_rank)
                .then(strategy_rank(lhs.strategy).cmp(&strategy_rank(rhs.strategy)))
                .then(rhs.ready_now.cmp(&lhs.ready_now))
                .then(rhs_fits.cmp(&lhs_fits))
                .then(
                    rhs.estimated_group_free_bytes
                        .saturating_sub(rhs.estimated_required_bytes)
                        .cmp(
                            &lhs.estimated_group_free_bytes
                                .saturating_sub(lhs.estimated_required_bytes),
                        ),
                )
                .then(lhs.device_count.cmp(&rhs.device_count))
                .then(lhs.owner_control_addr.cmp(&rhs.owner_control_addr))
        });
    }

    fn connect_local_host(&mut self) {
        let preferred = preferred_local_control_addr(self.local_control_addr_edit.trim());
        if preferred != self.local_control_addr_edit.trim() {
            self.local_control_addr_edit = preferred;
        }
        self.rebuild_host_from_inputs();
        if self.local_connect_in_progress {
            self.status = "Connecting local host...".to_string();
            return;
        }
        let runtime_dir = PathBuf::from(self.runtime_dir_edit.trim());
        let control_addr = self.local_control_addr_edit.trim().to_string();
        let tx = self.controller_worker_tx.clone();
        self.local_connect_in_progress = true;
        self.status = format!(
            "Connecting local host {} using runtime {}...",
            control_addr,
            runtime_dir.display()
        );
        std::thread::spawn(move || {
            let result = ensure_local_agent(&runtime_dir, &control_addr)
                .map(|_| {
                    format!(
                        "Connected to local host {} using runtime {}",
                        control_addr,
                        runtime_dir.display()
                    )
                })
                .map_err(|err| format!("local host start/connect failed: {err}"));
            let _ = tx.send(ControllerEvent::LocalAgentConnected(result));
        });
    }

    fn connect_local_host_and_start_pair_discovery(&mut self, seconds: u64) {
        self.local_connect_pending_pair_discovery_seconds = Some(seconds);
        self.connect_local_host();
    }

    fn apply_multi_node_rpc_setting(&mut self) {
        self.persist_controller_settings_if_changed();
        if self.multi_node_rpc_enabled {
            self.status =
                "Multi-node RPC setting saved. The worker will start only when a remote split actually needs this node.".to_string();
        } else {
            let restart_required = self.local_rpc_restart_required();
            self.status = if restart_required {
                "Multi-node RPC disabled for future launches on this node. Restart Engine to fully stop the embedded RPC worker.".to_string()
            } else {
                "Multi-node RPC disabled for future launches on this node.".to_string()
            };
        }
        self.enqueue_cluster_refresh();
        self.enqueue_telemetry_refresh();
        self.refresh_placement_candidates();
    }

    fn local_rpc_restart_required(&self) -> bool {
        if self.multi_node_rpc_enabled {
            return false;
        }
        let Ok(addr) = format!("127.0.0.1:{CLUSTER_AGENT_RPC_PORT}").parse::<SocketAddr>() else {
            return false;
        };
        TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok()
    }

    fn local_models_dir(&self) -> PathBuf {
        models_root_dir()
            .ok()
            .or_else(|| default_models_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn current_local_model_store_marker_modified(&self) -> Option<SystemTime> {
        let marker_path = model_store_change_marker_path(&self.local_models_dir());
        fs::metadata(marker_path).ok()?.modified().ok()
    }

    fn sync_local_model_store_marker_state(&mut self) {
        self.last_model_store_marker_modified = self.current_local_model_store_marker_modified();
    }

    fn poll_local_model_store_changes(&mut self) {
        if self.last_model_store_marker_poll.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_model_store_marker_poll = Instant::now();
        let current_marker = self.current_local_model_store_marker_modified();
        if self.last_model_store_marker_modified == current_marker {
            return;
        }
        let previous_marker = self.last_model_store_marker_modified;
        self.last_model_store_marker_modified = current_marker;
        if previous_marker.is_none() && current_marker.is_none() {
            return;
        }
        self.refresh_model_packages();
        self.refresh_available_model_packages();
        if self.host.local_client().is_some() {
            self.enqueue_managed_models_refresh();
        }
        self.refresh_placement_candidates();
    }

    fn refresh_managed_models(&mut self) {
        if self.host.local_client().is_none() {
            self.managed_models.clear();
            self.selected_managed_model_id = None;
            self.status = "local host is not connected".to_string();
            return;
        }
        self.enqueue_managed_models_refresh();
    }

    fn refresh_model_packages(&mut self) {
        let models_dir = self.local_models_dir();
        let marker_modified = self.current_local_model_store_marker_modified();
        let result = discover_model_packages(&models_dir).map_err(|err| {
            format!(
                "failed to scan model folders in '{}': {err:#}",
                models_dir.display()
            )
        });
        self.apply_local_model_packages_scan_result(&models_dir, result, marker_modified);
    }

    fn refresh_available_model_packages(&mut self) {
        if self.host.local_client().is_none() {
            self.rebuild_available_model_packages_local_only();
            return;
        }
        self.enqueue_available_model_packages_refresh();
    }

    fn refresh_all_ui(&mut self) {
        self.manual_refresh_in_progress = true;
        self.last_manual_refresh_completed_at = None;
        self.refresh_model_packages();
        self.refresh_available_model_packages();
        if self.host.local_client().is_some() {
            self.enqueue_managed_models_refresh();
            self.enqueue_cluster_refresh();
            self.enqueue_telemetry_refresh();
            self.enqueue_server_status_refresh();
        }
        self.refresh_placement_candidates();
        self.last_auto_refresh = Instant::now();
        self.status = "Refreshing controller state...".to_string();
        self.sync_manual_refresh_state();
    }

    fn background_refresh_in_progress(&self) -> bool {
        self.cluster_refresh_in_progress
            || self.telemetry_refresh_in_progress
            || self.placement_refresh_in_progress
            || self.managed_models_refresh_in_progress
            || self.available_model_packages_refresh_in_progress
            || self.server_status_refresh_in_progress
    }

    fn sync_manual_refresh_state(&mut self) {
        if self.manual_refresh_in_progress && !self.background_refresh_in_progress() {
            self.manual_refresh_in_progress = false;
            self.last_manual_refresh_completed_at = Some(Instant::now());
        }
    }

    fn enqueue_available_model_packages_refresh(&mut self) {
        let Some(local_client) = self.host.local_client() else {
            self.rebuild_available_model_packages_local_only();
            return;
        };
        if self.available_model_packages_refresh_in_progress {
            self.available_model_packages_refresh_pending = true;
            return;
        }
        self.available_model_packages_refresh_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        std::thread::spawn(move || {
            let result = local_client
                .list_cluster_model_packages()
                .map_err(|err| format!("failed to query available models: {err:#}"));
            let _ = tx.send(ControllerEvent::AvailableModelPackages(result));
        });
    }

    fn apply_available_model_packages_result(&mut self, packages: Vec<ClusterModelPackageInfo>) {
        if packages.is_empty() {
            self.rebuild_available_model_packages_local_only();
            return;
        }

        let mut available_packages = Vec::with_capacity(packages.len());
        let mut package_details = HashMap::with_capacity(packages.len());
        for info in packages {
            available_packages.push(info.package.clone());
            package_details.insert(info.package.folder_name.clone(), info);
        }
        self.available_model_packages = available_packages;
        self.available_model_package_details = package_details;
        self.sync_selected_model_package();
    }

    fn rebuild_available_model_packages_local_only(&mut self) {
        let local_control_addr = self.host.control_addr().to_string();
        let local_display_name = self
            .nodes
            .iter()
            .find(|node| node.control_addr == local_control_addr)
            .map(|node| node.node.display_name.clone())
            .or_else(|| {
                self.preview_node
                    .as_ref()
                    .map(|node| node.node.display_name.clone())
            })
            .unwrap_or_else(|| "This node".to_string());

        self.available_model_package_details = self
            .model_packages
            .iter()
            .cloned()
            .map(|package| {
                let package_path = package.path.display().to_string();
                let model_file_availability = package
                    .model_files
                    .iter()
                    .cloned()
                    .map(|artifact| ClusterModelArtifactInfo {
                        available_on: vec![ModelFileNodeAvailability {
                            control_addr: local_control_addr.clone(),
                            display_name: local_display_name.clone(),
                            package_path: package_path.clone(),
                            full_path: package_file_path(&package, &artifact.relative_path)
                                .display()
                                .to_string(),
                            managed_model_id: None,
                        }],
                        artifact,
                    })
                    .collect::<Vec<_>>();
                let mmproj_file_availability = package
                    .mmproj_files
                    .iter()
                    .cloned()
                    .map(|artifact| ClusterModelArtifactInfo {
                        available_on: vec![ModelFileNodeAvailability {
                            control_addr: local_control_addr.clone(),
                            display_name: local_display_name.clone(),
                            package_path: package_path.clone(),
                            full_path: package_file_path(&package, &artifact.relative_path)
                                .display()
                                .to_string(),
                            managed_model_id: None,
                        }],
                        artifact,
                    })
                    .collect::<Vec<_>>();
                let folder_name = package.folder_name.clone();
                (
                    folder_name,
                    ClusterModelPackageInfo {
                        package,
                        available_on: vec![ModelPackageNodeAvailability {
                            control_addr: local_control_addr.clone(),
                            display_name: local_display_name.clone(),
                            package_path,
                        }],
                        model_file_availability,
                        mmproj_file_availability,
                    },
                )
            })
            .collect();
        self.available_model_packages = self.model_packages.clone();
        self.sync_selected_model_package();
    }

    fn selected_model_package(&self) -> Option<&ModelPackage> {
        let selected = self.selected_model_package_folder.as_ref()?;
        self.available_model_packages
            .iter()
            .find(|package| &package.folder_name == selected)
    }

    fn selected_diarization_package(&self) -> Option<&ModelPackage> {
        let selected = self.selected_diarization_package_folder.as_ref()?;
        self.available_model_packages
            .iter()
            .find(|package| &package.folder_name == selected)
    }

    fn selected_local_model_package(&self) -> Option<&ModelPackage> {
        let selected = self.selected_model_package_folder.as_ref()?;
        self.model_packages
            .iter()
            .find(|package| &package.folder_name == selected)
    }

    fn selected_model_package_detail(&self) -> Option<&ClusterModelPackageInfo> {
        let selected = self.selected_model_package_folder.as_ref()?;
        self.available_model_package_details.get(selected)
    }

    fn selected_diarization_package_detail(&self) -> Option<&ClusterModelPackageInfo> {
        let selected = self.selected_diarization_package_folder.as_ref()?;
        self.available_model_package_details.get(selected)
    }

    fn selected_package_readme(&self) -> Option<String> {
        let package = self.selected_local_model_package()?;
        load_local_package_readme(package)
    }

    fn current_suggested_instance_name(&self) -> Option<String> {
        let package = self.selected_model_package()?;
        let model_file = self.selected_model_file_path.as_deref()?;
        Some(suggested_instance_name(
            package,
            &self.instance_model_kind,
            model_file,
        ))
    }

    fn sync_load_on_demand_grace_for_kind_change(&mut self, previous_kind: &str) {
        let previous_default = InstanceModelKind::from_dropdown_value(previous_kind)
            .default_load_on_demand_grace_seconds();
        if self.create_params.load_on_demand_grace_seconds == previous_default {
            self.create_params.load_on_demand_grace_seconds =
                InstanceModelKind::from_dropdown_value(&self.instance_model_kind)
                    .default_load_on_demand_grace_seconds();
        }
    }

    fn looks_like_generated_instance_name(&self, value: &str) -> bool {
        const MODEL_KINDS: [&str; 7] = [
            "text",
            "vision",
            "embeddings",
            "rerank",
            "whisper",
            "realtime-audio",
            "diarization",
        ];

        self.available_model_packages.iter().any(|package| {
            package.model_files.iter().any(|file| {
                MODEL_KINDS.iter().any(|kind| {
                    suggested_instance_name(package, kind, &file.relative_path) == value
                })
            })
        })
    }

    fn sync_instance_name_edit_state(&mut self) {
        let current_name = self.create_params.name.trim().to_string();
        let suggested = self.current_suggested_instance_name();
        self.last_suggested_instance_name = suggested.clone();
        self.instance_name_customized = if current_name.is_empty() {
            false
        } else {
            suggested.as_deref() != Some(current_name.as_str())
        };
    }

    fn sync_selected_model_package(&mut self) {
        let current_name = self.create_params.name.trim().to_string();
        let should_update_instance_name = current_name.is_empty()
            || !self.instance_name_customized
            || (self.last_suggested_instance_name.is_none()
                && self.looks_like_generated_instance_name(&current_name));

        if self.available_model_packages.is_empty() {
            self.selected_model_package_folder = None;
            self.selected_model_file_path = None;
            self.selected_mmproj_file_path = None;
            self.selected_diarization_package_folder = None;
            self.selected_diarization_file_path = None;
            self.last_suggested_instance_name = None;
            self.instance_name_customized = false;
            self.create_params.managed_model_id = None;
            self.create_params.model_path.clear();
            self.create_params.mmproj_path = None;
            self.create_params.diarization_model_path = None;
            self.allowed_control_addrs.clear();
            self.refresh_placement_candidates();
            return;
        }

        if self
            .selected_model_package_folder
            .as_ref()
            .is_none_or(|selected| {
                !self
                    .available_model_packages
                    .iter()
                    .any(|package| &package.folder_name == selected)
            })
        {
            self.selected_model_package_folder = self
                .available_model_packages
                .first()
                .map(|package| package.folder_name.clone());
        }

        let Some(package) = self.selected_model_package().cloned() else {
            return;
        };

        if self
            .selected_model_file_path
            .as_ref()
            .is_none_or(|selected| {
                !package
                    .model_files
                    .iter()
                    .any(|file| &file.relative_path == selected)
            })
        {
            self.selected_model_file_path = package
                .model_files
                .first()
                .map(|file| file.relative_path.clone());
        }

        if self.instance_model_kind == "vision" {
            if self
                .selected_mmproj_file_path
                .as_ref()
                .is_none_or(|selected| {
                    !package
                        .mmproj_files
                        .iter()
                        .any(|file| &file.relative_path == selected)
                })
            {
                self.selected_mmproj_file_path = package
                    .mmproj_files
                    .first()
                    .map(|file| file.relative_path.clone());
            }
        } else {
            self.selected_mmproj_file_path = None;
        }

        self.selected_diarization_package_folder = None;
        self.selected_diarization_file_path = None;

        self.apply_selected_package_to_create_params(should_update_instance_name);
    }

    fn apply_selected_package_to_create_params(&mut self, should_update_instance_name: bool) {
        let Some(package) = self.selected_model_package().cloned() else {
            return;
        };
        let Some(model_file) = self.selected_model_file_path.clone() else {
            self.create_params.model_path.clear();
            self.create_params.mmproj_path = None;
            self.create_params.diarization_model_path = None;
            self.allowed_control_addrs.clear();
            self.last_plan = None;
            self.refresh_placement_candidates();
            return;
        };

        let selected_owner = self
            .create_params
            .preferred_owner_control_addr
            .clone()
            .filter(|value| !value.trim().is_empty());
        let local_control_addr = self.host.control_addr().to_string();
        let model_availability = self
            .selected_model_package_detail()
            .and_then(|details| {
                details
                    .model_file_availability
                    .iter()
                    .find(|item| item.artifact.relative_path == model_file)
            })
            .cloned();
        let mut had_availability = model_availability.is_some();

        let choose_file_location =
            |availability: Option<&ClusterModelArtifactInfo>| -> Option<ModelFileNodeAvailability> {
                let availability = availability?;
                if let Some(owner_control_addr) = selected_owner.as_deref() {
                    if let Some(location) = availability
                        .available_on
                        .iter()
                        .find(|location| location.control_addr == owner_control_addr)
                    {
                        return Some(location.clone());
                    }
                }
                if let Some(location) = availability
                    .available_on
                    .iter()
                    .find(|location| location.control_addr == local_control_addr)
                {
                    return Some(location.clone());
                }
                availability.available_on.first().cloned()
            };

        let mut allowed_addrs = model_availability
            .as_ref()
            .map(|availability| {
                availability
                    .available_on
                    .iter()
                    .map(|node| node.control_addr.clone())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_else(|| {
                let mut values = BTreeSet::new();
                values.insert(local_control_addr.clone());
                values
            });

        self.create_params.managed_model_id =
            model_availability.as_ref().and_then(|availability| {
                selected_owner
                    .as_deref()
                    .and_then(|owner_control_addr| {
                        availability
                            .available_on
                            .iter()
                            .find(|location| location.control_addr == owner_control_addr)
                            .and_then(|location| location.managed_model_id.clone())
                    })
                    .or_else(|| {
                        availability
                            .available_on
                            .iter()
                            .find_map(|location| location.managed_model_id.clone())
                    })
            });
        self.create_params.model_path = choose_file_location(model_availability.as_ref())
            .map(|location| location.full_path)
            .unwrap_or_else(|| {
                package_file_path(&package, &model_file)
                    .display()
                    .to_string()
            });

        self.create_params.mmproj_path = if self.instance_model_kind == "vision" {
            let selected_mmproj = self.selected_mmproj_file_path.clone();
            selected_mmproj.and_then(|value| {
                let availability = self.selected_model_package_detail().and_then(|details| {
                    details
                        .mmproj_file_availability
                        .iter()
                        .find(|item| item.artifact.relative_path == value)
                });
                if let Some(availability) = availability {
                    had_availability = true;
                    let mmproj_nodes = availability
                        .available_on
                        .iter()
                        .map(|node| node.control_addr.clone())
                        .collect::<BTreeSet<_>>();
                    allowed_addrs = allowed_addrs
                        .intersection(&mmproj_nodes)
                        .cloned()
                        .collect::<BTreeSet<_>>();
                }
                choose_file_location(availability)
                    .map(|location| location.full_path)
                    .or_else(|| Some(package_file_path(&package, &value).display().to_string()))
            })
        } else {
            None
        };
        self.create_params.diarization_model_path = None;
        self.create_params.model_kind =
            InstanceModelKind::from_dropdown_value(&self.instance_model_kind);
        self.create_params.embedding = self.instance_model_kind == "embeddings";
        self.create_params.reranking = self.instance_model_kind == "rerank";
        self.create_params.single_device_only = matches!(
            self.instance_model_kind.as_str(),
            "whisper" | "realtime-audio" | "diarization"
        );
        let suggested = suggested_instance_name(&package, &self.instance_model_kind, &model_file);
        self.last_suggested_instance_name = Some(suggested.clone());
        if should_update_instance_name {
            self.create_params.name = suggested.clone();
            self.instance_name_customized = false;
        }
        let default_allowed = self.default_allowed_node_addrs();
        self.allowed_control_addrs = if default_allowed.is_empty() {
            if allowed_addrs.is_empty() && !had_availability {
                BTreeSet::new()
            } else {
                allowed_addrs
            }
        } else {
            default_allowed
        };
        self.create_params.preferred_owner_control_addr = self
            .create_params
            .preferred_owner_control_addr
            .clone()
            .filter(|control_addr| {
                self.allowed_control_addrs.is_empty()
                    || self.allowed_control_addrs.contains(control_addr)
            })
            .or_else(|| {
                if self.allowed_control_addrs.len() == 1 {
                    self.allowed_control_addrs.iter().next().cloned()
                } else {
                    None
                }
            });
        self.last_plan = None;
        self.sync_defaults_from_selected_node();
        self.refresh_placement_candidates();
    }

    fn selected_instance_model_entry(&self) -> Option<ManagedModelEntry> {
        let package = self.selected_model_package()?;
        let model_file = self.selected_model_file_path.as_ref()?;
        Some(ManagedModelEntry {
            id: sanitize_folder_name(&suggested_instance_name(
                package,
                &self.instance_model_kind,
                model_file,
            )),
            display_name: format!(
                "{} ({})",
                package.display_name,
                instance_model_type_label(&self.instance_model_kind)
            ),
            family: package.folder_name.clone(),
            task: instance_model_task(&self.instance_model_kind),
            single_device_only: matches!(
                self.instance_model_kind.as_str(),
                "whisper" | "realtime-audio" | "diarization"
            ),
            model_path: self.create_params.model_path.clone(),
            mmproj_path: self.create_params.mmproj_path.clone(),
            diarization_model_path: self.create_params.diarization_model_path.clone(),
            execution_group_id: self.create_params.execution_group_id.clone(),
            retention_mode: self.create_params.retention_mode,
            load_on_demand_grace_seconds: self.create_params.load_on_demand_grace_seconds,
            n_ctx: self.create_params.n_ctx,
            n_batch: self.create_params.n_batch,
            n_ubatch: self.create_params.n_ubatch,
            n_parallel: self.create_params.n_parallel,
            n_threads: self.create_params.n_threads,
            n_threads_batch: self.create_params.n_threads_batch,
            n_gpu_layers: self.create_params.n_gpu_layers,
            allowed_control_addrs: self.selected_allowed_node_addrs(),
        })
    }

    fn start_model_repo_load(&mut self, repo_id: String, recommended: Option<SupportedAudioRepo>) {
        self.model_store_error = None;
        self.model_store_repo_preview = None;
        self.model_store_progress = ModelDownloadProgress::default();
        self.model_store_busy = ModelStoreBusyState::LoadingRepo;
        let (tx, rx) = mpsc::channel();
        self.model_store_worker_rx = Some(rx);
        std::thread::spawn(move || {
            let result = fetch_repo_preview(&repo_id, recommended).map_err(|err| err.to_string());
            let _ = tx.send(ModelStoreEvent::RepoLoaded(result));
        });
    }

    fn start_current_repo_load(&mut self) {
        self.start_model_repo_load(self.model_store_repo_input.clone(), None);
    }

    fn load_supported_audio_repo(&mut self, repo: SupportedAudioRepo) {
        self.model_store_mode = ModelStoreMode::SupportedAudio;
        self.selected_supported_audio_repo = repo;
        self.model_store_repo_input = repo.repo_id().to_string();
        self.model_store_repo_folder_name = suggested_folder_name_for_repo(repo.repo_id());
        self.start_model_repo_load(repo.repo_id().to_string(), Some(repo));
    }

    fn start_repo_download(&mut self) {
        let Some(preview) = self.model_store_repo_preview.clone() else {
            self.model_store_error = Some("Load a repo first.".to_string());
            return;
        };
        let folder_name = if self.model_store_repo_folder_name.trim().is_empty() {
            suggested_folder_name_for_repo(&preview.repo_id)
        } else {
            sanitize_folder_name(&self.model_store_repo_folder_name)
        };
        self.model_store_progress = ModelDownloadProgress {
            total_files: preview.files.iter().filter(|file| file.selected).count(),
            total_bytes: preview
                .files
                .iter()
                .filter(|file| file.selected)
                .map(|file| file.size.unwrap_or(0))
                .sum(),
            ..ModelDownloadProgress::default()
        };
        self.model_store_busy = ModelStoreBusyState::Downloading;
        self.model_store_error = None;
        let (tx, rx) = mpsc::channel();
        self.model_store_worker_rx = Some(rx);
        std::thread::spawn(move || {
            let progress_tx = tx.clone();
            let result = download_repo_files(
                &preview.repo_id,
                &preview.revision,
                &folder_name,
                &preview.files,
                move |progress| {
                    let _ = progress_tx.send(ModelStoreEvent::DownloadProgress(progress));
                },
            )
            .map_err(|err| err.to_string());
            let _ = tx.send(ModelStoreEvent::DownloadFinished(result));
        });
    }

    fn pick_import_files(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Model files", &["gguf", "bin", "md", "json", "txt"])
            .set_title("Pick model files")
            .pick_files()
        {
            self.model_store_import_files = files;
            if self.model_store_import_name.trim().is_empty() {
                self.model_store_import_name = "model".to_string();
            }
        }
    }

    fn start_local_import(&mut self) {
        let folder_name = sanitize_folder_name(&self.model_store_import_name);
        let files = self.model_store_import_files.clone();
        if files.is_empty() {
            self.model_store_error = Some("Pick local model files first.".to_string());
            return;
        }
        self.model_store_busy = ModelStoreBusyState::Importing;
        self.model_store_error = None;
        let (tx, rx) = mpsc::channel();
        self.model_store_worker_rx = Some(rx);
        std::thread::spawn(move || {
            let result =
                import_local_model_files(&folder_name, &files).map_err(|err| err.to_string());
            let _ = tx.send(ModelStoreEvent::ImportFinished(result));
        });
    }

    fn start_selected_owner_artifact_transfer(&mut self) {
        let Some(owner_control_addr) = self.manual_owner_control_addr().map(str::to_string) else {
            self.status = "Pick a primary GPU first.".to_string();
            return;
        };
        let Some(package) = self.selected_model_package().cloned() else {
            self.status = "Pick a model folder first.".to_string();
            return;
        };
        let mut relative_paths = self.selected_primary_transfer_relative_paths(&package);
        relative_paths.sort();
        relative_paths.dedup();
        match self.build_model_transfer_plans(
            &package.folder_name,
            &relative_paths,
            &owner_control_addr,
        ) {
            Ok(plans) => self.start_model_transfer(plans),
            Err(err) => self.status = err,
        }
    }

    fn start_single_artifact_transfer_to_node(
        &mut self,
        folder_name: &str,
        relative_path: &str,
        dest_control_addr: &str,
    ) {
        match self.build_model_transfer_plans(
            folder_name,
            &[relative_path.to_string()],
            dest_control_addr,
        ) {
            Ok(plans) => self.start_model_transfer(plans),
            Err(err) => self.status = err,
        }
    }

    fn start_model_transfer(&mut self, plans: Vec<ModelTransferFilePlan>) {
        if self.model_transfer_in_progress {
            self.status = "A model transfer is already running.".to_string();
            return;
        }
        if plans.is_empty() {
            self.status =
                "All selected files are already present on the destination node.".to_string();
            return;
        }

        let total_files = plans.len();
        let total_bytes = plans.iter().map(|plan| plan.size_bytes).sum();
        self.model_transfer_in_progress = true;
        self.model_transfer_progress = Some(ModelTransferProgress {
            current_file: plans.first().map(|plan| plan.relative_path.clone()),
            source_display_name: plans
                .first()
                .map(|plan| plan.source_display_name.clone())
                .unwrap_or_default(),
            dest_display_name: plans
                .first()
                .map(|plan| plan.dest_display_name.clone())
                .unwrap_or_default(),
            completed_files: 0,
            total_files,
            transferred_bytes: 0,
            total_bytes,
            bytes_per_second: 0,
        });

        let (tx, rx) = mpsc::channel();
        self.model_transfer_worker_rx = Some(rx);
        std::thread::spawn(move || {
            let started = Instant::now();
            let mut transferred_bytes_total = 0u64;
            let mut completed_files = 0usize;
            let mut skipped_files = 0usize;
            let mut last_emit = Instant::now()
                .checked_sub(Duration::from_millis(250))
                .unwrap_or_else(Instant::now);
            let mut source_names = BTreeSet::new();
            let dest_display_name = plans
                .first()
                .map(|plan| plan.dest_display_name.clone())
                .unwrap_or_default();

            for plan in &plans {
                source_names.insert(plan.source_display_name.clone());
                let base_transferred = transferred_bytes_total;
                let progress_file = plan.relative_path.clone();
                let progress_source = plan.source_display_name.clone();
                let progress_dest = plan.dest_display_name.clone();
                let total_files = plans.len();
                let total_bytes = plans.iter().map(|item| item.size_bytes).sum();
                let progress_tx = tx.clone();
                let result = transfer_model_artifact_between_agents(
                    &plan.source_control_addr,
                    &plan.dest_control_addr,
                    &plan.folder_name,
                    &plan.relative_path,
                    |copied, _file_total| {
                        let now = Instant::now();
                        if now.duration_since(last_emit) < Duration::from_millis(125)
                            && copied < plan.size_bytes
                        {
                            return;
                        }
                        last_emit = now;
                        let transferred_bytes = base_transferred.saturating_add(copied);
                        let bytes_per_second = if started.elapsed().as_secs_f64() <= 0.0 {
                            0
                        } else {
                            (transferred_bytes as f64 / started.elapsed().as_secs_f64()).round()
                                as u64
                        };
                        let _ =
                            progress_tx.send(ModelTransferEvent::Progress(ModelTransferProgress {
                                current_file: Some(progress_file.clone()),
                                source_display_name: progress_source.clone(),
                                dest_display_name: progress_dest.clone(),
                                completed_files,
                                total_files,
                                transferred_bytes,
                                total_bytes,
                                bytes_per_second,
                            }));
                    },
                )
                .map_err(|err| err.to_string());

                match result {
                    Ok(outcome) => {
                        if outcome.skipped {
                            skipped_files += 1;
                        } else {
                            transferred_bytes_total =
                                transferred_bytes_total.saturating_add(outcome.size_bytes);
                        }
                        completed_files += 1;
                        let bytes_per_second = if started.elapsed().as_secs_f64() <= 0.0 {
                            0
                        } else {
                            (transferred_bytes_total as f64 / started.elapsed().as_secs_f64())
                                .round() as u64
                        };
                        let _ = tx.send(ModelTransferEvent::Progress(ModelTransferProgress {
                            current_file: Some(plan.relative_path.clone()),
                            source_display_name: plan.source_display_name.clone(),
                            dest_display_name: plan.dest_display_name.clone(),
                            completed_files,
                            total_files,
                            transferred_bytes: transferred_bytes_total,
                            total_bytes,
                            bytes_per_second,
                        }));
                    }
                    Err(err) => {
                        let _ = tx.send(ModelTransferEvent::Finished(Err(err)));
                        return;
                    }
                }
            }

            let _ = tx.send(ModelTransferEvent::Finished(Ok(ModelTransferSummary {
                source_display_names: source_names.into_iter().collect(),
                dest_display_name,
                completed_files,
                skipped_files,
            })));
        });
    }

    fn selected_primary_transfer_relative_paths(&self, package: &ModelPackage) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(model_relative_path) = self.selected_model_file_path.as_deref() {
            for relative in expanded_model_dependency_relative_paths(
                model_relative_path,
                package
                    .model_files
                    .iter()
                    .map(|artifact| artifact.relative_path.as_str()),
            ) {
                if !relative.trim().is_empty() {
                    out.push(relative);
                }
            }
        }
        if let Some(mmproj_relative_path) = self.selected_mmproj_file_path.as_deref() {
            out.push(mmproj_relative_path.to_string());
        }
        out
    }

    fn build_model_transfer_plans(
        &self,
        folder_name: &str,
        relative_paths: &[String],
        dest_control_addr: &str,
    ) -> Result<Vec<ModelTransferFilePlan>, String> {
        let details = self
            .available_model_package_details
            .get(folder_name)
            .ok_or_else(|| format!("No availability data is loaded for '{folder_name}'."))?;
        let dest_display_name = self.node_display_name_for_control_addr(dest_control_addr);
        let local_control_addr = self.host.control_addr().to_string();
        let mut plans = Vec::new();

        for relative_path in relative_paths {
            let availability = details
                .model_file_availability
                .iter()
                .find(|entry| entry.artifact.relative_path == *relative_path)
                .or_else(|| {
                    details
                        .mmproj_file_availability
                        .iter()
                        .find(|entry| entry.artifact.relative_path == *relative_path)
                })
                .ok_or_else(|| {
                    format!("Could not find '{relative_path}' in package '{folder_name}'.")
                })?;

            if availability
                .available_on
                .iter()
                .any(|node| node.control_addr == dest_control_addr)
            {
                continue;
            }

            let source = availability
                .available_on
                .iter()
                .find(|node| {
                    node.control_addr == local_control_addr
                        && node.control_addr != dest_control_addr
                })
                .or_else(|| {
                    availability
                        .available_on
                        .iter()
                        .find(|node| node.control_addr != dest_control_addr)
                })
                .ok_or_else(|| {
                    format!(
                        "No connected source node currently reports '{}'.",
                        relative_path
                    )
                })?;

            plans.push(ModelTransferFilePlan {
                folder_name: folder_name.to_string(),
                relative_path: relative_path.clone(),
                source_control_addr: source.control_addr.clone(),
                source_display_name: source.display_name.clone(),
                dest_control_addr: dest_control_addr.to_string(),
                dest_display_name: dest_display_name.clone(),
                size_bytes: availability.artifact.size_bytes,
            });
        }

        Ok(plans)
    }

    fn node_display_name_for_control_addr(&self, control_addr: &str) -> String {
        self.nodes
            .iter()
            .find(|node| node.control_addr == control_addr)
            .map(|node| node.node.display_name.clone())
            .or_else(|| {
                self.peers
                    .iter()
                    .find(|peer| peer.control_addr == control_addr)
                    .map(|peer| peer.display_name.clone())
            })
            .unwrap_or_else(|| control_addr.to_string())
    }

    fn drain_model_store_events(&mut self) {
        let Some(rx) = self.model_store_worker_rx.take() else {
            return;
        };
        let mut finished = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                ModelStoreEvent::RepoLoaded(result) => {
                    self.model_store_busy = ModelStoreBusyState::Idle;
                    finished = true;
                    match result {
                        Ok(preview) => {
                            if self.model_store_repo_folder_name.trim().is_empty() {
                                self.model_store_repo_folder_name =
                                    suggested_folder_name_for_repo(&preview.repo_id);
                            }
                            self.model_store_repo_input = preview.repo_id.clone();
                            self.model_store_repo_preview = Some(preview);
                            self.model_store_error = None;
                        }
                        Err(err) => {
                            self.model_store_repo_preview = None;
                            self.model_store_error = Some(err);
                        }
                    }
                }
                ModelStoreEvent::DownloadProgress(progress) => {
                    self.model_store_progress = progress.clone();
                    self.status = format_download_progress_status(&progress);
                }
                ModelStoreEvent::DownloadFinished(result) => {
                    self.model_store_busy = ModelStoreBusyState::Idle;
                    finished = true;
                    match result {
                        Ok(path) => {
                            self.model_store_error = None;
                            self.status = format!("Downloaded files into '{}'.", path.display());
                            self.refresh_model_packages();
                            if let Some(folder_name) =
                                path.file_name().and_then(|value| value.to_str())
                            {
                                self.selected_model_package_folder = Some(folder_name.to_string());
                                self.sync_selected_model_package();
                            }
                            self.refresh_available_model_packages();
                            self.refresh_managed_models();
                        }
                        Err(err) => {
                            self.model_store_error = Some(err);
                        }
                    }
                }
                ModelStoreEvent::ImportFinished(result) => {
                    self.model_store_busy = ModelStoreBusyState::Idle;
                    finished = true;
                    match result {
                        Ok(path) => {
                            self.model_store_error = None;
                            self.status = format!("Imported files into '{}'.", path.display());
                            self.model_store_import_files.clear();
                            self.refresh_model_packages();
                            if let Some(folder_name) =
                                path.file_name().and_then(|value| value.to_str())
                            {
                                self.selected_model_package_folder = Some(folder_name.to_string());
                                self.sync_selected_model_package();
                            }
                            self.refresh_available_model_packages();
                            self.refresh_managed_models();
                        }
                        Err(err) => {
                            self.model_store_error = Some(err);
                        }
                    }
                }
            }
        }
        if !finished {
            self.model_store_worker_rx = Some(rx);
        }
    }

    fn drain_model_transfer_events(&mut self) {
        let Some(rx) = self.model_transfer_worker_rx.take() else {
            return;
        };
        let mut finished = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                ModelTransferEvent::Progress(progress) => {
                    self.model_transfer_progress = Some(progress);
                }
                ModelTransferEvent::Finished(result) => {
                    self.model_transfer_in_progress = false;
                    self.model_transfer_progress = None;
                    finished = true;
                    match result {
                        Ok(summary) => {
                            let source_label = if summary.source_display_names.len() == 1 {
                                summary
                                    .source_display_names
                                    .first()
                                    .cloned()
                                    .unwrap_or_else(|| "source node".to_string())
                            } else {
                                format!("{} nodes", summary.source_display_names.len())
                            };
                            self.status = if summary.skipped_files > 0 {
                                format!(
                                    "Model transfer finished: {} file(s) copied, {} already present on {} from {}.",
                                    summary
                                        .completed_files
                                        .saturating_sub(summary.skipped_files),
                                    summary.skipped_files,
                                    summary.dest_display_name,
                                    source_label
                                )
                            } else {
                                format!(
                                    "Model transfer finished: {} file(s) copied to {} from {}.",
                                    summary.completed_files,
                                    summary.dest_display_name,
                                    source_label
                                )
                            };
                            self.refresh_model_packages();
                            self.refresh_available_model_packages();
                            self.refresh_managed_models();
                        }
                        Err(err) => {
                            self.status = format!("Model transfer failed: {err}");
                        }
                    }
                }
            }
        }
        if !finished {
            self.model_transfer_worker_rx = Some(rx);
        }
    }

    fn drain_controller_events(&mut self) {
        while let Ok(event) = self.controller_worker_rx.try_recv() {
            match event {
                ControllerEvent::StartupPrepared(result) => {
                    self.handle_startup_prepared(result);
                }
                ControllerEvent::LocalAgentConnected(result) => {
                    self.local_connect_in_progress = false;
                    self.handle_local_agent_connected(result);
                }
                ControllerEvent::ClusterRefresh(result) => {
                    self.cluster_refresh_in_progress = false;
                    match result {
                        Ok(payload) => self.apply_cluster_refresh_payload(payload),
                        Err(err) => self.status = err,
                    }
                    if self.cluster_refresh_pending {
                        self.cluster_refresh_pending = false;
                        self.enqueue_cluster_refresh();
                    }
                }
                ControllerEvent::TelemetryRefresh(result) => {
                    self.telemetry_refresh_in_progress = false;
                    match result {
                        Ok(telemetry) => {
                            self.telemetry = telemetry;
                            self.last_telemetry_refresh = Instant::now();
                        }
                        Err(err) => self.status = err,
                    }
                    if self.telemetry_refresh_pending {
                        self.telemetry_refresh_pending = false;
                        self.enqueue_telemetry_refresh();
                    }
                }
                ControllerEvent::PairingPoll(result) => {
                    self.pairing_poll_in_progress = false;
                    match result {
                        Ok(payload) => self.apply_pairing_poll_payload(payload),
                        Err(err) => {
                            if self.discovery_status.active
                                || self.pairing_modal_request_id.is_some()
                            {
                                self.status = err;
                            }
                        }
                    }
                }
                ControllerEvent::LinkBenchmarks(result) => {
                    self.link_benchmark_in_progress = false;
                    match result {
                        Ok(message) => self.status = message,
                        Err(err) => self.status = err,
                    }
                    self.enqueue_telemetry_refresh();
                }
                ControllerEvent::PlacementCandidates(result) => {
                    self.placement_refresh_in_progress = false;
                    match result {
                        Ok(plans) => {
                            self.placement_candidates = plans;
                            self.apply_runtime_estimate_to_placement_candidates();
                        }
                        Err(err) => {
                            self.placement_candidates.clear();
                            self.status = err;
                        }
                    }
                    if self.placement_refresh_pending {
                        self.placement_refresh_pending = false;
                        self.refresh_placement_candidates();
                    }
                }
                ControllerEvent::ManagedModels(result) => {
                    self.managed_models_refresh_in_progress = false;
                    match result {
                        Ok(models) => self.apply_managed_models_result(models),
                        Err(err) => {
                            self.managed_models.clear();
                            self.selected_managed_model_id = None;
                            self.status = err;
                        }
                    }
                    if self.managed_models_refresh_pending {
                        self.managed_models_refresh_pending = false;
                        self.enqueue_managed_models_refresh();
                    }
                }
                ControllerEvent::AvailableModelPackages(result) => {
                    self.available_model_packages_refresh_in_progress = false;
                    match result {
                        Ok(packages) => self.apply_available_model_packages_result(packages),
                        Err(err) => {
                            self.rebuild_available_model_packages_local_only();
                            self.status = err;
                        }
                    }
                    if self.available_model_packages_refresh_pending {
                        self.available_model_packages_refresh_pending = false;
                        self.enqueue_available_model_packages_refresh();
                    }
                }
                ControllerEvent::ServerStatus(result) => {
                    self.server_status_refresh_in_progress = false;
                    match result {
                        Ok(status) => {
                            self.server_enabled = status.enabled;
                            self.server_bind_addr_edit = status.bind_addr.clone();
                            self.server_allow_cors = status.allow_cors;
                            self.server_allowed_origins_edit = status.allowed_origins.join("\n");
                            self.server_allowed_client_ips_edit =
                                status.allowed_client_ips.join("\n");
                            self.server_status = Some(status);
                        }
                        Err(err) => {
                            self.server_status = None;
                            self.status = err;
                        }
                    }
                    if self.server_status_refresh_pending {
                        self.server_status_refresh_pending = false;
                        self.enqueue_server_status_refresh();
                    }
                }
            }
        }
    }

    fn handle_local_agent_connected(&mut self, result: Result<String, String>) {
        match result {
            Ok(status) => {
                self.host
                    .set_local_client(AgentClient::new(self.host.control_addr().to_string()));
                self.status = status;
                self.refresh_available_model_packages();
                self.enqueue_cluster_refresh();
                self.enqueue_telemetry_refresh();
                self.enqueue_managed_models_refresh();
                self.enqueue_server_status_refresh();
                if let Some(seconds) = self.local_connect_pending_pair_discovery_seconds.take() {
                    self.start_pair_discovery_with_seconds(seconds);
                }
            }
            Err(err) => {
                self.host.clear_local_connection();
                self.refresh_runtime_state();
                self.server_status = None;
                self.local_connect_pending_pair_discovery_seconds = None;
                self.status = err;
            }
        }
    }

    fn enqueue_cluster_refresh(&mut self) {
        let Some(local_client) = self.host.local_client() else {
            return;
        };
        if self.cluster_refresh_in_progress {
            self.cluster_refresh_pending = true;
            return;
        }
        self.cluster_refresh_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        let local_control_addr = self.host.control_addr().to_string();
        let selected_control_addr = self.selected_control_addr.clone();
        let selected_rpc_peer_addrs = self.selected_rpc_peer_addrs.clone();
        std::thread::spawn(move || {
            let result = query_cluster_refresh(
                local_client,
                local_control_addr,
                selected_control_addr,
                selected_rpc_peer_addrs,
            );
            let _ = tx.send(ControllerEvent::ClusterRefresh(result));
        });
    }

    fn enqueue_telemetry_refresh(&mut self) {
        let Some(local_client) = self.host.local_client() else {
            return;
        };
        if self.telemetry_refresh_in_progress {
            self.telemetry_refresh_pending = true;
            return;
        }
        self.telemetry_refresh_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        let local_control_addr = self.host.control_addr().to_string();
        std::thread::spawn(move || {
            let result = query_cluster_telemetry(local_client, local_control_addr);
            let _ = tx.send(ControllerEvent::TelemetryRefresh(result));
        });
    }

    fn enqueue_pairing_poll(&mut self) {
        let Some(local_client) = self.host.local_client() else {
            return;
        };
        if self.pairing_poll_in_progress {
            return;
        }
        self.pairing_poll_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        let previous_trusted = self
            .peers
            .iter()
            .filter(|peer| peer.trusted)
            .map(|peer| peer.control_addr.clone())
            .collect::<BTreeSet<_>>();
        std::thread::spawn(move || {
            let result = query_pairing_poll(local_client, previous_trusted);
            let _ = tx.send(ControllerEvent::PairingPoll(result));
        });
    }

    fn enqueue_managed_models_refresh(&mut self) {
        let Some(local_client) = self.host.local_client() else {
            return;
        };
        if self.managed_models_refresh_in_progress {
            self.managed_models_refresh_pending = true;
            return;
        }
        self.managed_models_refresh_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        std::thread::spawn(move || {
            let result = local_client
                .list_cluster_managed_models()
                .map_err(|err| format!("failed to query cluster model library: {err:#}"));
            let _ = tx.send(ControllerEvent::ManagedModels(result));
        });
    }

    fn enqueue_server_status_refresh(&mut self) {
        let Some(local_client) = self.host.local_client() else {
            return;
        };
        if self.server_status_refresh_in_progress {
            self.server_status_refresh_pending = true;
            return;
        }
        self.server_status_refresh_in_progress = true;
        let tx = self.controller_worker_tx.clone();
        std::thread::spawn(move || {
            let result = local_client
                .get_public_api_status()
                .map_err(|err| format!("failed to query server status: {err:#}"));
            let _ = tx.send(ControllerEvent::ServerStatus(result));
        });
    }

    fn apply_cluster_refresh_payload(&mut self, payload: ClusterRefreshPayload) {
        self.peers = payload.peers;
        self.pairing_requests = payload.pairing_requests;
        self.discovery_status = payload.discovery_status;
        self.nodes = payload.nodes;
        self.telemetry = payload.telemetry;
        self.last_telemetry_refresh = Instant::now();
        self.selected_control_addr = payload.selected_control_addr;
        self.preview_node = payload.preview_node;
        self.handle_pairing_request_updates();

        if self.selected_instance_id.is_some() && self.selected_instance().is_none() {
            self.selected_instance_id = None;
        }
        self.sync_defaults_from_selected_node();
        if self.allowed_control_addrs.is_empty()
            || !self
                .allowed_control_addrs
                .iter()
                .any(|addr| self.nodes.iter().any(|node| &node.control_addr == addr))
        {
            self.allowed_control_addrs = self.default_allowed_node_addrs();
        }
        if !self.create_params.model_path.trim().is_empty() {
            self.refresh_placement_candidates();
        }
        if !payload.warnings.is_empty() {
            self.status = format!(
                "State refreshed with warnings: {}",
                payload.warnings.join(" | ")
            );
        }
    }

    fn apply_pairing_poll_payload(&mut self, payload: PairingPollPayload) {
        self.peers = payload.peers;
        self.pairing_requests = payload.pairing_requests;
        self.discovery_status = payload.discovery_status;
        self.handle_pairing_request_updates();
        if payload.topology_changed {
            self.enqueue_cluster_refresh();
            self.enqueue_telemetry_refresh();
            self.refresh_available_model_packages();
        }
    }

    fn apply_managed_models_result(&mut self, models: Vec<ManagedModelEntry>) {
        let previous = self.selected_managed_model_id.clone();
        self.managed_models = models;
        if self.managed_models.is_empty() {
            self.selected_managed_model_id = None;
            return;
        }
        let next = previous
            .filter(|selected| {
                self.managed_models
                    .iter()
                    .any(|model| &model.id == selected)
            })
            .or_else(|| self.managed_models.first().map(|model| model.id.clone()));
        if let Some(next) = next {
            let changed = self.selected_managed_model_id.as_deref() != Some(next.as_str());
            self.selected_managed_model_id = Some(next.clone());
            if changed || self.create_params.model_path.trim().is_empty() {
                self.select_managed_model(next);
            }
        }
    }

    fn refresh_runtime_state(&mut self) {
        let runtime_dir = PathBuf::from(self.runtime_dir_edit.trim());
        let runtime_install_backends = runtime_installer::available_runtime_backends();
        let runtime_missing = runtime_installer::runtime_missing_messages(&runtime_dir);
        let runtime_install_recommendation = runtime_installer::runtime_install_recommendation(
            &runtime_dir,
            &runtime_install_backends,
        );
        self.apply_runtime_state_snapshot(
            runtime_missing,
            runtime_install_backends,
            runtime_install_recommendation,
        );
    }

    fn refresh_server_status(&mut self) {
        if self.host.local_client().is_none() {
            self.server_status = None;
            self.status = "local host is not connected".to_string();
            return;
        }
        self.enqueue_server_status_refresh();
    }

    fn try_refresh_server_status(&mut self) -> Result<(), String> {
        let result = self
            .host
            .local_client()
            .ok_or_else(|| "local host is not connected".to_string())
            .and_then(|client| {
                client
                    .get_public_api_status()
                    .map_err(|err| format!("failed to query server status: {err:#}"))
            });
        match result {
            Ok(status) => {
                self.server_enabled = status.enabled;
                self.server_bind_addr_edit = status.bind_addr.clone();
                self.server_allow_cors = status.allow_cors;
                self.server_allowed_origins_edit = status.allowed_origins.join("\n");
                self.server_allowed_client_ips_edit = status.allowed_client_ips.join("\n");
                self.server_status = Some(status);
                Ok(())
            }
            Err(err) => {
                self.server_status = None;
                Err(err)
            }
        }
    }

    fn apply_server_config(&mut self) {
        let bind_addr = self.server_bind_addr_edit.trim().to_string();
        if bind_addr.is_empty() {
            self.status = "server bind address cannot be empty".to_string();
            return;
        }
        let update = PublicApiConfigUpdate {
            enabled: self.server_enabled,
            bind_addr,
            allow_cors: self.server_allow_cors,
            allowed_origins: parse_multiline_list(&self.server_allowed_origins_edit),
            allowed_client_ips: parse_multiline_list(&self.server_allowed_client_ips_edit),
            api_key: {
                let value = self.server_api_key_edit.trim().to_string();
                if value.is_empty() {
                    None
                } else {
                    Some(value)
                }
            },
            clear_api_key: false,
        };
        let result = self
            .host
            .local_client()
            .ok_or_else(|| "local host is not connected".to_string())
            .and_then(|client| {
                client
                    .update_public_api_config(update)
                    .map_err(|err| format!("failed to update server settings: {err:#}"))
            });
        match result {
            Ok(status) => {
                self.server_enabled = status.enabled;
                self.server_bind_addr_edit = status.bind_addr.clone();
                self.server_allow_cors = status.allow_cors;
                self.server_allowed_origins_edit = status.allowed_origins.join("\n");
                self.server_allowed_client_ips_edit = status.allowed_client_ips.join("\n");
                self.server_status = Some(status);
                self.server_api_key_edit.clear();
                self.server_generated_api_key = None;
                self.status = "Server settings applied.".to_string();
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err,
        }
    }

    fn clear_server_api_key(&mut self) {
        let update = PublicApiConfigUpdate {
            enabled: self.server_enabled,
            bind_addr: self.server_bind_addr_edit.trim().to_string(),
            allow_cors: self.server_allow_cors,
            allowed_origins: parse_multiline_list(&self.server_allowed_origins_edit),
            allowed_client_ips: parse_multiline_list(&self.server_allowed_client_ips_edit),
            api_key: None,
            clear_api_key: true,
        };
        let result = self
            .host
            .local_client()
            .ok_or_else(|| "local host is not connected".to_string())
            .and_then(|client| {
                client
                    .update_public_api_config(update)
                    .map_err(|err| format!("failed to clear API key: {err:#}"))
            });
        match result {
            Ok(status) => {
                self.server_enabled = status.enabled;
                self.server_bind_addr_edit = status.bind_addr.clone();
                self.server_allow_cors = status.allow_cors;
                self.server_allowed_origins_edit = status.allowed_origins.join("\n");
                self.server_allowed_client_ips_edit = status.allowed_client_ips.join("\n");
                self.server_status = Some(status);
                self.server_api_key_edit.clear();
                self.server_generated_api_key = None;
                self.status = "Server API key removed.".to_string();
            }
            Err(err) => self.status = err,
        }
    }

    fn generate_server_api_key(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let key = format!("engine_{}", now);
        self.server_api_key_edit = key.clone();
        self.server_generated_api_key = Some(key);
        self.status =
            "Generated a new API key. Copy it before applying if you need the full value."
                .to_string();
    }

    fn selected_managed_model(&self) -> Option<&ManagedModelEntry> {
        let selected = self.selected_managed_model_id.as_ref()?;
        self.managed_models
            .iter()
            .find(|model| &model.id == selected)
    }

    fn select_managed_model(&mut self, model_id: String) {
        let Some(model) = self
            .managed_models
            .iter()
            .find(|item| item.id == model_id)
            .cloned()
        else {
            return;
        };
        self.selected_managed_model_id = Some(model.id.clone());
        self.create_params = model.create_instance_params();
        self.create_params.name = model.id.clone();
        self.instance_model_kind = self
            .create_params
            .model_kind
            .as_dropdown_value()
            .to_string();
        self.last_suggested_instance_name = Some(model.id.clone());
        self.instance_name_customized = false;
        self.create_params.allow_cpu = self.show_cpu_devices;
        self.create_params.allow_integrated_gpu = self.show_integrated_gpus;
        self.allowed_control_addrs = model
            .allowed_control_addrs
            .clone()
            .map(|values| values.into_iter().collect())
            .unwrap_or_else(|| self.default_allowed_node_addrs());
        self.sync_defaults_from_selected_node();
        self.refresh_placement_candidates();
    }

    fn default_allowed_node_addrs(&self) -> BTreeSet<String> {
        self.nodes
            .iter()
            .filter(|node| self.node_has_default_gpu_target(node))
            .map(|node| node.control_addr.clone())
            .collect()
    }

    fn node_has_default_gpu_target(&self, node: &NodeSnapshot) -> bool {
        let devices = self
            .telemetry_for_control_addr(&node.control_addr)
            .map(|snapshot| snapshot.devices.as_slice())
            .unwrap_or(node.devices.as_slice());
        devices
            .iter()
            .any(|device| self.device_allowed_in_default_cluster_view(node, device))
    }

    fn device_allowed_in_default_cluster_view(
        &self,
        node: &NodeSnapshot,
        device: &cluster_api::DeviceInfo,
    ) -> bool {
        if device_is_cpu_for_defaults(device) {
            return self.show_cpu_devices;
        }
        if device_is_rpc_for_defaults(device) {
            return false;
        }
        if !self.show_integrated_gpus && device_is_integrated_for_defaults(node, device) {
            return false;
        }
        true
    }

    fn selected_allowed_node_addrs(&self) -> Option<Vec<String>> {
        let has_package_scoped_selection = self.instance_creation_open
            && (self.selected_model_package_detail().is_some()
                || self.selected_diarization_package_detail().is_some());
        let restrict_to_allowed =
            has_package_scoped_selection || !self.allowed_control_addrs.is_empty();
        let allowed = if self.allowed_control_addrs.is_empty() && !has_package_scoped_selection {
            self.default_allowed_node_addrs()
        } else {
            self.allowed_control_addrs.clone()
        };
        let mut values = self
            .nodes
            .iter()
            .map(|node| node.control_addr.clone())
            .filter(|control_addr| !restrict_to_allowed || allowed.contains(control_addr))
            .collect::<Vec<_>>();
        values.sort();
        values.dedup();
        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }

    fn refresh_placement_candidates(&mut self) {
        if self.create_params.model_path.trim().is_empty() {
            self.placement_candidates.clear();
            return;
        }
        let Some(local_client) = self.host.local_client() else {
            self.placement_candidates.clear();
            return;
        };
        if self.placement_refresh_in_progress {
            self.placement_refresh_pending = true;
            return;
        }
        let allowed = self.selected_allowed_node_addrs();
        let preview_params = self.placement_candidate_request_params();
        let tx = self.controller_worker_tx.clone();
        self.placement_refresh_in_progress = true;
        std::thread::spawn(move || {
            let result = local_client
                .list_placement_candidates(preview_params, allowed)
                .map_err(|err| err.to_string());
            let _ = tx.send(ControllerEvent::PlacementCandidates(result));
        });
    }

    fn start_pair_discovery(&mut self) {
        self.start_pair_discovery_with_seconds(180);
    }

    fn start_pair_discovery_with_seconds(&mut self, seconds: u64) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.start_discovery(DiscoveryMode::Pairing, seconds) {
            Ok(status) => {
                self.discovery_status = status;
                self.status = format!("Looking for nodes for {seconds} seconds.");
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn stop_pair_discovery(&mut self) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.start_discovery(DiscoveryMode::Off, 0) {
            Ok(status) => {
                self.discovery_status = status;
                self.status = "Pairing mode stopped.".to_string();
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn request_pairing(&mut self, control_addr: &str) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.request_pairing(control_addr.to_string()) {
            Ok(()) => {
                self.status =
                    format!("Pair request sent to {control_addr}. Approve it on the other node.");
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn handle_pairing_request_updates(&mut self) {
        let current_ids = self
            .pairing_requests
            .iter()
            .map(|request| request.request_id.clone())
            .collect::<BTreeSet<_>>();
        let new_request = self
            .pairing_requests
            .iter()
            .find(|request| !self.seen_pairing_request_ids.contains(&request.request_id))
            .cloned();
        if let Some(request) = new_request {
            self.pairing_modal_request_id = Some(request.request_id.clone());
            self.pairing_request_attention_pending = true;
            self.status = format!(
                "{} wants to pair with this node.",
                request.requester_display_name
            );
        }
        if current_ids.is_empty() {
            self.pairing_modal_request_id = None;
        } else if self
            .pairing_modal_request_id
            .as_ref()
            .is_none_or(|id| !current_ids.contains(id))
        {
            self.pairing_modal_request_id = self
                .pairing_requests
                .first()
                .map(|request| request.request_id.clone());
        }
        self.seen_pairing_request_ids = current_ids;
    }

    fn poll_pairing_and_discovery_state(&mut self) -> Result<bool, String> {
        let local_client = self
            .host
            .local_client()
            .ok_or_else(|| "local host is not connected".to_string())?;
        let previous_trusted = self
            .peers
            .iter()
            .filter(|peer| peer.trusted)
            .map(|peer| peer.control_addr.clone())
            .collect::<BTreeSet<_>>();
        let peers = local_client.list_peers().map_err(|e| e.to_string())?;
        let pairing_requests = local_client
            .list_pairing_requests()
            .map_err(|e| e.to_string())?;
        let discovery_status = local_client
            .get_discovery_status()
            .map_err(|e| e.to_string())?;
        let next_trusted = peers
            .iter()
            .filter(|peer| peer.trusted)
            .map(|peer| peer.control_addr.clone())
            .collect::<BTreeSet<_>>();
        self.peers = peers;
        self.pairing_requests = pairing_requests;
        self.discovery_status = discovery_status;
        self.handle_pairing_request_updates();
        Ok(previous_trusted != next_trusted)
    }

    fn apply_pairing_attention(&mut self, ctx: &egui::Context) {
        if !self.pairing_request_attention_pending {
            return;
        }
        if self.window_hidden_to_tray {
            self.show_window_from_tray(ctx);
        } else {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
                egui::UserAttentionType::Informational,
            ));
        }
        self.pairing_request_attention_pending = false;
    }

    fn accept_pairing_request(&mut self, request_id: &str) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.accept_pairing_request(request_id.to_string()) {
            Ok(()) => {
                self.status = "Node pairing completed.".to_string();
                let _ = self.refresh_cluster();
                let _ = self.refresh_telemetry();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn decline_pairing_request(&mut self, request_id: &str) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.decline_pairing_request(request_id.to_string()) {
            Ok(()) => {
                self.status = "Pair request dismissed.".to_string();
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn run_cluster_link_benchmarks(&mut self, full: bool) {
        if self.link_benchmark_in_progress {
            self.status = "Link benchmark already running.".to_string();
            return;
        }
        let mut warnings = Vec::new();
        let mut addrs = BTreeSet::new();
        for node in &self.nodes {
            let addr = node
                .advertised_control_addr
                .clone()
                .unwrap_or_else(|| node.control_addr.clone());
            if !addr.trim().is_empty() {
                addrs.insert(addr);
            }
        }
        if addrs.is_empty() {
            return;
        }
        self.link_benchmark_in_progress = true;
        self.status = if full {
            "Running link benchmark in the background...".to_string()
        } else {
            "Running startup link probe in the background...".to_string()
        };
        let tx = self.controller_worker_tx.clone();
        std::thread::spawn(move || {
            for addr in addrs {
                let client = AgentClient::new(addr.clone());
                if let Err(err) = client.run_link_benchmarks(full) {
                    warnings.push(format!("{addr}: {err}"));
                }
            }
            let message = if warnings.is_empty() {
                if full {
                    "Link benchmark refreshed.".to_string()
                } else {
                    "Startup link benchmark completed.".to_string()
                }
            } else {
                format!(
                    "{} with warnings: {}",
                    if full {
                        "Link benchmark refreshed"
                    } else {
                        "Startup link benchmark completed"
                    },
                    warnings.join(" | ")
                )
            };
            let _ = tx.send(ControllerEvent::LinkBenchmarks(Ok(message)));
        });
    }

    fn start_runtime_install(&mut self) {
        if self.runtime_install_in_progress {
            return;
        }
        let runtime_dir = PathBuf::from(self.runtime_dir_edit.trim());
        let preferred_backend = self
            .runtime_install_backends
            .get(self.selected_runtime_install_backend)
            .cloned();
        let (tx, rx) = mpsc::channel();
        self.runtime_install_in_progress = true;
        self.runtime_install_status = Some("Preparing runtime install...".to_string());
        self.runtime_install_rx = Some(rx);
        std::thread::spawn(move || {
            let status_tx = tx.clone();
            let result = runtime_installer::install_or_repair_runtime_with_backend(
                &runtime_dir,
                preferred_backend.as_deref(),
                move |status| {
                    let _ = status_tx.send(RuntimeInstallEvent::Status(status));
                },
            )
            .map_err(|err: anyhow::Error| err.to_string());
            let _ = tx.send(RuntimeInstallEvent::Finished(result));
        });
    }

    fn poll_runtime_install_events(&mut self) {
        let Some(rx) = &self.runtime_install_rx else {
            return;
        };
        let mut finished = None;
        while let Ok(event) = rx.try_recv() {
            match event {
                RuntimeInstallEvent::Status(status) => {
                    self.runtime_install_status = Some(status);
                }
                RuntimeInstallEvent::Finished(result) => finished = Some(result),
            }
        }
        if let Some(result) = finished {
            self.runtime_install_in_progress = false;
            self.runtime_install_rx = None;
            match result {
                Ok(path) => {
                    self.runtime_dir_edit = path.display().to_string();
                    self.status = format!("Runtime installed to '{}'.", path.display());
                    self.refresh_runtime_state();
                    self.connect_local_host();
                }
                Err(err) => {
                    self.status = format!("Runtime install failed: {err}");
                    self.runtime_install_status = Some(err);
                    self.refresh_runtime_state();
                }
            }
        }
    }

    fn refresh_cluster(&mut self) -> Result<(), String> {
        if self.host.local_client().is_none() {
            return Err("local host is not connected".to_string());
        }
        self.enqueue_cluster_refresh();
        Ok(())
    }

    fn refresh_telemetry(&mut self) -> Result<(), String> {
        if self.host.local_client().is_none() {
            return Err("local host is not connected".to_string());
        }
        self.enqueue_telemetry_refresh();
        Ok(())
    }

    fn refresh_selected_preview(&mut self) -> Result<(), String> {
        let Some(client) = self.selected_client() else {
            self.preview_node = None;
            return Ok(());
        };
        let endpoints = self.selected_rpc_endpoints();
        let snapshot = client
            .get_snapshot_with_rpc(if endpoints.is_empty() {
                None
            } else {
                Some(endpoints.join(","))
            })
            .map_err(|e| e.to_string())?;
        self.preview_node = Some(snapshot);
        Ok(())
    }

    fn sync_defaults_from_selected_node(&mut self) {
        self.create_params.allow_cpu = self.show_cpu_devices;
        self.create_params.allow_integrated_gpu = self.show_integrated_gpus;

        if self.create_params.preferred_owner_control_addr.is_some()
            || self
                .create_params
                .rpc_servers
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            || (!self.create_params.execution_group_id.trim().is_empty()
                && self.create_params.execution_group_id != "cluster:auto")
        {
            return;
        }

        let Some(node) = self.selected_preview_node() else {
            return;
        };

        if self.create_params.execution_group_id.is_empty()
            || !node
                .execution_groups
                .iter()
                .any(|group| group.id == self.create_params.execution_group_id)
        {
            if let Some(auto) = node
                .execution_groups
                .iter()
                .find(|group| group.id == "cluster:auto")
            {
                self.create_params.execution_group_id = auto.id.clone();
            } else if let Some(split) = node
                .execution_groups
                .iter()
                .find(|group| group.uses_local_split)
            {
                self.create_params.execution_group_id = split.id.clone();
            } else if let Some(group) = node.execution_groups.first() {
                self.create_params.execution_group_id = group.id.clone();
            }
        }
    }

    fn selected_node(&self) -> Option<&NodeSnapshot> {
        let selected = self.selected_control_addr.as_ref()?;
        self.nodes
            .iter()
            .find(|node| &node.control_addr == selected)
    }

    fn selected_preview_node(&self) -> Option<&NodeSnapshot> {
        self.preview_node.as_ref().or_else(|| self.selected_node())
    }

    fn telemetry_for_control_addr(&self, control_addr: &str) -> Option<&TelemetrySnapshot> {
        self.telemetry.iter().find(|snapshot| {
            snapshot.control_addr == control_addr
                || snapshot.advertised_control_addr.as_deref() == Some(control_addr)
        })
    }

    fn selected_telemetry(&self) -> Option<&TelemetrySnapshot> {
        let selected = self.selected_control_addr.as_deref()?;
        self.telemetry_for_control_addr(selected)
    }

    fn selected_instance(&self) -> Option<&cluster_api::InstanceInfo> {
        let instance_id = self.selected_instance_id?;
        self.selected_node()?
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)
    }

    fn selected_is_local(&self) -> bool {
        self.selected_control_addr.as_deref() == Some(self.host.control_addr())
    }

    fn selected_client(&self) -> Option<AgentClient> {
        let selected = self.selected_control_addr.as_ref()?;
        if selected == self.host.control_addr() {
            return self.host.local_client();
        }
        Some(AgentClient::new(selected.clone()))
    }

    fn selected_rpc_endpoints(&self) -> Vec<String> {
        let selected_host = match self.selected_control_addr.as_deref() {
            Some(value) => value,
            None => return Vec::new(),
        };

        self.nodes
            .iter()
            .filter(|node| node.control_addr != selected_host)
            .filter(|node| self.selected_rpc_peer_addrs.contains(&node.control_addr))
            .filter_map(rpc_endpoint_for_node)
            .collect()
    }

    fn scheduler_allowed_nodes(&self) -> Option<Vec<String>> {
        let mut addrs: Vec<String> = self
            .nodes
            .iter()
            .filter_map(|node| {
                if node.control_addr == self.host.control_addr() {
                    Some(node.control_addr.clone())
                } else if node.rpc_running {
                    Some(node.control_addr.clone())
                } else {
                    None
                }
            })
            .collect();
        addrs.sort();
        addrs.dedup();
        if addrs.is_empty() {
            None
        } else {
            Some(addrs)
        }
    }

    fn plan_instance_cluster(&mut self) {
        let allowed = self.selected_allowed_node_addrs();
        let params = self.runtime_safe_create_params();
        match self.host.plan_instance(&params, allowed) {
            Ok(plan) => {
                self.create_params.execution_group_id = plan.execution_group_id.clone();
                self.create_params.rpc_servers = if plan.rpc_servers.is_empty() {
                    None
                } else {
                    Some(plan.rpc_servers.clone())
                };
                self.selected_control_addr = Some(plan.owner_control_addr.clone());
                self.last_plan = Some(plan.clone());
                let placement_label = if plan.display_label.trim().is_empty() {
                    plan.execution_group_id.clone()
                } else {
                    plan.display_label.clone()
                };
                self.status = format!(
                    "Planned {} on {} via {}",
                    placement_strategy_label(plan.strategy),
                    plan.owner_display_name,
                    placement_label
                );
                let _ = self.refresh_cluster();
                self.refresh_placement_candidates();
            }
            Err(err) => self.status = err,
        }
    }

    fn schedule_instance_cluster(&mut self) {
        let allowed = self.selected_allowed_node_addrs();
        let params = self.runtime_safe_create_params();
        match self.host.schedule_instance(&params, allowed, true) {
            Ok(scheduled) => {
                self.selected_control_addr = Some(scheduled.owner_control_addr.clone());
                self.selected_instance_id = Some(scheduled.instance_id);
                self.last_plan = Some(PlacementPlan {
                    owner_control_addr: scheduled.owner_control_addr.clone(),
                    owner_display_name: scheduled.owner_display_name.clone(),
                    execution_group_id: scheduled.execution_group_id.clone(),
                    rpc_servers: scheduled.rpc_servers.clone(),
                    display_label: String::new(),
                    strategy: scheduled.strategy,
                    device_count: 0,
                    remote_node_count: scheduled
                        .rpc_servers
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .count() as i32,
                    estimated_required_bytes: 0,
                    estimated_group_free_bytes: 0,
                    reusable_instance_id: if scheduled.reused_existing {
                        Some(scheduled.instance_id)
                    } else {
                        None
                    },
                    ready_now: scheduled.waited_ms == 0,
                    requires_eviction: false,
                });
                self.status = format!(
                    "Scheduled instance {} on {} via {}{}{}",
                    scheduled.instance_id,
                    scheduled.owner_display_name,
                    placement_strategy_label(scheduled.strategy),
                    if scheduled.reused_existing {
                        " (reused existing)"
                    } else {
                        ""
                    },
                    if scheduled.waited_ms > 0 {
                        format!(" after {} ms wait", scheduled.waited_ms)
                    } else {
                        String::new()
                    }
                );
                let _ = self.refresh_cluster();
                self.refresh_placement_candidates();
            }
            Err(err) => self.status = err,
        }
    }

    fn load_manual_instance_cluster(&mut self) {
        let Some(owner_control_addr) = self.manual_owner_control_addr().map(str::to_string) else {
            self.status = "Pick a primary GPU first.".to_string();
            return;
        };
        let params = self.runtime_safe_create_params();
        if params
            .manual_devices_csv
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            self.status = "Pick at least one device before loading.".to_string();
            return;
        }

        self.selected_control_addr = Some(owner_control_addr.clone());
        self.selected_rpc_peer_addrs = self.manual_remote_peer_control_addrs_for_owner();
        let result = if owner_control_addr == self.host.control_addr() {
            self.host.create_instance(&params).and_then(|instance_id| {
                self.host.load_instance(instance_id)?;
                Ok(instance_id)
            })
        } else {
            let client = AgentClient::new(owner_control_addr.clone());
            client
                .create_instance(params)
                .map_err(|err| err.to_string())
                .and_then(|instance_id| {
                    client
                        .load_instance(instance_id)
                        .map_err(|err| err.to_string())?;
                    Ok(instance_id)
                })
        };

        match result {
            Ok(instance_id) => {
                self.selected_instance_id = Some(instance_id);
                self.status = format!("Loaded instance {instance_id} on {owner_control_addr}");
                let _ = self.refresh_cluster();
                let _ = self.refresh_selected_preview();
                self.refresh_placement_candidates();
            }
            Err(err) => self.status = err,
        }
    }

    fn create_instance(&mut self) {
        let mut params = self.create_params.clone();
        let endpoints = self.selected_rpc_endpoints();
        params.rpc_servers = if endpoints.is_empty() {
            None
        } else {
            Some(endpoints.join(","))
        };
        let result = if self.selected_is_local() {
            self.host.create_instance(&params)
        } else {
            let Some(client) = self.selected_client() else {
                self.status = "select a node first".to_string();
                return;
            };
            client
                .create_instance(params)
                .map_err(|err| err.to_string())
        };
        match result {
            Ok(instance_id) => {
                self.selected_instance_id = Some(instance_id);
                self.status = format!("Created instance {instance_id}");
                let _ = self.refresh_cluster();
                self.refresh_placement_candidates();
            }
            Err(err) => self.status = err,
        }
    }

    fn forget_peer(&mut self, control_addr: &str) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.remove_peer(control_addr.to_string()) {
            Ok(()) => {
                self.status = format!("Forgot paired node {control_addr}");
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn configure_local_firewall(&mut self) {
        let Some(client) = self.host.local_client() else {
            self.status = "local host is not connected".to_string();
            return;
        };
        match client.configure_firewall() {
            Ok(()) => {
                self.status = "Firewall setup completed.".to_string();
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err.to_string(),
        }
    }

    fn select_all_remote_rpc_peers(&mut self) {
        self.selected_rpc_peer_addrs = self
            .nodes
            .iter()
            .filter(|node| {
                self.selected_control_addr.as_deref() != Some(node.control_addr.as_str())
            })
            .filter(|node| node.rpc_running)
            .map(|node| node.control_addr.clone())
            .collect();
        if let Err(err) = self.refresh_selected_preview() {
            self.status = err;
        } else {
            self.sync_defaults_from_selected_node();
        }
    }

    fn clear_remote_rpc_peers(&mut self) {
        self.selected_rpc_peer_addrs.clear();
        if let Err(err) = self.refresh_selected_preview() {
            self.status = err;
        } else {
            self.sync_defaults_from_selected_node();
        }
    }

    fn load_selected(&mut self) {
        let Some(instance_id) = self.selected_instance_id else {
            self.status = "select an instance first".to_string();
            return;
        };
        let result = if self.selected_is_local() {
            self.host.load_instance(instance_id)
        } else {
            let Some(client) = self.selected_client() else {
                return;
            };
            client
                .load_instance(instance_id)
                .map_err(|err| err.to_string())
        };
        match result {
            Ok(()) => {
                self.status = format!("Loaded instance {instance_id}");
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err,
        }
    }

    fn unload_selected(&mut self) {
        let Some(instance_id) = self.selected_instance_id else {
            self.status = "select an instance first".to_string();
            return;
        };
        let result = if self.selected_is_local() {
            self.host.unload_instance(instance_id)
        } else {
            let Some(client) = self.selected_client() else {
                return;
            };
            client
                .unload_instance(instance_id)
                .map_err(|err| err.to_string())
        };
        match result {
            Ok(()) => {
                self.status = format!("Unloaded instance {instance_id}");
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err,
        }
    }

    fn remove_selected(&mut self) {
        let Some(instance_id) = self.selected_instance_id else {
            self.status = "select an instance first".to_string();
            return;
        };
        let result = if self.selected_is_local() {
            self.host.remove_instance(instance_id)
        } else {
            let Some(client) = self.selected_client() else {
                return;
            };
            client
                .remove_instance(instance_id)
                .map_err(|err| err.to_string())
        };
        match result {
            Ok(()) => {
                self.selected_instance_id = None;
                self.status = format!("Removed instance {instance_id}");
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err,
        }
    }

    fn toggle_retention_selected(&mut self) {
        let Some(instance) = self.selected_instance().cloned() else {
            self.status = "select an instance first".to_string();
            return;
        };
        let next_mode = match instance.retention_mode {
            RetentionMode::KeepLoaded => RetentionMode::LoadOnDemand,
            RetentionMode::LoadOnDemand => RetentionMode::KeepLoaded,
        };
        let result = if self.selected_is_local() {
            self.host
                .set_retention_mode(instance.instance_id, next_mode)
        } else {
            let Some(client) = self.selected_client() else {
                return;
            };
            client
                .set_retention_mode(instance.instance_id, next_mode)
                .map_err(|err| err.to_string())
        };
        match result {
            Ok(()) => {
                self.status = format!("Updated retention for instance {}", instance.instance_id);
                let _ = self.refresh_cluster();
            }
            Err(err) => self.status = err,
        }
    }

    fn run_chat(&mut self) {
        let Some(instance_id) = self.selected_instance_id else {
            self.status = "select an instance first".to_string();
            return;
        };
        self.chat_request.instance_id = instance_id;
        let result = if self.selected_is_local() {
            self.host.chat_complete(&self.chat_request)
        } else {
            let Some(client) = self.selected_client() else {
                return;
            };
            client
                .chat_complete(self.chat_request.clone())
                .map_err(|err| err.to_string())
        };
        match result {
            Ok(result) => {
                self.chat_response = result.text;
                self.last_chat_metrics = Some(result.metrics);
                self.status = format!("Chat completed on instance {instance_id}");
                let _ = self.refresh_cluster();
            }
            Err(err) => {
                self.last_chat_metrics = None;
                self.status = err;
            }
        }
    }

    fn has_system_tray(&self) -> bool {
        self.tray.is_some()
    }

    fn show_window_from_tray(&mut self, ctx: &egui::Context) {
        self.window_hidden_to_tray = false;
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
        ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
            egui::UserAttentionType::Informational,
        ));
        self.status = "Controller restored from tray.".to_string();
    }

    fn hide_window_to_tray(&mut self, ctx: &egui::Context) {
        if !self.has_system_tray() {
            return;
        }
        self.window_hidden_to_tray = true;
        self.status = "Controller hidden to tray. The local host keeps running.".to_string();
        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
    }

    fn request_exit(&mut self, ctx: &egui::Context) {
        if self.shutdown_requested_at.is_some() {
            return;
        }
        self.allow_exit = true;
        self.window_hidden_to_tray = false;
        self.shutdown_requested_at = Some(Instant::now());
        self.host.clear_local_connection();
        stop_local_support_processes(self.host.runtime_dir());
        self.status = "Shutting down controller...".to_string();
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        ctx.request_repaint();
    }

    fn restart_controller(&mut self, ctx: &egui::Context) {
        if self.shutdown_requested_at.is_some() {
            return;
        }
        let current_exe = match std::env::current_exe() {
            Ok(path) => path,
            Err(err) => {
                self.status = format!("failed to restart Engine: {err}");
                return;
            }
        };
        let mut command = Command::new(&current_exe);
        if let Ok(cwd) = std::env::current_dir() {
            command.current_dir(cwd);
        }
        match command.spawn() {
            Ok(_) => {
                self.request_exit(ctx);
                self.status = "Restarting Engine...".to_string();
                ctx.request_repaint();
            }
            Err(err) => {
                self.status = format!("failed to restart Engine: {err}");
            }
        }
    }

    fn open_local_models_folder(&mut self) {
        let path = self.local_models_dir();
        match open_path_in_file_manager(&path) {
            Ok(()) => {
                self.status = format!("Opened model folder '{}'.", path.display());
            }
            Err(err) => {
                self.status = err;
            }
        }
    }

    fn open_selected_model_package_folder(&mut self) {
        let Some(package) = self.selected_local_model_package() else {
            self.status = "Choose a model folder that exists on this node first.".to_string();
            return;
        };
        match open_path_in_file_manager(&package.path) {
            Ok(()) => {
                self.status = format!("Opened model package '{}'.", package.display_name);
            }
            Err(err) => {
                self.status = err;
            }
        }
    }

    fn handle_tray_actions(&mut self, ctx: &egui::Context) {
        let Some(tray) = &self.tray else {
            return;
        };

        for action in tray.poll_actions() {
            match action {
                TrayAction::OpenController => self.show_window_from_tray(ctx),
                TrayAction::RefreshCluster => {
                    self.enqueue_cluster_refresh();
                    self.enqueue_telemetry_refresh();
                    self.status = "Refreshing state from tray...".to_string();
                    self.last_auto_refresh = Instant::now();
                }
                TrayAction::QuitController => self.request_exit(ctx),
            }
        }
    }
}

impl App for ClusterControllerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.apply_theme_preference(ctx);
        self.run_startup_initialization();
        if self.host.local_client().is_none()
            && !self.local_connect_in_progress
            && self
                .startup_connect_due_at
                .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.startup_connect_due_at = None;
            self.connect_local_host();
        }
        ctx.request_repaint_after(Duration::from_millis(
            if self.host.local_client().is_some() {
                33
            } else {
                250
            },
        ));
        self.handle_tray_actions(ctx);
        self.poll_runtime_install_events();
        self.drain_model_store_events();
        self.drain_model_transfer_events();
        self.drain_controller_events();
        self.sync_manual_refresh_state();
        self.poll_local_model_store_changes();

        if self
            .shutdown_requested_at
            .is_some_and(|started| started.elapsed() >= Duration::from_secs(2))
        {
            stop_local_support_processes(self.host.runtime_dir());
            std::process::exit(0);
        }

        if self.host.local_client().is_some()
            && self.last_pairing_poll.elapsed() >= Duration::from_secs(1)
        {
            self.enqueue_pairing_poll();
            self.last_pairing_poll = Instant::now();
        }
        self.apply_pairing_attention(ctx);

        if ctx.input(|input| input.viewport().close_requested()) && !self.allow_exit {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            if self.has_system_tray() {
                self.hide_window_to_tray(ctx);
            } else {
                self.request_exit(ctx);
            }
        }

        if self.window_hidden_to_tray {
            return;
        }

        if self.host.local_client().is_some()
            && self.last_auto_refresh.elapsed() >= Duration::from_secs(8)
        {
            self.enqueue_cluster_refresh();
            self.last_auto_refresh = Instant::now();
        }
        if self.host.local_client().is_some()
            && self.last_telemetry_refresh.elapsed() >= Duration::from_millis(500)
        {
            self.enqueue_telemetry_refresh();
        }
        controller_ui::render_controller(self, ctx);
        self.persist_controller_settings_if_changed();
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.host.clear_local_connection();
        stop_local_support_processes(self.host.runtime_dir());
    }
}

fn query_cluster_refresh(
    local_client: AgentClient,
    local_control_addr: String,
    selected_control_addr: Option<String>,
    selected_rpc_peer_addrs: BTreeSet<String>,
) -> Result<ClusterRefreshPayload, String> {
    let mut local_snapshot = local_client.get_snapshot().map_err(|e| e.to_string())?;
    local_snapshot.control_addr = local_control_addr.clone();

    let peers = local_client.list_peers().map_err(|e| e.to_string())?;
    let pairing_requests = local_client
        .list_pairing_requests()
        .map_err(|e| e.to_string())?;
    let discovery_status = local_client
        .get_discovery_status()
        .map_err(|e| e.to_string())?;

    let mut nodes = vec![local_snapshot];
    let mut warnings = Vec::new();
    for peer in peers.iter().filter(|peer| peer.trusted) {
        let remote = AgentClient::new(peer.control_addr.clone());
        match remote.get_snapshot() {
            Ok(mut snapshot) => {
                snapshot.control_addr = peer.control_addr.clone();
                if snapshot.advertised_control_addr.is_none() {
                    snapshot.advertised_control_addr = peer.advertised_control_addr.clone();
                }
                if snapshot.advertised_rpc_endpoint.is_none() {
                    snapshot.advertised_rpc_endpoint = peer.advertised_rpc_endpoint.clone();
                }
                nodes.push(snapshot);
            }
            Err(err) => warnings.push(format!("{}: {}", peer.display_name, err)),
        }
    }

    nodes.sort_by(|a, b| {
        a.node
            .display_name
            .cmp(&b.node.display_name)
            .then(a.control_addr.cmp(&b.control_addr))
    });

    let selected_control_addr = selected_control_addr
        .filter(|selected| nodes.iter().any(|node| &node.control_addr == selected))
        .or_else(|| {
            nodes
                .iter()
                .find(|node| node.control_addr == local_control_addr)
                .or_else(|| nodes.first())
                .map(|node| node.control_addr.clone())
        });

    let preview_node = if let Some(selected) = selected_control_addr.as_ref() {
        let preview_client = if selected == &local_control_addr {
            local_client.clone()
        } else {
            AgentClient::new(selected.clone())
        };
        let endpoints = nodes
            .iter()
            .filter(|node| node.control_addr != *selected)
            .filter(|node| selected_rpc_peer_addrs.contains(&node.control_addr))
            .filter_map(rpc_endpoint_for_node)
            .collect::<Vec<_>>();
        preview_client
            .get_snapshot_with_rpc(if endpoints.is_empty() {
                None
            } else {
                Some(endpoints.join(","))
            })
            .ok()
    } else {
        None
    };

    let telemetry = query_cluster_telemetry(local_client, local_control_addr)?;

    Ok(ClusterRefreshPayload {
        peers,
        pairing_requests,
        discovery_status,
        nodes,
        telemetry,
        preview_node,
        selected_control_addr,
        warnings,
    })
}

fn query_cluster_telemetry(
    local_client: AgentClient,
    local_control_addr: String,
) -> Result<Vec<TelemetrySnapshot>, String> {
    let mut telemetry = local_client
        .get_cluster_telemetry()
        .map_err(|e| e.to_string())?;
    let mut seen = BTreeSet::new();

    for snapshot in &mut telemetry {
        if snapshot.control_addr == local_control_addr
            || control_addr_is_loopback(&snapshot.control_addr)
        {
            snapshot.control_addr = local_control_addr.clone();
        }
        if snapshot
            .advertised_control_addr
            .as_ref()
            .is_some_and(|addr| control_addr_is_loopback(addr))
        {
            snapshot.advertised_control_addr = Some(local_control_addr.clone());
        }
    }

    telemetry.sort_by(|lhs, rhs| {
        lhs.node
            .display_name
            .cmp(&rhs.node.display_name)
            .then(lhs.control_addr.cmp(&rhs.control_addr))
    });
    telemetry.retain(|snapshot| {
        if seen.contains(&snapshot.control_addr) {
            return false;
        }
        seen.insert(snapshot.control_addr.clone());
        true
    });
    Ok(telemetry)
}

fn query_pairing_poll(
    local_client: AgentClient,
    previous_trusted: BTreeSet<String>,
) -> Result<PairingPollPayload, String> {
    let peers = local_client.list_peers().map_err(|e| e.to_string())?;
    let pairing_requests = local_client
        .list_pairing_requests()
        .map_err(|e| e.to_string())?;
    let discovery_status = local_client
        .get_discovery_status()
        .map_err(|e| e.to_string())?;
    let next_trusted = peers
        .iter()
        .filter(|peer| peer.trusted)
        .map(|peer| peer.control_addr.clone())
        .collect::<BTreeSet<_>>();
    Ok(PairingPollPayload {
        peers,
        pairing_requests,
        discovery_status,
        topology_changed: previous_trusted != next_trusted,
    })
}

fn labeled_i32(ui: &mut egui::Ui, label: &str, value: &mut i32) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).speed(1));
}

fn labeled_f32(ui: &mut egui::Ui, label: &str, value: &mut f32) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).speed(0.05));
}

fn format_mib(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const TIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;

    let bytes = bytes as f64;
    if bytes >= TIB {
        format!("{:.2} TiB", bytes / TIB)
    } else if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes / KIB)
    } else {
        format!("{} B", bytes as u64)
    }
}

fn device_is_cpu_for_defaults(device: &cluster_api::DeviceInfo) -> bool {
    let lowered = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    lowered.contains("cpu") || lowered.contains("blas") || lowered.contains("accelerate")
}

fn device_is_rpc_for_defaults(device: &cluster_api::DeviceInfo) -> bool {
    let lowered = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    lowered.contains("rpc")
}

fn device_is_integrated_for_defaults(
    node: &NodeSnapshot,
    device: &cluster_api::DeviceInfo,
) -> bool {
    if device_is_cpu_for_defaults(device) || device_is_rpc_for_defaults(device) {
        return false;
    }
    let lowered_backend = format!("{} {}", device.backend, device.name).to_ascii_lowercase();
    if node.node.os_name.eq_ignore_ascii_case("macos") && lowered_backend.contains("metal") {
        return false;
    }
    let lowered = format!("{} {}", device.name, device.description).to_ascii_lowercase();
    let looks_intel_integrated = lowered.contains("intel") && !lowered.contains("arc");
    let looks_integrated_family = lowered.contains("integrated")
        || lowered.contains("uhd")
        || lowered.contains("iris")
        || lowered.contains("hd graphics")
        || lowered.contains("xe graphics")
        || lowered.contains("graphics controller")
        || lowered.contains("apu")
        || lowered.contains("uma");
    let looks_shared_memory = lowered.contains("shared")
        || lowered.contains("unified")
        || lowered.contains("system memory");
    looks_intel_integrated || looks_integrated_family || looks_shared_memory
}

fn format_bytes_compact(bytes: u64) -> String {
    if bytes == 0 {
        return "-".to_string();
    }
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    if bytes < 1024 * 1024 {
        return format!("{:.1} KiB", bytes as f64 / 1024.0);
    }
    format_mib(bytes)
}

fn strategy_rank(strategy: PlacementStrategy) -> i32 {
    match strategy {
        PlacementStrategy::SingleNode => 0,
        PlacementStrategy::LocalSplit => 1,
        PlacementStrategy::HybridTwoNode => 2,
        PlacementStrategy::HybridMultiNode => 3,
    }
}

fn format_duration_compact(seconds: u64) -> String {
    if seconds < 60 {
        return format!("{seconds}s");
    }
    if seconds < 3600 {
        return format!("{}m {}s", seconds / 60, seconds % 60);
    }
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    if minutes == 0 {
        format!("{hours}h")
    } else {
        format!("{hours}h {minutes}m")
    }
}

fn format_download_progress_status(progress: &ModelDownloadProgress) -> String {
    let current = progress
        .current_file
        .as_deref()
        .map(|value| format!("Downloading {value}"))
        .unwrap_or_else(|| "Downloading model files".to_string());
    let totals = if progress.total_bytes > 0 {
        format!(
            "{} / {}",
            format_bytes_compact(progress.downloaded_bytes),
            format_bytes_compact(progress.total_bytes),
        )
    } else {
        format_bytes_compact(progress.downloaded_bytes)
    };
    let mut parts = vec![
        current,
        format!(
            "{} / {} files",
            progress.completed_files,
            progress.total_files.max(progress.completed_files)
        ),
        totals,
    ];
    if progress.bytes_per_second > 0 {
        parts.push(format!(
            "{}/s",
            format_bytes_compact(progress.bytes_per_second)
        ));
    }
    if let Some(eta) = progress.eta_seconds {
        parts.push(format!("ETA {}", format_duration_compact(eta)));
    }
    parts.join(" | ")
}

fn format_mib_from_bytes(bytes: u64) -> String {
    if bytes == 0 {
        "-".to_string()
    } else {
        format_mib(bytes)
    }
}

fn control_addr_is_loopback(value: &str) -> bool {
    let host = value
        .trim()
        .rsplit_once(':')
        .map(|(host, _)| host.trim())
        .unwrap_or_else(|| value.trim())
        .trim_matches(|ch| ch == '[' || ch == ']');
    host.eq_ignore_ascii_case("localhost") || matches!(host, "127.0.0.1" | "::1")
}

fn open_path_in_file_manager(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("Path '{}' does not exist.", path.display()));
    }

    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("explorer");
        command.arg(path);
        command
    };

    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(path);
        command
    };

    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(path);
        command
    };

    command
        .spawn()
        .map_err(|err| format!("Failed to open '{}': {err}", path.display()))?;
    Ok(())
}

fn package_file_path(package: &ModelPackage, relative_path: &str) -> PathBuf {
    let mut path = package.path.clone();
    for segment in relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
    {
        path.push(segment);
    }
    path
}

fn instance_model_task(value: &str) -> ManagedModelTask {
    match value {
        "embeddings" => ManagedModelTask::Embeddings,
        "rerank" => ManagedModelTask::Rerank,
        "whisper" | "realtime-audio" | "diarization" => ManagedModelTask::Transcription,
        _ => ManagedModelTask::Responses,
    }
}

fn instance_model_type_label(value: &str) -> &'static str {
    match value {
        "text" => "Text",
        "vision" => "Vision",
        "embeddings" => "Embeddings",
        "rerank" => "Rerank",
        "whisper" => "Whisper",
        "realtime-audio" => "Realtime audio",
        "diarization" => "Diarization",
        _ => "Text",
    }
}

fn suggested_instance_name(package: &ModelPackage, kind: &str, model_file: &str) -> String {
    let stem = PathBuf::from(model_file)
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| package.folder_name.clone());
    sanitize_folder_name(&format!(
        "{}_{}_{}",
        package.folder_name,
        kind.replace('-', "_"),
        stem
    ))
}

fn format_ratio(used: u64, total: u64) -> String {
    if total == 0 {
        "-".to_string()
    } else {
        format!("{:.0}%", (used as f64 / total as f64) * 100.0)
    }
}

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn parse_multiline_list(value: &str) -> Vec<String> {
    value
        .split(['\n', '\r', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}

fn telemetry_age_ms(unix_ms: u64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    now.saturating_sub(unix_ms)
}

fn format_inference_metrics(metrics: &InferenceMetrics) -> String {
    format!(
        "Latency {:.1} ms | queue {:.1} ms | load {:.1} ms | prompt {:.2} tok/s ({} tok / {:.1} ms) | decode {:.2} tok/s ({} tok / {:.1} ms) | total {:.2} tok/s | rpc {} ({}) | payload {} | model {}{}{}",
        metrics.request_total_ms,
        metrics.queue_wait_ms,
        metrics.load_ms,
        metrics.prompt_tokens_per_second,
        metrics.prompt_tokens,
        metrics.prompt_ms,
        metrics.decode_tokens_per_second,
        metrics.decoded_tokens,
        metrics.predicted_ms,
        metrics.total_tokens_per_second,
        if metrics.used_rpc { "yes" } else { "no" },
        metrics.rpc_server_count,
        format_bytes_compact(metrics.request_bytes),
        format_mib_from_bytes(metrics.model_bytes),
        if metrics.mmproj_bytes > 0 { " | mmproj " } else { "" },
        if metrics.mmproj_bytes > 0 {
            format_mib_from_bytes(metrics.mmproj_bytes)
        } else {
            String::new()
        }
    )
}

fn format_link_metrics(link: &LinkMetrics) -> String {
    let age_ms = telemetry_age_ms(link.unix_ms);
    let probe_kind = if link.probe_kind.trim().is_empty() {
        "measured"
    } else {
        link.probe_kind.trim()
    };
    match &link.error {
        Some(error) => format!(
            "{} {} -> {} | {} ms ago | benchmark failed: {}",
            probe_kind, link.transport, link.peer_control_addr, age_ms, error
        ),
        None => {
            let throughput = if link.goodput_mbps >= 1_000.0 {
                format!("{:.2} Gbps", link.goodput_mbps / 1_000.0)
            } else {
                format!("{:.0} Mbps", link.goodput_mbps)
            };
            format!(
                "{} {} -> {} | {} ms ago | latency {:.2} ms | throughput {} | payload {} | duration {:.0} ms",
                probe_kind,
                link.transport,
                link.peer_control_addr,
                age_ms,
                link.latency_ms,
                throughput,
                format_bytes_compact(link.payload_bytes),
                link.duration_ms
            )
        }
    }
}

fn placement_strategy_label(strategy: PlacementStrategy) -> &'static str {
    match strategy {
        PlacementStrategy::SingleNode => "single-node",
        PlacementStrategy::LocalSplit => "local-split",
        PlacementStrategy::HybridTwoNode => "hybrid-2-node",
        PlacementStrategy::HybridMultiNode => "hybrid-multi-node",
    }
}

fn state_label(state: i32) -> &'static str {
    match state {
        0 => "unloaded",
        1 => "loading",
        2 => "loaded",
        3 => "serving",
        4 => "grace",
        5 => "failed",
        _ => "unknown",
    }
}

fn rpc_endpoint_for_node(node: &NodeSnapshot) -> Option<String> {
    if !node.rpc_running {
        return None;
    }
    if let Some(endpoint) = node.advertised_rpc_endpoint.clone() {
        return Some(endpoint);
    }

    let port = node
        .rpc_endpoint
        .as_deref()
        .and_then(|value| value.rsplit_once(':').map(|(_, port)| port.to_string()))
        .unwrap_or_else(|| CLUSTER_AGENT_RPC_PORT.to_string());
    let control_addr = node
        .advertised_control_addr
        .as_deref()
        .unwrap_or(&node.control_addr);
    let (host, _) = control_addr.rsplit_once(':')?;
    Some(format!("{host}:{port}"))
}
