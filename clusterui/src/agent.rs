use crate::catalog::{default_models_dir, discover_models, find_model_entry, ManagedModelEntry};
use crate::cluster_api::{
    AudioRawRequest, ChatRequest, ClusterApi, CreateInstanceParams, EmbeddingsRequest, JsonResult,
    NativeAudioTranscriptionRequest, RerankRequest, RetentionMode, TextGenerationResult,
    VlmRequest,
};
use crate::model_metadata::{estimate_runtime_vram, inspect_model_file};
use crate::model_store::{
    discover_model_packages, sanitize_folder_name, touch_model_store_change_marker, ModelArtifact,
    ModelPackage,
};
use crate::protocol::{
    AgentRequest, AgentResponse, ClusterModelArtifactInfo, ClusterModelPackageInfo,
    DiscoveryAnnouncement, DiscoveryMode, DiscoveryStatus, LinkMetrics, ModelFileNodeAvailability,
    ModelPackageNodeAvailability, NodeSnapshot, PairingRequestInfo, PathStat, PeerInfo,
    PlacementPlan, PlacementStrategy, PublicApiConfig, PublicApiConfigUpdate, PublicApiStatus,
    ResolvedClusterInstance, ScheduledInstance, TelemetrySnapshot, CLUSTER_AGENT_CONTROL_PORT,
    CLUSTER_AGENT_DISCOVERY_PORT, CLUSTER_AGENT_RPC_PORT, CLUSTER_AGENT_TELEMETRY_PORT,
};
use crate::public_server::{start_public_server, PublicServerHandle};
use crate::settings;
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::env;
#[cfg(target_os = "macos")]
use std::ffi::CStr;
use std::fs;
use std::io::{Read, Write};
use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream, ToSocketAddrs,
    UdpSocket,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{ProcessesToUpdate, System};

#[cfg(target_os = "windows")]
use windows_sys::Win32::NetworkManagement::IpHelper::{
    GetAdaptersAddresses, GAA_FLAG_INCLUDE_PREFIX, IP_ADAPTER_ADDRESSES_LH,
};
#[cfg(target_os = "windows")]
use windows_sys::Win32::Networking::WinSock::{AF_INET, SOCKADDR_IN};
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

const DISCOVERY_MULTICAST_IP: Ipv4Addr = Ipv4Addr::new(239, 255, 42, 99);
const PEER_TTL: Duration = Duration::from_secs(45);
const DISCOVERY_ANNOUNCE_INTERVAL: Duration = Duration::from_secs(10);
const STARTUP_KNOWN_DISCOVERY_DURATION: Duration = Duration::from_secs(60);
const PAIRING_REQUEST_TTL: Duration = Duration::from_secs(300);
const CONNECT_RETRY_TIMEOUT: Duration = Duration::from_secs(8);
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(300);
const REQUEST_TIMEOUT_FAST: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT_STATE: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT_LOAD: Duration = Duration::from_secs(300);
const REQUEST_TIMEOUT_CHAT: Duration = Duration::from_secs(600);
const REQUEST_TIMEOUT_LINK_BENCHMARK: Duration = Duration::from_secs(45);
const SCHEDULE_WAIT_TIMEOUT: Duration = Duration::from_secs(35);
const SCHEDULE_WAIT_POLL: Duration = Duration::from_millis(500);
const SNAPSHOT_QUERY_RETRIES: usize = 3;
const SNAPSHOT_QUERY_RETRY_DELAY: Duration = Duration::from_millis(350);
const TELEMETRY_TTL: Duration = Duration::from_secs(8);
const TELEMETRY_DIRECT_QUERY_TIMEOUT: Duration = Duration::from_millis(900);
const RPC_ENDPOINT_PROBE_TIMEOUT: Duration = Duration::from_millis(350);
const RPC_SERVER_REACHABILITY_CACHE_TTL: Duration = Duration::from_millis(750);
const LINK_PROBE_TIMEOUT: Duration = Duration::from_secs(15);
const LINK_BENCHMARK_PING_SAMPLES: usize = 5;
const LINK_BENCHMARK_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const LINK_BENCHMARK_STARTUP_WARMUP_BYTES: u64 = 16 * 1024 * 1024;
const LINK_BENCHMARK_FULL_WARMUP_BYTES: u64 = 64 * 1024 * 1024;

const INSTANCE_STATE_UNLOADED: i32 = 0;
const INSTANCE_STATE_LOADING: i32 = 1;
const INSTANCE_STATE_LOADED: i32 = 2;
const INSTANCE_STATE_SERVING: i32 = 3;
const INSTANCE_STATE_GRACE: i32 = 4;
const INSTANCE_STATE_FAILED: i32 = 5;

#[cfg(target_os = "windows")]
const FIREWALL_RULE_CONTROL: &str = "ENGINE Cluster Agent Control";
#[cfg(target_os = "windows")]
const FIREWALL_RULE_RPC: &str = "ENGINE Cluster RPC";
#[cfg(target_os = "windows")]
const FIREWALL_RULE_DISCOVERY: &str = "ENGINE Cluster Discovery";
#[cfg(target_os = "windows")]
const FIREWALL_RULE_TELEMETRY: &str = "ENGINE Cluster Telemetry";
#[cfg(target_os = "windows")]
const FIREWALL_RULE_PUBLIC_API: &str = "ENGINE Cluster Public API";
const CLUSTER_PUBLIC_API_PORT: u16 = 46310;

type SharedClusterApi = Arc<RwLock<ClusterApi>>;

fn api_read<'a>(api: &'a SharedClusterApi) -> Result<RwLockReadGuard<'a, ClusterApi>> {
    api.read()
        .map_err(|_| anyhow::anyhow!("cluster api rwlock poisoned"))
}

fn api_write<'a>(api: &'a SharedClusterApi) -> Result<RwLockWriteGuard<'a, ClusterApi>> {
    api.write()
        .map_err(|_| anyhow::anyhow!("cluster api rwlock poisoned"))
}

#[derive(Clone)]
struct DiscoveredPeer {
    info: PeerInfo,
    last_seen: Instant,
    manual: bool,
    shared_token: Option<String>,
}

#[derive(Clone)]
struct PendingPairingRequest {
    info: PairingRequestInfo,
    received_at: Instant,
}

#[derive(Clone)]
struct OutgoingPairingRequest {
    request_id: String,
    request_code: String,
    control_addr: String,
    requested_at: Instant,
}

#[derive(Clone)]
struct DiscoveryRuntimeState {
    mode: DiscoveryMode,
    active_until: Option<Instant>,
}

struct RpcServerProcess {
    started_at: Instant,
}

#[derive(Clone, Copy)]
struct CachedRpcServerReachability {
    reachable: bool,
    checked_at: Instant,
}

#[derive(Clone)]
struct TelemetryEntry {
    snapshot: TelemetrySnapshot,
    last_seen: Instant,
}

struct AgentRuntimeState {
    runtime_dir: PathBuf,
    bind_addr: String,
    local_control_addr: String,
    local_node: crate::cluster_api::NodeInfo,
    models_dir: PathBuf,
    rpc_server: Mutex<Option<RpcServerProcess>>,
    schedule_lock: Mutex<()>,
    public_api: Mutex<PublicApiRuntimeState>,
    link_metrics: Mutex<HashMap<String, LinkMetrics>>,
    discovery: Mutex<DiscoveryRuntimeState>,
    pairing_requests: Mutex<HashMap<String, PendingPairingRequest>>,
    outgoing_pairing_requests: Mutex<HashMap<String, OutgoingPairingRequest>>,
}

struct PublicApiRuntimeState {
    config: PublicApiConfig,
    running: bool,
    bound_addr: Option<String>,
    last_error: Option<String>,
    handle: Option<PublicServerHandle>,
}

#[derive(Clone)]
struct PlacementCandidate {
    connect_control_addr: String,
    plan: PlacementPlan,
}

enum PreparedScheduleTarget {
    Reuse(i64),
    CreateNew,
}

#[derive(Clone, Default)]
struct FirewallState {
    status: Option<String>,
    action_required: bool,
}

#[cfg(target_os = "windows")]
#[derive(Clone)]
struct CachedFirewallState {
    state: FirewallState,
    checked_at: Instant,
}

#[derive(Clone, Debug)]
struct InterfaceCandidate {
    ip: Ipv4Addr,
    name: String,
    description: String,
    score: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicApiBindHostOption {
    pub host: String,
    pub label: String,
}

#[derive(Clone)]
pub struct AgentClient {
    control_addr: String,
}

pub struct ModelArtifactTransferOutcome {
    pub size_bytes: u64,
    pub skipped: bool,
}

impl AgentClient {
    pub fn new(control_addr: impl Into<String>) -> Self {
        Self {
            control_addr: control_addr.into(),
        }
    }

    pub fn ping(&self) -> Result<()> {
        match self.send(AgentRequest::Ping, REQUEST_TIMEOUT_FAST)? {
            AgentResponse::Pong => Ok(()),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn get_snapshot(&self) -> Result<NodeSnapshot> {
        self.get_snapshot_with_rpc(None)
    }

    pub fn get_snapshot_with_rpc(&self, rpc_servers: Option<String>) -> Result<NodeSnapshot> {
        match self.send(
            AgentRequest::GetSnapshot { rpc_servers },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::Snapshot { snapshot } => Ok(snapshot),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_peers(&self) -> Result<Vec<PeerInfo>> {
        match self.send(AgentRequest::ListPeers, REQUEST_TIMEOUT_FAST)? {
            AgentResponse::Peers { peers } => Ok(peers),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn get_cluster_telemetry(&self) -> Result<Vec<TelemetrySnapshot>> {
        match self.send(AgentRequest::GetClusterTelemetry, REQUEST_TIMEOUT_FAST)? {
            AgentResponse::ClusterTelemetry { snapshots } => Ok(snapshots),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn run_link_benchmarks(&self, full: bool) -> Result<()> {
        match self.send(
            AgentRequest::RunLinkBenchmarks { full },
            REQUEST_TIMEOUT_LINK_BENCHMARK,
        )? {
            AgentResponse::Ok => Ok(()),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn measure_link_to(&self, control_addr: String, full: bool) -> Result<LinkMetrics> {
        match self.send(
            AgentRequest::MeasureLinkTo { control_addr, full },
            REQUEST_TIMEOUT_LINK_BENCHMARK,
        )? {
            AgentResponse::LinkMetrics { metrics } => Ok(metrics),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn get_local_telemetry(&self) -> Result<TelemetrySnapshot> {
        self.get_local_telemetry_with_timeout(REQUEST_TIMEOUT_FAST)
    }

    fn get_local_telemetry_with_timeout(&self, timeout: Duration) -> Result<TelemetrySnapshot> {
        match self.send(AgentRequest::GetLocalTelemetry, timeout)? {
            AgentResponse::LocalTelemetry { snapshot } => Ok(snapshot),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn add_peer(&self, control_addr: impl Into<String>) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::AddPeer {
                control_addr: control_addr.into(),
            },
            REQUEST_TIMEOUT_FAST,
        )?)
    }

    pub fn remove_peer(&self, control_addr: impl Into<String>) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::RemovePeer {
                control_addr: control_addr.into(),
            },
            REQUEST_TIMEOUT_FAST,
        )?)
    }

    pub fn start_discovery(&self, mode: DiscoveryMode, seconds: u64) -> Result<DiscoveryStatus> {
        match self.send(
            AgentRequest::StartDiscovery { mode, seconds },
            REQUEST_TIMEOUT_FAST,
        )? {
            AgentResponse::DiscoveryStatus { status } => Ok(status),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn get_discovery_status(&self) -> Result<DiscoveryStatus> {
        match self.send(AgentRequest::GetDiscoveryStatus, REQUEST_TIMEOUT_FAST)? {
            AgentResponse::DiscoveryStatus { status } => Ok(status),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_pairing_requests(&self) -> Result<Vec<PairingRequestInfo>> {
        match self.send(AgentRequest::ListPairingRequests, REQUEST_TIMEOUT_FAST)? {
            AgentResponse::PairingRequests { requests } => Ok(requests),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn request_pairing(&self, control_addr: impl Into<String>) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::RequestPairing {
                control_addr: control_addr.into(),
            },
            REQUEST_TIMEOUT_FAST,
        )?)
    }

    pub fn accept_pairing_request(&self, request_id: impl Into<String>) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::AcceptPairingRequest {
                request_id: request_id.into(),
            },
            REQUEST_TIMEOUT_FAST,
        )?)
    }

    pub fn decline_pairing_request(&self, request_id: impl Into<String>) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::DeclinePairingRequest {
                request_id: request_id.into(),
            },
            REQUEST_TIMEOUT_FAST,
        )?)
    }

    pub fn restart_rpc_server(&self) -> Result<()> {
        self.expect_ok(self.send(AgentRequest::RestartRpcServer, REQUEST_TIMEOUT_STATE)?)
    }

    pub fn configure_firewall(&self) -> Result<()> {
        self.expect_ok(self.send(AgentRequest::ConfigureFirewall, REQUEST_TIMEOUT_STATE)?)
    }

    pub fn stat_paths(&self, paths: Vec<String>) -> Result<Vec<PathStat>> {
        match self.send(AgentRequest::StatPaths { paths }, REQUEST_TIMEOUT_STATE)? {
            AgentResponse::PathStats { stats } => Ok(stats),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn plan_instance(
        &self,
        params: CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
    ) -> Result<PlacementPlan> {
        match self.send(
            AgentRequest::PlanInstance {
                params,
                allowed_control_addrs,
            },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::PlacementPlan { plan } => Ok(plan),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_placement_candidates(
        &self,
        params: CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
    ) -> Result<Vec<PlacementPlan>> {
        match self.send(
            AgentRequest::ListPlacementCandidates {
                params,
                allowed_control_addrs,
            },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::PlacementCandidates { plans } => Ok(plans),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn schedule_instance(
        &self,
        params: CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
        load_immediately: bool,
    ) -> Result<ScheduledInstance> {
        match self.send(
            AgentRequest::ScheduleInstance {
                params,
                allowed_control_addrs,
                load_immediately,
            },
            REQUEST_TIMEOUT_LOAD,
        )? {
            AgentResponse::ScheduledInstance { scheduled } => Ok(scheduled),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn resolve_cluster_instance(
        &self,
        name: impl Into<String>,
        load_if_managed: bool,
    ) -> Result<ResolvedClusterInstance> {
        match self.send(
            AgentRequest::ResolveClusterInstance {
                name: name.into(),
                load_if_managed,
            },
            REQUEST_TIMEOUT_LOAD,
        )? {
            AgentResponse::ResolvedClusterInstance { resolved } => Ok(resolved),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_managed_models(&self) -> Result<Vec<ManagedModelEntry>> {
        match self.send(AgentRequest::ListManagedModels, REQUEST_TIMEOUT_STATE)? {
            AgentResponse::ManagedModels { models } => Ok(models),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_cluster_managed_models(&self) -> Result<Vec<ManagedModelEntry>> {
        match self.send(
            AgentRequest::ListClusterManagedModels,
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::ManagedModels { models } => Ok(models),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn resolve_managed_model(
        &self,
        model_id: impl Into<String>,
    ) -> Result<Option<ManagedModelEntry>> {
        match self.send(
            AgentRequest::ResolveManagedModel {
                model_id: model_id.into(),
            },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::ManagedModel { model } => Ok(model),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn resolve_cluster_managed_model(
        &self,
        model_id: impl Into<String>,
    ) -> Result<Option<ManagedModelEntry>> {
        match self.send(
            AgentRequest::ResolveClusterManagedModel {
                model_id: model_id.into(),
            },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::ManagedModel { model } => Ok(model),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_model_packages(&self) -> Result<Vec<ModelPackage>> {
        match self.send(AgentRequest::ListModelPackages, REQUEST_TIMEOUT_STATE)? {
            AgentResponse::ModelPackages { packages } => Ok(packages),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn list_cluster_model_packages(&self) -> Result<Vec<ClusterModelPackageInfo>> {
        match self.send(
            AgentRequest::ListClusterModelPackages,
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::ClusterModelPackages { packages } => Ok(packages),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn get_public_api_status(&self) -> Result<PublicApiStatus> {
        match self.send(AgentRequest::GetPublicApiStatus, REQUEST_TIMEOUT_STATE)? {
            AgentResponse::PublicApiStatus { status } => Ok(status),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn update_public_api_config(
        &self,
        update: PublicApiConfigUpdate,
    ) -> Result<PublicApiStatus> {
        match self.send(
            AgentRequest::UpdatePublicApiConfig { update },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::PublicApiStatus { status } => Ok(status),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn create_instance(&self, params: CreateInstanceParams) -> Result<i64> {
        match self.send(
            AgentRequest::CreateInstance { params },
            REQUEST_TIMEOUT_STATE,
        )? {
            AgentResponse::CreatedInstance { instance_id } => Ok(instance_id),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn load_instance(&self, instance_id: i64) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::LoadInstance { instance_id },
            REQUEST_TIMEOUT_LOAD,
        )?)
    }

    pub fn unload_instance(&self, instance_id: i64) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::UnloadInstance { instance_id },
            REQUEST_TIMEOUT_STATE,
        )?)
    }

    pub fn remove_instance(&self, instance_id: i64) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::RemoveInstance { instance_id },
            REQUEST_TIMEOUT_STATE,
        )?)
    }

    pub fn set_retention_mode(
        &self,
        instance_id: i64,
        retention_mode: RetentionMode,
    ) -> Result<()> {
        self.expect_ok(self.send(
            AgentRequest::SetRetentionMode {
                instance_id,
                retention_mode,
            },
            REQUEST_TIMEOUT_STATE,
        )?)
    }

    pub fn chat_complete(&self, request: ChatRequest) -> Result<TextGenerationResult> {
        match self.send(AgentRequest::ChatComplete { request }, REQUEST_TIMEOUT_CHAT)? {
            AgentResponse::ChatResult { result } => Ok(result),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn vlm_complete(&self, request: VlmRequest) -> Result<TextGenerationResult> {
        match self.send(AgentRequest::VlmComplete { request }, REQUEST_TIMEOUT_CHAT)? {
            AgentResponse::VlmResult { result } => Ok(result),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn embeddings(&self, request: EmbeddingsRequest) -> Result<JsonResult> {
        match self.send(AgentRequest::Embeddings { request }, REQUEST_TIMEOUT_CHAT)? {
            AgentResponse::JsonResult { result } => Ok(result),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn rerank(&self, request: RerankRequest) -> Result<JsonResult> {
        match self.send(AgentRequest::Rerank { request }, REQUEST_TIMEOUT_CHAT)? {
            AgentResponse::JsonResult { result } => Ok(result),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn audio_transcriptions_raw(&self, request: AudioRawRequest) -> Result<JsonResult> {
        match self.send(
            AgentRequest::AudioTranscriptionsRaw { request },
            REQUEST_TIMEOUT_CHAT,
        )? {
            AgentResponse::JsonResult { result } => Ok(result),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    pub fn audio_transcriptions_native(
        &self,
        request: NativeAudioTranscriptionRequest,
    ) -> Result<JsonResult> {
        match self.send(
            AgentRequest::AudioTranscriptionsNative { request },
            REQUEST_TIMEOUT_CHAT,
        )? {
            AgentResponse::JsonResult { result } => Ok(result),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    fn expect_ok(&self, response: AgentResponse) -> Result<()> {
        match response {
            AgentResponse::Ok => Ok(()),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected response: {:?}", other),
        }
    }

    fn send(&self, request: AgentRequest, timeout: Duration) -> Result<AgentResponse> {
        send_request_once(&self.control_addr, &request, timeout)
    }
}

fn send_request_once(
    control_addr: &str,
    request: &AgentRequest,
    timeout: Duration,
) -> Result<AgentResponse> {
    let connect_timeout = timeout.min(Duration::from_secs(3));
    let resolved = control_addr
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve agent address '{}'", control_addr))?
        .collect::<Vec<_>>();
    if resolved.is_empty() {
        bail!("failed to resolve agent address '{}'", control_addr);
    }

    let mut last_error = None;
    let mut stream = None;
    for addr in resolved {
        match TcpStream::connect_timeout(&addr, connect_timeout) {
            Ok(value) => {
                stream = Some(value);
                break;
            }
            Err(err) => last_error = Some(err),
        }
    }
    let mut stream = stream.ok_or_else(|| {
        let detail = last_error
            .map(|err| err.to_string())
            .unwrap_or_else(|| "unknown connect failure".to_string());
        anyhow::anyhow!("failed to connect to agent '{}': {}", control_addr, detail)
    })?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    write_message(&mut stream, request)?;
    read_message(&mut stream)
}

pub fn transfer_model_artifact_between_agents<F>(
    source_control_addr: &str,
    dest_control_addr: &str,
    folder_name: &str,
    relative_path: &str,
    mut on_progress: F,
) -> Result<ModelArtifactTransferOutcome>
where
    F: FnMut(u64, u64),
{
    let mut source_stream = connect_agent_stream(source_control_addr, REQUEST_TIMEOUT_LOAD)?;
    write_message(
        &mut source_stream,
        &AgentRequest::StreamModelArtifact {
            folder_name: folder_name.to_string(),
            relative_path: relative_path.to_string(),
        },
    )?;
    let source_size = match read_message::<AgentResponse>(&mut source_stream)? {
        AgentResponse::ModelArtifactTransferReady { size_bytes } => size_bytes,
        AgentResponse::Error { message } => bail!(message),
        other => bail!("unexpected response: {:?}", other),
    };

    let mut dest_stream = connect_agent_stream(dest_control_addr, REQUEST_TIMEOUT_LOAD)?;
    write_message(
        &mut dest_stream,
        &AgentRequest::ReceiveModelArtifact {
            folder_name: folder_name.to_string(),
            relative_path: relative_path.to_string(),
            size_bytes: source_size,
        },
    )?;
    match read_message::<AgentResponse>(&mut dest_stream)? {
        AgentResponse::ModelArtifactTransferSkipped { .. } => Ok(ModelArtifactTransferOutcome {
            size_bytes: source_size,
            skipped: true,
        }),
        AgentResponse::ModelArtifactTransferReady { size_bytes } if size_bytes == source_size => {
            let mut copied = 0u64;
            let mut buffer = vec![0u8; LINK_BENCHMARK_CHUNK_BYTES];
            while copied < source_size {
                let max_read =
                    usize::try_from((source_size - copied).min(buffer.len() as u64)).unwrap_or(buffer.len());
                let read = source_stream
                    .read(&mut buffer[..max_read])
                    .context("failed to read source artifact bytes")?;
                if read == 0 {
                    bail!("source artifact stream ended unexpectedly");
                }
                dest_stream
                    .write_all(&buffer[..read])
                    .context("failed to write destination artifact bytes")?;
                copied += read as u64;
                on_progress(copied, source_size);
            }
            dest_stream.flush().ok();
            match read_message::<AgentResponse>(&mut dest_stream)? {
                AgentResponse::Ok => Ok(ModelArtifactTransferOutcome {
                    size_bytes: source_size,
                    skipped: false,
                }),
                AgentResponse::Error { message } => bail!(message),
                other => bail!("unexpected response: {:?}", other),
            }
        }
        AgentResponse::ModelArtifactTransferReady { size_bytes } => bail!(
            "artifact size mismatch during transfer handshake (source {source_size}, destination {size_bytes})"
        ),
        AgentResponse::Error { message } => bail!(message),
        other => bail!("unexpected response: {:?}", other),
    }
}

fn connect_agent_stream(control_addr: &str, timeout: Duration) -> Result<TcpStream> {
    let connect_timeout = timeout.min(Duration::from_secs(3));
    let resolved = control_addr
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve agent address '{}'", control_addr))?
        .collect::<Vec<_>>();
    if resolved.is_empty() {
        bail!("failed to resolve agent address '{}'", control_addr);
    }

    let mut last_error = None;
    for addr in resolved {
        match TcpStream::connect_timeout(&addr, connect_timeout) {
            Ok(mut stream) => {
                stream.set_read_timeout(Some(timeout)).ok();
                stream.set_write_timeout(Some(timeout)).ok();
                return Ok(stream);
            }
            Err(err) => last_error = Some(err),
        }
    }

    let detail = last_error
        .map(|err| err.to_string())
        .unwrap_or_else(|| "unknown connect failure".to_string());
    bail!("failed to connect to agent '{}': {}", control_addr, detail)
}

pub fn default_local_agent_addr() -> String {
    format!("127.0.0.1:{CLUSTER_AGENT_CONTROL_PORT}")
}

pub fn preferred_local_control_addr(control_addr: &str) -> String {
    let (host, port) = control_addr
        .rsplit_once(':')
        .unwrap_or(("127.0.0.1", "46211"));
    let host = host.trim();
    let auto_host = matches!(
        host,
        "" | "0.0.0.0" | "::" | "[::]" | "127.0.0.1" | "localhost" | "::1" | "[::1]"
    );
    if !auto_host && !saved_local_control_host_is_stale(host) {
        return control_addr.to_string();
    }
    preferred_paired_link_local_host(false)
        .or_else(preferred_direct_link_host)
        .or_else(|| {
            preferred_interface_candidates()
                .into_iter()
                .map(|candidate| candidate.ip.to_string())
                .next()
        })
        .or_else(|| default_route_local_network_host(false))
        .map(|preferred| format!("{preferred}:{port}"))
        .unwrap_or_else(|| control_addr.to_string())
}

fn saved_local_control_host_is_stale(host: &str) -> bool {
    let Ok(ipv4) = host.parse::<Ipv4Addr>() else {
        return false;
    };
    !preferred_interface_candidates()
        .iter()
        .any(|candidate| candidate.ip == ipv4)
}

fn default_public_api_bind_addr() -> String {
    format!("127.0.0.1:{CLUSTER_PUBLIC_API_PORT}")
}

fn default_public_api_config() -> PublicApiConfig {
    PublicApiConfig {
        enabled: false,
        bind_addr: default_public_api_bind_addr(),
        allow_cors: false,
        allowed_origins: Vec::new(),
        allowed_client_ips: Vec::new(),
        api_key: None,
    }
}

fn public_api_config_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join("cluster-public-api.json")
}

fn load_public_api_config(runtime_dir: &Path) -> PublicApiConfig {
    let path = public_api_config_path(runtime_dir);
    let Ok(text) = fs::read_to_string(&path) else {
        return default_public_api_config();
    };
    serde_json::from_str::<PublicApiConfig>(&text).unwrap_or_else(|_| default_public_api_config())
}

fn save_public_api_config(runtime_dir: &Path, config: &PublicApiConfig) -> Result<()> {
    let path = public_api_config_path(runtime_dir);
    let payload = serde_json::to_vec_pretty(config)?;
    fs::write(path, payload)?;
    Ok(())
}

fn public_api_fingerprint(value: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(value.as_bytes());
    digest[..4]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn public_api_status_for_state(state: &Arc<AgentRuntimeState>) -> PublicApiStatus {
    let guard = state.public_api.lock();
    if let Ok(public_api) = guard {
        PublicApiStatus {
            enabled: public_api.config.enabled,
            running: public_api.running,
            bind_addr: public_api.config.bind_addr.clone(),
            effective_bind_addr: public_api.bound_addr.clone(),
            advertised_addr: public_api.bound_addr.as_deref().and_then(|addr| {
                advertised_public_api_addr_for_bind(
                    Some(addr),
                    advertised_control_addr_for_bind(&state.bind_addr).as_deref(),
                )
            }),
            allow_cors: public_api.config.allow_cors,
            allowed_origins: public_api.config.allowed_origins.clone(),
            allowed_client_ips: public_api.config.allowed_client_ips.clone(),
            api_key_present: public_api.config.api_key.is_some(),
            api_key_fingerprint: public_api
                .config
                .api_key
                .as_deref()
                .map(public_api_fingerprint),
            last_error: public_api.last_error.clone(),
        }
    } else {
        PublicApiStatus {
            enabled: false,
            running: false,
            bind_addr: default_public_api_bind_addr(),
            effective_bind_addr: None,
            advertised_addr: None,
            allow_cors: false,
            allowed_origins: Vec::new(),
            allowed_client_ips: Vec::new(),
            api_key_present: false,
            api_key_fingerprint: None,
            last_error: Some("public API state lock poisoned".to_string()),
        }
    }
}

static LOCAL_AGENT_STARTING: OnceLock<Mutex<bool>> = OnceLock::new();
static LOCAL_AGENT_LAST_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();
static INTERFACE_CANDIDATE_CACHE: OnceLock<Mutex<Option<(Instant, Vec<InterfaceCandidate>)>>> =
    OnceLock::new();
static RPC_SERVER_REACHABILITY_CACHE: OnceLock<Mutex<Option<CachedRpcServerReachability>>> =
    OnceLock::new();
#[cfg(target_os = "windows")]
static FIREWALL_STATE_CACHE: OnceLock<Mutex<Option<CachedFirewallState>>> = OnceLock::new();
static HOST_NAME_CACHE: OnceLock<Option<String>> = OnceLock::new();

fn start_local_agent_thread(runtime_dir: PathBuf, control_addr: String) {
    let start_state = LOCAL_AGENT_STARTING.get_or_init(|| Mutex::new(false));
    let Ok(mut guard) = start_state.lock() else {
        return;
    };
    if *guard {
        return;
    }
    *guard = true;
    drop(guard);
    if let Ok(mut guard) = LOCAL_AGENT_LAST_ERROR
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
        *guard = None;
    }

    let bind_addr = bind_addr_from_local_control_addr(&control_addr);
    thread::spawn(move || {
        if let Err(err) = run_agent(runtime_dir, bind_addr) {
            if let Ok(mut guard) = LOCAL_AGENT_LAST_ERROR
                .get_or_init(|| Mutex::new(None))
                .lock()
            {
                *guard = Some(err.to_string());
            }
            eprintln!("embedded cluster agent failed: {err}");
        }
        if let Ok(mut guard) = LOCAL_AGENT_STARTING
            .get_or_init(|| Mutex::new(false))
            .lock()
        {
            *guard = false;
        }
    });
}

pub fn ensure_local_agent(runtime_dir: &Path, control_addr: &str) -> Result<AgentClient> {
    let client = AgentClient::new(control_addr);
    if client.ping().is_ok() {
        if let Err(err) = client.list_model_packages() {
            let detail = format!("{err:#}");
            if detail.contains("failed to read message length")
                || detail.contains("failed to decode message")
            {
                bail!(
                    "local control port {control_addr} is already serving an older controller build; quit every running controller instance and reconnect"
                );
            }
        }
        return Ok(client);
    }

    start_local_agent_thread(runtime_dir.to_path_buf(), control_addr.to_string());

    let start = Instant::now();
    while start.elapsed() < CONNECT_RETRY_TIMEOUT {
        if client.ping().is_ok() {
            return Ok(client);
        }
        thread::sleep(CONNECT_RETRY_INTERVAL);
    }

    if let Ok(guard) = LOCAL_AGENT_LAST_ERROR
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
        if let Some(err) = guard.as_deref() {
            bail!("local agent failed to start: {err}");
        }
    }

    bail!("local agent did not start at {control_addr}")
}

pub fn run_agent(runtime_dir: PathBuf, bind_addr: String) -> Result<()> {
    let api = ClusterApi::load(&runtime_dir)?;
    let api = Arc::new(RwLock::new(api));
    let models_dir = default_models_dir()?;
    let local_control_addr = local_control_addr_for_bind(&bind_addr);
    let public_api_config = load_public_api_config(&runtime_dir);
    let mut local_node = {
        let guard = api_read(&api)?;
        guard.get_local_node_info()?
    };
    apply_local_node_identity_override(&mut local_node);
    let paired_peers = settings::load_controller_settings_or_default().paired_peers;
    let discovery_mode = if paired_peers.is_empty() {
        DiscoveryMode::Pairing
    } else {
        DiscoveryMode::KnownPeers
    };
    let discovery_until = if paired_peers.is_empty() {
        None
    } else {
        Some(Instant::now() + STARTUP_KNOWN_DISCOVERY_DURATION)
    };
    let state = Arc::new(AgentRuntimeState {
        runtime_dir: runtime_dir.clone(),
        bind_addr: bind_addr.clone(),
        local_control_addr: local_control_addr.clone(),
        local_node: local_node.clone(),
        models_dir: models_dir.clone(),
        rpc_server: Mutex::new(None),
        schedule_lock: Mutex::new(()),
        public_api: Mutex::new(PublicApiRuntimeState {
            config: public_api_config,
            running: false,
            bound_addr: None,
            last_error: None,
            handle: None,
        }),
        link_metrics: Mutex::new(HashMap::new()),
        discovery: Mutex::new(DiscoveryRuntimeState {
            mode: discovery_mode,
            active_until: discovery_until,
        }),
        pairing_requests: Mutex::new(HashMap::new()),
        outgoing_pairing_requests: Mutex::new(HashMap::new()),
    });
    let peers: Arc<Mutex<HashMap<String, DiscoveredPeer>>> = Arc::new(Mutex::new(HashMap::new()));
    let telemetry_cache: Arc<Mutex<HashMap<String, TelemetryEntry>>> =
        Arc::new(Mutex::new(HashMap::new()));

    seed_manual_peers_from_env(&peers);
    load_persisted_paired_peers(&runtime_dir, &local_node.node_id, &peers);

    if let Err(err) = sync_public_api_server(&state) {
        eprintln!("cluster agent public API start failed: {err}");
    }

    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("failed to bind agent listener on '{bind_addr}'"))?;
    let discovery_socket =
        bind_cluster_udp_socket(CLUSTER_AGENT_DISCOVERY_PORT, "discovery")?;
    let telemetry_socket =
        bind_cluster_udp_socket(CLUSTER_AGENT_TELEMETRY_PORT, "telemetry")?;

    start_discovery_loop(
        local_node.clone(),
        bind_addr.clone(),
        peers.clone(),
        state.clone(),
        discovery_socket,
    );
    start_telemetry_loop(
        local_node.clone(),
        bind_addr.clone(),
        api.clone(),
        state.clone(),
        telemetry_cache.clone(),
        telemetry_socket,
    );
    start_link_benchmark_monitor(
        local_node.clone(),
        bind_addr.clone(),
        peers.clone(),
        state.clone(),
    );

    for accepted in listener.incoming() {
        match accepted {
            Ok(stream) => {
                let api = api.clone();
                let peers = peers.clone();
                let telemetry_cache = telemetry_cache.clone();
                let bind_addr = bind_addr.clone();
                let runtime_dir = runtime_dir.clone();
                let state = state.clone();
                thread::spawn(move || {
                    if let Err(err) = handle_connection(
                        stream,
                        api,
                        peers,
                        telemetry_cache,
                        &bind_addr,
                        &runtime_dir,
                        state,
                    ) {
                        eprintln!("cluster agent connection failed: {err}");
                    }
                });
            }
            Err(err) => {
                eprintln!("cluster agent accept failed: {err}");
                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    Ok(())
}

fn sync_public_api_server(state: &Arc<AgentRuntimeState>) -> Result<()> {
    let mut public_api = state
        .public_api
        .lock()
        .map_err(|_| anyhow::anyhow!("public API state lock poisoned"))?;

    if let Some(handle) = public_api.handle.take() {
        handle.shutdown();
    }
    public_api.running = false;
    public_api.bound_addr = None;

    if !public_api.config.enabled {
        public_api.last_error = None;
        return Ok(());
    }

    public_api.config.bind_addr = match normalize_public_api_bind_addr(&public_api.config.bind_addr)
    {
        Ok(bind_addr) => bind_addr,
        Err(err) => {
            public_api.last_error = Some(err.to_string());
            return Err(err);
        }
    };

    match start_public_server(
        public_api.config.clone(),
        // The managed HTTP server runs inside the same process as the local agent,
        // so it should always talk to the agent over loopback rather than a dynamic
        // advertised control address such as link-local Thunderbolt IPs.
        default_local_agent_addr(),
        state.models_dir.clone(),
    ) {
        Ok(handle) => {
            public_api.running = true;
            public_api.bound_addr = Some(handle.bound_addr.clone());
            public_api.last_error = None;
            public_api.handle = Some(handle);
            Ok(())
        }
        Err(err) => {
            public_api.running = false;
            public_api.bound_addr = None;
            public_api.last_error = Some(err.to_string());
            Err(err)
        }
    }
}

fn apply_local_node_identity_override(node: &mut crate::cluster_api::NodeInfo) {
    let Some(host_name) = best_local_host_name() else {
        return;
    };
    if !host_name.trim().is_empty() {
        node.display_name = host_name.clone();
        if node.node_id.trim().is_empty() || node.node_id.starts_with("local-node-") {
            node.node_id = format!("{}-{}-{}", host_name, node.os_name, node.arch);
        }
    }
}

fn best_local_host_name() -> Option<String> {
    HOST_NAME_CACHE
        .get_or_init(compute_best_local_host_name)
        .clone()
}

fn compute_best_local_host_name() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        for (program, args) in [
            ("scutil", &["--get", "LocalHostName"][..]),
            ("scutil", &["--get", "ComputerName"][..]),
            ("hostname", &[][..]),
        ] {
            let mut command = Command::new(program);
            command.args(args);
            configure_background_command(&mut command);
            if let Ok(output) = command.output() {
                if output.status.success() {
                    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !value.is_empty() {
                        return Some(value);
                    }
                }
            }
        }
    }

    for key in ["COMPUTERNAME", "HOSTNAME"] {
        if let Some(value) = env::var(key).ok().map(|value| value.trim().to_string()) {
            if !value.is_empty() {
                return Some(value);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut command = Command::new("hostname");
        configure_background_command(&mut command);
        if let Ok(output) = command.output() {
            if output.status.success() {
                let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
    }

    None
}

fn handle_connection(
    mut stream: TcpStream,
    api: SharedClusterApi,
    peers: Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    telemetry_cache: Arc<Mutex<HashMap<String, TelemetryEntry>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: Arc<AgentRuntimeState>,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(15))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(15))).ok();

    let request: AgentRequest = read_message(&mut stream)?;
    match request {
        AgentRequest::LinkProbe { bytes } => {
            let response = handle_link_probe_connection(&mut stream, bytes);
            write_agent_response(&mut stream, response)
        }
        AgentRequest::StreamModelArtifact {
            folder_name,
            relative_path,
        } => handle_stream_model_artifact_connection(
            &mut stream,
            &state.models_dir,
            &folder_name,
            &relative_path,
        ),
        AgentRequest::ReceiveModelArtifact {
            folder_name,
            relative_path,
            size_bytes,
        } => handle_receive_model_artifact_connection(
            &mut stream,
            &state.models_dir,
            &folder_name,
            &relative_path,
            size_bytes,
        ),
        other => {
            let response = match catch_unwind(AssertUnwindSafe(|| {
                handle_request(
                    other,
                    api,
                    peers,
                    telemetry_cache,
                    bind_addr,
                    runtime_dir,
                    state,
                )
            })) {
                Ok(response) => response,
                Err(payload) => {
                    let detail = panic_payload_message(payload);
                    eprintln!("cluster agent request panicked: {detail}");
                    AgentResponse::Error {
                        message: format!("cluster agent request panicked: {detail}"),
                    }
                }
            };
            write_agent_response(&mut stream, response)
        }
    }
}

fn write_agent_response(stream: &mut TcpStream, response: AgentResponse) -> Result<()> {
    match write_message(stream, &response) {
        Ok(()) => Ok(()),
        Err(err) => {
            let detail = format!("{err:#}");
            eprintln!("cluster agent failed to send response: {detail}");
            write_message(
                stream,
                &AgentResponse::Error {
                    message: format!("cluster agent failed to send response: {detail}"),
                },
            )
            .context("failed to send fallback agent error response")
        }
    }
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic".to_string()
}

fn handle_link_probe_connection(stream: &mut TcpStream, bytes: u64) -> AgentResponse {
    let mut remaining = bytes;
    let mut buffer = vec![0u8; LINK_BENCHMARK_CHUNK_BYTES];
    while remaining > 0 {
        let to_read = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        if let Err(err) = stream.read_exact(&mut buffer[..to_read]) {
            return AgentResponse::Error {
                message: format!("failed to read link probe payload: {err}"),
            };
        }
        remaining -= to_read as u64;
    }
    AgentResponse::LinkProbeAck { bytes, checksum: 0 }
}

fn handle_stream_model_artifact_connection(
    stream: &mut TcpStream,
    models_dir: &Path,
    folder_name: &str,
    relative_path: &str,
) -> Result<()> {
    let path = resolve_model_artifact_path(models_dir, folder_name, relative_path)?;
    let metadata =
        fs::metadata(&path).with_context(|| format!("failed to stat '{}'", path.display()))?;
    if !metadata.is_file() {
        bail!("'{}' is not a file", path.display());
    }
    let size_bytes = metadata.len();
    write_message(
        stream,
        &AgentResponse::ModelArtifactTransferReady { size_bytes },
    )?;
    let mut file =
        fs::File::open(&path).with_context(|| format!("failed to open '{}'", path.display()))?;
    let mut buffer = vec![0u8; LINK_BENCHMARK_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        stream
            .write_all(&buffer[..read])
            .with_context(|| format!("failed to stream '{}'", path.display()))?;
    }
    stream.flush().ok();
    Ok(())
}

fn handle_receive_model_artifact_connection(
    stream: &mut TcpStream,
    models_dir: &Path,
    folder_name: &str,
    relative_path: &str,
    size_bytes: u64,
) -> Result<()> {
    let destination = resolve_model_artifact_path(models_dir, folder_name, relative_path)?;
    if destination.exists() {
        let _ = touch_model_store_change_marker(models_dir);
        write_message(
            stream,
            &AgentResponse::ModelArtifactTransferSkipped {
                reason: "already present".to_string(),
            },
        )?;
        return Ok(());
    }

    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid destination path"))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create '{}'", parent.display()))?;
    write_message(
        stream,
        &AgentResponse::ModelArtifactTransferReady { size_bytes },
    )?;

    let mut part_name = destination
        .file_name()
        .map(|value| value.to_os_string())
        .ok_or_else(|| anyhow::anyhow!("invalid destination file name"))?;
    part_name.push(".part");
    let part_path = destination.with_file_name(part_name);
    let result = (|| -> Result<()> {
        let mut file = fs::File::create(&part_path)
            .with_context(|| format!("failed to create '{}'", part_path.display()))?;
        let mut remaining = size_bytes;
        let mut buffer = vec![0u8; LINK_BENCHMARK_CHUNK_BYTES];
        while remaining > 0 {
            let chunk_len =
                usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
            stream
                .read_exact(&mut buffer[..chunk_len])
                .with_context(|| {
                    format!(
                        "failed to read upload payload for '{}'",
                        destination.display()
                    )
                })?;
            file.write_all(&buffer[..chunk_len])
                .with_context(|| format!("failed to write '{}' chunk", destination.display()))?;
            remaining -= chunk_len as u64;
        }
        file.flush().ok();
        drop(file);
        fs::rename(&part_path, &destination).with_context(|| {
            format!(
                "failed to finalize uploaded artifact '{}' -> '{}'",
                part_path.display(),
                destination.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part_path);
    }
    result?;
    let _ = touch_model_store_change_marker(models_dir);
    write_message(stream, &AgentResponse::Ok)?;
    Ok(())
}

fn handle_request(
    request: AgentRequest,
    api: SharedClusterApi,
    peers: Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    telemetry_cache: Arc<Mutex<HashMap<String, TelemetryEntry>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: Arc<AgentRuntimeState>,
) -> AgentResponse {
    let result = || -> Result<AgentResponse> {
        match request {
            AgentRequest::Ping => Ok(AgentResponse::Pong),
            AgentRequest::GetSnapshot { rpc_servers } => {
                let snapshot = snapshot_local_state(
                    &api,
                    bind_addr,
                    runtime_dir,
                    &state,
                    rpc_servers.as_deref(),
                )?;
                Ok(AgentResponse::Snapshot { snapshot })
            }
            AgentRequest::GetLocalTelemetry => Ok(AgentResponse::LocalTelemetry {
                snapshot: build_local_telemetry(&api, bind_addr, &state)?,
            }),
            AgentRequest::GetClusterTelemetry => Ok(AgentResponse::ClusterTelemetry {
                snapshots: collect_cluster_telemetry(
                    &api,
                    &peers,
                    &telemetry_cache,
                    bind_addr,
                    &state,
                )?,
            }),
            AgentRequest::RunLinkBenchmarks { full } => {
                run_link_benchmarks_for_state(bind_addr, &peers, &state, full)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::MeasureLinkTo { control_addr, full } => {
                let metrics = measure_link_metrics(&control_addr, full)?;
                Ok(AgentResponse::LinkMetrics { metrics })
            }
            AgentRequest::ListPeers => {
                let peers = list_peers(&peers);
                Ok(AgentResponse::Peers { peers })
            }
            AgentRequest::AddPeer { control_addr } => {
                upsert_manual_peer(&peers, &control_addr)?;
                persist_paired_peers(runtime_dir, &state.local_node.node_id, &peers)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::RemovePeer { control_addr } => {
                remove_peer_by_addr(&peers, &control_addr);
                persist_paired_peers(runtime_dir, &state.local_node.node_id, &peers)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::StartDiscovery { mode, seconds } => {
                let status = start_discovery_session(&state, mode, seconds);
                Ok(AgentResponse::DiscoveryStatus { status })
            }
            AgentRequest::GetDiscoveryStatus => Ok(AgentResponse::DiscoveryStatus {
                status: discovery_status(&state),
            }),
            AgentRequest::ListPairingRequests => Ok(AgentResponse::PairingRequests {
                requests: list_pairing_requests(&state),
            }),
            AgentRequest::RequestPairing { control_addr } => {
                request_pairing_with_peer(&state, &control_addr)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::AcceptPairingRequest { request_id } => {
                accept_pairing_request(&state, &peers, runtime_dir, &request_id)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::DeclinePairingRequest { request_id } => {
                decline_pairing_request(&state, &request_id);
                Ok(AgentResponse::Ok)
            }
            AgentRequest::SubmitPairingRequest { request } => {
                receive_pairing_request(&state, &peers, request);
                Ok(AgentResponse::Ok)
            }
            AgentRequest::FinalizePairing {
                request_id,
                peer,
                shared_token,
            } => {
                finalize_pairing_request(
                    &state,
                    &peers,
                    runtime_dir,
                    &request_id,
                    peer,
                    &shared_token,
                )?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::RestartRpcServer => {
                ensure_rpc_server_running(&state)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::ConfigureFirewall => {
                configure_local_firewall(&state.runtime_dir)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::StatPaths { paths } => Ok(AgentResponse::PathStats {
                stats: stat_paths(paths),
            }),
            AgentRequest::PlanInstance {
                params,
                allowed_control_addrs,
            } => {
                let plan = plan_instance_for_cluster(
                    &api,
                    &peers,
                    bind_addr,
                    runtime_dir,
                    &state,
                    params,
                    allowed_control_addrs,
                )?;
                Ok(AgentResponse::PlacementPlan { plan })
            }
            AgentRequest::ListPlacementCandidates {
                params,
                allowed_control_addrs,
            } => {
                let plans = build_placement_candidates(
                    &api,
                    &peers,
                    bind_addr,
                    runtime_dir,
                    &state,
                    &params,
                    allowed_control_addrs.as_deref(),
                )?
                .into_iter()
                .map(|candidate| candidate.plan)
                .collect();
                Ok(AgentResponse::PlacementCandidates { plans })
            }
            AgentRequest::ScheduleInstance {
                params,
                allowed_control_addrs,
                load_immediately,
            } => {
                let scheduled = schedule_instance_for_cluster(
                    &api,
                    &peers,
                    bind_addr,
                    runtime_dir,
                    &state,
                    params,
                    allowed_control_addrs,
                    load_immediately,
                )?;
                Ok(AgentResponse::ScheduledInstance { scheduled })
            }
            AgentRequest::ResolveClusterInstance {
                name,
                load_if_managed,
            } => {
                let resolved = resolve_cluster_instance(
                    &api,
                    &peers,
                    bind_addr,
                    runtime_dir,
                    &state,
                    &name,
                    load_if_managed,
                )?;
                Ok(AgentResponse::ResolvedClusterInstance { resolved })
            }
            AgentRequest::ListManagedModels => Ok(AgentResponse::ManagedModels {
                models: list_local_managed_models(&state)?,
            }),
            AgentRequest::ListClusterManagedModels => Ok(AgentResponse::ManagedModels {
                models: list_cluster_managed_models(&api, &peers, bind_addr, runtime_dir, &state)?,
            }),
            AgentRequest::ResolveManagedModel { model_id } => {
                let model = find_model_entry(&state.models_dir, &model_id)?;
                Ok(AgentResponse::ManagedModel { model })
            }
            AgentRequest::ResolveClusterManagedModel { model_id } => {
                let model = resolve_cluster_managed_model(
                    &api,
                    &peers,
                    bind_addr,
                    runtime_dir,
                    &state,
                    &model_id,
                )?;
                Ok(AgentResponse::ManagedModel { model })
            }
            AgentRequest::ListModelPackages => Ok(AgentResponse::ModelPackages {
                packages: list_local_model_packages(&state)?,
            }),
            AgentRequest::ListClusterModelPackages => Ok(AgentResponse::ClusterModelPackages {
                packages: list_cluster_model_packages(bind_addr, &peers, &state),
            }),
            AgentRequest::GetPublicApiStatus => Ok(AgentResponse::PublicApiStatus {
                status: public_api_status_for_state(&state),
            }),
            AgentRequest::UpdatePublicApiConfig { update } => {
                {
                    let mut public_api = state
                        .public_api
                        .lock()
                        .map_err(|_| anyhow::anyhow!("public API state lock poisoned"))?;
                    public_api.config.enabled = update.enabled;
                    public_api.config.bind_addr =
                        normalize_public_api_bind_addr(&update.bind_addr)?;
                    public_api.config.allow_cors = update.allow_cors;
                    public_api.config.allowed_origins = update
                        .allowed_origins
                        .into_iter()
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty())
                        .collect();
                    public_api.config.allowed_client_ips = update
                        .allowed_client_ips
                        .into_iter()
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty())
                        .collect();
                    if update.clear_api_key {
                        public_api.config.api_key = None;
                    } else if let Some(api_key) = update
                        .api_key
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty())
                    {
                        public_api.config.api_key = Some(api_key);
                    }
                    save_public_api_config(runtime_dir, &public_api.config)?;
                }
                sync_public_api_server(&state)?;
                Ok(AgentResponse::PublicApiStatus {
                    status: public_api_status_for_state(&state),
                })
            }
            AgentRequest::CreateInstance { params } => {
                let guard = api_write(&api)?;
                let instance_id = guard.create_instance(&params)?;
                Ok(AgentResponse::CreatedInstance { instance_id })
            }
            AgentRequest::LoadInstance { instance_id } => {
                let rpc_servers = {
                    let guard = api_read(&api)?;
                    let instance = guard
                        .list_instances()?
                        .into_iter()
                        .find(|instance| instance.instance_id == instance_id)
                        .ok_or_else(|| anyhow::anyhow!("unknown instance_id"))?;
                    instance.rpc_servers
                };
                restart_matching_peer_rpc_servers(&peers, &rpc_servers)?;
                let guard = api_write(&api)?;
                guard.load_instance(instance_id)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::UnloadInstance { instance_id } => {
                let guard = api_write(&api)?;
                guard.unload_instance(instance_id)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::RemoveInstance { instance_id } => {
                let guard = api_write(&api)?;
                guard.remove_instance(instance_id)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::SetRetentionMode {
                instance_id,
                retention_mode,
            } => {
                let guard = api_write(&api)?;
                guard.set_retention_mode(instance_id, retention_mode)?;
                Ok(AgentResponse::Ok)
            }
            AgentRequest::ChatComplete { request } => {
                let guard = api_read(&api)?;
                let result = guard.chat_complete(&request)?;
                Ok(AgentResponse::ChatResult { result })
            }
            AgentRequest::VlmComplete { request } => {
                let guard = api_read(&api)?;
                let result = guard.vlm_complete(&request)?;
                Ok(AgentResponse::VlmResult { result })
            }
            AgentRequest::Embeddings { request } => {
                let guard = api_read(&api)?;
                let result = guard.embeddings(&request)?;
                Ok(AgentResponse::JsonResult { result })
            }
            AgentRequest::Rerank { request } => {
                let guard = api_read(&api)?;
                let result = guard.rerank(&request)?;
                Ok(AgentResponse::JsonResult { result })
            }
            AgentRequest::AudioTranscriptionsRaw { request } => {
                let guard = api_read(&api)?;
                let result = guard.audio_transcriptions_raw(&request)?;
                Ok(AgentResponse::JsonResult { result })
            }
            AgentRequest::AudioTranscriptionsNative { request } => {
                let guard = api_read(&api)?;
                let result = guard.audio_transcriptions_native(&request)?;
                Ok(AgentResponse::JsonResult { result })
            }
            AgentRequest::LinkProbe { .. } => {
                bail!("link probe must be handled at the socket layer")
            }
            AgentRequest::StreamModelArtifact { .. } => {
                bail!("model artifact streaming must be handled at the socket layer")
            }
            AgentRequest::ReceiveModelArtifact { .. } => {
                bail!("model artifact upload must be handled at the socket layer")
            }
        }
    }();

    match result {
        Ok(response) => response,
        Err(err) => AgentResponse::Error {
            message: format!("{err:#}"),
        },
    }
}

fn snapshot_local_state(
    api: &SharedClusterApi,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    rpc_servers: Option<&str>,
) -> Result<NodeSnapshot> {
    let (mut node, instances) = {
        let guard = api_read(api)?;
        let mut node = guard.get_local_node_info()?;
        apply_local_node_identity_override(&mut node);
        let instances = guard.list_instances()?;
        (node, instances)
    };
    let (devices, execution_groups) =
        query_local_devices_and_groups(api, runtime_dir, rpc_servers)?;
    let (rpc_endpoint, rpc_running) = rpc_server_snapshot(state);
    let known_control_addrs = advertised_control_addrs_for_bind(bind_addr);
    let advertised_control_addr = known_control_addrs.first().cloned();
    if interface_debug_enabled() {
        eprintln!(
            "snapshot_local_state bind_addr={bind_addr} advertised_control_addr={advertised_control_addr:?}"
        );
    }
    let advertised_rpc_endpoint = advertised_rpc_endpoint_for_bind(
        rpc_endpoint.as_deref(),
        advertised_control_addr.as_deref(),
    );
    let public_api_status = public_api_status_for_state(state);
    let public_api_addr = if public_api_status.enabled {
        Some(public_api_status.bind_addr.clone())
    } else {
        None
    };
    let advertised_public_api_addr = public_api_status.advertised_addr.clone();
    let firewall = firewall_state_for_runtime(runtime_dir);
    let mut link_metrics: Vec<_> = state
        .link_metrics
        .lock()
        .map(|value| value.values().cloned().collect())
        .unwrap_or_default();
    link_metrics.sort_by(|lhs, rhs| lhs.peer_control_addr.cmp(&rhs.peer_control_addr));
    let mut snapshot = NodeSnapshot {
        node,
        control_addr: state.local_control_addr.clone(),
        advertised_control_addr,
        known_control_addrs,
        runtime_dir: runtime_dir.display().to_string(),
        models_dir: state.models_dir.display().to_string(),
        rpc_endpoint,
        advertised_rpc_endpoint,
        rpc_running,
        public_api_addr,
        advertised_public_api_addr,
        public_api_running: public_api_status.running,
        firewall_status: firewall.status,
        firewall_action_required: firewall.action_required,
        devices,
        execution_groups,
        instances,
        link_metrics,
    };
    sanitize_snapshot_memory(&mut snapshot);
    Ok(snapshot)
}

fn query_local_devices_and_groups(
    api: &SharedClusterApi,
    runtime_dir: &Path,
    rpc_servers: Option<&str>,
) -> Result<(
    Vec<crate::cluster_api::DeviceInfo>,
    Vec<crate::cluster_api::ExecutionGroupInfo>,
)> {
    let rpc_servers = rpc_servers.map(str::trim).filter(|value| !value.is_empty());
    if let Some(rpc_servers) = rpc_servers {
        // Keep RPC preview state isolated from the shared long-lived cluster handle so
        // background telemetry/snapshot refreshes don't inherit dead remote RPC backends.
        let preview_api = ClusterApi::load(runtime_dir)
            .context("failed to load isolated cluster api for rpc preview")?;
        let execution_groups = preview_api.list_execution_groups()?;
        let reachable_rpc_servers = reachable_rpc_preview_servers(rpc_servers);
        if reachable_rpc_servers.is_empty() {
            let devices = preview_api.list_devices()?;
            return Ok((devices, execution_groups));
        }
        let reachable_rpc_servers_csv = reachable_rpc_servers.join(",");
        let devices = preview_api.list_devices_with_rpc(Some(&reachable_rpc_servers_csv))?;
        let execution_groups =
            synthesize_preview_execution_groups_with_rpc(&devices, execution_groups);
        return Ok((devices, execution_groups));
    }

    let guard = api_read(api)?;
    let devices = guard.list_devices()?;
    let execution_groups = guard.list_execution_groups()?;
    Ok((devices, execution_groups))
}

fn reachable_rpc_preview_servers(rpc_servers: &str) -> Vec<String> {
    let mut reachable = Vec::new();
    for endpoint in split_csv(rpc_servers) {
        if rpc_endpoint_is_reachable(&endpoint) {
            reachable.push(endpoint);
        } else {
            eprintln!("cluster rpc preview skipping unreachable endpoint '{endpoint}'");
        }
    }
    reachable
}

fn synthesize_preview_execution_groups_with_rpc(
    devices: &[crate::cluster_api::DeviceInfo],
    mut execution_groups: Vec<crate::cluster_api::ExecutionGroupInfo>,
) -> Vec<crate::cluster_api::ExecutionGroupInfo> {
    let rpc_devices = devices
        .iter()
        .filter(|device| {
            is_rpc_backend_name(&device.backend) || is_rpc_backend_name(&device.name)
        })
        .collect::<Vec<_>>();
    if rpc_devices.is_empty() {
        return execution_groups;
    }

    let rpc_indices = rpc_devices
        .iter()
        .map(|device| device.bridge_device_index)
        .collect::<Vec<_>>();
    let rpc_memory_free = rpc_devices
        .iter()
        .fold(0u64, |total, device| total.saturating_add(device.memory_free));
    let rpc_memory_total = rpc_devices
        .iter()
        .fold(0u64, |total, device| total.saturating_add(device.memory_total));

    for group in &mut execution_groups {
        if group.id == "cluster:auto" {
            continue;
        }

        let mut merged_indices = group
            .devices_csv
            .split(',')
            .map(|part| part.trim())
            .filter(|part| !part.is_empty())
            .filter_map(|part| part.parse::<i32>().ok())
            .collect::<Vec<_>>();
        for rpc_index in &rpc_indices {
            if !merged_indices.contains(rpc_index) {
                merged_indices.push(*rpc_index);
            }
        }
        if merged_indices.is_empty() {
            continue;
        }

        group.devices_csv = merged_indices
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        group.device_count = i32::try_from(merged_indices.len()).unwrap_or(i32::MAX);
        group.uses_local_split = group.uses_local_split || merged_indices.len() > 1;
        group.memory_free = group.memory_free.saturating_add(rpc_memory_free);
        group.memory_total = group.memory_total.saturating_add(rpc_memory_total);
        if !group.backend_summary.to_ascii_lowercase().contains("rpc") {
            group.backend_summary = if group.backend_summary.trim().is_empty() {
                "RPC".to_string()
            } else {
                format!("{} + RPC", group.backend_summary)
            };
        }
        let lowered_label = group.label.to_ascii_lowercase();
        if !lowered_label.contains("remote") && !lowered_label.contains("rpc") {
            group.label = if group.label.trim().is_empty() {
                "Remote".to_string()
            } else {
                format!("{} + remote", group.label)
            };
        }
    }

    execution_groups
}

fn build_local_telemetry(
    api: &SharedClusterApi,
    bind_addr: &str,
    state: &Arc<AgentRuntimeState>,
) -> Result<TelemetrySnapshot> {
    let guard = api_read(api)?;
    let mut devices = guard.list_devices()?;
    for device in &mut devices {
        if device.memory_free > device.memory_total {
            device.memory_free = device.memory_total;
        }
    }
    let instances = guard.list_instances()?;
    let mut node = guard.get_local_node_info()?;
    apply_local_node_identity_override(&mut node);
    drop(guard);

    let (rpc_endpoint, rpc_running) = rpc_server_snapshot(state);
    let known_control_addrs = advertised_control_addrs_for_bind(bind_addr);
    let advertised_control_addr = known_control_addrs.first().cloned();
    let _advertised_rpc_endpoint = advertised_rpc_endpoint_for_bind(
        rpc_endpoint.as_deref(),
        advertised_control_addr.as_deref(),
    );

    let mut system = System::new();
    system.refresh_memory();

    let mut process_memory_bytes = 0u64;
    let mut process_virtual_memory_bytes = 0u64;
    let mut process_cpu_percent = 0.0f32;
    if let Ok(pid) = sysinfo::get_current_pid() {
        let _ = system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        if let Some(process) = system.process(pid) {
            process_memory_bytes = process.memory();
            process_virtual_memory_bytes = process.virtual_memory();
            process_cpu_percent = process.cpu_usage();
        }
    }

    let mut link_metrics: Vec<_> = state
        .link_metrics
        .lock()
        .map(|value| value.values().cloned().collect())
        .unwrap_or_default();
    link_metrics.sort_by(|lhs, rhs| lhs.peer_control_addr.cmp(&rhs.peer_control_addr));

    let mut snapshot = TelemetrySnapshot {
        node,
        control_addr: state.local_control_addr.clone(),
        advertised_control_addr,
        known_control_addrs,
        unix_ms: unix_ms_now(),
        process_memory_bytes,
        process_virtual_memory_bytes,
        process_cpu_percent,
        system_memory_total_bytes: system.total_memory(),
        system_memory_available_bytes: system.available_memory(),
        rpc_running,
        public_api_running: public_api_status_for_state(state).running,
        devices,
        instances,
        link_metrics,
    };
    sanitize_telemetry_snapshot(&mut snapshot);
    Ok(snapshot)
}

fn collect_cluster_telemetry(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    telemetry_cache: &Arc<Mutex<HashMap<String, TelemetryEntry>>>,
    bind_addr: &str,
    state: &Arc<AgentRuntimeState>,
) -> Result<Vec<TelemetrySnapshot>> {
    let mut out = vec![build_local_telemetry(api, bind_addr, state)?];
    let now = Instant::now();
    let local_control_addr = local_control_addr_for_bind(bind_addr);
    let mut cached_by_addr = HashMap::new();
    if let Ok(mut guard) = telemetry_cache.lock() {
        guard.retain(|_, entry| now.duration_since(entry.last_seen) <= TELEMETRY_TTL);
        for entry in guard.values() {
            for control_addr in telemetry_control_addr_candidates(&entry.snapshot) {
                cached_by_addr.insert(control_addr.clone(), entry.snapshot.clone());
            }
            out.push(entry.snapshot.clone());
        }
    }

    let mut seen_remote = HashSet::new();
    for peer in list_peers(peers) {
        let candidates = peer_control_addr_candidates(&peer);
        let connect_addr =
            preferred_control_addr_from_candidates(&candidates, &peer.control_addr);
        if connect_addr == local_control_addr || !seen_remote.insert(connect_addr.clone()) {
            continue;
        }
        if candidates.iter().any(|value| cached_by_addr.contains_key(value)) {
            continue;
        }

        let mut telemetry_result = None;
        for candidate in &candidates {
            match AgentClient::new(candidate.clone())
                .get_local_telemetry_with_timeout(TELEMETRY_DIRECT_QUERY_TIMEOUT)
            {
                Ok(snapshot) => {
                    telemetry_result = Some((candidate.clone(), snapshot));
                    break;
                }
                Err(err) => {
                    eprintln!("cluster telemetry fallback from '{candidate}' failed: {err}");
                }
            }
        }
        if let Some((resolved_addr, mut snapshot)) = telemetry_result {
            snapshot.control_addr = resolved_addr.clone();
            snapshot.known_control_addrs = dedup_sorted_control_addrs(
                std::iter::once(resolved_addr)
                    .chain(snapshot.known_control_addrs.into_iter())
                    .chain(candidates.into_iter())
                    .collect(),
            );
            if snapshot.advertised_control_addr.is_none() {
                snapshot.advertised_control_addr = snapshot.known_control_addrs.first().cloned();
            }
            sanitize_telemetry_snapshot(&mut snapshot);
            if let Ok(mut guard) = telemetry_cache.lock() {
                guard.insert(
                    snapshot.control_addr.clone(),
                    TelemetryEntry {
                        snapshot: snapshot.clone(),
                        last_seen: now,
                    },
                );
                guard.retain(|_, entry| now.duration_since(entry.last_seen) <= TELEMETRY_TTL);
            }
            out.push(snapshot);
        }
    }
    out.sort_by(|lhs, rhs| {
        lhs.node
            .display_name
            .cmp(&rhs.node.display_name)
            .then(lhs.control_addr.cmp(&rhs.control_addr))
    });
    out.dedup_by(|lhs, rhs| lhs.control_addr == rhs.control_addr);
    Ok(out)
}

fn preferred_peer_control_addr(peer: &PeerInfo) -> String {
    preferred_control_addr_from_candidates(&peer_control_addr_candidates(peer), &peer.control_addr)
}

fn peer_info_preference_score(peer: &PeerInfo) -> i32 {
    let base = preferred_peer_control_addr(peer);
    host_preference_score(&addr_host(&base)) * 100
        + if peer.rpc_running { 10 } else { 0 }
        + if peer.trusted { 5 } else { 0 }
}

fn better_control_addr<'a>(lhs: Option<&'a str>, rhs: Option<&'a str>) -> Option<String> {
    match (lhs, rhs) {
        (Some(left), Some(right)) => {
            let left_score = host_preference_score(&addr_host(left));
            let right_score = host_preference_score(&addr_host(right));
            if left_score >= right_score {
                Some(left.to_string())
            } else {
                Some(right.to_string())
            }
        }
        (Some(left), None) => Some(left.to_string()),
        (None, Some(right)) => Some(right.to_string()),
        (None, None) => None,
    }
}

fn dedup_sorted_control_addrs(addrs: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for addr in addrs {
        let trimmed = addr.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.to_string();
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out.sort_by(|lhs, rhs| {
        host_preference_score(&addr_host(rhs))
            .cmp(&host_preference_score(&addr_host(lhs)))
            .then(lhs.cmp(rhs))
    });
    out
}

fn preferred_control_addr_from_candidates(candidates: &[String], fallback: &str) -> String {
    candidates
        .first()
        .cloned()
        .unwrap_or_else(|| fallback.to_string())
}

fn control_addr_candidates_overlap(lhs: &[String], rhs: &[String]) -> bool {
    lhs.iter().any(|candidate| rhs.iter().any(|other| other == candidate))
}

fn peer_control_addr_candidates(peer: &PeerInfo) -> Vec<String> {
    let mut addrs = vec![peer.control_addr.clone()];
    if let Some(advertised) = peer.advertised_control_addr.clone() {
        addrs.push(advertised);
    }
    addrs.extend(peer.known_control_addrs.iter().cloned());
    dedup_sorted_control_addrs(addrs)
}

fn snapshot_control_addr_candidates(snapshot: &NodeSnapshot) -> Vec<String> {
    let mut addrs = vec![snapshot.control_addr.clone()];
    if let Some(advertised) = snapshot.advertised_control_addr.clone() {
        addrs.push(advertised);
    }
    addrs.extend(snapshot.known_control_addrs.iter().cloned());
    dedup_sorted_control_addrs(addrs)
}

fn telemetry_control_addr_candidates(snapshot: &TelemetrySnapshot) -> Vec<String> {
    let mut addrs = vec![snapshot.control_addr.clone()];
    if let Some(advertised) = snapshot.advertised_control_addr.clone() {
        addrs.push(advertised);
    }
    addrs.extend(snapshot.known_control_addrs.iter().cloned());
    dedup_sorted_control_addrs(addrs)
}

fn discovery_control_addr_candidates(
    announcement: &DiscoveryAnnouncement,
    sender_control_addr: &str,
) -> Vec<String> {
    let mut addrs = vec![sender_control_addr.to_string()];
    if let Some(advertised) = announcement.advertised_control_addr.clone() {
        addrs.push(advertised);
    }
    addrs.extend(announcement.known_control_addrs.iter().cloned());
    dedup_sorted_control_addrs(addrs)
}

fn host_preference_score(host: &str) -> i32 {
    let Ok(ip) = host.parse::<Ipv4Addr>() else {
        return 0;
    };
    if ip.is_loopback() || ip.is_unspecified() {
        return -10_000;
    }
    if is_link_local_ipv4(ip) {
        return 4_000;
    }
    if is_private_ipv4(ip) {
        return 3_000;
    }
    1_000
}

fn is_private_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 10
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 168)
}

fn list_peers(peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>) -> Vec<PeerInfo> {
    let mut guard = match peers.lock() {
        Ok(guard) => guard,
        Err(_) => return Vec::new(),
    };

    let now = Instant::now();
    guard.retain(|_, peer| peer.manual || now.duration_since(peer.last_seen) <= PEER_TTL);

    let mut best_by_key: HashMap<String, PeerInfo> = HashMap::new();
    for peer in guard.values() {
        let mut info = peer.info.clone();
        let candidates = peer_control_addr_candidates(&info);
        let preferred_control = preferred_control_addr_from_candidates(&candidates, &info.control_addr);
        info.control_addr = preferred_control.clone();
        info.known_control_addrs = candidates;
        let key = if info.node_id.trim().is_empty() {
            info.control_addr.clone()
        } else {
            info.node_id.clone()
        };
        match best_by_key.get_mut(&key) {
            Some(existing) => {
                if peer_info_preference_score(&info) > peer_info_preference_score(existing) {
                    *existing = info;
                }
            }
            None => {
                best_by_key.insert(key, info);
            }
        }
    }

    let mut out: Vec<_> = best_by_key.into_values().collect();
    out.sort_by(|a, b| {
        a.display_name
            .cmp(&b.display_name)
            .then(a.control_addr.cmp(&b.control_addr))
    });
    out
}

fn local_control_addr_for_bind(bind_addr: &str) -> String {
    preferred_local_control_addr(bind_addr)
}

fn align_snapshot_to_connect_addr(snapshot: &mut NodeSnapshot, connect_addr: &str) {
    let connect_host = addr_host(connect_addr);
    let advertised_host = snapshot
        .advertised_control_addr
        .as_deref()
        .map(addr_host)
        .unwrap_or_default();
    if host_preference_score(&connect_host) < host_preference_score(&advertised_host) {
        return;
    }

    snapshot.control_addr = connect_addr.to_string();
    snapshot.advertised_control_addr = Some(connect_addr.to_string());
    snapshot.known_control_addrs = dedup_sorted_control_addrs(
        std::iter::once(connect_addr.to_string())
            .chain(snapshot.known_control_addrs.iter().cloned())
            .collect(),
    );

    if let Some(rpc_endpoint) = snapshot
        .advertised_rpc_endpoint
        .as_deref()
        .or(snapshot.rpc_endpoint.as_deref())
        .and_then(|value| {
            value
                .rsplit_once(':')
                .map(|(_, port)| format!("{connect_host}:{port}"))
        })
    {
        snapshot.advertised_rpc_endpoint = Some(rpc_endpoint);
    }

    if let Some(public_api_addr) = snapshot
        .advertised_public_api_addr
        .as_deref()
        .or(snapshot.public_api_addr.as_deref())
        .and_then(|value| {
            value
                .rsplit_once(':')
                .map(|(_, port)| format!("{connect_host}:{port}"))
        })
    {
        snapshot.advertised_public_api_addr = Some(public_api_addr);
    }
}

fn align_telemetry_to_connect_addr(snapshot: &mut TelemetrySnapshot, connect_addr: &str) {
    let connect_host = addr_host(connect_addr);
    let advertised_host = snapshot
        .advertised_control_addr
        .as_deref()
        .map(addr_host)
        .unwrap_or_default();
    if host_preference_score(&connect_host) < host_preference_score(&advertised_host) {
        return;
    }
    snapshot.control_addr = connect_addr.to_string();
    snapshot.advertised_control_addr = Some(connect_addr.to_string());
}

fn sanitize_snapshot_memory(snapshot: &mut NodeSnapshot) {
    for device in &mut snapshot.devices {
        if device.memory_free > device.memory_total {
            device.memory_free = device.memory_total;
        }
    }
    for group in &mut snapshot.execution_groups {
        if group.memory_free > group.memory_total {
            group.memory_free = group.memory_total;
        }
    }
    for metrics in &mut snapshot.link_metrics {
        sanitize_link_metrics(metrics);
    }
}

fn sanitize_telemetry_snapshot(snapshot: &mut TelemetrySnapshot) {
    if !snapshot.process_cpu_percent.is_finite() {
        snapshot.process_cpu_percent = 0.0;
    }
    for device in &mut snapshot.devices {
        if device.memory_free > device.memory_total {
            device.memory_free = device.memory_total;
        }
    }
    for metrics in &mut snapshot.link_metrics {
        sanitize_link_metrics(metrics);
    }
}

fn sanitize_link_metrics(metrics: &mut LinkMetrics) {
    if !metrics.latency_ms.is_finite() {
        metrics.latency_ms = 0.0;
    }
    if !metrics.goodput_mbps.is_finite() {
        metrics.goodput_mbps = 0.0;
    }
    if !metrics.duration_ms.is_finite() {
        metrics.duration_ms = 0.0;
    }
}

fn rpc_endpoint_for_snapshot(node: &NodeSnapshot) -> Option<String> {
    if !node.rpc_running {
        return None;
    }

    let endpoint = node
        .advertised_rpc_endpoint
        .clone()
        .or_else(|| node.rpc_endpoint.clone());
    if let Some(endpoint) = endpoint {
        if rpc_endpoint_is_reachable(&endpoint) {
            return Some(endpoint);
        }
    }
    let port = node
        .advertised_rpc_endpoint
        .as_deref()
        .or(node.rpc_endpoint.as_deref())
        .and_then(|value| value.rsplit_once(':').map(|(_, port)| port.to_string()))
        .unwrap_or_else(|| CLUSTER_AGENT_RPC_PORT.to_string());
    snapshot_control_addr_candidates(node)
        .into_iter()
        .filter_map(|control_addr| {
            let (host, _) = control_addr.rsplit_once(':')?;
            Some(format!("{host}:{port}"))
        })
        .find(|endpoint| rpc_endpoint_is_reachable(endpoint))
}

fn rpc_endpoint_is_reachable(endpoint: &str) -> bool {
    endpoint
        .to_socket_addrs()
        .ok()
        .and_then(|mut addrs| addrs.next())
        .map(|addr| TcpStream::connect_timeout(&addr, RPC_ENDPOINT_PROBE_TIMEOUT).is_ok())
        .unwrap_or(false)
}

fn query_remote_snapshot_with_retry(
    control_addr: &str,
    rpc_servers: Option<&str>,
) -> Result<NodeSnapshot> {
    query_remote_snapshot_with_candidates(&[control_addr.to_string()], rpc_servers)
        .map(|(_, snapshot)| snapshot)
}

fn query_remote_snapshot_with_candidates(
    control_addrs: &[String],
    rpc_servers: Option<&str>,
) -> Result<(String, NodeSnapshot)> {
    let mut last_error = None;
    for control_addr in dedup_sorted_control_addrs(control_addrs.to_vec()) {
        for attempt in 0..SNAPSHOT_QUERY_RETRIES {
            match AgentClient::new(control_addr.clone())
                .get_snapshot_with_rpc(rpc_servers.map(|value| value.to_string()))
            {
                Ok(snapshot) => return Ok((control_addr.clone(), snapshot)),
                Err(err) => {
                    last_error = Some(err);
                    if attempt + 1 < SNAPSHOT_QUERY_RETRIES {
                        thread::sleep(SNAPSHOT_QUERY_RETRY_DELAY);
                    }
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("snapshot query failed")))
}

fn control_addr_allowed(node: &NodeSnapshot, allowed: &HashSet<String>) -> bool {
    allowed.is_empty()
        || allowed.contains(&node.control_addr)
        || node
            .advertised_control_addr
            .as_ref()
            .map(|value| allowed.contains(value))
            .unwrap_or(false)
        || node
            .known_control_addrs
            .iter()
            .any(|value| allowed.contains(value))
}

fn collect_cluster_snapshots(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    allowed_control_addrs: Option<&[String]>,
) -> Result<Vec<NodeSnapshot>> {
    let allowed: HashSet<String> = allowed_control_addrs
        .unwrap_or(&[])
        .iter()
        .filter_map(|value| normalize_control_addr(value).ok())
        .collect();

    let mut nodes = Vec::new();
    let mut local = snapshot_local_state(api, bind_addr, runtime_dir, state, None)?;
    local.control_addr = local_control_addr_for_bind(bind_addr);
    if control_addr_allowed(&local, &allowed) {
        nodes.push(local);
    }

    for peer in list_peers(peers).into_iter().filter(|peer| peer.trusted) {
        let candidates = peer_control_addr_candidates(&peer);
        let connect_addr =
            preferred_control_addr_from_candidates(&candidates, &peer.control_addr);
        let candidate = NodeSnapshot {
            node: crate::cluster_api::NodeInfo {
                node_id: peer.node_id.clone(),
                display_name: peer.display_name.clone(),
                os_name: peer.os_name.clone(),
                arch: peer.arch.clone(),
            },
            control_addr: connect_addr.clone(),
            advertised_control_addr: peer.advertised_control_addr.clone(),
            known_control_addrs: candidates.clone(),
            runtime_dir: String::new(),
            models_dir: String::new(),
            rpc_endpoint: peer.rpc_endpoint.clone(),
            advertised_rpc_endpoint: peer.advertised_rpc_endpoint.clone(),
            rpc_running: peer.rpc_running,
            public_api_addr: None,
            advertised_public_api_addr: None,
            public_api_running: false,
            firewall_status: None,
            firewall_action_required: false,
            devices: Vec::new(),
            execution_groups: Vec::new(),
            instances: Vec::new(),
            link_metrics: Vec::new(),
        };
        if !control_addr_allowed(&candidate, &allowed) {
            continue;
        }

        let (resolved_addr, mut snapshot) = query_remote_snapshot_with_candidates(&candidates, None)
            .with_context(|| format!("failed to query node snapshot from '{connect_addr}'"))?;
        align_snapshot_to_connect_addr(&mut snapshot, &resolved_addr);
        sanitize_snapshot_memory(&mut snapshot);
        if snapshot.advertised_control_addr.is_none() {
            snapshot.advertised_control_addr = peer.advertised_control_addr.clone();
        }
        snapshot.known_control_addrs = dedup_sorted_control_addrs(
            snapshot
                .known_control_addrs
                .into_iter()
                .chain(candidates.into_iter())
                .collect(),
        );
        if snapshot.advertised_rpc_endpoint.is_none() {
            snapshot.advertised_rpc_endpoint = peer.advertised_rpc_endpoint.clone();
        }
        let persist_needed = update_peer_from_snapshot(peers, &snapshot, &resolved_addr);
        if persist_needed {
            let _ = persist_paired_peers(runtime_dir, &state.local_node.node_id, peers);
        }
        nodes.push(snapshot);
    }

    nodes.sort_by(|lhs, rhs| {
        lhs.node
            .display_name
            .cmp(&rhs.node.display_name)
            .then(lhs.control_addr.cmp(&rhs.control_addr))
    });
    nodes.dedup_by(|lhs, rhs| lhs.control_addr == rhs.control_addr);
    Ok(nodes)
}

fn list_local_managed_models(state: &Arc<AgentRuntimeState>) -> Result<Vec<ManagedModelEntry>> {
    discover_models(&state.models_dir)
}

fn list_local_model_packages(state: &Arc<AgentRuntimeState>) -> Result<Vec<ModelPackage>> {
    discover_model_packages(&state.models_dir)
}

fn list_cluster_model_packages(
    bind_addr: &str,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    state: &Arc<AgentRuntimeState>,
) -> Vec<ClusterModelPackageInfo> {
    let mut merged_by_folder: HashMap<String, ClusterModelPackageInfo> = HashMap::new();
    let local_control_addr = local_control_addr_for_bind(bind_addr);
    let local_display_name = state.local_node.display_name.clone();
    if let Ok(packages) = list_local_model_packages(state) {
        let models = list_local_managed_models(state).unwrap_or_default();
        merge_model_packages_for_node(
            &mut merged_by_folder,
            &local_control_addr,
            &local_display_name,
            packages,
            models,
        );
    }

    for peer in list_peers(peers).into_iter().filter(|peer| peer.trusted) {
        let control_addr = preferred_peer_control_addr(&peer);
        let client = AgentClient::new(control_addr.clone());
        let Ok(packages) = client.list_model_packages() else {
            continue;
        };
        let models = client.list_managed_models().unwrap_or_default();
        merge_model_packages_for_node(
            &mut merged_by_folder,
            &control_addr,
            &peer.display_name,
            packages,
            models,
        );
    }

    let mut packages = merged_by_folder.into_values().collect::<Vec<_>>();
    packages.sort_by(|lhs, rhs| {
        lhs.package
            .display_name
            .cmp(&rhs.package.display_name)
            .then(lhs.package.folder_name.cmp(&rhs.package.folder_name))
    });
    packages
}

fn merge_model_packages_for_node(
    merged_by_folder: &mut HashMap<String, ClusterModelPackageInfo>,
    control_addr: &str,
    display_name: &str,
    packages: Vec<ModelPackage>,
    models: Vec<ManagedModelEntry>,
) {
    let managed_by_path = models
        .into_iter()
        .map(|model| (normalized_path_key(&model.model_path), model.id))
        .collect::<HashMap<_, _>>();

    for package in packages {
        let package_location = ModelPackageNodeAvailability {
            control_addr: control_addr.to_string(),
            display_name: display_name.to_string(),
            package_path: package.path.display().to_string(),
        };
        let entry = merged_by_folder
            .entry(package.folder_name.clone())
            .or_insert_with(|| ClusterModelPackageInfo {
                package: package.clone(),
                available_on: Vec::new(),
                model_file_availability: Vec::new(),
                mmproj_file_availability: Vec::new(),
            });
        merge_package_basics(&mut entry.package, &package);
        push_unique_package_location(&mut entry.available_on, &package_location);

        for artifact in &package.model_files {
            let full_path = package_file_path_string(&package, &artifact.relative_path);
            let file_location = ModelFileNodeAvailability {
                control_addr: control_addr.to_string(),
                display_name: display_name.to_string(),
                package_path: package.path.display().to_string(),
                full_path: full_path.clone(),
                managed_model_id: managed_by_path
                    .get(&normalized_path_key(&full_path))
                    .cloned(),
            };
            merge_cluster_artifact(
                &mut entry.package.model_files,
                &mut entry.model_file_availability,
                artifact.clone(),
                file_location,
            );
        }

        for artifact in &package.mmproj_files {
            let full_path = package_file_path_string(&package, &artifact.relative_path);
            let file_location = ModelFileNodeAvailability {
                control_addr: control_addr.to_string(),
                display_name: display_name.to_string(),
                package_path: package.path.display().to_string(),
                full_path,
                managed_model_id: None,
            };
            merge_cluster_artifact(
                &mut entry.package.mmproj_files,
                &mut entry.mmproj_file_availability,
                artifact.clone(),
                file_location,
            );
        }
    }
}

fn merge_package_basics(existing: &mut ModelPackage, incoming: &ModelPackage) {
    if existing.path.as_os_str().is_empty() {
        existing.path = incoming.path.clone();
    }
    if existing.readme_path.is_none() {
        existing.readme_path = incoming.readme_path.clone();
    }
    if existing.guessed_repo_id.is_none() {
        existing.guessed_repo_id = incoming.guessed_repo_id.clone();
    }
}

fn merge_cluster_artifact(
    package_artifacts: &mut Vec<ModelArtifact>,
    availability: &mut Vec<ClusterModelArtifactInfo>,
    artifact: ModelArtifact,
    location: ModelFileNodeAvailability,
) {
    if let Some(existing) = package_artifacts
        .iter_mut()
        .find(|existing| existing.relative_path == artifact.relative_path)
    {
        merge_artifact_basics(existing, &artifact);
    } else {
        package_artifacts.push(artifact.clone());
        package_artifacts.sort_by(|lhs, rhs| lhs.relative_path.cmp(&rhs.relative_path));
    }

    match availability
        .iter_mut()
        .find(|existing| existing.artifact.relative_path == artifact.relative_path)
    {
        Some(existing) => {
            merge_artifact_basics(&mut existing.artifact, &artifact);
            if !existing.available_on.iter().any(|node| {
                node.control_addr == location.control_addr && node.full_path == location.full_path
            }) {
                existing.available_on.push(location);
                existing.available_on.sort_by(|lhs, rhs| {
                    lhs.display_name
                        .cmp(&rhs.display_name)
                        .then(lhs.control_addr.cmp(&rhs.control_addr))
                });
            }
        }
        None => availability.push(ClusterModelArtifactInfo {
            artifact,
            available_on: vec![location],
        }),
    }

    availability.sort_by(|lhs, rhs| lhs.artifact.relative_path.cmp(&rhs.artifact.relative_path));
}

fn merge_artifact_basics(existing: &mut ModelArtifact, incoming: &ModelArtifact) {
    if existing.file_name.trim().is_empty() {
        existing.file_name = incoming.file_name.clone();
    }
    if existing.size_bytes == 0 {
        existing.size_bytes = incoming.size_bytes;
    }
    if existing.metadata.is_none() {
        existing.metadata = incoming.metadata.clone();
    }
}

fn push_unique_package_location(
    locations: &mut Vec<ModelPackageNodeAvailability>,
    location: &ModelPackageNodeAvailability,
) {
    if locations.iter().any(|existing| {
        existing.control_addr == location.control_addr
            && existing.package_path == location.package_path
    }) {
        return;
    }
    locations.push(location.clone());
    locations.sort_by(|lhs, rhs| {
        lhs.display_name
            .cmp(&rhs.display_name)
            .then(lhs.control_addr.cmp(&rhs.control_addr))
    });
}

fn package_file_path_string(package: &ModelPackage, relative_path: &str) -> String {
    let mut path = package.path.clone();
    for segment in relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
    {
        path.push(segment);
    }
    path.display().to_string()
}

fn normalized_path_key(path: &str) -> String {
    path.replace('\\', "/").to_ascii_lowercase()
}

fn list_cluster_managed_models(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
) -> Result<Vec<ManagedModelEntry>> {
    let nodes = collect_cluster_snapshots(api, peers, bind_addr, runtime_dir, state, None)?;
    let mut merged_by_id: HashMap<String, ManagedModelEntry> = HashMap::new();
    for node in nodes {
        let models = if node.control_addr == local_control_addr_for_bind(bind_addr) {
            list_local_managed_models(state)?
        } else {
            AgentClient::new(node.control_addr.clone()).list_managed_models()?
        };
        for model in models {
            match merged_by_id.get_mut(&model.id) {
                Some(existing) => merge_managed_model_entries(existing, &model, &node.control_addr),
                None => {
                    let mut base = model.clone();
                    let mut allowed = base.allowed_control_addrs.clone().unwrap_or_default();
                    allowed.push(node.control_addr.clone());
                    allowed.sort();
                    allowed.dedup();
                    base.allowed_control_addrs = Some(allowed);
                    merged_by_id.insert(base.id.clone(), base);
                }
            }
        }
    }

    let mut models: Vec<_> = merged_by_id.into_values().collect();
    models.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
    Ok(models)
}

fn merge_managed_model_entries(
    existing: &mut ManagedModelEntry,
    incoming: &ManagedModelEntry,
    owner_control_addr: &str,
) {
    if existing.model_path.trim().is_empty() {
        existing.model_path = incoming.model_path.clone();
    }
    if existing.mmproj_path.is_none() && incoming.mmproj_path.is_some() {
        existing.mmproj_path = incoming.mmproj_path.clone();
    }
    if existing.diarization_model_path.is_none() && incoming.diarization_model_path.is_some() {
        existing.diarization_model_path = incoming.diarization_model_path.clone();
    }
    existing.single_device_only = existing.single_device_only || incoming.single_device_only;
    existing.allowed_control_addrs = Some({
        let mut values = existing.allowed_control_addrs.clone().unwrap_or_default();
        values.push(owner_control_addr.to_string());
        if let Some(extra) = &incoming.allowed_control_addrs {
            values.extend(extra.iter().cloned());
        }
        values.sort();
        values.dedup();
        values
    });
}

fn resolve_cluster_managed_model(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    model_id: &str,
) -> Result<Option<ManagedModelEntry>> {
    let normalized = model_id.trim();
    if normalized.is_empty() {
        return Ok(None);
    }

    let nodes = collect_cluster_snapshots(api, peers, bind_addr, runtime_dir, state, None)?;
    let mut merged: Option<ManagedModelEntry> = None;
    for node in nodes {
        let entry = if node.control_addr == local_control_addr_for_bind(bind_addr) {
            find_model_entry(&state.models_dir, normalized)?
        } else {
            AgentClient::new(node.control_addr.clone())
                .resolve_managed_model(normalized.to_string())?
        };
        let Some(entry) = entry else {
            continue;
        };
        match &mut merged {
            Some(existing) => merge_managed_model_entries(existing, &entry, &node.control_addr),
            None => {
                let mut base = entry.clone();
                let mut allowed = base.allowed_control_addrs.clone().unwrap_or_default();
                allowed.push(node.control_addr.clone());
                allowed.sort();
                allowed.dedup();
                base.allowed_control_addrs = Some(allowed);
                merged = Some(base);
            }
        }
    }
    Ok(merged)
}

fn resolve_cluster_instance(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    name: &str,
    load_if_managed: bool,
) -> Result<ResolvedClusterInstance> {
    let normalized = name.trim();
    if normalized.is_empty() {
        bail!("instance name is required");
    }

    let nodes = collect_cluster_snapshots(api, peers, bind_addr, runtime_dir, state, None)?;
    let mut matches = Vec::new();
    for node in &nodes {
        for instance in node
            .instances
            .iter()
            .filter(|instance| instance.name == normalized)
        {
            matches.push((node, instance));
        }
    }
    match matches.as_slice() {
        [(node, instance)] => {
            return Ok(ResolvedClusterInstance {
                owner_control_addr: node.control_addr.clone(),
                owner_display_name: node.node.display_name.clone(),
                instance_id: instance.instance_id,
                model_id: None,
                auto_loaded: false,
            });
        }
        [] => {}
        _ => {
            let owners = matches
                .iter()
                .map(|(node, _)| format!("{} ({})", node.node.display_name, node.control_addr))
                .collect::<Vec<_>>()
                .join(", ");
            bail!("cluster instance name '{normalized}' is ambiguous across nodes: {owners}");
        }
    }

    if !load_if_managed {
        bail!("unknown cluster instance '{normalized}'");
    }

    let model_entry =
        resolve_cluster_managed_model(api, peers, bind_addr, runtime_dir, state, normalized)?
            .ok_or_else(|| {
                anyhow::anyhow!("unknown cluster instance or managed model '{normalized}'")
            })?;
    let scheduled = schedule_instance_for_cluster(
        api,
        peers,
        bind_addr,
        runtime_dir,
        state,
        model_entry.create_instance_params(),
        model_entry.allowed_control_addrs.clone(),
        true,
    )?;
    Ok(ResolvedClusterInstance {
        owner_control_addr: scheduled.owner_control_addr,
        owner_display_name: scheduled.owner_display_name,
        instance_id: scheduled.instance_id,
        model_id: Some(model_entry.id),
        auto_loaded: true,
    })
}

fn build_remote_rpc_sets(owner: &NodeSnapshot, nodes: &[NodeSnapshot]) -> Vec<Vec<String>> {
    let remote_endpoints: Vec<String> = nodes
        .iter()
        .filter(|node| node.control_addr != owner.control_addr)
        .filter_map(rpc_endpoint_for_snapshot)
        .collect();

    let mut sets = vec![Vec::new()];
    if remote_endpoints.len() <= 10 {
        for mask in 1usize..(1usize << remote_endpoints.len()) {
            let mut set = Vec::new();
            for (index, endpoint) in remote_endpoints.iter().enumerate() {
                if (mask & (1usize << index)) != 0 {
                    set.push(endpoint.clone());
                }
            }
            sets.push(set);
        }
    } else {
        for endpoint in &remote_endpoints {
            sets.push(vec![endpoint.clone()]);
        }
        for i in 0..remote_endpoints.len() {
            for j in (i + 1)..remote_endpoints.len() {
                sets.push(vec![
                    remote_endpoints[i].clone(),
                    remote_endpoints[j].clone(),
                ]);
            }
        }
        sets.push(remote_endpoints);
    }

    let mut seen = HashSet::new();
    sets.retain(|set| seen.insert(set.join(",")));
    sets
}

fn query_owner_snapshot_for_scheduler(
    api: &SharedClusterApi,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    owner: &NodeSnapshot,
    rpc_servers: Option<&str>,
) -> Result<NodeSnapshot> {
    query_snapshot_for_control_addr(
        api,
        bind_addr,
        runtime_dir,
        state,
        &owner.control_addr,
        rpc_servers,
    )
    .with_context(|| {
        format!(
            "failed to query preview snapshot from '{}'",
            owner.control_addr
        )
    })
}

fn query_snapshot_for_control_addr(
    api: &SharedClusterApi,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    control_addr: &str,
    rpc_servers: Option<&str>,
) -> Result<NodeSnapshot> {
    if control_addr == local_control_addr_for_bind(bind_addr) {
        let mut snapshot = snapshot_local_state(api, bind_addr, runtime_dir, state, rpc_servers)?;
        snapshot.control_addr = local_control_addr_for_bind(bind_addr);
        return Ok(snapshot);
    }

    let mut snapshot = query_remote_snapshot_with_retry(control_addr, rpc_servers)
        .with_context(|| format!("failed to query preview snapshot from '{}'", control_addr))?;
    align_snapshot_to_connect_addr(&mut snapshot, control_addr);
    sanitize_snapshot_memory(&mut snapshot);
    Ok(snapshot)
}

fn file_size_bytes(path: &str) -> u64 {
    fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}

fn required_paths(params: &CreateInstanceParams) -> Vec<String> {
    let mut paths = expand_model_path_dependencies(&params.model_path);
    if let Some(mmproj) = params
        .mmproj_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        paths.push(mmproj.to_string());
    }
    paths
}

fn expand_model_path_dependencies(model_path: &str) -> Vec<String> {
    let mut paths = vec![model_path.to_string()];
    let model_path = PathBuf::from(model_path);
    let Some(stem) = model_path
        .file_stem()
        .map(|value| value.to_string_lossy().into_owned())
    else {
        return paths;
    };
    let Some((base, index, total)) = parse_gguf_shard_suffix(&stem.to_ascii_lowercase()) else {
        return paths;
    };
    if index != 1 || total <= 1 {
        return paths;
    }
    let Some(parent) = model_path.parent() else {
        return paths;
    };
    let ext = model_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("gguf");
    paths.clear();
    for shard_index in 1..=total {
        paths.push(
            parent
                .join(format!("{base}-{shard_index:05}-of-{total:05}.{ext}"))
                .display()
                .to_string(),
        );
    }
    paths
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

fn stat_paths(paths: Vec<String>) -> Vec<PathStat> {
    paths
        .into_iter()
        .map(|path| {
            let meta = fs::metadata(&path).ok();
            PathStat {
                model_metadata: meta.as_ref().and_then(|value| {
                    if value.is_file() {
                        inspect_model_file(Path::new(&path))
                    } else {
                        None
                    }
                }),
                path,
                exists: meta.is_some(),
                size_bytes: meta.map(|value| value.len()).unwrap_or(0),
            }
        })
        .collect()
}

fn owner_required_path_stats(
    api: &SharedClusterApi,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    owner: &NodeSnapshot,
    params: &CreateInstanceParams,
) -> Result<Vec<PathStat>> {
    let resolved = resolve_owner_instance_params(bind_addr, state, &owner.control_addr, params)?;
    let paths = required_paths(&resolved);
    if owner.control_addr == local_control_addr_for_bind(bind_addr) {
        let _ = state;
        let _ = api;
        let _ = runtime_dir;
        return Ok(stat_paths(paths));
    }

    AgentClient::new(owner.control_addr.clone())
        .stat_paths(paths)
        .with_context(|| format!("failed to stat required paths on '{}'", owner.control_addr))
}

fn resolve_managed_model_for_owner(
    bind_addr: &str,
    state: &Arc<AgentRuntimeState>,
    owner_control_addr: &str,
    model_id: &str,
) -> Result<Option<ManagedModelEntry>> {
    if owner_control_addr == local_control_addr_for_bind(bind_addr) {
        return find_model_entry(&state.models_dir, model_id);
    }

    AgentClient::new(owner_control_addr.to_string()).resolve_managed_model(model_id.to_string())
}

fn resolve_owner_instance_params(
    bind_addr: &str,
    state: &Arc<AgentRuntimeState>,
    owner_control_addr: &str,
    params: &CreateInstanceParams,
) -> Result<CreateInstanceParams> {
    let Some(model_id) = params
        .managed_model_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(params.clone());
    };

    let Some(owner_entry) =
        resolve_managed_model_for_owner(bind_addr, state, owner_control_addr, model_id)?
    else {
        return Ok(params.clone());
    };

    let mut resolved = params.clone();
    resolved.model_path = owner_entry.model_path;
    resolved.mmproj_path = owner_entry.mmproj_path;
    resolved.diarization_model_path = owner_entry.diarization_model_path;
    Ok(resolved)
}

fn estimated_required_bytes_for_owner(
    api: &SharedClusterApi,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    owner: &NodeSnapshot,
    params: &CreateInstanceParams,
) -> Result<u64> {
    let resolved = resolve_owner_instance_params(bind_addr, state, &owner.control_addr, params)?;
    let stats = owner_required_path_stats(api, bind_addr, runtime_dir, state, owner, &resolved)?;
    if let Some(missing) = stats.iter().find(|stat| !stat.exists) {
        bail!(
            "required path '{}' is not available on node '{}'",
            missing.path,
            owner.node.display_name
        );
    }

    let model_paths = expand_model_path_dependencies(&resolved.model_path);
    let model_bytes = stats
        .iter()
        .filter(|stat| model_paths.iter().any(|path| path == &stat.path))
        .map(|stat| stat.size_bytes)
        .sum::<u64>();
    let mmproj_bytes = resolved
        .mmproj_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|path| {
            stats
                .iter()
                .find(|stat| stat.path == path)
                .map(|stat| stat.size_bytes)
        })
        .unwrap_or(0);
    let metadata = stats
        .iter()
        .find(|stat| stat.path == resolved.model_path)
        .and_then(|stat| stat.model_metadata.clone());

    Ok(estimate_runtime_vram(
        model_bytes,
        mmproj_bytes,
        metadata.as_ref(),
        resolved.n_ctx,
        resolved.n_batch,
        resolved.n_parallel,
        resolved.n_gpu_layers,
    )
    .required_gpu_bytes)
}

fn normalize_optional_text(value: Option<&str>) -> String {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("")
        .to_string()
}

fn format_mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / 1024.0 / 1024.0)
}

fn normalize_csv_value(value: &str) -> String {
    let mut parts = split_csv(value);
    parts.sort();
    parts.dedup();
    parts.join(",")
}

fn instance_matches_pool(
    instance: &crate::cluster_api::InstanceInfo,
    plan: &PlacementPlan,
) -> bool {
    instance.execution_group_id == plan.execution_group_id
        && normalize_csv_value(&instance.rpc_servers) == normalize_csv_value(&plan.rpc_servers)
}

fn instance_is_resident(instance: &crate::cluster_api::InstanceInfo) -> bool {
    matches!(
        instance.state,
        INSTANCE_STATE_LOADING
            | INSTANCE_STATE_LOADED
            | INSTANCE_STATE_SERVING
            | INSTANCE_STATE_GRACE
    )
}

fn instance_is_evictable_load_on_demand(instance: &crate::cluster_api::InstanceInfo) -> bool {
    instance.retention_mode == RetentionMode::LoadOnDemand
        && instance.active_request_count == 0
        && matches!(instance.state, INSTANCE_STATE_LOADED | INSTANCE_STATE_GRACE)
}

fn instance_is_waitable_load_on_demand(instance: &crate::cluster_api::InstanceInfo) -> bool {
    instance.retention_mode == RetentionMode::LoadOnDemand
        && (instance.active_request_count > 0
            || matches!(
                instance.state,
                INSTANCE_STATE_LOADING | INSTANCE_STATE_SERVING
            ))
}

fn instance_is_keep_loaded_blocker(instance: &crate::cluster_api::InstanceInfo) -> bool {
    instance.retention_mode == RetentionMode::KeepLoaded && instance_is_resident(instance)
}

fn find_matching_named_instance(
    instances: &[crate::cluster_api::InstanceInfo],
    params: &CreateInstanceParams,
    plan: &PlacementPlan,
) -> Option<crate::cluster_api::InstanceInfo> {
    let requested_name = params.name.trim();
    if requested_name.is_empty() {
        return None;
    }

    let requested_mmproj = normalize_optional_text(params.mmproj_path.as_deref());
    let requested_diarization = normalize_optional_text(params.diarization_model_path.as_deref());
    let requested_model_kind = params.effective_model_kind();
    instances
        .iter()
        .find(|instance| {
            instance.name == requested_name
                && instance.model_path == params.model_path
                && normalize_optional_text(Some(instance.mmproj_path.as_str())) == requested_mmproj
                && normalize_optional_text(instance.diarization_model_path.as_deref())
                    == requested_diarization
                && instance.model_kind == requested_model_kind
                && instance_matches_pool(instance, plan)
                && instance.state != INSTANCE_STATE_FAILED
        })
        .cloned()
}

fn has_conflicting_named_instance(
    instances: &[crate::cluster_api::InstanceInfo],
    params: &CreateInstanceParams,
    plan: &PlacementPlan,
) -> bool {
    let requested_name = params.name.trim();
    if requested_name.is_empty() {
        return false;
    }

    let requested_mmproj = normalize_optional_text(params.mmproj_path.as_deref());
    let requested_diarization = normalize_optional_text(params.diarization_model_path.as_deref());
    let requested_model_kind = params.effective_model_kind();
    instances.iter().any(|instance| {
        instance.name == requested_name
            && (instance.model_path != params.model_path
                || normalize_optional_text(Some(instance.mmproj_path.as_str())) != requested_mmproj
                || normalize_optional_text(instance.diarization_model_path.as_deref())
                    != requested_diarization
                || instance.model_kind != requested_model_kind
                || !instance_matches_pool(instance, plan))
    })
}

fn list_evictable_pool_instances(
    instances: &[crate::cluster_api::InstanceInfo],
    params: &CreateInstanceParams,
    plan: &PlacementPlan,
) -> Vec<crate::cluster_api::InstanceInfo> {
    instances
        .iter()
        .filter(|instance| instance_matches_pool(instance, plan))
        .filter(|instance| instance.name != params.name.trim())
        .filter(|instance| instance_is_evictable_load_on_demand(instance))
        .cloned()
        .collect()
}

fn has_waitable_pool_instances(
    instances: &[crate::cluster_api::InstanceInfo],
    params: &CreateInstanceParams,
    plan: &PlacementPlan,
) -> bool {
    instances
        .iter()
        .filter(|instance| instance_matches_pool(instance, plan))
        .filter(|instance| instance.name != params.name.trim())
        .any(|instance| instance_is_waitable_load_on_demand(instance))
}

fn has_keep_loaded_pool_blockers(
    instances: &[crate::cluster_api::InstanceInfo],
    params: &CreateInstanceParams,
    plan: &PlacementPlan,
) -> bool {
    instances
        .iter()
        .filter(|instance| instance_matches_pool(instance, plan))
        .filter(|instance| instance.name != params.name.trim())
        .any(|instance| instance_is_keep_loaded_blocker(instance))
}

fn is_cpu_backend_name(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    lowered.contains("cpu") || lowered.contains("blas") || lowered.contains("accelerate")
}

fn is_rpc_backend_name(value: &str) -> bool {
    value.to_ascii_lowercase().contains("rpc")
}

fn is_metal_backend_name(value: &str) -> bool {
    value.to_ascii_lowercase().contains("metal")
}

fn device_looks_integrated(
    snapshot: &NodeSnapshot,
    device: &crate::cluster_api::DeviceInfo,
) -> bool {
    if is_cpu_backend_name(&device.backend)
        || is_cpu_backend_name(&device.name)
        || is_rpc_backend_name(&device.backend)
        || is_rpc_backend_name(&device.name)
    {
        return false;
    }
    if snapshot.node.os_name.eq_ignore_ascii_case("macos")
        && (is_metal_backend_name(&device.backend) || is_metal_backend_name(&device.name))
    {
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

fn classify_group(
    snapshot: &NodeSnapshot,
    group: &crate::cluster_api::ExecutionGroupInfo,
    allow_cpu: bool,
    allow_integrated_gpu: bool,
) -> GroupClassification {
    let mut out = GroupClassification::default();

    for token in group
        .devices_csv
        .split(',')
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
    {
        let Ok(device_index) = token.parse::<i32>() else {
            continue;
        };
        let Some(device) = snapshot
            .devices
            .iter()
            .find(|device| device.bridge_device_index == device_index)
        else {
            continue;
        };

        let cpu = is_cpu_backend_name(&device.backend) || is_cpu_backend_name(&device.name);
        if cpu {
            if allow_cpu {
                out.accel_device_count += 1;
                out.has_local_accel = true;
                out.allowed_memory_free =
                    out.allowed_memory_free.saturating_add(device.memory_free);
                out.allowed_memory_total =
                    out.allowed_memory_total.saturating_add(device.memory_total);
            } else {
                out.has_disallowed_device = true;
            }
            continue;
        }

        if !allow_integrated_gpu && device_looks_integrated(snapshot, device) {
            out.has_disallowed_device = true;
            continue;
        }

        out.accel_device_count += 1;
        out.allowed_memory_free = out.allowed_memory_free.saturating_add(device.memory_free);
        out.allowed_memory_total = out.allowed_memory_total.saturating_add(device.memory_total);
        if is_rpc_backend_name(&device.backend) || is_rpc_backend_name(&device.name) {
            out.rpc_device_count += 1;
        } else {
            out.has_local_accel = true;
        }
    }

    out
}

#[derive(Default)]
struct GroupClassification {
    has_local_accel: bool,
    rpc_device_count: i32,
    accel_device_count: i32,
    has_disallowed_device: bool,
    allowed_memory_free: u64,
    allowed_memory_total: u64,
}

fn candidate_device_label(device: &crate::cluster_api::DeviceInfo) -> String {
    let description = device.description.trim();
    if !description.is_empty() && !description.eq_ignore_ascii_case(device.name.trim()) {
        description.to_string()
    } else {
        device.name.trim().to_string()
    }
}

fn candidate_device_key(
    snapshot: &NodeSnapshot,
    device: &crate::cluster_api::DeviceInfo,
) -> String {
    let lowered = candidate_device_label(device).to_ascii_lowercase();
    if is_cpu_backend_name(&device.backend) || is_cpu_backend_name(&device.name) {
        return format!("cpu|{lowered}|{}", device.memory_total);
    }
    if is_rpc_backend_name(&device.backend) || is_rpc_backend_name(&device.name) {
        return format!("rpc|{lowered}|{}", device.memory_total);
    }
    if device_looks_integrated(snapshot, device) {
        return format!("integrated|{lowered}|{}", device.memory_total);
    }
    format!("gpu|{lowered}|{}", device.memory_total)
}

fn candidate_group_device_entries(
    snapshot: &NodeSnapshot,
    group: &crate::cluster_api::ExecutionGroupInfo,
    allow_cpu: bool,
    allow_integrated_gpu: bool,
) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    for token in group
        .devices_csv
        .split(',')
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
    {
        let Ok(device_index) = token.parse::<i32>() else {
            continue;
        };
        let Some(device) = snapshot
            .devices
            .iter()
            .find(|device| device.bridge_device_index == device_index)
        else {
            continue;
        };
        if (is_cpu_backend_name(&device.backend) || is_cpu_backend_name(&device.name)) && !allow_cpu
        {
            continue;
        }
        if !allow_integrated_gpu && device_looks_integrated(snapshot, device) {
            continue;
        }
        entries.push((
            candidate_device_key(snapshot, device),
            candidate_device_label(device),
        ));
    }
    entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));
    entries.dedup_by(|lhs, rhs| lhs.0 == rhs.0);
    entries
}

fn candidate_display_label(
    snapshot: &NodeSnapshot,
    nodes: &[NodeSnapshot],
    owner_display_name: &str,
    group: &crate::cluster_api::ExecutionGroupInfo,
    rpc_servers: &str,
    allow_cpu: bool,
    allow_integrated_gpu: bool,
) -> String {
    let mut labels =
        candidate_group_device_entries(snapshot, group, allow_cpu, allow_integrated_gpu)
            .into_iter()
            .filter(|(key, _)| !key.starts_with("rpc|"))
            .map(|(_, label)| label)
            .collect::<Vec<_>>();
    labels.extend(remote_node_labels_for_candidate(nodes, rpc_servers));
    labels.sort();
    labels.dedup();
    if labels.is_empty() {
        format!("{owner_display_name}: {}", group.label)
    } else {
        format!("{owner_display_name}: {}", labels.join(" + "))
    }
}

fn remote_node_labels_for_candidate(nodes: &[NodeSnapshot], rpc_servers: &str) -> Vec<String> {
    split_csv(rpc_servers)
        .into_iter()
        .filter_map(|rpc_server| {
            nodes.iter().find(|node| {
                node.rpc_endpoint.as_deref() == Some(rpc_server.as_str())
                    || node.advertised_rpc_endpoint.as_deref() == Some(rpc_server.as_str())
            })
        })
        .map(|node| {
            let gpu_label = node
                .devices
                .iter()
                .find(|device| {
                    !is_cpu_backend_name(&device.backend)
                        && !is_cpu_backend_name(&device.name)
                        && !is_rpc_backend_name(&device.backend)
                        && !is_rpc_backend_name(&device.name)
                        && !device_looks_integrated(node, device)
                })
                .map(candidate_device_label);
            match gpu_label {
                Some(device) => format!("{} {}", node.node.display_name, device),
                None => node.node.display_name.clone(),
            }
        })
        .collect()
}

fn candidate_signature_key(
    snapshot: &NodeSnapshot,
    owner_control_addr: &str,
    group: &crate::cluster_api::ExecutionGroupInfo,
    rpc_servers: &str,
    allow_cpu: bool,
    allow_integrated_gpu: bool,
) -> String {
    let mut device_keys =
        candidate_group_device_entries(snapshot, group, allow_cpu, allow_integrated_gpu)
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
    device_keys.sort();
    let mut rpc_keys = split_csv(rpc_servers);
    rpc_keys.sort();
    rpc_keys.dedup();
    format!(
        "{}|{}|{}",
        owner_control_addr.to_ascii_lowercase(),
        device_keys.join("+"),
        rpc_keys.join(",")
    )
}

fn strategy_rank(strategy: PlacementStrategy) -> i32 {
    match strategy {
        PlacementStrategy::SingleNode => 0,
        PlacementStrategy::LocalSplit => 1,
        PlacementStrategy::HybridTwoNode => 2,
        PlacementStrategy::HybridMultiNode => 3,
    }
}

fn plan_is_schedulable(plan: &PlacementPlan) -> bool {
    plan.ready_now || plan.requires_eviction
}

fn build_placement_candidates(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    params: &CreateInstanceParams,
    allowed_control_addrs: Option<&[String]>,
) -> Result<Vec<PlacementCandidate>> {
    ensure_cluster_rpc_candidates(state, peers);
    let nodes = collect_cluster_snapshots(api, peers, bind_addr, runtime_dir, state, None)?;
    let allowed: HashSet<String> = allowed_control_addrs
        .unwrap_or(&[])
        .iter()
        .filter_map(|value| normalize_control_addr(value).ok())
        .collect();
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut owner_params_cache: HashMap<String, CreateInstanceParams> = HashMap::new();
    let requested_group_id = params.execution_group_id.trim();
    let wants_auto_group = requested_group_id.is_empty() || requested_group_id == "cluster:auto";
    let requested_rpc_servers =
        normalize_csv_value(&normalize_optional_text(params.rpc_servers.as_deref()));

    for owner in &nodes {
        if !allowed.is_empty() && !control_addr_allowed(owner, &allowed) {
            continue;
        }
        if let Some(preferred_owner) = params.preferred_owner_control_addr.as_deref() {
            let preferred_owner = match normalize_control_addr(preferred_owner) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let owner_matches = control_addr_allowed(owner, &HashSet::from([preferred_owner]));
            if !owner_matches {
                continue;
            }
        }

        let owner_params = match owner_params_cache.get(&owner.control_addr) {
            Some(cached) => cached.clone(),
            None => {
                let resolved =
                    resolve_owner_instance_params(bind_addr, state, &owner.control_addr, params)?;
                owner_params_cache.insert(owner.control_addr.clone(), resolved.clone());
                resolved
            }
        };

        let required_bytes = match estimated_required_bytes_for_owner(
            api,
            bind_addr,
            runtime_dir,
            state,
            owner,
            &owner_params,
        ) {
            Ok(value) => value,
            Err(_) => continue,
        };

        for remote_set in build_remote_rpc_sets(owner, &nodes) {
            let rpc_servers = remote_set.join(",");
            if !requested_rpc_servers.is_empty()
                && normalize_csv_value(&rpc_servers) != requested_rpc_servers
            {
                continue;
            }
            let Ok(preview) = query_owner_snapshot_for_scheduler(
                api,
                bind_addr,
                runtime_dir,
                state,
                owner,
                if rpc_servers.is_empty() {
                    None
                } else {
                    Some(rpc_servers.as_str())
                },
            ) else {
                continue;
            };

            for group in &preview.execution_groups {
                if group.id == "cluster:auto" {
                    continue;
                }
                if !wants_auto_group && group.id != requested_group_id {
                    continue;
                }

                let classification = classify_group(
                    &preview,
                    group,
                    params.allow_cpu,
                    params.allow_integrated_gpu,
                );
                if classification.accel_device_count == 0 || classification.has_disallowed_device {
                    continue;
                }
                if params.single_device_only
                    && (classification.accel_device_count != 1
                        || classification.rpc_device_count != 0)
                {
                    continue;
                }
                if rpc_servers.is_empty() && classification.rpc_device_count > 0 {
                    continue;
                }
                if !rpc_servers.is_empty() && classification.rpc_device_count == 0 {
                    continue;
                }
                if classification.rpc_device_count > 0 && !classification.has_local_accel {
                    continue;
                }
                let strategy = if classification.rpc_device_count == 0 {
                    if classification.accel_device_count == 1 {
                        PlacementStrategy::SingleNode
                    } else {
                        PlacementStrategy::LocalSplit
                    }
                } else if classification.rpc_device_count == 1 {
                    PlacementStrategy::HybridTwoNode
                } else {
                    PlacementStrategy::HybridMultiNode
                };
                let display_label = candidate_display_label(
                    &preview,
                    &nodes,
                    &owner.node.display_name,
                    group,
                    &rpc_servers,
                    params.allow_cpu,
                    params.allow_integrated_gpu,
                );

                let provisional_plan = PlacementPlan {
                    owner_control_addr: String::new(),
                    owner_display_name: owner.node.display_name.clone(),
                    execution_group_id: group.id.clone(),
                    rpc_servers: rpc_servers.clone(),
                    display_label: display_label.clone(),
                    strategy,
                    device_count: classification.accel_device_count,
                    remote_node_count: classification.rpc_device_count,
                    estimated_required_bytes: required_bytes,
                    estimated_group_free_bytes: classification.allowed_memory_free,
                    reusable_instance_id: None,
                    ready_now: false,
                    requires_eviction: false,
                };

                let reusable_instance = find_matching_named_instance(
                    &owner.instances,
                    &owner_params,
                    &provisional_plan,
                );
                let requires_eviction = classification.allowed_memory_free < required_bytes
                    && !list_evictable_pool_instances(
                        &owner.instances,
                        &owner_params,
                        &provisional_plan,
                    )
                    .is_empty();
                let ready_now = reusable_instance.is_some()
                    || classification.allowed_memory_free >= required_bytes;

                let public_owner_addr =
                    if owner.control_addr == local_control_addr_for_bind(bind_addr) {
                        local_control_addr_for_bind(bind_addr)
                    } else {
                        owner.control_addr.clone()
                    };

                let key = candidate_signature_key(
                    &preview,
                    &public_owner_addr,
                    group,
                    &rpc_servers,
                    params.allow_cpu,
                    params.allow_integrated_gpu,
                );
                if !seen.insert(key) {
                    continue;
                }

                candidates.push(PlacementCandidate {
                    connect_control_addr: owner.control_addr.clone(),
                    plan: PlacementPlan {
                        owner_control_addr: public_owner_addr,
                        owner_display_name: owner.node.display_name.clone(),
                        execution_group_id: group.id.clone(),
                        rpc_servers: rpc_servers.clone(),
                        display_label,
                        strategy,
                        device_count: classification.accel_device_count,
                        remote_node_count: classification.rpc_device_count,
                        estimated_required_bytes: required_bytes,
                        estimated_group_free_bytes: classification.allowed_memory_free,
                        reusable_instance_id: reusable_instance
                            .as_ref()
                            .map(|instance| instance.instance_id),
                        ready_now,
                        requires_eviction,
                    },
                });
            }
        }
    }

    candidates.sort_by(|lhs, rhs| {
        let lhs_reuse_rank = match lhs.plan.reusable_instance_id {
            Some(_) => 0,
            None => 1,
        };
        let rhs_reuse_rank = match rhs.plan.reusable_instance_id {
            Some(_) => 0,
            None => 1,
        };
        let lhs_rank = strategy_rank(lhs.plan.strategy);
        let rhs_rank = strategy_rank(rhs.plan.strategy);
        let lhs_fits = lhs.plan.estimated_group_free_bytes >= lhs.plan.estimated_required_bytes;
        let rhs_fits = rhs.plan.estimated_group_free_bytes >= rhs.plan.estimated_required_bytes;
        lhs_reuse_rank
            .cmp(&rhs_reuse_rank)
            .then(lhs_rank.cmp(&rhs_rank))
            .then(rhs.plan.ready_now.cmp(&lhs.plan.ready_now))
            .then(rhs_fits.cmp(&lhs_fits))
            .then(
                rhs.plan
                    .estimated_group_free_bytes
                    .saturating_sub(rhs.plan.estimated_required_bytes)
                    .cmp(
                        &lhs.plan
                            .estimated_group_free_bytes
                            .saturating_sub(lhs.plan.estimated_required_bytes),
                    ),
            )
            .then(lhs.plan.device_count.cmp(&rhs.plan.device_count))
            .then(
                lhs.plan
                    .owner_control_addr
                    .cmp(&rhs.plan.owner_control_addr),
            )
    });

    Ok(candidates)
}

fn ensure_cluster_rpc_candidates(
    state: &Arc<AgentRuntimeState>,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
) {
    if !settings::multi_node_rpc_enabled() {
        return;
    }

    if let Err(err) = ensure_rpc_server_running(state) {
        eprintln!(
            "cluster agent: failed to ensure local rpc host for placement discovery: {err:#}"
        );
    }

    let targets = list_peers(peers)
        .into_iter()
        .filter(|peer| peer.trusted)
        .filter(|peer| !peer.rpc_running || peer.advertised_rpc_endpoint.is_none())
        .map(|peer| preferred_peer_control_addr(&peer))
        .collect::<Vec<_>>();

    let mut seen = HashSet::new();
    for control_addr in targets {
        if !seen.insert(control_addr.clone()) {
            continue;
        }
        if let Err(err) = AgentClient::new(control_addr.clone()).restart_rpc_server() {
            eprintln!(
                "cluster agent: failed to ensure peer rpc host for placement discovery via {control_addr}: {err:#}"
            );
        }
    }
}

fn owner_set_retention_mode(
    api: &SharedClusterApi,
    bind_addr: &str,
    control_addr: &str,
    instance_id: i64,
    retention_mode: RetentionMode,
) -> Result<()> {
    if control_addr == local_control_addr_for_bind(bind_addr) {
        let guard = api_write(api)?;
        guard.set_retention_mode(instance_id, retention_mode)
    } else {
        AgentClient::new(control_addr.to_string()).set_retention_mode(instance_id, retention_mode)
    }
}

fn owner_load_instance(
    api: &SharedClusterApi,
    bind_addr: &str,
    control_addr: &str,
    instance_id: i64,
) -> Result<()> {
    if control_addr == local_control_addr_for_bind(bind_addr) {
        let guard = api_write(api)?;
        guard.load_instance(instance_id)
    } else {
        AgentClient::new(control_addr.to_string()).load_instance(instance_id)
    }
}

fn owner_unload_instance(
    api: &SharedClusterApi,
    bind_addr: &str,
    control_addr: &str,
    instance_id: i64,
) -> Result<()> {
    if control_addr == local_control_addr_for_bind(bind_addr) {
        let guard = api_write(api)?;
        guard.unload_instance(instance_id)
    } else {
        AgentClient::new(control_addr.to_string()).unload_instance(instance_id)
    }
}

fn owner_create_instance(
    api: &SharedClusterApi,
    bind_addr: &str,
    control_addr: &str,
    params: &CreateInstanceParams,
) -> Result<i64> {
    if control_addr == local_control_addr_for_bind(bind_addr) {
        let guard = api_write(api)?;
        guard.create_instance(params)
    } else {
        AgentClient::new(control_addr.to_string()).create_instance(params.clone())
    }
}

fn owner_remove_instance(
    api: &SharedClusterApi,
    bind_addr: &str,
    control_addr: &str,
    instance_id: i64,
) -> Result<()> {
    if control_addr == local_control_addr_for_bind(bind_addr) {
        let guard = api_write(api)?;
        guard.remove_instance(instance_id)
    } else {
        AgentClient::new(control_addr.to_string()).remove_instance(instance_id)
    }
}

fn prepare_candidate_for_schedule(
    api: &SharedClusterApi,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    candidate: &PlacementCandidate,
    params: &CreateInstanceParams,
    load_immediately: bool,
) -> Result<(PreparedScheduleTarget, u64)> {
    let started = Instant::now();
    let deadline = if load_immediately {
        Some(started + SCHEDULE_WAIT_TIMEOUT)
    } else {
        None
    };

    loop {
        let snapshot = query_snapshot_for_control_addr(
            api,
            bind_addr,
            runtime_dir,
            state,
            &candidate.connect_control_addr,
            if candidate.plan.rpc_servers.is_empty() {
                None
            } else {
                Some(candidate.plan.rpc_servers.as_str())
            },
        )?;

        if has_conflicting_named_instance(&snapshot.instances, params, &candidate.plan) {
            bail!(
                "instance name '{}' already exists on '{}' with a different placement or model config",
                params.name,
                candidate.plan.owner_display_name
            );
        }

        if let Some(existing) =
            find_matching_named_instance(&snapshot.instances, params, &candidate.plan)
        {
            return Ok((
                PreparedScheduleTarget::Reuse(existing.instance_id),
                started.elapsed().as_millis() as u64,
            ));
        }

        let group = snapshot
            .execution_groups
            .iter()
            .find(|group| group.id == candidate.plan.execution_group_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "execution group '{}' is no longer available on '{}'",
                    candidate.plan.execution_group_id,
                    candidate.plan.owner_display_name
                )
            })?;

        if !load_immediately || group.memory_free >= candidate.plan.estimated_required_bytes {
            return Ok((
                PreparedScheduleTarget::CreateNew,
                started.elapsed().as_millis() as u64,
            ));
        }

        let evictable = list_evictable_pool_instances(&snapshot.instances, params, &candidate.plan);
        if !evictable.is_empty() {
            for instance in evictable {
                let _ = owner_unload_instance(
                    api,
                    bind_addr,
                    &candidate.connect_control_addr,
                    instance.instance_id,
                );
            }
            thread::sleep(SCHEDULE_WAIT_POLL);
            continue;
        }

        let has_waitable =
            has_waitable_pool_instances(&snapshot.instances, params, &candidate.plan);
        let has_keep_loaded =
            has_keep_loaded_pool_blockers(&snapshot.instances, params, &candidate.plan);
        if has_waitable {
            if deadline.is_some_and(|until| Instant::now() < until) {
                thread::sleep(SCHEDULE_WAIT_POLL);
                continue;
            }
            bail!(
                "resource pool '{}' on '{}' stayed busy with active load_on_demand work",
                candidate.plan.execution_group_id,
                candidate.plan.owner_display_name
            );
        }

        if has_keep_loaded {
            bail!(
                "resource pool '{}' on '{}' is occupied by keep_loaded instances",
                candidate.plan.execution_group_id,
                candidate.plan.owner_display_name
            );
        }

        bail!(
            "not enough free memory on '{}' for group '{}' ({})",
            candidate.plan.owner_display_name,
            candidate.plan.execution_group_id,
            format_mib(candidate.plan.estimated_required_bytes)
        );
    }
}

fn plan_instance_for_cluster(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    params: CreateInstanceParams,
    allowed_control_addrs: Option<Vec<String>>,
) -> Result<PlacementPlan> {
    build_placement_candidates(
        api,
        peers,
        bind_addr,
        runtime_dir,
        state,
        &params,
        allowed_control_addrs.as_deref(),
    )?
    .into_iter()
    .find(|candidate| plan_is_schedulable(&candidate.plan))
    .map(|candidate| candidate.plan)
    .ok_or_else(|| anyhow::anyhow!("no viable placement candidates were found"))
}

fn schedule_instance_for_cluster(
    api: &SharedClusterApi,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    bind_addr: &str,
    runtime_dir: &Path,
    state: &Arc<AgentRuntimeState>,
    params: CreateInstanceParams,
    allowed_control_addrs: Option<Vec<String>>,
    load_immediately: bool,
) -> Result<ScheduledInstance> {
    let _schedule_guard = match state.schedule_lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("cluster agent: schedule mutex poisoned, recovering");
            poisoned.into_inner()
        }
    };
    let mut owner_params_cache: HashMap<String, CreateInstanceParams> = HashMap::new();

    let candidates = build_placement_candidates(
        api,
        peers,
        bind_addr,
        runtime_dir,
        state,
        &params,
        allowed_control_addrs.as_deref(),
    )?
    .into_iter()
    .filter(|candidate| plan_is_schedulable(&candidate.plan))
    .collect::<Vec<_>>();
    if candidates.is_empty() {
        bail!("no viable placement candidates were found");
    }

    let mut errors = Vec::new();
    for candidate in candidates {
        if load_immediately && !candidate.plan.rpc_servers.is_empty() {
            let _ = restart_matching_peer_rpc_servers(peers, &candidate.plan.rpc_servers);
        }

        let mut scheduled_params = match owner_params_cache.get(&candidate.connect_control_addr) {
            Some(cached) => cached.clone(),
            None => {
                let resolved = resolve_owner_instance_params(
                    bind_addr,
                    state,
                    &candidate.connect_control_addr,
                    &params,
                )?;
                owner_params_cache.insert(candidate.connect_control_addr.clone(), resolved.clone());
                resolved
            }
        };
        scheduled_params.execution_group_id = candidate.plan.execution_group_id.clone();
        scheduled_params.rpc_servers = if candidate.plan.rpc_servers.is_empty() {
            None
        } else {
            Some(candidate.plan.rpc_servers.clone())
        };

        let attempt = (|| -> Result<(i64, bool, u64)> {
            let (target, waited_ms) = prepare_candidate_for_schedule(
                api,
                bind_addr,
                runtime_dir,
                state,
                &candidate,
                &scheduled_params,
                load_immediately,
            )?;

            match target {
                PreparedScheduleTarget::Reuse(instance_id) => {
                    if scheduled_params.retention_mode == RetentionMode::KeepLoaded {
                        owner_set_retention_mode(
                            api,
                            bind_addr,
                            &candidate.connect_control_addr,
                            instance_id,
                            RetentionMode::KeepLoaded,
                        )?;
                    }
                    if load_immediately {
                        owner_load_instance(
                            api,
                            bind_addr,
                            &candidate.connect_control_addr,
                            instance_id,
                        )?;
                    }
                    Ok((instance_id, true, waited_ms))
                }
                PreparedScheduleTarget::CreateNew => {
                    let instance_id = owner_create_instance(
                        api,
                        bind_addr,
                        &candidate.connect_control_addr,
                        &scheduled_params,
                    )?;
                    if load_immediately {
                        if let Err(err) = owner_load_instance(
                            api,
                            bind_addr,
                            &candidate.connect_control_addr,
                            instance_id,
                        ) {
                            let _ = owner_remove_instance(
                                api,
                                bind_addr,
                                &candidate.connect_control_addr,
                                instance_id,
                            );
                            return Err(err);
                        }
                    }
                    Ok((instance_id, false, waited_ms))
                }
            }
        })();

        match attempt {
            Ok((instance_id, reused_existing, waited_ms)) => {
                return Ok(ScheduledInstance {
                    owner_control_addr: candidate.plan.owner_control_addr.clone(),
                    owner_display_name: candidate.plan.owner_display_name.clone(),
                    instance_id,
                    execution_group_id: candidate.plan.execution_group_id.clone(),
                    rpc_servers: candidate.plan.rpc_servers.clone(),
                    strategy: candidate.plan.strategy,
                    reused_existing,
                    waited_ms,
                });
            }
            Err(err) => {
                errors.push(format!(
                    "{} [{}] {}",
                    candidate.plan.owner_display_name, candidate.plan.execution_group_id, err
                ));
            }
        }
    }

    bail!("all placement candidates failed: {}", errors.join(" | "))
}

fn upsert_manual_peer(
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    control_addr: &str,
) -> Result<()> {
    let normalized = normalize_control_addr(control_addr)?;
    let key = format!("manual@{normalized}");
    let peer = DiscoveredPeer {
        info: placeholder_peer_info(&normalized),
        last_seen: Instant::now(),
        manual: true,
        shared_token: None,
    };
    let mut guard = peers
        .lock()
        .map_err(|_| anyhow::anyhow!("peer registry mutex poisoned"))?;
    guard.insert(key, peer);
    Ok(())
}

fn remove_peer_by_addr(peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>, control_addr: &str) {
    let Ok(normalized) = normalize_control_addr(control_addr) else {
        return;
    };
    if let Ok(mut guard) = peers.lock() {
        guard.retain(|_, peer| {
            !peer_control_addr_candidates(&peer.info)
                .iter()
                .any(|candidate| candidate == &normalized)
        });
    }
}

fn placeholder_peer_info(control_addr: &str) -> PeerInfo {
    PeerInfo {
        node_id: format!("manual:{control_addr}"),
        display_name: control_addr.to_string(),
        os_name: String::new(),
        arch: String::new(),
        control_addr: control_addr.to_string(),
        advertised_control_addr: Some(control_addr.to_string()),
        known_control_addrs: vec![control_addr.to_string()],
        rpc_endpoint: None,
        advertised_rpc_endpoint: None,
        rpc_running: false,
        trusted: true,
        last_seen_unix_ms: unix_ms_now(),
    }
}

fn normalize_control_addr(control_addr: &str) -> Result<String> {
    let trimmed = control_addr.trim();
    if trimmed.is_empty() {
        bail!("control_addr is required");
    }
    if trimmed.rsplit_once(':').is_none() {
        return Ok(format!("{trimmed}:{CLUSTER_AGENT_CONTROL_PORT}"));
    }
    Ok(trimmed.to_string())
}

fn seed_manual_peers_from_env(peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>) {
    let Ok(raw) = std::env::var("ENGINE_CLUSTER_SEED_PEERS") else {
        return;
    };
    for item in raw.split([',', ';', ' ']) {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let _ = upsert_manual_peer(peers, item);
    }
}

fn persisted_peers_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join("cluster-peers.txt")
}

fn load_persisted_paired_peers(
    runtime_dir: &Path,
    local_node_id: &str,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
) {
    let settings = settings::load_controller_settings_or_default();
    if !settings.paired_peers.is_empty() {
        for peer in settings.paired_peers {
            let control_addr = match normalize_control_addr(&peer.control_addr) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let info = PeerInfo {
                node_id: peer.node_id.clone(),
                display_name: if peer.display_name.trim().is_empty() {
                    control_addr.clone()
                } else {
                    peer.display_name.clone()
                },
                os_name: String::new(),
                arch: String::new(),
                control_addr: control_addr.clone(),
                advertised_control_addr: Some(control_addr.clone()),
                known_control_addrs: dedup_sorted_control_addrs(
                    std::iter::once(control_addr.clone())
                        .chain(peer.known_control_addrs.into_iter())
                        .collect(),
                ),
                rpc_endpoint: None,
                advertised_rpc_endpoint: None,
                rpc_running: false,
                trusted: true,
                last_seen_unix_ms: unix_ms_now(),
            };
            insert_or_update_peer(
                peers,
                info,
                true,
                decode_paired_token(local_node_id, &peer.shared_token_obfuscated),
            );
        }
        return;
    }

    let path = persisted_peers_path(runtime_dir);
    let Ok(raw) = fs::read_to_string(&path) else {
        return;
    };

    for line in raw.lines() {
        let item = line.trim();
        if item.is_empty() || item.starts_with('#') {
            continue;
        }
        let _ = upsert_manual_peer(peers, item);
    }
    let _ = persist_paired_peers(runtime_dir, local_node_id, peers);
}

fn persist_paired_peers(
    runtime_dir: &Path,
    local_node_id: &str,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
) -> Result<()> {
    let entries = {
        let guard = peers
            .lock()
            .map_err(|_| anyhow::anyhow!("peer registry mutex poisoned"))?;
        let items = guard
            .values()
            .filter(|peer| peer.manual)
            .map(|peer| settings::PairedPeerSettings {
                node_id: peer.info.node_id.clone(),
                display_name: peer.info.display_name.clone(),
                control_addr: preferred_peer_control_addr(&peer.info),
                known_control_addrs: peer_control_addr_candidates(&peer.info),
                shared_token_obfuscated: peer
                    .shared_token
                    .as_deref()
                    .map(|value| encode_paired_token(local_node_id, value))
                    .unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        let mut merged = Vec::<settings::PairedPeerSettings>::new();
        for mut item in items {
            item.known_control_addrs = dedup_sorted_control_addrs(
                std::iter::once(item.control_addr.clone())
                    .chain(item.known_control_addrs.into_iter())
                    .collect(),
            );
            if let Some(existing) = merged.iter_mut().find(|existing| {
                (!item.node_id.trim().is_empty() && existing.node_id == item.node_id)
                    || control_addr_candidates_overlap(
                        &existing.known_control_addrs,
                        &item.known_control_addrs,
                    )
            }) {
                if existing.node_id.trim().is_empty() && !item.node_id.trim().is_empty() {
                    existing.node_id = item.node_id.clone();
                }
                if existing.display_name.trim().is_empty() && !item.display_name.trim().is_empty() {
                    existing.display_name = item.display_name.clone();
                }
                if existing.shared_token_obfuscated.trim().is_empty()
                    && !item.shared_token_obfuscated.trim().is_empty()
                {
                    existing.shared_token_obfuscated = item.shared_token_obfuscated.clone();
                }
                existing.known_control_addrs = dedup_sorted_control_addrs(
                    existing
                        .known_control_addrs
                        .iter()
                        .cloned()
                        .chain(item.known_control_addrs.iter().cloned())
                        .collect(),
                );
                existing.control_addr = preferred_control_addr_from_candidates(
                    &existing.known_control_addrs,
                    &existing.control_addr,
                );
            } else {
                item.control_addr =
                    preferred_control_addr_from_candidates(&item.known_control_addrs, &item.control_addr);
                merged.push(item);
            }
        }
        merged.sort_by(|lhs, rhs| {
            lhs.display_name
                .cmp(&rhs.display_name)
                .then(lhs.control_addr.cmp(&rhs.control_addr))
        });
        merged
    };

    settings::update_controller_settings(|settings| {
        settings.paired_peers = entries.clone();
    })
    .map_err(anyhow::Error::msg)?;

    let legacy_path = persisted_peers_path(runtime_dir);
    if legacy_path.exists() {
        fs::remove_file(&legacy_path).ok();
    }
    Ok(())
}

fn encode_paired_token(local_node_id: &str, token: &str) -> String {
    let key = pairing_obfuscation_key(local_node_id);
    let bytes = token
        .as_bytes()
        .iter()
        .enumerate()
        .map(|(index, byte)| byte ^ key[index % key.len()])
        .collect::<Vec<_>>();
    BASE64.encode(bytes)
}

fn decode_paired_token(local_node_id: &str, encoded: &str) -> Option<String> {
    if encoded.trim().is_empty() {
        return None;
    }
    let key = pairing_obfuscation_key(local_node_id);
    let bytes = BASE64.decode(encoded).ok()?;
    let plain = bytes
        .iter()
        .enumerate()
        .map(|(index, byte)| byte ^ key[index % key.len()])
        .collect::<Vec<_>>();
    String::from_utf8(plain).ok()
}

fn pairing_obfuscation_key(local_node_id: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(local_node_id.as_bytes());
    hasher.update(b":openresearchtools:paired-peer");
    hasher.finalize().to_vec()
}

fn start_discovery_session(
    state: &Arc<AgentRuntimeState>,
    mode: DiscoveryMode,
    seconds: u64,
) -> DiscoveryStatus {
    let mut guard = match state.discovery.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return DiscoveryStatus {
                mode: DiscoveryMode::Off,
                active: false,
                expires_unix_ms: 0,
            }
        }
    };
    if mode == DiscoveryMode::Off || seconds == 0 {
        guard.mode = DiscoveryMode::Off;
        guard.active_until = None;
    } else {
        guard.mode = mode;
        guard.active_until = Some(Instant::now() + Duration::from_secs(seconds.min(600)));
    }
    discovery_status_from_guard(&mut guard)
}

fn discovery_status(state: &Arc<AgentRuntimeState>) -> DiscoveryStatus {
    let mut guard = match state.discovery.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return DiscoveryStatus {
                mode: DiscoveryMode::Off,
                active: false,
                expires_unix_ms: 0,
            }
        }
    };
    discovery_status_from_guard(&mut guard)
}

fn discovery_status_from_guard(guard: &mut DiscoveryRuntimeState) -> DiscoveryStatus {
    if guard
        .active_until
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        guard.mode = DiscoveryMode::Off;
        guard.active_until = None;
    }
    let expires_unix_ms = guard
        .active_until
        .map(|deadline| {
            let remaining = deadline.saturating_duration_since(Instant::now());
            unix_ms_now().saturating_add(remaining.as_millis() as u64)
        })
        .unwrap_or(0);
    DiscoveryStatus {
        mode: guard.mode,
        active: guard.mode != DiscoveryMode::Off,
        expires_unix_ms,
    }
}

fn list_pairing_requests(state: &Arc<AgentRuntimeState>) -> Vec<PairingRequestInfo> {
    let mut guard = match state.pairing_requests.lock() {
        Ok(guard) => guard,
        Err(_) => return Vec::new(),
    };
    let now = Instant::now();
    guard.retain(|_, request| now.duration_since(request.received_at) <= PAIRING_REQUEST_TTL);
    let mut requests = guard
        .values()
        .map(|request| request.info.clone())
        .collect::<Vec<_>>();
    requests.sort_by(|lhs, rhs| {
        lhs.requester_display_name
            .cmp(&rhs.requester_display_name)
            .then(lhs.requester_control_addr.cmp(&rhs.requester_control_addr))
    });
    requests
}

fn request_pairing_with_peer(state: &Arc<AgentRuntimeState>, control_addr: &str) -> Result<()> {
    let control_addr = normalize_control_addr(control_addr)?;
    let request_id = generate_pairing_token(
        "pair-request",
        &[
            state.local_node.node_id.as_str(),
            control_addr.as_str(),
            &unix_ms_now().to_string(),
        ],
    );
    let request_code = format_pairing_code(&request_id);
    {
        let mut guard = state
            .outgoing_pairing_requests
            .lock()
            .map_err(|_| anyhow::anyhow!("outgoing pairing registry mutex poisoned"))?;
        guard.retain(|_, request| request.requested_at.elapsed() <= PAIRING_REQUEST_TTL);
        guard.insert(
            request_id.clone(),
            OutgoingPairingRequest {
                request_id: request_id.clone(),
                request_code: request_code.clone(),
                control_addr: control_addr.clone(),
                requested_at: Instant::now(),
            },
        );
    }
    let request = PairingRequestInfo {
        request_id,
        request_code,
        requester_node_id: state.local_node.node_id.clone(),
        requester_display_name: state.local_node.display_name.clone(),
        requester_os_name: state.local_node.os_name.clone(),
        requester_arch: state.local_node.arch.clone(),
        requester_control_addr: preferred_control_addr_for_pairing(state),
        requested_unix_ms: unix_ms_now(),
    };
    let remote = AgentClient::new(control_addr.clone());
    remote.expect_ok(remote.send(
        AgentRequest::SubmitPairingRequest { request },
        REQUEST_TIMEOUT_FAST,
    )?)
}

fn receive_pairing_request(
    state: &Arc<AgentRuntimeState>,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    request: PairingRequestInfo,
) {
    let candidate = PeerInfo {
        node_id: request.requester_node_id.clone(),
        display_name: request.requester_display_name.clone(),
        os_name: request.requester_os_name.clone(),
        arch: request.requester_arch.clone(),
        control_addr: request.requester_control_addr.clone(),
        advertised_control_addr: Some(request.requester_control_addr.clone()),
        known_control_addrs: vec![request.requester_control_addr.clone()],
        rpc_endpoint: None,
        advertised_rpc_endpoint: None,
        rpc_running: false,
        trusted: false,
        last_seen_unix_ms: unix_ms_now(),
    };
    insert_or_update_peer(peers, candidate, false, None);
    if let Ok(mut guard) = state.pairing_requests.lock() {
        guard.retain(|_, pending| pending.received_at.elapsed() <= PAIRING_REQUEST_TTL);
        guard.insert(
            request.request_id.clone(),
            PendingPairingRequest {
                info: request,
                received_at: Instant::now(),
            },
        );
    }
}

fn accept_pairing_request(
    state: &Arc<AgentRuntimeState>,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    runtime_dir: &Path,
    request_id: &str,
) -> Result<()> {
    let request = {
        let mut guard = state
            .pairing_requests
            .lock()
            .map_err(|_| anyhow::anyhow!("pairing request registry mutex poisoned"))?;
        guard
            .remove(request_id)
            .ok_or_else(|| anyhow::anyhow!("pairing request '{request_id}' no longer exists"))?
            .info
    };
    let requester_control_addr = normalize_control_addr(&request.requester_control_addr)?;
    let shared_token = generate_pairing_token(
        "paired-peer",
        &[
            request_id,
            state.local_node.node_id.as_str(),
            request.requester_node_id.as_str(),
            &unix_ms_now().to_string(),
        ],
    );
    let local_peer = PeerInfo {
        node_id: state.local_node.node_id.clone(),
        display_name: state.local_node.display_name.clone(),
        os_name: state.local_node.os_name.clone(),
        arch: state.local_node.arch.clone(),
        control_addr: preferred_control_addr_for_pairing(state),
        advertised_control_addr: advertised_control_addr_for_bind(&state.bind_addr)
            .or_else(|| Some(preferred_control_addr_for_pairing(state))),
        known_control_addrs: advertised_control_addrs_for_bind(&state.bind_addr),
        rpc_endpoint: None,
        advertised_rpc_endpoint: None,
        rpc_running: settings::multi_node_rpc_enabled(),
        trusted: true,
        last_seen_unix_ms: unix_ms_now(),
    };
    AgentClient::new(requester_control_addr.clone()).expect_ok(
        AgentClient::new(requester_control_addr.clone()).send(
            AgentRequest::FinalizePairing {
                request_id: request.request_id.clone(),
                peer: local_peer.clone(),
                shared_token: shared_token.clone(),
            },
            REQUEST_TIMEOUT_STATE,
        )?,
    )?;

    let requester_peer = PeerInfo {
        node_id: request.requester_node_id.clone(),
        display_name: request.requester_display_name.clone(),
        os_name: request.requester_os_name.clone(),
        arch: request.requester_arch.clone(),
        control_addr: requester_control_addr.clone(),
        advertised_control_addr: Some(requester_control_addr),
        known_control_addrs: vec![request.requester_control_addr.clone()],
        rpc_endpoint: None,
        advertised_rpc_endpoint: None,
        rpc_running: false,
        trusted: true,
        last_seen_unix_ms: unix_ms_now(),
    };
    insert_or_update_peer(peers, requester_peer, true, Some(shared_token));
    persist_paired_peers(runtime_dir, &state.local_node.node_id, peers)
}

fn decline_pairing_request(state: &Arc<AgentRuntimeState>, request_id: &str) {
    if let Ok(mut guard) = state.pairing_requests.lock() {
        guard.remove(request_id);
    }
}

fn finalize_pairing_request(
    state: &Arc<AgentRuntimeState>,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    runtime_dir: &Path,
    request_id: &str,
    peer: PeerInfo,
    shared_token: &str,
) -> Result<()> {
    let expected = {
        let mut guard = state
            .outgoing_pairing_requests
            .lock()
            .map_err(|_| anyhow::anyhow!("outgoing pairing registry mutex poisoned"))?;
        guard.retain(|_, request| request.requested_at.elapsed() <= PAIRING_REQUEST_TTL);
        guard.remove(request_id).ok_or_else(|| {
            anyhow::anyhow!("pairing request '{request_id}' is unknown or expired")
        })?
    };
    let peer_control_addr = normalize_control_addr(&peer.control_addr)?;
    if normalize_control_addr(&expected.control_addr)? != peer_control_addr {
        bail!(
            "pairing reply for '{}' came from '{}' instead of '{}'",
            expected.request_code,
            peer_control_addr,
            expected.control_addr
        );
    }
    let mut paired_peer = peer;
    paired_peer.control_addr = peer_control_addr.clone();
    paired_peer.advertised_control_addr = Some(peer_control_addr);
    paired_peer.known_control_addrs = dedup_sorted_control_addrs(
        std::iter::once(paired_peer.control_addr.clone())
            .chain(paired_peer.known_control_addrs.iter().cloned())
            .collect(),
    );
    paired_peer.trusted = true;
    paired_peer.last_seen_unix_ms = unix_ms_now();
    insert_or_update_peer(peers, paired_peer, true, Some(shared_token.to_string()));
    persist_paired_peers(runtime_dir, &state.local_node.node_id, peers)
}

fn preferred_control_addr_for_pairing(state: &AgentRuntimeState) -> String {
    advertised_control_addr_for_bind(&state.bind_addr)
        .unwrap_or_else(|| state.local_control_addr.clone())
}

fn generate_pairing_token(prefix: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix.as_bytes());
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

fn format_pairing_code(request_id: &str) -> String {
    let compact = request_id
        .chars()
        .filter(|value| value.is_ascii_alphanumeric())
        .take(6)
        .collect::<String>()
        .to_ascii_uppercase();
    if compact.len() <= 3 {
        compact
    } else {
        format!("{}-{}", &compact[..3], &compact[3..])
    }
}

fn insert_or_update_peer(
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    mut info: PeerInfo,
    manual: bool,
    shared_token: Option<String>,
) {
    if let Ok(mut guard) = peers.lock() {
        let incoming_candidates = peer_control_addr_candidates(&info);
        let existing = guard.iter().find_map(|(key, peer)| {
            let same_node = !info.node_id.trim().is_empty() && peer.info.node_id == info.node_id;
            let same_addr = peer_control_addr_candidates(&peer.info)
                .iter()
                .any(|addr| incoming_candidates.iter().any(|candidate| candidate == addr));
            if same_node || same_addr {
                Some((key.clone(), peer.clone()))
            } else {
                None
            }
        });
        let key = existing
            .as_ref()
            .map(|(key, _)| key.clone())
            .unwrap_or_else(|| {
                if info.node_id.trim().is_empty() {
                    if manual {
                        format!("manual@{}", info.control_addr)
                    } else {
                        info.control_addr.clone()
                    }
                } else {
                    format!("{}@{}", info.node_id, info.control_addr)
                }
            });
        if let Some((_, current)) = existing.as_ref() {
            if info.os_name.trim().is_empty() {
                info.os_name = current.info.os_name.clone();
            }
            if info.arch.trim().is_empty() {
                info.arch = current.info.arch.clone();
            }
            if info.display_name.trim().is_empty() {
                info.display_name = current.info.display_name.clone();
            }
            if info.rpc_endpoint.is_none() {
                info.rpc_endpoint = current.info.rpc_endpoint.clone();
            }
            if info.advertised_rpc_endpoint.is_none() {
                info.advertised_rpc_endpoint = current.info.advertised_rpc_endpoint.clone();
            }
            info.rpc_running = info.rpc_running || current.info.rpc_running;
            info.known_control_addrs = dedup_sorted_control_addrs(
                current
                    .info
                    .known_control_addrs
                    .iter()
                    .cloned()
                    .chain(peer_control_addr_candidates(&current.info))
                    .chain(info.known_control_addrs.iter().cloned())
                    .chain(incoming_candidates.iter().cloned())
                    .collect(),
            );
        } else {
            info.known_control_addrs = dedup_sorted_control_addrs(
                info.known_control_addrs
                    .iter()
                    .cloned()
                    .chain(incoming_candidates.iter().cloned())
                    .collect(),
            );
        }
        let preferred_control =
            preferred_control_addr_from_candidates(&peer_control_addr_candidates(&info), &info.control_addr);
        info.control_addr = preferred_control.clone();
        if info.advertised_control_addr.is_none() {
            info.advertised_control_addr = Some(preferred_control);
        }
        guard.insert(
            key,
            DiscoveredPeer {
                info,
                last_seen: Instant::now(),
                manual: manual || existing.as_ref().is_some_and(|(_, peer)| peer.manual),
                shared_token: shared_token.or_else(|| {
                    existing
                        .as_ref()
                        .and_then(|(_, peer)| peer.shared_token.clone())
                }),
            },
        );
    }
}

fn update_peer_from_snapshot(
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    snapshot: &NodeSnapshot,
    connect_addr: &str,
) -> bool {
    let snapshot_candidates = dedup_sorted_control_addrs(
        std::iter::once(connect_addr.to_string())
            .chain(snapshot_control_addr_candidates(snapshot))
            .collect(),
    );
    let existing = peers.lock().ok().and_then(|guard| {
        guard.values().find_map(|peer| {
            let same_node = !snapshot.node.node_id.trim().is_empty()
                && peer.info.node_id == snapshot.node.node_id;
            let same_addr = peer_control_addr_candidates(&peer.info)
                .iter()
                .any(|addr| snapshot_candidates.iter().any(|candidate| candidate == addr));
            if same_node || same_addr {
                Some((peer.info.clone(), peer.manual, peer.shared_token.clone()))
            } else {
                None
            }
        })
    });

    let previous_preferred = existing
        .as_ref()
        .map(|(info, _, _)| preferred_peer_control_addr(info))
        .unwrap_or_default();
    let manual = existing
        .as_ref()
        .map(|(_, manual, _)| *manual)
        .unwrap_or(false);
    let shared_token = existing.as_ref().and_then(|(_, _, token)| token.clone());

    let info = PeerInfo {
        node_id: snapshot.node.node_id.clone(),
        display_name: snapshot.node.display_name.clone(),
        os_name: snapshot.node.os_name.clone(),
        arch: snapshot.node.arch.clone(),
        control_addr: connect_addr.to_string(),
        advertised_control_addr: better_control_addr(
            snapshot.advertised_control_addr.as_deref(),
            Some(connect_addr),
        ),
        known_control_addrs: snapshot_candidates,
        rpc_endpoint: snapshot.rpc_endpoint.clone(),
        advertised_rpc_endpoint: snapshot
            .advertised_rpc_endpoint
            .clone()
            .or_else(|| snapshot.rpc_endpoint.clone()),
        rpc_running: snapshot.rpc_running,
        trusted: manual,
        last_seen_unix_ms: unix_ms_now(),
    };
    insert_or_update_peer(peers, info, manual, shared_token);

    let next_preferred = peers
        .lock()
        .ok()
        .and_then(|guard| {
            guard.values().find_map(|peer| {
                if !snapshot.node.node_id.trim().is_empty()
                    && peer.info.node_id == snapshot.node.node_id
                {
                    Some(preferred_peer_control_addr(&peer.info))
                } else {
                    None
                }
            })
        })
        .unwrap_or_default();

    manual && !previous_preferred.is_empty() && previous_preferred != next_preferred
}

fn handle_discovery_announcement(
    state: &Arc<AgentRuntimeState>,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    mode: DiscoveryMode,
    announcement: DiscoveryAnnouncement,
    sender_control_addr: String,
) {
    let announcement_candidates =
        discovery_control_addr_candidates(&announcement, sender_control_addr.as_str());
    let control_addr =
        preferred_control_addr_from_candidates(&announcement_candidates, sender_control_addr.as_str());
    let mut persist_needed = false;
    let mut matched_known_pair = false;
    if let Ok(mut guard) = peers.lock() {
        let existing = guard.iter_mut().find_map(|(_, peer)| {
            let same_node = !announcement.node.node_id.trim().is_empty()
                && peer.info.node_id == announcement.node.node_id;
            let same_addr = peer_control_addr_candidates(&peer.info)
                .iter()
                .any(|addr| announcement_candidates.iter().any(|candidate| candidate == addr));
            if same_node || same_addr {
                Some(peer)
            } else {
                None
            }
        });
        if let Some(peer) = existing {
            matched_known_pair = peer.manual;
            let previous_addr = preferred_peer_control_addr(&peer.info);
            let next_candidates = dedup_sorted_control_addrs(
                peer_control_addr_candidates(&peer.info)
                    .into_iter()
                    .chain(announcement_candidates.iter().cloned())
                    .collect(),
            );
            let next_addr = preferred_control_addr_from_candidates(&next_candidates, &control_addr);
            peer.info.node_id = announcement.node.node_id.clone();
            peer.info.display_name = announcement.node.display_name.clone();
            peer.info.os_name = announcement.node.os_name.clone();
            peer.info.arch = announcement.node.arch.clone();
            peer.info.control_addr = next_addr.clone();
            peer.info.advertised_control_addr = Some(next_addr.clone());
            peer.info.known_control_addrs = next_candidates;
            peer.info.advertised_rpc_endpoint = announcement.advertised_rpc_endpoint.clone();
            peer.info.rpc_running = announcement.rpc_running;
            peer.info.last_seen_unix_ms = unix_ms_now();
            peer.last_seen = Instant::now();
            if peer.manual && previous_addr != next_addr {
                persist_needed = true;
            }
        } else {
            let peer_info = PeerInfo {
                node_id: announcement.node.node_id,
                display_name: announcement.node.display_name,
                os_name: announcement.node.os_name,
                arch: announcement.node.arch,
                control_addr: control_addr.clone(),
                advertised_control_addr: Some(control_addr.clone()),
                known_control_addrs: announcement_candidates.clone(),
                rpc_endpoint: None,
                advertised_rpc_endpoint: announcement.advertised_rpc_endpoint.clone(),
                rpc_running: announcement.rpc_running,
                trusted: false,
                last_seen_unix_ms: unix_ms_now(),
            };
            guard.insert(
                format!("{}@{}", peer_info.node_id, control_addr),
                DiscoveredPeer {
                    info: peer_info,
                    last_seen: Instant::now(),
                    manual: false,
                    shared_token: None,
                },
            );
        }
        let now = Instant::now();
        guard.retain(|_, peer| peer.manual || now.duration_since(peer.last_seen) <= PEER_TTL);
    }

    if persist_needed && matched_known_pair {
        let _ = persist_paired_peers(&state.runtime_dir, &state.local_node.node_id, peers);
    }
}

fn current_discovery_announcement(
    local_node: &crate::cluster_api::NodeInfo,
    bind_addr: &str,
    control_port: u16,
) -> DiscoveryAnnouncement {
    let known_control_addrs = advertised_control_addrs_for_bind(bind_addr);
    DiscoveryAnnouncement {
        protocol_version: 1,
        node: local_node.clone(),
        control_port,
        advertised_control_addr: known_control_addrs.first().cloned(),
        known_control_addrs: known_control_addrs.clone(),
        advertised_rpc_endpoint: {
            let rpc_endpoint = rpc_loopback_endpoint();
            advertised_rpc_endpoint_for_bind(
                Some(&rpc_endpoint),
                known_control_addrs.first().map(|value| value.as_str()),
            )
        },
        rpc_running: rpc_server_is_reachable(),
        announced_unix_ms: unix_ms_now(),
    }
}

fn start_discovery_loop(
    local_node: crate::cluster_api::NodeInfo,
    bind_addr: String,
    peers: Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    state: Arc<AgentRuntimeState>,
    socket: UdpSocket,
) {
    let control_port = bind_addr
        .rsplit_once(':')
        .and_then(|(_, port)| port.parse::<u16>().ok())
        .unwrap_or(CLUSTER_AGENT_CONTROL_PORT);

    thread::spawn(move || {
        let multicast_addr =
            SocketAddrV4::new(DISCOVERY_MULTICAST_IP, CLUSTER_AGENT_DISCOVERY_PORT);
        let broadcast_addr = SocketAddrV4::new(Ipv4Addr::BROADCAST, CLUSTER_AGENT_DISCOVERY_PORT);
        let mut last_announce = Instant::now() - DISCOVERY_ANNOUNCE_INTERVAL;
        let mut buffer = vec![0u8; 8192];

        loop {
            let status = discovery_status(&state);
            if status.active && last_announce.elapsed() >= DISCOVERY_ANNOUNCE_INTERVAL {
                let message = current_discovery_announcement(&local_node, &bind_addr, control_port);
                if let Ok(payload) = bincode::serialize(&message) {
                    let _ = socket.send_to(&payload, multicast_addr);
                    let _ = socket.send_to(&payload, broadcast_addr);
                }
                last_announce = Instant::now();
            }

            match socket.recv_from(&mut buffer) {
                Ok((bytes, sender)) => {
                    let Ok(announcement) =
                        bincode::deserialize::<DiscoveryAnnouncement>(&buffer[..bytes])
                    else {
                        continue;
                    };
                    if announcement.protocol_version != 1 {
                        continue;
                    }
                    if announcement.node.node_id == local_node.node_id
                        && announcement.control_port == control_port
                    {
                        continue;
                    }

                    if !status.active {
                        let reply =
                            current_discovery_announcement(&local_node, &bind_addr, control_port);
                        if let Ok(payload) = bincode::serialize(&reply) {
                            let _ = socket.send_to(&payload, sender);
                        }
                    }

                    let sender_ip = match sender {
                        SocketAddr::V4(addr) => *addr.ip(),
                        SocketAddr::V6(_) => continue,
                    };
                    let control_addr = format!("{}:{}", sender_ip, announcement.control_port);
                    handle_discovery_announcement(
                        &state,
                        &peers,
                        status.mode,
                        announcement,
                        control_addr,
                    );
                }
                Err(err)
                    if err.kind() == std::io::ErrorKind::WouldBlock
                        || err.kind() == std::io::ErrorKind::TimedOut => {}
                Err(err) => {
                    eprintln!("cluster discovery receive failed: {err}");
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    });
}

fn start_telemetry_loop(
    local_node: crate::cluster_api::NodeInfo,
    bind_addr: String,
    api: SharedClusterApi,
    state: Arc<AgentRuntimeState>,
    telemetry_cache: Arc<Mutex<HashMap<String, TelemetryEntry>>>,
    socket: UdpSocket,
) {
    thread::spawn(move || {
        let multicast_addr =
            SocketAddrV4::new(DISCOVERY_MULTICAST_IP, CLUSTER_AGENT_TELEMETRY_PORT);
        let broadcast_addr = SocketAddrV4::new(Ipv4Addr::BROADCAST, CLUSTER_AGENT_TELEMETRY_PORT);
        let announce_interval = Duration::from_millis(500);
        let mut last_announce = Instant::now() - announce_interval;
        let mut buffer = vec![0u8; 65536];

        loop {
            if last_announce.elapsed() >= announce_interval {
                if let Ok(message) = build_local_telemetry(&api, &bind_addr, &state) {
                    if let Ok(payload) = bincode::serialize(&message) {
                        let _ = socket.send_to(&payload, multicast_addr);
                        let _ = socket.send_to(&payload, broadcast_addr);
                    }
                }
                last_announce = Instant::now();
            }

            match socket.recv_from(&mut buffer) {
                Ok((bytes, sender)) => {
                    let Ok(snapshot) = bincode::deserialize::<TelemetrySnapshot>(&buffer[..bytes])
                    else {
                        continue;
                    };
                    if snapshot.node.node_id == local_node.node_id {
                        continue;
                    }
                    let sender_ip = match sender {
                        SocketAddr::V4(addr) => *addr.ip(),
                        SocketAddr::V6(_) => continue,
                    };
                    let sender_control_addr =
                        format!("{}:{}", sender_ip, CLUSTER_AGENT_CONTROL_PORT);
                    let control_addr = better_control_addr(
                        snapshot.advertised_control_addr.as_deref(),
                        Some(sender_control_addr.as_str()),
                    )
                    .unwrap_or(sender_control_addr);
                    let mut snapshot = snapshot;
                    align_telemetry_to_connect_addr(&mut snapshot, &control_addr);

                    if let Ok(mut guard) = telemetry_cache.lock() {
                        guard.insert(
                            control_addr,
                            TelemetryEntry {
                                snapshot,
                                last_seen: Instant::now(),
                            },
                        );
                        let now = Instant::now();
                        guard.retain(|_, entry| {
                            now.duration_since(entry.last_seen) <= TELEMETRY_TTL
                        });
                    }
                }
                Err(err)
                    if err.kind() == std::io::ErrorKind::WouldBlock
                        || err.kind() == std::io::ErrorKind::TimedOut => {}
                Err(err) => {
                    eprintln!("cluster telemetry receive failed: {err}");
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    });
}

fn bind_cluster_udp_socket(port: u16, purpose: &str) -> Result<UdpSocket> {
    let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port))
        .with_context(|| format!("failed to bind {purpose} socket on UDP {port}"))?;
    socket.set_nonblocking(false).ok();
    socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok();
    socket.set_broadcast(true).ok();
    socket
        .join_multicast_v4(&DISCOVERY_MULTICAST_IP, &Ipv4Addr::UNSPECIFIED)
        .ok();
    for candidate in preferred_interface_candidates() {
        socket
            .join_multicast_v4(&DISCOVERY_MULTICAST_IP, &candidate.ip)
            .ok();
    }
    socket.set_multicast_ttl_v4(1).ok();
    socket.set_multicast_loop_v4(false).ok();
    Ok(socket)
}

fn start_link_benchmark_monitor(
    local_node: crate::cluster_api::NodeInfo,
    bind_addr: String,
    peers: Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    state: Arc<AgentRuntimeState>,
) {
    thread::spawn(move || {
        let mut previous_remote = HashSet::new();
        loop {
            let current_remote = connected_remote_peer_addrs(&peers, &local_node.node_id);
            if current_remote != previous_remote {
                let added = current_remote
                    .difference(&previous_remote)
                    .cloned()
                    .collect::<Vec<_>>();
                for control_addr in added {
                    if has_link_metrics_for_peer(&state, &control_addr) {
                        continue;
                    }
                    let bind_addr = bind_addr.clone();
                    let state = state.clone();
                    thread::spawn(move || {
                        benchmark_connected_peer_with_retry(
                            &bind_addr,
                            &state,
                            &control_addr,
                            false,
                        );
                    });
                }
                previous_remote = current_remote;
            }
            thread::sleep(Duration::from_millis(750));
        }
    });
}

fn run_link_benchmarks_for_state(
    bind_addr: &str,
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    state: &Arc<AgentRuntimeState>,
    full: bool,
) -> Result<()> {
    let local_control_addr = local_control_addr_for_bind(bind_addr);
    let mut next_metrics = HashMap::new();
    let peers_snapshot = connected_remote_peer_addrs(peers, &state.local_node.node_id);

    for connect_addr in peers_snapshot {
        if connect_addr == local_control_addr {
            continue;
        }

        let metrics =
            measure_link_metrics_with_reverse_fallback(&connect_addr, &local_control_addr, full)
                .unwrap_or_else(|err| LinkMetrics {
                    peer_control_addr: connect_addr.clone(),
                    transport: transport_label_for_addr(&connect_addr),
                    probe_kind: if full {
                        "manual".to_string()
                    } else {
                        "startup".to_string()
                    },
                    payload_bytes: 0,
                    rounds: 0,
                    latency_ms: 0.0,
                    goodput_mbps: 0.0,
                    duration_ms: 0.0,
                    unix_ms: unix_ms_now(),
                    error: Some(format!("{err:#}")),
                });
        next_metrics.insert(connect_addr, metrics);
    }

    if let Ok(mut guard) = state.link_metrics.lock() {
        *guard = next_metrics;
    }
    Ok(())
}

fn connected_remote_peer_addrs(
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    local_node_id: &str,
) -> HashSet<String> {
    let Ok(guard) = peers.lock() else {
        return HashSet::new();
    };
    let now = Instant::now();
    guard
        .values()
        .filter(|peer| peer.info.node_id != local_node_id)
        .filter(|peer| now.duration_since(peer.last_seen) <= PEER_TTL)
        .map(|peer| preferred_peer_control_addr(&peer.info))
        .filter(|control_addr| !control_addr.trim().is_empty())
        .collect()
}

fn has_link_metrics_for_peer(state: &Arc<AgentRuntimeState>, control_addr: &str) -> bool {
    state
        .link_metrics
        .lock()
        .map(|guard| guard.contains_key(control_addr))
        .unwrap_or(false)
}

fn benchmark_connected_peer_with_retry(
    bind_addr: &str,
    state: &Arc<AgentRuntimeState>,
    control_addr: &str,
    full: bool,
) {
    thread::sleep(if full {
        Duration::from_millis(900)
    } else {
        Duration::from_secs(10)
    });
    let local_control_addr = local_control_addr_for_bind(bind_addr);
    let mut last_error = None;
    for _ in 0..3 {
        match measure_link_metrics_with_reverse_fallback(control_addr, &local_control_addr, full) {
            Ok(metrics) => {
                store_link_metrics(state, control_addr, metrics);
                return;
            }
            Err(err) => {
                last_error = Some(format!("{err:#}"));
                thread::sleep(Duration::from_millis(850));
            }
        }
    }

    store_link_metrics(
        state,
        control_addr,
        LinkMetrics {
            peer_control_addr: control_addr.to_string(),
            transport: transport_label_for_addr(control_addr),
            probe_kind: if full {
                "manual".to_string()
            } else {
                "startup".to_string()
            },
            payload_bytes: 0,
            rounds: 0,
            latency_ms: 0.0,
            goodput_mbps: 0.0,
            duration_ms: 0.0,
            unix_ms: unix_ms_now(),
            error: last_error.or_else(|| Some("benchmark failed".to_string())),
        },
    );
}

fn store_link_metrics(
    state: &Arc<AgentRuntimeState>,
    control_addr: &str,
    mut metrics: LinkMetrics,
) {
    metrics.peer_control_addr = control_addr.to_string();
    if let Ok(mut guard) = state.link_metrics.lock() {
        guard.insert(control_addr.to_string(), metrics);
    }
}

fn measure_link_metrics_with_reverse_fallback(
    peer_control_addr: &str,
    local_control_addr: &str,
    full: bool,
) -> Result<LinkMetrics> {
    match measure_link_metrics(peer_control_addr, full) {
        Ok(mut metrics) => {
            metrics.peer_control_addr = peer_control_addr.to_string();
            Ok(metrics)
        }
        Err(local_err) => {
            let client = AgentClient::new(peer_control_addr.to_string());
            let mut metrics = client
                .measure_link_to(local_control_addr.to_string(), full)
                .with_context(|| {
                    format!(
                        "local probe failed: {local_err:#}; reverse probe from '{peer_control_addr}' also failed"
                    )
                })?;
            metrics.peer_control_addr = peer_control_addr.to_string();
            metrics.unix_ms = unix_ms_now();
            Ok(metrics)
        }
    }
}

fn measure_link_metrics(control_addr: &str, full: bool) -> Result<LinkMetrics> {
    let latency_ms = measure_ping_latency_ms(control_addr)?;
    let transport = transport_label_for_addr(control_addr);
    let benchmark_deadline = benchmark_deadline_for_transport(&transport, full);

    let warmup_bytes = if full {
        LINK_BENCHMARK_FULL_WARMUP_BYTES
    } else {
        LINK_BENCHMARK_STARTUP_WARMUP_BYTES
    };
    let warmup_elapsed = transfer_probe_bytes(control_addr, warmup_bytes, benchmark_deadline)?;
    let warmup_bps = if warmup_elapsed > 0.0 {
        (warmup_bytes as f64) / warmup_elapsed
    } else {
        0.0
    };

    let (min_bytes, max_bytes, target_seconds) = benchmark_profile_for_transport(&transport, full);
    let mut measured_bytes = (warmup_bps * target_seconds) as u64;
    measured_bytes = measured_bytes.clamp(min_bytes, max_bytes);
    measured_bytes = align_up_to_chunk(measured_bytes, LINK_BENCHMARK_CHUNK_BYTES as u64);

    let elapsed = transfer_probe_bytes(control_addr, measured_bytes, benchmark_deadline)?;
    let goodput_mbps = if elapsed > 0.0 {
        ((measured_bytes as f64) * 8.0) / elapsed / 1_000_000.0
    } else {
        0.0
    };

    Ok(LinkMetrics {
        peer_control_addr: control_addr.to_string(),
        transport,
        probe_kind: if full {
            "manual".to_string()
        } else {
            "startup".to_string()
        },
        payload_bytes: measured_bytes,
        rounds: 1,
        latency_ms,
        goodput_mbps,
        duration_ms: elapsed * 1000.0,
        unix_ms: unix_ms_now(),
        error: None,
    })
}

fn measure_ping_latency_ms(control_addr: &str) -> Result<f64> {
    let mut samples = Vec::with_capacity(LINK_BENCHMARK_PING_SAMPLES);
    for _ in 0..LINK_BENCHMARK_PING_SAMPLES {
        let started = Instant::now();
        match send_request_once(control_addr, &AgentRequest::Ping, LINK_PROBE_TIMEOUT)? {
            AgentResponse::Pong => samples.push(started.elapsed().as_secs_f64() * 1000.0),
            AgentResponse::Error { message } => bail!(message),
            other => bail!("unexpected ping response: {:?}", other),
        }
    }
    samples.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    Ok(samples[samples.len() / 2])
}

fn transfer_probe_bytes(
    control_addr: &str,
    total_bytes: u64,
    max_elapsed: Duration,
) -> Result<f64> {
    let mut stream = TcpStream::connect(control_addr)
        .with_context(|| format!("failed to connect to peer '{control_addr}' for link probe"))?;
    stream.set_read_timeout(Some(LINK_PROBE_TIMEOUT)).ok();
    stream.set_write_timeout(Some(LINK_PROBE_TIMEOUT)).ok();
    stream.set_nodelay(true).ok();
    write_message(&mut stream, &AgentRequest::LinkProbe { bytes: total_bytes })?;

    let chunk = vec![0xA5u8; LINK_BENCHMARK_CHUNK_BYTES];
    let started = Instant::now();
    let mut remaining = total_bytes;
    while remaining > 0 {
        if started.elapsed() > max_elapsed {
            bail!(
                "link probe exceeded {:?} while sending {} bytes",
                max_elapsed,
                total_bytes
            );
        }
        let to_write = usize::try_from(remaining.min(chunk.len() as u64))
            .context("invalid link probe chunk size")?;
        stream
            .write_all(&chunk[..to_write])
            .context("failed to write link probe payload")?;
        remaining -= to_write as u64;
    }
    stream.flush().ok();
    match read_message::<AgentResponse>(&mut stream)? {
        AgentResponse::LinkProbeAck { bytes, .. } if bytes == total_bytes => {}
        AgentResponse::LinkProbeAck { bytes, .. } => {
            bail!("unexpected link probe ack size: {bytes}")
        }
        AgentResponse::Error { message } => bail!(message),
        other => bail!("unexpected link probe response: {:?}", other),
    }
    Ok(started.elapsed().as_secs_f64())
}

fn benchmark_profile_for_transport(transport: &str, full: bool) -> (u64, u64, f64) {
    match (transport, full) {
        ("thunderbolt/link-local", true) => (1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 2.5),
        ("thunderbolt/link-local", false) => (256 * 1024 * 1024, 512 * 1024 * 1024, 1.0),
        ("lan", true) => (512 * 1024 * 1024, 1024 * 1024 * 1024, 2.0),
        ("lan", false) => (128 * 1024 * 1024, 256 * 1024 * 1024, 1.0),
        (_, true) => (128 * 1024 * 1024, 512 * 1024 * 1024, 2.0),
        (_, false) => (64 * 1024 * 1024, 128 * 1024 * 1024, 1.0),
    }
}

fn benchmark_deadline_for_transport(transport: &str, full: bool) -> Duration {
    match (transport, full) {
        ("thunderbolt/link-local", true) => Duration::from_secs(15),
        ("thunderbolt/link-local", false) => Duration::from_secs(8),
        ("lan", true) => Duration::from_secs(25),
        ("lan", false) => Duration::from_secs(12),
        (_, true) => Duration::from_secs(20),
        (_, false) => Duration::from_secs(10),
    }
}

fn align_up_to_chunk(value: u64, chunk: u64) -> u64 {
    if chunk == 0 {
        return value;
    }
    let remainder = value % chunk;
    if remainder == 0 {
        value
    } else {
        value + (chunk - remainder)
    }
}

fn transport_label_for_addr(control_addr: &str) -> String {
    let host = addr_host(control_addr);
    let Ok(ip) = host.parse::<Ipv4Addr>() else {
        return "network".to_string();
    };
    if ip.octets()[0] == 169 && ip.octets()[1] == 254 {
        return "thunderbolt/link-local".to_string();
    }
    if is_private_ipv4(ip) {
        return "lan".to_string();
    }
    "network".to_string()
}

#[cfg(target_os = "windows")]
fn ps_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(target_os = "windows")]
fn query_windows_firewall_state() -> FirewallState {
    let cache = FIREWALL_STATE_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = cache.lock() {
        if let Some(cached) = guard.as_ref() {
            return cached.state.clone();
        }
    }

    let mut rules = vec![
        FIREWALL_RULE_CONTROL,
        FIREWALL_RULE_DISCOVERY,
        FIREWALL_RULE_TELEMETRY,
        FIREWALL_RULE_PUBLIC_API,
    ];
    if settings::multi_node_rpc_enabled() {
        rules.push(FIREWALL_RULE_RPC);
    }
    let mut missing = Vec::new();

    for rule in rules {
        let mut command = Command::new("netsh");
        command.args([
            "advfirewall",
            "firewall",
            "show",
            "rule",
            &format!("name={rule}"),
        ]);
        configure_background_command(&mut command);
        let output = command.output();
        match output {
            Ok(result) if result.status.success() => {
                let stdout = String::from_utf8_lossy(&result.stdout).to_ascii_lowercase();
                let profiles_line = stdout
                    .lines()
                    .find(|line| line.trim_start().starts_with("profiles:"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let allows_public =
                    profiles_line.contains("public") || profiles_line.contains("any");
                if !allows_public {
                    missing.push(rule.to_string());
                }
            }
            _ => missing.push(rule.to_string()),
        }
    }

    let state = if missing.is_empty() {
        FirewallState {
            status: Some("Direct LAN control is ready on this Windows node.".to_string()),
            action_required: false,
        }
    } else {
        FirewallState {
            status: Some(format!(
                "Firewall setup is needed for direct LAN control/RPC: {}",
                missing.join(", ")
            )),
            action_required: true,
        }
    };

    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedFirewallState {
            state: state.clone(),
            checked_at: Instant::now(),
        });
    }

    state
}

#[cfg(not(target_os = "windows"))]
fn query_windows_firewall_state() -> FirewallState {
    FirewallState::default()
}

fn firewall_state_for_runtime(_runtime_dir: &Path) -> FirewallState {
    query_windows_firewall_state()
}

#[cfg(target_os = "windows")]
fn configure_local_firewall(runtime_dir: &Path) -> Result<()> {
    let logs_dir = runtime_dir.join("logs");
    fs::create_dir_all(&logs_dir).ok();
    let script_path = logs_dir.join("configure-cluster-firewall.ps1");
    let script = format!(
        concat!(
            "$ErrorActionPreference='Stop'\n",
            "$rules = @(\n",
            "  @{{ Name = '{control}'; Protocol = 'TCP'; Port = 46211 }},\n",
            "  @{{ Name = '{rpc}'; Protocol = 'TCP'; Port = {rpc_port} }},\n",
            "  @{{ Name = '{discovery}'; Protocol = 'UDP'; Port = 46212 }},\n",
            "  @{{ Name = '{telemetry}'; Protocol = 'UDP'; Port = 46213 }},\n",
            "  @{{ Name = '{public_api}'; Protocol = 'TCP'; Port = 46310 }}\n",
            ")\n",
            "foreach ($rule in $rules) {{\n",
            "  Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue | Remove-NetFirewallRule | Out-Null\n",
            "  New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Action Allow -Protocol $rule.Protocol -LocalPort $rule.Port -Profile Any -RemoteAddress LocalSubnet | Out-Null\n",
            "}}\n"
        ),
        control = FIREWALL_RULE_CONTROL,
        rpc = FIREWALL_RULE_RPC,
        rpc_port = CLUSTER_AGENT_RPC_PORT,
        discovery = FIREWALL_RULE_DISCOVERY,
        telemetry = FIREWALL_RULE_TELEMETRY,
        public_api = FIREWALL_RULE_PUBLIC_API,
    );
    fs::write(&script_path, script)
        .with_context(|| format!("failed to write '{}'", script_path.display()))?;

    let command = format!(
        "Start-Process powershell -WindowStyle Hidden -Verb RunAs -Wait -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',{})",
        ps_single_quote(&script_path.display().to_string())
    );
    let mut process = Command::new("powershell");
    process
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-Command")
        .arg(command);
    configure_background_command(&mut process);
    let status = process
        .status()
        .context("failed to launch elevated firewall setup")?;
    if !status.success() {
        bail!("firewall setup was cancelled or failed");
    }

    if let Some(cache) = FIREWALL_STATE_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            *guard = None;
        }
    }

    let state = query_windows_firewall_state();
    if state.action_required {
        bail!(state
            .status
            .unwrap_or_else(|| "firewall setup did not complete".to_string()));
    }
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn configure_local_firewall(_runtime_dir: &Path) -> Result<()> {
    Ok(())
}

fn ensure_rpc_server_running(state: &Arc<AgentRuntimeState>) -> Result<()> {
    if !settings::multi_node_rpc_enabled() {
        bail!("multi-node RPC is disabled in settings");
    }

    stop_legacy_rpc_server_processes();

    if refresh_rpc_server_reachability() {
        let mut guard = state
            .rpc_server
            .lock()
            .map_err(|_| anyhow::anyhow!("rpc server mutex poisoned"))?;
        *guard = Some(RpcServerProcess {
            started_at: Instant::now(),
        });
        return Ok(());
    }

    if let Ok(guard) = state.rpc_server.lock() {
        if let Some(process) = guard.as_ref() {
            if process.started_at.elapsed() < Duration::from_secs(5) {
                drop(guard);
                wait_for_rpc_server_ready(Duration::from_secs(5))?;
                return Ok(());
            }
        }
    }

    spawn_managed_rpc_server(state)?;
    wait_for_rpc_server_ready(Duration::from_secs(5))
}

fn spawn_managed_rpc_server(state: &Arc<AgentRuntimeState>) -> Result<()> {
    let runtime_dir = state.runtime_dir.clone();
    clear_rpc_server_reachability_cache();
    thread::Builder::new()
        .name("engine-rpc-host".to_string())
        .spawn(move || {
            let result = ClusterApi::load(&runtime_dir).and_then(|api| {
                api.run_local_rpc_server("0.0.0.0", i32::from(CLUSTER_AGENT_RPC_PORT), 0)
            });
            if let Err(err) = result {
                eprintln!("cluster agent: embedded rpc host failed: {err:#}");
            }
        })
        .context("failed to launch embedded rpc host thread")?;
    let mut guard = state
        .rpc_server
        .lock()
        .map_err(|_| anyhow::anyhow!("rpc server mutex poisoned"))?;
    *guard = Some(RpcServerProcess {
        started_at: Instant::now(),
    });
    Ok(())
}

fn wait_for_rpc_server_ready(timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if refresh_rpc_server_reachability() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!("embedded rpc host did not become ready")
}

#[cfg(not(target_os = "windows"))]
fn sh_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

pub fn stop_local_support_processes(_runtime_dir: &Path) {
    clear_rpc_server_reachability_cache();
    stop_legacy_rpc_server_processes();
}

fn split_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn addr_host(value: &str) -> String {
    value
        .rsplit_once(':')
        .map(|(host, _)| host.to_string())
        .unwrap_or_else(|| value.to_string())
}

fn restart_matching_peer_rpc_servers(
    peers: &Arc<Mutex<HashMap<String, DiscoveredPeer>>>,
    rpc_servers: &str,
) -> Result<()> {
    let endpoints = split_csv(rpc_servers);
    if endpoints.is_empty() {
        return Ok(());
    }

    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    if let Ok(guard) = peers.lock() {
        for endpoint in endpoints {
            let endpoint_norm = endpoint.to_ascii_lowercase();
            let endpoint_host = addr_host(&endpoint_norm);
            for peer in guard.values() {
                let advertised_match = peer
                    .info
                    .advertised_rpc_endpoint
                    .as_ref()
                    .map(|value| value.to_ascii_lowercase() == endpoint_norm)
                    .unwrap_or(false);
                let direct_match = peer
                    .info
                    .rpc_endpoint
                    .as_ref()
                    .map(|value| value.to_ascii_lowercase() == endpoint_norm)
                    .unwrap_or(false);
                let control_host_match = peer_control_addr_candidates(&peer.info)
                    .iter()
                    .any(|value| addr_host(&value.to_ascii_lowercase()) == endpoint_host);

                if !(advertised_match || direct_match || control_host_match) {
                    continue;
                }

                let control_addr = preferred_peer_control_addr(&peer.info);
                if seen.insert(control_addr.clone()) {
                    targets.push(control_addr);
                }
                break;
            }
        }
    }

    for control_addr in targets {
        eprintln!("cluster agent: ensuring peer rpc server via {control_addr}");
        AgentClient::new(control_addr.clone())
            .restart_rpc_server()
            .with_context(|| format!("failed to ensure peer rpc server via '{control_addr}'"))?;
    }

    Ok(())
}

fn rpc_loopback_endpoint() -> String {
    format!("127.0.0.1:{CLUSTER_AGENT_RPC_PORT}")
}

fn rpc_bind_endpoint() -> String {
    format!("0.0.0.0:{CLUSTER_AGENT_RPC_PORT}")
}

fn rpc_server_is_reachable() -> bool {
    cached_rpc_server_reachability(false)
}

fn refresh_rpc_server_reachability() -> bool {
    cached_rpc_server_reachability(true)
}

fn clear_rpc_server_reachability_cache() {
    if let Ok(mut guard) = RPC_SERVER_REACHABILITY_CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
        *guard = None;
    }
}

fn cached_rpc_server_reachability(force_refresh: bool) -> bool {
    let cache = RPC_SERVER_REACHABILITY_CACHE.get_or_init(|| Mutex::new(None));
    if !force_refresh {
        if let Ok(guard) = cache.lock() {
            if let Some(cached) = guard.as_ref() {
                if cached.checked_at.elapsed() <= RPC_SERVER_REACHABILITY_CACHE_TTL {
                    return cached.reachable;
                }
            }
        }
    }

    let reachable = rpc_endpoint_is_reachable(&rpc_loopback_endpoint());
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedRpcServerReachability {
            reachable,
            checked_at: Instant::now(),
        });
    }
    reachable
}

fn rpc_server_snapshot(state: &Arc<AgentRuntimeState>) -> (Option<String>, bool) {
    if !settings::multi_node_rpc_enabled() {
        if let Ok(mut guard) = state.rpc_server.lock() {
            *guard = None;
        }
        return (None, false);
    }

    let running = rpc_server_is_reachable();
    let endpoint = if running {
        Some(rpc_bind_endpoint())
    } else {
        None
    };

    if running {
        return (endpoint, true);
    }

    if let Ok(mut guard) = state.rpc_server.lock() {
        if guard
            .as_ref()
            .is_some_and(|process| process.started_at.elapsed() > Duration::from_secs(5))
        {
            *guard = None;
        }
    }

    (None, false)
}

#[cfg(target_os = "windows")]
fn stop_legacy_rpc_server_processes() {
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::All, true);
    for process in system.processes().values() {
        let name = process.name().to_string_lossy().to_ascii_lowercase();
        if name == "rpc-server" || name == "rpc-server.exe" {
            let _ = process.kill();
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn stop_legacy_rpc_server_processes() {
    let _ = Command::new("pkill").args(["-x", "rpc-server"]).output();
    let _ = Command::new("pkill").args(["-f", "/rpc-server"]).output();
}

fn configure_background_command(_command: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        _command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn write_message<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let payload = serde_json::to_vec(value).context("failed to encode message")?;
    let len = u32::try_from(payload.len()).context("message too large")?;
    stream
        .write_all(&len.to_le_bytes())
        .context("failed to write message length")?;
    stream
        .write_all(&payload)
        .context("failed to write message payload")?;
    stream.flush().ok();
    Ok(())
}

fn resolve_model_artifact_path(
    models_dir: &Path,
    folder_name: &str,
    relative_path: &str,
) -> Result<PathBuf> {
    let folder_name = sanitize_folder_name(folder_name);
    let relative = Path::new(relative_path);
    let mut sanitized_relative = PathBuf::new();
    let mut extension = None::<String>;
    for component in relative.components() {
        match component {
            Component::Normal(value) => {
                let text = value.to_string_lossy();
                if text.trim().is_empty() {
                    bail!("artifact path contains an empty segment");
                }
                extension = Path::new(text.as_ref())
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(|value| value.to_ascii_lowercase());
                sanitized_relative.push(value);
            }
            _ => bail!("artifact path must stay inside the model package"),
        }
    }
    if sanitized_relative.as_os_str().is_empty() {
        bail!("artifact path cannot be empty");
    }
    if !matches!(extension.as_deref(), Some("gguf") | Some("bin")) {
        bail!("only model artifact files (.gguf/.bin) can be transferred");
    }
    Ok(models_dir.join(folder_name).join(sanitized_relative))
}

fn read_message<T: DeserializeOwned>(stream: &mut TcpStream) -> Result<T> {
    let mut len_bytes = [0u8; 4];
    stream
        .read_exact(&mut len_bytes)
        .context("failed to read message length")?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .context("failed to read message payload")?;
    match serde_json::from_slice(&payload) {
        Ok(value) => Ok(value),
        Err(err) => {
            let preview = String::from_utf8_lossy(&payload);
            Err(anyhow::anyhow!(
                "failed to decode message len={} error={} preview={}",
                len,
                err,
                preview.chars().take(512).collect::<String>()
            ))
        }
    }
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn bind_addr_from_local_control_addr(control_addr: &str) -> String {
    let (host, port) = control_addr
        .rsplit_once(':')
        .unwrap_or(("127.0.0.1", "46211"));
    let host = match host.trim() {
        "" | "localhost" => "127.0.0.1",
        value => value,
    };
    if matches!(host, "0.0.0.0" | "::" | "[::]" | "127.0.0.1" | "::1" | "[::1]") {
        return format!("0.0.0.0:{port}");
    }
    if host
        .parse::<Ipv4Addr>()
        .ok()
        .is_some_and(|ip| preferred_interface_candidates().iter().any(|candidate| candidate.ip == ip))
    {
        return format!("0.0.0.0:{port}");
    }
    format!("{host}:{port}")
}

fn normalize_public_api_bind_addr(value: &str) -> Result<String> {
    let trimmed = value.trim();
    let (host, port_text) = trimmed
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("managed HTTP bind address must include host and port"))?;
    if host.trim().is_empty() {
        bail!("managed HTTP bind host is required");
    }
    let port = port_text
        .trim()
        .parse::<u16>()
        .context("managed HTTP port must be a valid number")?;
    let host = normalize_public_api_bind_host(host.trim())?;
    Ok(format!("{host}:{port}"))
}

fn normalize_public_api_bind_host(host: &str) -> Result<String> {
    let trimmed = host.trim();
    if matches!(trimmed, "127.0.0.1" | "localhost" | "::1" | "[::1]") {
        return Ok("127.0.0.1".to_string());
    }
    if matches!(trimmed, "0.0.0.0" | "::" | "[::]") {
        return preferred_local_network_bind_host().ok_or_else(|| {
            anyhow::anyhow!("no local-network interface is available for managed HTTP binding")
        });
    }
    if let Ok(ipv4) = trimmed.parse::<Ipv4Addr>() {
        if is_local_network_ipv4(ipv4) {
            return Ok(ipv4.to_string());
        }
        bail!("managed HTTP bind host must stay on 127.0.0.1 or a local-network IPv4 address");
    }
    let ipv6_text = trimmed.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ipv6) = ipv6_text.parse::<Ipv6Addr>() {
        if is_local_network_ipv6(ipv6) {
            return Ok(format!("[{ipv6}]"));
        }
        bail!("managed HTTP bind host must stay on 127.0.0.1 or a local-network IPv6 address");
    }
    bail!("managed HTTP bind host must be 127.0.0.1 or a concrete local-network IP address")
}

pub fn available_public_api_bind_hosts() -> Vec<PublicApiBindHostOption> {
    let mut options = vec![
        PublicApiBindHostOption {
            host: "127.0.0.1".to_string(),
            label: "127.0.0.1 (this machine only)".to_string(),
        },
        PublicApiBindHostOption {
            host: "0.0.0.0".to_string(),
            label: "local network (auto-detect)".to_string(),
        },
    ];
    options.extend(
        preferred_interface_candidates()
            .into_iter()
            .map(|candidate| {
                let descriptor = if candidate.description.trim().is_empty() {
                    candidate.name.trim().to_string()
                } else if candidate.name.trim().is_empty()
                    || candidate
                        .name
                        .trim()
                        .eq_ignore_ascii_case(candidate.description.trim())
                {
                    candidate.description.trim().to_string()
                } else {
                    format!(
                        "{} | {}",
                        candidate.description.trim(),
                        candidate.name.trim()
                    )
                };
                let label = if descriptor.is_empty() {
                    candidate.ip.to_string()
                } else {
                    format!("{} ({descriptor})", candidate.ip)
                };
                PublicApiBindHostOption {
                    host: candidate.ip.to_string(),
                    label,
                }
            }),
    );
    options.dedup_by(|lhs, rhs| lhs.host == rhs.host);
    options
}

fn preferred_local_network_bind_host() -> Option<String> {
    preferred_paired_link_local_host(true)
        .or_else(|| preferred_direct_link_host())
        .or_else(|| {
            preferred_interface_candidates()
                .into_iter()
                .map(|candidate| candidate.ip.to_string())
                .next()
        })
        .or_else(|| default_route_local_network_host(true))
}

fn advertised_control_addrs_for_bind(bind_addr: &str) -> Vec<String> {
    let Some((_, port)) = bind_addr.rsplit_once(':') else {
        return Vec::new();
    };
    let mut hosts = Vec::new();
    let bind_host = bind_addr
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(bind_addr)
        .trim()
        .to_string();
    if is_cluster_reachable_host(&bind_host) {
        hosts.push(bind_host);
    }
    if let Some(host) = std::env::var("ENGINE_CLUSTER_ADVERTISE_HOST")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && is_cluster_reachable_host(value))
    {
        hosts.push(host);
    }
    if let Some(host) = preferred_paired_link_local_host(false) {
        hosts.push(host);
    }
    if let Some(host) = preferred_direct_link_host() {
        hosts.push(host);
    }
    hosts.extend(
        preferred_interface_candidates()
            .into_iter()
            .map(|candidate| candidate.ip.to_string()),
    );
    if let Some(host) = default_route_local_network_host(false) {
        hosts.push(host);
    }
    dedup_sorted_control_addrs(
        hosts
            .into_iter()
            .map(|host| format!("{host}:{port}"))
            .collect(),
    )
}

fn advertised_control_addr_for_bind(bind_addr: &str) -> Option<String> {
    advertised_control_addrs_for_bind(bind_addr).into_iter().next()
}

fn advertised_rpc_endpoint_for_bind(
    rpc_endpoint: Option<&str>,
    advertised_control_addr: Option<&str>,
) -> Option<String> {
    let rpc_endpoint = rpc_endpoint?;
    let (_, rpc_port) = rpc_endpoint.rsplit_once(':')?;
    let (host, _) = advertised_control_addr?.rsplit_once(':')?;
    Some(format!("{host}:{rpc_port}"))
}

fn advertised_public_api_addr_for_bind(
    public_api_bind_addr: Option<&str>,
    advertised_control_addr: Option<&str>,
) -> Option<String> {
    let bind_addr = public_api_bind_addr?;
    let (bind_host, port) = bind_addr.rsplit_once(':')?;
    let bind_host = bind_host.trim();
    if matches!(bind_host, "127.0.0.1" | "localhost" | "::1" | "[::1]") {
        return None;
    }
    if matches!(bind_host, "0.0.0.0" | "::" | "[::]") {
        let (host, _) = advertised_control_addr?.rsplit_once(':')?;
        return Some(format!("{host}:{port}"));
    }
    if is_cluster_reachable_host(bind_host) {
        return Some(format!("{bind_host}:{port}"));
    }
    None
}

fn advertised_host_for_bind(bind_addr: &str) -> Option<String> {
    let bind_host = bind_addr
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(bind_addr);
    if interface_debug_enabled() {
        eprintln!("advertised_host_for_bind bind_addr={bind_addr} bind_host={bind_host}");
    }
    if is_cluster_reachable_host(bind_host) {
        if interface_debug_enabled() {
            eprintln!("advertised_host_for_bind using bind_host={bind_host}");
        }
        return Some(bind_host.to_string());
    }

    if let Some(host) = std::env::var("ENGINE_CLUSTER_ADVERTISE_HOST")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && is_cluster_reachable_host(value))
    {
        if interface_debug_enabled() {
            eprintln!("advertised_host_for_bind using env host={host}");
        }
        return Some(host);
    }

    let selected = preferred_paired_link_local_host(false)
        .or_else(|| preferred_direct_link_host())
        .or_else(|| {
            preferred_interface_candidates()
                .into_iter()
                .map(|candidate| candidate.ip.to_string())
                .next()
        })
        .or_else(|| default_route_local_network_host(false));
    if interface_debug_enabled() {
        eprintln!("advertised_host_for_bind selected={selected:?}");
    }
    selected
}

fn preferred_direct_link_host() -> Option<String> {
    let selected = preferred_interface_candidates()
        .into_iter()
        .find(|candidate| is_link_local_ipv4(candidate.ip))
        .map(|candidate| candidate.ip.to_string());
    if interface_debug_enabled() {
        eprintln!("preferred_direct_link_host={selected:?}");
    }
    selected
}

fn preferred_paired_link_local_host(bracket_ipv6: bool) -> Option<String> {
    settings::load_controller_settings()?
        .paired_peers
        .into_iter()
        .filter_map(|peer| {
            let host = addr_host(&peer.control_addr);
            let ip = host.parse::<Ipv4Addr>().ok()?;
            if !is_link_local_ipv4(ip) {
                return None;
            }
            local_network_host_for_route(&host, bracket_ipv6)
        })
        .next()
}

fn default_route_local_network_host(bracket_ipv6: bool) -> Option<String> {
    local_network_host_for_route("8.8.8.8", bracket_ipv6)
}

fn local_network_host_for_route(remote_host: &str, bracket_ipv6: bool) -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect(format!("{remote_host}:80")).ok()?;
    let local_addr = socket.local_addr().ok()?;
    match local_addr.ip() {
        IpAddr::V4(ip) if is_local_network_ipv4(ip) => Some(ip.to_string()),
        IpAddr::V6(ip) if is_local_network_ipv6(ip) => {
            if bracket_ipv6 {
                Some(format!("[{ip}]"))
            } else {
                Some(ip.to_string())
            }
        }
        _ => None,
    }
}

fn is_cluster_reachable_host(host: &str) -> bool {
    let trimmed = host.trim();
    if trimmed.is_empty()
        || matches!(
            trimmed,
            "0.0.0.0" | "::" | "[::]" | "127.0.0.1" | "localhost" | "::1" | "[::1]"
        )
    {
        return false;
    }
    if let Ok(ipv4) = trimmed.parse::<Ipv4Addr>() {
        return is_local_network_ipv4(ipv4);
    }
    let ipv6_text = trimmed.trim_start_matches('[').trim_end_matches(']');
    ipv6_text
        .parse::<Ipv6Addr>()
        .map(is_local_network_ipv6)
        .unwrap_or(false)
}

fn preferred_interface_candidates() -> Vec<InterfaceCandidate> {
    let cache = INTERFACE_CANDIDATE_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cache.lock() {
        if let Some((at, cached)) = &*guard {
            if at.elapsed() < Duration::from_secs(10) {
                return cached.clone();
            }
        }
        let discovered = discover_interface_candidates();
        *guard = Some((Instant::now(), discovered.clone()));
        return discovered;
    }
    discover_interface_candidates()
}

fn discover_interface_candidates() -> Vec<InterfaceCandidate> {
    #[cfg(target_os = "windows")]
    {
        let mut candidates = discover_windows_interface_candidates();
        candidates.sort_by(|lhs, rhs| {
            rhs.score
                .cmp(&lhs.score)
                .then(lhs.ip.octets().cmp(&rhs.ip.octets()))
        });
        candidates.dedup_by(|lhs, rhs| lhs.ip == rhs.ip);
        return candidates;
    }
    #[cfg(target_os = "macos")]
    {
        let mut candidates = discover_macos_interface_candidates();
        candidates.sort_by(|lhs, rhs| {
            rhs.score
                .cmp(&lhs.score)
                .then(lhs.ip.octets().cmp(&rhs.ip.octets()))
        });
        candidates.dedup_by(|lhs, rhs| lhs.ip == rhs.ip);
        return candidates;
    }
    #[allow(unreachable_code)]
    Vec::new()
}

#[cfg(target_os = "windows")]
fn discover_windows_interface_candidates() -> Vec<InterfaceCandidate> {
    let mut output_len = 15_000u32;
    let mut buffer = vec![0u8; output_len as usize];
    let family = AF_INET as u32;
    let flags = GAA_FLAG_INCLUDE_PREFIX;
    let result = unsafe {
        GetAdaptersAddresses(
            family,
            flags,
            std::ptr::null_mut(),
            buffer.as_mut_ptr() as *mut IP_ADAPTER_ADDRESSES_LH,
            &mut output_len,
        )
    };
    if result == 111u32 {
        buffer.resize(output_len as usize, 0);
    }
    let result = unsafe {
        GetAdaptersAddresses(
            family,
            flags,
            std::ptr::null_mut(),
            buffer.as_mut_ptr() as *mut IP_ADAPTER_ADDRESSES_LH,
            &mut output_len,
        )
    };
    if result != 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut adapter = buffer.as_ptr() as *const IP_ADAPTER_ADDRESSES_LH;
    while !adapter.is_null() {
        let friendly_name = unsafe { wide_ptr_to_string((*adapter).FriendlyName) };
        let description = unsafe { wide_ptr_to_string((*adapter).Description) };
        let metadata = format!("{friendly_name} {description}").to_ascii_lowercase();
        let is_virtual = metadata.contains("virtual")
            || metadata.contains("hyper-v")
            || metadata.contains("virtualbox")
            || metadata.contains("vmware")
            || metadata.contains("bluetooth")
            || metadata.contains("wsl");
        let is_up = unsafe { (*adapter).OperStatus == 1 };
        if !is_up || is_virtual {
            adapter = unsafe { (*adapter).Next };
            continue;
        }

        let mut unicast = unsafe { (*adapter).FirstUnicastAddress };
        while !unicast.is_null() {
            let sockaddr = unsafe { (*unicast).Address.lpSockaddr };
            if !sockaddr.is_null() && unsafe { (*sockaddr).sa_family } == AF_INET {
                let sockaddr_in = sockaddr as *const SOCKADDR_IN;
                let raw = unsafe { (*sockaddr_in).sin_addr.S_un.S_addr };
                let ip = Ipv4Addr::from(u32::from_be(raw));
                if is_candidate_ipv4(ip) {
                    out.push(InterfaceCandidate {
                        ip,
                        name: friendly_name.clone(),
                        description: description.clone(),
                        score: score_interface_candidate(&friendly_name, &description, ip),
                    });
                }
            }
            unicast = unsafe { (*unicast).Next };
        }

        adapter = unsafe { (*adapter).Next };
    }
    out
}

#[cfg(target_os = "windows")]
unsafe fn wide_ptr_to_string(ptr: *mut u16) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    while *ptr.add(len) != 0 {
        len += 1;
    }
    String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len))
}

#[cfg(target_os = "macos")]
fn macos_default_route_interface() -> Option<String> {
    let output = run_command_output("/usr/sbin/route", &["-n", "get", "default"])?;
    output.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("interface:")
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

#[cfg(target_os = "macos")]
fn discover_macos_interface_candidates() -> Vec<InterfaceCandidate> {
    let hardware_ports = run_command_output("/usr/sbin/networksetup", &["-listallhardwareports"])
        .unwrap_or_default();
    let port_labels = parse_macos_hardware_ports(&hardware_ports);
    let default_interface = macos_default_route_interface();
    let mut out = Vec::new();
    let mut ifap: *mut libc::ifaddrs = std::ptr::null_mut();
    if unsafe { libc::getifaddrs(&mut ifap) } != 0 {
        return out;
    }

    let mut cursor = ifap;
    while !cursor.is_null() {
        let item = unsafe { &*cursor };
        let flags = item.ifa_flags as i32;
        let sockaddr = item.ifa_addr;
        if !sockaddr.is_null()
            && unsafe { (*sockaddr).sa_family as i32 } == libc::AF_INET
            && (flags & libc::IFF_UP) != 0
            && (flags & libc::IFF_LOOPBACK) == 0
        {
            let name = unsafe { CStr::from_ptr(item.ifa_name) }
                .to_string_lossy()
                .into_owned();
            let addr = sockaddr as *const libc::sockaddr_in;
            let ip = Ipv4Addr::from(u32::from_be(unsafe { (*addr).sin_addr.s_addr }));
            if is_candidate_ipv4(ip) {
                let description = port_labels.get(&name).cloned().unwrap_or_default();
                let mut score = score_interface_candidate(&name, &description, ip);
                if default_interface.as_deref() == Some(name.as_str()) {
                    score += 5_000;
                }
                if interface_debug_enabled() {
                    eprintln!(
                        "macos_interface_candidate name={name} description={description} ip={ip} score={score}"
                    );
                }
                out.push(InterfaceCandidate {
                    ip,
                    name: name.clone(),
                    description: description.clone(),
                    score,
                });
            }
        }
        cursor = item.ifa_next;
    }

    unsafe { libc::freeifaddrs(ifap) };
    out
}

#[cfg(target_os = "macos")]
fn parse_macos_hardware_ports(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut current_port = String::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Hardware Port:") {
            current_port = value.trim().to_string();
        } else if let Some(value) = trimmed.strip_prefix("Device:") {
            let device = value.trim().to_string();
            if !device.is_empty() && !current_port.is_empty() {
                out.insert(device, current_port.clone());
            }
        }
    }
    out
}

fn is_candidate_ipv4(ip: Ipv4Addr) -> bool {
    is_local_network_ipv4(ip)
}

fn is_link_local_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 169 && octets[1] == 254
}

fn is_local_network_ipv4(ip: Ipv4Addr) -> bool {
    is_private_ipv4(ip) || is_link_local_ipv4(ip)
}

fn is_local_network_ipv6(ip: Ipv6Addr) -> bool {
    ip.is_unique_local() || ip.is_unicast_link_local()
}

fn score_interface_candidate(name: &str, description: &str, ip: Ipv4Addr) -> i32 {
    let text = format!("{name} {description}").to_ascii_lowercase();
    let mut score = 0;
    if text.contains("virtual")
        || text.contains("hyper-v")
        || text.contains("virtualbox")
        || text.contains("vmware")
        || text.contains("wsl")
        || text.contains("bluetooth")
        || text.contains("loopback")
        || text.contains("awdl")
        || text.contains("llw")
        || text.contains("utun")
    {
        score -= 2_000;
    }
    if text.contains("thunderbolt") || text.contains("bridge0") || text.contains("ethernet 3") {
        score += 5_000;
    } else if text.contains("ethernet") {
        score += 3_000;
    } else if text.contains("wi-fi")
        || text.contains("wifi")
        || text.contains("wireless")
        || text.contains("wlan")
    {
        score += 2_000;
    }
    if is_link_local_ipv4(ip) {
        score += 4_000;
    } else if is_private_ipv4(ip) {
        score += 1_000;
    }
    score
}

fn interface_debug_enabled() -> bool {
    std::env::var("ENGINE_DEBUG_INTERFACES")
        .ok()
        .map(|value| value == "1")
        .unwrap_or(false)
}

#[cfg(not(target_os = "windows"))]
fn run_command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}
