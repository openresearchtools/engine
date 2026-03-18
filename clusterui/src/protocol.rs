use crate::catalog::ManagedModelEntry;
use crate::cluster_api::{
    AudioRawRequest, ChatRequest, CreateInstanceParams, DeviceInfo, EmbeddingsRequest,
    ExecutionGroupInfo, InstanceInfo, JsonResult, NativeAudioTranscriptionRequest, NodeInfo,
    RerankRequest, RetentionMode, TextGenerationResult, VlmRequest,
};
use crate::model_metadata::ModelFileMetadata;
use crate::model_store::{ModelArtifact, ModelPackage};
use serde::{Deserialize, Serialize};

pub const CLUSTER_AGENT_CONTROL_PORT: u16 = 46211;
pub const CLUSTER_AGENT_DISCOVERY_PORT: u16 = 46212;
pub const CLUSTER_AGENT_TELEMETRY_PORT: u16 = 46213;
pub const CLUSTER_AGENT_RPC_PORT: u16 = 46214;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: String,
    pub display_name: String,
    pub os_name: String,
    pub arch: String,
    pub control_addr: String,
    pub advertised_control_addr: Option<String>,
    pub rpc_endpoint: Option<String>,
    pub advertised_rpc_endpoint: Option<String>,
    pub rpc_running: bool,
    pub trusted: bool,
    pub last_seen_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub node: NodeInfo,
    pub control_addr: String,
    pub advertised_control_addr: Option<String>,
    pub runtime_dir: String,
    #[serde(default)]
    pub models_dir: String,
    pub rpc_endpoint: Option<String>,
    pub advertised_rpc_endpoint: Option<String>,
    pub rpc_running: bool,
    #[serde(default)]
    pub public_api_addr: Option<String>,
    #[serde(default)]
    pub advertised_public_api_addr: Option<String>,
    #[serde(default)]
    pub public_api_running: bool,
    pub firewall_status: Option<String>,
    pub firewall_action_required: bool,
    pub devices: Vec<DeviceInfo>,
    pub execution_groups: Vec<ExecutionGroupInfo>,
    pub instances: Vec<InstanceInfo>,
    #[serde(default)]
    pub link_metrics: Vec<LinkMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub node: NodeInfo,
    pub control_addr: String,
    pub advertised_control_addr: Option<String>,
    pub unix_ms: u64,
    pub process_memory_bytes: u64,
    pub process_virtual_memory_bytes: u64,
    pub process_cpu_percent: f32,
    pub system_memory_total_bytes: u64,
    pub system_memory_available_bytes: u64,
    pub rpc_running: bool,
    pub public_api_running: bool,
    pub devices: Vec<DeviceInfo>,
    pub instances: Vec<InstanceInfo>,
    #[serde(default)]
    pub link_metrics: Vec<LinkMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkMetrics {
    pub peer_control_addr: String,
    pub transport: String,
    #[serde(default)]
    pub probe_kind: String,
    pub payload_bytes: u64,
    pub rounds: u32,
    pub latency_ms: f64,
    pub goodput_mbps: f64,
    #[serde(default)]
    pub duration_ms: f64,
    pub unix_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryAnnouncement {
    pub protocol_version: u32,
    pub node: NodeInfo,
    pub control_port: u16,
    #[serde(default)]
    pub advertised_control_addr: Option<String>,
    #[serde(default)]
    pub advertised_rpc_endpoint: Option<String>,
    #[serde(default)]
    pub rpc_running: bool,
    pub announced_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DiscoveryMode {
    Off,
    KnownPeers,
    Pairing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStatus {
    pub mode: DiscoveryMode,
    pub active: bool,
    pub expires_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingRequestInfo {
    pub request_id: String,
    pub request_code: String,
    pub requester_node_id: String,
    pub requester_display_name: String,
    pub requester_os_name: String,
    pub requester_arch: String,
    pub requester_control_addr: String,
    pub requested_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathStat {
    pub path: String,
    pub exists: bool,
    pub size_bytes: u64,
    #[serde(default)]
    pub model_metadata: Option<ModelFileMetadata>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlacementStrategy {
    SingleNode,
    LocalSplit,
    HybridTwoNode,
    HybridMultiNode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub owner_control_addr: String,
    pub owner_display_name: String,
    pub execution_group_id: String,
    pub rpc_servers: String,
    #[serde(default)]
    pub display_label: String,
    pub strategy: PlacementStrategy,
    pub device_count: i32,
    pub remote_node_count: i32,
    pub estimated_required_bytes: u64,
    pub estimated_group_free_bytes: u64,
    pub reusable_instance_id: Option<i64>,
    pub ready_now: bool,
    pub requires_eviction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledInstance {
    pub owner_control_addr: String,
    pub owner_display_name: String,
    pub instance_id: i64,
    pub execution_group_id: String,
    pub rpc_servers: String,
    pub strategy: PlacementStrategy,
    pub reused_existing: bool,
    pub waited_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackageNodeAvailability {
    pub control_addr: String,
    pub display_name: String,
    pub package_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileNodeAvailability {
    pub control_addr: String,
    pub display_name: String,
    pub package_path: String,
    pub full_path: String,
    #[serde(default)]
    pub managed_model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterModelArtifactInfo {
    pub artifact: ModelArtifact,
    #[serde(default)]
    pub available_on: Vec<ModelFileNodeAvailability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterModelPackageInfo {
    pub package: ModelPackage,
    #[serde(default)]
    pub available_on: Vec<ModelPackageNodeAvailability>,
    #[serde(default)]
    pub model_file_availability: Vec<ClusterModelArtifactInfo>,
    #[serde(default)]
    pub mmproj_file_availability: Vec<ClusterModelArtifactInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedClusterInstance {
    pub owner_control_addr: String,
    pub owner_display_name: String,
    pub instance_id: i64,
    pub model_id: Option<String>,
    pub auto_loaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicApiConfig {
    pub enabled: bool,
    pub bind_addr: String,
    pub allow_cors: bool,
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    #[serde(default)]
    pub allowed_client_ips: Vec<String>,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicApiConfigUpdate {
    pub enabled: bool,
    pub bind_addr: String,
    pub allow_cors: bool,
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    #[serde(default)]
    pub allowed_client_ips: Vec<String>,
    pub api_key: Option<String>,
    pub clear_api_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicApiStatus {
    pub enabled: bool,
    pub running: bool,
    pub bind_addr: String,
    pub effective_bind_addr: Option<String>,
    pub advertised_addr: Option<String>,
    pub allow_cors: bool,
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    #[serde(default)]
    pub allowed_client_ips: Vec<String>,
    pub api_key_present: bool,
    pub api_key_fingerprint: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentRequest {
    Ping,
    GetSnapshot {
        rpc_servers: Option<String>,
    },
    GetLocalTelemetry,
    GetClusterTelemetry,
    RunLinkBenchmarks {
        full: bool,
    },
    MeasureLinkTo {
        control_addr: String,
        full: bool,
    },
    ListPeers,
    AddPeer {
        control_addr: String,
    },
    RemovePeer {
        control_addr: String,
    },
    StartDiscovery {
        mode: DiscoveryMode,
        seconds: u64,
    },
    GetDiscoveryStatus,
    ListPairingRequests,
    RequestPairing {
        control_addr: String,
    },
    AcceptPairingRequest {
        request_id: String,
    },
    DeclinePairingRequest {
        request_id: String,
    },
    SubmitPairingRequest {
        request: PairingRequestInfo,
    },
    FinalizePairing {
        request_id: String,
        peer: PeerInfo,
        shared_token: String,
    },
    RestartRpcServer,
    ConfigureFirewall,
    StatPaths {
        paths: Vec<String>,
    },
    PlanInstance {
        params: CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
    },
    ListPlacementCandidates {
        params: CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
    },
    ScheduleInstance {
        params: CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
        load_immediately: bool,
    },
    ResolveClusterInstance {
        name: String,
        load_if_managed: bool,
    },
    ListManagedModels,
    ListClusterManagedModels,
    ResolveManagedModel {
        model_id: String,
    },
    ResolveClusterManagedModel {
        model_id: String,
    },
    ListModelPackages,
    ListClusterModelPackages,
    StreamModelArtifact {
        folder_name: String,
        relative_path: String,
    },
    ReceiveModelArtifact {
        folder_name: String,
        relative_path: String,
        size_bytes: u64,
    },
    GetPublicApiStatus,
    UpdatePublicApiConfig {
        update: PublicApiConfigUpdate,
    },
    CreateInstance {
        params: CreateInstanceParams,
    },
    LoadInstance {
        instance_id: i64,
    },
    UnloadInstance {
        instance_id: i64,
    },
    RemoveInstance {
        instance_id: i64,
    },
    SetRetentionMode {
        instance_id: i64,
        retention_mode: RetentionMode,
    },
    ChatComplete {
        request: ChatRequest,
    },
    VlmComplete {
        request: VlmRequest,
    },
    Embeddings {
        request: EmbeddingsRequest,
    },
    Rerank {
        request: RerankRequest,
    },
    AudioTranscriptionsRaw {
        request: AudioRawRequest,
    },
    AudioTranscriptionsNative {
        request: NativeAudioTranscriptionRequest,
    },
    LinkProbe {
        bytes: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentResponse {
    Pong,
    Snapshot {
        snapshot: NodeSnapshot,
    },
    LocalTelemetry {
        snapshot: TelemetrySnapshot,
    },
    ClusterTelemetry {
        snapshots: Vec<TelemetrySnapshot>,
    },
    Peers {
        peers: Vec<PeerInfo>,
    },
    DiscoveryStatus {
        status: DiscoveryStatus,
    },
    PairingRequests {
        requests: Vec<PairingRequestInfo>,
    },
    PathStats {
        stats: Vec<PathStat>,
    },
    PlacementPlan {
        plan: PlacementPlan,
    },
    PlacementCandidates {
        plans: Vec<PlacementPlan>,
    },
    ScheduledInstance {
        scheduled: ScheduledInstance,
    },
    ResolvedClusterInstance {
        resolved: ResolvedClusterInstance,
    },
    ManagedModels {
        models: Vec<ManagedModelEntry>,
    },
    ManagedModel {
        model: Option<ManagedModelEntry>,
    },
    ModelPackages {
        packages: Vec<ModelPackage>,
    },
    ClusterModelPackages {
        packages: Vec<ClusterModelPackageInfo>,
    },
    ModelArtifactTransferReady {
        size_bytes: u64,
    },
    ModelArtifactTransferSkipped {
        reason: String,
    },
    PublicApiStatus {
        status: PublicApiStatus,
    },
    CreatedInstance {
        instance_id: i64,
    },
    ChatResult {
        result: TextGenerationResult,
    },
    VlmResult {
        result: TextGenerationResult,
    },
    JsonResult {
        result: JsonResult,
    },
    LinkMetrics {
        metrics: LinkMetrics,
    },
    LinkProbeAck {
        bytes: u64,
        checksum: u64,
    },
    Ok,
    Error {
        message: String,
    },
}
