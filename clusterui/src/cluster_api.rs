use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use std::thread;

#[cfg(target_os = "windows")]
use std::ffi::c_void;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;

#[repr(C)]
pub struct llama_server_cluster {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_node_info {
    node_id: *mut c_char,
    display_name: *mut c_char,
    os_name: *mut c_char,
    arch: *mut c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_device_info {
    bridge_device_index: i32,
    r#type: i32,
    memory_free: u64,
    memory_total: u64,
    backend: *mut c_char,
    name: *mut c_char,
    description: *mut c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_execution_group_info {
    id: *mut c_char,
    label: *mut c_char,
    backend_summary: *mut c_char,
    devices_csv: *mut c_char,
    device_count: i32,
    uses_local_split: i32,
    memory_free: u64,
    memory_total: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_instance_params {
    name: *const c_char,
    model_path: *const c_char,
    mmproj_path: *const c_char,
    diarization_model_path: *const c_char,
    execution_group_id: *const c_char,
    rpc_servers: *const c_char,
    manual_devices_csv: *const c_char,
    manual_tensor_split: *const c_char,
    retention_mode: i32,
    load_on_demand_grace_seconds: i32,
    embedding: i32,
    reranking: i32,
    model_kind: i32,
    allow_cpu: i32,
    allow_integrated_gpu: i32,
    n_ctx: i32,
    n_batch: i32,
    n_ubatch: i32,
    n_parallel: i32,
    n_threads: i32,
    n_threads_batch: i32,
    n_gpu_layers: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_instance_info {
    instance_id: i64,
    name: *mut c_char,
    model_path: *mut c_char,
    mmproj_path: *mut c_char,
    diarization_model_path: *mut c_char,
    execution_group_id: *mut c_char,
    rpc_servers: *mut c_char,
    retention_mode: i32,
    load_on_demand_grace_seconds: i32,
    model_kind: i32,
    state: i32,
    active_request_count: i32,
    queued_request_count: i32,
    n_parallel: i32,
    grace_deadline_unix_ms: i64,
    last_error: *mut c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_chat_request {
    instance_id: i64,
    prompt: *const c_char,
    n_predict: i32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    repeat_last_n: i32,
    repeat_penalty: f32,
    reasoning: *const c_char,
    reasoning_budget: i32,
    reasoning_format: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_vlm_request {
    instance_id: i64,
    prompt: *const c_char,
    image_bytes: *const u8,
    image_bytes_len: usize,
    n_predict: i32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    repeat_last_n: i32,
    repeat_penalty: f32,
    reasoning: *const c_char,
    reasoning_budget: i32,
    reasoning_format: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_embeddings_request {
    instance_id: i64,
    body_json: *const c_char,
    oai_compat: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_rerank_request {
    instance_id: i64,
    body_json: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_audio_raw_request {
    instance_id: i64,
    audio_bytes: *const u8,
    audio_bytes_len: usize,
    audio_format: *const c_char,
    metadata_json: *const c_char,
    ffmpeg_convert: i32,
    enable_diarization: i32,
    diarization_model_path: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_native_audio_transcription_request {
    model_path: *const c_char,
    execution_group_id: *const c_char,
    audio_bytes: *const u8,
    audio_bytes_len: usize,
    audio_format: *const c_char,
    metadata_json: *const c_char,
    ffmpeg_convert: i32,
    enable_diarization: i32,
    diarization_model_path: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct llama_server_cluster_inference_metrics {
    loaded_this_call: i32,
    used_rpc: i32,
    rpc_server_count: i32,
    prompt_tokens: i32,
    decoded_tokens: i32,
    request_bytes: u64,
    model_bytes: u64,
    mmproj_bytes: u64,
    queue_wait_ms: f64,
    load_ms: f64,
    prompt_ms: f64,
    predicted_ms: f64,
    request_total_ms: f64,
    prompt_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    total_tokens_per_second: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_chat_result {
    ok: i32,
    text: *mut c_char,
    error: *mut c_char,
    metrics: llama_server_cluster_inference_metrics,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_vlm_result {
    ok: i32,
    text: *mut c_char,
    error: *mut c_char,
    metrics: llama_server_cluster_inference_metrics,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_server_cluster_json_result {
    ok: i32,
    status: i32,
    json: *mut c_char,
    error: *mut c_char,
    metrics: llama_server_cluster_inference_metrics,
}

type FnDefaultInstanceParams = unsafe extern "C" fn() -> llama_server_cluster_instance_params;
type FnDefaultChatRequest = unsafe extern "C" fn() -> llama_server_cluster_chat_request;
type FnDefaultVlmRequest = unsafe extern "C" fn() -> llama_server_cluster_vlm_request;
type FnDefaultEmbeddingsRequest = unsafe extern "C" fn() -> llama_server_cluster_embeddings_request;
type FnDefaultRerankRequest = unsafe extern "C" fn() -> llama_server_cluster_rerank_request;
type FnDefaultAudioRawRequest = unsafe extern "C" fn() -> llama_server_cluster_audio_raw_request;
type FnDefaultNativeAudioTranscriptionRequest =
    unsafe extern "C" fn() -> llama_server_cluster_native_audio_transcription_request;
type FnEmptyChatResult = unsafe extern "C" fn() -> llama_server_cluster_chat_result;
type FnEmptyVlmResult = unsafe extern "C" fn() -> llama_server_cluster_vlm_result;
type FnEmptyJsonResult = unsafe extern "C" fn() -> llama_server_cluster_json_result;
type FnCreate = unsafe extern "C" fn() -> *mut llama_server_cluster;
type FnDestroy = unsafe extern "C" fn(*mut llama_server_cluster);
type FnLastError = unsafe extern "C" fn(*const llama_server_cluster) -> *const c_char;
type FnGetLocalNodeInfo =
    unsafe extern "C" fn(*mut llama_server_cluster, *mut llama_server_cluster_node_info) -> i32;
type FnFreeNodeInfo = unsafe extern "C" fn(*mut llama_server_cluster_node_info);
type FnListDevices = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *mut *mut llama_server_cluster_device_info,
    *mut usize,
) -> i32;
type FnListDevicesWithRpc = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const c_char,
    *mut *mut llama_server_cluster_device_info,
    *mut usize,
) -> i32;
type FnFreeDevices = unsafe extern "C" fn(*mut llama_server_cluster_device_info, usize);
type FnListExecutionGroups = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *mut *mut llama_server_cluster_execution_group_info,
    *mut usize,
) -> i32;
type FnListExecutionGroupsWithRpc = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const c_char,
    *mut *mut llama_server_cluster_execution_group_info,
    *mut usize,
) -> i32;
type FnRunLocalRpcServer =
    unsafe extern "C" fn(*mut llama_server_cluster, *const c_char, i32, i32) -> i32;
type FnFreeExecutionGroups =
    unsafe extern "C" fn(*mut llama_server_cluster_execution_group_info, usize);
type FnCreateInstance = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_instance_params,
) -> i64;
type FnRemoveInstance = unsafe extern "C" fn(*mut llama_server_cluster, i64) -> i32;
type FnListInstances = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *mut *mut llama_server_cluster_instance_info,
    *mut usize,
) -> i32;
type FnFreeInstances = unsafe extern "C" fn(*mut llama_server_cluster_instance_info, usize);
type FnSetRetention = unsafe extern "C" fn(*mut llama_server_cluster, i64, i32) -> i32;
type FnLoadInstance = unsafe extern "C" fn(*mut llama_server_cluster, i64) -> i32;
type FnUnloadInstance = unsafe extern "C" fn(*mut llama_server_cluster, i64) -> i32;
type FnChatComplete = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_chat_request,
    *mut llama_server_cluster_chat_result,
) -> i32;
type FnChatResultFree = unsafe extern "C" fn(*mut llama_server_cluster_chat_result);
type FnVlmComplete = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_vlm_request,
    *mut llama_server_cluster_vlm_result,
) -> i32;
type FnVlmResultFree = unsafe extern "C" fn(*mut llama_server_cluster_vlm_result);
type FnEmbeddings = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_embeddings_request,
    *mut llama_server_cluster_json_result,
) -> i32;
type FnRerank = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_rerank_request,
    *mut llama_server_cluster_json_result,
) -> i32;
type FnAudioTranscriptionsRaw = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_audio_raw_request,
    *mut llama_server_cluster_json_result,
) -> i32;
type FnAudioTranscriptionsNative = unsafe extern "C" fn(
    *mut llama_server_cluster,
    *const llama_server_cluster_native_audio_transcription_request,
    *mut llama_server_cluster_json_result,
) -> i32;
type FnJsonResultFree = unsafe extern "C" fn(*mut llama_server_cluster_json_result);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub display_name: String,
    pub os_name: String,
    pub arch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub bridge_device_index: i32,
    pub backend: String,
    pub name: String,
    pub description: String,
    pub memory_free: u64,
    pub memory_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGroupInfo {
    pub id: String,
    pub label: String,
    pub backend_summary: String,
    pub devices_csv: String,
    pub device_count: i32,
    pub uses_local_split: bool,
    pub memory_free: u64,
    pub memory_total: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManualDeviceAllocation {
    pub bridge_device_index: i32,
    #[serde(default)]
    pub device_label: String,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub layer_count: u32,
    #[serde(default)]
    pub rpc_device: bool,
    #[serde(default)]
    pub source_node_id: String,
    #[serde(default)]
    pub source_control_addr: String,
    #[serde(default = "manual_device_unknown_index")]
    pub source_bridge_device_index: i32,
}

const fn manual_device_unknown_index() -> i32 {
    -1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionMode {
    KeepLoaded,
    LoadOnDemand,
}

impl RetentionMode {
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::KeepLoaded => 1,
            Self::LoadOnDemand => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InstanceModelKind {
    #[default]
    Text,
    Vision,
    Embeddings,
    Rerank,
    Whisper,
    RealtimeAudio,
    Diarization,
}

impl InstanceModelKind {
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::Text => 0,
            Self::Vision => 1,
            Self::Embeddings => 2,
            Self::Rerank => 3,
            Self::Whisper => 4,
            Self::RealtimeAudio => 5,
            Self::Diarization => 6,
        }
    }

    pub fn from_ffi(value: i32) -> Self {
        match value {
            1 => Self::Vision,
            2 => Self::Embeddings,
            3 => Self::Rerank,
            4 => Self::Whisper,
            5 => Self::RealtimeAudio,
            6 => Self::Diarization,
            _ => Self::Text,
        }
    }

    pub fn from_dropdown_value(value: &str) -> Self {
        match value {
            "vision" => Self::Vision,
            "embeddings" => Self::Embeddings,
            "rerank" => Self::Rerank,
            "whisper" => Self::Whisper,
            "realtime-audio" => Self::RealtimeAudio,
            "diarization" => Self::Diarization,
            _ => Self::Text,
        }
    }

    pub fn as_dropdown_value(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Embeddings => "embeddings",
            Self::Rerank => "rerank",
            Self::Whisper => "whisper",
            Self::RealtimeAudio => "realtime-audio",
            Self::Diarization => "diarization",
        }
    }

    pub fn default_load_on_demand_grace_seconds(self) -> i32 {
        match self {
            Self::Whisper | Self::Diarization => 0,
            _ => 30,
        }
    }

    pub fn supports_responses(self) -> bool {
        matches!(self, Self::Text | Self::Vision)
    }

    pub fn supports_vision(self) -> bool {
        matches!(self, Self::Vision)
    }

    pub fn supports_embeddings(self) -> bool {
        matches!(self, Self::Embeddings)
    }

    pub fn supports_rerank(self) -> bool {
        matches!(self, Self::Rerank)
    }

    pub fn supports_transcription(self) -> bool {
        matches!(self, Self::Whisper | Self::RealtimeAudio)
    }

    pub fn supports_diarization(self) -> bool {
        matches!(self, Self::Diarization)
    }
}

pub fn default_load_on_demand_grace_seconds() -> i32 {
    InstanceModelKind::Text.default_load_on_demand_grace_seconds()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub instance_id: i64,
    pub name: String,
    pub model_path: String,
    pub mmproj_path: String,
    pub diarization_model_path: Option<String>,
    pub execution_group_id: String,
    pub rpc_servers: String,
    pub retention_mode: RetentionMode,
    #[serde(default = "default_load_on_demand_grace_seconds")]
    pub load_on_demand_grace_seconds: i32,
    #[serde(default)]
    pub model_kind: InstanceModelKind,
    pub state: i32,
    pub active_request_count: i32,
    pub queued_request_count: i32,
    pub n_parallel: i32,
    pub grace_deadline_unix_ms: i64,
    pub last_error: String,
}

impl InstanceInfo {
    pub fn supports_responses(&self) -> bool {
        self.model_kind.supports_responses()
    }

    pub fn supports_vision(&self) -> bool {
        self.model_kind.supports_vision()
    }

    pub fn supports_embeddings(&self) -> bool {
        self.model_kind.supports_embeddings()
    }

    pub fn supports_rerank(&self) -> bool {
        self.model_kind.supports_rerank()
    }

    pub fn supports_transcription(&self) -> bool {
        self.model_kind.supports_transcription()
    }

    pub fn supports_diarization(&self) -> bool {
        self.model_kind.supports_diarization()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateInstanceParams {
    pub name: String,
    pub managed_model_id: Option<String>,
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub diarization_model_path: Option<String>,
    pub execution_group_id: String,
    pub rpc_servers: Option<String>,
    #[serde(default)]
    pub manual_device_allocations: Vec<ManualDeviceAllocation>,
    #[serde(default)]
    pub manual_devices_csv: Option<String>,
    #[serde(default)]
    pub manual_tensor_split: Option<String>,
    pub preferred_owner_control_addr: Option<String>,
    pub retention_mode: RetentionMode,
    #[serde(default = "default_load_on_demand_grace_seconds")]
    pub load_on_demand_grace_seconds: i32,
    pub embedding: bool,
    pub reranking: bool,
    #[serde(default)]
    pub model_kind: InstanceModelKind,
    pub single_device_only: bool,
    pub allow_cpu: bool,
    pub allow_integrated_gpu: bool,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
}

impl CreateInstanceParams {
    pub fn effective_model_kind(&self) -> InstanceModelKind {
        if self.model_kind != InstanceModelKind::Text {
            return self.model_kind;
        }
        if self.reranking {
            return InstanceModelKind::Rerank;
        }
        if self.embedding {
            return InstanceModelKind::Embeddings;
        }
        if self
            .diarization_model_path
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        {
            return InstanceModelKind::Whisper;
        }
        if self
            .mmproj_path
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        {
            return InstanceModelKind::Vision;
        }
        InstanceModelKind::Text
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub instance_id: i64,
    pub prompt: String,
    pub n_predict: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub reasoning: Option<String>,
    pub reasoning_budget: i32,
    pub reasoning_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmRequest {
    pub instance_id: i64,
    pub prompt: String,
    pub image_bytes: Vec<u8>,
    pub n_predict: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub reasoning: Option<String>,
    pub reasoning_budget: i32,
    pub reasoning_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub instance_id: i64,
    pub body_json: String,
    pub oai_compat: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    pub instance_id: i64,
    pub body_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRawRequest {
    pub instance_id: i64,
    pub audio_bytes: Vec<u8>,
    pub audio_format: String,
    pub metadata_json: Option<String>,
    pub ffmpeg_convert: bool,
    pub enable_diarization: bool,
    pub diarization_model_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeAudioTranscriptionRequest {
    pub model_path: String,
    pub execution_group_id: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub audio_format: String,
    pub metadata_json: Option<String>,
    pub ffmpeg_convert: bool,
    pub enable_diarization: bool,
    pub diarization_model_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub loaded_this_call: bool,
    pub used_rpc: bool,
    pub rpc_server_count: i32,
    pub prompt_tokens: i32,
    pub decoded_tokens: i32,
    pub request_bytes: u64,
    pub model_bytes: u64,
    pub mmproj_bytes: u64,
    pub queue_wait_ms: f64,
    pub load_ms: f64,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub request_total_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub total_tokens_per_second: f64,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            loaded_this_call: false,
            used_rpc: false,
            rpc_server_count: 0,
            prompt_tokens: 0,
            decoded_tokens: 0,
            request_bytes: 0,
            model_bytes: 0,
            mmproj_bytes: 0,
            queue_wait_ms: 0.0,
            load_ms: 0.0,
            prompt_ms: 0.0,
            predicted_ms: 0.0,
            request_total_ms: 0.0,
            prompt_tokens_per_second: 0.0,
            decode_tokens_per_second: 0.0,
            total_tokens_per_second: 0.0,
        }
    }
}

impl From<llama_server_cluster_inference_metrics> for InferenceMetrics {
    fn from(value: llama_server_cluster_inference_metrics) -> Self {
        Self {
            loaded_this_call: value.loaded_this_call != 0,
            used_rpc: value.used_rpc != 0,
            rpc_server_count: value.rpc_server_count,
            prompt_tokens: value.prompt_tokens,
            decoded_tokens: value.decoded_tokens,
            request_bytes: value.request_bytes,
            model_bytes: value.model_bytes,
            mmproj_bytes: value.mmproj_bytes,
            queue_wait_ms: value.queue_wait_ms,
            load_ms: value.load_ms,
            prompt_ms: value.prompt_ms,
            predicted_ms: value.predicted_ms,
            request_total_ms: value.request_total_ms,
            prompt_tokens_per_second: value.prompt_tokens_per_second,
            decode_tokens_per_second: value.decode_tokens_per_second,
            total_tokens_per_second: value.total_tokens_per_second,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationResult {
    pub text: String,
    pub metrics: InferenceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonResult {
    pub status: i32,
    pub json: String,
    pub metrics: InferenceMetrics,
}

pub struct ClusterApi {
    pub runtime_dir: PathBuf,
    _lib: Library,
    cluster: *mut llama_server_cluster,
    default_instance_params: FnDefaultInstanceParams,
    default_chat_request: FnDefaultChatRequest,
    default_vlm_request: FnDefaultVlmRequest,
    default_embeddings_request: FnDefaultEmbeddingsRequest,
    default_rerank_request: FnDefaultRerankRequest,
    default_audio_raw_request: FnDefaultAudioRawRequest,
    default_native_audio_transcription_request: FnDefaultNativeAudioTranscriptionRequest,
    empty_chat_result: FnEmptyChatResult,
    empty_vlm_result: FnEmptyVlmResult,
    empty_json_result: FnEmptyJsonResult,
    destroy: FnDestroy,
    last_error: FnLastError,
    get_local_node_info: FnGetLocalNodeInfo,
    free_node_info: FnFreeNodeInfo,
    list_devices: FnListDevices,
    list_devices_with_rpc: FnListDevicesWithRpc,
    free_devices: FnFreeDevices,
    list_execution_groups: FnListExecutionGroups,
    list_execution_groups_with_rpc: FnListExecutionGroupsWithRpc,
    run_local_rpc_server_fn: FnRunLocalRpcServer,
    free_execution_groups: FnFreeExecutionGroups,
    create_instance: FnCreateInstance,
    remove_instance: FnRemoveInstance,
    list_instances: FnListInstances,
    free_instances: FnFreeInstances,
    set_retention: FnSetRetention,
    load_instance: FnLoadInstance,
    unload_instance: FnUnloadInstance,
    chat_complete: FnChatComplete,
    chat_result_free: FnChatResultFree,
    vlm_complete: FnVlmComplete,
    vlm_result_free: FnVlmResultFree,
    embeddings: FnEmbeddings,
    rerank: FnRerank,
    audio_transcriptions_raw: FnAudioTranscriptionsRaw,
    audio_transcriptions_native: FnAudioTranscriptionsNative,
    json_result_free: FnJsonResultFree,
}

unsafe impl Send for ClusterApi {}
unsafe impl Sync for ClusterApi {}

impl Drop for ClusterApi {
    fn drop(&mut self) {
        if !self.cluster.is_null() {
            unsafe { (self.destroy)(self.cluster) };
        }
    }
}

impl ClusterApi {
    fn run_with_large_stack<T, F>(&self, label: &str, func: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let handle = thread::Builder::new()
            .name(format!("cluster-api-{label}"))
            .stack_size(16 * 1024 * 1024)
            .spawn(func)
            .map_err(|err| anyhow!("failed to spawn {label} worker: {err}"))?;
        handle
            .join()
            .map_err(|_| anyhow!("{label} worker panicked"))
    }

    pub fn load(runtime_dir: &Path) -> Result<Self> {
        let runtime_dir = runtime_dir
            .canonicalize()
            .unwrap_or_else(|_| runtime_dir.to_path_buf());
        configure_runtime_loader_paths(&runtime_dir);
        let library_path = runtime_library_path(&runtime_dir);
        if !library_path.exists() {
            bail!("missing cluster library: '{}'", library_path.display());
        }

        let lib = unsafe { Library::new(&library_path) }
            .with_context(|| format!("failed to load '{}'", library_path.display()))?;

        unsafe {
            let default_instance_params = *lib.get::<FnDefaultInstanceParams>(
                b"llama_server_cluster_default_instance_params\0",
            )?;
            let default_chat_request =
                *lib.get::<FnDefaultChatRequest>(b"llama_server_cluster_default_chat_request\0")?;
            let default_vlm_request =
                *lib.get::<FnDefaultVlmRequest>(b"llama_server_cluster_default_vlm_request\0")?;
            let default_embeddings_request = *lib.get::<FnDefaultEmbeddingsRequest>(
                b"llama_server_cluster_default_embeddings_request\0",
            )?;
            let default_rerank_request = *lib
                .get::<FnDefaultRerankRequest>(b"llama_server_cluster_default_rerank_request\0")?;
            let default_audio_raw_request = *lib.get::<FnDefaultAudioRawRequest>(
                b"llama_server_cluster_default_audio_raw_request\0",
            )?;
            let default_native_audio_transcription_request =
                *lib.get::<FnDefaultNativeAudioTranscriptionRequest>(
                    b"llama_server_cluster_default_native_audio_transcription_request\0",
                )?;
            let empty_chat_result =
                *lib.get::<FnEmptyChatResult>(b"llama_server_cluster_empty_chat_result\0")?;
            let empty_vlm_result =
                *lib.get::<FnEmptyVlmResult>(b"llama_server_cluster_empty_vlm_result\0")?;
            let empty_json_result =
                *lib.get::<FnEmptyJsonResult>(b"llama_server_cluster_empty_json_result\0")?;
            let create = *lib.get::<FnCreate>(b"llama_server_cluster_create\0")?;
            let destroy = *lib.get::<FnDestroy>(b"llama_server_cluster_destroy\0")?;
            let last_error = *lib.get::<FnLastError>(b"llama_server_cluster_last_error\0")?;
            let get_local_node_info =
                *lib.get::<FnGetLocalNodeInfo>(b"llama_server_cluster_get_local_node_info\0")?;
            let free_node_info =
                *lib.get::<FnFreeNodeInfo>(b"llama_server_cluster_free_node_info\0")?;
            let list_devices = *lib.get::<FnListDevices>(b"llama_server_cluster_list_devices\0")?;
            let list_devices_with_rpc =
                *lib.get::<FnListDevicesWithRpc>(b"llama_server_cluster_list_devices_with_rpc\0")?;
            let free_devices = *lib.get::<FnFreeDevices>(b"llama_server_cluster_free_devices\0")?;
            let list_execution_groups =
                *lib.get::<FnListExecutionGroups>(b"llama_server_cluster_list_execution_groups\0")?;
            let list_execution_groups_with_rpc = *lib.get::<FnListExecutionGroupsWithRpc>(
                b"llama_server_cluster_list_execution_groups_with_rpc\0",
            )?;
            let run_local_rpc_server_fn =
                *lib.get::<FnRunLocalRpcServer>(b"llama_server_cluster_run_local_rpc_server\0")?;
            let free_execution_groups =
                *lib.get::<FnFreeExecutionGroups>(b"llama_server_cluster_free_execution_groups\0")?;
            let create_instance =
                *lib.get::<FnCreateInstance>(b"llama_server_cluster_create_instance\0")?;
            let remove_instance =
                *lib.get::<FnRemoveInstance>(b"llama_server_cluster_remove_instance\0")?;
            let list_instances =
                *lib.get::<FnListInstances>(b"llama_server_cluster_list_instances\0")?;
            let free_instances =
                *lib.get::<FnFreeInstances>(b"llama_server_cluster_free_instances\0")?;
            let set_retention =
                *lib.get::<FnSetRetention>(b"llama_server_cluster_set_instance_retention_mode\0")?;
            let load_instance =
                *lib.get::<FnLoadInstance>(b"llama_server_cluster_load_instance\0")?;
            let unload_instance =
                *lib.get::<FnUnloadInstance>(b"llama_server_cluster_unload_instance\0")?;
            let chat_complete =
                *lib.get::<FnChatComplete>(b"llama_server_cluster_chat_complete\0")?;
            let chat_result_free =
                *lib.get::<FnChatResultFree>(b"llama_server_cluster_chat_result_free\0")?;
            let vlm_complete = *lib.get::<FnVlmComplete>(b"llama_server_cluster_vlm_complete\0")?;
            let vlm_result_free =
                *lib.get::<FnVlmResultFree>(b"llama_server_cluster_vlm_result_free\0")?;
            let embeddings = *lib.get::<FnEmbeddings>(b"llama_server_cluster_embeddings\0")?;
            let rerank = *lib.get::<FnRerank>(b"llama_server_cluster_rerank\0")?;
            let audio_transcriptions_raw = *lib.get::<FnAudioTranscriptionsRaw>(
                b"llama_server_cluster_audio_transcriptions_raw\0",
            )?;
            let audio_transcriptions_native = *lib.get::<FnAudioTranscriptionsNative>(
                b"llama_server_cluster_audio_transcriptions_native\0",
            )?;
            let json_result_free =
                *lib.get::<FnJsonResultFree>(b"llama_server_cluster_json_result_free\0")?;

            let cluster = create();
            if cluster.is_null() {
                bail!("llama_server_cluster_create returned null");
            }

            Ok(Self {
                runtime_dir,
                _lib: lib,
                cluster,
                default_instance_params,
                default_chat_request,
                default_vlm_request,
                default_embeddings_request,
                default_rerank_request,
                default_audio_raw_request,
                default_native_audio_transcription_request,
                empty_chat_result,
                empty_vlm_result,
                empty_json_result,
                destroy,
                last_error,
                get_local_node_info,
                free_node_info,
                list_devices,
                list_devices_with_rpc,
                free_devices,
                list_execution_groups,
                list_execution_groups_with_rpc,
                run_local_rpc_server_fn,
                free_execution_groups,
                create_instance,
                remove_instance,
                list_instances,
                free_instances,
                set_retention,
                load_instance,
                unload_instance,
                chat_complete,
                chat_result_free,
                vlm_complete,
                vlm_result_free,
                embeddings,
                rerank,
                audio_transcriptions_raw,
                audio_transcriptions_native,
                json_result_free,
            })
        }
    }

    pub fn default_instance_params(&self) -> CreateInstanceParams {
        let defaults = unsafe { (self.default_instance_params)() };
        CreateInstanceParams {
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
            load_on_demand_grace_seconds: defaults.load_on_demand_grace_seconds,
            embedding: defaults.embedding != 0,
            reranking: defaults.reranking != 0,
            model_kind: InstanceModelKind::from_ffi(defaults.model_kind),
            single_device_only: false,
            allow_cpu: false,
            allow_integrated_gpu: false,
            n_ctx: defaults.n_ctx,
            n_batch: defaults.n_batch,
            n_ubatch: defaults.n_ubatch,
            n_parallel: defaults.n_parallel,
            n_threads: defaults.n_threads,
            n_threads_batch: defaults.n_threads_batch,
            n_gpu_layers: defaults.n_gpu_layers,
        }
    }

    pub fn default_chat_request(&self, instance_id: i64) -> ChatRequest {
        let defaults = unsafe { (self.default_chat_request)() };
        ChatRequest {
            instance_id,
            prompt: String::new(),
            n_predict: defaults.n_predict,
            temperature: defaults.temperature,
            top_p: defaults.top_p,
            top_k: defaults.top_k,
            min_p: defaults.min_p,
            repeat_last_n: defaults.repeat_last_n,
            repeat_penalty: defaults.repeat_penalty,
            reasoning: cstr_opt(defaults.reasoning),
            reasoning_budget: defaults.reasoning_budget,
            reasoning_format: cstr_opt(defaults.reasoning_format),
        }
    }

    pub fn default_vlm_request(&self, instance_id: i64) -> VlmRequest {
        let defaults = unsafe { (self.default_vlm_request)() };
        VlmRequest {
            instance_id,
            prompt: String::new(),
            image_bytes: Vec::new(),
            n_predict: defaults.n_predict,
            temperature: defaults.temperature,
            top_p: defaults.top_p,
            top_k: defaults.top_k,
            min_p: defaults.min_p,
            repeat_last_n: defaults.repeat_last_n,
            repeat_penalty: defaults.repeat_penalty,
            reasoning: cstr_opt(defaults.reasoning),
            reasoning_budget: defaults.reasoning_budget,
            reasoning_format: cstr_opt(defaults.reasoning_format),
        }
    }

    pub fn default_embeddings_request(&self, instance_id: i64) -> EmbeddingsRequest {
        let defaults = unsafe { (self.default_embeddings_request)() };
        EmbeddingsRequest {
            instance_id,
            body_json: String::new(),
            oai_compat: defaults.oai_compat != 0,
        }
    }

    pub fn default_rerank_request(&self, instance_id: i64) -> RerankRequest {
        let defaults = unsafe { (self.default_rerank_request)() };
        RerankRequest {
            instance_id,
            body_json: cstr_opt(defaults.body_json).unwrap_or_default(),
        }
    }

    pub fn default_audio_raw_request(&self, instance_id: i64) -> AudioRawRequest {
        let defaults = unsafe { (self.default_audio_raw_request)() };
        AudioRawRequest {
            instance_id,
            audio_bytes: Vec::new(),
            audio_format: cstr_opt(defaults.audio_format).unwrap_or_default(),
            metadata_json: cstr_opt(defaults.metadata_json),
            ffmpeg_convert: defaults.ffmpeg_convert != 0,
            enable_diarization: defaults.enable_diarization != 0,
            diarization_model_path: cstr_opt(defaults.diarization_model_path),
        }
    }

    pub fn default_native_audio_transcription_request(&self) -> NativeAudioTranscriptionRequest {
        let defaults = unsafe { (self.default_native_audio_transcription_request)() };
        NativeAudioTranscriptionRequest {
            model_path: String::new(),
            execution_group_id: (!defaults.execution_group_id.is_null())
                .then(|| cstr_from_const(defaults.execution_group_id))
                .filter(|value| !value.is_empty()),
            audio_bytes: Vec::new(),
            audio_format: cstr_from_const(defaults.audio_format),
            metadata_json: (!defaults.metadata_json.is_null())
                .then(|| cstr_from_const(defaults.metadata_json))
                .filter(|value| !value.is_empty()),
            ffmpeg_convert: defaults.ffmpeg_convert != 0,
            enable_diarization: defaults.enable_diarization != 0,
            diarization_model_path: (!defaults.diarization_model_path.is_null())
                .then(|| cstr_from_const(defaults.diarization_model_path))
                .filter(|value| !value.is_empty()),
        }
    }

    pub fn get_last_error(&self) -> String {
        unsafe { cstr_from_const((self.last_error)(self.cluster)) }
    }

    pub fn get_local_node_info(&self) -> Result<NodeInfo> {
        let mut raw = llama_server_cluster_node_info {
            node_id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            os_name: ptr::null_mut(),
            arch: ptr::null_mut(),
        };
        let rc = unsafe { (self.get_local_node_info)(self.cluster, &mut raw) };
        if rc != 0 {
            bail!(self.get_last_error());
        }
        let out = NodeInfo {
            node_id: cstr_from_mut(raw.node_id),
            display_name: cstr_from_mut(raw.display_name),
            os_name: cstr_from_mut(raw.os_name),
            arch: cstr_from_mut(raw.arch),
        };
        unsafe { (self.free_node_info)(&mut raw) };
        Ok(out)
    }

    pub fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
        let mut ptr_devices = ptr::null_mut();
        let mut count = 0usize;
        let rc = unsafe { (self.list_devices)(self.cluster, &mut ptr_devices, &mut count) };
        if rc != 0 {
            bail!(self.get_last_error());
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let raw = unsafe { &*ptr_devices.add(i) };
            out.push(DeviceInfo {
                bridge_device_index: raw.bridge_device_index,
                backend: cstr_from_mut(raw.backend),
                name: cstr_from_mut(raw.name),
                description: cstr_from_mut(raw.description),
                memory_free: raw.memory_free,
                memory_total: raw.memory_total,
            });
        }
        unsafe { (self.free_devices)(ptr_devices, count) };
        Ok(out)
    }

    pub fn list_devices_with_rpc(&self, rpc_servers: Option<&str>) -> Result<Vec<DeviceInfo>> {
        let rpc_c = rpc_servers
            .filter(|v| !v.is_empty())
            .map(CString::new)
            .transpose()
            .context("invalid rpc server list")?;

        let mut ptr_devices = ptr::null_mut();
        let mut count = 0usize;
        let rc = unsafe {
            (self.list_devices_with_rpc)(
                self.cluster,
                rpc_c.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                &mut ptr_devices,
                &mut count,
            )
        };
        if rc != 0 {
            bail!(self.get_last_error());
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let raw = unsafe { &*ptr_devices.add(i) };
            out.push(DeviceInfo {
                bridge_device_index: raw.bridge_device_index,
                backend: cstr_from_mut(raw.backend),
                name: cstr_from_mut(raw.name),
                description: cstr_from_mut(raw.description),
                memory_free: raw.memory_free,
                memory_total: raw.memory_total,
            });
        }
        unsafe { (self.free_devices)(ptr_devices, count) };
        Ok(out)
    }

    pub fn list_execution_groups(&self) -> Result<Vec<ExecutionGroupInfo>> {
        let mut ptr_groups = ptr::null_mut();
        let mut count = 0usize;
        let rc = unsafe { (self.list_execution_groups)(self.cluster, &mut ptr_groups, &mut count) };
        if rc != 0 {
            bail!(self.get_last_error());
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let raw = unsafe { &*ptr_groups.add(i) };
            out.push(ExecutionGroupInfo {
                id: cstr_from_mut(raw.id),
                label: cstr_from_mut(raw.label),
                backend_summary: cstr_from_mut(raw.backend_summary),
                devices_csv: cstr_from_mut(raw.devices_csv),
                device_count: raw.device_count,
                uses_local_split: raw.uses_local_split != 0,
                memory_free: raw.memory_free,
                memory_total: raw.memory_total,
            });
        }
        unsafe { (self.free_execution_groups)(ptr_groups, count) };
        Ok(out)
    }

    pub fn list_execution_groups_with_rpc(
        &self,
        rpc_servers: Option<&str>,
    ) -> Result<Vec<ExecutionGroupInfo>> {
        let rpc_c = rpc_servers
            .filter(|v| !v.is_empty())
            .map(CString::new)
            .transpose()
            .context("invalid rpc server list")?;

        let mut ptr_groups = ptr::null_mut();
        let mut count = 0usize;
        let rc = unsafe {
            (self.list_execution_groups_with_rpc)(
                self.cluster,
                rpc_c.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                &mut ptr_groups,
                &mut count,
            )
        };
        if rc != 0 {
            bail!(self.get_last_error());
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let raw = unsafe { &*ptr_groups.add(i) };
            out.push(ExecutionGroupInfo {
                id: cstr_from_mut(raw.id),
                label: cstr_from_mut(raw.label),
                backend_summary: cstr_from_mut(raw.backend_summary),
                devices_csv: cstr_from_mut(raw.devices_csv),
                device_count: raw.device_count,
                uses_local_split: raw.uses_local_split != 0,
                memory_free: raw.memory_free,
                memory_total: raw.memory_total,
            });
        }
        unsafe { (self.free_execution_groups)(ptr_groups, count) };
        Ok(out)
    }

    pub fn run_local_rpc_server(&self, host: &str, port: i32, n_threads: i32) -> Result<()> {
        let host_c = CString::new(host).context("invalid rpc host")?;
        let rc = unsafe {
            (self.run_local_rpc_server_fn)(self.cluster, host_c.as_ptr(), port, n_threads)
        };
        if rc != 0 {
            bail!(self.get_last_error());
        }
        Ok(())
    }

    pub fn list_instances(&self) -> Result<Vec<InstanceInfo>> {
        let mut ptr_instances = ptr::null_mut();
        let mut count = 0usize;
        let rc = unsafe { (self.list_instances)(self.cluster, &mut ptr_instances, &mut count) };
        if rc != 0 {
            bail!(self.get_last_error());
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let raw = unsafe { &*ptr_instances.add(i) };
            out.push(InstanceInfo {
                instance_id: raw.instance_id,
                name: cstr_from_mut(raw.name),
                model_path: cstr_from_mut(raw.model_path),
                mmproj_path: cstr_from_mut(raw.mmproj_path),
                diarization_model_path: {
                    let value = cstr_from_mut(raw.diarization_model_path);
                    if value.is_empty() {
                        None
                    } else {
                        Some(value)
                    }
                },
                execution_group_id: cstr_from_mut(raw.execution_group_id),
                rpc_servers: cstr_from_mut(raw.rpc_servers),
                retention_mode: if raw.retention_mode == 2 {
                    RetentionMode::LoadOnDemand
                } else {
                    RetentionMode::KeepLoaded
                },
                load_on_demand_grace_seconds: raw.load_on_demand_grace_seconds,
                model_kind: InstanceModelKind::from_ffi(raw.model_kind),
                state: raw.state,
                active_request_count: raw.active_request_count,
                queued_request_count: raw.queued_request_count,
                n_parallel: raw.n_parallel,
                grace_deadline_unix_ms: raw.grace_deadline_unix_ms,
                last_error: cstr_from_mut(raw.last_error),
            });
        }
        unsafe { (self.free_instances)(ptr_instances, count) };
        Ok(out)
    }

    pub fn create_instance(&self, params: &CreateInstanceParams) -> Result<i64> {
        let name_c = CString::new(params.name.as_str()).context("invalid instance name")?;
        let model_c = CString::new(params.model_path.as_str()).context("invalid model path")?;
        let mmproj_c = params
            .mmproj_path
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid mmproj path")?;
        let diarization_c = params
            .diarization_model_path
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid diarization model path")?;
        let rpc_servers_c = params
            .rpc_servers
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid rpc server list")?;
        let manual_devices_c = params
            .manual_devices_csv
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid manual devices list")?;
        let manual_tensor_split_c = params
            .manual_tensor_split
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid manual tensor split")?;
        let group_c = CString::new(params.execution_group_id.as_str())
            .context("invalid execution group id")?;
        let effective_kind = params.effective_model_kind();

        let raw = llama_server_cluster_instance_params {
            name: if params.name.is_empty() {
                ptr::null()
            } else {
                name_c.as_ptr()
            },
            model_path: model_c.as_ptr(),
            mmproj_path: mmproj_c.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
            diarization_model_path: diarization_c.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
            execution_group_id: group_c.as_ptr(),
            rpc_servers: rpc_servers_c.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
            manual_devices_csv: manual_devices_c
                .as_ref()
                .map_or(ptr::null(), |v| v.as_ptr()),
            manual_tensor_split: manual_tensor_split_c
                .as_ref()
                .map_or(ptr::null(), |v| v.as_ptr()),
            retention_mode: params.retention_mode.as_ffi(),
            load_on_demand_grace_seconds: params.load_on_demand_grace_seconds.max(0),
            embedding: if effective_kind.supports_embeddings() {
                1
            } else {
                0
            },
            reranking: if effective_kind.supports_rerank() {
                1
            } else {
                0
            },
            model_kind: effective_kind.as_ffi(),
            allow_cpu: if params.allow_cpu { 1 } else { 0 },
            allow_integrated_gpu: if params.allow_integrated_gpu { 1 } else { 0 },
            n_ctx: params.n_ctx,
            n_batch: params.n_batch,
            n_ubatch: params.n_ubatch,
            n_parallel: params.n_parallel,
            n_threads: params.n_threads,
            n_threads_batch: params.n_threads_batch,
            n_gpu_layers: params.n_gpu_layers,
        };

        let instance_id = unsafe { (self.create_instance)(self.cluster, &raw) };
        if instance_id <= 0 {
            bail!(self.get_last_error());
        }
        Ok(instance_id)
    }

    pub fn remove_instance(&self, instance_id: i64) -> Result<()> {
        let rc = unsafe { (self.remove_instance)(self.cluster, instance_id) };
        if rc != 0 {
            bail!(self.get_last_error());
        }
        Ok(())
    }

    pub fn set_retention_mode(&self, instance_id: i64, mode: RetentionMode) -> Result<()> {
        let rc = unsafe { (self.set_retention)(self.cluster, instance_id, mode.as_ffi()) };
        if rc != 0 {
            bail!(self.get_last_error());
        }
        Ok(())
    }

    pub fn load_instance(&self, instance_id: i64) -> Result<()> {
        let cluster = self.cluster as usize;
        let load_instance = self.load_instance;
        let rc = self.run_with_large_stack("load-instance", move || unsafe {
            load_instance(cluster as *mut llama_server_cluster, instance_id)
        })?;
        if rc != 0 {
            bail!(self.get_last_error());
        }
        Ok(())
    }

    pub fn unload_instance(&self, instance_id: i64) -> Result<()> {
        let rc = unsafe { (self.unload_instance)(self.cluster, instance_id) };
        if rc != 0 {
            bail!(self.get_last_error());
        }
        Ok(())
    }

    pub fn chat_complete(&self, req: &ChatRequest) -> Result<TextGenerationResult> {
        let cluster = self.cluster as usize;
        let chat_complete = self.chat_complete;
        let empty_chat_result = self.empty_chat_result;
        let chat_result_free = self.chat_result_free;
        let default_chat_request = self.default_chat_request;
        let prompt_c = CString::new(req.prompt.as_str()).context("invalid prompt")?;
        let reasoning_c = req
            .reasoning
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid reasoning")?;
        let reasoning_format_c = req
            .reasoning_format
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid reasoning format")?;
        let instance_id = req.instance_id;
        let n_predict = req.n_predict;
        let temperature = req.temperature;
        let top_p = req.top_p;
        let top_k = req.top_k;
        let min_p = req.min_p;
        let repeat_last_n = req.repeat_last_n;
        let repeat_penalty = req.repeat_penalty;
        let reasoning_budget = req.reasoning_budget;
        let (rc, text, error, metrics) = self.run_with_large_stack("chat-complete", move || {
            let mut raw_req = unsafe { default_chat_request() };
            raw_req.instance_id = instance_id;
            raw_req.prompt = prompt_c.as_ptr();
            raw_req.n_predict = n_predict;
            raw_req.temperature = temperature;
            raw_req.top_p = top_p;
            raw_req.top_k = top_k;
            raw_req.min_p = min_p;
            raw_req.repeat_last_n = repeat_last_n;
            raw_req.repeat_penalty = repeat_penalty;
            raw_req.reasoning = reasoning_c.as_ref().map_or(ptr::null(), |v| v.as_ptr());
            raw_req.reasoning_budget = reasoning_budget;
            raw_req.reasoning_format = reasoning_format_c
                .as_ref()
                .map_or(ptr::null(), |v| v.as_ptr());
            unsafe {
                let mut out = empty_chat_result();
                let rc = chat_complete(cluster as *mut llama_server_cluster, &raw_req, &mut out);
                let text = cstr_from_mut(out.text);
                let error = cstr_from_mut(out.error);
                let metrics = InferenceMetrics::from(out.metrics);
                chat_result_free(&mut out);
                (rc, text, error, metrics)
            }
        })?;

        if rc != 0 {
            bail!(if error.is_empty() {
                self.get_last_error()
            } else {
                error
            });
        }
        Ok(TextGenerationResult { text, metrics })
    }

    pub fn vlm_complete(&self, req: &VlmRequest) -> Result<TextGenerationResult> {
        if req.image_bytes.is_empty() {
            bail!("image_bytes are required");
        }

        let prompt_c = CString::new(req.prompt.as_str()).context("invalid prompt")?;
        let reasoning_c = req
            .reasoning
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid reasoning mode")?;
        let reasoning_format_c = req
            .reasoning_format
            .as_ref()
            .filter(|v| !v.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid reasoning format")?;

        let mut raw_req = unsafe { (self.default_vlm_request)() };
        raw_req.instance_id = req.instance_id;
        raw_req.prompt = prompt_c.as_ptr();
        raw_req.image_bytes = req.image_bytes.as_ptr();
        raw_req.image_bytes_len = req.image_bytes.len();
        raw_req.n_predict = req.n_predict;
        raw_req.temperature = req.temperature;
        raw_req.top_p = req.top_p;
        raw_req.top_k = req.top_k;
        raw_req.min_p = req.min_p;
        raw_req.repeat_last_n = req.repeat_last_n;
        raw_req.repeat_penalty = req.repeat_penalty;
        raw_req.reasoning = reasoning_c.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        raw_req.reasoning_budget = req.reasoning_budget;
        raw_req.reasoning_format = reasoning_format_c
            .as_ref()
            .map_or(ptr::null(), |v| v.as_ptr());

        let mut out = unsafe { (self.empty_vlm_result)() };
        let rc = unsafe { (self.vlm_complete)(self.cluster, &raw_req, &mut out) };
        let text = cstr_from_mut(out.text);
        let error = cstr_from_mut(out.error);
        let metrics = InferenceMetrics::from(out.metrics);
        unsafe { (self.vlm_result_free)(&mut out) };

        if rc != 0 {
            bail!(if error.is_empty() {
                self.get_last_error()
            } else {
                error
            });
        }
        Ok(TextGenerationResult { text, metrics })
    }

    pub fn embeddings(&self, req: &EmbeddingsRequest) -> Result<JsonResult> {
        let body_json_c =
            CString::new(req.body_json.as_str()).context("invalid embeddings body_json")?;

        let mut raw_req = unsafe { (self.default_embeddings_request)() };
        raw_req.instance_id = req.instance_id;
        raw_req.body_json = body_json_c.as_ptr();
        raw_req.oai_compat = if req.oai_compat { 1 } else { 0 };

        let mut out = unsafe { (self.empty_json_result)() };
        let rc = unsafe { (self.embeddings)(self.cluster, &raw_req, &mut out) };
        let json = cstr_from_mut(out.json);
        let error = cstr_from_mut(out.error);
        let status = out.status;
        let metrics = InferenceMetrics::from(out.metrics);
        unsafe { (self.json_result_free)(&mut out) };

        if rc != 0 {
            bail!(if error.is_empty() {
                self.get_last_error()
            } else {
                error
            });
        }
        Ok(JsonResult {
            status,
            json,
            metrics,
        })
    }

    pub fn rerank(&self, req: &RerankRequest) -> Result<JsonResult> {
        let body_json_c =
            CString::new(req.body_json.as_str()).context("invalid rerank body_json")?;

        let mut raw_req = unsafe { (self.default_rerank_request)() };
        raw_req.instance_id = req.instance_id;
        raw_req.body_json = body_json_c.as_ptr();

        let mut out = unsafe { (self.empty_json_result)() };
        let rc = unsafe { (self.rerank)(self.cluster, &raw_req, &mut out) };
        let json = cstr_from_mut(out.json);
        let error = cstr_from_mut(out.error);
        let status = out.status;
        let metrics = InferenceMetrics::from(out.metrics);
        unsafe { (self.json_result_free)(&mut out) };

        if rc != 0 {
            bail!(if error.is_empty() {
                self.get_last_error()
            } else {
                error
            });
        }
        Ok(JsonResult {
            status,
            json,
            metrics,
        })
    }

    pub fn audio_transcriptions_raw(&self, req: &AudioRawRequest) -> Result<JsonResult> {
        if req.audio_bytes.is_empty() {
            bail!("audio_bytes are required");
        }

        let audio_format_c =
            CString::new(req.audio_format.as_str()).context("invalid audio_format")?;
        let metadata_json_c = req
            .metadata_json
            .as_ref()
            .filter(|value| !value.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid metadata_json")?;
        let diarization_model_path_c = req
            .diarization_model_path
            .as_ref()
            .filter(|value| !value.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid diarization_model_path")?;

        let mut raw_req = unsafe { (self.default_audio_raw_request)() };
        raw_req.instance_id = req.instance_id;
        raw_req.audio_bytes = req.audio_bytes.as_ptr();
        raw_req.audio_bytes_len = req.audio_bytes.len();
        raw_req.audio_format = audio_format_c.as_ptr();
        raw_req.metadata_json = metadata_json_c
            .as_ref()
            .map_or(ptr::null(), |value| value.as_ptr());
        raw_req.ffmpeg_convert = if req.ffmpeg_convert { 1 } else { 0 };
        raw_req.enable_diarization = if req.enable_diarization { 1 } else { 0 };
        raw_req.diarization_model_path = diarization_model_path_c
            .as_ref()
            .map_or(ptr::null(), |value| value.as_ptr());

        let mut out = unsafe { (self.empty_json_result)() };
        let rc = unsafe { (self.audio_transcriptions_raw)(self.cluster, &raw_req, &mut out) };
        let json = cstr_from_mut(out.json);
        let error = cstr_from_mut(out.error);
        let status = out.status;
        let metrics = InferenceMetrics::from(out.metrics);
        unsafe { (self.json_result_free)(&mut out) };

        if rc != 0 {
            bail!(if error.is_empty() {
                self.get_last_error()
            } else {
                error
            });
        }
        Ok(JsonResult {
            status,
            json,
            metrics,
        })
    }

    pub fn audio_transcriptions_native(
        &self,
        req: &NativeAudioTranscriptionRequest,
    ) -> Result<JsonResult> {
        if req.audio_bytes.is_empty() {
            bail!("audio_bytes are required");
        }

        let model_path_c = CString::new(req.model_path.as_str()).context("invalid model_path")?;
        let execution_group_c = req
            .execution_group_id
            .as_ref()
            .filter(|value| !value.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid execution_group_id")?;
        let audio_format_c =
            CString::new(req.audio_format.as_str()).context("invalid audio_format")?;
        let metadata_json_c = req
            .metadata_json
            .as_ref()
            .filter(|value| !value.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid metadata_json")?;
        let diarization_model_path_c = req
            .diarization_model_path
            .as_ref()
            .filter(|value| !value.is_empty())
            .map(|value| CString::new(value.as_str()))
            .transpose()
            .context("invalid diarization_model_path")?;

        let raw_req = llama_server_cluster_native_audio_transcription_request {
            model_path: model_path_c.as_ptr(),
            execution_group_id: execution_group_c
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
            audio_bytes: req.audio_bytes.as_ptr(),
            audio_bytes_len: req.audio_bytes.len(),
            audio_format: audio_format_c.as_ptr(),
            metadata_json: metadata_json_c
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
            ffmpeg_convert: if req.ffmpeg_convert { 1 } else { 0 },
            enable_diarization: if req.enable_diarization { 1 } else { 0 },
            diarization_model_path: diarization_model_path_c
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
        };

        let mut out = unsafe { (self.empty_json_result)() };
        let rc = unsafe { (self.audio_transcriptions_native)(self.cluster, &raw_req, &mut out) };
        let json = cstr_from_mut(out.json);
        let error = cstr_from_mut(out.error);
        let status = out.status;
        let metrics = InferenceMetrics::from(out.metrics);
        unsafe { (self.json_result_free)(&mut out) };

        if rc != 0 {
            bail!(if error.is_empty() {
                self.get_last_error()
            } else {
                error
            });
        }
        Ok(JsonResult {
            status,
            json,
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CreateInstanceParams, InstanceModelKind, RetentionMode};

    #[test]
    fn dropdown_mapping_matches_ui_values() {
        assert_eq!(
            InstanceModelKind::from_dropdown_value("text"),
            InstanceModelKind::Text
        );
        assert_eq!(
            InstanceModelKind::from_dropdown_value("vision"),
            InstanceModelKind::Vision
        );
        assert_eq!(
            InstanceModelKind::from_dropdown_value("embeddings"),
            InstanceModelKind::Embeddings
        );
        assert_eq!(
            InstanceModelKind::from_dropdown_value("rerank"),
            InstanceModelKind::Rerank
        );
        assert_eq!(
            InstanceModelKind::from_dropdown_value("whisper"),
            InstanceModelKind::Whisper
        );
        assert_eq!(
            InstanceModelKind::from_dropdown_value("realtime-audio"),
            InstanceModelKind::RealtimeAudio
        );
        assert_eq!(
            InstanceModelKind::from_dropdown_value("diarization"),
            InstanceModelKind::Diarization
        );
    }

    #[test]
    fn effective_model_kind_uses_existing_capability_flags_as_fallback() {
        let params = CreateInstanceParams {
            name: String::new(),
            managed_model_id: None,
            model_path: "model.gguf".to_string(),
            mmproj_path: None,
            diarization_model_path: None,
            execution_group_id: "cluster:auto".to_string(),
            rpc_servers: None,
            manual_device_allocations: Vec::new(),
            manual_devices_csv: None,
            manual_tensor_split: None,
            preferred_owner_control_addr: None,
            retention_mode: RetentionMode::KeepLoaded,
            load_on_demand_grace_seconds: InstanceModelKind::Text
                .default_load_on_demand_grace_seconds(),
            embedding: false,
            reranking: true,
            model_kind: InstanceModelKind::Text,
            single_device_only: false,
            allow_cpu: false,
            allow_integrated_gpu: false,
            n_ctx: 0,
            n_batch: 0,
            n_ubatch: 0,
            n_parallel: 1,
            n_threads: 0,
            n_threads_batch: 0,
            n_gpu_layers: -1,
        };
        assert_eq!(params.effective_model_kind(), InstanceModelKind::Rerank);
    }
}

pub fn default_runtime_dir() -> Result<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let base = env::var_os("APPDATA").ok_or_else(|| anyhow!("APPDATA is not set"))?;
        Ok(PathBuf::from(base).join("OpenResearchTools").join("engine"))
    }
    #[cfg(target_os = "macos")]
    {
        let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("OpenResearchTools")
            .join("engine"))
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("OpenResearchTools")
            .join("engine"))
    }
}

fn runtime_library_path(runtime_dir: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let preferred = runtime_dir.join("multi-node-server.dll");
        if preferred.exists() {
            return preferred;
        }
        runtime_dir.join("llama-server-bridge.dll")
    }
    #[cfg(target_os = "macos")]
    {
        for candidate in [
            "libmulti-node-server.dylib",
            "libllama-server-cluster.dylib",
            "libllama-server-bridge.dylib",
        ] {
            let preferred = runtime_dir.join(candidate);
            if preferred.exists() {
                return preferred;
            }
        }
        runtime_dir.join("libllama-server-cluster.dylib")
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        for candidate in [
            "libmulti-node-server.so",
            "libllama-server-cluster.so",
            "libllama-server-bridge.so",
        ] {
            let preferred = runtime_dir.join(candidate);
            if preferred.exists() {
                return preferred;
            }
        }
        runtime_dir.join("libllama-server-cluster.so")
    }
}

fn configure_runtime_loader_paths(runtime_dir: &Path) {
    #[cfg(target_os = "windows")]
    {
        let dll_dirs = [
            runtime_dir.to_path_buf(),
            runtime_dir.join("vendor").join("ffmpeg").join("bin"),
            runtime_dir.join("vendor").join("pdfium"),
            runtime_dir.join("vendor").join("webrtc-audio-processing"),
        ];

        let mut parts = dll_dirs
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        if let Some(existing) = env::var_os("PATH") {
            parts.push(existing.to_string_lossy().to_string());
        }
        env::set_var("PATH", parts.join(";"));

        unsafe {
            let _ = SetDefaultDllDirectories(
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
            );
            for dir in dll_dirs.iter().filter(|path| path.exists()) {
                let wide = to_wide_null(dir);
                let _ = AddDllDirectory(wide.as_ptr());
            }
        }
    }
}

#[cfg(target_os = "windows")]
const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x00001000;
#[cfg(target_os = "windows")]
const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x00000400;

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
    fn AddDllDirectory(new_directory: *const u16) -> *mut c_void;
}

#[cfg(target_os = "windows")]
fn to_wide_null(path: &Path) -> Vec<u16> {
    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn cstr_from_mut(ptr: *mut c_char) -> String {
    cstr_from_const(ptr.cast_const())
}

fn cstr_from_const(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn cstr_opt(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}
