use crate::audio_assembler::{AssembleOptions, SpeakerSpan};
use crate::audio_orchestrator::{DiarizedTranscriptOrchestrator, OrchestratorSnapshot};
use serde_json::{json, Map, Value};
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{self, Read, Write};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;
use std::time::Instant;

const DEFAULT_VLM_PROMPT: &str = "Convert this page to markdown. Do not miss any text and only output the bare markdown! Any graphs or figures found convert to markdown table. If figure is image without details, describe what you see in the image. For tables, pay attention to whitespace: some cells may be intentionally empty, so keep empty and filled cells in the correct columns. Ensure correct assignment of column headings and subheadings for tables.";

#[repr(C)]
pub struct llama_server_bridge {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_server_bridge_audio_session {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_params {
    pub model_path: *const c_char,
    pub mmproj_path: *const c_char,

    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,

    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub gpu: i32,
    pub no_kv_offload: i32,
    pub mmproj_use_gpu: i32,
    pub cache_ram_mib: i32,

    pub seed: i32,
    pub ctx_shift: i32,
    pub kv_unified: i32,

    pub devices: *const c_char,
    pub tensor_split: *const c_char,
    pub split_mode: i32,

    pub embedding: i32,
    pub reranking: i32,
    pub pooling_type: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_chat_request {
    pub prompt: *const c_char,

    pub n_predict: i32,
    pub id_slot: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub seed: i32,

    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_vlm_request {
    pub prompt: *const c_char,
    pub image_bytes: *const u8,
    pub image_bytes_len: usize,

    pub n_predict: i32,
    pub id_slot: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub seed: i32,

    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_vlm_result {
    pub ok: i32,
    pub truncated: i32,
    pub stop: i32,
    pub n_decoded: i32,
    pub n_prompt_tokens: i32,
    pub n_tokens_cached: i32,
    pub eos_reached: i32,

    pub prompt_ms: f64,
    pub predicted_ms: f64,

    pub text: *mut c_char,
    pub error_json: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_embeddings_request {
    pub body_json: *const c_char,
    pub oai_compat: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_rerank_request {
    pub body_json: *const c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_request {
    pub body_json: *const c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_raw_request {
    pub audio_bytes: *const u8,
    pub audio_bytes_len: usize,
    pub audio_format: *const c_char,
    pub metadata_json: *const c_char,
    pub ffmpeg_convert: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_json_result {
    pub ok: i32,
    pub status: i32,
    pub json: *mut c_char,
    pub error_json: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_device_info {
    pub index: i32,
    pub r#type: i32,
    pub memory_free: u64,
    pub memory_total: u64,
    pub backend: *mut c_char,
    pub name: *mut c_char,
    pub description: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_realtime_params {
    pub backend_kind: i32,
    pub model_path: *const c_char,
    pub backend_name: *const c_char,
    pub expected_sample_rate_hz: u32,
    pub audio_ring_capacity_samples: u32,
    pub capture_debug: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_session_params {
    pub expected_input_sample_rate_hz: u32,
    pub expected_input_channels: u32,
    pub max_buffered_audio_samples: u32,
    pub event_queue_capacity: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_transcription_params {
    pub bridge_params: llama_server_bridge_params,
    pub metadata_json: *const c_char,
    pub mode: i32,
    pub realtime_params: llama_server_bridge_realtime_params,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_event {
    pub seq_no: u64,
    pub kind: i32,
    pub flags: u32,
    pub start_sample: u64,
    pub end_sample: u64,
    pub speaker_id: i32,
    pub item_id: u32,
    pub text: *mut c_char,
    pub detail: *mut c_char,
}

pub const LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_F32: i32 = 1;
pub const LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16: i32 = 2;

pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE: i32 = 0;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STARTED: i32 = 1;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED: i32 = 2;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT: i32 = 3;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT: i32 = 4;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_STATUS: i32 = 5;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR: i32 = 6;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STARTED: i32 = 7;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT: i32 = 8;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT: i32 = 9;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON: i32 = 10;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED: i32 = 11;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_STREAM_FLUSHED: i32 = 12;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR: i32 = 13;

pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL: u32 = 1u32 << 0;
pub const LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY: u32 = 1u32 << 1;
pub const LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE: i32 = 0;
pub const LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE: i32 = 1;

#[link(name = "llama-server-bridge")]
unsafe extern "C" {
    pub fn llama_server_bridge_default_params() -> llama_server_bridge_params;
    pub fn llama_server_bridge_default_chat_request() -> llama_server_bridge_chat_request;
    pub fn llama_server_bridge_default_vlm_request() -> llama_server_bridge_vlm_request;
    pub fn llama_server_bridge_empty_vlm_result() -> llama_server_bridge_vlm_result;
    pub fn llama_server_bridge_default_embeddings_request() -> llama_server_bridge_embeddings_request;
    pub fn llama_server_bridge_default_rerank_request() -> llama_server_bridge_rerank_request;
    pub fn llama_server_bridge_default_audio_request() -> llama_server_bridge_audio_request;
    pub fn llama_server_bridge_default_audio_raw_request() -> llama_server_bridge_audio_raw_request;
    pub fn llama_server_bridge_default_audio_session_params(
    ) -> llama_server_bridge_audio_session_params;
    pub fn llama_server_bridge_default_audio_transcription_params(
    ) -> llama_server_bridge_audio_transcription_params;
    pub fn llama_server_bridge_empty_json_result() -> llama_server_bridge_json_result;
    pub fn llama_server_bridge_default_realtime_params() -> llama_server_bridge_realtime_params;

    pub fn llama_server_bridge_create(
        params: *const llama_server_bridge_params,
    ) -> *mut llama_server_bridge;
    pub fn llama_server_bridge_destroy(bridge: *mut llama_server_bridge);

    pub fn llama_server_bridge_chat_complete(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_chat_request,
        out: *mut llama_server_bridge_vlm_result,
    ) -> i32;

    pub fn llama_server_bridge_vlm_complete(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_vlm_request,
        out: *mut llama_server_bridge_vlm_result,
    ) -> i32;

    pub fn llama_server_bridge_embeddings(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_embeddings_request,
        out: *mut llama_server_bridge_json_result,
    ) -> i32;

    pub fn llama_server_bridge_rerank(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_rerank_request,
        out: *mut llama_server_bridge_json_result,
    ) -> i32;

    pub fn llama_server_bridge_audio_transcriptions(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_audio_request,
        out: *mut llama_server_bridge_json_result,
    ) -> i32;

    pub fn llama_server_bridge_audio_transcriptions_raw(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_audio_raw_request,
        out: *mut llama_server_bridge_json_result,
    ) -> i32;

    pub fn llama_server_bridge_result_free(out: *mut llama_server_bridge_vlm_result);
    pub fn llama_server_bridge_json_result_free(out: *mut llama_server_bridge_json_result);
    pub fn llama_server_bridge_last_error(bridge: *const llama_server_bridge) -> *const c_char;

    pub fn llama_server_bridge_list_devices(
        out_devices: *mut *mut llama_server_bridge_device_info,
        out_count: *mut usize,
    ) -> i32;
    pub fn llama_server_bridge_free_devices(
        devices: *mut llama_server_bridge_device_info,
        count: usize,
    );

    pub fn llama_server_bridge_audio_session_create(
        params: *const llama_server_bridge_audio_session_params,
    ) -> *mut llama_server_bridge_audio_session;
    pub fn llama_server_bridge_audio_session_destroy(
        session: *mut llama_server_bridge_audio_session,
    );
    pub fn llama_server_bridge_audio_session_push_audio(
        session: *mut llama_server_bridge_audio_session,
        audio_bytes: *const c_void,
        frame_count: usize,
        sample_rate_hz: u32,
        channels: u32,
        sample_format: i32,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_push_encoded(
        session: *mut llama_server_bridge_audio_session,
        audio_bytes: *const u8,
        audio_bytes_len: usize,
        audio_format: *const c_char,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_flush_audio(
        session: *mut llama_server_bridge_audio_session,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_start_diarization(
        session: *mut llama_server_bridge_audio_session,
        params: *const llama_server_bridge_realtime_params,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_stop_diarization(
        session: *mut llama_server_bridge_audio_session,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_start_transcription(
        session: *mut llama_server_bridge_audio_session,
        params: *const llama_server_bridge_audio_transcription_params,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_stop_transcription(
        session: *mut llama_server_bridge_audio_session,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_wait_events(
        session: *mut llama_server_bridge_audio_session,
        timeout_ms: u32,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_drain_events(
        session: *mut llama_server_bridge_audio_session,
        out_events: *mut *mut llama_server_bridge_audio_event,
        out_count: *mut usize,
        max_events: usize,
    ) -> i32;
    pub fn llama_server_bridge_audio_session_free_events(
        events: *mut llama_server_bridge_audio_event,
        count: usize,
    );
    pub fn llama_server_bridge_audio_session_last_error(
        session: *const llama_server_bridge_audio_session,
    ) -> *const c_char;

    pub fn llama_server_bridge_realtime_model_cache_clear();
}

struct BridgeHandle {
    ptr: *mut llama_server_bridge,
}

impl Drop for BridgeHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                llama_server_bridge_destroy(self.ptr);
            }
        }
    }
}

struct AudioSessionHandle {
    ptr: *mut llama_server_bridge_audio_session,
}

impl Drop for AudioSessionHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                llama_server_bridge_audio_session_destroy(self.ptr);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioSessionEventOwned {
    pub seq_no: u64,
    pub kind: i32,
    pub flags: u32,
    pub start_sample: u64,
    pub end_sample: u64,
    pub speaker_id: i32,
    pub item_id: u32,
    pub text: String,
    pub detail: String,
}

pub struct AudioSession {
    handle: AudioSessionHandle,
}

struct RealtimeModelCacheGuard;

impl Drop for RealtimeModelCacheGuard {
    fn drop(&mut self) {
        unsafe {
            llama_server_bridge_realtime_model_cache_clear();
        }
    }
}

impl AudioSession {
    pub fn create(
        params: Option<llama_server_bridge_audio_session_params>,
    ) -> Result<Self, String> {
        let params =
            params.unwrap_or_else(|| unsafe { llama_server_bridge_default_audio_session_params() });
        let ptr = unsafe { llama_server_bridge_audio_session_create(&params) };
        if ptr.is_null() {
            return Err("llama_server_bridge_audio_session_create() failed".to_string());
        }
        Ok(Self {
            handle: AudioSessionHandle { ptr },
        })
    }

    pub fn raw_ptr(&self) -> *mut llama_server_bridge_audio_session {
        self.handle.ptr
    }

    pub fn last_error(&self) -> String {
        cstr_ptr_to_string(unsafe { llama_server_bridge_audio_session_last_error(self.handle.ptr) })
    }

    pub fn push_audio_f32(&mut self, samples: &[f32], sample_rate_hz: u32) -> Result<(), String> {
        let rc = unsafe {
            llama_server_bridge_audio_session_push_audio(
                self.handle.ptr,
                samples.as_ptr() as *const c_void,
                samples.len(),
                sample_rate_hz,
                1,
                LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_F32,
            )
        };
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn push_audio_s16(&mut self, samples: &[i16], sample_rate_hz: u32) -> Result<(), String> {
        let rc = unsafe {
            llama_server_bridge_audio_session_push_audio(
                self.handle.ptr,
                samples.as_ptr() as *const c_void,
                samples.len(),
                sample_rate_hz,
                1,
                LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16,
            )
        };
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn push_encoded(&mut self, audio_bytes: &[u8], audio_format: &str) -> Result<(), String> {
        let audio_format_c =
            CString::new(audio_format).map_err(|_| "audio format contains NUL byte".to_string())?;
        let rc = unsafe {
            llama_server_bridge_audio_session_push_encoded(
                self.handle.ptr,
                audio_bytes.as_ptr(),
                audio_bytes.len(),
                audio_format_c.as_ptr(),
            )
        };
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn flush_audio(&mut self) -> Result<(), String> {
        let rc = unsafe { llama_server_bridge_audio_session_flush_audio(self.handle.ptr) };
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn start_diarization(&mut self, params: &mut OwnedRealtimeParams) -> Result<(), String> {
        unsafe { self.start_diarization_raw(params.as_raw()) }
    }

    pub unsafe fn start_diarization_raw(
        &mut self,
        params: &llama_server_bridge_realtime_params,
    ) -> Result<(), String> {
        let rc = llama_server_bridge_audio_session_start_diarization(self.handle.ptr, params);
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn stop_diarization(&mut self) -> Result<(), String> {
        let rc = unsafe { llama_server_bridge_audio_session_stop_diarization(self.handle.ptr) };
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn start_transcription(
        &mut self,
        params: &mut OwnedAudioTranscriptionParams,
    ) -> Result<(), String> {
        unsafe { self.start_transcription_raw(params.as_raw()) }
    }

    pub unsafe fn start_transcription_raw(
        &mut self,
        params: &llama_server_bridge_audio_transcription_params,
    ) -> Result<(), String> {
        let rc = llama_server_bridge_audio_session_start_transcription(self.handle.ptr, params);
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn stop_transcription(&mut self) -> Result<(), String> {
        let rc = unsafe { llama_server_bridge_audio_session_stop_transcription(self.handle.ptr) };
        if rc != 0 {
            return Err(self.last_error());
        }
        Ok(())
    }

    pub fn wait_events(&mut self, timeout_ms: u32) -> Result<i32, String> {
        let rc =
            unsafe { llama_server_bridge_audio_session_wait_events(self.handle.ptr, timeout_ms) };
        if rc < 0 {
            return Err(self.last_error());
        }
        Ok(rc)
    }

    pub fn drain_events(
        &mut self,
        max_events: usize,
    ) -> Result<Vec<AudioSessionEventOwned>, String> {
        let mut ptr_events: *mut llama_server_bridge_audio_event = ptr::null_mut();
        let mut count: usize = 0;
        let rc = unsafe {
            llama_server_bridge_audio_session_drain_events(
                self.handle.ptr,
                &mut ptr_events,
                &mut count,
                max_events,
            )
        };
        if rc != 0 {
            return Err(self.last_error());
        }
        if ptr_events.is_null() || count == 0 {
            return Ok(Vec::new());
        }

        let slice = unsafe { std::slice::from_raw_parts(ptr_events, count) };
        let mut out = Vec::with_capacity(count);
        for ev in slice {
            out.push(AudioSessionEventOwned {
                seq_no: ev.seq_no,
                kind: ev.kind,
                flags: ev.flags,
                start_sample: ev.start_sample,
                end_sample: ev.end_sample,
                speaker_id: ev.speaker_id,
                item_id: ev.item_id,
                text: cstr_mut_ptr_to_string(ev.text),
                detail: cstr_mut_ptr_to_string(ev.detail),
            });
        }
        unsafe {
            llama_server_bridge_audio_session_free_events(ptr_events, count);
        }
        Ok(out)
    }
}

pub struct OwnedBridgeParams {
    raw: llama_server_bridge_params,
    model_path: Option<CString>,
    mmproj_path: Option<CString>,
    devices: Option<CString>,
    tensor_split: Option<CString>,
}

impl OwnedBridgeParams {
    pub fn new() -> Self {
        Self {
            raw: unsafe { llama_server_bridge_default_params() },
            model_path: None,
            mmproj_path: None,
            devices: None,
            tensor_split: None,
        }
    }

    pub fn set_model_path(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.model_path = match value {
            Some(v) => {
                Some(CString::new(v).map_err(|_| "model_path contains NUL byte".to_string())?)
            }
            None => None,
        };
        Ok(self)
    }

    pub fn set_mmproj_path(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.mmproj_path = match value {
            Some(v) => {
                Some(CString::new(v).map_err(|_| "mmproj_path contains NUL byte".to_string())?)
            }
            None => None,
        };
        Ok(self)
    }

    pub fn set_devices(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.devices = match value {
            Some(v) => Some(CString::new(v).map_err(|_| "devices contains NUL byte".to_string())?),
            None => None,
        };
        Ok(self)
    }

    pub fn set_tensor_split(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.tensor_split = match value {
            Some(v) => {
                Some(CString::new(v).map_err(|_| "tensor_split contains NUL byte".to_string())?)
            }
            None => None,
        };
        Ok(self)
    }

    pub fn raw_mut(&mut self) -> &mut llama_server_bridge_params {
        self.raw.model_path = self.model_path.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        self.raw.mmproj_path = self
            .mmproj_path
            .as_ref()
            .map_or(ptr::null(), |v| v.as_ptr());
        self.raw.devices = self.devices.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        self.raw.tensor_split = self
            .tensor_split
            .as_ref()
            .map_or(ptr::null(), |v| v.as_ptr());
        &mut self.raw
    }
}

impl Default for OwnedBridgeParams {
    fn default() -> Self {
        Self::new()
    }
}

pub struct OwnedRealtimeParams {
    raw: llama_server_bridge_realtime_params,
    model_path: Option<CString>,
    backend_name: Option<CString>,
}

impl OwnedRealtimeParams {
    pub fn new() -> Self {
        Self {
            raw: unsafe { llama_server_bridge_default_realtime_params() },
            model_path: None,
            backend_name: None,
        }
    }

    pub fn set_backend_kind(&mut self, backend_kind: i32) -> &mut Self {
        self.raw.backend_kind = backend_kind;
        self
    }

    pub fn set_model_path(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.model_path = match value {
            Some(v) => Some(
                CString::new(v).map_err(|_| "realtime model path contains NUL byte".to_string())?,
            ),
            None => None,
        };
        Ok(self)
    }

    pub fn set_backend_name(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.backend_name = match value {
            Some(v) => {
                Some(CString::new(v).map_err(|_| "backend_name contains NUL byte".to_string())?)
            }
            None => None,
        };
        Ok(self)
    }

    pub fn set_expected_sample_rate_hz(&mut self, value: u32) -> &mut Self {
        self.raw.expected_sample_rate_hz = value;
        self
    }

    pub fn set_audio_ring_capacity_samples(&mut self, value: u32) -> &mut Self {
        self.raw.audio_ring_capacity_samples = value;
        self
    }

    pub fn set_capture_debug(&mut self, value: bool) -> &mut Self {
        self.raw.capture_debug = if value { 1 } else { 0 };
        self
    }

    pub fn as_raw(&mut self) -> &llama_server_bridge_realtime_params {
        self.raw.model_path = self.model_path.as_ref().map_or(ptr::null(), |v| v.as_ptr());
        self.raw.backend_name = self
            .backend_name
            .as_ref()
            .map_or(ptr::null(), |v| v.as_ptr());
        &self.raw
    }
}

impl Default for OwnedRealtimeParams {
    fn default() -> Self {
        Self::new()
    }
}

pub struct OwnedAudioTranscriptionParams {
    raw: llama_server_bridge_audio_transcription_params,
    bridge_params: OwnedBridgeParams,
    metadata_json: Option<CString>,
    realtime_params: OwnedRealtimeParams,
}

impl OwnedAudioTranscriptionParams {
    pub fn new() -> Self {
        Self {
            raw: unsafe { llama_server_bridge_default_audio_transcription_params() },
            bridge_params: OwnedBridgeParams::new(),
            metadata_json: None,
            realtime_params: OwnedRealtimeParams::new(),
        }
    }

    pub fn bridge_params_mut(&mut self) -> &mut OwnedBridgeParams {
        &mut self.bridge_params
    }

    pub fn set_metadata_json(&mut self, value: Option<&str>) -> Result<&mut Self, String> {
        self.metadata_json = match value {
            Some(v) => {
                Some(CString::new(v).map_err(|_| "metadata_json contains NUL byte".to_string())?)
            }
            None => None,
        };
        Ok(self)
    }

    pub fn set_mode(&mut self, value: i32) -> &mut Self {
        self.raw.mode = value;
        self
    }

    pub fn realtime_params_mut(&mut self) -> &mut OwnedRealtimeParams {
        &mut self.realtime_params
    }

    pub fn as_raw(&mut self) -> &llama_server_bridge_audio_transcription_params {
        self.raw.bridge_params = *self.bridge_params.raw_mut();
        self.raw.metadata_json = self
            .metadata_json
            .as_ref()
            .map_or(ptr::null(), |v| v.as_ptr());
        self.raw.realtime_params = *self.realtime_params.as_raw();
        &self.raw
    }
}

impl Default for OwnedAudioTranscriptionParams {
    fn default() -> Self {
        Self::new()
    }
}

fn cstr_ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn cstr_mut_ptr_to_string(ptr: *mut c_char) -> String {
    cstr_ptr_to_string(ptr as *const c_char)
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    let mut i = 0usize;
    while i + 1 < args.len() {
        if args[i] == key {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}

fn arg_value_spanned(args: &[String], key: &str, stop_keys: &[&str]) -> Option<String> {
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == key {
            let start = i + 1;
            if start >= args.len() {
                return Some(String::new());
            }

            let mut end = start;
            while end < args.len() {
                let token = args[end].as_str();
                if stop_keys.iter().any(|k| *k == token) {
                    break;
                }
                end += 1;
            }
            return Some(args[start..end].join(" "));
        }
        i += 1;
    }
    None
}

fn has_arg(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

#[cfg(windows)]
fn prepare_windows_bridge_runtime_paths() {
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::iter;
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use std::sync::Once;

    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x00001000;
    const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x00000400;

    unsafe extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut c_void;
        fn SetDllDirectoryW(path_name: *const u16) -> i32;
    }

    fn as_wide_null(path: &Path) -> Vec<u16> {
        path.as_os_str()
            .encode_wide()
            .chain(iter::once(0))
            .collect()
    }

    fn candidate_dirs() -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(exe_path) = env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                out.push(exe_dir.join("vendor").join("ffmpeg").join("bin"));
            }
        }
        if let Ok(cwd) = env::current_dir() {
            out.push(cwd.join("vendor").join("ffmpeg").join("bin"));
        }
        out
    }

    fn add_search_dir(path: &Path) {
        if !path.is_dir() {
            return;
        }
        let wide = as_wide_null(path);
        unsafe {
            let cookie = AddDllDirectory(wide.as_ptr());
            if cookie.is_null() {
                let _ = SetDllDirectoryW(wide.as_ptr());
            }
        }
    }

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe {
            let _ = SetDefaultDllDirectories(
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
            );
        }

        let mut seen = HashSet::new();
        for dir in candidate_dirs() {
            let key = dir.to_string_lossy().to_string();
            if seen.insert(key) {
                add_search_dir(&dir);
            }
        }
    });
}

#[cfg(not(windows))]
fn prepare_windows_bridge_runtime_paths() {}

fn parse_usize_arg(args: &[String], key: &str, default_value: usize) -> Result<usize, String> {
    match arg_value(args, key) {
        Some(v) => v
            .parse::<usize>()
            .map_err(|e| format!("invalid value for {key}: {e}")),
        None => Ok(default_value),
    }
}

fn parse_i32_arg(args: &[String], key: &str, default_value: i32) -> Result<i32, String> {
    match arg_value(args, key) {
        Some(v) => v
            .parse::<i32>()
            .map_err(|e| format!("invalid value for {key}: {e}")),
        None => Ok(default_value),
    }
}

fn parse_optional_i32_arg(args: &[String], key: &str) -> Result<Option<i32>, String> {
    match arg_value(args, key) {
        Some(v) => v
            .parse::<i32>()
            .map(Some)
            .map_err(|e| format!("invalid value for {key}: {e}")),
        None => Ok(None),
    }
}

fn parse_optional_positive_i32_arg(args: &[String], key: &str) -> Result<Option<i32>, String> {
    let value = parse_optional_i32_arg(args, key)?;
    if let Some(v) = value {
        if v <= 0 {
            return Err(format!("{key} must be > 0"));
        }
    }
    Ok(value)
}

fn parse_optional_f64_arg(args: &[String], key: &str) -> Result<Option<f64>, String> {
    match arg_value(args, key) {
        Some(v) => v
            .parse::<f64>()
            .map(Some)
            .map_err(|e| format!("invalid value for {key}: {e}")),
        None => Ok(None),
    }
}

fn parse_u32_arg(args: &[String], key: &str, default_value: u32) -> Result<u32, String> {
    match arg_value(args, key) {
        Some(v) => v
            .parse::<u32>()
            .map_err(|e| format!("invalid value for {key}: {e}")),
        None => Ok(default_value),
    }
}

fn split_mode_arg_to_i32(value: &str) -> Result<i32, String> {
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(0),
        "layer" => Ok(1),
        "row" => Ok(2),
        _ => Err("invalid --split-mode, expected none/layer/row".to_string()),
    }
}

fn resolve_device_name_by_index(index: i32) -> Result<String, String> {
    if index < 0 {
        return Err("--gpu must be >= 0".to_string());
    }

    prepare_windows_bridge_runtime_paths();

    let mut ptr_devices = ptr::null_mut();
    let mut count: usize = 0;
    let rc = unsafe { llama_server_bridge_list_devices(&mut ptr_devices, &mut count) };
    if rc != 0 {
        return Err("llama_server_bridge_list_devices() failed".to_string());
    }

    let mut resolved: Option<String> = None;
    for i in 0..count {
        let dev = unsafe { &*ptr_devices.add(i) };
        if dev.index == index {
            resolved = Some(cstr_mut_ptr_to_string(dev.name));
            break;
        }
    }

    unsafe {
        llama_server_bridge_free_devices(ptr_devices, count);
    }

    match resolved {
        Some(v) if !v.trim().is_empty() => Ok(v),
        _ => Err(format!("invalid --gpu index: {index}")),
    }
}

fn apply_audio_cli_overrides(
    args: &[String],
    body_obj: &mut Map<String, Value>,
) -> Result<(), String> {
    let gpu_index = parse_optional_i32_arg(args, "--gpu")?;
    if let Some(gpu) = gpu_index {
        if gpu < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if arg_value(args, "--whisper-gpu-device").is_none() && !has_arg(args, "--whisper-no-gpu") {
            body_obj.insert("whisper_gpu_device".to_string(), json!(gpu));
        }
        if arg_value(args, "--diarization-device").is_none() {
            let device_name = resolve_device_name_by_index(gpu)?;
            body_obj.insert("diarization_device".to_string(), json!(device_name));
        }
    }

    if let Some(v) = arg_value(args, "--transcription-backend") {
        body_obj.insert("transcription_backend".to_string(), json!(v));
    }
    if let Some(v) = arg_value(args, "--diarization-backend") {
        body_obj.insert("diarization_backend".to_string(), json!(v));
    }
    if let Some(v) = arg_value(args, "--diarization-device") {
        body_obj.insert("diarization_device".to_string(), json!(v));
    }
    if let Some(v) = arg_value(args, "--diarization-model-path") {
        body_obj.insert("diarization_model_path".to_string(), json!(v));
    }
    if let Some(v) = arg_value(args, "--diarization-models-dir")
        .or_else(|| arg_value(args, "--diarization-model-dir"))
    {
        body_obj.insert("diarization_models_dir".to_string(), json!(v));
    }

    if let Some(v) = parse_optional_i32_arg(args, "--whisper-threads")? {
        body_obj.insert("whisper_threads".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--whisper-processors")? {
        body_obj.insert("whisper_processors".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--whisper-max-len")? {
        body_obj.insert("whisper_max_len".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--whisper-audio-ctx")? {
        body_obj.insert("whisper_audio_ctx".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--whisper-best-of")? {
        body_obj.insert("whisper_best_of".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--whisper-beam-size")? {
        body_obj.insert("whisper_beam_size".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--whisper-gpu-device")? {
        body_obj.insert("whisper_gpu_device".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--whisper-temperature")? {
        body_obj.insert("whisper_temperature".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--whisper-word-time-offset-sec")? {
        body_obj.insert("whisper_word_time_offset_sec".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--seconds-per-timeline-token")? {
        body_obj.insert("seconds_per_timeline_token".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--source-audio-seconds")? {
        body_obj.insert("source_audio_seconds".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--diarization-feed-ms")? {
        body_obj.insert("diarization_feed_ms".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--speaker-seg-max-gap-sec")? {
        body_obj.insert("speaker_seg_max_gap_sec".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_i32_arg(args, "--speaker-seg-max-words")? {
        body_obj.insert("speaker_seg_max_words".to_string(), json!(v));
    }
    if let Some(v) = parse_optional_f64_arg(args, "--speaker-seg-max-duration-sec")? {
        body_obj.insert("speaker_seg_max_duration_sec".to_string(), json!(v));
    }

    if let Some(v) = arg_value(args, "--whisper-language") {
        body_obj.insert("whisper_language".to_string(), json!(v));
    }
    if let Some(v) = arg_value(args, "--whisper-prompt") {
        body_obj.insert("whisper_prompt".to_string(), json!(v));
    }

    if has_arg(args, "--whisper-translate") {
        body_obj.insert("whisper_translate".to_string(), json!(true));
    }
    if has_arg(args, "--whisper-no-fallback") {
        body_obj.insert("whisper_no_fallback".to_string(), json!(true));
    }
    if has_arg(args, "--whisper-suppress-nst") {
        body_obj.insert("whisper_suppress_nst".to_string(), json!(true));
    }
    if has_arg(args, "--whisper-no-gpu") {
        body_obj.insert("whisper_no_gpu".to_string(), json!(true));
    }
    if has_arg(args, "--whisper-offline") {
        body_obj.insert("whisper_offline".to_string(), json!(true));
    }

    let set_flash_attn = has_arg(args, "--whisper-flash-attn");
    let unset_flash_attn = has_arg(args, "--whisper-no-flash-attn");
    if set_flash_attn && unset_flash_attn {
        return Err("choose one: --whisper-flash-attn OR --whisper-no-flash-attn".to_string());
    }
    if set_flash_attn {
        body_obj.insert("whisper_flash_attn".to_string(), json!(true));
    }
    if unset_flash_attn {
        body_obj.insert("whisper_flash_attn".to_string(), json!(false));
    }

    let split_on_hard_break = has_arg(args, "--speaker-seg-split-on-hard-break");
    let no_split_on_hard_break = has_arg(args, "--speaker-seg-no-split-on-hard-break");
    if split_on_hard_break && no_split_on_hard_break {
        return Err(
            "choose one: --speaker-seg-split-on-hard-break OR --speaker-seg-no-split-on-hard-break"
                .to_string(),
        );
    }
    if split_on_hard_break {
        body_obj.insert("speaker_seg_split_on_hard_break".to_string(), json!(true));
    }
    if no_split_on_hard_break {
        body_obj.insert("speaker_seg_split_on_hard_break".to_string(), json!(false));
    }

    Ok(())
}

fn chunk_text_by_words(input: &str, words_per_chunk: usize) -> Vec<String> {
    if words_per_chunk == 0 {
        return Vec::new();
    }

    let words: Vec<&str> = input.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < words.len() {
        let end = (i + words_per_chunk).min(words.len());
        out.push(words[i..end].join(" "));
        i = end;
    }
    out
}

fn build_text_pool(markdown_path: &str, words_per_chunk: usize) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(markdown_path)
        .map_err(|e| format!("failed to read markdown file '{markdown_path}': {e}"))?;
    let chunks = chunk_text_by_words(&raw, words_per_chunk);
    if chunks.is_empty() {
        return Err("markdown chunking produced 0 chunks".to_string());
    }
    Ok(chunks)
}

fn read_inline_or_file_json(raw_or_path: &str, label: &str) -> Result<String, String> {
    let path = std::path::Path::new(raw_or_path);
    if path.exists() {
        fs::read_to_string(path)
            .map_err(|e| format!("failed to read {label} file '{raw_or_path}': {e}"))
    } else {
        Ok(raw_or_path.to_string())
    }
}

fn make_bridge(
    model_path: &str,
    mmproj_path: Option<&str>,
    mmproj_use_gpu: Option<i32>,
    gpu: i32,
    devices: Option<&str>,
    tensor_split: Option<&str>,
    split_mode: i32,
    n_ctx: i32,
    n_batch: i32,
    n_ubatch: i32,
    n_parallel: i32,
    n_threads: Option<i32>,
    n_threads_batch: Option<i32>,
    n_gpu_layers: i32,
    main_gpu: i32,
    embedding: bool,
    reranking: bool,
) -> Result<BridgeHandle, String> {
    prepare_windows_bridge_runtime_paths();

    let model_c =
        CString::new(model_path).map_err(|_| "model path contains NUL byte".to_string())?;
    let mmproj_c = if let Some(v) = mmproj_path {
        Some(CString::new(v).map_err(|_| "mmproj path contains NUL byte".to_string())?)
    } else {
        None
    };
    let devices_c = if let Some(v) = devices {
        Some(CString::new(v).map_err(|_| "devices string contains NUL byte".to_string())?)
    } else {
        None
    };
    let tensor_split_c = if let Some(v) = tensor_split {
        Some(CString::new(v).map_err(|_| "tensor_split string contains NUL byte".to_string())?)
    } else {
        None
    };

    let mut params: llama_server_bridge_params = unsafe { llama_server_bridge_default_params() };
    params.model_path = model_c.as_ptr();
    params.mmproj_path = mmproj_c.as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null());
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.n_parallel = n_parallel;
    if let Some(v) = n_threads {
        params.n_threads = v;
    }
    if let Some(v) = n_threads_batch {
        params.n_threads_batch = v;
    }
    params.n_gpu_layers = n_gpu_layers;
    params.main_gpu = main_gpu;
    params.gpu = gpu;
    params.no_kv_offload = 0;
    params.mmproj_use_gpu = if let Some(v) = mmproj_use_gpu {
        v
    } else if mmproj_c.is_some() {
        1
    } else {
        0
    };
    params.cache_ram_mib = 0;
    params.ctx_shift = 1;
    params.kv_unified = 1;
    params.devices = devices_c
        .as_ref()
        .map(|s| s.as_ptr())
        .unwrap_or(ptr::null());
    params.tensor_split = tensor_split_c
        .as_ref()
        .map(|s| s.as_ptr())
        .unwrap_or(ptr::null());
    params.split_mode = split_mode;
    params.embedding = if embedding { 1 } else { 0 };
    params.reranking = if reranking { 1 } else { 0 };
    params.pooling_type = -1;

    let ptr = unsafe { llama_server_bridge_create(&params) };
    if ptr.is_null() {
        return Err("llama_server_bridge_create() failed".to_string());
    }

    Ok(BridgeHandle { ptr })
}

fn bridge_cli_usage() -> &'static str {
    "bridge usage:
  list-devices
  vlm --model <gguf> --mmproj <gguf> --image <png/jpg/webp> [--prompt <text>] [--out <md>] [--n-predict 5000] [--mmproj-use-gpu <-1|0|1>]
  audio --audio-file <audio-file> [--audio-format <format-hint>] [--output-dir <dir>] --mode <subtitle|speech|transcript> --custom <value> [whisper source + diarization source flags + advanced audio knobs]
  audio-session (--audio-file <audio-file> [--audio-format <format-hint>] | --stdin-pcm-s16le | --stdin-pcm-f32le) [--diarization-model-path <gguf>] [--whisper-model <bin> | --whisper-hf-repo <repo> --whisper-hf-file <file> | --transcription-realtime-model <gguf>] [--out <md>] [--timeline-json-out <json>] [--staged]
  chat --model <gguf> [--prompt <text>] [--markdown <md>] [--out <md>] [--devices <csv>] [--n-predict 10000]
  embed --model <gguf> (--markdown <md> | --body-json <json-or-path>) [--out <json>] [--devices <csv>] [--batch-size 32] [--chunk-words 500] [--batches 8]
  rerank --model <gguf> (--markdown <md> | --body-json <json-or-path>) [--out <json>] [--devices <csv>] [--docs-per-query 32] [--chunk-words 500] [--batches 8]
shared optional flags:
  --gpu <device-index> (single-device shortcut)
  --n-ctx <int> --n-batch <int> --n-ubatch <int> --n-parallel <int> --n-gpu-layers <int> --main-gpu <int>
  --threads <int> --threads-batch <int>
  --split-mode <none|layer|row> --tensor-split <csv>
defaults:
  if neither --gpu nor --devices is set: macOS => first GPU, others => CPU-only
  if --gpu is set, defaults are single-device full offload on that device
audio advanced flags:
  --ffmpeg-convert --no-ffmpeg-convert (--audio-only is accepted but no longer required)
  --transcription-backend <auto|whisper_cpp_inproc>
  --whisper-threads <int> --whisper-processors <int> --whisper-max-len <int> --whisper-audio-ctx <int>
  --whisper-best-of <int> --whisper-beam-size <int> --whisper-temperature <float>
  --whisper-language <lang|auto> --whisper-prompt <text> --whisper-translate --whisper-no-fallback --whisper-suppress-nst
  --whisper-no-gpu --whisper-gpu-device <int> --whisper-flash-attn|--whisper-no-flash-attn --whisper-offline
  --whisper-word-time-offset-sec <float> --seconds-per-timeline-token <float> --source-audio-seconds <float>
  --diarization-backend <native_cpp|auto> --diarization-offline --diarization-device <device>
  --diarization-embedding-min-segment-duration-sec <float> --diarization-embedding-max-segments-per-speaker <int>
  --diarization-min-duration-off-sec <float> --speaker-seg-max-gap-sec <float> --speaker-seg-max-words <int>
  --speaker-seg-max-duration-sec <float> --speaker-seg-split-on-hard-break|--speaker-seg-no-split-on-hard-break
audio-session specific flags:
  --no-diarization --no-transcription --staged
  --session-sample-rate <hz> --session-channels <n> --session-max-buffered-samples <n> --session-event-queue-capacity <n>
  --stdin-pcm-s16le | --stdin-pcm-f32le --stdin-chunk-frames <frames>
  --diarization-model <gguf> | --diarization-model-path <gguf> --diarization-device <name> --diarization-ring-capacity-samples <n>
  --metadata-json <json-or-path> --timeline-json-out <path>
  --alignment-offset-ms <float> --nearest-tolerance-ms <float> --wait-ms <ms>"
}

fn run_list_devices() -> Result<(), String> {
    prepare_windows_bridge_runtime_paths();

    let mut ptr_devices = ptr::null_mut();
    let mut count: usize = 0;
    let rc = unsafe { llama_server_bridge_list_devices(&mut ptr_devices, &mut count) };
    if rc != 0 {
        return Err("llama_server_bridge_list_devices() failed".to_string());
    }

    println!("devices_count={count}");
    for i in 0..count {
        let dev = unsafe { &*ptr_devices.add(i) };
        let free_mib = (dev.memory_free as f64) / 1024.0 / 1024.0;
        let total_mib = (dev.memory_total as f64) / 1024.0 / 1024.0;
        println!(
            "device index={} backend={} name={} desc={} type={} free_mib={:.1} total_mib={:.1}",
            dev.index,
            cstr_mut_ptr_to_string(dev.backend),
            cstr_mut_ptr_to_string(dev.name),
            cstr_mut_ptr_to_string(dev.description),
            dev.r#type,
            free_mib,
            total_mib
        );
    }

    unsafe {
        llama_server_bridge_free_devices(ptr_devices, count);
    }
    Ok(())
}

fn infer_audio_format(audio_path: &str, override_value: Option<String>) -> Result<String, String> {
    if let Some(value) = override_value {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err("--audio-format must not be empty".to_string());
        }
        return Ok(trimmed.to_ascii_lowercase());
    }

    let extension = Path::new(audio_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .filter(|ext| !ext.is_empty());
    extension
        .ok_or("--audio-format is required when the audio file extension is missing".to_string())
}

fn read_le_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let slice = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn try_parse_wav_mono_s16le(bytes: &[u8]) -> Result<Option<(Vec<i16>, u32)>, String> {
    if bytes.len() < 12 {
        return Ok(None);
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Ok(None);
    }

    let mut fmt_channels: Option<u16> = None;
    let mut fmt_sample_rate: Option<u32> = None;
    let mut fmt_audio_format: Option<u16> = None;
    let mut fmt_bits_per_sample: Option<u16> = None;
    let mut data_chunk: Option<&[u8]> = None;

    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let Some(chunk_size_u32) = read_le_u32(bytes, offset + 4) else {
            break;
        };
        let chunk_size = chunk_size_u32 as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start.saturating_add(chunk_size);
        if chunk_end > bytes.len() {
            return Err("WAV chunk extends past end of file".to_string());
        }

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err("WAV fmt chunk is too small".to_string());
            }
            fmt_audio_format = read_le_u16(bytes, chunk_start);
            fmt_channels = read_le_u16(bytes, chunk_start + 2);
            fmt_sample_rate = read_le_u32(bytes, chunk_start + 4);
            fmt_bits_per_sample = read_le_u16(bytes, chunk_start + 14);
        } else if chunk_id == b"data" {
            data_chunk = Some(&bytes[chunk_start..chunk_end]);
        }

        let padded_size = if chunk_size % 2 == 0 {
            chunk_size
        } else {
            chunk_size + 1
        };
        offset = chunk_start.saturating_add(padded_size);
    }

    let Some(audio_format) = fmt_audio_format else {
        return Err("WAV fmt chunk is missing".to_string());
    };
    let Some(channels) = fmt_channels else {
        return Err("WAV channel count is missing".to_string());
    };
    let Some(sample_rate_hz) = fmt_sample_rate else {
        return Err("WAV sample rate is missing".to_string());
    };
    let Some(bits_per_sample) = fmt_bits_per_sample else {
        return Err("WAV bits_per_sample is missing".to_string());
    };
    let Some(data) = data_chunk else {
        return Err("WAV data chunk is missing".to_string());
    };

    if audio_format != 1 {
        return Err(format!(
            "WAV audio format {} is unsupported for direct PCM ingest; expected PCM",
            audio_format
        ));
    }
    if channels != 1 {
        return Err(format!(
            "WAV channel count {} is unsupported for direct PCM ingest; expected mono",
            channels
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "WAV bits_per_sample {} is unsupported for direct PCM ingest; expected 16",
            bits_per_sample
        ));
    }
    if data.len() % 2 != 0 {
        return Err("WAV data chunk has an odd byte length".to_string());
    }

    let mut samples = Vec::with_capacity(data.len() / 2);
    let mut idx = 0usize;
    while idx + 1 < data.len() {
        samples.push(i16::from_le_bytes([data[idx], data[idx + 1]]));
        idx += 2;
    }
    Ok(Some((samples, sample_rate_hz)))
}

fn sample_to_hms(sample: u64, sample_rate_hz: u32) -> String {
    if sample_rate_hz == 0 {
        return "00:00.000".to_string();
    }
    let total_ms = sample.saturating_mul(1000) / sample_rate_hz as u64;
    let minutes = total_ms / 60_000;
    let seconds = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    format!("{minutes:02}:{seconds:02}.{millis:03}")
}

fn render_speaker_spans(spans: &[SpeakerSpan], sample_rate_hz: u32) -> String {
    let mut out = String::new();
    for span in spans {
        if !out.is_empty() {
            out.push('\n');
            out.push('\n');
        }
        out.push_str("### ");
        out.push_str(&span.speaker);
        out.push(' ');
        out.push('[');
        out.push_str(&sample_to_hms(span.start_sample, sample_rate_hz));
        out.push_str(" - ");
        out.push_str(&sample_to_hms(span.end_sample, sample_rate_hz));
        out.push(']');
    }
    out
}

#[derive(Clone, Debug)]
enum AudioSessionCliInput {
    AudioFile {
        path: String,
        audio_format: String,
        chunk_frames: usize,
    },
    StdinPcmS16Le { chunk_frames: usize },
    StdinPcmF32Le { chunk_frames: usize },
}

impl AudioSessionCliInput {
    fn summary_audio_format(&self) -> &str {
        match self {
            Self::AudioFile { audio_format, .. } => audio_format.as_str(),
            Self::StdinPcmS16Le { .. } => "stdin-pcm-s16le",
            Self::StdinPcmF32Le { .. } => "stdin-pcm-f32le",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AudioSessionRunResult {
    pub snapshot: OrchestratorSnapshot,
    pub output_text: String,
}

struct RollingAudioSessionReporter {
    out_path: Option<String>,
    timeline_json_out: Option<String>,
    print_stdout: bool,
    last_output_text: Option<String>,
    last_timeline_json: Option<String>,
}

impl RollingAudioSessionReporter {
    fn new(out_path: Option<String>, timeline_json_out: Option<String>, print_stdout: bool) -> Self {
        Self {
            out_path,
            timeline_json_out,
            print_stdout,
            last_output_text: None,
            last_timeline_json: None,
        }
    }

    fn emit(&mut self, result: &AudioSessionRunResult) -> Result<(), String> {
        if let (Some(path), Some(raw_json)) = (
            self.timeline_json_out.as_ref(),
            result.snapshot.latest_transcription_json.as_ref(),
        ) {
            if self.last_timeline_json.as_deref() != Some(raw_json.as_str()) {
                if let Some(parent) = std::path::Path::new(path).parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent).map_err(|e| {
                            format!("failed to create timeline JSON parent directory '{path}': {e}")
                        })?;
                    }
                }
                fs::write(path, raw_json)
                    .map_err(|e| format!("failed to write timeline JSON '{path}': {e}"))?;
                self.last_timeline_json = Some(raw_json.clone());
            }
        }

        if self.last_output_text.as_deref() == Some(result.output_text.as_str()) {
            return Ok(());
        }

        if let Some(path) = self.out_path.as_ref() {
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(|e| {
                        format!("failed to create output parent directory '{path}': {e}")
                    })?;
                }
            }
            fs::write(path, &result.output_text)
                .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
        }

        if self.print_stdout && !result.output_text.is_empty() {
            let mut stdout = io::stdout().lock();
            match self.last_output_text.as_deref() {
                Some(previous) => {
                    if let Some(delta) = result.output_text.strip_prefix(previous) {
                        stdout
                            .write_all(delta.as_bytes())
                            .map_err(|e| format!("failed to write rolling stdout update: {e}"))?;
                        if !delta.ends_with('\n') {
                            stdout
                                .write_all(b"\n")
                                .map_err(|e| format!("failed to write rolling stdout newline: {e}"))?;
                        }
                    } else {
                        stdout
                            .write_all(b"\n--- audio-session update ---\n")
                            .and_then(|_| stdout.write_all(result.output_text.as_bytes()))
                            .map_err(|e| format!("failed to write rolling stdout snapshot: {e}"))?;
                        if !result.output_text.ends_with('\n') {
                            stdout
                                .write_all(b"\n")
                                .map_err(|e| format!("failed to write rolling stdout newline: {e}"))?;
                        }
                    }
                }
                None => {
                    stdout
                        .write_all(result.output_text.as_bytes())
                        .map_err(|e| format!("failed to write rolling stdout output: {e}"))?;
                    if !result.output_text.ends_with('\n') {
                        stdout
                            .write_all(b"\n")
                            .map_err(|e| format!("failed to write rolling stdout newline: {e}"))?;
                    }
                }
            }
            stdout
                .flush()
                .map_err(|e| format!("failed to flush rolling stdout output: {e}"))?;
        }

        self.last_output_text = Some(result.output_text.clone());
        Ok(())
    }
}

pub struct AudioSessionRunner {
    session: AudioSession,
    orchestrator: DiarizedTranscriptOrchestrator,
    transcription_params: Option<OwnedAudioTranscriptionParams>,
    sample_rate_hz: u32,
    wait_ms: u32,
    want_diarization: bool,
    want_transcription: bool,
    staged: bool,
    audio_flushed: bool,
    diarization_running: bool,
    transcription_running: bool,
    diarization_done: bool,
    transcription_done: bool,
    fatal_event_error: Option<String>,
}

impl AudioSessionRunner {
    pub fn new(
        session_params: llama_server_bridge_audio_session_params,
        sample_rate_hz: u32,
        wait_ms: u32,
        staged: bool,
        options: AssembleOptions,
        diarization_params: Option<OwnedRealtimeParams>,
        transcription_params: Option<OwnedAudioTranscriptionParams>,
    ) -> Result<Self, String> {
        let want_diarization = diarization_params.is_some();
        let want_transcription = transcription_params.is_some();
        let mut session = AudioSession::create(Some(session_params))?;

        let mut diarization_running = false;
        if let Some(mut params) = diarization_params {
            session.start_diarization(&mut params)?;
            diarization_running = true;
        }

        let mut out = Self {
            session,
            orchestrator: DiarizedTranscriptOrchestrator::with_options(sample_rate_hz, options),
            transcription_params,
            sample_rate_hz,
            wait_ms,
            want_diarization,
            want_transcription,
            staged,
            audio_flushed: false,
            diarization_running,
            transcription_running: false,
            diarization_done: !want_diarization,
            transcription_done: !want_transcription,
            fatal_event_error: None,
        };

        if want_transcription && !staged {
            out.start_transcription_if_needed()?;
        }

        Ok(out)
    }

    fn current_result(&self) -> AudioSessionRunResult {
        let snapshot = self.orchestrator.snapshot();
        let output_text = render_audio_session_output(
            &snapshot,
            self.want_diarization,
            self.want_transcription,
            self.sample_rate_hz,
        );
        AudioSessionRunResult {
            snapshot,
            output_text,
        }
    }

    fn start_transcription_if_needed_with_progress<F>(&mut self, on_update: &mut F) -> Result<(), String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        if !self.want_transcription || self.transcription_running {
            return Ok(());
        }
        let params = self
            .transcription_params
            .as_mut()
            .ok_or("internal transcription parameter state missing".to_string())?;
        self.session.start_transcription(params)?;
        self.transcription_running = true;
        self.pump_pending_events_with_progress(on_update)?;
        Ok(())
    }

    fn start_transcription_if_needed(&mut self) -> Result<(), String> {
        self.start_transcription_if_needed_with_progress(&mut |_| Ok(()))
    }

    fn ingest_event(&mut self, event: &AudioSessionEventOwned) -> bool {
        match event.kind {
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED => {
                self.diarization_running = false;
                self.diarization_done = true;
            }
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED => {
                self.transcription_running = false;
                self.transcription_done = true;
            }
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR
            | LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR => {
                self.fatal_event_error = Some(if event.detail.trim().is_empty() {
                    "audio session reported an error".to_string()
                } else {
                    event.detail.clone()
                });
            }
            _ => {}
        }
        let orchestrator_changed = self.orchestrator.ingest_event(event);
        orchestrator_changed
            || matches!(event.kind, LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON)
    }

    fn take_fatal_error(&self) -> Result<(), String> {
        if let Some(message) = self.fatal_event_error.as_ref() {
            return Err(message.clone());
        }
        Ok(())
    }

    pub fn pump_pending_events_with_progress<F>(&mut self, on_update: &mut F) -> Result<(), String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        loop {
            let pending = self.session.wait_events(0)?;
            if pending == 0 {
                break;
            }
            let mut changed = false;
            for event in self.session.drain_events(0)? {
                changed |= self.ingest_event(&event);
            }
            self.take_fatal_error()?;
            if changed {
                let result = self.current_result();
                on_update(&result)?;
            }
        }
        Ok(())
    }

    pub fn pump_pending_events(&mut self) -> Result<(), String> {
        self.pump_pending_events_with_progress(&mut |_| Ok(()))
    }

    pub fn push_audio_s16(&mut self, samples: &[i16]) -> Result<(), String> {
        self.push_audio_s16_with_sample_rate(samples, self.sample_rate_hz)
    }

    pub fn push_audio_s16_with_sample_rate(
        &mut self,
        samples: &[i16],
        sample_rate_hz: u32,
    ) -> Result<(), String> {
        self.push_audio_s16_with_sample_rate_and_progress(
            samples,
            sample_rate_hz,
            &mut |_| Ok(()),
        )
    }

    pub fn push_audio_s16_with_sample_rate_and_progress<F>(
        &mut self,
        samples: &[i16],
        sample_rate_hz: u32,
        on_update: &mut F,
    ) -> Result<(), String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        self.session.push_audio_s16(samples, sample_rate_hz)?;
        self.pump_pending_events_with_progress(on_update)
    }

    pub fn push_audio_f32(&mut self, samples: &[f32]) -> Result<(), String> {
        self.push_audio_f32_with_sample_rate(samples, self.sample_rate_hz)
    }

    pub fn push_audio_f32_with_sample_rate(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(), String> {
        self.push_audio_f32_with_sample_rate_and_progress(
            samples,
            sample_rate_hz,
            &mut |_| Ok(()),
        )
    }

    pub fn push_audio_f32_with_sample_rate_and_progress<F>(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
        on_update: &mut F,
    ) -> Result<(), String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        self.session.push_audio_f32(samples, sample_rate_hz)?;
        self.pump_pending_events_with_progress(on_update)
    }

    pub fn push_encoded(&mut self, audio_bytes: &[u8], audio_format: &str) -> Result<(), String> {
        self.push_encoded_with_progress(audio_bytes, audio_format, &mut |_| Ok(()))
    }

    pub fn push_encoded_with_progress<F>(
        &mut self,
        audio_bytes: &[u8],
        audio_format: &str,
        on_update: &mut F,
    ) -> Result<(), String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        self.session.push_encoded(audio_bytes, audio_format)?;
        self.pump_pending_events_with_progress(on_update)
    }

    pub fn flush_audio(&mut self) -> Result<(), String> {
        self.flush_audio_with_progress(&mut |_| Ok(()))
    }

    pub fn flush_audio_with_progress<F>(&mut self, on_update: &mut F) -> Result<(), String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        if !self.audio_flushed {
            self.session.flush_audio()?;
            self.audio_flushed = true;
        }
        self.pump_pending_events_with_progress(on_update)
    }

    pub fn finish(mut self) -> Result<AudioSessionRunResult, String> {
        self.finish_with_progress(&mut |_| Ok(()))
    }

    pub fn finish_with_progress<F>(mut self, on_update: &mut F) -> Result<AudioSessionRunResult, String>
    where
        F: FnMut(&AudioSessionRunResult) -> Result<(), String>,
    {
        self.flush_audio_with_progress(on_update)?;

        if self.diarization_running {
            self.session.stop_diarization()?;
            self.diarization_running = false;
            self.pump_pending_events_with_progress(on_update)?;
        }

        if self.want_transcription && self.staged {
            self.start_transcription_if_needed_with_progress(on_update)?;
        }

        while !(self.diarization_done && self.transcription_done) {
            let pending = self.session.wait_events(self.wait_ms)?;
            if pending == 0 {
                continue;
            }
            let mut changed = false;
            for event in self.session.drain_events(0)? {
                changed |= self.ingest_event(&event);
            }
            self.take_fatal_error()?;
            if changed {
                let result = self.current_result();
                on_update(&result)?;
            }
        }

        Ok(self.current_result())
    }
}

fn render_audio_session_output(
    snapshot: &OrchestratorSnapshot,
    want_diarization: bool,
    want_transcription: bool,
    sample_rate_hz: u32,
) -> String {
    if want_diarization && want_transcription {
        snapshot.markdown.clone()
    } else if want_diarization {
        render_speaker_spans(&snapshot.spans, sample_rate_hz)
    } else if !snapshot.pieces.is_empty() {
        snapshot
            .pieces
            .iter()
            .map(|piece| piece.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        snapshot.latest_transcription_json.clone().unwrap_or_default()
    }
}

fn infer_audio_session_input(args: &[String], session_sample_rate_hz: u32) -> Result<AudioSessionCliInput, String> {
    let audio_file = arg_value(args, "--audio-file");
    let stdin_pcm_s16le = has_arg(args, "--stdin-pcm-s16le");
    let stdin_pcm_f32le = has_arg(args, "--stdin-pcm-f32le");
    let input_count =
        usize::from(audio_file.is_some()) + usize::from(stdin_pcm_s16le) + usize::from(stdin_pcm_f32le);
    if input_count == 0 {
        return Err(
            "audio-session requires exactly one input source: --audio-file, --stdin-pcm-s16le, or --stdin-pcm-f32le"
                .to_string(),
        );
    }
    if input_count > 1 {
        return Err(
            "choose exactly one input source: --audio-file, --stdin-pcm-s16le, or --stdin-pcm-f32le"
                .to_string(),
        );
    }

    let default_chunk_frames = (session_sample_rate_hz / 10).max(1) as usize;
    let chunk_frames = parse_usize_arg(args, "--stdin-chunk-frames", default_chunk_frames)?;
    if chunk_frames == 0 {
        return Err("--stdin-chunk-frames must be > 0".to_string());
    }
    if let Some(path) = audio_file {
        let audio_format = infer_audio_format(&path, arg_value(args, "--audio-format"))?;
        return Ok(AudioSessionCliInput::AudioFile {
            path,
            audio_format,
            chunk_frames,
        });
    }
    if stdin_pcm_s16le {
        Ok(AudioSessionCliInput::StdinPcmS16Le { chunk_frames })
    } else {
        Ok(AudioSessionCliInput::StdinPcmF32Le { chunk_frames })
    }
}

fn ingest_audio_session_stdin_s16le(
    runner: &mut AudioSessionRunner,
    chunk_frames: usize,
    on_update: &mut impl FnMut(&AudioSessionRunResult) -> Result<(), String>,
) -> Result<&'static str, String> {
    let chunk_bytes = chunk_frames
        .checked_mul(std::mem::size_of::<i16>())
        .ok_or("--stdin-chunk-frames is too large".to_string())?;
    let mut stdin = io::stdin().lock();
    let mut read_buf = vec![0u8; chunk_bytes.max(2)];
    let mut carry = Vec::<u8>::new();
    let mut samples = Vec::<i16>::new();

    loop {
        let read = stdin
            .read(&mut read_buf)
            .map_err(|e| format!("failed to read stdin PCM stream: {e}"))?;
        if read == 0 {
            break;
        }
        carry.extend_from_slice(&read_buf[..read]);
        let aligned_len = carry.len() / 2 * 2;
        if aligned_len == 0 {
            continue;
        }

        samples.clear();
        samples.reserve(aligned_len / 2);
        for chunk in carry[..aligned_len].chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        runner.push_audio_s16_with_sample_rate_and_progress(&samples, runner.sample_rate_hz, on_update)?;
        carry.drain(..aligned_len);
    }

    if !carry.is_empty() {
        return Err("stdin PCM s16le stream ended on a partial sample".to_string());
    }

    Ok("stdin_pcm_s16le")
}

fn ingest_audio_session_stdin_f32le(
    runner: &mut AudioSessionRunner,
    chunk_frames: usize,
    on_update: &mut impl FnMut(&AudioSessionRunResult) -> Result<(), String>,
) -> Result<&'static str, String> {
    let chunk_bytes = chunk_frames
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("--stdin-chunk-frames is too large".to_string())?;
    let mut stdin = io::stdin().lock();
    let mut read_buf = vec![0u8; chunk_bytes.max(4)];
    let mut carry = Vec::<u8>::new();
    let mut samples = Vec::<f32>::new();

    loop {
        let read = stdin
            .read(&mut read_buf)
            .map_err(|e| format!("failed to read stdin PCM stream: {e}"))?;
        if read == 0 {
            break;
        }
        carry.extend_from_slice(&read_buf[..read]);
        let aligned_len = carry.len() / 4 * 4;
        if aligned_len == 0 {
            continue;
        }

        samples.clear();
        samples.reserve(aligned_len / 4);
        for chunk in carry[..aligned_len].chunks_exact(4) {
            samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        runner.push_audio_f32_with_sample_rate_and_progress(&samples, runner.sample_rate_hz, on_update)?;
        carry.drain(..aligned_len);
    }

    if !carry.is_empty() {
        return Err("stdin PCM f32le stream ended on a partial sample".to_string());
    }

    Ok("stdin_pcm_f32le")
}

fn ingest_audio_session_s16_chunks(
    runner: &mut AudioSessionRunner,
    samples: &[i16],
    sample_rate_hz: u32,
    chunk_frames: usize,
    on_update: &mut impl FnMut(&AudioSessionRunResult) -> Result<(), String>,
) -> Result<&'static str, String> {
    if chunk_frames == 0 {
        return Err("audio-session chunk_frames must be > 0".to_string());
    }

    let mut offset = 0usize;
    while offset < samples.len() {
        let end = (offset + chunk_frames).min(samples.len());
        runner.push_audio_s16_with_sample_rate_and_progress(
            &samples[offset..end],
            sample_rate_hz,
            on_update,
        )?;
        offset = end;
    }

    Ok("wav_pcm_chunked")
}

fn read_audio_session_metadata_json(args: &[String]) -> Result<Option<String>, String> {
    if let Some(raw_body) = arg_value(args, "--body-json") {
        let body_string = read_inline_or_file_json(&raw_body, "--body-json")?;
        let mut body_json: Value = serde_json::from_str(&body_string)
            .map_err(|e| format!("invalid --body-json payload: {e}"))?;
        let body_obj = body_json
            .as_object_mut()
            .ok_or("--body-json payload must be a JSON object".to_string())?;
        apply_audio_cli_overrides(args, body_obj)?;
        return Ok(Some(body_json.to_string()));
    }

    let whisper_model =
        arg_value(args, "--whisper-model").or_else(|| arg_value(args, "--whisper-model-path"));
    let whisper_hf_repo = arg_value(args, "--whisper-hf-repo");
    let whisper_hf_file = arg_value(args, "--whisper-hf-file");
    let metadata_json = arg_value(args, "--metadata-json")
        .map(|raw| read_inline_or_file_json(&raw, "--metadata-json"))
        .transpose()?;

    let has_local = whisper_model.is_some();
    let has_hf = whisper_hf_repo.is_some() || whisper_hf_file.is_some();
    if has_local && has_hf {
        return Err(
            "choose exactly one whisper source: local (--whisper-model) OR HF (--whisper-hf-repo + --whisper-hf-file)"
                .to_string(),
        );
    }
    if has_hf && (whisper_hf_repo.is_none() || whisper_hf_file.is_none()) {
        return Err(
            "HF whisper source requires both --whisper-hf-repo and --whisper-hf-file".to_string(),
        );
    }

    if whisper_model.is_none()
        && whisper_hf_repo.is_none()
        && whisper_hf_file.is_none()
        && metadata_json.is_none()
    {
        return Ok(None);
    }

    let mut body_json = match metadata_json {
        Some(raw) => serde_json::from_str::<Value>(&raw)
            .map_err(|e| format!("invalid --metadata-json payload: {e}"))?,
        None => Value::Object(Map::new()),
    };
    let body_obj = body_json
        .as_object_mut()
        .ok_or("--metadata-json payload must be a JSON object".to_string())?;
    if let Some(value) = whisper_model {
        body_obj.insert("whisper_model".to_string(), json!(value));
    }
    if let Some(value) = whisper_hf_repo {
        body_obj.insert("whisper_hf_repo".to_string(), json!(value));
    }
    if let Some(value) = whisper_hf_file {
        body_obj.insert("whisper_hf_file".to_string(), json!(value));
    }
    if let Some(value) = arg_value(args, "--custom") {
        body_obj.insert("custom".to_string(), json!(value));
    }
    apply_audio_cli_overrides(args, body_obj)?;
    Ok(Some(body_json.to_string()))
}

fn run_vlm(args: &[String]) -> Result<(), String> {
    let model = arg_value(args, "--model").ok_or("--model is required".to_string())?;
    let mmproj = arg_value(args, "--mmproj").ok_or("--mmproj is required".to_string())?;
    let image = arg_value(args, "--image").ok_or("--image is required".to_string())?;
    let out_path = arg_value(args, "--out");
    let gpu = parse_optional_i32_arg(args, "--gpu")?;
    let devices = arg_value(args, "--devices");
    if let Some(gpu_index) = gpu {
        if gpu_index < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if devices.is_some() {
            return Err("choose one: --gpu OR --devices".to_string());
        }
    }
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => {
            if gpu.is_some() {
                0
            } else {
                -1
            }
        }
    };

    let prompt = arg_value(args, "--prompt").unwrap_or_else(|| DEFAULT_VLM_PROMPT.to_string());
    let n_predict = parse_i32_arg(args, "--n-predict", 5_000)?;
    let n_ctx = parse_i32_arg(args, "--n-ctx", 32_768)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2_048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2_048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_threads = parse_optional_positive_i32_arg(args, "--threads")?;
    let n_threads_batch = parse_optional_positive_i32_arg(args, "--threads-batch")?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", -1)?;
    let mmproj_use_gpu = parse_i32_arg(args, "--mmproj-use-gpu", -1)?;
    if mmproj_use_gpu != -1 && mmproj_use_gpu != 0 && mmproj_use_gpu != 1 {
        return Err("--mmproj-use-gpu must be -1, 0 or 1".to_string());
    }

    let image_bytes =
        fs::read(&image).map_err(|e| format!("failed to read image file '{image}': {e}"))?;
    if image_bytes.is_empty() {
        return Err("image is empty".to_string());
    }

    let prompt_c = CString::new(prompt).map_err(|_| "prompt contains NUL byte".to_string())?;

    println!(
        "vlm_start model={} mmproj={} image={} bytes={} devices={} n_ctx={} n_predict={} n_parallel={}",
        model,
        mmproj,
        image,
        image_bytes.len(),
        devices.clone().unwrap_or_else(|| "<default>".to_string()),
        n_ctx,
        n_predict,
        n_parallel
    );

    let t_load = Instant::now();
    let bridge = make_bridge(
        &model,
        Some(&mmproj),
        Some(mmproj_use_gpu),
        gpu.unwrap_or(-1),
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        false,
        false,
    )?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let mut req = unsafe { llama_server_bridge_default_vlm_request() };
    req.prompt = prompt_c.as_ptr();
    req.image_bytes = image_bytes.as_ptr();
    req.image_bytes_len = image_bytes.len();
    req.n_predict = n_predict;
    req.id_slot = -1;
    req.temperature = 0.0;
    req.top_p = 1.0;
    req.top_k = -1;
    req.min_p = -1.0;
    req.seed = -1;

    let t_infer = Instant::now();
    let mut out = unsafe { llama_server_bridge_empty_vlm_result() };
    let rc = unsafe { llama_server_bridge_vlm_complete(bridge.ptr, &req, &mut out) };
    let infer_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

    let text = cstr_mut_ptr_to_string(out.text);
    let out_err = cstr_mut_ptr_to_string(out.error_json);
    if rc != 0 || out.ok == 0 {
        let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
        unsafe {
            llama_server_bridge_result_free(&mut out);
        }
        return Err(format!(
            "vlm call failed rc={} ok={} bridge_err='{}' out_err='{}'",
            rc, out.ok, bridge_err, out_err
        ));
    }

    if let Some(path) = out_path {
        fs::write(&path, &text)
            .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
    } else {
        println!("{text}");
    }

    let prompt_sec = out.prompt_ms / 1000.0;
    let output_sec = out.predicted_ms / 1000.0;
    let total_sec = (load_ms + infer_ms) / 1000.0;
    let total_tokens = out.n_prompt_tokens + out.n_decoded;
    let prompt_tps = if prompt_sec > 0.0 {
        out.n_prompt_tokens as f64 / prompt_sec
    } else {
        0.0
    };
    let output_tps = if output_sec > 0.0 {
        out.n_decoded as f64 / output_sec
    } else {
        0.0
    };
    let end_to_end_tps = if total_sec > 0.0 {
        total_tokens as f64 / total_sec
    } else {
        0.0
    };

    println!(
        "vlm_summary load_ms={:.2} infer_ms={:.2} total_ms={:.2} prompt_tokens={} output_tokens={} total_tokens={} prompt_tps={:.2} output_tps={:.2} end_to_end_tps={:.2} eos={} truncated={} stop={} output_chars={}",
        load_ms,
        infer_ms,
        load_ms + infer_ms,
        out.n_prompt_tokens,
        out.n_decoded,
        total_tokens,
        prompt_tps,
        output_tps,
        end_to_end_tps,
        out.eos_reached,
        out.truncated,
        out.stop,
        text.chars().count()
    );

    unsafe {
        llama_server_bridge_result_free(&mut out);
    }
    Ok(())
}

fn run_audio(args: &[String]) -> Result<(), String> {
    // Audio command always runs model-free in audio-only runtime mode.
    // A text LLM --model is not required for transcription/diarization.
    let model = String::new();
    let audio_only_mode = true;
    let out_path = arg_value(args, "--out");
    let output_dir_override = arg_value(args, "--output-dir");
    let gpu = parse_optional_i32_arg(args, "--gpu")?;
    let devices = arg_value(args, "--devices");
    if let Some(gpu_index) = gpu {
        if gpu_index < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if devices.is_some() {
            return Err("choose one: --gpu OR --devices".to_string());
        }
    }
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => {
            if gpu.is_some() {
                0
            } else {
                -1
            }
        }
    };

    let n_ctx = parse_i32_arg(args, "--n-ctx", 32_768)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2_048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2_048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_threads = parse_optional_positive_i32_arg(args, "--threads")?;
    let n_threads_batch = parse_optional_positive_i32_arg(args, "--threads-batch")?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", -1)?;
    let ffmpeg_convert_enabled = {
        let force_convert = has_arg(args, "--ffmpeg-convert");
        let no_convert = has_arg(args, "--no-ffmpeg-convert");
        if force_convert && no_convert {
            return Err("choose one: --ffmpeg-convert OR --no-ffmpeg-convert".to_string());
        }
        !no_convert
    };

    let mut use_raw_audio = false;
    let mut raw_audio_bytes: Vec<u8> = Vec::new();
    let mut raw_audio_format_c: Option<CString> = None;
    let mut raw_metadata_c: Option<CString> = None;
    let mut body_c: Option<CString> = None;

    if let Some(raw_body) = arg_value(args, "--body-json") {
        let path = std::path::Path::new(&raw_body);
        let body_string = if path.exists() {
            fs::read_to_string(path)
                .map_err(|e| format!("failed to read body json file '{}': {e}", raw_body))?
        } else {
            raw_body
        };
        let mut body_json: Value = serde_json::from_str(&body_string)
            .map_err(|e| format!("invalid --body-json payload: {e}"))?;
        let body_obj = body_json
            .as_object_mut()
            .ok_or("--body-json payload must be a JSON object".to_string())?;
        apply_audio_cli_overrides(args, body_obj)?;
        if let Some(output_dir) = output_dir_override.as_ref() {
            body_obj.insert("output_dir".to_string(), json!(output_dir));
        }
        body_c = Some(
            CString::new(body_json.to_string())
                .map_err(|_| "audio body json contains NUL byte".to_string())?,
        );
    } else {
        use_raw_audio = true;

        let audio_file = arg_value(args, "--audio-file")
            .ok_or("--audio-file is required (or use --body-json)".to_string())?;
        let mode = arg_value(args, "--mode").ok_or("--mode is required".to_string())?;
        let custom = arg_value(args, "--custom").unwrap_or_else(|| "default".to_string());

        raw_audio_bytes = fs::read(&audio_file)
            .map_err(|e| format!("failed to read audio file '{}': {e}", audio_file))?;
        if raw_audio_bytes.is_empty() {
            return Err("audio file is empty".to_string());
        }

        let audio_format = if let Some(fmt) = arg_value(args, "--audio-format") {
            fmt.to_ascii_lowercase()
        } else {
            std::path::Path::new(&audio_file)
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_else(|| "wav".to_string())
        };
        if audio_format.trim().is_empty() {
            return Err("audio format must be a non-empty string".to_string());
        }
        let output_dir = if let Some(v) = output_dir_override.as_ref() {
            v.clone()
        } else {
            std::path::Path::new(&audio_file)
                .parent()
                .map(|p| {
                    if p.as_os_str().is_empty() {
                        ".".to_string()
                    } else {
                        p.to_string_lossy().into_owned()
                    }
                })
                .unwrap_or_else(|| ".".to_string())
        };

        let whisper_local =
            arg_value(args, "--whisper-model").or_else(|| arg_value(args, "--whisper-model-path"));
        let whisper_hf_repo = arg_value(args, "--whisper-hf-repo");
        let whisper_hf_file = arg_value(args, "--whisper-hf-file");

        let has_local = whisper_local.is_some();
        let has_hf = whisper_hf_repo.is_some() || whisper_hf_file.is_some();
        if has_local && has_hf {
            return Err(
                "choose exactly one whisper source: local (--whisper-model) OR HF (--whisper-hf-repo + --whisper-hf-file)"
                    .to_string(),
            );
        }
        if !has_local && !has_hf {
            return Err(
                "missing whisper source: provide --whisper-model or (--whisper-hf-repo + --whisper-hf-file)"
                    .to_string(),
            );
        }
        if has_hf && (whisper_hf_repo.is_none() || whisper_hf_file.is_none()) {
            return Err(
                "HF whisper source requires both --whisper-hf-repo and --whisper-hf-file"
                    .to_string(),
            );
        }

        let diarization_model_path = arg_value(args, "--diarization-model-path");
        let diarization_models_dir = arg_value(args, "--diarization-models-dir")
            .or_else(|| arg_value(args, "--diarization-model-dir"));
        if diarization_model_path.is_some() && diarization_models_dir.is_some() {
            return Err(
                "choose exactly one diarization source: --diarization-model-path OR --diarization-models-dir"
                    .to_string(),
            );
        }

        let mut metadata = json!({
            "mode": mode,
            "custom": custom,
            "output_dir": output_dir,
            "audio_source_path": audio_file
        });

        if let Some(local) = whisper_local {
            metadata["whisper_model"] = json!(local);
        } else {
            metadata["whisper_hf_repo"] = json!(whisper_hf_repo.unwrap());
            metadata["whisper_hf_file"] = json!(whisper_hf_file.unwrap());
        }

        if let Some(path) = diarization_model_path {
            metadata["diarization_model_path"] = json!(path);
        }
        if let Some(dir) = diarization_models_dir {
            metadata["diarization_models_dir"] = json!(dir);
        }
        let metadata_obj = metadata
            .as_object_mut()
            .ok_or("audio metadata JSON must be an object".to_string())?;
        apply_audio_cli_overrides(args, metadata_obj)?;

        raw_audio_format_c = Some(
            CString::new(audio_format).map_err(|_| "audio format contains NUL byte".to_string())?,
        );
        raw_metadata_c = Some(
            CString::new(metadata.to_string())
                .map_err(|_| "audio metadata json contains NUL byte".to_string())?,
        );
    }

    println!(
        "audio_start model={} devices={} n_ctx={} n_parallel={}",
        "<audio-only>",
        devices.clone().unwrap_or_else(|| "<default>".to_string()),
        n_ctx,
        n_parallel
    );

    let t_load = Instant::now();
    // Audio-only runs do not require a text model path; enable bridge audio-only mode
    // for this process when --model is omitted.
    let prev_audio_only_env = if audio_only_mode {
        env::var("LLAMA_SERVER_AUDIO_ONLY").ok()
    } else {
        None
    };
    if audio_only_mode {
        env::set_var("LLAMA_SERVER_AUDIO_ONLY", "1");
    }

    let bridge_res = make_bridge(
        &model,
        None,
        None,
        gpu.unwrap_or(-1),
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        false,
        false,
    );

    if audio_only_mode {
        if let Some(v) = prev_audio_only_env {
            env::set_var("LLAMA_SERVER_AUDIO_ONLY", v);
        } else {
            env::remove_var("LLAMA_SERVER_AUDIO_ONLY");
        }
    }

    let bridge = bridge_res?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let mut out = unsafe { llama_server_bridge_empty_json_result() };
    let t_req = Instant::now();
    let rc = if use_raw_audio {
        let mut req = unsafe { llama_server_bridge_default_audio_raw_request() };
        req.audio_bytes = raw_audio_bytes.as_ptr();
        req.audio_bytes_len = raw_audio_bytes.len();
        req.audio_format = raw_audio_format_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
        req.metadata_json = raw_metadata_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
        req.ffmpeg_convert = if ffmpeg_convert_enabled { 1 } else { 0 };
        unsafe { llama_server_bridge_audio_transcriptions_raw(bridge.ptr, &req, &mut out) }
    } else {
        let mut req = unsafe { llama_server_bridge_default_audio_request() };
        req.body_json = body_c.as_ref().map(|v| v.as_ptr()).unwrap_or(ptr::null());
        unsafe { llama_server_bridge_audio_transcriptions(bridge.ptr, &req, &mut out) }
    };
    let req_ms = t_req.elapsed().as_secs_f64() * 1000.0;

    let response_text = cstr_mut_ptr_to_string(out.json);
    let out_err = cstr_mut_ptr_to_string(out.error_json);
    if rc != 0 || out.ok == 0 {
        let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
        unsafe {
            llama_server_bridge_json_result_free(&mut out);
        }
        return Err(format!(
            "audio transcriptions call failed rc={} status={} bridge_err='{}' out_err='{}'",
            rc, out.status, bridge_err, out_err
        ));
    }

    if let Some(path) = out_path {
        fs::write(&path, &response_text)
            .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
    } else {
        println!("{response_text}");
    }

    println!(
        "audio_summary load_ms={:.2} request_ms={:.2} status={}",
        load_ms, req_ms, out.status
    );

    unsafe {
        llama_server_bridge_json_result_free(&mut out);
    }
    Ok(())
}

fn run_audio_session(args: &[String]) -> Result<(), String> {
    let _realtime_model_cache_guard = RealtimeModelCacheGuard;
    let out_path = arg_value(args, "--out");
    let timeline_json_out = arg_value(args, "--timeline-json-out");
    let staged = has_arg(args, "--staged");
    let no_diarization = has_arg(args, "--no-diarization");
    let no_transcription = has_arg(args, "--no-transcription");

    let session_sample_rate_hz = parse_u32_arg(args, "--session-sample-rate", 16_000)?;
    let session_channels = parse_u32_arg(args, "--session-channels", 1)?;
    if session_channels != 1 {
        return Err("audio session CLI currently supports mono only".to_string());
    }
    if session_sample_rate_hz != 16_000 {
        return Err("audio session CLI currently expects --session-sample-rate 16000".to_string());
    }

    let input = infer_audio_session_input(args, session_sample_rate_hz)?;

    let max_buffered_audio_samples = parse_usize_arg(args, "--session-max-buffered-samples", 0)?;
    let event_queue_capacity = parse_usize_arg(args, "--session-event-queue-capacity", 0)?;
    if max_buffered_audio_samples > u32::MAX as usize {
        return Err("--session-max-buffered-samples exceeds u32 range".to_string());
    }
    if event_queue_capacity > u32::MAX as usize {
        return Err("--session-event-queue-capacity exceeds u32 range".to_string());
    }

    let diarization_model = arg_value(args, "--diarization-model")
        .or_else(|| arg_value(args, "--diarization-model-path"));
    let diarization_ring_capacity = parse_u32_arg(
        args,
        "--diarization-ring-capacity-samples",
        session_sample_rate_hz.saturating_mul(60),
    )?;
    let transcription_realtime_model = arg_value(args, "--transcription-realtime-model")
        .or_else(|| arg_value(args, "--transcription-model-path"));
    let transcription_ring_capacity = parse_u32_arg(
        args,
        "--transcription-ring-capacity-samples",
        session_sample_rate_hz.saturating_mul(60),
    )?;

    let metadata_json = if no_transcription {
        None
    } else {
        read_audio_session_metadata_json(args)?
    };
    if metadata_json.is_some() && transcription_realtime_model.is_some() {
        return Err(
            "choose one transcription path: offline metadata (--whisper-model/--metadata-json/etc) OR --transcription-realtime-model"
                .to_string(),
        );
    }

    let want_diarization = !no_diarization && diarization_model.is_some();
    let want_transcription =
        !no_transcription && (metadata_json.is_some() || transcription_realtime_model.is_some());
    if !want_diarization && !want_transcription {
        return Err(
            "nothing to run: provide --diarization-model-path and/or transcription input (--whisper-model/--metadata-json for offline route, or --transcription-realtime-model for native realtime)"
                .to_string(),
        );
    }

    let gpu = parse_optional_i32_arg(args, "--gpu")?;
    let devices = arg_value(args, "--devices");
    if let Some(gpu_index) = gpu {
        if gpu_index < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if devices.is_some() {
            return Err("choose one: --gpu OR --devices".to_string());
        }
    }
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => {
            if gpu.is_some() {
                0
            } else {
                -1
            }
        }
    };
    let n_ctx = parse_i32_arg(args, "--n-ctx", 32_768)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2_048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2_048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_threads = parse_optional_positive_i32_arg(args, "--threads")?;
    let n_threads_batch = parse_optional_positive_i32_arg(args, "--threads-batch")?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", -1)?;
    let wait_ms = parse_u32_arg(args, "--wait-ms", 50)?;
    let resolved_gpu_device_name = match gpu {
        Some(gpu_index) => Some(resolve_device_name_by_index(gpu_index)?),
        None => None,
    };
    let diarization_backend_name = arg_value(args, "--diarization-device")
        .or_else(|| arg_value(args, "--diarization-backend-name"))
        .or_else(|| arg_value(args, "--backend-name"))
        .or_else(|| resolved_gpu_device_name.clone());
    let transcription_backend_name = arg_value(args, "--transcription-device")
        .or_else(|| arg_value(args, "--transcription-backend-name"))
        .or_else(|| resolved_gpu_device_name.clone());

    let nearest_tolerance_ms =
        parse_optional_f64_arg(args, "--nearest-tolerance-ms")?.unwrap_or(100.0);
    let alignment_offset_ms = parse_optional_f64_arg(args, "--alignment-offset-ms")?.unwrap_or(0.0);

    let mut session_params = unsafe { llama_server_bridge_default_audio_session_params() };
    session_params.expected_input_sample_rate_hz = session_sample_rate_hz;
    session_params.expected_input_channels = session_channels;
    session_params.max_buffered_audio_samples = max_buffered_audio_samples as u32;
    session_params.event_queue_capacity = event_queue_capacity as u32;
    let options = AssembleOptions {
        nearest_tolerance_samples: ((nearest_tolerance_ms / 1000.0)
            * f64::from(session_sample_rate_hz))
        .round()
        .max(0.0) as u64,
        alignment_offset_samples: ((alignment_offset_ms / 1000.0)
            * f64::from(session_sample_rate_hz))
        .round() as i64,
    };

    let diarization_params = if want_diarization {
        let mut diar_params = OwnedRealtimeParams::new();
        diar_params.set_expected_sample_rate_hz(session_sample_rate_hz);
        diar_params.set_audio_ring_capacity_samples(diarization_ring_capacity);
        diar_params.set_model_path(diarization_model.as_deref())?;
        diar_params.set_backend_name(diarization_backend_name.as_deref())?;
        Some(diar_params)
    } else {
        None
    };

    let transcription_params = if want_transcription {
        let mut tx = OwnedAudioTranscriptionParams::new();
        if let Some(model_path) = transcription_realtime_model.as_deref() {
            tx.set_mode(LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE);
            let rt = tx.realtime_params_mut();
            rt.set_expected_sample_rate_hz(session_sample_rate_hz);
            rt.set_audio_ring_capacity_samples(transcription_ring_capacity);
            rt.set_model_path(Some(model_path))?;
            rt.set_backend_name(transcription_backend_name.as_deref())?;
        } else {
            tx.set_mode(LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE);
            {
                let bridge_params = tx.bridge_params_mut();
                bridge_params.set_devices(devices.as_deref())?;
                bridge_params.set_tensor_split(tensor_split.as_deref())?;
                let raw = bridge_params.raw_mut();
                raw.n_ctx = n_ctx;
                raw.n_batch = n_batch;
                raw.n_ubatch = n_ubatch;
                raw.n_parallel = n_parallel;
                raw.n_gpu_layers = n_gpu_layers;
                raw.main_gpu = main_gpu;
                raw.gpu = gpu.unwrap_or(-1);
                raw.split_mode = split_mode;
                if let Some(value) = n_threads {
                    raw.n_threads = value;
                }
                if let Some(value) = n_threads_batch {
                    raw.n_threads_batch = value;
                }
            }
            tx.set_metadata_json(metadata_json.as_deref())?;
        }
        Some(tx)
    } else {
        None
    };

    let mut runner = AudioSessionRunner::new(
        session_params,
        session_sample_rate_hz,
        wait_ms,
        staged,
        options,
        diarization_params,
        transcription_params,
    )?;
    let rolling_stdout =
        out_path.is_none() && !matches!(&input, AudioSessionCliInput::AudioFile { .. });
    let mut reporter =
        RollingAudioSessionReporter::new(out_path.clone(), timeline_json_out.clone(), rolling_stdout);

    let audio_format_summary = input.summary_audio_format().to_string();
    let ingest_mode = match &input {
        AudioSessionCliInput::AudioFile {
            path,
            audio_format,
            chunk_frames,
        } => {
            let audio_bytes = fs::read(&path)
                .map_err(|e| format!("failed to read audio file '{path}': {e}"))?;
            if audio_bytes.is_empty() {
                return Err("audio file is empty".to_string());
            }
            if audio_format == "wav" {
                match try_parse_wav_mono_s16le(&audio_bytes)? {
                    Some((samples, sample_rate_hz)) => {
                        ingest_audio_session_s16_chunks(
                            &mut runner,
                            &samples,
                            sample_rate_hz,
                            *chunk_frames,
                            &mut |result| reporter.emit(result),
                        )?
                        .to_string()
                    }
                    None => {
                        runner.push_encoded_with_progress(
                            &audio_bytes,
                            &audio_format,
                            &mut |result| reporter.emit(result),
                        )?;
                        "encoded".to_string()
                    }
                }
            } else {
                runner.push_encoded_with_progress(
                    &audio_bytes,
                    &audio_format,
                    &mut |result| reporter.emit(result),
                )?;
                "encoded".to_string()
            }
        }
        AudioSessionCliInput::StdinPcmS16Le { chunk_frames } => {
            ingest_audio_session_stdin_s16le(
                &mut runner,
                *chunk_frames,
                &mut |result| reporter.emit(result),
            )?
            .to_string()
        }
        AudioSessionCliInput::StdinPcmF32Le { chunk_frames } => {
            ingest_audio_session_stdin_f32le(
                &mut runner,
                *chunk_frames,
                &mut |result| reporter.emit(result),
            )?
            .to_string()
        }
    };

    let result = runner.finish_with_progress(&mut |result| reporter.emit(result))?;
    let snapshot = result.snapshot;
    let output_text = result.output_text;
    reporter.emit(&AudioSessionRunResult {
        snapshot: snapshot.clone(),
        output_text: output_text.clone(),
    })?;

    if out_path.is_none() && !rolling_stdout {
        println!("{output_text}");
    }

    println!(
        "audio_session_summary diarization={} transcription={} spans={} pieces={} words={} turns={} staged={} audio_format={} sample_rate_hz={} ingest_mode={}",
        want_diarization,
        want_transcription,
        snapshot.spans.len(),
        snapshot.pieces.len(),
        snapshot.words.len(),
        snapshot.turns.len(),
        staged,
        audio_format_summary,
        session_sample_rate_hz,
        ingest_mode
    );
    Ok(())
}

fn run_chat(args: &[String]) -> Result<(), String> {
    const CHAT_STOP_KEYS: &[&str] = &[
        "--model",
        "--prompt",
        "--markdown",
        "--out",
        "--gpu",
        "--devices",
        "--tensor-split",
        "--split-mode",
        "--n-predict",
        "--n-ctx",
        "--n-batch",
        "--n-ubatch",
        "--n-parallel",
        "--threads",
        "--threads-batch",
        "--n-gpu-layers",
        "--main-gpu",
    ];

    let model = arg_value(args, "--model").ok_or("--model is required".to_string())?;
    let markdown = arg_value(args, "--markdown");
    let user_prompt = arg_value_spanned(args, "--prompt", CHAT_STOP_KEYS).and_then(|s| {
        if s.trim().is_empty() {
            None
        } else {
            Some(s)
        }
    });
    if markdown.is_none() && user_prompt.is_none() {
        return Err("chat requires --prompt and/or --markdown".to_string());
    }
    let out_path = arg_value(args, "--out");
    let gpu = parse_optional_i32_arg(args, "--gpu")?;
    let devices = arg_value(args, "--devices");
    if let Some(gpu_index) = gpu {
        if gpu_index < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if devices.is_some() {
            return Err("choose one: --gpu OR --devices".to_string());
        }
    }
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => {
            if gpu.is_some() {
                0
            } else {
                -1
            }
        }
    };

    let n_predict = parse_i32_arg(args, "--n-predict", 10_000)?;
    let n_ctx = parse_i32_arg(args, "--n-ctx", 50_000)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 1024)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 1024)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_threads = parse_optional_positive_i32_arg(args, "--threads")?;
    let n_threads_batch = parse_optional_positive_i32_arg(args, "--threads-batch")?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", -1)?;

    let markdown_text = if let Some(path) = &markdown {
        Some(
            fs::read_to_string(path)
                .map_err(|e| format!("failed to read markdown file '{path}': {e}"))?,
        )
    } else {
        None
    };

    let prompt = match (user_prompt, markdown_text) {
        (Some(p), Some(md)) => format!("{p}\n\nContext markdown:\n\n{md}"),
        (Some(p), None) => p,
        (None, Some(md)) => format!(
            "Summarize the following markdown in about 1000 words. Keep methods, core findings, key statistics, and table takeaways accurate.\n\n{}",
            md
        ),
        (None, None) => return Err("chat requires --prompt and/or --markdown".to_string()),
    };
    let prompt_c = CString::new(prompt).map_err(|_| "prompt contains NUL byte".to_string())?;

    println!(
        "chat_start model={} markdown={} prompt_chars={} devices={} n_ctx={} n_predict={} n_parallel={}",
        model,
        markdown.unwrap_or_else(|| "<none>".to_string()),
        prompt_c.as_bytes().len(),
        devices.clone().unwrap_or_else(|| "<default>".to_string()),
        n_ctx,
        n_predict,
        n_parallel
    );

    let t_load = Instant::now();
    let bridge = make_bridge(
        &model,
        None,
        None,
        gpu.unwrap_or(-1),
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        false,
        false,
    )?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let mut req = unsafe { llama_server_bridge_default_chat_request() };
    req.prompt = prompt_c.as_ptr();
    req.n_predict = n_predict;
    req.id_slot = -1;
    req.temperature = 0.0;
    req.top_p = 1.0;
    req.top_k = -1;
    req.min_p = 0.0;
    req.seed = -1;

    let t_infer = Instant::now();
    let mut out = unsafe { llama_server_bridge_empty_vlm_result() };
    let rc = unsafe { llama_server_bridge_chat_complete(bridge.ptr, &req, &mut out) };
    let infer_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

    let text = cstr_mut_ptr_to_string(out.text);
    let out_err = cstr_mut_ptr_to_string(out.error_json);
    if rc != 0 || out.ok == 0 {
        let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
        unsafe {
            llama_server_bridge_result_free(&mut out);
        }
        return Err(format!(
            "chat call failed rc={} ok={} bridge_err='{}' out_err='{}'",
            rc, out.ok, bridge_err, out_err
        ));
    }

    if let Some(path) = out_path {
        fs::write(&path, &text)
            .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
    }

    let prompt_sec = out.prompt_ms / 1000.0;
    let output_sec = out.predicted_ms / 1000.0;
    let total_sec = (load_ms + infer_ms) / 1000.0;
    let total_tokens = out.n_prompt_tokens + out.n_decoded;
    let prompt_tps = if prompt_sec > 0.0 {
        out.n_prompt_tokens as f64 / prompt_sec
    } else {
        0.0
    };
    let output_tps = if output_sec > 0.0 {
        out.n_decoded as f64 / output_sec
    } else {
        0.0
    };
    let end_to_end_tps = if total_sec > 0.0 {
        total_tokens as f64 / total_sec
    } else {
        0.0
    };

    println!(
        "chat_summary load_ms={:.2} infer_ms={:.2} total_ms={:.2} prompt_tokens={} output_tokens={} total_tokens={} prompt_tps={:.2} output_tps={:.2} end_to_end_tps={:.2} eos={} truncated={} stop={} output_chars={}",
        load_ms,
        infer_ms,
        load_ms + infer_ms,
        out.n_prompt_tokens,
        out.n_decoded,
        total_tokens,
        prompt_tps,
        output_tps,
        end_to_end_tps,
        out.eos_reached,
        out.truncated,
        out.stop,
        text.chars().count()
    );

    unsafe {
        llama_server_bridge_result_free(&mut out);
    }
    Ok(())
}

fn run_embed(args: &[String]) -> Result<(), String> {
    let model = arg_value(args, "--model").ok_or("--model is required".to_string())?;
    let markdown = arg_value(args, "--markdown");
    let body_json = arg_value(args, "--body-json");
    if markdown.is_none() && body_json.is_none() {
        return Err("embed requires --markdown or --body-json".to_string());
    }
    if markdown.is_some() && body_json.is_some() {
        return Err("embed accepts only one input source: --markdown OR --body-json".to_string());
    }
    let out_path = arg_value(args, "--out");
    let gpu = parse_optional_i32_arg(args, "--gpu")?;
    let devices = arg_value(args, "--devices");
    if let Some(gpu_index) = gpu {
        if gpu_index < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if devices.is_some() {
            return Err("choose one: --gpu OR --devices".to_string());
        }
    }
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => {
            if gpu.is_some() {
                0
            } else {
                -1
            }
        }
    };

    let batch_size = parse_usize_arg(args, "--batch-size", 32)?;
    let chunk_words = parse_usize_arg(args, "--chunk-words", 500)?;
    let batches = parse_usize_arg(args, "--batches", 8)?;

    let n_ctx = parse_i32_arg(args, "--n-ctx", 8192)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_threads = parse_optional_positive_i32_arg(args, "--threads")?;
    let n_threads_batch = parse_optional_positive_i32_arg(args, "--threads-batch")?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", -1)?;

    let bridge = make_bridge(
        &model,
        None,
        None,
        gpu.unwrap_or(-1),
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        true,
        false,
    )?;

    if let Some(raw_body) = body_json {
        let body_string = read_inline_or_file_json(&raw_body, "embedding body json")?;
        let body_c = CString::new(body_string)
            .map_err(|_| "embedding body json contains NUL byte".to_string())?;

        println!(
            "embed_start model={} source=body-json devices={}",
            model,
            devices.clone().unwrap_or_else(|| "<default>".to_string())
        );

        let mut req = unsafe { llama_server_bridge_default_embeddings_request() };
        req.body_json = body_c.as_ptr();
        req.oai_compat = 1;

        let mut out = unsafe { llama_server_bridge_empty_json_result() };
        let t0 = Instant::now();
        let rc = unsafe { llama_server_bridge_embeddings(bridge.ptr, &req, &mut out) };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let response_text = cstr_mut_ptr_to_string(out.json);
        let out_err = cstr_mut_ptr_to_string(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err =
                cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
            unsafe {
                llama_server_bridge_json_result_free(&mut out);
            }
            return Err(format!(
                "embeddings call failed rc={} status={} bridge_err='{}' out_err='{}'",
                rc, out.status, bridge_err, out_err
            ));
        }

        let items = match serde_json::from_str::<Value>(&response_text) {
            Ok(v) => v
                .get("data")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .or_else(|| v.as_array().map(|x| x.len()))
                .unwrap_or(0),
            Err(_) => 0,
        };

        if let Some(path) = &out_path {
            fs::write(path, &response_text)
                .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
        } else {
            println!("{response_text}");
        }

        let sec = ms / 1000.0;
        let items_s = if sec > 0.0 { items as f64 / sec } else { 0.0 };
        println!(
            "embed_summary batches=1 total_items={} total_ms={:.2} items_per_s={:.2}",
            items, ms, items_s
        );

        unsafe {
            llama_server_bridge_json_result_free(&mut out);
        }
        return Ok(());
    }

    let markdown = markdown.expect("checked above");
    let texts = build_text_pool(&markdown, chunk_words)?;
    println!(
        "embed_start model={} source=markdown markdown={} chunks={} batch_size={} batches={} devices={}",
        model,
        markdown,
        texts.len(),
        batch_size,
        batches,
        devices.unwrap_or_else(|| "<default>".to_string())
    );

    let mut total_items = 0usize;
    let mut total_ms = 0.0f64;
    let mut responses: Vec<String> = Vec::new();

    for b in 0..batches {
        let mut batch = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let idx = (b * batch_size + i) % texts.len();
            batch.push(texts[idx].clone());
        }
        let body = json!({
            "input": batch,
            "encoding_format": "float"
        });
        let body_c = CString::new(body.to_string())
            .map_err(|_| "embedding body has NUL byte".to_string())?;

        let mut req = unsafe { llama_server_bridge_default_embeddings_request() };
        req.body_json = body_c.as_ptr();
        req.oai_compat = 1;

        let mut out = unsafe { llama_server_bridge_empty_json_result() };
        let t0 = Instant::now();
        let rc = unsafe { llama_server_bridge_embeddings(bridge.ptr, &req, &mut out) };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let response_text = cstr_mut_ptr_to_string(out.json);
        let out_err = cstr_mut_ptr_to_string(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err =
                cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
            unsafe {
                llama_server_bridge_json_result_free(&mut out);
            }
            return Err(format!(
                "embeddings call failed batch={} rc={} status={} bridge_err='{}' out_err='{}'",
                b + 1,
                rc,
                out.status,
                bridge_err,
                out_err
            ));
        }

        let items = match serde_json::from_str::<Value>(&response_text) {
            Ok(v) => v
                .get("data")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .or_else(|| v.as_array().map(|x| x.len()))
                .unwrap_or(batch_size),
            Err(_) => batch_size,
        };

        total_items += items;
        total_ms += ms;
        responses.push(response_text);

        println!(
            "embed_batch index={} items={} ms={:.2} status={}",
            b + 1,
            items,
            ms,
            out.status
        );

        unsafe {
            llama_server_bridge_json_result_free(&mut out);
        }
    }

    if let Some(path) = &out_path {
        fs::write(path, format!("[{}]", responses.join(",")))
            .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
    }

    let sec = total_ms / 1000.0;
    let items_s = if sec > 0.0 {
        total_items as f64 / sec
    } else {
        0.0
    };
    println!(
        "embed_summary batches={} total_items={} total_ms={:.2} items_per_s={:.2}",
        batches, total_items, total_ms, items_s
    );
    Ok(())
}

fn run_rerank(args: &[String]) -> Result<(), String> {
    let model = arg_value(args, "--model").ok_or("--model is required".to_string())?;
    let markdown = arg_value(args, "--markdown");
    let body_json = arg_value(args, "--body-json");
    if markdown.is_none() && body_json.is_none() {
        return Err("rerank requires --markdown or --body-json".to_string());
    }
    if markdown.is_some() && body_json.is_some() {
        return Err("rerank accepts only one input source: --markdown OR --body-json".to_string());
    }
    let out_path = arg_value(args, "--out");
    let gpu = parse_optional_i32_arg(args, "--gpu")?;
    let devices = arg_value(args, "--devices");
    if let Some(gpu_index) = gpu {
        if gpu_index < 0 {
            return Err("--gpu must be >= 0".to_string());
        }
        if devices.is_some() {
            return Err("choose one: --gpu OR --devices".to_string());
        }
    }
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => {
            if gpu.is_some() {
                0
            } else {
                -1
            }
        }
    };

    let docs_per_query = parse_usize_arg(args, "--docs-per-query", 32)?;
    let chunk_words = parse_usize_arg(args, "--chunk-words", 500)?;
    let batches = parse_usize_arg(args, "--batches", 8)?;

    let n_ctx = parse_i32_arg(args, "--n-ctx", 8192)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_threads = parse_optional_positive_i32_arg(args, "--threads")?;
    let n_threads_batch = parse_optional_positive_i32_arg(args, "--threads-batch")?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", -1)?;

    let bridge = make_bridge(
        &model,
        None,
        None,
        gpu.unwrap_or(-1),
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_threads,
        n_threads_batch,
        n_gpu_layers,
        main_gpu,
        true,
        true,
    )?;

    if let Some(raw_body) = body_json {
        let body_string = read_inline_or_file_json(&raw_body, "rerank body json")?;
        let body_c = CString::new(body_string)
            .map_err(|_| "rerank body json contains NUL byte".to_string())?;

        println!(
            "rerank_start model={} source=body-json devices={}",
            model,
            devices.clone().unwrap_or_else(|| "<default>".to_string())
        );

        let mut req = unsafe { llama_server_bridge_default_rerank_request() };
        req.body_json = body_c.as_ptr();

        let mut out = unsafe { llama_server_bridge_empty_json_result() };
        let t0 = Instant::now();
        let rc = unsafe { llama_server_bridge_rerank(bridge.ptr, &req, &mut out) };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let response_text = cstr_mut_ptr_to_string(out.json);
        let out_err = cstr_mut_ptr_to_string(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err =
                cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
            unsafe {
                llama_server_bridge_json_result_free(&mut out);
            }
            return Err(format!(
                "rerank call failed rc={} status={} bridge_err='{}' out_err='{}'",
                rc, out.status, bridge_err, out_err
            ));
        }

        let items = match serde_json::from_str::<Value>(&response_text) {
            Ok(v) => v
                .get("results")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .or_else(|| v.as_array().map(|x| x.len()))
                .unwrap_or(0),
            Err(_) => 0,
        };

        if let Some(path) = &out_path {
            fs::write(path, &response_text)
                .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
        } else {
            println!("{response_text}");
        }

        let sec = ms / 1000.0;
        let docs_s = if sec > 0.0 { items as f64 / sec } else { 0.0 };
        println!(
            "rerank_summary batches=1 total_docs={} total_ms={:.2} docs_per_s={:.2}",
            items, ms, docs_s
        );

        unsafe {
            llama_server_bridge_json_result_free(&mut out);
        }
        return Ok(());
    }

    let markdown = markdown.expect("checked above");
    let texts = build_text_pool(&markdown, chunk_words)?;
    if texts.len() < 2 {
        return Err("not enough chunks for rerank".to_string());
    }

    println!(
        "rerank_start model={} source=markdown markdown={} chunks={} docs_per_query={} batches={} devices={}",
        model,
        markdown,
        texts.len(),
        docs_per_query,
        batches,
        devices.unwrap_or_else(|| "<default>".to_string())
    );

    let mut total_docs = 0usize;
    let mut total_ms = 0.0f64;
    let mut responses: Vec<String> = Vec::new();

    for b in 0..batches {
        let qidx = (b * (docs_per_query + 1)) % texts.len();
        let query = texts[qidx].clone();
        let mut docs = Vec::with_capacity(docs_per_query);
        for i in 0..docs_per_query {
            let didx = (qidx + 1 + i) % texts.len();
            docs.push(texts[didx].clone());
        }

        let body = json!({
            "query": query,
            "documents": docs,
            "top_n": docs_per_query
        });
        let body_c =
            CString::new(body.to_string()).map_err(|_| "rerank body has NUL byte".to_string())?;

        let mut req = unsafe { llama_server_bridge_default_rerank_request() };
        req.body_json = body_c.as_ptr();

        let mut out = unsafe { llama_server_bridge_empty_json_result() };
        let t0 = Instant::now();
        let rc = unsafe { llama_server_bridge_rerank(bridge.ptr, &req, &mut out) };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let response_text = cstr_mut_ptr_to_string(out.json);
        let out_err = cstr_mut_ptr_to_string(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err =
                cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
            unsafe {
                llama_server_bridge_json_result_free(&mut out);
            }
            return Err(format!(
                "rerank call failed batch={} rc={} status={} bridge_err='{}' out_err='{}'",
                b + 1,
                rc,
                out.status,
                bridge_err,
                out_err
            ));
        }

        let items = match serde_json::from_str::<Value>(&response_text) {
            Ok(v) => v
                .get("results")
                .and_then(|x| x.as_array())
                .map(|x| x.len())
                .or_else(|| v.as_array().map(|x| x.len()))
                .unwrap_or(docs_per_query),
            Err(_) => docs_per_query,
        };
        total_docs += items;
        total_ms += ms;
        responses.push(response_text);

        println!(
            "rerank_batch index={} docs={} ms={:.2} status={}",
            b + 1,
            items,
            ms,
            out.status
        );

        unsafe {
            llama_server_bridge_json_result_free(&mut out);
        }
    }

    if let Some(path) = &out_path {
        fs::write(path, format!("[{}]", responses.join(",")))
            .map_err(|e| format!("failed to write output file '{path}': {e}"))?;
    }

    let sec = total_ms / 1000.0;
    let docs_s = if sec > 0.0 {
        total_docs as f64 / sec
    } else {
        0.0
    };
    println!(
        "rerank_summary batches={} total_docs={} total_ms={:.2} docs_per_s={:.2}",
        batches, total_docs, total_ms, docs_s
    );
    Ok(())
}

pub fn run_bridge_cli_subcommand(sub: &str, sub_args: &[String]) -> Result<(), String> {
    prepare_windows_bridge_runtime_paths();

    match sub {
        "list-devices" => run_list_devices(),
        "vlm" => run_vlm(sub_args),
        "audio" | "transcribe" | "transcriptions" => run_audio(sub_args),
        "audio-session" | "audio-orchestrate" => run_audio_session(sub_args),
        "convert" => {
            Err("use 'engine pdf ...' for PDF conversion; bridge vlm is image-only".to_string())
        }
        "chat" => run_chat(sub_args),
        "embed" => run_embed(sub_args),
        "rerank" => run_rerank(sub_args),
        _ => {
            println!("{}", bridge_cli_usage());
            Err(format!("unknown subcommand: {sub}"))
        }
    }
}

pub fn run_bridge_cli_from_env() -> Result<(), String> {
    prepare_windows_bridge_runtime_paths();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("{}", bridge_cli_usage());
        return Err("missing subcommand".to_string());
    }
    run_bridge_cli_subcommand(args[1].as_str(), &args[2..])
}
