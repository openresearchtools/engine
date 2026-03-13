use std::os::raw::c_char;

#[repr(C)]
pub struct llama_server_bridge {
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

    pub reasoning: *const c_char,
    pub reasoning_budget: i32,
    pub reasoning_format: *const c_char,
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

#[link(name = "llama-server-bridge")]
unsafe extern "C" {
    pub fn llama_server_bridge_default_params() -> llama_server_bridge_params;
    pub fn llama_server_bridge_default_vlm_request() -> llama_server_bridge_vlm_request;
    pub fn llama_server_bridge_empty_vlm_result() -> llama_server_bridge_vlm_result;

    pub fn llama_server_bridge_create(
        params: *const llama_server_bridge_params,
    ) -> *mut llama_server_bridge;
    pub fn llama_server_bridge_destroy(bridge: *mut llama_server_bridge);

    pub fn llama_server_bridge_vlm_complete(
        bridge: *mut llama_server_bridge,
        req: *const llama_server_bridge_vlm_request,
        out: *mut llama_server_bridge_vlm_result,
    ) -> i32;

    pub fn llama_server_bridge_result_free(out: *mut llama_server_bridge_vlm_result);
    pub fn llama_server_bridge_last_error(bridge: *const llama_server_bridge) -> *const c_char;
}
