use serde_json::{json, Value};
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::c_char;
use std::ptr;
use std::time::Instant;

const DEFAULT_VLM_PROMPT: &str = "Convert this page to markdown. Do not miss any text and only output the bare markdown! Any graphs or figures found convert to markdown table. If figure is image without details, describe what you see in the image. For tables, pay attention to whitespace: some cells may be intentionally empty, so keep empty and filled cells in the correct columns. Ensure correct assignment of column headings and subheadings for tables.";

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
    pub fn llama_server_bridge_empty_json_result() -> llama_server_bridge_json_result;

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

fn cstr_ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
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

fn has_arg(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

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

fn split_mode_arg_to_i32(value: &str) -> Result<i32, String> {
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(0),
        "layer" => Ok(1),
        "row" => Ok(2),
        _ => Err("invalid --split-mode, expected none/layer/row".to_string()),
    }
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
        fs::read_to_string(path).map_err(|e| format!("failed to read {label} file '{raw_or_path}': {e}"))
    } else {
        Ok(raw_or_path.to_string())
    }
}

fn make_bridge(
    model_path: &str,
    mmproj_path: Option<&str>,
    devices: Option<&str>,
    tensor_split: Option<&str>,
    split_mode: i32,
    n_ctx: i32,
    n_batch: i32,
    n_ubatch: i32,
    n_parallel: i32,
    n_gpu_layers: i32,
    main_gpu: i32,
    embedding: bool,
    reranking: bool,
) -> Result<BridgeHandle, String> {
    let model_c = CString::new(model_path).map_err(|_| "model path contains NUL byte".to_string())?;
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
    params.mmproj_path = mmproj_c
        .as_ref()
        .map(|s| s.as_ptr())
        .unwrap_or(ptr::null());
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.n_parallel = n_parallel;
    params.n_gpu_layers = n_gpu_layers;
    params.main_gpu = main_gpu;
    params.no_kv_offload = 0;
    params.mmproj_use_gpu = if mmproj_c.is_some() { 1 } else { 0 };
    params.cache_ram_mib = 0;
    params.ctx_shift = 1;
    params.kv_unified = 1;
    params.devices = devices_c.as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null());
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
  vlm --model <gguf> --mmproj <gguf> --image <png/jpg/webp> [--prompt <text>] [--out <md>] [--n-predict 5000]
  audio [--model <gguf>] --audio-file <wav/mp3> [--audio-format wav|mp3] --mode <subtitle|speech|transcript> --custom <value> [--no-ffmpeg-convert] [whisper source + diarization source flags]
  chat --model <gguf> [--prompt <text>] [--markdown <md>] [--out <md>] [--devices <csv>] [--n-predict 10000]
  embed --model <gguf> (--markdown <md> | --body-json <json-or-path>) [--out <json>] [--devices <csv>] [--batch-size 32] [--chunk-words 500] [--batches 8]
  rerank --model <gguf> (--markdown <md> | --body-json <json-or-path>) [--out <json>] [--devices <csv>] [--docs-per-query 32] [--chunk-words 500] [--batches 8]
shared optional flags:
  --n-ctx <int> --n-batch <int> --n-ubatch <int> --n-parallel <int> --n-gpu-layers <int> --main-gpu <int>
  --split-mode <none|layer|row> --tensor-split <csv>"
}

fn run_list_devices() -> Result<(), String> {
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

fn run_vlm(args: &[String]) -> Result<(), String> {
    let model = arg_value(args, "--model").ok_or("--model is required".to_string())?;
    let mmproj = arg_value(args, "--mmproj").ok_or("--mmproj is required".to_string())?;
    let image = arg_value(args, "--image").ok_or("--image is required".to_string())?;
    let out_path = arg_value(args, "--out");
    let devices = arg_value(args, "--devices");
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => -1,
    };

    let prompt = arg_value(args, "--prompt").unwrap_or_else(|| DEFAULT_VLM_PROMPT.to_string());
    let n_predict = parse_i32_arg(args, "--n-predict", 5_000)?;
    let n_ctx = parse_i32_arg(args, "--n-ctx", 32_768)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2_048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2_048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", 0)?;

    let image_bytes = fs::read(&image)
        .map_err(|e| format!("failed to read image file '{image}': {e}"))?;
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
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
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
        fs::write(&path, &text).map_err(|e| format!("failed to write output file '{path}': {e}"))?;
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
    let model = arg_value(args, "--model").unwrap_or_default();
    let audio_only_mode = model.trim().is_empty();
    let out_path = arg_value(args, "--out");
    let devices = arg_value(args, "--devices");
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => -1,
    };

    let n_ctx = parse_i32_arg(args, "--n-ctx", 32_768)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2_048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2_048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", 0)?;

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
        body_c = Some(
            CString::new(body_string).map_err(|_| "audio body json contains NUL byte".to_string())?,
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
        if audio_format != "wav" && audio_format != "mp3" {
            return Err("audio format must be wav or mp3".to_string());
        }

        let whisper_local = arg_value(args, "--whisper-model")
            .or_else(|| arg_value(args, "--whisper-model-path"));
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

        let diarization_models_dir = arg_value(args, "--diarization-models-dir");
        let diarization_hf_repo = arg_value(args, "--diarization-hf-repo");
        let diarization_device =
            arg_value(args, "--diarization-device").unwrap_or_else(|| "auto".to_string());

        let mut metadata = json!({
            "mode": mode,
            "custom": custom
        });

        if let Some(local) = whisper_local {
            metadata["whisper_model"] = json!(local);
        } else {
            metadata["whisper_hf_repo"] = json!(whisper_hf_repo.unwrap());
            metadata["whisper_hf_file"] = json!(whisper_hf_file.unwrap());
        }

        if let Some(dir) = diarization_models_dir {
            metadata["diarization_models_dir"] = json!(dir);
        }
        if let Some(repo) = diarization_hf_repo {
            metadata["diarization_hf_repo"] = json!(repo);
        }
        if has_arg(args, "--diarization-device") {
            metadata["diarization_device"] = json!(diarization_device);
        }

        raw_audio_format_c =
            Some(CString::new(audio_format).map_err(|_| "audio format contains NUL byte".to_string())?);
        raw_metadata_c = Some(
            CString::new(metadata.to_string()).map_err(|_| "audio metadata json contains NUL byte".to_string())?,
        );
    }

    println!(
        "audio_start model={} devices={} n_ctx={} n_parallel={}",
        if model.is_empty() { "<empty>" } else { &model },
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
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
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
        req.ffmpeg_convert = if has_arg(args, "--no-ffmpeg-convert") { 0 } else { 1 };
        unsafe { llama_server_bridge_audio_transcriptions_raw(bridge.ptr, &req, &mut out) }
    } else {
        let mut req = unsafe { llama_server_bridge_default_audio_request() };
        req.body_json = body_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
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

fn run_chat(args: &[String]) -> Result<(), String> {
    let model = arg_value(args, "--model").ok_or("--model is required".to_string())?;
    let markdown = arg_value(args, "--markdown");
    let user_prompt = arg_value(args, "--prompt");
    if markdown.is_none() && user_prompt.is_none() {
        return Err("chat requires --prompt and/or --markdown".to_string());
    }
    let out_path = arg_value(args, "--out");
    let devices = arg_value(args, "--devices");
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => -1,
    };

    let n_predict = parse_i32_arg(args, "--n-predict", 10_000)?;
    let n_ctx = parse_i32_arg(args, "--n-ctx", 50_000)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 1024)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 1024)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", 0)?;

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
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
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
        fs::write(&path, &text).map_err(|e| format!("failed to write output file '{path}': {e}"))?;
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
    let devices = arg_value(args, "--devices");
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => -1,
    };

    let batch_size = parse_usize_arg(args, "--batch-size", 32)?;
    let chunk_words = parse_usize_arg(args, "--chunk-words", 500)?;
    let batches = parse_usize_arg(args, "--batches", 8)?;

    let n_ctx = parse_i32_arg(args, "--n-ctx", 8192)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", 0)?;

    let bridge = make_bridge(
        &model,
        None,
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_gpu_layers,
        main_gpu,
        true,
        false,
    )?;

    if let Some(raw_body) = body_json {
        let body_string = read_inline_or_file_json(&raw_body, "embedding body json")?;
        let body_c =
            CString::new(body_string).map_err(|_| "embedding body json contains NUL byte".to_string())?;

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
            let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
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
        let body_c = CString::new(body.to_string()).map_err(|_| "embedding body has NUL byte".to_string())?;

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
            let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
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
    let items_s = if sec > 0.0 { total_items as f64 / sec } else { 0.0 };
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
    let devices = arg_value(args, "--devices");
    let tensor_split = arg_value(args, "--tensor-split");
    let split_mode = match arg_value(args, "--split-mode") {
        Some(v) => split_mode_arg_to_i32(&v)?,
        None => -1,
    };

    let docs_per_query = parse_usize_arg(args, "--docs-per-query", 32)?;
    let chunk_words = parse_usize_arg(args, "--chunk-words", 500)?;
    let batches = parse_usize_arg(args, "--batches", 8)?;

    let n_ctx = parse_i32_arg(args, "--n-ctx", 8192)?;
    let n_batch = parse_i32_arg(args, "--n-batch", 2048)?;
    let n_ubatch = parse_i32_arg(args, "--n-ubatch", 2048)?;
    let n_parallel = parse_i32_arg(args, "--n-parallel", 1)?;
    let n_gpu_layers = parse_i32_arg(args, "--n-gpu-layers", -1)?;
    let main_gpu = parse_i32_arg(args, "--main-gpu", 0)?;

    let bridge = make_bridge(
        &model,
        None,
        devices.as_deref(),
        tensor_split.as_deref(),
        split_mode,
        n_ctx,
        n_batch,
        n_ubatch,
        n_parallel,
        n_gpu_layers,
        main_gpu,
        true,
        true,
    )?;

    if let Some(raw_body) = body_json {
        let body_string = read_inline_or_file_json(&raw_body, "rerank body json")?;
        let body_c = CString::new(body_string).map_err(|_| "rerank body json contains NUL byte".to_string())?;

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
            let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
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
        let body_c = CString::new(body.to_string()).map_err(|_| "rerank body has NUL byte".to_string())?;

        let mut req = unsafe { llama_server_bridge_default_rerank_request() };
        req.body_json = body_c.as_ptr();

        let mut out = unsafe { llama_server_bridge_empty_json_result() };
        let t0 = Instant::now();
        let rc = unsafe { llama_server_bridge_rerank(bridge.ptr, &req, &mut out) };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let response_text = cstr_mut_ptr_to_string(out.json);
        let out_err = cstr_mut_ptr_to_string(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err = cstr_ptr_to_string(unsafe { llama_server_bridge_last_error(bridge.ptr) });
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
    let docs_s = if sec > 0.0 { total_docs as f64 / sec } else { 0.0 };
    println!(
        "rerank_summary batches={} total_docs={} total_ms={:.2} docs_per_s={:.2}",
        batches, total_docs, total_ms, docs_s
    );
    Ok(())
}

pub fn run_bridge_cli_subcommand(sub: &str, sub_args: &[String]) -> Result<(), String> {
    match sub {
        "list-devices" => run_list_devices(),
        "vlm" => run_vlm(sub_args),
        "audio" | "transcribe" | "transcriptions" => run_audio(sub_args),
        "convert" => Err("use 'engine pdf ...' for PDF conversion; bridge vlm is image-only".to_string()),
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
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("{}", bridge_cli_usage());
        return Err("missing subcommand".to_string());
    }
    run_bridge_cli_subcommand(args[1].as_str(), &args[2..])
}
