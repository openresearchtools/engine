#[path = "../cluster_api.rs"]
mod cluster_api;

use anyhow::{bail, Context, Result};
use cluster_api::{
    AudioRawRequest, ChatRequest, ClusterApi, CreateInstanceParams, InstanceModelKind,
    RetentionMode, VlmRequest,
};
use std::env;
use std::fs;
use std::path::PathBuf;

fn env_required(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("missing env var {name}"))
}

fn env_optional(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn main() -> Result<()> {
    let runtime_dir = PathBuf::from(env_required("CLUSTER_RUNTIME_DIR")?);
    let action = env::var("CLUSTER_ACTION").unwrap_or_else(|_| "run".to_string());
    let rpc_servers = env::var("CLUSTER_RPC_SERVERS")
        .ok()
        .filter(|v| !v.trim().is_empty());

    let api = ClusterApi::load(&runtime_dir).with_context(|| {
        format!(
            "failed to load cluster runtime from '{}'",
            runtime_dir.display()
        )
    })?;

    if action == "snapshot" {
        let groups = api
            .list_execution_groups_with_rpc(rpc_servers.as_deref())
            .context("list_execution_groups_with_rpc failed")?;
        let devices = api
            .list_devices_with_rpc(rpc_servers.as_deref())
            .context("list_devices_with_rpc failed")?;
        println!("devices={}", devices.len());
        for device in devices {
            println!(
                "device index={} backend={} name={} free={} total={}",
                device.bridge_device_index,
                device.backend,
                device.name,
                device.memory_free,
                device.memory_total
            );
        }
        println!("execution_groups={}", groups.len());
        for group in groups {
            println!(
                "group id={} split={} devices={} free={} total={} backend={}",
                group.id,
                group.uses_local_split,
                group.devices_csv,
                group.memory_free,
                group.memory_total,
                group.backend_summary
            );
        }
        return Ok(());
    }

    let model_path = env_required("CLUSTER_MODEL_PATH")?;
    let execution_group_id =
        env::var("CLUSTER_GROUP_ID").unwrap_or_else(|_| "cluster:auto".to_string());
    let prompt = if let Some(prompt_file) = env_optional("CLUSTER_PROMPT_FILE") {
        fs::read_to_string(&prompt_file)
            .with_context(|| format!("failed to read prompt file '{}'", prompt_file))?
    } else {
        env::var("CLUSTER_PROMPT").unwrap_or_else(|_| "Reply with ok.".to_string())
    };
    let retention_mode = match env::var("CLUSTER_RETENTION_MODE").ok().as_deref() {
        Some("load_on_demand") => RetentionMode::LoadOnDemand,
        _ => RetentionMode::KeepLoaded,
    };

    let params = CreateInstanceParams {
        name: env::var("CLUSTER_INSTANCE_NAME").unwrap_or_else(|_| "direct-harness".to_string()),
        managed_model_id: None,
        model_path,
        mmproj_path: env::var("CLUSTER_MMPROJ_PATH")
            .ok()
            .filter(|v| !v.trim().is_empty()),
        diarization_model_path: env::var("CLUSTER_DIARIZATION_MODEL_PATH")
            .ok()
            .filter(|v| !v.trim().is_empty()),
        execution_group_id,
        rpc_servers,
        manual_device_allocations: Vec::new(),
        manual_devices_csv: None,
        manual_tensor_split: None,
        preferred_owner_control_addr: None,
        retention_mode,
        load_on_demand_grace_seconds: env_i32(
            "CLUSTER_LOAD_ON_DEMAND_GRACE_SECONDS",
            InstanceModelKind::Text.default_load_on_demand_grace_seconds(),
        ),
        embedding: env_flag("CLUSTER_EMBEDDING"),
        reranking: env_flag("CLUSTER_RERANKING"),
        model_kind: InstanceModelKind::Text,
        single_device_only: env_flag("CLUSTER_SINGLE_DEVICE_ONLY"),
        allow_cpu: env_flag("CLUSTER_ALLOW_CPU"),
        allow_integrated_gpu: env_flag("CLUSTER_ALLOW_INTEGRATED_GPU"),
        n_ctx: env_i32("CLUSTER_N_CTX", 2048),
        n_batch: env_i32("CLUSTER_N_BATCH", 512),
        n_ubatch: env_i32("CLUSTER_N_UBATCH", 512),
        n_parallel: env_i32("CLUSTER_N_PARALLEL", 1),
        n_threads: env_i32("CLUSTER_N_THREADS", 8),
        n_threads_batch: env_i32("CLUSTER_N_THREADS_BATCH", 8),
        n_gpu_layers: env_i32("CLUSTER_N_GPU_LAYERS", -1),
    };

    let instance_id = api
        .create_instance(&params)
        .context("create_instance failed")?;
    println!("created instance_id={instance_id}");

    let load_result = api.load_instance(instance_id);
    if let Err(err) = load_result {
        let _ = api.remove_instance(instance_id);
        bail!("load_instance failed: {err}");
    }
    println!("loaded instance_id={instance_id}");

    let result = if action == "vlm" {
        let image_path = env_required("CLUSTER_IMAGE_PATH")?;
        let image_bytes = fs::read(&image_path)
            .with_context(|| format!("failed to read image '{}'", image_path))?;
        let request = VlmRequest {
            instance_id,
            prompt,
            image_bytes,
            n_predict: env_i32("CLUSTER_N_PREDICT", 32),
            temperature: env_f32("CLUSTER_TEMPERATURE", 0.2),
            top_p: env_f32("CLUSTER_TOP_P", 0.95),
            top_k: env_i32("CLUSTER_TOP_K", 40),
            min_p: env_f32("CLUSTER_MIN_P", 0.05),
            repeat_last_n: env_i32("CLUSTER_REPEAT_LAST_N", 64),
            repeat_penalty: env_f32("CLUSTER_REPEAT_PENALTY", 1.05),
            reasoning: None,
            reasoning_budget: -2147483648,
            reasoning_format: None,
        };
        api.vlm_complete(&request).context("vlm_complete failed")?
    } else if action == "audio_raw" {
        let audio_path = env_required("CLUSTER_AUDIO_PATH")?;
        let audio_bytes = fs::read(&audio_path)
            .with_context(|| format!("failed to read audio '{}'", audio_path))?;
        let request = AudioRawRequest {
            instance_id,
            audio_bytes,
            audio_format: env::var("CLUSTER_AUDIO_FORMAT").unwrap_or_else(|_| "wav".to_string()),
            metadata_json: env::var("CLUSTER_METADATA_JSON")
                .ok()
                .filter(|v| !v.trim().is_empty()),
            ffmpeg_convert: env_flag("CLUSTER_FFMPEG_CONVERT"),
            enable_diarization: env_flag("CLUSTER_ENABLE_DIARIZATION"),
            diarization_model_path: env::var("CLUSTER_DIARIZATION_MODEL_PATH")
                .ok()
                .filter(|v| !v.trim().is_empty()),
        };
        let json = api
            .audio_transcriptions_raw(&request)
            .context("audio_transcriptions_raw failed")?;
        println!("json_result_begin");
        println!("{}", json.json);
        println!("json_result_end");
        println!(
            "metrics total_ms={:.2} queue_ms={:.2} load_ms={:.2} prompt_tps={:.2} decode_tps={:.2} total_tps={:.2} prompt_tokens={} decoded_tokens={} used_rpc={} rpc_count={}",
            json.metrics.request_total_ms,
            json.metrics.queue_wait_ms,
            json.metrics.load_ms,
            json.metrics.prompt_tokens_per_second,
            json.metrics.decode_tokens_per_second,
            json.metrics.total_tokens_per_second,
            json.metrics.prompt_tokens,
            json.metrics.decoded_tokens,
            json.metrics.used_rpc,
            json.metrics.rpc_server_count
        );
        let _ = api.unload_instance(instance_id);
        let _ = api.remove_instance(instance_id);
        return Ok(());
    } else {
        let request = ChatRequest {
            instance_id,
            prompt,
            n_predict: env_i32("CLUSTER_N_PREDICT", 32),
            temperature: env_f32("CLUSTER_TEMPERATURE", 0.2),
            top_p: env_f32("CLUSTER_TOP_P", 0.95),
            top_k: env_i32("CLUSTER_TOP_K", 40),
            min_p: env_f32("CLUSTER_MIN_P", 0.05),
            repeat_last_n: env_i32("CLUSTER_REPEAT_LAST_N", 64),
            repeat_penalty: env_f32("CLUSTER_REPEAT_PENALTY", 1.05),
            reasoning: env_optional("CLUSTER_REASONING"),
            reasoning_budget: env_i32("CLUSTER_REASONING_BUDGET", -2147483648),
            reasoning_format: env_optional("CLUSTER_REASONING_FORMAT"),
        };
        match api.chat_complete(&request) {
            Ok(result) => result,
            Err(err) => {
                let _ = api.unload_instance(instance_id);
                let _ = api.remove_instance(instance_id);
                return Err(err).context("chat_complete failed");
            }
        }
    };
    println!("chat_result_begin");
    println!("{}", result.text);
    println!("chat_result_end");
    println!(
        "metrics total_ms={:.2} queue_ms={:.2} load_ms={:.2} prompt_tps={:.2} decode_tps={:.2} total_tps={:.2} prompt_tokens={} decoded_tokens={} used_rpc={} rpc_count={}",
        result.metrics.request_total_ms,
        result.metrics.queue_wait_ms,
        result.metrics.load_ms,
        result.metrics.prompt_tokens_per_second,
        result.metrics.decode_tokens_per_second,
        result.metrics.total_tokens_per_second,
        result.metrics.prompt_tokens,
        result.metrics.decoded_tokens,
        result.metrics.used_rpc,
        result.metrics.rpc_server_count
    );

    let _ = api.unload_instance(instance_id);
    let _ = api.remove_instance(instance_id);
    Ok(())
}
