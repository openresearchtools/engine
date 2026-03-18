#[path = "../agent.rs"]
mod agent;
#[path = "../catalog.rs"]
mod catalog;
#[path = "../cluster_api.rs"]
mod cluster_api;
#[path = "../model_metadata.rs"]
mod model_metadata;
#[path = "../model_store.rs"]
mod model_store;
#[path = "../protocol.rs"]
mod protocol;
#[path = "../public_server.rs"]
mod public_server;
#[path = "../settings.rs"]
mod settings;

use agent::AgentClient;
use anyhow::{bail, Context, Result};
use cluster_api::{
    ChatRequest, CreateInstanceParams, InstanceModelKind, NativeAudioTranscriptionRequest,
    RetentionMode,
};
use std::env;
use std::fs;
use std::thread;
use std::time::Duration;

fn env_required(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("missing env var {name}"))
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(default)
}

fn env_optional(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    env_bool(name, false)
}

fn main() -> Result<()> {
    let agent_addr = env_required("CLUSTER_AGENT_ADDR")?;
    let action = env::var("CLUSTER_ACTION").unwrap_or_else(|_| "run".to_string());
    let model_path = env::var("CLUSTER_MODEL_PATH").unwrap_or_default();
    let execution_group_id =
        env::var("CLUSTER_GROUP_ID").unwrap_or_else(|_| "cluster:auto".to_string());
    let rpc_servers = env::var("CLUSTER_RPC_SERVERS")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let prompt =
        env::var("CLUSTER_PROMPT").unwrap_or_else(|_| "Reply with the single word ok.".to_string());
    let name = env::var("CLUSTER_INSTANCE_NAME").unwrap_or_else(|_| "agent-harness".to_string());
    let retention_mode = match env::var("CLUSTER_RETENTION_MODE").ok().as_deref() {
        Some("load_on_demand") => RetentionMode::LoadOnDemand,
        _ => RetentionMode::KeepLoaded,
    };
    let post_chat_sleep_seconds = env_u64("CLUSTER_POST_CHAT_SLEEP_SECONDS", 0);
    let skip_cleanup = env_bool("CLUSTER_SKIP_CLEANUP", false);

    let client = AgentClient::new(agent_addr.clone());
    client
        .ping()
        .with_context(|| format!("agent ping failed at {agent_addr}"))?;

    if action == "snapshot" {
        let snapshot = client
            .get_snapshot_with_rpc(rpc_servers.clone())
            .context("get_snapshot_with_rpc failed")?;
        println!(
            "snapshot node={} addr={} advertised={} rpc_running={} rpc_endpoint={} advertised_rpc={}",
            snapshot.node.display_name,
            snapshot.control_addr,
            snapshot.advertised_control_addr.unwrap_or_default(),
            snapshot.rpc_running,
            snapshot.rpc_endpoint.unwrap_or_default(),
            snapshot.advertised_rpc_endpoint.unwrap_or_default(),
        );
        println!("devices={}", snapshot.devices.len());
        for device in snapshot.devices {
            println!(
                "device index={} backend={} name={} free={} total={}",
                device.bridge_device_index,
                device.backend,
                device.name,
                device.memory_free,
                device.memory_total
            );
        }
        println!("execution_groups={}", snapshot.execution_groups.len());
        for group in snapshot.execution_groups {
            println!(
                "group id={} split={} devices={} free={} total={}",
                group.id,
                group.uses_local_split,
                group.devices_csv,
                group.memory_free,
                group.memory_total
            );
        }
        println!("instances={}", snapshot.instances.len());
        for instance in snapshot.instances {
            println!(
                "instance id={} name={} state={} retention={:?} group={}",
                instance.instance_id,
                instance.name,
                instance.state,
                instance.retention_mode,
                instance.execution_group_id
            );
        }
        return Ok(());
    }

    if action == "restart-rpc" {
        client
            .restart_rpc_server()
            .context("restart_rpc_server failed")?;
        println!("restart-rpc ok");
        return Ok(());
    }

    if action == "audio-native" {
        let model_path = env_required("CLUSTER_MODEL_PATH")?;
        let audio_file = env_required("CLUSTER_AUDIO_FILE")?;
        let audio_bytes = fs::read(&audio_file)
            .with_context(|| format!("failed to read audio file '{audio_file}'"))?;
        let audio_format = env_optional("CLUSTER_AUDIO_FORMAT").unwrap_or_else(|| {
            std::path::Path::new(&audio_file)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("wav")
                .to_ascii_lowercase()
        });
        let result = client
            .audio_transcriptions_native(NativeAudioTranscriptionRequest {
                model_path,
                execution_group_id: env_optional("CLUSTER_GROUP_ID"),
                audio_bytes,
                audio_format,
                metadata_json: env_optional("CLUSTER_METADATA_JSON"),
                ffmpeg_convert: env_bool("CLUSTER_FFMPEG_CONVERT", false),
                enable_diarization: env_bool("CLUSTER_ENABLE_DIARIZATION", false),
                diarization_model_path: env_optional("CLUSTER_DIARIZATION_MODEL_PATH"),
            })
            .context("audio_transcriptions_native failed")?;
        println!("audio_result_begin");
        println!("{}", result.json);
        println!("audio_result_end");
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
        return Ok(());
    }

    if model_path.is_empty() {
        bail!("missing env var CLUSTER_MODEL_PATH");
    }

    let params = CreateInstanceParams {
        name,
        managed_model_id: None,
        model_path,
        mmproj_path: None,
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

    let allowed_control_addrs = env::var("CLUSTER_ALLOWED_CONTROL_ADDRS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| part.trim().to_string())
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty());

    if action == "plan" {
        let plan = client
            .plan_instance(params, allowed_control_addrs)
            .context("plan_instance failed")?;
        println!(
            "plan owner={} display={} label={} group={} rpc={} strategy={:?} required={} free={} ready_now={} reuse={:?} evict={}",
            plan.owner_control_addr,
            plan.owner_display_name,
            plan.display_label,
            plan.execution_group_id,
            plan.rpc_servers,
            plan.strategy,
            plan.estimated_required_bytes,
            plan.estimated_group_free_bytes,
            plan.ready_now,
            plan.reusable_instance_id,
            plan.requires_eviction
        );
        return Ok(());
    }

    if action == "list-candidates" {
        let plans = client
            .list_placement_candidates(params, allowed_control_addrs)
            .context("list_placement_candidates failed")?;
        for (index, plan) in plans.iter().enumerate() {
            println!(
                "candidate[{index}] owner={} display={} label={} group={} rpc={} strategy={:?} required={} free={} ready_now={} reuse={:?} evict={}",
                plan.owner_control_addr,
                plan.owner_display_name,
                plan.display_label,
                plan.execution_group_id,
                plan.rpc_servers,
                plan.strategy,
                plan.estimated_required_bytes,
                plan.estimated_group_free_bytes,
                plan.ready_now,
                plan.reusable_instance_id,
                plan.requires_eviction
            );
        }
        return Ok(());
    }

    if action == "schedule" {
        let scheduled = client
            .schedule_instance(params, allowed_control_addrs, true)
            .context("schedule_instance failed")?;
        println!(
            "scheduled owner={} display={} instance_id={} group={} rpc={} strategy={:?} reused_existing={} waited_ms={}",
            scheduled.owner_control_addr,
            scheduled.owner_display_name,
            scheduled.instance_id,
            scheduled.execution_group_id,
            scheduled.rpc_servers,
            scheduled.strategy,
            scheduled.reused_existing,
            scheduled.waited_ms
        );
        if !skip_cleanup {
            let owner = AgentClient::new(scheduled.owner_control_addr.clone());
            if let Err(err) = owner.unload_instance(scheduled.instance_id) {
                eprintln!("unload_instance warning: {err}");
            }
            if let Err(err) = owner.remove_instance(scheduled.instance_id) {
                eprintln!("remove_instance warning: {err}");
            }
        }
        return Ok(());
    }

    let instance_id = client
        .create_instance(params)
        .context("create_instance failed")?;
    println!("created instance_id={instance_id}");

    let load_result = client.load_instance(instance_id);
    if let Err(err) = load_result {
        let _ = client.remove_instance(instance_id);
        bail!("load_instance failed: {err}");
    }
    println!("loaded instance_id={instance_id}");

    let chat = ChatRequest {
        instance_id,
        prompt,
        n_predict: env_i32("CLUSTER_N_PREDICT", 64),
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

    let chat_result = client.chat_complete(chat);

    match chat_result {
        Ok(result) => {
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

            if post_chat_sleep_seconds > 0 {
                thread::sleep(Duration::from_secs(post_chat_sleep_seconds));
                let snapshot = client
                    .get_snapshot_with_rpc(
                        env::var("CLUSTER_RPC_SERVERS")
                            .ok()
                            .filter(|v| !v.trim().is_empty()),
                    )
                    .context("post-chat snapshot failed")?;
                if let Some(instance) = snapshot
                    .instances
                    .into_iter()
                    .find(|instance| instance.instance_id == instance_id)
                {
                    println!(
                        "post_chat_instance id={} state={} retention={:?} grace_deadline_unix_ms={}",
                        instance.instance_id,
                        instance.state,
                        instance.retention_mode,
                        instance.grace_deadline_unix_ms
                    );
                } else {
                    println!("post_chat_instance missing");
                }
            }

            if !skip_cleanup {
                if let Err(err) = client.unload_instance(instance_id) {
                    eprintln!("unload_instance warning: {err}");
                }
                if let Err(err) = client.remove_instance(instance_id) {
                    eprintln!("remove_instance warning: {err}");
                }
            }
            Ok(())
        }
        Err(err) => {
            if !skip_cleanup {
                if let Err(cleanup_err) = client.unload_instance(instance_id) {
                    eprintln!("unload_instance warning after chat error: {cleanup_err}");
                }
                if let Err(cleanup_err) = client.remove_instance(instance_id) {
                    eprintln!("remove_instance warning after chat error: {cleanup_err}");
                }
            }
            bail!("chat_complete failed: {err}")
        }
    }
}
