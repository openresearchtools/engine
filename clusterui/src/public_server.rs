use crate::agent::AgentClient;
use crate::cluster_api::{
    AudioRawRequest, ChatRequest, EmbeddingsRequest, InferenceMetrics, InstanceInfo,
    InstanceModelKind, RerankRequest, RetentionMode, VlmRequest,
};
use crate::protocol::{PublicApiConfig, TelemetrySnapshot};
use anyhow::{anyhow, Result};
use axum::extract::{ConnectInfo, Multipart, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};
use tower_http::cors::{Any, CorsLayer};

const CLUSTER_INSTANCES_CACHE_TTL: Duration = Duration::from_millis(750);
const CLUSTER_INSTANCES_STALE_FALLBACK_TTL: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct ManagedApiState {
    local_control_addr: String,
    models_dir: PathBuf,
    api_key: Option<String>,
    allowed_client_ips: Vec<String>,
    request_gate: Arc<Semaphore>,
    instances_cache: Arc<Mutex<Option<CachedClusterInstances>>>,
}

#[derive(Clone)]
struct CachedClusterInstances {
    instances: Vec<ResolvedApiInstance>,
    fetched_at: Instant,
}

pub struct PublicServerHandle {
    pub bound_addr: String,
    shutdown_tx: Option<oneshot::Sender<()>>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl PublicServerHandle {
    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequestBody {
    model: String,
    input: Value,
    max_output_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    x_engine_reasoning: Option<String>,
    x_engine_reasoning_budget: Option<i32>,
    x_engine_reasoning_format: Option<String>,
    x_engine_n_parallel: Option<i32>,
    x_engine_retention: Option<String>,
    x_engine_allowed_nodes: Option<Value>,
    x_engine_preferred_owner: Option<String>,
    x_engine_execution_group: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsRequestBody {
    model: String,
    messages: Vec<Value>,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    stream: Option<bool>,
    x_engine_reasoning: Option<String>,
    x_engine_reasoning_budget: Option<i32>,
    x_engine_reasoning_format: Option<String>,
    x_engine_n_parallel: Option<i32>,
    x_engine_retention: Option<String>,
    x_engine_allowed_nodes: Option<Value>,
    x_engine_preferred_owner: Option<String>,
    x_engine_execution_group: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsRequestBody {
    model: String,
    input: Value,
    encoding_format: Option<String>,
    x_engine_n_parallel: Option<i32>,
    x_engine_retention: Option<String>,
    x_engine_allowed_nodes: Option<Value>,
    x_engine_preferred_owner: Option<String>,
    x_engine_execution_group: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RerankRequestBody {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
    x_engine_n_parallel: Option<i32>,
    x_engine_retention: Option<String>,
    x_engine_allowed_nodes: Option<Value>,
    x_engine_preferred_owner: Option<String>,
    x_engine_execution_group: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct ScheduleOverrides {
    allowed_control_addrs: Option<Vec<String>>,
    preferred_owner_control_addr: Option<String>,
    execution_group_id: Option<String>,
    n_parallel: Option<i32>,
    retention_mode: Option<RetentionMode>,
}

#[derive(Debug, Clone)]
struct ResolvedApiInstance {
    owner_control_addr: String,
    owner_display_name: String,
    instance: InstanceInfo,
}

fn supports_public_transcription(instance: &InstanceInfo) -> bool {
    matches!(instance.model_kind, InstanceModelKind::Whisper)
}

fn managed_reasoning_mode(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| Some("off".to_string()))
}

fn managed_reasoning_budget(value: Option<i32>) -> i32 {
    value.unwrap_or(i32::MIN)
}

fn managed_reasoning_format(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("{value:#}"),
        }
    }
}

pub fn start_public_server(
    config: PublicApiConfig,
    local_control_addr: String,
    models_dir: PathBuf,
) -> Result<PublicServerHandle> {
    let (tx, rx) = mpsc::sync_channel(1);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = Arc::new(ManagedApiState {
        local_control_addr,
        models_dir,
        api_key: config.api_key.clone(),
        allowed_client_ips: config.allowed_client_ips.clone(),
        request_gate: Arc::new(Semaphore::new(1)),
        instances_cache: Arc::new(Mutex::new(None)),
    });

    let thread_handle = thread::spawn(move || {
        let worker_threads = thread::available_parallelism()
            .map(|value| value.get().max(4))
            .unwrap_or(4);
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .max_blocking_threads(worker_threads.max(8))
            .thread_name("engine-public-api")
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(err) => {
                let _ = tx.send(Err(anyhow!("failed to build public HTTP runtime: {err}")));
                return;
            }
        };

        runtime.block_on(async move {
            let mut app = Router::new()
                .route("/v1/models", get(list_models))
                .route("/v1/responses", post(run_responses))
                .route("/v1/chat/completions", post(run_chat_completions))
                .route("/v1/embeddings", post(run_embeddings))
                .route("/v1/rerank", post(run_rerank))
                .route("/v1/audio/transcriptions", post(run_transcriptions))
                .with_state(state);
            if config.allow_cors {
                let cors = if config.allowed_origins.is_empty() {
                    CorsLayer::new()
                        .allow_methods(Any)
                        .allow_headers(Any)
                        .allow_origin(Any)
                } else {
                    let origins = config
                        .allowed_origins
                        .iter()
                        .map(|value| {
                            HeaderValue::from_str(value)
                                .map_err(|err| anyhow!("invalid CORS origin '{}': {err}", value))
                        })
                        .collect::<Result<Vec<_>, _>>();
                    match origins {
                        Ok(origins) => CorsLayer::new()
                            .allow_methods(Any)
                            .allow_headers(Any)
                            .allow_origin(origins),
                        Err(err) => {
                            let _ = tx.send(Err(err));
                            return;
                        }
                    }
                };
                app = app.layer(cors);
            }

            match tokio::net::TcpListener::bind(&config.bind_addr).await {
                Ok(listener) => {
                    let bound_addr = listener
                        .local_addr()
                        .map(|addr| addr.to_string())
                        .unwrap_or_else(|_| config.bind_addr.clone());
                    let _ = tx.send(Ok(bound_addr));
                    let shutdown = async move {
                        let _ = shutdown_rx.await;
                    };
                    if let Err(err) = axum::serve(
                        listener,
                        app.into_make_service_with_connect_info::<SocketAddr>(),
                    )
                    .with_graceful_shutdown(shutdown)
                    .await
                    {
                        eprintln!("cluster public HTTP failed: {err}");
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(anyhow!(
                        "failed to bind public HTTP '{}': {err}",
                        config.bind_addr
                    )));
                }
            }
        });
    });

    let bound_addr = rx
        .recv_timeout(Duration::from_secs(10))
        .map_err(|_| anyhow!("timed out starting public HTTP server"))??;
    Ok(PublicServerHandle {
        bound_addr,
        shutdown_tx: Some(shutdown_tx),
        thread_handle: Some(thread_handle),
    })
}

fn authorize_request(
    state: &ManagedApiState,
    headers: &HeaderMap,
    remote_addr: SocketAddr,
) -> Result<(), ApiError> {
    if !client_ip_allowed(&state.allowed_client_ips, remote_addr.ip()) {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            message: format!("client IP '{}' is not allowed", remote_addr.ip()),
        });
    }
    let Some(expected) = state.api_key.as_deref() else {
        return Ok(());
    };
    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string());
    let api_key_header = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string());
    let bearer = auth_header
        .as_deref()
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim);
    let supplied = bearer.or(api_key_header.as_deref()).unwrap_or_default();
    if supplied == expected {
        Ok(())
    } else {
        Err(ApiError {
            status: StatusCode::UNAUTHORIZED,
            message: "missing or invalid API key".to_string(),
        })
    }
}

fn client_ip_allowed(allowed_client_ips: &[String], remote_ip: IpAddr) -> bool {
    allowed_client_ips.is_empty()
        || allowed_client_ips
            .iter()
            .any(|entry| allowed_ip_entry_matches(entry, remote_ip))
}

fn allowed_ip_entry_matches(entry: &str, remote_ip: IpAddr) -> bool {
    let trimmed = entry.trim();
    if trimmed.is_empty() {
        return false;
    }
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return ip == remote_ip;
    }
    let Some((base_text, prefix_text)) = trimmed.split_once('/') else {
        return false;
    };
    let Ok(base_ip) = base_text.trim().parse::<IpAddr>() else {
        return false;
    };
    let Ok(prefix) = prefix_text.trim().parse::<u8>() else {
        return false;
    };
    ip_in_cidr(remote_ip, base_ip, prefix)
}

fn ip_in_cidr(remote_ip: IpAddr, base_ip: IpAddr, prefix: u8) -> bool {
    match (remote_ip, base_ip) {
        (IpAddr::V4(remote), IpAddr::V4(base)) if prefix <= 32 => {
            let remote = u32::from(remote);
            let base = u32::from(base);
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix)
            };
            (remote & mask) == (base & mask)
        }
        (IpAddr::V6(remote), IpAddr::V6(base)) if prefix <= 128 => {
            let remote = u128::from_be_bytes(remote.octets());
            let base = u128::from_be_bytes(base.octets());
            let mask = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - prefix)
            };
            (remote & mask) == (base & mask)
        }
        _ => false,
    }
}

async fn list_models(
    State(state): State<Arc<ManagedApiState>>,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorize_request(&state, &headers, remote_addr)?;
    let state_for_lookup = state.clone();
    let instances = run_blocking(move || list_cluster_instances(&state_for_lookup)).await?;
    let data = instances
        .into_iter()
        .map(|resolved| {
            let instance = resolved.instance;
            json!({
                "id": instance.name,
                "object": "model",
                "owned_by": resolved.owner_display_name,
                "owner_control_addr": resolved.owner_control_addr,
                "kind": instance.model_kind.as_dropdown_value(),
                "state": instance.state,
                "vision": instance.supports_vision(),
                "embeddings": instance.supports_embeddings(),
                "rerank": instance.supports_rerank(),
                "transcription": supports_public_transcription(&instance),
                "diarization": instance.supports_diarization(),
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({ "object": "list", "data": data })))
}

async fn run_responses(
    State(state): State<Arc<ManagedApiState>>,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<ResponsesRequestBody>,
) -> Result<Json<Value>, ApiError> {
    authorize_request(&state, &headers, remote_addr)?;
    let _permit = acquire_request_slot(&state).await?;
    let overrides = schedule_overrides_from_json(
        body.x_engine_n_parallel,
        body.x_engine_retention,
        body.x_engine_allowed_nodes,
        body.x_engine_preferred_owner,
        body.x_engine_execution_group,
    )?;
    ensure_no_schedule_overrides(&overrides)?;
    let resolved = resolve_instance_target_async(state.clone(), body.model.clone()).await?;
    ensure_instance_supports(
        &resolved,
        "/v1/responses",
        resolved.instance.supports_responses(),
    )?;
    let owner_control_addr = resolved.owner_control_addr.clone();
    let instance_id = resolved.instance.instance_id;
    let model_name = resolved.instance.name.clone();
    let supports_vision = resolved.instance.supports_vision();

    let (prompt, image_bytes) = extract_responses_input(&body.input)?;
    let response = if let Some(image_bytes) = image_bytes {
        ensure_instance_supports(&resolved, "vision input", supports_vision)?;
        run_blocking(move || {
            AgentClient::new(owner_control_addr)
                .vlm_complete(VlmRequest {
                    instance_id,
                    prompt: if prompt.trim().is_empty() {
                        "Describe the image.".to_string()
                    } else {
                        prompt
                    },
                    image_bytes,
                    n_predict: body.max_output_tokens.unwrap_or(512),
                    temperature: body.temperature.unwrap_or(0.7),
                    top_p: body.top_p.unwrap_or(0.95),
                    top_k: body.top_k.unwrap_or(40),
                    min_p: 0.05,
                    repeat_last_n: 64,
                    repeat_penalty: 1.05,
                    reasoning: managed_reasoning_mode(body.x_engine_reasoning.clone()),
                    reasoning_budget: managed_reasoning_budget(body.x_engine_reasoning_budget),
                    reasoning_format: managed_reasoning_format(
                        body.x_engine_reasoning_format.clone(),
                    ),
                })
                .map_err(map_runtime_error)
        })
        .await?
    } else {
        let owner_control_addr = resolved.owner_control_addr.clone();
        run_blocking(move || {
            AgentClient::new(owner_control_addr)
                .chat_complete(ChatRequest {
                    instance_id,
                    prompt,
                    n_predict: body.max_output_tokens.unwrap_or(512),
                    temperature: body.temperature.unwrap_or(0.7),
                    top_p: body.top_p.unwrap_or(0.95),
                    top_k: body.top_k.unwrap_or(40),
                    min_p: 0.05,
                    repeat_last_n: 64,
                    repeat_penalty: 1.05,
                    reasoning: managed_reasoning_mode(body.x_engine_reasoning.clone()),
                    reasoning_budget: managed_reasoning_budget(body.x_engine_reasoning_budget),
                    reasoning_format: managed_reasoning_format(
                        body.x_engine_reasoning_format.clone(),
                    ),
                })
                .map_err(map_runtime_error)
        })
        .await?
    };
    let response_text = response.text.clone();
    let metrics_json = metrics_value(&response.metrics);

    let now = unix_time_secs();
    Ok(Json(json!({
        "id": format!("resp_{}_{}", now, instance_id),
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model_name,
        "output": [
            {
                "id": format!("msg_{}_{}", now, instance_id),
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_text.clone(),
                        "annotations": []
                    }
                ]
            }
        ],
        "output_text": response_text,
        "x_engine_metrics": metrics_json
    })))
}

async fn run_chat_completions(
    State(state): State<Arc<ManagedApiState>>,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<ChatCompletionsRequestBody>,
) -> Result<Json<Value>, ApiError> {
    authorize_request(&state, &headers, remote_addr)?;
    let _permit = acquire_request_slot(&state).await?;
    if body.stream.unwrap_or(false) {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "stream=true is not implemented on this instance endpoint yet".to_string(),
        });
    }

    let overrides = schedule_overrides_from_json(
        body.x_engine_n_parallel,
        body.x_engine_retention,
        body.x_engine_allowed_nodes,
        body.x_engine_preferred_owner,
        body.x_engine_execution_group,
    )?;
    ensure_no_schedule_overrides(&overrides)?;
    let resolved = resolve_instance_target_async(state.clone(), body.model.clone()).await?;
    ensure_instance_supports(
        &resolved,
        "/v1/chat/completions",
        resolved.instance.supports_responses(),
    )?;
    let owner_control_addr = resolved.owner_control_addr.clone();
    let instance_id = resolved.instance.instance_id;
    let model_name = resolved.instance.name.clone();
    let supports_vision = resolved.instance.supports_vision();
    let (prompt, image_bytes) = extract_chat_messages_input(&body.messages)?;

    let response = if let Some(image_bytes) = image_bytes {
        ensure_instance_supports(&resolved, "vision input", supports_vision)?;
        run_blocking(move || {
            AgentClient::new(owner_control_addr)
                .vlm_complete(VlmRequest {
                    instance_id,
                    prompt: if prompt.trim().is_empty() {
                        "Describe the image.".to_string()
                    } else {
                        prompt
                    },
                    image_bytes,
                    n_predict: body.max_tokens.unwrap_or(512),
                    temperature: body.temperature.unwrap_or(0.7),
                    top_p: body.top_p.unwrap_or(0.95),
                    top_k: body.top_k.unwrap_or(40),
                    min_p: 0.05,
                    repeat_last_n: 64,
                    repeat_penalty: 1.05,
                    reasoning: managed_reasoning_mode(body.x_engine_reasoning.clone()),
                    reasoning_budget: managed_reasoning_budget(body.x_engine_reasoning_budget),
                    reasoning_format: managed_reasoning_format(
                        body.x_engine_reasoning_format.clone(),
                    ),
                })
                .map_err(map_runtime_error)
        })
        .await?
    } else {
        let owner_control_addr = resolved.owner_control_addr.clone();
        run_blocking(move || {
            AgentClient::new(owner_control_addr)
                .chat_complete(ChatRequest {
                    instance_id,
                    prompt,
                    n_predict: body.max_tokens.unwrap_or(512),
                    temperature: body.temperature.unwrap_or(0.7),
                    top_p: body.top_p.unwrap_or(0.95),
                    top_k: body.top_k.unwrap_or(40),
                    min_p: 0.05,
                    repeat_last_n: 64,
                    repeat_penalty: 1.05,
                    reasoning: managed_reasoning_mode(body.x_engine_reasoning.clone()),
                    reasoning_budget: managed_reasoning_budget(body.x_engine_reasoning_budget),
                    reasoning_format: managed_reasoning_format(
                        body.x_engine_reasoning_format.clone(),
                    ),
                })
                .map_err(map_runtime_error)
        })
        .await?
    };

    let now = unix_time_secs();
    Ok(Json(json!({
        "id": format!("chatcmpl-{}_{}", now, instance_id),
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": response.text
                }
            }
        ],
        "usage": {
            "prompt_tokens": response.metrics.prompt_tokens,
            "completion_tokens": response.metrics.decoded_tokens,
            "total_tokens": response.metrics.prompt_tokens + response.metrics.decoded_tokens
        },
        "x_engine_metrics": metrics_value(&response.metrics)
    })))
}

async fn run_embeddings(
    State(state): State<Arc<ManagedApiState>>,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<EmbeddingsRequestBody>,
) -> Result<Json<Value>, ApiError> {
    authorize_request(&state, &headers, remote_addr)?;
    let _permit = acquire_request_slot(&state).await?;
    let overrides = schedule_overrides_from_json(
        body.x_engine_n_parallel,
        body.x_engine_retention,
        body.x_engine_allowed_nodes,
        body.x_engine_preferred_owner,
        body.x_engine_execution_group,
    )?;
    ensure_no_schedule_overrides(&overrides)?;
    let resolved = resolve_instance_target_async(state.clone(), body.model.clone()).await?;
    ensure_instance_supports(
        &resolved,
        "/v1/embeddings",
        resolved.instance.supports_embeddings(),
    )?;
    let owner_control_addr = resolved.owner_control_addr.clone();
    let instance_id = resolved.instance.instance_id;
    let payload = json!({
        "input": body.input,
        "encoding_format": body.encoding_format.unwrap_or_else(|| "float".to_string())
    });
    let result = run_blocking(move || {
        AgentClient::new(owner_control_addr)
            .embeddings(EmbeddingsRequest {
                instance_id,
                body_json: payload.to_string(),
                oai_compat: true,
            })
            .map_err(map_runtime_error)
    })
    .await?;
    Ok(Json(insert_engine_metrics(
        parse_json_body(&result.json)?,
        &result.metrics,
    )))
}

async fn run_rerank(
    State(state): State<Arc<ManagedApiState>>,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<RerankRequestBody>,
) -> Result<Json<Value>, ApiError> {
    authorize_request(&state, &headers, remote_addr)?;
    let _permit = acquire_request_slot(&state).await?;
    let overrides = schedule_overrides_from_json(
        body.x_engine_n_parallel,
        body.x_engine_retention,
        body.x_engine_allowed_nodes,
        body.x_engine_preferred_owner,
        body.x_engine_execution_group,
    )?;
    ensure_no_schedule_overrides(&overrides)?;
    let resolved = resolve_instance_target_async(state.clone(), body.model.clone()).await?;
    ensure_instance_supports(&resolved, "/v1/rerank", resolved.instance.supports_rerank())?;
    let owner_control_addr = resolved.owner_control_addr.clone();
    let instance_id = resolved.instance.instance_id;
    let payload = json!({
        "query": body.query,
        "documents": body.documents,
        "top_n": body.top_n.unwrap_or(5),
    });
    let result = run_blocking(move || {
        AgentClient::new(owner_control_addr)
            .rerank(RerankRequest {
                instance_id,
                body_json: payload.to_string(),
            })
            .map_err(map_runtime_error)
    })
    .await?;
    Ok(Json(insert_engine_metrics(
        parse_json_body(&result.json)?,
        &result.metrics,
    )))
}

async fn run_transcriptions(
    State(state): State<Arc<ManagedApiState>>,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize_request(&state, &headers, remote_addr)?;
    let _permit = acquire_request_slot(&state).await?;
    let mut model = None::<String>;
    let mut metadata = Map::new();
    let mut audio_bytes = Vec::new();
    let mut audio_format = None::<String>;
    let mut enable_diarization = false;
    let mut ffmpeg_convert = None::<bool>;
    let mut preferred_owner_control_addr = None::<String>;
    let mut execution_group_id = None::<String>;
    let mut allowed_control_addrs = None::<Vec<String>>;
    let mut n_parallel = None::<i32>;
    let mut retention_mode = None::<RetentionMode>;

    while let Some(field) = multipart.next_field().await.map_err(|err| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: format!("invalid multipart body: {err}"),
    })? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            if let Some(file_name) = field.file_name() {
                audio_format = infer_audio_format(file_name);
                metadata.insert(
                    "file_name".to_string(),
                    Value::String(file_name.to_string()),
                );
            }
            let content_type = field.content_type().map(|value| value.to_string());
            if let Some(content_type) = content_type {
                metadata.insert(
                    "content_type".to_string(),
                    Value::String(content_type.clone()),
                );
                if audio_format.is_none() {
                    audio_format = infer_audio_format(&content_type);
                }
            }
            audio_bytes = field
                .bytes()
                .await
                .map_err(|err| ApiError {
                    status: StatusCode::BAD_REQUEST,
                    message: format!("failed to read audio upload: {err}"),
                })?
                .to_vec();
        } else {
            let value = field.text().await.map_err(|err| ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!("failed to read multipart field '{name}': {err}"),
            })?;
            if name == "model" {
                model = Some(value);
            } else if name == "diarization" || name == "x_engine_diarization" {
                enable_diarization = parse_bool_field(&value);
            } else if name == "ffmpeg_convert" || name == "x_engine_ffmpeg_convert" {
                ffmpeg_convert = Some(parse_bool_field(&value));
            } else if name == "x_engine_preferred_owner" {
                preferred_owner_control_addr = Some(value);
            } else if name == "x_engine_execution_group" {
                execution_group_id = Some(value);
            } else if name == "x_engine_allowed_nodes" {
                allowed_control_addrs = Some(parse_allowed_control_addrs_string(&value)?);
            } else if name == "x_engine_n_parallel" {
                n_parallel = Some(value.trim().parse::<i32>().map_err(|_| ApiError {
                    status: StatusCode::BAD_REQUEST,
                    message: "x_engine_n_parallel must be an integer".to_string(),
                })?);
            } else if name == "x_engine_retention" {
                retention_mode = Some(parse_retention_mode(&value)?);
            } else {
                metadata.insert(name, Value::String(value));
            }
        }
    }

    let model = model.ok_or_else(|| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: "multipart field 'model' is required".to_string(),
    })?;
    if audio_bytes.is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "multipart field 'file' is required".to_string(),
        });
    }

    if !metadata.contains_key("output_dir") && !metadata.contains_key("output_path") {
        metadata.insert(
            "output_dir".to_string(),
            Value::String(default_managed_audio_output_dir()?.display().to_string()),
        );
    }

    let metadata_json = if metadata.is_empty() {
        None
    } else {
        Some(Value::Object(metadata).to_string())
    };
    let overrides = ScheduleOverrides {
        allowed_control_addrs,
        preferred_owner_control_addr,
        execution_group_id,
        n_parallel,
        retention_mode,
    };
    ensure_no_schedule_overrides(&overrides)?;
    let resolved = resolve_instance_target_async(state.clone(), model.clone()).await?;
    ensure_instance_supports(
        &resolved,
        "/v1/audio/transcriptions",
        supports_public_transcription(&resolved.instance),
    )?;
    let owner_control_addr = resolved.owner_control_addr.clone();
    let instance_id = resolved.instance.instance_id;
    let model_name = resolved.instance.name.clone();
    let diarization_model_path = if enable_diarization {
        Some(
            resolve_same_owner_diarization_path_async(state.clone(), owner_control_addr.clone())
                .await?,
        )
    } else {
        None
    };
    let audio_format = audio_format.unwrap_or_else(|| "wav".to_string());
    let ffmpeg_convert = ffmpeg_convert
        .unwrap_or_else(|| matches!(audio_format.as_str(), "mp3" | "m4a" | "ogg" | "webm"));
    let result = run_blocking(move || {
        AgentClient::new(owner_control_addr)
            .audio_transcriptions_raw(AudioRawRequest {
                instance_id,
                audio_bytes,
                audio_format,
                metadata_json,
                ffmpeg_convert,
                enable_diarization,
                diarization_model_path,
            })
            .map_err(map_runtime_error)
    })
    .await?;

    let mut response = Json(insert_engine_metrics(
        parse_json_body(&result.json)?,
        &result.metrics,
    ))
    .into_response();
    response.headers_mut().insert(
        "x-engine-model",
        HeaderValue::from_str(&model_name).unwrap_or(HeaderValue::from_static("unknown")),
    );
    Ok(response)
}

fn list_cluster_instances(state: &ManagedApiState) -> Result<Vec<ResolvedApiInstance>, ApiError> {
    if let Some(cached) = cached_cluster_instances(state, CLUSTER_INSTANCES_CACHE_TTL) {
        return Ok(cached);
    }

    match AgentClient::new(state.local_control_addr.clone())
        .get_cluster_telemetry()
        .map(flatten_cluster_instances)
        .map_err(map_runtime_error)
    {
        Ok(instances) => {
            store_cluster_instances_cache(state, &instances);
            Ok(instances)
        }
        Err(err) => {
            if let Some(cached) =
                cached_cluster_instances(state, CLUSTER_INSTANCES_STALE_FALLBACK_TTL)
            {
                return Ok(cached);
            }
            Err(err)
        }
    }
}

fn cached_cluster_instances(
    state: &ManagedApiState,
    max_age: Duration,
) -> Option<Vec<ResolvedApiInstance>> {
    let guard = state.instances_cache.lock().ok()?;
    let cached = guard.as_ref()?;
    if cached.fetched_at.elapsed() > max_age {
        return None;
    }
    Some(cached.instances.clone())
}

fn store_cluster_instances_cache(state: &ManagedApiState, instances: &[ResolvedApiInstance]) {
    if let Ok(mut guard) = state.instances_cache.lock() {
        *guard = Some(CachedClusterInstances {
            instances: instances.to_vec(),
            fetched_at: Instant::now(),
        });
    }
}

fn flatten_cluster_instances(telemetry: Vec<TelemetrySnapshot>) -> Vec<ResolvedApiInstance> {
    let mut instances = Vec::new();
    for snapshot in telemetry {
        let owner_control_addr = snapshot.control_addr;
        let owner_display_name = snapshot.node.display_name;
        for instance in snapshot.instances {
            instances.push(ResolvedApiInstance {
                owner_control_addr: owner_control_addr.clone(),
                owner_display_name: owner_display_name.clone(),
                instance,
            });
        }
    }
    instances.sort_by(|lhs, rhs| {
        lhs.instance
            .name
            .cmp(&rhs.instance.name)
            .then(lhs.owner_display_name.cmp(&rhs.owner_display_name))
            .then(lhs.instance.instance_id.cmp(&rhs.instance.instance_id))
    });
    instances
}

fn resolve_instance_target(
    state: &ManagedApiState,
    instance_name: &str,
) -> Result<ResolvedApiInstance, ApiError> {
    let normalized = instance_name.trim();
    if normalized.is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "model must be a non-empty instance name".to_string(),
        });
    }

    let matches = list_cluster_instances(state)?
        .into_iter()
        .filter(|resolved| resolved.instance.name == normalized)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Err(ApiError {
            status: StatusCode::NOT_FOUND,
            message: format!("unknown cluster instance '{normalized}'"),
        }),
        [resolved] => Ok(resolved.clone()),
        _ => {
            let owners = matches
                .iter()
                .map(|resolved| {
                    format!(
                        "{} ({})",
                        resolved.owner_display_name, resolved.owner_control_addr
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(ApiError {
                status: StatusCode::CONFLICT,
                message: format!(
                    "cluster instance name '{normalized}' is ambiguous across nodes: {owners}"
                ),
            })
        }
    }
}

async fn resolve_instance_target_async(
    state: Arc<ManagedApiState>,
    instance_name: String,
) -> Result<ResolvedApiInstance, ApiError> {
    run_blocking(move || resolve_instance_target(&state, &instance_name)).await
}

fn resolve_same_owner_diarization_path(
    state: &ManagedApiState,
    owner_control_addr: &str,
) -> Result<String, ApiError> {
    let matches = list_cluster_instances(state)?
        .into_iter()
        .filter(|resolved| {
            resolved.owner_control_addr == owner_control_addr
                && resolved.instance.supports_diarization()
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: format!(
                "no diarization instance is available on owner '{}'",
                owner_control_addr
            ),
        }),
        [resolved] => Ok(resolved.instance.model_path.clone()),
        _ => Err(ApiError {
            status: StatusCode::CONFLICT,
            message: format!(
                "multiple diarization instances are loaded on owner '{}'; pick a single diarization companion",
                owner_control_addr
            ),
        }),
    }
}

async fn resolve_same_owner_diarization_path_async(
    state: Arc<ManagedApiState>,
    owner_control_addr: String,
) -> Result<String, ApiError> {
    run_blocking(move || resolve_same_owner_diarization_path(&state, &owner_control_addr)).await
}

fn ensure_instance_supports(
    resolved: &ResolvedApiInstance,
    endpoint: &str,
    supported: bool,
) -> Result<(), ApiError> {
    if supported {
        return Ok(());
    }
    Err(ApiError {
        status: StatusCode::BAD_REQUEST,
        message: format!(
            "instance '{}' has kind '{}' and cannot serve {}",
            resolved.instance.name,
            resolved.instance.model_kind.as_dropdown_value(),
            endpoint
        ),
    })
}

fn ensure_no_schedule_overrides(overrides: &ScheduleOverrides) -> Result<(), ApiError> {
    let mut fields = Vec::new();
    if overrides.n_parallel.is_some() {
        fields.push("x_engine_n_parallel");
    }
    if overrides.retention_mode.is_some() {
        fields.push("x_engine_retention");
    }
    if overrides.allowed_control_addrs.is_some() {
        fields.push("x_engine_allowed_nodes");
    }
    if overrides.preferred_owner_control_addr.is_some() {
        fields.push("x_engine_preferred_owner");
    }
    if overrides.execution_group_id.is_some() {
        fields.push("x_engine_execution_group");
    }
    if fields.is_empty() {
        return Ok(());
    }
    Err(ApiError {
        status: StatusCode::BAD_REQUEST,
        message: format!(
            "instance-targeted API does not support scheduling overrides: {}",
            fields.join(", ")
        ),
    })
}

async fn run_blocking<T, F>(func: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiError> + Send + 'static,
{
    tokio::task::spawn_blocking(func)
        .await
        .map_err(|err| ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("public API worker join failed: {err}"),
        })?
}

fn map_runtime_error<E: std::fmt::Display>(message: E) -> ApiError {
    ApiError {
        status: StatusCode::BAD_GATEWAY,
        message: message.to_string(),
    }
}

fn parse_json_body(value: &str) -> Result<Value, ApiError> {
    serde_json::from_str::<Value>(value).map_err(|err| ApiError {
        status: StatusCode::BAD_GATEWAY,
        message: format!("runtime returned invalid JSON: {err}"),
    })
}

fn metrics_value(metrics: &InferenceMetrics) -> Value {
    serde_json::to_value(metrics).unwrap_or(Value::Null)
}

fn insert_engine_metrics(mut value: Value, metrics: &InferenceMetrics) -> Value {
    if let Value::Object(ref mut map) = value {
        map.insert("x_engine_metrics".to_string(), metrics_value(metrics));
    }
    value
}

fn extract_responses_input(input: &Value) -> Result<(String, Option<Vec<u8>>), ApiError> {
    match input {
        Value::String(text) => Ok((text.clone(), None)),
        Value::Array(items) => {
            let mut prompt_parts = Vec::new();
            let mut image_bytes = None;
            for item in items {
                if let Some((prompt, image)) = extract_input_item(item)? {
                    if !prompt.is_empty() {
                        prompt_parts.push(prompt);
                    }
                    if image_bytes.is_none() {
                        image_bytes = image;
                    }
                }
            }
            Ok((prompt_parts.join("\n"), image_bytes))
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                return Ok((text.to_string(), None));
            }
            extract_input_item(input)?.ok_or_else(|| ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "unsupported responses input object".to_string(),
            })
        }
        _ => Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "unsupported responses input shape".to_string(),
        }),
    }
}

fn extract_chat_messages_input(messages: &[Value]) -> Result<(String, Option<Vec<u8>>), ApiError> {
    let mut prompt_parts = Vec::new();
    let mut image_bytes = None;
    for item in messages {
        if let Some((prompt, image)) = extract_input_item(item)? {
            if !prompt.is_empty() {
                prompt_parts.push(prompt);
            }
            if image_bytes.is_none() {
                image_bytes = image;
            }
        }
    }
    Ok((prompt_parts.join("\n"), image_bytes))
}

fn extract_input_item(value: &Value) -> Result<Option<(String, Option<Vec<u8>>)>, ApiError> {
    let Value::Object(map) = value else {
        return Ok(None);
    };

    let role = map
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user")
        .to_string();
    let Some(content) = map.get("content").or_else(|| map.get("input")) else {
        return Ok(None);
    };

    match content {
        Value::String(text) => Ok(Some((format!("{role}: {text}"), None))),
        Value::Array(parts) => {
            let mut prompt = Vec::new();
            let mut image_bytes = None;
            for part in parts {
                let Value::Object(part_map) = part else {
                    continue;
                };
                let part_type = part_map
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                match part_type {
                    "input_text" | "text" | "output_text" => {
                        if let Some(text) = part_map.get("text").and_then(Value::as_str) {
                            prompt.push(text.to_string());
                        }
                    }
                    "input_image" | "image" | "image_url" => {
                        if image_bytes.is_none() {
                            image_bytes = Some(decode_image_part(part_map)?);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Some((
                format!("{role}: {}", prompt.join("\n")),
                image_bytes,
            )))
        }
        Value::Object(part_map) => {
            if let Some(text) = part_map.get("text").and_then(Value::as_str) {
                return Ok(Some((format!("{role}: {text}"), None)));
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn decode_image_part(part_map: &Map<String, Value>) -> Result<Vec<u8>, ApiError> {
    if let Some(text) = part_map.get("image_base64").and_then(Value::as_str) {
        return base64::engine::general_purpose::STANDARD
            .decode(text)
            .map_err(|err| ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!("invalid image_base64 payload: {err}"),
            });
    }

    let image_url = match part_map.get("image_url") {
        Some(Value::String(value)) => value.to_string(),
        Some(Value::Object(map)) => map
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    if image_url.is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "input_image requires image_url or image_base64".to_string(),
        });
    }
    decode_data_url(&image_url)
}

fn decode_data_url(value: &str) -> Result<Vec<u8>, ApiError> {
    let (_, payload) = value.split_once(',').ok_or_else(|| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: "only data: image URLs are currently supported".to_string(),
    })?;
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| ApiError {
            status: StatusCode::BAD_REQUEST,
            message: format!("invalid base64 image payload: {err}"),
        })
}

fn infer_audio_format(value: &str) -> Option<String> {
    let value = value.to_ascii_lowercase();
    for ext in ["wav", "mp3", "m4a", "flac", "ogg", "webm"] {
        if value.ends_with(ext) || value.contains(ext) {
            return Some(ext.to_string());
        }
    }
    None
}

fn default_managed_audio_output_dir() -> Result<PathBuf, ApiError> {
    let path = std::env::temp_dir()
        .join("OpenResearchTools")
        .join("audio-transcriptions");
    fs::create_dir_all(&path).map_err(|err| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!(
            "failed to prepare managed audio output directory '{}': {err}",
            path.display()
        ),
    })?;
    Ok(path)
}

async fn acquire_request_slot(
    state: &Arc<ManagedApiState>,
) -> Result<OwnedSemaphorePermit, ApiError> {
    state
        .request_gate
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "public inference gate is unavailable".to_string(),
        })
}

fn parse_bool_field(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn parse_retention_mode(value: &str) -> Result<RetentionMode, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "keep_loaded" | "keep-loaded" | "keep loaded" | "keep" | "warm" => {
            Ok(RetentionMode::KeepLoaded)
        }
        "load_on_demand" | "load-on-demand" | "load on demand" | "ondemand" | "lazy" => {
            Ok(RetentionMode::LoadOnDemand)
        }
        _ => Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "x_engine_retention must be 'keep_loaded' or 'load_on_demand'".to_string(),
        }),
    }
}

fn parse_allowed_control_addrs_string(value: &str) -> Result<Vec<String>, ApiError> {
    let values = value
        .split(',')
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "x_engine_allowed_nodes must contain at least one control address".to_string(),
        });
    }
    Ok(values)
}

fn schedule_overrides_from_json(
    n_parallel: Option<i32>,
    retention: Option<String>,
    allowed_nodes: Option<Value>,
    preferred_owner_control_addr: Option<String>,
    execution_group_id: Option<String>,
) -> Result<ScheduleOverrides, ApiError> {
    let allowed_control_addrs = match allowed_nodes {
        Some(Value::String(value)) => Some(parse_allowed_control_addrs_string(&value)?),
        Some(Value::Array(values)) => {
            let addrs = values
                .into_iter()
                .filter_map(|value| value.as_str().map(|value| value.trim().to_string()))
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>();
            if addrs.is_empty() {
                return Err(ApiError {
                    status: StatusCode::BAD_REQUEST,
                    message:
                        "x_engine_allowed_nodes array must contain at least one string address"
                            .to_string(),
                });
            }
            Some(addrs)
        }
        Some(_) => {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "x_engine_allowed_nodes must be a string or array of strings".to_string(),
            })
        }
        None => None,
    };

    Ok(ScheduleOverrides {
        allowed_control_addrs,
        preferred_owner_control_addr,
        execution_group_id,
        n_parallel: n_parallel.map(|value| value.max(1)),
        retention_mode: retention.as_deref().map(parse_retention_mode).transpose()?,
    })
}

fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
