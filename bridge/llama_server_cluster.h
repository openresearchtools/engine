#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef LLAMA_SERVER_CLUSTER_BUILD
#        define LLAMA_SERVER_CLUSTER_API __declspec(dllexport)
#    else
#        define LLAMA_SERVER_CLUSTER_API __declspec(dllimport)
#    endif
#else
#    define LLAMA_SERVER_CLUSTER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llama_server_cluster;

enum llama_server_cluster_instance_retention_mode {
    LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED = 1,
    LLAMA_SERVER_CLUSTER_INSTANCE_LOAD_ON_DEMAND = 2,
};

enum llama_server_cluster_instance_state {
    LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED = 0,
    LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADING = 1,
    LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADED = 2,
    LLAMA_SERVER_CLUSTER_INSTANCE_STATE_SERVING = 3,
    LLAMA_SERVER_CLUSTER_INSTANCE_STATE_GRACE = 4,
    LLAMA_SERVER_CLUSTER_INSTANCE_STATE_FAILED = 5,
};

enum llama_server_cluster_instance_model_kind {
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_TEXT = 0,
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_VISION = 1,
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_EMBEDDINGS = 2,
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_RERANK = 3,
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_WHISPER = 4,
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_REALTIME_AUDIO = 5,
    LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_DIARIZATION = 6,
};

struct llama_server_cluster_node_info {
    char * node_id;
    char * display_name;
    char * os_name;
    char * arch;
};

struct llama_server_cluster_device_info {
    int32_t bridge_device_index;
    int32_t type;
    uint64_t memory_free;
    uint64_t memory_total;
    char * backend;
    char * name;
    char * description;
};

struct llama_server_cluster_execution_group_info {
    char * id;
    char * label;
    char * backend_summary;
    char * devices_csv;
    int32_t device_count;
    int32_t uses_local_split;
    uint64_t memory_free;
    uint64_t memory_total;
};

struct llama_server_cluster_instance_params {
    const char * name;
    const char * model_path;
    const char * mmproj_path;
    const char * diarization_model_path;
    const char * execution_group_id;
    const char * rpc_servers;    // comma-separated host:port RPC endpoints
    const char * manual_devices_csv;   // ordered bridge device indices for this instance
    const char * manual_tensor_split;  // comma-separated weights aligned with manual_devices_csv
    int32_t retention_mode;
    int32_t load_on_demand_grace_seconds;
    int32_t embedding;
    int32_t reranking;
    int32_t model_kind;
    int32_t allow_cpu;
    int32_t allow_integrated_gpu;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_parallel;
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t n_gpu_layers;
};

struct llama_server_cluster_instance_info {
    int64_t instance_id;
    char * name;
    char * model_path;
    char * mmproj_path;
    char * diarization_model_path;
    char * execution_group_id;
    char * rpc_servers;
    int32_t retention_mode;
    int32_t load_on_demand_grace_seconds;
    int32_t model_kind;
    int32_t state;
    int32_t active_request_count;
    int32_t queued_request_count;
    int32_t n_parallel;
    int64_t grace_deadline_unix_ms;
    char * last_error;
};

struct llama_server_cluster_chat_request {
    int64_t instance_id;
    const char * prompt;
    int32_t n_predict;
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    int32_t repeat_last_n;
    float repeat_penalty;
    const char * reasoning;
    int32_t reasoning_budget;
    const char * reasoning_format;
};

struct llama_server_cluster_vlm_request {
    int64_t instance_id;
    const char * prompt;
    const uint8_t * image_bytes;
    size_t image_bytes_len;
    int32_t n_predict;
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    int32_t repeat_last_n;
    float repeat_penalty;
    const char * reasoning;
    int32_t reasoning_budget;
    const char * reasoning_format;
};

struct llama_server_cluster_embeddings_request {
    int64_t instance_id;
    const char * body_json;
    int32_t oai_compat;
};

struct llama_server_cluster_rerank_request {
    int64_t instance_id;
    const char * body_json;
};

struct llama_server_cluster_audio_raw_request {
    int64_t instance_id;
    const uint8_t * audio_bytes;
    size_t audio_bytes_len;
    const char * audio_format;
    const char * metadata_json;
    int32_t ffmpeg_convert;
    int32_t enable_diarization;
    const char * diarization_model_path;
};

struct llama_server_cluster_native_audio_transcription_request {
    const char * model_path;
    const char * execution_group_id;
    const uint8_t * audio_bytes;
    size_t audio_bytes_len;
    const char * audio_format;
    const char * metadata_json;
    int32_t ffmpeg_convert;
    int32_t enable_diarization;
    const char * diarization_model_path;
};

struct llama_server_cluster_inference_metrics {
    int32_t loaded_this_call;
    int32_t used_rpc;
    int32_t rpc_server_count;
    int32_t prompt_tokens;
    int32_t decoded_tokens;
    uint64_t request_bytes;
    uint64_t model_bytes;
    uint64_t mmproj_bytes;
    double queue_wait_ms;
    double load_ms;
    double prompt_ms;
    double predicted_ms;
    double request_total_ms;
    double prompt_tokens_per_second;
    double decode_tokens_per_second;
    double total_tokens_per_second;
};

struct llama_server_cluster_chat_result {
    int32_t ok;
    char * text;
    char * error;
    struct llama_server_cluster_inference_metrics metrics;
};

struct llama_server_cluster_vlm_result {
    int32_t ok;
    char * text;
    char * error;
    struct llama_server_cluster_inference_metrics metrics;
};

struct llama_server_cluster_json_result {
    int32_t ok;
    int32_t status;
    char * json;
    char * error;
    struct llama_server_cluster_inference_metrics metrics;
};

LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_instance_params llama_server_cluster_default_instance_params(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_chat_request llama_server_cluster_default_chat_request(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_chat_result llama_server_cluster_empty_chat_result(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_vlm_request llama_server_cluster_default_vlm_request(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_vlm_result llama_server_cluster_empty_vlm_result(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_embeddings_request llama_server_cluster_default_embeddings_request(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_rerank_request llama_server_cluster_default_rerank_request(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_audio_raw_request llama_server_cluster_default_audio_raw_request(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_native_audio_transcription_request llama_server_cluster_default_native_audio_transcription_request(void);
LLAMA_SERVER_CLUSTER_API struct llama_server_cluster_json_result llama_server_cluster_empty_json_result(void);

LLAMA_SERVER_CLUSTER_API struct llama_server_cluster * llama_server_cluster_create(void);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_destroy(struct llama_server_cluster * cluster);
LLAMA_SERVER_CLUSTER_API const char * llama_server_cluster_last_error(const struct llama_server_cluster * cluster);

LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_get_local_node_info(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_node_info * out_info);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_free_node_info(struct llama_server_cluster_node_info * info);

LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_list_devices(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_device_info ** out_devices,
    size_t * out_count);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_free_devices(
    struct llama_server_cluster_device_info * devices,
    size_t count);

LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_list_execution_groups(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_execution_group_info ** out_groups,
    size_t * out_count);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_free_execution_groups(
    struct llama_server_cluster_execution_group_info * groups,
    size_t count);

LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_list_devices_with_rpc(
    struct llama_server_cluster * cluster,
    const char * rpc_servers,
    struct llama_server_cluster_device_info ** out_devices,
    size_t * out_count);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_list_execution_groups_with_rpc(
    struct llama_server_cluster * cluster,
    const char * rpc_servers,
    struct llama_server_cluster_execution_group_info ** out_groups,
    size_t * out_count);

LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_run_local_rpc_server(
    struct llama_server_cluster * cluster,
    const char * host,
    int32_t port,
    int32_t n_threads);

LLAMA_SERVER_CLUSTER_API int64_t llama_server_cluster_create_instance(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_instance_params * params);
LLAMA_SERVER_CLUSTER_API int64_t llama_server_cluster_find_instance_by_name(
    struct llama_server_cluster * cluster,
    const char * name);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_remove_instance(
    struct llama_server_cluster * cluster,
    int64_t instance_id);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_list_instances(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_instance_info ** out_instances,
    size_t * out_count);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_free_instances(
    struct llama_server_cluster_instance_info * instances,
    size_t count);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_set_instance_retention_mode(
    struct llama_server_cluster * cluster,
    int64_t instance_id,
    int32_t retention_mode);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_load_instance(
    struct llama_server_cluster * cluster,
    int64_t instance_id);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_unload_instance(
    struct llama_server_cluster * cluster,
    int64_t instance_id);

LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_chat_complete(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_chat_request * req,
    struct llama_server_cluster_chat_result * out);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_chat_result_free(
    struct llama_server_cluster_chat_result * out);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_vlm_complete(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_vlm_request * req,
    struct llama_server_cluster_vlm_result * out);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_vlm_result_free(
    struct llama_server_cluster_vlm_result * out);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_embeddings(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_embeddings_request * req,
    struct llama_server_cluster_json_result * out);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_rerank(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_rerank_request * req,
    struct llama_server_cluster_json_result * out);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_audio_transcriptions_raw(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_audio_raw_request * req,
    struct llama_server_cluster_json_result * out);
LLAMA_SERVER_CLUSTER_API int32_t llama_server_cluster_audio_transcriptions_native(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_native_audio_transcription_request * req,
    struct llama_server_cluster_json_result * out);
LLAMA_SERVER_CLUSTER_API void llama_server_cluster_json_result_free(
    struct llama_server_cluster_json_result * out);

#ifdef __cplusplus
}
#endif
