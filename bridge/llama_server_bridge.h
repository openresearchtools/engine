#pragma once

#include "llama.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef LLAMA_SERVER_BRIDGE_BUILD
#        define LLAMA_SERVER_BRIDGE_API __declspec(dllexport)
#    else
#        define LLAMA_SERVER_BRIDGE_API __declspec(dllimport)
#    endif
#else
#    define LLAMA_SERVER_BRIDGE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llama_server_bridge;

struct llama_server_bridge_params {
    const char * model_path;
    const char * mmproj_path;

    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_parallel;
    int32_t n_threads;
    int32_t n_threads_batch;

    int32_t n_gpu_layers;
    int32_t main_gpu;
    int32_t gpu; // -1 unset/default, >=0 single-device selector from list-devices
    int32_t no_kv_offload;
    int32_t mmproj_use_gpu;  // -1 auto, 0 CPU, 1 GPU
    int32_t cache_ram_mib;

    int32_t seed;
    int32_t ctx_shift;
    int32_t kv_unified;

    // optional multi-device / split controls
    const char * devices;       // comma-separated device names or indices
    const char * tensor_split;  // comma-separated floats, one per device
    int32_t split_mode;         // -1 unset, 0 none, 1 layer, 2 row

    // optional embedding/rerank mode controls
    int32_t embedding;          // 0/1
    int32_t reranking;          // 0/1
    int32_t pooling_type;       // -1 unset, otherwise llama_pooling_type
};

struct llama_server_bridge_chat_request {
    const char * prompt;

    int32_t n_predict;
    int32_t id_slot; // -1 = any slot
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    int32_t seed;

    int32_t repeat_last_n;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;
    float dry_multiplier;
    int32_t dry_allowed_length;
    int32_t dry_penalty_last_n;
};

struct llama_server_bridge_vlm_request {
    const char * prompt;
    const uint8_t * image_bytes;
    size_t image_bytes_len;

    int32_t n_predict;
    int32_t id_slot; // -1 = any slot
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    int32_t seed;

    int32_t repeat_last_n;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;
    float dry_multiplier;
    int32_t dry_allowed_length;
    int32_t dry_penalty_last_n;
};

struct llama_server_bridge_vlm_result {
    int32_t ok;
    int32_t truncated;
    int32_t stop;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_tokens_cached;
    int32_t eos_reached;

    double prompt_ms;
    double predicted_ms;

    char * text;
    char * error_json;
};

struct llama_server_bridge_embeddings_request {
    const char * body_json; // OpenAI-compatible embeddings body JSON
    int32_t oai_compat;     // 0 => /embeddings, 1 => /v1/embeddings
};

struct llama_server_bridge_rerank_request {
    const char * body_json; // rerank body JSON: query + documents/texts (+ top_n)
};

struct llama_server_bridge_audio_request {
    const char * body_json; // /v1/audio/transcriptions body JSON
};

struct llama_server_bridge_audio_raw_request {
    const uint8_t * audio_bytes;
    size_t audio_bytes_len;
    const char * audio_format;   // input format (wav/mp3/...)
    const char * metadata_json;  // body without "audio", e.g. mode/custom/model fields
    int32_t ffmpeg_convert;      // 0/1: convert to WAV 16-bit mono 16 kHz in RAM before request
};

struct llama_server_bridge_json_result {
    int32_t ok;
    int32_t status;
    char * json;
    char * error_json;
};

struct llama_server_bridge_device_info {
    int32_t index;
    int32_t type; // ggml_backend_dev_type
    uint64_t memory_free;
    uint64_t memory_total;
    char * backend;
    char * name;
    char * description;
};

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_params llama_server_bridge_default_params(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_chat_request llama_server_bridge_default_chat_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_vlm_request llama_server_bridge_default_vlm_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_vlm_result llama_server_bridge_empty_vlm_result(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_embeddings_request llama_server_bridge_default_embeddings_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_rerank_request llama_server_bridge_default_rerank_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_request llama_server_bridge_default_audio_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_raw_request llama_server_bridge_default_audio_raw_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_json_result llama_server_bridge_empty_json_result(void);

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge * llama_server_bridge_create(const struct llama_server_bridge_params * params);
LLAMA_SERVER_BRIDGE_API void llama_server_bridge_destroy(struct llama_server_bridge * bridge);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_vlm_complete(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_vlm_request * req,
    struct llama_server_bridge_vlm_result * out);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_chat_complete(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_chat_request * req,
    struct llama_server_bridge_vlm_result * out);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_embeddings(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_embeddings_request * req,
    struct llama_server_bridge_json_result * out);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_rerank(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_rerank_request * req,
    struct llama_server_bridge_json_result * out);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_transcriptions(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_audio_request * req,
    struct llama_server_bridge_json_result * out);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_transcriptions_raw(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_audio_raw_request * req,
    struct llama_server_bridge_json_result * out);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_result_free(struct llama_server_bridge_vlm_result * out);
LLAMA_SERVER_BRIDGE_API void llama_server_bridge_json_result_free(struct llama_server_bridge_json_result * out);
LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_last_error(const struct llama_server_bridge * bridge);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_list_devices(
    struct llama_server_bridge_device_info ** out_devices,
    size_t * out_count);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_free_devices(
    struct llama_server_bridge_device_info * devices,
    size_t count);

#ifdef __cplusplus
}
#endif
