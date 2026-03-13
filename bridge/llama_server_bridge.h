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
struct llama_server_bridge_audio_session;
struct llama_server_bridge_realtime;
struct llama_server_bridge_realtime_model_impl;
typedef struct llama_server_bridge_realtime_model_impl llama_server_bridge_realtime_model;
typedef struct llama_server_bridge_realtime_model_impl llama_server_bridge_realtime_sortformer_model;

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

    const char * reasoning;        // null/unset, or: on | off | auto
    int32_t reasoning_budget;      // INT32_MIN = unset, -1 = unlimited, 0 = disable, >0 = requested limit
    const char * reasoning_format; // null/unset, or: none | deepseek | deepseek-legacy
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

    const char * reasoning;        // null/unset, or: on | off | auto
    int32_t reasoning_budget;      // INT32_MIN = unset, -1 = unlimited, 0 = disable, >0 = requested limit
    const char * reasoning_format; // null/unset, or: none | deepseek | deepseek-legacy
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

struct llama_server_bridge_realtime_sortformer_params {
    const char * gguf_path;
    const char * backend_name; // e.g. "Vulkan0", "CPU"
    uint32_t expected_sample_rate_hz;
    uint32_t audio_ring_capacity_samples;
};

enum llama_server_bridge_realtime_backend_kind {
    LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_AUTO = 0,
    LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER = 1,
    LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_VOXTRAL = 2,
};

struct llama_server_bridge_realtime_params {
    int32_t backend_kind;       // llama_server_bridge_realtime_backend_kind, 0 = auto-detect from model_path
    const char * model_path;    // backend-specific primary model path, e.g. Sortformer GGUF
    const char * backend_name;  // e.g. "Vulkan0", "CPU"
    uint32_t expected_sample_rate_hz;
    uint32_t audio_ring_capacity_samples;
    uint32_t capture_debug;     // 0/1: request backend debug capture/logging for parity tooling
};

struct llama_server_bridge_realtime_backend_info {
    int32_t backend_kind;
    const char * name;
    const char * default_runtime_backend_name; // actual loaded runtime backend for model_get_info
    int32_t supports_model_preload;
    int32_t emits_transcript;
    int32_t emits_speaker_spans;
    uint32_t default_sample_rate_hz;
    uint32_t default_audio_ring_capacity_samples;
    uint32_t required_input_channels;
};

struct llama_server_bridge_realtime_event {
    int32_t type;
    int64_t session_id;
    double begin_sec;
    double end_sec;
    int32_t speaker_id;
    char * text;
    char * detail;
};

enum llama_server_bridge_audio_sample_format {
    LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_F32 = 1,
    LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16 = 2,
};

enum llama_server_bridge_audio_event_kind {
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE = 0,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STARTED = 1,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED = 2,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT = 3,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT = 4,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_STATUS = 5,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR = 6,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STARTED = 7,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT = 8,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT = 9,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON = 10,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED = 11,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_STREAM_FLUSHED = 12,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR = 13,
};

enum llama_server_bridge_audio_event_flags {
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL = 1u << 0,
    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY = 1u << 1,
};

struct llama_server_bridge_audio_session_params {
    uint32_t expected_input_sample_rate_hz;
    uint32_t expected_input_channels;
    uint32_t max_buffered_audio_samples; // 0 = unbounded
    uint32_t event_queue_capacity;       // 0 = unbounded
};

enum llama_server_bridge_audio_transcription_mode {
    LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE = 0,
    LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE = 1,
};

struct llama_server_bridge_audio_transcription_params {
    struct llama_server_bridge_params bridge_params;
    const char * metadata_json; // body JSON without the "audio" field
    int32_t mode;               // llama_server_bridge_audio_transcription_mode
    struct llama_server_bridge_realtime_params realtime_params;
};

struct llama_server_bridge_audio_event {
    uint64_t seq_no;
    int32_t kind;
    uint32_t flags;
    uint64_t start_sample;
    uint64_t end_sample;
    int32_t speaker_id;
    uint32_t item_id;
    char * text;
    char * detail;
};

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_params llama_server_bridge_default_params(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_chat_request llama_server_bridge_default_chat_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_vlm_request llama_server_bridge_default_vlm_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_vlm_result llama_server_bridge_empty_vlm_result(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_embeddings_request llama_server_bridge_default_embeddings_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_rerank_request llama_server_bridge_default_rerank_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_request llama_server_bridge_default_audio_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_raw_request llama_server_bridge_default_audio_raw_request(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_session_params llama_server_bridge_default_audio_session_params(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_transcription_params llama_server_bridge_default_audio_transcription_params(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_json_result llama_server_bridge_empty_json_result(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime_sortformer_params llama_server_bridge_default_realtime_sortformer_params(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime_params llama_server_bridge_default_realtime_params_for_backend(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime_params llama_server_bridge_default_realtime_params(void);
LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime_backend_info llama_server_bridge_empty_realtime_backend_info(void);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_count(void);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_kind_at(size_t index);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_get_info(
    int32_t backend_kind,
    struct llama_server_bridge_realtime_backend_info * out_info);
LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_realtime_backend_name(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_kind_from_name(const char * name);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_kind_from_model_path(const char * model_path);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_supports_model_preload(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_emits_transcript(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_backend_emits_speaker_spans(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_realtime_backend_default_runtime_backend_name(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API uint32_t llama_server_bridge_realtime_backend_default_sample_rate_hz(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API uint32_t llama_server_bridge_realtime_backend_default_audio_ring_capacity_samples(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API uint32_t llama_server_bridge_realtime_backend_required_input_channels(int32_t backend_kind);
LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_model_cache_entry_count(void);
LLAMA_SERVER_BRIDGE_API void llama_server_bridge_realtime_model_cache_clear(void);

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

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_audio_session * llama_server_bridge_audio_session_create(
    const struct llama_server_bridge_audio_session_params * params);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_audio_session_destroy(
    struct llama_server_bridge_audio_session * session);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_push_audio(
    struct llama_server_bridge_audio_session * session,
    const void * audio_bytes,
    size_t frame_count,
    uint32_t sample_rate_hz,
    uint32_t channels,
    int32_t sample_format);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_push_encoded(
    struct llama_server_bridge_audio_session * session,
    const uint8_t * audio_bytes,
    size_t audio_bytes_len,
    const char * audio_format);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_flush_audio(
    struct llama_server_bridge_audio_session * session);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_start_diarization(
    struct llama_server_bridge_audio_session * session,
    const struct llama_server_bridge_realtime_params * params);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_stop_diarization(
    struct llama_server_bridge_audio_session * session);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_start_transcription(
    struct llama_server_bridge_audio_session * session,
    const struct llama_server_bridge_audio_transcription_params * params);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_stop_transcription(
    struct llama_server_bridge_audio_session * session);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_wait_events(
    struct llama_server_bridge_audio_session * session,
    uint32_t timeout_ms);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_audio_session_drain_events(
    struct llama_server_bridge_audio_session * session,
    struct llama_server_bridge_audio_event ** out_events,
    size_t * out_count,
    size_t max_events);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_audio_session_free_events(
    struct llama_server_bridge_audio_event * events,
    size_t count);

LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_audio_session_last_error(
    const struct llama_server_bridge_audio_session * session);

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime * llama_server_bridge_realtime_sortformer_create(
    const struct llama_server_bridge_realtime_sortformer_params * params);

LLAMA_SERVER_BRIDGE_API llama_server_bridge_realtime_sortformer_model * llama_server_bridge_realtime_sortformer_model_create(
    const struct llama_server_bridge_realtime_sortformer_params * params);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_realtime_sortformer_model_destroy(
    llama_server_bridge_realtime_sortformer_model * model);

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime * llama_server_bridge_realtime_sortformer_create_from_model(
    const llama_server_bridge_realtime_sortformer_model * model,
    const struct llama_server_bridge_realtime_sortformer_params * params);

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime * llama_server_bridge_realtime_create(
    const struct llama_server_bridge_realtime_params * params);

LLAMA_SERVER_BRIDGE_API llama_server_bridge_realtime_model * llama_server_bridge_realtime_model_create(
    const struct llama_server_bridge_realtime_params * params);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_model_get_info(
    const llama_server_bridge_realtime_model * model,
    struct llama_server_bridge_realtime_backend_info * out_info);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_realtime_model_destroy(
    llama_server_bridge_realtime_model * model);

LLAMA_SERVER_BRIDGE_API struct llama_server_bridge_realtime * llama_server_bridge_realtime_create_from_model(
    const llama_server_bridge_realtime_model * model,
    const struct llama_server_bridge_realtime_params * params);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_realtime_destroy(
    struct llama_server_bridge_realtime * bridge);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_push_audio_f32(
    struct llama_server_bridge_realtime * bridge,
    const float * samples,
    size_t n_samples,
    uint32_t sample_rate_hz);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_flush(
    struct llama_server_bridge_realtime * bridge);

LLAMA_SERVER_BRIDGE_API int32_t llama_server_bridge_realtime_drain_events(
    struct llama_server_bridge_realtime * bridge,
    struct llama_server_bridge_realtime_event ** out_events,
    size_t * out_count,
    size_t max_events);

LLAMA_SERVER_BRIDGE_API void llama_server_bridge_realtime_free_events(
    struct llama_server_bridge_realtime_event * events,
    size_t count);

LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_realtime_last_error(
    const struct llama_server_bridge_realtime * bridge);

LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_realtime_model_last_error(
    const llama_server_bridge_realtime_model * model);

LLAMA_SERVER_BRIDGE_API const char * llama_server_bridge_realtime_sortformer_model_last_error(
    const llama_server_bridge_realtime_sortformer_model * model);

#ifdef __cplusplus
}
#endif
