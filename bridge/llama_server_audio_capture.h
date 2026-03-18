#pragma once

#include "llama_server_bridge.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef LLAMA_SERVER_AUDIO_BUILD
#        define LLAMA_SERVER_AUDIO_API __declspec(dllexport)
#    else
#        define LLAMA_SERVER_AUDIO_API __declspec(dllimport)
#    endif
#else
#    define LLAMA_SERVER_AUDIO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llama_server_audio_live;

struct llama_server_audio_capture_device_info {
    int32_t index;
    int32_t is_default;
    char * name;
};

struct llama_server_audio_output_paths {
    char * output_dir;
    char * cleaned_wav_path;
    char * transcript_path;
    char * preview_path;
};

struct llama_server_audio_live_params {
    const char * output_dir;
    const char * session_name;
    const char * capture_device_name;
    int32_t capture_device_index; // -1 => default capture device
    uint32_t bridge_push_samples; // defaults to 7680 samples at 16 kHz
    int32_t enable_webrtc;        // 0/1
    int32_t enable_transcription; // 0/1
    int32_t enable_diarization;   // 0/1
    int32_t write_clean_wav;      // 0/1
    int32_t write_preview_file;   // 0/1
    uint32_t event_queue_capacity; // 0 => unbounded
    struct llama_server_bridge_audio_session_params session_params;
    struct llama_server_bridge_audio_transcription_params transcription_params;
    struct llama_server_bridge_realtime_params diarization_params;
};

LLAMA_SERVER_AUDIO_API struct llama_server_audio_live_params llama_server_audio_default_live_params(void);
LLAMA_SERVER_AUDIO_API struct llama_server_audio_output_paths llama_server_audio_empty_output_paths(void);

LLAMA_SERVER_AUDIO_API int32_t llama_server_audio_list_capture_devices(
    struct llama_server_audio_capture_device_info ** out_devices,
    size_t * out_count);

LLAMA_SERVER_AUDIO_API void llama_server_audio_free_capture_devices(
    struct llama_server_audio_capture_device_info * devices,
    size_t count);

LLAMA_SERVER_AUDIO_API struct llama_server_audio_live * llama_server_audio_live_create(
    const struct llama_server_audio_live_params * params);

LLAMA_SERVER_AUDIO_API void llama_server_audio_live_destroy(
    struct llama_server_audio_live * live);

LLAMA_SERVER_AUDIO_API int32_t llama_server_audio_live_start(
    struct llama_server_audio_live * live);

LLAMA_SERVER_AUDIO_API int32_t llama_server_audio_live_stop(
    struct llama_server_audio_live * live);

LLAMA_SERVER_AUDIO_API int32_t llama_server_audio_live_wait_events(
    struct llama_server_audio_live * live,
    uint32_t timeout_ms);

LLAMA_SERVER_AUDIO_API int32_t llama_server_audio_live_drain_events(
    struct llama_server_audio_live * live,
    struct llama_server_bridge_audio_event ** out_events,
    size_t * out_count,
    size_t max_events);

LLAMA_SERVER_AUDIO_API void llama_server_audio_live_free_events(
    struct llama_server_bridge_audio_event * events,
    size_t count);

LLAMA_SERVER_AUDIO_API int32_t llama_server_audio_live_get_output_paths(
    const struct llama_server_audio_live * live,
    struct llama_server_audio_output_paths * out_paths);

LLAMA_SERVER_AUDIO_API void llama_server_audio_output_paths_free(
    struct llama_server_audio_output_paths * paths);

LLAMA_SERVER_AUDIO_API const char * llama_server_audio_live_last_error(
    const struct llama_server_audio_live * live);

#ifdef __cplusplus
}
#endif
