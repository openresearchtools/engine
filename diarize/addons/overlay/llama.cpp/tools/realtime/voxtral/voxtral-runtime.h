#ifndef LLAMA_REALTIME_VOXTRAL_RUNTIME_H
#define LLAMA_REALTIME_VOXTRAL_RUNTIME_H

// Adapted from the MIT-licensed voxtral-cpp reference implementation.

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#include <functional>
#endif

// ============================================================================
// Constants & Configuration
// ============================================================================

// Encoder configuration
#define VOXTRAL_ENC_DIM         1280
#define VOXTRAL_ENC_LAYERS      32
#define VOXTRAL_ENC_HEADS       32
#define VOXTRAL_ENC_HEAD_DIM    64
#define VOXTRAL_ENC_HIDDEN      5120
#define VOXTRAL_ENC_KV_HEADS    32
#define VOXTRAL_ENC_WINDOW      750
#define VOXTRAL_ENC_NORM_EPS    1e-5f
#define VOXTRAL_ENC_ROPE_THETA  1000000.0f

// Decoder configuration
#define VOXTRAL_DEC_DIM         3072
#define VOXTRAL_DEC_LAYERS      26
#define VOXTRAL_DEC_HEADS       32
#define VOXTRAL_DEC_HEAD_DIM    128
#define VOXTRAL_DEC_HIDDEN      9216
#define VOXTRAL_DEC_KV_HEADS    8
#define VOXTRAL_DEC_WINDOW      8192
#define VOXTRAL_DEC_NORM_EPS    1e-5f
#define VOXTRAL_DEC_ROPE_THETA  1000000.0f
#define VOXTRAL_VOCAB_SIZE      131072

// Audio configuration
#define VOXTRAL_SAMPLE_RATE         16000
#define VOXTRAL_FRAME_RATE          12.5f
#define VOXTRAL_NUM_MEL_BINS        128
#define VOXTRAL_HOP_LENGTH          160
#define VOXTRAL_WINDOW_SIZE         400
#define VOXTRAL_GLOBAL_LOG_MEL_MAX  1.5f
#define VOXTRAL_DOWNSAMPLE_FACTOR   4

// Adaptive normalization
#define VOXTRAL_ADA_NORM_DIM    32

// Streaming configuration
#define VOXTRAL_N_LEFT_PAD_TOKENS   32
#define VOXTRAL_TRANSCRIPTION_DELAY_MS  480
#define VOXTRAL_N_DELAY_TOKENS      6
#define VOXTRAL_N_RIGHT_PAD_TOKENS  17
#define VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK  1280

// Special tokens
#define VOXTRAL_TOKEN_BOS           1
#define VOXTRAL_TOKEN_EOS           2
#define VOXTRAL_TOKEN_STREAMING_PAD 32
#define VOXTRAL_TOKEN_STREAMING_WORD 33
#define VOXTRAL_TOKEN_BEGIN_AUDIO   25
#define VOXTRAL_TOKEN_AUDIO         24

// ============================================================================
// C++ API
// ============================================================================

#ifdef __cplusplus

enum class voxtral_log_level : int {
    error = 0,
    warn  = 1,
    info  = 2,
    debug = 3,
};

enum class voxtral_gpu_backend : int {
    none = 0,
    auto_detect,
    cuda,
    metal,
    vulkan,
};

using voxtral_log_callback = std::function<void(voxtral_log_level, const std::string &)>;

// ============================================================================
// Model Weights (opaque in header, defined in .cpp)
// ============================================================================

struct voxtral_model;

// ============================================================================
// Context Parameters
// ============================================================================

struct voxtral_context_params {
    int32_t              n_threads  = 0;
    voxtral_log_level    log_level  = voxtral_log_level::info;
    voxtral_log_callback logger     = nullptr;
    voxtral_gpu_backend  gpu        = voxtral_gpu_backend::none;
};

// ============================================================================
// Inference Result
// ============================================================================

struct voxtral_result {
    std::string          text;
    std::vector<int32_t> tokens;
    std::vector<float>   first_step_logits;   // full vocab logits from first decode step
};

enum class voxtral_stream_event_kind : int {
    piece_commit = 0,
    notice = 1,
    error = 2,
};

struct voxtral_stream_event {
    voxtral_stream_event_kind kind = voxtral_stream_event_kind::notice;
    double begin_sec = 0.0;
    double end_sec = 0.0;
    std::string text;
    std::string detail;
};

struct voxtral_stream_params {
    int32_t delay_ms = VOXTRAL_TRANSCRIPTION_DELAY_MS;
    int32_t processing_interval_ms = VOXTRAL_TRANSCRIPTION_DELAY_MS;
    int32_t right_pad_tokens = VOXTRAL_N_RIGHT_PAD_TOKENS;
    int32_t max_tokens_per_step = 256;
    bool continuous_mode = false;
};

// ============================================================================
// Context (opaque in header, defined in .cpp)
// ============================================================================

struct voxtral_context;
struct voxtral_stream;

// ============================================================================
// Public API
// ============================================================================

voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger = nullptr,
    voxtral_gpu_backend    gpu = voxtral_gpu_backend::none);

void voxtral_model_free(voxtral_model * model);

voxtral_context * voxtral_init_from_model(
    voxtral_model              * model,
    const voxtral_context_params & params);

void voxtral_free(voxtral_context * ctx);

bool voxtral_transcribe_file(
    voxtral_context  & ctx,
    const std::string & audio_path,
    int32_t            max_tokens,
    voxtral_result   & result);

bool voxtral_transcribe_audio(
    voxtral_context  & ctx,
    const std::vector<float> & audio,
    int32_t            max_tokens,
    voxtral_result   & result);

bool voxtral_transcribe_audio_events(
    voxtral_context  & ctx,
    const std::vector<float> & audio,
    int32_t            max_tokens,
    std::vector<voxtral_stream_event> & out_events,
    voxtral_result   * out_result = nullptr);

voxtral_stream * voxtral_stream_init(
    voxtral_context & ctx,
    const voxtral_stream_params & params = {});

void voxtral_stream_free(voxtral_stream * stream);

bool voxtral_stream_reset(
    voxtral_stream & stream,
    std::string * error = nullptr);

bool voxtral_stream_push_audio(
    voxtral_stream & stream,
    const float * samples,
    int32_t n_samples,
    std::vector<voxtral_stream_event> & out_events,
    std::string * error = nullptr);

bool voxtral_stream_flush(
    voxtral_stream & stream,
    std::vector<voxtral_stream_event> & out_events,
    std::string * error = nullptr);

#endif // __cplusplus

#endif // LLAMA_REALTIME_VOXTRAL_RUNTIME_H
