// Adapted from the MIT-licensed voxtral-cpp reference implementation.
#include "voxtral-runtime.h"
#include "gguf.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

struct voxtral_context;

// ============================================================================
// Internal constants
// ============================================================================

static constexpr float VOXTRAL_PI = 3.14159265358979323846f;
static constexpr int32_t VOXTRAL_N_FFT       = VOXTRAL_WINDOW_SIZE;         // 400
static constexpr int32_t VOXTRAL_N_FREQ      = VOXTRAL_N_FFT / 2 + 1;      // 201
static constexpr int32_t VOXTRAL_ENC_CHUNK_MEL     = 3000;  // mel frames per encoder chunk
static constexpr int32_t VOXTRAL_ENC_CHUNK_OVERLAP  = 750;  // overlap in encoder-token space (= window)
static constexpr int32_t VOXTRAL_MAX_ENC_CHUNK      = 2000; // max enc tokens per single chunk

// ============================================================================
// Logging helper
// ============================================================================

#define LOG(ctx_ptr, lvl, ...) \
    do { \
        if ((ctx_ptr)->logger && static_cast<int>(lvl) <= static_cast<int>((ctx_ptr)->log_level)) { \
            char _buf[2048]; \
            snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
            (ctx_ptr)->logger(lvl, std::string(_buf)); \
        } \
    } while (0)

#define LOG_INFO(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::info,  __VA_ARGS__)
#define LOG_WARN(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::warn,  __VA_ARGS__)
#define LOG_ERR(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::error, __VA_ARGS__)
#define LOG_DBG(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::debug, __VA_ARGS__)

static ggml_tensor * ggml_rowwise_scale(
    ggml_context * gctx,
    ggml_tensor  * x,
    ggml_tensor  * weight) {
    ggml_tensor * weight_2d = ggml_reshape_2d(gctx, weight, weight->ne[0], 1);
    return ggml_mul(gctx, x, weight_2d);
}

static ggml_tensor * ggml_rowwise_bias(
    ggml_context * gctx,
    ggml_tensor  * x,
    ggml_tensor  * bias) {
    ggml_tensor * bias_2d = ggml_reshape_2d(gctx, bias, bias->ne[0], 1);
    return ggml_add(gctx, x, bias_2d);
}

static ggml_tensor * ctx_runtime_tensor(
    voxtral_context * ctx,
    ggml_tensor     * src);

static ggml_tensor * build_adapter_project_tensor(
    voxtral_context * ctx,
    ggml_context    * gctx,
    ggml_tensor     * enc_out,
    int32_t           enc_seq);

// ============================================================================
// Weight structures (internal)
// ============================================================================

struct voxtral_encoder_layer {
    ggml_tensor * attn_norm_weight;  // [enc_dim]
    ggml_tensor * attn_q_weight;     // [enc_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_q_bias;       // [enc_heads*enc_head_dim]
    ggml_tensor * attn_k_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_bias;       // [enc_kv_heads*enc_head_dim]
    ggml_tensor * attn_o_weight;     // [enc_dim, enc_heads*enc_head_dim]
    ggml_tensor * attn_o_bias;       // [enc_dim]
    ggml_tensor * ffn_norm_weight;   // [enc_dim]
    ggml_tensor * ffn_w1_weight;     // [enc_hidden, enc_dim]
    ggml_tensor * ffn_w2_weight;     // [enc_dim, enc_hidden]
    ggml_tensor * ffn_w2_bias;       // [enc_dim]
    ggml_tensor * ffn_w3_weight;     // [enc_hidden, enc_dim]
};

struct voxtral_decoder_layer {
    ggml_tensor * attn_norm_weight;  // [dec_dim]
    ggml_tensor * attn_q_weight;     // [dec_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_k_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_v_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_o_weight;     // [dec_dim, dec_heads*dec_head_dim]
    ggml_tensor * ffn_norm_weight;   // [dec_dim]
    ggml_tensor * ffn_w1_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ffn_w2_weight;     // [dec_dim, dec_hidden]
    ggml_tensor * ffn_w3_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ada0_weight;       // [ada_dim, dec_dim]
    ggml_tensor * ada2_weight;       // [dec_dim, ada_dim]
};

// ============================================================================
// Model structure
// ============================================================================

struct voxtral_model {
    // Encoder conv stem
    ggml_tensor * enc_conv0_weight;  // [enc_dim, num_mel_bins, 3]
    ggml_tensor * enc_conv0_bias;    // [enc_dim]
    ggml_tensor * enc_conv1_weight;  // [enc_dim, enc_dim, 3]
    ggml_tensor * enc_conv1_bias;    // [enc_dim]
    std::vector<voxtral_encoder_layer> enc_layers;
    ggml_tensor * enc_norm_weight;   // [enc_dim]

    // Adapter
    ggml_tensor * adapter_0_weight;  // [dec_dim, enc_dim*downsample]
    ggml_tensor * adapter_2_weight;  // [dec_dim, dec_dim]

    // Decoder
    ggml_tensor * tok_embeddings_weight; // [vocab_size, dec_dim]
    std::vector<voxtral_decoder_layer> dec_layers;
    ggml_tensor * dec_norm_weight;   // [dec_dim]

    // Mel filters (stored in GGUF)
    ggml_tensor * mel_filters;       // [n_freq, n_mel] = [201, 128]

    // Tokenizer (Tekken vocab)
    int32_t tokenizer_num_special_tokens = 1000;
    std::unordered_set<int32_t> tokenizer_special_ranks;
    std::vector<std::string> tokenizer_vocab_b64;
    mutable std::unordered_map<int32_t, std::string> tokenizer_bytes_cache;

    // Owning contexts
    ggml_context * ctx_gguf   = nullptr;
    gguf_context * gguf_ctx   = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_backend_t         backend_weights = nullptr;
    bool                   weights_on_gpu = false;
    voxtral_gpu_backend    gpu_type = voxtral_gpu_backend::none;
    std::unordered_map<std::string, std::vector<float>> cpu_1d_tensors;
};

// ============================================================================
// Context structure
// ============================================================================

struct voxtral_context {
    voxtral_model        * model     = nullptr;
    voxtral_log_level      log_level = voxtral_log_level::info;
    voxtral_log_callback   logger    = nullptr;
    int32_t                n_threads = 4;

    // Backend
    ggml_backend_t         backend      = nullptr;
    ggml_backend_t         backend_cpu  = nullptr;
    ggml_backend_t         blas_backend = nullptr;
    voxtral_gpu_backend    gpu_type     = voxtral_gpu_backend::none;
    bool                   owns_backend = false;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> runtime_1d_tensors;

    // Persistent device tensors (allocated once)
    ggml_context       * ctx_persistent = nullptr;
    ggml_backend_buffer_t buf_persistent = nullptr;

    // Per-chunk encoder output (fixed size, reused each chunk)
    ggml_tensor * encoder_chunk_input  = nullptr;  // [enc_dim, MAX_ENC_CHUNK]
    ggml_tensor * encoder_chunk_output = nullptr;  // [enc_dim, MAX_ENC_CHUNK]
    ggml_tensor * decoder_logits  = nullptr;  // [vocab_size]
    ggml_tensor * decoder_prev_token = nullptr; // [1]
    ggml_tensor * decoder_time_emb = nullptr; // [dec_dim]
    ggml_tensor * decoder_step_position = nullptr; // [1]
    ggml_tensor * decoder_step_audio_emb = nullptr; // [dec_dim, 1]

    // Encoder KV cache: [enc_kv_heads*head_dim, enc_window, enc_layers]
    ggml_tensor * enc_kv_self_k   = nullptr;
    ggml_tensor * enc_kv_self_v   = nullptr;
    int32_t enc_kv_used           = 0;
    int32_t enc_kv_pos_base       = 0;

    // KV cache: [kv_heads*head_dim, dec_window, dec_layers]
    ggml_tensor * kv_self_k       = nullptr;
    ggml_tensor * kv_self_v       = nullptr;

    // Full accumulated encoder output (dynamic, allocated per utterance ON DEVICE)
    ggml_context       * ctx_enc_full = nullptr;
    ggml_backend_buffer_t buf_enc_full = nullptr;
    ggml_tensor        * encoder_output = nullptr;  // [enc_dim, total_enc_tokens]
    int32_t total_enc_tokens = 0;

    // Dynamic decoder memory (allocated per utterance ON DEVICE)
    ggml_context       * ctx_dec_mem = nullptr;
    ggml_backend_buffer_t buf_dec_mem = nullptr;
    ggml_tensor        * decoder_memory = nullptr;  // [dec_dim, dec_seq]
    int32_t dec_mem_capacity = 0;

    // Actual sizes (set per utterance)
    int32_t enc_seq_len  = 0;  // after conv, before left-trunc
    int32_t enc_seq_used = 0;  // after left-trunc (multiple of downsample_factor)
    int32_t dec_seq_len  = 0;  // adapter output length

    // KV ring buffer state
    int32_t kv_used      = 0;  // tokens currently in KV cache

    // Schedulers
    ggml_backend_sched_t sched_encoder  = nullptr;
    ggml_backend_sched_t sched_adapter  = nullptr;
    ggml_backend_sched_t sched_dec_pre  = nullptr;
    ggml_backend_sched_t sched_dec_step = nullptr;

    // CPU scratch
    std::vector<float> hann_window;     // [window_size]
    std::vector<float> mel_filters_cpu; // [n_freq * n_mel]
    std::vector<float> time_emb_cpu;    // [dec_dim]

    // CPU-side conv stem weights used by the incremental streaming frontend.
    std::vector<float> enc_conv0_weight_cpu;
    std::vector<float> enc_conv0_bias_cpu;
    std::vector<float> enc_conv1_weight_cpu;
    std::vector<float> enc_conv1_bias_cpu;

    // Cached saturated decoder-step graph for the long post-warmup streaming tail.
    ggml_context * ctx_dec_step_cached = nullptr;
    ggml_cgraph  * gf_dec_step_cached = nullptr;
    std::vector<uint8_t> dec_step_cached_meta;
    bool dec_step_cached_sched_ready = false;
};

static ggml_tensor * ctx_runtime_tensor(
    voxtral_context * ctx,
    ggml_tensor     * src) {
    if (ctx == nullptr || src == nullptr) {
        return src;
    }
    auto it = ctx->runtime_1d_tensors.find(src);
    return it != ctx->runtime_1d_tensors.end() ? it->second : src;
}

static void upload_decoder_time_embedding(voxtral_context * ctx) {
    if (ctx == nullptr || ctx->decoder_time_emb == nullptr || ctx->time_emb_cpu.empty()) {
        return;
    }
    ggml_backend_tensor_set(
        ctx->decoder_time_emb,
        ctx->time_emb_cpu.data(),
        0,
        ctx->time_emb_cpu.size() * sizeof(float));
}

// ============================================================================
// Mel filterbank computation (Slaney-style, matches Python reference)
// ============================================================================

static float hertz_to_mel(float freq_hz) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = 27.0f / logf(6.4f);
    float mels = 3.0f * freq_hz / 200.0f;
    if (freq_hz >= min_log_hertz) {
        mels = min_log_mel + logf(freq_hz / min_log_hertz) * logstep;
    }
    return mels;
}

static float mel_to_hertz(float mels) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = logf(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) {
        freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    }
    return freq;
}

static void compute_mel_filters_slaney(std::vector<float> & filters) {
    // Output: filters[k * n_mel + m] for k in [0..n_freq), m in [0..n_mel)
    // Matches Python compute_mel_filters() exactly
    constexpr int32_t n_freq = VOXTRAL_N_FREQ;  // 201
    constexpr int32_t n_mel  = VOXTRAL_NUM_MEL_BINS;  // 128

    filters.resize(n_freq * n_mel, 0.0f);

    // FFT frequencies: linspace(0, sr/2, n_freq)
    std::vector<float> fft_freqs(n_freq);
    for (int32_t i = 0; i < n_freq; i++) {
        fft_freqs[i] = (float)(VOXTRAL_SAMPLE_RATE / 2) * (float)i / (float)(n_freq - 1);
    }

    // Mel frequencies: linspace(mel(0), mel(8000), n_mel+2)
    const float mel_min = hertz_to_mel(0.0f);
    const float mel_max = hertz_to_mel(8000.0f);

    std::vector<float> mel_pts(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mel + 1);
    }

    std::vector<float> filter_freqs(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        filter_freqs[i] = mel_to_hertz(mel_pts[i]);
    }

    // Build triangular filters (matching Python slopes approach)
    for (int32_t m = 0; m < n_mel; m++) {
        const float f_left   = filter_freqs[m];
        const float f_center = filter_freqs[m + 1];
        const float f_right  = filter_freqs[m + 2];
        const float enorm    = 2.0f / (f_right - f_left);

        for (int32_t k = 0; k < n_freq; k++) {
            const float f = fft_freqs[k];
            float down_slope = -(f - f_center) / (f_center - f_left);   // -slopes[:, :-2] / filter_diff[:-1]
            float up_slope   =  (f_right - f)  / (f_right - f_center);  // slopes[:, 2:] / filter_diff[1:]

            float val = std::max(0.0f, std::min(down_slope, up_slope));
            filters[k * n_mel + m] = val * enorm;
        }
    }
}

// ============================================================================
// Time embedding (sinusoidal, matches Python compute_time_embedding)
// ============================================================================

static void compute_time_embedding(std::vector<float> & out, float t, int32_t dim) {
    // Python: inv_freq = exp(-log(10000) * arange(half) / half)
    //         emb = t * inv_freq;  return cat([cos(emb), sin(emb)])
    out.resize(dim);
    const int32_t half = dim / 2;
    for (int32_t i = 0; i < half; i++) {
        const float inv_freq = expf(-logf(10000.0f) * (float)i / (float)half);
        const float angle = t * inv_freq;
        out[i]        = cosf(angle);   // cos first half
        out[i + half] = sinf(angle);   // sin second half
    }
}

static double elapsed_ms(const std::chrono::steady_clock::time_point & t0) {
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool read_tensor_f32(
    ggml_tensor * tensor,
    std::vector<float> & out,
    std::string * error = nullptr) {

    if (tensor == nullptr) {
        if (error) {
            *error = "tensor is null";
        }
        return false;
    }

    const int64_t n = ggml_nelements(tensor);
    if (n < 0) {
        if (error) {
            *error = "tensor has invalid element count";
        }
        return false;
    }

    out.resize(static_cast<size_t>(n));
    const size_t nbytes = ggml_nbytes(tensor);

    switch (tensor->type) {
        case GGML_TYPE_F32:
            if (nbytes != out.size() * sizeof(float)) {
                if (error) {
                    *error = "unexpected F32 tensor byte size";
                }
                return false;
            }
            ggml_backend_tensor_get(tensor, out.data(), 0, nbytes);
            return true;

        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n));
            if (nbytes != tmp.size() * sizeof(ggml_fp16_t)) {
                if (error) {
                    *error = "unexpected F16 tensor byte size";
                }
                return false;
            }
            ggml_backend_tensor_get(tensor, tmp.data(), 0, nbytes);
            ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
            return true;
        }

        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp(static_cast<size_t>(n));
            if (nbytes != tmp.size() * sizeof(ggml_bf16_t)) {
                if (error) {
                    *error = "unexpected BF16 tensor byte size";
                }
                return false;
            }
            ggml_backend_tensor_get(tensor, tmp.data(), 0, nbytes);
            ggml_bf16_to_fp32_row(tmp.data(), out.data(), n);
            return true;
        }

        default:
            if (error) {
                *error = "unsupported tensor type for f32 staging";
            }
            return false;
    }
}

static bool decode_tensor_bytes_to_f32(
    enum ggml_type           type,
    const uint8_t          * bytes,
    size_t                   nbytes,
    int64_t                  n,
    std::vector<float>     & out) {
    out.resize((size_t) n);

    switch (type) {
        case GGML_TYPE_F32:
            if (nbytes != out.size() * sizeof(float)) {
                return false;
            }
            memcpy(out.data(), bytes, nbytes);
            return true;

        case GGML_TYPE_F16: {
            if (nbytes != out.size() * sizeof(ggml_fp16_t)) {
                return false;
            }
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) bytes, out.data(), n);
            return true;
        }

        case GGML_TYPE_BF16: {
            if (nbytes != out.size() * sizeof(ggml_bf16_t)) {
                return false;
            }
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) bytes, out.data(), n);
            return true;
        }

        default:
            out.clear();
            return false;
    }
}

static bool file_seek_absolute(FILE * fp, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

static void zero_tensor_bytes(ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(tensor), 0);
    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
}

// ============================================================================
// Token decode helpers (Tekken vocab from GGUF metadata)
// ============================================================================

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(26 + (c - 'a'));
        for (int c = '0'; c <= '9'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(52 + (c - '0'));
        t[static_cast<size_t>('+')] = 62;
        t[static_cast<size_t>('/')] = 63;
        return t;
    }();

    std::vector<uint8_t> out;
    out.reserve((in.size() * 3) / 4 + 4);

    uint32_t acc = 0;
    int bits = 0;

    for (char ch : in) {
        if (ch == '=') {
            break;
        }

        const uint8_t uch = static_cast<uint8_t>(ch);
        const int8_t val = table[uch];
        if (val < 0) {
            continue;
        }

        acc = (acc << 6) | static_cast<uint32_t>(val);
        bits += 6;

        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }

    return out;
}

static const std::string & token_bytes_for_id(const voxtral_model & model, int32_t token_id) {
    auto it_cached = model.tokenizer_bytes_cache.find(token_id);
    if (it_cached != model.tokenizer_bytes_cache.end()) {
        return it_cached->second;
    }

    std::string decoded;
    if (token_id >= 0 &&
        token_id >= model.tokenizer_num_special_tokens &&
        model.tokenizer_special_ranks.find(token_id) == model.tokenizer_special_ranks.end()) {
        const int64_t vocab_id = static_cast<int64_t>(token_id) -
                                 static_cast<int64_t>(model.tokenizer_num_special_tokens);
        if (vocab_id >= 0 && vocab_id < static_cast<int64_t>(model.tokenizer_vocab_b64.size())) {
            const std::vector<uint8_t> bytes =
                base64_decode(model.tokenizer_vocab_b64[static_cast<size_t>(vocab_id)]);
            decoded.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }
    }

    auto [it_new, _] = model.tokenizer_bytes_cache.emplace(token_id, std::move(decoded));
    return it_new->second;
}

static std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens) {
    if (model.tokenizer_vocab_b64.empty()) {
        return {};
    }

    std::string out;
    out.reserve(tokens.size() * 3);

    for (int32_t token : tokens) {
        if (token < model.tokenizer_num_special_tokens) {
            continue;
        }
        if (model.tokenizer_special_ranks.find(token) != model.tokenizer_special_ranks.end()) {
            continue;
        }

        const std::string & token_bytes = token_bytes_for_id(model, token);
        out.append(token_bytes);
    }

    return out;
}

// ============================================================================
// Reflect padding helper (matches PyTorch pad(mode="reflect"))
// ============================================================================

static inline int32_t reflect_index(int32_t idx, int32_t len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * len - 2 - idx;
        }
    }
    return idx;
}

// ============================================================================
// WAV file loading (16-bit PCM or 32-bit float, mono/stereo)
// ============================================================================

static bool load_wav_file(const std::string & path, std::vector<float> & audio_out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;

    // RIFF header
    char riff[4]; fin.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return false;

    uint32_t chunk_size; fin.read(reinterpret_cast<char*>(&chunk_size), 4);
    char wave[4]; fin.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) return false;

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    while (fin.good() && !(found_fmt && found_data)) {
        char sub_id[4]; fin.read(sub_id, 4);
        uint32_t sub_size; fin.read(reinterpret_cast<char*>(&sub_size), 4);
        if (!fin.good()) break;

        if (memcmp(sub_id, "fmt ", 4) == 0) {
            fin.read(reinterpret_cast<char*>(&audio_format),    2);
            fin.read(reinterpret_cast<char*>(&num_channels),    2);
            fin.read(reinterpret_cast<char*>(&sample_rate),     4);
            uint32_t byte_rate; fin.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align; fin.read(reinterpret_cast<char*>(&block_align), 2);
            fin.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (sub_size > 16) fin.seekg(sub_size - 16, std::ios::cur);
            found_fmt = true;
        } else if (memcmp(sub_id, "data", 4) == 0) {
            data_size = sub_size;
            found_data = true;
        } else {
            fin.seekg(sub_size, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) return false;
    if (audio_format != 1 && audio_format != 3) return false; // 1=PCM, 3=IEEE float

    const int32_t n_samples_total = data_size / (bits_per_sample / 8);
    const int32_t n_samples = n_samples_total / num_channels;

    if (audio_format == 1 && bits_per_sample == 16) {
        std::vector<int16_t> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += (float)raw[i * num_channels + c] / 32768.0f;
            }
            audio_out[i] = sum / num_channels;
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        std::vector<float> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c];
            }
            audio_out[i] = sum / num_channels;
        }
    } else {
        return false;
    }

    return true;
}

// ============================================================================
// Mel spectrogram computation (CPU, matches Python compute_mel_spectrogram)
// ============================================================================

struct stft_plan {
    int32_t n_fft = 0;
    int32_t n_bins = 0;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
};

static const stft_plan & get_stft_plan() {
    static stft_plan plan = []() {
        stft_plan p;
        p.n_fft  = VOXTRAL_N_FFT;
        p.n_bins = VOXTRAL_N_FREQ;
        p.cos_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        p.sin_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        for (int32_t k = 0; k < p.n_bins; ++k) {
            for (int32_t n = 0; n < p.n_fft; ++n) {
                const float angle = 2.0f * VOXTRAL_PI * (float) k * (float) n / (float) p.n_fft;
                const size_t idx = (size_t) k * (size_t) p.n_fft + (size_t) n;
                p.cos_table[idx] = cosf(angle);
                p.sin_table[idx] = sinf(angle);
            }
        }
        return p;
    }();
    return plan;
}

static void compute_mel_spectrogram(
    const float * audio,
    int32_t       n_samples,
    const float * mel_filters,   // [n_freq * n_mel]
    const float * hann_window,   // [window_size]
    float       * mel_out,       // [n_mel, n_frames]  (pre-allocated)
    int32_t     * out_n_frames)
{
    // torch.stft with window_size, hop_length, return_complex=True
    // produces (n_freq, n_stft_frames) where n_stft_frames = n_samples/hop + 1
    // Then magnitudes = stft[..., :-1].abs()**2  -> drops last frame
    const int32_t n_stft_frames = n_samples / VOXTRAL_HOP_LENGTH + 1;
    const int32_t n_frames = n_stft_frames - 1;  // drop last frame (matching Python [:-1])
    *out_n_frames = n_frames;

    constexpr int32_t n_freq = VOXTRAL_N_FREQ;
    constexpr int32_t n_mel  = VOXTRAL_NUM_MEL_BINS;
    constexpr int32_t n_fft  = VOXTRAL_N_FFT;
    constexpr int32_t hop    = VOXTRAL_HOP_LENGTH;
    constexpr int32_t pad    = n_fft / 2;

    if (n_frames <= 0) {
        return;
    }

    const stft_plan & plan = get_stft_plan();

    // Reflect padding once (equivalent to center=True, pad_mode="reflect")
    const int32_t centered_len = n_samples + 2 * pad;
    std::vector<float> centered((size_t) centered_len, 0.0f);
    if (n_samples > 0) {
        for (int32_t i = 0; i < centered_len; ++i) {
            const int32_t src = i - pad;
            const int32_t ridx = (src >= 0 && src < n_samples) ? src : reflect_index(src, n_samples);
            centered[(size_t) i] = audio[(size_t) ridx];
        }
    }

    // Pre-allocate per-call buffers
    std::vector<float> windowed((size_t) n_fft);
    std::vector<float> power((size_t) n_freq);
    std::vector<float> mel_accum((size_t) n_mel);

    for (int32_t frame = 0; frame < n_frames; ++frame) {
        const int32_t start = frame * hop;
        const float * frame_ptr = centered.data() + (size_t) start;

        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[(size_t) i] = frame_ptr[(size_t) i] * hann_window[(size_t) i];
        }

        // DFT with precomputed sin/cos tables
        for (int32_t k = 0; k < n_freq; ++k) {
            const float * cos_row = plan.cos_table.data() + (size_t) k * (size_t) n_fft;
            const float * sin_row = plan.sin_table.data() + (size_t) k * (size_t) n_fft;
            float re = 0.0f;
            float im = 0.0f;

            int32_t i = 0;
            for (; i + 3 < n_fft; i += 4) {
                const float x0 = windowed[(size_t) i + 0];
                const float x1 = windowed[(size_t) i + 1];
                const float x2 = windowed[(size_t) i + 2];
                const float x3 = windowed[(size_t) i + 3];

                re += x0 * cos_row[i + 0] + x1 * cos_row[i + 1] + x2 * cos_row[i + 2] + x3 * cos_row[i + 3];
                im -= x0 * sin_row[i + 0] + x1 * sin_row[i + 1] + x2 * sin_row[i + 2] + x3 * sin_row[i + 3];
            }
            for (; i < n_fft; ++i) {
                const float x = windowed[(size_t) i];
                re += x * cos_row[i];
                im -= x * sin_row[i];
            }

            power[(size_t) k] = re * re + im * im;
        }

        // Apply mel filterbank (k-major for cache-friendly access)
        std::fill(mel_accum.begin(), mel_accum.end(), 0.0f);
        for (int32_t k = 0; k < n_freq; ++k) {
            const float * w = mel_filters + (size_t) k * (size_t) n_mel;
            const float  pk = power[(size_t) k];
            for (int32_t m = 0; m < n_mel; ++m) {
                mel_accum[(size_t) m] += w[m] * pk;
            }
        }

        for (int32_t m = 0; m < n_mel; ++m) {
            float val = mel_accum[(size_t) m];
            val = std::max(val, 1e-10f);
            val = log10f(val);
            val = std::max(val, VOXTRAL_GLOBAL_LOG_MEL_MAX - 8.0f);
            val = (val + 4.0f) / 4.0f;
            mel_out[(size_t) m * (size_t) n_frames + (size_t) frame] = val;
        }
    }
}

// ============================================================================
// GGUF tensor loading helper
// ============================================================================

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "voxtral: tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

// ============================================================================
// Model loading
// ============================================================================

voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger,
    voxtral_gpu_backend    gpu)
{
    auto log_info = [&](const std::string & msg) {
        if (logger) logger(voxtral_log_level::info, msg);
    };

    const auto t_load_start = std::chrono::steady_clock::now();
    log_info("loading model from " + path);

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "voxtral: failed to open GGUF file: %s\n", path.c_str());
        return nullptr;
    }

    voxtral_model * model = new voxtral_model();
    model->gguf_ctx  = gguf_ctx;
    model->ctx_gguf  = ctx_meta;

    // Allocate a backend buffer for all the weights
    ggml_backend_t weights_backend = nullptr;
    voxtral_gpu_backend resolved_gpu = voxtral_gpu_backend::none;

    auto try_cuda = [&]() -> bool {
#ifdef GGML_USE_CUDA
        weights_backend = ggml_backend_cuda_init(0);
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::cuda; return true; }
        log_info("CUDA backend init failed");
#endif
        return false;
    };

    auto try_metal = [&]() -> bool {
#ifdef GGML_USE_METAL
        weights_backend = ggml_backend_metal_init();
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::metal; return true; }
        log_info("Metal backend init failed");
#endif
        return false;
    };

    auto try_vulkan = [&]() -> bool {
#ifdef GGML_USE_VULKAN
        weights_backend = ggml_backend_vk_init(0);
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::vulkan; return true; }
        log_info("Vulkan backend init failed");
#endif
        return false;
    };

    switch (gpu) {
        case voxtral_gpu_backend::cuda:
            if (!try_cuda()) {
                log_info("CUDA not available in this build, falling back to CPU");
            }
            break;
        case voxtral_gpu_backend::metal:
            if (!try_metal()) {
                log_info("Metal not available in this build, falling back to CPU");
            }
            break;
        case voxtral_gpu_backend::vulkan:
            if (!try_vulkan()) {
                log_info("Vulkan not available in this build, falling back to CPU");
            }
            break;
        case voxtral_gpu_backend::auto_detect:
            if (!try_cuda() && !try_metal() && !try_vulkan()) {
                log_info("no GPU backend available, using CPU");
            }
            break;
        case voxtral_gpu_backend::none:
        default:
            break;
    }

    if (!weights_backend) {
        weights_backend = ggml_backend_cpu_init();
    }

    model->backend_weights = weights_backend;
    model->weights_on_gpu = (resolved_gpu != voxtral_gpu_backend::none);
    model->gpu_type = resolved_gpu;
    model->buf_weights = ggml_backend_alloc_ctx_tensors(ctx_meta, weights_backend);

    if (!model->buf_weights) {
        fprintf(stderr, "voxtral: failed to allocate weight buffer\n");
        if (model->backend_weights) {
            ggml_backend_free(model->backend_weights);
            model->backend_weights = nullptr;
        }
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        delete model;
        return nullptr;
    }

    // Load tensor data from file into buffer
    {
        FILE * fp = fopen(path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "voxtral: failed to open file for reading weights\n");
            voxtral_model_free(model);
            return nullptr;
        }

        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) continue;

            const uint64_t offset = (uint64_t) gguf_get_data_offset(gguf_ctx) + (uint64_t) gguf_get_tensor_offset(gguf_ctx, i);
            const size_t nbytes = ggml_nbytes(t);

            std::vector<uint8_t> tmp(nbytes);
            if (!file_seek_absolute(fp, offset)) {
                fprintf(stderr, "voxtral: failed to seek tensor '%s' to offset %" PRIu64 "\n", name, offset);
                fclose(fp);
                voxtral_model_free(model);
                return nullptr;
            }
            if (fread(tmp.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "voxtral: failed to read tensor '%s'\n", name);
                fclose(fp);
                voxtral_model_free(model);
                return nullptr;
            }

            if (t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1) {
                std::vector<float> decoded;
                if (decode_tensor_bytes_to_f32(t->type, tmp.data(), nbytes, ggml_nelements(t), decoded)) {
                    model->cpu_1d_tensors.emplace(name, std::move(decoded));
                }
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    // Map weight tensors
    model->enc_conv0_weight = get_tensor(ctx_meta, "enc.conv0.weight");
    model->enc_conv0_bias   = get_tensor(ctx_meta, "enc.conv0.bias");
    model->enc_conv1_weight = get_tensor(ctx_meta, "enc.conv1.weight");
    model->enc_conv1_bias   = get_tensor(ctx_meta, "enc.conv1.bias");
    model->enc_norm_weight  = get_tensor(ctx_meta, "enc.norm.weight");

    model->enc_layers.resize(VOXTRAL_ENC_LAYERS);
    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; i++) {
        char nm[256];
        auto & L = model->enc_layers[i];
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.bias",i);      L.attn_q_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.bias",i);      L.attn_v_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.bias",i);      L.attn_o_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.bias",i);      L.ffn_w2_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
    }

    model->adapter_0_weight = get_tensor(ctx_meta, "adapter.0.weight");
    model->adapter_2_weight = get_tensor(ctx_meta, "adapter.2.weight");

    model->tok_embeddings_weight = get_tensor(ctx_meta, "tok_embeddings.weight");
    model->dec_norm_weight       = get_tensor(ctx_meta, "norm.weight");

    model->dec_layers.resize(VOXTRAL_DEC_LAYERS);
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        char nm[256];
        auto & L = model->dec_layers[i];
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ada0.weight",i);      L.ada0_weight   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ada2.weight",i);      L.ada2_weight   = get_tensor(ctx_meta,nm);
    }

    model->mel_filters = get_tensor(ctx_meta, "audio.mel_filters");

    // Tokenizer metadata (Tekken)
    {
        const int64_t key_num_special = gguf_find_key(gguf_ctx, "voxtral.tokenizer.num_special_tokens");
        if (key_num_special >= 0) {
            model->tokenizer_num_special_tokens = gguf_get_val_i32(gguf_ctx, key_num_special);
        }

        const int64_t key_special = gguf_find_key(gguf_ctx, "voxtral.tokenizer.special_token_ranks");
        if (key_special >= 0 && gguf_get_kv_type(gguf_ctx, key_special) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(gguf_ctx, key_special) == GGUF_TYPE_INT32) {
                const size_t n = gguf_get_arr_n(gguf_ctx, key_special);
                const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf_ctx, key_special);
                if (data) {
                    for (size_t i = 0; i < n; ++i) {
                        model->tokenizer_special_ranks.insert(data[i]);
                    }
                }
            }
        }

        const int64_t key_vocab = gguf_find_key(gguf_ctx, "voxtral.tokenizer.vocab_token_bytes_b64");
        if (key_vocab >= 0 && gguf_get_kv_type(gguf_ctx, key_vocab) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(gguf_ctx, key_vocab) == GGUF_TYPE_STRING) {
                const size_t n = gguf_get_arr_n(gguf_ctx, key_vocab);
                model->tokenizer_vocab_b64.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    const char * s = gguf_get_arr_str(gguf_ctx, key_vocab, i);
                    model->tokenizer_vocab_b64.emplace_back(s ? s : "");
                }
            }
        }
    }

    log_info("model loaded: enc_layers=" + std::to_string(VOXTRAL_ENC_LAYERS) +
             " dec_layers=" + std::to_string(VOXTRAL_DEC_LAYERS) +
             " vocab=" + std::to_string(VOXTRAL_VOCAB_SIZE));

    if (model->buf_weights) {
        const double sz_mb = (double) ggml_backend_buffer_get_size(model->buf_weights) / 1e6;
        log_info("model weights: " + std::to_string(sz_mb) + " MB");
    }
    log_info("encoder: dim=" + std::to_string(VOXTRAL_ENC_DIM) +
             " heads=" + std::to_string(VOXTRAL_ENC_HEADS) +
             " head_dim=" + std::to_string(VOXTRAL_ENC_HEAD_DIM) +
             " hidden=" + std::to_string(VOXTRAL_ENC_HIDDEN));
    log_info("decoder: dim=" + std::to_string(VOXTRAL_DEC_DIM) +
             " heads=" + std::to_string(VOXTRAL_DEC_HEADS) +
             " head_dim=" + std::to_string(VOXTRAL_DEC_HEAD_DIM) +
             " hidden=" + std::to_string(VOXTRAL_DEC_HIDDEN) +
             " kv_heads=" + std::to_string(VOXTRAL_DEC_KV_HEADS));

    {
        char buf[128];
        snprintf(buf, sizeof(buf), "model load time: %.2f ms", elapsed_ms(t_load_start));
        log_info(std::string(buf));
    }

    return model;
}

void voxtral_model_free(voxtral_model * model) {
    if (!model) return;
    if (model->buf_weights) ggml_backend_buffer_free(model->buf_weights);
    if (model->backend_weights) ggml_backend_free(model->backend_weights);
    if (model->ctx_gguf)    ggml_free(model->ctx_gguf);
    if (model->gguf_ctx)    gguf_free(model->gguf_ctx);
    delete model;
}

// ============================================================================
// Context initialization
// ============================================================================

voxtral_context * voxtral_init_from_model(
    voxtral_model              * model,
    const voxtral_context_params & params)
{
    voxtral_context * ctx = new voxtral_context();
    ctx->model     = model;
    ctx->log_level = params.log_level;
    ctx->logger    = params.logger;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Select GPU backend — inherit from model if params say none
    voxtral_gpu_backend gpu = params.gpu;
    if (gpu == voxtral_gpu_backend::none && model && model->weights_on_gpu) {
        gpu = model->gpu_type;
    }
    ctx->gpu_type = voxtral_gpu_backend::none;

    auto can_borrow_model_backend = [&]() -> bool {
        if (model == nullptr || model->backend_weights == nullptr) {
            return false;
        }
        if (gpu == voxtral_gpu_backend::none || gpu == voxtral_gpu_backend::auto_detect) {
            return true;
        }
        return gpu == model->gpu_type;
    };

    if (can_borrow_model_backend()) {
        ctx->backend = model->backend_weights;
        ctx->gpu_type = model->gpu_type;
        ctx->owns_backend = false;
        if (ctx->gpu_type == voxtral_gpu_backend::none) {
            LOG_INFO(ctx, "backend: CPU (borrowed from model)");
        } else {
            const char * gpu_name = "GPU";
            if (ctx->gpu_type == voxtral_gpu_backend::cuda)   gpu_name = "CUDA";
            if (ctx->gpu_type == voxtral_gpu_backend::metal)  gpu_name = "METAL";
            if (ctx->gpu_type == voxtral_gpu_backend::vulkan) gpu_name = "VULKAN";
            LOG_INFO(ctx, "backend: %s (borrowed from model, CPU fallback %d threads)", gpu_name, ctx->n_threads);
        }
    }

    auto try_cuda_ctx = [&]() -> bool {
#ifdef GGML_USE_CUDA
        ctx->backend = ggml_backend_cuda_init(0);
        if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::cuda; ctx->owns_backend = true; return true; }
        LOG_WARN(ctx, "CUDA backend init failed");
#endif
        return false;
    };
    auto try_metal_ctx = [&]() -> bool {
#ifdef GGML_USE_METAL
        ctx->backend = ggml_backend_metal_init();
        if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::metal; ctx->owns_backend = true; return true; }
        LOG_WARN(ctx, "Metal backend init failed");
#endif
        return false;
    };
    auto try_vulkan_ctx = [&]() -> bool {
#ifdef GGML_USE_VULKAN
        ctx->backend = ggml_backend_vk_init(0);
        if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::vulkan; ctx->owns_backend = true; return true; }
        LOG_WARN(ctx, "Vulkan backend init failed");
#endif
        return false;
    };

    if (!ctx->backend) {
        switch (gpu) {
            case voxtral_gpu_backend::cuda:    try_cuda_ctx();   break;
            case voxtral_gpu_backend::metal:   try_metal_ctx();  break;
            case voxtral_gpu_backend::vulkan:  try_vulkan_ctx(); break;
            case voxtral_gpu_backend::auto_detect:
                if (!try_cuda_ctx() && !try_metal_ctx() && !try_vulkan_ctx()) {
                    LOG_INFO(ctx, "no GPU backend available, using CPU");
                }
                break;
            case voxtral_gpu_backend::none:
            default:
                break;
        }
    }

    bool has_gpu = (ctx->gpu_type != voxtral_gpu_backend::none);

    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
        ctx->owns_backend = true;
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        LOG_INFO(ctx, "backend: CPU with %d threads", ctx->n_threads);
    } else if (ctx->owns_backend) {
        ctx->backend_cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
        const char * gpu_name = "GPU";
        if (ctx->gpu_type == voxtral_gpu_backend::cuda)   gpu_name = "CUDA";
        if (ctx->gpu_type == voxtral_gpu_backend::metal)  gpu_name = "METAL";
        if (ctx->gpu_type == voxtral_gpu_backend::vulkan) gpu_name = "VULKAN";
        LOG_INFO(ctx, "backend: %s (CPU fallback %d threads)", gpu_name, ctx->n_threads);
    } else if (has_gpu) {
        ctx->backend_cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }

    // Try to init BLAS backend for accelerated CPU matmuls
#ifdef GGML_USE_BLAS
    ctx->blas_backend = ggml_backend_blas_init();
    if (ctx->blas_backend) {
        ggml_backend_blas_set_n_threads(ctx->blas_backend, ctx->n_threads);
        LOG_INFO(ctx, "BLAS backend enabled with %d threads", ctx->n_threads);
    }
#endif

    // Allocate persistent tensors: encoder chunk output, decoder logits, KV caches
    {
        constexpr size_t n_base_tensors = 11;
        constexpr size_t n_runtime_1d_tensors =
            1 + VOXTRAL_ENC_LAYERS * 6 +
            1 + VOXTRAL_DEC_LAYERS * 2;
        constexpr size_t n_tensors = n_base_tensors + n_runtime_1d_tensors;
        ggml_init_params p = {
            /*.mem_size  =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc  =*/ true,
        };
        ctx->ctx_persistent = ggml_init(p);

        // encoder_chunk_input: [enc_dim, MAX_ENC_CHUNK] (device input staging for incremental encoder)
        ctx->encoder_chunk_input = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_ENC_DIM, VOXTRAL_MAX_ENC_CHUNK);
        ggml_set_name(ctx->encoder_chunk_input, "encoder_chunk_input");

        // encoder_chunk_output: [enc_dim, MAX_ENC_CHUNK] (reused per chunk)
        ctx->encoder_chunk_output = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_ENC_DIM, VOXTRAL_MAX_ENC_CHUNK);
        ggml_set_name(ctx->encoder_chunk_output, "encoder_chunk_output");

        // decoder_logits: [vocab_size]
        ctx->decoder_logits = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_F32,
            VOXTRAL_VOCAB_SIZE);
        ggml_set_name(ctx->decoder_logits, "decoder_logits");

        ctx->decoder_prev_token = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_I32, 1);
        ggml_set_name(ctx->decoder_prev_token, "decoder_prev_token");

        ctx->decoder_time_emb = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_DEC_DIM);
        ggml_set_name(ctx->decoder_time_emb, "decoder_time_emb");

        ctx->decoder_step_position = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_I32, 1);
        ggml_set_name(ctx->decoder_step_position, "decoder_step_position");

        ctx->decoder_step_audio_emb = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_DEC_DIM, 1);
        ggml_set_name(ctx->decoder_step_audio_emb, "decoder_step_audio_emb");

        // Encoder KV cache: [enc_kv_dim, enc_window, enc_layers]
        const int32_t enc_kv_dim = VOXTRAL_ENC_KV_HEADS * VOXTRAL_ENC_HEAD_DIM;  // 2048
        ctx->enc_kv_self_k = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32,
            enc_kv_dim, VOXTRAL_ENC_WINDOW, VOXTRAL_ENC_LAYERS);
        ggml_set_name(ctx->enc_kv_self_k, "enc_kv_self_k");

        ctx->enc_kv_self_v = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32,
            enc_kv_dim, VOXTRAL_ENC_WINDOW, VOXTRAL_ENC_LAYERS);
        ggml_set_name(ctx->enc_kv_self_v, "enc_kv_self_v");

        // KV cache: [kv_dim, dec_window, dec_layers]
        const int32_t kv_dim = VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM;  // 1024
        ctx->kv_self_k = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32,
            kv_dim, VOXTRAL_DEC_WINDOW, VOXTRAL_DEC_LAYERS);
        ggml_set_name(ctx->kv_self_k, "kv_self_k");

        ctx->kv_self_v = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F32,
            kv_dim, VOXTRAL_DEC_WINDOW, VOXTRAL_DEC_LAYERS);
        ggml_set_name(ctx->kv_self_v, "kv_self_v");

        auto create_runtime_1d = [&](ggml_tensor * src, const char * name) {
            if (src == nullptr) {
                return;
            }
            ggml_tensor * dst = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32, src->ne[0], 1);
            ggml_set_name(dst, name);
            ctx->runtime_1d_tensors[src] = dst;
        };

        create_runtime_1d(model->enc_norm_weight, "rt.enc.norm.weight");
        for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; ++i) {
            char nm[96];
            const auto & L = model->enc_layers[(size_t) i];
            snprintf(nm, sizeof(nm), "rt.enc.blk.%d.attn_norm.weight", i);
            create_runtime_1d(L.attn_norm_weight, nm);
            snprintf(nm, sizeof(nm), "rt.enc.blk.%d.attn_q.bias", i);
            create_runtime_1d(L.attn_q_bias, nm);
            snprintf(nm, sizeof(nm), "rt.enc.blk.%d.attn_v.bias", i);
            create_runtime_1d(L.attn_v_bias, nm);
            snprintf(nm, sizeof(nm), "rt.enc.blk.%d.attn_o.bias", i);
            create_runtime_1d(L.attn_o_bias, nm);
            snprintf(nm, sizeof(nm), "rt.enc.blk.%d.ffn_norm.weight", i);
            create_runtime_1d(L.ffn_norm_weight, nm);
            snprintf(nm, sizeof(nm), "rt.enc.blk.%d.ffn_w2.bias", i);
            create_runtime_1d(L.ffn_w2_bias, nm);
        }

        create_runtime_1d(model->dec_norm_weight, "rt.dec.norm.weight");
        for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; ++i) {
            char nm[96];
            const auto & L = model->dec_layers[(size_t) i];
            snprintf(nm, sizeof(nm), "rt.dec.blk.%d.attn_norm.weight", i);
            create_runtime_1d(L.attn_norm_weight, nm);
            snprintf(nm, sizeof(nm), "rt.dec.blk.%d.ffn_norm.weight", i);
            create_runtime_1d(L.ffn_norm_weight, nm);
        }

        ctx->buf_persistent = ggml_backend_alloc_ctx_tensors(ctx->ctx_persistent, ctx->backend);
        if (!ctx->buf_persistent) {
            fprintf(stderr, "voxtral: failed to allocate persistent buffer\n");
            voxtral_free(ctx);
            return nullptr;
        }

        // Zero persistent buffer (KV cache etc.)
        ggml_backend_buffer_clear(ctx->buf_persistent, 0);

        for (const auto & entry : ctx->runtime_1d_tensors) {
            const ggml_tensor * src = entry.first;
            ggml_tensor * dst = entry.second;
            const char * src_name = ggml_get_name(src);
            auto it = model->cpu_1d_tensors.find(src_name ? src_name : "");
            if (it == model->cpu_1d_tensors.end()) {
                LOG_ERR(ctx, "missing cached 1d tensor: %s", src_name ? src_name : "<unnamed>");
                voxtral_free(ctx);
                return nullptr;
            }
            if ((int64_t) it->second.size() != dst->ne[0]) {
                LOG_ERR(ctx, "cached 1d tensor size mismatch for %s: cached=%zu runtime=%lld",
                    src_name ? src_name : "<unnamed>", it->second.size(), (long long) dst->ne[0]);
                voxtral_free(ctx);
                return nullptr;
            }
            ggml_backend_tensor_set(dst, it->second.data(), 0, it->second.size() * sizeof(float));
        }

    }

    {
        const double chunk_mb = (double) ggml_nbytes(ctx->encoder_chunk_output) / 1e6;
        const double enc_kv_mb =
            (double) (ggml_nbytes(ctx->enc_kv_self_k) + ggml_nbytes(ctx->enc_kv_self_v)) / 1e6;
        const double dec_kv_mb =
            (double) (ggml_nbytes(ctx->kv_self_k) + ggml_nbytes(ctx->kv_self_v)) / 1e6;
        LOG_INFO(ctx, "buffers: encoder_chunk=%.2f MB enc_kv=%.2f MB dec_kv=%.2f MB",
            chunk_mb, enc_kv_mb, dec_kv_mb);
    }

    // Schedulers — ggml requires the last backend to be CPU.
    // With GPU:    [GPU, BLAS?, CPU]
    // Without GPU: [BLAS?, CPU]  (ctx->backend IS the CPU backend)
    ggml_backend_t backends[4];
    int n_backends = 0;
    if (has_gpu) {
        backends[n_backends++] = ctx->backend;           // GPU first
    }
    if (ctx->blas_backend) {
        backends[n_backends++] = ctx->blas_backend;      // BLAS before CPU
    }
    // CPU must be last
    ggml_backend_t cpu_be = has_gpu ? ctx->backend_cpu : ctx->backend;
    backends[n_backends++] = cpu_be;
    const bool op_offload = has_gpu;

    ctx->sched_encoder  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_adapter  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_dec_pre  = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);
    ctx->sched_dec_step = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, op_offload);

    // Hann window
    ctx->hann_window.resize(VOXTRAL_WINDOW_SIZE);
    for (int32_t i = 0; i < VOXTRAL_WINDOW_SIZE; i++) {
        // Match torch.hann_window(W, periodic=True)
        ctx->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * VOXTRAL_PI * (float)i / (float)(VOXTRAL_WINDOW_SIZE)));
    }

    // Mel filters (compute on CPU if not available from model, else load from GGUF)
    if (model->mel_filters) {
        constexpr int32_t n = VOXTRAL_N_FREQ * VOXTRAL_NUM_MEL_BINS;
        ctx->mel_filters_cpu.resize(n);
        ggml_backend_tensor_get(model->mel_filters, ctx->mel_filters_cpu.data(), 0, n * sizeof(float));
    } else {
        compute_mel_filters_slaney(ctx->mel_filters_cpu);
    }

    // Cache the small conv stem weights on CPU for the incremental frontend.
    {
        std::string tensor_error;
        if (!read_tensor_f32(model->enc_conv0_weight, ctx->enc_conv0_weight_cpu, &tensor_error) ||
            !read_tensor_f32(model->enc_conv0_bias, ctx->enc_conv0_bias_cpu, &tensor_error) ||
            !read_tensor_f32(model->enc_conv1_weight, ctx->enc_conv1_weight_cpu, &tensor_error) ||
            !read_tensor_f32(model->enc_conv1_bias, ctx->enc_conv1_bias_cpu, &tensor_error)) {
            LOG_ERR(ctx, "failed to stage conv stem weights: %s", tensor_error.c_str());
            voxtral_free(ctx);
            return nullptr;
        }
    }

    if (static_cast<int>(ctx->log_level) >= static_cast<int>(voxtral_log_level::debug)) {
        auto log_weight_head = [&](const char * label, ggml_tensor * tensor) {
            std::vector<float> values;
            if (!read_tensor_f32(tensor, values, nullptr) || values.empty()) {
                return;
            }
            const int32_t n = std::min<int32_t>(8, (int32_t) values.size());
            std::ostringstream oss;
            oss << "[";
            for (int32_t i = 0; i < n; ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << values[(size_t) i];
            }
            oss << "]";
            LOG_DBG(ctx, "%s head=%s", label, oss.str().c_str());
        };

        auto log_cached_head = [&](const char * label, const char * name) {
            auto it = model->cpu_1d_tensors.find(name);
            if (it == model->cpu_1d_tensors.end() || it->second.empty()) {
                return;
            }
            const int32_t n = std::min<int32_t>(8, (int32_t) it->second.size());
            std::ostringstream oss;
            oss << "[";
            for (int32_t i = 0; i < n; ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << it->second[(size_t) i];
            }
            oss << "]";
            LOG_DBG(ctx, "%s head=%s", label, oss.str().c_str());
        };

        log_cached_head("cached enc.norm.weight", "enc.norm.weight");
        log_weight_head("enc.norm.weight", ctx_runtime_tensor(ctx, model->enc_norm_weight));
        log_weight_head("enc.blk.0.attn_norm.weight", model->enc_layers.empty() ? nullptr : ctx_runtime_tensor(ctx, model->enc_layers[0].attn_norm_weight));
        log_weight_head("enc.blk.0.attn_q.weight", model->enc_layers.empty() ? nullptr : model->enc_layers[0].attn_q_weight);
    }

    // Time embedding for t = N_DELAY_TOKENS
    compute_time_embedding(ctx->time_emb_cpu, (float)VOXTRAL_N_DELAY_TOKENS, VOXTRAL_DEC_DIM);
    upload_decoder_time_embedding(ctx);

    LOG_INFO(ctx, "context initialized");
    return ctx;
}

void voxtral_free(voxtral_context * ctx) {
    if (!ctx) return;
    if (ctx->ctx_dec_step_cached) ggml_free(ctx->ctx_dec_step_cached);
    if (ctx->sched_encoder)  ggml_backend_sched_free(ctx->sched_encoder);
    if (ctx->sched_adapter)  ggml_backend_sched_free(ctx->sched_adapter);
    if (ctx->sched_dec_pre)  ggml_backend_sched_free(ctx->sched_dec_pre);
    if (ctx->sched_dec_step) ggml_backend_sched_free(ctx->sched_dec_step);
    if (ctx->buf_enc_full)   ggml_backend_buffer_free(ctx->buf_enc_full);
    if (ctx->ctx_enc_full)   ggml_free(ctx->ctx_enc_full);
    if (ctx->buf_dec_mem)    ggml_backend_buffer_free(ctx->buf_dec_mem);
    if (ctx->ctx_dec_mem)    ggml_free(ctx->ctx_dec_mem);
    if (ctx->buf_persistent) ggml_backend_buffer_free(ctx->buf_persistent);
    if (ctx->ctx_persistent) ggml_free(ctx->ctx_persistent);
    if (ctx->blas_backend)   ggml_backend_free(ctx->blas_backend);
    if (ctx->backend_cpu)    ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend && ctx->owns_backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ============================================================================
// KV cache helpers
// ============================================================================

static void clear_kv_cache(voxtral_context * ctx) {
    if (!ctx || !ctx->kv_self_k || !ctx->kv_self_v) {
        return;
    }
    zero_tensor_bytes(ctx->kv_self_k);
    zero_tensor_bytes(ctx->kv_self_v);
    if (ctx->decoder_prev_token != nullptr) {
        zero_tensor_bytes(ctx->decoder_prev_token);
    }
    ctx->kv_used = 0;
}

static void clear_encoder_kv_cache(voxtral_context * ctx) {
    if (!ctx || !ctx->enc_kv_self_k || !ctx->enc_kv_self_v) {
        return;
    }
    zero_tensor_bytes(ctx->enc_kv_self_k);
    zero_tensor_bytes(ctx->enc_kv_self_v);
    ctx->enc_kv_used = 0;
    ctx->enc_kv_pos_base = 0;
}

static bool shift_tensor_3d_window_left_device(
    ggml_backend_t backend,
    ggml_tensor  * tensor,
    int32_t        window,
    int32_t        n_layers,
    int32_t        shift);

static void kv_cache_shift_left(voxtral_context * ctx, int32_t shift) {
    if (!ctx || shift <= 0 || !ctx->kv_self_k || !ctx->kv_self_v) {
        return;
    }
    const int32_t window = VOXTRAL_DEC_WINDOW;
    if (shift >= window) {
        clear_kv_cache(ctx);
        return;
    }
    if (!shift_tensor_3d_window_left_device(ctx->backend, ctx->kv_self_k, window, VOXTRAL_DEC_LAYERS, shift) ||
        !shift_tensor_3d_window_left_device(ctx->backend, ctx->kv_self_v, window, VOXTRAL_DEC_LAYERS, shift)) {
        std::vector<uint8_t> k_bytes(ggml_nbytes(ctx->kv_self_k));
        std::vector<uint8_t> v_bytes(ggml_nbytes(ctx->kv_self_v));
        ggml_backend_tensor_get(ctx->kv_self_k, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_get(ctx->kv_self_v, v_bytes.data(), 0, v_bytes.size());

        const size_t row_bytes = ctx->kv_self_k->nb[1];
        const size_t layer_stride = ctx->kv_self_k->nb[2];

        for (int32_t l = 0; l < VOXTRAL_DEC_LAYERS; ++l) {
            uint8_t * k_base = k_bytes.data() + (size_t) l * layer_stride;
            uint8_t * v_base = v_bytes.data() + (size_t) l * layer_stride;

            memmove(k_base, k_base + (size_t) shift * row_bytes, (size_t) (window - shift) * row_bytes);
            memmove(v_base, v_base + (size_t) shift * row_bytes, (size_t) (window - shift) * row_bytes);

            memset(k_base + (size_t) (window - shift) * row_bytes, 0, (size_t) shift * row_bytes);
            memset(v_base + (size_t) (window - shift) * row_bytes, 0, (size_t) shift * row_bytes);
        }

        ggml_backend_tensor_set(ctx->kv_self_k, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_set(ctx->kv_self_v, v_bytes.data(), 0, v_bytes.size());
    }
}

struct voxtral_incremental_mel {
    std::vector<float> samples;
    int64_t sample_offset = 0;
    std::vector<float> mel; // [n_frames, n_mel] row-major
    int32_t mel_frame_offset = 0;
    bool finished = false;
};

static void incremental_mel_init(
    voxtral_incremental_mel & mel,
    int32_t left_pad_samples) {

    mel.samples.assign((size_t) (VOXTRAL_WINDOW_SIZE / 2 + left_pad_samples), 0.0f);
    mel.samples.reserve(std::max<size_t>(mel.samples.capacity(), (size_t) (VOXTRAL_SAMPLE_RATE * 4)));
    mel.sample_offset = 0;
    mel.mel.clear();
    mel.mel.reserve(std::max<size_t>(mel.mel.capacity(), (size_t) VOXTRAL_NUM_MEL_BINS * 4096u));
    mel.mel_frame_offset = 0;
    mel.finished = false;
}

static void incremental_mel_compact_samples(voxtral_incremental_mel & mel) {
    if (mel.samples.size() <= VOXTRAL_WINDOW_SIZE) {
        return;
    }
    const int64_t next_frame_global =
        (int64_t) mel.mel_frame_offset + (int64_t) (mel.mel.size() / VOXTRAL_NUM_MEL_BINS);
    const int64_t needed_from_global = next_frame_global * VOXTRAL_HOP_LENGTH;
    const int64_t discard64 = needed_from_global - mel.sample_offset;
    if (discard64 <= 0) {
        return;
    }
    const size_t discard = (size_t) std::min<int64_t>(discard64, (int64_t) mel.samples.size());
    if (discard < (size_t) VOXTRAL_SAMPLE_RATE && mel.samples.size() < 3 * VOXTRAL_SAMPLE_RATE) {
        return;
    }
    mel.samples.erase(mel.samples.begin(), mel.samples.begin() + static_cast<ptrdiff_t>(discard));
    mel.sample_offset += (int64_t) discard;
}

static int32_t incremental_mel_compute_available(
    const voxtral_context & ctx,
    voxtral_incremental_mel & mel) {

    std::array<float, VOXTRAL_N_FFT> windowed;
    std::array<float, VOXTRAL_N_FREQ> power;
    const stft_plan & plan = get_stft_plan();
    const int32_t local_frame = static_cast<int32_t>(mel.mel.size() / VOXTRAL_NUM_MEL_BINS);
    const int64_t global_frame0 = (int64_t) mel.mel_frame_offset + local_frame;
    const int64_t start640 = global_frame0 * VOXTRAL_HOP_LENGTH - mel.sample_offset;
    const int64_t last_start64 = (int64_t) mel.samples.size() - VOXTRAL_WINDOW_SIZE;

    if (start640 < 0 || last_start64 < start640) {
        return 0;
    }

    const int64_t frame_count64 = ((last_start64 - start640) / VOXTRAL_HOP_LENGTH) + 1;
    if (frame_count64 <= 0) {
        return 0;
    }

    const int32_t new_frames = static_cast<int32_t>(frame_count64);
    const size_t mel_base = mel.mel.size();
    mel.mel.resize(mel_base + (size_t) new_frames * (size_t) VOXTRAL_NUM_MEL_BINS);

    for (int32_t frame_idx = 0; frame_idx < new_frames; ++frame_idx) {
        const int64_t start64 = start640 + (int64_t) frame_idx * VOXTRAL_HOP_LENGTH;
        const size_t start = (size_t) start64;
        for (int32_t i = 0; i < VOXTRAL_N_FFT; ++i) {
            windowed[(size_t) i] = mel.samples[start + (size_t) i] * ctx.hann_window[(size_t) i];
        }

        for (int32_t k = 0; k < VOXTRAL_N_FREQ; ++k) {
            const float * cos_row = plan.cos_table.data() + (size_t) k * (size_t) VOXTRAL_N_FFT;
            const float * sin_row = plan.sin_table.data() + (size_t) k * (size_t) VOXTRAL_N_FFT;
            float re = 0.0f;
            float im = 0.0f;
            for (int32_t i = 0; i < VOXTRAL_N_FFT; ++i) {
                const float x = windowed[(size_t) i];
                re += x * cos_row[i];
                im -= x * sin_row[i];
            }
            power[(size_t) k] = re * re + im * im;
        }

        const size_t frame_base = mel_base + (size_t) frame_idx * (size_t) VOXTRAL_NUM_MEL_BINS;
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            float sum = 0.0f;
            for (int32_t k = 0; k < VOXTRAL_N_FREQ; ++k) {
                sum += ctx.mel_filters_cpu[(size_t) k * (size_t) VOXTRAL_NUM_MEL_BINS + (size_t) m] * power[(size_t) k];
            }
            sum = std::max(sum, 1e-10f);
            float val = log10f(sum);
            val = std::max(val, VOXTRAL_GLOBAL_LOG_MEL_MAX - 8.0f);
            mel.mel[frame_base + (size_t) m] = (val + 4.0f) / 4.0f;
        }
    }

    return new_frames;
}

static int32_t incremental_mel_feed(
    const voxtral_context & ctx,
    voxtral_incremental_mel & mel,
    const float * samples,
    int32_t n_samples) {

    if (samples == nullptr || n_samples <= 0) {
        return 0;
    }
    const size_t dst_offset = mel.samples.size();
    mel.samples.resize(dst_offset + (size_t) n_samples);
    std::memcpy(mel.samples.data() + dst_offset, samples, (size_t) n_samples * sizeof(float));
    const int32_t new_frames = incremental_mel_compute_available(ctx, mel);
    incremental_mel_compact_samples(mel);
    return new_frames;
}

static void incremental_mel_discard_before(
    voxtral_incremental_mel & mel,
    int32_t keep_from_frame) {

    if (keep_from_frame <= mel.mel_frame_offset) {
        return;
    }
    const int32_t total_frames = static_cast<int32_t>(mel.mel.size() / VOXTRAL_NUM_MEL_BINS);
    int32_t discard = keep_from_frame - mel.mel_frame_offset;
    discard = std::min(discard, total_frames);
    if (discard <= 0) {
        return;
    }

    const size_t discard_values = (size_t) discard * (size_t) VOXTRAL_NUM_MEL_BINS;
    mel.mel.erase(mel.mel.begin(), mel.mel.begin() + static_cast<ptrdiff_t>(discard_values));
    mel.mel_frame_offset += discard;
    incremental_mel_compact_samples(mel);
}

static void incremental_mel_finish(
    const voxtral_context & ctx,
    voxtral_incremental_mel & mel,
    int32_t right_pad_samples) {

    if (mel.finished) {
        return;
    }
    if (right_pad_samples > 0) {
        const size_t dst_offset = mel.samples.size();
        mel.samples.resize(dst_offset + (size_t) right_pad_samples);
        std::fill(
            mel.samples.begin() + static_cast<ptrdiff_t>(dst_offset),
            mel.samples.end(),
            0.0f);
    }
    const size_t real_end = mel.samples.size() - (size_t) std::max(0, right_pad_samples);
    const size_t reflect_count = (size_t) (VOXTRAL_WINDOW_SIZE / 2);
    const size_t reflect_offset = mel.samples.size();
    mel.samples.resize(reflect_offset + reflect_count);
    for (size_t i = 0; i < reflect_count; ++i) {
        const int64_t src = (int64_t) real_end - 2 - (int64_t) i;
        mel.samples[reflect_offset + i] = src >= 0 ? mel.samples[(size_t) src] : 0.0f;
    }
    (void) incremental_mel_compute_available(ctx, mel);
    const size_t total_frames = mel.mel.size() / VOXTRAL_NUM_MEL_BINS;
    if (total_frames > 0) {
        mel.mel.resize((total_frames - 1) * (size_t) VOXTRAL_NUM_MEL_BINS);
    }
    mel.finished = true;
}

static void gelu_erf_inplace(float * data, size_t n) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    for (size_t i = 0; i < n; ++i) {
        const float x = data[i];
        data[i] = 0.5f * x * (1.0f + erff(x * kInvSqrt2));
    }
}

static void cpu_causal_conv1d_col_major(
    float * out,                  // [out_channels, out_len] col-major
    const float * in,             // [in_channels, in_len] col-major
    const float * weight,         // [kernel, in_channels, out_channels]
    const float * bias,           // [out_channels]
    int32_t in_channels,
    int32_t out_channels,
    int32_t in_len,
    int32_t kernel_size,
    int32_t stride) {

    const int32_t padding_total = kernel_size - stride;
    const float n_frames =
        (static_cast<float>(in_len - kernel_size + padding_total) / static_cast<float>(stride)) + 1.0f;
    const int32_t target_length =
        (static_cast<int32_t>(std::ceil(n_frames)) - 1) * stride + (kernel_size - padding_total);
    const int32_t extra_padding = target_length - in_len;
    const int32_t pad_left = padding_total;
    const int32_t pad_right = std::max<int32_t>(0, extra_padding);
    const int32_t padded_len = in_len + pad_left + pad_right;
    const int32_t out_len = (padded_len - kernel_size) / stride + 1;

    for (int32_t oc = 0; oc < out_channels; ++oc) {
        for (int32_t out_idx = 0; out_idx < out_len; ++out_idx) {
            float sum = bias != nullptr ? bias[oc] : 0.0f;
            const int32_t origin = out_idx * stride - pad_left;
            for (int32_t ic = 0; ic < in_channels; ++ic) {
                const float * in_row = in + (size_t) ic * (size_t) in_len;
                for (int32_t k = 0; k < kernel_size; ++k) {
                    const int32_t src_idx = origin + k;
                    if (src_idx < 0 || src_idx >= in_len) {
                        continue;
                    }
                    const size_t widx =
                        (size_t) k +
                        (size_t) kernel_size * ((size_t) ic + (size_t) in_channels * (size_t) oc);
                    sum += in_row[src_idx] * weight[widx];
                }
            }
            out[(size_t) oc * (size_t) out_len + (size_t) out_idx] = sum;
        }
    }
}

static void encoder_kv_cache_shift_left(voxtral_context * ctx, int32_t shift) {
    if (!ctx || shift <= 0 || !ctx->enc_kv_self_k || !ctx->enc_kv_self_v) {
        return;
    }
    const int32_t window = VOXTRAL_ENC_WINDOW;
    if (shift >= window) {
        clear_encoder_kv_cache(ctx);
        return;
    }
    if (!shift_tensor_3d_window_left_device(ctx->backend, ctx->enc_kv_self_k, window, VOXTRAL_ENC_LAYERS, shift) ||
        !shift_tensor_3d_window_left_device(ctx->backend, ctx->enc_kv_self_v, window, VOXTRAL_ENC_LAYERS, shift)) {
        std::vector<uint8_t> k_bytes(ggml_nbytes(ctx->enc_kv_self_k));
        std::vector<uint8_t> v_bytes(ggml_nbytes(ctx->enc_kv_self_v));
        ggml_backend_tensor_get(ctx->enc_kv_self_k, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_get(ctx->enc_kv_self_v, v_bytes.data(), 0, v_bytes.size());

        const size_t row_bytes = ctx->enc_kv_self_k->nb[1];
        const size_t layer_stride = ctx->enc_kv_self_k->nb[2];

        for (int32_t l = 0; l < VOXTRAL_ENC_LAYERS; ++l) {
            uint8_t * k_base = k_bytes.data() + (size_t) l * layer_stride;
            uint8_t * v_base = v_bytes.data() + (size_t) l * layer_stride;

            memmove(k_base, k_base + (size_t) shift * row_bytes, (size_t) (window - shift) * row_bytes);
            memmove(v_base, v_base + (size_t) shift * row_bytes, (size_t) (window - shift) * row_bytes);

            memset(k_base + (size_t) (window - shift) * row_bytes, 0, (size_t) shift * row_bytes);
            memset(v_base + (size_t) (window - shift) * row_bytes, 0, (size_t) shift * row_bytes);
        }

        ggml_backend_tensor_set(ctx->enc_kv_self_k, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_set(ctx->enc_kv_self_v, v_bytes.data(), 0, v_bytes.size());
    }
    ctx->enc_kv_pos_base += shift;
}

// ============================================================================
// Graph Building: Encoder
// ============================================================================


struct causal_conv1d_dims {
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int32_t padded_len = 0;
    int32_t out_len = 0;
};

causal_conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t kernel_size, int32_t stride) {
    causal_conv1d_dims out{};
    if (in_len <= 0 || kernel_size <= 0 || stride <= 0) {
        return out;
    }

    const int32_t padding_total = kernel_size - stride;
    const float n_frames = (static_cast<float>(in_len - kernel_size + padding_total) / static_cast<float>(stride)) + 1.0f;
    const int32_t target_length =
        (static_cast<int32_t>(std::ceil(n_frames)) - 1) * stride + (kernel_size - padding_total);
    const int32_t extra_padding = target_length - in_len;

    out.pad_left = padding_total;
    out.pad_right = std::max<int32_t>(0, extra_padding);
    out.padded_len = in_len + out.pad_left + out.pad_right;
    out.out_len = (out.padded_len - kernel_size) / stride + 1;
    return out;
}

// Compute the number of encoder tokens from mel frames (accounting for conv and truncation)
static int32_t mel_frames_to_enc_tokens(int32_t n_frames) {
    auto d0 = compute_causal_conv1d_dims(n_frames, 3, 1);  // conv0
    auto d1 = compute_causal_conv1d_dims(d0.out_len, 3, 2); // conv1 (stride 2)
    int32_t trunc = d1.out_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    return d1.out_len - trunc;
}

// Pre-compute total encoder tokens for a given mel frame count (for buffer allocation)
static int32_t compute_total_enc_tokens(int32_t total_mel_frames) {
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - VOXTRAL_ENC_CHUNK_OVERLAP * 2;
    int32_t total = 0;
    int32_t mel_offset = 0;
    bool first = true;

    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);
        int32_t chunk_tokens = mel_frames_to_enc_tokens(chunk_mel);
        int32_t skip = first ? 0 : VOXTRAL_ENC_CHUNK_OVERLAP;
        int32_t stride = chunk_tokens - skip;
        if (stride <= 0) break;
        total += stride;
        mel_offset += mel_stride;
        first = false;
    }
    return total;
}

// Allocate per-utterance encoder output buffer on device
static bool alloc_encoder_output(voxtral_context * ctx, int32_t n_tokens) {
    // Free previous allocation
    if (ctx->buf_enc_full) { ggml_backend_buffer_free(ctx->buf_enc_full); ctx->buf_enc_full = nullptr; }
    if (ctx->ctx_enc_full) { ggml_free(ctx->ctx_enc_full); ctx->ctx_enc_full = nullptr; }
    ctx->encoder_output = nullptr;

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ctx->ctx_enc_full = ggml_init(p);
    ctx->encoder_output = ggml_new_tensor_2d(ctx->ctx_enc_full, GGML_TYPE_F32,
        VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(ctx->encoder_output, "encoder_output");
    ctx->buf_enc_full = ggml_backend_alloc_ctx_tensors(ctx->ctx_enc_full, ctx->backend);
    if (!ctx->buf_enc_full) return false;

    ctx->total_enc_tokens = n_tokens;
    return true;
}

// Allocate per-utterance decoder memory buffer on device
static bool create_decoder_memory_allocation(
    voxtral_context      * ctx,
    int32_t                dec_seq,
    ggml_context       ** out_ctx,
    ggml_backend_buffer_t * out_buf,
    ggml_tensor        ** out_tensor) {

    if (out_ctx == nullptr || out_buf == nullptr || out_tensor == nullptr) {
        return false;
    }

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * dec_ctx = ggml_init(p);
    if (dec_ctx == nullptr) {
        return false;
    }

    ggml_tensor * decoder_memory = ggml_new_tensor_2d(dec_ctx, GGML_TYPE_F32,
        VOXTRAL_DEC_DIM, dec_seq);
    ggml_set_name(decoder_memory, "decoder_memory");
    ggml_backend_buffer_t dec_buf = ggml_backend_alloc_ctx_tensors(dec_ctx, ctx->backend);
    if (!dec_buf) {
        ggml_free(dec_ctx);
        return false;
    }

    *out_ctx = dec_ctx;
    *out_buf = dec_buf;
    *out_tensor = decoder_memory;
    return true;
}

static bool copy_tensor_2d_prefix(
    ggml_tensor * src,
    ggml_tensor * dst,
    int64_t       ne0,
    int64_t       ne1) {

    if (src == nullptr || dst == nullptr || ne0 <= 0 || ne1 <= 0) {
        return true;
    }

    ggml_init_params view_params = {
        /*.mem_size  =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * view_ctx = ggml_init(view_params);
    if (view_ctx == nullptr) {
        return false;
    }

    ggml_tensor * src_view = ggml_view_2d(view_ctx, src, ne0, ne1, src->nb[1], 0);
    ggml_tensor * dst_view = ggml_view_2d(view_ctx, dst, ne0, ne1, dst->nb[1], 0);
    const enum ggml_status src_status = ggml_backend_view_init(src_view);
    const enum ggml_status dst_status = ggml_backend_view_init(dst_view);
    if (src_status != GGML_STATUS_SUCCESS || dst_status != GGML_STATUS_SUCCESS) {
        ggml_free(view_ctx);
        return false;
    }

    ggml_backend_tensor_copy(src_view, dst_view);
    ggml_free(view_ctx);
    return true;
}

static bool copy_tensor_2d_region(
    ggml_tensor * src,
    ggml_tensor * dst,
    int64_t       ne0,
    int64_t       ne1,
    size_t        src_offset,
    size_t        dst_offset) {

    if (src == nullptr || dst == nullptr || ne0 <= 0 || ne1 <= 0) {
        return true;
    }

    ggml_init_params view_params = {
        /*.mem_size  =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * view_ctx = ggml_init(view_params);
    if (view_ctx == nullptr) {
        return false;
    }

    ggml_tensor * src_view = ggml_view_2d(view_ctx, src, ne0, ne1, src->nb[1], src_offset);
    ggml_tensor * dst_view = ggml_view_2d(view_ctx, dst, ne0, ne1, dst->nb[1], dst_offset);
    const enum ggml_status src_status = ggml_backend_view_init(src_view);
    const enum ggml_status dst_status = ggml_backend_view_init(dst_view);
    if (src_status != GGML_STATUS_SUCCESS || dst_status != GGML_STATUS_SUCCESS) {
        ggml_free(view_ctx);
        return false;
    }

    ggml_backend_tensor_copy(src_view, dst_view);
    ggml_free(view_ctx);
    return true;
}

static bool shift_tensor_3d_window_left_device(
    ggml_backend_t backend,
    ggml_tensor  * tensor,
    int32_t        window,
    int32_t        n_layers,
    int32_t        shift) {

    if (backend == nullptr || tensor == nullptr || shift <= 0) {
        return true;
    }
    if (shift >= window) {
        zero_tensor_bytes(tensor);
        return true;
    }

    const int32_t keep = window - shift;
    const int64_t ne0 = tensor->ne[0];
    const size_t row_bytes = tensor->nb[1];
    const size_t layer_stride = tensor->nb[2];

    ggml_init_params tmp_params = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * tmp_ctx = ggml_init(tmp_params);
    if (tmp_ctx == nullptr) {
        return false;
    }

    ggml_tensor * tmp = ggml_new_tensor_2d(tmp_ctx, tensor->type, ne0, keep);
    ggml_backend_buffer_t tmp_buf = ggml_backend_alloc_ctx_tensors(tmp_ctx, backend);
    if (tmp_buf == nullptr) {
        ggml_free(tmp_ctx);
        return false;
    }

    std::vector<uint8_t> zeros((size_t) shift * row_bytes, 0);
    for (int32_t layer = 0; layer < n_layers; ++layer) {
        const size_t layer_offset = (size_t) layer * layer_stride;
        const size_t src_offset = layer_offset + (size_t) shift * row_bytes;
        const size_t dst_offset = layer_offset;
        if (!copy_tensor_2d_region(tensor, tmp, ne0, keep, src_offset, 0) ||
            !copy_tensor_2d_region(tmp, tensor, ne0, keep, 0, dst_offset)) {
            ggml_backend_buffer_free(tmp_buf);
            ggml_free(tmp_ctx);
            return false;
        }
        ggml_backend_tensor_set(
            tensor,
            zeros.data(),
            layer_offset + (size_t) keep * row_bytes,
            zeros.size());
    }

    ggml_backend_buffer_free(tmp_buf);
    ggml_free(tmp_ctx);
    return true;
}

static bool read_tensor_2d_columns_f32(
    ggml_tensor          * tensor,
    int64_t                col_begin,
    int64_t                col_count,
    std::vector<float>   & out,
    std::string          * error = nullptr) {

    if (tensor == nullptr) {
        if (error) {
            *error = "tensor is null";
        }
        return false;
    }
    if (col_begin < 0 || col_count < 0 || col_begin + col_count > tensor->ne[1]) {
        if (error) {
            *error = "invalid column slice";
        }
        return false;
    }
    if (col_count == 0) {
        out.clear();
        return true;
    }
    if (tensor->type != GGML_TYPE_F32) {
        if (error) {
            *error = "read_tensor_2d_columns_f32 requires F32 tensor";
        }
        return false;
    }

    out.resize((size_t) tensor->ne[0] * (size_t) col_count);
    const size_t byte_offset = (size_t) col_begin * tensor->nb[1];
    const size_t byte_count = (size_t) tensor->ne[0] * (size_t) col_count * sizeof(float);
    ggml_backend_tensor_get(tensor, out.data(), byte_offset, byte_count);
    return true;
}

static bool alloc_decoder_memory(voxtral_context * ctx, int32_t dec_seq) {
    ggml_context * new_ctx = nullptr;
    ggml_backend_buffer_t new_buf = nullptr;
    ggml_tensor * new_tensor = nullptr;
    if (!create_decoder_memory_allocation(ctx, dec_seq, &new_ctx, &new_buf, &new_tensor)) {
        return false;
    }

    if (ctx->buf_dec_mem) { ggml_backend_buffer_free(ctx->buf_dec_mem); }
    if (ctx->ctx_dec_mem) { ggml_free(ctx->ctx_dec_mem); }

    ctx->ctx_dec_mem = new_ctx;
    ctx->buf_dec_mem = new_buf;
    ctx->decoder_memory = new_tensor;
    ctx->dec_seq_len = dec_seq;
    ctx->dec_mem_capacity = dec_seq;
    return true;
}

static bool ensure_decoder_memory_capacity(voxtral_context * ctx, int32_t required_tokens) {
    if (required_tokens <= 0) {
        return true;
    }
    if (ctx->decoder_memory != nullptr && ctx->dec_mem_capacity >= required_tokens) {
        ctx->dec_seq_len = std::max(ctx->dec_seq_len, required_tokens);
        return true;
    }

    const int32_t old_capacity = ctx->dec_mem_capacity;
    const int32_t new_capacity = std::max<int32_t>(
        required_tokens,
        old_capacity > 0 ? old_capacity * 2 : 256);

    ggml_context * old_ctx = ctx->ctx_dec_mem;
    ggml_backend_buffer_t old_buf = ctx->buf_dec_mem;
    ggml_tensor * old_tensor = ctx->decoder_memory;
    const int32_t old_seq_len = ctx->dec_seq_len;

    ggml_context * new_ctx = nullptr;
    ggml_backend_buffer_t new_buf = nullptr;
    ggml_tensor * new_tensor = nullptr;
    if (!create_decoder_memory_allocation(ctx, new_capacity, &new_ctx, &new_buf, &new_tensor)) {
        return false;
    }

    if (old_tensor != nullptr && old_seq_len > 0) {
        if (!copy_tensor_2d_prefix(old_tensor, new_tensor, VOXTRAL_DEC_DIM, old_seq_len)) {
            ggml_backend_buffer_free(new_buf);
            ggml_free(new_ctx);
            return false;
        }
    }

    if (old_buf != nullptr) {
        ggml_backend_buffer_free(old_buf);
    }
    if (old_ctx != nullptr) {
        ggml_free(old_ctx);
    }

    ctx->ctx_dec_mem = new_ctx;
    ctx->buf_dec_mem = new_buf;
    ctx->decoder_memory = new_tensor;
    ctx->dec_mem_capacity = new_capacity;
    ctx->dec_seq_len = std::max<int32_t>(old_seq_len, required_tokens);
    return true;
}

ggml_tensor * causal_conv1d_graph(
    ggml_context * ctx0,
    ggml_tensor * x,
    int32_t in_len,
    ggml_tensor * weight,
    ggml_tensor * bias,
    int32_t out_channels,
    int32_t kernel_size,
    int32_t stride,
    int32_t & out_len) {
    out_len = 0;
    if (ctx0 == nullptr || x == nullptr || weight == nullptr || kernel_size <= 0 || stride <= 0) {
        return nullptr;
    }
    if (in_len <= 0 || out_channels <= 0) {
        return nullptr;
    }

    const auto dims = compute_causal_conv1d_dims(in_len, kernel_size, stride);
    if (dims.out_len <= 0) {
        return nullptr;
    }

    ggml_tensor * x_pad = ggml_pad_ext(ctx0, x, dims.pad_left, dims.pad_right, 0, 0, 0, 0, 0, 0);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = ggml_conv_1d(ctx0, weight, x_pad, stride, 0, 1);
    if (y == nullptr) {
        return nullptr;
    }

    if (bias != nullptr) {
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, bias, 1, out_channels, 1));
    }

    out_len = dims.out_len;
    return y;
}


void print_tensor_info(struct ggml_tensor * tensor) {
    printf("Tensor name: %s\n", tensor->name);
    printf("Tensor type: %s\n", ggml_type_name(tensor->type));
    printf("Number of dimensions: %d\n", ggml_n_dims(tensor));
    printf("Total elements: %" PRId64 "\n", ggml_nelements(tensor));
    printf("Shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
}

static void log_tensor_info(voxtral_context * ctx, const char * tag, struct ggml_tensor * t) {
    if (t == nullptr) {
        LOG_DBG(ctx, "%s: <null>", tag);
        return;
    }
    LOG_DBG(ctx, "%s: type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu] n_dims=%d nbytes=%zu",
        tag,
        ggml_type_name(t->type),
        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3],
        ggml_n_dims(t),
        (size_t) ggml_nbytes(t));
}

static void log_graph_info(voxtral_context * ctx, const char * name, struct ggml_cgraph * gf) {
    if (gf == nullptr) {
        return;
    }
    const int size  = ggml_graph_size(gf);
    const int nodes = ggml_graph_n_nodes(gf);
    LOG_DBG(ctx, "%s graph: size=%d nodes=%d", name, size, nodes);
}

static std::string format_float_slice(const float * data, int32_t n) {
    std::ostringstream oss;
    oss << "[";
    for (int32_t i = 0; i < n; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << data[i];
    }
    oss << "]";
    return oss.str();
}

static void log_row_major_sample(
    voxtral_context * ctx,
    const char * label,
    const float * data,
    int32_t rows,
    int32_t cols,
    int32_t stats_rows) {

    if (ctx == nullptr || data == nullptr || rows <= 0 || cols <= 0) {
        return;
    }

    constexpr int32_t sample_dims = 8;
    const int32_t dims = std::min<int32_t>(sample_dims, cols);
    const int32_t sample_rows = std::max<int32_t>(1, std::min<int32_t>(stats_rows, rows));
    const float * row0 = data;
    const float * row_last = data + (size_t) (sample_rows - 1) * (size_t) cols;

    double sum = 0.0;
    double sq_sum = 0.0;
    const size_t total = (size_t) sample_rows * (size_t) cols;
    for (size_t i = 0; i < total; ++i) {
        const double v = data[i];
        sum += v;
        sq_sum += v * v;
    }
    const double n = (double) total;
    const double mean = n > 0.0 ? sum / n : 0.0;
    const double variance = n > 0.0 ? std::max(0.0, (sq_sum / n) - (mean * mean)) : 0.0;
    const double stddev = std::sqrt(variance);

    LOG_INFO(
        ctx,
        "%s: rows=%d cols=%d row0_first%d=%s row%d_first%d=%s mean=%.6f std=%.6f",
        label,
        sample_rows,
        cols,
        dims,
        format_float_slice(row0, dims).c_str(),
        sample_rows - 1,
        dims,
        format_float_slice(row_last, dims).c_str(),
        mean,
        stddev);
}

static void log_col_major_tensor_sample(
    voxtral_context * ctx,
    const char * label,
    ggml_tensor * tensor,
    int32_t rows,
    int32_t cols,
    int32_t stats_rows) {

    if (ctx == nullptr ||
        static_cast<int>(ctx->log_level) < static_cast<int>(voxtral_log_level::debug) ||
        tensor == nullptr ||
        rows <= 0 ||
        cols <= 0) {
        return;
    }

    constexpr int32_t sample_dims = 8;
    const int32_t dims = std::min<int32_t>(sample_dims, cols);
    const int32_t sample_rows = std::max<int32_t>(1, std::min<int32_t>(stats_rows, rows));
    std::vector<float> row0((size_t) dims, 0.0f);
    std::vector<float> row_last((size_t) dims, 0.0f);
    std::vector<float> stats((size_t) sample_rows * (size_t) cols, 0.0f);

    for (int32_t d = 0; d < dims; ++d) {
        ggml_backend_tensor_get(
            tensor,
            &row0[(size_t) d],
            (size_t) d * tensor->nb[0],
            sizeof(float));
        ggml_backend_tensor_get(
            tensor,
            &row_last[(size_t) d],
            (size_t) d * tensor->nb[0] + (size_t) (sample_rows - 1) * tensor->nb[1],
            sizeof(float));
    }

    for (int32_t row = 0; row < sample_rows; ++row) {
        ggml_backend_tensor_get(
            tensor,
            stats.data() + (size_t) row * (size_t) cols,
            (size_t) row * tensor->nb[1],
            (size_t) cols * sizeof(float));
    }

    double sum = 0.0;
    double sq_sum = 0.0;
    for (float v : stats) {
        sum += v;
        sq_sum += (double) v * (double) v;
    }
    const double n = (double) stats.size();
    const double mean = n > 0.0 ? sum / n : 0.0;
    const double variance = n > 0.0 ? std::max(0.0, (sq_sum / n) - (mean * mean)) : 0.0;
    const double stddev = std::sqrt(variance);

    LOG_INFO(
        ctx,
        "%s: rows=%d cols=%d row0_first%d=%s row%d_first%d=%s mean=%.6f std=%.6f",
        label,
        sample_rows,
        cols,
        dims,
        format_float_slice(row0.data(), dims).c_str(),
        sample_rows - 1,
        dims,
        format_float_slice(row_last.data(), dims).c_str(),
        mean,
        stddev);
}

static void log_decoder_memory_window_sample(
    voxtral_context * ctx,
    const char * label,
    int32_t stats_tokens) {

    if (ctx == nullptr ||
        static_cast<int>(ctx->log_level) < static_cast<int>(voxtral_log_level::debug) ||
        ctx->decoder_memory == nullptr ||
        stats_tokens <= 0) {
        return;
    }

    const int32_t token_count = std::min<int32_t>(stats_tokens, ctx->dec_seq_len);
    if (token_count <= 0) {
        return;
    }

    constexpr int32_t sample_dims = 8;
    const int32_t dims = std::min<int32_t>(sample_dims, VOXTRAL_DEC_DIM);
    std::vector<float> token0((size_t) dims, 0.0f);
    std::vector<float> token_last((size_t) dims, 0.0f);
    std::vector<float> stats((size_t) token_count * (size_t) VOXTRAL_DEC_DIM, 0.0f);

    ggml_backend_tensor_get(
        ctx->decoder_memory,
        token0.data(),
        0,
        (size_t) dims * sizeof(float));
    ggml_backend_tensor_get(
        ctx->decoder_memory,
        token_last.data(),
        (size_t) (token_count - 1) * ctx->decoder_memory->nb[1],
        (size_t) dims * sizeof(float));
    ggml_backend_tensor_get(
        ctx->decoder_memory,
        stats.data(),
        0,
        stats.size() * sizeof(float));

    double sum = 0.0;
    double sq_sum = 0.0;
    for (float v : stats) {
        sum += v;
        sq_sum += (double) v * (double) v;
    }
    const double n = (double) stats.size();
    const double mean = n > 0.0 ? sum / n : 0.0;
    const double variance = n > 0.0 ? std::max(0.0, (sq_sum / n) - (mean * mean)) : 0.0;
    const double stddev = std::sqrt(variance);

    LOG_INFO(
        ctx,
        "%s: tokens=%d token0_first%d=%s token%d_first%d=%s mean=%.6f std=%.6f",
        label,
        token_count,
        dims,
        format_float_slice(token0.data(), dims).c_str(),
        token_count - 1,
        dims,
        format_float_slice(token_last.data(), dims).c_str(),
        mean,
        stddev);
}

static ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name);

// Build encoder graph that writes output into ctx->encoder_chunk_output
static ggml_cgraph * build_encoder_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    const float * mel_data,   // [n_mel, n_frames] on CPU
    int32_t n_frames,
    int32_t * out_seq_len)    // output: encoder tokens produced by this chunk
{
    LOG_DBG(ctx, "Building encoder graph");
    voxtral_model * model = ctx->model;

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // ggml_conv_1d expects input as [length, in_channels, batch]
    // mel_data is [n_mel, n_frames] on CPU; we transpose on upload.
    ggml_tensor * mel_input = ggml_new_tensor_3d(
        gctx, GGML_TYPE_F32, n_frames, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(mel_input, "mel_input");

    // We need to set data after sched_alloc, mark as input
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, mel_input, ctx->backend);

    // Conv stem: mel is [n_frames, n_mel, 1], weights are [k, in_ch, out_ch]
    log_tensor_info(ctx, "enc.conv0.weight", model->enc_conv0_weight);
    log_tensor_info(ctx, "enc.conv1.weight", model->enc_conv1_weight);
    log_tensor_info(ctx, "mel_input", mel_input);

    int32_t conv0_len = 0;
    ggml_tensor * conv0_out = causal_conv1d_graph(
        gctx, mel_input, n_frames,
        model->enc_conv0_weight, model->enc_conv0_bias,
        VOXTRAL_ENC_DIM, 3, 1, conv0_len);
    if (conv0_out == nullptr) {
        LOG_ERR(ctx, "conv0_out is null");
        return gf;
    }
    log_tensor_info(ctx, "conv0_out(pre_act)", conv0_out);
    conv0_out = ggml_gelu_erf(gctx, conv0_out);

    int32_t conv_out_len = 0;
    ggml_tensor * conv1_out = causal_conv1d_graph(
        gctx, conv0_out, conv0_len,
        model->enc_conv1_weight, model->enc_conv1_bias,
        VOXTRAL_ENC_DIM, 3, 2, conv_out_len);
    if (conv1_out == nullptr) {
        LOG_ERR(ctx, "conv1_out is null");
        return gf;
    }
    log_tensor_info(ctx, "conv1_out(pre_act)", conv1_out);
    conv1_out = ggml_gelu_erf(gctx, conv1_out);
    log_tensor_info(ctx, "conv1_out", conv1_out);

    // Transpose for transformer: [enc_dim, seq] -> [enc_dim, seq] (already correct for ggml)
    // In ggml, tensor is [ne0=enc_dim, ne1=seq], which means each "row" (token) has enc_dim elements
    // This is what we need for mul_mat: ggml_mul_mat(weight[out,in], x[in,seq]) -> [out,seq]

    // Left-truncate to multiple of downsample_factor (matching Python)
    const int32_t trunc = conv_out_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    ggml_tensor * x_len_first = conv1_out;
    int32_t seq_len = conv_out_len;
    if (trunc > 0) {
        // Skip first 'trunc' frames along length dimension (ne0)
        x_len_first = ggml_view_3d(gctx, conv1_out,
            conv_out_len - trunc, VOXTRAL_ENC_DIM, 1,
            conv1_out->nb[1], conv1_out->nb[2],
            (size_t) trunc * conv1_out->nb[0]); // [len, enc_dim, 1]
        seq_len = conv_out_len - trunc;
    }
    LOG_DBG(ctx, "encoder conv: in_frames=%d conv0_len=%d conv1_len=%d trunc=%d seq_len=%d",
        n_frames, conv0_len, conv_out_len, trunc, seq_len);

    // Transpose to [enc_dim, seq_len] for transformer blocks
    ggml_tensor * x = ggml_permute(gctx, x_len_first, 1, 0, 2, 3); // [enc_dim, seq_len, 1]
    x = ggml_cont(gctx, x);
    x = ggml_reshape_2d(gctx, x, VOXTRAL_ENC_DIM, seq_len);
    ggml_set_name(x, "encoder_x");
    log_tensor_info(ctx, "encoder_x", x);

    // Position tensor for RoPE: [seq_len] int32
    ggml_tensor * enc_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(enc_positions, "enc_positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_positions, ctx->backend);

    // Encoder attention mask (sliding causal window)
    ggml_tensor * enc_attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(enc_attn_mask, "enc_attn_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_attn_mask, ctx->backend);

    // Cast mask to F16 for flash attention
    ggml_tensor * enc_attn_mask_f16 = ggml_cast(gctx, enc_attn_mask, GGML_TYPE_F16);

    // Transformer layers
    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; i++) {
        auto & L = model->enc_layers[i];

        // Pre-attention RMS norm
        ggml_tensor * residual = x; // [enc_dim, seq_len]
        ggml_tensor * x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS); // [enc_dim, seq_len]
        x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.attn_norm_weight)); // [enc_dim, seq_len]

        // Q, K, V projections
        ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm); // [enc_heads*head_dim, seq_len]
        q = ggml_rowwise_bias(gctx, q, ctx_runtime_tensor(ctx, L.attn_q_bias)); // [enc_heads*head_dim, seq_len]

        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm); // [enc_kv_heads*head_dim, seq_len]
        // k has no bias in encoder

        ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm); // [enc_kv_heads*head_dim, seq_len]
        v = ggml_rowwise_bias(gctx, v, ctx_runtime_tensor(ctx, L.attn_v_bias)); // [enc_kv_heads*head_dim, seq_len]

        // Reshape for RoPE: [head_dim, n_heads, seq_len]
        q = ggml_reshape_3d(gctx, q, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_HEADS, seq_len);
        k = ggml_reshape_3d(gctx, k, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len);

        // Apply RoPE (interleaved, mode=0)
        // ggml_rope_ext expects: a=[head_dim, n_heads, seq], b=[seq] positions
        q = ggml_rope_ext(gctx, q, enc_positions, nullptr,
            VOXTRAL_ENC_HEAD_DIM, 0, 0,
            VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); // [head_dim, n_heads, seq_len]
        k = ggml_rope_ext(gctx, k, enc_positions, nullptr,
            VOXTRAL_ENC_HEAD_DIM, 0, 0,
            VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); // [head_dim, n_kv_heads, seq_len]

        // Flash attention
        // Q: [head_dim, n_heads, seq_len] -> [head_dim, seq_len, n_heads]
        q = ggml_permute(gctx, q, 0, 2, 1, 3); // [head_dim, seq_len, n_heads]
        k = ggml_permute(gctx, k, 0, 2, 1, 3); // [head_dim, seq_len, n_kv_heads]

        // V: [enc_kv_heads*head_dim, seq_len] -> [head_dim, n_kv_heads, seq_len] -> [head_dim, seq_len, n_kv_heads]
        v = ggml_reshape_3d(gctx, v, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len);
        v = ggml_permute(gctx, v, 0, 2, 1, 3); // [head_dim, seq_len, n_kv_heads]

        const float scale = 1.0f / sqrtf((float)VOXTRAL_ENC_HEAD_DIM);

        // ggml_flash_attn_ext fuses Q@K^T, scale, mask, softmax, @V
        ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q, k, v, enc_attn_mask_f16, scale, 0.0f, 0.0f);
        // Output: [head_dim, n_heads, seq_len] (already permuted by flash_attn_ext)
        attn_out = ggml_cont(gctx, attn_out);
        attn_out = ggml_reshape_2d(gctx, attn_out, VOXTRAL_ENC_HEADS * VOXTRAL_ENC_HEAD_DIM, seq_len); // [n_heads*head_dim, seq_len]

        // Output projection + residual
        ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out); // [enc_dim, seq_len]
        attn_proj = ggml_rowwise_bias(gctx, attn_proj, ctx_runtime_tensor(ctx, L.attn_o_bias)); // [enc_dim, seq_len]
        x = ggml_add(gctx, residual, attn_proj); // [enc_dim, seq_len]

        // FFN
        residual = x; // [enc_dim, seq_len]
        x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS); // [enc_dim, seq_len]
        x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.ffn_norm_weight)); // [enc_dim, seq_len]

        // SwiGLU: silu(w1(x)) * w3(x), then w2
        ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm); // [enc_hidden, seq_len]
        gate = ggml_silu(gctx, gate); // [enc_hidden, seq_len]
        ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, x_norm); // [enc_hidden, seq_len]
        ggml_tensor * ffn_out = ggml_mul(gctx, gate, up); // [enc_hidden, seq_len]
        ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out); // [enc_dim, seq_len]
        ffn_out = ggml_rowwise_bias(gctx, ffn_out, ctx_runtime_tensor(ctx, L.ffn_w2_bias)); // [enc_dim, seq_len]

        x = ggml_add(gctx, residual, ffn_out); // [enc_dim, seq_len]
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS); // [enc_dim, seq_len]
    x = ggml_rowwise_scale(gctx, x, ctx_runtime_tensor(ctx, model->enc_norm_weight)); // [enc_dim, seq_len]

    // Copy result to encoder_chunk_output (per-chunk buffer, reused each chunk)
    ggml_tensor * enc_out_view = ggml_view_2d(gctx, ctx->encoder_chunk_output,
        VOXTRAL_ENC_DIM, seq_len,
        ctx->encoder_chunk_output->nb[1], 0); // [enc_dim, seq_len]
    ggml_tensor * cpy = ggml_cpy(gctx, x, enc_out_view);
    ggml_build_forward_expand(gf, cpy);

    if (out_seq_len) *out_seq_len = seq_len;

    return gf;
}

static ggml_cgraph * build_encoder_direct_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    int32_t n_tokens) {

    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    ggml_tensor * encoder_direct_input = ggml_new_tensor_2d(
        gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(encoder_direct_input, "encoder_direct_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, encoder_direct_input, ctx->backend);

    ggml_tensor * enc_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(enc_positions, "enc_positions");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_positions, ctx->backend);

    ggml_tensor * enc_attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(enc_attn_mask, "enc_attn_mask");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_attn_mask, ctx->backend);

    ggml_tensor * enc_attn_mask_f16 = ggml_cast(gctx, enc_attn_mask, GGML_TYPE_F16);
    ggml_tensor * x = encoder_direct_input;

    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; ++i) {
        auto & L = model->enc_layers[i];

        ggml_tensor * residual = x;
        ggml_tensor * x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS);
        x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.attn_norm_weight));
        if (i == 0) {
            ggml_tensor * dbg_attn_norm = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
            ggml_set_name(dbg_attn_norm, "encoder_direct_layer0_attn_norm");
            ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_attn_norm, ctx->backend);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, x_norm, dbg_attn_norm));
        }

        ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm);
        q = ggml_rowwise_bias(gctx, q, ctx_runtime_tensor(ctx, L.attn_q_bias));

        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm);

        ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm);
        v = ggml_rowwise_bias(gctx, v, ctx_runtime_tensor(ctx, L.attn_v_bias));

        q = ggml_reshape_3d(gctx, q, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_HEADS, n_tokens);
        k = ggml_reshape_3d(gctx, k, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, n_tokens);

        q = ggml_rope_ext(gctx, q, enc_positions, nullptr,
            VOXTRAL_ENC_HEAD_DIM, 0, 0,
            VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, enc_positions, nullptr,
            VOXTRAL_ENC_HEAD_DIM, 0, 0,
            VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        q = ggml_permute(gctx, q, 0, 2, 1, 3);
        k = ggml_permute(gctx, k, 0, 2, 1, 3);

        v = ggml_reshape_3d(gctx, v, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, n_tokens);
        v = ggml_permute(gctx, v, 0, 2, 1, 3);

        const float scale = 1.0f / sqrtf((float) VOXTRAL_ENC_HEAD_DIM);
        ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q, k, v, enc_attn_mask_f16, scale, 0.0f, 0.0f);
        attn_out = ggml_cont(gctx, attn_out);
        attn_out = ggml_reshape_2d(gctx, attn_out, VOXTRAL_ENC_HEADS * VOXTRAL_ENC_HEAD_DIM, n_tokens);

        ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out);
        attn_proj = ggml_rowwise_bias(gctx, attn_proj, ctx_runtime_tensor(ctx, L.attn_o_bias));
        x = ggml_add(gctx, residual, attn_proj);
        if (i == 0) {
            ggml_tensor * dbg_after_attn = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
            ggml_set_name(dbg_after_attn, "encoder_direct_layer0_after_attn");
            ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_after_attn, ctx->backend);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_after_attn));
        }

        residual = x;
        x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS);
        x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.ffn_norm_weight));
        if (i == 0) {
            ggml_tensor * dbg_ffn_norm = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
            ggml_set_name(dbg_ffn_norm, "encoder_direct_layer0_ffn_norm");
            ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_ffn_norm, ctx->backend);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, x_norm, dbg_ffn_norm));
        }

        ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm);
        gate = ggml_silu(gctx, gate);
        ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, x_norm);
        ggml_tensor * ffn_out = ggml_mul(gctx, gate, up);
        ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out);
        ffn_out = ggml_rowwise_bias(gctx, ffn_out, ctx_runtime_tensor(ctx, L.ffn_w2_bias));

        x = ggml_add(gctx, residual, ffn_out);
        if (i == 0) {
            ggml_tensor * dbg_after_ffn = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
            ggml_set_name(dbg_after_ffn, "encoder_direct_layer0_after_ffn");
            ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_after_ffn, ctx->backend);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_after_ffn));
        }
        if (i == 1) {
            ggml_tensor * dbg_after_ffn = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
            ggml_set_name(dbg_after_ffn, "encoder_direct_layer1_after_ffn");
            ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_after_ffn, ctx->backend);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_after_ffn));
        }
        if (i == VOXTRAL_ENC_LAYERS - 1) {
            ggml_tensor * dbg_after_ffn = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
            ggml_set_name(dbg_after_ffn, "encoder_direct_last_after_ffn");
            ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_after_ffn, ctx->backend);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_after_ffn));
        }
    }

    ggml_tensor * dbg_pre_final = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(dbg_pre_final, "encoder_direct_pre_final_norm");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_pre_final, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_pre_final));

    x = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS);
    ggml_tensor * dbg_post_rms = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(dbg_post_rms, "encoder_direct_post_final_rms");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_post_rms, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_post_rms));

    x = ggml_rowwise_scale(gctx, x, ctx_runtime_tensor(ctx, model->enc_norm_weight));
    ggml_tensor * dbg_post_scale = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(dbg_post_scale, "encoder_direct_post_final_scale");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, dbg_post_scale, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dbg_post_scale));

    ggml_tensor * enc_out = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(enc_out, "encoder_direct_output");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, enc_out, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, enc_out));

    return gf;
}

static ggml_tensor * build_i32_arange(
    ggml_context * gctx,
    int32_t        start,
    int32_t        count);

static ggml_tensor * build_causal_mask(
    ggml_context * gctx,
    int32_t        n_kv,
    int32_t        n_tokens,
    int32_t        n_past);

static ggml_tensor * build_encoder_incremental_layer(
    voxtral_context * ctx,
    ggml_context * gctx,
    ggml_cgraph * gf,
    ggml_tensor * x,          // [enc_dim, n_tokens]
    ggml_tensor * positions,  // [n_tokens] int32
    ggml_tensor * attn_mask,  // [n_kv, n_tokens]
    int32_t layer_idx,
    int32_t n_tokens,
    int32_t kv_offset) {

    voxtral_model * model = ctx->model;
    auto & L = model->enc_layers[layer_idx];
    const int32_t kv_dim = VOXTRAL_ENC_KV_HEADS * VOXTRAL_ENC_HEAD_DIM; // 2048

    ggml_tensor * residual = x;
    ggml_tensor * x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS);
    x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.attn_norm_weight));

    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm);
    q = ggml_rowwise_bias(gctx, q, ctx_runtime_tensor(ctx, L.attn_q_bias));

    ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm);

    ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm);
    v = ggml_rowwise_bias(gctx, v, ctx_runtime_tensor(ctx, L.attn_v_bias));

    q = ggml_reshape_3d(gctx, q, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_HEADS, n_tokens);
    k = ggml_reshape_3d(gctx, k, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, n_tokens);

    q = ggml_rope_ext(gctx, q, positions, nullptr,
        VOXTRAL_ENC_HEAD_DIM, 0, 0,
        VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(gctx, k, positions, nullptr,
        VOXTRAL_ENC_HEAD_DIM, 0, 0,
        VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    const int32_t n_kv = kv_offset + n_tokens;
    const float scale = 1.0f / sqrtf((float) VOXTRAL_ENC_HEAD_DIM);
    ggml_tensor * attn_mask_f16 = ggml_cast(gctx, attn_mask, GGML_TYPE_F16);
    ggml_tensor * attn_out = nullptr;

    ggml_tensor * q_flat = ggml_cont(gctx, ggml_reshape_2d(
        gctx, q, VOXTRAL_ENC_HEADS * VOXTRAL_ENC_HEAD_DIM, n_tokens));
    ggml_tensor * k_flat = ggml_cont(gctx, ggml_reshape_2d(
        gctx, k, kv_dim, n_tokens));
    ggml_tensor * v_flat = ggml_cont(gctx, v);

    {
        ggml_tensor * k_cache_slice = ggml_view_2d(gctx, ctx->enc_kv_self_k,
            kv_dim, n_tokens,
            ctx->enc_kv_self_k->nb[1],
            layer_idx * ctx->enc_kv_self_k->nb[2] + (size_t) kv_offset * ctx->enc_kv_self_k->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, k_flat, k_cache_slice));

        ggml_tensor * v_cache_slice = ggml_view_2d(gctx, ctx->enc_kv_self_v,
            kv_dim, n_tokens,
            ctx->enc_kv_self_v->nb[1],
            layer_idx * ctx->enc_kv_self_v->nb[2] + (size_t) kv_offset * ctx->enc_kv_self_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, v_flat, v_cache_slice));
    }

    ggml_tensor * q3 = ggml_permute(gctx, q, 0, 2, 1, 3);
    if (kv_offset == 0) {
        ggml_tensor * k3 = ggml_permute(gctx, k, 0, 2, 1, 3);
        ggml_tensor * v3 = ggml_reshape_3d(gctx, v_flat, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, n_tokens);
        v3 = ggml_permute(gctx, v3, 0, 2, 1, 3);
        attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask_f16, scale, 0.0f, 0.0f);
    } else {
        ggml_tensor * k_full = ggml_view_2d(gctx, ctx->enc_kv_self_k,
            kv_dim, n_kv,
            ctx->enc_kv_self_k->nb[1],
            layer_idx * ctx->enc_kv_self_k->nb[2]);
        ggml_tensor * v_full = ggml_view_2d(gctx, ctx->enc_kv_self_v,
            kv_dim, n_kv,
            ctx->enc_kv_self_v->nb[1],
            layer_idx * ctx->enc_kv_self_v->nb[2]);

        ggml_tensor * k3 = ggml_reshape_3d(gctx, k_full, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, n_kv);
        k3 = ggml_permute(gctx, k3, 0, 2, 1, 3);

        ggml_tensor * v3 = ggml_reshape_3d(gctx, v_full, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, n_kv);
        v3 = ggml_permute(gctx, v3, 0, 2, 1, 3);
        attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask_f16, scale, 0.0f, 0.0f);
    }
    attn_out = ggml_cont(gctx, attn_out);
    attn_out = ggml_reshape_2d(gctx, attn_out, VOXTRAL_ENC_HEADS * VOXTRAL_ENC_HEAD_DIM, n_tokens);

    ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out);
    attn_proj = ggml_rowwise_bias(gctx, attn_proj, ctx_runtime_tensor(ctx, L.attn_o_bias));
    x = ggml_add(gctx, residual, attn_proj);

    residual = x;
    x_norm = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS);
    x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.ffn_norm_weight));

    ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, x_norm);
    gate = ggml_silu(gctx, gate);
    ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, x_norm);
    ggml_tensor * ffn_out = ggml_mul(gctx, gate, up);
    ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out);
    ffn_out = ggml_rowwise_bias(gctx, ffn_out, ctx_runtime_tensor(ctx, L.ffn_w2_bias));

    x = ggml_add(gctx, residual, ffn_out);
    return x;
}

static ggml_cgraph * build_encoder_incremental_graph_impl(
    voxtral_context * ctx,
    ggml_context * gctx,
    ggml_tensor * encoder_input,
    int32_t n_tokens,
    int32_t kv_offset,
    int32_t abs_pos_begin) {

    voxtral_model * model = ctx->model;
    const int32_t n_kv = kv_offset + n_tokens;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    ggml_tensor * enc_positions = build_i32_arange(gctx, abs_pos_begin, n_tokens);
    ggml_set_name(enc_positions, "enc_positions");

    ggml_tensor * enc_attn_mask = build_causal_mask(gctx, n_kv, n_tokens, kv_offset);
    ggml_set_name(enc_attn_mask, "enc_attn_mask");

    ggml_tensor * x = encoder_input;
    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; ++i) {
        x = build_encoder_incremental_layer(ctx, gctx, gf, x, enc_positions, enc_attn_mask, i, n_tokens, kv_offset);
    }

    x = ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS);
    x = ggml_rowwise_scale(gctx, x, ctx_runtime_tensor(ctx, model->enc_norm_weight));

    ggml_tensor * encoder_incremental_output = ggml_view_2d(
        gctx,
        ctx->encoder_chunk_output,
        VOXTRAL_ENC_DIM,
        n_tokens,
        ctx->encoder_chunk_output->nb[1],
        0);
    ggml_set_name(encoder_incremental_output, "encoder_incremental_output");
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, encoder_incremental_output));
    return gf;
}

static ggml_cgraph * build_encoder_incremental_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    int32_t n_tokens,
    int32_t kv_offset,
    int32_t abs_pos_begin) {

    ggml_tensor * encoder_new_input = ggml_new_tensor_2d(
        gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ggml_set_name(encoder_new_input, "encoder_new_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, encoder_new_input, ctx->backend);
    return build_encoder_incremental_graph_impl(ctx, gctx, encoder_new_input, n_tokens, kv_offset, abs_pos_begin);
}

static ggml_cgraph * build_encoder_incremental_graph_from_chunk_input(
    voxtral_context * ctx,
    ggml_context * gctx,
    int32_t n_tokens,
    int32_t kv_offset,
    int32_t abs_pos_begin) {

    ggml_tensor * encoder_chunk_input_view = ggml_view_2d(
        gctx,
        ctx->encoder_chunk_input,
        VOXTRAL_ENC_DIM,
        n_tokens,
        ctx->encoder_chunk_input->nb[1],
        0);
    ggml_set_name(encoder_chunk_input_view, "encoder_chunk_input_view");
    return build_encoder_incremental_graph_impl(ctx, gctx, encoder_chunk_input_view, n_tokens, kv_offset, abs_pos_begin);
}

static void run_encoder_direct_debug(
    voxtral_context * ctx,
    const float * x_new_row_major,
    int32_t n_tokens,
    int32_t abs_pos_begin) {

    if (ctx == nullptr || static_cast<int>(ctx->log_level) < static_cast<int>(voxtral_log_level::debug)) {
        return;
    }

    const size_t direct_meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                                    ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> direct_meta_buf(direct_meta_size);
    ggml_init_params direct_params = {
        /*.mem_size  =*/ direct_meta_size,
        /*.mem_buffer=*/ direct_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * direct_ctx = ggml_init(direct_params);
    ggml_cgraph * direct_gf = build_encoder_direct_graph(ctx, direct_ctx, n_tokens);

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, direct_gf)) {
        LOG_ERR(ctx, "encoder direct: failed to allocate graph");
        ggml_free(direct_ctx);
        return;
    }

    ggml_tensor * direct_x_t = find_tensor_in_graph(direct_gf, "encoder_direct_input");
    if (direct_x_t) {
        ggml_backend_tensor_set(
            direct_x_t,
            x_new_row_major,
            0,
            (size_t) n_tokens * (size_t) VOXTRAL_ENC_DIM * sizeof(float));
    }

    ggml_tensor * direct_pos_t = find_tensor_in_graph(direct_gf, "enc_positions");
    if (direct_pos_t) {
        std::vector<int32_t> pos((size_t) n_tokens);
        std::iota(pos.begin(), pos.end(), abs_pos_begin);
        ggml_backend_tensor_set(direct_pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));
    }

    ggml_tensor * direct_mask_t = find_tensor_in_graph(direct_gf, "enc_attn_mask");
    if (direct_mask_t) {
        std::vector<float> direct_mask((size_t) n_tokens * (size_t) n_tokens, -INFINITY);
        for (int32_t q = 0; q < n_tokens; ++q) {
            const int32_t abs_q = abs_pos_begin + q;
            const int32_t min_k = std::max<int32_t>(0, abs_q - (VOXTRAL_ENC_WINDOW - 1));
            for (int32_t kv = 0; kv < n_tokens; ++kv) {
                const int32_t abs_k = abs_pos_begin + kv;
                const bool allow = abs_k <= abs_q && abs_k >= min_k;
                direct_mask[(size_t) q * (size_t) n_tokens + (size_t) kv] = allow ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(direct_mask_t, direct_mask.data(), 0, direct_mask.size() * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx->sched_encoder, direct_gf);
    ggml_tensor * direct_attn_norm_t = find_tensor_in_graph(direct_gf, "encoder_direct_layer0_attn_norm");
    if (direct_attn_norm_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct layer0 attn_norm",
            direct_attn_norm_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_after_attn_t = find_tensor_in_graph(direct_gf, "encoder_direct_layer0_after_attn");
    if (direct_after_attn_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct layer0 after_attn",
            direct_after_attn_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_ffn_norm_t = find_tensor_in_graph(direct_gf, "encoder_direct_layer0_ffn_norm");
    if (direct_ffn_norm_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct layer0 ffn_norm",
            direct_ffn_norm_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_after_ffn_t = find_tensor_in_graph(direct_gf, "encoder_direct_layer0_after_ffn");
    if (direct_after_ffn_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct layer0 after_ffn",
            direct_after_ffn_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_layer1_t = find_tensor_in_graph(direct_gf, "encoder_direct_layer1_after_ffn");
    if (direct_layer1_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct layer1 after_ffn",
            direct_layer1_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_last_t = find_tensor_in_graph(direct_gf, "encoder_direct_last_after_ffn");
    if (direct_last_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct last after_ffn",
            direct_last_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_pre_final_t = find_tensor_in_graph(direct_gf, "encoder_direct_pre_final_norm");
    if (direct_pre_final_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct pre_final_norm",
            direct_pre_final_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_post_rms_t = find_tensor_in_graph(direct_gf, "encoder_direct_post_final_rms");
    if (direct_post_rms_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct post_final_rms",
            direct_post_rms_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_post_scale_t = find_tensor_in_graph(direct_gf, "encoder_direct_post_final_scale");
    if (direct_post_scale_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct post_final_scale",
            direct_post_scale_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }
    ggml_tensor * direct_out_t = find_tensor_in_graph(direct_gf, "encoder_direct_output");
    if (direct_out_t != nullptr) {
        log_col_major_tensor_sample(
            ctx,
            "encoder direct graph output",
            direct_out_t,
            n_tokens,
            VOXTRAL_ENC_DIM,
            n_tokens);
    }

    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(direct_ctx);
}

static bool run_encoder_incremental(
    voxtral_context * ctx,
    const float * x_new_row_major, // [n_tokens, enc_dim]
    int32_t n_tokens,
    int32_t abs_pos_begin) {

    if (ctx == nullptr || x_new_row_major == nullptr || n_tokens <= 0) {
        return false;
    }
    if (n_tokens > VOXTRAL_MAX_ENC_CHUNK) {
        LOG_ERR(ctx, "encoder incremental: %d tokens exceed max chunk %d", n_tokens, VOXTRAL_MAX_ENC_CHUNK);
        return false;
    }

    if (ctx->enc_kv_used + n_tokens > VOXTRAL_ENC_WINDOW) {
        const int32_t shift = ctx->enc_kv_used + n_tokens - VOXTRAL_ENC_WINDOW;
        encoder_kv_cache_shift_left(ctx, shift);
        ctx->enc_kv_used = std::max<int32_t>(0, ctx->enc_kv_used - shift);
    }
    const int32_t kv_offset = ctx->enc_kv_used;
    const int32_t n_kv = kv_offset + n_tokens;

    if (abs_pos_begin == 0) {
        run_encoder_direct_debug(ctx, x_new_row_major, n_tokens, abs_pos_begin);
    }

    static thread_local std::vector<uint8_t> encoder_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    if (encoder_meta_buf.size() < meta_size) {
        encoder_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ encoder_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_encoder_incremental_graph(ctx, gctx, n_tokens, kv_offset, abs_pos_begin);

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        LOG_ERR(ctx, "encoder incremental: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_tensor * x_t = find_tensor_in_graph(gf, "encoder_new_input");
    if (x_t) {
        ggml_backend_tensor_set(
            x_t,
            x_new_row_major,
            0,
            (size_t) n_tokens * (size_t) VOXTRAL_ENC_DIM * sizeof(float));
        if (abs_pos_begin == 0) {
            log_col_major_tensor_sample(
                ctx,
                "encoder incremental input buffer",
                x_t,
                n_tokens,
                VOXTRAL_ENC_DIM,
                n_tokens);
        }
    }

    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);
    if (abs_pos_begin == 0) {
        ggml_tensor * out_t = find_tensor_in_graph(gf, "encoder_incremental_output");
        if (out_t != nullptr) {
            log_col_major_tensor_sample(
                ctx,
                "encoder incremental graph output",
                out_t,
                n_tokens,
                VOXTRAL_ENC_DIM,
                n_tokens);
        }
    }

    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);

    ctx->enc_kv_used = n_kv;
    return true;
}

static bool run_encoder_incremental_from_chunk_input(
    voxtral_context * ctx,
    int32_t n_tokens,
    int32_t abs_pos_begin) {

    if (ctx == nullptr || n_tokens <= 0) {
        return false;
    }
    if (n_tokens > VOXTRAL_MAX_ENC_CHUNK) {
        LOG_ERR(ctx, "encoder incremental device input: %d tokens exceed max chunk %d", n_tokens, VOXTRAL_MAX_ENC_CHUNK);
        return false;
    }

    if (ctx->enc_kv_used + n_tokens > VOXTRAL_ENC_WINDOW) {
        const int32_t shift = ctx->enc_kv_used + n_tokens - VOXTRAL_ENC_WINDOW;
        encoder_kv_cache_shift_left(ctx, shift);
        ctx->enc_kv_used = std::max<int32_t>(0, ctx->enc_kv_used - shift);
    }
    const int32_t kv_offset = ctx->enc_kv_used;
    const int32_t n_kv = kv_offset + n_tokens;

    static thread_local std::vector<uint8_t> encoder_device_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    if (encoder_device_meta_buf.size() < meta_size) {
        encoder_device_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ encoder_device_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_encoder_incremental_graph_from_chunk_input(ctx, gctx, n_tokens, kv_offset, abs_pos_begin);

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        LOG_ERR(ctx, "encoder incremental device input: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);
    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);

    ctx->enc_kv_used = n_kv;
    return true;
}

// ============================================================================
// Graph Building: Adapter
// ============================================================================

static ggml_cgraph * build_adapter_append_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    int32_t enc_src_offset,
    int32_t enc_seq,
    int32_t dec_dst_offset) {

    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;
    ggml_cgraph * gf = ggml_new_graph(gctx);

    ggml_tensor * enc_out = ggml_view_2d(gctx, ctx->encoder_chunk_output,
        VOXTRAL_ENC_DIM, enc_seq,
        ctx->encoder_chunk_output->nb[1],
        (size_t) enc_src_offset * ctx->encoder_chunk_output->nb[1]);

    ggml_tensor * x = build_adapter_project_tensor(ctx, gctx, enc_out, enc_seq);

    ggml_tensor * dec_mem_view = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, dec_seq,
        ctx->decoder_memory->nb[1],
        (size_t) dec_dst_offset * ctx->decoder_memory->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dec_mem_view));
    return gf;
}

static bool run_adapter_append(
    voxtral_context * ctx,
    int32_t enc_src_offset,
    int32_t enc_seq,
    int32_t dec_dst_offset,
    int32_t * out_dec_tokens) {

    if (out_dec_tokens) {
        *out_dec_tokens = 0;
    }
    if (ctx == nullptr || enc_seq <= 0) {
        return true;
    }
    if ((enc_seq % VOXTRAL_DOWNSAMPLE_FACTOR) != 0) {
        LOG_ERR(ctx, "adapter append: enc_seq=%d is not divisible by %d", enc_seq, VOXTRAL_DOWNSAMPLE_FACTOR);
        return false;
    }

    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;
    if (!ensure_decoder_memory_capacity(ctx, dec_dst_offset + dec_seq)) {
        LOG_ERR(ctx, "adapter append: failed to grow decoder memory to %d tokens", dec_dst_offset + dec_seq);
        return false;
    }

    static thread_local std::vector<uint8_t> adapter_append_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
    if (adapter_append_meta_buf.size() < meta_size) {
        adapter_append_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ adapter_append_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_adapter_append_graph(ctx, gctx, enc_src_offset, enc_seq, dec_dst_offset);

    ggml_backend_sched_reset(ctx->sched_adapter);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_adapter, gf)) {
        LOG_ERR(ctx, "adapter append: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_backend_sched_graph_compute(ctx->sched_adapter, gf);
    ggml_backend_sched_reset(ctx->sched_adapter);
    ggml_free(gctx);

    ctx->dec_seq_len = std::max<int32_t>(ctx->dec_seq_len, dec_dst_offset + dec_seq);
    if (out_dec_tokens) {
        *out_dec_tokens = dec_seq;
    }
    return true;
}

static ggml_tensor * build_adapter_project_tensor(
    voxtral_context * ctx,
    ggml_context    * gctx,
    ggml_tensor     * enc_out,
    int32_t           enc_seq) {

    if (ctx == nullptr || gctx == nullptr || enc_out == nullptr || enc_seq <= 0) {
        return nullptr;
    }

    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;
    ggml_tensor * x = enc_out;
    if (!ggml_is_contiguous_0(x)) {
        x = ggml_cont(gctx, x);
    }

    voxtral_model * model = ctx->model;
    x = ggml_reshape_2d(gctx, x, VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, dec_seq);
    x = ggml_mul_mat(gctx, model->adapter_0_weight, x);
    x = ggml_gelu_erf(gctx, x);
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x);
    return x;
}

static ggml_cgraph * build_adapter_upload_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    int32_t enc_seq,
    int32_t dec_dst_offset) {

    voxtral_model * model = ctx->model;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;
    ggml_cgraph * gf = ggml_new_graph(gctx);

    ggml_tensor * encoder_upload_input = ggml_new_tensor_2d(
        gctx, GGML_TYPE_F32, VOXTRAL_ENC_DIM, enc_seq);
    ggml_set_name(encoder_upload_input, "encoder_upload_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_adapter, encoder_upload_input, ctx->backend);

    ggml_tensor * x = ggml_reshape_2d(gctx, encoder_upload_input,
        VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, dec_seq);
    x = ggml_mul_mat(gctx, model->adapter_0_weight, x);
    x = ggml_gelu_erf(gctx, x);
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x);
    ggml_backend_sched_set_tensor_backend(ctx->sched_adapter, x, ctx->backend);

    ggml_tensor * dec_mem_view = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, dec_seq,
        ctx->decoder_memory->nb[1],
        (size_t) dec_dst_offset * ctx->decoder_memory->nb[1]);
    ggml_backend_sched_set_tensor_backend(ctx->sched_adapter, dec_mem_view, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dec_mem_view));
    return gf;
}

static bool run_adapter_upload(
    voxtral_context * ctx,
    const float * enc_row_major,
    int32_t enc_seq,
    int32_t dec_dst_offset,
    int32_t * out_dec_tokens) {

    if (out_dec_tokens) {
        *out_dec_tokens = 0;
    }
    if (ctx == nullptr || enc_row_major == nullptr || enc_seq <= 0) {
        return true;
    }
    if ((enc_seq % VOXTRAL_DOWNSAMPLE_FACTOR) != 0) {
        LOG_ERR(ctx, "adapter upload: enc_seq=%d is not divisible by %d", enc_seq, VOXTRAL_DOWNSAMPLE_FACTOR);
        return false;
    }
    if (enc_seq > VOXTRAL_MAX_ENC_CHUNK) {
        LOG_ERR(ctx, "adapter upload: enc_seq=%d exceeds staging capacity %d", enc_seq, VOXTRAL_MAX_ENC_CHUNK);
        return false;
    }

    // Stage the uploaded encoder rows into the persistent encoder chunk buffer and
    // reuse the normal adapter append graph. This avoids allocating a fresh graph
    // input tensor on the scheduler hot path while keeping the adapter compute on
    // the selected device backend.
    if (dec_dst_offset == 0) {
        log_row_major_sample(ctx, "adapter upload encoder rows", enc_row_major, enc_seq, VOXTRAL_ENC_DIM, enc_seq);
    }
    ggml_backend_tensor_set(
        ctx->encoder_chunk_output,
        enc_row_major,
        0,
        (size_t) enc_seq * (size_t) VOXTRAL_ENC_DIM * sizeof(float));
    if (dec_dst_offset == 0) {
        log_col_major_tensor_sample(
            ctx,
            "adapter upload encoder chunk buffer",
            ctx->encoder_chunk_output,
            enc_seq,
            VOXTRAL_ENC_DIM,
            enc_seq);
    }

    const bool ok = run_adapter_append(ctx, 0, enc_seq, dec_dst_offset, out_dec_tokens);
    if (ok && dec_dst_offset == 0 && ctx->dec_seq_len >= 39) {
        log_decoder_memory_window_sample(ctx, "adapter upload decoder_memory sample", 39);
    }
    return ok;
}

static ggml_cgraph * build_adapter_graph(
    voxtral_context * ctx,
    ggml_context * gctx)
{
    voxtral_model * model = ctx->model;
    const int32_t enc_seq = ctx->enc_seq_used;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;

    ggml_cgraph * gf = ggml_new_graph(gctx);

    // Read encoder_output: [enc_dim, enc_seq]
    ggml_tensor * enc_out = ggml_view_2d(gctx, ctx->encoder_output,
        VOXTRAL_ENC_DIM, enc_seq,
        ctx->encoder_output->nb[1], 0); // [enc_dim, enc_seq]

    // Reshape for downsample: [enc_dim, enc_seq] -> [enc_dim * 4, enc_seq/4]
    ggml_tensor * x = ggml_reshape_2d(gctx, enc_out,
        VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, dec_seq); // [enc_dim*4, dec_seq]

    // Linear 0: [enc_dim*4, dec_seq] -> [dec_dim, dec_seq]
    x = ggml_mul_mat(gctx, model->adapter_0_weight, x); // [dec_dim, dec_seq]
    x = ggml_gelu_erf(gctx, x); // [dec_dim, dec_seq]

    // Linear 2: [dec_dim, dec_seq] -> [dec_dim, dec_seq]
    x = ggml_mul_mat(gctx, model->adapter_2_weight, x); // [dec_dim, dec_seq]

    // Copy to persistent decoder_memory
    ggml_tensor * dec_mem_view = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, dec_seq,
        ctx->decoder_memory->nb[1], 0); // [dec_dim, dec_seq]
    ggml_tensor * cpy = ggml_cpy(gctx, x, dec_mem_view);
    ggml_build_forward_expand(gf, cpy);

    ctx->dec_seq_len = dec_seq;

    return gf;
}

// ============================================================================
// Graph Building: Decoder (common layer forward)
// ============================================================================

// Build one decoder layer. Returns updated hidden state.
// For prefill: n_tokens > 1, positions = [0..n_tokens-1]
// For step: n_tokens = 1
static ggml_tensor * build_decoder_layer(
    voxtral_context     * ctx,
    ggml_context * gctx,
    ggml_cgraph  * gf,
    ggml_tensor  * x,          // [dec_dim, n_tokens]
    ggml_tensor  * positions,  // [n_tokens] int32
    ggml_tensor  * time_emb,   // [dec_dim]
    int32_t layer_idx,
    int32_t n_tokens,
    int32_t kv_offset,                // starting position in KV cache
    ggml_tensor  * attn_mask)  // [n_kv, n_tokens] or nullptr
{
    voxtral_model * model = ctx->model;
    auto & L = model->dec_layers[layer_idx];

    const int32_t kv_dim = VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM; // 1024

    // Pre-attention RMS norm
    ggml_tensor * residual = x; // [dec_dim, n_tokens]
    ggml_tensor * x_norm = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, n_tokens]
    x_norm = ggml_rowwise_scale(gctx, x_norm, ctx_runtime_tensor(ctx, L.attn_norm_weight)); // [dec_dim, n_tokens]

    // Q, K, V (no bias in decoder)
    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, x_norm); // [dec_heads*head_dim, n_tokens]
    ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, x_norm); // [kv_dim, n_tokens]
    ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, x_norm); // [kv_dim, n_tokens]

    // Reshape for RoPE: [head_dim, n_heads, n_tokens]
    q = ggml_reshape_3d(gctx, q, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_HEADS, n_tokens);
    k = ggml_reshape_3d(gctx, k, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_tokens);

    // RoPE (interleaved, mode=0)
    q = ggml_rope_ext(gctx, q, positions, nullptr,
        VOXTRAL_DEC_HEAD_DIM, 0, 0,
        VOXTRAL_DEC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(gctx, k, positions, nullptr,
        VOXTRAL_DEC_HEAD_DIM, 0, 0,
        VOXTRAL_DEC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Flatten Q back: [head_dim, n_heads, n_tokens] -> [n_heads*head_dim, n_tokens]
    q = ggml_cont(gctx, ggml_reshape_2d(gctx, q, VOXTRAL_DEC_HEADS * VOXTRAL_DEC_HEAD_DIM, n_tokens));
    k = ggml_cont(gctx, ggml_reshape_2d(gctx, k, kv_dim, n_tokens));

    // Store K, V in KV cache at positions [kv_offset .. kv_offset+n_tokens-1]
    // KV cache layout: [kv_dim, dec_window, dec_layers]
    // Layer slice: offset = layer_idx * kv_dim * dec_window * sizeof(float)
    {
        ggml_tensor * k_cache_slice = ggml_view_2d(gctx, ctx->kv_self_k,
            kv_dim, n_tokens,
            ctx->kv_self_k->nb[1],
            layer_idx * ctx->kv_self_k->nb[2] + (size_t)kv_offset * ctx->kv_self_k->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, k, k_cache_slice));

        ggml_tensor * v_cache_slice = ggml_view_2d(gctx, ctx->kv_self_v,
            kv_dim, n_tokens,
            ctx->kv_self_v->nb[1],
            layer_idx * ctx->kv_self_v->nb[2] + (size_t)kv_offset * ctx->kv_self_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, v, v_cache_slice));
    }

    // Read full KV from cache: [kv_dim, n_kv] where n_kv = kv_offset + n_tokens
    const int32_t n_kv = kv_offset + n_tokens;
    ggml_tensor * k_full = ggml_view_2d(gctx, ctx->kv_self_k,
        kv_dim, n_kv,
        ctx->kv_self_k->nb[1],
        layer_idx * ctx->kv_self_k->nb[2]); // [kv_dim, n_kv]
    ggml_tensor * v_full = ggml_view_2d(gctx, ctx->kv_self_v,
        kv_dim, n_kv,
        ctx->kv_self_v->nb[1],
        layer_idx * ctx->kv_self_v->nb[2]); // [kv_dim, n_kv]

    // Flash attention with GQA
    // Q: [n_heads*head_dim, n_tokens] -> [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    ggml_tensor * q3 = ggml_reshape_3d(gctx, q, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_HEADS, n_tokens);
    q3 = ggml_permute(gctx, q3, 0, 2, 1, 3); // [head_dim, n_tokens, n_heads]

    // K: [kv_dim, n_kv] -> [head_dim, n_kv_heads, n_kv] -> [head_dim, n_kv, n_kv_heads]
    ggml_tensor * k3 = ggml_reshape_3d(gctx, k_full, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_kv);
    k3 = ggml_permute(gctx, k3, 0, 2, 1, 3); // [head_dim, n_kv, n_kv_heads]

    // V: [kv_dim, n_kv] -> [head_dim, n_kv_heads, n_kv] -> [head_dim, n_kv, n_kv_heads]
    ggml_tensor * v3 = ggml_reshape_3d(gctx, v_full, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_kv);
    v3 = ggml_permute(gctx, v3, 0, 2, 1, 3); // [head_dim, n_kv, n_kv_heads]

    const float scale = 1.0f / sqrtf((float)VOXTRAL_DEC_HEAD_DIM);

    // ggml_flash_attn_ext fuses Q@K^T, scale, mask, softmax, @V in one op
    // GQA broadcast is built-in (n_heads % n_kv_heads == 0)
    // Mask is cast to F16 inside the graph if provided
    ggml_tensor * attn_mask_f16 = attn_mask ? ggml_cast(gctx, attn_mask, GGML_TYPE_F16) : nullptr;
    ggml_tensor * attn_out = ggml_flash_attn_ext(gctx, q3, k3, v3, attn_mask_f16, scale, 0.0f, 0.0f);
    // Output: [head_dim, n_heads, n_tokens] (already permuted by flash_attn_ext)
    attn_out = ggml_cont(gctx, attn_out);
    attn_out = ggml_reshape_2d(gctx, attn_out, VOXTRAL_DEC_HEADS * VOXTRAL_DEC_HEAD_DIM, n_tokens);

    // Output projection + residual
    ggml_tensor * attn_proj = ggml_mul_mat(gctx, L.attn_o_weight, attn_out); // [dec_dim, n_tokens]
    x = ggml_add(gctx, residual, attn_proj); // [dec_dim, n_tokens]

    // Pre-FFN RMS norm
    residual = x; // [dec_dim, n_tokens]
    ggml_tensor * h_norm = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, n_tokens]
    h_norm = ggml_rowwise_scale(gctx, h_norm, ctx_runtime_tensor(ctx, L.ffn_norm_weight)); // [dec_dim, n_tokens]

    // Ada time conditioning: h_norm = h_norm * (1 + ada_mlp(time_emb))
    // = h_norm + h_norm * ada_scale
    // ada_mlp: Linear(3072->32) -> GELU -> Linear(32->3072)
    {
        ggml_tensor * ada_hidden = ggml_mul_mat(gctx, L.ada0_weight, time_emb); // [ada_dim]
        ada_hidden = ggml_gelu_erf(gctx, ada_hidden); // [ada_dim]
        ggml_tensor * ada_scale = ggml_mul_mat(gctx, L.ada2_weight, ada_hidden); // [dec_dim]

        // h_norm * (1 + ada_scale) = h_norm + h_norm * ada_scale
        ggml_tensor * scaled = ggml_rowwise_scale(gctx, h_norm, ada_scale); // [dec_dim, n_tokens]
        h_norm = ggml_add(gctx, h_norm, scaled); // [dec_dim, n_tokens]
    }

    // SwiGLU FFN
    ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_w1_weight, h_norm); // [dec_hidden, n_tokens]
    gate = ggml_silu(gctx, gate); // [dec_hidden, n_tokens]
    ggml_tensor * up = ggml_mul_mat(gctx, L.ffn_w3_weight, h_norm); // [dec_hidden, n_tokens]
    ggml_tensor * ffn_out = ggml_mul(gctx, gate, up); // [dec_hidden, n_tokens]
    ffn_out = ggml_mul_mat(gctx, L.ffn_w2_weight, ffn_out); // [dec_dim, n_tokens]

    x = ggml_add(gctx, residual, ffn_out); // [dec_dim, n_tokens]

    return x;
}

// ============================================================================
// Graph Building: Decoder Prefill
// ============================================================================

static ggml_cgraph * build_decoder_prefill_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               n_tokens,
    bool                  emit_logits,
    bool                  emit_token)  // number of prompt tokens
{
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    // Token IDs input: [n_tokens] int32
    ggml_tensor * token_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids, "token_ids");
    ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, token_ids, ctx->backend);

    // Position indices: [n_tokens] int32
    ggml_tensor * positions = build_i32_arange(gctx, 0, n_tokens);
    ggml_set_name(positions, "positions");

    ggml_tensor * time_emb = ctx->decoder_time_emb;

    // Token embeddings: [dec_dim, n_tokens]
    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_ids); // [dec_dim, n_tokens]

    // Audio embeddings from decoder_memory: [dec_dim, n_tokens]
    ggml_tensor * audio_emb = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, n_tokens,
        ctx->decoder_memory->nb[1], 0); // [dec_dim, n_tokens]

    // Combined input: tok_emb + audio_emb
    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb); // [dec_dim, n_tokens]

    // Causal mask for prefill: [n_tokens, n_tokens] additive mask
    // -inf for positions that should not attend
    ggml_tensor * causal_mask = build_causal_mask(gctx, n_tokens, n_tokens, 0);
    ggml_set_name(causal_mask, "causal_mask");

    // Decoder layers
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, positions, time_emb,
            i, n_tokens, /*kv_offset=*/0, causal_mask);
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, n_tokens]
    x = ggml_rowwise_scale(gctx, x, ctx_runtime_tensor(ctx, model->dec_norm_weight)); // [dec_dim, n_tokens]

    if (emit_logits || emit_token) {
        // Logits for last token only: extract last token -> matmul with embeddings
        ggml_tensor * last_hidden = ggml_view_1d(gctx, x, VOXTRAL_DEC_DIM,
            (n_tokens - 1) * x->nb[1]); // [dec_dim]

        ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, last_hidden); // [vocab_size]
        if (emit_logits) {
            ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));
        }
        if (emit_token && ctx->decoder_prev_token != nullptr) {
            ggml_tensor * logits_matrix = ggml_reshape_2d(gctx, logits, VOXTRAL_VOCAB_SIZE, 1);
            ggml_tensor * argmax = ggml_argmax(gctx, logits_matrix);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, argmax, ctx->decoder_prev_token));
        }
    }

    return gf;
}

// ============================================================================
// Graph Building: Decoder Step (single token)
// ============================================================================

static ggml_cgraph * build_decoder_step_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               position,    // absolute position
    int32_t               audio_pos,   // position in audio embeddings (may differ)
    bool                  emit_logits,
    bool                  use_prev_token_device)
{
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    const int32_t kv_used = ctx->kv_used;  // tokens already in KV cache

    ggml_tensor * token_id = nullptr;
    ggml_tensor * token_src = nullptr;
    if (use_prev_token_device) {
        token_src = ctx->decoder_prev_token;
    } else {
        token_id = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
        ggml_set_name(token_id, "token_id");
        ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, token_id, ctx->backend);
        token_src = token_id;
    }

    // Position: [1] int32
    ggml_tensor * pos_tensor = build_i32_arange(gctx, position, 1);
    ggml_set_name(pos_tensor, "position");

    ggml_tensor * time_emb = ctx->decoder_time_emb;

    // Token embedding: [dec_dim, 1]
    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_src); // [dec_dim, 1]

    // Audio embedding from decoder_memory at audio_pos
    ggml_tensor * audio_emb = ggml_view_2d(gctx, ctx->decoder_memory,
        VOXTRAL_DEC_DIM, 1,
        ctx->decoder_memory->nb[1],
        (size_t)audio_pos * ctx->decoder_memory->nb[1]); // [dec_dim, 1]

    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb); // [dec_dim, 1]

    // Decoder layers (no mask needed for single token - all KV positions are valid)
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        x = build_decoder_layer(ctx, gctx, gf, x, pos_tensor, time_emb,
            i, 1, /*kv_offset=*/kv_used, /*attn_mask=*/nullptr);
    }

    // Final norm
    x = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS); // [dec_dim, 1]
    x = ggml_rowwise_scale(gctx, x, ctx_runtime_tensor(ctx, model->dec_norm_weight)); // [dec_dim, 1]

    // Logits
    ggml_tensor * x_flat = ggml_reshape_1d(gctx, x, VOXTRAL_DEC_DIM); // [dec_dim]
    ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, x_flat); // [vocab_size]
    ggml_tensor * logits_matrix = ggml_reshape_2d(gctx, logits, VOXTRAL_VOCAB_SIZE, 1);
    if (emit_logits) {
        ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));
    }
    if (ctx->decoder_prev_token != nullptr) {
        ggml_tensor * argmax = ggml_argmax(gctx, logits_matrix);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, argmax, ctx->decoder_prev_token));
    }

    return gf;
}

static ggml_cgraph * build_decoder_step_graph_saturated_cached(
    voxtral_context * ctx,
    ggml_context    * gctx)
{
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

    ggml_tensor * token_src = ctx->decoder_prev_token;
    ggml_tensor * pos_tensor = ctx->decoder_step_position;
    ggml_tensor * time_emb = ctx->decoder_time_emb;
    ggml_tensor * audio_emb = ctx->decoder_step_audio_emb;

    ggml_tensor * tok_emb = ggml_get_rows(gctx, model->tok_embeddings_weight, token_src);
    ggml_tensor * x = ggml_add(gctx, tok_emb, audio_emb);

    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        x = build_decoder_layer(
            ctx,
            gctx,
            gf,
            x,
            pos_tensor,
            time_emb,
            i,
            1,
            /*kv_offset=*/VOXTRAL_DEC_WINDOW - 1,
            /*attn_mask=*/nullptr);
    }

    x = ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS);
    x = ggml_rowwise_scale(gctx, x, ctx_runtime_tensor(ctx, model->dec_norm_weight));

    ggml_tensor * x_flat = ggml_reshape_1d(gctx, x, VOXTRAL_DEC_DIM);
    ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, x_flat);
    ggml_tensor * logits_matrix = ggml_reshape_2d(gctx, logits, VOXTRAL_VOCAB_SIZE, 1);
    ggml_tensor * argmax = ggml_argmax(gctx, logits_matrix);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, argmax, ctx->decoder_prev_token));

    return gf;
}

static bool ensure_decoder_step_graph_saturated_cached(voxtral_context * ctx) {
    if (ctx == nullptr) {
        return false;
    }
    if (ctx->gf_dec_step_cached != nullptr && ctx->ctx_dec_step_cached != nullptr) {
        return true;
    }

    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    ctx->dec_step_cached_meta.resize(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ ctx->dec_step_cached_meta.data(),
        /*.no_alloc  =*/ true,
    };
    ctx->ctx_dec_step_cached = ggml_init(p);
    if (ctx->ctx_dec_step_cached == nullptr) {
        return false;
    }

    ctx->gf_dec_step_cached = build_decoder_step_graph_saturated_cached(ctx, ctx->ctx_dec_step_cached);
    if (ctx->gf_dec_step_cached == nullptr) {
        ggml_free(ctx->ctx_dec_step_cached);
        ctx->ctx_dec_step_cached = nullptr;
        ctx->dec_step_cached_meta.clear();
        return false;
    }

    ctx->dec_step_cached_sched_ready = false;
    return true;
}

// ============================================================================
// Helper: set named input tensors in a graph
// ============================================================================

static ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name) {
    return ggml_graph_get_tensor(gf, name);
}

static ggml_tensor * build_i32_arange(
    ggml_context * gctx,
    int32_t        start,
    int32_t        count) {

    ggml_tensor * arange_f32 = ggml_arange(
        gctx,
        (float) start,
        (float) (start + count),
        1.0f);
    return ggml_cast(gctx, arange_f32, GGML_TYPE_I32);
}

static ggml_tensor * build_causal_mask(
    ggml_context * gctx,
    int32_t        n_kv,
    int32_t        n_tokens,
    int32_t        n_past) {

    ggml_tensor * zero_scalar = ggml_arange(gctx, 0.0f, 1.0f, 1.0f);
    ggml_tensor * mask_shape = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_tensor * mask_base = ggml_repeat(gctx, zero_scalar, mask_shape);
    return ggml_diag_mask_inf(gctx, mask_base, n_past);
}

// ============================================================================
// Run Encoder
// ============================================================================

// Run a single encoder chunk: build graph, set inputs, compute, return seq_len
static bool run_encoder_chunk(
    voxtral_context * ctx,
    const float * chunk_mel_data,  // [n_mel, chunk_mel_frames]
    int32_t chunk_mel_frames,
    int32_t rope_pos_offset,
    int32_t * out_seq_len)
{
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    int32_t chunk_seq_len = 0;
    ggml_cgraph * gf = build_encoder_graph(ctx, gctx, chunk_mel_data, chunk_mel_frames, &chunk_seq_len);

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        LOG_ERR(ctx, "encoder chunk: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set mel input
    ggml_tensor * mel_t = find_tensor_in_graph(gf, "mel_input");
    if (mel_t) {
        const int64_t expected_ne0 = chunk_mel_frames;
        const int64_t expected_ne1 = VOXTRAL_NUM_MEL_BINS;
        if (mel_t->ne[0] == expected_ne0 && mel_t->ne[1] == expected_ne1) {
            ggml_backend_tensor_set(mel_t, chunk_mel_data, 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        } else if (mel_t->ne[0] == expected_ne1 && mel_t->ne[1] == expected_ne0) {
            std::vector<float> mel_tbuf((size_t) chunk_mel_frames * VOXTRAL_NUM_MEL_BINS);
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                const float * src = chunk_mel_data + (size_t) m * chunk_mel_frames;
                for (int32_t f = 0; f < chunk_mel_frames; ++f) {
                    mel_tbuf[(size_t) m + (size_t) VOXTRAL_NUM_MEL_BINS * f] = src[f];
                }
            }
            ggml_backend_tensor_set(mel_t, mel_tbuf.data(), 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        } else {
            ggml_backend_tensor_set(mel_t, chunk_mel_data, 0,
                (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
        }
    }

    // Set positions with RoPE offset for absolute positions across chunks
    ggml_tensor * pos_t = find_tensor_in_graph(gf, "enc_positions");
    if (pos_t) {
        std::vector<int32_t> pos(chunk_seq_len);
        std::iota(pos.begin(), pos.end(), rope_pos_offset);
        ggml_backend_tensor_set(pos_t, pos.data(), 0, chunk_seq_len * sizeof(int32_t));
    }

    // Set encoder sliding causal mask (local to chunk)
    ggml_tensor * mask_t = find_tensor_in_graph(gf, "enc_attn_mask");
    if (mask_t) {
        std::vector<float> mask((size_t) chunk_seq_len * chunk_seq_len);
        for (int32_t q = 0; q < chunk_seq_len; ++q) {
            const int32_t min_kv = std::max<int32_t>(0, q - (VOXTRAL_ENC_WINDOW - 1));
            for (int32_t kv = 0; kv < chunk_seq_len; ++kv) {
                const bool allow = (kv <= q) && (kv >= min_kv);
                mask[(size_t) q * chunk_seq_len + kv] = allow ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);

    if (rope_pos_offset == 0) {
        ggml_tensor * enc_x_t = find_tensor_in_graph(gf, "encoder_x");
        if (enc_x_t != nullptr) {
            log_col_major_tensor_sample(
                ctx,
                "encoder offline graph input",
                enc_x_t,
                chunk_seq_len,
                VOXTRAL_ENC_DIM,
                chunk_seq_len);
        }
        log_col_major_tensor_sample(
            ctx,
            "encoder offline graph output",
            ctx->encoder_chunk_output,
            chunk_seq_len,
            VOXTRAL_ENC_DIM,
            chunk_seq_len);
    }

    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);

    if (out_seq_len) *out_seq_len = chunk_seq_len;
    return true;
}

// Process mel spectrogram in overlapping chunks, accumulating encoder output on device
static bool run_encoder_chunked(voxtral_context * ctx, const float * mel_data, int32_t total_mel_frames) {
    const int32_t mel_overlap = VOXTRAL_ENC_CHUNK_OVERLAP * 2;  // mel frames of overlap (1500)
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - mel_overlap;  // 1500

    // Pre-compute total encoder tokens for allocation
    int32_t alloc_total = compute_total_enc_tokens(total_mel_frames);
    if (alloc_total <= 0) {
        LOG_ERR(ctx, "encoder: audio too short to produce encoder tokens");
        return false;
    }

    // Allocate encoder_output on device
    if (!alloc_encoder_output(ctx, alloc_total)) {
        LOG_ERR(ctx, "encoder: failed to allocate encoder output (%d tokens, %.2f MB)",
                alloc_total, (double) alloc_total * VOXTRAL_ENC_DIM * sizeof(float) / 1e6);
        return false;
    }

    LOG_INFO(ctx, "encoder chunked: %d mel frames, %d alloc enc tokens, mel_stride=%d",
             total_mel_frames, alloc_total, mel_stride);

    int32_t mel_offset = 0;
    int32_t enc_write_offset = 0;
    int32_t chunk_idx = 0;

    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel_frames = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);

        // Pre-check: will this chunk contribute any new tokens?
        // This avoids building and running the full encoder graph for nothing.
        int32_t skip = (chunk_idx > 0) ? VOXTRAL_ENC_CHUNK_OVERLAP : 0;
        {
            int32_t expected_tokens = mel_frames_to_enc_tokens(chunk_mel_frames);
            if (expected_tokens - skip <= 0) {
                LOG_DBG(ctx, "encoder chunk %d: skipped (expected %d tokens, skip=%d)",
                        chunk_idx, expected_tokens, skip);
                break;
            }
        }

        // For single-chunk case (entire mel fits), use mel_data directly to avoid copy
        const float * chunk_mel_ptr = nullptr;
        std::vector<float> chunk_mel_buf;
        if (mel_offset == 0 && chunk_mel_frames == total_mel_frames) {
            // Single chunk — mel_data is already in [n_mel, total_frames] layout
            chunk_mel_ptr = mel_data;
        } else {
            // Multi-chunk — extract sub-range of frames for this chunk
            chunk_mel_buf.resize((size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames);
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) {
                memcpy(chunk_mel_buf.data() + (size_t) m * chunk_mel_frames,
                       mel_data + (size_t) m * total_mel_frames + mel_offset,
                       chunk_mel_frames * sizeof(float));
            }
            chunk_mel_ptr = chunk_mel_buf.data();
        }

        int32_t rope_offset = enc_write_offset - skip;

        // Run encoder for this chunk
        int32_t chunk_seq_len = 0;
        if (!run_encoder_chunk(ctx, chunk_mel_ptr, chunk_mel_frames, rope_offset, &chunk_seq_len)) {
            LOG_ERR(ctx, "encoder chunk %d: failed", chunk_idx);
            return false;
        }

        int32_t stride = chunk_seq_len - skip;
        if (stride <= 0) {
            LOG_DBG(ctx, "encoder chunk %d: no new tokens (seq_len=%d, skip=%d), stopping",
                    chunk_idx, chunk_seq_len, skip);
            break;
        }

        // Clamp stride to not overflow pre-allocated buffer
        if (enc_write_offset + stride > alloc_total) {
            stride = alloc_total - enc_write_offset;
            if (stride <= 0) break;
        }

        LOG_INFO(ctx, "encoder chunk %d: mel[%d..%d) enc_tokens=%d skip=%d stride=%d rope_offset=%d",
                 chunk_idx, mel_offset, mel_offset + chunk_mel_frames,
                 chunk_seq_len, skip, stride, rope_offset);

        // Copy stride portion from encoder_chunk_output to encoder_output directly on device.
        {
            const size_t elem_bytes = VOXTRAL_ENC_DIM * sizeof(float);
            const size_t src_offset = (size_t) skip * elem_bytes;
            const size_t dst_offset = (size_t) enc_write_offset * elem_bytes;
            if (!copy_tensor_2d_region(
                    ctx->encoder_chunk_output,
                    ctx->encoder_output,
                    VOXTRAL_ENC_DIM,
                    stride,
                    src_offset,
                    dst_offset)) {
                LOG_ERR(ctx, "encoder chunk %d: device copy into encoder_output failed", chunk_idx);
                return false;
            }
        }

        enc_write_offset += stride;
        mel_offset += mel_stride;
        chunk_idx++;
    }

    // Trim to multiple of downsample factor for adapter compatibility
    ctx->enc_seq_used = (enc_write_offset / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
    ctx->total_enc_tokens = ctx->enc_seq_used;

    LOG_INFO(ctx, "encoder done: %d chunks, enc_seq_used=%d (raw=%d)",
             chunk_idx, ctx->enc_seq_used, enc_write_offset);
    return true;
}

// ============================================================================
// Run Adapter
// ============================================================================

static bool run_adapter(voxtral_context * ctx) {
    const int32_t enc_seq = ctx->enc_seq_used;
    const int32_t dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;

    LOG_INFO(ctx, "running adapter: enc_seq=%d -> dec_seq=%d", enc_seq, dec_seq);

    // Allocate decoder_memory for this utterance
    if (!alloc_decoder_memory(ctx, dec_seq)) {
        LOG_ERR(ctx, "adapter: failed to allocate decoder memory (%d tokens, %.2f MB)",
                dec_seq, (double) dec_seq * VOXTRAL_DEC_DIM * sizeof(float) / 1e6);
        return false;
    }

    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_adapter_graph(ctx, gctx);
    log_graph_info(ctx, "adapter", gf);

    ggml_backend_sched_reset(ctx->sched_adapter);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_adapter, gf)) {
        LOG_ERR(ctx, "adapter: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_backend_sched_graph_compute(ctx->sched_adapter, gf);
    ggml_backend_sched_reset(ctx->sched_adapter);
    ggml_free(gctx);

    LOG_INFO(ctx, "adapter done: dec_seq_len=%d (%.2f MB on device)",
             ctx->dec_seq_len,
             (double) ggml_nbytes(ctx->decoder_memory) / 1e6);
    return true;
}

// ============================================================================
// Run Decoder Prefill
// ============================================================================

static bool run_decoder_prefill(
    voxtral_context * ctx,
    const int32_t   * token_ids,
    int32_t           n_tokens,
    float           * logits_out,  // [vocab_size]
    int32_t         * token_out)
{
    LOG_DBG(ctx, "decoder prefill: %d tokens", n_tokens);

    if (n_tokens > VOXTRAL_DEC_WINDOW) {
        LOG_ERR(ctx, "decoder prefill: n_tokens=%d exceeds DEC_WINDOW=%d", n_tokens, VOXTRAL_DEC_WINDOW);
        return false;
    }

    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> meta_buf(meta_size);

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_decoder_prefill_graph(ctx, gctx, n_tokens, logits_out != nullptr, token_out != nullptr);
    log_graph_info(ctx, "decoder prefill", gf);

    ggml_backend_sched_reset(ctx->sched_dec_pre);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_pre, gf)) {
        LOG_ERR(ctx, "decoder prefill: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set inputs
    ggml_tensor * tok_t = find_tensor_in_graph(gf, "token_ids");
    if (tok_t) {
        ggml_backend_tensor_set(tok_t, token_ids, 0, n_tokens * sizeof(int32_t));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_dec_pre, gf);

    if (logits_out != nullptr) {
        ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    }
    if (token_out != nullptr && ctx->decoder_prev_token != nullptr) {
        ggml_backend_tensor_get(ctx->decoder_prev_token, token_out, 0, sizeof(int32_t));
    }

    ctx->kv_used = std::min(n_tokens, VOXTRAL_DEC_WINDOW);

    ggml_backend_sched_reset(ctx->sched_dec_pre);
    ggml_free(gctx);

    LOG_DBG(ctx, "decoder prefill done");
    return true;
}

// ============================================================================
// Run Decoder Step
// ============================================================================

static bool run_decoder_step(
    voxtral_context * ctx,
    int32_t           token_id,
    int32_t           position,     // absolute position in decoder sequence
    int32_t           audio_pos,    // position in adapter output for audio embedding
    float           * logits_out,   // [vocab_size]
    int32_t         * token_out)
{
    if (ctx->kv_used >= VOXTRAL_DEC_WINDOW) {
        kv_cache_shift_left(ctx, 1);
        ctx->kv_used = VOXTRAL_DEC_WINDOW - 1;
    }
    ctx->dec_step_cached_sched_ready = false;

    // Use thread-local buffer to avoid per-step heap allocation
    static thread_local std::vector<uint8_t> step_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    if (step_meta_buf.size() < meta_size) {
        step_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ step_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_decoder_step_graph(ctx, gctx, position, audio_pos, logits_out != nullptr, false);

    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_step, gf)) {
        LOG_ERR(ctx, "decoder step: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    // Set inputs
    ggml_tensor * tok_t = find_tensor_in_graph(gf, "token_id");
    if (tok_t) {
        ggml_backend_tensor_set(tok_t, &token_id, 0, sizeof(int32_t));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched_dec_step, gf);

    if (logits_out != nullptr) {
        ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    }
    if (token_out != nullptr && ctx->decoder_prev_token != nullptr) {
        ggml_backend_tensor_get(ctx->decoder_prev_token, token_out, 0, sizeof(int32_t));
    }

    ctx->kv_used += 1;

    ggml_backend_sched_reset(ctx->sched_dec_step);
    ggml_free(gctx);

    return true;
}

static bool run_decoder_step_from_prev_token(
    voxtral_context * ctx,
    int32_t           position,
    int32_t           audio_pos,
    float           * logits_out,
    int32_t         * token_out)
{
    if (ctx == nullptr || ctx->decoder_prev_token == nullptr) {
        return false;
    }
    if (ctx->kv_used >= VOXTRAL_DEC_WINDOW) {
        kv_cache_shift_left(ctx, 1);
        ctx->kv_used = VOXTRAL_DEC_WINDOW - 1;
    }

    const bool can_use_cached_saturated_step =
        logits_out == nullptr &&
        ctx->kv_used == VOXTRAL_DEC_WINDOW - 1 &&
        ctx->decoder_step_position != nullptr &&
        ctx->decoder_step_audio_emb != nullptr;

    if (can_use_cached_saturated_step) {
        if (audio_pos < 0 || ctx->decoder_memory == nullptr || audio_pos >= ctx->dec_seq_len) {
            return false;
        }
        if (!ensure_decoder_step_graph_saturated_cached(ctx)) {
            return false;
        }

        const size_t audio_offset = (size_t) audio_pos * ctx->decoder_memory->nb[1];
        if (!copy_tensor_2d_region(
                ctx->decoder_memory,
                ctx->decoder_step_audio_emb,
                VOXTRAL_DEC_DIM,
                1,
                audio_offset,
                0)) {
            return false;
        }
        ggml_backend_tensor_set(ctx->decoder_step_position, &position, 0, sizeof(int32_t));

        if (!ctx->dec_step_cached_sched_ready) {
            ggml_backend_sched_reset(ctx->sched_dec_step);
            if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_step, ctx->gf_dec_step_cached)) {
                LOG_ERR(ctx, "decoder step cached saturated: failed to allocate graph");
                return false;
            }
            ctx->dec_step_cached_sched_ready = true;
        }

        ggml_backend_sched_graph_compute(ctx->sched_dec_step, ctx->gf_dec_step_cached);

        if (token_out != nullptr && ctx->decoder_prev_token != nullptr) {
            ggml_backend_tensor_get(ctx->decoder_prev_token, token_out, 0, sizeof(int32_t));
        }

        ctx->kv_used += 1;
        return true;
    }

    ctx->dec_step_cached_sched_ready = false;

    static thread_local std::vector<uint8_t> step_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    if (step_meta_buf.size() < meta_size) {
        step_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ step_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);

    ggml_cgraph * gf = build_decoder_step_graph(ctx, gctx, position, audio_pos, logits_out != nullptr, true);

    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_dec_step, gf)) {
        LOG_ERR(ctx, "decoder step device token: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_backend_sched_graph_compute(ctx->sched_dec_step, gf);

    if (logits_out != nullptr) {
        ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    }
    if (token_out != nullptr && ctx->decoder_prev_token != nullptr) {
        ggml_backend_tensor_get(ctx->decoder_prev_token, token_out, 0, sizeof(int32_t));
    }

    ctx->kv_used += 1;

    ggml_backend_sched_reset(ctx->sched_dec_step);
    ggml_free(gctx);
    return true;
}

// ============================================================================
// Streaming API
// ============================================================================

struct voxtral_stream {
    voxtral_context * ctx = nullptr;
    voxtral_stream_params params = {};
    voxtral_incremental_mel mel = {};
    int64_t real_samples_fed = 0;
    int32_t mel_cursor = 0;

    std::vector<float> mel_tail;        // [n_mel, 2] col-major
    std::vector<float> conv0_tail;      // [enc_dim, 2] col-major
    std::vector<float> conv0_residual;  // [enc_dim]
    std::vector<float> conv0_input_scratch; // [n_mel, n_frames] col-major
    std::vector<float> cpu_conv_in_scratch;
    std::vector<float> cpu_conv0_new_scratch;
    std::vector<float> cpu_conv0_full_scratch;
    std::vector<float> cpu_feed_scratch;
    std::vector<float> cpu_conv1_in_scratch;
    std::vector<float> cpu_conv1_out_scratch;
    int32_t conv0_residual_count = 0;
    bool conv_stem_initialized = false;

    ggml_context * conv_state_ctx = nullptr;
    ggml_backend_buffer_t conv_state_buf = nullptr;
    std::array<ggml_tensor *, 2> conv0_tail_tensors = { nullptr, nullptr };      // [enc_dim, 2]
    std::array<ggml_tensor *, 2> conv0_residual_tensors = { nullptr, nullptr };  // [enc_dim, 1]
    int32_t conv_state_slot = 0;

    ggml_context * enc_residual_ctx = nullptr;
    ggml_backend_buffer_t enc_residual_buf = nullptr;
    ggml_tensor * enc_residual_tensor = nullptr; // [enc_dim, downsample_factor]
    int32_t enc_residual_count = 0;
    int32_t total_encoder_positions = 0;

    int32_t total_audio_tokens = 0;
    bool decoder_started = false;
    bool eos_seen = false;
    bool finished = false;
    int32_t prev_token = VOXTRAL_TOKEN_BOS;
    int32_t gen_pos = 0;
    int32_t generated_token_index = 0;  // counts all generated token positions in the active segment
    int32_t timeline_token_base = 0;    // absolute 80 ms token index for this segment
    int32_t min_new_mel_frames = 1;

    std::string pending_piece_text;
    int32_t pending_piece_begin_token_index = -1;
    int32_t pending_piece_end_token_index = -1;
};

static void free_stream_conv_state(voxtral_stream & stream) {
    if (stream.conv_state_buf != nullptr) {
        ggml_backend_buffer_free(stream.conv_state_buf);
        stream.conv_state_buf = nullptr;
    }
    if (stream.conv_state_ctx != nullptr) {
        ggml_free(stream.conv_state_ctx);
        stream.conv_state_ctx = nullptr;
    }
    stream.conv0_tail_tensors = { nullptr, nullptr };
    stream.conv0_residual_tensors = { nullptr, nullptr };
    stream.conv_state_slot = 0;
}

static ggml_tensor * active_stream_conv0_tail_tensor(voxtral_stream & stream) {
    return stream.conv0_tail_tensors[(size_t) stream.conv_state_slot];
}

static ggml_tensor * active_stream_conv0_residual_tensor(voxtral_stream & stream) {
    return stream.conv0_residual_tensors[(size_t) stream.conv_state_slot];
}

static ggml_tensor * inactive_stream_conv0_tail_tensor(voxtral_stream & stream) {
    return stream.conv0_tail_tensors[(size_t) (1 - stream.conv_state_slot)];
}

static ggml_tensor * inactive_stream_conv0_residual_tensor(voxtral_stream & stream) {
    return stream.conv0_residual_tensors[(size_t) (1 - stream.conv_state_slot)];
}

static void clear_stream_conv_state(voxtral_stream & stream) {
    for (ggml_tensor * t : stream.conv0_tail_tensors) {
        if (t != nullptr) {
            zero_tensor_bytes(t);
        }
    }
    for (ggml_tensor * t : stream.conv0_residual_tensors) {
        if (t != nullptr) {
            zero_tensor_bytes(t);
        }
    }
    stream.conv_state_slot = 0;
}

static bool ensure_stream_conv_state(voxtral_stream & stream) {
    if (stream.conv0_tail_tensors[0] != nullptr &&
        stream.conv0_tail_tensors[1] != nullptr &&
        stream.conv0_residual_tensors[0] != nullptr &&
        stream.conv0_residual_tensors[1] != nullptr) {
        return true;
    }
    if (stream.ctx == nullptr || stream.ctx->backend == nullptr || stream.ctx->gpu_type == voxtral_gpu_backend::none) {
        return false;
    }

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead() * 4,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    stream.conv_state_ctx = ggml_init(p);
    if (stream.conv_state_ctx == nullptr) {
        return false;
    }

    for (int i = 0; i < 2; ++i) {
        char nm_tail[64];
        char nm_residual[64];
        snprintf(nm_tail, sizeof(nm_tail), "stream_conv0_tail_%d", i);
        snprintf(nm_residual, sizeof(nm_residual), "stream_conv0_residual_%d", i);

        stream.conv0_tail_tensors[(size_t) i] = ggml_new_tensor_2d(
            stream.conv_state_ctx,
            GGML_TYPE_F32,
            VOXTRAL_ENC_DIM,
            2);
        ggml_set_name(stream.conv0_tail_tensors[(size_t) i], nm_tail);

        stream.conv0_residual_tensors[(size_t) i] = ggml_new_tensor_2d(
            stream.conv_state_ctx,
            GGML_TYPE_F32,
            VOXTRAL_ENC_DIM,
            1);
        ggml_set_name(stream.conv0_residual_tensors[(size_t) i], nm_residual);
    }

    stream.conv_state_buf = ggml_backend_alloc_ctx_tensors(stream.conv_state_ctx, stream.ctx->backend);
    if (stream.conv_state_buf == nullptr) {
        free_stream_conv_state(stream);
        return false;
    }

    clear_stream_conv_state(stream);
    return true;
}

static void free_stream_encoder_residual(voxtral_stream & stream) {
    if (stream.enc_residual_buf != nullptr) {
        ggml_backend_buffer_free(stream.enc_residual_buf);
        stream.enc_residual_buf = nullptr;
    }
    if (stream.enc_residual_ctx != nullptr) {
        ggml_free(stream.enc_residual_ctx);
        stream.enc_residual_ctx = nullptr;
    }
    stream.enc_residual_tensor = nullptr;
    stream.enc_residual_count = 0;
}

static bool ensure_stream_encoder_residual(voxtral_stream & stream) {
    if (stream.enc_residual_tensor != nullptr) {
        return true;
    }
    if (stream.ctx == nullptr || stream.ctx->backend == nullptr) {
        return false;
    }

    ggml_init_params p = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    stream.enc_residual_ctx = ggml_init(p);
    if (stream.enc_residual_ctx == nullptr) {
        return false;
    }

    stream.enc_residual_tensor = ggml_new_tensor_2d(
        stream.enc_residual_ctx,
        GGML_TYPE_F32,
        VOXTRAL_ENC_DIM,
        VOXTRAL_DOWNSAMPLE_FACTOR);
    ggml_set_name(stream.enc_residual_tensor, "stream_enc_residual");

    stream.enc_residual_buf = ggml_backend_alloc_ctx_tensors(stream.enc_residual_ctx, stream.ctx->backend);
    if (stream.enc_residual_buf == nullptr) {
        free_stream_encoder_residual(stream);
        return false;
    }
    return true;
}

static ggml_cgraph * build_stream_adapter_append_graph(
    voxtral_stream & stream,
    ggml_context   * gctx,
    int32_t          new_enc_seq,
    int32_t          usable_enc_seq,
    int32_t          leftover_enc_seq,
    int32_t          dec_dst_offset) {

    voxtral_context * ctx = stream.ctx;
    ggml_cgraph * gf = ggml_new_graph(gctx);

    ggml_tensor * new_enc = ggml_view_2d(
        gctx,
        ctx->encoder_chunk_output,
        VOXTRAL_ENC_DIM,
        new_enc_seq,
        ctx->encoder_chunk_output->nb[1],
        0);

    ggml_tensor * merged = new_enc;
    if (stream.enc_residual_count > 0) {
        ggml_tensor * residual = ggml_view_2d(
            gctx,
            stream.enc_residual_tensor,
            VOXTRAL_ENC_DIM,
            stream.enc_residual_count,
            stream.enc_residual_tensor->nb[1],
            0);
        merged = ggml_concat(gctx, residual, new_enc, 1);
    }

    if (usable_enc_seq > 0) {
        ggml_tensor * usable = merged;
        if (usable_enc_seq != stream.enc_residual_count + new_enc_seq) {
            usable = ggml_view_2d(
                gctx,
                merged,
                VOXTRAL_ENC_DIM,
                usable_enc_seq,
                merged->nb[1],
                0);
        }

        ggml_tensor * x = build_adapter_project_tensor(ctx, gctx, usable, usable_enc_seq);
        ggml_tensor * dec_mem_view = ggml_view_2d(
            gctx,
            ctx->decoder_memory,
            VOXTRAL_DEC_DIM,
            usable_enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR,
            ctx->decoder_memory->nb[1],
            (size_t) dec_dst_offset * ctx->decoder_memory->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, x, dec_mem_view));
    }

    if (leftover_enc_seq > 0) {
        ggml_tensor * leftover = ggml_view_2d(
            gctx,
            merged,
            VOXTRAL_ENC_DIM,
            leftover_enc_seq,
            merged->nb[1],
            (size_t) usable_enc_seq * merged->nb[1]);
        ggml_tensor * residual_dst = ggml_view_2d(
            gctx,
            stream.enc_residual_tensor,
            VOXTRAL_ENC_DIM,
            leftover_enc_seq,
            stream.enc_residual_tensor->nb[1],
            0);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, leftover, residual_dst));
    }

    return gf;
}

static bool run_stream_adapter_append(
    voxtral_stream * stream,
    int32_t          new_enc_seq,
    int32_t          dec_dst_offset,
    int32_t        * out_dec_tokens) {

    if (out_dec_tokens != nullptr) {
        *out_dec_tokens = 0;
    }
    if (stream == nullptr || stream->ctx == nullptr || new_enc_seq <= 0) {
        return true;
    }
    if (!ensure_stream_encoder_residual(*stream)) {
        LOG_ERR(stream->ctx, "stream adapter append: failed to allocate residual tensor");
        return false;
    }

    const int32_t total_enc = stream->enc_residual_count + new_enc_seq;
    const int32_t usable_enc = (total_enc / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
    const int32_t leftover_enc = total_enc - usable_enc;
    if (leftover_enc > VOXTRAL_DOWNSAMPLE_FACTOR) {
        LOG_ERR(stream->ctx, "stream adapter append: leftover=%d exceeds residual capacity", leftover_enc);
        return false;
    }

    if (usable_enc > 0) {
        if (!ensure_decoder_memory_capacity(stream->ctx, dec_dst_offset + usable_enc / VOXTRAL_DOWNSAMPLE_FACTOR)) {
            LOG_ERR(
                stream->ctx,
                "stream adapter append: failed to grow decoder memory to %d",
                dec_dst_offset + usable_enc / VOXTRAL_DOWNSAMPLE_FACTOR);
            return false;
        }
    }
    if (usable_enc <= 0 && leftover_enc <= 0) {
        return true;
    }

    static thread_local std::vector<uint8_t> stream_adapter_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
    if (stream_adapter_meta_buf.size() < meta_size) {
        stream_adapter_meta_buf.resize(meta_size);
    }

    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ stream_adapter_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);
    ggml_cgraph * gf = build_stream_adapter_append_graph(
        *stream, gctx, new_enc_seq, usable_enc, leftover_enc, dec_dst_offset);

    ggml_backend_sched_reset(stream->ctx->sched_adapter);
    if (!ggml_backend_sched_alloc_graph(stream->ctx->sched_adapter, gf)) {
        LOG_ERR(stream->ctx, "stream adapter append: failed to allocate graph");
        ggml_free(gctx);
        return false;
    }

    ggml_backend_sched_graph_compute(stream->ctx->sched_adapter, gf);
    ggml_backend_sched_reset(stream->ctx->sched_adapter);
    ggml_free(gctx);

    if (usable_enc > 0) {
        stream->ctx->dec_seq_len = std::max<int32_t>(
            stream->ctx->dec_seq_len,
            dec_dst_offset + usable_enc / VOXTRAL_DOWNSAMPLE_FACTOR);
        if (out_dec_tokens != nullptr) {
            *out_dec_tokens = usable_enc / VOXTRAL_DOWNSAMPLE_FACTOR;
        }
    }

    stream->enc_residual_count = leftover_enc;
    return true;
}

static int32_t clamp_delay_ms(int32_t value) {
    int32_t clamped = value <= 0 ? VOXTRAL_TRANSCRIPTION_DELAY_MS : value;
    clamped = std::max<int32_t>(80, clamped);
    clamped = std::min<int32_t>(2400, clamped);
    clamped = (clamped / 80) * 80;
    return std::max<int32_t>(80, clamped);
}

static int32_t delay_ms_to_tokens(int32_t delay_ms) {
    return clamp_delay_ms(delay_ms) / 80;
}

static bool decode_visible_token_piece(
    const voxtral_model & model,
    int32_t token_id,
    std::string & out_text) {

    out_text.clear();
    if (token_id == VOXTRAL_TOKEN_EOS ||
        token_id == VOXTRAL_TOKEN_STREAMING_PAD ||
        token_id == VOXTRAL_TOKEN_STREAMING_WORD) {
        return false;
    }
    if (token_id < model.tokenizer_num_special_tokens) {
        return false;
    }
    if (model.tokenizer_special_ranks.find(token_id) != model.tokenizer_special_ranks.end()) {
        return false;
    }

    const std::string & piece = token_bytes_for_id(model, token_id);
    if (piece.empty()) {
        return false;
    }
    bool has_visible = false;
    for (unsigned char ch : piece) {
        if (!std::iscntrl(ch) || ch == '\n' || ch == '\t' || ch == '\r') {
            has_visible = true;
            break;
        }
    }
    if (!has_visible) {
        return false;
    }
    out_text = piece;
    return true;
}

static void emit_stream_piece_event(
    voxtral_stream & stream,
    int32_t begin_token_index,
    int32_t end_token_index,
    const std::string & text,
    std::vector<voxtral_stream_event> & out_events) {

    voxtral_stream_event ev;
    ev.kind = voxtral_stream_event_kind::piece_commit;
    ev.begin_sec = (double) (stream.timeline_token_base + begin_token_index) / VOXTRAL_FRAME_RATE;
    ev.end_sec = (double) (stream.timeline_token_base + end_token_index) / VOXTRAL_FRAME_RATE;
    ev.text = text;
    out_events.push_back(std::move(ev));
}

static void flush_stream_piece_buffer(
    voxtral_stream & stream,
    std::vector<voxtral_stream_event> & out_events) {

    if (stream.pending_piece_text.empty() ||
        stream.pending_piece_begin_token_index < 0 ||
        stream.pending_piece_end_token_index <= stream.pending_piece_begin_token_index) {
        stream.pending_piece_text.clear();
        stream.pending_piece_begin_token_index = -1;
        stream.pending_piece_end_token_index = -1;
        return;
    }

    emit_stream_piece_event(
        stream,
        stream.pending_piece_begin_token_index,
        stream.pending_piece_end_token_index,
        stream.pending_piece_text,
        out_events);

    stream.pending_piece_text.clear();
    stream.pending_piece_begin_token_index = -1;
    stream.pending_piece_end_token_index = -1;
}

static void append_stream_piece_token(
    voxtral_stream & stream,
    int32_t token_index,
    const std::string & text) {

    if (stream.pending_piece_begin_token_index < 0) {
        stream.pending_piece_begin_token_index = token_index;
    }
    stream.pending_piece_end_token_index = token_index + 1;
    stream.pending_piece_text.append(text);
}

static bool pending_piece_ends_sentence(const std::string & text) {
    for (size_t i = text.size(); i > 0; --i) {
        const unsigned char ch = static_cast<unsigned char>(text[i - 1]);
        if (std::isspace(ch)) {
            continue;
        }
        return ch == '.' || ch == ';' || ch == '?' || ch == '!';
    }
    return false;
}

static bool text_begins_sentence(const std::string & text) {
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (std::isspace(ch) || ch == '"' || ch == '\'' || ch == '(' || ch == '[') {
            ++i;
            continue;
        }
        return std::isupper(ch) != 0;
    }
    return false;
}

struct voxtral_stream_conv_debug_state {
    bool conv_stem_initialized = false;
    std::vector<float> mel_tail;
    std::vector<float> conv0_tail;
    std::vector<float> conv0_residual;
    int32_t conv0_residual_count = 0;
};

static voxtral_stream_conv_debug_state capture_stream_conv_debug_state(const voxtral_stream & stream) {
    voxtral_stream_conv_debug_state state;
    state.conv_stem_initialized = stream.conv_stem_initialized;
    state.mel_tail = stream.mel_tail;
    state.conv0_residual_count = stream.conv0_residual_count;
    ggml_tensor * active_tail = const_cast<ggml_tensor *>(stream.conv0_tail_tensors[(size_t) stream.conv_state_slot]);
    if (active_tail != nullptr) {
        std::string ignored_error;
        if (!read_tensor_2d_columns_f32(active_tail, 0, 2, state.conv0_tail, &ignored_error)) {
            state.conv0_tail = stream.conv0_tail;
        }
    } else {
        state.conv0_tail = stream.conv0_tail;
    }
    ggml_tensor * active_residual = const_cast<ggml_tensor *>(stream.conv0_residual_tensors[(size_t) stream.conv_state_slot]);
    if (stream.conv0_residual_count > 0 && active_residual != nullptr) {
        std::string ignored_error;
        if (!read_tensor_f32(active_residual, state.conv0_residual, &ignored_error)) {
            state.conv0_residual = stream.conv0_residual;
        }
    } else {
        state.conv0_residual = stream.conv0_residual;
    }
    return state;
}

static void log_stream_conv_input_upload_debug(
    voxtral_context * ctx,
    ggml_tensor     * tensor,
    const float     * host_col_major,
    int32_t           in_len,
    int32_t           in_channels) {

    if (ctx == nullptr ||
        tensor == nullptr ||
        host_col_major == nullptr ||
        in_len <= 0 ||
        in_channels <= 0 ||
        static_cast<int>(ctx->log_level) < static_cast<int>(voxtral_log_level::debug)) {
        return;
    }

    std::vector<float> device_col_major;
    std::string error;
    if (!read_tensor_2d_columns_f32(tensor, 0, in_channels, device_col_major, &error)) {
        LOG_ERR(ctx, "stream conv input debug: failed to read uploaded tensor: %s", error.c_str());
        return;
    }
    if (device_col_major.size() != (size_t) in_len * (size_t) in_channels) {
        LOG_ERR(
            ctx,
            "stream conv input debug: size mismatch host=%zu device=%zu",
            (size_t) in_len * (size_t) in_channels,
            device_col_major.size());
        return;
    }

    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    int32_t max_row = 0;
    int32_t max_col = 0;
    for (int32_t col = 0; col < in_channels; ++col) {
        for (int32_t row = 0; row < in_len; ++row) {
            const size_t idx = (size_t) col * (size_t) in_len + (size_t) row;
            const double abs_diff = std::fabs((double) device_col_major[idx] - (double) host_col_major[idx]);
            mean_abs_diff += abs_diff;
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                max_row = row;
                max_col = col;
            }
        }
    }
    mean_abs_diff /= (double) in_len * (double) in_channels;

    LOG_INFO(
        ctx,
        "stream conv input debug: len=%d chans=%d max_abs_diff=%.9g mean_abs_diff=%.9g sample[%d,%d] host=%.9g device=%.9g",
        in_len,
        in_channels,
        max_abs_diff,
        mean_abs_diff,
        max_row,
        max_col,
        host_col_major[(size_t) max_col * (size_t) in_len + (size_t) max_row],
        device_col_major[(size_t) max_col * (size_t) in_len + (size_t) max_row]);
}

static bool run_device_causal_conv1d_col_major(
    voxtral_context      * ctx,
    const float          * input_col_major,
    int32_t                in_channels,
    int32_t                in_len,
    ggml_tensor          * weight,
    ggml_tensor          * bias,
    int32_t                out_channels,
    int32_t                kernel_size,
    int32_t                stride,
    bool                   apply_gelu,
    std::vector<float>   * out_col_major,
    int32_t              * out_len) {

    if (ctx == nullptr || input_col_major == nullptr || weight == nullptr || out_col_major == nullptr || out_len == nullptr) {
        return false;
    }

    *out_len = 0;
    out_col_major->clear();

    const auto dims = compute_causal_conv1d_dims(in_len, kernel_size, stride);
    if (dims.out_len <= 0) {
        return true;
    }

    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
    std::vector<uint8_t> meta_buf(meta_size);
    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);
    if (gctx == nullptr) {
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
    ggml_tensor * input = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, in_len, in_channels, 1);
    ggml_set_name(input, "stream_conv_device_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, input, ctx->backend);

    int32_t device_out_len = 0;
    ggml_tensor * y = causal_conv1d_graph(
        gctx,
        input,
        in_len,
        weight,
        bias,
        out_channels,
        kernel_size,
        stride,
        device_out_len);
    if (y == nullptr) {
        ggml_free(gctx);
        return false;
    }
    if (apply_gelu) {
        y = ggml_gelu_erf(gctx, y);
    }

    ggml_tensor * output = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, device_out_len, out_channels, 1);
    ggml_set_name(output, "stream_conv_device_output");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, output, ctx->backend);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, y, output));

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        ggml_free(gctx);
        return false;
    }

    ggml_backend_tensor_set(input, input_col_major, 0, (size_t) in_channels * (size_t) in_len * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);

    out_col_major->resize((size_t) out_channels * (size_t) device_out_len);
    ggml_backend_tensor_get(output, out_col_major->data(), 0, out_col_major->size() * sizeof(float));
    *out_len = device_out_len;

    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);
    return true;
}

static void run_stream_conv_stem_device_debug(
    voxtral_context                         * ctx,
    const voxtral_stream_conv_debug_state   & state,
    const float                             * mel_new,
    int32_t                                   n_new_mel,
    const float                             * cpu_out_row_major,
    int32_t                                   cpu_out_len) {

    if (ctx == nullptr || mel_new == nullptr || cpu_out_row_major == nullptr || cpu_out_len <= 0) {
        return;
    }
    if (static_cast<int>(ctx->log_level) < static_cast<int>(voxtral_log_level::debug)) {
        return;
    }

    const bool first_chunk = !state.conv_stem_initialized;
    const int32_t dim = VOXTRAL_ENC_DIM;

    std::vector<float> conv0_new;
    int32_t conv0_new_len = 0;

    if (first_chunk) {
        std::vector<float> conv_in((size_t) VOXTRAL_NUM_MEL_BINS * (size_t) n_new_mel);
        for (int32_t frame = 0; frame < n_new_mel; ++frame) {
            const float * src = mel_new + (size_t) frame * VOXTRAL_NUM_MEL_BINS;
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                conv_in[(size_t) m * (size_t) n_new_mel + (size_t) frame] = src[m];
            }
        }
        if (!run_device_causal_conv1d_col_major(
                ctx,
                conv_in.data(),
                VOXTRAL_NUM_MEL_BINS,
                n_new_mel,
                ctx->model->enc_conv0_weight,
                ctx->model->enc_conv0_bias,
                dim,
                3,
                1,
                /*apply_gelu=*/true,
                &conv0_new,
                &conv0_new_len)) {
            LOG_ERR(ctx, "stream conv device debug: conv0 failed");
            return;
        }
    } else {
        const int32_t padded_mel_len = n_new_mel + 2;
        std::vector<float> conv_in((size_t) VOXTRAL_NUM_MEL_BINS * (size_t) padded_mel_len);
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            conv_in[(size_t) m * (size_t) padded_mel_len + 0] = state.mel_tail[(size_t) m * 2u + 0u];
            conv_in[(size_t) m * (size_t) padded_mel_len + 1] = state.mel_tail[(size_t) m * 2u + 1u];
            for (int32_t f = 0; f < n_new_mel; ++f) {
                conv_in[(size_t) m * (size_t) padded_mel_len + (size_t) (f + 2)] =
                    mel_new[(size_t) f * VOXTRAL_NUM_MEL_BINS + (size_t) m];
            }
        }

        std::vector<float> conv0_full;
        int32_t conv0_full_len = 0;
        if (!run_device_causal_conv1d_col_major(
                ctx,
                conv_in.data(),
                VOXTRAL_NUM_MEL_BINS,
                padded_mel_len,
                ctx->model->enc_conv0_weight,
                ctx->model->enc_conv0_bias,
                dim,
                3,
                1,
                /*apply_gelu=*/true,
                &conv0_full,
                &conv0_full_len)) {
            LOG_ERR(ctx, "stream conv device debug: conv0 padded failed");
            return;
        }
        if (conv0_full_len < n_new_mel + 2) {
            LOG_ERR(ctx, "stream conv device debug: conv0 padded len too small (%d)", conv0_full_len);
            return;
        }

        conv0_new_len = n_new_mel;
        conv0_new.resize((size_t) dim * (size_t) conv0_new_len);
        for (int32_t d = 0; d < dim; ++d) {
            memcpy(
                conv0_new.data() + (size_t) d * (size_t) conv0_new_len,
                conv0_full.data() + (size_t) d * (size_t) conv0_full_len + 2,
                (size_t) conv0_new_len * sizeof(float));
        }
    }

    const int32_t prev_residual = state.conv0_residual_count;
    const int32_t total_avail = prev_residual + conv0_new_len;
    const int32_t new_residual = total_avail & 1;
    const int32_t feed_from_new = conv0_new_len - new_residual;
    const int32_t feed_total = prev_residual + feed_from_new;
    if (feed_total <= 0) {
        LOG_ERR(ctx, "stream conv device debug: feed_total <= 0 while cpu_out_len=%d", cpu_out_len);
        return;
    }

    std::vector<float> feed((size_t) dim * (size_t) feed_total);
    int32_t feed_pos = 0;
    if (prev_residual == 1) {
        for (int32_t d = 0; d < dim; ++d) {
            feed[(size_t) d * (size_t) feed_total] = state.conv0_residual[(size_t) d];
        }
        feed_pos = 1;
    }
    for (int32_t d = 0; d < dim; ++d) {
        memcpy(
            feed.data() + (size_t) d * (size_t) feed_total + (size_t) feed_pos,
            conv0_new.data() + (size_t) d * (size_t) conv0_new_len,
            (size_t) feed_from_new * sizeof(float));
    }

    std::vector<float> conv1_in;
    int32_t conv1_in_len = 0;
    int32_t conv1_discard = 0;
    if (first_chunk) {
        conv1_in = std::move(feed);
        conv1_in_len = feed_total;
    } else {
        conv1_in_len = feed_total + 2;
        conv1_in.resize((size_t) dim * (size_t) conv1_in_len);
        for (int32_t d = 0; d < dim; ++d) {
            conv1_in[(size_t) d * (size_t) conv1_in_len + 0] = state.conv0_tail[(size_t) d * 2u + 0u];
            conv1_in[(size_t) d * (size_t) conv1_in_len + 1] = state.conv0_tail[(size_t) d * 2u + 1u];
            memcpy(
                conv1_in.data() + (size_t) d * (size_t) conv1_in_len + 2u,
                feed.data() + (size_t) d * (size_t) feed_total,
                (size_t) feed_total * sizeof(float));
        }
        conv1_discard = 1;
    }

    std::vector<float> conv1_out;
    int32_t conv1_out_len = 0;
    if (!run_device_causal_conv1d_col_major(
            ctx,
            conv1_in.data(),
            dim,
            conv1_in_len,
            ctx->model->enc_conv1_weight,
            ctx->model->enc_conv1_bias,
            dim,
            3,
            2,
            /*apply_gelu=*/true,
            &conv1_out,
            &conv1_out_len)) {
        LOG_ERR(ctx, "stream conv device debug: conv1 failed");
        return;
    }

    const int32_t result_len = conv1_out_len - conv1_discard;
    if (result_len != cpu_out_len) {
        LOG_ERR(ctx, "stream conv device debug: len mismatch cpu=%d gpu=%d discard=%d",
            cpu_out_len, result_len, conv1_discard);
        return;
    }

    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    int32_t max_row = 0;
    int32_t max_col = 0;
    for (int32_t row = 0; row < result_len; ++row) {
        for (int32_t d = 0; d < dim; ++d) {
            const float gpu = conv1_out[(size_t) d * (size_t) conv1_out_len + (size_t) (conv1_discard + row)];
            const float cpu = cpu_out_row_major[(size_t) row * (size_t) dim + (size_t) d];
            const double abs_diff = std::fabs((double) gpu - (double) cpu);
            mean_abs_diff += abs_diff;
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                max_row = row;
                max_col = d;
            }
        }
    }
    mean_abs_diff /= (double) result_len * (double) dim;
    const float cpu_sample = cpu_out_row_major[(size_t) max_row * (size_t) dim + (size_t) max_col];
    const float gpu_sample = conv1_out[(size_t) max_col * (size_t) conv1_out_len + (size_t) (conv1_discard + max_row)];
    LOG_INFO(
        ctx,
        "stream conv device debug: rows=%d max_abs_diff=%.9g mean_abs_diff=%.9g sample[%d,%d] cpu=%.9g gpu=%.9g",
        result_len,
        max_abs_diff,
        mean_abs_diff,
        max_row,
        max_col,
        cpu_sample,
        gpu_sample);
}

static bool run_stream_conv_stem_device(
    voxtral_stream & stream,
    const float * mel_new,
    int32_t n_new_mel,
    int32_t * out_len,
    std::string * error) {

    voxtral_context * ctx = stream.ctx;
    if (out_len != nullptr) {
        *out_len = 0;
    }
    if (ctx == nullptr || mel_new == nullptr || n_new_mel <= 0 || out_len == nullptr) {
        if (error) {
            *error = "invalid stream conv device input";
        }
        return false;
    }
    if (ctx->gpu_type == voxtral_gpu_backend::none || ctx->backend == nullptr) {
        if (error) {
            *error = "stream conv device path requires a GPU backend";
        }
        return false;
    }
    if (!ensure_stream_conv_state(stream)) {
        if (error) {
            *error = "stream conv device path failed to allocate persistent stream state";
        }
        return false;
    }
    const int32_t dim = VOXTRAL_ENC_DIM;
    const bool first_chunk = !stream.conv_stem_initialized;
    const int32_t conv0_input_len = first_chunk ? n_new_mel : (n_new_mel + 2);
    std::vector<float> & conv0_input = stream.conv0_input_scratch;
    conv0_input.resize((size_t) VOXTRAL_NUM_MEL_BINS * (size_t) conv0_input_len);
    if (first_chunk) {
        for (int32_t frame = 0; frame < n_new_mel; ++frame) {
            const float * src = mel_new + (size_t) frame * VOXTRAL_NUM_MEL_BINS;
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                conv0_input[(size_t) m * (size_t) conv0_input_len + (size_t) frame] = src[m];
            }
        }
    } else {
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            conv0_input[(size_t) m * (size_t) conv0_input_len + 0] = stream.mel_tail[(size_t) m * 2u + 0u];
            conv0_input[(size_t) m * (size_t) conv0_input_len + 1] = stream.mel_tail[(size_t) m * 2u + 1u];
            for (int32_t frame = 0; frame < n_new_mel; ++frame) {
                conv0_input[(size_t) m * (size_t) conv0_input_len + (size_t) (frame + 2)] =
                    mel_new[(size_t) frame * VOXTRAL_NUM_MEL_BINS + (size_t) m];
            }
        }
    }

    const int32_t prev_residual = stream.conv0_residual_count;
    const int32_t conv0_new_len = n_new_mel;
    const int32_t total_avail = prev_residual + conv0_new_len;
    const int32_t new_residual = total_avail & 1;
    const int32_t feed_from_new = conv0_new_len - new_residual;
    const int32_t feed_total = prev_residual + feed_from_new;
    const int32_t tail_count = std::min<int32_t>(2, std::max<int32_t>(0, feed_total));
    const int32_t conv1_discard = first_chunk ? 0 : 1;

    static thread_local std::vector<uint8_t> stream_conv_meta_buf;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 6 +
                             ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 6, false);
    if (stream_conv_meta_buf.size() < meta_size) {
        stream_conv_meta_buf.resize(meta_size);
    }
    ggml_init_params p = {
        /*.mem_size  =*/ meta_size,
        /*.mem_buffer=*/ stream_conv_meta_buf.data(),
        /*.no_alloc  =*/ true,
    };
    ggml_context * gctx = ggml_init(p);
    if (gctx == nullptr) {
        if (error) {
            *error = "failed to allocate ggml context for stream conv device path";
        }
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 6, false);

    ggml_tensor * conv0_input_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, conv0_input_len, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(conv0_input_t, "stream_conv0_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, conv0_input_t, ctx->backend);

    ggml_tensor * conv0_full = causal_conv1d_graph(
        gctx,
        conv0_input_t,
        conv0_input_len,
        ctx->model->enc_conv0_weight,
        ctx->model->enc_conv0_bias,
        dim,
        3,
        1,
        *out_len);
    if (conv0_full == nullptr) {
        ggml_free(gctx);
        if (error) {
            *error = "stream conv device conv0 graph build failed";
        }
        return false;
    }
    const int32_t conv0_full_len = *out_len;
    conv0_full = ggml_gelu_erf(gctx, conv0_full);
    ggml_tensor * conv0_full_2d = ggml_permute(gctx, conv0_full, 1, 0, 2, 3);
    conv0_full_2d = ggml_cont(gctx, conv0_full_2d);
    conv0_full_2d = ggml_reshape_2d(gctx, conv0_full_2d, dim, conv0_full_len);

    ggml_tensor * conv0_new = first_chunk
        ? conv0_full_2d
        : ggml_view_2d(
            gctx,
            conv0_full_2d,
            dim,
            conv0_new_len,
            conv0_full_2d->nb[1],
            2 * conv0_full_2d->nb[1]);

    ggml_tensor * active_residual_tensor = active_stream_conv0_residual_tensor(stream);
    ggml_tensor * active_tail_tensor = active_stream_conv0_tail_tensor(stream);
    ggml_tensor * next_residual_tensor = inactive_stream_conv0_residual_tensor(stream);
    ggml_tensor * next_tail_tensor = inactive_stream_conv0_tail_tensor(stream);

    ggml_tensor * prev_residual_t = nullptr;
    ggml_tensor * prev_tail_t = nullptr;
    if (prev_residual == 1) {
        prev_residual_t = ggml_view_2d(
            gctx,
            active_residual_tensor,
            dim,
            1,
            active_residual_tensor->nb[1],
            0);
        ggml_set_name(prev_residual_t, "stream_prev_residual_view");
    }
    if (!first_chunk) {
        prev_tail_t = ggml_view_2d(
            gctx,
            active_tail_tensor,
            dim,
            2,
            active_tail_tensor->nb[1],
            0);
        ggml_set_name(prev_tail_t, "stream_prev_tail_view");
    }

    ggml_tensor * residual_state = nullptr;
    if (new_residual == 1) {
        ggml_tensor * residual_src = ggml_view_2d(
            gctx,
            conv0_new,
            dim,
            1,
            conv0_new->nb[1],
            (size_t) feed_from_new * conv0_new->nb[1]);
        residual_state = residual_src;
    } else {
        residual_state = ggml_scale(gctx, active_residual_tensor, 0.0f);
    }
    ggml_build_forward_expand(gf, ggml_cpy(gctx, residual_state, next_residual_tensor));

    ggml_tensor * feed = nullptr;
    if (feed_from_new > 0) {
        feed = ggml_view_2d(
            gctx,
            conv0_new,
            dim,
            feed_from_new,
            conv0_new->nb[1],
            0);
    }
    if (prev_residual == 1) {
        feed = feed != nullptr ? ggml_concat(gctx, prev_residual_t, feed, 1) : prev_residual_t;
    }

    ggml_tensor * tail_state = nullptr;
    if (feed != nullptr) {
        tail_state = ggml_scale(gctx, active_tail_tensor, 0.0f);
        if (tail_count > 0) {
            ggml_tensor * tail_src = ggml_view_2d(
                gctx,
                feed,
                dim,
                tail_count,
                feed->nb[1],
                (size_t) (feed_total - tail_count) * feed->nb[1]);
            const size_t tail_offset = (size_t) (2 - tail_count) * (size_t) active_tail_tensor->nb[1];
            tail_state = ggml_set(
                gctx,
                tail_state,
                tail_src,
                active_tail_tensor->nb[1],
                active_tail_tensor->nb[2],
                active_tail_tensor->nb[3],
                tail_offset);
        }
    } else if (!first_chunk) {
        tail_state = ggml_scale(gctx, active_tail_tensor, 0.0f);
    }
    if (tail_state != nullptr) {
        ggml_build_forward_expand(gf, ggml_cpy(gctx, tail_state, next_tail_tensor));
    }

    int32_t result_len = 0;
    if (feed_total > 0 && feed != nullptr) {
        ggml_tensor * conv1_in = first_chunk ? feed : ggml_concat(gctx, prev_tail_t, feed, 1);
        const int32_t conv1_in_len = feed_total + (first_chunk ? 0 : 2);
        ggml_tensor * conv1_in_t = ggml_transpose(gctx, conv1_in);
        conv1_in_t = ggml_cont(gctx, conv1_in_t);
        conv1_in_t = ggml_reshape_3d(gctx, conv1_in_t, conv1_in_len, dim, 1);

        int32_t conv1_full_len = 0;
        ggml_tensor * conv1_full = causal_conv1d_graph(
            gctx,
            conv1_in_t,
            conv1_in_len,
            ctx->model->enc_conv1_weight,
            ctx->model->enc_conv1_bias,
            dim,
            3,
            2,
            conv1_full_len);
        if (conv1_full == nullptr) {
            ggml_free(gctx);
            if (error) {
                *error = "stream conv device conv1 graph build failed";
            }
            return false;
        }
        conv1_full = ggml_gelu_erf(gctx, conv1_full);
        ggml_tensor * conv1_full_2d = ggml_permute(gctx, conv1_full, 1, 0, 2, 3);
        conv1_full_2d = ggml_cont(gctx, conv1_full_2d);
        conv1_full_2d = ggml_reshape_2d(gctx, conv1_full_2d, dim, conv1_full_len);

        result_len = conv1_full_len - conv1_discard;
        if (result_len < 0 || result_len > VOXTRAL_MAX_ENC_CHUNK) {
            ggml_free(gctx);
            if (error) {
                *error = "stream conv device produced invalid encoder chunk length";
            }
            return false;
        }

        if (result_len > 0) {
            ggml_tensor * result = conv1_discard > 0
                ? ggml_view_2d(
                    gctx,
                    conv1_full_2d,
                    dim,
                    result_len,
                    conv1_full_2d->nb[1],
                    (size_t) conv1_discard * conv1_full_2d->nb[1])
                : conv1_full_2d;
            ggml_tensor * encoder_chunk_input_view = ggml_view_2d(
                gctx,
                ctx->encoder_chunk_input,
                dim,
                result_len,
                ctx->encoder_chunk_input->nb[1],
                0);
            ggml_build_forward_expand(gf, ggml_cpy(gctx, result, encoder_chunk_input_view));
        }
    }

    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!ggml_backend_sched_alloc_graph(ctx->sched_encoder, gf)) {
        ggml_free(gctx);
        if (error) {
            *error = "stream conv device failed to allocate graph";
        }
        return false;
    }

    ggml_backend_tensor_set(conv0_input_t, conv0_input.data(), 0, conv0_input.size() * sizeof(float));
    if (static_cast<int>(ctx->log_level) >= static_cast<int>(voxtral_log_level::debug)) {
        log_stream_conv_input_upload_debug(
            ctx,
            conv0_input_t,
            conv0_input.data(),
            conv0_input_len,
            VOXTRAL_NUM_MEL_BINS);
    }

    ggml_backend_sched_graph_compute(ctx->sched_encoder, gf);
    stream.conv_state_slot = 1 - stream.conv_state_slot;

    std::fill(stream.mel_tail.begin(), stream.mel_tail.end(), 0.0f);
    const int32_t mel_tail_count = std::min<int32_t>(2, n_new_mel);
    const int32_t mel_tail_start = n_new_mel - mel_tail_count;
    for (int32_t f = 0; f < mel_tail_count; ++f) {
        const float * src = mel_new + (size_t) (mel_tail_start + f) * VOXTRAL_NUM_MEL_BINS;
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            stream.mel_tail[(size_t) m * 2u + (size_t) (2 - mel_tail_count + f)] = src[m];
        }
    }

    stream.conv0_residual_count = new_residual;
    if (static_cast<int>(ctx->log_level) >= static_cast<int>(voxtral_log_level::debug)) {
        if (new_residual == 1 && active_stream_conv0_residual_tensor(stream) != nullptr) {
            std::vector<float> residual_values;
            std::string ignored_error;
            if (read_tensor_f32(active_stream_conv0_residual_tensor(stream), residual_values, &ignored_error) &&
                residual_values.size() == (size_t) dim) {
                stream.conv0_residual = std::move(residual_values);
            }
        }
        if (active_stream_conv0_tail_tensor(stream) != nullptr) {
            std::vector<float> tail_values;
            std::string ignored_error;
            if (read_tensor_2d_columns_f32(active_stream_conv0_tail_tensor(stream), 0, 2, tail_values, &ignored_error) &&
                tail_values.size() == (size_t) dim * 2u) {
                stream.conv0_tail = std::move(tail_values);
            }
        }
    }

    stream.conv_stem_initialized = true;
    *out_len = result_len;

    ggml_backend_sched_reset(ctx->sched_encoder);
    ggml_free(gctx);
    return true;
}

static float * stream_conv_stem_cpu(
    voxtral_stream & stream,
    const float * mel_new,  // [n_new_mel, n_mel] row-major
    int32_t n_new_mel,
    int32_t * out_len) {

    voxtral_context * ctx = stream.ctx;
    *out_len = 0;
    if (ctx == nullptr || mel_new == nullptr || n_new_mel <= 0) {
        return nullptr;
    }

    const int32_t dim = VOXTRAL_ENC_DIM;
    bool first_chunk = false;
    std::vector<float> & conv0_new = stream.cpu_conv0_new_scratch;
    int32_t conv0_new_len = 0;

    if (!stream.conv_stem_initialized) {
        first_chunk = true;

        std::vector<float> & conv_in = stream.cpu_conv_in_scratch;
        conv_in.resize((size_t) VOXTRAL_NUM_MEL_BINS * (size_t) n_new_mel);
        for (int32_t frame = 0; frame < n_new_mel; ++frame) {
            const float * src = mel_new + (size_t) frame * VOXTRAL_NUM_MEL_BINS;
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                conv_in[(size_t) m * (size_t) n_new_mel + (size_t) frame] = src[m];
            }
        }

        conv0_new_len = n_new_mel;
        conv0_new.resize((size_t) dim * (size_t) conv0_new_len);
        cpu_causal_conv1d_col_major(
            conv0_new.data(),
            conv_in.data(),
            ctx->enc_conv0_weight_cpu.data(),
            ctx->enc_conv0_bias_cpu.data(),
            VOXTRAL_NUM_MEL_BINS,
            dim,
            n_new_mel,
            3,
            1);
        gelu_erf_inplace(conv0_new.data(), conv0_new.size());

        std::fill(stream.mel_tail.begin(), stream.mel_tail.end(), 0.0f);
        const int32_t tail_count = std::min<int32_t>(2, n_new_mel);
        const int32_t tail_start = n_new_mel - tail_count;
        for (int32_t f = 0; f < tail_count; ++f) {
            const float * src = mel_new + (size_t) (tail_start + f) * VOXTRAL_NUM_MEL_BINS;
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                stream.mel_tail[(size_t) m * 2u + (size_t) (2 - tail_count + f)] = src[m];
            }
        }

        stream.conv_stem_initialized = true;
    } else {
        const int32_t padded_mel_len = n_new_mel + 2;
        std::vector<float> & conv_in = stream.cpu_conv_in_scratch;
        conv_in.resize((size_t) VOXTRAL_NUM_MEL_BINS * (size_t) padded_mel_len);
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            conv_in[(size_t) m * (size_t) padded_mel_len + 0] = stream.mel_tail[(size_t) m * 2u + 0u];
            conv_in[(size_t) m * (size_t) padded_mel_len + 1] = stream.mel_tail[(size_t) m * 2u + 1u];
            for (int32_t f = 0; f < n_new_mel; ++f) {
                conv_in[(size_t) m * (size_t) padded_mel_len + (size_t) (f + 2)] =
                    mel_new[(size_t) f * VOXTRAL_NUM_MEL_BINS + (size_t) m];
            }
        }

        std::vector<float> & conv0_full = stream.cpu_conv0_full_scratch;
        conv0_full.resize((size_t) dim * (size_t) padded_mel_len);
        cpu_causal_conv1d_col_major(
            conv0_full.data(),
            conv_in.data(),
            ctx->enc_conv0_weight_cpu.data(),
            ctx->enc_conv0_bias_cpu.data(),
            VOXTRAL_NUM_MEL_BINS,
            dim,
            padded_mel_len,
            3,
            1);
        gelu_erf_inplace(conv0_full.data(), conv0_full.size());

        conv0_new_len = n_new_mel;
        conv0_new.resize((size_t) dim * (size_t) conv0_new_len);
        for (int32_t d = 0; d < dim; ++d) {
            memcpy(
                conv0_new.data() + (size_t) d * (size_t) conv0_new_len,
                conv0_full.data() + (size_t) d * (size_t) padded_mel_len + 2,
                (size_t) conv0_new_len * sizeof(float));
        }

        const int32_t tail_count = std::min<int32_t>(2, n_new_mel);
        const int32_t tail_start = n_new_mel - tail_count;
        std::fill(stream.mel_tail.begin(), stream.mel_tail.end(), 0.0f);
        for (int32_t f = 0; f < tail_count; ++f) {
            const float * src = mel_new + (size_t) (tail_start + f) * VOXTRAL_NUM_MEL_BINS;
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                stream.mel_tail[(size_t) m * 2u + (size_t) (2 - tail_count + f)] = src[m];
            }
        }
    }

    const int32_t prev_residual = stream.conv0_residual_count;
    const int32_t total_avail = prev_residual + conv0_new_len;
    const int32_t new_residual = total_avail & 1;
    const int32_t feed_from_new = conv0_new_len - new_residual;
    const int32_t feed_total = prev_residual + feed_from_new;

    if (feed_total <= 0) {
        if (new_residual && conv0_new_len > 0) {
            for (int32_t d = 0; d < dim; ++d) {
                stream.conv0_residual[(size_t) d] =
                    conv0_new[(size_t) d * (size_t) conv0_new_len + (size_t) (conv0_new_len - 1)];
            }
        }
        stream.conv0_residual_count = new_residual;
        return nullptr;
    }

    std::vector<float> & feed = stream.cpu_feed_scratch;
    feed.resize((size_t) dim * (size_t) feed_total);
    int32_t feed_pos = 0;
    if (prev_residual == 1) {
        for (int32_t d = 0; d < dim; ++d) {
            feed[(size_t) d * (size_t) feed_total] = stream.conv0_residual[(size_t) d];
        }
        feed_pos = 1;
    }
    for (int32_t d = 0; d < dim; ++d) {
        memcpy(
            feed.data() + (size_t) d * (size_t) feed_total + (size_t) feed_pos,
            conv0_new.data() + (size_t) d * (size_t) conv0_new_len,
            (size_t) feed_from_new * sizeof(float));
    }

    if (new_residual) {
        for (int32_t d = 0; d < dim; ++d) {
            stream.conv0_residual[(size_t) d] =
                conv0_new[(size_t) d * (size_t) conv0_new_len + (size_t) (conv0_new_len - 1)];
        }
    }
    stream.conv0_residual_count = new_residual;

    std::vector<float> & conv1_in = stream.cpu_conv1_in_scratch;
    int32_t conv1_in_len = 0;
    int32_t conv1_discard = 0;
    if (first_chunk) {
        conv1_in_len = feed_total;
        conv1_in.resize((size_t) dim * (size_t) conv1_in_len);
        memcpy(conv1_in.data(), feed.data(), conv1_in.size() * sizeof(float));
    } else {
        conv1_in_len = feed_total + 2;
        conv1_in.resize((size_t) dim * (size_t) conv1_in_len);
        for (int32_t d = 0; d < dim; ++d) {
            conv1_in[(size_t) d * (size_t) conv1_in_len + 0] = stream.conv0_tail[(size_t) d * 2u + 0u];
            conv1_in[(size_t) d * (size_t) conv1_in_len + 1] = stream.conv0_tail[(size_t) d * 2u + 1u];
            memcpy(
                conv1_in.data() + (size_t) d * (size_t) conv1_in_len + 2u,
                feed.data() + (size_t) d * (size_t) feed_total,
                (size_t) feed_total * sizeof(float));
        }
        conv1_discard = 1;
    }

    std::fill(stream.conv0_tail.begin(), stream.conv0_tail.end(), 0.0f);
    for (int32_t d = 0; d < dim; ++d) {
        const float * src = first_chunk ? conv1_in.data() + (size_t) d * (size_t) conv1_in_len
                                        : feed.data() + (size_t) d * (size_t) feed_total;
        const int32_t src_len = first_chunk ? conv1_in_len : feed_total;
        if (src_len >= 2) {
            stream.conv0_tail[(size_t) d * 2u + 0u] = src[(size_t) (src_len - 2)];
            stream.conv0_tail[(size_t) d * 2u + 1u] = src[(size_t) (src_len - 1)];
        } else if (src_len == 1) {
            stream.conv0_tail[(size_t) d * 2u + 1u] = src[0];
        }
    }

    const int32_t conv1_out_len = conv1_in_len / 2;
    std::vector<float> & conv1_out = stream.cpu_conv1_out_scratch;
    conv1_out.resize((size_t) dim * (size_t) conv1_out_len);
    cpu_causal_conv1d_col_major(
        conv1_out.data(),
        conv1_in.data(),
        ctx->enc_conv1_weight_cpu.data(),
        ctx->enc_conv1_bias_cpu.data(),
        dim,
        dim,
        conv1_in_len,
        3,
        2);
    gelu_erf_inplace(conv1_out.data(), conv1_out.size());

    const int32_t result_len = conv1_out_len - conv1_discard;
    if (result_len <= 0) {
        return nullptr;
    }

    float * result = new float[(size_t) result_len * (size_t) dim];
    for (int32_t row = 0; row < result_len; ++row) {
        for (int32_t d = 0; d < dim; ++d) {
            result[(size_t) row * (size_t) dim + (size_t) d] =
                conv1_out[(size_t) d * (size_t) conv1_out_len + (size_t) (conv1_discard + row)];
        }
    }
    *out_len = result_len;
    return result;
}

static bool stream_run_encoder_bounded(
    voxtral_stream & stream,
    int32_t max_new_mel_frames,
    std::string * error) {

    voxtral_context * ctx = stream.ctx;
    LOG_DBG(ctx, "stream_run_encoder: begin");
    const int32_t total_mel =
        stream.mel.mel_frame_offset + (int32_t) (stream.mel.mel.size() / VOXTRAL_NUM_MEL_BINS);
    if (stream.mel_cursor < stream.mel.mel_frame_offset) {
        stream.mel_cursor = stream.mel.mel_frame_offset;
    }
    const int32_t available_new_mel = total_mel - stream.mel_cursor;
    const int32_t delay_tokens = delay_ms_to_tokens(stream.params.delay_ms);
    const int32_t prompt_len = 1 + VOXTRAL_N_LEFT_PAD_TOKENS + delay_tokens;
    const int32_t first_chunk_min_mel = prompt_len * 8;
    const int32_t need_mel = stream.conv_stem_initialized ? stream.min_new_mel_frames : first_chunk_min_mel;

    if (!stream.finished && available_new_mel < need_mel) {
        LOG_DBG(ctx, "stream_run_encoder: waiting new_mel=%d need_mel=%d", available_new_mel, need_mel);
        return true;
    }
    if (available_new_mel <= 0) {
        LOG_DBG(ctx, "stream_run_encoder: no new mel");
        return true;
    }
    const int32_t new_mel = max_new_mel_frames > 0
        ? std::min(available_new_mel, max_new_mel_frames)
        : available_new_mel;
    const int32_t process_mel_end = stream.mel_cursor + new_mel;

    const int32_t mel_start = stream.mel_cursor - stream.mel.mel_frame_offset;
    const float * mel_ptr = stream.mel.mel.data() + (size_t) mel_start * VOXTRAL_NUM_MEL_BINS;
    const bool use_device_conv_stem = ctx->gpu_type != voxtral_gpu_backend::none;
    const bool run_conv_debug =
        static_cast<int>(ctx->log_level) >= static_cast<int>(voxtral_log_level::debug);
    const voxtral_stream_conv_debug_state conv_debug_state =
        run_conv_debug ? capture_stream_conv_debug_state(stream) : voxtral_stream_conv_debug_state{};
    std::unique_ptr<float[]> conv_out_cpu;
    std::unique_ptr<float[]> conv_debug_cpu_out;
    int32_t conv_debug_cpu_len = 0;

    int32_t conv_out_len = 0;
    if (use_device_conv_stem) {
        if (!run_stream_conv_stem_device(stream, mel_ptr, new_mel, &conv_out_len, error)) {
            return false;
        }
    } else {
        conv_out_cpu.reset(stream_conv_stem_cpu(stream, mel_ptr, new_mel, &conv_out_len));
    }
    stream.mel_cursor = process_mel_end;
    if (conv_out_len <= 0) {
        LOG_DBG(ctx, "stream_run_encoder: conv stem produced no output");
        incremental_mel_discard_before(stream.mel, stream.mel_cursor);
        return true;
    }
    if (use_device_conv_stem) {
        log_col_major_tensor_sample(
            ctx,
            "stream conv stem output",
            ctx->encoder_chunk_input,
            conv_out_len,
            VOXTRAL_ENC_DIM,
            conv_out_len);
        if (run_conv_debug) {
            voxtral_stream conv_debug_cpu_stream = stream;
            conv_debug_cpu_out.reset(stream_conv_stem_cpu(conv_debug_cpu_stream, mel_ptr, new_mel, &conv_debug_cpu_len));
            if (conv_debug_cpu_len == conv_out_len && conv_debug_cpu_out != nullptr) {
                run_stream_conv_stem_device_debug(
                    ctx,
                    conv_debug_state,
                    mel_ptr,
                    new_mel,
                    conv_debug_cpu_out.get(),
                    conv_debug_cpu_len);
            } else {
                LOG_ERR(ctx, "stream_run_encoder: cpu conv debug len mismatch cpu=%d gpu=%d",
                    conv_debug_cpu_len, conv_out_len);
            }
        }
    } else {
        log_row_major_sample(
            ctx,
            "stream conv stem output",
            conv_out_cpu.get(),
            conv_out_len,
            VOXTRAL_ENC_DIM,
            conv_out_len);
        if (run_conv_debug) {
            run_stream_conv_stem_device_debug(ctx, conv_debug_state, mel_ptr, new_mel, conv_out_cpu.get(), conv_out_len);
        }
    }

    if (stream.total_encoder_positions == 0 &&
        static_cast<int>(ctx->log_level) >= static_cast<int>(voxtral_log_level::debug)) {
        int32_t offline_seq_len = 0;
        if (run_encoder_chunk(ctx, mel_ptr, new_mel, 0, &offline_seq_len)) {
            LOG_DBG(ctx, "stream_run_encoder: offline encoder debug seq_len=%d", offline_seq_len);
        } else {
            LOG_ERR(ctx, "stream_run_encoder: offline encoder debug failed");
        }
    }

    LOG_DBG(ctx, "stream_run_encoder: run_encoder_incremental conv_out_len=%d total_encoder_positions=%d",
        conv_out_len, stream.total_encoder_positions);
    const bool encoder_ok = use_device_conv_stem
        ? run_encoder_incremental_from_chunk_input(ctx, conv_out_len, stream.total_encoder_positions)
        : run_encoder_incremental(ctx, conv_out_cpu.get(), conv_out_len, stream.total_encoder_positions);
    if (!encoder_ok) {
        if (error) {
            *error = "voxtral incremental encoder failed";
        }
        return false;
    }
    stream.total_encoder_positions += conv_out_len;
    int32_t new_audio_tokens = 0;
    LOG_DBG(
        ctx,
        "stream_run_encoder: adapter append conv_out=%d residual=%d total_audio_tokens=%d",
        conv_out_len,
        stream.enc_residual_count,
        stream.total_audio_tokens);
    if (!run_stream_adapter_append(&stream, conv_out_len, stream.total_audio_tokens, &new_audio_tokens)) {
        if (error) {
            *error = "voxtral adapter append failed";
        }
        return false;
    }
    stream.total_audio_tokens += new_audio_tokens;

    incremental_mel_discard_before(stream.mel, stream.mel_cursor);
    LOG_DBG(ctx, "stream_run_encoder: end total_audio_tokens=%d leftover=%d",
        stream.total_audio_tokens, stream.enc_residual_count);
    return true;
}

static bool stream_run_encoder(
    voxtral_stream & stream,
    std::string * error) {

    return stream_run_encoder_bounded(stream, 0, error);
}

static bool stream_run_decoder(
    voxtral_stream & stream,
    std::vector<voxtral_stream_event> & out_events,
    std::string * error) {

    voxtral_context * ctx = stream.ctx;
    const int32_t delay_tokens = delay_ms_to_tokens(stream.params.delay_ms);
    const int32_t prompt_len = 1 + VOXTRAL_N_LEFT_PAD_TOKENS + delay_tokens;

    if (!stream.decoder_started && stream.total_audio_tokens < prompt_len) {
        LOG_DBG(ctx, "stream_run_decoder: waiting total_audio_tokens=%d prompt_len=%d",
            stream.total_audio_tokens, prompt_len);
        return true;
    }

    auto handle_generated_token = [&](int32_t token_id, int32_t token_index) {
        std::string text;
        if (decode_visible_token_piece(*ctx->model, token_id, text)) {
            if (!stream.pending_piece_text.empty() &&
                pending_piece_ends_sentence(stream.pending_piece_text) &&
                text_begins_sentence(text)) {
                flush_stream_piece_buffer(stream, out_events);
            }
            append_stream_piece_token(stream, token_index, text);
            return;
        }

        flush_stream_piece_buffer(stream, out_events);
    };

    if (!stream.decoder_started) {
        LOG_DBG(ctx, "stream_run_decoder: prefill prompt_len=%d", prompt_len);
        clear_kv_cache(ctx);
        std::vector<int32_t> prompt_ids((size_t) prompt_len, VOXTRAL_TOKEN_STREAMING_PAD);
        prompt_ids[0] = VOXTRAL_TOKEN_BOS;

        if (prompt_len > 1 && !run_decoder_prefill(ctx, prompt_ids.data(), prompt_len - 1, nullptr, nullptr)) {
            if (error) {
                *error = "voxtral decoder prefill failed";
            }
            return false;
        }
        if (!run_decoder_step(ctx, prompt_ids[(size_t) prompt_len - 1], prompt_len - 1, prompt_len - 1, nullptr, &stream.prev_token)) {
            if (error) {
                *error = "voxtral decoder initial step failed";
            }
            return false;
        }

        handle_generated_token(stream.prev_token, stream.generated_token_index);
        stream.generated_token_index += 1;
        stream.eos_seen = (stream.prev_token == VOXTRAL_TOKEN_EOS);
        stream.gen_pos = prompt_len;
        stream.decoder_started = true;
    }

    int32_t steps = 0;
    while (!stream.eos_seen &&
           stream.gen_pos < stream.total_audio_tokens &&
           steps < std::max(1, stream.params.max_tokens_per_step)) {

        LOG_DBG(ctx, "stream_run_decoder: step gen_pos=%d prev_token=%d total_audio_tokens=%d",
            stream.gen_pos, stream.prev_token, stream.total_audio_tokens);
        if (!run_decoder_step_from_prev_token(ctx, stream.gen_pos, stream.gen_pos, nullptr, &stream.prev_token)) {
            if (error) {
                *error = "voxtral decoder step failed";
            }
            return false;
        }

        handle_generated_token(stream.prev_token, stream.generated_token_index);
        stream.generated_token_index += 1;
        stream.eos_seen = (stream.prev_token == VOXTRAL_TOKEN_EOS);
        stream.gen_pos += 1;
        steps += 1;
    }

    return true;
}

voxtral_stream * voxtral_stream_init(
    voxtral_context & ctx,
    const voxtral_stream_params & params) {

    auto * stream = new voxtral_stream();
    stream->ctx = &ctx;
    stream->params = params;
    stream->params.delay_ms = clamp_delay_ms(stream->params.delay_ms);
    if (stream->params.processing_interval_ms <= 0) {
        stream->params.processing_interval_ms = stream->params.delay_ms;
    }
    if (stream->params.right_pad_tokens <= 0) {
        stream->params.right_pad_tokens = VOXTRAL_N_RIGHT_PAD_TOKENS;
    }
    if (stream->params.max_tokens_per_step <= 0) {
        stream->params.max_tokens_per_step = 256;
    }

    compute_time_embedding(ctx.time_emb_cpu, (float) delay_ms_to_tokens(stream->params.delay_ms), VOXTRAL_DEC_DIM);
    upload_decoder_time_embedding(&ctx);
    incremental_mel_init(stream->mel, VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK);
    stream->mel_tail.assign((size_t) VOXTRAL_NUM_MEL_BINS * 2u, 0.0f);
    stream->conv0_tail.assign((size_t) VOXTRAL_ENC_DIM * 2u, 0.0f);
    stream->conv0_residual.assign((size_t) VOXTRAL_ENC_DIM, 0.0f);
    stream->min_new_mel_frames = std::max<int32_t>(1, stream->params.processing_interval_ms / 10);
    LOG_INFO(&ctx, "stream_init: delay_ms=%d processing_interval_ms=%d min_new_mel_frames=%d",
        stream->params.delay_ms,
        stream->params.processing_interval_ms,
        stream->min_new_mel_frames);
    if (ctx.gpu_type != voxtral_gpu_backend::none && !ensure_stream_conv_state(*stream)) {
        LOG_ERR(&ctx, "stream_init: failed to allocate conv stem state buffer");
        free_stream_conv_state(*stream);
        delete stream;
        return nullptr;
    }
    if (!ensure_stream_encoder_residual(*stream)) {
        LOG_ERR(&ctx, "stream_init: failed to allocate encoder residual buffer");
        free_stream_conv_state(*stream);
        free_stream_encoder_residual(*stream);
        delete stream;
        return nullptr;
    }
    clear_encoder_kv_cache(&ctx);
    clear_kv_cache(&ctx);
    ctx.dec_seq_len = 0;
    LOG_INFO(&ctx, "stream_init: ready");
    return stream;
}

void voxtral_stream_free(voxtral_stream * stream) {
    if (stream != nullptr) {
        free_stream_conv_state(*stream);
        free_stream_encoder_residual(*stream);
    }
    delete stream;
}

bool voxtral_stream_reset(
    voxtral_stream & stream,
    std::string * error) {

    if (stream.ctx == nullptr) {
        if (error) {
            *error = "voxtral stream has no context";
        }
        return false;
    }

    compute_time_embedding(
        stream.ctx->time_emb_cpu,
        (float) delay_ms_to_tokens(stream.params.delay_ms),
        VOXTRAL_DEC_DIM);
    upload_decoder_time_embedding(stream.ctx);
    incremental_mel_init(stream.mel, VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK);
    stream.real_samples_fed = 0;
    stream.mel_cursor = 0;
    std::fill(stream.mel_tail.begin(), stream.mel_tail.end(), 0.0f);
    std::fill(stream.conv0_tail.begin(), stream.conv0_tail.end(), 0.0f);
    std::fill(stream.conv0_residual.begin(), stream.conv0_residual.end(), 0.0f);
    stream.conv0_residual_count = 0;
    stream.conv_stem_initialized = false;
    stream.enc_residual_count = 0;
    stream.total_encoder_positions = 0;
    stream.total_audio_tokens = 0;
    stream.decoder_started = false;
    stream.eos_seen = false;
    stream.finished = false;
    stream.prev_token = VOXTRAL_TOKEN_BOS;
    stream.gen_pos = 0;
    stream.generated_token_index = 0;
    stream.timeline_token_base = 0;
    stream.pending_piece_text.clear();
    stream.pending_piece_begin_token_index = -1;
    stream.pending_piece_end_token_index = -1;
    clear_stream_conv_state(stream);
    clear_encoder_kv_cache(stream.ctx);
    clear_kv_cache(stream.ctx);
    stream.ctx->dec_seq_len = 0;
    return true;
}

bool voxtral_stream_push_audio(
    voxtral_stream & stream,
    const float * samples,
    int32_t n_samples,
    std::vector<voxtral_stream_event> & out_events,
    std::string * error) {

    if (stream.finished) {
        if (error) {
            *error = "voxtral stream already flushed";
        }
        return false;
    }
    if (samples == nullptr || n_samples <= 0) {
        return true;
    }

    (void) incremental_mel_feed(*stream.ctx, stream.mel, samples, n_samples);
    stream.real_samples_fed += n_samples;
    LOG_DBG(stream.ctx, "stream_push_audio: n_samples=%d total_samples=%lld",
        n_samples, (long long) stream.real_samples_fed);
    if (!stream_run_encoder(stream, error)) {
        return false;
    }
    if (!stream_run_decoder(stream, out_events, error)) {
        return false;
    }
    return true;
}

bool voxtral_stream_flush(
    voxtral_stream & stream,
    std::vector<voxtral_stream_event> & out_events,
    std::string * error) {

    if (stream.finished) {
        return true;
    }
    stream.finished = true;
    incremental_mel_finish(
        *stream.ctx,
        stream.mel,
        stream.params.right_pad_tokens * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK);

    if (!stream_run_encoder(stream, error)) {
        return false;
    }
    while (!stream.eos_seen && stream.gen_pos < stream.total_audio_tokens) {
        const int32_t prev_gen_pos = stream.gen_pos;
        const int32_t prev_generated = stream.generated_token_index;
        const bool prev_started = stream.decoder_started;
        if (!stream_run_decoder(stream, out_events, error)) {
            return false;
        }
        if (stream.gen_pos == prev_gen_pos &&
            stream.generated_token_index == prev_generated &&
            stream.decoder_started == prev_started) {
            break;
        }
    }
    flush_stream_piece_buffer(stream, out_events);
    return true;
}

// ============================================================================
// High-level: Transcribe
// ============================================================================

static bool voxtral_transcribe_from_audio(
    voxtral_context & ctx,
    const float     * audio,
    int32_t           n_samples,
    int32_t           max_tokens,
    voxtral_result  & result,
    bool              log_audio)
{
    result.text.clear();
    result.tokens.clear();
    result.first_step_logits.clear();

    if (audio == nullptr || n_samples <= 0) {
        LOG_ERR(&ctx, "audio input is empty");
        return false;
    }

    if (log_audio) {
        LOG_INFO(&ctx, "audio input: %d samples (%.1f s)", n_samples,
            (float)n_samples / VOXTRAL_SAMPLE_RATE);
    }

    // 2. Streaming padding (matching Python pad_audio_streaming)
    constexpr int32_t mult_of   = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;   // 1280
    const int32_t n_raw     = n_samples;
    const int32_t align_pad = (mult_of - (n_raw % mult_of)) % mult_of;
    const int32_t right_pad = align_pad + VOXTRAL_N_RIGHT_PAD_TOKENS * mult_of;
    constexpr int32_t left_pad  = VOXTRAL_N_LEFT_PAD_TOKENS * mult_of;

    std::vector<float> padded(left_pad + n_raw + right_pad, 0.0f);
    memcpy(padded.data() + left_pad, audio, n_raw * sizeof(float));

    LOG_INFO(&ctx, "padded audio: %d samples (left=%d, right=%d)", (int)padded.size(), left_pad, right_pad);

    // 3. Compute mel spectrogram
    int32_t n_frames = 0;
    const int32_t max_frames = (int32_t)padded.size() / VOXTRAL_HOP_LENGTH + 1;
    std::vector<float> mel_data(VOXTRAL_NUM_MEL_BINS * max_frames);

    compute_mel_spectrogram(
        padded.data(), (int32_t)padded.size(),
        ctx.mel_filters_cpu.data(),
        ctx.hann_window.data(),
        mel_data.data(), &n_frames);

    LOG_INFO(&ctx, "mel spectrogram: %d frames", n_frames);

    // Truncate to even number of frames (for conv stride=2)
    if (n_frames % 2 != 0) {
        // Drop first frame (matching Python mel[:, 1:])
        // Shift mel data left by 1 frame
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) {
            memmove(mel_data.data() + m * (n_frames - 1),
                    mel_data.data() + m * n_frames + 1,
                    (n_frames - 1) * sizeof(float));
        }
        n_frames -= 1;
        LOG_INFO(&ctx, "mel truncated to %d frames (even)", n_frames);
    }

    // 4. Run encoder (chunked for arbitrarily long audio)
    auto t_encoder = std::chrono::steady_clock::now();
    if (!run_encoder_chunked(&ctx, mel_data.data(), n_frames)) {
        return false;
    }
    LOG_INFO(&ctx, "encoder time: %.1f ms", elapsed_ms(t_encoder));

    // 5. Run adapter
    auto t_adapter = std::chrono::steady_clock::now();
    if (!run_adapter(&ctx)) {
        return false;
    }
    LOG_INFO(&ctx, "adapter time: %.1f ms", elapsed_ms(t_adapter));

    const int32_t n_audio = ctx.dec_seq_len;

    // 6. Build prompt tokens: [BOS] + [STREAMING_PAD] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
    std::vector<int32_t> prompt_ids;
    prompt_ids.push_back(VOXTRAL_TOKEN_BOS);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; i++) {
        prompt_ids.push_back(VOXTRAL_TOKEN_STREAMING_PAD);
    }
    const int32_t L = (int32_t)prompt_ids.size();  // 39

    LOG_INFO(&ctx, "prompt: %d tokens, audio_tokens: %d", L, n_audio);

    if (L > n_audio) {
        LOG_ERR(&ctx, "prompt length %d exceeds audio tokens %d", L, n_audio);
        return false;
    }

    // 7. Reset KV cache
    clear_kv_cache(&ctx);

    // 8. Decoder prefill
    auto t_prefill = std::chrono::steady_clock::now();
    std::vector<float> logits(VOXTRAL_VOCAB_SIZE);
    if (L > 1) {
        if (!run_decoder_prefill(&ctx, prompt_ids.data(), L - 1, logits.data(), nullptr)) {
            return false;
        }
    }

    // 8b. One step with last prefix token (matches Python prefill + forward_one)
    int32_t token = -1;
    if (!run_decoder_step(&ctx, prompt_ids[L - 1], L - 1, L - 1, logits.data(), &token)) {
        return false;
    }
    LOG_INFO(&ctx, "prefill time: %.1f ms", elapsed_ms(t_prefill));

    // Store first step logits
    result.first_step_logits = logits;
    result.tokens.push_back(token);

    LOG_INFO(&ctx, "first token: %d", token);

    // 9. Autoregressive decoding
    auto t_decode = std::chrono::steady_clock::now();
    int32_t consecutive_pad = 0;
    bool seen_text = false;
    for (int32_t pos = L; pos < n_audio && (int32_t)result.tokens.size() < max_tokens; pos++) {
        if (token == VOXTRAL_TOKEN_EOS) break;

        if (!run_decoder_step(&ctx, token, pos, pos, logits.data(), &token)) {
            return false;
        }

        result.tokens.push_back(token);

        // Early stopping: if we've seen real text and then get enough
        // consecutive streaming pad tokens, the transcript is complete.
        if (token == VOXTRAL_TOKEN_STREAMING_PAD) {
            consecutive_pad++;
        } else {
            consecutive_pad = 0;
            if (token >= ctx.model->tokenizer_num_special_tokens) {
                seen_text = true;
            }
        }
        if (seen_text && consecutive_pad >= VOXTRAL_N_RIGHT_PAD_TOKENS) {
            LOG_INFO(&ctx, "early stop: %d consecutive pad tokens after text", consecutive_pad);
            break;
        }
    }
    LOG_INFO(&ctx, "decode time: %.1f ms (%d steps, %.1f ms/step)",
        elapsed_ms(t_decode), (int)result.tokens.size() - 1,
        result.tokens.size() > 1 ? elapsed_ms(t_decode) / (result.tokens.size() - 1) : 0.0);

    // Remove trailing EOS
    if (!result.tokens.empty() && result.tokens.back() == VOXTRAL_TOKEN_EOS) {
        result.tokens.pop_back();
    }

    LOG_INFO(&ctx, "generated %d tokens", (int)result.tokens.size());

    // 10. Decode tokens to text (Tekken vocab from GGUF metadata)
    result.text = decode_tokens(*ctx.model, result.tokens);

    return true;
}

static std::string join_piece_commit_text(
    const std::vector<voxtral_stream_event> & events) {

    std::string text;
    for (const auto & ev : events) {
        if (ev.kind == voxtral_stream_event_kind::piece_commit) {
            text += ev.text;
        }
    }
    return text;
}

static bool voxtral_transcribe_audio_events_stream_offline(
    voxtral_context & ctx,
    const std::vector<float> & audio,
    std::vector<voxtral_stream_event> & out_events,
    voxtral_result * out_result) {

    out_events.clear();
    if (out_result != nullptr) {
        out_result->text.clear();
        out_result->tokens.clear();
        out_result->first_step_logits.clear();
    }

    if (audio.empty()) {
        LOG_ERR(&ctx, "offline stream transcription: audio input is empty");
        return false;
    }

    voxtral_stream_params params = {};
    std::unique_ptr<voxtral_stream, decltype(&voxtral_stream_free)> stream(
        voxtral_stream_init(ctx, params),
        &voxtral_stream_free);
    if (!stream) {
        LOG_ERR(&ctx, "offline stream transcription: failed to initialize stream");
        return false;
    }

    voxtral_incremental_mel mel = {};
    incremental_mel_init(mel, VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK);
    (void) incremental_mel_feed(ctx, mel, audio.data(), static_cast<int32_t>(audio.size()));
    incremental_mel_finish(
        ctx,
        mel,
        params.right_pad_tokens * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK);

    const int32_t total_mel =
        mel.mel_frame_offset + static_cast<int32_t>(mel.mel.size() / VOXTRAL_NUM_MEL_BINS);
    const int32_t delay_tokens = delay_ms_to_tokens(stream->params.delay_ms);
    const int32_t prompt_len = 1 + VOXTRAL_N_LEFT_PAD_TOKENS + delay_tokens;
    const int32_t first_chunk_min_mel = prompt_len * 8;
    LOG_INFO(
        &ctx,
        "offline stream transcription: samples=%d mel_frames=%d first_chunk_mel=%d step_mel=%d",
        static_cast<int32_t>(audio.size()),
        total_mel,
        first_chunk_min_mel,
        stream->min_new_mel_frames);

    stream->mel = std::move(mel);
    stream->real_samples_fed = static_cast<int64_t>(audio.size());
    stream->finished = true;

    std::string error;
    for (;;) {
        const int32_t available_total_mel =
            stream->mel.mel_frame_offset + static_cast<int32_t>(stream->mel.mel.size() / VOXTRAL_NUM_MEL_BINS);
        if (stream->mel_cursor >= available_total_mel) {
            break;
        }
        const int32_t offline_step_mel =
            stream->conv_stem_initialized ? stream->min_new_mel_frames : first_chunk_min_mel;

        const int32_t prev_cursor = stream->mel_cursor;
        const int32_t prev_audio_tokens = stream->total_audio_tokens;
        const int32_t prev_generated = stream->generated_token_index;
        if (!stream_run_encoder_bounded(*stream, offline_step_mel, &error)) {
            LOG_ERR(&ctx, "offline stream transcription: encoder failed: %s", error.c_str());
            return false;
        }
        if (!stream_run_decoder(*stream, out_events, &error)) {
            LOG_ERR(&ctx, "offline stream transcription: decoder failed: %s", error.c_str());
            return false;
        }
        if (stream->mel_cursor == prev_cursor &&
            stream->total_audio_tokens == prev_audio_tokens &&
            stream->generated_token_index == prev_generated) {
            LOG_ERR(&ctx, "offline stream transcription: stream loop made no progress");
            return false;
        }
    }

    while (!stream->eos_seen && stream->gen_pos < stream->total_audio_tokens) {
        const int32_t prev_gen_pos = stream->gen_pos;
        const int32_t prev_generated = stream->generated_token_index;
        const bool prev_started = stream->decoder_started;
        if (!stream_run_decoder(*stream, out_events, &error)) {
            LOG_ERR(&ctx, "offline stream transcription: tail decode failed: %s", error.c_str());
            return false;
        }
        if (stream->gen_pos == prev_gen_pos &&
            stream->generated_token_index == prev_generated &&
            stream->decoder_started == prev_started) {
            break;
        }
    }

    flush_stream_piece_buffer(*stream, out_events);

    if (out_result != nullptr) {
        out_result->text = join_piece_commit_text(out_events);
    }

    return true;
}

static void build_piece_events_from_tokens(
    voxtral_context & ctx,
    const std::vector<int32_t> & tokens,
    std::vector<voxtral_stream_event> & out_events) {

    out_events.clear();

    voxtral_stream stream = {};
    stream.ctx = &ctx;
    stream.timeline_token_base = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t token_id = tokens[i];
        const int32_t token_index = static_cast<int32_t>(i);

        std::string text;
        if (decode_visible_token_piece(*ctx.model, token_id, text)) {
            if (!stream.pending_piece_text.empty() &&
                pending_piece_ends_sentence(stream.pending_piece_text) &&
                text_begins_sentence(text)) {
                flush_stream_piece_buffer(stream, out_events);
            }
            append_stream_piece_token(stream, token_index, text);
            continue;
        }

        flush_stream_piece_buffer(stream, out_events);
    }

    flush_stream_piece_buffer(stream, out_events);
}

bool voxtral_transcribe_audio(
    voxtral_context   & ctx,
    const std::vector<float> & audio,
    int32_t             max_tokens,
    voxtral_result    & result)
{
    return voxtral_transcribe_from_audio(
        ctx, audio.data(), (int32_t) audio.size(), max_tokens, result, true);
}

bool voxtral_transcribe_audio_events(
    voxtral_context & ctx,
    const std::vector<float> & audio,
    int32_t max_tokens,
    std::vector<voxtral_stream_event> & out_events,
    voxtral_result * out_result) {

    voxtral_result local_result;
    voxtral_result & result = out_result != nullptr ? *out_result : local_result;
    (void) max_tokens;
    if (!voxtral_transcribe_audio_events_stream_offline(
            ctx,
            audio,
            out_events,
            &result)) {
        return false;
    }
    return true;
}

bool voxtral_transcribe_file(
    voxtral_context   & ctx,
    const std::string & audio_path,
    int32_t             max_tokens,
    voxtral_result    & result)
{
    std::vector<float> audio;
    if (!load_wav_file(audio_path, audio)) {
        LOG_ERR(&ctx, "failed to load WAV: %s", audio_path.c_str());
        return false;
    }
    LOG_INFO(&ctx, "audio loaded: %d samples (%.1f s)", (int)audio.size(),
        (float)audio.size() / VOXTRAL_SAMPLE_RATE);

    return voxtral_transcribe_from_audio(
        ctx, audio.data(), (int32_t) audio.size(), max_tokens, result, false);
}
