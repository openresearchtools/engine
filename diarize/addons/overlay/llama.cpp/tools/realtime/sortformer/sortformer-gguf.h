#pragma once

#include <cstdint>
#include <string>

namespace llama::realtime {

struct sortformer_model_metadata {
    std::string path;
    std::string architecture;

    uint32_t sample_rate_hz = 16000;
    float window_size_sec = 0.025f;
    float window_stride_sec = 0.010f;
    std::string window_type = "hann";
    uint32_t mel_bins = 128;
    uint32_t n_fft = 512;
    uint32_t pad_to = 16;
    float preemph = 0.97f;
    float dither = 0.0f;
    float log_zero_guard_value = 5.96046448e-08f;
    bool log_zero_guard_add = true;
    bool log_features = true;
    std::string normalize_mode = "NA";

    bool streaming_mode = false;
    uint32_t max_speakers = 4;

    uint32_t fc_d_model = 512;
    uint32_t tf_d_model = 192;
    uint32_t chunk_len_frames = 0;
    uint32_t fifo_len_frames = 0;
    uint32_t spkcache_len_frames = 0;
    uint32_t spkcache_update_period_frames = 0;
    uint32_t chunk_left_context = 0;
    uint32_t chunk_right_context = 0;
    uint32_t spkcache_sil_frames_per_spk = 0;

    uint32_t encoder_layers = 0;
    uint32_t encoder_d_model = 0;
    uint32_t encoder_heads = 0;
    uint32_t encoder_subsampling_factor = 0;

    uint32_t transformer_layers = 0;
    uint32_t transformer_heads = 0;

    float pred_score_threshold = 0.25f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_pos_scores_rate = 0.5f;
    float sil_threshold = 0.2f;
    float scores_boost_latest = 0.0f;

    int64_t tensor_count = 0;
};

sortformer_model_metadata load_sortformer_gguf(const std::string & path);
std::string sortformer_metadata_summary(const sortformer_model_metadata & meta);

} // namespace llama::realtime
