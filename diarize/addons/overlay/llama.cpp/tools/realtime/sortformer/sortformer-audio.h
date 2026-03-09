#pragma once

#include "sortformer-preencode.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama::realtime {

class sortformer_model;

struct sortformer_audio_frontend_config {
    uint32_t sample_rate_hz = 16000;
    uint32_t n_fft = 512;
    uint32_t win_length_samples = 400;
    uint32_t hop_length_samples = 160;
    uint32_t mel_bins = 128;
    uint32_t pad_to = 16;
    float preemph = 0.97f;
    float log_zero_guard_value = 0.0f;
    bool log_zero_guard_add = true;
    bool log_features = true;
    std::string normalize_mode = "NA";
};

class sortformer_audio_frontend {
public:
    explicit sortformer_audio_frontend(const sortformer_model & model);

    void reset();
    void push_audio(const float * samples, size_t n_samples);
    void flush();

    bool flushing() const;
    uint64_t total_samples() const;
    uint32_t available_feature_frames() const;
    uint32_t final_feature_frames() const;

    const sortformer_audio_frontend_config & config() const;
    sortformer_matrix_f32 copy_feature_rows(uint64_t begin_frame, uint32_t rows) const;

private:
    void compute_ready_frames();
    uint32_t stable_target_frames() const;
    uint32_t final_target_frames() const;
    void compute_frame(uint32_t frame_index, float * dst_row);
    void fft_inplace(float * real, float * imag) const;
    void load_reference_tensors(const sortformer_model & model);

    sortformer_audio_frontend_config config_;
    bool flushing_ = false;

    std::vector<float> raw_pcm_;
    std::vector<float> preemph_pcm_;
    sortformer_matrix_f32 features_;

    std::vector<float> padded_window_;
    std::vector<float> mel_filters_;
    std::vector<uint32_t> bit_reversed_;
    std::vector<float> twiddle_real_;
    std::vector<float> twiddle_imag_;
    std::vector<float> fft_real_scratch_;
    std::vector<float> fft_imag_scratch_;
    std::vector<float> power_scratch_;
};

} // namespace llama::realtime
