#include "sortformer-audio.h"

#include "sortformer-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llama::realtime {

namespace {

constexpr float kPi = 3.14159265358979323846f;

void require(bool cond, const char * message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

bool is_power_of_two(uint32_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

uint32_t bit_reverse(uint32_t value, uint32_t bits) {
    uint32_t out = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        out = (out << 1) | (value & 1u);
        value >>= 1u;
    }
    return out;
}

std::vector<float> read_tensor_f32(ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("Sortformer audio frontend expects F32 frontend tensors");
    }
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace

sortformer_audio_frontend::sortformer_audio_frontend(const sortformer_model & model) {
    const auto & meta = model.metadata();
    config_.sample_rate_hz = meta.sample_rate_hz;
    config_.n_fft = meta.n_fft;
    config_.win_length_samples = static_cast<uint32_t>(std::llround(meta.window_size_sec * meta.sample_rate_hz));
    config_.hop_length_samples = static_cast<uint32_t>(std::llround(meta.window_stride_sec * meta.sample_rate_hz));
    config_.mel_bins = meta.mel_bins;
    config_.pad_to = meta.pad_to;
    config_.preemph = meta.preemph;
    config_.log_zero_guard_value = meta.log_zero_guard_value;
    config_.log_zero_guard_add = meta.log_zero_guard_add;
    config_.log_features = meta.log_features;
    config_.normalize_mode = meta.normalize_mode;

    require(is_power_of_two(config_.n_fft), "Sortformer audio frontend expects power-of-two n_fft");
    require(config_.win_length_samples > 0 && config_.win_length_samples <= config_.n_fft, "invalid Sortformer win_length");
    require(config_.hop_length_samples > 0, "invalid Sortformer hop_length");
    require(config_.normalize_mode == "NA", "only normalize=NA is supported for native Sortformer frontend");

    load_reference_tensors(model);

    const uint32_t bits = static_cast<uint32_t>(std::log2(config_.n_fft));
    bit_reversed_.resize(config_.n_fft);
    for (uint32_t i = 0; i < config_.n_fft; ++i) {
        bit_reversed_[i] = bit_reverse(i, bits);
    }

    twiddle_real_.resize(config_.n_fft / 2);
    twiddle_imag_.resize(config_.n_fft / 2);
    for (uint32_t i = 0; i < config_.n_fft / 2; ++i) {
        const float angle = -2.0f * kPi * static_cast<float>(i) / static_cast<float>(config_.n_fft);
        twiddle_real_[i] = std::cos(angle);
        twiddle_imag_[i] = std::sin(angle);
    }

    fft_real_scratch_.assign(config_.n_fft, 0.0f);
    fft_imag_scratch_.assign(config_.n_fft, 0.0f);
    power_scratch_.assign(config_.n_fft / 2 + 1, 0.0f);

    features_.cols = config_.mel_bins;
}

void sortformer_audio_frontend::load_reference_tensors(const sortformer_model & model) {
    auto * window = model.tensor("prep.feat.win");
    auto * fb = model.tensor("prep.feat.fb");
    const auto win = read_tensor_f32(window);
    const auto mel = read_tensor_f32(fb);

    require(win.size() == config_.win_length_samples, "unexpected Sortformer frontend window size");
    require(mel.size() == static_cast<size_t>(config_.mel_bins) * (config_.n_fft / 2 + 1), "unexpected Sortformer mel filter size");

    padded_window_.assign(config_.n_fft, 0.0f);
    const uint32_t pad_left = (config_.n_fft - config_.win_length_samples) / 2;
    std::copy(win.begin(), win.end(), padded_window_.begin() + pad_left);
    mel_filters_ = mel;
}

void sortformer_audio_frontend::reset() {
    flushing_ = false;
    raw_pcm_.clear();
    preemph_pcm_.clear();
    features_.rows = 0;
    features_.cols = config_.mel_bins;
    features_.data.clear();
}

void sortformer_audio_frontend::push_audio(const float * samples, size_t n_samples) {
    if (samples == nullptr && n_samples != 0) {
        throw std::invalid_argument("null audio pointer");
    }

    raw_pcm_.reserve(raw_pcm_.size() + n_samples);
    preemph_pcm_.reserve(preemph_pcm_.size() + n_samples);

    float prev = raw_pcm_.empty() ? 0.0f : raw_pcm_.back();
    for (size_t i = 0; i < n_samples; ++i) {
        const float current = samples[i];
        raw_pcm_.push_back(current);
        if (raw_pcm_.size() == 1) {
            preemph_pcm_.push_back(current);
        } else {
            preemph_pcm_.push_back(current - config_.preemph * prev);
        }
        prev = current;
    }

    compute_ready_frames();
}

void sortformer_audio_frontend::flush() {
    flushing_ = true;
    compute_ready_frames();
}

bool sortformer_audio_frontend::flushing() const {
    return flushing_;
}

uint64_t sortformer_audio_frontend::total_samples() const {
    return raw_pcm_.size();
}

uint32_t sortformer_audio_frontend::available_feature_frames() const {
    return features_.rows;
}

uint32_t sortformer_audio_frontend::final_feature_frames() const {
    return final_target_frames();
}

const sortformer_audio_frontend_config & sortformer_audio_frontend::config() const {
    return config_;
}

sortformer_matrix_f32 sortformer_audio_frontend::copy_feature_rows(uint64_t begin_frame, uint32_t rows) const {
    sortformer_matrix_f32 out;
    out.rows = rows;
    out.cols = config_.mel_bins;
    out.data.assign(static_cast<size_t>(rows) * out.cols, 0.0f);

    if (rows == 0 || begin_frame >= features_.rows) {
        return out;
    }

    const uint32_t available = std::min<uint32_t>(rows, features_.rows - static_cast<uint32_t>(begin_frame));
    const size_t cols = out.cols;
    const size_t src_offset = static_cast<size_t>(begin_frame) * cols;
    std::copy_n(features_.data.data() + src_offset, static_cast<size_t>(available) * cols, out.data.data());
    return out;
}

uint32_t sortformer_audio_frontend::stable_target_frames() const {
    if (raw_pcm_.size() < config_.win_length_samples / 2) {
        return 0;
    }
    return 1u + static_cast<uint32_t>((raw_pcm_.size() - config_.win_length_samples / 2) / config_.hop_length_samples);
}

uint32_t sortformer_audio_frontend::final_target_frames() const {
    return static_cast<uint32_t>(raw_pcm_.size() / config_.hop_length_samples);
}

void sortformer_audio_frontend::compute_ready_frames() {
    const uint32_t target = flushing_ ? final_target_frames() : stable_target_frames();
    if (target <= features_.rows) {
        return;
    }

    const uint32_t old_rows = features_.rows;
    features_.rows = target;
    features_.data.resize(static_cast<size_t>(target) * features_.cols, 0.0f);

    for (uint32_t row = old_rows; row < target; ++row) {
        compute_frame(row, features_.data.data() + static_cast<size_t>(row) * features_.cols);
    }
}

void sortformer_audio_frontend::fft_inplace(float * real, float * imag) const {
    for (uint32_t i = 0; i < config_.n_fft; ++i) {
        const uint32_t j = bit_reversed_[i];
        if (j > i) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    for (uint32_t len = 2; len <= config_.n_fft; len <<= 1u) {
        const uint32_t half = len >> 1u;
        const uint32_t step = config_.n_fft / len;
        for (uint32_t offset = 0; offset < config_.n_fft; offset += len) {
            for (uint32_t j = 0; j < half; ++j) {
                const uint32_t tw_idx = j * step;
                const float wr = twiddle_real_[tw_idx];
                const float wi = twiddle_imag_[tw_idx];

                const uint32_t even = offset + j;
                const uint32_t odd = even + half;
                const float tr = wr * real[odd] - wi * imag[odd];
                const float ti = wr * imag[odd] + wi * real[odd];

                const float ur = real[even];
                const float ui = imag[even];
                real[even] = ur + tr;
                imag[even] = ui + ti;
                real[odd] = ur - tr;
                imag[odd] = ui - ti;
            }
        }
    }
}

void sortformer_audio_frontend::compute_frame(uint32_t frame_index, float * dst_row) {
    std::fill(fft_real_scratch_.begin(), fft_real_scratch_.end(), 0.0f);
    std::fill(fft_imag_scratch_.begin(), fft_imag_scratch_.end(), 0.0f);

    float * real = fft_real_scratch_.data();
    float * imag = fft_imag_scratch_.data();
    const int64_t base = static_cast<int64_t>(frame_index) * static_cast<int64_t>(config_.hop_length_samples) - static_cast<int64_t>(config_.n_fft / 2);

    for (uint32_t i = 0; i < config_.n_fft; ++i) {
        const int64_t sample_idx = base + static_cast<int64_t>(i);
        float sample = 0.0f;
        if (sample_idx >= 0 && sample_idx < static_cast<int64_t>(preemph_pcm_.size())) {
            sample = preemph_pcm_[static_cast<size_t>(sample_idx)];
        }
        real[i] = sample * padded_window_[i];
    }

    fft_inplace(real, imag);

    const uint32_t fft_bins = config_.n_fft / 2 + 1;
    float * power = power_scratch_.data();
    for (uint32_t i = 0; i < fft_bins; ++i) {
        power[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    for (uint32_t mel = 0; mel < config_.mel_bins; ++mel) {
        const float * fb_row = mel_filters_.data() + static_cast<size_t>(mel) * fft_bins;
        double accum = 0.0;
        for (uint32_t k = 0; k < fft_bins; ++k) {
            accum += static_cast<double>(fb_row[k]) * static_cast<double>(power[k]);
        }

        float value = static_cast<float>(accum);
        if (config_.log_features) {
            value = config_.log_zero_guard_add
                ? std::log(value + config_.log_zero_guard_value)
                : std::log(std::max(value, config_.log_zero_guard_value));
        }
        dst_row[mel] = value;
    }
}

} // namespace llama::realtime
