#include "sortformer-state.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llama::realtime {

namespace {

uint64_t conv_output_size(uint64_t input_size, uint64_t kernel_size, uint64_t stride, uint64_t pad_left, uint64_t pad_right) {
    return (input_size + pad_left + pad_right - kernel_size) / stride + 1;
}

uint64_t ceil_div(uint64_t num, uint64_t den) {
    return den == 0 ? 0 : (num + den - 1) / den;
}

}

sortformer_stream_state::sortformer_stream_state(const sortformer_model_metadata & meta)
    : meta_(meta) {
    if (meta_.sample_rate_hz == 0 || meta_.chunk_len_frames == 0 || meta_.encoder_subsampling_factor == 0) {
        throw std::invalid_argument("invalid Sortformer metadata for stream state");
    }
}

void sortformer_stream_state::reset() {
    flushing_ = false;
    available_pcm_samples_ = 0;
    available_feature_frames_ = 0;
    next_chunk_index_ = 0;
    completed_chunks_ = 0;
}

void sortformer_stream_state::set_flushing(bool enabled) {
    flushing_ = enabled;
    available_feature_frames_ = enabled
        ? final_feature_frames_from_pcm(available_pcm_samples_)
        : stable_feature_frames_from_pcm(available_pcm_samples_);
}

void sortformer_stream_state::set_available_pcm_samples(uint64_t sample_count) {
    available_pcm_samples_ = sample_count;
    available_feature_frames_ = flushing_
        ? final_feature_frames_from_pcm(available_pcm_samples_)
        : stable_feature_frames_from_pcm(available_pcm_samples_);
}

void sortformer_stream_state::set_available_feature_frames(uint64_t frame_count) {
    available_feature_frames_ = frame_count;
}

uint64_t sortformer_stream_state::available_pcm_samples() const {
    return available_pcm_samples_;
}

uint64_t sortformer_stream_state::available_feature_frames() const {
    return available_feature_frames_;
}

uint64_t sortformer_stream_state::completed_chunks() const {
    return completed_chunks_;
}

bool sortformer_stream_state::has_ready_chunk() const {
    const uint64_t chunk_feature_len = meta_.chunk_len_frames * meta_.encoder_subsampling_factor;
    const uint64_t context_feature_len = meta_.chunk_right_context * meta_.encoder_subsampling_factor;
    const uint64_t stt_feat = next_chunk_index_ * chunk_feature_len;
    if (stt_feat >= available_feature_frames_) {
        return false;
    }

    const uint64_t core_end_feat = std::min<uint64_t>(stt_feat + chunk_feature_len, available_feature_frames_);
    if (core_end_feat <= stt_feat) {
        return false;
    }

    if (!flushing_) {
        return available_feature_frames_ >= core_end_feat + context_feature_len;
    }

    return true;
}

sortformer_chunk_request sortformer_stream_state::next_chunk() const {
    if (!has_ready_chunk()) {
        throw std::runtime_error("Sortformer chunk requested before it was ready");
    }

    const uint64_t chunk_len = meta_.chunk_len_frames;
    const uint64_t chunk_feature_len = meta_.chunk_len_frames * meta_.encoder_subsampling_factor;
    const uint64_t context_feature_len = meta_.chunk_left_context * meta_.encoder_subsampling_factor;
    const uint64_t chunk_index = next_chunk_index_;
    const uint64_t stt_feat = chunk_index * chunk_feature_len;
    const uint64_t left_offset_feat = std::min<uint64_t>(context_feature_len, stt_feat);
    const uint64_t core_end_feat = std::min<uint64_t>(stt_feat + chunk_feature_len, available_feature_frames_);
    const uint64_t right_offset_feat = flushing_
        ? std::min<uint64_t>(context_feature_len, available_feature_frames_ - core_end_feat)
        : context_feature_len;
    const uint64_t input_begin = stt_feat - left_offset_feat;
    const uint64_t input_end = core_end_feat + right_offset_feat;
    const uint64_t chunk_feat_len = input_end - input_begin;
    const uint64_t valid_input_len = std::min<uint64_t>(
        std::max<uint64_t>(available_feature_frames_ + left_offset_feat, stt_feat) - stt_feat,
        chunk_feat_len);
    const uint64_t emit_begin = chunk_index * chunk_len;
    const uint64_t emit_end = emit_begin + std::min<uint64_t>(chunk_len, preencode_valid_rows(valid_input_len));

    sortformer_chunk_request request;
    request.chunk_index = chunk_index;
    request.input_begin_frame = input_begin;
    request.nominal_input_end_frame = input_end;
    request.valid_input_feature_frames = valid_input_len;
    request.emit_begin_frame = emit_begin;
    request.nominal_emit_end_frame = emit_end;
    request.available_feature_frames = available_feature_frames_;
    request.left_context_rows = left_offset_feat / meta_.encoder_subsampling_factor;
    request.right_context_rows = ceil_div(right_offset_feat, meta_.encoder_subsampling_factor);
    request.final_partial = core_end_feat < stt_feat + chunk_feature_len;
    return request;
}

void sortformer_stream_state::mark_chunk_complete() {
    if (!has_ready_chunk()) {
        throw std::runtime_error("Sortformer chunk completion called with no ready chunk");
    }

    ++next_chunk_index_;
    ++completed_chunks_;
}

uint64_t sortformer_stream_state::stable_feature_frames_from_pcm(uint64_t sample_count) const {
    const uint64_t hop_samples = static_cast<uint64_t>(std::llround(meta_.window_stride_sec * meta_.sample_rate_hz));
    const uint64_t win_samples = static_cast<uint64_t>(std::llround(meta_.window_size_sec * meta_.sample_rate_hz));
    if (hop_samples == 0 || win_samples == 0 || sample_count < (win_samples / 2)) {
        return 0;
    }
    return 1 + (sample_count - (win_samples / 2)) / hop_samples;
}

uint64_t sortformer_stream_state::final_feature_frames_from_pcm(uint64_t sample_count) const {
    return pcm_to_feature_frames(sample_count);
}

uint64_t sortformer_stream_state::pcm_to_feature_frames(uint64_t sample_count) const {
    const uint64_t hop_samples = static_cast<uint64_t>(std::llround(meta_.window_stride_sec * meta_.sample_rate_hz));
    if (hop_samples == 0 || sample_count < hop_samples) {
        return 0;
    }
    return sample_count / hop_samples;
}

uint64_t sortformer_stream_state::preencode_valid_rows(uint64_t valid_input_feature_frames) const {
    if (valid_input_feature_frames == 0) {
        return 0;
    }
    uint64_t rows = valid_input_feature_frames;
    rows = conv_output_size(rows, 3, 2, 1, 1);
    rows = conv_output_size(rows, 3, 2, 1, 1);
    rows = conv_output_size(rows, 3, 2, 1, 1);
    return rows;
}

} // namespace llama::realtime
