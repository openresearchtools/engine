#pragma once

#include "sortformer-gguf.h"

#include <cstddef>
#include <cstdint>

namespace llama::realtime {

struct sortformer_chunk_request {
    uint64_t chunk_index = 0;
    uint64_t input_begin_frame = 0;
    uint64_t nominal_input_end_frame = 0;
    uint64_t valid_input_feature_frames = 0;
    uint64_t emit_begin_frame = 0;
    uint64_t nominal_emit_end_frame = 0;
    uint64_t available_feature_frames = 0;
    uint64_t left_context_rows = 0;
    uint64_t right_context_rows = 0;
    bool final_partial = false;
};

class sortformer_stream_state {
public:
    explicit sortformer_stream_state(const sortformer_model_metadata & meta);

    void reset();
    void set_flushing(bool enabled);
    void set_available_pcm_samples(uint64_t sample_count);
    void set_available_feature_frames(uint64_t frame_count);

    uint64_t available_pcm_samples() const;
    uint64_t available_feature_frames() const;
    uint64_t completed_chunks() const;
    bool has_ready_chunk() const;
    sortformer_chunk_request next_chunk() const;
    void mark_chunk_complete();

private:
    uint64_t stable_feature_frames_from_pcm(uint64_t sample_count) const;
    uint64_t final_feature_frames_from_pcm(uint64_t sample_count) const;
    uint64_t pcm_to_feature_frames(uint64_t sample_count) const;
    uint64_t preencode_valid_rows(uint64_t valid_input_feature_frames) const;

    sortformer_model_metadata meta_;
    bool flushing_ = false;
    uint64_t available_pcm_samples_ = 0;
    uint64_t available_feature_frames_ = 0;
    uint64_t next_chunk_index_ = 0;
    uint64_t completed_chunks_ = 0;
};

} // namespace llama::realtime
