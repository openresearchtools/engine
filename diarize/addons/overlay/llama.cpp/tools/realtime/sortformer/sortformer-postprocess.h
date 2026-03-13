#pragma once

#include "sortformer-model.h"
#include "sortformer-preencode.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llama::realtime {

struct sortformer_postprocess_params {
    float onset = 0.5f;
    float offset = 0.5f;
    float pad_onset = 0.0f;
    float pad_offset = 0.0f;
    float min_duration_on = 0.0f;
    float min_duration_off = 0.0f;
    bool filter_speech_first = true;
    uint32_t unit_10ms_frame_count = 8;
    uint32_t round_precision = 2;
};

struct sortformer_speaker_span {
    int32_t speaker_id = -1;
    double begin_sec = 0.0;
    double end_sec = 0.0;
};

sortformer_postprocess_params sortformer_default_postprocess_params(const sortformer_model_metadata & meta);

std::vector<sortformer_speaker_span> sortformer_postprocess_speaker_spans(
    const sortformer_matrix_f32 & speaker_probs,
    const sortformer_postprocess_params & params,
    double offset_sec = 0.0);

} // namespace llama::realtime
