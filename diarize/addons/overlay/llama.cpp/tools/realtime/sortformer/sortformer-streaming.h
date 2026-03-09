#pragma once

#include "sortformer-frontend.h"
#include "sortformer-postnet.h"

#include <vector>

namespace llama::realtime {

struct sortformer_stream_cache_state {
    sortformer_matrix_f32 spkcache;
    sortformer_matrix_f32 fifo;
};

struct sortformer_stream_fastpath_inputs {
    uint32_t rows = 0;
    uint32_t valid_rows = 0;
    sortformer_matrix_f32 pos_emb;
    sortformer_matrix_f32 pad_mask;
    sortformer_matrix_f32 att_mask;
};

struct sortformer_stream_runtime_state {
    sortformer_matrix_f32 spkcache;
    sortformer_matrix_f32 spkcache_preds;
    sortformer_matrix_f32 fifo;
    sortformer_matrix_f32 fifo_preds;
    std::vector<float> mean_sil_emb;
    sortformer_stream_fastpath_inputs fastpath_inputs;
    sortformer_encoder_postnet_plan encoder_postnet_plan;
    uint64_t n_sil_frames = 0;
    bool have_spkcache_preds = false;
};

struct sortformer_stream_step_outputs {
    sortformer_matrix_f32 chunk_preencode;
    uint32_t chunk_preencode_valid_rows = 0;
    uint32_t left_context_rows = 0;
    uint32_t right_context_rows = 0;

    sortformer_matrix_f32 concat_preencode;
    uint32_t concat_valid_rows = 0;

    sortformer_frontend_outputs frontend;
    sortformer_postnet_outputs postnet;

    sortformer_matrix_f32 preds_all;
    sortformer_matrix_f32 chunk_preds;
    sortformer_matrix_f32 chunk_core_preencode;
};

sortformer_stream_step_outputs sortformer_run_stream_step(
    const sortformer_model & model,
    const sortformer_matrix_f32 & chunk_features,
    uint32_t chunk_valid_feature_rows,
    const sortformer_stream_cache_state & cache_state,
    uint32_t left_context_rows = 0,
    uint32_t right_context_rows = 0,
    bool capture_debug = false,
    sortformer_stream_fastpath_inputs * fastpath_inputs = nullptr,
    sortformer_encoder_postnet_plan * encoder_postnet_plan = nullptr);

sortformer_stream_runtime_state sortformer_make_stream_runtime_state(const sortformer_model_metadata & meta);

sortformer_stream_step_outputs sortformer_streaming_update(
    const sortformer_model & model,
    const sortformer_matrix_f32 & chunk_features,
    uint32_t chunk_valid_feature_rows,
    uint32_t left_context_rows,
    uint32_t right_context_rows,
    sortformer_stream_runtime_state & state,
    bool capture_debug = false);

} // namespace llama::realtime
