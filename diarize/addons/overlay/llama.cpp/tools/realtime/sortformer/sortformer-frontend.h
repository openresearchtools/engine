#pragma once

#include "sortformer-model.h"
#include "sortformer-preencode.h"

namespace llama::realtime {

struct sortformer_frontend_outputs {
    sortformer_matrix_f32 preencode_out;
    sortformer_matrix_f32 posenc_x;
    sortformer_matrix_f32 pos_emb;
    sortformer_matrix_f32 pad_mask;
    sortformer_matrix_f32 att_mask;
    sortformer_matrix_f32 encoder_mask;
    sortformer_matrix_f32 encoder_out;
};

sortformer_frontend_outputs sortformer_run_frontend_encoder(
    const sortformer_model & model,
    const sortformer_matrix_f32 & features,
    bool capture_debug = false);

sortformer_frontend_outputs sortformer_run_frontend_encoder_from_preencoded(
    const sortformer_model & model,
    const sortformer_matrix_f32 & preencoded,
    uint32_t valid_frames,
    bool capture_debug = false);

} // namespace llama::realtime
