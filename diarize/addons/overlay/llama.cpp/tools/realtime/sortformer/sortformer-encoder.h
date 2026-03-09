#pragma once

#include "sortformer-model.h"
#include "sortformer-preencode.h"

namespace llama::realtime {

sortformer_matrix_f32 sortformer_run_encoder(
    const sortformer_model & model,
    const sortformer_matrix_f32 & preencoded,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask);

} // namespace llama::realtime
