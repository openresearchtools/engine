#pragma once

#include "sortformer-model.h"
#include "sortformer-preencode.h"

namespace llama::realtime {

struct sortformer_layer0_outputs {
    sortformer_matrix_f32 ff1_norm;
    sortformer_matrix_f32 ff1_mm;
    sortformer_matrix_f32 ff1_l1;
    sortformer_matrix_f32 ff1_act;
    sortformer_matrix_f32 ff1_out_mm;
    sortformer_matrix_f32 ff1_out;
    sortformer_matrix_f32 ff1_res;
    sortformer_matrix_f32 att_norm;
    sortformer_matrix_f32 matrix_ac_head0;
    sortformer_matrix_f32 matrix_bd_head0;
    sortformer_matrix_f32 scores_head0;
    sortformer_matrix_f32 attn_head0;
    sortformer_matrix_f32 att_value_head0;
    sortformer_matrix_f32 att_x;
    sortformer_matrix_f32 att_out;
    sortformer_matrix_f32 att_res;
    sortformer_matrix_f32 conv_norm;
    sortformer_matrix_f32 conv_pw1;
    sortformer_matrix_f32 conv_glu;
    sortformer_matrix_f32 conv_dw;
    sortformer_matrix_f32 conv_bn;
    sortformer_matrix_f32 conv_act;
    sortformer_matrix_f32 conv_pw2;
    sortformer_matrix_f32 conv_out;
    sortformer_matrix_f32 conv_res;
    sortformer_matrix_f32 ff2_norm;
    sortformer_matrix_f32 ff2_out;
    sortformer_matrix_f32 ff2_res;
    sortformer_matrix_f32 out;
};

sortformer_layer0_outputs sortformer_run_layer0(
    const sortformer_model & model,
    const sortformer_matrix_f32 & posenc_x,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask);

} // namespace llama::realtime
