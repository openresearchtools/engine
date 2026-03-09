#pragma once

#include "sortformer-model.h"
#include "sortformer-preencode.h"

namespace llama::realtime {

struct sortformer_postnet_outputs {
    sortformer_matrix_f32 encoder_proj_out;

    sortformer_matrix_f32 te_layer0_q;
    sortformer_matrix_f32 te_layer0_k;
    sortformer_matrix_f32 te_layer0_v;
    sortformer_matrix_f32 te_layer0_scores_head0;
    sortformer_matrix_f32 te_layer0_probs_head0;
    sortformer_matrix_f32 te_layer0_context;
    sortformer_matrix_f32 te_layer0_att_out;
    sortformer_matrix_f32 te_layer0_att_res;
    sortformer_matrix_f32 te_layer0_ln1;
    sortformer_matrix_f32 te_layer0_ff_di;
    sortformer_matrix_f32 te_layer0_ff_act;
    sortformer_matrix_f32 te_layer0_ff_do;
    sortformer_matrix_f32 te_layer0_out;

    sortformer_matrix_f32 transformer_out;
    sortformer_matrix_f32 head_hidden1;
    sortformer_matrix_f32 head_hidden2;
    sortformer_matrix_f32 head_hidden3;
    sortformer_matrix_f32 head_logits;
    sortformer_matrix_f32 preds;
};

struct sortformer_full_step_outputs {
    sortformer_matrix_f32 chunk_preencode;
    sortformer_matrix_f32 preds_all;
};

class sortformer_encoder_postnet_plan {
public:
    sortformer_encoder_postnet_plan();
    ~sortformer_encoder_postnet_plan();

    sortformer_encoder_postnet_plan(const sortformer_encoder_postnet_plan & other);
    sortformer_encoder_postnet_plan & operator=(const sortformer_encoder_postnet_plan & other);
    sortformer_encoder_postnet_plan(sortformer_encoder_postnet_plan && other) noexcept;
    sortformer_encoder_postnet_plan & operator=(sortformer_encoder_postnet_plan && other) noexcept;

private:
    struct impl;
    impl * impl_ = nullptr;

    friend sortformer_matrix_f32 sortformer_run_encoder_postnet_concat_cached(
        sortformer_encoder_postnet_plan & plan,
        const sortformer_model & model,
        const sortformer_matrix_f32 & spkcache,
        const sortformer_matrix_f32 & fifo,
        const sortformer_matrix_f32 & chunk_preencode,
        const sortformer_matrix_f32 & pos_emb,
        const sortformer_matrix_f32 & pad_mask,
        const sortformer_matrix_f32 & att_mask);
};

sortformer_postnet_outputs sortformer_run_postnet(
    const sortformer_model & model,
    const sortformer_matrix_f32 & fc_encoder_out,
    const sortformer_matrix_f32 & encoder_mask,
    bool capture_debug = false);

sortformer_matrix_f32 sortformer_run_encoder_postnet(
    const sortformer_model & model,
    const sortformer_matrix_f32 & preencoded,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask);

sortformer_matrix_f32 sortformer_run_encoder_postnet_concat(
    const sortformer_model & model,
    const sortformer_matrix_f32 & spkcache,
    const sortformer_matrix_f32 & fifo,
    const sortformer_matrix_f32 & chunk_preencode,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask);

sortformer_matrix_f32 sortformer_run_encoder_postnet_concat_cached(
    sortformer_encoder_postnet_plan & plan,
    const sortformer_model & model,
    const sortformer_matrix_f32 & spkcache,
    const sortformer_matrix_f32 & fifo,
    const sortformer_matrix_f32 & chunk_preencode,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask);

sortformer_full_step_outputs sortformer_run_full_step_concat(
    const sortformer_model & model,
    const sortformer_matrix_f32 & chunk_features,
    uint32_t chunk_valid_feature_rows,
    const sortformer_matrix_f32 & spkcache,
    const sortformer_matrix_f32 & fifo,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask);

} // namespace llama::realtime
