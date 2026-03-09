#include "sortformer-layer0.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace llama::realtime {

namespace {

ggml_init_params make_graph_ctx_params() {
    ggml_init_params params = {};
    params.mem_size = 256ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    return params;
}

void require(bool cond, const char * message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

ggml_tensor * add_bias_2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    ggml_tensor * b = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, x, b);
}

ggml_tensor * mul_mat_checked(
    ggml_context * ctx,
    ggml_tensor * a,
    ggml_tensor * b,
    const char * label) {
    if (!(a->ne[0] == b->ne[0] && (b->ne[2] % a->ne[2] == 0) && (b->ne[3] % a->ne[3] == 0))) {
        throw std::runtime_error(std::string("ggml_can_mul_mat failed at ") + label);
    }
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

ggml_tensor * apply_layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, ggml_reshape_2d(ctx, weight, weight->ne[0], 1));
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, bias->ne[0], 1));
    return y;
}

ggml_tensor * mul_linear_project(
    ggml_context * ctx,
    ggml_tensor * w,
    ggml_tensor * x,
    bool use_columnwise_matvec,
    const char * label) {
    x = ggml_cont(ctx, x);
    if (!use_columnwise_matvec || x->ne[1] <= 1) {
        return mul_mat_checked(ctx, w, x, label);
    }

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < x->ne[1]; ++i) {
        ggml_tensor * col = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (size_t) i * x->nb[1]);
        col = ggml_cont(ctx, col);
        ggml_tensor * yi = mul_mat_checked(ctx, w, col, label);
        out = (out == nullptr) ? yi : ggml_concat(ctx, out, yi, 1);
    }
    return out;
}

ggml_tensor * apply_feed_forward(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w1,
    ggml_tensor * b1,
    ggml_tensor * w2,
    ggml_tensor * b2,
    bool use_columnwise_matvec,
    ggml_tensor ** mm1_out = nullptr,
    ggml_tensor ** l1_out = nullptr,
    ggml_tensor ** act_out = nullptr,
    ggml_tensor ** mm2_out = nullptr) {
    ggml_tensor * y = mul_linear_project(ctx, w1, x, use_columnwise_matvec, "feedforward.linear1");
    if (mm1_out != nullptr) {
        *mm1_out = y;
    }
    y = add_bias_2d(ctx, y, b1);
    if (l1_out != nullptr) {
        *l1_out = y;
    }
    y = ggml_silu(ctx, y);
    y = ggml_cont(ctx, y);
    if (act_out != nullptr) {
        *act_out = y;
    }
    y = mul_linear_project(ctx, w2, y, use_columnwise_matvec, "feedforward.linear2");
    if (mm2_out != nullptr) {
        *mm2_out = y;
    }
    y = add_bias_2d(ctx, y, b2);
    return y;
}

sortformer_matrix_f32 tensor_to_matrix_2d(ggml_tensor * tensor) {
    require(tensor->ne[2] == 1 && tensor->ne[3] == 1, "expected rank-2 tensor output");

    sortformer_matrix_f32 out;
    out.rows = static_cast<uint32_t>(tensor->ne[1]);
    out.cols = static_cast<uint32_t>(tensor->ne[0]);
    out.data.resize((size_t) out.rows * (size_t) out.cols);
    ggml_backend_tensor_get(tensor, out.data.data(), 0, out.data.size() * sizeof(float));
    return out;
}

} // namespace

sortformer_layer0_outputs sortformer_run_layer0(
    const sortformer_model & model,
    const sortformer_matrix_f32 & posenc_x,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask) {
    require(posenc_x.rows > 0 && posenc_x.cols == model.metadata().encoder_d_model, "invalid posenc_x shape");
    require(pos_emb.cols == model.metadata().encoder_d_model, "invalid pos_emb shape");
    require(pad_mask.rows == 1 && pad_mask.cols == posenc_x.rows, "invalid pad_mask shape");
    require(att_mask.rows == posenc_x.rows && att_mask.cols == posenc_x.rows, "invalid att_mask shape");

    const int64_t d_model = model.metadata().encoder_d_model;
    const int64_t n_head = model.metadata().encoder_heads;
    require(d_model > 0 && n_head > 0 && (d_model % n_head) == 0, "invalid encoder head configuration");
    const int64_t d_head = d_model / n_head;
    const std::string backend_name = ggml_backend_name(model.backend());
    const bool use_columnwise_matvec = backend_name.rfind("Vulkan", 0) == 0;

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer layer0 graph context");
    }

    sortformer_layer0_outputs out;

    try {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, posenc_x.cols, posenc_x.rows);
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        ggml_tensor * pad_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * att_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_set_name(x, "sortformer.layer0.x");
        ggml_set_name(pos, "sortformer.layer0.pos");
        ggml_set_name(pad_keep, "sortformer.layer0.pad_keep");
        ggml_set_name(att_bias, "sortformer.layer0.att_bias");
        ggml_set_name(att_keep, "sortformer.layer0.att_keep");
        ggml_set_input(x);
        ggml_set_input(pos);
        ggml_set_input(pad_keep);
        ggml_set_input(att_bias);
        ggml_set_input(att_keep);

        std::vector<float> pad_keep_data((size_t) pad_mask.cols, 1.0f);
        for (uint32_t i = 0; i < pad_mask.cols; ++i) {
            pad_keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
        }
        std::vector<float> att_bias_data((size_t) att_mask.rows * (size_t) att_mask.cols, 0.0f);
        std::vector<float> att_keep_data((size_t) att_mask.rows * (size_t) att_mask.cols, 1.0f);
        for (uint32_t r = 0; r < att_mask.rows; ++r) {
            const bool padded_query = pad_mask.data[r] > 0.5f;
            for (uint32_t c = 0; c < att_mask.cols; ++c) {
                const size_t idx = (size_t) r * (size_t) att_mask.cols + (size_t) c;
                att_bias_data[idx] = padded_query ? 0.0f : (att_mask.data[idx] > 0.5f ? -10000.0f : 0.0f);
                att_keep_data[idx] = att_mask.data[idx] > 0.5f ? 0.0f : 1.0f;
            }
        }
        ggml_tensor * residual = x;

        ggml_tensor * ff1_norm = apply_layer_norm(
            ctx, residual,
            model.tensor("enc.l0.nff1.w"),
            model.tensor("enc.l0.nff1.b"),
            1e-5f);
        ggml_tensor * ff1_mm = nullptr;
        ggml_tensor * ff1_l1 = nullptr;
        ggml_tensor * ff1_act = nullptr;
        ggml_tensor * ff1_out_mm = nullptr;
        ggml_tensor * ff1_out = apply_feed_forward(
            ctx,
            ff1_norm,
            model.tensor("enc.l0.ff1.l1.w"),
            model.tensor("enc.l0.ff1.l1.b"),
            model.tensor("enc.l0.ff1.l2.w"),
            model.tensor("enc.l0.ff1.l2.b"),
            use_columnwise_matvec,
            &ff1_mm,
            &ff1_l1,
            &ff1_act,
            &ff1_out_mm);
        ggml_tensor * ff1_res = ggml_add(ctx, residual, ggml_scale(ctx, ff1_out, 0.5f));

        ggml_tensor * att_norm = apply_layer_norm(
            ctx, ff1_res,
            model.tensor("enc.l0.nsa.w"),
            model.tensor("enc.l0.nsa.b"),
            1e-5f);

        ggml_tensor * Qcur = mul_linear_project(ctx, model.tensor("enc.l0.att.q.w"), att_norm, use_columnwise_matvec, "att.q");
        Qcur = add_bias_2d(ctx, Qcur, model.tensor("enc.l0.att.q.b"));
        Qcur = ggml_reshape_3d(ctx, Qcur, d_head, n_head, Qcur->ne[1]);
        ggml_tensor * pos_bias_u = ggml_reshape_3d(ctx, model.tensor("enc.l0.att.pbu"), d_head, n_head, 1);
        ggml_tensor * pos_bias_v = ggml_reshape_3d(ctx, model.tensor("enc.l0.att.pbv"), d_head, n_head, 1);
        ggml_tensor * Q_bias_u = ggml_add(ctx, Qcur, pos_bias_u);
        Q_bias_u = ggml_permute(ctx, Q_bias_u, 0, 2, 1, 3);
        ggml_tensor * Q_bias_v = ggml_add(ctx, Qcur, pos_bias_v);
        Q_bias_v = ggml_permute(ctx, Q_bias_v, 0, 2, 1, 3);

        ggml_tensor * Kcur = mul_linear_project(ctx, model.tensor("enc.l0.att.k.w"), att_norm, use_columnwise_matvec, "att.k");
        Kcur = add_bias_2d(ctx, Kcur, model.tensor("enc.l0.att.k.b"));
        Kcur = ggml_reshape_3d(ctx, Kcur, d_head, n_head, Kcur->ne[1]);
        Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

        ggml_tensor * Vcur = mul_linear_project(ctx, model.tensor("enc.l0.att.v.w"), att_norm, use_columnwise_matvec, "att.v");
        Vcur = add_bias_2d(ctx, Vcur, model.tensor("enc.l0.att.v.b"));
        Vcur = ggml_reshape_3d(ctx, Vcur, d_head, n_head, Vcur->ne[1]);
        Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

        ggml_tensor * matrix_ac = mul_mat_checked(ctx, Q_bias_u, Kcur, "att.matrix_ac");
        matrix_ac = ggml_cont(ctx, ggml_permute(ctx, matrix_ac, 1, 0, 2, 3));

        ggml_tensor * p = mul_linear_project(ctx, model.tensor("enc.l0.att.p.w"), pos, use_columnwise_matvec, "att.linear_pos");
        p = ggml_reshape_3d(ctx, p, d_head, n_head, p->ne[1]);
        p = ggml_permute(ctx, p, 0, 2, 1, 3);

        ggml_tensor * matrix_bd = mul_mat_checked(ctx, Q_bias_v, p, "att.matrix_bd");
        matrix_bd = ggml_cont(ctx, ggml_permute(ctx, matrix_bd, 1, 0, 2, 3));
        {
            const auto pos_len = matrix_bd->ne[0];
            const auto q_len = matrix_bd->ne[1];
            const auto h = matrix_bd->ne[2];
            matrix_bd = ggml_pad(ctx, matrix_bd, 1, 0, 0, 0);
            matrix_bd = ggml_roll(ctx, matrix_bd, 1, 0, 0, 0);
            matrix_bd = ggml_reshape_3d(ctx, matrix_bd, q_len, pos_len + 1, h);
            matrix_bd = ggml_view_3d(ctx, matrix_bd, q_len, pos_len, h, matrix_bd->nb[1], matrix_bd->nb[2], matrix_bd->nb[0] * q_len);
            matrix_bd = ggml_cont_3d(ctx, matrix_bd, pos_len, q_len, h);
        }
        matrix_bd = ggml_view_3d(ctx, matrix_bd, matrix_ac->ne[0], matrix_bd->ne[1], matrix_bd->ne[2], matrix_bd->nb[1], matrix_bd->nb[2], 0);

        ggml_tensor * scores = ggml_add(ctx, matrix_ac, matrix_bd);
        scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) d_head));
        scores = ggml_add(ctx, scores, att_bias);
        ggml_tensor * attn = ggml_soft_max(ctx, scores);
        attn = ggml_mul(ctx, attn, att_keep);

        ggml_tensor * matrix_ac_head0 = ggml_cont(ctx, ggml_view_2d(ctx, matrix_ac, matrix_ac->ne[0], matrix_ac->ne[1], matrix_ac->nb[1], 0));
        ggml_tensor * matrix_bd_head0 = ggml_cont(ctx, ggml_view_2d(ctx, matrix_bd, matrix_bd->ne[0], matrix_bd->ne[1], matrix_bd->nb[1], 0));
        ggml_tensor * scores_head0 = ggml_cont(ctx, ggml_view_2d(ctx, scores, scores->ne[0], scores->ne[1], scores->nb[1], 0));
        ggml_tensor * attn_head0 = ggml_cont(ctx, ggml_view_2d(ctx, attn, attn->ne[0], attn->ne[1], attn->nb[1], 0));

        ggml_tensor * att_value = mul_mat_checked(ctx, attn, Vcur, "att.value_mix");
        ggml_tensor * att_value_head0 = ggml_cont(ctx, ggml_transpose(ctx, ggml_view_2d(ctx, att_value, att_value->ne[0], att_value->ne[1], att_value->nb[1], 0)));
        ggml_tensor * att_x = att_value;
        att_x = ggml_permute(ctx, att_x, 2, 0, 1, 3);
        att_x = ggml_cont_2d(ctx, att_x, att_x->ne[0] * att_x->ne[1], att_x->ne[2]);

        ggml_tensor * att_out = mul_linear_project(ctx, model.tensor("enc.l0.att.o.w"), att_x, use_columnwise_matvec, "att.o");
        att_out = add_bias_2d(ctx, att_out, model.tensor("enc.l0.att.o.b"));
        ggml_tensor * att_res = ggml_add(ctx, ff1_res, att_out);

        ggml_tensor * conv_norm = apply_layer_norm(
            ctx, att_res,
            model.tensor("enc.l0.nc.w"),
            model.tensor("enc.l0.nc.b"),
            1e-5f);

        ggml_tensor * conv_pw1_w = ggml_reshape_2d(
            ctx,
            model.tensor("enc.l0.conv.pw1.w"),
            model.tensor("enc.l0.conv.pw1.w")->ne[1],
            model.tensor("enc.l0.conv.pw1.w")->ne[2]);
        ggml_tensor * conv_pw1 = mul_linear_project(ctx, conv_pw1_w, conv_norm, use_columnwise_matvec, "conv.pw1");
        conv_pw1 = add_bias_2d(ctx, conv_pw1, model.tensor("enc.l0.conv.pw1.b"));
        ggml_tensor * conv_x = conv_pw1;
        {
            const int64_t d = conv_x->ne[0] / 2;
            ggml_tensor * left = ggml_cont(ctx, ggml_view_2d(ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], 0));
            ggml_tensor * right = ggml_cont(ctx, ggml_view_2d(ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], d * conv_x->nb[0]));
            ggml_tensor * gate = ggml_sigmoid(ctx, right);
            conv_x = ggml_mul(ctx, left, gate);
        }
        ggml_tensor * conv_glu = conv_x;
        conv_x = ggml_mul(ctx, conv_x, pad_keep);
        conv_x = ggml_cont(ctx, ggml_transpose(ctx, conv_x)); // [time, channels]
        conv_x = ggml_reshape_3d(ctx, conv_x, conv_x->ne[0], conv_x->ne[1], 1);
        conv_x = ggml_pad(ctx, conv_x, 4, 0, 0, 0);
        conv_x = ggml_roll(ctx, conv_x, 4, 0, 0, 0);
        conv_x = ggml_pad(ctx, conv_x, 4, 0, 0, 0);

        ggml_tensor * conv_dw_w = ggml_reshape_2d(
            ctx,
            model.tensor("enc.l0.conv.dw.w"),
            model.tensor("enc.l0.conv.dw.w")->ne[0],
            model.tensor("enc.l0.conv.dw.w")->ne[2]);
        ggml_tensor * conv_dw = ggml_ssm_conv(ctx, conv_x, conv_dw_w);
        conv_dw = ggml_add(ctx, conv_dw, model.tensor("enc.l0.conv.dw.b"));
        conv_dw = ggml_reshape_2d(ctx, conv_dw, conv_dw->ne[0], conv_dw->ne[1]); // [channels, time]
        ggml_tensor * conv_dw_t = conv_dw;
        conv_dw = ggml_cont(ctx, ggml_transpose(ctx, conv_dw)); // [time, channels]
        ggml_tensor * conv_bn = ggml_mul(
            ctx,
            conv_dw,
            ggml_reshape_2d(ctx, model.tensor("enc.l0.conv.bn.sc"), 1, model.tensor("enc.l0.conv.bn.sc")->ne[0]));
        conv_bn = ggml_add(
            ctx,
            conv_bn,
            ggml_reshape_2d(ctx, model.tensor("enc.l0.conv.bn.sh"), 1, model.tensor("enc.l0.conv.bn.sh")->ne[0]));
        ggml_tensor * conv_act = ggml_silu(ctx, conv_bn);
        ggml_tensor * conv_bn_t = ggml_cont(ctx, ggml_transpose(ctx, conv_bn));
        ggml_tensor * conv_act_t = ggml_cont(ctx, ggml_transpose(ctx, conv_act));
        conv_dw = ggml_cont(ctx, ggml_transpose(ctx, conv_act)); // [channels, time]

        ggml_tensor * conv_pw2_w = ggml_reshape_2d(
            ctx,
            model.tensor("enc.l0.conv.pw2.w"),
            model.tensor("enc.l0.conv.pw2.w")->ne[1],
            model.tensor("enc.l0.conv.pw2.w")->ne[2]);
        ggml_tensor * conv_out = mul_linear_project(ctx, conv_pw2_w, conv_dw, use_columnwise_matvec, "conv.pw2");
        conv_out = add_bias_2d(ctx, conv_out, model.tensor("enc.l0.conv.pw2.b"));
        ggml_tensor * conv_pw2 = conv_out;
        ggml_tensor * conv_res = ggml_add(ctx, att_res, conv_out);

        ggml_tensor * ff2_norm = apply_layer_norm(
            ctx, conv_res,
            model.tensor("enc.l0.nff2.w"),
            model.tensor("enc.l0.nff2.b"),
            1e-5f);
        ggml_tensor * ff2_out = apply_feed_forward(
            ctx,
            ff2_norm,
            model.tensor("enc.l0.ff2.l1.w"),
            model.tensor("enc.l0.ff2.l1.b"),
            model.tensor("enc.l0.ff2.l2.w"),
            model.tensor("enc.l0.ff2.l2.b"),
            use_columnwise_matvec);
        ggml_tensor * ff2_res = ggml_add(ctx, conv_res, ggml_scale(ctx, ff2_out, 0.5f));

        ggml_tensor * layer_out = apply_layer_norm(
            ctx, ff2_res,
            model.tensor("enc.l0.no.w"),
            model.tensor("enc.l0.no.b"),
            1e-5f);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer layer0 graph tensors");
        }

        ggml_backend_tensor_set(x, posenc_x.data.data(), 0, posenc_x.data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
        ggml_backend_tensor_set(pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(gf, matrix_ac_head0);
        ggml_build_forward_expand(gf, matrix_bd_head0);
        ggml_build_forward_expand(gf, scores_head0);
        ggml_build_forward_expand(gf, attn_head0);
        ggml_build_forward_expand(gf, att_value_head0);
        ggml_build_forward_expand(gf, att_x);
        ggml_build_forward_expand(gf, layer_out);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer layer0");
        }

        out.ff1_norm = tensor_to_matrix_2d(ff1_norm);
        out.ff1_mm = tensor_to_matrix_2d(ff1_mm);
        out.ff1_l1 = tensor_to_matrix_2d(ff1_l1);
        out.ff1_act = tensor_to_matrix_2d(ff1_act);
        out.ff1_out_mm = tensor_to_matrix_2d(ff1_out_mm);
        out.ff1_out = tensor_to_matrix_2d(ff1_out);
        out.ff1_res = tensor_to_matrix_2d(ff1_res);
        out.att_norm = tensor_to_matrix_2d(att_norm);
        out.matrix_ac_head0 = tensor_to_matrix_2d(matrix_ac_head0);
        out.matrix_bd_head0 = tensor_to_matrix_2d(matrix_bd_head0);
        out.scores_head0 = tensor_to_matrix_2d(scores_head0);
        out.attn_head0 = tensor_to_matrix_2d(attn_head0);
        out.att_value_head0 = tensor_to_matrix_2d(att_value_head0);
        out.att_x = tensor_to_matrix_2d(att_x);
        out.att_out = tensor_to_matrix_2d(att_out);
        out.att_res = tensor_to_matrix_2d(att_res);
        out.conv_norm = tensor_to_matrix_2d(conv_norm);
        out.conv_pw1 = tensor_to_matrix_2d(conv_pw1);
        out.conv_glu = tensor_to_matrix_2d(conv_glu);
        out.conv_dw = tensor_to_matrix_2d(conv_dw_t);
        out.conv_bn = tensor_to_matrix_2d(conv_bn_t);
        out.conv_act = tensor_to_matrix_2d(conv_act_t);
        out.conv_pw2 = tensor_to_matrix_2d(conv_pw2);
        out.conv_out = tensor_to_matrix_2d(conv_out);
        out.conv_res = tensor_to_matrix_2d(conv_res);
        out.ff2_norm = tensor_to_matrix_2d(ff2_norm);
        out.ff2_out = tensor_to_matrix_2d(ff2_out);
        out.ff2_res = tensor_to_matrix_2d(ff2_res);
        out.out = tensor_to_matrix_2d(layer_out);

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

} // namespace llama::realtime
