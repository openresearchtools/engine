#include "sortformer-encoder.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama::realtime {

namespace {

ggml_init_params make_graph_ctx_params() {
    ggml_init_params params = {};
    params.mem_size = 512ull * 1024ull * 1024ull;
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
    const std::string & label) {
    if (!(a->ne[0] == b->ne[0] && (b->ne[2] % a->ne[2] == 0) && (b->ne[3] % a->ne[3] == 0))) {
        throw std::runtime_error("ggml_can_mul_mat failed at " + label);
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
    const std::string & label) {
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
    const std::string & label_prefix) {
    ggml_tensor * y = mul_linear_project(ctx, w1, x, use_columnwise_matvec, label_prefix + ".linear1");
    y = add_bias_2d(ctx, y, b1);
    y = ggml_silu(ctx, y);
    y = ggml_cont(ctx, y);
    y = mul_linear_project(ctx, w2, y, use_columnwise_matvec, label_prefix + ".linear2");
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

sortformer_matrix_f32 run_encoder_layer_vulkan(
    const sortformer_model & model,
    uint32_t il,
    const sortformer_matrix_f32 & x_in,
    const sortformer_matrix_f32 & pos_emb,
    const std::vector<float> & pad_keep_data,
    const std::vector<float> & att_bias_data,
    const std::vector<float> & att_keep_data,
    bool use_columnwise_matvec,
    int64_t d_head,
    int64_t n_head) {
    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer encoder layer graph context");
    }

    sortformer_matrix_f32 out;

    try {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_in.cols, x_in.rows);
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        ggml_tensor * pad_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, (int64_t) pad_keep_data.size());
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_in.rows, x_in.rows);
        ggml_tensor * att_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_in.rows, x_in.rows);
        ggml_set_input(x);
        ggml_set_input(pos);
        ggml_set_input(pad_keep);
        ggml_set_input(att_bias);
        ggml_set_input(att_keep);

        ggml_tensor * residual = x;
        const std::string prefix = "enc.l" + std::to_string(il);
        auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
            return model.tensor(prefix + suffix);
        };

        ggml_tensor * ff1_norm = apply_layer_norm(ctx, residual, tensor(".nff1.w"), tensor(".nff1.b"), 1e-5f);
        ggml_tensor * ff1_out = apply_feed_forward(
            ctx,
            ff1_norm,
            tensor(".ff1.l1.w"),
            tensor(".ff1.l1.b"),
            tensor(".ff1.l2.w"),
            tensor(".ff1.l2.b"),
            use_columnwise_matvec,
            prefix + ".ff1");
        ggml_tensor * ff1_res = ggml_add(ctx, residual, ggml_scale(ctx, ff1_out, 0.5f));

        ggml_tensor * att_norm = apply_layer_norm(ctx, ff1_res, tensor(".nsa.w"), tensor(".nsa.b"), 1e-5f);

        ggml_tensor * Qcur = mul_linear_project(ctx, tensor(".att.q.w"), att_norm, use_columnwise_matvec, prefix + ".att.q");
        Qcur = add_bias_2d(ctx, Qcur, tensor(".att.q.b"));
        Qcur = ggml_reshape_3d(ctx, Qcur, d_head, n_head, Qcur->ne[1]);
        ggml_tensor * pos_bias_u = ggml_reshape_3d(ctx, tensor(".att.pbu"), d_head, n_head, 1);
        ggml_tensor * pos_bias_v = ggml_reshape_3d(ctx, tensor(".att.pbv"), d_head, n_head, 1);
        ggml_tensor * Q_bias_u = ggml_add(ctx, Qcur, pos_bias_u);
        Q_bias_u = ggml_permute(ctx, Q_bias_u, 0, 2, 1, 3);
        ggml_tensor * Q_bias_v = ggml_add(ctx, Qcur, pos_bias_v);
        Q_bias_v = ggml_permute(ctx, Q_bias_v, 0, 2, 1, 3);

        ggml_tensor * Kcur = mul_linear_project(ctx, tensor(".att.k.w"), att_norm, use_columnwise_matvec, prefix + ".att.k");
        Kcur = add_bias_2d(ctx, Kcur, tensor(".att.k.b"));
        Kcur = ggml_reshape_3d(ctx, Kcur, d_head, n_head, Kcur->ne[1]);
        Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

        ggml_tensor * Vcur = mul_linear_project(ctx, tensor(".att.v.w"), att_norm, use_columnwise_matvec, prefix + ".att.v");
        Vcur = add_bias_2d(ctx, Vcur, tensor(".att.v.b"));
        Vcur = ggml_reshape_3d(ctx, Vcur, d_head, n_head, Vcur->ne[1]);
        Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

        ggml_tensor * matrix_ac = mul_mat_checked(ctx, Q_bias_u, Kcur, prefix + ".att.matrix_ac");
        matrix_ac = ggml_cont(ctx, ggml_permute(ctx, matrix_ac, 1, 0, 2, 3));

        ggml_tensor * p = mul_linear_project(ctx, tensor(".att.p.w"), pos, use_columnwise_matvec, prefix + ".att.linear_pos");
        p = ggml_reshape_3d(ctx, p, d_head, n_head, p->ne[1]);
        p = ggml_permute(ctx, p, 0, 2, 1, 3);

        ggml_tensor * matrix_bd = mul_mat_checked(ctx, Q_bias_v, p, prefix + ".att.matrix_bd");
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

        ggml_tensor * att_x = mul_mat_checked(ctx, attn, Vcur, prefix + ".att.value_mix");
        att_x = ggml_permute(ctx, att_x, 2, 0, 1, 3);
        att_x = ggml_cont_2d(ctx, att_x, att_x->ne[0] * att_x->ne[1], att_x->ne[2]);

        ggml_tensor * att_out = mul_linear_project(ctx, tensor(".att.o.w"), att_x, use_columnwise_matvec, prefix + ".att.o");
        att_out = add_bias_2d(ctx, att_out, tensor(".att.o.b"));
        ggml_tensor * att_res = ggml_add(ctx, ff1_res, att_out);

        ggml_tensor * conv_norm = apply_layer_norm(ctx, att_res, tensor(".nc.w"), tensor(".nc.b"), 1e-5f);
        ggml_tensor * conv_pw1_w = ggml_reshape_2d(ctx, tensor(".conv.pw1.w"), tensor(".conv.pw1.w")->ne[1], tensor(".conv.pw1.w")->ne[2]);
        ggml_tensor * conv_x = mul_linear_project(ctx, conv_pw1_w, conv_norm, use_columnwise_matvec, prefix + ".conv.pw1");
        conv_x = add_bias_2d(ctx, conv_x, tensor(".conv.pw1.b"));
        {
            const int64_t d = conv_x->ne[0] / 2;
            ggml_tensor * left = ggml_cont(ctx, ggml_view_2d(ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], 0));
            ggml_tensor * right = ggml_cont(ctx, ggml_view_2d(ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], d * conv_x->nb[0]));
            ggml_tensor * gate = ggml_sigmoid(ctx, right);
            conv_x = ggml_mul(ctx, left, gate);
        }
        conv_x = ggml_mul(ctx, conv_x, pad_keep);
        conv_x = ggml_cont(ctx, ggml_transpose(ctx, conv_x));
        conv_x = ggml_reshape_3d(ctx, conv_x, conv_x->ne[0], conv_x->ne[1], 1);
        conv_x = ggml_pad(ctx, conv_x, 4, 0, 0, 0);
        conv_x = ggml_roll(ctx, conv_x, 4, 0, 0, 0);
        conv_x = ggml_pad(ctx, conv_x, 4, 0, 0, 0);

        ggml_tensor * conv_dw_w = ggml_reshape_2d(ctx, tensor(".conv.dw.w"), tensor(".conv.dw.w")->ne[0], tensor(".conv.dw.w")->ne[2]);
        ggml_tensor * conv_dw = ggml_ssm_conv(ctx, conv_x, conv_dw_w);
        conv_dw = ggml_add(ctx, conv_dw, tensor(".conv.dw.b"));
        conv_dw = ggml_reshape_2d(ctx, conv_dw, conv_dw->ne[0], conv_dw->ne[1]);

        conv_dw = ggml_cont(ctx, ggml_transpose(ctx, conv_dw));
        conv_dw = ggml_mul(ctx, conv_dw, ggml_reshape_2d(ctx, tensor(".conv.bn.sc"), 1, tensor(".conv.bn.sc")->ne[0]));
        conv_dw = ggml_add(ctx, conv_dw, ggml_reshape_2d(ctx, tensor(".conv.bn.sh"), 1, tensor(".conv.bn.sh")->ne[0]));
        conv_dw = ggml_silu(ctx, conv_dw);
        conv_dw = ggml_cont(ctx, ggml_transpose(ctx, conv_dw));

        ggml_tensor * conv_pw2_w = ggml_reshape_2d(ctx, tensor(".conv.pw2.w"), tensor(".conv.pw2.w")->ne[1], tensor(".conv.pw2.w")->ne[2]);
        ggml_tensor * conv_out = mul_linear_project(ctx, conv_pw2_w, conv_dw, use_columnwise_matvec, prefix + ".conv.pw2");
        conv_out = add_bias_2d(ctx, conv_out, tensor(".conv.pw2.b"));
        ggml_tensor * conv_res = ggml_add(ctx, att_res, conv_out);

        ggml_tensor * ff2_norm = apply_layer_norm(ctx, conv_res, tensor(".nff2.w"), tensor(".nff2.b"), 1e-5f);
        ggml_tensor * ff2_out = apply_feed_forward(
            ctx,
            ff2_norm,
            tensor(".ff2.l1.w"),
            tensor(".ff2.l1.b"),
            tensor(".ff2.l2.w"),
            tensor(".ff2.l2.b"),
            use_columnwise_matvec,
            prefix + ".ff2");
        ggml_tensor * ff2_res = ggml_add(ctx, conv_res, ggml_scale(ctx, ff2_out, 0.5f));

        ggml_tensor * cur = apply_layer_norm(ctx, ff2_res, tensor(".no.w"), tensor(".no.b"), 1e-5f);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer encoder layer graph tensors");
        }

        ggml_backend_tensor_set(x, x_in.data.data(), 0, x_in.data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
        ggml_backend_tensor_set(pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, cur);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer encoder layer");
        }

        out = tensor_to_matrix_2d(cur);

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

} // namespace

sortformer_matrix_f32 sortformer_run_encoder(
    const sortformer_model & model,
    const sortformer_matrix_f32 & preencoded,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask) {
    require(preencoded.rows > 0 && preencoded.cols == model.metadata().encoder_d_model, "invalid preencoded shape");
    require(pos_emb.cols == model.metadata().encoder_d_model, "invalid pos_emb shape");
    require(pad_mask.rows == 1 && pad_mask.cols == preencoded.rows, "invalid pad_mask shape");
    require(att_mask.rows == preencoded.rows && att_mask.cols == preencoded.rows, "invalid att_mask shape");

    const int64_t d_model = model.metadata().encoder_d_model;
    const int64_t n_head = model.metadata().encoder_heads;
    require(d_model > 0 && n_head > 0 && (d_model % n_head) == 0, "invalid encoder head configuration");
    const int64_t d_head = d_model / n_head;
    const bool use_columnwise_matvec = false;

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer encoder graph context");
    }

    sortformer_matrix_f32 out;

    try {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, preencoded.cols, preencoded.rows);
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        ggml_tensor * pad_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * att_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
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

        ggml_tensor * cur = ggml_scale(ctx, x, std::sqrt((float) d_model));

        for (uint32_t il = 0; il < model.metadata().encoder_layers; ++il) {
            const std::string prefix = "enc.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * ff1_norm = apply_layer_norm(ctx, residual, tensor(".nff1.w"), tensor(".nff1.b"), 1e-5f);
            ggml_tensor * ff1_out = apply_feed_forward(
                ctx,
                ff1_norm,
                tensor(".ff1.l1.w"),
                tensor(".ff1.l1.b"),
                tensor(".ff1.l2.w"),
                tensor(".ff1.l2.b"),
                use_columnwise_matvec,
                prefix + ".ff1");
            ggml_tensor * ff1_res = ggml_add(ctx, residual, ggml_scale(ctx, ff1_out, 0.5f));

            ggml_tensor * att_norm = apply_layer_norm(ctx, ff1_res, tensor(".nsa.w"), tensor(".nsa.b"), 1e-5f);

            ggml_tensor * Qcur = mul_linear_project(ctx, tensor(".att.q.w"), att_norm, use_columnwise_matvec, prefix + ".att.q");
            Qcur = add_bias_2d(ctx, Qcur, tensor(".att.q.b"));
            Qcur = ggml_reshape_3d(ctx, Qcur, d_head, n_head, Qcur->ne[1]);
            ggml_tensor * pos_bias_u = ggml_reshape_3d(ctx, tensor(".att.pbu"), d_head, n_head, 1);
            ggml_tensor * pos_bias_v = ggml_reshape_3d(ctx, tensor(".att.pbv"), d_head, n_head, 1);
            ggml_tensor * Q_bias_u = ggml_add(ctx, Qcur, pos_bias_u);
            Q_bias_u = ggml_permute(ctx, Q_bias_u, 0, 2, 1, 3);
            ggml_tensor * Q_bias_v = ggml_add(ctx, Qcur, pos_bias_v);
            Q_bias_v = ggml_permute(ctx, Q_bias_v, 0, 2, 1, 3);

            ggml_tensor * Kcur = mul_linear_project(ctx, tensor(".att.k.w"), att_norm, use_columnwise_matvec, prefix + ".att.k");
            Kcur = add_bias_2d(ctx, Kcur, tensor(".att.k.b"));
            Kcur = ggml_reshape_3d(ctx, Kcur, d_head, n_head, Kcur->ne[1]);
            Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = mul_linear_project(ctx, tensor(".att.v.w"), att_norm, use_columnwise_matvec, prefix + ".att.v");
            Vcur = add_bias_2d(ctx, Vcur, tensor(".att.v.b"));
            Vcur = ggml_reshape_3d(ctx, Vcur, d_head, n_head, Vcur->ne[1]);
            Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

            ggml_tensor * matrix_ac = mul_mat_checked(ctx, Q_bias_u, Kcur, prefix + ".att.matrix_ac");
            matrix_ac = ggml_cont(ctx, ggml_permute(ctx, matrix_ac, 1, 0, 2, 3));

            ggml_tensor * p = mul_linear_project(ctx, tensor(".att.p.w"), pos, use_columnwise_matvec, prefix + ".att.linear_pos");
            p = ggml_reshape_3d(ctx, p, d_head, n_head, p->ne[1]);
            p = ggml_permute(ctx, p, 0, 2, 1, 3);

            ggml_tensor * matrix_bd = mul_mat_checked(ctx, Q_bias_v, p, prefix + ".att.matrix_bd");
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

            ggml_tensor * att_x = mul_mat_checked(ctx, attn, Vcur, prefix + ".att.value_mix");
            att_x = ggml_permute(ctx, att_x, 2, 0, 1, 3);
            att_x = ggml_cont_2d(ctx, att_x, att_x->ne[0] * att_x->ne[1], att_x->ne[2]);

            ggml_tensor * att_out = mul_linear_project(ctx, tensor(".att.o.w"), att_x, use_columnwise_matvec, prefix + ".att.o");
            att_out = add_bias_2d(ctx, att_out, tensor(".att.o.b"));
            ggml_tensor * att_res = ggml_add(ctx, ff1_res, att_out);

            ggml_tensor * conv_norm = apply_layer_norm(ctx, att_res, tensor(".nc.w"), tensor(".nc.b"), 1e-5f);
            ggml_tensor * conv_pw1_w = ggml_reshape_2d(ctx, tensor(".conv.pw1.w"), tensor(".conv.pw1.w")->ne[1], tensor(".conv.pw1.w")->ne[2]);
            ggml_tensor * conv_x = mul_linear_project(ctx, conv_pw1_w, conv_norm, use_columnwise_matvec, prefix + ".conv.pw1");
            conv_x = add_bias_2d(ctx, conv_x, tensor(".conv.pw1.b"));
            {
                const int64_t d = conv_x->ne[0] / 2;
                ggml_tensor * left = ggml_cont(ctx, ggml_view_2d(ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], 0));
                ggml_tensor * right = ggml_cont(ctx, ggml_view_2d(ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], d * conv_x->nb[0]));
                ggml_tensor * gate = ggml_sigmoid(ctx, right);
                conv_x = ggml_mul(ctx, left, gate);
            }
            conv_x = ggml_mul(ctx, conv_x, pad_keep);
            conv_x = ggml_cont(ctx, ggml_transpose(ctx, conv_x));
            conv_x = ggml_reshape_3d(ctx, conv_x, conv_x->ne[0], conv_x->ne[1], 1);
            conv_x = ggml_pad(ctx, conv_x, 4, 0, 0, 0);
            conv_x = ggml_roll(ctx, conv_x, 4, 0, 0, 0);
            conv_x = ggml_pad(ctx, conv_x, 4, 0, 0, 0);

            ggml_tensor * conv_dw_w = ggml_reshape_2d(ctx, tensor(".conv.dw.w"), tensor(".conv.dw.w")->ne[0], tensor(".conv.dw.w")->ne[2]);
            ggml_tensor * conv_dw = ggml_ssm_conv(ctx, conv_x, conv_dw_w);
            conv_dw = ggml_add(ctx, conv_dw, tensor(".conv.dw.b"));
            conv_dw = ggml_reshape_2d(ctx, conv_dw, conv_dw->ne[0], conv_dw->ne[1]);

            conv_dw = ggml_cont(ctx, ggml_transpose(ctx, conv_dw));
            conv_dw = ggml_mul(ctx, conv_dw, ggml_reshape_2d(ctx, tensor(".conv.bn.sc"), 1, tensor(".conv.bn.sc")->ne[0]));
            conv_dw = ggml_add(ctx, conv_dw, ggml_reshape_2d(ctx, tensor(".conv.bn.sh"), 1, tensor(".conv.bn.sh")->ne[0]));
            conv_dw = ggml_silu(ctx, conv_dw);
            conv_dw = ggml_cont(ctx, ggml_transpose(ctx, conv_dw));

            ggml_tensor * conv_pw2_w = ggml_reshape_2d(ctx, tensor(".conv.pw2.w"), tensor(".conv.pw2.w")->ne[1], tensor(".conv.pw2.w")->ne[2]);
            ggml_tensor * conv_out = mul_linear_project(ctx, conv_pw2_w, conv_dw, use_columnwise_matvec, prefix + ".conv.pw2");
            conv_out = add_bias_2d(ctx, conv_out, tensor(".conv.pw2.b"));
            ggml_tensor * conv_res = ggml_add(ctx, att_res, conv_out);

            ggml_tensor * ff2_norm = apply_layer_norm(ctx, conv_res, tensor(".nff2.w"), tensor(".nff2.b"), 1e-5f);
            ggml_tensor * ff2_out = apply_feed_forward(
                ctx,
                ff2_norm,
                tensor(".ff2.l1.w"),
                tensor(".ff2.l1.b"),
                tensor(".ff2.l2.w"),
                tensor(".ff2.l2.b"),
                use_columnwise_matvec,
                prefix + ".ff2");
            ggml_tensor * ff2_res = ggml_add(ctx, conv_res, ggml_scale(ctx, ff2_out, 0.5f));

            cur = apply_layer_norm(ctx, ff2_res, tensor(".no.w"), tensor(".no.b"), 1e-5f);
        }

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer encoder graph tensors");
        }

        ggml_backend_tensor_set(x, preencoded.data.data(), 0, preencoded.data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
        ggml_backend_tensor_set(pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 262144, false);
        ggml_build_forward_expand(gf, cur);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer encoder");
        }

        out = tensor_to_matrix_2d(cur);

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

} // namespace llama::realtime
