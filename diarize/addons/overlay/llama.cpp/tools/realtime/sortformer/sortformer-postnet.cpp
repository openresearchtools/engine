#include "sortformer-postnet.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llama::realtime {

namespace {

ggml_init_params make_graph_ctx_params() {
    ggml_init_params params = {};
    params.mem_size = 768ull * 1024ull * 1024ull;
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

ggml_tensor * add_bias_4d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    ggml_tensor * b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
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

ggml_tensor * conv2d_exact(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * x,
    bool use_blocked_matmul,
    int stride0,
    int stride1,
    int padding0,
    int padding1,
    int dilation0,
    int dilation1) {
    ggml_tensor * im2col = ggml_im2col(ctx, weight, x, stride0, stride1, padding0, padding1, dilation0, dilation1, true, GGML_TYPE_F32);
    ggml_tensor * result = mul_linear_project(
        ctx,
        ggml_reshape_2d(ctx, weight, weight->ne[0] * weight->ne[1] * weight->ne[2], weight->ne[3]),
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
        use_blocked_matmul,
        "sortformer.pre.conv");
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], weight->ne[3]);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
    return result;
}

ggml_tensor * conv2d_dw_exact(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * x,
    bool use_blocked_matmul,
    int stride0,
    int stride1,
    int padding0,
    int padding1,
    int dilation0,
    int dilation1) {
    ggml_tensor * new_w = ggml_reshape_4d(ctx, weight, weight->ne[0], weight->ne[1], 1, weight->ne[2] * weight->ne[3]);
    ggml_tensor * im2col = ggml_im2col(
        ctx,
        new_w,
        ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], 1, x->ne[2] * x->ne[3]),
        stride0,
        stride1,
        padding0,
        padding1,
        dilation0,
        dilation1,
        true,
        GGML_TYPE_F32);
    ggml_tensor * new_x = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], x->ne[2], x->ne[3]);
    new_w = ggml_reshape_4d(ctx, new_w, new_w->ne[0] * new_w->ne[1], new_w->ne[2], new_w->ne[3], 1);
    GGML_UNUSED(use_blocked_matmul);
    ggml_tensor * result = mul_mat_checked(ctx, new_w, new_x, "sortformer.pre.conv_dw");
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], x->ne[2], x->ne[3]);
    return result;
}

ggml_tensor * conv2d_relu(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias,
    bool use_blocked_matmul,
    int stride0,
    int stride1,
    int padding0,
    int padding1,
    bool depthwise,
    bool direct,
    bool relu) {
    GGML_UNUSED(direct);
    ggml_tensor * y = depthwise
        ? conv2d_dw_exact(ctx, weight, x, use_blocked_matmul, stride0, stride1, padding0, padding1, 1, 1)
        : conv2d_exact(ctx, weight, x, use_blocked_matmul, stride0, stride1, padding0, padding1, 1, 1);
    y = add_bias_4d(ctx, y, bias);
    if (relu) {
        y = ggml_relu_inplace(ctx, y);
    }
    return y;
}

uint32_t calculate_conv_output_size(uint32_t input_size, uint32_t kernel_size, uint32_t stride, uint32_t pad_left, uint32_t pad_right) {
    return (input_size + pad_left + pad_right - kernel_size) / stride + 1;
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

struct sortformer_encoder_postnet_plan::impl {
    const sortformer_model * model = nullptr;
    uint32_t spk_rows = 0;
    uint32_t fifo_rows = 0;
    uint32_t chunk_rows = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t graph_buf = nullptr;
    ggml_cgraph * gf = nullptr;

    ggml_tensor * spk_t = nullptr;
    ggml_tensor * fifo_t = nullptr;
    ggml_tensor * chunk_t = nullptr;
    ggml_tensor * pos = nullptr;
    ggml_tensor * pad_keep = nullptr;
    ggml_tensor * att_bias = nullptr;
    ggml_tensor * att_keep = nullptr;
    ggml_tensor * keep_t = nullptr;
    ggml_tensor * tf_att_bias = nullptr;
    ggml_tensor * preds = nullptr;

    ~impl() {
        if (graph_buf != nullptr) {
            ggml_backend_buffer_free(graph_buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

sortformer_encoder_postnet_plan::sortformer_encoder_postnet_plan() = default;

sortformer_encoder_postnet_plan::~sortformer_encoder_postnet_plan() {
    delete impl_;
}

sortformer_encoder_postnet_plan::sortformer_encoder_postnet_plan(const sortformer_encoder_postnet_plan & other) {
    GGML_UNUSED(other);
    impl_ = nullptr;
}

sortformer_encoder_postnet_plan & sortformer_encoder_postnet_plan::operator=(const sortformer_encoder_postnet_plan & other) {
    GGML_UNUSED(other);
    delete impl_;
    impl_ = nullptr;
    return *this;
}

sortformer_encoder_postnet_plan::sortformer_encoder_postnet_plan(sortformer_encoder_postnet_plan && other) noexcept {
    *this = std::move(other);
}

sortformer_encoder_postnet_plan & sortformer_encoder_postnet_plan::operator=(sortformer_encoder_postnet_plan && other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

sortformer_matrix_f32 sortformer_run_encoder_postnet(
    const sortformer_model & model,
    const sortformer_matrix_f32 & preencoded,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask) {
    require(preencoded.rows > 0 && preencoded.cols == model.metadata().encoder_d_model, "invalid preencoded shape");
    require(pos_emb.cols == model.metadata().encoder_d_model, "invalid pos_emb shape");
    require(pad_mask.rows == 1 && pad_mask.cols == preencoded.rows, "invalid pad_mask shape");
    require(att_mask.rows == preencoded.rows && att_mask.cols == preencoded.rows, "invalid att_mask shape");

    const auto & meta = model.metadata();
    const int64_t enc_d_model = meta.encoder_d_model;
    const int64_t enc_n_head = meta.encoder_heads;
    require(enc_d_model > 0 && enc_n_head > 0 && (enc_d_model % enc_n_head) == 0, "invalid encoder head configuration");
    const int64_t enc_d_head = enc_d_model / enc_n_head;

    require(meta.tf_d_model > 0 && meta.transformer_heads > 0, "invalid transformer metadata");
    require((meta.tf_d_model % meta.transformer_heads) == 0, "invalid transformer head configuration");
    const int64_t tf_d_model = meta.tf_d_model;
    const int64_t tf_n_head = meta.transformer_heads;
    const int64_t tf_d_head = tf_d_model / tf_n_head;
    const bool use_columnwise_matvec = false;

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer fused encoder/postnet graph context");
    }

    sortformer_matrix_f32 out;

    try {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, preencoded.cols, preencoded.rows);
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        ggml_tensor * pad_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * att_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * keep_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * tf_att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pad_mask.cols, pad_mask.cols);
        ggml_set_input(x);
        ggml_set_input(pos);
        ggml_set_input(pad_keep);
        ggml_set_input(att_bias);
        ggml_set_input(att_keep);
        ggml_set_input(keep_t);
        ggml_set_input(tf_att_bias);

        std::vector<float> pad_keep_data((size_t) pad_mask.cols, 1.0f);
        std::vector<float> keep_data((size_t) pad_mask.cols, 0.0f);
        for (uint32_t i = 0; i < pad_mask.cols; ++i) {
            pad_keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
            keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
        }

        std::vector<float> att_bias_data((size_t) att_mask.rows * (size_t) att_mask.cols, 0.0f);
        std::vector<float> att_keep_data((size_t) att_mask.rows * (size_t) att_mask.cols, 1.0f);
        std::vector<float> tf_att_bias_data((size_t) pad_mask.cols * (size_t) pad_mask.cols, 0.0f);
        for (uint32_t r = 0; r < att_mask.rows; ++r) {
            const bool padded_query = pad_mask.data[r] > 0.5f;
            for (uint32_t c = 0; c < att_mask.cols; ++c) {
                const size_t idx = (size_t) r * (size_t) att_mask.cols + (size_t) c;
                att_bias_data[idx] = padded_query ? 0.0f : (att_mask.data[idx] > 0.5f ? -10000.0f : 0.0f);
                att_keep_data[idx] = att_mask.data[idx] > 0.5f ? 0.0f : 1.0f;
                tf_att_bias_data[idx] = pad_mask.data[c] > 0.5f ? 0.0f : -10000.0f;
            }
        }

        ggml_tensor * cur = ggml_scale(ctx, x, std::sqrt((float) enc_d_model));

        for (uint32_t il = 0; il < meta.encoder_layers; ++il) {
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
            Qcur = ggml_reshape_3d(ctx, Qcur, enc_d_head, enc_n_head, Qcur->ne[1]);
            ggml_tensor * pos_bias_u = ggml_reshape_3d(ctx, tensor(".att.pbu"), enc_d_head, enc_n_head, 1);
            ggml_tensor * pos_bias_v = ggml_reshape_3d(ctx, tensor(".att.pbv"), enc_d_head, enc_n_head, 1);
            ggml_tensor * Q_bias_u = ggml_add(ctx, Qcur, pos_bias_u);
            Q_bias_u = ggml_permute(ctx, Q_bias_u, 0, 2, 1, 3);
            ggml_tensor * Q_bias_v = ggml_add(ctx, Qcur, pos_bias_v);
            Q_bias_v = ggml_permute(ctx, Q_bias_v, 0, 2, 1, 3);

            ggml_tensor * Kcur = mul_linear_project(ctx, tensor(".att.k.w"), att_norm, use_columnwise_matvec, prefix + ".att.k");
            Kcur = add_bias_2d(ctx, Kcur, tensor(".att.k.b"));
            Kcur = ggml_reshape_3d(ctx, Kcur, enc_d_head, enc_n_head, Kcur->ne[1]);
            Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = mul_linear_project(ctx, tensor(".att.v.w"), att_norm, use_columnwise_matvec, prefix + ".att.v");
            Vcur = add_bias_2d(ctx, Vcur, tensor(".att.v.b"));
            Vcur = ggml_reshape_3d(ctx, Vcur, enc_d_head, enc_n_head, Vcur->ne[1]);
            Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

            ggml_tensor * matrix_ac = mul_mat_checked(ctx, Q_bias_u, Kcur, prefix + ".att.matrix_ac");
            matrix_ac = ggml_cont(ctx, ggml_permute(ctx, matrix_ac, 1, 0, 2, 3));

            ggml_tensor * p = mul_linear_project(ctx, tensor(".att.p.w"), pos, use_columnwise_matvec, prefix + ".att.linear_pos");
            p = ggml_reshape_3d(ctx, p, enc_d_head, enc_n_head, p->ne[1]);
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
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) enc_d_head));
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

        if (preencoded.cols == meta.fc_d_model) {
            cur = mul_linear_project(ctx, model.tensor("mods.ep.w"), cur, use_columnwise_matvec, "mods.ep");
            cur = add_bias_2d(ctx, cur, model.tensor("mods.ep.b"));
        } else if (preencoded.cols != meta.tf_d_model) {
            throw std::runtime_error("encoder output columns do not match fc_d_model or tf_d_model");
        }

        for (uint32_t il = 0; il < meta.transformer_layers; ++il) {
            const std::string prefix = "te.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * q = mul_linear_project(ctx, tensor(".sa.q.w"), cur, use_columnwise_matvec, prefix + ".sa.q");
            q = add_bias_2d(ctx, q, tensor(".sa.q.b"));
            ggml_tensor * k = mul_linear_project(ctx, tensor(".sa.k.w"), cur, use_columnwise_matvec, prefix + ".sa.k");
            k = add_bias_2d(ctx, k, tensor(".sa.k.b"));
            ggml_tensor * v = mul_linear_project(ctx, tensor(".sa.v.w"), cur, use_columnwise_matvec, prefix + ".sa.v");
            v = add_bias_2d(ctx, v, tensor(".sa.v.b"));

            ggml_tensor * qh = ggml_reshape_3d(ctx, q, tf_d_head, tf_n_head, q->ne[1]);
            qh = ggml_permute(ctx, qh, 0, 2, 1, 3);
            ggml_tensor * kh = ggml_reshape_3d(ctx, k, tf_d_head, tf_n_head, k->ne[1]);
            kh = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));
            ggml_tensor * vh = ggml_reshape_3d(ctx, v, tf_d_head, tf_n_head, v->ne[1]);
            vh = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 2, 0, 3));

            ggml_tensor * scores = mul_mat_checked(ctx, qh, kh, prefix + ".sa.scores");
            scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) tf_d_head));
            scores = ggml_add(ctx, scores, tf_att_bias);
            ggml_tensor * probs = ggml_soft_max(ctx, scores);

            ggml_tensor * context = mul_mat_checked(ctx, probs, vh, prefix + ".sa.value_mix");
            context = ggml_permute(ctx, context, 2, 0, 1, 3);
            context = ggml_cont_2d(ctx, context, context->ne[0] * context->ne[1], context->ne[2]);

            ggml_tensor * att_out = mul_linear_project(ctx, tensor(".sa.o.w"), context, use_columnwise_matvec, prefix + ".sa.o");
            att_out = add_bias_2d(ctx, att_out, tensor(".sa.o.b"));
            ggml_tensor * att_res = ggml_add(ctx, residual, att_out);

            ggml_tensor * ln1 = apply_layer_norm(ctx, att_res, tensor(".ln1.w"), tensor(".ln1.b"), 1e-5f);

            ggml_tensor * ff_di = mul_linear_project(ctx, tensor(".ff.di.w"), ln1, use_columnwise_matvec, prefix + ".ff.di");
            ff_di = add_bias_2d(ctx, ff_di, tensor(".ff.di.b"));
            ggml_tensor * ff_act = ggml_relu(ctx, ff_di);
            ggml_tensor * ff_do = mul_linear_project(ctx, tensor(".ff.do.w"), ff_act, use_columnwise_matvec, prefix + ".ff.do");
            ff_do = add_bias_2d(ctx, ff_do, tensor(".ff.do.b"));
            ggml_tensor * ff_res = ggml_add(ctx, ln1, ff_do);

            cur = apply_layer_norm(ctx, ff_res, tensor(".ln2.w"), tensor(".ln2.b"), 1e-5f);
        }

        ggml_tensor * head_hidden1 = ggml_relu(ctx, cur);
        ggml_tensor * head_hidden2 = mul_linear_project(ctx, model.tensor("mods.fh2h.w"), head_hidden1, use_columnwise_matvec, "mods.fh2h");
        head_hidden2 = add_bias_2d(ctx, head_hidden2, model.tensor("mods.fh2h.b"));
        ggml_tensor * head_hidden3 = ggml_relu(ctx, head_hidden2);
        ggml_tensor * head_logits = mul_linear_project(ctx, model.tensor("mods.sh2s.w"), head_hidden3, use_columnwise_matvec, "mods.sh2s");
        head_logits = add_bias_2d(ctx, head_logits, model.tensor("mods.sh2s.b"));
        ggml_tensor * preds = ggml_sigmoid(ctx, head_logits);
        preds = ggml_mul(ctx, preds, keep_t);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer fused encoder/postnet graph tensors");
        }

        ggml_backend_tensor_set(x, preencoded.data.data(), 0, preencoded.data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
        ggml_backend_tensor_set(pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(keep_t, keep_data.data(), 0, keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(tf_att_bias, tf_att_bias_data.data(), 0, tf_att_bias_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1048576, false);
        ggml_build_forward_expand(gf, preds);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer fused encoder/postnet");
        }

        out = tensor_to_matrix_2d(preds);

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

sortformer_postnet_outputs sortformer_run_postnet(
    const sortformer_model & model,
    const sortformer_matrix_f32 & fc_encoder_out,
    const sortformer_matrix_f32 & encoder_mask,
    bool capture_debug) {
    require(fc_encoder_out.rows > 0, "invalid fc_encoder_out shape");
    require(encoder_mask.rows == 1 && encoder_mask.cols == fc_encoder_out.rows, "invalid encoder_mask shape");

    const auto & meta = model.metadata();
    require(meta.tf_d_model > 0 && meta.transformer_heads > 0, "invalid transformer metadata");
    require((meta.tf_d_model % meta.transformer_heads) == 0, "invalid transformer head configuration");
    const int64_t d_model = meta.tf_d_model;
    const int64_t n_head = meta.transformer_heads;
    const int64_t d_head = d_model / n_head;
    const bool use_columnwise_matvec = false;

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer postnet graph context");
    }

    sortformer_postnet_outputs out;

    try {
        std::vector<ggml_tensor *> input_tensors;

        ggml_tensor * fc_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fc_encoder_out.cols, fc_encoder_out.rows);
        ggml_tensor * keep_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, encoder_mask.cols);
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, encoder_mask.cols, encoder_mask.cols);
        ggml_set_input(fc_in);
        ggml_set_input(keep_t);
        ggml_set_input(att_bias);
        input_tensors.push_back(fc_in);
        input_tensors.push_back(keep_t);
        input_tensors.push_back(att_bias);

        std::vector<float> keep_data((size_t) encoder_mask.cols, 0.0f);
        std::vector<float> att_bias_data((size_t) encoder_mask.cols * (size_t) encoder_mask.cols, 0.0f);
        for (uint32_t c = 0; c < encoder_mask.cols; ++c) {
            keep_data[c] = encoder_mask.data[c];
        }
        for (uint32_t r = 0; r < encoder_mask.cols; ++r) {
            for (uint32_t c = 0; c < encoder_mask.cols; ++c) {
                const size_t idx = (size_t) r * (size_t) encoder_mask.cols + (size_t) c;
                att_bias_data[idx] = encoder_mask.data[c] > 0.5f ? 0.0f : -10000.0f;
            }
        }

        ggml_tensor * cur = fc_in;
        if (fc_encoder_out.cols == meta.fc_d_model) {
            cur = mul_linear_project(ctx, model.tensor("mods.ep.w"), cur, use_columnwise_matvec, "mods.ep");
            cur = add_bias_2d(ctx, cur, model.tensor("mods.ep.b"));
        } else if (fc_encoder_out.cols != meta.tf_d_model) {
            throw std::runtime_error("fc_encoder_out columns do not match fc_d_model or tf_d_model");
        }
        ggml_tensor * encoder_proj_out = cur;

        ggml_tensor * te_layer0_q = nullptr;
        ggml_tensor * te_layer0_k = nullptr;
        ggml_tensor * te_layer0_v = nullptr;
        ggml_tensor * te_layer0_scores_head0 = nullptr;
        ggml_tensor * te_layer0_probs_head0 = nullptr;
        ggml_tensor * te_layer0_context = nullptr;
        ggml_tensor * te_layer0_att_out = nullptr;
        ggml_tensor * te_layer0_att_res = nullptr;
        ggml_tensor * te_layer0_ln1 = nullptr;
        ggml_tensor * te_layer0_ff_di = nullptr;
        ggml_tensor * te_layer0_ff_act = nullptr;
        ggml_tensor * te_layer0_ff_do = nullptr;
        ggml_tensor * te_layer0_out = nullptr;

        for (uint32_t il = 0; il < meta.transformer_layers; ++il) {
            const std::string prefix = "te.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * q = mul_linear_project(ctx, tensor(".sa.q.w"), cur, use_columnwise_matvec, prefix + ".sa.q");
            q = add_bias_2d(ctx, q, tensor(".sa.q.b"));
            ggml_tensor * k = mul_linear_project(ctx, tensor(".sa.k.w"), cur, use_columnwise_matvec, prefix + ".sa.k");
            k = add_bias_2d(ctx, k, tensor(".sa.k.b"));
            ggml_tensor * v = mul_linear_project(ctx, tensor(".sa.v.w"), cur, use_columnwise_matvec, prefix + ".sa.v");
            v = add_bias_2d(ctx, v, tensor(".sa.v.b"));

            ggml_tensor * qh = ggml_reshape_3d(ctx, q, d_head, n_head, q->ne[1]);
            qh = ggml_permute(ctx, qh, 0, 2, 1, 3);
            ggml_tensor * kh = ggml_reshape_3d(ctx, k, d_head, n_head, k->ne[1]);
            kh = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));
            ggml_tensor * vh = ggml_reshape_3d(ctx, v, d_head, n_head, v->ne[1]);
            vh = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 2, 0, 3));

            ggml_tensor * scores = mul_mat_checked(ctx, qh, kh, prefix + ".sa.scores");
            scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) d_head));
            scores = ggml_add(ctx, scores, att_bias);
            ggml_tensor * probs = ggml_soft_max(ctx, scores);

            ggml_tensor * context = mul_mat_checked(ctx, probs, vh, prefix + ".sa.value_mix");
            context = ggml_permute(ctx, context, 2, 0, 1, 3);
            context = ggml_cont_2d(ctx, context, context->ne[0] * context->ne[1], context->ne[2]);

            ggml_tensor * att_out = mul_linear_project(ctx, tensor(".sa.o.w"), context, use_columnwise_matvec, prefix + ".sa.o");
            att_out = add_bias_2d(ctx, att_out, tensor(".sa.o.b"));
            ggml_tensor * att_res = ggml_add(ctx, residual, att_out);

            ggml_tensor * ln1 = apply_layer_norm(ctx, att_res, tensor(".ln1.w"), tensor(".ln1.b"), 1e-5f);

            ggml_tensor * ff_di = mul_linear_project(ctx, tensor(".ff.di.w"), ln1, use_columnwise_matvec, prefix + ".ff.di");
            ff_di = add_bias_2d(ctx, ff_di, tensor(".ff.di.b"));
            ggml_tensor * ff_act = ggml_relu(ctx, ff_di);
            ggml_tensor * ff_do = mul_linear_project(ctx, tensor(".ff.do.w"), ff_act, use_columnwise_matvec, prefix + ".ff.do");
            ff_do = add_bias_2d(ctx, ff_do, tensor(".ff.do.b"));
            ggml_tensor * ff_res = ggml_add(ctx, ln1, ff_do);

            cur = apply_layer_norm(ctx, ff_res, tensor(".ln2.w"), tensor(".ln2.b"), 1e-5f);

            if (il == 0) {
                te_layer0_q = q;
                te_layer0_k = k;
                te_layer0_v = v;
                te_layer0_scores_head0 = ggml_view_2d(ctx, scores, scores->ne[0], scores->ne[1], scores->nb[1], 0);
                te_layer0_probs_head0 = ggml_view_2d(ctx, probs, probs->ne[0], probs->ne[1], probs->nb[1], 0);
                te_layer0_context = context;
                te_layer0_att_out = att_out;
                te_layer0_att_res = att_res;
                te_layer0_ln1 = ln1;
                te_layer0_ff_di = ff_di;
                te_layer0_ff_act = ff_act;
                te_layer0_ff_do = ff_do;
                te_layer0_out = cur;
            }
        }

        ggml_tensor * head_hidden1 = ggml_relu(ctx, cur);
        ggml_tensor * head_hidden2 = mul_linear_project(ctx, model.tensor("mods.fh2h.w"), head_hidden1, use_columnwise_matvec, "mods.fh2h");
        head_hidden2 = add_bias_2d(ctx, head_hidden2, model.tensor("mods.fh2h.b"));
        ggml_tensor * head_hidden3 = ggml_relu(ctx, head_hidden2);
        ggml_tensor * head_logits = mul_linear_project(ctx, model.tensor("mods.sh2s.w"), head_hidden3, use_columnwise_matvec, "mods.sh2s");
        head_logits = add_bias_2d(ctx, head_logits, model.tensor("mods.sh2s.b"));
        ggml_tensor * preds = ggml_sigmoid(ctx, head_logits);
        preds = ggml_mul(ctx, preds, keep_t);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer postnet graph tensors");
        }

        ggml_backend_tensor_set(fc_in, fc_encoder_out.data.data(), 0, fc_encoder_out.data.size() * sizeof(float));
        ggml_backend_tensor_set(keep_t, keep_data.data(), 0, keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1048576, false);
        ggml_build_forward_expand(gf, preds);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer postnet");
        }

        out.preds = tensor_to_matrix_2d(preds);
        if (capture_debug) {
            out.encoder_proj_out = tensor_to_matrix_2d(encoder_proj_out);
            out.te_layer0_q = tensor_to_matrix_2d(te_layer0_q);
            out.te_layer0_k = tensor_to_matrix_2d(te_layer0_k);
            out.te_layer0_v = tensor_to_matrix_2d(te_layer0_v);
            out.te_layer0_scores_head0 = tensor_to_matrix_2d(te_layer0_scores_head0);
            out.te_layer0_probs_head0 = tensor_to_matrix_2d(te_layer0_probs_head0);
            out.te_layer0_context = tensor_to_matrix_2d(te_layer0_context);
            out.te_layer0_att_out = tensor_to_matrix_2d(te_layer0_att_out);
            out.te_layer0_att_res = tensor_to_matrix_2d(te_layer0_att_res);
            out.te_layer0_ln1 = tensor_to_matrix_2d(te_layer0_ln1);
            out.te_layer0_ff_di = tensor_to_matrix_2d(te_layer0_ff_di);
            out.te_layer0_ff_act = tensor_to_matrix_2d(te_layer0_ff_act);
            out.te_layer0_ff_do = tensor_to_matrix_2d(te_layer0_ff_do);
            out.te_layer0_out = tensor_to_matrix_2d(te_layer0_out);
            out.transformer_out = tensor_to_matrix_2d(cur);
            out.head_hidden1 = tensor_to_matrix_2d(head_hidden1);
            out.head_hidden2 = tensor_to_matrix_2d(head_hidden2);
            out.head_hidden3 = tensor_to_matrix_2d(head_hidden3);
            out.head_logits = tensor_to_matrix_2d(head_logits);
        }

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

sortformer_matrix_f32 sortformer_run_encoder_postnet_concat(
    const sortformer_model & model,
    const sortformer_matrix_f32 & spkcache,
    const sortformer_matrix_f32 & fifo,
    const sortformer_matrix_f32 & chunk_preencode,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask) {
    require(chunk_preencode.rows > 0, "invalid chunk_preencode shape");
    const uint32_t total_rows = spkcache.rows + fifo.rows + chunk_preencode.rows;
    require(total_rows > 0, "invalid concatenated preencoded shape");
    require(pos_emb.cols == model.metadata().encoder_d_model, "invalid pos_emb shape");
    require(pad_mask.rows == 1 && pad_mask.cols == total_rows, "invalid pad_mask shape");
    require(att_mask.rows == total_rows && att_mask.cols == total_rows, "invalid att_mask shape");
    require((spkcache.rows == 0 || spkcache.cols == model.metadata().encoder_d_model) &&
            (fifo.rows == 0 || fifo.cols == model.metadata().encoder_d_model) &&
            chunk_preencode.cols == model.metadata().encoder_d_model,
            "invalid concat input shape");

    const auto & meta = model.metadata();
    const int64_t enc_d_model = meta.encoder_d_model;
    const int64_t enc_n_head = meta.encoder_heads;
    require(enc_d_model > 0 && enc_n_head > 0 && (enc_d_model % enc_n_head) == 0, "invalid encoder head configuration");
    const int64_t enc_d_head = enc_d_model / enc_n_head;

    require(meta.tf_d_model > 0 && meta.transformer_heads > 0, "invalid transformer metadata");
    require((meta.tf_d_model % meta.transformer_heads) == 0, "invalid transformer head configuration");
    const int64_t tf_d_model = meta.tf_d_model;
    const int64_t tf_n_head = meta.transformer_heads;
    const int64_t tf_d_head = tf_d_model / tf_n_head;
    const bool use_columnwise_matvec = false;

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer fused concat encoder/postnet graph context");
    }

    sortformer_matrix_f32 out;

    try {
        ggml_tensor * spk_t = nullptr;
        ggml_tensor * fifo_t = nullptr;
        if (spkcache.rows > 0) {
            spk_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spkcache.cols, spkcache.rows);
            ggml_set_input(spk_t);
        }
        if (fifo.rows > 0) {
            fifo_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fifo.cols, fifo.rows);
            ggml_set_input(fifo_t);
        }
        ggml_tensor * chunk_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, chunk_preencode.cols, chunk_preencode.rows);
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        ggml_tensor * pad_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * att_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * keep_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * tf_att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pad_mask.cols, pad_mask.cols);
        ggml_set_input(chunk_t);
        ggml_set_input(pos);
        ggml_set_input(pad_keep);
        ggml_set_input(att_bias);
        ggml_set_input(att_keep);
        ggml_set_input(keep_t);
        ggml_set_input(tf_att_bias);

        std::vector<float> pad_keep_data((size_t) pad_mask.cols, 1.0f);
        std::vector<float> keep_data((size_t) pad_mask.cols, 0.0f);
        for (uint32_t i = 0; i < pad_mask.cols; ++i) {
            pad_keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
            keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
        }

        std::vector<float> att_bias_data((size_t) att_mask.rows * (size_t) att_mask.cols, 0.0f);
        std::vector<float> att_keep_data((size_t) att_mask.rows * (size_t) att_mask.cols, 1.0f);
        std::vector<float> tf_att_bias_data((size_t) pad_mask.cols * (size_t) pad_mask.cols, 0.0f);
        for (uint32_t r = 0; r < att_mask.rows; ++r) {
            const bool padded_query = pad_mask.data[r] > 0.5f;
            for (uint32_t c = 0; c < att_mask.cols; ++c) {
                const size_t idx = (size_t) r * (size_t) att_mask.cols + (size_t) c;
                att_bias_data[idx] = padded_query ? 0.0f : (att_mask.data[idx] > 0.5f ? -10000.0f : 0.0f);
                att_keep_data[idx] = att_mask.data[idx] > 0.5f ? 0.0f : 1.0f;
                tf_att_bias_data[idx] = pad_mask.data[c] > 0.5f ? 0.0f : -10000.0f;
            }
        }

        ggml_tensor * x = nullptr;
        if (spk_t != nullptr && fifo_t != nullptr) {
            x = ggml_concat(ctx, spk_t, fifo_t, 1);
            x = ggml_concat(ctx, x, chunk_t, 1);
        } else if (spk_t != nullptr) {
            x = ggml_concat(ctx, spk_t, chunk_t, 1);
        } else if (fifo_t != nullptr) {
            x = ggml_concat(ctx, fifo_t, chunk_t, 1);
        } else {
            x = chunk_t;
        }
        x = ggml_cont(ctx, x);
        ggml_tensor * cur = ggml_scale(ctx, x, std::sqrt((float) enc_d_model));

        for (uint32_t il = 0; il < meta.encoder_layers; ++il) {
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
            Qcur = ggml_reshape_3d(ctx, Qcur, enc_d_head, enc_n_head, Qcur->ne[1]);
            ggml_tensor * pos_bias_u = ggml_reshape_3d(ctx, tensor(".att.pbu"), enc_d_head, enc_n_head, 1);
            ggml_tensor * pos_bias_v = ggml_reshape_3d(ctx, tensor(".att.pbv"), enc_d_head, enc_n_head, 1);
            ggml_tensor * Q_bias_u = ggml_add(ctx, Qcur, pos_bias_u);
            Q_bias_u = ggml_permute(ctx, Q_bias_u, 0, 2, 1, 3);
            ggml_tensor * Q_bias_v = ggml_add(ctx, Qcur, pos_bias_v);
            Q_bias_v = ggml_permute(ctx, Q_bias_v, 0, 2, 1, 3);

            ggml_tensor * Kcur = mul_linear_project(ctx, tensor(".att.k.w"), att_norm, use_columnwise_matvec, prefix + ".att.k");
            Kcur = add_bias_2d(ctx, Kcur, tensor(".att.k.b"));
            Kcur = ggml_reshape_3d(ctx, Kcur, enc_d_head, enc_n_head, Kcur->ne[1]);
            Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = mul_linear_project(ctx, tensor(".att.v.w"), att_norm, use_columnwise_matvec, prefix + ".att.v");
            Vcur = add_bias_2d(ctx, Vcur, tensor(".att.v.b"));
            Vcur = ggml_reshape_3d(ctx, Vcur, enc_d_head, enc_n_head, Vcur->ne[1]);
            Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

            ggml_tensor * matrix_ac = mul_mat_checked(ctx, Q_bias_u, Kcur, prefix + ".att.matrix_ac");
            matrix_ac = ggml_cont(ctx, ggml_permute(ctx, matrix_ac, 1, 0, 2, 3));

            ggml_tensor * p = mul_linear_project(ctx, tensor(".att.p.w"), pos, use_columnwise_matvec, prefix + ".att.linear_pos");
            p = ggml_reshape_3d(ctx, p, enc_d_head, enc_n_head, p->ne[1]);
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
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) enc_d_head));
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

        if (chunk_preencode.cols == meta.fc_d_model) {
            cur = mul_linear_project(ctx, model.tensor("mods.ep.w"), cur, use_columnwise_matvec, "mods.ep");
            cur = add_bias_2d(ctx, cur, model.tensor("mods.ep.b"));
        } else if (chunk_preencode.cols != meta.tf_d_model) {
            throw std::runtime_error("encoder output columns do not match fc_d_model or tf_d_model");
        }

        for (uint32_t il = 0; il < meta.transformer_layers; ++il) {
            const std::string prefix = "te.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * q = mul_linear_project(ctx, tensor(".sa.q.w"), cur, use_columnwise_matvec, prefix + ".sa.q");
            q = add_bias_2d(ctx, q, tensor(".sa.q.b"));
            ggml_tensor * k = mul_linear_project(ctx, tensor(".sa.k.w"), cur, use_columnwise_matvec, prefix + ".sa.k");
            k = add_bias_2d(ctx, k, tensor(".sa.k.b"));
            ggml_tensor * v = mul_linear_project(ctx, tensor(".sa.v.w"), cur, use_columnwise_matvec, prefix + ".sa.v");
            v = add_bias_2d(ctx, v, tensor(".sa.v.b"));

            ggml_tensor * qh = ggml_reshape_3d(ctx, q, tf_d_head, tf_n_head, q->ne[1]);
            qh = ggml_permute(ctx, qh, 0, 2, 1, 3);
            ggml_tensor * kh = ggml_reshape_3d(ctx, k, tf_d_head, tf_n_head, k->ne[1]);
            kh = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));
            ggml_tensor * vh = ggml_reshape_3d(ctx, v, tf_d_head, tf_n_head, v->ne[1]);
            vh = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 2, 0, 3));

            ggml_tensor * scores = mul_mat_checked(ctx, qh, kh, prefix + ".sa.scores");
            scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) tf_d_head));
            scores = ggml_add(ctx, scores, tf_att_bias);
            ggml_tensor * probs = ggml_soft_max(ctx, scores);

            ggml_tensor * context = mul_mat_checked(ctx, probs, vh, prefix + ".sa.value_mix");
            context = ggml_permute(ctx, context, 2, 0, 1, 3);
            context = ggml_cont_2d(ctx, context, context->ne[0] * context->ne[1], context->ne[2]);

            ggml_tensor * att_out = mul_linear_project(ctx, tensor(".sa.o.w"), context, use_columnwise_matvec, prefix + ".sa.o");
            att_out = add_bias_2d(ctx, att_out, tensor(".sa.o.b"));
            ggml_tensor * att_res = ggml_add(ctx, residual, att_out);

            ggml_tensor * ln1 = apply_layer_norm(ctx, att_res, tensor(".ln1.w"), tensor(".ln1.b"), 1e-5f);

            ggml_tensor * ff_di = mul_linear_project(ctx, tensor(".ff.di.w"), ln1, use_columnwise_matvec, prefix + ".ff.di");
            ff_di = add_bias_2d(ctx, ff_di, tensor(".ff.di.b"));
            ggml_tensor * ff_act = ggml_relu(ctx, ff_di);
            ggml_tensor * ff_do = mul_linear_project(ctx, tensor(".ff.do.w"), ff_act, use_columnwise_matvec, prefix + ".ff.do");
            ff_do = add_bias_2d(ctx, ff_do, tensor(".ff.do.b"));
            ggml_tensor * ff_res = ggml_add(ctx, ln1, ff_do);

            cur = apply_layer_norm(ctx, ff_res, tensor(".ln2.w"), tensor(".ln2.b"), 1e-5f);
        }

        ggml_tensor * head_hidden1 = ggml_relu(ctx, cur);
        ggml_tensor * head_hidden2 = mul_linear_project(ctx, model.tensor("mods.fh2h.w"), head_hidden1, use_columnwise_matvec, "mods.fh2h");
        head_hidden2 = add_bias_2d(ctx, head_hidden2, model.tensor("mods.fh2h.b"));
        ggml_tensor * head_hidden3 = ggml_relu(ctx, head_hidden2);
        ggml_tensor * head_logits = mul_linear_project(ctx, model.tensor("mods.sh2s.w"), head_hidden3, use_columnwise_matvec, "mods.sh2s");
        head_logits = add_bias_2d(ctx, head_logits, model.tensor("mods.sh2s.b"));
        ggml_tensor * preds = ggml_sigmoid(ctx, head_logits);
        preds = ggml_mul(ctx, preds, keep_t);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer fused concat encoder/postnet graph tensors");
        }

        if (spk_t != nullptr) {
            ggml_backend_tensor_set(spk_t, spkcache.data.data(), 0, spkcache.data.size() * sizeof(float));
        }
        if (fifo_t != nullptr) {
            ggml_backend_tensor_set(fifo_t, fifo.data.data(), 0, fifo.data.size() * sizeof(float));
        }
        ggml_backend_tensor_set(chunk_t, chunk_preencode.data.data(), 0, chunk_preencode.data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
        ggml_backend_tensor_set(pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(keep_t, keep_data.data(), 0, keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(tf_att_bias, tf_att_bias_data.data(), 0, tf_att_bias_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1048576, false);
        ggml_build_forward_expand(gf, preds);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer fused concat encoder/postnet");
        }

        out = tensor_to_matrix_2d(preds);

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

sortformer_matrix_f32 sortformer_run_encoder_postnet_concat_cached(
    sortformer_encoder_postnet_plan & plan,
    const sortformer_model & model,
    const sortformer_matrix_f32 & spkcache,
    const sortformer_matrix_f32 & fifo,
    const sortformer_matrix_f32 & chunk_preencode,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask) {
    require(chunk_preencode.rows > 0, "invalid chunk_preencode shape");
    const uint32_t total_rows = spkcache.rows + fifo.rows + chunk_preencode.rows;
    require(total_rows > 0, "invalid concatenated preencoded shape");
    require(pos_emb.cols == model.metadata().encoder_d_model, "invalid pos_emb shape");
    require(pad_mask.rows == 1 && pad_mask.cols == total_rows, "invalid pad_mask shape");
    require(att_mask.rows == total_rows && att_mask.cols == total_rows, "invalid att_mask shape");
    require((spkcache.rows == 0 || spkcache.cols == model.metadata().encoder_d_model) &&
            (fifo.rows == 0 || fifo.cols == model.metadata().encoder_d_model) &&
            chunk_preencode.cols == model.metadata().encoder_d_model,
            "invalid concat input shape");

    const auto & meta = model.metadata();
    const int64_t enc_d_model = meta.encoder_d_model;
    const int64_t enc_n_head = meta.encoder_heads;
    require(enc_d_model > 0 && enc_n_head > 0 && (enc_d_model % enc_n_head) == 0, "invalid encoder head configuration");
    const int64_t enc_d_head = enc_d_model / enc_n_head;

    require(meta.tf_d_model > 0 && meta.transformer_heads > 0, "invalid transformer metadata");
    require((meta.tf_d_model % meta.transformer_heads) == 0, "invalid transformer head configuration");
    const int64_t tf_d_model = meta.tf_d_model;
    const int64_t tf_n_head = meta.transformer_heads;
    const int64_t tf_d_head = tf_d_model / tf_n_head;
    const bool use_columnwise_matvec = false;

    auto rebuild_plan = [&]() {
        delete plan.impl_;
        plan.impl_ = new sortformer_encoder_postnet_plan::impl();
        auto & pi = *plan.impl_;
        pi.model = &model;
        pi.spk_rows = spkcache.rows;
        pi.fifo_rows = fifo.rows;
        pi.chunk_rows = chunk_preencode.rows;
        pi.ctx = ggml_init(make_graph_ctx_params());
        if (pi.ctx == nullptr) {
            throw std::runtime_error("failed to allocate cached Sortformer encoder/postnet graph context");
        }

        if (spkcache.rows > 0) {
            pi.spk_t = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, spkcache.cols, spkcache.rows);
            ggml_set_input(pi.spk_t);
        }
        if (fifo.rows > 0) {
            pi.fifo_t = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, fifo.cols, fifo.rows);
            ggml_set_input(pi.fifo_t);
        }
        pi.chunk_t = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, chunk_preencode.cols, chunk_preencode.rows);
        pi.pos = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        pi.pad_keep = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        pi.att_bias = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        pi.att_keep = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        pi.keep_t = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        pi.tf_att_bias = ggml_new_tensor_2d(pi.ctx, GGML_TYPE_F32, pad_mask.cols, pad_mask.cols);
        ggml_set_input(pi.chunk_t);
        ggml_set_input(pi.pos);
        ggml_set_input(pi.pad_keep);
        ggml_set_input(pi.att_bias);
        ggml_set_input(pi.att_keep);
        ggml_set_input(pi.keep_t);
        ggml_set_input(pi.tf_att_bias);

        ggml_tensor * x = nullptr;
        if (pi.spk_t != nullptr && pi.fifo_t != nullptr) {
            x = ggml_concat(pi.ctx, pi.spk_t, pi.fifo_t, 1);
            x = ggml_concat(pi.ctx, x, pi.chunk_t, 1);
        } else if (pi.spk_t != nullptr) {
            x = ggml_concat(pi.ctx, pi.spk_t, pi.chunk_t, 1);
        } else if (pi.fifo_t != nullptr) {
            x = ggml_concat(pi.ctx, pi.fifo_t, pi.chunk_t, 1);
        } else {
            x = pi.chunk_t;
        }
        x = ggml_cont(pi.ctx, x);
        ggml_tensor * cur = ggml_scale(pi.ctx, x, std::sqrt((float) enc_d_model));

        for (uint32_t il = 0; il < meta.encoder_layers; ++il) {
            const std::string prefix = "enc.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * ff1_norm = apply_layer_norm(pi.ctx, residual, tensor(".nff1.w"), tensor(".nff1.b"), 1e-5f);
            ggml_tensor * ff1_out = apply_feed_forward(
                pi.ctx,
                ff1_norm,
                tensor(".ff1.l1.w"),
                tensor(".ff1.l1.b"),
                tensor(".ff1.l2.w"),
                tensor(".ff1.l2.b"),
                use_columnwise_matvec,
                prefix + ".ff1");
            ggml_tensor * ff1_res = ggml_add(pi.ctx, residual, ggml_scale(pi.ctx, ff1_out, 0.5f));

            ggml_tensor * att_norm = apply_layer_norm(pi.ctx, ff1_res, tensor(".nsa.w"), tensor(".nsa.b"), 1e-5f);

            ggml_tensor * Qcur = mul_linear_project(pi.ctx, tensor(".att.q.w"), att_norm, use_columnwise_matvec, prefix + ".att.q");
            Qcur = add_bias_2d(pi.ctx, Qcur, tensor(".att.q.b"));
            Qcur = ggml_reshape_3d(pi.ctx, Qcur, enc_d_head, enc_n_head, Qcur->ne[1]);
            ggml_tensor * pos_bias_u = ggml_reshape_3d(pi.ctx, tensor(".att.pbu"), enc_d_head, enc_n_head, 1);
            ggml_tensor * pos_bias_v = ggml_reshape_3d(pi.ctx, tensor(".att.pbv"), enc_d_head, enc_n_head, 1);
            ggml_tensor * Q_bias_u = ggml_add(pi.ctx, Qcur, pos_bias_u);
            Q_bias_u = ggml_permute(pi.ctx, Q_bias_u, 0, 2, 1, 3);
            ggml_tensor * Q_bias_v = ggml_add(pi.ctx, Qcur, pos_bias_v);
            Q_bias_v = ggml_permute(pi.ctx, Q_bias_v, 0, 2, 1, 3);

            ggml_tensor * Kcur = mul_linear_project(pi.ctx, tensor(".att.k.w"), att_norm, use_columnwise_matvec, prefix + ".att.k");
            Kcur = add_bias_2d(pi.ctx, Kcur, tensor(".att.k.b"));
            Kcur = ggml_reshape_3d(pi.ctx, Kcur, enc_d_head, enc_n_head, Kcur->ne[1]);
            Kcur = ggml_cont(pi.ctx, ggml_permute(pi.ctx, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = mul_linear_project(pi.ctx, tensor(".att.v.w"), att_norm, use_columnwise_matvec, prefix + ".att.v");
            Vcur = add_bias_2d(pi.ctx, Vcur, tensor(".att.v.b"));
            Vcur = ggml_reshape_3d(pi.ctx, Vcur, enc_d_head, enc_n_head, Vcur->ne[1]);
            Vcur = ggml_cont(pi.ctx, ggml_permute(pi.ctx, Vcur, 1, 2, 0, 3));

            ggml_tensor * matrix_ac = mul_mat_checked(pi.ctx, Q_bias_u, Kcur, prefix + ".att.matrix_ac");
            matrix_ac = ggml_cont(pi.ctx, ggml_permute(pi.ctx, matrix_ac, 1, 0, 2, 3));

            ggml_tensor * p = mul_linear_project(pi.ctx, tensor(".att.p.w"), pi.pos, use_columnwise_matvec, prefix + ".att.linear_pos");
            p = ggml_reshape_3d(pi.ctx, p, enc_d_head, enc_n_head, p->ne[1]);
            p = ggml_permute(pi.ctx, p, 0, 2, 1, 3);

            ggml_tensor * matrix_bd = mul_mat_checked(pi.ctx, Q_bias_v, p, prefix + ".att.matrix_bd");
            matrix_bd = ggml_cont(pi.ctx, ggml_permute(pi.ctx, matrix_bd, 1, 0, 2, 3));
            {
                const auto pos_len = matrix_bd->ne[0];
                const auto q_len = matrix_bd->ne[1];
                const auto h = matrix_bd->ne[2];
                matrix_bd = ggml_pad(pi.ctx, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_roll(pi.ctx, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_reshape_3d(pi.ctx, matrix_bd, q_len, pos_len + 1, h);
                matrix_bd = ggml_view_3d(pi.ctx, matrix_bd, q_len, pos_len, h, matrix_bd->nb[1], matrix_bd->nb[2], matrix_bd->nb[0] * q_len);
                matrix_bd = ggml_cont_3d(pi.ctx, matrix_bd, pos_len, q_len, h);
            }
            matrix_bd = ggml_view_3d(pi.ctx, matrix_bd, matrix_ac->ne[0], matrix_bd->ne[1], matrix_bd->ne[2], matrix_bd->nb[1], matrix_bd->nb[2], 0);

            ggml_tensor * scores = ggml_add(pi.ctx, matrix_ac, matrix_bd);
            scores = ggml_scale(pi.ctx, scores, 1.0f / std::sqrt((float) enc_d_head));
            scores = ggml_add(pi.ctx, scores, pi.att_bias);
            ggml_tensor * attn = ggml_soft_max(pi.ctx, scores);
            attn = ggml_mul(pi.ctx, attn, pi.att_keep);

            ggml_tensor * att_x = mul_mat_checked(pi.ctx, attn, Vcur, prefix + ".att.value_mix");
            att_x = ggml_permute(pi.ctx, att_x, 2, 0, 1, 3);
            att_x = ggml_cont_2d(pi.ctx, att_x, att_x->ne[0] * att_x->ne[1], att_x->ne[2]);

            ggml_tensor * att_out = mul_linear_project(pi.ctx, tensor(".att.o.w"), att_x, use_columnwise_matvec, prefix + ".att.o");
            att_out = add_bias_2d(pi.ctx, att_out, tensor(".att.o.b"));
            ggml_tensor * att_res = ggml_add(pi.ctx, ff1_res, att_out);

            ggml_tensor * conv_norm = apply_layer_norm(pi.ctx, att_res, tensor(".nc.w"), tensor(".nc.b"), 1e-5f);
            ggml_tensor * conv_pw1_w = ggml_reshape_2d(pi.ctx, tensor(".conv.pw1.w"), tensor(".conv.pw1.w")->ne[1], tensor(".conv.pw1.w")->ne[2]);
            ggml_tensor * conv_x = mul_linear_project(pi.ctx, conv_pw1_w, conv_norm, use_columnwise_matvec, prefix + ".conv.pw1");
            conv_x = add_bias_2d(pi.ctx, conv_x, tensor(".conv.pw1.b"));
            {
                const int64_t d = conv_x->ne[0] / 2;
                ggml_tensor * left = ggml_cont(pi.ctx, ggml_view_2d(pi.ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], 0));
                ggml_tensor * right = ggml_cont(pi.ctx, ggml_view_2d(pi.ctx, conv_x, d, conv_x->ne[1], conv_x->nb[1], d * conv_x->nb[0]));
                ggml_tensor * gate = ggml_sigmoid(pi.ctx, right);
                conv_x = ggml_mul(pi.ctx, left, gate);
            }
            conv_x = ggml_mul(pi.ctx, conv_x, pi.pad_keep);
            conv_x = ggml_cont(pi.ctx, ggml_transpose(pi.ctx, conv_x));
            conv_x = ggml_reshape_3d(pi.ctx, conv_x, conv_x->ne[0], conv_x->ne[1], 1);
            conv_x = ggml_pad(pi.ctx, conv_x, 4, 0, 0, 0);
            conv_x = ggml_roll(pi.ctx, conv_x, 4, 0, 0, 0);
            conv_x = ggml_pad(pi.ctx, conv_x, 4, 0, 0, 0);

            ggml_tensor * conv_dw_w = ggml_reshape_2d(pi.ctx, tensor(".conv.dw.w"), tensor(".conv.dw.w")->ne[0], tensor(".conv.dw.w")->ne[2]);
            ggml_tensor * conv_dw = ggml_ssm_conv(pi.ctx, conv_x, conv_dw_w);
            conv_dw = ggml_add(pi.ctx, conv_dw, tensor(".conv.dw.b"));
            conv_dw = ggml_reshape_2d(pi.ctx, conv_dw, conv_dw->ne[0], conv_dw->ne[1]);

            conv_dw = ggml_cont(pi.ctx, ggml_transpose(pi.ctx, conv_dw));
            conv_dw = ggml_mul(pi.ctx, conv_dw, ggml_reshape_2d(pi.ctx, tensor(".conv.bn.sc"), 1, tensor(".conv.bn.sc")->ne[0]));
            conv_dw = ggml_add(pi.ctx, conv_dw, ggml_reshape_2d(pi.ctx, tensor(".conv.bn.sh"), 1, tensor(".conv.bn.sh")->ne[0]));
            conv_dw = ggml_silu(pi.ctx, conv_dw);
            conv_dw = ggml_cont(pi.ctx, ggml_transpose(pi.ctx, conv_dw));

            ggml_tensor * conv_pw2_w = ggml_reshape_2d(pi.ctx, tensor(".conv.pw2.w"), tensor(".conv.pw2.w")->ne[1], tensor(".conv.pw2.w")->ne[2]);
            ggml_tensor * conv_out = mul_linear_project(pi.ctx, conv_pw2_w, conv_dw, use_columnwise_matvec, prefix + ".conv.pw2");
            conv_out = add_bias_2d(pi.ctx, conv_out, tensor(".conv.pw2.b"));
            ggml_tensor * conv_res = ggml_add(pi.ctx, att_res, conv_out);

            ggml_tensor * ff2_norm = apply_layer_norm(pi.ctx, conv_res, tensor(".nff2.w"), tensor(".nff2.b"), 1e-5f);
            ggml_tensor * ff2_out = apply_feed_forward(
                pi.ctx,
                ff2_norm,
                tensor(".ff2.l1.w"),
                tensor(".ff2.l1.b"),
                tensor(".ff2.l2.w"),
                tensor(".ff2.l2.b"),
                use_columnwise_matvec,
                prefix + ".ff2");
            ggml_tensor * ff2_res = ggml_add(pi.ctx, conv_res, ggml_scale(pi.ctx, ff2_out, 0.5f));

            cur = apply_layer_norm(pi.ctx, ff2_res, tensor(".no.w"), tensor(".no.b"), 1e-5f);
        }

        if (chunk_preencode.cols == meta.fc_d_model) {
            cur = mul_linear_project(pi.ctx, model.tensor("mods.ep.w"), cur, use_columnwise_matvec, "mods.ep");
            cur = add_bias_2d(pi.ctx, cur, model.tensor("mods.ep.b"));
        } else if (chunk_preencode.cols != meta.tf_d_model) {
            throw std::runtime_error("encoder output columns do not match fc_d_model or tf_d_model");
        }

        for (uint32_t il = 0; il < meta.transformer_layers; ++il) {
            const std::string prefix = "te.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * q = mul_linear_project(pi.ctx, tensor(".sa.q.w"), cur, use_columnwise_matvec, prefix + ".sa.q");
            q = add_bias_2d(pi.ctx, q, tensor(".sa.q.b"));
            ggml_tensor * k = mul_linear_project(pi.ctx, tensor(".sa.k.w"), cur, use_columnwise_matvec, prefix + ".sa.k");
            k = add_bias_2d(pi.ctx, k, tensor(".sa.k.b"));
            ggml_tensor * v = mul_linear_project(pi.ctx, tensor(".sa.v.w"), cur, use_columnwise_matvec, prefix + ".sa.v");
            v = add_bias_2d(pi.ctx, v, tensor(".sa.v.b"));

            ggml_tensor * qh = ggml_reshape_3d(pi.ctx, q, tf_d_head, tf_n_head, q->ne[1]);
            qh = ggml_permute(pi.ctx, qh, 0, 2, 1, 3);
            ggml_tensor * kh = ggml_reshape_3d(pi.ctx, k, tf_d_head, tf_n_head, k->ne[1]);
            kh = ggml_cont(pi.ctx, ggml_permute(pi.ctx, kh, 0, 2, 1, 3));
            ggml_tensor * vh = ggml_reshape_3d(pi.ctx, v, tf_d_head, tf_n_head, v->ne[1]);
            vh = ggml_cont(pi.ctx, ggml_permute(pi.ctx, vh, 1, 2, 0, 3));

            ggml_tensor * scores = mul_mat_checked(pi.ctx, qh, kh, prefix + ".sa.scores");
            scores = ggml_cont(pi.ctx, ggml_permute(pi.ctx, scores, 1, 0, 2, 3));
            scores = ggml_scale(pi.ctx, scores, 1.0f / std::sqrt((float) tf_d_head));
            scores = ggml_add(pi.ctx, scores, pi.tf_att_bias);
            ggml_tensor * probs = ggml_soft_max(pi.ctx, scores);

            ggml_tensor * context = mul_mat_checked(pi.ctx, probs, vh, prefix + ".sa.value_mix");
            context = ggml_permute(pi.ctx, context, 2, 0, 1, 3);
            context = ggml_cont_2d(pi.ctx, context, context->ne[0] * context->ne[1], context->ne[2]);

            ggml_tensor * att_out = mul_linear_project(pi.ctx, tensor(".sa.o.w"), context, use_columnwise_matvec, prefix + ".sa.o");
            att_out = add_bias_2d(pi.ctx, att_out, tensor(".sa.o.b"));
            ggml_tensor * att_res = ggml_add(pi.ctx, residual, att_out);

            ggml_tensor * ln1 = apply_layer_norm(pi.ctx, att_res, tensor(".ln1.w"), tensor(".ln1.b"), 1e-5f);

            ggml_tensor * ff_di = mul_linear_project(pi.ctx, tensor(".ff.di.w"), ln1, use_columnwise_matvec, prefix + ".ff.di");
            ff_di = add_bias_2d(pi.ctx, ff_di, tensor(".ff.di.b"));
            ggml_tensor * ff_act = ggml_relu(pi.ctx, ff_di);
            ggml_tensor * ff_do = mul_linear_project(pi.ctx, tensor(".ff.do.w"), ff_act, use_columnwise_matvec, prefix + ".ff.do");
            ff_do = add_bias_2d(pi.ctx, ff_do, tensor(".ff.do.b"));
            ggml_tensor * ff_res = ggml_add(pi.ctx, ln1, ff_do);

            cur = apply_layer_norm(pi.ctx, ff_res, tensor(".ln2.w"), tensor(".ln2.b"), 1e-5f);
        }

        ggml_tensor * head_hidden1 = ggml_relu(pi.ctx, cur);
        ggml_tensor * head_hidden2 = mul_linear_project(pi.ctx, model.tensor("mods.fh2h.w"), head_hidden1, use_columnwise_matvec, "mods.fh2h");
        head_hidden2 = add_bias_2d(pi.ctx, head_hidden2, model.tensor("mods.fh2h.b"));
        ggml_tensor * head_hidden3 = ggml_relu(pi.ctx, head_hidden2);
        ggml_tensor * head_logits = mul_linear_project(pi.ctx, model.tensor("mods.sh2s.w"), head_hidden3, use_columnwise_matvec, "mods.sh2s");
        head_logits = add_bias_2d(pi.ctx, head_logits, model.tensor("mods.sh2s.b"));
        pi.preds = ggml_sigmoid(pi.ctx, head_logits);
        pi.preds = ggml_mul(pi.ctx, pi.preds, pi.keep_t);

        pi.graph_buf = ggml_backend_alloc_ctx_tensors(pi.ctx, model.backend());
        if (pi.graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate cached Sortformer encoder/postnet graph tensors");
        }

        pi.gf = ggml_new_graph_custom(pi.ctx, 1048576, false);
        ggml_build_forward_expand(pi.gf, pi.preds);
    };

    const bool needs_rebuild =
        plan.impl_ == nullptr ||
        plan.impl_->model != &model ||
        plan.impl_->spk_rows != spkcache.rows ||
        plan.impl_->fifo_rows != fifo.rows ||
        plan.impl_->chunk_rows != chunk_preencode.rows;
    if (needs_rebuild) {
        rebuild_plan();
    }

    auto & pi = *plan.impl_;

    std::vector<float> pad_keep_data((size_t) pad_mask.cols, 1.0f);
    std::vector<float> keep_data((size_t) pad_mask.cols, 0.0f);
    for (uint32_t i = 0; i < pad_mask.cols; ++i) {
        pad_keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
        keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
    }

    std::vector<float> att_bias_data((size_t) att_mask.rows * (size_t) att_mask.cols, 0.0f);
    std::vector<float> att_keep_data((size_t) att_mask.rows * (size_t) att_mask.cols, 1.0f);
    std::vector<float> tf_att_bias_data((size_t) pad_mask.cols * (size_t) pad_mask.cols, 0.0f);
    for (uint32_t r = 0; r < att_mask.rows; ++r) {
        const bool padded_query = pad_mask.data[r] > 0.5f;
        for (uint32_t c = 0; c < att_mask.cols; ++c) {
            const size_t idx = (size_t) r * (size_t) att_mask.cols + (size_t) c;
            att_bias_data[idx] = padded_query ? 0.0f : (att_mask.data[idx] > 0.5f ? -10000.0f : 0.0f);
            att_keep_data[idx] = att_mask.data[idx] > 0.5f ? 0.0f : 1.0f;
            tf_att_bias_data[idx] = pad_mask.data[c] > 0.5f ? 0.0f : -10000.0f;
        }
    }

    if (pi.spk_t != nullptr) {
        ggml_backend_tensor_set(pi.spk_t, spkcache.data.data(), 0, spkcache.data.size() * sizeof(float));
    }
    if (pi.fifo_t != nullptr) {
        ggml_backend_tensor_set(pi.fifo_t, fifo.data.data(), 0, fifo.data.size() * sizeof(float));
    }
    ggml_backend_tensor_set(pi.chunk_t, chunk_preencode.data.data(), 0, chunk_preencode.data.size() * sizeof(float));
    ggml_backend_tensor_set(pi.pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
    ggml_backend_tensor_set(pi.pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
    ggml_backend_tensor_set(pi.att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
    ggml_backend_tensor_set(pi.att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));
    ggml_backend_tensor_set(pi.keep_t, keep_data.data(), 0, keep_data.size() * sizeof(float));
    ggml_backend_tensor_set(pi.tf_att_bias, tf_att_bias_data.data(), 0, tf_att_bias_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(model.backend(), pi.gf);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed for cached Sortformer encoder/postnet");
    }

    return tensor_to_matrix_2d(pi.preds);
}

sortformer_full_step_outputs sortformer_run_full_step_concat(
    const sortformer_model & model,
    const sortformer_matrix_f32 & chunk_features,
    uint32_t chunk_valid_feature_rows,
    const sortformer_matrix_f32 & spkcache,
    const sortformer_matrix_f32 & fifo,
    const sortformer_matrix_f32 & pos_emb,
    const sortformer_matrix_f32 & pad_mask,
    const sortformer_matrix_f32 & att_mask) {
    require(chunk_features.rows > 0 && chunk_features.cols == model.metadata().mel_bins, "invalid chunk_features shape");
    require(chunk_valid_feature_rows <= chunk_features.rows, "invalid chunk_valid_feature_rows");

    const auto & meta = model.metadata();
    const uint32_t pre_rows =
        calculate_conv_output_size(
            calculate_conv_output_size(
                calculate_conv_output_size(chunk_valid_feature_rows, 3, 2, 1, 1),
                3,
                2,
                1,
                1),
            3,
            2,
            1,
            1);

    const uint32_t total_rows = spkcache.rows + fifo.rows + pre_rows;
    require(total_rows > 0, "invalid concatenated preencoded shape");
    require(pos_emb.cols == meta.encoder_d_model, "invalid pos_emb shape");
    require(pad_mask.rows == 1 && pad_mask.cols == total_rows, "invalid pad_mask shape");
    require(att_mask.rows == total_rows && att_mask.cols == total_rows, "invalid att_mask shape");
    require((spkcache.rows == 0 || spkcache.cols == meta.encoder_d_model) &&
            (fifo.rows == 0 || fifo.cols == meta.encoder_d_model),
            "invalid cache input shape");

    const int64_t enc_d_model = meta.encoder_d_model;
    const int64_t enc_n_head = meta.encoder_heads;
    require(enc_d_model > 0 && enc_n_head > 0 && (enc_d_model % enc_n_head) == 0, "invalid encoder head configuration");
    const int64_t enc_d_head = enc_d_model / enc_n_head;

    require(meta.tf_d_model > 0 && meta.transformer_heads > 0, "invalid transformer metadata");
    require((meta.tf_d_model % meta.transformer_heads) == 0, "invalid transformer head configuration");
    const int64_t tf_d_model = meta.tf_d_model;
    const int64_t tf_n_head = meta.transformer_heads;
    const int64_t tf_d_head = tf_d_model / tf_n_head;

    const std::string backend_name = ggml_backend_name(model.backend());
    const bool use_blocked_matmul = backend_name.rfind("Vulkan", 0) == 0;
    const bool use_columnwise_matvec = false;

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer full-step graph context");
    }

    sortformer_full_step_outputs out;

    try {
        struct pending_mask {
            ggml_tensor * tensor;
            std::vector<float> data;
        };
        std::vector<pending_mask> pending_masks;

        ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, chunk_features.cols, chunk_features.rows, 1);
        ggml_set_name(inp, "sortformer.features");
        ggml_set_input(inp);

        auto apply_time_mask = [&](ggml_tensor * tensor, uint32_t valid_time, const char * name) -> ggml_tensor * {
            ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
            ggml_set_name(mask, name);

            std::vector<float> mask_data((size_t) tensor->ne[0] * (size_t) tensor->ne[1] * (size_t) tensor->ne[2] * (size_t) tensor->ne[3], 0.0f);
            for (int64_t b = 0; b < tensor->ne[3]; ++b) {
                for (int64_t c = 0; c < tensor->ne[2]; ++c) {
                    for (int64_t t = 0; t < tensor->ne[1]; ++t) {
                        if ((uint32_t) t >= valid_time) {
                            continue;
                        }
                        for (int64_t f = 0; f < tensor->ne[0]; ++f) {
                            const size_t idx =
                                (((size_t) b * (size_t) tensor->ne[2] + (size_t) c) * (size_t) tensor->ne[1] + (size_t) t) * (size_t) tensor->ne[0] +
                                (size_t) f;
                            mask_data[idx] = 1.0f;
                        }
                    }
                }
            }

            pending_masks.push_back({ mask, std::move(mask_data) });
            return ggml_mul(ctx, tensor, mask);
        };

        ggml_tensor * pre_x = inp;
        ggml_tensor * conv0 = conv2d_relu(
            ctx,
            pre_x,
            model.tensor("enc.pre.conv.0.w"),
            model.tensor("enc.pre.conv.0.b"),
            use_blocked_matmul,
            2,
            2,
            1,
            1,
            false,
            true,
            true);
        pre_x = apply_time_mask(conv0, calculate_conv_output_size(chunk_valid_feature_rows, 3, 2, 1, 1), "sortformer.mask.conv0");

        ggml_tensor * conv2 = conv2d_relu(
            ctx,
            pre_x,
            model.tensor("enc.pre.conv.2.w"),
            model.tensor("enc.pre.conv.2.b"),
            use_blocked_matmul,
            2,
            2,
            1,
            1,
            true,
            false,
            false);
        pre_x = apply_time_mask(
            conv2,
            calculate_conv_output_size(calculate_conv_output_size(chunk_valid_feature_rows, 3, 2, 1, 1), 3, 2, 1, 1),
            "sortformer.mask.conv2");

        ggml_tensor * conv3 = conv2d_relu(
            ctx,
            pre_x,
            model.tensor("enc.pre.conv.3.w"),
            model.tensor("enc.pre.conv.3.b"),
            use_blocked_matmul,
            1,
            1,
            0,
            0,
            false,
            true,
            true);
        pre_x = apply_time_mask(
            conv3,
            calculate_conv_output_size(calculate_conv_output_size(chunk_valid_feature_rows, 3, 2, 1, 1), 3, 2, 1, 1),
            "sortformer.mask.conv3");

        ggml_tensor * conv5 = conv2d_relu(
            ctx,
            pre_x,
            model.tensor("enc.pre.conv.5.w"),
            model.tensor("enc.pre.conv.5.b"),
            use_blocked_matmul,
            2,
            2,
            1,
            1,
            true,
            false,
            false);
        pre_x = apply_time_mask(
            conv5,
            calculate_conv_output_size(
                calculate_conv_output_size(calculate_conv_output_size(chunk_valid_feature_rows, 3, 2, 1, 1), 3, 2, 1, 1),
                3,
                2,
                1,
                1),
            "sortformer.mask.conv5");

        ggml_tensor * conv6 = conv2d_relu(
            ctx,
            pre_x,
            model.tensor("enc.pre.conv.6.w"),
            model.tensor("enc.pre.conv.6.b"),
            use_blocked_matmul,
            1,
            1,
            0,
            0,
            false,
            true,
            true);
        pre_x = apply_time_mask(
            conv6,
            calculate_conv_output_size(
                calculate_conv_output_size(calculate_conv_output_size(chunk_valid_feature_rows, 3, 2, 1, 1), 3, 2, 1, 1),
                3,
                2,
                1,
                1),
            "sortformer.mask.conv6");

        require(pre_x->ne[2] == 256, "unexpected preencode channel count");

        ggml_tensor * flat = ggml_permute(ctx, pre_x, 0, 2, 1, 3);
        flat = ggml_cont(ctx, flat);
        flat = ggml_reshape_2d(ctx, flat, flat->ne[0] * flat->ne[1], flat->ne[2]);

        ggml_tensor * proj_w = model.tensor("enc.pre.out.w");
        require(proj_w->ne[0] == flat->ne[0], "preencode projection shape mismatch");
        ggml_tensor * chunk_proj = ggml_mul_mat(ctx, proj_w, flat);
        chunk_proj = add_bias_2d(ctx, chunk_proj, model.tensor("enc.pre.out.b"));
        ggml_set_name(chunk_proj, "sortformer.preencode.out");

        ggml_tensor * spk_t = nullptr;
        ggml_tensor * fifo_t = nullptr;
        if (spkcache.rows > 0) {
            spk_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spkcache.cols, spkcache.rows);
            ggml_set_input(spk_t);
        }
        if (fifo.rows > 0) {
            fifo_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fifo.cols, fifo.rows);
            ggml_set_input(fifo_t);
        }
        ggml_tensor * pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pos_emb.cols, pos_emb.rows);
        ggml_tensor * pad_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * att_keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, att_mask.cols, att_mask.rows);
        ggml_tensor * keep_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, pad_mask.cols);
        ggml_tensor * tf_att_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pad_mask.cols, pad_mask.cols);
        ggml_set_input(pos);
        ggml_set_input(pad_keep);
        ggml_set_input(att_bias);
        ggml_set_input(att_keep);
        ggml_set_input(keep_t);
        ggml_set_input(tf_att_bias);

        std::vector<float> pad_keep_data((size_t) pad_mask.cols, 1.0f);
        std::vector<float> keep_data((size_t) pad_mask.cols, 0.0f);
        for (uint32_t i = 0; i < pad_mask.cols; ++i) {
            pad_keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
            keep_data[i] = pad_mask.data[i] > 0.5f ? 0.0f : 1.0f;
        }

        std::vector<float> att_bias_data((size_t) att_mask.rows * (size_t) att_mask.cols, 0.0f);
        std::vector<float> att_keep_data((size_t) att_mask.rows * (size_t) att_mask.cols, 1.0f);
        std::vector<float> tf_att_bias_data((size_t) pad_mask.cols * (size_t) pad_mask.cols, 0.0f);
        for (uint32_t r = 0; r < att_mask.rows; ++r) {
            const bool padded_query = pad_mask.data[r] > 0.5f;
            for (uint32_t c = 0; c < att_mask.cols; ++c) {
                const size_t idx = (size_t) r * (size_t) att_mask.cols + (size_t) c;
                att_bias_data[idx] = padded_query ? 0.0f : (att_mask.data[idx] > 0.5f ? -10000.0f : 0.0f);
                att_keep_data[idx] = att_mask.data[idx] > 0.5f ? 0.0f : 1.0f;
                tf_att_bias_data[idx] = pad_mask.data[c] > 0.5f ? 0.0f : -10000.0f;
            }
        }

        ggml_tensor * x = nullptr;
        if (spk_t != nullptr && fifo_t != nullptr) {
            x = ggml_concat(ctx, spk_t, fifo_t, 1);
            x = ggml_concat(ctx, x, chunk_proj, 1);
        } else if (spk_t != nullptr) {
            x = ggml_concat(ctx, spk_t, chunk_proj, 1);
        } else if (fifo_t != nullptr) {
            x = ggml_concat(ctx, fifo_t, chunk_proj, 1);
        } else {
            x = chunk_proj;
        }
        x = ggml_cont(ctx, x);
        ggml_tensor * cur = ggml_scale(ctx, x, std::sqrt((float) enc_d_model));

        for (uint32_t il = 0; il < meta.encoder_layers; ++il) {
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
            Qcur = ggml_reshape_3d(ctx, Qcur, enc_d_head, enc_n_head, Qcur->ne[1]);
            ggml_tensor * pos_bias_u = ggml_reshape_3d(ctx, tensor(".att.pbu"), enc_d_head, enc_n_head, 1);
            ggml_tensor * pos_bias_v = ggml_reshape_3d(ctx, tensor(".att.pbv"), enc_d_head, enc_n_head, 1);
            ggml_tensor * Q_bias_u = ggml_add(ctx, Qcur, pos_bias_u);
            Q_bias_u = ggml_permute(ctx, Q_bias_u, 0, 2, 1, 3);
            ggml_tensor * Q_bias_v = ggml_add(ctx, Qcur, pos_bias_v);
            Q_bias_v = ggml_permute(ctx, Q_bias_v, 0, 2, 1, 3);

            ggml_tensor * Kcur = mul_linear_project(ctx, tensor(".att.k.w"), att_norm, use_columnwise_matvec, prefix + ".att.k");
            Kcur = add_bias_2d(ctx, Kcur, tensor(".att.k.b"));
            Kcur = ggml_reshape_3d(ctx, Kcur, enc_d_head, enc_n_head, Kcur->ne[1]);
            Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = mul_linear_project(ctx, tensor(".att.v.w"), att_norm, use_columnwise_matvec, prefix + ".att.v");
            Vcur = add_bias_2d(ctx, Vcur, tensor(".att.v.b"));
            Vcur = ggml_reshape_3d(ctx, Vcur, enc_d_head, enc_n_head, Vcur->ne[1]);
            Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

            ggml_tensor * matrix_ac = mul_mat_checked(ctx, Q_bias_u, Kcur, prefix + ".att.matrix_ac");
            matrix_ac = ggml_cont(ctx, ggml_permute(ctx, matrix_ac, 1, 0, 2, 3));

            ggml_tensor * p = mul_linear_project(ctx, tensor(".att.p.w"), pos, use_columnwise_matvec, prefix + ".att.linear_pos");
            p = ggml_reshape_3d(ctx, p, enc_d_head, enc_n_head, p->ne[1]);
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
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) enc_d_head));
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

        if (chunk_proj->ne[0] == meta.fc_d_model) {
            cur = mul_linear_project(ctx, model.tensor("mods.ep.w"), cur, use_columnwise_matvec, "mods.ep");
            cur = add_bias_2d(ctx, cur, model.tensor("mods.ep.b"));
        } else if (chunk_proj->ne[0] != meta.tf_d_model) {
            throw std::runtime_error("encoder output columns do not match fc_d_model or tf_d_model");
        }

        for (uint32_t il = 0; il < meta.transformer_layers; ++il) {
            const std::string prefix = "te.l" + std::to_string(il);
            auto tensor = [&](const std::string & suffix) -> ggml_tensor * {
                return model.tensor(prefix + suffix);
            };

            ggml_tensor * residual = cur;

            ggml_tensor * q = mul_linear_project(ctx, tensor(".sa.q.w"), cur, use_columnwise_matvec, prefix + ".sa.q");
            q = add_bias_2d(ctx, q, tensor(".sa.q.b"));
            ggml_tensor * k = mul_linear_project(ctx, tensor(".sa.k.w"), cur, use_columnwise_matvec, prefix + ".sa.k");
            k = add_bias_2d(ctx, k, tensor(".sa.k.b"));
            ggml_tensor * v = mul_linear_project(ctx, tensor(".sa.v.w"), cur, use_columnwise_matvec, prefix + ".sa.v");
            v = add_bias_2d(ctx, v, tensor(".sa.v.b"));

            ggml_tensor * qh = ggml_reshape_3d(ctx, q, tf_d_head, tf_n_head, q->ne[1]);
            qh = ggml_permute(ctx, qh, 0, 2, 1, 3);
            ggml_tensor * kh = ggml_reshape_3d(ctx, k, tf_d_head, tf_n_head, k->ne[1]);
            kh = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));
            ggml_tensor * vh = ggml_reshape_3d(ctx, v, tf_d_head, tf_n_head, v->ne[1]);
            vh = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 2, 0, 3));

            ggml_tensor * scores = mul_mat_checked(ctx, qh, kh, prefix + ".sa.scores");
            scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) tf_d_head));
            scores = ggml_add(ctx, scores, tf_att_bias);
            ggml_tensor * probs = ggml_soft_max(ctx, scores);

            ggml_tensor * context = mul_mat_checked(ctx, probs, vh, prefix + ".sa.value_mix");
            context = ggml_permute(ctx, context, 2, 0, 1, 3);
            context = ggml_cont_2d(ctx, context, context->ne[0] * context->ne[1], context->ne[2]);

            ggml_tensor * att_out = mul_linear_project(ctx, tensor(".sa.o.w"), context, use_columnwise_matvec, prefix + ".sa.o");
            att_out = add_bias_2d(ctx, att_out, tensor(".sa.o.b"));
            ggml_tensor * att_res = ggml_add(ctx, residual, att_out);

            ggml_tensor * ln1 = apply_layer_norm(ctx, att_res, tensor(".ln1.w"), tensor(".ln1.b"), 1e-5f);

            ggml_tensor * ff_di = mul_linear_project(ctx, tensor(".ff.di.w"), ln1, use_columnwise_matvec, prefix + ".ff.di");
            ff_di = add_bias_2d(ctx, ff_di, tensor(".ff.di.b"));
            ggml_tensor * ff_act = ggml_relu(ctx, ff_di);
            ggml_tensor * ff_do = mul_linear_project(ctx, tensor(".ff.do.w"), ff_act, use_columnwise_matvec, prefix + ".ff.do");
            ff_do = add_bias_2d(ctx, ff_do, tensor(".ff.do.b"));
            ggml_tensor * ff_res = ggml_add(ctx, ln1, ff_do);

            cur = apply_layer_norm(ctx, ff_res, tensor(".ln2.w"), tensor(".ln2.b"), 1e-5f);
        }

        ggml_tensor * head_hidden1 = ggml_relu(ctx, cur);
        ggml_tensor * head_hidden2 = mul_linear_project(ctx, model.tensor("mods.fh2h.w"), head_hidden1, use_columnwise_matvec, "mods.fh2h");
        head_hidden2 = add_bias_2d(ctx, head_hidden2, model.tensor("mods.fh2h.b"));
        ggml_tensor * head_hidden3 = ggml_relu(ctx, head_hidden2);
        ggml_tensor * head_logits = mul_linear_project(ctx, model.tensor("mods.sh2s.w"), head_hidden3, use_columnwise_matvec, "mods.sh2s");
        head_logits = add_bias_2d(ctx, head_logits, model.tensor("mods.sh2s.b"));
        ggml_tensor * preds = ggml_sigmoid(ctx, head_logits);
        preds = ggml_mul(ctx, preds, keep_t);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer full-step graph tensors");
        }

        ggml_backend_tensor_set(inp, chunk_features.data.data(), 0, chunk_features.data.size() * sizeof(float));
        for (const auto & mask : pending_masks) {
            ggml_backend_tensor_set(mask.tensor, mask.data.data(), 0, mask.data.size() * sizeof(float));
        }
        if (spk_t != nullptr) {
            ggml_backend_tensor_set(spk_t, spkcache.data.data(), 0, spkcache.data.size() * sizeof(float));
        }
        if (fifo_t != nullptr) {
            ggml_backend_tensor_set(fifo_t, fifo.data.data(), 0, fifo.data.size() * sizeof(float));
        }
        ggml_backend_tensor_set(pos, pos_emb.data.data(), 0, pos_emb.data.size() * sizeof(float));
        ggml_backend_tensor_set(pad_keep, pad_keep_data.data(), 0, pad_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_bias, att_bias_data.data(), 0, att_bias_data.size() * sizeof(float));
        ggml_backend_tensor_set(att_keep, att_keep_data.data(), 0, att_keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(keep_t, keep_data.data(), 0, keep_data.size() * sizeof(float));
        ggml_backend_tensor_set(tf_att_bias, tf_att_bias_data.data(), 0, tf_att_bias_data.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1048576, false);
        ggml_build_forward_expand(gf, preds);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer full-step graph");
        }

        out.chunk_preencode = tensor_to_matrix_2d(chunk_proj);
        out.preds_all = tensor_to_matrix_2d(preds);

        ggml_backend_buffer_free(graph_buf);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    return out;
}

} // namespace llama::realtime
