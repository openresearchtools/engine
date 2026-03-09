#include "sortformer-preencode.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

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

ggml_tensor * add_bias_4d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    ggml_tensor * b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    return ggml_add(ctx, x, b);
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

ggml_tensor * mul_mat_blocked(
    ggml_context * ctx,
    ggml_tensor * a,
    ggml_tensor * b,
    bool use_blocked_matmul,
    const char * label) {
    b = ggml_cont(ctx, b);
    if (!use_blocked_matmul || b->ne[1] <= 128) {
        return mul_mat_checked(ctx, a, b, label);
    }

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < b->ne[1]; i += 128) {
        const int64_t cols = std::min<int64_t>(128, b->ne[1] - i);
        ggml_tensor * block = ggml_view_2d(ctx, b, b->ne[0], cols, b->nb[1], (size_t) i * b->nb[1]);
        block = ggml_cont(ctx, block);
        ggml_tensor * yi = mul_mat_checked(ctx, a, block, label);
        out = (out == nullptr) ? yi : ggml_concat(ctx, out, yi, 1);
    }
    return out;
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
    ggml_tensor * result = mul_mat_blocked(
        ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, weight, weight->ne[0] * weight->ne[1] * weight->ne[2], weight->ne[3]),
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

std::vector<char> read_file_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> bytes((size_t) size);
    if (size > 0) {
        in.read(bytes.data(), size);
    }
    return bytes;
}

void write_file_bytes(const std::string & path, const void * data, size_t nbytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out.write(reinterpret_cast<const char *>(data), (std::streamsize) nbytes);
}

void maybe_log_tensor_layout(const char * label, ggml_tensor * t) {
    if (std::getenv("SORTFORMER_DEBUG_LAYOUT") == nullptr) {
        return;
    }
    std::fprintf(
        stderr,
        "%s: ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu] cont=%d cont_ch=%d\n",
        label,
        (long long) t->ne[0],
        (long long) t->ne[1],
        (long long) t->ne[2],
        (long long) t->ne[3],
        (unsigned long long) t->nb[0],
        (unsigned long long) t->nb[1],
        (unsigned long long) t->nb[2],
        (unsigned long long) t->nb[3],
        ggml_is_contiguous(t) ? 1 : 0,
        ggml_is_contiguous_channels(t) ? 1 : 0);
}

void dump_tensor_as_time_feature_matrix(
    ggml_tensor * tensor,
    const std::string & path) {
    const int64_t freq = tensor->ne[0];
    const int64_t time = tensor->ne[1];
    const int64_t chan = tensor->ne[2];
    const int64_t batch = tensor->ne[3];
    if (batch != 1) {
        throw std::runtime_error("debug tensor dump currently expects batch=1");
    }

    std::vector<float> raw((size_t) freq * (size_t) time * (size_t) chan * (size_t) batch);
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(float));

    sortformer_matrix_f32 out;
    out.rows = static_cast<uint32_t>(time);
    out.cols = static_cast<uint32_t>(freq * chan);
    out.data.resize((size_t) out.rows * (size_t) out.cols);

    for (int64_t t = 0; t < time; ++t) {
        for (int64_t c = 0; c < chan; ++c) {
            for (int64_t f = 0; f < freq; ++f) {
                const size_t src_idx = (size_t) f + (size_t) t * (size_t) freq + (size_t) c * (size_t) freq * (size_t) time;
                const size_t dst_idx = (size_t) t * (size_t) out.cols + (size_t) c * (size_t) freq + (size_t) f;
                out.data[dst_idx] = raw[src_idx];
            }
        }
    }

    save_matrix_f32_bin(path, out);
}

uint32_t count_valid_feature_rows(const sortformer_matrix_f32 & features) {
    uint32_t valid = features.rows;
    while (valid > 0) {
        bool any_nonzero = false;
        const size_t row_offset = (size_t) (valid - 1) * features.cols;
        for (uint32_t col = 0; col < features.cols; ++col) {
            if (features.data[row_offset + col] != 0.0f) {
                any_nonzero = true;
                break;
            }
        }
        if (any_nonzero) {
            break;
        }
        --valid;
    }
    return valid;
}

uint32_t calculate_conv_output_size(uint32_t input_size, uint32_t kernel_size, uint32_t stride, uint32_t pad_left, uint32_t pad_right) {
    return (input_size + pad_left + pad_right - kernel_size) / stride + 1;
}

} // namespace

sortformer_matrix_f32 load_matrix_f32_bin(const std::string & path) {
    const auto bytes = read_file_bytes(path);
    require(bytes.size() >= sizeof(uint32_t) * 2, "matrix file too small");

    const uint32_t * header = reinterpret_cast<const uint32_t *>(bytes.data());
    sortformer_matrix_f32 matrix;
    matrix.rows = header[0];
    matrix.cols = header[1];

    const size_t expected = sizeof(uint32_t) * 2 + (size_t) matrix.rows * (size_t) matrix.cols * sizeof(float);
    require(bytes.size() == expected, "matrix file size does not match header");

    matrix.data.resize((size_t) matrix.rows * (size_t) matrix.cols);
    std::memcpy(matrix.data.data(), bytes.data() + sizeof(uint32_t) * 2, matrix.data.size() * sizeof(float));
    return matrix;
}

void save_matrix_f32_bin(const std::string & path, const sortformer_matrix_f32 & matrix) {
    std::vector<char> bytes(sizeof(uint32_t) * 2 + matrix.data.size() * sizeof(float));
    uint32_t * header = reinterpret_cast<uint32_t *>(bytes.data());
    header[0] = matrix.rows;
    header[1] = matrix.cols;
    if (!matrix.data.empty()) {
        std::memcpy(bytes.data() + sizeof(uint32_t) * 2, matrix.data.data(), matrix.data.size() * sizeof(float));
    }
    write_file_bytes(path, bytes.data(), bytes.size());
}

sortformer_matrix_f32 sortformer_run_preencode(
    const sortformer_model & model,
    const sortformer_matrix_f32 & features) {
    require(features.rows > 0 && features.cols == model.metadata().mel_bins, "invalid feature matrix shape");

    ggml_context * ctx = ggml_init(make_graph_ctx_params());
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer preencode graph context");
    }

    sortformer_matrix_f32 result;

    try {
        const std::string backend_name = ggml_backend_name(model.backend());
        const bool use_blocked_matmul = backend_name.rfind("Vulkan", 0) == 0;
        struct pending_mask {
            ggml_tensor * tensor;
            std::vector<float> data;
        };
        std::vector<pending_mask> pending_masks;

        ggml_tensor * inp = ggml_new_tensor_3d(
            ctx,
            GGML_TYPE_F32,
            features.cols,
            features.rows,
            1);
        ggml_set_name(inp, "sortformer.features");
        ggml_set_input(inp);

        const uint32_t valid_input_rows = count_valid_feature_rows(features);
        auto apply_time_mask = [&](ggml_tensor * tensor, uint32_t valid_time, const char * name) -> ggml_tensor * {
            if (std::getenv("SORTFORMER_DISABLE_MASKS") != nullptr) {
                return tensor;
            }
            const int64_t freq = tensor->ne[0];
            const int64_t time = tensor->ne[1];
            const int64_t channels = tensor->ne[2];
            const int64_t batch = tensor->ne[3];

            ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, freq, time, channels, batch);
            ggml_set_name(mask, name);

            std::vector<float> mask_data((size_t) freq * (size_t) time * (size_t) channels * (size_t) batch, 0.0f);
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t t = 0; t < time; ++t) {
                        if ((uint32_t) t >= valid_time) {
                            continue;
                        }
                        for (int64_t f = 0; f < freq; ++f) {
                            const size_t idx =
                                (((size_t) b * (size_t) channels + (size_t) c) * (size_t) time + (size_t) t) * (size_t) freq +
                                (size_t) f;
                            mask_data[idx] = 1.0f;
                        }
                    }
                }
            }

            pending_masks.push_back({ mask, std::move(mask_data) });
            return ggml_mul(ctx, tensor, mask);
        };

        ggml_tensor * x = inp;
        ggml_tensor * conv0 = conv2d_relu(
            ctx,
            x,
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
        maybe_log_tensor_layout("conv0", conv0);
        x = apply_time_mask(
            conv0,
            calculate_conv_output_size(valid_input_rows, 3, 2, 1, 1),
            "sortformer.mask.conv0");
        ggml_tensor * conv2 = conv2d_relu(
            ctx,
            x,
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
        maybe_log_tensor_layout("conv2", conv2);
        x = apply_time_mask(
            conv2,
            calculate_conv_output_size(calculate_conv_output_size(valid_input_rows, 3, 2, 1, 1), 3, 2, 1, 1),
            "sortformer.mask.conv2");
        ggml_tensor * conv3 = conv2d_relu(
            ctx,
            x,
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
        maybe_log_tensor_layout("conv3", conv3);
        x = apply_time_mask(
            conv3,
            calculate_conv_output_size(calculate_conv_output_size(valid_input_rows, 3, 2, 1, 1), 3, 2, 1, 1),
            "sortformer.mask.conv3");
        ggml_tensor * conv5 = conv2d_relu(
            ctx,
            x,
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
        maybe_log_tensor_layout("conv5", conv5);
        x = apply_time_mask(
            conv5,
            calculate_conv_output_size(calculate_conv_output_size(calculate_conv_output_size(valid_input_rows, 3, 2, 1, 1), 3, 2, 1, 1), 3, 2, 1, 1),
            "sortformer.mask.conv5");
        ggml_tensor * conv6 = conv2d_relu(
            ctx,
            x,
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
        maybe_log_tensor_layout("conv6", conv6);
        x = apply_time_mask(
            conv6,
            calculate_conv_output_size(calculate_conv_output_size(calculate_conv_output_size(valid_input_rows, 3, 2, 1, 1), 3, 2, 1, 1), 3, 2, 1, 1),
            "sortformer.mask.conv6");

        require(x->ne[2] == 256, "unexpected preencode channel count");

        ggml_tensor * flat = ggml_permute(ctx, x, 0, 2, 1, 3); // [freq, chan, time, batch]
        flat = ggml_cont(ctx, flat);
        flat = ggml_reshape_2d(ctx, flat, flat->ne[0] * flat->ne[1], flat->ne[2]); // [freq*chan, time]

        ggml_tensor * proj_w = model.tensor("enc.pre.out.w");
        if (proj_w->ne[0] != flat->ne[0]) {
            std::ostringstream oss;
            oss
                << "preencode projection shape mismatch: weight=[" << proj_w->ne[0] << "," << proj_w->ne[1]
                << "," << proj_w->ne[2] << "," << proj_w->ne[3] << "]"
                << " flat=[" << flat->ne[0] << "," << flat->ne[1]
                << "," << flat->ne[2] << "," << flat->ne[3] << "]"
                << " x=[" << x->ne[0] << "," << x->ne[1]
                << "," << x->ne[2] << "," << x->ne[3] << "]";
            throw std::runtime_error(oss.str());
        }

        ggml_tensor * proj = ggml_mul_mat(ctx, proj_w, flat);
        proj = add_bias_2d(ctx, proj, model.tensor("enc.pre.out.b"));
        ggml_set_name(proj, "sortformer.preencode.out");

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors(ctx, model.backend());
        if (graph_buf == nullptr) {
            throw std::runtime_error("failed to allocate Sortformer preencode graph tensors");
        }

        ggml_backend_tensor_set(inp, features.data.data(), 0, features.data.size() * sizeof(float));
        for (const auto & mask : pending_masks) {
            ggml_backend_tensor_set(mask.tensor, mask.data.data(), 0, mask.data.size() * sizeof(float));
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, proj);

        const ggml_status status = ggml_backend_graph_compute(model.backend(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(graph_buf);
            throw std::runtime_error("ggml_backend_graph_compute failed for Sortformer preencode");
        }

        if (const char * conv0_dump = std::getenv("SORTFORMER_DEBUG_CONV0_BIN")) {
            dump_tensor_as_time_feature_matrix(conv0, conv0_dump);
        }
        if (const char * conv2_dump = std::getenv("SORTFORMER_DEBUG_CONV2_BIN")) {
            dump_tensor_as_time_feature_matrix(conv2, conv2_dump);
        }
        if (const char * conv3_dump = std::getenv("SORTFORMER_DEBUG_CONV3_BIN")) {
            dump_tensor_as_time_feature_matrix(conv3, conv3_dump);
        }
        if (const char * conv5_dump = std::getenv("SORTFORMER_DEBUG_CONV5_BIN")) {
            dump_tensor_as_time_feature_matrix(conv5, conv5_dump);
        }
        if (const char * conv6_dump = std::getenv("SORTFORMER_DEBUG_CONV6_BIN")) {
            dump_tensor_as_time_feature_matrix(conv6, conv6_dump);
        }
        if (const char * flat_dump = std::getenv("SORTFORMER_DEBUG_FLAT_BIN")) {
            sortformer_matrix_f32 flat_out;
            flat_out.rows = static_cast<uint32_t>(flat->ne[1]);
            flat_out.cols = static_cast<uint32_t>(flat->ne[0]);
            flat_out.data.resize((size_t) flat_out.rows * (size_t) flat_out.cols);
            ggml_backend_tensor_get(flat, flat_out.data.data(), 0, flat_out.data.size() * sizeof(float));
            save_matrix_f32_bin(flat_dump, flat_out);
        }

        result.rows = static_cast<uint32_t>(proj->ne[1]);
        result.cols = static_cast<uint32_t>(proj->ne[0]);
        result.data.resize((size_t) result.rows * (size_t) result.cols);
        ggml_backend_tensor_get(proj, result.data.data(), 0, result.data.size() * sizeof(float));

        ggml_backend_buffer_free(graph_buf);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }

    ggml_free(ctx);
    return result;
}

double sortformer_max_abs_diff(const sortformer_matrix_f32 & a, const sortformer_matrix_f32 & b) {
    require(a.rows == b.rows && a.cols == b.cols, "matrix shape mismatch");
    double best = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const double av = (double) a.data[i];
        const double bv = (double) b.data[i];
        const bool a_finite = std::isfinite(av);
        const bool b_finite = std::isfinite(bv);
        if (!a_finite || !b_finite) {
            if ((std::isnan(av) && std::isnan(bv)) || (std::isinf(av) && std::isinf(bv) && std::signbit(av) == std::signbit(bv))) {
                continue;
            }
            return std::numeric_limits<double>::infinity();
        }
        best = std::max(best, std::abs(av - bv));
    }
    return best;
}

double sortformer_rmse(const sortformer_matrix_f32 & a, const sortformer_matrix_f32 & b) {
    require(a.rows == b.rows && a.cols == b.cols, "matrix shape mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const double av = (double) a.data[i];
        const double bv = (double) b.data[i];
        const bool a_finite = std::isfinite(av);
        const bool b_finite = std::isfinite(bv);
        if (!a_finite || !b_finite) {
            if ((std::isnan(av) && std::isnan(bv)) || (std::isinf(av) && std::isinf(bv) && std::signbit(av) == std::signbit(bv))) {
                continue;
            }
            return std::numeric_limits<double>::infinity();
        }
        const double d = av - bv;
        sum += d * d;
    }
    return std::sqrt(sum / std::max<size_t>(1, a.data.size()));
}

} // namespace llama::realtime
