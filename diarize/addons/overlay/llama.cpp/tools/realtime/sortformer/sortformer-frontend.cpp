#include "sortformer-frontend.h"

#include "sortformer-encoder.h"

#include <cmath>
#include <stdexcept>

namespace llama::realtime {

namespace {

uint32_t count_valid_rows(const sortformer_matrix_f32 & matrix) {
    uint32_t valid = matrix.rows;
    while (valid > 0) {
        bool any_nonzero = false;
        const size_t row_offset = (size_t) (valid - 1) * matrix.cols;
        for (uint32_t col = 0; col < matrix.cols; ++col) {
            if (matrix.data[row_offset + col] != 0.0f) {
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

uint32_t calculate_preencode_valid_rows(uint32_t valid_input_rows) {
    uint32_t rows = valid_input_rows;
    rows = calculate_conv_output_size(rows, 3, 2, 1, 1);
    rows = calculate_conv_output_size(rows, 3, 2, 1, 1);
    rows = calculate_conv_output_size(rows, 3, 2, 1, 1);
    return rows;
}

sortformer_matrix_f32 make_rel_pos_emb(uint32_t seq_len, uint32_t d_model) {
    sortformer_matrix_f32 out;
    out.rows = 2 * seq_len - 1;
    out.cols = d_model;
    out.data.resize((size_t) out.rows * (size_t) out.cols, 0.0f);

    for (uint32_t row = 0; row < out.rows; ++row) {
        const float pos = (float) ((int32_t) seq_len - 1 - (int32_t) row);
        for (uint32_t col = 0; col < d_model; col += 2) {
            const float div = std::exp(-(std::log(10000.0f) / (float) d_model) * (float) col);
            out.data[(size_t) row * out.cols + col] = std::sin(pos * div);
            if (col + 1 < d_model) {
                out.data[(size_t) row * out.cols + col + 1] = std::cos(pos * div);
            }
        }
    }
    return out;
}

} // namespace

sortformer_frontend_outputs sortformer_run_frontend_encoder(
    const sortformer_model & model,
    const sortformer_matrix_f32 & features,
    bool capture_debug) {
    const uint32_t valid_input_rows = count_valid_rows(features);
    auto preencode_out = sortformer_run_preencode(model, features);
    auto out = sortformer_run_frontend_encoder_from_preencoded(
        model,
        preencode_out,
        calculate_preencode_valid_rows(valid_input_rows),
        capture_debug);
    if (capture_debug) {
        out.preencode_out = std::move(preencode_out);
    }
    return out;
}

sortformer_frontend_outputs sortformer_run_frontend_encoder_from_preencoded(
    const sortformer_model & model,
    const sortformer_matrix_f32 & preencoded,
    uint32_t valid_frames,
    bool capture_debug) {
    sortformer_frontend_outputs out;
    if (capture_debug) {
        out.preencode_out = preencoded;
    }

    const uint32_t total_frames = preencoded.rows;
    if (total_frames == 0 || preencoded.cols != model.metadata().encoder_d_model) {
        throw std::runtime_error("unexpected preencode output shape for frontend encoder");
    }
    if (valid_frames > total_frames) {
        throw std::runtime_error("calculated valid frontend frames exceed total preencode frames");
    }

    if (capture_debug) {
        auto posenc_x = preencoded;
        const float xscale = std::sqrt((float) model.metadata().encoder_d_model);
        for (float & v : posenc_x.data) {
            v *= xscale;
        }
        out.posenc_x = posenc_x;
    }

    auto pos_emb = make_rel_pos_emb(total_frames, model.metadata().encoder_d_model);
    if (capture_debug) {
        out.pos_emb = pos_emb;
    }

    sortformer_matrix_f32 pad_mask;
    pad_mask.rows = 1;
    pad_mask.cols = total_frames;
    pad_mask.data.resize((size_t) total_frames, 1.0f);

    out.encoder_mask.rows = 1;
    out.encoder_mask.cols = total_frames;
    out.encoder_mask.data.resize((size_t) total_frames, 0.0f);

    for (uint32_t i = 0; i < total_frames; ++i) {
        const float valid = i < valid_frames ? 1.0f : 0.0f;
        out.encoder_mask.data[i] = valid;
        pad_mask.data[i] = valid > 0.5f ? 0.0f : 1.0f;
    }
    if (capture_debug) {
        out.pad_mask = pad_mask;
    }

    sortformer_matrix_f32 att_mask;
    att_mask.rows = total_frames;
    att_mask.cols = total_frames;
    att_mask.data.resize((size_t) total_frames * (size_t) total_frames, 1.0f);
    for (uint32_t r = 0; r < total_frames; ++r) {
        for (uint32_t c = 0; c < total_frames; ++c) {
            const bool valid = out.encoder_mask.data[r] > 0.5f && out.encoder_mask.data[c] > 0.5f;
            att_mask.data[(size_t) r * total_frames + c] = valid ? 0.0f : 1.0f;
        }
    }
    if (capture_debug) {
        out.att_mask = att_mask;
    }

    out.encoder_out = sortformer_run_encoder(model, preencoded, pos_emb, pad_mask, att_mask);
    return out;
}

} // namespace llama::realtime
