#include "sortformer-streaming.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llama::realtime {

namespace {

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

sortformer_matrix_f32 make_empty_matrix(uint32_t cols) {
    sortformer_matrix_f32 out;
    out.rows = 0;
    out.cols = cols;
    return out;
}

sortformer_matrix_f32 concat_rows(
    const sortformer_matrix_f32 & a,
    const sortformer_matrix_f32 & b) {
    if (a.rows == 0) {
        return b;
    }
    if (b.rows == 0) {
        return a;
    }
    if (a.cols != b.cols) {
        throw std::runtime_error("concat_rows column mismatch");
    }

    sortformer_matrix_f32 out;
    out.rows = a.rows + b.rows;
    out.cols = a.cols;
    out.data.resize((size_t) out.rows * (size_t) out.cols);

    std::copy(a.data.begin(), a.data.end(), out.data.begin());
    std::copy(b.data.begin(), b.data.end(), out.data.begin() + (ptrdiff_t) a.data.size());
    return out;
}

sortformer_matrix_f32 slice_rows(
    const sortformer_matrix_f32 & in,
    uint32_t row_begin,
    uint32_t row_end) {
    if (row_begin > row_end || row_end > in.rows) {
        throw std::runtime_error("slice_rows range out of bounds");
    }

    sortformer_matrix_f32 out;
    out.rows = row_end - row_begin;
    out.cols = in.cols;
    out.data.resize((size_t) out.rows * (size_t) out.cols);

    const size_t cols = in.cols;
    const size_t begin = (size_t) row_begin * cols;
    const size_t end = (size_t) row_end * cols;
    std::copy(in.data.begin() + (ptrdiff_t) begin, in.data.begin() + (ptrdiff_t) end, out.data.begin());
    return out;
}

void validate_cache_matrix(const sortformer_matrix_f32 & m, uint32_t cols, const char * label) {
    if (m.cols != 0 && m.cols != cols) {
        throw std::runtime_error(std::string("invalid ") + label + " column dimension");
    }
}

void ensure_runtime_state_shape(sortformer_stream_runtime_state & state, const sortformer_model_metadata & meta) {
    if (state.spkcache.cols == 0) {
        state.spkcache.cols = meta.encoder_d_model;
    }
    if (state.fifo.cols == 0) {
        state.fifo.cols = meta.encoder_d_model;
    }
    if (state.spkcache_preds.cols == 0) {
        state.spkcache_preds.cols = meta.max_speakers;
    }
    if (state.fifo_preds.cols == 0) {
        state.fifo_preds.cols = meta.max_speakers;
    }
    if (state.mean_sil_emb.empty()) {
        state.mean_sil_emb.assign(meta.encoder_d_model, 0.0f);
    }

    validate_cache_matrix(state.spkcache, meta.encoder_d_model, "spkcache");
    validate_cache_matrix(state.fifo, meta.encoder_d_model, "fifo");
    validate_cache_matrix(state.spkcache_preds, meta.max_speakers, "spkcache_preds");
    validate_cache_matrix(state.fifo_preds, meta.max_speakers, "fifo_preds");
    if (state.mean_sil_emb.size() != meta.encoder_d_model) {
        throw std::runtime_error("invalid mean_sil_emb size");
    }
}

void ensure_fastpath_inputs(
    sortformer_stream_fastpath_inputs & inputs,
    const sortformer_model_metadata & meta,
    uint32_t rows,
    uint32_t valid_rows) {
    if (inputs.rows == rows &&
        inputs.valid_rows == valid_rows &&
        inputs.pos_emb.rows == (rows == 0 ? 0u : (2 * rows - 1)) &&
        inputs.pos_emb.cols == meta.encoder_d_model &&
        inputs.pad_mask.rows == 1 &&
        inputs.pad_mask.cols == rows &&
        inputs.att_mask.rows == rows &&
        inputs.att_mask.cols == rows) {
        return;
    }

    inputs.rows = rows;
    inputs.valid_rows = valid_rows;
    inputs.pos_emb = make_rel_pos_emb(rows, meta.encoder_d_model);

    inputs.pad_mask.rows = 1;
    inputs.pad_mask.cols = rows;
    inputs.pad_mask.data.resize((size_t) rows, 1.0f);
    for (uint32_t i = 0; i < rows; ++i) {
        const float valid = i < valid_rows ? 1.0f : 0.0f;
        inputs.pad_mask.data[i] = valid > 0.5f ? 0.0f : 1.0f;
    }

    inputs.att_mask.rows = rows;
    inputs.att_mask.cols = rows;
    inputs.att_mask.data.resize((size_t) rows * (size_t) rows, 1.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < rows; ++c) {
            const bool valid = inputs.pad_mask.data[r] < 0.5f && inputs.pad_mask.data[c] < 0.5f;
            inputs.att_mask.data[(size_t) r * rows + c] = valid ? 0.0f : 1.0f;
        }
    }
}

void update_silence_profile(
    const sortformer_model_metadata & meta,
    const sortformer_matrix_f32 & embs,
    const sortformer_matrix_f32 & preds,
    std::vector<float> & mean_sil_emb,
    uint64_t & n_sil_frames) {
    if (embs.rows != preds.rows || embs.cols != meta.encoder_d_model || preds.cols != meta.max_speakers) {
        throw std::runtime_error("silence profile update shape mismatch");
    }

    for (uint32_t row = 0; row < preds.rows; ++row) {
        float pred_sum = 0.0f;
        const size_t pred_off = (size_t) row * preds.cols;
        for (uint32_t spk = 0; spk < preds.cols; ++spk) {
            pred_sum += preds.data[pred_off + spk];
        }
        if (pred_sum >= meta.sil_threshold) {
            continue;
        }

        const size_t emb_off = (size_t) row * embs.cols;
        const float old_count = static_cast<float>(n_sil_frames);
        const float new_count = static_cast<float>(n_sil_frames + 1);
        for (uint32_t i = 0; i < embs.cols; ++i) {
            mean_sil_emb[i] = (mean_sil_emb[i] * old_count + embs.data[emb_off + i]) / new_count;
        }
        ++n_sil_frames;
    }
}

sortformer_matrix_f32 get_log_pred_scores(
    const sortformer_model_metadata & meta,
    const sortformer_matrix_f32 & preds) {
    sortformer_matrix_f32 scores;
    scores.rows = preds.rows;
    scores.cols = preds.cols;
    scores.data.resize(preds.data.size(), 0.0f);

    for (uint32_t row = 0; row < preds.rows; ++row) {
        const size_t off = (size_t) row * preds.cols;
        double log_1_probs_sum = 0.0;
        for (uint32_t spk = 0; spk < preds.cols; ++spk) {
            const double p_raw = preds.data[off + spk];
            log_1_probs_sum += std::log(std::max(1.0 - p_raw, (double) meta.pred_score_threshold));
        }
        for (uint32_t spk = 0; spk < preds.cols; ++spk) {
            const double p_raw = preds.data[off + spk];
            const double p = std::max<double>(p_raw, meta.pred_score_threshold);
            const double log_p = std::log(p);
            const double log_1_p = std::log(std::max(1.0 - p_raw, (double) meta.pred_score_threshold));
            scores.data[off + spk] = static_cast<float>(log_p - log_1_p + log_1_probs_sum - std::log(0.5));
        }
    }

    return scores;
}

void disable_low_scores(
    const sortformer_model_metadata & meta,
    const sortformer_matrix_f32 & preds,
    sortformer_matrix_f32 & scores) {
    if (preds.rows != scores.rows || preds.cols != scores.cols) {
        throw std::runtime_error("disable_low_scores shape mismatch");
    }

    const uint32_t n_spk = preds.cols;
    const uint32_t n_frames = preds.rows;
    const uint32_t spkcache_len_per_spk = meta.spkcache_len_frames / std::max<uint32_t>(1, n_spk);
    const uint32_t min_pos_scores_per_spk = (spkcache_len_per_spk > meta.spkcache_sil_frames_per_spk)
        ? static_cast<uint32_t>(std::floor((float) (spkcache_len_per_spk - meta.spkcache_sil_frames_per_spk) * meta.min_pos_scores_rate))
        : 0;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (uint32_t row = 0; row < n_frames; ++row) {
        const size_t off = (size_t) row * n_spk;
        for (uint32_t spk = 0; spk < n_spk; ++spk) {
            const bool is_speech = preds.data[off + spk] > 0.5f;
            if (!is_speech) {
                scores.data[off + spk] = neg_inf;
            }
        }
    }

    std::vector<uint32_t> pos_counts(n_spk, 0);
    for (uint32_t row = 0; row < n_frames; ++row) {
        const size_t off = (size_t) row * n_spk;
        for (uint32_t spk = 0; spk < n_spk; ++spk) {
            if (scores.data[off + spk] > 0.0f) {
                ++pos_counts[spk];
            }
        }
    }

    for (uint32_t row = 0; row < n_frames; ++row) {
        const size_t off = (size_t) row * n_spk;
        for (uint32_t spk = 0; spk < n_spk; ++spk) {
            const bool is_speech = preds.data[off + spk] > 0.5f;
            const bool is_pos = scores.data[off + spk] > 0.0f;
            if (!is_pos && is_speech && pos_counts[spk] >= min_pos_scores_per_spk) {
                scores.data[off + spk] = neg_inf;
            }
        }
    }
}

void boost_topk_scores(sortformer_matrix_f32 & scores, uint32_t n_boost_per_spk, float scale_factor) {
    if (n_boost_per_spk == 0 || scores.rows == 0 || scores.cols == 0) {
        return;
    }

    for (uint32_t spk = 0; spk < scores.cols; ++spk) {
        std::vector<std::pair<float, uint32_t>> ordered;
        ordered.reserve(scores.rows);
        for (uint32_t row = 0; row < scores.rows; ++row) {
            ordered.emplace_back(scores.data[(size_t) row * scores.cols + spk], row);
        }
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) {
            if (a.first == b.first) {
                return a.second < b.second;
            }
            return a.first > b.first;
        });
        const uint32_t limit = std::min<uint32_t>(n_boost_per_spk, (uint32_t) ordered.size());
        for (uint32_t i = 0; i < limit; ++i) {
            const float score = ordered[i].first;
            if (score == -std::numeric_limits<float>::infinity()) {
                continue;
            }
            const uint32_t row = ordered[i].second;
            scores.data[(size_t) row * scores.cols + spk] -= static_cast<float>(scale_factor * std::log(0.5));
        }
    }
}

std::pair<std::vector<uint32_t>, std::vector<bool>> get_topk_indices(
    const sortformer_model_metadata & meta,
    const sortformer_matrix_f32 & scores) {
    const uint32_t n_frames = scores.rows;
    const uint32_t n_spk = scores.cols;
    const uint32_t n_frames_no_sil = n_frames - std::min<uint32_t>(n_frames, meta.spkcache_sil_frames_per_spk);
    const size_t max_index = std::numeric_limits<size_t>::max();

    std::vector<std::pair<size_t, float>> flat_scores;
    flat_scores.reserve((size_t) n_frames * (size_t) n_spk);
    for (uint32_t spk = 0; spk < n_spk; ++spk) {
        for (uint32_t row = 0; row < n_frames; ++row) {
            const size_t flat_idx = (size_t) spk * (size_t) n_frames + (size_t) row;
            flat_scores.emplace_back(flat_idx, scores.data[(size_t) row * n_spk + spk]);
        }
    }
    std::stable_sort(flat_scores.begin(), flat_scores.end(), [](const auto & a, const auto & b) {
        if (a.second == b.second) {
            return a.first < b.first;
        }
        return a.second > b.second;
    });

    std::vector<size_t> topk_flat(meta.spkcache_len_frames, max_index);
    for (uint32_t i = 0; i < meta.spkcache_len_frames && i < flat_scores.size(); ++i) {
        topk_flat[i] = std::isinf(flat_scores[i].second) && flat_scores[i].second < 0 ? max_index : flat_scores[i].first;
    }
    std::sort(topk_flat.begin(), topk_flat.end());

    std::vector<uint32_t> frame_indices(meta.spkcache_len_frames, 0);
    std::vector<bool> disabled(meta.spkcache_len_frames, false);
    for (uint32_t i = 0; i < meta.spkcache_len_frames; ++i) {
        const size_t flat_idx = topk_flat[i];
        if (flat_idx == max_index) {
            disabled[i] = true;
            continue;
        }
        const uint32_t frame_idx = static_cast<uint32_t>(flat_idx % n_frames);
        if (frame_idx >= n_frames_no_sil) {
            disabled[i] = true;
            continue;
        }
        frame_indices[i] = frame_idx;
    }

    return {std::move(frame_indices), std::move(disabled)};
}

std::pair<sortformer_matrix_f32, sortformer_matrix_f32> gather_spkcache_and_preds(
    const sortformer_model_metadata & meta,
    const sortformer_matrix_f32 & emb_seq,
    const sortformer_matrix_f32 & preds,
    const std::vector<uint32_t> & topk_indices,
    const std::vector<bool> & is_disabled,
    const std::vector<float> & mean_sil_emb) {
    sortformer_matrix_f32 gathered_embs;
    gathered_embs.rows = meta.spkcache_len_frames;
    gathered_embs.cols = emb_seq.cols;
    gathered_embs.data.resize((size_t) gathered_embs.rows * gathered_embs.cols, 0.0f);

    sortformer_matrix_f32 gathered_preds;
    gathered_preds.rows = meta.spkcache_len_frames;
    gathered_preds.cols = preds.cols;
    gathered_preds.data.resize((size_t) gathered_preds.rows * gathered_preds.cols, 0.0f);

    for (uint32_t i = 0; i < meta.spkcache_len_frames; ++i) {
        float * emb_dst = gathered_embs.data.data() + (size_t) i * gathered_embs.cols;
        float * pred_dst = gathered_preds.data.data() + (size_t) i * gathered_preds.cols;

        if (is_disabled[i]) {
            std::copy(mean_sil_emb.begin(), mean_sil_emb.end(), emb_dst);
            std::fill(pred_dst, pred_dst + gathered_preds.cols, 0.0f);
            continue;
        }

        const uint32_t row = topk_indices[i];
        std::copy_n(emb_seq.data.data() + (size_t) row * emb_seq.cols, emb_seq.cols, emb_dst);
        std::copy_n(preds.data.data() + (size_t) row * preds.cols, preds.cols, pred_dst);
    }

    return {std::move(gathered_embs), std::move(gathered_preds)};
}

void compress_spkcache(
    const sortformer_model_metadata & meta,
    sortformer_stream_runtime_state & state) {
    if (!state.have_spkcache_preds) {
        return;
    }
    if (state.spkcache.rows <= meta.spkcache_len_frames) {
        return;
    }

    sortformer_matrix_f32 scores = get_log_pred_scores(meta, state.spkcache_preds);
    disable_low_scores(meta, state.spkcache_preds, scores);

    if (meta.scores_boost_latest > 0.0f && state.spkcache.rows > meta.spkcache_len_frames) {
        for (uint32_t row = meta.spkcache_len_frames; row < scores.rows; ++row) {
            for (uint32_t spk = 0; spk < scores.cols; ++spk) {
                const size_t idx = (size_t) row * scores.cols + spk;
                if (scores.data[idx] != -std::numeric_limits<float>::infinity()) {
                    scores.data[idx] += meta.scores_boost_latest;
                }
            }
        }
    }

    const uint32_t n_spk = std::max<uint32_t>(1, meta.max_speakers);
    const uint32_t spkcache_len_per_spk = meta.spkcache_len_frames / n_spk;
    const uint32_t usable_per_spk = (spkcache_len_per_spk > meta.spkcache_sil_frames_per_spk)
        ? (spkcache_len_per_spk - meta.spkcache_sil_frames_per_spk)
        : 0;
    const uint32_t strong_boost_per_spk = static_cast<uint32_t>(std::floor((float) usable_per_spk * meta.strong_boost_rate));
    const uint32_t weak_boost_per_spk = static_cast<uint32_t>(std::floor((float) usable_per_spk * meta.weak_boost_rate));

    boost_topk_scores(scores, strong_boost_per_spk, 2.0f);
    boost_topk_scores(scores, weak_boost_per_spk, 1.0f);

    if (meta.spkcache_sil_frames_per_spk > 0) {
        sortformer_matrix_f32 padded;
        padded.rows = scores.rows + meta.spkcache_sil_frames_per_spk;
        padded.cols = scores.cols;
        padded.data.resize((size_t) padded.rows * padded.cols, -std::numeric_limits<float>::infinity());
        std::copy(scores.data.begin(), scores.data.end(), padded.data.begin());
        for (uint32_t row = scores.rows; row < padded.rows; ++row) {
            for (uint32_t spk = 0; spk < padded.cols; ++spk) {
                padded.data[(size_t) row * padded.cols + spk] = std::numeric_limits<float>::infinity();
            }
        }
        scores = std::move(padded);
    }

    auto [topk_indices, is_disabled] = get_topk_indices(meta, scores);
    auto [new_embs, new_preds] = gather_spkcache_and_preds(
        meta,
        state.spkcache,
        state.spkcache_preds,
        topk_indices,
        is_disabled,
        state.mean_sil_emb);

    state.spkcache = std::move(new_embs);
    state.spkcache_preds = std::move(new_preds);
    state.have_spkcache_preds = true;
}

} // namespace

sortformer_stream_step_outputs sortformer_run_stream_step(
    const sortformer_model & model,
    const sortformer_matrix_f32 & chunk_features,
    uint32_t chunk_valid_feature_rows,
    const sortformer_stream_cache_state & cache_state,
    uint32_t left_context_rows,
    uint32_t right_context_rows,
    bool capture_debug,
    sortformer_stream_fastpath_inputs * fastpath_inputs,
    sortformer_encoder_postnet_plan * encoder_postnet_plan) {
    if (chunk_features.rows == 0 || chunk_features.cols != model.metadata().mel_bins) {
        throw std::runtime_error("invalid Sortformer chunk feature matrix");
    }
    if (chunk_valid_feature_rows > chunk_features.rows) {
        throw std::runtime_error("chunk_valid_feature_rows exceeds chunk feature rows");
    }
    if (cache_state.spkcache.cols != 0 && cache_state.spkcache.cols != model.metadata().encoder_d_model) {
        throw std::runtime_error("invalid spkcache embedding dimension");
    }
    if (cache_state.fifo.cols != 0 && cache_state.fifo.cols != model.metadata().encoder_d_model) {
        throw std::runtime_error("invalid fifo embedding dimension");
    }

    sortformer_stream_step_outputs out;
    out.chunk_preencode_valid_rows = calculate_preencode_valid_rows(chunk_valid_feature_rows);
    if (capture_debug) {
        out.chunk_preencode = sortformer_run_preencode(model, chunk_features);
        if (out.chunk_preencode_valid_rows > out.chunk_preencode.rows) {
            throw std::runtime_error("computed valid preencode rows exceed chunk_preencode rows");
        }
        if (left_context_rows + right_context_rows > out.chunk_preencode_valid_rows) {
            throw std::runtime_error("stream-step context rows exceed valid preencode rows");
        }
        out.left_context_rows = left_context_rows;
        out.right_context_rows = right_context_rows;

        auto concat_preencode = concat_rows(concat_rows(cache_state.spkcache, cache_state.fifo), out.chunk_preencode);
        out.concat_valid_rows = cache_state.spkcache.rows + cache_state.fifo.rows + out.chunk_preencode_valid_rows;
        if (out.concat_valid_rows > concat_preencode.rows) {
            throw std::runtime_error("concat_valid_rows exceed concat_preencode rows");
        }
        out.concat_preencode = concat_preencode;
        out.frontend = sortformer_run_frontend_encoder_from_preencoded(model, concat_preencode, out.concat_valid_rows, true);
        out.postnet = sortformer_run_postnet(model, out.frontend.encoder_out, out.frontend.encoder_mask, true);
        out.preds_all = out.postnet.preds;
    } else {
        out.chunk_preencode = sortformer_run_preencode(model, chunk_features);
        if (out.chunk_preencode_valid_rows > out.chunk_preencode.rows) {
            throw std::runtime_error("computed valid preencode rows exceed chunk_preencode rows");
        }
        if (left_context_rows + right_context_rows > out.chunk_preencode_valid_rows) {
            throw std::runtime_error("stream-step context rows exceed valid preencode rows");
        }
        out.left_context_rows = left_context_rows;
        out.right_context_rows = right_context_rows;

        sortformer_stream_fastpath_inputs local_inputs;
        sortformer_stream_fastpath_inputs * inputs = fastpath_inputs != nullptr ? fastpath_inputs : &local_inputs;
        out.concat_valid_rows = cache_state.spkcache.rows + cache_state.fifo.rows + out.chunk_preencode_valid_rows;
        ensure_fastpath_inputs(*inputs, model.metadata(), cache_state.spkcache.rows + cache_state.fifo.rows + out.chunk_preencode.rows, out.concat_valid_rows);
        if (encoder_postnet_plan != nullptr) {
            out.preds_all = sortformer_run_encoder_postnet_concat_cached(
                *encoder_postnet_plan,
                model,
                cache_state.spkcache,
                cache_state.fifo,
                out.chunk_preencode,
                inputs->pos_emb,
                inputs->pad_mask,
                inputs->att_mask);
        } else {
            out.preds_all = sortformer_run_encoder_postnet_concat(
                model,
                cache_state.spkcache,
                cache_state.fifo,
                out.chunk_preencode,
                inputs->pos_emb,
                inputs->pad_mask,
                inputs->att_mask);
        }
    }

    const uint32_t core_valid_rows = out.chunk_preencode_valid_rows - left_context_rows - right_context_rows;
    const uint32_t emit_count = std::min<uint32_t>(model.metadata().chunk_len_frames, core_valid_rows);
    const uint32_t emit_begin = cache_state.spkcache.rows + cache_state.fifo.rows + left_context_rows;
    if (capture_debug) {
        out.chunk_core_preencode = slice_rows(out.chunk_preencode, left_context_rows, left_context_rows + emit_count);
    }
    out.chunk_preds = slice_rows(out.preds_all, emit_begin, emit_begin + emit_count);
    return out;
}

sortformer_stream_runtime_state sortformer_make_stream_runtime_state(const sortformer_model_metadata & meta) {
    sortformer_stream_runtime_state state;
    state.spkcache = make_empty_matrix(meta.encoder_d_model);
    state.spkcache_preds = make_empty_matrix(meta.max_speakers);
    state.fifo = make_empty_matrix(meta.encoder_d_model);
    state.fifo_preds = make_empty_matrix(meta.max_speakers);
    state.mean_sil_emb.assign(meta.encoder_d_model, 0.0f);
    return state;
}

sortformer_stream_step_outputs sortformer_streaming_update(
    const sortformer_model & model,
    const sortformer_matrix_f32 & chunk_features,
    uint32_t chunk_valid_feature_rows,
    uint32_t left_context_rows,
    uint32_t right_context_rows,
    sortformer_stream_runtime_state & state,
    bool capture_debug) {
    const auto & meta = model.metadata();
    ensure_runtime_state_shape(state, meta);

    const uint32_t spkcache_len = state.spkcache.rows;
    const uint32_t fifo_len = state.fifo.rows;

    sortformer_stream_cache_state cache_view;
    cache_view.spkcache = state.spkcache;
    cache_view.fifo = state.fifo;
    auto outputs = sortformer_run_stream_step(
        model,
        chunk_features,
        chunk_valid_feature_rows,
        cache_view,
        left_context_rows,
        right_context_rows,
        capture_debug,
        capture_debug ? nullptr : &state.fastpath_inputs,
        capture_debug ? nullptr : &state.encoder_postnet_plan);

    if (fifo_len > 0) {
        state.fifo_preds = slice_rows(outputs.preds_all, spkcache_len, spkcache_len + fifo_len);
    } else {
        state.fifo_preds = make_empty_matrix(meta.max_speakers);
    }

    const uint32_t cache_chunk_rows = outputs.chunk_preencode.rows - left_context_rows - right_context_rows;
    const auto cache_chunk_preencode = slice_rows(
        outputs.chunk_preencode,
        left_context_rows,
        left_context_rows + cache_chunk_rows);
    const auto cache_chunk_preds = slice_rows(
        outputs.preds_all,
        spkcache_len + fifo_len + left_context_rows,
        spkcache_len + fifo_len + left_context_rows + cache_chunk_rows);

    state.fifo = concat_rows(state.fifo, cache_chunk_preencode);
    state.fifo_preds = concat_rows(state.fifo_preds, cache_chunk_preds);

    if (fifo_len + cache_chunk_rows > meta.fifo_len_frames) {
        uint32_t pop_out_len = meta.spkcache_update_period_frames;
        pop_out_len = std::max<uint32_t>(pop_out_len, cache_chunk_rows - meta.fifo_len_frames + fifo_len);
        pop_out_len = std::min<uint32_t>(pop_out_len, fifo_len + cache_chunk_rows);

        const auto pop_out_embs = slice_rows(state.fifo, 0, pop_out_len);
        const auto pop_out_preds = slice_rows(state.fifo_preds, 0, pop_out_len);

        update_silence_profile(meta, pop_out_embs, pop_out_preds, state.mean_sil_emb, state.n_sil_frames);

        state.fifo = slice_rows(state.fifo, pop_out_len, state.fifo.rows);
        state.fifo_preds = slice_rows(state.fifo_preds, pop_out_len, state.fifo_preds.rows);

        state.spkcache = concat_rows(state.spkcache, pop_out_embs);
        if (state.have_spkcache_preds) {
            state.spkcache_preds = concat_rows(state.spkcache_preds, pop_out_preds);
        }

        if (state.spkcache.rows > meta.spkcache_len_frames) {
            if (!state.have_spkcache_preds) {
                const auto current_cache_preds = slice_rows(outputs.preds_all, 0, spkcache_len);
                state.spkcache_preds = concat_rows(current_cache_preds, pop_out_preds);
                state.have_spkcache_preds = true;
            }
            compress_spkcache(meta, state);
        }
    }

    return outputs;
}

} // namespace llama::realtime
