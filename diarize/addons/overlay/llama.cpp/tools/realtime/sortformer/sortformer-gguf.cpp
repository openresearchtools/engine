#include "sortformer-gguf.h"

#include "gguf.h"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace llama::realtime {

namespace {

int64_t find_key_any(const gguf_context * ctx, const std::vector<const char *> & keys) {
    for (const char * key : keys) {
        const int64_t id = gguf_find_key(ctx, key);
        if (id >= 0) {
            return id;
        }
    }
    return -1;
}

int64_t require_key_any(const gguf_context * ctx, const std::vector<const char *> & keys, const char * label) {
    const int64_t id = find_key_any(ctx, keys);
    if (id >= 0) {
        return id;
    }
    std::ostringstream oss;
    oss << "missing GGUF metadata key for " << label << " (candidates:";
    for (const char * key : keys) {
        oss << " " << key;
    }
    oss << ")";
    throw std::runtime_error(oss.str());
}

uint32_t read_u32_flexible(const gguf_context * ctx, int64_t key_id, const char * label) {
    switch (gguf_get_kv_type(ctx, key_id)) {
    case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, key_id);
    case GGUF_TYPE_INT32: {
        const int32_t v = gguf_get_val_i32(ctx, key_id);
        if (v >= 0) {
            return static_cast<uint32_t>(v);
        }
        break;
    }
    case GGUF_TYPE_UINT64: return static_cast<uint32_t>(gguf_get_val_u64(ctx, key_id));
    case GGUF_TYPE_INT64: {
        const int64_t v = gguf_get_val_i64(ctx, key_id);
        if (v >= 0) {
            return static_cast<uint32_t>(v);
        }
        break;
    }
    default:
        break;
    }
    throw std::runtime_error(std::string("unexpected GGUF integer type for ") + label);
}

float read_f32_flexible(const gguf_context * ctx, int64_t key_id, const char * label) {
    switch (gguf_get_kv_type(ctx, key_id)) {
    case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, key_id);
    case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx, key_id));
    case GGUF_TYPE_UINT32: return static_cast<float>(gguf_get_val_u32(ctx, key_id));
    case GGUF_TYPE_INT32: return static_cast<float>(gguf_get_val_i32(ctx, key_id));
    default:
        throw std::runtime_error(std::string("unexpected GGUF float type for ") + label);
    }
}

bool read_bool_flexible(const gguf_context * ctx, int64_t key_id, const char * label) {
    switch (gguf_get_kv_type(ctx, key_id)) {
    case GGUF_TYPE_BOOL: return gguf_get_val_bool(ctx, key_id);
    case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, key_id) != 0;
    case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx, key_id) != 0;
    default:
        throw std::runtime_error(std::string("unexpected GGUF bool type for ") + label);
    }
}

std::string read_string_any(const gguf_context * ctx, const std::vector<const char *> & keys, const char * label) {
    const int64_t key_id = require_key_any(ctx, keys, label);
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("unexpected GGUF string type for ") + label);
    }
    return gguf_get_val_str(ctx, key_id);
}

uint32_t read_u32_any(const gguf_context * ctx, const std::vector<const char *> & keys, const char * label) {
    return read_u32_flexible(ctx, require_key_any(ctx, keys, label), label);
}

float read_f32_any(const gguf_context * ctx, const std::vector<const char *> & keys, const char * label) {
    return read_f32_flexible(ctx, require_key_any(ctx, keys, label), label);
}

bool read_bool_any(const gguf_context * ctx, const std::vector<const char *> & keys, const char * label) {
    return read_bool_flexible(ctx, require_key_any(ctx, keys, label), label);
}

bool try_read_bool_any(const gguf_context * ctx, const std::vector<const char *> & keys, bool & value) {
    const int64_t key_id = find_key_any(ctx, keys);
    if (key_id < 0) {
        return false;
    }
    value = read_bool_flexible(ctx, key_id, keys.front());
    return true;
}

bool try_read_u32_any(const gguf_context * ctx, const std::vector<const char *> & keys, uint32_t & value) {
    const int64_t key_id = find_key_any(ctx, keys);
    if (key_id < 0) {
        return false;
    }
    value = read_u32_flexible(ctx, key_id, keys.front());
    return true;
}

bool try_read_f32_any(const gguf_context * ctx, const std::vector<const char *> & keys, float & value) {
    const int64_t key_id = find_key_any(ctx, keys);
    if (key_id < 0) {
        return false;
    }
    value = read_f32_flexible(ctx, key_id, keys.front());
    return true;
}

bool try_read_string_any(const gguf_context * ctx, const std::vector<const char *> & keys, std::string & value) {
    const int64_t key_id = find_key_any(ctx, keys);
    if (key_id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("unexpected GGUF string type for ") + keys.front());
    }
    value = gguf_get_val_str(ctx, key_id);
    return true;
}

} // namespace

sortformer_model_metadata load_sortformer_gguf(const std::string & path) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to open Sortformer GGUF: " + path);
    }

    sortformer_model_metadata meta;
    meta.path = path;

    try {
        meta.architecture = read_string_any(ctx, {"general.architecture"}, "architecture");
        if (meta.architecture != "sortformer") {
            throw std::runtime_error("GGUF architecture is not sortformer");
        }

        meta.sample_rate_hz = read_u32_any(ctx, {
            "sortformer.config.sample_rate",
            "sortformer.config.preprocessor.sample_rate",
        }, "sample_rate");
        meta.window_size_sec = read_f32_any(ctx, {
            "sortformer.config.preprocessor.window_size",
        }, "window_size");
        meta.window_stride_sec = read_f32_any(ctx, {
            "sortformer.config.preprocessor.window_stride",
        }, "window_stride");
        meta.mel_bins = read_u32_any(ctx, {
            "sortformer.config.preprocessor.features",
        }, "mel_bins");
        try_read_string_any(ctx, {
            "sortformer.config.preprocessor.window",
        }, meta.window_type);
        try_read_u32_any(ctx, {
            "sortformer.config.preprocessor.n_fft",
        }, meta.n_fft);
        try_read_u32_any(ctx, {
            "sortformer.config.preprocessor.pad_to",
        }, meta.pad_to);
        try_read_f32_any(ctx, {
            "sortformer.config.preprocessor.preemph",
        }, meta.preemph);
        try_read_f32_any(ctx, {
            "sortformer.config.preprocessor.dither",
        }, meta.dither);
        try_read_string_any(ctx, {
            "sortformer.config.preprocessor.normalize",
        }, meta.normalize_mode);
        try_read_bool_any(ctx, {
            "sortformer.config.preprocessor.log",
        }, meta.log_features);
        try_read_f32_any(ctx, {
            "sortformer.config.preprocessor.log_zero_guard_value",
        }, meta.log_zero_guard_value);
        std::string log_zero_guard_type = "add";
        if (try_read_string_any(ctx, {
            "sortformer.config.preprocessor.log_zero_guard_type",
        }, log_zero_guard_type)) {
            meta.log_zero_guard_add = log_zero_guard_type != "clamp";
        }

        meta.streaming_mode = read_bool_any(ctx, {
            "sortformer.config.streaming_mode",
        }, "streaming_mode");
        meta.max_speakers = read_u32_any(ctx, {
            "sortformer.config.max_num_of_spks",
            "sortformer.config.sortformer_modules.num_spks",
        }, "max_speakers");

        meta.fc_d_model = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.fc_d_model",
        }, "fc_d_model");
        meta.tf_d_model = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.tf_d_model",
        }, "tf_d_model");
        meta.chunk_len_frames = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.chunk_len",
        }, "chunk_len");
        meta.fifo_len_frames = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.fifo_len",
        }, "fifo_len");
        meta.spkcache_len_frames = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.spkcache_len",
        }, "spkcache_len");
        meta.spkcache_update_period_frames = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.spkcache_update_period",
        }, "spkcache_update_period");
        meta.chunk_left_context = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.chunk_left_context",
        }, "chunk_left_context");
        meta.chunk_right_context = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.chunk_right_context",
        }, "chunk_right_context");
        meta.spkcache_sil_frames_per_spk = read_u32_any(ctx, {
            "sortformer.config.sortformer_modules.spkcache_sil_frames_per_spk",
        }, "spkcache_sil_frames_per_spk");
        meta.pred_score_threshold = read_f32_any(ctx, {
            "sortformer.config.sortformer_modules.pred_score_threshold",
        }, "pred_score_threshold");
        meta.strong_boost_rate = read_f32_any(ctx, {
            "sortformer.config.sortformer_modules.strong_boost_rate",
        }, "strong_boost_rate");
        meta.weak_boost_rate = read_f32_any(ctx, {
            "sortformer.config.sortformer_modules.weak_boost_rate",
        }, "weak_boost_rate");
        meta.min_pos_scores_rate = read_f32_any(ctx, {
            "sortformer.config.sortformer_modules.min_pos_scores_rate",
        }, "min_pos_scores_rate");
        meta.sil_threshold = read_f32_any(ctx, {
            "sortformer.config.sortformer_modules.sil_threshold",
        }, "sil_threshold");
        meta.scores_boost_latest = read_f32_any(ctx, {
            "sortformer.config.sortformer_modules.scores_boost_latest",
        }, "scores_boost_latest");

        meta.encoder_layers = read_u32_any(ctx, {
            "sortformer.config.encoder.n_layers",
        }, "encoder_layers");
        meta.encoder_d_model = read_u32_any(ctx, {
            "sortformer.config.encoder.d_model",
        }, "encoder_d_model");
        meta.encoder_heads = read_u32_any(ctx, {
            "sortformer.config.encoder.n_heads",
        }, "encoder_heads");
        meta.encoder_subsampling_factor = read_u32_any(ctx, {
            "sortformer.config.encoder.subsampling_factor",
        }, "encoder_subsampling_factor");

        meta.transformer_layers = read_u32_any(ctx, {
            "sortformer.config.transformer_encoder.num_layers",
        }, "transformer_layers");
        meta.transformer_heads = read_u32_any(ctx, {
            "sortformer.config.transformer_encoder.num_attention_heads",
        }, "transformer_heads");

        meta.tensor_count = gguf_get_n_tensors(ctx);
    } catch (...) {
        gguf_free(ctx);
        throw;
    }

    gguf_free(ctx);
    return meta;
}

std::string sortformer_metadata_summary(const sortformer_model_metadata & meta) {
    std::ostringstream oss;
    oss
        << "architecture=" << meta.architecture
        << ", sample_rate_hz=" << meta.sample_rate_hz
        << ", n_fft=" << meta.n_fft
        << ", mel_bins=" << meta.mel_bins
        << ", normalize=" << meta.normalize_mode
        << ", streaming_mode=" << (meta.streaming_mode ? "true" : "false")
        << ", max_speakers=" << meta.max_speakers
        << ", chunk_len_frames=" << meta.chunk_len_frames
        << ", fifo_len_frames=" << meta.fifo_len_frames
        << ", spkcache_len_frames=" << meta.spkcache_len_frames
        << ", spkcache_update_period_frames=" << meta.spkcache_update_period_frames
        << ", left_context_chunks=" << meta.chunk_left_context
        << ", right_context_chunks=" << meta.chunk_right_context
        << ", encoder_layers=" << meta.encoder_layers
        << ", encoder_subsampling_factor=" << meta.encoder_subsampling_factor
        << ", transformer_layers=" << meta.transformer_layers
        << ", tensor_count=" << meta.tensor_count;
    return oss.str();
}

} // namespace llama::realtime
