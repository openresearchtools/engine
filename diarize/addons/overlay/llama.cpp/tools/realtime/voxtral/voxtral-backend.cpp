#include "voxtral-backend.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace llama::realtime {

namespace {

std::string lowercase_copy(const std::string & value) {
    std::string out = value;
    std::transform(
        out.begin(),
        out.end(),
        out.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return out;
}

voxtral_gpu_backend parse_voxtral_gpu_backend_name(const std::string & backend_name) {
    const std::string lowered = lowercase_copy(backend_name);
    if (lowered.empty() || lowered == "auto") {
        return voxtral_gpu_backend::auto_detect;
    }
    if (lowered == "cpu" || lowered == "none") {
        return voxtral_gpu_backend::none;
    }
    if (lowered.rfind("vulkan", 0) == 0) {
        return voxtral_gpu_backend::vulkan;
    }
    if (lowered.rfind("cuda", 0) == 0) {
        return voxtral_gpu_backend::cuda;
    }
    if (lowered.rfind("metal", 0) == 0) {
        return voxtral_gpu_backend::metal;
    }
    return voxtral_gpu_backend::auto_detect;
}

const char * voxtral_log_level_name(voxtral_log_level level) {
    switch (level) {
        case voxtral_log_level::error: return "error";
        case voxtral_log_level::warn:  return "warn";
        case voxtral_log_level::info:  return "info";
        case voxtral_log_level::debug: return "debug";
        default: return "log";
    }
}

void emit_voxtral_backend_log(voxtral_log_level level, const std::string & message) {
    std::fprintf(stderr, "voxtral[%s]: %s\n", voxtral_log_level_name(level), message.c_str());
}

void append_stream_events(
    const std::vector<voxtral_stream_event> & stream_events,
    std::vector<event> & out_events) {

    for (const auto & src : stream_events) {
        event dst;
        dst.begin_sec = src.begin_sec;
        dst.end_sec = src.end_sec;
        dst.text = src.text;
        dst.detail = src.detail;

        switch (src.kind) {
            case voxtral_stream_event_kind::piece_commit:
                dst.type = event_type::transcript_piece_commit;
                break;
            case voxtral_stream_event_kind::error:
                dst.type = event_type::backend_error;
                break;
            case voxtral_stream_event_kind::notice:
            default:
                dst.type = event_type::backend_status;
                break;
        }

        out_events.push_back(std::move(dst));
    }
}

} // namespace

std::shared_ptr<voxtral_loaded_model> voxtral_loaded_model::load_from_gguf(
    const std::string & gguf_path,
    const std::string & backend_name) {

    if (gguf_path.empty()) {
        throw std::invalid_argument("voxtral_loaded_model::load_from_gguf requires a model path");
    }

    const auto gpu_backend = parse_voxtral_gpu_backend_name(backend_name);
    voxtral_model * model = voxtral_model_load_from_file(
        gguf_path,
        nullptr,
        gpu_backend);
    if (model == nullptr) {
        throw std::runtime_error("failed to load Voxtral GGUF model");
    }
    return std::shared_ptr<voxtral_loaded_model>(
        new voxtral_loaded_model(model, backend_name));
}

voxtral_loaded_model::voxtral_loaded_model(
    voxtral_model * model,
    std::string backend_name)
    : model_(model),
      backend_name_(std::move(backend_name)) {

    if (model_ == nullptr) {
        throw std::invalid_argument("voxtral_loaded_model requires a loaded model");
    }
}

voxtral_loaded_model::~voxtral_loaded_model() {
    if (model_ != nullptr) {
        voxtral_model_free(model_);
        model_ = nullptr;
    }
}

voxtral_model * voxtral_loaded_model::model() const {
    return model_;
}

const std::string & voxtral_loaded_model::backend_name() const {
    return backend_name_;
}

std::mutex & voxtral_loaded_model::mutex() {
    return mutex_;
}

voxtral_stream_backend::voxtral_stream_backend(
    const std::string & gguf_path,
    const std::string & backend_name,
    bool capture_debug)
    : voxtral_stream_backend(
        voxtral_loaded_model::load_from_gguf(gguf_path, backend_name),
        capture_debug) {
}

voxtral_stream_backend::voxtral_stream_backend(
    std::shared_ptr<voxtral_loaded_model> loaded_model,
    bool capture_debug)
    : loaded_model_(std::move(loaded_model)),
      capture_debug_(capture_debug) {

    if (!loaded_model_) {
        throw std::invalid_argument("voxtral_stream_backend requires a loaded model");
    }

    voxtral_context_params params = {};
    params.gpu = parse_voxtral_gpu_backend_name(loaded_model_->backend_name());
    params.log_level = capture_debug_ ? voxtral_log_level::debug : voxtral_log_level::info;
    params.logger = emit_voxtral_backend_log;

    ctx_ = voxtral_init_from_model(loaded_model_->model(), params);
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize Voxtral runtime context");
    }
    stream_ = voxtral_stream_init(*ctx_, {});
    if (stream_ == nullptr) {
        voxtral_free(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error("failed to initialize Voxtral streaming session");
    }
    backend_name_ = format_backend_name(ctx_, loaded_model_->backend_name());
}

voxtral_stream_backend::~voxtral_stream_backend() {
    if (stream_ != nullptr) {
        voxtral_stream_free(stream_);
        stream_ = nullptr;
    }
    if (ctx_ != nullptr) {
        voxtral_free(ctx_);
        ctx_ = nullptr;
    }
}

std::string voxtral_stream_backend::backend_name() const {
    return backend_name_;
}

backend_limits voxtral_stream_backend::limits() const {
    backend_limits out;
    out.sample_rate_hz = VOXTRAL_SAMPLE_RATE;
    out.preferred_push_samples = static_cast<size_t>(VOXTRAL_SAMPLE_RATE / 2);
    out.max_buffered_samples = static_cast<size_t>(VOXTRAL_SAMPLE_RATE) * 120u;
    out.emits_transcript = true;
    out.emits_speaker_spans = false;
    return out;
}

void voxtral_stream_backend::reset() {
    if (stream_ == nullptr) {
        throw std::runtime_error("voxtral realtime stream is not initialized");
    }
    std::string error;
    if (!voxtral_stream_reset(*stream_, &error)) {
        throw std::runtime_error(
            error.empty()
                ? "failed to reset Voxtral realtime stream"
                : error);
    }
}

void voxtral_stream_backend::push_audio(
    const float * samples,
    size_t n_samples,
    std::vector<event> & out_events) {

    if (stream_ == nullptr) {
        throw std::runtime_error("voxtral realtime stream is not initialized");
    }
    std::vector<voxtral_stream_event> stream_events;
    std::string error;
    if (!voxtral_stream_push_audio(
            *stream_,
            samples,
            static_cast<int32_t>(n_samples),
            stream_events,
            &error)) {
        throw std::runtime_error(
            error.empty()
                ? "voxtral realtime push failed"
                : error);
    }
    append_stream_events(stream_events, out_events);
}

void voxtral_stream_backend::flush(std::vector<event> & out_events) {
    if (stream_ == nullptr) {
        throw std::runtime_error("voxtral realtime stream is not initialized");
    }
    std::vector<voxtral_stream_event> stream_events;
    std::string error;
    if (!voxtral_stream_flush(*stream_, stream_events, &error)) {
        throw std::runtime_error(
            error.empty()
                ? "voxtral realtime flush failed"
                : error);
    }
    append_stream_events(stream_events, out_events);
}

void voxtral_transcribe_offline_to_events(
    const std::shared_ptr<voxtral_loaded_model> & loaded_model,
    const std::vector<float> & audio,
    std::vector<event> & out_events,
    bool capture_debug) {

    if (!loaded_model) {
        throw std::invalid_argument("voxtral offline transcription requires a loaded model");
    }

    std::lock_guard<std::mutex> lock(loaded_model->mutex());

    voxtral_context_params params = {};
    params.gpu = parse_voxtral_gpu_backend_name(loaded_model->backend_name());
    params.log_level = capture_debug ? voxtral_log_level::debug : voxtral_log_level::info;
    params.logger = emit_voxtral_backend_log;

    voxtral_context * ctx = voxtral_init_from_model(loaded_model->model(), params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize Voxtral offline transcription context");
    }

    try {
        std::vector<voxtral_stream_event> stream_events;
        if (!voxtral_transcribe_audio_events(
                *ctx,
                audio,
                std::numeric_limits<int32_t>::max() / 4,
                stream_events,
                nullptr)) {
            throw std::runtime_error("voxtral offline transcription failed");
        }
        out_events.clear();
        append_stream_events(stream_events, out_events);
    } catch (...) {
        voxtral_free(ctx);
        throw;
    }

    voxtral_free(ctx);
}

voxtral_gpu_backend voxtral_stream_backend::parse_gpu_backend_name(const std::string & backend_name) {
    return parse_voxtral_gpu_backend_name(backend_name);
}

std::string voxtral_stream_backend::format_backend_name(
    const voxtral_context * /*ctx*/,
    const std::string & requested) {

    if (!requested.empty()) {
        return std::string("voxtral/") + requested;
    }
    return "voxtral/auto";
}

} // namespace llama::realtime
