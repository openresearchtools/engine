#include "backend-factory.h"

#include "sortformer/sortformer-backend.h"
#include "voxtral/voxtral-backend.h"
#include "gguf.h"

#include <array>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llama::realtime {

void loaded_backend_model::transcribe_audio_offline(
    const std::vector<float> &,
    std::vector<event> &,
    bool) const {

    throw std::runtime_error("offline transcription is not supported by this backend model");
}

namespace {

using create_backend_fn = std::unique_ptr<stream_backend> (*)(
    const backend_model_params & params,
    bool capture_debug);

using load_model_fn = std::shared_ptr<loaded_backend_model> (*)(
    const backend_model_params & params);
using resolve_runtime_fn = bool (*)(
    const backend_model_params & params,
    uint32_t requested_sample_rate_hz,
    uint32_t requested_audio_ring_capacity_samples,
    backend_model_params & out_model_params,
    stream_session_config & out_session_config,
    std::string & error);

class sortformer_loaded_backend_model final : public loaded_backend_model {
public:
    explicit sortformer_loaded_backend_model(
        std::shared_ptr<sortformer_loaded_model> loaded_model,
        std::string backend_name)
        : loaded_model_(std::move(loaded_model)),
          backend_name_(std::move(backend_name)) {

        if (!loaded_model_) {
            throw std::invalid_argument("sortformer loaded backend model requires a loaded model");
        }
    }

    backend_kind kind() const override {
        return backend_kind::sortformer;
    }

    std::string backend_name() const override {
        return backend_name_;
    }

    stream_session_config default_session_config() const override {
        stream_session_config cfg = {};
        cfg.expected_sample_rate_hz = 16000u;
        cfg.audio_ring_capacity_samples = 16000u * 120u;
        cfg.continuous_mode = true;
        return cfg;
    }

    std::unique_ptr<stream_backend> create_backend(bool capture_debug) const override {
        std::lock_guard<std::mutex> lock(loaded_model_->mutex());
        return std::make_unique<sortformer_stream_backend>(loaded_model_, capture_debug);
    }

private:
    std::shared_ptr<sortformer_loaded_model> loaded_model_;
    std::string backend_name_;
};

class voxtral_loaded_backend_model final : public loaded_backend_model {
public:
    explicit voxtral_loaded_backend_model(
        std::shared_ptr<voxtral_loaded_model> loaded_model,
        std::string backend_name)
        : loaded_model_(std::move(loaded_model)),
          backend_name_(std::move(backend_name)) {

        if (!loaded_model_) {
            throw std::invalid_argument("voxtral loaded backend model requires a loaded model");
        }
    }

    backend_kind kind() const override {
        return backend_kind::voxtral_realtime;
    }

    std::string backend_name() const override {
        return backend_name_;
    }

    stream_session_config default_session_config() const override {
        stream_session_config cfg = {};
        cfg.expected_sample_rate_hz = VOXTRAL_SAMPLE_RATE;
        cfg.audio_ring_capacity_samples = VOXTRAL_SAMPLE_RATE * 120u;
        cfg.continuous_mode = true;
        return cfg;
    }

    std::unique_ptr<stream_backend> create_backend(bool capture_debug) const override {
        std::lock_guard<std::mutex> lock(loaded_model_->mutex());
        return std::make_unique<voxtral_stream_backend>(loaded_model_, capture_debug);
    }

    bool supports_offline_transcription() const override {
        return true;
    }

    void transcribe_audio_offline(
        const std::vector<float> & audio,
        std::vector<event> & out_events,
        bool capture_debug) const override {

        voxtral_transcribe_offline_to_events(loaded_model_, audio, out_events, capture_debug);
    }

private:
    std::shared_ptr<voxtral_loaded_model> loaded_model_;
    std::string backend_name_;
};

std::unique_ptr<stream_backend> create_sortformer_backend(
    const backend_model_params & params,
    bool capture_debug) {

    return std::make_unique<sortformer_stream_backend>(
        params.model_path,
        params.backend_name,
        capture_debug);
}

std::shared_ptr<loaded_backend_model> load_sortformer_backend_model(
    const backend_model_params & params) {

    return std::make_shared<sortformer_loaded_backend_model>(
        sortformer_loaded_model::load_from_gguf(params.model_path, params.backend_name),
        params.backend_name);
}

std::unique_ptr<stream_backend> create_voxtral_backend(
    const backend_model_params & params,
    bool capture_debug) {

    return std::make_unique<voxtral_stream_backend>(
        params.model_path,
        params.backend_name,
        capture_debug);
}

std::shared_ptr<loaded_backend_model> load_voxtral_backend_model(
    const backend_model_params & params) {

    return std::make_shared<voxtral_loaded_backend_model>(
        voxtral_loaded_model::load_from_gguf(params.model_path, params.backend_name),
        params.backend_name);
}

bool resolve_voxtral_runtime(
    const backend_model_params & params,
    uint32_t requested_sample_rate_hz,
    uint32_t requested_audio_ring_capacity_samples,
    backend_model_params & out_model_params,
    stream_session_config & out_session_config,
    std::string & error) {

    constexpr const char * k_default_backend_name = "Vulkan0";
    constexpr uint32_t k_default_sample_rate_hz = VOXTRAL_SAMPLE_RATE;
    constexpr uint32_t k_default_audio_ring_capacity_samples = VOXTRAL_SAMPLE_RATE * 120u;

    if (params.model_path.empty()) {
        error = "model_path is empty";
        return false;
    }

    out_model_params = params;
    if (out_model_params.backend_name.empty()) {
        out_model_params.backend_name = k_default_backend_name;
    }

    out_session_config.expected_sample_rate_hz =
        requested_sample_rate_hz == 0 ? k_default_sample_rate_hz : requested_sample_rate_hz;
    out_session_config.audio_ring_capacity_samples =
        requested_audio_ring_capacity_samples == 0
            ? k_default_audio_ring_capacity_samples
            : requested_audio_ring_capacity_samples;
    out_session_config.continuous_mode = true;
    error.clear();
    return true;
}

bool resolve_sortformer_runtime(
    const backend_model_params & params,
    uint32_t requested_sample_rate_hz,
    uint32_t requested_audio_ring_capacity_samples,
    backend_model_params & out_model_params,
    stream_session_config & out_session_config,
    std::string & error) {

    constexpr const char * k_default_backend_name = "Vulkan0";
    constexpr uint32_t k_default_sample_rate_hz = 16000u;
    constexpr uint32_t k_default_audio_ring_capacity_samples = 16000u * 120u;

    if (params.model_path.empty()) {
        error = "model_path is empty";
        return false;
    }

    out_model_params = params;
    if (out_model_params.backend_name.empty()) {
        out_model_params.backend_name = k_default_backend_name;
    }

    out_session_config.expected_sample_rate_hz =
        requested_sample_rate_hz == 0 ? k_default_sample_rate_hz : requested_sample_rate_hz;
    out_session_config.audio_ring_capacity_samples =
        requested_audio_ring_capacity_samples == 0
            ? k_default_audio_ring_capacity_samples
            : requested_audio_ring_capacity_samples;
    out_session_config.continuous_mode = true;
    error.clear();
    return true;
}

struct backend_registry_entry {
    backend_descriptor descriptor;
    create_backend_fn create_backend = nullptr;
    load_model_fn load_model = nullptr;
    resolve_runtime_fn resolve_runtime = nullptr;
};

const std::array<backend_registry_entry, 2> & backend_registry(void) {
    static const std::array<backend_registry_entry, 2> registry = {{
        {
            { backend_kind::sortformer, "sortformer", "Vulkan0", true, false, true, 16000u, 16000u * 120u, 1u },
            &create_sortformer_backend,
            &load_sortformer_backend_model,
            &resolve_sortformer_runtime,
        },
        {
            { backend_kind::voxtral_realtime, "voxtral_realtime", "Vulkan0", true, true, false, VOXTRAL_SAMPLE_RATE, VOXTRAL_SAMPLE_RATE * 120u, 1u },
            &create_voxtral_backend,
            &load_voxtral_backend_model,
            &resolve_voxtral_runtime,
        },
    }};
    return registry;
}

const backend_registry_entry * find_backend_entry(backend_kind kind) {
    const auto & registry = backend_registry();
    for (const auto & entry : registry) {
        if (entry.descriptor.kind == kind) {
            return &entry;
        }
    }
    return nullptr;
}

const backend_registry_entry * find_backend_entry(const std::string & name) {
    const auto & registry = backend_registry();
    for (const auto & entry : registry) {
        if (entry.descriptor.name != nullptr && name == entry.descriptor.name) {
            return &entry;
        }
    }
    return nullptr;
}

bool resolve_backend_kind_internal(
    const backend_model_params & params,
    backend_kind & out_kind,
    std::string & error) {

    if (params.kind != backend_kind::unknown) {
        out_kind = params.kind;
        error.clear();
        return true;
    }

    if (!detect_backend_kind_from_model_path(params.model_path, out_kind, error)) {
        return false;
    }

    error.clear();
    return true;
}

} // namespace

size_t backend_descriptor_count(void) {
    return backend_registry().size();
}

const backend_descriptor * backend_descriptor_at(size_t index) {
    const auto & registry = backend_registry();
    if (index >= registry.size()) {
        return nullptr;
    }
    return &registry[index].descriptor;
}

const backend_descriptor * find_backend_descriptor(backend_kind kind) {
    const auto * entry = find_backend_entry(kind);
    return entry != nullptr ? &entry->descriptor : nullptr;
}

const backend_descriptor * find_backend_descriptor(const std::string & name) {
    const auto * entry = find_backend_entry(name);
    return entry != nullptr ? &entry->descriptor : nullptr;
}

const char * backend_kind_name(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr ? descriptor->name : nullptr;
}

bool parse_backend_kind_name(const std::string & name, backend_kind & out_kind) {
    const auto * descriptor = find_backend_descriptor(name);
    if (descriptor == nullptr) {
        return false;
    }
    out_kind = descriptor->kind;
    return true;
}

bool detect_backend_kind_from_model_path(const std::string & model_path, backend_kind & out_kind, std::string & error) {
    if (model_path.empty()) {
        error = "model_path is empty";
        return false;
    }

    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (ctx == nullptr) {
        error = "failed to open GGUF model: " + model_path;
        return false;
    }

    struct gguf_context_guard {
        gguf_context * ctx = nullptr;
        ~gguf_context_guard() {
            if (ctx != nullptr) {
                gguf_free(ctx);
            }
        }
    } guard{ctx};

    const int64_t key_id = gguf_find_key(ctx, "general.architecture");
    if (key_id < 0) {
        error = "missing GGUF key general.architecture";
        return false;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        error = "unexpected GGUF type for general.architecture";
        return false;
    }

    const std::string architecture = gguf_get_val_str(ctx, key_id);
    const auto * descriptor = find_backend_descriptor(architecture);
    if (descriptor == nullptr) {
        error = "unsupported realtime backend architecture: " + architecture;
        return false;
    }

    out_kind = descriptor->kind;
    error.clear();
    return true;
}

bool backend_supports_model_preload(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr && descriptor->supports_model_preload;
}

bool backend_emits_transcript(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr && descriptor->emits_transcript;
}

bool backend_emits_speaker_spans(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr && descriptor->emits_speaker_spans;
}

bool backend_info(backend_kind kind, backend_descriptor & out_info) {
    const auto * descriptor = find_backend_descriptor(kind);
    if (descriptor == nullptr) {
        out_info = {};
        return false;
    }
    out_info = *descriptor;
    return true;
}

const char * backend_default_runtime_backend_name(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr ? descriptor->default_runtime_backend_name : nullptr;
}

uint32_t backend_default_sample_rate_hz(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr ? descriptor->default_sample_rate_hz : 0u;
}

uint32_t backend_default_audio_ring_capacity_samples(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr ? descriptor->default_audio_ring_capacity_samples : 0u;
}

uint32_t backend_required_input_channels(backend_kind kind) {
    const auto * descriptor = find_backend_descriptor(kind);
    return descriptor != nullptr ? descriptor->required_input_channels : 0u;
}

bool backend_resolve_runtime(
    const backend_model_params & params,
    uint32_t requested_sample_rate_hz,
    uint32_t requested_audio_ring_capacity_samples,
    backend_model_params & out_model_params,
    stream_session_config & out_session_config,
    std::string & error) {

    backend_model_params resolved_params = params;
    if (!resolve_backend_kind_internal(params, resolved_params.kind, error)) {
        return false;
    }

    const auto * entry = find_backend_entry(resolved_params.kind);
    if (entry == nullptr || entry->resolve_runtime == nullptr) {
        error = "unsupported realtime backend kind";
        return false;
    }
    return entry->resolve_runtime(
        resolved_params,
        requested_sample_rate_hz,
        requested_audio_ring_capacity_samples,
        out_model_params,
        out_session_config,
        error);
}

std::unique_ptr<stream_backend> create_backend(
    const backend_model_params & params,
    bool capture_debug) {

    backend_model_params resolved_params = params;
    std::string error;
    if (!resolve_backend_kind_internal(params, resolved_params.kind, error)) {
        throw std::invalid_argument(error);
    }

    const auto * entry = find_backend_entry(resolved_params.kind);
    if (entry == nullptr || entry->create_backend == nullptr) {
        throw std::invalid_argument("unsupported realtime backend kind");
    }
    return entry->create_backend(resolved_params, capture_debug);
}

std::shared_ptr<loaded_backend_model> load_backend_model(
    const backend_model_params & params) {

    backend_model_params resolved_params = params;
    std::string error;
    if (!resolve_backend_kind_internal(params, resolved_params.kind, error)) {
        throw std::invalid_argument(error);
    }

    const auto * entry = find_backend_entry(resolved_params.kind);
    if (entry == nullptr || entry->load_model == nullptr) {
        throw std::invalid_argument("unsupported realtime backend kind");
    }
    return entry->load_model(resolved_params);
}

} // namespace llama::realtime
