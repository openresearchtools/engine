#pragma once

#include "stream-session.h"
#include "stream-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace llama::realtime {

enum class backend_kind : int32_t {
    unknown = 0,
    sortformer = 1,
    voxtral_realtime = 2,
};

struct backend_descriptor {
    backend_kind kind = backend_kind::sortformer;
    const char * name = nullptr;
    const char * default_runtime_backend_name = nullptr;
    bool supports_model_preload = false;
    bool emits_transcript = false;
    bool emits_speaker_spans = false;
    uint32_t default_sample_rate_hz = 0;
    uint32_t default_audio_ring_capacity_samples = 0;
    uint32_t required_input_channels = 0;
};

struct backend_model_params {
    backend_kind kind = backend_kind::sortformer;
    std::string model_path;
    std::string backend_name;
};

class loaded_backend_model {
public:
    virtual ~loaded_backend_model() = default;

    virtual backend_kind kind() const = 0;
    virtual std::string backend_name() const = 0;
    virtual stream_session_config default_session_config() const = 0;
    virtual std::unique_ptr<stream_backend> create_backend(bool capture_debug = false) const = 0;
    virtual bool supports_offline_transcription() const {
        return false;
    }
    virtual void transcribe_audio_offline(
        const std::vector<float> & /*audio*/,
        std::vector<event> & /*out_events*/,
        bool /*capture_debug*/ = false) const;
};

size_t backend_descriptor_count(void);
const backend_descriptor * backend_descriptor_at(size_t index);
const backend_descriptor * find_backend_descriptor(backend_kind kind);
const backend_descriptor * find_backend_descriptor(const std::string & name);
const char * backend_kind_name(backend_kind kind);
bool parse_backend_kind_name(const std::string & name, backend_kind & out_kind);
bool detect_backend_kind_from_model_path(const std::string & model_path, backend_kind & out_kind, std::string & error);
bool backend_supports_model_preload(backend_kind kind);
bool backend_emits_transcript(backend_kind kind);
bool backend_emits_speaker_spans(backend_kind kind);
bool backend_info(backend_kind kind, backend_descriptor & out_info);
const char * backend_default_runtime_backend_name(backend_kind kind);
uint32_t backend_default_sample_rate_hz(backend_kind kind);
uint32_t backend_default_audio_ring_capacity_samples(backend_kind kind);
uint32_t backend_required_input_channels(backend_kind kind);
bool backend_resolve_runtime(
    const backend_model_params & params,
    uint32_t requested_sample_rate_hz,
    uint32_t requested_audio_ring_capacity_samples,
    backend_model_params & out_model_params,
    stream_session_config & out_session_config,
    std::string & error);

std::unique_ptr<stream_backend> create_backend(
    const backend_model_params & params,
    bool capture_debug = false);

std::shared_ptr<loaded_backend_model> load_backend_model(
    const backend_model_params & params);

} // namespace llama::realtime
