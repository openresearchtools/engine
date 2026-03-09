#pragma once

#include "../stream-backend.h"
#include "voxtral-runtime.h"

#include <memory>
#include <mutex>
#include <string>

namespace llama::realtime {

class voxtral_loaded_model {
public:
    static std::shared_ptr<voxtral_loaded_model> load_from_gguf(
        const std::string & gguf_path,
        const std::string & backend_name);

    ~voxtral_loaded_model();

    voxtral_model * model() const;
    const std::string & backend_name() const;
    std::mutex & mutex();

private:
    voxtral_loaded_model(
        voxtral_model * model,
        std::string backend_name);

    voxtral_model * model_ = nullptr;
    std::string backend_name_;
    std::mutex mutex_;
};

class voxtral_stream_backend final : public stream_backend {
public:
    voxtral_stream_backend(
        const std::string & gguf_path,
        const std::string & backend_name,
        bool capture_debug = false);
    voxtral_stream_backend(
        std::shared_ptr<voxtral_loaded_model> loaded_model,
        bool capture_debug = false);
    ~voxtral_stream_backend() override;

    std::string backend_name() const override;
    backend_limits limits() const override;

    void reset() override;
    void push_audio(const float * samples, size_t n_samples, std::vector<event> & out_events) override;
    void flush(std::vector<event> & out_events) override;

private:
    static voxtral_gpu_backend parse_gpu_backend_name(const std::string & backend_name);
    static std::string format_backend_name(const voxtral_context * ctx, const std::string & requested);

    std::shared_ptr<voxtral_loaded_model> loaded_model_;
    voxtral_context * ctx_ = nullptr;
    voxtral_stream * stream_ = nullptr;
    std::string backend_name_;
    bool capture_debug_ = false;
};

void voxtral_transcribe_offline_to_events(
    const std::shared_ptr<voxtral_loaded_model> & loaded_model,
    const std::vector<float> & audio,
    std::vector<event> & out_events,
    bool capture_debug = false);

} // namespace llama::realtime
