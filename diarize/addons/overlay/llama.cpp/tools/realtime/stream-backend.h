#pragma once

#include "stream-events.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama::realtime {

struct backend_limits {
    uint32_t sample_rate_hz = 16000;
    size_t preferred_push_samples = 1600;
    size_t max_buffered_samples = 16000 * 30;
    bool emits_transcript = false;
    bool emits_speaker_spans = false;
};

class stream_backend {
public:
    virtual ~stream_backend() = default;

    virtual std::string backend_name() const = 0;
    virtual backend_limits limits() const = 0;

    virtual void reset() = 0;
    virtual void push_audio(const float * samples, size_t n_samples, std::vector<event> & out_events) = 0;
    virtual void flush(std::vector<event> & out_events) = 0;
};

} // namespace llama::realtime
