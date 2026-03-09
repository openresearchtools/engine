#pragma once

#include "audio-ring.h"
#include "stream-backend.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace llama::realtime {

struct stream_session_config {
    uint32_t expected_sample_rate_hz = 16000;
    size_t audio_ring_capacity_samples = 16000 * 30;
    bool continuous_mode = true;
};

class stream_session {
public:
    stream_session(int64_t session_id, std::unique_ptr<stream_backend> backend, const stream_session_config & config = {});

    int64_t id() const;
    const stream_session_config & config() const;
    backend_limits limits() const;
    std::string backend_name() const;

    void reset();
    void push_audio(const float * samples, size_t n_samples, uint32_t sample_rate_hz);
    void flush();

    size_t queued_event_count() const;
    std::vector<event> drain_events(size_t max_events);

    uint64_t samples_received() const;

private:
    void enqueue_events_locked(std::vector<event> & events);

    const int64_t session_id_;
    const stream_session_config config_;
    std::unique_ptr<stream_backend> backend_;

    mutable std::mutex mutex_;
    audio_ring_buffer audio_ring_;
    std::deque<event> event_queue_;
    uint64_t samples_received_ = 0;
};

} // namespace llama::realtime
