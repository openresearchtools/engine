#include "stream-session.h"

#include <stdexcept>

namespace llama::realtime {

stream_session::stream_session(int64_t session_id, std::unique_ptr<stream_backend> backend, const stream_session_config & config)
    : session_id_(session_id), config_(config), backend_(std::move(backend)), audio_ring_(config.audio_ring_capacity_samples) {
    if (!backend_) {
        throw std::invalid_argument("stream_session requires a backend");
    }
}

int64_t stream_session::id() const {
    return session_id_;
}

const stream_session_config & stream_session::config() const {
    return config_;
}

backend_limits stream_session::limits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_->limits();
}

std::string stream_session::backend_name() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_->backend_name();
}

void stream_session::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    audio_ring_.clear();
    event_queue_.clear();
    samples_received_ = 0;
    backend_->reset();
}

void stream_session::push_audio(const float * samples, size_t n_samples, uint32_t sample_rate_hz) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto backend_limits = backend_->limits();
    if (sample_rate_hz != config_.expected_sample_rate_hz || sample_rate_hz != backend_limits.sample_rate_hz) {
        throw std::invalid_argument("stream_session sample rate mismatch");
    }

    audio_ring_.push(samples, n_samples);
    samples_received_ += n_samples;

    std::vector<event> events;
    backend_->push_audio(samples, n_samples, events);
    enqueue_events_locked(events);
}

void stream_session::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<event> events;
    backend_->flush(events);
    enqueue_events_locked(events);
}

size_t stream_session::queued_event_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return event_queue_.size();
}

std::vector<event> stream_session::drain_events(size_t max_events) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t n = max_events == 0 ? event_queue_.size() : std::min(max_events, event_queue_.size());
    std::vector<event> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(std::move(event_queue_.front()));
        event_queue_.pop_front();
    }
    return out;
}

uint64_t stream_session::samples_received() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_received_;
}

void stream_session::enqueue_events_locked(std::vector<event> & events) {
    for (auto & ev : events) {
        ev.session_id = session_id_;
        event_queue_.push_back(std::move(ev));
    }
    events.clear();
}

} // namespace llama::realtime
