#include "stream-manager.h"

#include <stdexcept>

namespace llama::realtime {

int64_t stream_manager::create_session(std::unique_ptr<stream_backend> backend, const stream_session_config & config) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t session_id = next_session_id_++;
    sessions_.emplace(session_id, std::make_unique<stream_session>(session_id, std::move(backend), config));
    return session_id;
}

bool stream_manager::has_session(int64_t session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.find(session_id) != sessions_.end();
}

size_t stream_manager::session_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
}

void stream_manager::push_audio(int64_t session_id, const float * samples, size_t n_samples, uint32_t sample_rate_hz) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_session_locked(session_id).push_audio(samples, n_samples, sample_rate_hz);
}

void stream_manager::flush_session(int64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_session_locked(session_id).flush();
}

std::vector<event> stream_manager::drain_events(int64_t session_id, size_t max_events) {
    std::lock_guard<std::mutex> lock(mutex_);
    return require_session_locked(session_id).drain_events(max_events);
}

void stream_manager::close_session(int64_t session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        throw std::out_of_range("stream session not found");
    }
    sessions_.erase(it);
}

stream_session & stream_manager::require_session_locked(int64_t session_id) {
    const auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        throw std::out_of_range("stream session not found");
    }
    return *it->second;
}

} // namespace llama::realtime
