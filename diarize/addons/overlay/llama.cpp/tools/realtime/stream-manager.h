#pragma once

#include "stream-session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace llama::realtime {

class stream_manager {
public:
    int64_t create_session(std::unique_ptr<stream_backend> backend, const stream_session_config & config = {});
    bool has_session(int64_t session_id) const;
    size_t session_count() const;

    void push_audio(int64_t session_id, const float * samples, size_t n_samples, uint32_t sample_rate_hz);
    void flush_session(int64_t session_id);
    std::vector<event> drain_events(int64_t session_id, size_t max_events = 0);
    void close_session(int64_t session_id);

private:
    stream_session & require_session_locked(int64_t session_id);

    mutable std::mutex mutex_;
    int64_t next_session_id_ = 1;
    std::unordered_map<int64_t, std::unique_ptr<stream_session>> sessions_;
};

} // namespace llama::realtime
