#pragma once

#include <cstdint>
#include <string>

namespace llama::realtime {

enum class event_type {
    backend_status,
    transcript_commit,
    transcript_piece_commit,
    transcript_word_commit,
    speaker_span_commit,
    session_notice,
    backend_error,
};

enum event_flags : uint32_t {
    event_flag_none = 0,
    event_flag_preview = 1u << 0,
    event_flag_snapshot_start = 1u << 1,
    event_flag_snapshot_end = 1u << 2,
};

struct event {
    event_type type = event_type::session_notice;
    int64_t session_id = 0;
    double begin_sec = 0.0;
    double end_sec = 0.0;
    int32_t speaker_id = -1;
    std::string text;
    std::string detail;
    uint32_t flags = event_flag_none;
};

} // namespace llama::realtime
