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

struct event {
    event_type type = event_type::session_notice;
    int64_t session_id = 0;
    double begin_sec = 0.0;
    double end_sec = 0.0;
    int32_t speaker_id = -1;
    std::string text;
    std::string detail;
};

} // namespace llama::realtime
