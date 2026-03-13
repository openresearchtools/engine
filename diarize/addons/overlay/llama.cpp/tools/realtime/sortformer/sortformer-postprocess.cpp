#include "sortformer-postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llama::realtime {

namespace {

using interval = std::pair<double, double>;

double round_to(double value, uint32_t precision) {
    const double scale = std::pow(10.0, static_cast<double>(precision));
    return std::round(value * scale) / scale;
}

int64_t fl2int(double value, int decimals) {
    const double scale = std::pow(10.0, static_cast<double>(decimals));
    return static_cast<int64_t>(std::llround(value * scale));
}

double int2fl(int64_t value, int decimals) {
    const double scale = std::pow(10.0, static_cast<double>(decimals));
    return round_to(static_cast<double>(value) / scale, static_cast<uint32_t>(decimals));
}

std::vector<interval> merge_overlap_segments(std::vector<interval> segments) {
    if (segments.size() <= 1) {
        return segments;
    }

    std::sort(segments.begin(), segments.end(), [](const interval & a, const interval & b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });

    std::vector<interval> merged;
    merged.reserve(segments.size());
    merged.push_back(segments[0]);
    for (size_t i = 1; i < segments.size(); ++i) {
        interval & tail = merged.back();
        if (tail.second >= segments[i].first) {
            tail.second = std::max(tail.second, segments[i].second);
        } else {
            merged.push_back(segments[i]);
        }
    }
    return merged;
}

std::vector<interval> filter_short_segments(const std::vector<interval> & segments, double threshold) {
    std::vector<interval> out;
    out.reserve(segments.size());
    for (const auto & seg : segments) {
        if (seg.second - seg.first >= threshold) {
            out.push_back(seg);
        }
    }
    return out;
}

std::vector<interval> get_gap_segments(std::vector<interval> segments) {
    if (segments.size() <= 1) {
        return {};
    }
    std::sort(segments.begin(), segments.end(), [](const interval & a, const interval & b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    std::vector<interval> gaps;
    gaps.reserve(segments.size() - 1);
    for (size_t i = 0; i + 1 < segments.size(); ++i) {
        gaps.emplace_back(segments[i].second, segments[i + 1].first);
    }
    return gaps;
}

std::vector<interval> filtering(std::vector<interval> speech_segments, const sortformer_postprocess_params & params) {
    if (speech_segments.empty()) {
        return speech_segments;
    }

    if (params.filter_speech_first) {
        if (params.min_duration_on > 0.0f) {
            speech_segments = filter_short_segments(speech_segments, params.min_duration_on);
        }
        if (params.min_duration_off > 0.0f && !speech_segments.empty()) {
            const auto non_speech_segments = get_gap_segments(speech_segments);
            std::vector<interval> short_non_speech;
            short_non_speech.reserve(non_speech_segments.size());
            for (const auto & gap : non_speech_segments) {
                if (gap.second - gap.first < params.min_duration_off) {
                    short_non_speech.push_back(gap);
                }
            }
            speech_segments.insert(speech_segments.end(), short_non_speech.begin(), short_non_speech.end());
            speech_segments = merge_overlap_segments(std::move(speech_segments));
        }
    } else {
        if (params.min_duration_off > 0.0f) {
            const auto non_speech_segments = get_gap_segments(speech_segments);
            std::vector<interval> short_non_speech;
            short_non_speech.reserve(non_speech_segments.size());
            for (const auto & gap : non_speech_segments) {
                if (gap.second - gap.first < params.min_duration_off) {
                    short_non_speech.push_back(gap);
                }
            }
            speech_segments.insert(speech_segments.end(), short_non_speech.begin(), short_non_speech.end());
            speech_segments = merge_overlap_segments(std::move(speech_segments));
        }
        if (params.min_duration_on > 0.0f) {
            speech_segments = filter_short_segments(speech_segments, params.min_duration_on);
        }
    }

    return speech_segments;
}

std::vector<interval> binarization(const std::vector<float> & sequence, const sortformer_postprocess_params & params) {
    if (sequence.empty()) {
        return {};
    }

    constexpr double frame_length_in_sec = 0.01;
    bool speech = false;
    double start = 0.0;
    size_t i = 0;
    std::vector<interval> speech_segments;

    for (i = 0; i < sequence.size(); ++i) {
        if (speech) {
            if (sequence[i] < params.offset) {
                const double seg_end = static_cast<double>(i) * frame_length_in_sec + params.pad_offset;
                const double seg_begin = std::max(0.0, start - params.pad_onset);
                if (seg_end > seg_begin) {
                    speech_segments.emplace_back(seg_begin, seg_end);
                }
                start = static_cast<double>(i) * frame_length_in_sec;
                speech = false;
            }
        } else {
            if (sequence[i] > params.onset) {
                start = static_cast<double>(i) * frame_length_in_sec;
                speech = true;
            }
        }
    }

    if (speech) {
        const double seg_begin = std::max(0.0, start - params.pad_onset);
        const double seg_end = static_cast<double>(i == 0 ? 0 : (i - 1)) * frame_length_in_sec + params.pad_offset;
        if (seg_end > seg_begin) {
            speech_segments.emplace_back(seg_begin, seg_end);
        }
    }

    return merge_overlap_segments(std::move(speech_segments));
}

std::vector<interval> ts_vad_post_processing(
    const sortformer_matrix_f32 & speaker_probs,
    uint32_t speaker_id,
    const sortformer_postprocess_params & params) {
    if (speaker_id >= speaker_probs.cols) {
        throw std::runtime_error("speaker index out of range in Sortformer postprocess");
    }

    std::vector<float> repeated;
    repeated.reserve(static_cast<size_t>(speaker_probs.rows) * params.unit_10ms_frame_count);
    for (uint32_t row = 0; row < speaker_probs.rows; ++row) {
        const float value = speaker_probs.data[static_cast<size_t>(row) * speaker_probs.cols + speaker_id];
        for (uint32_t k = 0; k < params.unit_10ms_frame_count; ++k) {
            repeated.push_back(value);
        }
    }

    auto speech_segments = binarization(repeated, params);
    speech_segments = filtering(std::move(speech_segments), params);
    return speech_segments;
}

std::vector<interval> merge_float_intervals(const std::vector<interval> & ranges, int decimals = 5, int margin = 2) {
    std::vector<std::pair<int64_t, int64_t>> ranges_int;
    ranges_int.reserve(ranges.size());
    for (const auto & x : ranges) {
        const int64_t stt = fl2int(x.first, decimals) + margin;
        const int64_t end = fl2int(x.second, decimals);
        if (stt < end) {
            ranges_int.emplace_back(stt, end);
        }
    }
    if (ranges_int.empty()) {
        return {};
    }

    std::sort(ranges_int.begin(), ranges_int.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });

    std::vector<std::pair<int64_t, int64_t>> merged_int;
    merged_int.reserve(ranges_int.size());
    merged_int.push_back(ranges_int[0]);
    for (size_t i = 1; i < ranges_int.size(); ++i) {
        auto & tail = merged_int.back();
        if (tail.second >= ranges_int[i].first) {
            tail.second = std::max(tail.second, ranges_int[i].second);
        } else {
            merged_int.push_back(ranges_int[i]);
        }
    }

    std::vector<interval> merged_float;
    merged_float.reserve(merged_int.size());
    for (const auto & x : merged_int) {
        merged_float.emplace_back(int2fl(x.first - margin, decimals), int2fl(x.second, decimals));
    }
    return merged_float;
}

} // namespace

sortformer_postprocess_params sortformer_default_postprocess_params(const sortformer_model_metadata & meta) {
    sortformer_postprocess_params params;
    params.unit_10ms_frame_count = meta.encoder_subsampling_factor;
    return params;
}

std::vector<sortformer_speaker_span> sortformer_postprocess_speaker_spans(
    const sortformer_matrix_f32 & speaker_probs,
    const sortformer_postprocess_params & params,
    double offset_sec) {
    std::vector<sortformer_speaker_span> out;
    if (speaker_probs.cols == 0) {
        return out;
    }

    for (uint32_t spk = 0; spk < speaker_probs.cols; ++spk) {
        auto segments = ts_vad_post_processing(speaker_probs, spk, params);
        for (auto & seg : segments) {
            seg.first = round_to(seg.first + offset_sec, params.round_precision);
            seg.second = round_to(seg.second + offset_sec, params.round_precision);
        }
        segments = merge_float_intervals(segments);
        for (const auto & seg : segments) {
            if (seg.second <= seg.first) {
                continue;
            }
            sortformer_speaker_span span;
            span.speaker_id = static_cast<int32_t>(spk);
            span.begin_sec = seg.first;
            span.end_sec = seg.second;
            out.push_back(span);
        }
    }

    std::sort(out.begin(), out.end(), [](const sortformer_speaker_span & a, const sortformer_speaker_span & b) {
        if (a.begin_sec != b.begin_sec) {
            return a.begin_sec < b.begin_sec;
        }
        if (a.end_sec != b.end_sec) {
            return a.end_sec < b.end_sec;
        }
        return a.speaker_id < b.speaker_id;
    });

    return out;
}

} // namespace llama::realtime
