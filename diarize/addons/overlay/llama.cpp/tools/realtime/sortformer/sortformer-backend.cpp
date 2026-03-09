#include "sortformer-backend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llama::realtime {

namespace {

event make_span_event(int32_t speaker_id, double begin_sec, double end_sec) {
    event ev;
    ev.type = event_type::speaker_span_commit;
    ev.speaker_id = speaker_id;
    ev.begin_sec = begin_sec;
    ev.end_sec = end_sec;
    return ev;
}

} // namespace

sortformer_stream_backend::sortformer_stream_backend(const std::string & gguf_path, const std::string & backend_name, bool capture_debug)
    : sortformer_stream_backend(sortformer_loaded_model::load_from_gguf(gguf_path, backend_name), capture_debug) {
}

sortformer_stream_backend::sortformer_stream_backend(std::shared_ptr<sortformer_loaded_model> loaded_model, bool capture_debug)
    : loaded_model_(std::move(loaded_model))
    , audio_frontend_(model_ref())
    , stream_state_(model_ref().metadata())
    , runtime_state_(sortformer_make_stream_runtime_state(model_ref().metadata()))
    , postprocess_params_(sortformer_default_postprocess_params(model_ref().metadata()))
    , capture_debug_(capture_debug) {
    if (!loaded_model_) {
        throw std::invalid_argument("sortformer_stream_backend requires a loaded model");
    }
    all_chunk_preds_.cols = model_ref().metadata().max_speakers;
}

std::string sortformer_stream_backend::backend_name() const {
    return std::string("sortformer/") + ggml_backend_name(model_ref().backend());
}

backend_limits sortformer_stream_backend::limits() const {
    backend_limits out;
    out.sample_rate_hz = model_ref().metadata().sample_rate_hz;
    out.preferred_push_samples = static_cast<size_t>(std::llround(model_ref().metadata().window_stride_sec * model_ref().metadata().sample_rate_hz));
    out.max_buffered_samples = static_cast<size_t>(model_ref().metadata().sample_rate_hz) * 120;
    out.emits_speaker_spans = true;
    return out;
}

void sortformer_stream_backend::reset() {
    audio_frontend_.reset();
    stream_state_.reset();
    runtime_state_ = sortformer_make_stream_runtime_state(model_ref().metadata());
    debug_steps_.clear();
    all_chunk_preds_.rows = 0;
    all_chunk_preds_.cols = model_ref().metadata().max_speakers;
    all_chunk_preds_.data.clear();
}

void sortformer_stream_backend::push_audio(const float * samples, size_t n_samples, std::vector<event> & out_events) {
    audio_frontend_.push_audio(samples, n_samples);
    stream_state_.set_available_pcm_samples(audio_frontend_.total_samples());
    stream_state_.set_available_feature_frames(audio_frontend_.available_feature_frames());
    process_ready_chunks(out_events);
}

void sortformer_stream_backend::flush(std::vector<event> & out_events) {
    audio_frontend_.flush();
    stream_state_.set_flushing(true);
    stream_state_.set_available_pcm_samples(audio_frontend_.total_samples());
    stream_state_.set_available_feature_frames(audio_frontend_.available_feature_frames());
    process_ready_chunks(out_events);
    emit_postprocessed_spans(out_events);
}

const sortformer_model & sortformer_stream_backend::model() const {
    return model_ref();
}

const sortformer_audio_frontend & sortformer_stream_backend::audio_frontend() const {
    return audio_frontend_;
}

const std::vector<sortformer_backend_step_debug> & sortformer_stream_backend::debug_steps() const {
    return debug_steps_;
}

void sortformer_stream_backend::process_ready_chunks(std::vector<event> & out_events) {
    (void) out_events;
    while (stream_state_.has_ready_chunk()) {
        const auto request = stream_state_.next_chunk();
        const uint32_t nominal_rows = static_cast<uint32_t>(request.nominal_input_end_frame - request.input_begin_frame);
        auto chunk_features = audio_frontend_.copy_feature_rows(request.input_begin_frame, nominal_rows);

        std::lock_guard<std::mutex> model_lock(loaded_model_->mutex());
        auto outputs = sortformer_streaming_update(
            model_ref(),
            chunk_features,
            static_cast<uint32_t>(request.valid_input_feature_frames),
            static_cast<uint32_t>(request.left_context_rows),
            static_cast<uint32_t>(request.right_context_rows),
            runtime_state_,
            capture_debug_);

        append_chunk_predictions(request.emit_begin_frame, outputs.chunk_preds);

        if (capture_debug_) {
            sortformer_backend_step_debug dbg;
            dbg.request = request;
            dbg.chunk_features = chunk_features;
            dbg.outputs = outputs;
            dbg.state_after = runtime_state_;
            debug_steps_.push_back(std::move(dbg));
        }

        stream_state_.mark_chunk_complete();
    }
}

void sortformer_stream_backend::append_chunk_predictions(uint64_t begin_frame, const sortformer_matrix_f32 & chunk_preds) {
    if (chunk_preds.cols != all_chunk_preds_.cols) {
        throw std::runtime_error("Sortformer chunk prediction speaker dimension mismatch");
    }

    const uint64_t needed_rows = begin_frame + chunk_preds.rows;
    if (needed_rows > all_chunk_preds_.rows) {
        const size_t old_size = all_chunk_preds_.data.size();
        all_chunk_preds_.data.resize(static_cast<size_t>(needed_rows) * all_chunk_preds_.cols, 0.0f);
        if (all_chunk_preds_.rows == 0) {
            std::fill(all_chunk_preds_.data.begin(), all_chunk_preds_.data.end(), 0.0f);
        } else if (all_chunk_preds_.data.size() > old_size) {
            std::fill(all_chunk_preds_.data.begin() + static_cast<ptrdiff_t>(old_size), all_chunk_preds_.data.end(), 0.0f);
        }
        all_chunk_preds_.rows = static_cast<uint32_t>(needed_rows);
    }

    for (uint32_t row = 0; row < chunk_preds.rows; ++row) {
        const size_t src_offset = static_cast<size_t>(row) * chunk_preds.cols;
        const size_t dst_offset = (static_cast<size_t>(begin_frame) + row) * all_chunk_preds_.cols;
        std::copy_n(chunk_preds.data.begin() + static_cast<ptrdiff_t>(src_offset), chunk_preds.cols, all_chunk_preds_.data.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
}

void sortformer_stream_backend::emit_postprocessed_spans(std::vector<event> & out_events) const {
    const auto spans = sortformer_postprocess_speaker_spans(all_chunk_preds_, postprocess_params_);
    for (const auto & span : spans) {
        out_events.push_back(make_span_event(span.speaker_id, span.begin_sec, span.end_sec));
    }
}

const sortformer_model & sortformer_stream_backend::model_ref() const {
    return loaded_model_->model();
}

} // namespace llama::realtime
