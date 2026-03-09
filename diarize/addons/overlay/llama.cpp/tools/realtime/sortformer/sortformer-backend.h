#pragma once

#include "../stream-backend.h"
#include "sortformer-audio.h"
#include "sortformer-model.h"
#include "sortformer-postprocess.h"
#include "sortformer-state.h"
#include "sortformer-streaming.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama::realtime {

struct sortformer_backend_step_debug {
    sortformer_chunk_request request;
    sortformer_matrix_f32 chunk_features;
    sortformer_stream_step_outputs outputs;
    sortformer_stream_runtime_state state_after;
};

class sortformer_stream_backend final : public stream_backend {
public:
    sortformer_stream_backend(const std::string & gguf_path, const std::string & backend_name, bool capture_debug = false);
    sortformer_stream_backend(std::shared_ptr<sortformer_loaded_model> loaded_model, bool capture_debug = false);

    std::string backend_name() const override;
    backend_limits limits() const override;

    void reset() override;
    void push_audio(const float * samples, size_t n_samples, std::vector<event> & out_events) override;
    void flush(std::vector<event> & out_events) override;

    const sortformer_model & model() const;
    const sortformer_audio_frontend & audio_frontend() const;
    const std::vector<sortformer_backend_step_debug> & debug_steps() const;

private:
    const sortformer_model & model_ref() const;
    void process_ready_chunks(std::vector<event> & out_events);
    void append_chunk_predictions(uint64_t begin_frame, const sortformer_matrix_f32 & chunk_preds);
    void emit_postprocessed_spans(std::vector<event> & out_events) const;

    std::shared_ptr<sortformer_loaded_model> loaded_model_;
    sortformer_audio_frontend audio_frontend_;
    sortformer_stream_state stream_state_;
    sortformer_stream_runtime_state runtime_state_;
    sortformer_postprocess_params postprocess_params_;
    sortformer_matrix_f32 all_chunk_preds_;
    bool capture_debug_ = false;
    std::vector<sortformer_backend_step_debug> debug_steps_;
};

} // namespace llama::realtime
