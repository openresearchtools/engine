#include "server-context.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "common.h"
#include "download.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "base64.hpp"
#include "../realtime/backend-factory.h"
#include "../realtime/stream-manager.h"
#include "../whisper/whisper-cli-entrypoint.h"

#include <cstddef>
#include <cinttypes>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_map>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

bool read_audio_data(
        const std::string & fname,
        std::vector<float> & pcmf32,
        std::vector<std::vector<float>> & pcmf32s,
        bool stereo);

constexpr int HTTP_POLLING_SECONDS = 1;

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    // TODO: change to unique_ptrs for consistency:
    llama_context * ctx = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    common_speculative * spec = nullptr;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    llama_tokens generated_tokens;

    // idx of draft tokens in the main batch
    // non-empty if we went to evaluate draft tokens
    // ref: https://github.com/ggml-org/llama.cpp/pull/17808
    std::vector<int32_t> i_batch_dft;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size = llama_state_seq_get_size_ext(ctx, id, 0);

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx, cur->data.data(), cur_size, id, 0);
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        llama_memory_seq_rm(llama_get_memory(ctx), id, -1, -1);
        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token  sampled; // in speculative mode, this is the last accepted token
    llama_tokens drafted;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        drafted.clear();
        i_batch_dft.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_INF(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx) && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        int n_draft_max = task->params.speculative.n_max;

        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < task->params.speculative.n_min) {
            SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, task->params.speculative.n_min);
            n_draft_max = 0;
        }

        return n_draft_max;
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total
            );
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx, true);
                res["generated"] = generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        llama_memory_seq_rm(llama_get_memory(ctx), other.id,     -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;

    bool add_bos_token  = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;

    int slots_debug = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API

    bool sleeping = false;

    void destroy() {
        llama_init.reset();
        ctx = nullptr;
        model = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        // Clear any sampling context
        for (server_slot & slot : slots) {
            common_speculative_free(slot.spec);
            slot.spec = nullptr;
        }

        llama_batch_free(batch);
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(const common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;

        llama_init = common_init_from_params(params_base);

        model = llama_init->model();
        ctx   = llama_init->context();

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            SRV_INF("loading draft model '%s'\n", params_base.speculative.mparams_dft.path.c_str());

            const auto & params_spec = params_base.speculative;

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams_dft;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            std::string mmproj_backend_device;
            if (params_base.mmproj_use_gpu) {
                for (auto * dev : params_base.devices) {
                    if (dev == nullptr) {
                        continue;
                    }
                    const auto dev_type = ggml_backend_dev_type(dev);
                    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                        continue;
                    }
                    const char * dev_name = ggml_backend_dev_name(dev);
                    if (dev_name != nullptr && dev_name[0] != '\0') {
                        mmproj_backend_device = dev_name;
                    }
                    break;
                }
            }

            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.backend_device   = mmproj_backend_device.empty() ? nullptr : mmproj_backend_device.c_str();
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }

            if (params_base.speculative.type != COMMON_SPECULATIVE_TYPE_NONE) {
                params_base.speculative.type =  COMMON_SPECULATIVE_TYPE_NONE;
                SRV_WRN("%s\n", "speculative decoding is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model);

        int n_ctx_slot = llama_n_ctx_seq(ctx);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        const bool can_spec = common_speculative_is_compat(ctx);
        if (!can_spec) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot slot;

            slot.id    = i;
            slot.ctx   = ctx;
            slot.n_ctx = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            // try speculative decoding
            if (can_spec) {
                slot.spec = common_speculative_init(params_base.speculative, slot.ctx);
                if (slot.spec) {
                    if (mctx) {
                        SRV_ERR("%s\n", "speculative decoding is not supported with multimodal");
                        return false;
                    }
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                } else {
                    SLT_INF(slot, "%s", "speculative decoding context not initialized\n");
                }
            }

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();

            slots.push_back(std::move(slot));
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("slots debug = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_WRN("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_WRN("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (!params_base.model_alias.empty()) {
            // user explicitly specified model name
            model_name = params_base.model_alias;
        } else if (!params_base.model.name.empty()) {
            // use model name in registry format (for models in cache)
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(model != nullptr);
        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled (reasoning_budget == 0)
            // 2. The chat template supports it
            const bool enable_thinking = params_base.use_jinja && params_base.reasoning_budget != 0 && common_chat_templates_support_enable_thinking(chat_templates.get());
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* media_path            */ params_base.media_path,
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            // don't update the cache if the slot's context is empty
            update_cache = update_cache && tokens.size() > 0;

            // TODO: mtmd does not support prompt cache
            update_cache = update_cache && (ret->mctx == nullptr);

            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_INF(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_INF(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            slot.smpl.reset(common_sampler_init(model, task.params.sampling));

            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_logits = task.params.sampling.n_probs > 0;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0);

            // TODO: getting post/pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx, slot.id, nullptr);
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.task->n_tokens();
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index           = slot.task->index;
        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated           = slot.truncated;
        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.task->n_tokens();
        res->n_tokens_cached     = slot.prompt.n_tokens();
        res->has_new_line        = slot.has_new_line;
        res->stopping_word       = slot.stopping_word;
        res->stop                = slot.stop;
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(ctx, i);
            } else {
                embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1
                                            ? get_slot_by_id(id_slot)
                                            : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_text_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_memory_seq_rm (llama_get_memory(ctx), slot.id, n_keep            , n_keep + n_discard);
                llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_text_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            // generate draft tokens in speculative decoding mode
            // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
            //       perform the speculative drafting for all sequences at the same time in a single batch
            const int n_draft_max = slot.get_n_draft_max();
            if (n_draft_max > 0) {
                if (mctx) {
                    // we should never reach this, as speculative is automatically disabled if mmproj is loaded
                    GGML_ABORT("not supported by multimodal");
                }

                const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();

                const auto & params_spec = slot.task->params.speculative;

                llama_tokens draft = common_speculative_draft(slot.spec, params_spec, cached_text_tokens, slot.sampled);

                if (draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int) draft.size(), n_draft_max);
                    draft.resize(n_draft_max);
                }

                // add the sampled token to the batch
                slot.i_batch_dft.push_back(batch.n_tokens);
                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);
                slot.prompt.tokens.push_back(slot.sampled);

                if (slot.task->params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) draft.size(), slot.task->params.speculative.n_min);
                    // fallback to normal decoding
                    slot.i_batch = slot.i_batch_dft[0];
                    slot.drafted.clear();
                    slot.i_batch_dft.clear();
                } else {
                    // keep track of total number of drafted tokens tested
                    slot.n_draft_total += draft.size();

                    // add all drafted tokens to the batch
                    for (size_t i = 0; i < draft.size(); i++) {
                        slot.i_batch_dft.push_back(batch.n_tokens);
                        common_batch_add(batch, draft[i], slot.prompt.tokens.pos_next(), { slot.id }, true);
                        slot.prompt.tokens.push_back(draft[i]);
                    }
                    slot.drafted = std::move(draft);
                }
            } else {
                // no speculative decoding
                slot.i_batch = batch.n_tokens;

                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);

                slot.prompt.tokens.push_back(slot.sampled);

                SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                        slot.n_ctx, slot.prompt.n_tokens(), slot.truncated);
            }
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
                                            llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            // note: when n_swa == 0, the model does not use SWA, which is equivalent to a window of 1
                            const auto n_swa = std::max(1, llama_model_n_swa(model));

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, n_past - n_swa);

                            // note: disallow with mtmd contexts for now
                            //       https://github.com/ggml-org/llama.cpp/issues/17043
                            if (!mctx && n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min > pos_min_thold) {
                                    // TODO: support can be added in the future when corresponding vision models get released
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min, n_swa);

                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            return cur.pos_min < pos_min_thold;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        const size_t checkpoint_size = it->data.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                                        } else {
                                            n_past = std::min(n_past, std::max(it->pos_min + 1, it->pos_max));
                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, (float) checkpoint_size / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_min > pos_min_thold
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_min > pos_min_thold) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_swa = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, n_swa, (float) cur.data.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache     = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_INF(slot, "n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, p0, -1)) {
                        SLT_WRN(slot, "failed to truncate tokens with position >= %d - clearing the memory\n", p0);

                        slot.prompt_clear(true);

                        // there is no common part left
                        slot.n_prompt_tokens_cache = 0;
                    }

                    // check if we should process the image
                    if (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model uses SWA and we are not using `swa_full`
                    // - the model architecture is marked as recurrent or hybrid
                    //
                    // TODO: try to make this conditional on the context or the memory module, instead of the model type
                    do_checkpoint = do_checkpoint && (
                            llama_model_is_recurrent(model) ||
                            llama_model_is_hybrid(model) ||
                            (llama_model_n_swa(model) > 0 && !params_base.swa_full)
                            );

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        if (do_checkpoint && slot.task->n_tokens() - slot.prompt.n_tokens() == 64) {
                            break;
                        }
                    }

                    // SLT_INF(slot, "new slot.prompt.tokens: %s\n", slot.slot.prompt.tokens.str().c_str());

                    SLT_INF(slot, "prompt processing progress, n_tokens = %d, batch.n_tokens = %d, progress = %f\n", slot.prompt.n_tokens(), batch.n_tokens, (float) slot.prompt.n_tokens() / slot.task->n_tokens());

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        SLT_INF(slot, "prompt done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);

                        slot.init_sampler();

                        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                        // no need for empty or small checkpoints
                        do_checkpoint = do_checkpoint && (pos_min >= 0 && pos_max >= 64);

                        // no need to create checkpoints that are too close together
                        do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || pos_max > slot.prompt.checkpoints.back().pos_max + 64);

                        if (do_checkpoint) {
                            while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
                                // make room for the new checkpoint, if needed
                                const auto & cur = slot.prompt.checkpoints.front();

                                SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n",
                                        cur.pos_min, cur.pos_max, (float) cur.data.size() / 1024 / 1024);

                                slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
                            }

                            const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            auto & cur = slot.prompt.checkpoints.emplace_back(server_prompt_checkpoint{
                                /*.pos_min = */ pos_min,
                                /*.pos_max = */ pos_max,
                                /*.data    = */ std::vector<uint8_t>(checkpoint_size),
                            });

                            llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            SLT_WRN(slot, "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, size = %.3f MiB)\n",
                                    (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min, cur.pos_max, (float) cur.data.size() / 1024 / 1024);
                        }
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx, slot_batched->task->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.i_batch_dft.size() > 0) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || slot.i_batch_dft.empty()) {
                    continue;
                }

                const size_t n_draft = slot.drafted.size();

                // the accepted tokens from the speculation
                const auto ids = common_sampler_sample_and_accept_n(slot.smpl.get(), ctx, slot.i_batch_dft, slot.drafted);
                slot.i_batch_dft.clear();
                slot.drafted.clear();

                const int64_t t_current = ggml_time_us();

                slot.n_decoded += ids.size();

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // inform the speculative decoding about the number of accepted tokens
                common_speculative_accept(slot.spec, ids.size() - 1);

                // rollback to the state before sampling the draft tokens
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);

                // add accepted tokens to the prompt
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});
                slot.sampled = ids.back(); // last accepted token

                llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.prompt.n_tokens(), -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(const common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ build_info,
        /* model_name             */ impl->model_name,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model),
        /* model_n_params         */ llama_model_n_params(impl->model),
        /* model_size             */ llama_model_size(impl->model),
    };
}



static std::string srv_audio_make_run_id() {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::mt19937_64 rng(static_cast<uint64_t>(ms));
    const uint64_t rnd = rng() & 0xffffULL;
    std::ostringstream oss;
    oss << ms << "_" << std::hex << rnd;
    return oss.str();
}

struct srv_audio_temp_dir_guard {
    std::filesystem::path path;
    ~srv_audio_temp_dir_guard() {
        std::error_code ec;
        if (!path.empty()) {
            std::filesystem::remove_all(path, ec);
        }
    }
};

struct srv_audio_inproc_raw_payload {
    const raw_buffer * audio_bytes = nullptr;
    std::string audio_format;
};

static thread_local const srv_audio_inproc_raw_payload * g_srv_audio_inproc_raw_payload = nullptr;

struct srv_audio_inproc_raw_scope {
    explicit srv_audio_inproc_raw_scope(const srv_audio_inproc_raw_payload * payload)
        : prev(g_srv_audio_inproc_raw_payload) {
        g_srv_audio_inproc_raw_payload = payload;
    }

    ~srv_audio_inproc_raw_scope() {
        g_srv_audio_inproc_raw_payload = prev;
    }

    const srv_audio_inproc_raw_payload * prev = nullptr;
};

static std::string srv_audio_shell_quote(const std::string & s) {
    std::string out = "\"";
    out.reserve(s.size() + 2);
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

static std::string srv_audio_join_args_for_log(const std::vector<std::string> & args) {
    std::ostringstream oss;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            oss << " ";
        }
        oss << srv_audio_shell_quote(args[i]);
    }
    return oss.str();
}

static int srv_audio_run_inproc_main(
        int (*entrypoint)(int, char **),
        const std::vector<std::string> & args) {
    if (entrypoint == nullptr || args.empty()) {
        return 2;
    }

    // Use mutable C-string buffers instead of std::string storage to avoid
    // lifetime/capacity coupling if an in-process entrypoint mutates argv text.
    std::vector<std::vector<char>> argv_storage;
    argv_storage.reserve(args.size());
    for (const auto & s : args) {
        std::vector<char> buf;
        buf.reserve(s.size() + 1);
        buf.insert(buf.end(), s.begin(), s.end());
        buf.push_back('\0');
        argv_storage.push_back(std::move(buf));
    }

    const int argc = static_cast<int>(argv_storage.size());

    std::vector<char *> argv;
    argv.reserve(argv_storage.size() + 1);
    for (auto & s : argv_storage) {
        argv.push_back(s.data());
    }
    // Match C runtime argv contract: argv[argc] must be null.
    argv.push_back(nullptr);

    return entrypoint(argc, argv.data());
}

static std::string srv_audio_norm_win_path(std::string s) {
#if defined(_WIN32) || defined(WIN32)
    std::replace(s.begin(), s.end(), '/', '\\');
#endif
    return s;
}

static void srv_audio_write_binary(const std::filesystem::path & path, const raw_buffer & data) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    ofs.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

static void srv_audio_write_text(const std::filesystem::path & path, const std::string & text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    ofs << text;
}

static std::string srv_audio_read_text(const std::filesystem::path & path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static bool srv_audio_is_utf8_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

static std::string srv_audio_repair_utf8(const std::string & text, size_t * replacements) {
    if (replacements != nullptr) {
        *replacements = 0;
    }
    std::string out;
    out.reserve(text.size());

    auto append_replacement = [&](size_t * i, size_t n) {
        out.append("\xEF\xBF\xBD");
        if (replacements != nullptr) {
            (*replacements)++;
        }
        ++(*i);
        while (*i < n && srv_audio_is_utf8_continuation((unsigned char) text[*i])) {
            ++(*i);
        }
    };

    const size_t n = text.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c0 = (unsigned char) text[i];
        if (c0 <= 0x7F) {
            out.push_back((char) c0);
            ++i;
            continue;
        }

        if (c0 >= 0xC2 && c0 <= 0xDF) {
            if (i + 1 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                if (srv_audio_is_utf8_continuation(c1)) {
                    out.append(text, i, 2);
                    i += 2;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 == 0xE0) {
            if (i + 2 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                if (c1 >= 0xA0 && c1 <= 0xBF && srv_audio_is_utf8_continuation(c2)) {
                    out.append(text, i, 3);
                    i += 3;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 >= 0xE1 && c0 <= 0xEC) {
            if (i + 2 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                if (srv_audio_is_utf8_continuation(c1) && srv_audio_is_utf8_continuation(c2)) {
                    out.append(text, i, 3);
                    i += 3;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 == 0xED) {
            if (i + 2 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                if (c1 >= 0x80 && c1 <= 0x9F && srv_audio_is_utf8_continuation(c2)) {
                    out.append(text, i, 3);
                    i += 3;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 >= 0xEE && c0 <= 0xEF) {
            if (i + 2 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                if (srv_audio_is_utf8_continuation(c1) && srv_audio_is_utf8_continuation(c2)) {
                    out.append(text, i, 3);
                    i += 3;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 == 0xF0) {
            if (i + 3 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                const unsigned char c3 = (unsigned char) text[i + 3];
                if (c1 >= 0x90 && c1 <= 0xBF &&
                    srv_audio_is_utf8_continuation(c2) &&
                    srv_audio_is_utf8_continuation(c3)) {
                    out.append(text, i, 4);
                    i += 4;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 >= 0xF1 && c0 <= 0xF3) {
            if (i + 3 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                const unsigned char c3 = (unsigned char) text[i + 3];
                if (srv_audio_is_utf8_continuation(c1) &&
                    srv_audio_is_utf8_continuation(c2) &&
                    srv_audio_is_utf8_continuation(c3)) {
                    out.append(text, i, 4);
                    i += 4;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        if (c0 == 0xF4) {
            if (i + 3 < n) {
                const unsigned char c1 = (unsigned char) text[i + 1];
                const unsigned char c2 = (unsigned char) text[i + 2];
                const unsigned char c3 = (unsigned char) text[i + 3];
                if (c1 >= 0x80 && c1 <= 0x8F &&
                    srv_audio_is_utf8_continuation(c2) &&
                    srv_audio_is_utf8_continuation(c3)) {
                    out.append(text, i, 4);
                    i += 4;
                    continue;
                }
            }
            append_replacement(&i, n);
            continue;
        }

        append_replacement(&i, n);
    }

    return out;
}

struct srv_audio_json_parse_result {
    json value = json::object();
    size_t utf8_repairs = 0;
    size_t escaped_non_ascii_bytes = 0;
    bool byte_escaped_strings = false;
    bool raw_parse_ok = false;
    bool fallback_used = false;
    std::string parse_mode = "raw";
};

static void srv_audio_append_json_u00_escape(std::string & out, unsigned char c) {
    static const char * kHex = "0123456789ABCDEF";
    out.push_back('\\');
    out.push_back('u');
    out.push_back('0');
    out.push_back('0');
    out.push_back(kHex[(c >> 4) & 0x0F]);
    out.push_back(kHex[c & 0x0F]);
}

static std::string srv_audio_escape_non_ascii_json_string_bytes(const std::string & raw_json, size_t * escaped_bytes) {
    if (escaped_bytes != nullptr) {
        *escaped_bytes = 0;
    }
    std::string out;
    out.reserve(raw_json.size() + raw_json.size() / 2);

    bool in_string = false;
    bool escape_next = false;
    for (size_t i = 0; i < raw_json.size(); ++i) {
        const unsigned char c = (unsigned char) raw_json[i];
        if (!in_string) {
            out.push_back((char) c);
            if (c == '"') {
                in_string = true;
                escape_next = false;
            }
            continue;
        }

        if (escape_next) {
            out.push_back((char) c);
            escape_next = false;
            continue;
        }

        if (c == '\\') {
            out.push_back((char) c);
            escape_next = true;
            continue;
        }
        if (c == '"') {
            out.push_back((char) c);
            in_string = false;
            continue;
        }

        if (c < 0x20 || c >= 0x80) {
            srv_audio_append_json_u00_escape(out, c);
            if (escaped_bytes != nullptr) {
                (*escaped_bytes)++;
            }
            continue;
        }

        out.push_back((char) c);
    }

    return out;
}

static srv_audio_json_parse_result srv_audio_parse_json_utf8_safe(const std::string & raw_json) {
    srv_audio_json_parse_result result;

    try {
        result.raw_parse_ok = true;
        result.parse_mode = "raw";
        result.value = json::parse(raw_json);
        return result;
    } catch (const std::exception & first_error) {
        std::string escaped_error;
        size_t escaped_non_ascii_bytes = 0;
        const std::string byte_escaped_json = srv_audio_escape_non_ascii_json_string_bytes(
            raw_json,
            &escaped_non_ascii_bytes);
        if (escaped_non_ascii_bytes > 0) {
            try {
                result.value = json::parse(byte_escaped_json);
                result.byte_escaped_strings = true;
                result.escaped_non_ascii_bytes = escaped_non_ascii_bytes;
                result.fallback_used = true;
                result.parse_mode = "byte_escape";
                return result;
            } catch (const std::exception & e) {
                escaped_error = e.what();
            }
        }

        size_t repairs = 0;
        const std::string repaired = srv_audio_repair_utf8(raw_json, &repairs);
        if (repairs == 0) {
            throw;
        }
        try {
            result.value = json::parse(repaired);
            result.utf8_repairs = repairs;
            result.fallback_used = true;
            result.parse_mode = "utf8_repair";
            return result;
        } catch (const std::exception & repaired_error) {
            throw std::runtime_error(
                "JSON parse failed after fallback(s): escaped_non_ascii=" + std::to_string(escaped_non_ascii_bytes) +
                ", utf8_repairs=" + std::to_string(repairs) +
                ", escaped_error=" + escaped_error +
                ", repaired_error=" + std::string(repaired_error.what()) +
                " | original error: " + std::string(first_error.what()));
        }
    }
}

static std::filesystem::path srv_audio_find_latest_file(
        const std::filesystem::path & dir,
        const std::string & prefix,
        const std::string & suffix) {
    std::filesystem::path best_path;
    std::filesystem::file_time_type best_time;
    bool found = false;

    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return {};
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!prefix.empty() && name.rfind(prefix, 0) != 0) {
            continue;
        }
        if (!suffix.empty()) {
            if (name.size() < suffix.size() || name.substr(name.size() - suffix.size()) != suffix) {
                continue;
            }
        }
        const auto mtime = entry.last_write_time();
        if (!found || mtime > best_time) {
            best_time = mtime;
            best_path = entry.path();
            found = true;
        }
    }

    return best_path;
}

static std::string srv_audio_collapse_ws(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = false;
    for (char c : s) {
        const bool is_space = c == ' ' || c == '\n' || c == '\r' || c == '\t';
        if (is_space) {
            if (!prev_space) {
                out.push_back(' ');
            }
            prev_space = true;
        } else {
            out.push_back(c);
            prev_space = false;
        }
    }
    while (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

static std::vector<std::string> srv_audio_split_ws(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        const bool is_space = c == ' ' || c == '\n' || c == '\r' || c == '\t';
        if (is_space) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

static void srv_audio_replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::string srv_audio_join_words(const std::vector<std::string> & tokens) {
    if (tokens.empty()) {
        return "";
    }
    std::string text;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            text.push_back(' ');
        }
        text += tokens[i];
    }
    for (const char * punct : {".", ",", "?", "!", ";", ":", "%"}) {
        srv_audio_replace_all(text, std::string(" ") + punct, punct);
    }
    srv_audio_replace_all(text, " n't", "n't");
    srv_audio_replace_all(text, " '", "'");
    return srv_audio_collapse_ws(text);
}

static double srv_audio_round3(double x) {
    return std::round(x * 1000.0) / 1000.0;
}

static std::string srv_audio_format_hms(double seconds) {
    const int64_t total = (int64_t) std::llround(std::max(0.0, seconds));
    int64_t rem = total;
    const int64_t h = rem / 3600;
    rem -= h * 3600;
    const int64_t m = rem / 60;
    rem -= m * 60;
    const int64_t s = rem;
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << h << ":"
        << std::setw(2) << m << ":"
        << std::setw(2) << s;
    return oss.str();
}

static std::string srv_audio_sanitize_word_token(std::string token_text) {
    token_text = srv_audio_collapse_ws(token_text);
    if (token_text.rfind("<|", 0) == 0 && token_text.find("|>") != std::string::npos) {
        return "";
    }
    if (token_text.size() >= 3 &&
        token_text.front() == '[' &&
        token_text[1] == '_' &&
        token_text.back() == ']') {
        return "";
    }

    srv_audio_replace_all(token_text, "<s>", "");
    srv_audio_replace_all(token_text, "</s>", "");
    srv_audio_replace_all(token_text, "[STREAMING_PAD]", "");
    srv_audio_replace_all(token_text, "[STREAMING_WORD]", "");
    srv_audio_replace_all(token_text, "---", "");
    token_text = srv_audio_collapse_ws(token_text);
    return token_text;
}

static std::string srv_audio_sanitize_transcript_text(std::string text) {
    srv_audio_replace_all(text, "<s>", "");
    srv_audio_replace_all(text, "</s>", "");
    srv_audio_replace_all(text, "[STREAMING_PAD]", "");
    srv_audio_replace_all(text, "[STREAMING_WORD]", "");
    srv_audio_replace_all(text, "---", "");
    auto tokens = srv_audio_split_ws(srv_audio_collapse_ws(text));
    std::vector<std::string> kept;
    kept.reserve(tokens.size());
    for (const auto & t : tokens) {
        if (t.empty()) {
            continue;
        }
        kept.push_back(t);
    }
    return srv_audio_join_words(kept);
}

static void srv_audio_append_distributed_words(json & words, const std::string & text, double start_sec, double end_sec) {
    auto raw = srv_audio_split_ws(text);
    std::vector<std::string> clean;
    clean.reserve(raw.size());
    for (const auto & token : raw) {
        std::string w = srv_audio_sanitize_word_token(token);
        if (!w.empty()) {
            clean.push_back(std::move(w));
        }
    }
    if (clean.empty()) {
        return;
    }

    start_sec = std::max(0.0, start_sec);
    end_sec = std::max(start_sec, end_sec);
    if (end_sec <= start_sec) {
        end_sec = start_sec + 0.001;
    }

    const double span = std::max(1e-6, end_sec - start_sec);
    const double step = span / (double) clean.size();
    for (size_t i = 0; i < clean.size(); ++i) {
        double ws = start_sec + step * (double) i;
        double we = (i + 1 < clean.size()) ? (start_sec + step * (double) (i + 1)) : end_sec;
        if (we <= ws) {
            we = ws + 0.001;
        }
        words.push_back({
            {"word", clean[i]},
            {"start_sec", srv_audio_round3(ws)},
            {"end_sec", srv_audio_round3(we)},
        });
    }
}

struct srv_audio_whisper_parse {
    std::string transcript;
    json words = json::array();
    json pieces = json::array();
};

static std::string srv_audio_join_word_range(const json & words, size_t begin, size_t end) {
    if (!words.is_array() || begin >= end || begin >= words.size()) {
        return "";
    }

    end = std::min(end, words.size());
    std::vector<std::string> toks;
    toks.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        if (!words[i].is_object()) {
            continue;
        }
        std::string tok = srv_audio_sanitize_word_token(json_value(words[i], "word", std::string()));
        if (!tok.empty()) {
            toks.push_back(std::move(tok));
        }
    }
    return srv_audio_join_words(toks);
}

static std::string srv_audio_decode_byte_escaped_raw_bytes(const std::string & text) {
    std::string raw;
    raw.reserve(text.size());

    auto append_decoded = [&](uint32_t cp, size_t pos, size_t len) {
        if (cp <= 0xFF) {
            raw.push_back((char) cp);
        } else {
            raw.append(text, pos, len);
        }
    };

    const size_t n = text.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c0 = (unsigned char) text[i];
        if (c0 <= 0x7F) {
            raw.push_back((char) c0);
            ++i;
            continue;
        }

        if (c0 >= 0xC2 && c0 <= 0xDF && i + 1 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            if (srv_audio_is_utf8_continuation(c1)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x1F) << 6) | (uint32_t) (c1 & 0x3F);
                append_decoded(cp, i, 2);
                i += 2;
                continue;
            }
        }

        if (c0 == 0xE0 && i + 2 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            if (c1 >= 0xA0 && c1 <= 0xBF && srv_audio_is_utf8_continuation(c2)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x0F) << 12) |
                                    ((uint32_t) (c1 & 0x3F) << 6) |
                                    (uint32_t) (c2 & 0x3F);
                append_decoded(cp, i, 3);
                i += 3;
                continue;
            }
        }
        if (c0 >= 0xE1 && c0 <= 0xEC && i + 2 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            if (srv_audio_is_utf8_continuation(c1) && srv_audio_is_utf8_continuation(c2)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x0F) << 12) |
                                    ((uint32_t) (c1 & 0x3F) << 6) |
                                    (uint32_t) (c2 & 0x3F);
                append_decoded(cp, i, 3);
                i += 3;
                continue;
            }
        }
        if (c0 == 0xED && i + 2 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            if (c1 >= 0x80 && c1 <= 0x9F && srv_audio_is_utf8_continuation(c2)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x0F) << 12) |
                                    ((uint32_t) (c1 & 0x3F) << 6) |
                                    (uint32_t) (c2 & 0x3F);
                append_decoded(cp, i, 3);
                i += 3;
                continue;
            }
        }
        if (c0 >= 0xEE && c0 <= 0xEF && i + 2 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            if (srv_audio_is_utf8_continuation(c1) && srv_audio_is_utf8_continuation(c2)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x0F) << 12) |
                                    ((uint32_t) (c1 & 0x3F) << 6) |
                                    (uint32_t) (c2 & 0x3F);
                append_decoded(cp, i, 3);
                i += 3;
                continue;
            }
        }

        if (c0 == 0xF0 && i + 3 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            const unsigned char c3 = (unsigned char) text[i + 3];
            if (c1 >= 0x90 && c1 <= 0xBF &&
                srv_audio_is_utf8_continuation(c2) &&
                srv_audio_is_utf8_continuation(c3)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x07) << 18) |
                                    ((uint32_t) (c1 & 0x3F) << 12) |
                                    ((uint32_t) (c2 & 0x3F) << 6) |
                                    (uint32_t) (c3 & 0x3F);
                append_decoded(cp, i, 4);
                i += 4;
                continue;
            }
        }
        if (c0 >= 0xF1 && c0 <= 0xF3 && i + 3 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            const unsigned char c3 = (unsigned char) text[i + 3];
            if (srv_audio_is_utf8_continuation(c1) &&
                srv_audio_is_utf8_continuation(c2) &&
                srv_audio_is_utf8_continuation(c3)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x07) << 18) |
                                    ((uint32_t) (c1 & 0x3F) << 12) |
                                    ((uint32_t) (c2 & 0x3F) << 6) |
                                    (uint32_t) (c3 & 0x3F);
                append_decoded(cp, i, 4);
                i += 4;
                continue;
            }
        }
        if (c0 == 0xF4 && i + 3 < n) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const unsigned char c2 = (unsigned char) text[i + 2];
            const unsigned char c3 = (unsigned char) text[i + 3];
            if (c1 >= 0x80 && c1 <= 0x8F &&
                srv_audio_is_utf8_continuation(c2) &&
                srv_audio_is_utf8_continuation(c3)) {
                const uint32_t cp = ((uint32_t) (c0 & 0x07) << 18) |
                                    ((uint32_t) (c1 & 0x3F) << 12) |
                                    ((uint32_t) (c2 & 0x3F) << 6) |
                                    (uint32_t) (c3 & 0x3F);
                append_decoded(cp, i, 4);
                i += 4;
                continue;
            }
        }

        raw.push_back((char) c0);
        ++i;
    }

    return raw;
}

static std::string srv_audio_decode_byte_escaped_utf8_text(const std::string & text, size_t * utf8_repairs) {
    std::string raw = srv_audio_decode_byte_escaped_raw_bytes(text);
    size_t repairs = 0;
    std::string repaired = srv_audio_repair_utf8(raw, &repairs);
    if (utf8_repairs != nullptr) {
        *utf8_repairs += repairs;
    }
    return repaired;
}

static std::string srv_audio_decode_whisper_json_text(
    const std::string & text,
    bool byte_escaped_strings,
    size_t * utf8_repairs) {
    if (!byte_escaped_strings) {
        return text;
    }
    return srv_audio_decode_byte_escaped_utf8_text(text, utf8_repairs);
}

static srv_audio_whisper_parse srv_audio_parse_whisper_json(
        const json & whisper_json,
        bool byte_escaped_strings,
        size_t * utf8_repairs) {
    srv_audio_whisper_parse out;
    if (!whisper_json.is_object()) {
        return out;
    }

    std::vector<std::string> transcript_parts;
    const json trans = json_value(whisper_json, "transcription", json::array());
    if (!trans.is_array()) {
        return out;
    }

    for (const auto & seg : trans) {
        if (!seg.is_object()) {
            continue;
        }

        std::string seg_text = json_value(seg, "text", std::string());
        seg_text = srv_audio_decode_whisper_json_text(seg_text, byte_escaped_strings, utf8_repairs);
        seg_text = srv_audio_collapse_ws(seg_text);
        if (!seg_text.empty()) {
            transcript_parts.push_back(seg_text);
        }

        double seg_start = 0.0;
        double seg_end = 0.0;
        if (seg.contains("offsets") && seg.at("offsets").is_object()) {
            const auto & offsets = seg.at("offsets");
            seg_start = std::max(0.0, json_value(offsets, "from", 0.0) / 1000.0);
            seg_end = std::max(seg_start, json_value(offsets, "to", seg_start * 1000.0) / 1000.0);
        }

        const size_t piece_word_begin = out.words.size();
        bool added_token_words = false;
        if (seg.contains("tokens") && seg.at("tokens").is_array()) {
            auto trim_copy = [](std::string s) {
                size_t b = 0;
                while (b < s.size() && std::isspace((unsigned char) s[b])) {
                    ++b;
                }
                size_t e = s.size();
                while (e > b && std::isspace((unsigned char) s[e - 1])) {
                    --e;
                }
                return s.substr(b, e - b);
            };
            auto is_punct_only = [](const std::string & s) {
                bool has = false;
                for (char ch : s) {
                    const unsigned char c = (unsigned char) ch;
                    if (std::isspace(c)) {
                        continue;
                    }
                    has = true;
                    if (c >= 0x80) {
                        return false;
                    }
                    if (std::isalnum(c)) {
                        return false;
                    }
                }
                return has;
            };

            std::string cur_word;
            double cur_start = 0.0;
            double cur_end = 0.0;
            bool cur_has_leading_space = false;
            bool has_cur = false;

            auto flush_cur = [&]() {
                std::string word_text = cur_word;
                if (byte_escaped_strings) {
                    size_t repairs = 0;
                    word_text = srv_audio_repair_utf8(word_text, &repairs);
                    if (utf8_repairs != nullptr) {
                        *utf8_repairs += repairs;
                    }
                }
                const std::string w = srv_audio_sanitize_word_token(word_text);
                if (!w.empty()) {
                    out.words.push_back({
                        {"word", w},
                        {"has_leading_space", cur_has_leading_space},
                        {"start_sec", srv_audio_round3(cur_start)},
                        {"end_sec", srv_audio_round3(std::max(cur_start, cur_end))},
                    });
                    added_token_words = true;
                }
                cur_word.clear();
                cur_start = 0.0;
                cur_end = 0.0;
                cur_has_leading_space = false;
                has_cur = false;
            };

            for (const auto & tok : seg.at("tokens")) {
                if (!tok.is_object()) {
                    continue;
                }

                std::string tok_text = json_value(tok, "text", std::string());
                if (byte_escaped_strings) {
                    tok_text = srv_audio_decode_byte_escaped_raw_bytes(tok_text);
                } else {
                    tok_text = srv_audio_decode_whisper_json_text(tok_text, false, utf8_repairs);
                }
                if (tok_text.empty()) {
                    continue;
                }
                if (tok_text.rfind("<|", 0) == 0 && tok_text.find("|>") != std::string::npos) {
                    continue;
                }
                if (tok_text.rfind("[_", 0) == 0 && tok_text.back() == ']') {
                    continue;
                }

                double tok_start = seg_start;
                double tok_end = seg_end;
                if (tok.contains("offsets") && tok.at("offsets").is_object()) {
                    const auto & offsets = tok.at("offsets");
                    tok_start = std::max(0.0, json_value(offsets, "from", seg_start * 1000.0) / 1000.0);
                    tok_end = std::max(tok_start, json_value(offsets, "to", tok_end * 1000.0) / 1000.0);
                }

                const bool has_leading_space = !tok_text.empty() && std::isspace((unsigned char) tok_text.front());
                tok_text = trim_copy(tok_text);
                if (tok_text.empty()) {
                    continue;
                }
                tok_text = srv_audio_sanitize_word_token(tok_text);
                if (tok_text.empty()) {
                    continue;
                }

                const bool punct_only = is_punct_only(tok_text);

                if (!has_cur) {
                    cur_word = tok_text;
                    cur_start = tok_start;
                    cur_end = tok_end;
                    cur_has_leading_space = has_leading_space;
                    has_cur = true;
                    continue;
                }

                if (has_leading_space && !punct_only) {
                    flush_cur();
                    cur_word = tok_text;
                    cur_start = tok_start;
                    cur_end = tok_end;
                    cur_has_leading_space = has_leading_space;
                    has_cur = true;
                    continue;
                }

                cur_word += tok_text;
                cur_end = std::max(cur_end, tok_end);
            }

            if (has_cur) {
                flush_cur();
            }
        }

        if (!added_token_words && !seg_text.empty()) {
            srv_audio_append_distributed_words(out.words, seg_text, seg_start, seg_end);
        }

        const size_t piece_word_end = out.words.size();
        std::string piece_text = srv_audio_sanitize_transcript_text(seg_text);
        if (piece_text.empty() && piece_word_end > piece_word_begin) {
            piece_text = srv_audio_join_word_range(out.words, piece_word_begin, piece_word_end);
        }

        double piece_start = std::max(0.0, seg_start);
        double piece_end = std::max(piece_start, seg_end);
        if (piece_word_end > piece_word_begin &&
                out.words[piece_word_begin].is_object() &&
                out.words[piece_word_end - 1].is_object()) {
            piece_start = std::max(0.0, json_value(out.words[piece_word_begin], "start_sec", piece_start));
            piece_end = std::max(piece_start, json_value(out.words[piece_word_end - 1], "end_sec", piece_end));
        } else if (piece_end <= piece_start) {
            piece_end = piece_start + 0.001;
        }

        if (!piece_text.empty() || piece_word_end > piece_word_begin) {
            out.pieces.push_back(json{
                {"text", piece_text},
                {"start_sec", srv_audio_round3(piece_start)},
                {"end_sec", srv_audio_round3(std::max(piece_start, piece_end))},
                {"start_word_index", (int64_t) piece_word_begin},
                {"end_word_index", (int64_t) piece_word_end},
                {"num_words", (int64_t) (piece_word_end - piece_word_begin)},
            });
        }
    }

    out.transcript = srv_audio_sanitize_transcript_text(srv_audio_join_words(transcript_parts));
    if (out.transcript.empty() && out.words.is_array() && !out.words.empty()) {
        std::vector<std::string> words_for_text;
        words_for_text.reserve(out.words.size());
        for (const auto & w : out.words) {
            if (!w.is_object()) {
                continue;
            }
            std::string token = srv_audio_sanitize_word_token(json_value(w, "word", std::string()));
            if (!token.empty()) {
                words_for_text.push_back(std::move(token));
            }
        }
        out.transcript = srv_audio_join_words(words_for_text);
    }

    return out;
}

static void srv_audio_add_hms_fields(json & items) {
    if (!items.is_array()) {
        return;
    }
    for (auto & item : items) {
        if (!item.is_object()) {
            continue;
        }
        const double start_sec = item.value("start_sec", 0.0);
        const double end_sec = item.value("end_sec", start_sec);
        item["start_hms"] = srv_audio_format_hms(start_sec);
        item["end_hms"] = srv_audio_format_hms(end_sec);
    }
}

static std::string srv_audio_format_srt_time(double seconds) {
    const int64_t total_ms = (int64_t) std::llround(std::max(0.0, seconds) * 1000.0);
    int64_t rem = total_ms;
    const int64_t h = rem / 3600000;
    rem -= h * 3600000;
    const int64_t m = rem / 60000;
    rem -= m * 60000;
    const int64_t s = rem / 1000;
    rem -= s * 1000;
    const int64_t ms = rem;

    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << h << ":"
        << std::setw(2) << m << ":"
        << std::setw(2) << s << ","
        << std::setw(3) << ms;
    return oss.str();
}

static std::string srv_audio_segments_to_srt(const json & segments) {
    if (!segments.is_array()) {
        return "";
    }
    std::ostringstream oss;
    int idx = 1;
    for (const auto & seg : segments) {
        if (!seg.is_object()) {
            continue;
        }
        const double start_sec = seg.value("start_sec", 0.0);
        const double end_sec = std::max(start_sec, seg.value("end_sec", start_sec));
        const std::string text = srv_audio_collapse_ws(seg.value("text", std::string()));
        if (text.empty()) {
            continue;
        }
        oss << idx++ << "\n";
        oss << srv_audio_format_srt_time(start_sec) << " --> " << srv_audio_format_srt_time(end_sec) << "\n";
        oss << text << "\n\n";
    }
    return oss.str();
}

static std::string srv_audio_transcript_to_md(const std::string & transcript) {
    return srv_audio_collapse_ws(transcript) + "\n";
}

static std::string srv_audio_speaker_segments_to_md(const json & speaker_segments) {
    std::ostringstream oss;
    if (!speaker_segments.is_array() || speaker_segments.empty()) {
        return "";
    }
    for (const auto & seg : speaker_segments) {
        if (!seg.is_object()) {
            continue;
        }
        const std::string speaker = seg.value("speaker", std::string("SPEAKER_00"));
        const double start_sec = seg.value("start_sec", 0.0);
        const double end_sec = std::max(start_sec, seg.value("end_sec", start_sec));
        const std::string start_hms = seg.contains("start_hms")
            ? seg.value("start_hms", srv_audio_format_hms(start_sec))
            : srv_audio_format_hms(start_sec);
        const std::string end_hms = seg.contains("end_hms")
            ? seg.value("end_hms", srv_audio_format_hms(end_sec))
            : srv_audio_format_hms(end_sec);
        const std::string text = srv_audio_collapse_ws(seg.value("text", std::string()));
        if (text.empty()) {
            continue;
        }
        oss << "### " << speaker << " [" << start_hms << " - " << end_hms << "]\n";
        oss << text << "\n\n";
    }
    return oss.str();
}

static bool srv_audio_word_ends_sentence(const std::string & word);
static json srv_audio_build_fallback_whisper_pieces(const json & words);

static std::string srv_audio_realtime_speaker_label(int32_t speaker_id) {
    std::ostringstream oss;
    oss << "SPEAKER_" << std::setfill('0') << std::setw(2) << std::max<int32_t>(0, speaker_id);
    return oss.str();
}

static bool srv_audio_resolve_realtime_model_path(
        const json & body,
        std::string & out_model_path,
        std::string & out_error) {
    const std::string explicit_model_path = body.value("diarization_model_path", std::string());
    const std::string models_dir = body.value(
        "diarization_models_dir",
        body.value("diarization_model_dir", std::string()));
    const std::string hf_repo = body.value("diarization_hf_repo", std::string());

    if (!hf_repo.empty()) {
        out_error = "diarization_hf_repo is no longer supported for diarization. Provide diarization_model_path or diarization_models_dir.";
        return false;
    }
    if (!explicit_model_path.empty() && !models_dir.empty()) {
        out_error = "Choose one diarization model source: diarization_model_path OR diarization_models_dir.";
        return false;
    }
    if (explicit_model_path.empty() && models_dir.empty()) {
        out_error = "Missing diarization model source. Provide diarization_model_path or diarization_models_dir.";
        return false;
    }

    std::filesystem::path resolved_path;
    if (!explicit_model_path.empty()) {
        resolved_path = std::filesystem::path(explicit_model_path);
    } else {
        const std::filesystem::path dir_path(models_dir);
        if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
            out_error = "diarization_models_dir is not a directory: " + models_dir;
            return false;
        }

        const std::vector<std::filesystem::path> preferred = {
            dir_path / "model.gguf",
            dir_path / "sortformer.gguf",
            dir_path / "diarization.gguf",
        };
        for (const auto & candidate : preferred) {
            if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate)) {
                resolved_path = candidate;
                break;
            }
        }

        if (resolved_path.empty()) {
            std::vector<std::filesystem::path> ggufs;
            for (const auto & entry : std::filesystem::directory_iterator(dir_path)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                if (entry.path().extension() == ".gguf") {
                    ggufs.push_back(entry.path());
                }
            }
            std::sort(ggufs.begin(), ggufs.end(), [](const auto & a, const auto & b) {
                return a.filename().string() < b.filename().string();
            });
            if (!ggufs.empty()) {
                resolved_path = ggufs.front();
            }
        }
    }

    if (resolved_path.empty() || !std::filesystem::exists(resolved_path) || !std::filesystem::is_regular_file(resolved_path)) {
        out_error = "Realtime diarization GGUF was not found";
        return false;
    }

    out_model_path = srv_audio_norm_win_path(resolved_path.string());
    out_error.clear();
    return true;
}

static bool srv_audio_resolve_realtime_backend_kind(
        const std::string & requested_backend,
        const std::string & model_path,
        llama::realtime::backend_kind & out_kind,
        std::string & out_error) {
    auto trim_copy = [](std::string s) {
        size_t b = 0;
        while (b < s.size() && std::isspace((unsigned char) s[b])) {
            ++b;
        }
        size_t e = s.size();
        while (e > b && std::isspace((unsigned char) s[e - 1])) {
            --e;
        }
        return s.substr(b, e - b);
    };
    const std::string requested = trim_copy(requested_backend);
    if (requested.empty() || requested == "auto" || requested == "native_cpp" || requested == "realtime") {
        if (!llama::realtime::detect_backend_kind_from_model_path(model_path, out_kind, out_error)) {
            return false;
        }
        return true;
    }

    if (!llama::realtime::parse_backend_kind_name(requested, out_kind)) {
        out_error = "Unsupported diarization backend: " + requested;
        return false;
    }

    out_error.clear();
    return true;
}

static bool srv_audio_collect_realtime_speaker_spans(
        const std::filesystem::path & audio_path,
        const std::string & model_path,
        const std::string & backend_requested,
        const std::string & runtime_backend_name,
        double feed_ms,
        json & out_spans,
        double & out_elapsed_sec,
        std::string & out_backend_name,
        std::string & out_error) {
    out_spans = json::array();
    out_elapsed_sec = 0.0;
    out_backend_name.clear();
    out_error.clear();

    if (feed_ms <= 0.0 || !std::isfinite(feed_ms)) {
        out_error = "diarization_feed_ms must be a positive finite number";
        return false;
    }

    llama::realtime::backend_kind kind = llama::realtime::backend_kind::unknown;
    if (!srv_audio_resolve_realtime_backend_kind(backend_requested, model_path, kind, out_error)) {
        return false;
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(audio_path.string(), pcmf32, pcmf32s, /* stereo */ false)) {
        out_error = "failed to decode diarization audio input";
        return false;
    }

    llama::realtime::backend_model_params backend_params = {};
    backend_params.kind = kind;
    backend_params.model_path = model_path;
    backend_params.backend_name = runtime_backend_name;

    llama::realtime::backend_model_params resolved_backend_params = {};
    llama::realtime::stream_session_config session_config = {};
    if (!llama::realtime::backend_resolve_runtime(
            backend_params,
            /* requested_sample_rate_hz */ 16000,
            /* requested_audio_ring_capacity_samples */ 0,
            resolved_backend_params,
            session_config,
            out_error)) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        auto loaded_model = llama::realtime::load_backend_model(resolved_backend_params);
        if (!loaded_model) {
            out_error = "failed to load realtime diarization backend model";
            return false;
        }

        out_backend_name = loaded_model->backend_name();

        llama::realtime::stream_manager manager;
        const auto session_id = manager.create_session(loaded_model->create_backend(), session_config);
        const size_t push_samples = std::max<size_t>(
            1,
            (size_t) std::llround((feed_ms / 1000.0) * (double) session_config.expected_sample_rate_hz));

        for (size_t offset = 0; offset < pcmf32.size(); offset += push_samples) {
            const size_t count = std::min(push_samples, pcmf32.size() - offset);
            manager.push_audio(session_id, pcmf32.data() + static_cast<ptrdiff_t>(offset), count, session_config.expected_sample_rate_hz);
        }
        manager.flush_session(session_id);
        const auto events = manager.drain_events(session_id, 0);
        manager.close_session(session_id);

        for (const auto & ev : events) {
            if (ev.type != llama::realtime::event_type::speaker_span_commit) {
                continue;
            }
            out_spans.push_back(json{
                {"speaker_id", ev.speaker_id},
                {"speaker", srv_audio_realtime_speaker_label(ev.speaker_id)},
                {"start_sec", srv_audio_round3(ev.begin_sec)},
                {"end_sec", srv_audio_round3(std::max(ev.begin_sec, ev.end_sec))},
            });
        }
        std::sort(out_spans.begin(), out_spans.end(), [](const auto & a, const auto & b) {
            const double a_start = a.value("start_sec", 0.0);
            const double b_start = b.value("start_sec", 0.0);
            if (a_start != b_start) {
                return a_start < b_start;
            }
            const double a_end = a.value("end_sec", a_start);
            const double b_end = b.value("end_sec", b_start);
            if (a_end != b_end) {
                return a_end < b_end;
            }
            return a.value("speaker_id", -1) < b.value("speaker_id", -1);
        });
    } catch (const std::exception & e) {
        out_error = e.what();
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    out_elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    return true;
}

static std::string srv_audio_assign_word_to_realtime_speaker(
        const json & word,
        const json & speaker_spans) {
    if (!word.is_object() || !speaker_spans.is_array() || speaker_spans.empty()) {
        return "UNASSIGNED";
    }

    const double start_sec = word.value("start_sec", 0.0);
    const double end_sec = std::max(start_sec, word.value("end_sec", start_sec));
    const double anchor_sec = 0.5 * (start_sec + end_sec);

    const json * best_span = nullptr;
    double best_score = -1.0;
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));
        if (anchor_sec >= span_start && anchor_sec <= span_end) {
            return span.value("speaker", std::string("UNASSIGNED"));
        }
        const double overlap = std::max(0.0, std::min(end_sec, span_end) - std::max(start_sec, span_start));
        if (overlap > best_score) {
            best_score = overlap;
            best_span = &span;
        }
    }
    if (best_span != nullptr && best_score > 1e-9) {
        return best_span->value("speaker", std::string("UNASSIGNED"));
    }

    constexpr double k_nearest_assign_tolerance_sec = 0.08;
    const json * nearest_span = nullptr;
    double nearest_dist = std::numeric_limits<double>::infinity();
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));
        double dist = 0.0;
        if (anchor_sec < span_start) {
            dist = span_start - anchor_sec;
        } else if (anchor_sec > span_end) {
            dist = anchor_sec - span_end;
        }
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_span = &span;
        }
    }

    if (nearest_span != nullptr && nearest_dist <= k_nearest_assign_tolerance_sec) {
        return nearest_span->value("speaker", std::string("UNASSIGNED"));
    }

    return "UNASSIGNED";
}

static bool srv_audio_word_starts_with_lower_or_continuation(const std::string & word) {
    for (char ch : word) {
        const unsigned char c = (unsigned char) ch;
        if (std::isspace(c)) {
            continue;
        }
        if (ch == '"' || ch == '\'' || ch == '(' || ch == '[' || ch == '{') {
            continue;
        }
        if (ch == ',' || ch == ';' || ch == ':' || ch == ')' || ch == ']' || ch == '}' || ch == '-') {
            return true;
        }
        if (std::isalpha(c)) {
            return std::islower(c) != 0;
        }
        return false;
    }
    return false;
}

static bool srv_audio_word_starts_with_sentence_start(const std::string & word) {
    for (char ch : word) {
        const unsigned char c = (unsigned char) ch;
        if (std::isspace(c)) {
            continue;
        }
        if (ch == '"' || ch == '\'' || ch == '(' || ch == '[' || ch == '{') {
            continue;
        }
        if (std::isalpha(c)) {
            return std::isupper(c) != 0;
        }
        if (std::isdigit(c)) {
            return true;
        }
        return false;
    }
    return false;
}

static void srv_audio_smooth_realtime_sentence_speakers(std::vector<json> & assigned_words) {
    if (assigned_words.empty()) {
        return;
    }

    struct speaker_run {
        size_t begin = 0;
        size_t end = 0;
        std::string speaker;
        double duration = 0.0;
    };

    auto reassign_range = [&](size_t begin, size_t end, const std::string & speaker) {
        for (size_t i = begin; i < end; ++i) {
            assigned_words[i]["speaker"] = speaker;
        }
    };

    auto process_sentence = [&](size_t sentence_begin, size_t sentence_end) {
        if (sentence_begin >= sentence_end) {
            return;
        }

        std::vector<speaker_run> runs;
        for (size_t i = sentence_begin; i < sentence_end;) {
            const std::string speaker = assigned_words[i].value("speaker", std::string("UNASSIGNED"));
            size_t j = i + 1;
            while (j < sentence_end &&
                   assigned_words[j].value("speaker", std::string("UNASSIGNED")) == speaker) {
                ++j;
            }

            double duration = 0.0;
            for (size_t k = i; k < j; ++k) {
                const double ws = assigned_words[k].value("start_sec", 0.0);
                const double we = std::max(ws, assigned_words[k].value("end_sec", ws));
                duration += std::max(0.001, we - ws);
            }

            runs.push_back(speaker_run{ i, j, speaker, duration });
            i = j;
        }

        for (size_t idx = 1; idx < runs.size(); ++idx) {
            auto & run = runs[idx];
            auto & prev = runs[idx - 1];
            if (run.speaker == prev.speaker) {
                continue;
            }
            const size_t run_words = run.end - run.begin;
            if (run_words > 4 || run.duration > 1.25) {
                continue;
            }
            const std::string first_word = assigned_words[run.begin].value("word", std::string());
            if (!srv_audio_word_starts_with_lower_or_continuation(first_word)) {
                continue;
            }
            reassign_range(run.begin, run.end, prev.speaker);
            run.speaker = prev.speaker;
        }

        runs.clear();
        for (size_t i = sentence_begin; i < sentence_end;) {
            const std::string speaker = assigned_words[i].value("speaker", std::string("UNASSIGNED"));
            size_t j = i + 1;
            while (j < sentence_end &&
                   assigned_words[j].value("speaker", std::string("UNASSIGNED")) == speaker) {
                ++j;
            }

            double duration = 0.0;
            for (size_t k = i; k < j; ++k) {
                const double ws = assigned_words[k].value("start_sec", 0.0);
                const double we = std::max(ws, assigned_words[k].value("end_sec", ws));
                duration += std::max(0.001, we - ws);
            }

            runs.push_back(speaker_run{ i, j, speaker, duration });
            i = j;
        }

        for (size_t idx = 0; idx + 1 < runs.size(); ++idx) {
            auto & run = runs[idx];
            auto & next = runs[idx + 1];
            if (run.speaker == next.speaker) {
                continue;
            }
            const size_t run_words = run.end - run.begin;
            if (run_words > 4 || run.duration > 1.25) {
                continue;
            }
            const size_t next_words = next.end - next.begin;
            if (next_words == 0) {
                continue;
            }
            const std::string first_word = assigned_words[run.begin].value("word", std::string());
            bool sentence_start = run.begin == sentence_begin;
            if (!sentence_start && !srv_audio_word_ends_sentence(assigned_words[run.begin - 1].value("word", std::string()))) {
                continue;
            }
            if (!srv_audio_word_starts_with_sentence_start(first_word)) {
                continue;
            }
            if (next_words < run_words + 2 && next.duration < run.duration + 0.4) {
                continue;
            }
            reassign_range(run.begin, run.end, next.speaker);
            run.speaker = next.speaker;
        }

        std::unordered_map<std::string, double> weights;
        size_t assigned_words_count = 0;
        for (size_t i = sentence_begin; i < sentence_end; ++i) {
            const std::string speaker = assigned_words[i].value("speaker", std::string("UNASSIGNED"));
            if (speaker == "UNASSIGNED") {
                continue;
            }
            const double ws = assigned_words[i].value("start_sec", 0.0);
            const double we = std::max(ws, assigned_words[i].value("end_sec", ws));
            weights[speaker] += std::max(0.001, we - ws);
            ++assigned_words_count;
        }

        if (weights.empty()) {
            return;
        }

        std::string best_speaker = "UNASSIGNED";
        double best_weight = -1.0;
        double second_weight = -1.0;
        for (const auto & it : weights) {
            if (it.second > best_weight) {
                second_weight = best_weight;
                best_weight = it.second;
                best_speaker = it.first;
            } else if (it.second > second_weight) {
                second_weight = it.second;
            }
        }

        const double total_weight = std::max(1e-9, [&]() {
            double total = 0.0;
            for (const auto & it : weights) {
                total += it.second;
            }
            return total;
        }());

        const bool single_speaker = weights.size() == 1;
        const bool clear_majority =
            best_weight / total_weight >= 0.60 ||
            (best_weight >= 2.0 * std::max(0.0, second_weight)) ||
            (assigned_words_count <= 4 && best_weight / total_weight >= 0.50);

        if (single_speaker || clear_majority) {
            reassign_range(sentence_begin, sentence_end, best_speaker);
        }
    };

    size_t sentence_begin = 0;
    for (size_t i = 0; i < assigned_words.size(); ++i) {
        const std::string word = assigned_words[i].value("word", std::string());
        if (srv_audio_word_ends_sentence(word)) {
            process_sentence(sentence_begin, i + 1);
            sentence_begin = i + 1;
        }
    }
    process_sentence(sentence_begin, assigned_words.size());
}

static std::string srv_audio_dominant_speaker_for_word_range(
        const std::vector<json> & assigned_words,
        size_t begin,
        size_t end) {
    std::unordered_map<std::string, double> weights;
    for (size_t i = begin; i < end; ++i) {
        const std::string speaker = assigned_words[i].value("speaker", std::string("UNASSIGNED"));
        if (speaker == "UNASSIGNED") {
            continue;
        }
        const double ws = assigned_words[i].value("start_sec", 0.0);
        const double we = std::max(ws, assigned_words[i].value("end_sec", ws));
        weights[speaker] += std::max(0.001, we - ws);
    }

    std::string best_speaker = "UNASSIGNED";
    double best_weight = -1.0;
    for (const auto & it : weights) {
        if (it.second > best_weight) {
            best_weight = it.second;
            best_speaker = it.first;
        }
    }
    return best_speaker;
}

static std::string srv_audio_previous_span_speaker_near(
        const json & speaker_spans,
        double anchor_sec,
        double max_gap_sec,
        double * out_gap_sec = nullptr) {
    std::string best_speaker;
    double best_gap = std::numeric_limits<double>::infinity();
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));
        if (span_end > anchor_sec) {
            continue;
        }
        const double gap = anchor_sec - span_end;
        if (gap <= max_gap_sec && gap < best_gap) {
            best_gap = gap;
            best_speaker = span.value("speaker", std::string("UNASSIGNED"));
        }
    }
    if (out_gap_sec != nullptr) {
        *out_gap_sec = best_gap;
    }
    return best_speaker.empty() ? std::string("UNASSIGNED") : best_speaker;
}

static std::string srv_audio_next_span_speaker_near(
        const json & speaker_spans,
        double anchor_sec,
        double max_gap_sec,
        double * out_gap_sec = nullptr) {
    std::string best_speaker;
    double best_gap = std::numeric_limits<double>::infinity();
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double gap = span_start - anchor_sec;
        if (gap < 0.0) {
            continue;
        }
        if (gap <= max_gap_sec && gap < best_gap) {
            best_gap = gap;
            best_speaker = span.value("speaker", std::string("UNASSIGNED"));
        }
    }
    if (out_gap_sec != nullptr) {
        *out_gap_sec = best_gap;
    }
    return best_speaker.empty() ? std::string("UNASSIGNED") : best_speaker;
}

static bool srv_audio_previous_span_end_for_speaker_near(
        const json & speaker_spans,
        const std::string & speaker,
        double anchor_sec,
        double max_gap_sec,
        double * out_end_sec = nullptr,
        double * out_gap_sec = nullptr) {
    if (!speaker_spans.is_array() || speaker.empty() || speaker == "UNASSIGNED") {
        return false;
    }

    double best_end = -std::numeric_limits<double>::infinity();
    double best_gap = std::numeric_limits<double>::infinity();
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        if (span.value("speaker", std::string()) != speaker) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));
        if (span_end > anchor_sec) {
            continue;
        }
        const double gap = anchor_sec - span_end;
        if (gap <= max_gap_sec && (gap < best_gap || (gap == best_gap && span_end > best_end))) {
            best_gap = gap;
            best_end = span_end;
        }
    }

    if (!std::isfinite(best_end)) {
        return false;
    }
    if (out_end_sec != nullptr) {
        *out_end_sec = best_end;
    }
    if (out_gap_sec != nullptr) {
        *out_gap_sec = best_gap;
    }
    return true;
}

static bool srv_audio_covering_or_previous_span_end_for_speaker_near(
        const json & speaker_spans,
        const std::string & speaker,
        double anchor_sec,
        double max_gap_sec,
        double * out_end_sec = nullptr,
        double * out_gap_sec = nullptr) {
    if (!speaker_spans.is_array() || speaker.empty() || speaker == "UNASSIGNED") {
        return false;
    }

    double best_end = -std::numeric_limits<double>::infinity();
    double best_gap = std::numeric_limits<double>::infinity();
    bool found = false;
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        if (span.value("speaker", std::string()) != speaker) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));

        double gap = std::numeric_limits<double>::infinity();
        if (anchor_sec >= span_start && anchor_sec <= span_end) {
            gap = 0.0;
        } else if (span_end <= anchor_sec) {
            gap = anchor_sec - span_end;
        } else {
            continue;
        }

        if (gap <= max_gap_sec && (!found || gap < best_gap || (gap == best_gap && span_end > best_end))) {
            best_gap = gap;
            best_end = span_end;
            found = true;
        }
    }

    if (!found) {
        return false;
    }
    if (out_end_sec != nullptr) {
        *out_end_sec = best_end;
    }
    if (out_gap_sec != nullptr) {
        *out_gap_sec = best_gap;
    }
    return true;
}

static bool srv_audio_next_span_start_for_speaker_near(
        const json & speaker_spans,
        const std::string & speaker,
        double anchor_sec,
        double max_gap_sec,
        double * out_start_sec = nullptr,
        double * out_gap_sec = nullptr) {
    if (!speaker_spans.is_array() || speaker.empty() || speaker == "UNASSIGNED") {
        return false;
    }

    double best_start = std::numeric_limits<double>::infinity();
    double best_gap = std::numeric_limits<double>::infinity();
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        if (span.value("speaker", std::string()) != speaker) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        if (span_start < anchor_sec) {
            continue;
        }
        const double gap = span_start - anchor_sec;
        if (gap <= max_gap_sec && (gap < best_gap || (gap == best_gap && span_start < best_start))) {
            best_gap = gap;
            best_start = span_start;
        }
    }

    if (!std::isfinite(best_start)) {
        return false;
    }
    if (out_start_sec != nullptr) {
        *out_start_sec = best_start;
    }
    if (out_gap_sec != nullptr) {
        *out_gap_sec = best_gap;
    }
    return true;
}

static void srv_audio_refine_realtime_sentence_boundaries(
        std::vector<json> & assigned_words,
        const json & speaker_spans) {
    if (assigned_words.empty() || !speaker_spans.is_array() || speaker_spans.empty()) {
        return;
    }

    struct sentence_range {
        size_t begin = 0;
        size_t end = 0;
        double start_sec = 0.0;
        double end_sec = 0.0;
        double duration_sec = 0.0;
        size_t num_words = 0;
        std::string dominant_speaker = "UNASSIGNED";
    };

    auto reassign_range = [&](size_t begin, size_t end, const std::string & speaker) {
        for (size_t i = begin; i < end; ++i) {
            assigned_words[i]["speaker"] = speaker;
        }
    };

    std::vector<sentence_range> ranges;
    size_t sentence_begin = 0;
    for (size_t i = 0; i < assigned_words.size(); ++i) {
        const std::string word = assigned_words[i].value("word", std::string());
        if (!srv_audio_word_ends_sentence(word) && i + 1 != assigned_words.size()) {
            continue;
        }

        const size_t sentence_end = i + 1;
        sentence_range range;
        range.begin = sentence_begin;
        range.end = sentence_end;
        range.num_words = sentence_end - sentence_begin;
        range.start_sec = assigned_words[sentence_begin].value("start_sec", 0.0);
        range.end_sec = std::max(
            range.start_sec,
            assigned_words[sentence_end - 1].value("end_sec", range.start_sec));
        range.duration_sec = std::max(0.0, range.end_sec - range.start_sec);
        range.dominant_speaker = srv_audio_dominant_speaker_for_word_range(assigned_words, sentence_begin, sentence_end);
        ranges.push_back(std::move(range));
        sentence_begin = sentence_end;
    }

    constexpr double k_boundary_gap_sec = 0.24;
    constexpr size_t k_short_sentence_words = 6;
    constexpr double k_short_sentence_duration_sec = 1.75;
    constexpr double k_prev_carry_gap_sec = 0.24;
    constexpr double k_next_span_delay_sec = 0.20;
    constexpr double k_next_span_search_sec = 0.80;

    for (size_t i = 0; i < ranges.size(); ++i) {
        auto & range = ranges[i];
        if (range.begin >= range.end) {
            continue;
        }

        const bool short_sentence =
            range.num_words <= k_short_sentence_words ||
            range.duration_sec <= k_short_sentence_duration_sec;
        if (!short_sentence) {
            continue;
        }

        const std::string first_word = assigned_words[range.begin].value("word", std::string());
        if (!srv_audio_word_starts_with_sentence_start(first_word)) {
            continue;
        }

        const std::string prev_sentence_speaker =
            i > 0 ? ranges[i - 1].dominant_speaker : std::string("UNASSIGNED");
        const std::string next_sentence_speaker =
            (i + 1 < ranges.size()) ? ranges[i + 1].dominant_speaker : std::string("UNASSIGNED");

        double prev_gap_sec = std::numeric_limits<double>::infinity();
        double next_gap_sec = std::numeric_limits<double>::infinity();
        const std::string prev_span_speaker =
            srv_audio_previous_span_speaker_near(speaker_spans, range.start_sec, k_boundary_gap_sec, &prev_gap_sec);
        const std::string next_span_speaker =
            srv_audio_next_span_speaker_near(speaker_spans, range.end_sec, k_boundary_gap_sec, &next_gap_sec);

        if (prev_sentence_speaker != "UNASSIGNED" &&
                prev_sentence_speaker == next_sentence_speaker &&
                range.dominant_speaker != prev_sentence_speaker) {
            reassign_range(range.begin, range.end, prev_sentence_speaker);
            range.dominant_speaker = prev_sentence_speaker;
            continue;
        }

        if (prev_sentence_speaker != "UNASSIGNED" &&
                prev_sentence_speaker == prev_span_speaker &&
                range.dominant_speaker != prev_sentence_speaker &&
                next_span_speaker == range.dominant_speaker &&
                prev_gap_sec <= k_boundary_gap_sec) {
            reassign_range(range.begin, range.end, prev_sentence_speaker);
            range.dominant_speaker = prev_sentence_speaker;
            continue;
        }

        if (next_sentence_speaker != "UNASSIGNED" &&
                next_sentence_speaker == next_span_speaker &&
                range.dominant_speaker != next_sentence_speaker &&
                prev_span_speaker == range.dominant_speaker &&
                next_gap_sec <= k_boundary_gap_sec) {
            reassign_range(range.begin, range.end, next_sentence_speaker);
            range.dominant_speaker = next_sentence_speaker;
            continue;
        }

        // Keep very short standalone sentences with the previous speaker when the
        // current dominant speaker only starts after the sentence has already begun.
        if (i > 0 &&
                prev_sentence_speaker != "UNASSIGNED" &&
                range.dominant_speaker != "UNASSIGNED" &&
                range.dominant_speaker != prev_sentence_speaker &&
                (next_sentence_speaker == range.dominant_speaker || next_sentence_speaker == "UNASSIGNED")) {
            double prev_same_end_sec = 0.0;
            double prev_same_gap_sec = std::numeric_limits<double>::infinity();
            double next_dom_start_sec = std::numeric_limits<double>::infinity();
            double next_dom_gap_sec = std::numeric_limits<double>::infinity();
            const bool has_prev_same = srv_audio_covering_or_previous_span_end_for_speaker_near(
                speaker_spans,
                prev_sentence_speaker,
                range.start_sec,
                k_prev_carry_gap_sec,
                &prev_same_end_sec,
                &prev_same_gap_sec);
            const bool has_next_dom = srv_audio_next_span_start_for_speaker_near(
                speaker_spans,
                range.dominant_speaker,
                range.start_sec,
                k_next_span_search_sec,
                &next_dom_start_sec,
                &next_dom_gap_sec);
            if (has_prev_same &&
                    has_next_dom &&
                    prev_same_gap_sec <= k_prev_carry_gap_sec &&
                    next_dom_start_sec - range.start_sec >= k_next_span_delay_sec) {
                reassign_range(range.begin, range.end, prev_sentence_speaker);
                range.dominant_speaker = prev_sentence_speaker;
                continue;
            }
        }
    }
}

static std::string srv_audio_piece_first_token(const json & piece) {
    const auto tokens = srv_audio_split_ws(srv_audio_collapse_ws(piece.value("text", std::string())));
    for (const auto & token : tokens) {
        const std::string clean = srv_audio_sanitize_word_token(token);
        if (!clean.empty()) {
            return clean;
        }
    }
    return "";
}

static std::string srv_audio_piece_last_token(const json & piece) {
    const auto tokens = srv_audio_split_ws(srv_audio_collapse_ws(piece.value("text", std::string())));
    for (size_t i = tokens.size(); i > 0; --i) {
        const std::string clean = srv_audio_sanitize_word_token(tokens[i - 1]);
        if (!clean.empty()) {
            return clean;
        }
    }
    return "";
}

static int64_t srv_audio_piece_num_words(const json & piece) {
    const int64_t num_words = piece.value("num_words", int64_t(-1));
    if (num_words >= 0) {
        return num_words;
    }
    int64_t count = 0;
    for (const auto & token : srv_audio_split_ws(srv_audio_collapse_ws(piece.value("text", std::string())))) {
        if (!srv_audio_sanitize_word_token(token).empty()) {
            ++count;
        }
    }
    return count;
}

static double srv_audio_piece_duration_sec(const json & piece) {
    const double start_sec = piece.value("start_sec", 0.0);
    const double end_sec = std::max(start_sec, piece.value("end_sec", start_sec));
    return std::max(0.0, end_sec - start_sec);
}

static bool srv_audio_piece_is_tiny(const json & piece) {
    const int64_t num_words = srv_audio_piece_num_words(piece);
    const double duration_sec = srv_audio_piece_duration_sec(piece);
    if (num_words <= 1) {
        return duration_sec <= 2.0;
    }
    return num_words <= 4 && duration_sec <= 1.6;
}

static char srv_audio_last_meaningful_char(const std::string & text) {
    for (size_t i = text.size(); i > 0; --i) {
        const char ch = text[i - 1];
        if (std::isspace((unsigned char) ch) ||
                ch == '"' ||
                ch == '\'' ||
                ch == ')' ||
                ch == ']' ||
                ch == '}') {
            continue;
        }
        return ch;
    }
    return '\0';
}

static bool srv_audio_piece_starts_with_lower_or_continuation(const json & piece) {
    return srv_audio_word_starts_with_lower_or_continuation(srv_audio_piece_first_token(piece));
}

static bool srv_audio_piece_starts_with_sentence_start(const json & piece) {
    return srv_audio_word_starts_with_sentence_start(srv_audio_piece_first_token(piece));
}

static bool srv_audio_piece_ends_sentence(const json & piece) {
    return srv_audio_word_ends_sentence(srv_audio_piece_last_token(piece));
}

static bool srv_audio_piece_flows_into_next(const json & current_piece, const json & next_piece) {
    if (srv_audio_piece_ends_sentence(current_piece)) {
        return false;
    }

    if (srv_audio_piece_starts_with_lower_or_continuation(next_piece)) {
        return true;
    }

    const char last = srv_audio_last_meaningful_char(current_piece.value("text", std::string()));
    return last == ',' || last == ';' || last == ':' || last == '-';
}

static std::string srv_audio_assign_piece_to_realtime_speaker(
        const json & piece,
        const json & speaker_spans) {
    if (!piece.is_object() || !speaker_spans.is_array() || speaker_spans.empty()) {
        return "UNASSIGNED";
    }

    const double start_sec = piece.value("start_sec", 0.0);
    const double end_sec = std::max(start_sec, piece.value("end_sec", start_sec));
    const double anchor_sec = 0.5 * (start_sec + end_sec);

    const json * best_overlap_span = nullptr;
    double best_overlap_sec = 0.0;
    bool best_anchor_inside = false;
    double best_anchor_dist = std::numeric_limits<double>::infinity();

    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));
        const double overlap_sec = std::max(0.0, std::min(end_sec, span_end) - std::max(start_sec, span_start));
        if (overlap_sec <= 1e-9) {
            continue;
        }

        const bool anchor_inside = anchor_sec >= span_start && anchor_sec <= span_end;
        double anchor_dist = 0.0;
        if (!anchor_inside) {
            anchor_dist = std::min(std::abs(anchor_sec - span_start), std::abs(anchor_sec - span_end));
        }

        const bool better_overlap = overlap_sec > best_overlap_sec + 1e-9;
        const bool same_overlap = std::abs(overlap_sec - best_overlap_sec) <= 1e-9;
        const bool better_tie_break =
            same_overlap && (
                (anchor_inside && !best_anchor_inside) ||
                (anchor_inside == best_anchor_inside && anchor_dist < best_anchor_dist));

        if (best_overlap_span == nullptr || better_overlap || better_tie_break) {
            best_overlap_span = &span;
            best_overlap_sec = overlap_sec;
            best_anchor_inside = anchor_inside;
            best_anchor_dist = anchor_dist;
        }
    }

    if (best_overlap_span != nullptr) {
        return best_overlap_span->value("speaker", std::string("UNASSIGNED"));
    }

    constexpr double k_nearest_assign_tolerance_sec = 0.12;
    const json * nearest_span = nullptr;
    double nearest_dist = std::numeric_limits<double>::infinity();
    for (const auto & span : speaker_spans) {
        if (!span.is_object()) {
            continue;
        }
        const double span_start = span.value("start_sec", 0.0);
        const double span_end = std::max(span_start, span.value("end_sec", span_start));
        double dist = 0.0;
        if (anchor_sec < span_start) {
            dist = span_start - anchor_sec;
        } else if (anchor_sec > span_end) {
            dist = anchor_sec - span_end;
        }
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_span = &span;
        }
    }

    if (nearest_span != nullptr && nearest_dist <= k_nearest_assign_tolerance_sec) {
        return nearest_span->value("speaker", std::string("UNASSIGNED"));
    }

    return "UNASSIGNED";
}

static void srv_audio_carry_piece_prefixes_backward(
        std::vector<json> & assigned_pieces,
        const json & words);

static void srv_audio_refine_realtime_piece_boundaries(
        std::vector<json> & assigned_pieces,
        const json & words) {
    if (assigned_pieces.size() < 2) {
        return;
    }

    srv_audio_carry_piece_prefixes_backward(assigned_pieces, words);

    for (size_t i = 1; i + 1 < assigned_pieces.size(); ++i) {
        auto & cur = assigned_pieces[i];
        if (cur.value("speaker", std::string("UNASSIGNED")) != "UNASSIGNED" || !srv_audio_piece_is_tiny(cur)) {
            continue;
        }
        const std::string prev_speaker = assigned_pieces[i - 1].value("speaker", std::string("UNASSIGNED"));
        const std::string next_speaker = assigned_pieces[i + 1].value("speaker", std::string("UNASSIGNED"));
        if (prev_speaker != "UNASSIGNED" && prev_speaker == next_speaker) {
            cur["speaker"] = prev_speaker;
        }
    }

    for (size_t i = 1; i < assigned_pieces.size(); ++i) {
        auto & cur = assigned_pieces[i];
        const std::string prev_speaker = assigned_pieces[i - 1].value("speaker", std::string("UNASSIGNED"));
        const std::string cur_speaker = cur.value("speaker", std::string("UNASSIGNED"));
        if (prev_speaker == "UNASSIGNED" || cur_speaker == prev_speaker) {
            continue;
        }
        if (!srv_audio_piece_is_tiny(cur) || !srv_audio_piece_starts_with_lower_or_continuation(cur)) {
            continue;
        }
        cur["speaker"] = prev_speaker;
    }

    for (size_t i = 1; i + 1 < assigned_pieces.size(); ++i) {
        auto & cur = assigned_pieces[i];
        const std::string prev_speaker = assigned_pieces[i - 1].value("speaker", std::string("UNASSIGNED"));
        const std::string cur_speaker = cur.value("speaker", std::string("UNASSIGNED"));
        const std::string next_speaker = assigned_pieces[i + 1].value("speaker", std::string("UNASSIGNED"));
        if (prev_speaker == "UNASSIGNED" || next_speaker == "UNASSIGNED") {
            continue;
        }
        if (cur_speaker != prev_speaker || cur_speaker == next_speaker) {
            continue;
        }
        if (!srv_audio_piece_is_tiny(cur) ||
                !srv_audio_piece_starts_with_sentence_start(cur) ||
                !srv_audio_piece_flows_into_next(cur, assigned_pieces[i + 1])) {
            continue;
        }
        cur["speaker"] = next_speaker;
    }
}

static double srv_audio_assigned_words_duration_sec(
        const std::vector<json> & assigned_words,
        size_t begin,
        size_t end) {
    double duration_sec = 0.0;
    end = std::min(end, assigned_words.size());
    for (size_t i = begin; i < end; ++i) {
        const double start_sec = assigned_words[i].value("start_sec", 0.0);
        const double end_sec = std::max(start_sec, assigned_words[i].value("end_sec", start_sec));
        duration_sec += std::max(0.001, end_sec - start_sec);
    }
    return duration_sec;
}

static bool srv_audio_piece_edge_run_is_small(
        const std::vector<json> & assigned_words,
        size_t begin,
        size_t end) {
    constexpr size_t k_max_fragment_words = 8;
    constexpr double k_max_fragment_duration_sec = 3.0;
    return end > begin &&
        end - begin <= k_max_fragment_words &&
        srv_audio_assigned_words_duration_sec(assigned_words, begin, end) <= k_max_fragment_duration_sec;
}

static bool srv_audio_piece_edge_run_speaker(
        const std::vector<json> & assigned_words,
        size_t begin,
        size_t end,
        const std::string & dominant_speaker,
        std::string & out_speaker) {
    out_speaker.clear();
    end = std::min(end, assigned_words.size());
    for (size_t i = begin; i < end; ++i) {
        const std::string speaker = assigned_words[i].value("speaker", std::string("UNASSIGNED"));
        if (speaker.empty() || speaker == "UNASSIGNED" || speaker == dominant_speaker) {
            return false;
        }
        if (out_speaker.empty()) {
            out_speaker = speaker;
            continue;
        }
        if (speaker != out_speaker) {
            return false;
        }
    }
    return !out_speaker.empty();
}

static json srv_audio_make_piece_from_assigned_words(
        const std::vector<json> & assigned_words,
        size_t begin,
        size_t end,
        const std::string & speaker) {
    std::vector<std::string> toks;
    toks.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        toks.push_back(assigned_words[i].value("word", std::string()));
    }

    const int64_t start_word_index = assigned_words[begin].value("_global_word_index", int64_t(-1));
    const int64_t end_word_index = assigned_words[end - 1].value("_global_word_index", int64_t(-1)) + 1;
    return json{
        {"speaker", speaker},
        {"text", srv_audio_join_words(toks)},
        {"start_sec", srv_audio_round3(assigned_words[begin].value("start_sec", 0.0))},
        {"end_sec", srv_audio_round3(std::max(
            assigned_words[begin].value("start_sec", 0.0),
            assigned_words[end - 1].value("end_sec", assigned_words[begin].value("start_sec", 0.0))))},
        {"start_word_index", start_word_index},
        {"end_word_index", end_word_index},
        {"num_words", (int64_t) (end - begin)},
    };
}

static json srv_audio_make_piece_from_word_range(
        const json & words,
        int64_t begin_word_index,
        int64_t end_word_index,
        const std::string & speaker) {
    std::vector<std::string> toks;
    toks.reserve((size_t) std::max<int64_t>(0, end_word_index - begin_word_index));
    for (int64_t idx = begin_word_index; idx < end_word_index; ++idx) {
        if (!words[(size_t) idx].is_object()) {
            continue;
        }
        toks.push_back(srv_audio_sanitize_word_token(json_value(words[(size_t) idx], "word", std::string())));
    }

    const double start_sec = json_value(words[(size_t) begin_word_index], "start_sec", 0.0);
    const double end_sec = std::max(
        start_sec,
        json_value(words[(size_t) end_word_index - 1], "end_sec", start_sec));
    return json{
        {"speaker", speaker},
        {"text", srv_audio_join_words(toks)},
        {"start_sec", srv_audio_round3(start_sec)},
        {"end_sec", srv_audio_round3(end_sec)},
        {"start_word_index", begin_word_index},
        {"end_word_index", end_word_index},
        {"num_words", end_word_index - begin_word_index},
    };
}

static bool srv_audio_word_range_is_small(
        const json & words,
        int64_t begin_word_index,
        int64_t end_word_index) {
    if (!words.is_array() || begin_word_index < 0 || end_word_index <= begin_word_index || (size_t) end_word_index > words.size()) {
        return false;
    }
    constexpr int64_t k_max_fragment_words = 4;
    constexpr double k_max_fragment_duration_sec = 1.6;
    const double start_sec = json_value(words[(size_t) begin_word_index], "start_sec", 0.0);
    const double end_sec = std::max(
        start_sec,
        json_value(words[(size_t) end_word_index - 1], "end_sec", start_sec));
    return end_word_index - begin_word_index <= k_max_fragment_words &&
        end_sec - start_sec <= k_max_fragment_duration_sec;
}

static int64_t srv_audio_piece_prefix_sentence_end_word_index(
        const json & piece,
        const json & words) {
    const int64_t start_word_index = piece.value("start_word_index", int64_t(-1));
    const int64_t end_word_index = piece.value("end_word_index", int64_t(-1));
    if (!words.is_array() || start_word_index < 0 || end_word_index <= start_word_index || (size_t) end_word_index > words.size()) {
        return -1;
    }

    for (int64_t idx = start_word_index; idx < end_word_index; ++idx) {
        if (!words[(size_t) idx].is_object()) {
            continue;
        }
        if (srv_audio_word_ends_sentence(json_value(words[(size_t) idx], "word", std::string()))) {
            return idx + 1;
        }
    }
    return -1;
}

static void srv_audio_carry_piece_prefixes_backward(
        std::vector<json> & assigned_pieces,
        const json & words) {
    if (assigned_pieces.size() < 2 || !words.is_array()) {
        return;
    }

    std::vector<json> adjusted;
    adjusted.reserve(assigned_pieces.size() + 4);
    adjusted.push_back(assigned_pieces.front());

    for (size_t i = 1; i < assigned_pieces.size(); ++i) {
        json cur = assigned_pieces[i];
        json & prev = adjusted.back();

        const std::string prev_speaker = prev.value("speaker", std::string("UNASSIGNED"));
        const std::string cur_speaker = cur.value("speaker", std::string("UNASSIGNED"));
        if (prev_speaker == "UNASSIGNED" ||
                cur_speaker == prev_speaker ||
                srv_audio_piece_ends_sentence(prev)) {
            adjusted.push_back(std::move(cur));
            continue;
        }

        const int64_t prefix_end_word_index = srv_audio_piece_prefix_sentence_end_word_index(cur, words);
        const int64_t start_word_index = cur.value("start_word_index", int64_t(-1));
        const int64_t end_word_index = cur.value("end_word_index", int64_t(-1));
        if (prefix_end_word_index <= start_word_index ||
                prefix_end_word_index >= end_word_index ||
                !srv_audio_word_range_is_small(words, start_word_index, prefix_end_word_index)) {
            adjusted.push_back(std::move(cur));
            continue;
        }

        adjusted.push_back(srv_audio_make_piece_from_word_range(
            words,
            start_word_index,
            prefix_end_word_index,
            prev_speaker));
        adjusted.push_back(srv_audio_make_piece_from_word_range(
            words,
            prefix_end_word_index,
            end_word_index,
            cur_speaker));
    }

    assigned_pieces = std::move(adjusted);
}

static std::vector<json> srv_audio_split_piece_edges_by_word_speakers(
        const json & piece,
        const json & words,
        const json & speaker_spans) {
    std::vector<json> out;

    const int64_t start_word_index = piece.value("start_word_index", int64_t(-1));
    const int64_t end_word_index = piece.value("end_word_index", int64_t(-1));
    const bool has_word_range =
        words.is_array() &&
        start_word_index >= 0 &&
        end_word_index > start_word_index &&
        (size_t) end_word_index <= words.size();
    if (!has_word_range) {
        out.push_back(piece);
        return out;
    }

    std::vector<json> assigned_words;
    assigned_words.reserve((size_t) (end_word_index - start_word_index));
    for (int64_t idx = start_word_index; idx < end_word_index; ++idx) {
        if (!words[(size_t) idx].is_object()) {
            continue;
        }
        json assigned_word = words[(size_t) idx];
        assigned_word["word"] = srv_audio_sanitize_word_token(json_value(assigned_word, "word", std::string()));
        if (assigned_word.value("word", std::string()).empty()) {
            continue;
        }
        assigned_word["speaker"] = srv_audio_assign_word_to_realtime_speaker(assigned_word, speaker_spans);
        assigned_word["_global_word_index"] = idx;
        assigned_words.push_back(std::move(assigned_word));
    }

    if (assigned_words.size() < 2) {
        out.push_back(piece);
        return out;
    }

    const std::string word_dominant_speaker =
        srv_audio_dominant_speaker_for_word_range(assigned_words, 0, assigned_words.size());
    const std::string dominant_speaker =
        word_dominant_speaker != "UNASSIGNED" ? word_dominant_speaker : piece.value("speaker", std::string("UNASSIGNED"));
    if (dominant_speaker == "UNASSIGNED") {
        out.push_back(piece);
        return out;
    }

    size_t leading_end = 0;
    while (leading_end < assigned_words.size() &&
            assigned_words[leading_end].value("speaker", std::string("UNASSIGNED")) != dominant_speaker) {
        ++leading_end;
    }

    size_t trailing_begin = assigned_words.size();
    while (trailing_begin > 0 &&
            assigned_words[trailing_begin - 1].value("speaker", std::string("UNASSIGNED")) != dominant_speaker) {
        --trailing_begin;
    }

    bool have_leading = false;
    std::string leading_speaker;
    if (leading_end > 0 &&
            leading_end < assigned_words.size() &&
            srv_audio_piece_edge_run_is_small(assigned_words, 0, leading_end) &&
            srv_audio_piece_edge_run_speaker(assigned_words, 0, leading_end, dominant_speaker, leading_speaker)) {
        have_leading = true;
    } else {
        leading_end = 0;
    }

    bool have_trailing = false;
    std::string trailing_speaker;
    if (trailing_begin < assigned_words.size() &&
            trailing_begin > leading_end &&
            srv_audio_piece_edge_run_is_small(assigned_words, trailing_begin, assigned_words.size()) &&
            srv_audio_piece_edge_run_speaker(assigned_words, trailing_begin, assigned_words.size(), dominant_speaker, trailing_speaker)) {
        have_trailing = true;
    } else {
        trailing_begin = assigned_words.size();
    }

    if (!have_leading && !have_trailing) {
        json updated_piece = piece;
        updated_piece["speaker"] = dominant_speaker;
        out.push_back(std::move(updated_piece));
        return out;
    }

    if (have_leading) {
        out.push_back(srv_audio_make_piece_from_assigned_words(assigned_words, 0, leading_end, leading_speaker));
    }

    const size_t core_begin = leading_end;
    const size_t core_end = trailing_begin;
    if (core_begin < core_end) {
        out.push_back(srv_audio_make_piece_from_assigned_words(assigned_words, core_begin, core_end, dominant_speaker));
    }

    if (have_trailing) {
        out.push_back(srv_audio_make_piece_from_assigned_words(
            assigned_words,
            trailing_begin,
            assigned_words.size(),
            trailing_speaker));
    }

    if (out.empty()) {
        out.push_back(piece);
    }
    return out;
}

static json srv_audio_build_realtime_speaker_segments(
        const json & pieces,
        const json & words,
        const json & speaker_spans,
        double max_gap_sec,
        int max_words,
        double max_duration_sec,
        bool split_on_hard_break) {
    json speaker_segments = json::array();
    json effective_pieces = pieces;
    if ((!effective_pieces.is_array() || effective_pieces.empty()) && words.is_array() && !words.empty()) {
        effective_pieces = srv_audio_build_fallback_whisper_pieces(words);
    }
    if (!effective_pieces.is_array() || effective_pieces.empty()) {
        return speaker_segments;
    }

    std::vector<json> assigned_pieces;
    assigned_pieces.reserve(effective_pieces.size());
    for (const auto & piece : effective_pieces) {
        if (!piece.is_object()) {
            continue;
        }

        json normalized_piece = piece;
        const int64_t start_word_index = normalized_piece.value("start_word_index", int64_t(-1));
        const int64_t end_word_index = normalized_piece.value("end_word_index", int64_t(-1));
        const bool has_word_range =
            words.is_array() &&
            start_word_index >= 0 &&
            end_word_index > start_word_index &&
            (size_t) end_word_index <= words.size() &&
            words[(size_t) start_word_index].is_object() &&
            words[(size_t) end_word_index - 1].is_object();

        std::string piece_text = srv_audio_sanitize_transcript_text(
            normalized_piece.value("text", std::string()));
        if (piece_text.empty() && has_word_range) {
            piece_text = srv_audio_join_word_range(words, (size_t) start_word_index, (size_t) end_word_index);
        }
        if (piece_text.empty()) {
            continue;
        }

        double piece_start = normalized_piece.value("start_sec", 0.0);
        double piece_end = std::max(piece_start, normalized_piece.value("end_sec", piece_start));
        if (has_word_range) {
            piece_start = std::max(
                0.0,
                json_value(words[(size_t) start_word_index], "start_sec", piece_start));
            piece_end = std::max(
                piece_start,
                json_value(words[(size_t) end_word_index - 1], "end_sec", piece_end));
        }
        if (piece_end <= piece_start) {
            piece_end = piece_start + 0.001;
        }

        int64_t piece_num_words = srv_audio_piece_num_words(normalized_piece);
        if (piece_num_words < 0 && has_word_range) {
            piece_num_words = end_word_index - start_word_index;
        }
        if (piece_num_words < 0) {
            piece_num_words = 0;
        }

        normalized_piece["text"] = piece_text;
        normalized_piece["start_sec"] = srv_audio_round3(piece_start);
        normalized_piece["end_sec"] = srv_audio_round3(piece_end);
        normalized_piece["num_words"] = piece_num_words;
        normalized_piece["speaker"] = srv_audio_assign_piece_to_realtime_speaker(normalized_piece, speaker_spans);
        auto split_pieces = srv_audio_split_piece_edges_by_word_speakers(normalized_piece, words, speaker_spans);
        for (auto & split_piece : split_pieces) {
            assigned_pieces.push_back(std::move(split_piece));
        }
    }

    if (assigned_pieces.empty()) {
        return speaker_segments;
    }

    srv_audio_refine_realtime_piece_boundaries(assigned_pieces, words);

    std::vector<json> bucket;
    std::string current_speaker = "UNASSIGNED";
    double current_start = 0.0;
    int64_t current_num_words = 0;

    auto flush = [&]() {
        if (bucket.empty()) {
            return;
        }
        std::vector<std::string> piece_texts;
        piece_texts.reserve(bucket.size());
        int64_t total_words = 0;
        for (const auto & w : bucket) {
            piece_texts.push_back(w.value("text", std::string()));
            total_words += w.value("num_words", int64_t(0));
        }

        const std::string merged_text = srv_audio_sanitize_transcript_text(srv_audio_join_words(piece_texts));
        if (merged_text.empty()) {
            bucket.clear();
            current_num_words = 0;
            return;
        }

        speaker_segments.push_back(json{
            {"speaker", current_speaker},
            {"start_sec", srv_audio_round3(bucket.front().value("start_sec", 0.0))},
            {"end_sec", srv_audio_round3(bucket.back().value("end_sec", bucket.front().value("start_sec", 0.0)))},
            {"text", merged_text},
            {"num_words", total_words},
            {"num_pieces", (int64_t) bucket.size()},
        });
        bucket.clear();
        current_num_words = 0;
    };

    for (const auto & normalized_piece : assigned_pieces) {
        const std::string speaker = normalized_piece.value("speaker", std::string("UNASSIGNED"));
        const int64_t piece_num_words = normalized_piece.value("num_words", int64_t(0));

        if (bucket.empty()) {
            bucket.push_back(normalized_piece);
            current_speaker = speaker;
            current_start = normalized_piece.value("start_sec", 0.0);
            current_num_words = piece_num_words;
            continue;
        }

        const double prev_end = bucket.back().value("end_sec", bucket.back().value("start_sec", 0.0));
        const double next_start = normalized_piece.value("start_sec", prev_end);
        const double next_end = normalized_piece.value("end_sec", next_start);

        bool should_split = speaker != current_speaker;
        if (!should_split && max_gap_sec >= 0.0 && next_start - prev_end > max_gap_sec) {
            should_split = true;
        }
        if (!should_split && max_words > 0 && current_num_words + piece_num_words > max_words) {
            should_split = true;
        }
        if (!should_split && max_duration_sec > 0.0 && next_end - current_start > max_duration_sec) {
            should_split = true;
        }
        if (!should_split &&
                split_on_hard_break &&
                srv_audio_piece_ends_sentence(bucket.back()) &&
                srv_audio_piece_starts_with_sentence_start(normalized_piece)) {
            should_split = true;
        }
        if (should_split) {
            flush();
            current_speaker = speaker;
            current_start = next_start;
        }

        bucket.push_back(normalized_piece);
        current_num_words += piece_num_words;
    }

    flush();
    return speaker_segments;
}

static void srv_audio_remove_if_exists(const std::filesystem::path & path) {
    std::error_code ec;
    if (!path.empty()) {
        std::filesystem::remove(path, ec);
    }
}

static void srv_audio_shift_word_timestamps(json & words, double offset_sec, double source_audio_seconds) {
    if (!words.is_array() || std::abs(offset_sec) < 1e-9) {
        return;
    }

    for (auto & w : words) {
        if (!w.is_object()) {
            continue;
        }

        double s = w.value("start_sec", 0.0);
        double e = w.value("end_sec", s);

        s = std::max(0.0, s + offset_sec);
        e = std::max(s, e + offset_sec);

        if (source_audio_seconds > 0.0) {
            s = std::min(s, source_audio_seconds);
            e = std::min(std::max(e, s), source_audio_seconds);
        }

        if (e <= s) {
            e = s + 0.001;
            if (source_audio_seconds > 0.0) {
                e = std::min(e, source_audio_seconds);
            }
        }

        w["start_sec"] = srv_audio_round3(s);
        w["end_sec"] = srv_audio_round3(e);
    }
}

static void srv_audio_shift_piece_timestamps(json & pieces, double offset_sec, double source_audio_seconds) {
    if (!pieces.is_array() || std::abs(offset_sec) < 1e-9) {
        return;
    }

    for (auto & piece : pieces) {
        if (!piece.is_object()) {
            continue;
        }

        double s = piece.value("start_sec", 0.0);
        double e = piece.value("end_sec", s);

        s = std::max(0.0, s + offset_sec);
        e = std::max(s, e + offset_sec);

        if (source_audio_seconds > 0.0) {
            s = std::min(s, source_audio_seconds);
            e = std::min(std::max(e, s), source_audio_seconds);
        }

        if (e <= s) {
            e = s + 0.001;
            if (source_audio_seconds > 0.0) {
                e = std::min(e, source_audio_seconds);
            }
        }

        piece["start_sec"] = srv_audio_round3(s);
        piece["end_sec"] = srv_audio_round3(e);
    }
}

static json srv_audio_fallback_words(
        const std::string & transcript,
        double source_audio_seconds,
        double seconds_per_token) {
    json words = json::array();
    std::vector<std::string> ws = srv_audio_split_ws(transcript);
    if (ws.empty()) {
        return words;
    }
    const double duration = source_audio_seconds > 0.0
        ? source_audio_seconds
        : std::max(seconds_per_token * (double) ws.size(), 1e-3);
    const double dt = duration / (double) ws.size();
    for (size_t i = 0; i < ws.size(); ++i) {
        std::string clean_w = srv_audio_sanitize_word_token(ws[i]);
        if (clean_w.empty()) {
            continue;
        }
        const double start = dt * (double) i;
        const double end = dt * (double) (i + 1);
        words.push_back(json{
            {"word", clean_w},
            {"start_sec", srv_audio_round3(start)},
            {"end_sec", srv_audio_round3(end)},
            {"start_token_index", (int64_t) i},
            {"end_token_index", (int64_t) (i + 1)},
        });
    }
    return words;
}

static json srv_audio_build_fallback_whisper_pieces(const json & words) {
    json pieces = json::array();
    if (!words.is_array() || words.empty()) {
        return pieces;
    }

    constexpr size_t k_max_words = 16;
    constexpr double k_max_duration_sec = 5.0;

    size_t begin = 0;
    while (begin < words.size()) {
        if (!words[begin].is_object()) {
            ++begin;
            continue;
        }

        const double piece_start = words[begin].value("start_sec", 0.0);
        size_t end = begin + 1;
        for (; end < words.size(); ++end) {
            if (!words[end].is_object()) {
                continue;
            }
            const double candidate_end = words[end].value("end_sec", piece_start);
            const size_t candidate_words = end + 1 - begin;
            const bool sentence_end = srv_audio_word_ends_sentence(words[end].value("word", std::string()));
            const bool too_many_words = candidate_words >= k_max_words;
            const bool too_long = candidate_end - piece_start >= k_max_duration_sec;
            if (sentence_end || too_many_words || too_long) {
                ++end;
                break;
            }
        }

        end = std::min(end, words.size());
        while (end > begin && !words[end - 1].is_object()) {
            --end;
        }
        if (end <= begin) {
            begin = std::min(words.size(), begin + 1);
            continue;
        }

        const double piece_end = std::max(
            piece_start,
            words[end - 1].value("end_sec", piece_start));
        pieces.push_back(json{
            {"text", srv_audio_join_word_range(words, begin, end)},
            {"start_sec", srv_audio_round3(piece_start)},
            {"end_sec", srv_audio_round3(piece_end)},
            {"start_word_index", (int64_t) begin},
            {"end_word_index", (int64_t) end},
            {"num_words", (int64_t) (end - begin)},
        });
        begin = end;
    }

    return pieces;
}

static json srv_audio_build_segments(const json & words, int max_words = 12, double max_duration_sec = 4.0) {
    json segments = json::array();
    if (!words.is_array() || words.empty()) {
        return segments;
    }

    std::vector<json> bucket;
    bucket.reserve((size_t) std::max(4, max_words));
    double seg_start = words[0].value("start_sec", 0.0);

    auto flush = [&]() {
        if (bucket.empty()) {
            return;
        }
        std::vector<std::string> toks;
        toks.reserve(bucket.size());
        for (const auto & w : bucket) {
            toks.push_back(w.value("word", std::string()));
        }
        segments.push_back(json{
            {"start_sec", srv_audio_round3(bucket.front().value("start_sec", 0.0))},
            {"end_sec", srv_audio_round3(bucket.back().value("end_sec", 0.0))},
            {"text", srv_audio_join_words(toks)},
            {"num_words", (int64_t) bucket.size()},
        });
        bucket.clear();
    };

    for (const auto & word : words) {
        if (bucket.empty()) {
            bucket.push_back(word);
            seg_start = word.value("start_sec", 0.0);
            continue;
        }

        const double candidate_duration = word.value("end_sec", seg_start) - seg_start;
        if ((int) bucket.size() >= max_words || candidate_duration > max_duration_sec) {
            flush();
            bucket.push_back(word);
            seg_start = word.value("start_sec", 0.0);
            continue;
        }

        bucket.push_back(word);
        const std::string wtxt = word.value("word", std::string());
        if (bucket.size() >= 4 && !wtxt.empty()) {
            const char c = wtxt.back();
            if (c == '.' || c == '?' || c == '!') {
                flush();
            }
        }
    }
    flush();

    return segments;
}

static bool srv_audio_word_ends_sentence(const std::string & word) {
    for (size_t i = word.size(); i > 0; --i) {
        const char c = word[i - 1];
        if (std::isspace((unsigned char) c) || c == '"' || c == '\'' || c == ')' || c == ']' || c == '}') {
            continue;
        }
        return c == '.' || c == '!' || c == '?';
    }
    return false;
}

static json srv_audio_build_sentence_segments(const json & words) {
    json segments = json::array();
    if (!words.is_array() || words.empty()) {
        return segments;
    }

    std::vector<json> bucket;
    auto flush = [&]() {
        if (bucket.empty()) {
            return;
        }
        std::vector<std::string> toks;
        toks.reserve(bucket.size());
        for (const auto & w : bucket) {
            toks.push_back(w.value("word", std::string()));
        }
        segments.push_back(json{
            {"start_sec", srv_audio_round3(bucket.front().value("start_sec", 0.0))},
            {"end_sec", srv_audio_round3(bucket.back().value("end_sec", 0.0))},
            {"text", srv_audio_join_words(toks)},
            {"num_words", (int64_t) bucket.size()},
        });
        bucket.clear();
    };

    for (const auto & word : words) {
        if (!word.is_object()) {
            continue;
        }
        bucket.push_back(word);
        if (srv_audio_word_ends_sentence(word.value("word", std::string()))) {
            flush();
        }
    }
    flush();

    return segments;
}

static json srv_audio_build_subtitle_segments_window(const json & words, double max_duration_sec) {
    if (max_duration_sec <= 0.0) {
        return srv_audio_build_segments(words);
    }

    json segments = json::array();
    if (!words.is_array() || words.empty()) {
        return segments;
    }

    std::vector<json> bucket;
    double seg_start = 0.0;

    auto flush = [&]() {
        if (bucket.empty()) {
            return;
        }
        std::vector<std::string> toks;
        toks.reserve(bucket.size());
        for (const auto & w : bucket) {
            toks.push_back(w.value("word", std::string()));
        }
        segments.push_back(json{
            {"start_sec", srv_audio_round3(bucket.front().value("start_sec", 0.0))},
            {"end_sec", srv_audio_round3(bucket.back().value("end_sec", 0.0))},
            {"text", srv_audio_join_words(toks)},
            {"num_words", (int64_t) bucket.size()},
        });
        bucket.clear();
    };

    for (const auto & word : words) {
        if (!word.is_object()) {
            continue;
        }
        if (bucket.empty()) {
            bucket.push_back(word);
            seg_start = word.value("start_sec", 0.0);
            continue;
        }

        const double candidate_duration = word.value("end_sec", seg_start) - seg_start;
        if (candidate_duration > max_duration_sec) {
            flush();
            bucket.push_back(word);
            seg_start = word.value("start_sec", 0.0);
            continue;
        }

        bucket.push_back(word);
    }
    flush();

    return segments;
}

static std::string srv_audio_transcript_to_md_chunked(const json & sentence_segments, double max_chunk_sec) {
    if (!sentence_segments.is_array() || sentence_segments.empty()) {
        return "";
    }
    if (max_chunk_sec <= 0.0) {
        std::string text;
        for (const auto & seg : sentence_segments) {
            if (!seg.is_object()) {
                continue;
            }
            const std::string s = srv_audio_collapse_ws(seg.value("text", std::string()));
            if (s.empty()) {
                continue;
            }
            if (!text.empty()) {
                text += " ";
            }
            text += s;
        }
        return text.empty() ? std::string() : text + "\n";
    }

    std::vector<std::string> paragraphs;
    std::string current;
    bool has_open_chunk = false;
    double chunk_start = 0.0;

    auto flush_chunk = [&]() {
        const std::string c = srv_audio_collapse_ws(current);
        if (!c.empty()) {
            paragraphs.push_back(c);
        }
        current.clear();
        has_open_chunk = false;
        chunk_start = 0.0;
    };

    for (const auto & seg : sentence_segments) {
        if (!seg.is_object()) {
            continue;
        }
        const std::string seg_text = srv_audio_collapse_ws(seg.value("text", std::string()));
        if (seg_text.empty()) {
            continue;
        }
        const double seg_start = seg.value("start_sec", 0.0);
        const double seg_end = std::max(seg_start, seg.value("end_sec", seg_start));

        if (!has_open_chunk) {
            chunk_start = seg_start;
            current = seg_text;
            has_open_chunk = true;
            continue;
        }

        const double candidate_duration = seg_end - chunk_start;
        if (candidate_duration > max_chunk_sec) {
            flush_chunk();
            chunk_start = seg_start;
            current = seg_text;
            has_open_chunk = true;
            continue;
        }

        current += " ";
        current += seg_text;
    }
    if (has_open_chunk) {
        flush_chunk();
    }

    if (paragraphs.empty()) {
        return "";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < paragraphs.size(); ++i) {
        if (i > 0) {
            oss << "\n\n";
        }
        oss << paragraphs[i];
    }
    oss << "\n";
    return oss.str();
}

static std::string srv_audio_clean_cache_name(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    for (const unsigned char c : name) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == '-') {
            out.push_back((char) c);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

static bool srv_audio_resolve_hf_model(
        const std::string & hf_repo_with_tag,
        const std::string & hf_file,
        const std::string & bearer_token,
        bool offline,
        std::string & out_path,
        std::string & out_repo_no_tag,
        std::string & out_err) {
    out_path.clear();
    out_repo_no_tag.clear();
    out_err.clear();

    if (hf_repo_with_tag.empty() || hf_file.empty()) {
        out_err = "hf_repo and hf_file must be non-empty";
        return false;
    }

    auto [hf_repo, hf_tag] = common_download_split_repo_tag(hf_repo_with_tag);
    if (hf_repo.empty()) {
        out_err = "invalid hf_repo value: " + hf_repo_with_tag;
        return false;
    }

    if (hf_tag.empty() || hf_tag == "latest") {
        hf_tag = "main";
    }

    const std::string model_url = get_model_endpoint() + hf_repo + "/resolve/" + hf_tag + "/" + hf_file;
    const std::string model_path = fs_get_cache_file(srv_audio_clean_cache_name(hf_repo + "_" + hf_file));

    const int status = common_download_file_single(model_url, model_path, bearer_token, offline);
    if (status < 200 || status >= 400) {
        out_err = "failed to download model from " + model_url;
        return false;
    }

    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        out_err = "resolved model path not found after download: " + model_path;
        return false;
    }

    out_path = model_path;
    out_repo_no_tag = hf_repo;
    return true;
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};



//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->get_api_show = [this](const server_http_req &) {
        auto res = create_response();
        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        json data = {
            {
                "model_info", {
                    { "llama.context_length", meta->slot_n_ctx },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", tmpl_default},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})}
        };

        res->ok(data);
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_text_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_responses_to_chatcmpl(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       meta->model_name},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     {
                        {"vocab_type",  meta->model_vocab_type},
                        {"n_vocab",     meta->model_vocab_n_tokens},
                        {"n_ctx_train", meta->model_n_ctx_train},
                        {"n_embd",      meta->model_n_embd_inp},
                        {"n_params",    meta->model_n_params},
                        {"size",        meta->model_size},
                    }},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->post_audio_transcriptions = [this](const server_http_req & req) {
        auto res = create_response();
        const auto pipeline_t0 = std::chrono::steady_clock::now();

        const json body = json::parse(req.body);
        const json generation = json_value(body, "generation", json::object());

        json audio_obj = json::object();
        if (body.contains("audio") && body.at("audio").is_object()) {
            audio_obj = body.at("audio");
        } else if (body.contains("input_audio") && body.at("input_audio").is_object()) {
            audio_obj = body.at("input_audio");
        } else {
            res->error(format_error_response(
                "Request must contain 'audio' or 'input_audio' object with fields {data, format}.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const srv_audio_inproc_raw_payload * inproc_raw_payload = g_srv_audio_inproc_raw_payload;
        std::string audio_data_b64 = json_value(audio_obj, "data", std::string());
        std::string audio_format = json_value(audio_obj, "format", std::string("wav"));
        std::transform(audio_format.begin(), audio_format.end(), audio_format.begin(), [](unsigned char c) {
            return (char) std::tolower(c);
        });
        if (inproc_raw_payload != nullptr && !inproc_raw_payload->audio_format.empty()) {
            audio_format = inproc_raw_payload->audio_format;
            std::transform(audio_format.begin(), audio_format.end(), audio_format.begin(), [](unsigned char c) {
                return (char) std::tolower(c);
            });
        }
        const bool has_raw_audio = inproc_raw_payload != nullptr
            && inproc_raw_payload->audio_bytes != nullptr
            && !inproc_raw_payload->audio_bytes->empty();
        if (!has_raw_audio && audio_data_b64.empty()) {
            res->error(format_error_response(
                "'audio.data' must be non-empty when raw in-process audio bytes are not provided.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (audio_format != "wav" && audio_format != "mp3") {
            res->error(format_error_response(
                "'audio.format' must be 'wav' or 'mp3'.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string mode = body.value("mode", std::string());
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
            return (char) std::tolower(c);
        });
        if (mode.empty()) {
            const bool legacy_enable_diarization = body.value("enable_diarization", false);
            std::string legacy_format = body.value("transcript_format", body.value("output_format", std::string("md")));
            std::transform(legacy_format.begin(), legacy_format.end(), legacy_format.begin(), [](unsigned char c) {
                return (char) std::tolower(c);
            });
            if (legacy_enable_diarization) {
                mode = "transcript";
            } else if (legacy_format == "srt") {
                mode = "subtitle";
            } else {
                mode = "speech";
            }
        }

        const std::string audio_source_path = body.value("audio_source_path", std::string());
        std::filesystem::path output_dir = std::filesystem::path(body.value("output_dir", std::string()));
        if (mode != "timeline") {
            if (output_dir.empty() && !audio_source_path.empty()) {
                output_dir = std::filesystem::path(audio_source_path).parent_path();
            }
            if (output_dir.empty()) {
                output_dir = std::filesystem::current_path();
            }
            std::error_code ec_output_dir;
            std::filesystem::create_directories(output_dir, ec_output_dir);
            if (ec_output_dir) {
                res->error(format_error_response(
                    "Failed to create output_dir: " + output_dir.string(),
                    ERROR_TYPE_SERVER));
                return res;
            }
        }

        std::string output_stem;
        if (!audio_source_path.empty()) {
            output_stem = std::filesystem::path(audio_source_path).stem().string();
        }
        if (output_stem.empty()) {
            output_stem = "audio";
        }

        const std::filesystem::path temp_root = std::filesystem::temp_directory_path() / ("engine_audio_" + srv_audio_make_run_id());
        std::error_code ec_temp_dir;
        std::filesystem::create_directories(temp_root, ec_temp_dir);
        if (ec_temp_dir) {
            res->error(format_error_response(
                "Failed to create temporary directory for audio pipeline",
                ERROR_TYPE_SERVER));
            return res;
        }
        srv_audio_temp_dir_guard temp_dir_guard{temp_root};

        const std::filesystem::path diarization_out_dir = temp_root / "diarization";
        std::filesystem::create_directories(diarization_out_dir, ec_temp_dir);
        if (ec_temp_dir) {
            res->error(format_error_response(
                "Failed to create temporary diarization directory",
                ERROR_TYPE_SERVER));
            return res;
        }

        raw_buffer audio_bytes;
        if (has_raw_audio) {
            audio_bytes = *inproc_raw_payload->audio_bytes;
        } else {
            std::string decoded_audio;
            try {
                decoded_audio = base64::decode(audio_data_b64);
            } catch (const std::exception & ex) {
                res->error(format_error_response(
                    std::string("invalid base64 audio payload: ") + ex.what(),
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            audio_bytes.assign(decoded_audio.begin(), decoded_audio.end());
        }
        const std::filesystem::path audio_path = temp_root / ("input_audio." + audio_format);
        srv_audio_write_binary(audio_path, audio_bytes);

        std::string transcript;
        json words = json::array();

        const std::string requested_backend = body.value("transcription_backend", std::string("auto"));
        std::string transcription_backend = requested_backend;
        std::transform(
            transcription_backend.begin(),
            transcription_backend.end(),
            transcription_backend.begin(),
            [](unsigned char c) { return (char) std::tolower(c); });

        if (transcription_backend.empty() || transcription_backend == "auto" ||
            transcription_backend == "whisper" || transcription_backend == "whisper_cpp") {
            transcription_backend = "whisper_cpp_inproc";
        }
        if (transcription_backend == "whisper_cpp_cli") {
            // Backward-compatible alias.
            transcription_backend = "whisper_cpp_inproc";
        }
        if (transcription_backend != "whisper_cpp_inproc") {
            res->error(format_error_response(
                "Unknown transcription_backend '" + transcription_backend + "'. Supported: auto, whisper_cpp_inproc.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string whisper_model_local = body.value(
            "whisper_model",
            body.value("whisper_model_path", std::string()));
        const std::string whisper_hf_repo = body.value("whisper_hf_repo", std::string());
        const std::string whisper_hf_file = body.value("whisper_hf_file", std::string());
        const bool whisper_offline = generation.value("whisper_offline", body.value("whisper_offline", params.offline));

        const bool whisper_has_local = !whisper_model_local.empty();
        const bool whisper_has_hf = !whisper_hf_repo.empty() || !whisper_hf_file.empty();

        if (whisper_has_local && whisper_has_hf) {
            res->error(format_error_response(
                "Choose one whisper source: local whisper_model/whisper_model_path OR whisper_hf_repo + whisper_hf_file.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!whisper_has_local && !whisper_has_hf) {
            res->error(format_error_response(
                "Missing whisper model source. Provide local whisper_model/whisper_model_path OR whisper_hf_repo + whisper_hf_file.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string whisper_model_path;
        std::string whisper_model_source = "local";
        if (whisper_has_local) {
            whisper_model_path = whisper_model_local;
            if (!std::filesystem::exists(whisper_model_path)) {
                res->error(format_error_response(
                    "whisper_model path not found: " + whisper_model_path,
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            if (whisper_hf_repo.empty() || whisper_hf_file.empty()) {
                res->error(format_error_response(
                    "whisper_hf_repo and whisper_hf_file must both be set when using Hugging Face source.",
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            std::string resolve_err;
            std::string resolved_repo;
            if (!srv_audio_resolve_hf_model(
                    whisper_hf_repo,
                    whisper_hf_file,
                    params.hf_token,
                    whisper_offline,
                    whisper_model_path,
                    resolved_repo,
                    resolve_err)) {
                res->error(format_error_response(
                    "Failed to resolve whisper model from Hugging Face: " + resolve_err,
                    ERROR_TYPE_SERVER));
                return res;
            }
            whisper_model_source = "hf:" + resolved_repo;
        }
        SRV_INF("audio pipeline: whisper model source=%s path=%s\n", whisper_model_source.c_str(), whisper_model_path.c_str());

        const int whisper_threads = generation.value("whisper_threads", body.value("whisper_threads", 0));
        const int whisper_processors = generation.value("whisper_processors", body.value("whisper_processors", 0));
        const int whisper_max_len = generation.value("whisper_max_len", body.value("whisper_max_len", 0));
        const int whisper_audio_ctx = generation.value("whisper_audio_ctx", body.value("whisper_audio_ctx", 0));
        const int whisper_best_of = generation.value("whisper_best_of", body.value("whisper_best_of", 0));
        const int whisper_beam_size = generation.value("whisper_beam_size", body.value("whisper_beam_size", 0));
        const double whisper_temperature = generation.value("whisper_temperature", body.value("whisper_temperature", -1.0));
        const bool whisper_translate = generation.value("whisper_translate", body.value("whisper_translate", false));
        const bool whisper_no_gpu = generation.value("whisper_no_gpu", body.value("whisper_no_gpu", false));
        const int whisper_gpu_device = generation.value("whisper_gpu_device", body.value("whisper_gpu_device", 0));
        const bool whisper_flash_attn = generation.value("whisper_flash_attn", body.value("whisper_flash_attn", true));
        const bool whisper_no_fallback = generation.value("whisper_no_fallback", body.value("whisper_no_fallback", false));
        const bool whisper_suppress_nst = generation.value("whisper_suppress_nst", body.value("whisper_suppress_nst", false));
        const double whisper_word_time_offset_sec = generation.value(
            "whisper_word_time_offset_sec",
            body.value("whisper_word_time_offset_sec", 0.73));
        std::string whisper_language = generation.value("whisper_language", body.value("whisper_language", std::string("auto")));
        std::string whisper_prompt = generation.value("whisper_prompt", body.value("whisper_prompt", std::string()));
        if (whisper_language.empty()) {
            whisper_language = "auto";
        }

        const std::filesystem::path whisper_out_prefix = temp_root / "whisper_transcript";
        const std::filesystem::path whisper_json_path = std::filesystem::path(whisper_out_prefix.string() + ".json");

        std::vector<std::string> whisper_args = {
            "whisper-cli",
            "-m", whisper_model_path,
            "-f", audio_path.string(),
            "-ojf",
            "-of", whisper_out_prefix.string(),
            "-np",
            "-l", whisper_language,
        };

        if (whisper_threads > 0) {
            whisper_args.push_back("-t");
            whisper_args.push_back(std::to_string(whisper_threads));
        }
        if (whisper_processors > 0) {
            whisper_args.push_back("-p");
            whisper_args.push_back(std::to_string(whisper_processors));
        }
        if (whisper_max_len > 0) {
            whisper_args.push_back("-ml");
            whisper_args.push_back(std::to_string(whisper_max_len));
        }
        if (whisper_audio_ctx > 0) {
            whisper_args.push_back("-ac");
            whisper_args.push_back(std::to_string(whisper_audio_ctx));
        }
        if (whisper_best_of > 0) {
            whisper_args.push_back("-bo");
            whisper_args.push_back(std::to_string(whisper_best_of));
        }
        if (whisper_beam_size > 0) {
            whisper_args.push_back("-bs");
            whisper_args.push_back(std::to_string(whisper_beam_size));
        }
        if (whisper_temperature >= 0.0) {
            whisper_args.push_back("-tp");
            whisper_args.push_back(std::to_string(whisper_temperature));
        }
        if (whisper_translate) {
            whisper_args.push_back("-tr");
        }
        if (whisper_no_fallback) {
            whisper_args.push_back("-nf");
        }
        if (whisper_suppress_nst) {
            whisper_args.push_back("-sns");
        }
        if (!whisper_prompt.empty()) {
            whisper_args.push_back("--prompt");
            whisper_args.push_back(whisper_prompt);
        }

        if (whisper_no_gpu) {
            whisper_args.push_back("-ng");
        } else {
            whisper_args.push_back("-dev");
            whisper_args.push_back(std::to_string(whisper_gpu_device));
            whisper_args.push_back(whisper_flash_attn ? "-fa" : "-nfa");
        }

        const auto whisper_t0 = std::chrono::steady_clock::now();
        const int whisper_rc = srv_audio_run_inproc_main(whisper_cli_inproc_main, whisper_args);
        const auto whisper_t1 = std::chrono::steady_clock::now();
        const double whisper_elapsed_sec = std::chrono::duration<double>(whisper_t1 - whisper_t0).count();
        if (whisper_rc != 0) {
            res->error(format_error_response(
                "whisper in-process transcription failed with exit code " + std::to_string(whisper_rc),
                ERROR_TYPE_SERVER));
            return res;
        }
        if (!std::filesystem::exists(whisper_json_path)) {
            res->error(format_error_response(
                "whisper JSON output not found: " + whisper_json_path.string(),
                ERROR_TYPE_SERVER));
            return res;
        }

        const auto whisper_json_parse = srv_audio_parse_json_utf8_safe(
            srv_audio_read_text(whisper_json_path));
        size_t whisper_text_utf8_repairs = 0;
        const auto parsed_whisper = srv_audio_parse_whisper_json(
            whisper_json_parse.value,
            whisper_json_parse.byte_escaped_strings,
            &whisper_text_utf8_repairs);
        transcript = parsed_whisper.transcript;
        words = parsed_whisper.words;
        json whisper_pieces = parsed_whisper.pieces;
        srv_audio_shift_word_timestamps(words, whisper_word_time_offset_sec, /* source_audio_seconds */ -1.0);
        srv_audio_shift_piece_timestamps(whisper_pieces, whisper_word_time_offset_sec, /* source_audio_seconds */ -1.0);
        transcript = srv_audio_sanitize_transcript_text(transcript);

        srv_audio_remove_if_exists(whisper_json_path);

        const double seconds_per_token = body.value("seconds_per_timeline_token", 0.08);
        double source_audio_seconds = body.value("source_audio_seconds", -1.0);
        if (words.empty()) {
            words = srv_audio_fallback_words(transcript, source_audio_seconds, seconds_per_token);
        }
        if ((!whisper_pieces.is_array() || whisper_pieces.empty()) && words.is_array() && !words.empty()) {
            whisper_pieces = srv_audio_build_fallback_whisper_pieces(words);
        }
        if (source_audio_seconds <= 0.0 && words.is_array() && !words.empty()) {
            const auto & last = words.back();
            if (last.is_object()) {
                source_audio_seconds = std::max(0.0, json_value(last, "end_sec", 0.0));
            }
        }
        json segments = srv_audio_build_segments(words);
        srv_audio_add_hms_fields(words);
        srv_audio_add_hms_fields(whisper_pieces);
        srv_audio_add_hms_fields(segments);
        if (mode != "subtitle" && mode != "speech" && mode != "transcript" && mode != "timeline") {
            res->error(format_error_response(
                "mode must be one of: subtitle, speech, transcript, timeline.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (mode == "timeline") {
            const auto pipeline_t1 = std::chrono::steady_clock::now();
            const double total_elapsed_sec = std::chrono::duration<double>(pipeline_t1 - pipeline_t0).count();
            json root = {
                {"mode", "timeline"},
                {"diarization", {{"enabled", false}}},
                {"timeline", {
                    {"transcript", transcript},
                    {"words", words},
                    {"whisper_pieces", whisper_pieces},
                    {"segments", segments},
                }},
                {"stats", {
                    {"num_words", (int64_t) words.size()},
                    {"num_whisper_pieces", whisper_pieces.is_array() ? (int64_t) whisper_pieces.size() : int64_t(0)},
                    {"num_segments", (int64_t) segments.size()},
                    {"source_audio_seconds", source_audio_seconds},
                    {"whisper_utf8_repairs", (int64_t) whisper_text_utf8_repairs},
                }},
                {"timings_sec", {
                    {"total", total_elapsed_sec},
                    {"whisper", whisper_elapsed_sec},
                }},
            };
            res->ok(root);
            return res;
        }

        std::string custom = "default";
        if (body.contains("custom")) {
            const json & v = body.at("custom");
            if (v.is_string()) {
                custom = v.get<std::string>();
            } else if (v.is_number_integer()) {
                custom = std::to_string(v.get<int64_t>());
            } else if (v.is_number_unsigned()) {
                custom = std::to_string(v.get<uint64_t>());
            } else if (v.is_number_float()) {
                std::ostringstream oss;
                oss << v.get<double>();
                custom = oss.str();
            } else if (v.is_null()) {
                custom = "default";
            } else {
                res->error(format_error_response(
                    "custom must be a string, number, or null.",
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        custom = srv_audio_collapse_ws(custom);
        if (custom.empty()) {
            custom = "default";
        }
        std::string custom_lower = custom;
        std::transform(custom_lower.begin(), custom_lower.end(), custom_lower.begin(), [](unsigned char c) {
            return (char) std::tolower(c);
        });
        const bool custom_is_default = custom_lower == "default" || custom_lower == "auto";

        auto parse_positive_double = [&](double & out_value) -> bool {
            if (custom_is_default) {
                return false;
            }
            try {
                size_t pos = 0;
                const double parsed = std::stod(custom, &pos);
                if (pos != custom.size() || !std::isfinite(parsed) || parsed <= 0.0) {
                    return false;
                }
                out_value = parsed;
                return true;
            } catch (...) {
                return false;
            }
        };

        auto parse_positive_int = [&](int & out_value) -> bool {
            if (custom_is_default) {
                return false;
            }
            try {
                size_t pos = 0;
                const long long parsed = std::stoll(custom, &pos);
                if (pos != custom.size() || parsed <= 0 || parsed > INT_MAX) {
                    return false;
                }
                out_value = (int) parsed;
                return true;
            } catch (...) {
                return false;
            }
        };

        if (mode == "subtitle") {
            json subtitle_segments = segments;
            double subtitle_window_sec = -1.0;
            if (!custom_is_default) {
                if (!parse_positive_double(subtitle_window_sec)) {
                    res->error(format_error_response(
                        "For mode=subtitle, custom must be 'default' or a positive number of seconds.",
                        ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                subtitle_segments = srv_audio_build_subtitle_segments_window(words, subtitle_window_sec);
            }

            const std::filesystem::path final_output_path = output_dir / (output_stem + ".srt");
            srv_audio_write_text(final_output_path, srv_audio_segments_to_srt(subtitle_segments));

            const auto pipeline_t1 = std::chrono::steady_clock::now();
            const double total_elapsed_sec = std::chrono::duration<double>(pipeline_t1 - pipeline_t0).count();
            json root = {
                {"mode", "subtitle"},
                {"custom", custom},
                {"diarization", {{"enabled", false}}},
                {"output", {
                    {"path", final_output_path.string()},
                    {"format", "srt"},
                    {"speaker_turns", false},
                }},
                {"stats", {
                    {"num_words", (int64_t) words.size()},
                    {"num_segments", (int64_t) subtitle_segments.size()},
                    {"source_audio_seconds", source_audio_seconds},
                }},
                {"timings_sec", {
                    {"total", total_elapsed_sec},
                    {"whisper", whisper_elapsed_sec},
                }},
            };
            res->ok(root);
            return res;
        }

        if (mode == "speech") {
            std::string speech_md;
            if (custom_is_default) {
                speech_md = srv_audio_transcript_to_md(transcript);
            } else {
                double speech_chunk_sec = -1.0;
                if (!parse_positive_double(speech_chunk_sec)) {
                    res->error(format_error_response(
                        "For mode=speech, custom must be 'default' or a positive number of seconds.",
                        ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                const json sentence_segments = srv_audio_build_sentence_segments(words);
                speech_md = srv_audio_transcript_to_md_chunked(sentence_segments, speech_chunk_sec);
                if (speech_md.empty()) {
                    speech_md = srv_audio_transcript_to_md(transcript);
                }
            }

            const std::filesystem::path final_output_path = output_dir / (output_stem + ".md");
            srv_audio_write_text(final_output_path, speech_md);

            const auto pipeline_t1 = std::chrono::steady_clock::now();
            const double total_elapsed_sec = std::chrono::duration<double>(pipeline_t1 - pipeline_t0).count();
            json root = {
                {"mode", "speech"},
                {"custom", custom},
                {"diarization", {{"enabled", false}}},
                {"output", {
                    {"path", final_output_path.string()},
                    {"format", "md"},
                    {"speaker_turns", false},
                }},
                {"stats", {
                    {"num_words", (int64_t) words.size()},
                    {"num_segments", (int64_t) segments.size()},
                    {"source_audio_seconds", source_audio_seconds},
                }},
                {"timings_sec", {
                    {"total", total_elapsed_sec},
                    {"whisper", whisper_elapsed_sec},
                }},
            };
            res->ok(root);
            return res;
        }

        const std::string diar_backend_requested = body.value("diarization_backend", std::string("native_cpp"));
        if (!diar_backend_requested.empty() &&
            diar_backend_requested != "native_cpp" &&
            diar_backend_requested != "auto" &&
            diar_backend_requested != "realtime" &&
            diar_backend_requested != "sortformer") {
            res->error(format_error_response(
                "Unsupported diarization backend. Supported values: auto, native_cpp, realtime, sortformer.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (!custom_is_default) {
            int ignored_num_speakers = -1;
            if (!parse_positive_int(ignored_num_speakers)) {
                res->error(format_error_response(
                    "For mode=transcript, custom must be 'auto'/'default' or a positive integer speaker count.",
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            res->error(format_error_response(
                "Realtime diarization backend does not support overriding the speaker count.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string diar_model_path;
        std::string diar_model_error;
        if (!srv_audio_resolve_realtime_model_path(body, diar_model_path, diar_model_error)) {
            res->error(format_error_response(diar_model_error, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string diar_device = body.value("diarization_device", std::string("auto"));
        const double diarization_feed_ms = body.value("diarization_feed_ms", 100.0);
        if (!(diarization_feed_ms > 0.0) || !std::isfinite(diarization_feed_ms)) {
            res->error(format_error_response(
                "diarization_feed_ms must be a positive finite number.",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const double speaker_seg_max_gap_sec = body.value("speaker_seg_max_gap_sec", -1.0);
        const int speaker_seg_max_words = body.value("speaker_seg_max_words", 0);
        const double speaker_seg_max_duration_sec = body.value("speaker_seg_max_duration_sec", -1.0);
        const bool speaker_seg_split_on_hard_break = body.value("speaker_seg_split_on_hard_break", false);

        json speaker_spans = json::array();
        double diar_elapsed_sec = 0.0;
        std::string resolved_diar_backend_name;
        std::string diar_error;
        if (!srv_audio_collect_realtime_speaker_spans(
                audio_path,
                diar_model_path,
                diar_backend_requested,
                diar_device,
                diarization_feed_ms,
                speaker_spans,
                diar_elapsed_sec,
                resolved_diar_backend_name,
                diar_error)) {
            res->error(format_error_response(
                "native realtime diarization failed: " + diar_error,
                ERROR_TYPE_SERVER));
            return res;
        }

        json speaker_segments = srv_audio_build_realtime_speaker_segments(
            whisper_pieces,
            words,
            speaker_spans,
            speaker_seg_max_gap_sec,
            speaker_seg_max_words,
            speaker_seg_max_duration_sec,
            speaker_seg_split_on_hard_break);

        srv_audio_add_hms_fields(speaker_spans);
        srv_audio_add_hms_fields(speaker_segments);

        const std::filesystem::path final_output_path = output_dir / (output_stem + ".md");
        srv_audio_write_text(final_output_path, srv_audio_speaker_segments_to_md(speaker_segments));

        const auto pipeline_t1 = std::chrono::steady_clock::now();
        const double total_elapsed_sec = std::chrono::duration<double>(pipeline_t1 - pipeline_t0).count();
        json root = {
            {"mode", "transcript"},
            {"custom", custom},
            {"diarization", {
                {"enabled", true},
                {"backend", diar_backend_requested == "sortformer" ? "sortformer" : "realtime"},
                {"runtime_backend", resolved_diar_backend_name.empty() ? diar_device : resolved_diar_backend_name},
                {"model_path", diar_model_path},
                {"feed_ms", diarization_feed_ms},
                {"speaker_count", "model_default"},
            }},
            {"output", {
                {"path", final_output_path.string()},
                {"format", "md"},
                {"speaker_turns", true},
            }},
            {"stats", {
                {"num_words", (int64_t) words.size()},
                {"num_whisper_pieces", whisper_pieces.is_array() ? (int64_t) whisper_pieces.size() : int64_t(0)},
                {"num_segments", (int64_t) segments.size()},
                {"num_speaker_spans", (int64_t) speaker_spans.size()},
                {"num_speaker_segments", (int64_t) speaker_segments.size()},
                {"source_audio_seconds", source_audio_seconds},
            }},
            {"speaker_spans", speaker_spans},
            {"timings_sec", {
                {"total", total_elapsed_sec},
                {"whisper", whisper_elapsed_sec},
                {"diarization", diar_elapsed_sec},
            }},
        };

        res->ok(root);
        return res;
    };

    this->post_audio_transcriptions_raw =
        [this](const std::string & audio_format, const raw_buffer & audio_bytes, const json & body_in) {
            if (!this->post_audio_transcriptions) {
                return server_http_res_ptr{};
            }

            json body = body_in;
            if (!body.is_object()) {
                body = json::object();
            }

            json & audio_obj = body["audio"];
            if (!audio_obj.is_object()) {
                audio_obj = json::object();
            }

            std::string fmt = audio_format;
            std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
                return (char) std::tolower(c);
            });
            if (fmt.empty()) {
                fmt = "wav";
            }

            audio_obj["format"] = fmt;

            srv_audio_inproc_raw_payload raw_payload{&audio_bytes, fmt};
            srv_audio_inproc_raw_scope raw_scope(&raw_payload);

            const std::function<bool()> should_stop = []() -> bool { return false; };
            const server_http_req req{
                {},
                {},
                "/v1/audio/transcriptions",
                body.dump(),
                should_stop
            };

            return this->post_audio_transcriptions(req);
        };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}
