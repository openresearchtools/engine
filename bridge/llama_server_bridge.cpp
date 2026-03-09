#include "llama_server_bridge.h"

#include "common.h"
#include "server-common.h"
#include "server-context.h"
#include "tools/realtime/backend-factory.h"
#include "tools/realtime/stream-manager.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <climits>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef LLAMA_SERVER_BRIDGE_USE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}
#endif

struct llama_server_bridge {
    common_params params = common_params();
    server_context ctx = server_context();
    std::unique_ptr<server_routes> routes;

    std::thread loop_thread;

    mutable std::mutex error_mutex;
    std::string last_error;

    std::string model_name;
    int32_t primary_device_index = -1;
    std::string primary_device_name;
    bool primary_device_is_gpu = false;
    bool backend_acquired = false;
};

struct llama_server_bridge_realtime {
    common_params params = common_params();
    llama::realtime::stream_manager manager;
    int64_t session_id = 0;
    bool backend_acquired = false;
    std::shared_ptr<llama::realtime::loaded_backend_model> loaded_model;

    mutable std::mutex error_mutex;
    std::string last_error;
};

struct llama_server_bridge_realtime_model_impl {
    common_params params = common_params();
    bool backend_acquired = false;
    int32_t backend_kind = LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_AUTO;
    std::string model_path;
    std::string resolved_runtime_backend_name;
    llama::realtime::stream_session_config default_session_cfg = {};

    mutable std::mutex error_mutex;
    std::string last_error;
};

struct audio_session_event_record {
    uint64_t seq_no = 0;
    int32_t kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE;
    uint32_t flags = 0;
    uint64_t start_sample = 0;
    uint64_t end_sample = 0;
    int32_t speaker_id = -1;
    uint32_t item_id = 0;
    std::string text;
    std::string detail;
};

struct owned_bridge_params {
    llama_server_bridge_params raw = {};
    std::string model_path;
    std::string mmproj_path;
    std::string devices;
    std::string tensor_split;

    void assign_defaults(void) {
        raw = llama_server_bridge_default_params();
    }

    void assign(const llama_server_bridge_params * params) {
        assign_defaults();
        if (params == nullptr) {
            return;
        }
        raw = *params;
        model_path = params->model_path != nullptr ? params->model_path : "";
        mmproj_path = params->mmproj_path != nullptr ? params->mmproj_path : "";
        devices = params->devices != nullptr ? params->devices : "";
        tensor_split = params->tensor_split != nullptr ? params->tensor_split : "";
    }

    llama_server_bridge_params borrow(void) const {
        auto out = raw;
        out.model_path = model_path.empty() ? nullptr : model_path.c_str();
        out.mmproj_path = mmproj_path.empty() ? nullptr : mmproj_path.c_str();
        out.devices = devices.empty() ? nullptr : devices.c_str();
        out.tensor_split = tensor_split.empty() ? nullptr : tensor_split.c_str();
        return out;
    }
};

struct owned_realtime_params {
    llama_server_bridge_realtime_params raw = {};
    std::string model_path;
    std::string backend_name;

    void assign_defaults(void) {
        raw = llama_server_bridge_default_realtime_params();
    }

    void assign(const llama_server_bridge_realtime_params * params) {
        assign_defaults();
        if (params == nullptr) {
            return;
        }
        raw = *params;
        model_path = params->model_path != nullptr ? params->model_path : "";
        backend_name = params->backend_name != nullptr ? params->backend_name : "";
    }

    llama_server_bridge_realtime_params borrow(void) const {
        auto out = raw;
        out.model_path = model_path.empty() ? nullptr : model_path.c_str();
        out.backend_name = backend_name.empty() ? nullptr : backend_name.c_str();
        return out;
    }
};

struct owned_audio_transcription_params {
    owned_bridge_params bridge_params;
    std::string metadata_json;
    int32_t mode = LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE;
    owned_realtime_params realtime_params;

    void assign(const llama_server_bridge_audio_transcription_params * params) {
        bridge_params.assign(params != nullptr ? &params->bridge_params : nullptr);
        metadata_json = (params != nullptr && params->metadata_json != nullptr) ? params->metadata_json : "";
        mode = params != nullptr
            ? params->mode
            : LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE;
        realtime_params.assign(params != nullptr ? &params->realtime_params : nullptr);
    }
};

struct llama_server_bridge_audio_session {
    llama_server_bridge_audio_session_params params = {};
    std::vector<float> audio_samples;
    std::vector<size_t> chunk_sizes;
    bool audio_flushed = false;

    uint64_t next_seq_no = 1;
    uint32_t next_item_id = 1;

    std::deque<audio_session_event_record> event_queue;
    mutable std::mutex mutex;
    std::condition_variable cv;

    mutable std::mutex error_mutex;
    std::string last_error;

    llama_server_bridge_realtime * diarization_bridge = nullptr;
    bool diarization_started = false;
    bool diarization_stopped = false;

    llama_server_bridge_realtime * transcription_bridge = nullptr;
    owned_audio_transcription_params transcription_params = {};
    bool transcription_requested = false;
    bool transcription_started = false;
    bool transcription_running = false;
    bool transcription_completed = false;
    bool transcription_stop_requested = false;
    bool transcription_native_realtime = false;
    std::thread transcription_thread;
};

static std::mutex g_backend_mutex;
static int g_backend_refcount = 0;
static std::mutex g_realtime_model_cache_mutex;
struct cached_realtime_model_entry {
    std::shared_ptr<llama::realtime::loaded_backend_model> model;
    std::shared_ptr<void> backend_hold;
};
static std::unordered_map<std::string, std::shared_ptr<cached_realtime_model_entry>> g_realtime_model_cache;

static void acquire_backend(const common_params & params);
static void release_backend();

static bool realtime_trace_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("LLAMA_BRIDGE_RT_TRACE");
        enabled = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

#define RT_TRACE(...)                                                      \
    do {                                                                   \
        if (realtime_trace_enabled()) {                                    \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
            std::fflush(stderr);                                           \
        }                                                                  \
    } while (0)

static std::string make_realtime_model_cache_key(const llama::realtime::backend_model_params & params) {
    std::ostringstream out;
    out << static_cast<int>(params.kind) << '\n';
    out << params.backend_name << '\n';
    out << params.model_path;
    return out.str();
}

static void prune_realtime_model_cache_locked(void) {
    (void) g_realtime_model_cache;
}

static std::shared_ptr<llama::realtime::loaded_backend_model> load_backend_model_maybe_cached(
    const llama::realtime::backend_model_params & params,
    bool * out_cache_hit = nullptr) {

    if (out_cache_hit != nullptr) {
        *out_cache_hit = false;
    }

    if (!llama::realtime::backend_supports_model_preload(params.kind)) {
        RT_TRACE("rt_load_model: preload unsupported kind=%d", static_cast<int>(params.kind));
        return llama::realtime::load_backend_model(params);
    }

    const std::string cache_key = make_realtime_model_cache_key(params);
    {
        std::lock_guard<std::mutex> lock(g_realtime_model_cache_mutex);
        auto it = g_realtime_model_cache.find(cache_key);
        if (it != g_realtime_model_cache.end()) {
            RT_TRACE("rt_load_model: cache hit kind=%d model=%s backend=%s",
                static_cast<int>(params.kind),
                params.model_path.c_str(),
                params.backend_name.c_str());
            if (out_cache_hit != nullptr) {
                *out_cache_hit = true;
            }
            return it->second->model;
        }
    }

    common_params init_params = common_params();
    RT_TRACE("rt_load_model: cache miss kind=%d model=%s backend=%s",
        static_cast<int>(params.kind),
        params.model_path.c_str(),
        params.backend_name.c_str());
    acquire_backend(init_params);
    try {
        RT_TRACE("rt_load_model: backend acquired");
        auto loaded = llama::realtime::load_backend_model(params);
        RT_TRACE("rt_load_model: model loaded");
        auto entry = std::make_shared<cached_realtime_model_entry>();
        entry->model = std::move(loaded);
        entry->backend_hold = std::shared_ptr<void>(
            reinterpret_cast<void *>(static_cast<uintptr_t>(1)),
            [](void *) {
                release_backend();
            });

        std::lock_guard<std::mutex> lock(g_realtime_model_cache_mutex);
        auto it = g_realtime_model_cache.find(cache_key);
        if (it != g_realtime_model_cache.end()) {
            RT_TRACE("rt_load_model: duplicate cache fill kind=%d", static_cast<int>(params.kind));
            entry.reset();
            if (out_cache_hit != nullptr) {
                *out_cache_hit = true;
            }
            return it->second->model;
        }
        g_realtime_model_cache[cache_key] = entry;
        prune_realtime_model_cache_locked();
        RT_TRACE("rt_load_model: cache store complete size=%zu", g_realtime_model_cache.size());
        return entry->model;
    } catch (...) {
        RT_TRACE("rt_load_model: exception path, releasing backend");
        release_backend();
        throw;
    }
}

static void set_realtime_error(llama_server_bridge_realtime * bridge, const std::string & message) {
    if (bridge == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(bridge->error_mutex);
    bridge->last_error = message;
}

static void set_realtime_model_error(llama_server_bridge_realtime_model_impl * model, const std::string & message) {
    if (model == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(model->error_mutex);
    model->last_error = message;
}

template <typename T, typename = void>
struct has_post_audio_transcriptions : std::false_type {};

template <typename T>
struct has_post_audio_transcriptions<T, std::void_t<decltype(&T::post_audio_transcriptions)>> : std::true_type {};

using audio_transcriptions_raw_handler_t = std::function<server_http_res_ptr(
    const std::string &,
    const raw_buffer &,
    const json &)>;

template <typename T, typename = void>
struct has_post_audio_transcriptions_raw : std::false_type {};

template <typename T>
struct has_post_audio_transcriptions_raw<T, std::void_t<decltype(&T::post_audio_transcriptions_raw)>> : std::true_type {};

static server_http_context::handler_t resolve_audio_transcriptions_handler(server_routes * routes) {
    if (routes == nullptr) {
        return {};
    }

    if constexpr (has_post_audio_transcriptions<server_routes>::value) {
        return routes->post_audio_transcriptions;
    }

    return {};
}

static audio_transcriptions_raw_handler_t resolve_audio_transcriptions_raw_handler(server_routes * routes) {
    if (routes == nullptr) {
        return {};
    }

    if constexpr (has_post_audio_transcriptions_raw<server_routes>::value) {
        return routes->post_audio_transcriptions_raw;
    }

    return {};
}

static char * copy_to_c_string(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

static std::string trim_copy(const std::string & s) {
    size_t begin = 0;
    size_t end = s.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        begin += 1;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        end -= 1;
    }

    return s.substr(begin, end - begin);
}

static std::vector<std::string> split_csv(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : text) {
        if (ch == ',') {
            out.push_back(trim_copy(cur));
            cur.clear();
            continue;
        }
        cur.push_back(ch);
    }
    out.push_back(trim_copy(cur));
    return out;
}

static bool parse_i32(const std::string & text, int32_t * out) {
    if (out == nullptr || text.empty()) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE) {
        return false;
    }
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    if (value < INT32_MIN || value > INT32_MAX) {
        return false;
    }
    *out = static_cast<int32_t>(value);
    return true;
}

static bool parse_float32(const std::string & text, float * out) {
    if (out == nullptr || text.empty()) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    if (errno == ERANGE) {
        return false;
    }
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

static bool parse_devices_csv(
    const char * devices_csv,
    std::vector<ggml_backend_dev_t> & out_devices,
    std::string & error) {

    out_devices.clear();
    if (devices_csv == nullptr) {
        // Default to CPU-only unless the caller explicitly selects devices.
        out_devices.push_back(nullptr);
        return true;
    }
    const std::string raw = trim_copy(devices_csv);
    if (raw.empty()) {
        // Empty devices behaves the same as unset: CPU-only.
        out_devices.push_back(nullptr);
        return true;
    }

    const size_t device_count = ggml_backend_dev_count();
    const auto tokens = split_csv(raw);
    if (tokens.empty()) {
        error = "devices list is empty";
        return false;
    }

    bool saw_none = false;
    for (const std::string & token : tokens) {
        if (token.empty()) {
            error = "devices list contains an empty entry";
            return false;
        }
        if (token == "none" || token == "NONE") {
            saw_none = true;
            continue;
        }
        if (saw_none) {
            error = "devices list cannot mix 'none' with concrete devices";
            return false;
        }

        ggml_backend_dev_t dev = nullptr;
        int32_t idx = -1;
        if (parse_i32(token, &idx)) {
            if (idx < 0 || static_cast<size_t>(idx) >= device_count) {
                error = "invalid device index: " + token;
                return false;
            }
            dev = ggml_backend_dev_get(static_cast<size_t>(idx));
        } else {
            dev = ggml_backend_dev_by_name(token.c_str());
        }

        if (dev == nullptr) {
            error = "invalid device: " + token;
            return false;
        }
        out_devices.push_back(dev);
    }

    if (saw_none) {
        out_devices.clear();
    }
    // llama_model_params.devices expects NULL-terminated device list
    out_devices.push_back(nullptr);

    return true;
}

static bool parse_tensor_split_csv(
    const char * tensor_split_csv,
    float out_tensor_split[128],
    std::string & error) {

    if (tensor_split_csv == nullptr) {
        return true;
    }
    const std::string raw = trim_copy(tensor_split_csv);
    if (raw.empty()) {
        return true;
    }

    const auto tokens = split_csv(raw);
    if (tokens.empty()) {
        error = "tensor_split list is empty";
        return false;
    }

    const size_t max_devices = llama_max_devices();
    if (tokens.size() > max_devices) {
        error = "tensor_split has more entries than available max devices";
        return false;
    }

    std::fill(out_tensor_split, out_tensor_split + 128, 0.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
        float value = 0.0f;
        if (!parse_float32(tokens[i], &value)) {
            error = "invalid tensor_split value: " + tokens[i];
            return false;
        }
        if (value < 0.0f) {
            error = "tensor_split values must be >= 0";
            return false;
        }
        out_tensor_split[i] = value;
    }

    return true;
}

static bool parse_split_mode(int32_t input, enum llama_split_mode * out_mode, std::string & error) {
    if (out_mode == nullptr) {
        error = "split mode output is null";
        return false;
    }
    if (input < 0) {
        return true;
    }
    switch (input) {
        case 0:
            *out_mode = LLAMA_SPLIT_MODE_NONE;
            return true;
        case 1:
            *out_mode = LLAMA_SPLIT_MODE_LAYER;
            return true;
        case 2:
            *out_mode = LLAMA_SPLIT_MODE_ROW;
            return true;
        default:
            error = "invalid split_mode, expected -1/0/1/2";
            return false;
    }
}

static bool is_gpu_device_type(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static int32_t find_backend_device_index(ggml_backend_dev_t target) {
    if (target == nullptr) {
        return -1;
    }
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        if (ggml_backend_dev_get(i) == target) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

static bool lookup_backend_device_by_index(int32_t index, std::string & out_name, bool & out_is_gpu) {
    out_name.clear();
    out_is_gpu = false;

    if (index < 0) {
        return false;
    }

    const size_t n = ggml_backend_dev_count();
    if (static_cast<size_t>(index) >= n) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_get(static_cast<size_t>(index));
    if (dev == nullptr) {
        return false;
    }

    const char * dev_name = ggml_backend_dev_name(dev);
    if (dev_name != nullptr) {
        out_name = dev_name;
    }
    out_is_gpu = is_gpu_device_type(ggml_backend_dev_type(dev));
    return true;
}

static bool parse_gpu_selector_text(const std::string & raw, int32_t & out_index) {
    const std::string trimmed = trim_copy(raw);
    if (trimmed.empty()) {
        return false;
    }

    int32_t idx = -1;
    if (parse_i32(trimmed, &idx) && idx >= 0) {
        out_index = idx;
        return true;
    }

    std::string lowered = trimmed;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (lowered.rfind("gpu", 0) != 0) {
        return false;
    }

    std::string suffix = trim_copy(lowered.substr(3));
    if (!suffix.empty() && (suffix[0] == ':' || suffix[0] == '=')) {
        suffix = trim_copy(suffix.substr(1));
    }
    if (suffix.empty()) {
        return false;
    }

    if (!parse_i32(suffix, &idx) || idx < 0) {
        return false;
    }
    out_index = idx;
    return true;
}

static bool parse_audio_gpu_override(const json & body, int32_t & out_index) {
    if (!body.is_object()) {
        return false;
    }

    auto parse_value = [&](const json & value) -> bool {
        if (value.is_number_integer()) {
            const int64_t v = value.get<int64_t>();
            if (v < 0 || v > INT32_MAX) {
                return false;
            }
            out_index = static_cast<int32_t>(v);
            return true;
        }
        if (value.is_string()) {
            return parse_gpu_selector_text(value.get<std::string>(), out_index);
        }
        return false;
    };

    const auto it_gpu = body.find("gpu");
    if (it_gpu != body.end() && parse_value(*it_gpu)) {
        return true;
    }

    const auto it_device = body.find("device");
    if (it_device != body.end() && parse_value(*it_device)) {
        return true;
    }

    return false;
}

static void set_env_if_unset(const char * key, const std::string & value) {
    if (key == nullptr || key[0] == '\0' || value.empty()) {
        return;
    }
#if defined(_WIN32)
    // Use process-wide env APIs on Windows so all runtimes in-process
    // (including ggml/llama DLLs) observe the same value.
    (void) SetEnvironmentVariableA(key, value.c_str());
    (void) _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

static bool is_devices_unset_or_none(const char * devices_csv) {
    if (devices_csv == nullptr) {
        return true;
    }
    const std::string raw = trim_copy(devices_csv);
    return raw.empty() || raw == "none" || raw == "NONE";
}

static bool is_devices_unset(const char * devices_csv) {
    if (devices_csv == nullptr) {
        return true;
    }
    return trim_copy(devices_csv).empty();
}

static int32_t find_first_gpu_backend_device_index(void) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        if (is_gpu_device_type(ggml_backend_dev_type(dev))) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

static bool looks_like_zero_initialized_params(const llama_server_bridge_params * p) {
    if (p == nullptr) {
        return false;
    }
    return p->n_ctx == 0
        && p->n_batch == 0
        && p->n_ubatch == 0
        && p->n_parallel == 0
        && p->n_threads == 0
        && p->n_threads_batch == 0
        && p->n_gpu_layers == 0
        && p->gpu == 0
        && p->no_kv_offload == 0
        && p->mmproj_use_gpu == 0
        && p->cache_ram_mib == 0
        && p->seed == 0
        && p->ctx_shift == 0
        && p->kv_unified == 0
        && p->split_mode == 0
        && p->embedding == 0
        && p->reranking == 0
        && p->pooling_type == 0;
}

static void apply_audio_runtime_device_defaults(const llama_server_bridge * bridge, json & body) {
    if (bridge == nullptr || !body.is_object()) {
        return;
    }

    int32_t override_gpu_index = -1;
    const bool has_gpu_override = parse_audio_gpu_override(body, override_gpu_index);
    bool override_is_gpu = false;
    std::string override_device_name;
    const bool override_valid = has_gpu_override
        && lookup_backend_device_by_index(override_gpu_index, override_device_name, override_is_gpu);

    const auto it_no_gpu = body.find("whisper_no_gpu");
    const bool whisper_no_gpu = it_no_gpu != body.end() ? json_value(body, "whisper_no_gpu", false) : false;

    if (body.find("whisper_gpu_device") == body.end()) {
        if (!whisper_no_gpu && override_valid && override_is_gpu) {
            body["whisper_gpu_device"] = override_gpu_index;
        } else if (!whisper_no_gpu && bridge->primary_device_is_gpu && bridge->primary_device_index >= 0) {
            body["whisper_gpu_device"] = bridge->primary_device_index;
        } else if (it_no_gpu == body.end() && (!override_valid || !override_is_gpu) && !bridge->primary_device_is_gpu) {
            body["whisper_no_gpu"] = true;
        }
    }

    if (body.find("diarization_device") == body.end()) {
        if (override_valid && override_is_gpu && !override_device_name.empty()) {
            body["diarization_device"] = override_device_name;
        } else if (bridge->primary_device_is_gpu && !bridge->primary_device_name.empty()) {
            body["diarization_device"] = bridge->primary_device_name;
        } else {
            body["diarization_device"] = "cpu";
        }
    }
}

static bool is_valid_pooling_type(int32_t pooling_type) {
    return pooling_type >= static_cast<int32_t>(LLAMA_POOLING_TYPE_UNSPECIFIED)
        && pooling_type <= static_cast<int32_t>(LLAMA_POOLING_TYPE_RANK);
}

static void set_bridge_error(llama_server_bridge * bridge, const std::string & message) {
    if (bridge == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(bridge->error_mutex);
    bridge->last_error = message;
}

static void acquire_backend(const common_params & params) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    if (g_backend_refcount == 0) {
        common_init();
        ggml_backend_load_all();
        llama_backend_init();
        llama_numa_init(params.numa);
    }
    g_backend_refcount += 1;
}

static void release_backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    if (g_backend_refcount <= 0) {
        return;
    }
    g_backend_refcount -= 1;
    if (g_backend_refcount == 0) {
        llama_backend_free();
    }
}

static std::string normalize_error(const std::string & msg) {
    if (msg.empty()) {
        return "unknown bridge error";
    }
    return msg;
}

static void set_audio_session_error(
    llama_server_bridge_audio_session * session,
    const std::string & message) {

    if (session == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->error_mutex);
    session->last_error = message;
}

static uint64_t seconds_to_samples(double sec, uint32_t sample_rate_hz) {
    if (!(sec > 0.0) || !std::isfinite(sec) || sample_rate_hz == 0) {
        return 0;
    }
    const double scaled = sec * static_cast<double>(sample_rate_hz);
    if (scaled <= 0.0) {
        return 0;
    }
    if (scaled >= static_cast<double>(UINT64_MAX)) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(std::llround(scaled));
}

static void push_audio_session_event_locked(
    llama_server_bridge_audio_session * session,
    audio_session_event_record event) {

    if (session == nullptr) {
        return;
    }
    event.seq_no = session->next_seq_no++;
    if (session->params.event_queue_capacity > 0) {
        while (session->event_queue.size() >= session->params.event_queue_capacity) {
            session->event_queue.pop_front();
        }
    }
    session->event_queue.push_back(std::move(event));
    session->cv.notify_all();
}

static void push_audio_session_event(
    llama_server_bridge_audio_session * session,
    int32_t kind,
    uint32_t flags,
    uint64_t start_sample,
    uint64_t end_sample,
    int32_t speaker_id,
    uint32_t item_id,
    const std::string & text,
    const std::string & detail) {

    if (session == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    audio_session_event_record event = {};
    event.kind = kind;
    event.flags = flags;
    event.start_sample = start_sample;
    event.end_sample = end_sample;
    event.speaker_id = speaker_id;
    event.item_id = item_id;
    event.text = text;
    event.detail = detail;
    push_audio_session_event_locked(session, std::move(event));
}

static int32_t fail_audio_session(
    llama_server_bridge_audio_session * session,
    const std::string & message,
    bool emit_event = true) {

    const std::string normalized = normalize_error(message);
    set_audio_session_error(session, normalized);
    if (emit_event) {
        push_audio_session_event(
            session,
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR,
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL,
            0,
            0,
            -1,
            0,
            "",
            normalized);
    }
    return -1;
}

static std::vector<uint8_t> f32_mono_to_pcm16le_bytes(const std::vector<float> & samples) {
    std::vector<uint8_t> out;
    out.resize(samples.size() * sizeof(int16_t));
    auto * dst = reinterpret_cast<int16_t *>(out.data());
    for (size_t i = 0; i < samples.size(); ++i) {
        const float clamped = std::clamp(samples[i], -1.0f, 1.0f);
        const float scaled = clamped >= 0.0f ? clamped * 32767.0f : clamped * 32768.0f;
        dst[i] = static_cast<int16_t>(std::lrint(scaled));
    }
    return out;
}

static std::vector<float> pcm16le_bytes_to_f32_mono(const uint8_t * bytes, size_t byte_count) {
    std::vector<float> out;
    if (bytes == nullptr || byte_count < sizeof(int16_t)) {
        return out;
    }
    const size_t n = byte_count / sizeof(int16_t);
    out.resize(n);
    const auto * src = reinterpret_cast<const int16_t *>(bytes);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(src[i]) / 32768.0f;
    }
    return out;
}

static bool wav_payload_to_pcm16le(
    const std::vector<uint8_t> & wav_bytes,
    std::vector<uint8_t> & pcm16le,
    std::string & error) {

    pcm16le.clear();
    error.clear();
    if (wav_bytes.size() < 44) {
        error = "decoded WAV payload is too small";
        return false;
    }
    pcm16le.assign(wav_bytes.begin() + 44, wav_bytes.end());
    if (pcm16le.empty()) {
        error = "decoded WAV payload contains no PCM data";
        return false;
    }
    return true;
}

static uint32_t next_audio_session_item_id_locked(llama_server_bridge_audio_session * session) {
    if (session == nullptr) {
        return 0;
    }
    return session->next_item_id++;
}

static void join_audio_session_transcription_thread(llama_server_bridge_audio_session * session) {
    if (session == nullptr) {
        return;
    }
    if (session->transcription_thread.joinable()) {
        session->transcription_thread.join();
    }
}

static void destroy_realtime_bridge(llama_server_bridge_realtime * bridge) {
    if (bridge != nullptr) {
        llama_server_bridge_realtime_destroy(bridge);
    }
}

static std::string audio_session_speaker_label(int32_t speaker_id) {
    if (speaker_id < 0) {
        return "UNASSIGNED";
    }
    std::ostringstream oss;
    oss << "SPEAKER_" << std::setw(2) << std::setfill('0') << speaker_id;
    return oss.str();
}

static bool build_audio_session_timeline_metadata(
    const std::string & input_metadata_json,
    std::string & out_metadata_json,
    std::string & error) {

    error.clear();
    json metadata = json::object();
    if (!input_metadata_json.empty()) {
        try {
            metadata = json::parse(input_metadata_json);
        } catch (const std::exception & e) {
            error = std::string("invalid audio session transcription metadata_json: ") + e.what();
            return false;
        }
        if (!metadata.is_object()) {
            error = "audio session transcription metadata_json must be a JSON object";
            return false;
        }
    }

    metadata["mode"] = "timeline";
    if (metadata.find("custom") == metadata.end()) {
        metadata["custom"] = "default";
    }
    out_metadata_json = metadata.dump();
    return true;
}

static void emit_transcription_timeline_events(
    llama_server_bridge_audio_session * session,
    const json & root_json,
    uint32_t base_flags) {

    if (session == nullptr || !root_json.is_object()) {
        return;
    }

    const json timeline = json_value(root_json, "timeline", json::object());
    const json pieces = json_value(timeline, "whisper_pieces", json::array());
    const json words = json_value(timeline, "words", json::array());

    auto emit_items = [&](const json & items, int32_t kind, const char * text_key) {
        if (!items.is_array()) {
            return;
        }
        for (const auto & item : items) {
            if (!item.is_object()) {
                continue;
            }
            const double start_sec = std::max(0.0, json_value(item, "start_sec", 0.0));
            const double end_sec = std::max(start_sec, json_value(item, "end_sec", start_sec));
            const std::string text = json_value(item, text_key, std::string());
            json detail_json = json::object();
            if (item.contains("start_word_index")) {
                detail_json["start_word_index"] = json_value(item, "start_word_index", int64_t(0));
            }
            if (item.contains("end_word_index")) {
                detail_json["end_word_index"] = json_value(item, "end_word_index", int64_t(0));
            }
            if (item.contains("num_words")) {
                detail_json["num_words"] = json_value(item, "num_words", int64_t(0));
            }
            uint32_t item_id = 0;
            {
                std::lock_guard<std::mutex> lock(session->mutex);
                item_id = next_audio_session_item_id_locked(session);
            }
            push_audio_session_event(
                session,
                kind,
                base_flags | LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL,
                seconds_to_samples(start_sec, session->params.expected_input_sample_rate_hz),
                seconds_to_samples(end_sec, session->params.expected_input_sample_rate_hz),
                -1,
                item_id,
                text,
                detail_json.empty() ? std::string() : detail_json.dump());
        }
    };

    emit_items(words, LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT, "word");
    emit_items(pieces, LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT, "text");
}

static std::string base64_encode_bytes(const uint8_t * data, size_t len) {
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if (data == nullptr || len == 0) {
        return "";
    }

    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        const uint32_t chunk = (static_cast<uint32_t>(data[i]) << 16)
            | (static_cast<uint32_t>(data[i + 1]) << 8)
            | (static_cast<uint32_t>(data[i + 2]));
        out.push_back(kTable[(chunk >> 18) & 0x3F]);
        out.push_back(kTable[(chunk >> 12) & 0x3F]);
        out.push_back(kTable[(chunk >> 6) & 0x3F]);
        out.push_back(kTable[chunk & 0x3F]);
    }

    const size_t rem = len - i;
    if (rem == 1) {
        const uint32_t chunk = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(kTable[(chunk >> 18) & 0x3F]);
        out.push_back(kTable[(chunk >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        const uint32_t chunk = (static_cast<uint32_t>(data[i]) << 16)
            | (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(kTable[(chunk >> 18) & 0x3F]);
        out.push_back(kTable[(chunk >> 12) & 0x3F]);
        out.push_back(kTable[(chunk >> 6) & 0x3F]);
        out.push_back('=');
    }

    return out;
}

static void append_le16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

static void append_le32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

static std::vector<uint8_t> pcm16_mono_16k_to_wav(const std::vector<uint8_t> & pcm) {
    const uint32_t sample_rate = 16000;
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = static_cast<uint16_t>(channels * (bits_per_sample / 8));
    const uint32_t byte_rate = sample_rate * static_cast<uint32_t>(block_align);
    const uint32_t data_size = static_cast<uint32_t>(pcm.size());
    const uint32_t riff_size = 36u + data_size;

    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(44u + data_size));

    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_le32(out, riff_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E'});

    out.insert(out.end(), {'f', 'm', 't', ' '});
    append_le32(out, 16);
    append_le16(out, 1);
    append_le16(out, channels);
    append_le32(out, sample_rate);
    append_le32(out, byte_rate);
    append_le16(out, block_align);
    append_le16(out, bits_per_sample);

    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_le32(out, data_size);
    out.insert(out.end(), pcm.begin(), pcm.end());

    return out;
}

#ifdef LLAMA_SERVER_BRIDGE_USE_FFMPEG
struct ffmpeg_mem_reader {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t pos = 0;
};

static int ffmpeg_read_packet(void * opaque, uint8_t * buf, int buf_size) {
    auto * src = static_cast<ffmpeg_mem_reader *>(opaque);
    if (src == nullptr || buf == nullptr || buf_size <= 0) {
        return AVERROR(EINVAL);
    }
    if (src->pos >= src->size) {
        return AVERROR_EOF;
    }
    const size_t remain = src->size - src->pos;
    const size_t n = std::min(remain, static_cast<size_t>(buf_size));
    std::memcpy(buf, src->data + src->pos, n);
    src->pos += n;
    return static_cast<int>(n);
}

static int64_t ffmpeg_seek(void * opaque, int64_t offset, int whence) {
    auto * src = static_cast<ffmpeg_mem_reader *>(opaque);
    if (src == nullptr) {
        return AVERROR(EINVAL);
    }
    if (whence == AVSEEK_SIZE) {
        return static_cast<int64_t>(src->size);
    }

    size_t base = 0;
    switch (whence & ~AVSEEK_FORCE) {
        case SEEK_SET:
            base = 0;
            break;
        case SEEK_CUR:
            base = src->pos;
            break;
        case SEEK_END:
            base = src->size;
            break;
        default:
            return AVERROR(EINVAL);
    }

    const int64_t target = static_cast<int64_t>(base) + offset;
    if (target < 0 || static_cast<size_t>(target) > src->size) {
        return AVERROR(EINVAL);
    }
    src->pos = static_cast<size_t>(target);
    return static_cast<int64_t>(src->pos);
}

static bool ffmpeg_append_frame_s16mono16k(
    SwrContext * swr,
    AVCodecContext * codec_ctx,
    AVFrame * frame,
    std::vector<uint8_t> & pcm,
    std::string & error) {

    if (swr == nullptr || codec_ctx == nullptr || frame == nullptr) {
        error = "ffmpeg: invalid frame conversion state";
        return false;
    }

    const int in_rate = codec_ctx->sample_rate > 0 ? codec_ctx->sample_rate : 16000;
    const int64_t delay = swr_get_delay(swr, in_rate);
    const int out_samples = static_cast<int>(av_rescale_rnd(
        delay + frame->nb_samples,
        16000,
        in_rate,
        AV_ROUND_UP));
    if (out_samples <= 0) {
        return true;
    }

    uint8_t * out_buf = nullptr;
    int out_linesize = 0;
    int rc = av_samples_alloc(&out_buf, &out_linesize, 1, out_samples, AV_SAMPLE_FMT_S16, 0);
    if (rc < 0) {
        error = "ffmpeg: av_samples_alloc failed";
        return false;
    }

    const uint8_t ** in_data = const_cast<const uint8_t **>(frame->extended_data);
    const int converted = swr_convert(swr, &out_buf, out_samples, in_data, frame->nb_samples);
    if (converted < 0) {
        av_freep(&out_buf);
        error = "ffmpeg: swr_convert failed";
        return false;
    }

    const size_t bytes = static_cast<size_t>(converted) * 2;
    pcm.insert(pcm.end(), out_buf, out_buf + bytes);
    av_freep(&out_buf);
    return true;
}

static bool ffmpeg_convert_to_wav_pcm16_mono_16k(
    const uint8_t * input_data,
    size_t input_len,
    const char * input_format,
    std::vector<uint8_t> & out_wav,
    std::string & error) {

    out_wav.clear();
    error.clear();

    if (input_data == nullptr || input_len == 0) {
        error = "audio raw input is empty";
        return false;
    }

    AVFormatContext * fmt = nullptr;
    AVIOContext * avio = nullptr;
    AVCodecContext * codec_ctx = nullptr;
    AVPacket * pkt = nullptr;
    AVFrame * frame = nullptr;
    SwrContext * swr = nullptr;
    uint8_t * avio_buf = nullptr;
    std::vector<uint8_t> pcm;
    const AVInputFormat * in_fmt = nullptr;
    AVStream * st = nullptr;
    const AVCodec * codec = nullptr;
    int in_rate = 16000;
    const int avio_buf_size = 32768;
    int stream_idx = -1;
    int ret = 0;

    ffmpeg_mem_reader mem = {};
    mem.data = input_data;
    mem.size = input_len;
    mem.pos = 0;

    fmt = avformat_alloc_context();
    if (fmt == nullptr) {
        error = "ffmpeg: avformat_alloc_context failed";
        goto fail;
    }

    avio_buf = static_cast<uint8_t *>(av_malloc(avio_buf_size));
    if (avio_buf == nullptr) {
        error = "ffmpeg: av_malloc for AVIO buffer failed";
        goto fail;
    }

    avio = avio_alloc_context(avio_buf, avio_buf_size, 0, &mem, ffmpeg_read_packet, nullptr, ffmpeg_seek);
    if (avio == nullptr) {
        error = "ffmpeg: avio_alloc_context failed";
        goto fail;
    }
    fmt->pb = avio;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (input_format != nullptr && input_format[0] != '\0') {
        in_fmt = av_find_input_format(input_format);
    }

    ret = avformat_open_input(&fmt, nullptr, in_fmt, nullptr);
    if (ret < 0) {
        error = "ffmpeg: avformat_open_input failed";
        goto fail;
    }

    ret = avformat_find_stream_info(fmt, nullptr);
    if (ret < 0) {
        error = "ffmpeg: avformat_find_stream_info failed";
        goto fail;
    }

    stream_idx = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_idx < 0) {
        error = "ffmpeg: no audio stream found";
        goto fail;
    }

    st = fmt->streams[stream_idx];
    if (st == nullptr || st->codecpar == nullptr) {
        error = "ffmpeg: invalid audio stream codec parameters";
        goto fail;
    }

    codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (codec == nullptr) {
        error = "ffmpeg: avcodec_find_decoder failed";
        goto fail;
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (codec_ctx == nullptr) {
        error = "ffmpeg: avcodec_alloc_context3 failed";
        goto fail;
    }

    ret = avcodec_parameters_to_context(codec_ctx, st->codecpar);
    if (ret < 0) {
        error = "ffmpeg: avcodec_parameters_to_context failed";
        goto fail;
    }

    ret = avcodec_open2(codec_ctx, codec, nullptr);
    if (ret < 0) {
        error = "ffmpeg: avcodec_open2 failed";
        goto fail;
    }

    in_rate = codec_ctx->sample_rate > 0 ? codec_ctx->sample_rate : 16000;
#if LIBAVCODEC_VERSION_MAJOR >= 59
    {
        AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;
        const AVChannelLayout * in_ch_layout = &codec_ctx->ch_layout;
        AVChannelLayout fallback_in_ch_layout = {};
        if (in_ch_layout->nb_channels <= 0) {
            av_channel_layout_default(&fallback_in_ch_layout, 1);
            in_ch_layout = &fallback_in_ch_layout;
        }

        ret = swr_alloc_set_opts2(
            &swr,
            &out_ch_layout,
            AV_SAMPLE_FMT_S16,
            16000,
            in_ch_layout,
            codec_ctx->sample_fmt,
            in_rate,
            0,
            nullptr);
        av_channel_layout_uninit(&out_ch_layout);
        av_channel_layout_uninit(&fallback_in_ch_layout);
        if (ret < 0 || swr == nullptr) {
            error = "ffmpeg: swr_alloc_set_opts2 failed";
            goto fail;
        }
    }
#else
    int64_t in_ch_layout = codec_ctx->channel_layout;
    if (in_ch_layout == 0) {
        in_ch_layout = av_get_default_channel_layout(std::max(1, codec_ctx->channels));
    }

    swr = swr_alloc_set_opts(
        nullptr,
        AV_CH_LAYOUT_MONO,
        AV_SAMPLE_FMT_S16,
        16000,
        in_ch_layout,
        codec_ctx->sample_fmt,
        in_rate,
        0,
        nullptr);
    if (swr == nullptr) {
        error = "ffmpeg: swr_alloc_set_opts failed";
        goto fail;
    }
#endif
    ret = swr_init(swr);
    if (ret < 0) {
        error = "ffmpeg: swr_init failed";
        goto fail;
    }

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (pkt == nullptr || frame == nullptr) {
        error = "ffmpeg: packet/frame allocation failed";
        goto fail;
    }

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            ret = avcodec_send_packet(codec_ctx, pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                error = "ffmpeg: avcodec_send_packet failed";
                goto fail;
            }
            while (true) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    error = "ffmpeg: avcodec_receive_frame failed";
                    goto fail;
                }
                if (!ffmpeg_append_frame_s16mono16k(swr, codec_ctx, frame, pcm, error)) {
                    goto fail;
                }
                av_frame_unref(frame);
            }
        }
        av_packet_unref(pkt);
    }

    ret = avcodec_send_packet(codec_ctx, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) {
        error = "ffmpeg: final avcodec_send_packet failed";
        goto fail;
    }
    while (true) {
        ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            error = "ffmpeg: final avcodec_receive_frame failed";
            goto fail;
        }
        if (!ffmpeg_append_frame_s16mono16k(swr, codec_ctx, frame, pcm, error)) {
            goto fail;
        }
        av_frame_unref(frame);
    }

    if (pcm.empty()) {
        error = "ffmpeg: decoded PCM is empty";
        goto fail;
    }

    out_wav = pcm16_mono_16k_to_wav(pcm);

    if (pkt != nullptr) {
        av_packet_free(&pkt);
    }
    if (frame != nullptr) {
        av_frame_free(&frame);
    }
    if (swr != nullptr) {
        swr_free(&swr);
    }
    if (codec_ctx != nullptr) {
        avcodec_free_context(&codec_ctx);
    }
    if (fmt != nullptr) {
        avformat_close_input(&fmt);
    }
    if (avio != nullptr) {
        av_freep(&avio->buffer);
        avio_context_free(&avio);
    }
    return true;

fail:
    if (pkt != nullptr) {
        av_packet_free(&pkt);
    }
    if (frame != nullptr) {
        av_frame_free(&frame);
    }
    if (swr != nullptr) {
        swr_free(&swr);
    }
    if (codec_ctx != nullptr) {
        avcodec_free_context(&codec_ctx);
    }
    if (fmt != nullptr) {
        avformat_close_input(&fmt);
    }
    if (avio != nullptr) {
        av_freep(&avio->buffer);
        avio_context_free(&avio);
    } else if (avio_buf != nullptr) {
        av_freep(&avio_buf);
    }
    return false;
}
#else
static bool ffmpeg_convert_to_wav_pcm16_mono_16k(
    const uint8_t *,
    size_t,
    const char *,
    std::vector<uint8_t> &,
    std::string & error) {
    error = "bridge was built without FFmpeg support (LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=OFF)";
    return false;
}
#endif

static bool prepare_audio_raw_payload(
    const llama_server_bridge_audio_raw_request * req,
    json & out_metadata,
    std::vector<uint8_t> & out_audio,
    std::string & out_format,
    std::string & error) {

    out_metadata = json::object();
    out_audio.clear();
    out_format.clear();
    error.clear();

    if (req == nullptr || req->audio_bytes == nullptr || req->audio_bytes_len == 0) {
        error = "audio raw request is empty";
        return false;
    }

    if (req->metadata_json != nullptr && req->metadata_json[0] != '\0') {
        try {
            out_metadata = json::parse(req->metadata_json);
        } catch (const std::exception & e) {
            error = std::string("invalid metadata_json: ") + e.what();
            return false;
        }
        if (!out_metadata.is_object()) {
            error = "metadata_json must be a JSON object";
            return false;
        }
    }

    out_audio.assign(req->audio_bytes, req->audio_bytes + req->audio_bytes_len);
    out_format = (req->audio_format != nullptr && req->audio_format[0] != '\0')
        ? std::string(req->audio_format)
        : std::string("wav");

    if (req->ffmpeg_convert != 0) {
        std::vector<uint8_t> converted;
        if (!ffmpeg_convert_to_wav_pcm16_mono_16k(
                out_audio.data(),
                out_audio.size(),
                out_format.c_str(),
                converted,
                error)) {
            return false;
        }
        out_audio.swap(converted);
        out_format = "wav";
    }

    return true;
}

static std::string build_audio_body_with_base64(
    json metadata,
    const std::vector<uint8_t> & audio,
    const std::string & format) {

    if (!metadata.is_object()) {
        metadata = json::object();
    }

    metadata["audio"] = json::object();
    metadata["audio"]["format"] = format;
    metadata["audio"]["data"] = base64_encode_bytes(audio.data(), audio.size());
    return metadata.dump();
}

struct llama_server_bridge_params llama_server_bridge_default_params(void) {
    llama_server_bridge_params p = {};
    p.model_path = nullptr;
    p.mmproj_path = nullptr;
    p.n_ctx = 32768;
    p.n_batch = 2048;
    p.n_ubatch = 2048;
    p.n_parallel = 1;
    p.n_threads = 8;
    p.n_threads_batch = 8;
    p.n_gpu_layers = -1;
    p.main_gpu = -1;
    p.gpu = -1;
    p.no_kv_offload = 0;
    p.mmproj_use_gpu = -1;
    p.cache_ram_mib = -1;
    p.seed = -1;
    p.ctx_shift = 1;
    p.kv_unified = 1;
#if defined(__APPLE__)
    // On macOS/Metal, use "unset devices" as the default so create()
    // can auto-select the first GPU when the caller does not pass a device.
    p.devices = nullptr;
#else
    p.devices = "none";
#endif
    p.tensor_split = nullptr;
    p.split_mode = 0;
    p.embedding = 0;
    p.reranking = 0;
    p.pooling_type = -1;
    return p;
}

struct llama_server_bridge_chat_request llama_server_bridge_default_chat_request(void) {
    llama_server_bridge_chat_request req = {};
    req.prompt = nullptr;
    req.n_predict = 4096;
    req.id_slot = -1;
    req.temperature = 0.0f;
    req.top_p = 1.0f;
    req.top_k = -1;
    req.min_p = -1.0f;
    req.seed = -1;
    req.repeat_last_n = -1;
    req.repeat_penalty = -1.0f;
    req.presence_penalty = -1.0f;
    req.frequency_penalty = -1.0f;
    req.dry_multiplier = -1.0f;
    req.dry_allowed_length = -1;
    req.dry_penalty_last_n = -1;
    return req;
}

struct llama_server_bridge_vlm_request llama_server_bridge_default_vlm_request(void) {
    llama_server_bridge_vlm_request req = {};
    req.prompt = nullptr;
    req.image_bytes = nullptr;
    req.image_bytes_len = 0;
    req.n_predict = 4096;
    req.id_slot = -1;
    req.temperature = 0.0f;
    req.top_p = 1.0f;
    req.top_k = -1;
    req.min_p = -1.0f;
    req.seed = -1;
    req.repeat_last_n = -1;
    req.repeat_penalty = -1.0f;
    req.presence_penalty = -1.0f;
    req.frequency_penalty = -1.0f;
    req.dry_multiplier = -1.0f;
    req.dry_allowed_length = -1;
    req.dry_penalty_last_n = -1;
    return req;
}

struct llama_server_bridge_vlm_result llama_server_bridge_empty_vlm_result(void) {
    llama_server_bridge_vlm_result out = {};
    out.ok = 0;
    out.truncated = 0;
    out.stop = 0;
    out.n_decoded = 0;
    out.n_prompt_tokens = 0;
    out.n_tokens_cached = 0;
    out.eos_reached = 0;
    out.prompt_ms = 0.0;
    out.predicted_ms = 0.0;
    out.text = nullptr;
    out.error_json = nullptr;
    return out;
}

struct llama_server_bridge_embeddings_request llama_server_bridge_default_embeddings_request(void) {
    llama_server_bridge_embeddings_request req = {};
    req.body_json = nullptr;
    req.oai_compat = 1;
    return req;
}

struct llama_server_bridge_rerank_request llama_server_bridge_default_rerank_request(void) {
    llama_server_bridge_rerank_request req = {};
    req.body_json = nullptr;
    return req;
}

struct llama_server_bridge_audio_request llama_server_bridge_default_audio_request(void) {
    llama_server_bridge_audio_request req = {};
    req.body_json = nullptr;
    return req;
}

struct llama_server_bridge_audio_raw_request llama_server_bridge_default_audio_raw_request(void) {
    llama_server_bridge_audio_raw_request req = {};
    req.audio_bytes = nullptr;
    req.audio_bytes_len = 0;
    req.audio_format = nullptr;
    req.metadata_json = nullptr;
    req.ffmpeg_convert = 1;
    return req;
}

struct llama_server_bridge_audio_session_params llama_server_bridge_default_audio_session_params(void) {
    llama_server_bridge_audio_session_params out = {};
    out.expected_input_sample_rate_hz = 16000;
    out.expected_input_channels = 1;
    out.max_buffered_audio_samples = 0;
    out.event_queue_capacity = 1024;
    return out;
}

struct llama_server_bridge_audio_transcription_params llama_server_bridge_default_audio_transcription_params(void) {
    llama_server_bridge_audio_transcription_params out = {};
    out.bridge_params = llama_server_bridge_default_params();
    out.metadata_json = nullptr;
    out.mode = LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE;
    out.realtime_params = llama_server_bridge_default_realtime_params();
    return out;
}

struct llama_server_bridge_json_result llama_server_bridge_empty_json_result(void) {
    llama_server_bridge_json_result out = {};
    out.ok = 0;
    out.status = 0;
    out.json = nullptr;
    out.error_json = nullptr;
    return out;
}

struct llama_server_bridge_realtime_backend_info llama_server_bridge_empty_realtime_backend_info(void) {
    llama_server_bridge_realtime_backend_info out = {};
    out.backend_kind = 0;
    out.name = "";
    out.default_runtime_backend_name = "";
    out.supports_model_preload = 0;
    out.emits_transcript = 0;
    out.emits_speaker_spans = 0;
    out.default_sample_rate_hz = 0;
    out.default_audio_ring_capacity_samples = 0;
    out.required_input_channels = 0;
    return out;
}

static llama_server_bridge_realtime_backend_info make_realtime_backend_info(int32_t backend_kind) {
    llama_server_bridge_realtime_backend_info out = llama_server_bridge_empty_realtime_backend_info();
    const auto kind = static_cast<llama::realtime::backend_kind>(backend_kind);
    llama::realtime::backend_descriptor descriptor = {};
    if (!llama::realtime::backend_info(kind, descriptor)) {
        return out;
    }

    out.backend_kind = static_cast<int32_t>(descriptor.kind);
    out.name = descriptor.name != nullptr ? descriptor.name : "";
    out.default_runtime_backend_name =
        descriptor.default_runtime_backend_name != nullptr ? descriptor.default_runtime_backend_name : "";
    out.supports_model_preload = descriptor.supports_model_preload ? 1 : 0;
    out.emits_transcript = descriptor.emits_transcript ? 1 : 0;
    out.emits_speaker_spans = descriptor.emits_speaker_spans ? 1 : 0;
    out.default_sample_rate_hz = descriptor.default_sample_rate_hz;
    out.default_audio_ring_capacity_samples = descriptor.default_audio_ring_capacity_samples;
    out.required_input_channels = descriptor.required_input_channels;
    return out;
}

static llama_server_bridge_realtime_backend_info make_realtime_backend_info_for_model(
    const llama_server_bridge_realtime_model_impl & model) {

    auto out = make_realtime_backend_info(model.backend_kind);
    if (!model.resolved_runtime_backend_name.empty()) {
        out.default_runtime_backend_name = model.resolved_runtime_backend_name.c_str();
    }
    if (model.default_session_cfg.expected_sample_rate_hz != 0) {
        out.default_sample_rate_hz = model.default_session_cfg.expected_sample_rate_hz;
    }
    if (model.default_session_cfg.audio_ring_capacity_samples != 0) {
        out.default_audio_ring_capacity_samples = model.default_session_cfg.audio_ring_capacity_samples;
    }
    return out;
}

struct llama_server_bridge_realtime_sortformer_params llama_server_bridge_default_realtime_sortformer_params(void) {
    const auto info = make_realtime_backend_info(LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER);
    llama_server_bridge_realtime_sortformer_params out = {};
    out.gguf_path = nullptr;
    out.backend_name = info.default_runtime_backend_name;
    out.expected_sample_rate_hz = info.default_sample_rate_hz;
    out.audio_ring_capacity_samples = info.default_audio_ring_capacity_samples;
    return out;
}

struct llama_server_bridge_realtime_params llama_server_bridge_default_realtime_params_for_backend(int32_t backend_kind) {
    const auto info = make_realtime_backend_info(backend_kind);
    llama_server_bridge_realtime_params out = {};
    out.backend_kind = info.backend_kind != 0 ? info.backend_kind : backend_kind;
    out.model_path = nullptr;
    out.backend_name = info.default_runtime_backend_name;
    out.expected_sample_rate_hz = info.default_sample_rate_hz;
    out.audio_ring_capacity_samples = info.default_audio_ring_capacity_samples;
    out.capture_debug = 0;
    return out;
}

struct llama_server_bridge_realtime_params llama_server_bridge_default_realtime_params(void) {
    llama_server_bridge_realtime_params out = {};
    out.backend_kind = LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_AUTO;
    out.model_path = nullptr;
    out.backend_name = nullptr;
    out.expected_sample_rate_hz = 0;
    out.audio_ring_capacity_samples = 0;
    out.capture_debug = 0;
    return out;
}

int32_t llama_server_bridge_realtime_backend_count(void) {
    const size_t count = llama::realtime::backend_descriptor_count();
    if (count > static_cast<size_t>(INT32_MAX)) {
        return INT32_MAX;
    }
    return static_cast<int32_t>(count);
}

int32_t llama_server_bridge_realtime_backend_kind_at(size_t index) {
    const auto * descriptor = llama::realtime::backend_descriptor_at(index);
    if (descriptor == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(descriptor->kind);
}

int32_t llama_server_bridge_realtime_backend_get_info(
    int32_t backend_kind,
    struct llama_server_bridge_realtime_backend_info * out_info) {

    if (out_info == nullptr) {
        return 0;
    }
    *out_info = make_realtime_backend_info(backend_kind);
    return out_info->backend_kind != 0 ? 1 : 0;
}

const char * llama_server_bridge_realtime_backend_name(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).name;
}

int32_t llama_server_bridge_realtime_backend_kind_from_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return 0;
    }
    llama::realtime::backend_kind kind = llama::realtime::backend_kind::unknown;
    if (!llama::realtime::parse_backend_kind_name(name, kind)) {
        return 0;
    }
    return static_cast<int32_t>(kind);
}

int32_t llama_server_bridge_realtime_backend_kind_from_model_path(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return 0;
    }
    llama::realtime::backend_kind kind = llama::realtime::backend_kind::unknown;
    std::string error;
    if (!llama::realtime::detect_backend_kind_from_model_path(model_path, kind, error)) {
        return 0;
    }
    return static_cast<int32_t>(kind);
}

int32_t llama_server_bridge_realtime_backend_supports_model_preload(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).supports_model_preload;
}

int32_t llama_server_bridge_realtime_backend_emits_transcript(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).emits_transcript;
}

int32_t llama_server_bridge_realtime_backend_emits_speaker_spans(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).emits_speaker_spans;
}

const char * llama_server_bridge_realtime_backend_default_runtime_backend_name(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).default_runtime_backend_name;
}

uint32_t llama_server_bridge_realtime_backend_default_sample_rate_hz(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).default_sample_rate_hz;
}

uint32_t llama_server_bridge_realtime_backend_default_audio_ring_capacity_samples(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).default_audio_ring_capacity_samples;
}

uint32_t llama_server_bridge_realtime_backend_required_input_channels(int32_t backend_kind) {
    return make_realtime_backend_info(backend_kind).required_input_channels;
}

int32_t llama_server_bridge_realtime_model_cache_entry_count(void) {
    std::lock_guard<std::mutex> lock(g_realtime_model_cache_mutex);
    const size_t count = g_realtime_model_cache.size();
    return count > static_cast<size_t>(INT32_MAX) ? INT32_MAX : static_cast<int32_t>(count);
}

void llama_server_bridge_realtime_model_cache_clear(void) {
    decltype(g_realtime_model_cache) cleared;
    {
        std::lock_guard<std::mutex> lock(g_realtime_model_cache_mutex);
        g_realtime_model_cache.swap(cleared);
    }
    cleared.clear();
}

void llama_server_bridge_result_free(struct llama_server_bridge_vlm_result * out) {
    if (out == nullptr) {
        return;
    }
    if (out->text != nullptr) {
        std::free(out->text);
        out->text = nullptr;
    }
    if (out->error_json != nullptr) {
        std::free(out->error_json);
        out->error_json = nullptr;
    }
}

void llama_server_bridge_json_result_free(struct llama_server_bridge_json_result * out) {
    if (out == nullptr) {
        return;
    }
    if (out->json != nullptr) {
        std::free(out->json);
        out->json = nullptr;
    }
    if (out->error_json != nullptr) {
        std::free(out->error_json);
        out->error_json = nullptr;
    }
}

const char * llama_server_bridge_last_error(const struct llama_server_bridge * bridge) {
    if (bridge == nullptr) {
        return "";
    }
    std::lock_guard<std::mutex> lock(bridge->error_mutex);
    return bridge->last_error.c_str();
}

const char * llama_server_bridge_realtime_last_error(const struct llama_server_bridge_realtime * bridge) {
    if (bridge == nullptr) {
        return "";
    }
    std::lock_guard<std::mutex> lock(bridge->error_mutex);
    return bridge->last_error.c_str();
}

const char * llama_server_bridge_realtime_sortformer_model_last_error(
    const llama_server_bridge_realtime_sortformer_model * model) {
    return llama_server_bridge_realtime_model_last_error(
        reinterpret_cast<const llama_server_bridge_realtime_model *>(model));
}

const char * llama_server_bridge_realtime_model_last_error(
    const llama_server_bridge_realtime_model * model) {
    auto typed_model = reinterpret_cast<const struct llama_server_bridge_realtime_model_impl *>(model);
    if (typed_model == nullptr) {
        return "";
    }
    std::lock_guard<std::mutex> lock(typed_model->error_mutex);
    return typed_model->last_error.c_str();
}

struct realtime_resolved_params {
    llama::realtime::backend_model_params backend_params;
    llama::realtime::stream_session_config session_cfg;
    bool capture_debug = false;
};

static llama_server_bridge_realtime_params sortformer_params_to_generic(
    const struct llama_server_bridge_realtime_sortformer_params * params) {

    const auto defaults = llama_server_bridge_default_realtime_sortformer_params();
    llama_server_bridge_realtime_params out = llama_server_bridge_default_realtime_params_for_backend(
        LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER);
    out.model_path = params != nullptr ? params->gguf_path : defaults.gguf_path;
    out.backend_name = (params != nullptr && params->backend_name != nullptr && params->backend_name[0] != '\0')
        ? params->backend_name
        : defaults.backend_name;
    out.expected_sample_rate_hz =
        (params != nullptr && params->expected_sample_rate_hz != 0)
            ? params->expected_sample_rate_hz
            : defaults.expected_sample_rate_hz;
    out.audio_ring_capacity_samples =
        (params != nullptr && params->audio_ring_capacity_samples != 0)
            ? params->audio_ring_capacity_samples
            : defaults.audio_ring_capacity_samples;
    return out;
}

static bool translate_generic_realtime_params(
    const struct llama_server_bridge_realtime_params * params,
    const char * error_prefix,
    realtime_resolved_params & out,
    std::string & error) {

    if (params == nullptr) {
        error = std::string(error_prefix) + ": params is null";
        return false;
    }

    if (params->model_path != nullptr) {
        out.backend_params.model_path = params->model_path;
    }
    if (params->backend_name != nullptr) {
        out.backend_params.backend_name = params->backend_name;
    }

    const int32_t resolved_backend_kind = params->backend_kind;
    out.backend_params.kind = static_cast<llama::realtime::backend_kind>(resolved_backend_kind);

    if (!llama::realtime::backend_resolve_runtime(
            out.backend_params,
            params->expected_sample_rate_hz,
            params->audio_ring_capacity_samples,
            out.backend_params,
            out.session_cfg,
            error)) {
        error = std::string(error_prefix) + ": " + error;
        return false;
    }

    out.capture_debug = params->capture_debug != 0;

    error.clear();
    return true;
}

static bool translate_realtime_session_params_from_loaded_model(
    const llama_server_bridge_realtime_model_impl * model,
    const struct llama_server_bridge_realtime_params * params,
    const char * error_prefix,
    realtime_resolved_params & out,
    std::string & error) {

    if (model == nullptr) {
        error = std::string(error_prefix) + ": model is null";
        return false;
    }

    const int32_t loaded_backend_kind = model->backend_kind;
    const auto defaults = make_realtime_backend_info_for_model(*model);
    out.backend_params.model_path = model->model_path;
    out.backend_params.kind = static_cast<llama::realtime::backend_kind>(loaded_backend_kind);
    out.backend_params.backend_name = model->resolved_runtime_backend_name;
    out.session_cfg = model->default_session_cfg;

    if (params == nullptr) {
        if (out.session_cfg.expected_sample_rate_hz == 0) {
            out.session_cfg.expected_sample_rate_hz = defaults.default_sample_rate_hz;
        }
        if (out.session_cfg.audio_ring_capacity_samples == 0) {
            out.session_cfg.audio_ring_capacity_samples = defaults.default_audio_ring_capacity_samples;
        }
        out.capture_debug = false;
        error.clear();
        return true;
    }

    const int32_t requested_backend_kind =
        params->backend_kind != 0 ? params->backend_kind : loaded_backend_kind;
    if (requested_backend_kind != loaded_backend_kind) {
        error = std::string(error_prefix) + ": backend kind mismatch";
        return false;
    }

    if (params->backend_name != nullptr && params->backend_name[0] != '\0') {
        const std::string requested_backend_name = params->backend_name;
        if (!model->resolved_runtime_backend_name.empty() &&
            requested_backend_name != model->resolved_runtime_backend_name) {
            error = std::string(error_prefix) + ": runtime backend mismatch";
            return false;
        }
    }

    if (params->expected_sample_rate_hz != 0) {
        out.session_cfg.expected_sample_rate_hz = params->expected_sample_rate_hz;
    } else if (out.session_cfg.expected_sample_rate_hz == 0) {
        out.session_cfg.expected_sample_rate_hz = defaults.default_sample_rate_hz;
    }

    if (params->audio_ring_capacity_samples != 0) {
        out.session_cfg.audio_ring_capacity_samples = params->audio_ring_capacity_samples;
    } else if (out.session_cfg.audio_ring_capacity_samples == 0) {
        out.session_cfg.audio_ring_capacity_samples = defaults.default_audio_ring_capacity_samples;
    }

    out.capture_debug = params->capture_debug != 0;

    error.clear();
    return true;
}

struct llama_server_bridge * llama_server_bridge_create(const struct llama_server_bridge_params * params) {
    if (params == nullptr) {
        std::fprintf(stderr, "llama_server_bridge_create: params is null\n");
        return nullptr;
    }

    const bool has_model_path = params->model_path != nullptr && params->model_path[0] != '\0';
    const char * env_audio_only = std::getenv("LLAMA_SERVER_AUDIO_ONLY");
    // Default to audio-only mode when model_path is omitted. This keeps
    // transcriptions working even if the caller does not set an env flag.
    bool audio_only = !has_model_path;
    if (env_audio_only != nullptr && env_audio_only[0] != '\0') {
        const std::string v = trim_copy(env_audio_only);
        if (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON") {
            audio_only = true;
        } else if (v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF") {
            audio_only = false;
        }
    }
    if (!has_model_path && !audio_only) {
        std::fprintf(stderr, "llama_server_bridge_create: missing model_path and audio-only mode is disabled\n");
        return nullptr;
    }

    std::unique_ptr<llama_server_bridge> bridge = std::make_unique<llama_server_bridge>();
    if (has_model_path) {
        bridge->params.model.path = params->model_path;
    }

    if (params->mmproj_path != nullptr && params->mmproj_path[0] != '\0') {
        bridge->params.mmproj.path = params->mmproj_path;
    }

    const llama_server_bridge_params defaults = llama_server_bridge_default_params();
    const bool zero_init_params = looks_like_zero_initialized_params(params);

    auto choose_i32 = [&](int32_t raw, int32_t fallback) -> int32_t {
        return (zero_init_params && raw == 0) ? fallback : raw;
    };

    const int32_t requested_n_ctx = std::max<int32_t>(0, choose_i32(params->n_ctx, defaults.n_ctx));
    const int32_t requested_n_batch = std::max<int32_t>(32, choose_i32(params->n_batch, defaults.n_batch));
    const int32_t requested_n_ubatch = std::max<int32_t>(32, choose_i32(params->n_ubatch, defaults.n_ubatch));
    const int32_t requested_n_parallel = std::max<int32_t>(1, choose_i32(params->n_parallel, defaults.n_parallel));
    const int32_t requested_n_threads = choose_i32(params->n_threads, defaults.n_threads);
    const int32_t requested_n_threads_batch = choose_i32(params->n_threads_batch, defaults.n_threads_batch);
    const int32_t requested_n_gpu_layers = choose_i32(params->n_gpu_layers, defaults.n_gpu_layers);
    int32_t requested_main_gpu = choose_i32(params->main_gpu, defaults.main_gpu);
    requested_main_gpu = std::max<int32_t>(-1, requested_main_gpu);
    int32_t requested_gpu = choose_i32(params->gpu, defaults.gpu);
    requested_gpu = std::max<int32_t>(-1, requested_gpu);
    const int32_t requested_no_kv_offload = choose_i32(params->no_kv_offload, defaults.no_kv_offload);
    const int32_t requested_mmproj_use_gpu = choose_i32(params->mmproj_use_gpu, defaults.mmproj_use_gpu);
    const int32_t requested_cache_ram_mib = choose_i32(params->cache_ram_mib, defaults.cache_ram_mib);
    const int32_t requested_seed = choose_i32(params->seed, defaults.seed);
    const int32_t requested_ctx_shift = choose_i32(params->ctx_shift, defaults.ctx_shift);
    const int32_t requested_kv_unified = choose_i32(params->kv_unified, defaults.kv_unified);
    const int32_t requested_split_mode = choose_i32(params->split_mode, defaults.split_mode);
    const int32_t requested_pooling_type = choose_i32(params->pooling_type, defaults.pooling_type);

    const char * requested_devices = params->devices;
    if (zero_init_params && is_devices_unset_or_none(requested_devices)) {
        requested_devices = defaults.devices;
    }
    const char * requested_tensor_split = params->tensor_split;
    if (zero_init_params && requested_tensor_split != nullptr && trim_copy(requested_tensor_split).empty()) {
        requested_tensor_split = defaults.tensor_split;
    }

    if (requested_mmproj_use_gpu < -1 || requested_mmproj_use_gpu > 1) {
        set_bridge_error(bridge.get(), "invalid mmproj_use_gpu, expected -1/0/1");
        std::fprintf(stderr, "llama_server_bridge_create: invalid mmproj_use_gpu=%d\n", requested_mmproj_use_gpu);
        return nullptr;
    }
    if (requested_gpu >= 0 && !is_devices_unset_or_none(requested_devices)) {
        set_bridge_error(bridge.get(), "choose one: gpu OR devices");
        std::fprintf(stderr, "llama_server_bridge_create: both gpu=%d and devices are set\n", requested_gpu);
        return nullptr;
    }

    bridge->params.n_ctx = requested_n_ctx;
    bridge->params.n_batch = requested_n_batch;
    bridge->params.n_ubatch = requested_n_ubatch;
    bridge->params.n_parallel = requested_n_parallel;
    bridge->params.n_gpu_layers = requested_n_gpu_layers;
    bridge->params.main_gpu = requested_main_gpu;
    bridge->params.no_kv_offload = requested_no_kv_offload != 0;
    bridge->params.mmproj_use_gpu = requested_mmproj_use_gpu > 0;
    bridge->params.no_mmproj = bridge->params.mmproj.path.empty();
    bridge->params.cache_ram_mib = requested_cache_ram_mib;
    bridge->params.ctx_shift = requested_ctx_shift != 0;
    bridge->params.kv_unified = requested_kv_unified != 0;
    bridge->params.embedding = params->embedding != 0;
    // Default to single-device mode unless explicitly overridden by split_mode.
    bridge->params.split_mode = LLAMA_SPLIT_MODE_NONE;

    if (requested_pooling_type >= 0) {
        if (!is_valid_pooling_type(requested_pooling_type)) {
            set_bridge_error(bridge.get(), "invalid pooling_type, expected -1..4");
            std::fprintf(stderr, "llama_server_bridge_create: invalid pooling_type=%d\n", requested_pooling_type);
            return nullptr;
        }
        bridge->params.pooling_type = static_cast<enum llama_pooling_type>(requested_pooling_type);
    }
    if (params->reranking != 0) {
        bridge->params.embedding = true;
        bridge->params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }
    if (bridge->params.embedding) {
        bridge->params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    std::string parse_error;
    if (!parse_split_mode(requested_split_mode, &bridge->params.split_mode, parse_error)) {
        set_bridge_error(bridge.get(), parse_error);
        std::fprintf(stderr, "llama_server_bridge_create: split_mode parse failed: %s\n", parse_error.c_str());
        return nullptr;
    }
    if (!parse_tensor_split_csv(requested_tensor_split, bridge->params.tensor_split, parse_error)) {
        set_bridge_error(bridge.get(), parse_error);
        std::fprintf(stderr, "llama_server_bridge_create: tensor_split parse failed: %s\n", parse_error.c_str());
        return nullptr;
    }

    // NOTE: do not remap list-devices indices through *VISIBLE_DEVICES env vars.
    // Vulkan visibility uses a different index space (physical-device enumeration),
    // which can misroute selected devices (e.g. --gpu 1 landing on Vulkan0).
    std::string effective_devices_storage;
    const char * effective_devices_csv = requested_devices;
    if (requested_gpu >= 0) {
        effective_devices_storage = std::to_string(requested_gpu);
        effective_devices_csv = effective_devices_storage.c_str();
        if (bridge->params.split_mode == LLAMA_SPLIT_MODE_NONE) {
            // Single-device shortcut: main_gpu is index inside selected devices.
            bridge->params.main_gpu = 0;
        }
    } else if (is_devices_unset_or_none(effective_devices_csv)) {
        const bool infer_from_main_gpu = requested_main_gpu > 0 || (!zero_init_params && requested_main_gpu >= 0);
        if (infer_from_main_gpu) {
            // Allow host apps to pass only main_gpu as a simple device selector.
            effective_devices_storage = std::to_string(requested_main_gpu);
            effective_devices_csv = effective_devices_storage.c_str();
        }
    }

    if (requested_n_threads > 0) {
        bridge->params.cpuparams.n_threads = requested_n_threads;
    }
    if (requested_n_threads_batch > 0) {
        bridge->params.cpuparams_batch.n_threads = requested_n_threads_batch;
    } else if (requested_n_threads > 0) {
        bridge->params.cpuparams_batch.n_threads = requested_n_threads;
    }
    if (requested_seed >= 0) {
        bridge->params.sampling.seed = (uint32_t) requested_seed;
    }

    acquire_backend(bridge->params);
    bridge->backend_acquired = true;

#if defined(__APPLE__)
    if (is_devices_unset(effective_devices_csv)) {
        const int32_t mac_default_gpu = find_first_gpu_backend_device_index();
        if (mac_default_gpu >= 0) {
            // On macOS, no explicit device means "use the default Metal GPU".
            effective_devices_storage = std::to_string(mac_default_gpu);
            effective_devices_csv = effective_devices_storage.c_str();
        }
    }
#endif

    if (!parse_devices_csv(effective_devices_csv, bridge->params.devices, parse_error)) {
        set_bridge_error(bridge.get(), parse_error);
        std::fprintf(stderr, "llama_server_bridge_create: devices parse failed: %s\n", parse_error.c_str());
        release_backend();
        bridge->backend_acquired = false;
        return nullptr;
    }

    ggml_backend_dev_t primary_dev = nullptr;
    for (auto * dev : bridge->params.devices) {
        if (dev != nullptr) {
            primary_dev = dev;
            break;
        }
    }
    if (primary_dev != nullptr) {
        bridge->primary_device_index = find_backend_device_index(primary_dev);
        const char * dev_name = ggml_backend_dev_name(primary_dev);
        if (dev_name != nullptr) {
            bridge->primary_device_name = dev_name;
        }
        bridge->primary_device_is_gpu = is_gpu_device_type(ggml_backend_dev_type(primary_dev));
    }

    if (primary_dev == nullptr) {
        bridge->params.main_gpu = -1;
    } else if (bridge->params.split_mode == LLAMA_SPLIT_MODE_NONE && bridge->params.main_gpu < 0) {
        // In single-device mode, use the first selected device by default.
        bridge->params.main_gpu = 0;
    }
    if (!bridge->primary_device_is_gpu && bridge->params.n_gpu_layers < 0) {
        bridge->params.n_gpu_layers = 0;
    }
    if (requested_mmproj_use_gpu < 0) {
        bridge->params.mmproj_use_gpu = !bridge->params.no_mmproj && bridge->primary_device_is_gpu;
    }
    if (bridge->params.no_mmproj) {
        bridge->params.mmproj_use_gpu = false;
    }
    if (bridge->params.mmproj_use_gpu && bridge->primary_device_is_gpu && !bridge->primary_device_name.empty()) {
        // Let mtmd/mmproj follow the selected bridge device unless the host app already pinned it.
        set_env_if_unset("MTMD_BACKEND_DEVICE", bridge->primary_device_name);
    }

    if (has_model_path) {
        if (!bridge->ctx.load_model(bridge->params)) {
            set_bridge_error(bridge.get(), "failed to load model in llama_server_bridge_create()");
            std::fprintf(stderr, "llama_server_bridge_create: load_model failed for model path '%s'\n", params->model_path);
            release_backend();
            bridge->backend_acquired = false;
            return nullptr;
        }

        auto meta = bridge->ctx.get_meta();
        bridge->model_name = meta.model_name;
    } else {
        bridge->model_name = "audio-only";
    }

    bridge->routes = std::make_unique<server_routes>(bridge->params, bridge->ctx);
    if (has_model_path) {
        bridge->routes->update_meta(bridge->ctx);
    }

    // In audio-only/no-model mode, the transcription route runs directly and does
    // not require the server task loop. Starting it without a loaded model can
    // hit model-dependent update paths in server_context.
    if (has_model_path) {
        bridge->loop_thread = std::thread([raw = bridge.get()]() {
            raw->ctx.start_loop();
        });
    }

    return bridge.release();
}

void llama_server_bridge_destroy(struct llama_server_bridge * bridge) {
    if (bridge == nullptr) {
        return;
    }
    bridge->ctx.terminate();
    if (bridge->loop_thread.joinable()) {
        bridge->loop_thread.join();
    }
    bridge->routes.reset();
    if (bridge->backend_acquired) {
        release_backend();
        bridge->backend_acquired = false;
    }
    delete bridge;
}

struct llama_server_bridge_realtime * llama_server_bridge_realtime_sortformer_create(
    const struct llama_server_bridge_realtime_sortformer_params * params) {
    if (params == nullptr) {
        std::fprintf(stderr, "llama_server_bridge_realtime_sortformer_create: params is null\n");
        return nullptr;
    }
    const auto generic = sortformer_params_to_generic(params);
    return llama_server_bridge_realtime_create(&generic);
}

llama_server_bridge_realtime_sortformer_model * llama_server_bridge_realtime_sortformer_model_create(
    const struct llama_server_bridge_realtime_sortformer_params * params) {
    if (params == nullptr) {
        std::fprintf(stderr, "llama_server_bridge_realtime_sortformer_model_create: params is null\n");
        return nullptr;
    }
    const auto generic = sortformer_params_to_generic(params);
    return reinterpret_cast<llama_server_bridge_realtime_sortformer_model *>(
        llama_server_bridge_realtime_model_create(&generic));
}

void llama_server_bridge_realtime_sortformer_model_destroy(
    llama_server_bridge_realtime_sortformer_model * model) {
    llama_server_bridge_realtime_model_destroy(
        reinterpret_cast<llama_server_bridge_realtime_model *>(model));
}

struct llama_server_bridge_realtime * llama_server_bridge_realtime_sortformer_create_from_model(
    const llama_server_bridge_realtime_sortformer_model * model,
    const struct llama_server_bridge_realtime_sortformer_params * params) {
    if (model == nullptr) {
        std::fprintf(stderr, "llama_server_bridge_realtime_sortformer_create_from_model: model is null\n");
        return nullptr;
    }
    const auto generic = sortformer_params_to_generic(params);
    return llama_server_bridge_realtime_create_from_model(
        reinterpret_cast<const llama_server_bridge_realtime_model *>(model),
        &generic);
}

struct llama_server_bridge_realtime * llama_server_bridge_realtime_create(
    const struct llama_server_bridge_realtime_params * params) {

    realtime_resolved_params translated = {};
    std::string parse_error;
    if (!translate_generic_realtime_params(
            params,
            "llama_server_bridge_realtime_create",
            translated,
            parse_error)) {
        std::fprintf(stderr, "%s\n", parse_error.c_str());
        return nullptr;
    }
    auto bridge = std::make_unique<llama_server_bridge_realtime>();
    try {
        if (llama::realtime::backend_supports_model_preload(translated.backend_params.kind)) {
            bridge->loaded_model = load_backend_model_maybe_cached(translated.backend_params);
        } else {
            acquire_backend(bridge->params);
            bridge->backend_acquired = true;
        }
        auto backend = bridge->loaded_model
            ? bridge->loaded_model->create_backend(translated.capture_debug)
            : llama::realtime::create_backend(translated.backend_params, translated.capture_debug);
        bridge->session_id = bridge->manager.create_session(std::move(backend), translated.session_cfg);
        set_realtime_error(bridge.get(), "");
        return bridge.release();
    } catch (const std::exception & e) {
        const std::string msg = normalize_error(e.what());
        set_realtime_error(bridge.get(), msg);
        std::fprintf(stderr, "llama_server_bridge_realtime_create: %s\n", msg.c_str());
        if (bridge->backend_acquired) {
            release_backend();
            bridge->backend_acquired = false;
        }
        bridge->loaded_model.reset();
        return nullptr;
    } catch (...) {
        const std::string msg = "unknown realtime backend creation failure";
        set_realtime_error(bridge.get(), msg);
        std::fprintf(stderr, "llama_server_bridge_realtime_create: %s\n", msg.c_str());
        if (bridge->backend_acquired) {
            release_backend();
            bridge->backend_acquired = false;
        }
        bridge->loaded_model.reset();
        return nullptr;
    }
}

llama_server_bridge_realtime_model * llama_server_bridge_realtime_model_create(
    const struct llama_server_bridge_realtime_params * params) {

    realtime_resolved_params translated = {};
    std::string parse_error;
    if (!translate_generic_realtime_params(
            params,
            "llama_server_bridge_realtime_model_create",
            translated,
            parse_error)) {
        std::fprintf(stderr, "%s\n", parse_error.c_str());
        return nullptr;
    }

    const int32_t resolved_backend_kind = static_cast<int32_t>(translated.backend_params.kind);
    if (!llama_server_bridge_realtime_backend_supports_model_preload(resolved_backend_kind)) {
        std::fprintf(stderr, "llama_server_bridge_realtime_model_create: backend does not support model preload\n");
        return nullptr;
    }

    auto model = std::make_unique<llama_server_bridge_realtime_model_impl>();
    try {
        RT_TRACE("rt_model_create: begin kind=%d model=%s backend=%s",
            resolved_backend_kind,
            translated.backend_params.model_path.c_str(),
            translated.backend_params.backend_name.c_str());
        model->backend_kind = resolved_backend_kind;
        auto cached_model = load_backend_model_maybe_cached(translated.backend_params);
        (void) cached_model;
        model->model_path = translated.backend_params.model_path;
        model->resolved_runtime_backend_name = translated.backend_params.backend_name;
        model->default_session_cfg = translated.session_cfg;
        RT_TRACE("rt_model_create: loaded kind=%d runtime=%s sr=%u ring=%u",
            model->backend_kind,
            model->resolved_runtime_backend_name.c_str(),
            model->default_session_cfg.expected_sample_rate_hz,
            static_cast<unsigned>(model->default_session_cfg.audio_ring_capacity_samples));
        set_realtime_model_error(model.get(), "");
        return reinterpret_cast<llama_server_bridge_realtime_model *>(model.release());
    } catch (const std::exception & e) {
        const std::string msg = normalize_error(e.what());
        set_realtime_model_error(model.get(), msg);
        std::fprintf(stderr, "llama_server_bridge_realtime_model_create: %s\n", msg.c_str());
        if (model->backend_acquired) {
            release_backend();
            model->backend_acquired = false;
        }
        return nullptr;
    } catch (...) {
        const std::string msg = "unknown realtime model creation failure";
        set_realtime_model_error(model.get(), msg);
        std::fprintf(stderr, "llama_server_bridge_realtime_model_create: %s\n", msg.c_str());
        if (model->backend_acquired) {
            release_backend();
            model->backend_acquired = false;
        }
        return nullptr;
    }
}

int32_t llama_server_bridge_realtime_model_get_info(
    const llama_server_bridge_realtime_model * model,
    struct llama_server_bridge_realtime_backend_info * out_info) {

    if (out_info == nullptr) {
        return 0;
    }

    *out_info = llama_server_bridge_empty_realtime_backend_info();
    const auto * typed_model = reinterpret_cast<const struct llama_server_bridge_realtime_model_impl *>(model);
    if (typed_model == nullptr) {
        return 0;
    }

    *out_info = make_realtime_backend_info_for_model(*typed_model);
    RT_TRACE("rt_model_get_info: kind=%d runtime=%s sr=%u ring=%u",
        out_info->backend_kind,
        out_info->default_runtime_backend_name != nullptr ? out_info->default_runtime_backend_name : "(null)",
        out_info->default_sample_rate_hz,
        static_cast<unsigned>(out_info->default_audio_ring_capacity_samples));
    return out_info->backend_kind != 0 ? 1 : 0;
}

void llama_server_bridge_realtime_model_destroy(
    llama_server_bridge_realtime_model * model) {
    auto typed_model = reinterpret_cast<struct llama_server_bridge_realtime_model_impl *>(model);
    if (typed_model == nullptr) {
        return;
    }
    if (typed_model->backend_acquired) {
        release_backend();
        typed_model->backend_acquired = false;
    }
    delete typed_model;
}

struct llama_server_bridge_realtime * llama_server_bridge_realtime_create_from_model(
    const llama_server_bridge_realtime_model * model,
    const struct llama_server_bridge_realtime_params * params) {

    if (model == nullptr) {
        std::fprintf(stderr, "llama_server_bridge_realtime_create_from_model: model is null\n");
        return nullptr;
    }

    auto typed_model = reinterpret_cast<const struct llama_server_bridge_realtime_model_impl *>(model);
    realtime_resolved_params translated = {};
    std::string parse_error;
    if (!translate_realtime_session_params_from_loaded_model(
            typed_model,
            params,
            "llama_server_bridge_realtime_create_from_model",
            translated,
            parse_error)) {
        std::fprintf(stderr, "%s\n", parse_error.c_str());
        return nullptr;
    }

    auto bridge = std::make_unique<llama_server_bridge_realtime>();
    try {
        RT_TRACE("rt_create_from_model: begin kind=%d model=%s runtime=%s sr=%u ring=%u",
            typed_model->backend_kind,
            translated.backend_params.model_path.c_str(),
            translated.backend_params.backend_name.c_str(),
            translated.session_cfg.expected_sample_rate_hz,
            static_cast<unsigned>(translated.session_cfg.audio_ring_capacity_samples));
        bridge->loaded_model = load_backend_model_maybe_cached(translated.backend_params);
        RT_TRACE("rt_create_from_model: cache/model acquired");
        auto backend = bridge->loaded_model->create_backend(translated.capture_debug);
        RT_TRACE("rt_create_from_model: backend created");
        bridge->session_id = bridge->manager.create_session(std::move(backend), translated.session_cfg);
        RT_TRACE("rt_create_from_model: session created id=%lld", static_cast<long long>(bridge->session_id));
        set_realtime_error(bridge.get(), "");
        return bridge.release();
    } catch (const std::exception & e) {
        const std::string msg = normalize_error(e.what());
        set_realtime_error(bridge.get(), msg);
        std::fprintf(stderr, "llama_server_bridge_realtime_create_from_model: %s\n", msg.c_str());
        return nullptr;
    } catch (...) {
        const std::string msg = "unknown realtime session creation from model failure";
        set_realtime_error(bridge.get(), msg);
        std::fprintf(stderr, "llama_server_bridge_realtime_create_from_model: %s\n", msg.c_str());
        return nullptr;
    }
}

void llama_server_bridge_realtime_destroy(struct llama_server_bridge_realtime * bridge) {
    if (bridge == nullptr) {
        return;
    }

    try {
        RT_TRACE("rt_destroy: session=%lld has=%d",
            static_cast<long long>(bridge->session_id),
            (bridge->session_id != 0 && bridge->manager.has_session(bridge->session_id)) ? 1 : 0);
        if (bridge->session_id != 0 && bridge->manager.has_session(bridge->session_id)) {
            bridge->manager.close_session(bridge->session_id);
        }
    } catch (...) {
    }

    if (bridge->backend_acquired) {
        release_backend();
        bridge->backend_acquired = false;
    }
    delete bridge;
}

int32_t llama_server_bridge_realtime_push_audio_f32(
    struct llama_server_bridge_realtime * bridge,
    const float * samples,
    size_t n_samples,
    uint32_t sample_rate_hz) {

    if (bridge == nullptr || samples == nullptr) {
        return -1;
    }

    try {
        bridge->manager.push_audio(bridge->session_id, samples, n_samples, sample_rate_hz);
        set_realtime_error(bridge, "");
        return 0;
    } catch (const std::exception & e) {
        set_realtime_error(bridge, normalize_error(e.what()));
        return -1;
    } catch (...) {
        set_realtime_error(bridge, "unknown realtime push_audio failure");
        return -1;
    }
}

int32_t llama_server_bridge_realtime_flush(struct llama_server_bridge_realtime * bridge) {
    if (bridge == nullptr) {
        return -1;
    }

    try {
        bridge->manager.flush_session(bridge->session_id);
        set_realtime_error(bridge, "");
        return 0;
    } catch (const std::exception & e) {
        set_realtime_error(bridge, normalize_error(e.what()));
        return -1;
    } catch (...) {
        set_realtime_error(bridge, "unknown realtime flush failure");
        return -1;
    }
}

int32_t llama_server_bridge_realtime_drain_events(
    struct llama_server_bridge_realtime * bridge,
    struct llama_server_bridge_realtime_event ** out_events,
    size_t * out_count,
    size_t max_events) {

    if (bridge == nullptr || out_events == nullptr || out_count == nullptr) {
        return -1;
    }

    *out_events = nullptr;
    *out_count = 0;

    try {
        const auto events = bridge->manager.drain_events(bridge->session_id, max_events);
        if (events.empty()) {
            set_realtime_error(bridge, "");
            return 0;
        }

        auto * out = static_cast<llama_server_bridge_realtime_event *>(
            std::calloc(events.size(), sizeof(llama_server_bridge_realtime_event)));
        if (out == nullptr) {
            set_realtime_error(bridge, "failed to allocate realtime event array");
            return -1;
        }

        for (size_t i = 0; i < events.size(); ++i) {
            out[i].type = static_cast<int32_t>(events[i].type);
            out[i].session_id = events[i].session_id;
            out[i].begin_sec = events[i].begin_sec;
            out[i].end_sec = events[i].end_sec;
            out[i].speaker_id = events[i].speaker_id;
            out[i].text = copy_to_c_string(events[i].text);
            out[i].detail = copy_to_c_string(events[i].detail);
        }

        *out_events = out;
        *out_count = events.size();
        set_realtime_error(bridge, "");
        return 0;
    } catch (const std::exception & e) {
        set_realtime_error(bridge, normalize_error(e.what()));
        return -1;
    } catch (...) {
        set_realtime_error(bridge, "unknown realtime drain_events failure");
        return -1;
    }
}

void llama_server_bridge_realtime_free_events(
    struct llama_server_bridge_realtime_event * events,
    size_t count) {

    if (events == nullptr) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        std::free(events[i].text);
        std::free(events[i].detail);
    }
    std::free(events);
}

enum class audio_session_realtime_lane {
    diarization,
    transcription,
};

static void translate_realtime_events_locked(
    llama_server_bridge_audio_session * session,
    const std::vector<llama::realtime::event> & events,
    uint32_t extra_flags,
    audio_session_realtime_lane lane) {

    if (session == nullptr) {
        return;
    }

    for (const auto & ev : events) {
        audio_session_event_record out = {};
        out.flags = extra_flags | LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
        out.start_sample = seconds_to_samples(ev.begin_sec, session->params.expected_input_sample_rate_hz);
        out.end_sample = seconds_to_samples(ev.end_sec, session->params.expected_input_sample_rate_hz);
        out.speaker_id = ev.speaker_id;
        out.text = ev.text;
        out.detail = ev.detail;

        using llama::realtime::event_type;
        switch (ev.type) {
            case event_type::speaker_span_commit:
                out.kind = lane == audio_session_realtime_lane::diarization
                    ? LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT
                    : LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE;
                out.item_id = next_audio_session_item_id_locked(session);
                if (out.text.empty()) {
                    out.text = audio_session_speaker_label(ev.speaker_id);
                }
                break;
            case event_type::transcript_commit:
                out.kind = lane == audio_session_realtime_lane::diarization
                    ? LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT
                    : LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT;
                out.item_id = next_audio_session_item_id_locked(session);
                break;
            case event_type::transcript_piece_commit:
                out.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT;
                out.item_id = next_audio_session_item_id_locked(session);
                break;
            case event_type::transcript_word_commit:
                out.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT;
                out.item_id = next_audio_session_item_id_locked(session);
                break;
            case event_type::backend_status:
                out.kind = lane == audio_session_realtime_lane::diarization
                    ? LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_STATUS
                    : LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE;
                out.item_id = next_audio_session_item_id_locked(session);
                break;
            case event_type::backend_error:
                out.kind = lane == audio_session_realtime_lane::diarization
                    ? LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR
                    : LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR;
                out.item_id = next_audio_session_item_id_locked(session);
                break;
            case event_type::session_notice:
            default:
                out.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE;
                out.item_id = 0;
                break;
        }
        push_audio_session_event_locked(session, std::move(out));
    }
}

static void drain_realtime_bridge_events_locked(
    llama_server_bridge_audio_session * session,
    llama_server_bridge_realtime * bridge,
    uint32_t extra_flags,
    audio_session_realtime_lane lane) {

    if (session == nullptr || bridge == nullptr) {
        return;
    }
    const auto events = bridge->manager.drain_events(bridge->session_id);
    translate_realtime_events_locked(session, events, extra_flags, lane);
}

static void replay_audio_into_realtime_bridge_locked(
    llama_server_bridge_audio_session * session,
    llama_server_bridge_realtime * bridge,
    audio_session_realtime_lane lane) {

    if (session == nullptr || bridge == nullptr) {
        return;
    }

    size_t offset = 0;
    for (const size_t chunk_size : session->chunk_sizes) {
        if (chunk_size == 0) {
            continue;
        }
        bridge->manager.push_audio(
            bridge->session_id,
            session->audio_samples.data() + offset,
            chunk_size,
            session->params.expected_input_sample_rate_hz);
        offset += chunk_size;
        drain_realtime_bridge_events_locked(
            session,
            bridge,
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY,
            lane);
    }
}

static int append_audio_chunk_locked(
    llama_server_bridge_audio_session * session,
    const float * samples,
    size_t n_samples,
    uint32_t event_flags) {

    if (session == nullptr || samples == nullptr) {
        return -1;
    }

    if (n_samples == 0) {
        return 0;
    }
    if (session->audio_flushed) {
        return -1;
    }
    if (session->params.max_buffered_audio_samples > 0) {
        const size_t limit = static_cast<size_t>(session->params.max_buffered_audio_samples);
        if (session->audio_samples.size() + n_samples > limit) {
            return -1;
        }
    }

    session->audio_samples.insert(session->audio_samples.end(), samples, samples + n_samples);
    session->chunk_sizes.push_back(n_samples);

    if (session->diarization_bridge == nullptr) {
        if (session->transcription_bridge == nullptr) {
            return 0;
        }
    }

    if (session->diarization_bridge != nullptr) {
        session->diarization_bridge->manager.push_audio(
            session->diarization_bridge->session_id,
            samples,
            n_samples,
            session->params.expected_input_sample_rate_hz);
        drain_realtime_bridge_events_locked(
            session,
            session->diarization_bridge,
            event_flags,
            audio_session_realtime_lane::diarization);
    }
    if (session->transcription_bridge != nullptr) {
        session->transcription_bridge->manager.push_audio(
            session->transcription_bridge->session_id,
            samples,
            n_samples,
            session->params.expected_input_sample_rate_hz);
        drain_realtime_bridge_events_locked(
            session,
            session->transcription_bridge,
            event_flags,
            audio_session_realtime_lane::transcription);
    }
    return 0;
}

static int start_audio_session_native_transcription(
    llama_server_bridge_audio_session * session,
    const owned_audio_transcription_params & transcription_params) {

    if (session == nullptr) {
        return -1;
    }

    join_audio_session_transcription_thread(session);

    auto resolved = transcription_params.realtime_params.borrow();
    if (resolved.expected_sample_rate_hz == 0) {
        resolved.expected_sample_rate_hz = session->params.expected_input_sample_rate_hz;
    }
    realtime_resolved_params translated = {};
    std::string parse_error;
    if (!translate_generic_realtime_params(
            &resolved,
            "start_audio_session_native_transcription",
            translated,
            parse_error)) {
        return fail_audio_session(session, parse_error);
    }

    llama_server_bridge_realtime * bridge_to_destroy = nullptr;
    llama_server_bridge_realtime * bridge = nullptr;
    try {
        if (translated.session_cfg.expected_sample_rate_hz != session->params.expected_input_sample_rate_hz) {
            return fail_audio_session(
                session,
                "native transcription expected_sample_rate_hz must match the audio session sample rate");
        }

        bool audio_already_flushed = false;
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            audio_already_flushed = session->audio_flushed;
        }

        std::shared_ptr<llama::realtime::loaded_backend_model> offline_model;
        if (audio_already_flushed) {
            offline_model = load_backend_model_maybe_cached(translated.backend_params);
        }
        if (offline_model && offline_model->supports_offline_transcription()) {
            bool already_running = false;
            bool missing_audio = false;
            std::vector<float> audio_snapshot;
            {
                std::lock_guard<std::mutex> lock(session->mutex);
                if (session->transcription_running || session->transcription_bridge != nullptr) {
                    already_running = true;
                } else if (session->audio_samples.empty()) {
                    missing_audio = true;
                } else {
                    session->transcription_requested = true;
                    session->transcription_started = true;
                    session->transcription_running = true;
                    session->transcription_completed = false;
                    session->transcription_stop_requested = false;
                    session->transcription_native_realtime = true;
                    audio_snapshot = session->audio_samples;

                    audio_session_event_record started = {};
                    started.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STARTED;
                    started.detail = resolved.model_path != nullptr ? resolved.model_path : "native_offline";
                    push_audio_session_event_locked(session, std::move(started));
                }
            }
            if (already_running) {
                return fail_audio_session(session, "transcription is already running for this audio session");
            }
            if (missing_audio) {
                return fail_audio_session(session, "cannot start native offline transcription without buffered audio");
            }

            set_audio_session_error(session, "");
            session->transcription_thread = std::thread(
                [session,
                 audio_snapshot = std::move(audio_snapshot),
                 offline_model = std::move(offline_model),
                 capture_debug = translated.capture_debug]() mutable {
                    std::string error_message;
                    bool success = false;
                    std::vector<llama::realtime::event> native_events;

                    try {
                        offline_model->transcribe_audio_offline(
                            audio_snapshot,
                            native_events,
                            capture_debug);
                        success = true;
                    } catch (const std::exception & e) {
                        error_message = normalize_error(e.what());
                    } catch (...) {
                        error_message = "unknown native offline transcription failure";
                    }

                    if (!success) {
                        fail_audio_session(
                            session,
                            error_message.empty()
                                ? "native offline transcription failed"
                                : error_message);
                    } else {
                        set_audio_session_error(session, "");
                        std::lock_guard<std::mutex> lock(session->mutex);
                        translate_realtime_events_locked(
                            session,
                            native_events,
                            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY,
                            audio_session_realtime_lane::transcription);
                    }

                    {
                        std::lock_guard<std::mutex> lock(session->mutex);
                        session->transcription_running = false;
                        session->transcription_requested = false;
                        session->transcription_completed = success;
                        session->transcription_stop_requested = false;
                        session->transcription_native_realtime = false;
                        audio_session_event_record stopped = {};
                        stopped.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED;
                        stopped.flags = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
                        stopped.end_sample = static_cast<uint64_t>(audio_snapshot.size());
                        stopped.detail = success ? "completed" : "failed";
                        push_audio_session_event_locked(session, std::move(stopped));
                    }
                });
            return 0;
        }

        bridge = llama_server_bridge_realtime_create(&resolved);
        if (bridge == nullptr) {
            return fail_audio_session(session, "failed to create native realtime transcription bridge");
        }

        bool already_running = false;
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->transcription_running || session->transcription_bridge != nullptr) {
                already_running = true;
            } else {
                session->transcription_bridge = bridge;
                session->transcription_requested = true;
                session->transcription_started = true;
                session->transcription_running = true;
                session->transcription_completed = false;
                session->transcription_stop_requested = false;
                session->transcription_native_realtime = true;

                audio_session_event_record started = {};
                started.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STARTED;
                started.detail = resolved.model_path != nullptr ? resolved.model_path : "native_realtime";
                push_audio_session_event_locked(session, std::move(started));

                replay_audio_into_realtime_bridge_locked(
                    session,
                    session->transcription_bridge,
                    audio_session_realtime_lane::transcription);

                if (session->audio_flushed) {
                    session->transcription_bridge->manager.flush_session(session->transcription_bridge->session_id);
                    drain_realtime_bridge_events_locked(
                        session,
                        session->transcription_bridge,
                        LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY,
                        audio_session_realtime_lane::transcription);

                    bridge_to_destroy = session->transcription_bridge;
                    session->transcription_bridge = nullptr;
                    session->transcription_requested = false;
                    session->transcription_running = false;
                    session->transcription_completed = true;
                    session->transcription_native_realtime = false;

                    audio_session_event_record stopped = {};
                    stopped.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED;
                    stopped.flags = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
                    stopped.end_sample = static_cast<uint64_t>(session->audio_samples.size());
                    stopped.detail = "completed";
                    push_audio_session_event_locked(session, std::move(stopped));
                }
            }
        }
        if (already_running) {
            destroy_realtime_bridge(bridge);
            return fail_audio_session(session, "transcription is already running for this audio session");
        }
        if (bridge_to_destroy == nullptr) {
            bridge = nullptr;
        }
        destroy_realtime_bridge(bridge_to_destroy);
        set_audio_session_error(session, "");
        return 0;
    } catch (const std::exception & e) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->transcription_bridge == bridge) {
                session->transcription_bridge = nullptr;
            }
            session->transcription_requested = false;
            session->transcription_started = false;
            session->transcription_running = false;
            session->transcription_completed = false;
            session->transcription_native_realtime = false;
        }
        destroy_realtime_bridge(bridge);
        destroy_realtime_bridge(bridge_to_destroy);
        return fail_audio_session(session, normalize_error(e.what()));
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->transcription_bridge == bridge) {
                session->transcription_bridge = nullptr;
            }
            session->transcription_requested = false;
            session->transcription_started = false;
            session->transcription_running = false;
            session->transcription_completed = false;
            session->transcription_native_realtime = false;
        }
        destroy_realtime_bridge(bridge);
        destroy_realtime_bridge(bridge_to_destroy);
        return fail_audio_session(session, "unknown native transcription start failure");
    }
}

static int maybe_launch_audio_session_transcription(llama_server_bridge_audio_session * session) {
    if (session == nullptr) {
        return -1;
    }

    std::vector<float> audio_snapshot;
    owned_audio_transcription_params transcription_params = {};
    bool missing_audio = false;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->transcription_requested || !session->audio_flushed) {
            return 0;
        }
        if (session->transcription_params.mode == LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE) {
            return 0;
        }
        if (session->transcription_running || session->transcription_completed) {
            return 0;
        }
        if (session->audio_samples.empty()) {
            missing_audio = true;
        } else {
            session->transcription_running = true;
            session->transcription_started = true;
            session->transcription_stop_requested = false;
            audio_snapshot = session->audio_samples;
            transcription_params = session->transcription_params;
        }
    }
    if (missing_audio) {
        return fail_audio_session(session, "cannot start transcription without buffered audio");
    }

    join_audio_session_transcription_thread(session);
    set_audio_session_error(session, "");
    push_audio_session_event(
        session,
        LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STARTED,
        0,
        0,
        static_cast<uint64_t>(audio_snapshot.size()),
        -1,
        0,
        "",
        "offline_final_only");

    session->transcription_thread = std::thread(
        [session, audio_snapshot = std::move(audio_snapshot), transcription_params = std::move(transcription_params)]() mutable {
            std::string error_message;
            std::string result_json;
            int result_status = 0;
            bool success = false;

            try {
                const auto pcm16 = f32_mono_to_pcm16le_bytes(audio_snapshot);
                const auto wav = pcm16_mono_16k_to_wav(pcm16);

                llama_server_bridge * bridge = nullptr;
                llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
                std::string metadata_json;

                auto borrowed_bridge_params = transcription_params.bridge_params.borrow();
                bridge = llama_server_bridge_create(&borrowed_bridge_params);
                if (bridge == nullptr) {
                    error_message = "failed to create bridge for audio transcription";
                } else {
                    if (!build_audio_session_timeline_metadata(
                            transcription_params.metadata_json,
                            metadata_json,
                            error_message)) {
                        llama_server_bridge_destroy(bridge);
                        bridge = nullptr;
                    }
                }
                if (bridge != nullptr) {
                    llama_server_bridge_audio_raw_request req = llama_server_bridge_default_audio_raw_request();
                    req.audio_bytes = wav.data();
                    req.audio_bytes_len = wav.size();
                    req.audio_format = "wav";
                    req.metadata_json = metadata_json.c_str();
                    req.ffmpeg_convert = 0;

                    const int32_t rc = llama_server_bridge_audio_transcriptions_raw(bridge, &req, &out);
                    result_status = out.status;
                    if (rc == 0 && out.ok != 0 && out.json != nullptr) {
                        result_json = out.json;
                        try {
                            const json root_json = json::parse(result_json);
                            emit_transcription_timeline_events(session, root_json, 0);
                        } catch (const std::exception & e) {
                            success = false;
                            error_message = std::string("failed to parse transcription timeline JSON: ") + e.what();
                        }
                        if (error_message.empty()) {
                            success = true;
                        }
                    } else {
                        const char * bridge_err_ptr = llama_server_bridge_last_error(bridge);
                        const std::string bridge_err = bridge_err_ptr != nullptr ? bridge_err_ptr : "";
                        const std::string out_err = out.error_json != nullptr ? out.error_json : "";
                        error_message = !out_err.empty() ? out_err : bridge_err;
                        if (error_message.empty()) {
                            error_message = "audio transcription request failed";
                        }
                    }
                    llama_server_bridge_json_result_free(&out);
                    llama_server_bridge_destroy(bridge);
                }
            } catch (const std::exception & e) {
                error_message = normalize_error(e.what());
            } catch (...) {
                error_message = "unknown audio transcription failure";
            }

            if (!success) {
                fail_audio_session(session, error_message.empty() ? "audio transcription failed" : error_message);
            } else {
                set_audio_session_error(session, "");
                push_audio_session_event(
                    session,
                    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON,
                    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL,
                    0,
                    static_cast<uint64_t>(audio_snapshot.size()),
                    -1,
                    0,
                    result_json,
                    std::to_string(result_status));
            }

            {
                std::lock_guard<std::mutex> lock(session->mutex);
                session->transcription_running = false;
                session->transcription_requested = false;
                session->transcription_completed = success;
                session->transcription_stop_requested = false;
                audio_session_event_record stopped = {};
                stopped.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED;
                stopped.flags = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
                stopped.end_sample = static_cast<uint64_t>(audio_snapshot.size());
                stopped.detail = success ? "completed" : "failed";
                push_audio_session_event_locked(session, std::move(stopped));
            }
        });
    return 0;
}

struct llama_server_bridge_audio_session * llama_server_bridge_audio_session_create(
    const struct llama_server_bridge_audio_session_params * params) {

    auto session = std::make_unique<llama_server_bridge_audio_session>();
    session->params = params != nullptr
        ? *params
        : llama_server_bridge_default_audio_session_params();

    if (session->params.expected_input_sample_rate_hz == 0) {
        session->params.expected_input_sample_rate_hz = 16000;
    }
    if (session->params.expected_input_channels == 0) {
        session->params.expected_input_channels = 1;
    }
    if (session->params.expected_input_channels != 1) {
        set_audio_session_error(session.get(), "audio session currently supports mono PCM only");
        return nullptr;
    }

    set_audio_session_error(session.get(), "");
    return session.release();
}

void llama_server_bridge_audio_session_destroy(
    struct llama_server_bridge_audio_session * session) {

    if (session == nullptr) {
        return;
    }

    llama_server_bridge_realtime * diarization_bridge = nullptr;
    llama_server_bridge_realtime * transcription_bridge = nullptr;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        diarization_bridge = session->diarization_bridge;
        session->diarization_bridge = nullptr;
        transcription_bridge = session->transcription_bridge;
        session->transcription_bridge = nullptr;
    }
    destroy_realtime_bridge(diarization_bridge);
    destroy_realtime_bridge(transcription_bridge);

    join_audio_session_transcription_thread(session);
    delete session;
}

int32_t llama_server_bridge_audio_session_push_audio(
    struct llama_server_bridge_audio_session * session,
    const void * audio_bytes,
    size_t frame_count,
    uint32_t sample_rate_hz,
    uint32_t channels,
    int32_t sample_format) {

    if (session == nullptr || audio_bytes == nullptr) {
        return -1;
    }
    if (frame_count == 0) {
        set_audio_session_error(session, "");
        return 0;
    }
    if (sample_rate_hz != session->params.expected_input_sample_rate_hz) {
        return fail_audio_session(
            session,
            "audio session sample_rate_hz must match expected_input_sample_rate_hz");
    }
    if (channels != session->params.expected_input_channels) {
        return fail_audio_session(
            session,
            "audio session channel count must match expected_input_channels");
    }

    std::vector<float> mono_f32;
    switch (sample_format) {
        case LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_F32: {
            const auto * samples = static_cast<const float *>(audio_bytes);
            mono_f32.assign(samples, samples + frame_count);
            break;
        }
        case LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16: {
            const auto * samples = static_cast<const int16_t *>(audio_bytes);
            mono_f32.resize(frame_count);
            for (size_t i = 0; i < frame_count; ++i) {
                mono_f32[i] = static_cast<float>(samples[i]) / 32768.0f;
            }
            break;
        }
        default:
            return fail_audio_session(session, "unsupported audio session PCM sample format");
    }

    std::string append_error;
    try {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (append_audio_chunk_locked(session, mono_f32.data(), mono_f32.size(), 0) != 0) {
            append_error = session->audio_flushed
                ? "cannot push audio after audio session flush"
                : "audio session buffered audio limit exceeded";
        } else {
            set_audio_session_error(session, "");
            return 0;
        }
        if (append_error.empty()) {
            append_error = "audio session push failed";
        }
    } catch (const std::exception & e) {
        return fail_audio_session(session, normalize_error(e.what()));
    } catch (...) {
        return fail_audio_session(session, "unknown audio session push failure");
    }
    if (append_error.empty()) {
        append_error = "audio session push failed";
    }
    return fail_audio_session(session, append_error);
}

int32_t llama_server_bridge_audio_session_push_encoded(
    struct llama_server_bridge_audio_session * session,
    const uint8_t * audio_bytes,
    size_t audio_bytes_len,
    const char * audio_format) {

    if (session == nullptr || audio_bytes == nullptr || audio_bytes_len == 0) {
        return -1;
    }
    if (session->params.expected_input_sample_rate_hz != 16000 || session->params.expected_input_channels != 1) {
        return fail_audio_session(
            session,
            "push_encoded currently requires a 16 kHz mono audio session");
    }

    std::vector<uint8_t> wav;
    std::string error;
    if (!ffmpeg_convert_to_wav_pcm16_mono_16k(audio_bytes, audio_bytes_len, audio_format, wav, error)) {
        return fail_audio_session(session, error);
    }

    std::vector<uint8_t> pcm16le;
    if (!wav_payload_to_pcm16le(wav, pcm16le, error)) {
        return fail_audio_session(session, error);
    }
    const auto mono_f32 = pcm16le_bytes_to_f32_mono(pcm16le.data(), pcm16le.size());
    return llama_server_bridge_audio_session_push_audio(
        session,
        mono_f32.data(),
        mono_f32.size(),
        session->params.expected_input_sample_rate_hz,
        1,
        LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_F32);
}

int32_t llama_server_bridge_audio_session_flush_audio(
    struct llama_server_bridge_audio_session * session) {

    if (session == nullptr) {
        return -1;
    }

    llama_server_bridge_realtime * transcription_bridge_to_destroy = nullptr;
    try {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->audio_flushed) {
                set_audio_session_error(session, "");
                return 0;
            }
            session->audio_flushed = true;
            if (session->diarization_bridge != nullptr) {
                session->diarization_bridge->manager.flush_session(session->diarization_bridge->session_id);
                drain_realtime_bridge_events_locked(
                    session,
                    session->diarization_bridge,
                    0,
                    audio_session_realtime_lane::diarization);
            }
            if (session->transcription_bridge != nullptr) {
                session->transcription_bridge->manager.flush_session(session->transcription_bridge->session_id);
                drain_realtime_bridge_events_locked(
                    session,
                    session->transcription_bridge,
                    0,
                    audio_session_realtime_lane::transcription);
                transcription_bridge_to_destroy = session->transcription_bridge;
                session->transcription_bridge = nullptr;
                session->transcription_requested = false;
                session->transcription_running = false;
                session->transcription_completed = true;
                session->transcription_native_realtime = false;

                audio_session_event_record stopped = {};
                stopped.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED;
                stopped.flags = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
                stopped.end_sample = static_cast<uint64_t>(session->audio_samples.size());
                stopped.detail = "completed";
                push_audio_session_event_locked(session, std::move(stopped));
            }
            audio_session_event_record flushed = {};
            flushed.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_STREAM_FLUSHED;
            flushed.flags = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
            flushed.end_sample = static_cast<uint64_t>(session->audio_samples.size());
            push_audio_session_event_locked(session, std::move(flushed));
        }
        destroy_realtime_bridge(transcription_bridge_to_destroy);
        set_audio_session_error(session, "");
        return maybe_launch_audio_session_transcription(session);
    } catch (const std::exception & e) {
        return fail_audio_session(session, normalize_error(e.what()));
    } catch (...) {
        return fail_audio_session(session, "unknown audio session flush failure");
    }
}

int32_t llama_server_bridge_audio_session_start_diarization(
    struct llama_server_bridge_audio_session * session,
    const struct llama_server_bridge_realtime_params * params) {

    if (session == nullptr) {
        return -1;
    }

    const auto defaults = llama_server_bridge_default_realtime_params();
    auto resolved = params != nullptr ? *params : defaults;
    if (resolved.expected_sample_rate_hz == 0) {
        resolved.expected_sample_rate_hz = session->params.expected_input_sample_rate_hz;
    }
    if (resolved.expected_sample_rate_hz != session->params.expected_input_sample_rate_hz) {
        return fail_audio_session(
            session,
            "diarization expected_sample_rate_hz must match the audio session sample rate");
    }

    auto * bridge = llama_server_bridge_realtime_create(&resolved);
    if (bridge == nullptr) {
        return fail_audio_session(session, "failed to create realtime diarization bridge");
    }

    try {
        bool already_running = false;
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->diarization_bridge != nullptr) {
            already_running = true;
        } else {
            session->diarization_bridge = bridge;
            session->diarization_started = true;
            session->diarization_stopped = false;

            audio_session_event_record started = {};
            started.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STARTED;
            started.detail = resolved.model_path != nullptr ? resolved.model_path : "";
            push_audio_session_event_locked(session, std::move(started));

            replay_audio_into_realtime_bridge_locked(
                session,
                session->diarization_bridge,
                audio_session_realtime_lane::diarization);

            if (session->audio_flushed) {
                session->diarization_bridge->manager.flush_session(session->diarization_bridge->session_id);
                drain_realtime_bridge_events_locked(
                    session,
                    session->diarization_bridge,
                    LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY,
                    audio_session_realtime_lane::diarization);
            }
        }
        if (already_running) {
            // handled below outside the session mutex
        } else {
            set_audio_session_error(session, "");
            return 0;
        }
    } catch (const std::exception & e) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->diarization_bridge == bridge) {
                session->diarization_bridge = nullptr;
            }
        }
        llama_server_bridge_realtime_destroy(bridge);
        return fail_audio_session(session, normalize_error(e.what()));
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->diarization_bridge == bridge) {
                session->diarization_bridge = nullptr;
            }
        }
        llama_server_bridge_realtime_destroy(bridge);
        return fail_audio_session(session, "unknown audio session diarization start failure");
    }
    llama_server_bridge_realtime_destroy(bridge);
    return fail_audio_session(session, "diarization is already running for this audio session");
}

int32_t llama_server_bridge_audio_session_stop_diarization(
    struct llama_server_bridge_audio_session * session) {

    if (session == nullptr) {
        return -1;
    }

    llama_server_bridge_realtime * bridge = nullptr;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        bridge = session->diarization_bridge;
        session->diarization_bridge = nullptr;
        if (bridge != nullptr) {
            session->diarization_stopped = true;
            audio_session_event_record stopped = {};
            stopped.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED;
            stopped.flags = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL;
            push_audio_session_event_locked(session, std::move(stopped));
        }
    }
    if (bridge != nullptr) {
        llama_server_bridge_realtime_destroy(bridge);
    }
    set_audio_session_error(session, "");
    return 0;
}

int32_t llama_server_bridge_audio_session_start_transcription(
    struct llama_server_bridge_audio_session * session,
    const struct llama_server_bridge_audio_transcription_params * params) {

    if (session == nullptr) {
        return -1;
    }

    owned_audio_transcription_params owned = {};
    owned.assign(params);
    if (owned.mode == LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE) {
        return start_audio_session_native_transcription(session, owned);
    }

    bool already_running = false;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->transcription_running) {
            already_running = true;
        } else {
            session->transcription_params = std::move(owned);
            session->transcription_requested = true;
            session->transcription_started = false;
            session->transcription_completed = false;
            session->transcription_stop_requested = false;
        }
    }
    if (already_running) {
        return fail_audio_session(session, "transcription is already running for this audio session");
    }
    set_audio_session_error(session, "");
    return maybe_launch_audio_session_transcription(session);
}

int32_t llama_server_bridge_audio_session_stop_transcription(
    struct llama_server_bridge_audio_session * session) {

    if (session == nullptr) {
        return -1;
    }

    bool emit_stopped = false;
    bool running = false;
    llama_server_bridge_realtime * transcription_bridge = nullptr;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->transcription_native_realtime && session->transcription_bridge != nullptr) {
            transcription_bridge = session->transcription_bridge;
            session->transcription_bridge = nullptr;
            session->transcription_requested = false;
            session->transcription_started = false;
            session->transcription_running = false;
            session->transcription_completed = false;
            session->transcription_stop_requested = true;
            session->transcription_native_realtime = false;
            emit_stopped = true;
        } else if (session->transcription_running) {
            running = true;
        } else {
            emit_stopped = session->transcription_requested || session->transcription_started || session->transcription_completed;
            session->transcription_requested = false;
            session->transcription_started = false;
            session->transcription_completed = false;
            session->transcription_stop_requested = true;
            session->transcription_native_realtime = false;
        }
    }
    destroy_realtime_bridge(transcription_bridge);
    if (running) {
        return fail_audio_session(session, "cannot stop transcription while the offline job is already running");
    }

    join_audio_session_transcription_thread(session);

    if (emit_stopped) {
        push_audio_session_event(
            session,
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED,
            LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL,
            0,
            0,
            -1,
            0,
            "",
            "stopped");
    }
    set_audio_session_error(session, "");
    return 0;
}

int32_t llama_server_bridge_audio_session_wait_events(
    struct llama_server_bridge_audio_session * session,
    uint32_t timeout_ms) {

    if (session == nullptr) {
        return -1;
    }

    std::unique_lock<std::mutex> lock(session->mutex);
    if (session->event_queue.empty() && timeout_ms > 0) {
        session->cv.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [session]() { return !session->event_queue.empty(); });
    }
    const size_t count = session->event_queue.size();
    return count > static_cast<size_t>(INT32_MAX) ? INT32_MAX : static_cast<int32_t>(count);
}

int32_t llama_server_bridge_audio_session_drain_events(
    struct llama_server_bridge_audio_session * session,
    struct llama_server_bridge_audio_event ** out_events,
    size_t * out_count,
    size_t max_events) {

    if (session == nullptr || out_events == nullptr || out_count == nullptr) {
        return -1;
    }

    *out_events = nullptr;
    *out_count = 0;

    std::vector<audio_session_event_record> drained;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        const size_t limit = (max_events == 0)
            ? session->event_queue.size()
            : std::min(max_events, session->event_queue.size());
        drained.reserve(limit);
        for (size_t i = 0; i < limit; ++i) {
            drained.push_back(std::move(session->event_queue.front()));
            session->event_queue.pop_front();
        }
    }

    if (drained.empty()) {
        set_audio_session_error(session, "");
        return 0;
    }

    auto * out = static_cast<llama_server_bridge_audio_event *>(
        std::calloc(drained.size(), sizeof(llama_server_bridge_audio_event)));
    if (out == nullptr) {
        return fail_audio_session(session, "failed to allocate audio session event array");
    }

    for (size_t i = 0; i < drained.size(); ++i) {
        out[i].seq_no = drained[i].seq_no;
        out[i].kind = drained[i].kind;
        out[i].flags = drained[i].flags;
        out[i].start_sample = drained[i].start_sample;
        out[i].end_sample = drained[i].end_sample;
        out[i].speaker_id = drained[i].speaker_id;
        out[i].item_id = drained[i].item_id;
        out[i].text = copy_to_c_string(drained[i].text);
        out[i].detail = copy_to_c_string(drained[i].detail);
    }

    *out_events = out;
    *out_count = drained.size();
    set_audio_session_error(session, "");
    return 0;
}

void llama_server_bridge_audio_session_free_events(
    struct llama_server_bridge_audio_event * events,
    size_t count) {

    if (events == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(events[i].text);
        std::free(events[i].detail);
    }
    std::free(events);
}

const char * llama_server_bridge_audio_session_last_error(
    const struct llama_server_bridge_audio_session * session) {

    if (session == nullptr) {
        return "";
    }
    std::lock_guard<std::mutex> lock(session->error_mutex);
    return session->last_error.c_str();
}

static int32_t run_json_route(
    llama_server_bridge * bridge,
    const server_http_context::handler_t & handler,
    const std::string & path,
    const std::string & body,
    struct llama_server_bridge_json_result * out) {

    if (bridge == nullptr || out == nullptr) {
        return -1;
    }
    *out = llama_server_bridge_empty_json_result();

    if (!handler) {
        const std::string msg = "requested route is not available";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    auto finalize = [&](server_http_res_ptr res) -> int32_t {
        if (res == nullptr) {
            const std::string msg = "route returned null response";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        std::string payload = res->data;
        if (res->is_stream()) {
            while (true) {
                std::string chunk;
                const bool has_more = res->next(chunk);
                payload += chunk;
                if (!has_more) {
                    break;
                }
            }
        }

        out->status = res->status;
        if (res->status >= 200 && res->status < 300) {
            out->ok = 1;
            out->json = copy_to_c_string(payload);
            if (out->json == nullptr) {
                const std::string msg = "failed to allocate route JSON output";
                set_bridge_error(bridge, msg);
                out->ok = 0;
                out->error_json = copy_to_c_string(msg);
                return -1;
            }
            set_bridge_error(bridge, "");
            return 0;
        }

        const std::string err = payload.empty() ? "route returned error status with empty payload" : payload;
        set_bridge_error(bridge, err);
        out->error_json = copy_to_c_string(err);
        return -1;
    };

    try {
        const std::function<bool()> should_stop = []() -> bool { return false; };
        const server_http_req req{
            {},
            {},
            path,
            body,
            should_stop
        };
        return finalize(handler(req));
    } catch (const std::exception & e) {
        const std::string msg = normalize_error(e.what());
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    } catch (...) {
        const std::string msg = "unknown exception while running route";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
}

int32_t llama_server_bridge_embeddings(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_embeddings_request * req,
    struct llama_server_bridge_json_result * out) {

    if (bridge == nullptr || req == nullptr || out == nullptr) {
        return -1;
    }
    *out = llama_server_bridge_empty_json_result();

    if (bridge->routes == nullptr) {
        const std::string msg = "server routes are not initialized";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (req->body_json == nullptr || req->body_json[0] == '\0') {
        const std::string msg = "embeddings body_json is empty";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    const bool oai = req->oai_compat != 0;
    return run_json_route(
        bridge,
        oai ? bridge->routes->post_embeddings_oai : bridge->routes->post_embeddings,
        oai ? "/v1/embeddings" : "/embeddings",
        req->body_json,
        out);
}

int32_t llama_server_bridge_rerank(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_rerank_request * req,
    struct llama_server_bridge_json_result * out) {

    if (bridge == nullptr || req == nullptr || out == nullptr) {
        return -1;
    }
    *out = llama_server_bridge_empty_json_result();

    if (bridge->routes == nullptr) {
        const std::string msg = "server routes are not initialized";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (req->body_json == nullptr || req->body_json[0] == '\0') {
        const std::string msg = "rerank body_json is empty";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    return run_json_route(
        bridge,
        bridge->routes->post_rerank,
        "/rerank",
        req->body_json,
        out);
}

int32_t llama_server_bridge_audio_transcriptions(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_audio_request * req,
    struct llama_server_bridge_json_result * out) {

    if (bridge == nullptr || req == nullptr || out == nullptr) {
        return -1;
    }
    *out = llama_server_bridge_empty_json_result();

    if (bridge->routes == nullptr) {
        const std::string msg = "server routes are not initialized";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (req->body_json == nullptr || req->body_json[0] == '\0') {
        const std::string msg = "audio transcriptions body_json is empty";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    const auto handler = resolve_audio_transcriptions_handler(bridge->routes.get());
    if (!handler) {
        const std::string msg =
            "audio transcriptions route is unavailable in this llama.cpp build (missing server audio patch)";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    json body = json::object();
    try {
        body = json::parse(req->body_json);
    } catch (const std::exception & e) {
        const std::string msg = std::string("invalid audio transcriptions body_json: ") + e.what();
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (!body.is_object()) {
        const std::string msg = "audio transcriptions body_json must be a JSON object";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    apply_audio_runtime_device_defaults(bridge, body);
    const std::string body_json = body.dump();

    return run_json_route(
        bridge,
        handler,
        "/v1/audio/transcriptions",
        body_json,
        out);
}

int32_t llama_server_bridge_audio_transcriptions_raw(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_audio_raw_request * req,
    struct llama_server_bridge_json_result * out) {

    if (bridge == nullptr || req == nullptr || out == nullptr) {
        return -1;
    }
    *out = llama_server_bridge_empty_json_result();

    if (bridge->routes == nullptr) {
        const std::string msg = "server routes are not initialized";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    json metadata = json::object();
    std::vector<uint8_t> audio;
    std::string format;
    std::string prep_error;
    if (!prepare_audio_raw_payload(req, metadata, audio, format, prep_error)) {
        set_bridge_error(bridge, prep_error);
        out->error_json = copy_to_c_string(prep_error);
        return -1;
    }
    apply_audio_runtime_device_defaults(bridge, metadata);

    const auto raw_handler = resolve_audio_transcriptions_raw_handler(bridge->routes.get());
    if (raw_handler) {
        auto finalize = [&](server_http_res_ptr res) -> int32_t {
            if (res == nullptr) {
                const std::string msg = "route returned null response";
                set_bridge_error(bridge, msg);
                out->error_json = copy_to_c_string(msg);
                return -1;
            }

            std::string payload = res->data;
            if (res->is_stream()) {
                while (true) {
                    std::string chunk;
                    const bool has_more = res->next(chunk);
                    payload += chunk;
                    if (!has_more) {
                        break;
                    }
                }
            }

            out->status = res->status;
            if (res->status >= 200 && res->status < 300) {
                out->ok = 1;
                out->json = copy_to_c_string(payload);
                if (out->json == nullptr) {
                    const std::string msg = "failed to allocate route JSON output";
                    set_bridge_error(bridge, msg);
                    out->ok = 0;
                    out->error_json = copy_to_c_string(msg);
                    return -1;
                }
                set_bridge_error(bridge, "");
                return 0;
            }

            const std::string err = payload.empty() ? "route returned error status with empty payload" : payload;
            set_bridge_error(bridge, err);
            out->error_json = copy_to_c_string(err);
            return -1;
        };

        try {
            return finalize(raw_handler(format, audio, metadata));
        } catch (const std::exception & e) {
            const std::string msg = normalize_error(e.what());
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        } catch (...) {
            const std::string msg = "unknown exception while running raw audio route";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }
    }

    const auto handler = resolve_audio_transcriptions_handler(bridge->routes.get());
    if (!handler) {
        const std::string msg =
            "audio transcriptions route is unavailable in this llama.cpp build (missing server audio patch)";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    const std::string body_json = build_audio_body_with_base64(metadata, audio, format);
    return run_json_route(
        bridge,
        handler,
        "/v1/audio/transcriptions",
        body_json,
        out);
}

int32_t llama_server_bridge_list_devices(
    struct llama_server_bridge_device_info ** out_devices,
    size_t * out_count) {

    if (out_devices == nullptr || out_count == nullptr) {
        return -1;
    }

    *out_devices = nullptr;
    *out_count = 0;

    common_params params = common_params();
    acquire_backend(params);

    const size_t n = ggml_backend_dev_count();
    if (n == 0) {
        release_backend();
        return 0;
    }

    auto * devices = static_cast<llama_server_bridge_device_info *>(
        std::calloc(n, sizeof(llama_server_bridge_device_info)));
    if (devices == nullptr) {
        release_backend();
        return -1;
    }

    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }

        size_t mem_free = 0;
        size_t mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);

        auto reg = ggml_backend_dev_backend_reg(dev);
        devices[i].index = static_cast<int32_t>(i);
        devices[i].type = static_cast<int32_t>(ggml_backend_dev_type(dev));
        devices[i].memory_free = static_cast<uint64_t>(mem_free);
        devices[i].memory_total = static_cast<uint64_t>(mem_total);
        devices[i].backend = copy_to_c_string(reg != nullptr ? ggml_backend_reg_name(reg) : "");
        devices[i].name = copy_to_c_string(ggml_backend_dev_name(dev));
        devices[i].description = copy_to_c_string(ggml_backend_dev_description(dev));

        if (devices[i].backend == nullptr || devices[i].name == nullptr || devices[i].description == nullptr) {
            ok = false;
            break;
        }
    }

    if (!ok) {
        llama_server_bridge_free_devices(devices, n);
        release_backend();
        return -1;
    }

    *out_devices = devices;
    *out_count = n;

    release_backend();
    return 0;
}

void llama_server_bridge_free_devices(
    struct llama_server_bridge_device_info * devices,
    size_t count) {

    if (devices == nullptr) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        std::free(devices[i].backend);
        std::free(devices[i].name);
        std::free(devices[i].description);
        devices[i].backend = nullptr;
        devices[i].name = nullptr;
        devices[i].description = nullptr;
    }
    std::free(devices);
}

static std::string extract_markdown_from_oai_chat(const json & response) {
    if (!response.is_object()) {
        return "";
    }
    if (response.contains("choices") && response["choices"].is_array() && !response["choices"].empty()) {
        const json & c0 = response["choices"][0];
        if (c0.contains("message") && c0["message"].is_object()) {
            const json & msg = c0["message"];
            if (msg.contains("content") && msg["content"].is_string()) {
                return msg["content"].get<std::string>();
            }
        }
        if (c0.contains("text") && c0["text"].is_string()) {
            return c0["text"].get<std::string>();
        }
    }
    if (response.contains("content") && response["content"].is_string()) {
        return response["content"].get<std::string>();
    }
    return "";
}

int32_t llama_server_bridge_chat_complete(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_chat_request * req,
    struct llama_server_bridge_vlm_result * out) {

    if (bridge == nullptr || req == nullptr || out == nullptr) {
        return -1;
    }

    *out = llama_server_bridge_empty_vlm_result();

    if (bridge->routes == nullptr) {
        const std::string msg = "server routes are not initialized";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (req->prompt == nullptr || req->prompt[0] == '\0') {
        const std::string msg = "prompt is empty";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    try {
        const auto meta = bridge->ctx.get_meta();

        json body = {
            {"model", meta.model_name},
            {"stream", false},
            {"temperature", req->temperature >= 0.0f ? req->temperature : 0.0f},
            {"top_p", req->top_p >= 0.0f ? req->top_p : 1.0f},
            {"max_tokens", req->n_predict > 0 ? req->n_predict : 4096},
            {"messages", json::array({
                json{
                    {"role", "user"},
                    {"content", req->prompt}
                }
            })}
        };

        if (req->id_slot >= 0) {
            body["id_slot"] = req->id_slot;
        }
        if (req->seed >= 0) {
            body["seed"] = req->seed;
        }
        if (req->top_k >= 0) {
            body["top_k"] = req->top_k;
        }
        if (req->min_p >= 0.0f) {
            body["min_p"] = req->min_p;
        }
        if (req->repeat_last_n >= 0) {
            body["repeat_last_n"] = req->repeat_last_n;
        }
        if (req->repeat_penalty > 0.0f) {
            body["repeat_penalty"] = req->repeat_penalty;
        }
        if (req->presence_penalty >= 0.0f) {
            body["presence_penalty"] = req->presence_penalty;
        }
        if (req->frequency_penalty >= 0.0f) {
            body["frequency_penalty"] = req->frequency_penalty;
        }
        if (req->dry_multiplier > 0.0f) {
            body["dry_multiplier"] = req->dry_multiplier;
        }
        if (req->dry_allowed_length >= 0) {
            body["dry_allowed_length"] = req->dry_allowed_length;
        }
        if (req->dry_penalty_last_n >= 0) {
            body["dry_penalty_last_n"] = req->dry_penalty_last_n;
        }

        std::vector<raw_buffer> ignored_files;
        json llama_params = oaicompat_chat_params_parse(body, meta.chat_params, ignored_files);

        llama_context * lctx = bridge->ctx.get_llama_context();
        if (lctx == nullptr) {
            const std::string msg = "llama context is not available";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }
        const llama_model * model = llama_get_model(lctx);
        const llama_vocab * vocab = model != nullptr ? llama_model_get_vocab(model) : nullptr;
        if (vocab == nullptr) {
            const std::string msg = "failed to access llama vocab";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        const std::string cli_prompt = json_value(llama_params, "prompt", std::string());
        if (cli_prompt.empty()) {
            const std::string msg = "chat template produced empty prompt";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        server_response_reader rd = bridge->ctx.get_response_reader();

        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = rd.get_new_id();
        task.index = 0;
        task.params = server_task::params_from_json_cmpl(
            vocab,
            bridge->params,
            meta.slot_n_ctx,
            llama_params);
        task.id_slot = json_value(llama_params, "id_slot", -1);

        task.params.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
        task.params.oaicompat_cmpl_id = gen_chatcmplid();
        task.params.oaicompat_model = meta.model_name;

        task.cli = true;
        task.cli_prompt = cli_prompt;

        rd.post_task(std::move(task));

        const std::function<bool()> should_stop = []() -> bool { return false; };
        server_task_result_ptr result = rd.next(should_stop);
        server_task_result_cmpl_final * final_result = nullptr;
        while (result != nullptr) {
            if (result->is_error()) {
                const std::string err = safe_json_to_str(result->to_json());
                set_bridge_error(bridge, normalize_error(err));
                out->error_json = copy_to_c_string(err);
                return -1;
            }
            if (auto * r_final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                final_result = r_final;
                break;
            }
            result = rd.next(should_stop);
        }

        if (final_result == nullptr) {
            const std::string msg = "no final completion result received";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        std::string text = final_result->oaicompat_msg.content;
        if (text.empty()) {
            text = final_result->content;
        }
        if (text.empty()) {
            const json response_json = final_result->to_json();
            text = extract_markdown_from_oai_chat(response_json);
        }
        if (text.empty()) {
            const std::string err = "chat response missing content";
            set_bridge_error(bridge, err);
            out->error_json = copy_to_c_string(err);
            return -1;
        }

        out->ok = 1;
        out->truncated = final_result->truncated ? 1 : 0;
        out->stop = static_cast<int32_t>(final_result->stop);
        out->n_decoded = final_result->n_decoded;
        out->n_prompt_tokens = final_result->n_prompt_tokens;
        out->n_tokens_cached = final_result->n_tokens_cached;
        out->eos_reached = final_result->stop == STOP_TYPE_EOS ? 1 : 0;
        out->prompt_ms = final_result->timings.prompt_ms;
        out->predicted_ms = final_result->timings.predicted_ms;

        out->text = copy_to_c_string(text);
        if (out->text == nullptr) {
            const std::string msg = "failed to allocate output text";
            set_bridge_error(bridge, msg);
            out->ok = 0;
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        set_bridge_error(bridge, "");
        return 0;
    } catch (const std::exception & e) {
        const std::string msg = normalize_error(e.what());
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    } catch (...) {
        const std::string msg = "unknown exception in llama_server_bridge_chat_complete()";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
}

int32_t llama_server_bridge_vlm_complete(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_vlm_request * req,
    struct llama_server_bridge_vlm_result * out) {

    if (bridge == nullptr || req == nullptr || out == nullptr) {
        return -1;
    }

    *out = llama_server_bridge_empty_vlm_result();

    if (bridge->routes == nullptr) {
        const std::string msg = "server routes are not initialized";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (req->prompt == nullptr || req->prompt[0] == '\0') {
        const std::string msg = "prompt is empty";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
    if (req->image_bytes == nullptr || req->image_bytes_len == 0) {
        const std::string msg = "image bytes are empty";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }

    try {
        const auto meta = bridge->ctx.get_meta();

        json body = {
            {"model", meta.model_name},
            {"stream", false},
            {"temperature", req->temperature >= 0.0f ? req->temperature : 0.0f},
            {"top_p", req->top_p >= 0.0f ? req->top_p : 1.0f},
            {"max_tokens", req->n_predict > 0 ? req->n_predict : 4096},
            {"messages", json::array({
                json{
                    {"role", "user"},
                    {"content", json::array({
                        json{
                            {"type", "text"},
                            {"text", req->prompt}
                        },
                        json{
                            {"type", "text"},
                            {"text", mtmd_default_marker()}
                        }
                    })}
                }
            })}
        };

        if (req->id_slot >= 0) {
            body["id_slot"] = req->id_slot;
        }
        if (req->seed >= 0) {
            body["seed"] = req->seed;
        }
        if (req->top_k >= 0) {
            body["top_k"] = req->top_k;
        }
        if (req->min_p >= 0.0f) {
            body["min_p"] = req->min_p;
        }
        if (req->repeat_last_n >= 0) {
            body["repeat_last_n"] = req->repeat_last_n;
        }
        if (req->repeat_penalty > 0.0f) {
            body["repeat_penalty"] = req->repeat_penalty;
        }
        if (req->presence_penalty >= 0.0f) {
            body["presence_penalty"] = req->presence_penalty;
        }
        if (req->frequency_penalty >= 0.0f) {
            body["frequency_penalty"] = req->frequency_penalty;
        }
        if (req->dry_multiplier > 0.0f) {
            body["dry_multiplier"] = req->dry_multiplier;
        }
        if (req->dry_allowed_length >= 0) {
            body["dry_allowed_length"] = req->dry_allowed_length;
        }
        if (req->dry_penalty_last_n >= 0) {
            body["dry_penalty_last_n"] = req->dry_penalty_last_n;
        }

        std::vector<raw_buffer> ignored_files;
        json llama_params = oaicompat_chat_params_parse(body, meta.chat_params, ignored_files);

        llama_context * lctx = bridge->ctx.get_llama_context();
        if (lctx == nullptr) {
            const std::string msg = "llama context is not available";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }
        const llama_model * model = llama_get_model(lctx);
        const llama_vocab * vocab = model != nullptr ? llama_model_get_vocab(model) : nullptr;
        if (vocab == nullptr) {
            const std::string msg = "failed to access llama vocab";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        const std::string cli_prompt = json_value(llama_params, "prompt", std::string());
        if (cli_prompt.empty()) {
            const std::string msg = "chat template produced empty prompt";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        raw_buffer image_file(req->image_bytes, req->image_bytes + req->image_bytes_len);

        server_response_reader rd = bridge->ctx.get_response_reader();

        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = rd.get_new_id();
        task.index = 0;
        task.params = server_task::params_from_json_cmpl(
            vocab,
            bridge->params,
            meta.slot_n_ctx,
            llama_params);
        task.id_slot = json_value(llama_params, "id_slot", -1);

        task.params.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
        task.params.oaicompat_cmpl_id = gen_chatcmplid();
        task.params.oaicompat_model = meta.model_name;

        task.cli = true;
        task.cli_prompt = cli_prompt;
        task.cli_files.push_back(std::move(image_file));

        rd.post_task(std::move(task));

        const std::function<bool()> should_stop = []() -> bool { return false; };
        server_task_result_ptr result = rd.next(should_stop);
        server_task_result_cmpl_final * final_result = nullptr;
        while (result != nullptr) {
            if (result->is_error()) {
                const std::string err = safe_json_to_str(result->to_json());
                set_bridge_error(bridge, normalize_error(err));
                out->error_json = copy_to_c_string(err);
                return -1;
            }
            if (auto * r_final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                final_result = r_final;
                break;
            }
            result = rd.next(should_stop);
        }

        if (final_result == nullptr) {
            const std::string msg = "no final completion result received";
            set_bridge_error(bridge, msg);
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        std::string markdown = final_result->oaicompat_msg.content;
        if (markdown.empty()) {
            markdown = final_result->content;
        }
        if (markdown.empty()) {
            const json response_json = final_result->to_json();
            markdown = extract_markdown_from_oai_chat(response_json);
        }
        if (markdown.empty()) {
            const std::string err = "chat response missing markdown content";
            set_bridge_error(bridge, err);
            out->error_json = copy_to_c_string(err);
            return -1;
        }

        out->ok = 1;
        out->truncated = final_result->truncated ? 1 : 0;
        out->stop = static_cast<int32_t>(final_result->stop);
        out->n_decoded = final_result->n_decoded;
        out->n_prompt_tokens = final_result->n_prompt_tokens;
        out->n_tokens_cached = final_result->n_tokens_cached;
        out->eos_reached = final_result->stop == STOP_TYPE_EOS ? 1 : 0;
        out->prompt_ms = final_result->timings.prompt_ms;
        out->predicted_ms = final_result->timings.predicted_ms;

        out->text = copy_to_c_string(markdown);
        if (out->text == nullptr) {
            const std::string msg = "failed to allocate output text";
            set_bridge_error(bridge, msg);
            out->ok = 0;
            out->error_json = copy_to_c_string(msg);
            return -1;
        }

        set_bridge_error(bridge, "");
        return 0;
    } catch (const std::exception & e) {
        const std::string msg = normalize_error(e.what());
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    } catch (...) {
        const std::string msg = "unknown exception in llama_server_bridge_vlm_complete()";
        set_bridge_error(bridge, msg);
        out->error_json = copy_to_c_string(msg);
        return -1;
    }
}
