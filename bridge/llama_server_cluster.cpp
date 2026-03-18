#include "llama_server_cluster.h"

#include "llama_server_bridge.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace {

using json = nlohmann::ordered_json;
using steady_clock = std::chrono::steady_clock;
using system_clock = std::chrono::system_clock;

constexpr int32_t kDefaultLoadOnDemandGraceSeconds = 30;

struct bridge_handle_deleter {
    void operator()(llama_server_bridge * bridge) const {
        if (bridge != nullptr) {
            llama_server_bridge_destroy(bridge);
        }
    }
};

using bridge_handle_ptr = std::unique_ptr<llama_server_bridge, bridge_handle_deleter>;

struct owned_device_info {
    int32_t bridge_device_index = -1;
    int32_t type = 0;
    uint64_t memory_free = 0;
    uint64_t memory_total = 0;
    std::string backend;
    std::string name;
    std::string description;
};

struct owned_execution_group {
    std::string id;
    std::string label;
    std::string backend_summary;
    std::string devices_csv;
    int32_t device_count = 0;
    bool uses_local_split = false;
    uint64_t memory_free = 0;
    uint64_t memory_total = 0;
};

struct owned_instance_params {
    std::string name;
    std::string model_path;
    std::string mmproj_path;
    std::string diarization_model_path;
    std::string execution_group_id;
    std::string rpc_servers;
    std::string manual_devices_csv;
    std::string manual_tensor_split;
    int32_t embedding = 0;
    int32_t reranking = 0;
    int32_t model_kind = LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_TEXT;
    int32_t allow_cpu = 0;
    int32_t allow_integrated_gpu = 0;
    int32_t n_ctx = 0;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t n_parallel = 0;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    int32_t n_gpu_layers = 0;
    int32_t load_on_demand_grace_seconds = kDefaultLoadOnDemandGraceSeconds;
};

struct model_instance {
    explicit model_instance(int64_t id_value, owned_instance_params params_value, int32_t retention_value)
        : instance_id(id_value), params(std::move(params_value)), retention_mode(retention_value) {}

    std::mutex mutex;
    std::condition_variable cv;
    int64_t instance_id = 0;
    owned_instance_params params;
    int32_t retention_mode = LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED;
    int32_t state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
    int32_t active_request_count = 0;
    int32_t queued_request_count = 0;
    int64_t grace_deadline_unix_ms = 0;
    steady_clock::time_point grace_deadline_steady = steady_clock::time_point::min();
    std::string last_error;
    bridge_handle_ptr bridge;
};

char * dup_cstr(const char * value) {
    if (value == nullptr) {
        return nullptr;
    }
    const size_t len = std::strlen(value);
    char * copy = static_cast<char *>(std::malloc(len + 1));
    if (copy == nullptr) {
        return nullptr;
    }
    std::memcpy(copy, value, len + 1);
    return copy;
}

char * dup_cstr(const std::string & value) {
    return dup_cstr(value.c_str());
}

std::string safe_cstr(const char * value) {
    return value != nullptr ? value : "";
}

std::string trim_copy(const std::string & value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(start, end - start);
}

struct native_transcript_item {
    uint64_t start_sample = 0;
    uint64_t end_sample = 0;
    std::string text;
};

bool same_transcript_item(
    const native_transcript_item & lhs,
    const native_transcript_item & rhs) {
    return lhs.start_sample == rhs.start_sample
        && lhs.end_sample == rhs.end_sample
        && lhs.text == rhs.text;
}

void append_unique_transcript_item(
    std::vector<native_transcript_item> & items,
    native_transcript_item item) {
    const auto duplicate = std::find_if(
        items.begin(),
        items.end(),
        [&](const native_transcript_item & existing) {
            return same_transcript_item(existing, item);
        });
    if (duplicate == items.end()) {
        items.push_back(std::move(item));
    }
}

std::string join_transcript_items_text(const std::vector<native_transcript_item> & items) {
    std::ostringstream out;
    bool first = true;
    for (const native_transcript_item & item : items) {
        const std::string text = trim_copy(item.text);
        if (text.empty()) {
            continue;
        }
        if (!first) {
            out << ' ';
        }
        first = false;
        out << text;
    }
    return out.str();
}

void append_json_escaped(std::ostringstream & out, const std::string & value) {
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u"
                        << std::hex
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<int>(ch)
                        << std::dec
                        << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
}

void append_transcript_items_json_array(
    std::ostringstream & out,
    const std::vector<native_transcript_item> & items) {
    out << '[';
    bool first = true;
    for (const native_transcript_item & item : items) {
        const std::string text = trim_copy(item.text);
        if (text.empty()) {
            continue;
        }
        if (!first) {
            out << ',';
        }
        first = false;
        const double start_sec = static_cast<double>(item.start_sample) / 16000.0;
        const double end_sec = static_cast<double>(item.end_sample) / 16000.0;
        out << "{\"start_sec\":" << start_sec
            << ",\"end_sec\":" << end_sec
            << ",\"text\":\"";
        append_json_escaped(out, text);
        out << "\"}";
    }
    out << ']';
}

std::string build_native_transcription_result_json(
    const std::vector<native_transcript_item> & pieces,
    const std::vector<native_transcript_item> & words,
    const std::string & diarization_markdown) {
    std::ostringstream out;
    out << "{\"text\":\"";
    append_json_escaped(out, join_transcript_items_text(pieces));
    out << "\",\"timeline\":{\"whisper_pieces\":";
    append_transcript_items_json_array(out, pieces);
    out << ",\"words\":";
    append_transcript_items_json_array(out, words);
    out << "},\"segments\":";
    append_transcript_items_json_array(out, pieces);
    if (!trim_copy(diarization_markdown).empty()) {
        out << ",\"diarization\":{\"markdown\":\"";
        append_json_escaped(out, diarization_markdown);
        out << "\"}";
    }
    out << '}';
    return out.str();
}

uint16_t read_le_u16(const uint8_t * ptr) {
    return static_cast<uint16_t>(ptr[0]) |
        (static_cast<uint16_t>(ptr[1]) << 8);
}

uint32_t read_le_u32(const uint8_t * ptr) {
    return static_cast<uint32_t>(ptr[0]) |
        (static_cast<uint32_t>(ptr[1]) << 8) |
        (static_cast<uint32_t>(ptr[2]) << 16) |
        (static_cast<uint32_t>(ptr[3]) << 24);
}

struct decoded_wav_pcm16 {
    std::vector<int16_t> samples;
    uint32_t sample_rate_hz = 0;
    uint16_t channels = 0;
};

bool decode_wav_pcm16(
    const uint8_t * bytes,
    size_t len,
    decoded_wav_pcm16 & out,
    std::string * error_out) {

    if (bytes == nullptr || len < 44) {
        if (error_out != nullptr) {
            *error_out = "WAV payload is too small";
        }
        return false;
    }
    if (std::memcmp(bytes, "RIFF", 4) != 0 || std::memcmp(bytes + 8, "WAVE", 4) != 0) {
        if (error_out != nullptr) {
            *error_out = "audio payload is not a RIFF/WAVE file";
        }
        return false;
    }

    bool found_fmt = false;
    bool found_data = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate_hz = 0;
    const uint8_t * data_ptr = nullptr;
    size_t data_len = 0;

    size_t offset = 12;
    while (offset + 8 <= len) {
        const uint8_t * chunk = bytes + offset;
        const uint32_t chunk_size = read_le_u32(chunk + 4);
        const size_t data_offset = offset + 8;
        const size_t next_offset = data_offset + static_cast<size_t>(chunk_size) + (chunk_size & 1u);
        if (next_offset > len) {
            if (error_out != nullptr) {
                *error_out = "WAV chunk exceeds payload size";
            }
            return false;
        }

        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                if (error_out != nullptr) {
                    *error_out = "WAV fmt chunk is too small";
                }
                return false;
            }
            audio_format = read_le_u16(bytes + data_offset);
            channels = read_le_u16(bytes + data_offset + 2);
            sample_rate_hz = read_le_u32(bytes + data_offset + 4);
            bits_per_sample = read_le_u16(bytes + data_offset + 14);
            found_fmt = true;
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data_ptr = bytes + data_offset;
            data_len = static_cast<size_t>(chunk_size);
            found_data = true;
        }

        offset = next_offset;
    }

    if (!found_fmt || !found_data) {
        if (error_out != nullptr) {
            *error_out = "WAV payload is missing fmt or data chunk";
        }
        return false;
    }
    if (audio_format != 1 || bits_per_sample != 16) {
        if (error_out != nullptr) {
            *error_out = "only PCM16 WAV input is supported without FFmpeg";
        }
        return false;
    }
    if (channels != 1) {
        if (error_out != nullptr) {
            *error_out = "only mono WAV input is supported without FFmpeg";
        }
        return false;
    }
    if ((data_len % sizeof(int16_t)) != 0) {
        if (error_out != nullptr) {
            *error_out = "WAV PCM16 data size is invalid";
        }
        return false;
    }

    out.sample_rate_hz = sample_rate_hz;
    out.channels = channels;
    const size_t sample_count = data_len / sizeof(int16_t);
    out.samples.resize(sample_count);
    std::memcpy(out.samples.data(), data_ptr, data_len);
    return true;
}

void free_cstr(char * value) {
    if (value != nullptr) {
        std::free(value);
    }
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    const std::string lowered = lower_copy(value);
    return lowered == "1"
        || lowered == "true"
        || lowered == "yes"
        || lowered == "on";
}

int32_t env_i32_or_default(const char * name, int32_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0')) {
        return fallback;
    }

    return static_cast<int32_t>(parsed);
}

int32_t cluster_default_thread_count() {
    const unsigned int detected = std::thread::hardware_concurrency();
    return static_cast<int32_t>(std::max(1u, detected));
}

std::vector<std::string> split_csv(const std::string & text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(',', start);
        const size_t count = end == std::string::npos ? text.size() - start : end - start;
        std::string item = text.substr(start, count);
        const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), not_space));
        item.erase(std::find_if(item.rbegin(), item.rend(), not_space).base(), item.end());
        if (!item.empty()) {
            out.push_back(std::move(item));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

double duration_ms(const steady_clock::duration & duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double tokens_per_second(int32_t tokens, double elapsed_ms) {
    if (tokens <= 0 || elapsed_ms <= 0.0) {
        return 0.0;
    }
    return (static_cast<double>(tokens) * 1000.0) / elapsed_ms;
}

uint64_t file_size_or_zero(const std::string & path) {
    if (path.empty()) {
        return 0;
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

llama_server_cluster_inference_metrics make_base_metrics(
    const owned_instance_params & params,
    uint64_t request_bytes) {
    llama_server_cluster_inference_metrics metrics{};
    metrics.loaded_this_call = 0;
    metrics.used_rpc = params.rpc_servers.empty() ? 0 : 1;
    metrics.rpc_server_count = static_cast<int32_t>(split_csv(params.rpc_servers).size());
    metrics.request_bytes = request_bytes;
    metrics.model_bytes = file_size_or_zero(params.model_path);
    metrics.mmproj_bytes = file_size_or_zero(params.mmproj_path);
    return metrics;
}

llama_server_cluster_inference_metrics make_base_metrics_for_model(
    const std::string & model_path,
    const std::string & mmproj_path,
    uint64_t request_bytes) {
    llama_server_cluster_inference_metrics metrics{};
    metrics.request_bytes = request_bytes;
    metrics.model_bytes = file_size_or_zero(model_path);
    metrics.mmproj_bytes = file_size_or_zero(mmproj_path);
    return metrics;
}

void finalize_text_metrics(
    llama_server_cluster_inference_metrics & metrics,
    int32_t prompt_tokens,
    int32_t decoded_tokens,
    double prompt_ms,
    double predicted_ms,
    double request_total_ms) {
    metrics.prompt_tokens = prompt_tokens;
    metrics.decoded_tokens = decoded_tokens;
    metrics.prompt_ms = prompt_ms;
    metrics.predicted_ms = predicted_ms;
    metrics.request_total_ms = request_total_ms;
    metrics.prompt_tokens_per_second = tokens_per_second(prompt_tokens, prompt_ms);
    metrics.decode_tokens_per_second = tokens_per_second(decoded_tokens, predicted_ms);
    metrics.total_tokens_per_second =
        tokens_per_second(prompt_tokens + decoded_tokens, request_total_ms);
}

bool is_rpc_backend(const owned_device_info & device);

bool is_cpu_backend(const owned_device_info & device) {
    const std::string backend = lower_copy(device.backend);
    const std::string name = lower_copy(device.name);
    return backend.find("cpu") != std::string::npos
        || backend.find("blas") != std::string::npos
        || name.find("cpu") != std::string::npos
        || name.find("accelerate") != std::string::npos
        || name.find("blas") != std::string::npos;
}

bool is_integrated_gpu(const owned_device_info & device) {
    if (is_cpu_backend(device) || is_rpc_backend(device)) {
        return false;
    }
    const std::string backend = lower_copy(device.backend);
    if (backend.find("metal") != std::string::npos) {
        return false;
    }
    const std::string text = lower_copy(device.name + " " + device.description);
    const bool looks_intel_integrated =
        text.find("intel") != std::string::npos && text.find("arc") == std::string::npos;
    const bool looks_integrated_family =
        text.find("integrated") != std::string::npos
        || text.find("uhd") != std::string::npos
        || text.find("iris") != std::string::npos
        || text.find("hd graphics") != std::string::npos
        || text.find("xe graphics") != std::string::npos
        || text.find("graphics controller") != std::string::npos
        || text.find("apu") != std::string::npos
        || text.find("uma") != std::string::npos;
    const bool looks_shared_memory =
        text.find("shared") != std::string::npos
        || text.find("unified") != std::string::npos
        || text.find("system memory") != std::string::npos;
    return looks_intel_integrated || looks_integrated_family || looks_shared_memory;
}

bool is_split_accelerator(const owned_device_info & device) {
    if (is_cpu_backend(device)) {
        return false;
    }
    const std::string backend = lower_copy(device.backend);
    const std::string name = lower_copy(device.name);
    if (env_flag_enabled("ENGINE_CLUSTER_DISABLE_RPC_SPLIT")
        && (backend.find("rpc") != std::string::npos || name.find("rpc") != std::string::npos)) {
        return false;
    }
    return device.memory_total > 0 && device.memory_free > 0;
}

bool is_rpc_backend(const owned_device_info & device) {
    const std::string backend = lower_copy(device.backend);
    const std::string name = lower_copy(device.name);
    return backend.find("rpc") != std::string::npos || name.find("rpc") != std::string::npos;
}

int backend_preference_rank(const owned_device_info & device) {
    const std::string backend = lower_copy(device.backend);
    if (backend.find("cuda") != std::string::npos) {
        return 0;
    }
    if (backend.find("metal") != std::string::npos) {
        return 1;
    }
    if (backend.find("sycl") != std::string::npos) {
        return 2;
    }
    if (backend.find("vulkan") != std::string::npos) {
        return 3;
    }
    if (backend.find("kompute") != std::string::npos) {
        return 4;
    }
    if (backend.find("opencl") != std::string::npos) {
        return 5;
    }
    return 10;
}

bool same_physical_accelerator(
    const owned_device_info & lhs,
    const owned_device_info & rhs) {
    if (is_cpu_backend(lhs) || is_cpu_backend(rhs) || is_rpc_backend(lhs) || is_rpc_backend(rhs)) {
        return false;
    }

    const std::string lhs_description = lower_copy(lhs.description);
    const std::string rhs_description = lower_copy(rhs.description);
    if (!lhs_description.empty() && lhs_description == rhs_description) {
        return lhs.memory_total == rhs.memory_total;
    }

    const std::string lhs_name = lower_copy(lhs.name);
    const std::string rhs_name = lower_copy(rhs.name);
    if (!lhs_name.empty() && lhs_name == rhs_name) {
        return lhs.memory_total == rhs.memory_total;
    }

    return false;
}

bool should_prefer_device_choice(
    const owned_device_info & candidate,
    const owned_device_info & existing) {
    const int candidate_rank = backend_preference_rank(candidate);
    const int existing_rank = backend_preference_rank(existing);
    if (candidate_rank != existing_rank) {
        return candidate_rank < existing_rank;
    }
    if (candidate.memory_free != existing.memory_free) {
        return candidate.memory_free > existing.memory_free;
    }
    return candidate.bridge_device_index < existing.bridge_device_index;
}

const owned_device_info * find_device_by_index(
    const std::vector<owned_device_info> & devices,
    int32_t bridge_device_index) {
    for (const owned_device_info & device : devices) {
        if (device.bridge_device_index == bridge_device_index) {
            return &device;
        }
    }
    return nullptr;
}

std::vector<const owned_device_info *> devices_from_csv(
    const std::vector<owned_device_info> & devices,
    const std::string & devices_csv) {
    std::vector<const owned_device_info *> selected;
    for (const std::string & token : split_csv(devices_csv)) {
        char * end = nullptr;
        const long value = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || (end != nullptr && *end != '\0')) {
            continue;
        }
        if (const owned_device_info * device = find_device_by_index(devices, static_cast<int32_t>(value))) {
            selected.push_back(device);
        }
    }
    return selected;
}

double model_file_size_gib(const std::string & model_path) {
    if (model_path.empty()) {
        return 0.0;
    }
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(model_path, ec);
    if (ec) {
        return 0.0;
    }
    return static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
}

std::string build_tensor_split_csv_for_group(
    const std::vector<const owned_device_info *> & selected_devices,
    const std::string & model_path) {
    if (selected_devices.size() < 2) {
        return {};
    }

    double rpc_weight_sum = 0.0;
    double local_weight_sum = 0.0;
    double rpc_memory_total_gib = 0.0;
    size_t rpc_count = 0;
    size_t local_count = 0;

    std::vector<double> base_weights;
    base_weights.reserve(selected_devices.size());

    for (const owned_device_info * device : selected_devices) {
        const double weight = static_cast<double>(device->memory_total > 0 ? device->memory_total : device->memory_free);
        base_weights.push_back(weight > 0.0 ? weight : 1.0);
        if (is_rpc_backend(*device)) {
            ++rpc_count;
            rpc_weight_sum += base_weights.back();
            rpc_memory_total_gib += static_cast<double>(device->memory_total) / (1024.0 * 1024.0 * 1024.0);
        } else if (!is_cpu_backend(*device)) {
            ++local_count;
            local_weight_sum += base_weights.back();
        }
    }

    if (rpc_count == 0 || local_count == 0 || rpc_weight_sum <= 0.0 || local_weight_sum <= 0.0) {
        return {};
    }

    const double total_weight = rpc_weight_sum + local_weight_sum;
    if (total_weight <= 0.0) {
        return {};
    }

    const double model_size_gib = model_file_size_gib(model_path);
    const double default_rpc_fraction = rpc_weight_sum / total_weight;

    double min_rpc_fraction = 0.0;
    if (model_size_gib > 0.0) {
        if (model_size_gib <= 8.0) {
            min_rpc_fraction = 0.25;
        } else if (model_size_gib <= 16.0) {
            min_rpc_fraction = 0.20;
        } else {
            min_rpc_fraction = 0.15;
        }
    }

    constexpr double kRpcHeadroomGib = 1.25;
    double max_rpc_fraction = 0.50;
    if (model_size_gib > 0.0 && rpc_memory_total_gib > kRpcHeadroomGib) {
        max_rpc_fraction = std::clamp((rpc_memory_total_gib - kRpcHeadroomGib) / model_size_gib, 0.0, 0.50);
    }

    double target_rpc_fraction = default_rpc_fraction;
    if (target_rpc_fraction < min_rpc_fraction) {
        target_rpc_fraction = std::min(min_rpc_fraction, max_rpc_fraction);
    } else if (target_rpc_fraction > max_rpc_fraction) {
        target_rpc_fraction = max_rpc_fraction;
    }

    if (!(target_rpc_fraction > 0.0 && target_rpc_fraction < 1.0)) {
        return {};
    }

    std::ostringstream csv;
    csv.setf(std::ios::fixed);
    csv.precision(6);

    bool first = true;
    for (size_t i = 0; i < selected_devices.size(); ++i) {
        const owned_device_info * device = selected_devices[i];
        double normalized_weight = 0.0;
        if (is_rpc_backend(*device)) {
            normalized_weight = target_rpc_fraction * (base_weights[i] / rpc_weight_sum);
        } else {
            normalized_weight = (1.0 - target_rpc_fraction) * (base_weights[i] / local_weight_sum);
        }
        normalized_weight = std::max(normalized_weight, 0.0001);
        if (!first) {
            csv << ",";
        }
        first = false;
        csv << normalized_weight;
    }

    return csv.str();
}

std::mutex & rpc_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_set<std::string> & registered_rpc_servers() {
    static std::unordered_set<std::string> servers;
    return servers;
}

bool ensure_bridge_backend_registry_ready(std::string * error_out) {
    llama_server_bridge_device_info * raw_devices = nullptr;
    size_t raw_count = 0;
    const int32_t rc = llama_server_bridge_list_devices(&raw_devices, &raw_count);
    if (raw_devices != nullptr) {
        llama_server_bridge_free_devices(raw_devices, raw_count);
    }
    if (rc != 0) {
        if (error_out != nullptr) {
            *error_out = "failed to initialize bridge backend registry";
        }
        return false;
    }
    return true;
}

bool register_rpc_servers(const std::string & rpc_servers, std::string * error_out) {
    const std::vector<std::string> endpoints = split_csv(rpc_servers);
    if (endpoints.empty()) {
        return true;
    }

    if (!ensure_bridge_backend_registry_ready(error_out)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(rpc_registry_mutex());
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == nullptr) {
        if (error_out != nullptr) {
            *error_out = "RPC backend is not available in this runtime";
        }
        return false;
    }

    using ggml_backend_rpc_add_server_t = ggml_backend_reg_t (*)(const char * endpoint);
    auto * add_server = reinterpret_cast<ggml_backend_rpc_add_server_t>(
        ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server"));
    if (add_server == nullptr) {
        if (error_out != nullptr) {
            *error_out = "RPC backend does not expose ggml_backend_rpc_add_server";
        }
        return false;
    }

    for (const std::string & endpoint : endpoints) {
        if (registered_rpc_servers().count(endpoint) != 0) {
            continue;
        }

        ggml_backend_reg_t reg = add_server(endpoint.c_str());
        if (reg == nullptr) {
            if (error_out != nullptr) {
                *error_out = "failed to register RPC server: " + endpoint;
            }
            return false;
        }
        ggml_backend_register(reg);
        registered_rpc_servers().insert(endpoint);
    }

    return true;
}

std::string host_name_string() {
#if defined(_WIN32)
    char buffer[MAX_COMPUTERNAME_LENGTH + 1] = {};
    DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
    if (GetComputerNameA(buffer, &size) != 0 && buffer[0] != '\0') {
        return buffer;
    }
    const char * value = std::getenv("COMPUTERNAME");
#else
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer)) == 0) {
        buffer[sizeof(buffer) - 1] = '\0';
        if (buffer[0] != '\0') {
            return buffer;
        }
    }
    const char * value = std::getenv("HOSTNAME");
#endif
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return "local-node";
}

std::string os_name_string() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

std::string arch_name_string() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#else
    return "unknown";
#endif
}

int64_t unix_ms_from_steady_deadline(const steady_clock::time_point & deadline) {
    if (deadline == steady_clock::time_point::min()) {
        return 0;
    }
    const auto now_steady = steady_clock::now();
    const auto now_system = system_clock::now();
    if (deadline <= now_steady) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(now_system.time_since_epoch()).count();
    }
    const auto delta = deadline - now_steady;
    return std::chrono::duration_cast<std::chrono::milliseconds>((now_system + delta).time_since_epoch()).count();
}

std::vector<owned_device_info> query_devices(const std::string & rpc_servers, std::string * error_out) {
    if (!register_rpc_servers(rpc_servers, error_out)) {
        return {};
    }

    llama_server_bridge_device_info * raw_devices = nullptr;
    size_t raw_count = 0;
    const int32_t rc = llama_server_bridge_list_devices_ex(
        rpc_servers.empty() ? 0 : 1,
        &raw_devices,
        &raw_count);
    if (rc != 0) {
        if (error_out != nullptr) {
            *error_out = "llama_server_bridge_list_devices failed";
        }
        return {};
    }

    std::vector<owned_device_info> devices;
    devices.reserve(raw_count);
    for (size_t i = 0; i < raw_count; ++i) {
        const llama_server_bridge_device_info & raw = raw_devices[i];
        owned_device_info device;
        device.bridge_device_index = raw.index;
        device.type = raw.type;
        device.memory_free = raw.memory_free;
        device.memory_total = raw.memory_total;
        if (raw.backend != nullptr) {
            device.backend = raw.backend;
        }
        if (raw.name != nullptr) {
            device.name = raw.name;
        }
        if (raw.description != nullptr) {
            device.description = raw.description;
        }
        devices.push_back(std::move(device));
    }

    if (raw_devices != nullptr) {
        llama_server_bridge_free_devices(raw_devices, raw_count);
    }

    std::vector<owned_device_info> filtered;
    filtered.reserve(devices.size());

    size_t cpu_like_index = static_cast<size_t>(-1);
    for (owned_device_info & device : devices) {
        if (is_cpu_backend(device)) {
            if (cpu_like_index == static_cast<size_t>(-1)) {
                filtered.push_back(std::move(device));
                cpu_like_index = filtered.size() - 1;
            } else {
                owned_device_info & existing = filtered[cpu_like_index];
                const uint64_t existing_score = std::max(existing.memory_total, existing.memory_free);
                const uint64_t candidate_score = std::max(device.memory_total, device.memory_free);
                if (candidate_score > existing_score) {
                    existing = std::move(device);
                }
            }
            continue;
        }

        if (device.memory_total == 0 && device.memory_free == 0) {
            continue;
        }

        auto existing_it = std::find_if(
            filtered.begin(),
            filtered.end(),
            [&](const owned_device_info & existing) {
                return same_physical_accelerator(existing, device);
            });
        if (existing_it == filtered.end()) {
            filtered.push_back(std::move(device));
        } else if (should_prefer_device_choice(device, *existing_it)) {
            *existing_it = std::move(device);
        }
    }

    return filtered;
}

std::vector<owned_execution_group> build_execution_groups(
    const std::vector<owned_device_info> & devices,
    bool include_cluster_auto) {
    std::vector<owned_execution_group> groups;
    groups.reserve(devices.size() + 2);

    std::vector<const owned_device_info *> gpu_devices;
    bool has_rpc_accelerator = false;
    for (const owned_device_info & device : devices) {
        owned_execution_group group;
        group.id = "device:" + std::to_string(device.bridge_device_index);
        group.label = device.name.empty() ? ("Device " + std::to_string(device.bridge_device_index)) : device.name;
        group.backend_summary = device.backend;
        group.devices_csv = std::to_string(device.bridge_device_index);
        group.device_count = 1;
        group.uses_local_split = false;
        group.memory_free = device.memory_free;
        group.memory_total = device.memory_total;
        groups.push_back(std::move(group));

        if (is_split_accelerator(device)) {
            gpu_devices.push_back(&device);
            has_rpc_accelerator = has_rpc_accelerator || is_rpc_backend(device);
        }
    }

    auto make_split_group = [&](const std::vector<const owned_device_info *> & split_devices,
                                const std::string & id,
                                const std::string & label) {
        owned_execution_group split_group;
        split_group.id = id;
        split_group.label = label;
        split_group.device_count = static_cast<int32_t>(split_devices.size());
        split_group.uses_local_split = true;
        for (size_t i = 0; i < split_devices.size(); ++i) {
            const owned_device_info & device = *split_devices[i];
            if (!split_group.backend_summary.empty()) {
                split_group.backend_summary += ", ";
                split_group.devices_csv += ",";
            }
            split_group.backend_summary += device.backend;
            split_group.devices_csv += std::to_string(device.bridge_device_index);
            split_group.memory_free += device.memory_free;
            split_group.memory_total += device.memory_total;
        }
        groups.push_back(std::move(split_group));
    };

    if (gpu_devices.size() > 2) {
        const std::string id_prefix = include_cluster_auto ? "cluster-split:" : "local-split:";
        const uint64_t mask_limit = 1ULL << gpu_devices.size();
        for (uint64_t mask = 1; mask < mask_limit; ++mask) {
            std::vector<const owned_device_info *> subset;
            subset.reserve(gpu_devices.size());
            for (size_t bit = 0; bit < gpu_devices.size(); ++bit) {
                if ((mask & (1ULL << bit)) != 0) {
                    subset.push_back(gpu_devices[bit]);
                }
            }
            if (subset.size() < 2 || subset.size() == gpu_devices.size()) {
                continue;
            }

            std::string id = id_prefix;
            std::string label = "Split across ";
            for (size_t i = 0; i < subset.size(); ++i) {
                if (i > 0) {
                    id += ",";
                    label += " + ";
                }
                id += std::to_string(subset[i]->bridge_device_index);
                label += subset[i]->name;
            }
            make_split_group(subset, id, label);
        }
    }

    if (gpu_devices.size() >= 2) {
        make_split_group(
            gpu_devices,
            include_cluster_auto ? "cluster-split-gpu-all" : "local-split-gpu-all",
            include_cluster_auto
                ? (has_rpc_accelerator
                    ? "Cluster split across all local and remote accelerators"
                    : "Cluster split across all accelerators")
                : "Local split across all accelerators");
    }

    if (include_cluster_auto && !gpu_devices.empty()) {
        owned_execution_group auto_group;
        auto_group.id = "cluster:auto";
        auto_group.label = has_rpc_accelerator
            ? "Cluster auto placement across local and RPC accelerators"
            : "Cluster auto placement across local accelerators";
        auto_group.backend_summary = "auto";
        auto_group.device_count = static_cast<int32_t>(gpu_devices.size());
        auto_group.uses_local_split = true;
        for (size_t i = 0; i < gpu_devices.size(); ++i) {
            const owned_device_info & device = *gpu_devices[i];
            if (!auto_group.devices_csv.empty()) {
                auto_group.devices_csv += ",";
            }
            auto_group.devices_csv += std::to_string(device.bridge_device_index);
            auto_group.memory_free += device.memory_free;
            auto_group.memory_total += device.memory_total;
        }
        groups.push_back(std::move(auto_group));
    }

    return groups;
}

std::string default_execution_group_id(const std::vector<owned_execution_group> & groups) {
    for (const owned_execution_group & group : groups) {
        if (group.id == "cluster:auto") {
            return group.id;
        }
    }
    for (const owned_execution_group & group : groups) {
        if (group.uses_local_split) {
            return group.id;
        }
    }
    for (const owned_execution_group & group : groups) {
        if (lower_copy(group.backend_summary).find("cpu") == std::string::npos) {
            return group.id;
        }
    }
    if (!groups.empty()) {
        return groups.front().id;
    }
    return {};
}

const owned_execution_group * find_group_by_id(
    const std::vector<owned_execution_group> & groups,
    const std::string & group_id) {
    for (const owned_execution_group & group : groups) {
        if (group.id == group_id) {
            return &group;
        }
    }
    return nullptr;
}

std::vector<const owned_execution_group *> ordered_auto_groups(
    const std::vector<owned_device_info> & devices,
    const std::vector<owned_execution_group> & groups,
    uint64_t minimum_total_bytes) {
    struct ranked_group {
        const owned_execution_group * group = nullptr;
        int category = 99;
        uint64_t memory_total = 0;
        int32_t device_count = 0;
    };

    std::vector<ranked_group> ranked;
    ranked.reserve(groups.size());

    for (const owned_execution_group & group : groups) {
        if (group.id == "cluster:auto") {
            continue;
        }
        if (minimum_total_bytes > 0 && group.memory_total < minimum_total_bytes) {
            continue;
        }

        const std::vector<const owned_device_info *> selected = devices_from_csv(devices, group.devices_csv);
        bool has_rpc = false;
        bool has_local_accel = false;
        bool has_cpu = false;

        for (const owned_device_info * device : selected) {
            if (device == nullptr) {
                continue;
            }
            if (is_rpc_backend(*device)) {
                has_rpc = true;
            } else if (is_cpu_backend(*device)) {
                has_cpu = true;
            } else {
                has_local_accel = true;
            }
        }

        int category = 99;
        if (has_local_accel && !has_rpc && group.device_count == 1) {
            category = 0;
        } else if (has_local_accel && !has_rpc && group.device_count > 1) {
            category = 1;
        } else if (has_local_accel && has_rpc && group.device_count == 2) {
            category = 2;
        } else if (has_local_accel && has_rpc) {
            category = 3;
        } else if (has_rpc && !has_local_accel) {
            category = 4;
        } else if (has_cpu) {
            category = 5;
        }

        ranked.push_back(ranked_group {
            &group,
            category,
            group.memory_total,
            group.device_count,
        });
    }

    std::sort(ranked.begin(), ranked.end(), [](const ranked_group & lhs, const ranked_group & rhs) {
        if (lhs.category != rhs.category) {
            return lhs.category < rhs.category;
        }
        if (lhs.device_count != rhs.device_count) {
            return lhs.device_count < rhs.device_count;
        }
        if (lhs.memory_total != rhs.memory_total) {
            return lhs.memory_total > rhs.memory_total;
        }
        return lhs.group->id < rhs.group->id;
    });

    std::vector<const owned_execution_group *> ordered;
    ordered.reserve(ranked.size());
    for (const ranked_group & entry : ranked) {
        ordered.push_back(entry.group);
    }
    return ordered;
}

int32_t first_device_index_from_csv(const std::string & devices_csv) {
    const std::vector<std::string> parts = split_csv(devices_csv);
    if (parts.empty()) {
        return -1;
    }
    return static_cast<int32_t>(std::atoi(parts.front().c_str()));
}

bool group_allowed_for_params(
    const owned_execution_group & group,
    const std::vector<owned_device_info> & devices,
    const owned_instance_params & params) {
    const std::vector<const owned_device_info *> selected = devices_from_csv(devices, group.devices_csv);
    bool has_allowed_accelerator = false;

    for (const owned_device_info * device : selected) {
        if (device == nullptr) {
            continue;
        }
        if (is_rpc_backend(*device)) {
            has_allowed_accelerator = true;
            continue;
        }
        if (is_cpu_backend(*device)) {
            if (!params.allow_cpu) {
                return false;
            }
            continue;
        }
        if (is_integrated_gpu(*device) && !params.allow_integrated_gpu) {
            return false;
        }
        has_allowed_accelerator = true;
    }

    return has_allowed_accelerator || params.allow_cpu != 0;
}

bool model_uses_realtime_native_audio_backend(const std::string & model_path) {
    return llama_server_bridge_realtime_backend_kind_from_model_path(model_path.c_str()) != 0;
}

bool model_uses_offline_audio_bridge_backend(const std::string & model_path) {
    const std::string lowered = lower_copy(model_path);
    return lowered.find("whisper") != std::string::npos
        || std::filesystem::path(lowered).extension() == ".bin";
}

bool model_uses_native_audio_backend(const std::string & model_path) {
    return model_uses_realtime_native_audio_backend(model_path)
        || model_uses_offline_audio_bridge_backend(model_path);
}

int32_t normalize_instance_model_kind(const llama_server_cluster_instance_params & input) {
    switch (input.model_kind) {
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_TEXT:
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_VISION:
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_EMBEDDINGS:
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_RERANK:
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_WHISPER:
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_REALTIME_AUDIO:
        case LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_DIARIZATION:
            return input.model_kind;
        default:
            break;
    }

    if (input.reranking != 0) {
        return LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_RERANK;
    }
    if (input.embedding != 0) {
        return LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_EMBEDDINGS;
    }
    if (input.diarization_model_path != nullptr && input.diarization_model_path[0] != '\0') {
        return LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_WHISPER;
    }
    if (input.mmproj_path != nullptr && input.mmproj_path[0] != '\0') {
        return LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_VISION;
    }
    return LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_TEXT;
}

owned_instance_params normalize_instance_params(
    const llama_server_cluster_instance_params & input,
    const std::vector<owned_execution_group> & groups,
    const std::vector<owned_device_info> & devices,
    std::string * error_out) {
    owned_instance_params out;
    if (input.model_path == nullptr || input.model_path[0] == '\0') {
        if (error_out != nullptr) {
            *error_out = "model_path is required";
        }
        return out;
    }

    out.model_path = input.model_path;
    out.name = (input.name != nullptr && input.name[0] != '\0') ? input.name : out.model_path;
    if (input.mmproj_path != nullptr && input.mmproj_path[0] != '\0') {
        out.mmproj_path = input.mmproj_path;
    }
    if (input.diarization_model_path != nullptr && input.diarization_model_path[0] != '\0') {
        out.diarization_model_path = input.diarization_model_path;
    }
    if (input.rpc_servers != nullptr && input.rpc_servers[0] != '\0') {
        out.rpc_servers = input.rpc_servers;
    }
    if (input.manual_devices_csv != nullptr && input.manual_devices_csv[0] != '\0') {
        out.manual_devices_csv = input.manual_devices_csv;
    }
    if (input.manual_tensor_split != nullptr && input.manual_tensor_split[0] != '\0') {
        out.manual_tensor_split = input.manual_tensor_split;
    }
    if (out.manual_devices_csv.empty()) {
        if (input.execution_group_id != nullptr && input.execution_group_id[0] != '\0') {
            out.execution_group_id = input.execution_group_id;
        } else {
            out.execution_group_id = default_execution_group_id(groups);
        }
    } else if (input.execution_group_id != nullptr && input.execution_group_id[0] != '\0') {
        out.execution_group_id = input.execution_group_id;
    } else {
        out.execution_group_id = "cluster:manual";
    }

    out.n_ctx = input.n_ctx;
    out.n_batch = input.n_batch;
    out.n_ubatch = input.n_ubatch;
    out.n_parallel = input.n_parallel;
    out.n_threads = input.n_threads;
    out.n_threads_batch = input.n_threads_batch;
    out.n_gpu_layers = input.n_gpu_layers;
    out.load_on_demand_grace_seconds = std::max<int32_t>(0, input.load_on_demand_grace_seconds);
    out.embedding = input.embedding;
    out.reranking = input.reranking;
    out.model_kind = normalize_instance_model_kind(input);
    out.allow_cpu = input.allow_cpu;
    out.allow_integrated_gpu = input.allow_integrated_gpu;

    if (!out.manual_devices_csv.empty()) {
        const std::vector<std::string> requested_devices = split_csv(out.manual_devices_csv);
        const std::vector<const owned_device_info *> selected_devices =
            devices_from_csv(devices, out.manual_devices_csv);
        if (requested_devices.empty() || selected_devices.size() != requested_devices.size()) {
            if (error_out != nullptr) {
                *error_out = "manual device selection is no longer available";
            }
            return {};
        }
        return out;
    }

    if (out.execution_group_id.empty()) {
        if (error_out != nullptr) {
            *error_out = "no execution groups are available";
        }
        return {};
    }
    const owned_execution_group * selected_group = find_group_by_id(groups, out.execution_group_id);
    if (selected_group == nullptr) {
        if (error_out != nullptr) {
            *error_out = "unknown execution_group_id: " + out.execution_group_id;
        }
        return {};
    }
    if (!group_allowed_for_params(*selected_group, devices, out)) {
        if (error_out != nullptr) {
            *error_out = "selected execution group requires CPU or integrated GPU, but GPU-first mode is enabled";
        }
        return {};
    }
    return out;
}

class scoped_env_override {
public:
    scoped_env_override(const char * name, const char * value)
        : name_(name != nullptr ? name : "") {
        if (name_.empty()) {
            return;
        }
        const char * existing = std::getenv(name_.c_str());
        if (existing != nullptr) {
            had_previous_ = true;
            previous_ = existing;
        }
        apply(value != nullptr ? value : "");
    }

    ~scoped_env_override() {
        if (name_.empty()) {
            return;
        }
        if (had_previous_) {
            apply(previous_.c_str());
        } else {
            clear();
        }
    }

private:
    void apply(const char * value) {
#if defined(_WIN32)
        SetEnvironmentVariableA(name_.c_str(), value);
#else
        setenv(name_.c_str(), value, 1);
#endif
    }

    void clear() {
#if defined(_WIN32)
        SetEnvironmentVariableA(name_.c_str(), nullptr);
#else
        unsetenv(name_.c_str());
#endif
    }

    std::string name_;
    bool had_previous_ = false;
    std::string previous_;
};

const owned_device_info * select_native_audio_device(
    const std::vector<owned_device_info> & devices,
    const std::string & execution_group_id,
    std::string * error_out) {

    const auto choose_from_selected = [](const std::vector<const owned_device_info *> & selected) -> const owned_device_info * {
        const owned_device_info * best_local = nullptr;
        for (const owned_device_info * device : selected) {
            if (device == nullptr || is_rpc_backend(*device) || is_cpu_backend(*device)) {
                continue;
            }
            if (best_local == nullptr || device->memory_free > best_local->memory_free) {
                best_local = device;
            }
        }
        return best_local;
    };

    if (!execution_group_id.empty() && execution_group_id != "cluster:auto") {
        const std::vector<owned_execution_group> groups = build_execution_groups(devices, false);
        const owned_execution_group * group = find_group_by_id(groups, execution_group_id);
        if (group == nullptr) {
            if (error_out != nullptr) {
                *error_out = "unknown execution_group_id for native transcription: " + execution_group_id;
            }
            return nullptr;
        }
        const auto selected = devices_from_csv(devices, group->devices_csv);
        const owned_device_info * device = choose_from_selected(selected);
        if (device != nullptr) {
            return device;
        }
    }

    const owned_device_info * best = nullptr;
    for (const owned_device_info & device : devices) {
        if (is_rpc_backend(device) || is_cpu_backend(device)) {
            continue;
        }
        if (best == nullptr || device.memory_free > best->memory_free) {
            best = &device;
        }
    }
    if (best != nullptr) {
        return best;
    }

    if (error_out != nullptr) {
        *error_out = "no local GPU backend is available for native transcription";
    }
    return nullptr;
}

bridge_handle_ptr create_audio_only_bridge_for_execution_group(
    const std::string & execution_group_id,
    std::string * error_out) {

    std::string device_error;
    const std::vector<owned_device_info> devices = query_devices({}, &device_error);
    if (!device_error.empty()) {
        if (error_out != nullptr) {
            *error_out = device_error;
        }
        return bridge_handle_ptr(nullptr);
    }

    const owned_device_info * selected = select_native_audio_device(devices, execution_group_id, error_out);
    if (selected == nullptr) {
        return bridge_handle_ptr(nullptr);
    }

    llama_server_bridge_params bridge_params = llama_server_bridge_default_params();
    bridge_params.cache_ram_mib = 0;
    bridge_params.use_mlock = 0;
    bridge_params.use_mmap = 0;
    bridge_params.no_host = 1;
    bridge_params.no_extra_bufts = 1;
    bridge_params.gpu = selected->bridge_device_index;
    bridge_params.main_gpu = selected->bridge_device_index;
    bridge_params.split_mode = 0;
    bridge_params.n_gpu_layers = -1;

    scoped_env_override audio_only_env("LLAMA_SERVER_AUDIO_ONLY", "1");
    bridge_handle_ptr bridge(llama_server_bridge_create(&bridge_params));
    if (!bridge) {
        if (error_out != nullptr) {
            *error_out =
                "failed to create audio-only bridge for execution group "
                + (execution_group_id.empty() ? std::string("<auto>") : execution_group_id);
        }
        return bridge_handle_ptr(nullptr);
    }

    return bridge;
}

bool build_offline_audio_metadata_json(
    const llama_server_cluster_native_audio_transcription_request & req,
    std::string * metadata_json_out,
    std::string * error_out) {

    if (metadata_json_out == nullptr) {
        if (error_out != nullptr) {
            *error_out = "metadata_json_out is required";
        }
        return false;
    }

    json metadata = json::object();
    if (req.metadata_json != nullptr && req.metadata_json[0] != '\0') {
        try {
            metadata = json::parse(req.metadata_json);
        } catch (const std::exception & e) {
            if (error_out != nullptr) {
                *error_out = std::string("invalid metadata_json: ") + e.what();
            }
            return false;
        }
        if (!metadata.is_object()) {
            if (error_out != nullptr) {
                *error_out = "metadata_json must be a JSON object";
            }
            return false;
        }
    }

    const std::string model_path = safe_cstr(req.model_path);
    if (!model_path.empty()
        && model_uses_offline_audio_bridge_backend(model_path)
        && !metadata.contains("whisper_model")
        && !metadata.contains("whisper_hf_repo")
        && !metadata.contains("whisper_hf_file")) {
        metadata["whisper_model"] = model_path;
    }
    if (req.enable_diarization != 0
        && req.diarization_model_path != nullptr
        && req.diarization_model_path[0] != '\0'
        && !metadata.contains("diarization_model_path")) {
        metadata["diarization_model_path"] = req.diarization_model_path;
    }
    if (req.enable_diarization != 0) {
        if (!metadata.contains("enable_diarization")) {
            metadata["enable_diarization"] = true;
        }
        if (!metadata.contains("mode")) {
            metadata["mode"] = "transcript";
        }
        if (!metadata.contains("custom")) {
            metadata["custom"] = "auto";
        }
        if (!metadata.contains("output_dir")) {
            std::error_code ec;
            const std::filesystem::path output_dir =
                std::filesystem::temp_directory_path(ec) / "OpenResearchTools" / "audio-transcriptions";
            if (!ec) {
                metadata["output_dir"] = output_dir.string();
            }
        }
    }
    if (safe_cstr(req.execution_group_id).size() > 0 || model_uses_offline_audio_bridge_backend(model_path)) {
        std::string device_error;
        const std::vector<owned_device_info> devices = query_devices({}, &device_error);
        if (!device_error.empty()) {
            if (error_out != nullptr) {
                *error_out = device_error;
            }
            return false;
        }
        const owned_device_info * selected =
            select_native_audio_device(devices, safe_cstr(req.execution_group_id), nullptr);
        if (selected != nullptr) {
            if (!metadata.contains("whisper_gpu_device")) {
                metadata["whisper_gpu_device"] = selected->bridge_device_index;
            }
            if (!metadata.contains("diarization_device") && !selected->name.empty()) {
                metadata["diarization_device"] = selected->name;
            }
            if (metadata.contains("whisper_no_gpu")) {
                metadata.erase("whisper_no_gpu");
            }
        }
    }

    *metadata_json_out = metadata.dump();
    return true;
}

int32_t run_offline_audio_bridge_transcription(
    const llama_server_cluster_native_audio_transcription_request & req,
    llama_server_cluster_json_result * out) {

    if (out == nullptr) {
        return -1;
    }

    std::string metadata_json;
    std::string metadata_error;
    if (!build_offline_audio_metadata_json(req, &metadata_json, &metadata_error)) {
        out->error = dup_cstr(metadata_error);
        out->status = 500;
        return -1;
    }

    const auto load_started = steady_clock::now();
    std::string bridge_error;
    bridge_handle_ptr bridge = create_audio_only_bridge_for_execution_group(
        safe_cstr(req.execution_group_id),
        &bridge_error);
    out->metrics.loaded_this_call = 1;
    out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);
    if (!bridge) {
        out->error = dup_cstr(bridge_error.empty() ? "failed to create audio-only bridge" : bridge_error);
        out->status = 500;
        return -1;
    }

    llama_server_bridge_audio_raw_request bridge_req = llama_server_bridge_default_audio_raw_request();
    bridge_req.audio_bytes = req.audio_bytes;
    bridge_req.audio_bytes_len = req.audio_bytes_len;
    bridge_req.audio_format = req.audio_format;
    bridge_req.metadata_json = metadata_json.c_str();
    bridge_req.ffmpeg_convert = req.ffmpeg_convert;

    llama_server_bridge_json_result bridge_out = llama_server_bridge_empty_json_result();
    const auto request_started = steady_clock::now();
    const int32_t rc = llama_server_bridge_audio_transcriptions_raw(
        bridge.get(),
        &bridge_req,
        &bridge_out);
    out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);

    out->ok = (rc == 0 && bridge_out.ok != 0) ? 1 : 0;
    out->status = bridge_out.status;
    if (bridge_out.json != nullptr) {
        out->json = dup_cstr(bridge_out.json);
    }
    if (bridge_out.error_json != nullptr) {
        out->error = dup_cstr(bridge_out.error_json);
    }
    llama_server_bridge_json_result_free(&bridge_out);

    if (rc != 0 || out->ok == 0) {
        if (out->error == nullptr) {
            out->error = dup_cstr("offline audio bridge transcription request failed");
        }
        if (out->status == 0) {
            out->status = 500;
        }
        return -1;
    }

    return 0;
}

bridge_handle_ptr create_bridge_for_instance(
    model_instance & instance,
    const std::string & execution_group_id,
    std::string * error_out) {
    std::string device_error;
    const std::vector<owned_device_info> devices = query_devices(instance.params.rpc_servers, &device_error);
    if (!device_error.empty()) {
        if (error_out != nullptr) {
            *error_out = device_error;
        }
        return bridge_handle_ptr(nullptr);
    }

    const bool has_manual_devices = !instance.params.manual_devices_csv.empty();
    const std::vector<owned_execution_group> groups =
        has_manual_devices ? std::vector<owned_execution_group>{}
                           : build_execution_groups(devices, !instance.params.rpc_servers.empty());
    const owned_execution_group * group = nullptr;
    std::string active_devices_csv;
    std::vector<const owned_device_info *> selected_devices;
    bool uses_multi_device_runtime = false;

    if (has_manual_devices) {
        active_devices_csv = instance.params.manual_devices_csv;
        selected_devices = devices_from_csv(devices, active_devices_csv);
        const std::vector<std::string> requested_devices = split_csv(active_devices_csv);
        if (requested_devices.empty() || selected_devices.size() != requested_devices.size()) {
            if (error_out != nullptr) {
                *error_out = "manual device selection is no longer available";
            }
            return bridge_handle_ptr(nullptr);
        }
        uses_multi_device_runtime = selected_devices.size() > 1;
    } else {
        group = find_group_by_id(groups, execution_group_id);
        if (group == nullptr) {
            if (error_out != nullptr) {
                *error_out = "execution group is no longer available: " + execution_group_id;
            }
            return bridge_handle_ptr(nullptr);
        }
        active_devices_csv = group->devices_csv;
        selected_devices = devices_from_csv(devices, active_devices_csv);
        uses_multi_device_runtime = group->device_count > 1;
    }

    llama_server_bridge_params bridge_params = llama_server_bridge_default_params();
    bridge_params.model_path = instance.params.model_path.c_str();
    bridge_params.mmproj_path = instance.params.mmproj_path.empty() ? nullptr : instance.params.mmproj_path.c_str();
    bridge_params.embedding = instance.params.embedding;
    bridge_params.reranking = instance.params.reranking;
    bridge_params.cache_ram_mib = 0;
    bridge_params.use_mlock = 0;

    if (instance.params.n_ctx > 0) {
        bridge_params.n_ctx = instance.params.n_ctx;
    }
    if (instance.params.n_batch > 0) {
        bridge_params.n_batch = instance.params.n_batch;
    }
    if (instance.params.n_ubatch > 0) {
        bridge_params.n_ubatch = instance.params.n_ubatch;
    }
    if (instance.params.n_parallel > 0) {
        bridge_params.n_parallel = instance.params.n_parallel;
    }
    if (instance.params.n_threads > 0) {
        bridge_params.n_threads = instance.params.n_threads;
    } else {
        bridge_params.n_threads = cluster_default_thread_count();
    }
    if (instance.params.n_threads_batch > 0) {
        bridge_params.n_threads_batch = instance.params.n_threads_batch;
    } else {
        bridge_params.n_threads_batch = cluster_default_thread_count();
    }
    if (uses_multi_device_runtime) {
        const int32_t hybrid_batch_cap = std::max<int32_t>(
            32,
            env_i32_or_default("ENGINE_CLUSTER_HYBRID_BATCH_CAP", 64));
        bridge_params.n_batch = std::min<int32_t>(bridge_params.n_batch, hybrid_batch_cap);
        bridge_params.n_ubatch = std::min<int32_t>(bridge_params.n_ubatch, hybrid_batch_cap);
        bridge_params.kv_unified = 0;

        if (!env_flag_enabled("ENGINE_CLUSTER_DISABLE_LOW_RAM_LOAD")) {
            bridge_params.use_mmap = 0;
            bridge_params.no_host = 1;
            bridge_params.no_extra_bufts = 1;
            if (env_flag_enabled("ENGINE_CLUSTER_FORCE_DIRECT_IO")) {
                bridge_params.use_direct_io = 1;
            }
        }
    }

    std::string tensor_split_storage;
    const bool use_split_mode = has_manual_devices ? uses_multi_device_runtime : group->uses_local_split;
    if (use_split_mode) {
        bridge_params.devices = active_devices_csv.c_str();
        bridge_params.split_mode = 1;
        bridge_params.main_gpu = first_device_index_from_csv(active_devices_csv);
        bridge_params.gpu = -1;
        tensor_split_storage = instance.params.manual_tensor_split;
        if (tensor_split_storage.empty()) {
            tensor_split_storage =
                build_tensor_split_csv_for_group(selected_devices, instance.params.model_path);
        }
        bridge_params.tensor_split = tensor_split_storage.empty() ? nullptr : tensor_split_storage.c_str();
    } else {
        bridge_params.gpu = first_device_index_from_csv(active_devices_csv);
        bridge_params.split_mode = 0;
    }

    const bool cpu_only_runtime = std::all_of(
        selected_devices.begin(),
        selected_devices.end(),
        [](const owned_device_info * device) {
            return device == nullptr || is_cpu_backend(*device);
        });
    if (cpu_only_runtime) {
        bridge_params.n_gpu_layers = 0;
    } else if (instance.params.n_gpu_layers != 0) {
        bridge_params.n_gpu_layers = instance.params.n_gpu_layers;
    }

    bridge_handle_ptr bridge(llama_server_bridge_create(&bridge_params));
    if (!bridge) {
        if (error_out != nullptr) {
            *error_out = has_manual_devices
                ? "llama_server_bridge_create failed for manual device allocation"
                : "llama_server_bridge_create failed for execution group " + group->id;
        }
        return bridge_handle_ptr(nullptr);
    }

    return bridge;
}

bridge_handle_ptr create_bridge_for_instance(model_instance & instance, std::string * error_out) {
    return create_bridge_for_instance(instance, instance.params.execution_group_id, error_out);
}

std::string select_native_audio_backend_name(
    const std::string & execution_group_id,
    std::string * error_out) {

    std::string device_error;
    const std::vector<owned_device_info> devices = query_devices({}, &device_error);
    if (!device_error.empty()) {
        if (error_out != nullptr) {
            *error_out = device_error;
        }
        return {};
    }

    const owned_device_info * best = select_native_audio_device(devices, execution_group_id, error_out);
    if (best != nullptr) {
        return best->name;
    }
    return {};
}

int32_t run_native_audio_transcription(
    const llama_server_cluster_native_audio_transcription_request & req,
    llama_server_cluster_json_result * out) {

    if (out == nullptr) {
        return -1;
    }
    *out = llama_server_cluster_empty_json_result();
    const auto started_at = steady_clock::now();
    out->metrics = make_base_metrics_for_model(
        safe_cstr(req.model_path),
        safe_cstr(req.diarization_model_path),
        static_cast<uint64_t>(req.audio_bytes_len));

    if (req.model_path == nullptr || req.model_path[0] == '\0') {
        out->error = dup_cstr("model_path is required");
        return -1;
    }
    if (req.audio_bytes == nullptr || req.audio_bytes_len == 0) {
        out->error = dup_cstr("audio_bytes are required");
        return -1;
    }
    if (req.enable_diarization != 0 && (req.diarization_model_path == nullptr || req.diarization_model_path[0] == '\0')) {
        out->error = dup_cstr("diarization_model_path is required when enable_diarization=1");
        return -1;
    }

    if (!model_uses_realtime_native_audio_backend(safe_cstr(req.model_path))) {
        return run_offline_audio_bridge_transcription(req, out);
    }

    std::string backend_error;
    const std::string backend_name =
        select_native_audio_backend_name(safe_cstr(req.execution_group_id), &backend_error);
    if (backend_name.empty()) {
        out->error = dup_cstr(backend_error.empty() ? "failed to resolve native transcription backend" : backend_error);
        return -1;
    }

    llama_server_bridge_audio_session_params session_params =
        llama_server_bridge_default_audio_session_params();
    std::unique_ptr<llama_server_bridge_audio_session, void(*)(llama_server_bridge_audio_session *)> session(
        llama_server_bridge_audio_session_create(&session_params),
        llama_server_bridge_audio_session_destroy);
    if (!session) {
        out->error = dup_cstr("failed to create audio session");
        return -1;
    }

    llama_server_bridge_audio_transcription_params transcription_params =
        llama_server_bridge_default_audio_transcription_params();
    transcription_params.mode = LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE;
    transcription_params.metadata_json =
        (req.metadata_json != nullptr && req.metadata_json[0] != '\0') ? req.metadata_json : nullptr;
    transcription_params.realtime_params.model_path = req.model_path;
    transcription_params.realtime_params.backend_name = backend_name.c_str();
    if (transcription_params.realtime_params.expected_sample_rate_hz == 0) {
        transcription_params.realtime_params.expected_sample_rate_hz =
            session_params.expected_input_sample_rate_hz;
    }

    bool diarization_requested = req.enable_diarization != 0;
    bool diarization_stopped = !diarization_requested;
    std::string diarization_markdown;
    if (diarization_requested) {
        llama_server_bridge_realtime_params diarization_params =
            llama_server_bridge_default_realtime_params_for_backend(
                LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER);
        diarization_params.backend_kind = LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER;
        diarization_params.model_path = req.diarization_model_path;
        diarization_params.backend_name = backend_name.c_str();
        if (diarization_params.expected_sample_rate_hz == 0) {
            diarization_params.expected_sample_rate_hz = session_params.expected_input_sample_rate_hz;
        }
        if (llama_server_bridge_audio_session_start_diarization(session.get(), &diarization_params) != 0) {
            out->error = dup_cstr(llama_server_bridge_audio_session_last_error(session.get()));
            return -1;
        }
    }
    if (llama_server_bridge_audio_session_start_transcription(session.get(), &transcription_params) != 0) {
        out->error = dup_cstr(llama_server_bridge_audio_session_last_error(session.get()));
        return -1;
    }

    const std::string audio_format = lower_copy(safe_cstr(req.audio_format));
    if (audio_format.empty() || audio_format == "wav" || audio_format == "wave") {
        decoded_wav_pcm16 wav{};
        std::string wav_error;
        if (decode_wav_pcm16(req.audio_bytes, req.audio_bytes_len, wav, &wav_error)) {
            if (wav.sample_rate_hz != session_params.expected_input_sample_rate_hz) {
                out->error = dup_cstr("only 16 kHz mono WAV input is supported without FFmpeg");
                return -1;
            }
            if (llama_server_bridge_audio_session_push_audio(
                    session.get(),
                    wav.samples.data(),
                    wav.samples.size(),
                    wav.sample_rate_hz,
                    wav.channels,
                    LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16) != 0) {
                out->error = dup_cstr(llama_server_bridge_audio_session_last_error(session.get()));
                return -1;
            }
        } else if (llama_server_bridge_audio_session_push_encoded(
                session.get(),
                req.audio_bytes,
                req.audio_bytes_len,
                "wav") != 0) {
            out->error = dup_cstr(wav_error.empty()
                ? llama_server_bridge_audio_session_last_error(session.get())
                : wav_error);
            return -1;
        }
    } else if (llama_server_bridge_audio_session_push_encoded(
            session.get(),
            req.audio_bytes,
            req.audio_bytes_len,
            req.audio_format) != 0) {
        out->error = dup_cstr(llama_server_bridge_audio_session_last_error(session.get()));
        return -1;
    }
    if (llama_server_bridge_audio_session_flush_audio(session.get()) != 0) {
        out->error = dup_cstr(llama_server_bridge_audio_session_last_error(session.get()));
        return -1;
    }
    if (diarization_requested && llama_server_bridge_audio_session_stop_diarization(session.get()) != 0) {
        out->error = dup_cstr(llama_server_bridge_audio_session_last_error(session.get()));
        return -1;
    }

    std::string result_json;
    std::string last_error;
    std::vector<native_transcript_item> pieces;
    std::vector<native_transcript_item> words;
    int32_t status = 200;
    bool transcription_stopped = false;
    const auto deadline = steady_clock::now() + std::chrono::minutes(4);
    while (steady_clock::now() < deadline) {
        (void) llama_server_bridge_audio_session_wait_events(session.get(), 1000);

        llama_server_bridge_audio_event * events = nullptr;
        size_t event_count = 0;
        if (llama_server_bridge_audio_session_drain_events(session.get(), &events, &event_count, 256) != 0) {
            last_error = safe_cstr(llama_server_bridge_audio_session_last_error(session.get()));
            break;
        }

        for (size_t i = 0; i < event_count; ++i) {
            const auto & event = events[i];
            if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON && event.text != nullptr) {
                result_json = event.text;
                const std::string detail = safe_cstr(event.detail);
                if (!detail.empty()) {
                    status = std::atoi(detail.c_str());
                    if (status <= 0) {
                        status = 200;
                    }
                }
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT) {
                append_unique_transcript_item(
                    pieces,
                    native_transcript_item{
                        event.start_sample,
                        event.end_sample,
                        safe_cstr(event.text),
                    });
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT) {
                append_unique_transcript_item(
                    words,
                    native_transcript_item{
                        event.start_sample,
                        event.end_sample,
                        safe_cstr(event.text),
                    });
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT) {
                const std::string markdown = safe_cstr(event.text);
                if (!trim_copy(markdown).empty()) {
                    diarization_markdown = markdown;
                }
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR) {
                last_error = !safe_cstr(event.detail).empty() ? safe_cstr(event.detail) : safe_cstr(event.text);
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR) {
                last_error = !safe_cstr(event.detail).empty() ? safe_cstr(event.detail) : safe_cstr(event.text);
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED) {
                diarization_stopped = true;
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED) {
                transcription_stopped = true;
                if (safe_cstr(event.detail) == "failed" && last_error.empty()) {
                    last_error = safe_cstr(llama_server_bridge_audio_session_last_error(session.get()));
                }
            }
        }

        if (events != nullptr) {
            llama_server_bridge_audio_session_free_events(events, event_count);
        }

        if (transcription_stopped && diarization_stopped) {
            break;
        }
    }

    if (result_json.empty() && last_error.empty()
        && (!pieces.empty() || !words.empty() || !trim_copy(diarization_markdown).empty())) {
        result_json = build_native_transcription_result_json(pieces, words, diarization_markdown);
    }

    if (result_json.empty()) {
        if (last_error.empty()) {
            last_error = safe_cstr(llama_server_bridge_audio_session_last_error(session.get()));
        }
        if (last_error.empty()) {
            last_error = "native audio transcription did not return a final JSON result";
        }
        out->error = dup_cstr(last_error);
        out->status = 500;
        return -1;
    }

    out->ok = 1;
    out->status = status;
    out->json = dup_cstr(result_json);
    const double total_ms = duration_ms(steady_clock::now() - started_at);
    out->metrics.load_ms = total_ms;
    out->metrics.request_total_ms = total_ms;
    if (out->json == nullptr) {
        out->ok = 0;
        out->status = 500;
        out->error = dup_cstr("failed to allocate native transcription JSON result");
        return -1;
    }
    return 0;
}

std::chrono::seconds instance_load_on_demand_grace_period(const model_instance & instance) {
    return std::chrono::seconds(std::max<int32_t>(0, instance.params.load_on_demand_grace_seconds));
}

bool instance_should_unload_immediately_on_demand(const model_instance & instance) {
    return instance_load_on_demand_grace_period(instance).count() <= 0;
}

void set_instance_grace_state(model_instance & instance) {
    instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_GRACE;
    instance.grace_deadline_steady = steady_clock::now() + instance_load_on_demand_grace_period(instance);
    instance.grace_deadline_unix_ms = unix_ms_from_steady_deadline(instance.grace_deadline_steady);
}

void clear_instance_grace_state(model_instance & instance) {
    instance.grace_deadline_steady = steady_clock::time_point::min();
    instance.grace_deadline_unix_ms = 0;
}

bool ensure_instance_loaded_locked(model_instance & instance, bool for_request, std::string * error_out) {
    if (instance.bridge) {
        clear_instance_grace_state(instance);
        if (for_request) {
            instance.active_request_count += 1;
            instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_SERVING;
        } else if (instance.active_request_count == 0) {
            instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADED;
        }
        return true;
    }
    if (model_uses_native_audio_backend(instance.params.model_path)) {
        clear_instance_grace_state(instance);
        instance.last_error.clear();
        if (for_request) {
            instance.active_request_count += 1;
            instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_SERVING;
        } else if (instance.active_request_count == 0) {
            instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADED;
        }
        return true;
    }

    instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADING;
    instance.last_error.clear();
    clear_instance_grace_state(instance);

    std::string load_error;
    bridge_handle_ptr bridge(nullptr);
    if (instance.params.execution_group_id == "cluster:auto") {
        std::string device_error;
        const std::vector<owned_device_info> devices = query_devices(instance.params.rpc_servers, &device_error);
        if (!device_error.empty()) {
            instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_FAILED;
            instance.last_error = device_error;
            if (error_out != nullptr) {
                *error_out = device_error;
            }
            return false;
        }

        const std::vector<owned_execution_group> groups =
            build_execution_groups(devices, !instance.params.rpc_servers.empty());
        const uint64_t minimum_total_bytes =
            file_size_or_zero(instance.params.model_path) + file_size_or_zero(instance.params.mmproj_path);
        std::vector<const owned_execution_group *> candidates =
            ordered_auto_groups(devices, groups, minimum_total_bytes);
        if (candidates.empty()) {
            candidates = ordered_auto_groups(devices, groups, 0);
        }
        const owned_execution_group * chosen = nullptr;
        for (const owned_execution_group * candidate : candidates) {
            const std::vector<const owned_device_info *> selected =
                devices_from_csv(devices, candidate->devices_csv);
            const bool has_gpu_accel = std::any_of(
                selected.begin(),
                selected.end(),
                [](const owned_device_info * device) {
                    return device != nullptr && !is_cpu_backend(*device);
                });
            if (has_gpu_accel && group_allowed_for_params(*candidate, devices, instance.params)) {
                chosen = candidate;
                break;
            }
        }
        if (chosen == nullptr) {
            load_error = "no GPU execution groups are currently available";
        } else {
            bridge = create_bridge_for_instance(instance, chosen->id, &load_error);
            if (bridge) {
                instance.params.execution_group_id = chosen->id;
            } else if (load_error.empty()) {
                load_error = "load failed for selected automatic execution group " + chosen->id;
            }
        }
    } else {
        bridge = create_bridge_for_instance(instance, &load_error);
    }
    if (!bridge) {
        instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_FAILED;
        instance.last_error = load_error.empty() ? "load failed" : load_error;
        if (error_out != nullptr) {
            *error_out = instance.last_error;
        }
        return false;
    }

    instance.bridge = std::move(bridge);
    if (for_request) {
        instance.active_request_count += 1;
        instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_SERVING;
    } else {
        instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADED;
    }
    return true;
}

void finish_request_locked(model_instance & instance) {
    if (instance.active_request_count > 0) {
        instance.active_request_count -= 1;
    }
    const bool uses_native_audio_backend = model_uses_native_audio_backend(instance.params.model_path);
    if (instance.retention_mode == LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED) {
        clear_instance_grace_state(instance);
        instance.state = (instance.bridge || uses_native_audio_backend)
                             ? LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADED
                             : LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
    } else if (instance_should_unload_immediately_on_demand(instance)) {
        instance.bridge.reset();
        instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
        clear_instance_grace_state(instance);
    } else if (instance.bridge) {
        set_instance_grace_state(instance);
    } else {
        instance.state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
        clear_instance_grace_state(instance);
    }
}

int32_t instance_parallel_limit(const model_instance & instance) {
    return std::max<int32_t>(1, instance.params.n_parallel);
}

void wait_for_instance_slot_locked(
    model_instance & instance,
    std::unique_lock<std::mutex> & lock) {
    bool counted_as_queued = false;
    while (instance.active_request_count >= instance_parallel_limit(instance)) {
        if (!counted_as_queued) {
            instance.queued_request_count += 1;
            counted_as_queued = true;
        }
        instance.cv.wait(lock);
    }
    if (counted_as_queued && instance.queued_request_count > 0) {
        instance.queued_request_count -= 1;
    }
}

} // namespace

struct llama_server_cluster {
    std::mutex mutex;
    std::string last_error;
    std::unordered_map<int64_t, std::shared_ptr<model_instance>> instances;
    int64_t next_instance_id = 1;
    std::atomic<bool> stop_requested = false;
    std::thread housekeeper;

    llama_server_cluster();
    ~llama_server_cluster();
    void housekeeper_loop();
};

namespace {

llama_server_cluster & shared_cluster_instance() {
    static llama_server_cluster instance;
    return instance;
}

std::shared_ptr<model_instance> find_instance(
    llama_server_cluster * cluster,
    int64_t instance_id) {
    if (cluster == nullptr) {
        return {};
    }
    std::lock_guard<std::mutex> lock(cluster->mutex);
    const auto it = cluster->instances.find(instance_id);
    if (it == cluster->instances.end()) {
        return {};
    }
    return it->second;
}

void set_cluster_error(llama_server_cluster * cluster, const std::string & message) {
    if (cluster == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(cluster->mutex);
    cluster->last_error = message;
}

} // namespace

llama_server_cluster::llama_server_cluster() {
    housekeeper = std::thread([this]() { this->housekeeper_loop(); });
}

llama_server_cluster::~llama_server_cluster() {
    stop_requested.store(true);
    if (housekeeper.joinable()) {
        housekeeper.join();
    }

    std::vector<std::shared_ptr<model_instance>> instances_snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto & pair : instances) {
            instances_snapshot.push_back(pair.second);
        }
        instances.clear();
    }

    for (const auto & instance : instances_snapshot) {
        std::lock_guard<std::mutex> lock(instance->mutex);
        instance->bridge.reset();
        instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
        clear_instance_grace_state(*instance);
    }
}

void llama_server_cluster::housekeeper_loop() {
    while (!stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::vector<std::shared_ptr<model_instance>> instances_snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto & pair : instances) {
                instances_snapshot.push_back(pair.second);
            }
        }

        const auto now = steady_clock::now();
        for (const auto & instance : instances_snapshot) {
            std::lock_guard<std::mutex> lock(instance->mutex);
            if (instance->retention_mode != LLAMA_SERVER_CLUSTER_INSTANCE_LOAD_ON_DEMAND ||
                !instance->bridge ||
                instance->active_request_count != 0 ||
                instance->state != LLAMA_SERVER_CLUSTER_INSTANCE_STATE_GRACE ||
                instance->grace_deadline_steady == steady_clock::time_point::min() ||
                now < instance->grace_deadline_steady) {
                continue;
            }

            instance->bridge.reset();
            instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
            clear_instance_grace_state(*instance);
        }
    }
}

extern "C" {

struct llama_server_cluster_instance_params llama_server_cluster_default_instance_params(void) {
    const llama_server_bridge_params bridge_params = llama_server_bridge_default_params();
    llama_server_cluster_instance_params params{};
    params.name = nullptr;
    params.model_path = nullptr;
    params.mmproj_path = nullptr;
    params.diarization_model_path = nullptr;
    params.execution_group_id = nullptr;
    params.rpc_servers = nullptr;
    params.manual_devices_csv = nullptr;
    params.manual_tensor_split = nullptr;
    params.retention_mode = LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED;
    params.load_on_demand_grace_seconds = kDefaultLoadOnDemandGraceSeconds;
    params.embedding = bridge_params.embedding;
    params.reranking = bridge_params.reranking;
    params.model_kind = LLAMA_SERVER_CLUSTER_INSTANCE_MODEL_KIND_TEXT;
    params.allow_cpu = 0;
    params.allow_integrated_gpu = 0;
    params.n_ctx = bridge_params.n_ctx;
    params.n_batch = bridge_params.n_batch;
    params.n_ubatch = bridge_params.n_ubatch;
    params.n_parallel = bridge_params.n_parallel;
    params.n_threads = cluster_default_thread_count();
    params.n_threads_batch = cluster_default_thread_count();
    params.n_gpu_layers = bridge_params.n_gpu_layers;
    return params;
}

struct llama_server_cluster_chat_request llama_server_cluster_default_chat_request(void) {
    const llama_server_bridge_chat_request bridge_req = llama_server_bridge_default_chat_request();
    llama_server_cluster_chat_request req{};
    req.instance_id = 0;
    req.prompt = nullptr;
    req.n_predict = bridge_req.n_predict;
    req.temperature = bridge_req.temperature;
    req.top_p = bridge_req.top_p;
    req.top_k = bridge_req.top_k;
    req.min_p = bridge_req.min_p;
    req.repeat_last_n = bridge_req.repeat_last_n;
    req.repeat_penalty = bridge_req.repeat_penalty;
    req.reasoning = bridge_req.reasoning;
    req.reasoning_budget = bridge_req.reasoning_budget;
    req.reasoning_format = bridge_req.reasoning_format;
    return req;
}

struct llama_server_cluster_chat_result llama_server_cluster_empty_chat_result(void) {
    llama_server_cluster_chat_result out{};
    out.ok = 0;
    out.text = nullptr;
    out.error = nullptr;
    return out;
}

struct llama_server_cluster_vlm_request llama_server_cluster_default_vlm_request(void) {
    const llama_server_bridge_vlm_request bridge_req = llama_server_bridge_default_vlm_request();
    llama_server_cluster_vlm_request req{};
    req.instance_id = 0;
    req.prompt = nullptr;
    req.image_bytes = nullptr;
    req.image_bytes_len = 0;
    req.n_predict = bridge_req.n_predict;
    req.temperature = bridge_req.temperature;
    req.top_p = bridge_req.top_p;
    req.top_k = bridge_req.top_k;
    req.min_p = bridge_req.min_p;
    req.repeat_last_n = bridge_req.repeat_last_n;
    req.repeat_penalty = bridge_req.repeat_penalty;
    req.reasoning = bridge_req.reasoning;
    req.reasoning_budget = bridge_req.reasoning_budget;
    req.reasoning_format = bridge_req.reasoning_format;
    return req;
}

struct llama_server_cluster_vlm_result llama_server_cluster_empty_vlm_result(void) {
    llama_server_cluster_vlm_result out{};
    out.ok = 0;
    out.text = nullptr;
    out.error = nullptr;
    return out;
}

struct llama_server_cluster_embeddings_request llama_server_cluster_default_embeddings_request(void) {
    const llama_server_bridge_embeddings_request bridge_req = llama_server_bridge_default_embeddings_request();
    llama_server_cluster_embeddings_request req{};
    req.instance_id = 0;
    req.body_json = bridge_req.body_json;
    req.oai_compat = bridge_req.oai_compat;
    return req;
}

struct llama_server_cluster_rerank_request llama_server_cluster_default_rerank_request(void) {
    const llama_server_bridge_rerank_request bridge_req = llama_server_bridge_default_rerank_request();
    llama_server_cluster_rerank_request req{};
    req.instance_id = 0;
    req.body_json = bridge_req.body_json;
    return req;
}

struct llama_server_cluster_audio_raw_request llama_server_cluster_default_audio_raw_request(void) {
    const llama_server_bridge_audio_raw_request bridge_req = llama_server_bridge_default_audio_raw_request();
    llama_server_cluster_audio_raw_request req{};
    req.instance_id = 0;
    req.audio_bytes = bridge_req.audio_bytes;
    req.audio_bytes_len = bridge_req.audio_bytes_len;
    req.audio_format = bridge_req.audio_format;
    req.metadata_json = bridge_req.metadata_json;
    req.ffmpeg_convert = bridge_req.ffmpeg_convert;
    req.enable_diarization = 0;
    req.diarization_model_path = nullptr;
    return req;
}

struct llama_server_cluster_native_audio_transcription_request llama_server_cluster_default_native_audio_transcription_request(void) {
    llama_server_cluster_native_audio_transcription_request req{};
    req.model_path = nullptr;
    req.execution_group_id = nullptr;
    req.audio_bytes = nullptr;
    req.audio_bytes_len = 0;
    req.audio_format = "wav";
    req.metadata_json = nullptr;
    req.ffmpeg_convert = 1;
    req.enable_diarization = 0;
    req.diarization_model_path = nullptr;
    return req;
}

struct llama_server_cluster_json_result llama_server_cluster_empty_json_result(void) {
    llama_server_cluster_json_result out{};
    out.ok = 0;
    out.status = 0;
    out.json = nullptr;
    out.error = nullptr;
    return out;
}

struct llama_server_cluster * llama_server_cluster_create(void) {
    try {
        return &shared_cluster_instance();
    } catch (...) {
        return nullptr;
    }
}

void llama_server_cluster_destroy(struct llama_server_cluster * cluster) {
    (void) cluster;
}

const char * llama_server_cluster_last_error(const struct llama_server_cluster * cluster) {
    if (cluster == nullptr) {
        return "cluster is null";
    }
    return cluster->last_error.c_str();
}

int32_t llama_server_cluster_get_local_node_info(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_node_info * out_info) {
    if (cluster == nullptr || out_info == nullptr) {
        set_cluster_error(cluster, "cluster and out_info are required");
        return -1;
    }

    out_info->display_name = dup_cstr(host_name_string());
    out_info->node_id = dup_cstr(host_name_string() + "-" + os_name_string() + "-" + arch_name_string());
    out_info->os_name = dup_cstr(os_name_string());
    out_info->arch = dup_cstr(arch_name_string());
    return 0;
}

void llama_server_cluster_free_node_info(struct llama_server_cluster_node_info * info) {
    if (info == nullptr) {
        return;
    }
    free_cstr(info->node_id);
    free_cstr(info->display_name);
    free_cstr(info->os_name);
    free_cstr(info->arch);
    info->node_id = nullptr;
    info->display_name = nullptr;
    info->os_name = nullptr;
    info->arch = nullptr;
}

int32_t llama_server_cluster_list_devices(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_device_info ** out_devices,
    size_t * out_count) {
    if (cluster == nullptr || out_devices == nullptr || out_count == nullptr) {
        set_cluster_error(cluster, "cluster, out_devices, and out_count are required");
        return -1;
    }

    std::string error;
    const std::vector<owned_device_info> devices = query_devices("", &error);
    if (!error.empty()) {
        set_cluster_error(cluster, error);
        return -1;
    }

    auto * raw = static_cast<llama_server_cluster_device_info *>(
        std::calloc(devices.size(), sizeof(llama_server_cluster_device_info)));
    if (raw == nullptr && !devices.empty()) {
        set_cluster_error(cluster, "out of memory");
        return -1;
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        const owned_device_info & src = devices[i];
        raw[i].bridge_device_index = src.bridge_device_index;
        raw[i].type = src.type;
        raw[i].memory_free = src.memory_free;
        raw[i].memory_total = src.memory_total;
        raw[i].backend = dup_cstr(src.backend);
        raw[i].name = dup_cstr(src.name);
        raw[i].description = dup_cstr(src.description);
    }

    *out_devices = raw;
    *out_count = devices.size();
    return 0;
}

void llama_server_cluster_free_devices(
    struct llama_server_cluster_device_info * devices,
    size_t count) {
    if (devices == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free_cstr(devices[i].backend);
        free_cstr(devices[i].name);
        free_cstr(devices[i].description);
    }
    std::free(devices);
}

int32_t llama_server_cluster_list_execution_groups(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_execution_group_info ** out_groups,
    size_t * out_count) {
    if (cluster == nullptr || out_groups == nullptr || out_count == nullptr) {
        set_cluster_error(cluster, "cluster, out_groups, and out_count are required");
        return -1;
    }

    std::string error;
    const std::vector<owned_device_info> devices = query_devices("", &error);
    if (!error.empty()) {
        set_cluster_error(cluster, error);
        return -1;
    }
    const std::vector<owned_execution_group> groups = build_execution_groups(devices, false);

    auto * raw = static_cast<llama_server_cluster_execution_group_info *>(
        std::calloc(groups.size(), sizeof(llama_server_cluster_execution_group_info)));
    if (raw == nullptr && !groups.empty()) {
        set_cluster_error(cluster, "out of memory");
        return -1;
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        const owned_execution_group & src = groups[i];
        raw[i].id = dup_cstr(src.id);
        raw[i].label = dup_cstr(src.label);
        raw[i].backend_summary = dup_cstr(src.backend_summary);
        raw[i].devices_csv = dup_cstr(src.devices_csv);
        raw[i].device_count = src.device_count;
        raw[i].uses_local_split = src.uses_local_split ? 1 : 0;
        raw[i].memory_free = src.memory_free;
        raw[i].memory_total = src.memory_total;
    }

    *out_groups = raw;
    *out_count = groups.size();
    return 0;
}

void llama_server_cluster_free_execution_groups(
    struct llama_server_cluster_execution_group_info * groups,
    size_t count) {
    if (groups == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free_cstr(groups[i].id);
        free_cstr(groups[i].label);
        free_cstr(groups[i].backend_summary);
        free_cstr(groups[i].devices_csv);
    }
    std::free(groups);
}

int32_t llama_server_cluster_list_devices_with_rpc(
    struct llama_server_cluster * cluster,
    const char * rpc_servers,
    struct llama_server_cluster_device_info ** out_devices,
    size_t * out_count) {
    if (cluster == nullptr || out_devices == nullptr || out_count == nullptr) {
        set_cluster_error(cluster, "cluster, out_devices, and out_count are required");
        return -1;
    }

    std::string error;
    const std::vector<owned_device_info> devices = query_devices(rpc_servers != nullptr ? rpc_servers : "", &error);
    if (!error.empty()) {
        set_cluster_error(cluster, error);
        return -1;
    }

    auto * raw = static_cast<llama_server_cluster_device_info *>(
        std::calloc(devices.size(), sizeof(llama_server_cluster_device_info)));
    if (raw == nullptr && !devices.empty()) {
        set_cluster_error(cluster, "out of memory");
        return -1;
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        const owned_device_info & src = devices[i];
        raw[i].bridge_device_index = src.bridge_device_index;
        raw[i].type = src.type;
        raw[i].memory_free = src.memory_free;
        raw[i].memory_total = src.memory_total;
        raw[i].backend = dup_cstr(src.backend);
        raw[i].name = dup_cstr(src.name);
        raw[i].description = dup_cstr(src.description);
    }

    *out_devices = raw;
    *out_count = devices.size();
    return 0;
}

int32_t llama_server_cluster_list_execution_groups_with_rpc(
    struct llama_server_cluster * cluster,
    const char * rpc_servers,
    struct llama_server_cluster_execution_group_info ** out_groups,
    size_t * out_count) {
    if (cluster == nullptr || out_groups == nullptr || out_count == nullptr) {
        set_cluster_error(cluster, "cluster, out_groups, and out_count are required");
        return -1;
    }

    std::string error;
    const std::vector<owned_device_info> devices = query_devices(rpc_servers != nullptr ? rpc_servers : "", &error);
    if (!error.empty()) {
        set_cluster_error(cluster, error);
        return -1;
    }
    const std::vector<owned_execution_group> groups =
        build_execution_groups(devices, rpc_servers != nullptr && rpc_servers[0] != '\0');

    auto * raw = static_cast<llama_server_cluster_execution_group_info *>(
        std::calloc(groups.size(), sizeof(llama_server_cluster_execution_group_info)));
    if (raw == nullptr && !groups.empty()) {
        set_cluster_error(cluster, "out of memory");
        return -1;
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        const owned_execution_group & src = groups[i];
        raw[i].id = dup_cstr(src.id);
        raw[i].label = dup_cstr(src.label);
        raw[i].backend_summary = dup_cstr(src.backend_summary);
        raw[i].devices_csv = dup_cstr(src.devices_csv);
        raw[i].device_count = src.device_count;
        raw[i].uses_local_split = src.uses_local_split ? 1 : 0;
        raw[i].memory_free = src.memory_free;
        raw[i].memory_total = src.memory_total;
    }

    *out_groups = raw;
    *out_count = groups.size();
    return 0;
}

int32_t llama_server_cluster_run_local_rpc_server(
    struct llama_server_cluster * cluster,
    const char * host,
    int32_t port,
    int32_t n_threads) {
    if (cluster == nullptr || host == nullptr || host[0] == '\0') {
        set_cluster_error(cluster, "cluster and host are required");
        return -1;
    }
    if (port <= 0 || port > 65535) {
        set_cluster_error(cluster, "RPC port must be between 1 and 65535");
        return -1;
    }

    std::string error;
    if (!ensure_bridge_backend_registry_ready(&error)) {
        set_cluster_error(cluster, error);
        return -1;
    }

    using ggml_backend_rpc_start_server_t = void (*)(
        const char * endpoint,
        const char * cache_dir,
        size_t n_threads,
        size_t n_devices,
        ggml_backend_dev_t * devices);

    ggml_backend_rpc_start_server_t start_server = nullptr;
    {
        std::lock_guard<std::mutex> lock(rpc_registry_mutex());
        ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
        if (rpc_reg == nullptr) {
            set_cluster_error(cluster, "RPC backend is not available in this runtime");
            return -1;
        }

        start_server = reinterpret_cast<ggml_backend_rpc_start_server_t>(
            ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server"));
    }
    if (start_server == nullptr) {
        set_cluster_error(cluster, "RPC backend does not expose ggml_backend_rpc_start_server");
        return -1;
    }

    std::vector<ggml_backend_dev_t> devices;
    devices.reserve(ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (device != nullptr && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(device);
        }
    }
    if (devices.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            devices.push_back(cpu);
        }
    }
    if (devices.empty()) {
        set_cluster_error(cluster, "no backend devices are available for embedded RPC hosting");
        return -1;
    }

    const std::string endpoint = std::string(host) + ":" + std::to_string(port);
    set_cluster_error(cluster, "");
    start_server(
        endpoint.c_str(),
        nullptr,
        static_cast<size_t>(std::max(n_threads, 0)),
        devices.size(),
        devices.data());
    return 0;
}

int64_t llama_server_cluster_create_instance(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_instance_params * params) {
    if (cluster == nullptr || params == nullptr) {
        set_cluster_error(cluster, "cluster and params are required");
        return -1;
    }

    std::string device_error;
    const std::vector<owned_device_info> devices = query_devices(
        params->rpc_servers != nullptr ? params->rpc_servers : "",
        &device_error);
    if (!device_error.empty()) {
        set_cluster_error(cluster, device_error);
        return -1;
    }
    const std::vector<owned_execution_group> groups = build_execution_groups(
        devices,
        params->rpc_servers != nullptr && params->rpc_servers[0] != '\0');

    std::string normalize_error;
    owned_instance_params normalized = normalize_instance_params(*params, groups, devices, &normalize_error);
    if (!normalize_error.empty()) {
        set_cluster_error(cluster, normalize_error);
        return -1;
    }

    int32_t retention_mode = params->retention_mode;
    if (retention_mode != LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED &&
        retention_mode != LLAMA_SERVER_CLUSTER_INSTANCE_LOAD_ON_DEMAND) {
        retention_mode = LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED;
    }

    std::lock_guard<std::mutex> lock(cluster->mutex);
    const int64_t instance_id = cluster->next_instance_id++;
    cluster->instances.emplace(
        instance_id,
        std::make_shared<model_instance>(instance_id, std::move(normalized), retention_mode));
    cluster->last_error.clear();
    return instance_id;
}

int64_t llama_server_cluster_find_instance_by_name(
    struct llama_server_cluster * cluster,
    const char * name) {
    if (cluster == nullptr) {
        return -1;
    }
    if (name == nullptr || name[0] == '\0') {
        set_cluster_error(cluster, "name is required");
        return -1;
    }

    std::lock_guard<std::mutex> lock(cluster->mutex);
    for (const auto & entry : cluster->instances) {
        const auto & instance = entry.second;
        if (!instance) {
            continue;
        }
        std::lock_guard<std::mutex> instance_lock(instance->mutex);
        if (instance->params.name == name) {
            cluster->last_error.clear();
            return instance->instance_id;
        }
    }

    cluster->last_error = "unknown instance name";
    return -1;
}

int32_t llama_server_cluster_remove_instance(
    struct llama_server_cluster * cluster,
    int64_t instance_id) {
    if (cluster == nullptr) {
        return -1;
    }

    std::shared_ptr<model_instance> instance;
    {
        std::lock_guard<std::mutex> cluster_lock(cluster->mutex);
        const auto it = cluster->instances.find(instance_id);
        if (it == cluster->instances.end()) {
            cluster->last_error = "unknown instance_id";
            return -1;
        }
        instance = it->second;
        std::lock_guard<std::mutex> instance_lock(instance->mutex);
        if (instance->active_request_count != 0) {
            cluster->last_error = "instance is busy";
            return -1;
        }
        instance->bridge.reset();
        instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
        clear_instance_grace_state(*instance);
        cluster->instances.erase(it);
        cluster->last_error.clear();
    }

    return 0;
}

int32_t llama_server_cluster_list_instances(
    struct llama_server_cluster * cluster,
    struct llama_server_cluster_instance_info ** out_instances,
    size_t * out_count) {
    if (cluster == nullptr || out_instances == nullptr || out_count == nullptr) {
        set_cluster_error(cluster, "cluster, out_instances, and out_count are required");
        return -1;
    }

    std::vector<std::shared_ptr<model_instance>> instances;
    {
        std::lock_guard<std::mutex> lock(cluster->mutex);
        for (const auto & pair : cluster->instances) {
            instances.push_back(pair.second);
        }
    }

    std::sort(instances.begin(), instances.end(), [](const auto & lhs, const auto & rhs) {
        return lhs->instance_id < rhs->instance_id;
    });

    auto * raw = static_cast<llama_server_cluster_instance_info *>(
        std::calloc(instances.size(), sizeof(llama_server_cluster_instance_info)));
    if (raw == nullptr && !instances.empty()) {
        set_cluster_error(cluster, "out of memory");
        return -1;
    }

    for (size_t i = 0; i < instances.size(); ++i) {
        const auto & instance = instances[i];
        std::lock_guard<std::mutex> lock(instance->mutex);
        raw[i].instance_id = instance->instance_id;
        raw[i].name = dup_cstr(instance->params.name);
        raw[i].model_path = dup_cstr(instance->params.model_path);
        raw[i].mmproj_path = dup_cstr(instance->params.mmproj_path);
        raw[i].diarization_model_path = dup_cstr(instance->params.diarization_model_path);
        raw[i].execution_group_id = dup_cstr(instance->params.execution_group_id);
        raw[i].rpc_servers = dup_cstr(instance->params.rpc_servers);
        raw[i].retention_mode = instance->retention_mode;
        raw[i].load_on_demand_grace_seconds = instance->params.load_on_demand_grace_seconds;
        raw[i].model_kind = instance->params.model_kind;
        raw[i].state = instance->state;
        raw[i].active_request_count = instance->active_request_count;
        raw[i].queued_request_count = instance->queued_request_count;
        raw[i].n_parallel = std::max<int32_t>(1, instance->params.n_parallel);
        raw[i].grace_deadline_unix_ms = instance->grace_deadline_unix_ms;
        raw[i].last_error = dup_cstr(instance->last_error);
    }

    *out_instances = raw;
    *out_count = instances.size();
    return 0;
}

void llama_server_cluster_free_instances(
    struct llama_server_cluster_instance_info * instances,
    size_t count) {
    if (instances == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free_cstr(instances[i].name);
        free_cstr(instances[i].model_path);
        free_cstr(instances[i].mmproj_path);
        free_cstr(instances[i].diarization_model_path);
        free_cstr(instances[i].execution_group_id);
        free_cstr(instances[i].rpc_servers);
        free_cstr(instances[i].last_error);
    }
    std::free(instances);
}

int32_t llama_server_cluster_set_instance_retention_mode(
    struct llama_server_cluster * cluster,
    int64_t instance_id,
    int32_t retention_mode) {
    if (cluster == nullptr) {
        return -1;
    }
    if (retention_mode != LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED &&
        retention_mode != LLAMA_SERVER_CLUSTER_INSTANCE_LOAD_ON_DEMAND) {
        set_cluster_error(cluster, "invalid retention_mode");
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        return -1;
    }

    std::lock_guard<std::mutex> lock(instance->mutex);
    instance->retention_mode = retention_mode;
    if (retention_mode == LLAMA_SERVER_CLUSTER_INSTANCE_KEEP_LOADED && instance->bridge) {
        clear_instance_grace_state(*instance);
        if (instance->active_request_count == 0) {
            instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_LOADED;
        }
    } else if (retention_mode == LLAMA_SERVER_CLUSTER_INSTANCE_LOAD_ON_DEMAND &&
               instance->active_request_count == 0) {
        if (instance_should_unload_immediately_on_demand(*instance)) {
            instance->bridge.reset();
            instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
            clear_instance_grace_state(*instance);
        } else if (instance->bridge) {
            set_instance_grace_state(*instance);
        } else {
            instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
            clear_instance_grace_state(*instance);
        }
    }
    set_cluster_error(cluster, "");
    return 0;
}

int32_t llama_server_cluster_load_instance(
    struct llama_server_cluster * cluster,
    int64_t instance_id) {
    if (cluster == nullptr) {
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        return -1;
    }

    std::lock_guard<std::mutex> lock(instance->mutex);
    std::string load_error;
    if (!ensure_instance_loaded_locked(*instance, false, &load_error)) {
        set_cluster_error(cluster, load_error);
        return -1;
    }

    set_cluster_error(cluster, "");
    return 0;
}

int32_t llama_server_cluster_unload_instance(
    struct llama_server_cluster * cluster,
    int64_t instance_id) {
    if (cluster == nullptr) {
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        return -1;
    }

    std::lock_guard<std::mutex> lock(instance->mutex);
    if (instance->active_request_count != 0) {
        set_cluster_error(cluster, "instance is busy");
        return -1;
    }
    instance->bridge.reset();
    instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_UNLOADED;
    clear_instance_grace_state(*instance);
    set_cluster_error(cluster, "");
    return 0;
}

int32_t llama_server_cluster_chat_complete(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_chat_request * req,
    struct llama_server_cluster_chat_result * out) {
    if (cluster == nullptr || req == nullptr || out == nullptr) {
        set_cluster_error(cluster, "cluster, req, and out are required");
        return -1;
    }
    *out = llama_server_cluster_empty_chat_result();

    if (req->instance_id <= 0) {
        set_cluster_error(cluster, "instance_id is required");
        out->error = dup_cstr("instance_id is required");
        return -1;
    }
    if (req->prompt == nullptr || req->prompt[0] == '\0') {
        set_cluster_error(cluster, "prompt is required");
        out->error = dup_cstr("prompt is required");
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, req->instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        out->error = dup_cstr("unknown instance_id");
        return -1;
    }

    const auto request_started = steady_clock::now();
    const auto lock_started = steady_clock::now();
    std::unique_lock<std::mutex> lock(instance->mutex);
    out->metrics = make_base_metrics(
        instance->params,
        static_cast<uint64_t>(req->prompt != nullptr ? std::strlen(req->prompt) : 0));
    wait_for_instance_slot_locked(*instance, lock);
    out->metrics.queue_wait_ms = duration_ms(steady_clock::now() - lock_started);
    const bool was_loaded = static_cast<bool>(instance->bridge);
    std::string load_error;
    const auto load_started = steady_clock::now();
    if (!ensure_instance_loaded_locked(*instance, true, &load_error)) {
        out->metrics.loaded_this_call = was_loaded ? 0 : 1;
        out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);
        out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
        set_cluster_error(cluster, load_error);
        out->error = dup_cstr(load_error);
        return -1;
    }
    out->metrics.loaded_this_call = was_loaded ? 0 : 1;
    out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);

    llama_server_bridge_chat_request bridge_req = llama_server_bridge_default_chat_request();
    bridge_req.prompt = req->prompt;
    bridge_req.n_predict = req->n_predict;
    bridge_req.temperature = req->temperature;
    bridge_req.top_p = req->top_p;
    bridge_req.top_k = req->top_k;
    bridge_req.min_p = req->min_p;
    bridge_req.repeat_last_n = req->repeat_last_n;
    bridge_req.repeat_penalty = req->repeat_penalty;
    bridge_req.reasoning = req->reasoning;
    bridge_req.reasoning_budget = req->reasoning_budget;
    bridge_req.reasoning_format = req->reasoning_format;

    llama_server_bridge * bridge_handle = instance->bridge.get();
    lock.unlock();
    llama_server_bridge_vlm_result bridge_out = llama_server_bridge_empty_vlm_result();
    const int32_t rc = llama_server_bridge_chat_complete(
        bridge_handle,
        &bridge_req,
        &bridge_out);
    lock.lock();

    if (bridge_out.text != nullptr) {
        out->text = dup_cstr(bridge_out.text);
    }
    if (bridge_out.error_json != nullptr) {
        out->error = dup_cstr(bridge_out.error_json);
    }
    out->ok = (rc == 0 && bridge_out.ok != 0) ? 1 : 0;
    finalize_text_metrics(
        out->metrics,
        bridge_out.n_prompt_tokens,
        bridge_out.n_decoded,
        bridge_out.prompt_ms,
        bridge_out.predicted_ms,
        duration_ms(steady_clock::now() - request_started));
    llama_server_bridge_result_free(&bridge_out);

    finish_request_locked(*instance);
    lock.unlock();
    instance->cv.notify_all();

    if (rc != 0 || out->ok == 0) {
        if (out->error == nullptr) {
            out->error = dup_cstr("chat request failed");
        }
        instance->last_error = out->error != nullptr ? out->error : "chat request failed";
        set_cluster_error(cluster, instance->last_error);
        return -1;
    }

    instance->last_error.clear();
    set_cluster_error(cluster, "");
    return 0;
}

void llama_server_cluster_chat_result_free(
    struct llama_server_cluster_chat_result * out) {
    if (out == nullptr) {
        return;
    }
    free_cstr(out->text);
    free_cstr(out->error);
    out->text = nullptr;
    out->error = nullptr;
    out->ok = 0;
    out->metrics = {};
}

int32_t llama_server_cluster_vlm_complete(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_vlm_request * req,
    struct llama_server_cluster_vlm_result * out) {
    if (cluster == nullptr || req == nullptr || out == nullptr) {
        set_cluster_error(cluster, "cluster, req, and out are required");
        return -1;
    }
    *out = llama_server_cluster_empty_vlm_result();

    if (req->instance_id <= 0) {
        set_cluster_error(cluster, "instance_id is required");
        out->error = dup_cstr("instance_id is required");
        return -1;
    }
    if (req->image_bytes == nullptr || req->image_bytes_len == 0) {
        set_cluster_error(cluster, "image_bytes are required");
        out->error = dup_cstr("image_bytes are required");
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, req->instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        out->error = dup_cstr("unknown instance_id");
        return -1;
    }

    const auto request_started = steady_clock::now();
    const auto lock_started = steady_clock::now();
    std::unique_lock<std::mutex> lock(instance->mutex);
    out->metrics = make_base_metrics(
        instance->params,
        static_cast<uint64_t>(req->image_bytes_len)
            + static_cast<uint64_t>(req->prompt != nullptr ? std::strlen(req->prompt) : 0));
    wait_for_instance_slot_locked(*instance, lock);
    out->metrics.queue_wait_ms = duration_ms(steady_clock::now() - lock_started);
    const bool was_loaded = static_cast<bool>(instance->bridge);
    std::string load_error;
    const auto load_started = steady_clock::now();
    if (!ensure_instance_loaded_locked(*instance, true, &load_error)) {
        out->metrics.loaded_this_call = was_loaded ? 0 : 1;
        out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);
        out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
        set_cluster_error(cluster, load_error);
        out->error = dup_cstr(load_error);
        return -1;
    }
    out->metrics.loaded_this_call = was_loaded ? 0 : 1;
    out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);

    llama_server_bridge_vlm_request bridge_req = llama_server_bridge_default_vlm_request();
    bridge_req.prompt = req->prompt != nullptr ? req->prompt : "";
    bridge_req.image_bytes = req->image_bytes;
    bridge_req.image_bytes_len = req->image_bytes_len;
    bridge_req.n_predict = req->n_predict;
    bridge_req.temperature = req->temperature;
    bridge_req.top_p = req->top_p;
    bridge_req.top_k = req->top_k;
    bridge_req.min_p = req->min_p;
    bridge_req.repeat_last_n = req->repeat_last_n;
    bridge_req.repeat_penalty = req->repeat_penalty;
    bridge_req.reasoning = req->reasoning;
    bridge_req.reasoning_budget = req->reasoning_budget;
    bridge_req.reasoning_format = req->reasoning_format;

    llama_server_bridge * bridge_handle = instance->bridge.get();
    lock.unlock();
    llama_server_bridge_vlm_result bridge_out = llama_server_bridge_empty_vlm_result();
    const int32_t rc = llama_server_bridge_vlm_complete(
        bridge_handle,
        &bridge_req,
        &bridge_out);
    lock.lock();

    if (bridge_out.text != nullptr) {
        out->text = dup_cstr(bridge_out.text);
    }
    if (bridge_out.error_json != nullptr) {
        out->error = dup_cstr(bridge_out.error_json);
    }
    out->ok = (rc == 0 && bridge_out.ok != 0) ? 1 : 0;
    finalize_text_metrics(
        out->metrics,
        bridge_out.n_prompt_tokens,
        bridge_out.n_decoded,
        bridge_out.prompt_ms,
        bridge_out.predicted_ms,
        duration_ms(steady_clock::now() - request_started));
    llama_server_bridge_result_free(&bridge_out);

    finish_request_locked(*instance);
    lock.unlock();
    instance->cv.notify_all();

    if (rc != 0 || out->ok == 0) {
        if (out->error == nullptr) {
            out->error = dup_cstr("vlm request failed");
        }
        instance->last_error = out->error != nullptr ? out->error : "vlm request failed";
        set_cluster_error(cluster, instance->last_error);
        return -1;
    }

    instance->last_error.clear();
    set_cluster_error(cluster, "");
    return 0;
}

void llama_server_cluster_vlm_result_free(
    struct llama_server_cluster_vlm_result * out) {
    if (out == nullptr) {
        return;
    }
    free_cstr(out->text);
    free_cstr(out->error);
    out->text = nullptr;
    out->error = nullptr;
    out->ok = 0;
    out->metrics = {};
}

int32_t llama_server_cluster_embeddings(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_embeddings_request * req,
    struct llama_server_cluster_json_result * out) {
    if (cluster == nullptr || req == nullptr || out == nullptr) {
        set_cluster_error(cluster, "cluster, req, and out are required");
        return -1;
    }
    *out = llama_server_cluster_empty_json_result();

    if (req->instance_id <= 0) {
        set_cluster_error(cluster, "instance_id is required");
        out->error = dup_cstr("instance_id is required");
        return -1;
    }
    if (req->body_json == nullptr || req->body_json[0] == '\0') {
        set_cluster_error(cluster, "body_json is required");
        out->error = dup_cstr("body_json is required");
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, req->instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        out->error = dup_cstr("unknown instance_id");
        return -1;
    }

    const auto request_started = steady_clock::now();
    const auto lock_started = steady_clock::now();
    std::unique_lock<std::mutex> lock(instance->mutex);
    out->metrics = make_base_metrics(
        instance->params,
        static_cast<uint64_t>(req->body_json != nullptr ? std::strlen(req->body_json) : 0));
    wait_for_instance_slot_locked(*instance, lock);
    out->metrics.queue_wait_ms = duration_ms(steady_clock::now() - lock_started);
    const bool was_loaded = static_cast<bool>(instance->bridge);
    std::string load_error;
    const auto load_started = steady_clock::now();
    if (!ensure_instance_loaded_locked(*instance, true, &load_error)) {
        out->metrics.loaded_this_call = was_loaded ? 0 : 1;
        out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);
        out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
        set_cluster_error(cluster, load_error);
        out->error = dup_cstr(load_error);
        return -1;
    }
    out->metrics.loaded_this_call = was_loaded ? 0 : 1;
    out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);

    llama_server_bridge_embeddings_request bridge_req = llama_server_bridge_default_embeddings_request();
    bridge_req.body_json = req->body_json;
    bridge_req.oai_compat = req->oai_compat;

    llama_server_bridge * bridge_handle = instance->bridge.get();
    lock.unlock();
    llama_server_bridge_json_result bridge_out = llama_server_bridge_empty_json_result();
    const int32_t rc = llama_server_bridge_embeddings(bridge_handle, &bridge_req, &bridge_out);
    lock.lock();

    out->ok = (rc == 0 && bridge_out.ok != 0) ? 1 : 0;
    out->status = bridge_out.status;
    if (bridge_out.json != nullptr) {
        out->json = dup_cstr(bridge_out.json);
    }
    if (bridge_out.error_json != nullptr) {
        out->error = dup_cstr(bridge_out.error_json);
    }
    out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
    llama_server_bridge_json_result_free(&bridge_out);

    finish_request_locked(*instance);
    lock.unlock();
    instance->cv.notify_all();

    if (rc != 0 || out->ok == 0) {
        if (out->error == nullptr) {
            out->error = dup_cstr("embeddings request failed");
        }
        instance->last_error = out->error != nullptr ? out->error : "embeddings request failed";
        set_cluster_error(cluster, instance->last_error);
        return -1;
    }

    instance->last_error.clear();
    set_cluster_error(cluster, "");
    return 0;
}

int32_t llama_server_cluster_rerank(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_rerank_request * req,
    struct llama_server_cluster_json_result * out) {
    if (cluster == nullptr || req == nullptr || out == nullptr) {
        set_cluster_error(cluster, "cluster, req, and out are required");
        return -1;
    }
    *out = llama_server_cluster_empty_json_result();

    if (req->instance_id <= 0) {
        set_cluster_error(cluster, "instance_id is required");
        out->error = dup_cstr("instance_id is required");
        return -1;
    }
    if (req->body_json == nullptr || req->body_json[0] == '\0') {
        set_cluster_error(cluster, "body_json is required");
        out->error = dup_cstr("body_json is required");
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, req->instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        out->error = dup_cstr("unknown instance_id");
        return -1;
    }

    const auto request_started = steady_clock::now();
    const auto lock_started = steady_clock::now();
    std::unique_lock<std::mutex> lock(instance->mutex);
    out->metrics = make_base_metrics(
        instance->params,
        static_cast<uint64_t>(req->body_json != nullptr ? std::strlen(req->body_json) : 0));
    wait_for_instance_slot_locked(*instance, lock);
    out->metrics.queue_wait_ms = duration_ms(steady_clock::now() - lock_started);
    const bool was_loaded = static_cast<bool>(instance->bridge);
    std::string load_error;
    const auto load_started = steady_clock::now();
    if (!ensure_instance_loaded_locked(*instance, true, &load_error)) {
        out->metrics.loaded_this_call = was_loaded ? 0 : 1;
        out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);
        out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
        set_cluster_error(cluster, load_error);
        out->error = dup_cstr(load_error);
        return -1;
    }
    out->metrics.loaded_this_call = was_loaded ? 0 : 1;
    out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);

    llama_server_bridge_rerank_request bridge_req = llama_server_bridge_default_rerank_request();
    bridge_req.body_json = req->body_json;

    llama_server_bridge * bridge_handle = instance->bridge.get();
    lock.unlock();
    llama_server_bridge_json_result bridge_out = llama_server_bridge_empty_json_result();
    const int32_t rc = llama_server_bridge_rerank(bridge_handle, &bridge_req, &bridge_out);
    lock.lock();

    out->ok = (rc == 0 && bridge_out.ok != 0) ? 1 : 0;
    out->status = bridge_out.status;
    if (bridge_out.json != nullptr) {
        out->json = dup_cstr(bridge_out.json);
    }
    if (bridge_out.error_json != nullptr) {
        out->error = dup_cstr(bridge_out.error_json);
    }
    out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
    llama_server_bridge_json_result_free(&bridge_out);

    finish_request_locked(*instance);
    lock.unlock();
    instance->cv.notify_all();

    if (rc != 0 || out->ok == 0) {
        if (out->error == nullptr) {
            out->error = dup_cstr("rerank request failed");
        }
        instance->last_error = out->error != nullptr ? out->error : "rerank request failed";
        set_cluster_error(cluster, instance->last_error);
        return -1;
    }

    instance->last_error.clear();
    set_cluster_error(cluster, "");
    return 0;
}

int32_t llama_server_cluster_audio_transcriptions_raw(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_audio_raw_request * req,
    struct llama_server_cluster_json_result * out) {
    if (cluster == nullptr || req == nullptr || out == nullptr) {
        set_cluster_error(cluster, "cluster, req, and out are required");
        return -1;
    }
    *out = llama_server_cluster_empty_json_result();

    if (req->instance_id <= 0) {
        set_cluster_error(cluster, "instance_id is required");
        out->error = dup_cstr("instance_id is required");
        return -1;
    }
    if (req->audio_bytes == nullptr || req->audio_bytes_len == 0) {
        set_cluster_error(cluster, "audio_bytes are required");
        out->error = dup_cstr("audio_bytes are required");
        return -1;
    }
    if (req->audio_format == nullptr || req->audio_format[0] == '\0') {
        set_cluster_error(cluster, "audio_format is required");
        out->error = dup_cstr("audio_format is required");
        return -1;
    }

    const std::shared_ptr<model_instance> instance = find_instance(cluster, req->instance_id);
    if (!instance) {
        set_cluster_error(cluster, "unknown instance_id");
        out->error = dup_cstr("unknown instance_id");
        return -1;
    }

    const auto request_started = steady_clock::now();
    const auto lock_started = steady_clock::now();
    std::unique_lock<std::mutex> lock(instance->mutex);
    out->metrics = make_base_metrics(instance->params, static_cast<uint64_t>(req->audio_bytes_len));
    wait_for_instance_slot_locked(*instance, lock);
    out->metrics.queue_wait_ms = duration_ms(steady_clock::now() - lock_started);
    const bool use_native_audio_backend = model_uses_native_audio_backend(instance->params.model_path);
    if (use_native_audio_backend) {
        clear_instance_grace_state(*instance);
        instance->active_request_count += 1;
        instance->state = LLAMA_SERVER_CLUSTER_INSTANCE_STATE_SERVING;

        const char * effective_diarization_model_path = req->diarization_model_path;
        if ((effective_diarization_model_path == nullptr || effective_diarization_model_path[0] == '\0')
            && !instance->params.diarization_model_path.empty()) {
            effective_diarization_model_path = instance->params.diarization_model_path.c_str();
        }

        llama_server_cluster_native_audio_transcription_request native_req =
            llama_server_cluster_default_native_audio_transcription_request();
        native_req.model_path = instance->params.model_path.c_str();
        native_req.execution_group_id =
            instance->params.execution_group_id.empty() ? nullptr : instance->params.execution_group_id.c_str();
        native_req.audio_bytes = req->audio_bytes;
        native_req.audio_bytes_len = req->audio_bytes_len;
        native_req.audio_format = req->audio_format;
        native_req.metadata_json = req->metadata_json;
        native_req.ffmpeg_convert = req->ffmpeg_convert;
        native_req.enable_diarization = req->enable_diarization;
        native_req.diarization_model_path = effective_diarization_model_path;

        lock.unlock();
        llama_server_cluster_json_result native_out = llama_server_cluster_empty_json_result();
        const int32_t rc = run_native_audio_transcription(native_req, &native_out);
        lock.lock();
        out->ok = (rc == 0 && native_out.ok != 0) ? 1 : 0;
        out->status = native_out.status;
        if (native_out.json != nullptr) {
            out->json = dup_cstr(native_out.json);
        }
        if (native_out.error != nullptr) {
            out->error = dup_cstr(native_out.error);
        }
        out->metrics = native_out.metrics;
        out->metrics.queue_wait_ms = duration_ms(steady_clock::now() - lock_started);
        out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
        llama_server_cluster_json_result_free(&native_out);

        finish_request_locked(*instance);
        lock.unlock();
        instance->cv.notify_all();

        if (rc != 0 || out->ok == 0) {
            if (out->error == nullptr) {
                out->error = dup_cstr("native audio transcription request failed");
            }
            instance->last_error =
                out->error != nullptr ? out->error : "native audio transcription request failed";
            set_cluster_error(cluster, instance->last_error);
            return -1;
        }

        instance->last_error.clear();
        set_cluster_error(cluster, "");
        return 0;
    }

    if (req->enable_diarization != 0) {
        set_cluster_error(
            cluster,
            "diarization is only supported for native single-device audio backends");
        out->error =
            dup_cstr("diarization is only supported for native single-device audio backends");
        return -1;
    }

    const bool was_loaded = static_cast<bool>(instance->bridge);
    std::string load_error;
    const auto load_started = steady_clock::now();
    if (!ensure_instance_loaded_locked(*instance, true, &load_error)) {
        out->metrics.loaded_this_call = was_loaded ? 0 : 1;
        out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);
        out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
        set_cluster_error(cluster, load_error);
        out->error = dup_cstr(load_error);
        return -1;
    }
    out->metrics.loaded_this_call = was_loaded ? 0 : 1;
    out->metrics.load_ms = duration_ms(steady_clock::now() - load_started);

    llama_server_bridge_audio_raw_request bridge_req = llama_server_bridge_default_audio_raw_request();
    bridge_req.audio_bytes = req->audio_bytes;
    bridge_req.audio_bytes_len = req->audio_bytes_len;
    bridge_req.audio_format = req->audio_format;
    bridge_req.metadata_json = req->metadata_json;
    bridge_req.ffmpeg_convert = req->ffmpeg_convert;

    llama_server_bridge * bridge_handle = instance->bridge.get();
    lock.unlock();
    llama_server_bridge_json_result bridge_out = llama_server_bridge_empty_json_result();
    const int32_t rc =
        llama_server_bridge_audio_transcriptions_raw(bridge_handle, &bridge_req, &bridge_out);
    lock.lock();

    out->ok = (rc == 0 && bridge_out.ok != 0) ? 1 : 0;
    out->status = bridge_out.status;
    if (bridge_out.json != nullptr) {
        out->json = dup_cstr(bridge_out.json);
    }
    if (bridge_out.error_json != nullptr) {
        out->error = dup_cstr(bridge_out.error_json);
    }
    out->metrics.request_total_ms = duration_ms(steady_clock::now() - request_started);
    llama_server_bridge_json_result_free(&bridge_out);

    finish_request_locked(*instance);
    lock.unlock();
    instance->cv.notify_all();

    if (rc != 0 || out->ok == 0) {
        if (out->error == nullptr) {
            out->error = dup_cstr("audio transcription request failed");
        }
        instance->last_error = out->error != nullptr ? out->error : "audio transcription request failed";
        set_cluster_error(cluster, instance->last_error);
        return -1;
    }

    instance->last_error.clear();
    set_cluster_error(cluster, "");
    return 0;
}

int32_t llama_server_cluster_audio_transcriptions_native(
    struct llama_server_cluster * cluster,
    const struct llama_server_cluster_native_audio_transcription_request * req,
    struct llama_server_cluster_json_result * out) {
    if (cluster == nullptr || req == nullptr || out == nullptr) {
        set_cluster_error(cluster, "cluster, req, and out are required");
        return -1;
    }

    const int32_t rc = run_native_audio_transcription(*req, out);
    if (rc != 0 || out->ok == 0) {
        const std::string error = out->error != nullptr ? out->error : "native audio transcription failed";
        set_cluster_error(cluster, error);
        return -1;
    }

    set_cluster_error(cluster, "");
    return 0;
}

void llama_server_cluster_json_result_free(
    struct llama_server_cluster_json_result * out) {
    if (out == nullptr) {
        return;
    }
    free_cstr(out->json);
    free_cstr(out->error);
    out->json = nullptr;
    out->error = nullptr;
    out->ok = 0;
    out->status = 0;
    out->metrics = {};
}

} // extern "C"
