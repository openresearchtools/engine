#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include "pyannote-entrypoints.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#include "miniaudio/miniaudio.h"

using json = nlohmann::json;

struct sliding_window {
    double start = 0.0;
    double duration = 0.0;
    double step = 0.0;

    int closest_frame(double t) const {
        return static_cast<int>(std::llround((t - start - 0.5 * duration) / step));
    }
};

struct diar_seg {
    int speaker = -1;
    double start_sec = 0.0;
    double end_sec = 0.0;
};

struct feature_map {
    int channels = 0;
    int frames = 0;
    std::vector<float> data;

    float & at(int c, int t) {
        return data[static_cast<size_t>(c) * static_cast<size_t>(frames) + static_cast<size_t>(t)];
    }

    const float & at(int c, int t) const {
        return data[static_cast<size_t>(c) * static_cast<size_t>(frames) + static_cast<size_t>(t)];
    }
};

struct feature_map2d {
    int channels = 0;
    int height = 0;
    int width = 0;
    std::vector<float> data;

    float & at(int c, int h, int w) {
        return data[(static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(h)) * static_cast<size_t>(width) + static_cast<size_t>(w)];
    }

    const float & at(int c, int h, int w) const {
        return data[(static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(h)) * static_cast<size_t>(width) + static_cast<size_t>(w)];
    }
};

struct tensor_f32 {
    std::vector<float> data;
    int64_t ne0 = 1;
    int64_t ne1 = 1;
    int64_t ne2 = 1;
    int64_t ne3 = 1;
};

struct batch_norm_2d {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> running_mean;
    std::vector<float> running_var;
    float eps = 1e-5f;
};

struct conv2d_layer {
    tensor_f32 weight; // [kw, kh, in, out]
    int stride = 1;
    int kdim = 0;
    std::vector<float> weight_oc_k; // [out, kdim]
};

struct resnet_block {
    conv2d_layer conv1;
    batch_norm_2d bn1;
    conv2d_layer conv2;
    batch_norm_2d bn2;

    bool has_shortcut = false;
    conv2d_layer shortcut_conv;
    batch_norm_2d shortcut_bn;
};

struct embedding_model {
    int sample_rate = 16000;
    int num_mel_bins = 80;
    int frame_length_samples = 400;
    int frame_shift_samples = 160;
    int fft_size = 512;
    int min_num_samples = 400;
    int embed_dim = 256;
    std::string architecture_class = "unknown";

    std::vector<float> hamming_window;
    std::vector<float> mel_fbanks; // [num_mel_bins, fft_size / 2 + 1]

    conv2d_layer conv1;
    batch_norm_2d bn1;
    std::array<std::vector<resnet_block>, 4> layers; // 3,4,6,3 blocks
    tensor_f32 seg1_w; // [in_dim, embed_dim] in gguf ne0=in_dim ne1=embed_dim
    tensor_f32 seg1_b;
};

struct lstm_direction {
    int input_size = 0;
    int hidden_size = 0;

    std::vector<float> w_ih; // [4H, I]
    std::vector<float> w_hh; // [4H, H]
    std::vector<float> b_ih; // [4H]
    std::vector<float> b_hh; // [4H]
};

struct lstm_layer {
    lstm_direction fwd;
    lstm_direction rev;
};

struct segmentation_model {
    float wav_norm_weight = 1.0f;
    float wav_norm_bias = 0.0f;

    std::vector<float> sinc_low_hz;
    std::vector<float> sinc_band_hz;
    std::vector<float> sinc_window;
    std::vector<float> sinc_n;
    std::vector<float> sinc_filters; // [80, 251] flattened [out, k]

    tensor_f32 conv1_w; // ne0=5, ne1=80, ne2=60
    tensor_f32 conv1_b;
    tensor_f32 conv2_w; // ne0=5, ne1=60, ne2=60
    tensor_f32 conv2_b;

    tensor_f32 norm0_w;
    tensor_f32 norm0_b;
    tensor_f32 norm1_w;
    tensor_f32 norm1_b;
    tensor_f32 norm2_w;
    tensor_f32 norm2_b;

    std::array<lstm_layer, 4> lstm;

    tensor_f32 linear0_w; // [128,256] in GGUF ne0=256 ne1=128
    tensor_f32 linear0_b;
    tensor_f32 linear1_w; // [128,128]
    tensor_f32 linear1_b;
    tensor_f32 cls_w;     // [7,128] ne0=128 ne1=7
    tensor_f32 cls_b;

    int sample_rate = 16000;
    double frame_duration_sec = 991.0 / 16000.0;
    double frame_step_sec = 270.0 / 16000.0;
    double chunk_duration_sec = 10.0;
};

struct matmul_workspace {
    int rows = 0;
    int in_dim = 0;
    int out_dim = 0;
    const float * weight_key_ptr = nullptr;
    size_t weight_key_size = 0;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t_w = nullptr;
    ggml_tensor * t_x = nullptr;
    ggml_tensor * t_y = nullptr;
    ggml_cgraph * gf = nullptr;
    bool weight_uploaded = false;

    const float * cached_w_ptr = nullptr;
    size_t cached_w_size = 0;

    matmul_workspace() = default;
    matmul_workspace(const matmul_workspace &) = delete;
    matmul_workspace & operator=(const matmul_workspace &) = delete;

    ~matmul_workspace() {
        clear();
    }

    void clear() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        t_w = nullptr;
        t_x = nullptr;
        t_y = nullptr;
        gf = nullptr;
        weight_uploaded = false;
        weight_key_ptr = nullptr;
        weight_key_size = 0;
        cached_w_ptr = nullptr;
        cached_w_size = 0;
        rows = 0;
        in_dim = 0;
        out_dim = 0;
    }
};

struct conv1d_workspace {
    int in_frames = 0;
    int in_ch = 0;
    int out_ch = 0;
    int kernel = 0;
    int stride = 1;
    int padding = 0;
    const float * weight_key_ptr = nullptr;
    size_t weight_key_size = 0;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t_w = nullptr;
    ggml_tensor * t_x = nullptr;
    ggml_tensor * t_y = nullptr;
    ggml_cgraph * gf = nullptr;
    bool weight_uploaded = false;

    const float * cached_w_ptr = nullptr;
    size_t cached_w_size = 0;

    conv1d_workspace() = default;
    conv1d_workspace(const conv1d_workspace &) = delete;
    conv1d_workspace & operator=(const conv1d_workspace &) = delete;

    ~conv1d_workspace() {
        clear();
    }

    void clear() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        t_w = nullptr;
        t_x = nullptr;
        t_y = nullptr;
        gf = nullptr;
        weight_uploaded = false;
        cached_w_ptr = nullptr;
        cached_w_size = 0;
        in_frames = 0;
        in_ch = 0;
        out_ch = 0;
        kernel = 0;
        stride = 1;
        padding = 0;
        weight_key_ptr = nullptr;
        weight_key_size = 0;
    }
};

struct conv2d_workspace {
    int in_w = 0;
    int in_h = 0;
    int in_ch = 0;
    int out_ch = 0;
    int kw = 0;
    int kh = 0;
    int stride = 1;
    int padding = 0;
    const float * weight_key_ptr = nullptr;
    size_t weight_key_size = 0;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t_w = nullptr;
    ggml_tensor * t_x = nullptr;
    ggml_tensor * t_y = nullptr;
    ggml_cgraph * gf = nullptr;
    bool weight_uploaded = false;

    conv2d_workspace() = default;
    conv2d_workspace(const conv2d_workspace &) = delete;
    conv2d_workspace & operator=(const conv2d_workspace &) = delete;

    ~conv2d_workspace() {
        clear();
    }

    void clear() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        t_w = nullptr;
        t_x = nullptr;
        t_y = nullptr;
        gf = nullptr;
        weight_uploaded = false;
        in_w = 0;
        in_h = 0;
        in_ch = 0;
        out_ch = 0;
        kw = 0;
        kh = 0;
        stride = 1;
        padding = 0;
        weight_key_ptr = nullptr;
        weight_key_size = 0;
    }
};

struct diarization_runtime {
    ggml_backend_t backend = nullptr;
    bool owns_backend = true;
    std::string requested_device = "auto";
    std::string selected_backend;
    std::string selected_device;
    std::string selected_device_type;
    bool use_backend_matmul = false;
    mutable bool use_backend_conv2d_direct = true;
    mutable std::vector<std::unique_ptr<matmul_workspace>> matmul_workspaces;
    mutable std::vector<std::unique_ptr<conv1d_workspace>> conv1d_workspaces;
    mutable std::vector<std::unique_ptr<conv2d_workspace>> conv2d_workspaces;

    diarization_runtime() = default;
    diarization_runtime(const diarization_runtime &) = delete;
    diarization_runtime & operator=(const diarization_runtime &) = delete;

    ~diarization_runtime() {
        conv2d_workspaces.clear();
        conv1d_workspaces.clear();
        matmul_workspaces.clear();
        if (owns_backend && backend != nullptr) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
};

struct diarization_output {
    std::vector<diar_seg> regular;
    std::vector<diar_seg> exclusive;
    int speaker_count = 0;
    int num_chunks = 0;
    int num_frames_per_chunk = 0;
    double elapsed_build_chunks_sec = 0.0;
    double elapsed_segmentation_infer_sec = 0.0;
    double elapsed_global_mapping_sec = 0.0;
    double elapsed_aggregation_sec = 0.0;
    double elapsed_binarize_segments_sec = 0.0;
    double infer_norm_affine_sec = 0.0;
    double infer_frontend_sec = 0.0;
    double infer_conv_stack_sec = 0.0;
    double infer_seq_pack_sec = 0.0;
    double infer_lstm_sec = 0.0;
    double infer_linear_head_sec = 0.0;
    double infer_decode_sec = 0.0;
    int infer_profiled_chunks = 0;
};

struct segmentation_infer_profile {
    double norm_affine_sec = 0.0;
    double frontend_sec = 0.0;
    double conv_stack_sec = 0.0;
    double seq_pack_sec = 0.0;
    double lstm_sec = 0.0;
    double linear_head_sec = 0.0;
    double decode_sec = 0.0;
    int chunks = 0;
};

static std::vector<float> linear_forward(
    const std::vector<float> & in,
    int rows,
    int in_dim,
    const tensor_f32 & w,
    const tensor_f32 & b,
    const diarization_runtime * rt);

static std::vector<float> matmul_rows_backend(
    const diarization_runtime * rt,
    const std::vector<float> & in,
    int rows,
    int in_dim,
    const std::vector<float> & w,
    int out_dim,
    bool use_workspace_cache = true);

static std::vector<float> conv1d_forward_backend_ggml(
    const feature_map & in,
    const std::vector<float> & w,
    int in_ch,
    int out_ch,
    int kernel,
    int stride,
    int padding,
    const diarization_runtime * rt,
    const float * weight_key_ptr,
    size_t weight_key_size);

static std::vector<float> conv2d_forward_backend_direct(
    const feature_map2d & in,
    const conv2d_layer & conv,
    int padding,
    const diarization_runtime * rt);

static std::string to_lower_copy(std::string s) {
    for (char & ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

static bool str_contains_ci(const std::string & haystack, const std::string & needle) {
    if (needle.empty()) {
        return true;
    }
    const std::string h = to_lower_copy(haystack);
    const std::string n = to_lower_copy(needle);
    return h.find(n) != std::string::npos;
}

static bool env_flag_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr) {
        return false;
    }
    const std::string s = to_lower_copy(std::string(v));
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

static bool allow_vulkan_native_conv1d() {
    static const bool force_off = env_flag_enabled("PYANNOTE_VULKAN_NATIVE_CONV1D_OFF");
    static const bool force_on = env_flag_enabled("PYANNOTE_VULKAN_NATIVE_CONV1D");
    if (force_off) {
        return false;
    }
    if (force_on) {
        return true;
    }
    return true; // default on: improves Vulkan segmentation conv stack latency.
}

static bool is_gpuish_type(enum ggml_backend_dev_type t) {
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static std::string backend_dev_type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:return "accel";
        default:                            return "unknown";
    }
}

struct backend_cache_entry {
    ggml_backend_t backend = nullptr;
    std::string selected_backend;
    std::string selected_device;
    std::string selected_device_type;
    bool use_backend_matmul = false;
};

struct backend_cache_state {
    std::mutex mu;
    std::unordered_map<std::string, backend_cache_entry> entries;
};

static backend_cache_state & get_backend_cache_state() {
    // Intentionally leaked to avoid shutdown-order issues with backend globals.
    static backend_cache_state * state = new backend_cache_state();
    return *state;
}

static ggml_backend_dev_t find_backend_device(
    const std::string & backend_substr,
    bool allow_gpu,
    bool allow_igpu) {
    ggml_backend_dev_t igpu_match = nullptr;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU && !allow_gpu) {
            continue;
        }
        if (t == GGML_BACKEND_DEVICE_TYPE_IGPU && !allow_igpu) {
            continue;
        }
        if (!is_gpuish_type(t) && t != GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const std::string reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : "";
        const std::string dev_name = ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "";
        if (!str_contains_ci(reg_name, backend_substr) && !str_contains_ci(dev_name, backend_substr)) {
            continue;
        }

        if (t == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return dev;
        }
        if (t == GGML_BACKEND_DEVICE_TYPE_IGPU && igpu_match == nullptr) {
            igpu_match = dev;
        }
    }
    return igpu_match;
}

static ggml_backend_dev_t find_first_gpu_device() {
    ggml_backend_dev_t igpu_match = nullptr;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return dev;
        }
        if (t == GGML_BACKEND_DEVICE_TYPE_IGPU && igpu_match == nullptr) {
            igpu_match = dev;
        }
    }
    return igpu_match;
}

static void print_available_devices(std::ostream & os) {
    ggml_backend_load_all();

    const size_t n = ggml_backend_dev_count();
    os << "Available ggml backend devices (" << n << "):\n";
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : "unknown";
        const char * dev_name = ggml_backend_dev_name(dev);
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        size_t mem_free = 0;
        size_t mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);
        os << "  [" << i << "] "
           << (dev_name ? dev_name : "unknown")
           << "  backend=" << reg_name
           << "  type=" << backend_dev_type_name(t)
           << "  mem_total_mb=" << (mem_total / (1024ull * 1024ull))
           << "  mem_free_mb=" << (mem_free / (1024ull * 1024ull))
           << "\n";
    }
}

static void init_runtime_backend(const std::string & requested_device_raw, diarization_runtime & rt) {
    ggml_backend_load_all();

    const std::string requested = requested_device_raw.empty() ? "auto" : requested_device_raw;
    const std::string req = to_lower_copy(requested);
    ggml_backend_dev_t chosen = nullptr;

    if (req == "auto") {
        // strict preference order: Vulkan -> CUDA -> any GPU/IGPU.
        chosen = find_backend_device("vulkan", true, true);
        if (chosen == nullptr) {
            chosen = find_backend_device("cuda", true, true);
        }
        if (chosen == nullptr) {
            chosen = find_first_gpu_device();
        }
        if (chosen == nullptr) {
            throw std::runtime_error("no GPU backend found for --device auto (expected Vulkan or CUDA)");
        }
    } else if (req == "vulkan") {
        chosen = find_backend_device("vulkan", true, true);
        if (chosen == nullptr) {
            throw std::runtime_error("requested --device vulkan but no Vulkan backend device was found");
        }
    } else if (req == "cuda") {
        chosen = find_backend_device("cuda", true, true);
        if (chosen == nullptr) {
            throw std::runtime_error("requested --device cuda but no CUDA backend device was found");
        }
    } else if (req == "gpu") {
        chosen = find_first_gpu_device();
        if (chosen == nullptr) {
            throw std::runtime_error("requested --device gpu but no GPU backend device was found");
        }
    } else if (req == "igpu") {
        const size_t n = ggml_backend_dev_count();
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                chosen = dev;
                break;
            }
        }
        if (chosen == nullptr) {
            throw std::runtime_error("requested --device igpu but no IGPU backend device was found");
        }
    } else if (req == "cpu") {
        chosen = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (chosen == nullptr) {
            throw std::runtime_error("requested --device cpu but CPU backend device was not found");
        }
    } else {
        chosen = ggml_backend_dev_by_name(requested.c_str());
        if (chosen == nullptr) {
            throw std::runtime_error("requested --device '" + requested + "' was not found in ggml backend devices");
        }
    }

    const std::string selected_device = ggml_backend_dev_name(chosen) ? ggml_backend_dev_name(chosen) : "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(chosen);
    const std::string selected_backend = reg != nullptr ? ggml_backend_reg_name(reg) : "";
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(chosen);
    const std::string selected_device_type = backend_dev_type_name(t);
    const bool use_backend_matmul = is_gpuish_type(t);

    const std::string cache_key =
        to_lower_copy(selected_backend) + "|" +
        to_lower_copy(selected_device) + "|" +
        to_lower_copy(selected_device_type);

    auto & cache = get_backend_cache_state();
    {
        std::lock_guard<std::mutex> lock(cache.mu);
        auto it = cache.entries.find(cache_key);
        if (it == cache.entries.end()) {
            ggml_backend_t backend = ggml_backend_dev_init(chosen, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error("failed to initialize ggml backend device");
            }

            backend_cache_entry entry;
            entry.backend = backend;
            entry.selected_backend = selected_backend;
            entry.selected_device = selected_device;
            entry.selected_device_type = selected_device_type;
            entry.use_backend_matmul = use_backend_matmul;
            it = cache.entries.emplace(cache_key, std::move(entry)).first;
        }

        rt.backend = it->second.backend;
        rt.owns_backend = false;
        rt.requested_device = requested;
        rt.selected_backend = it->second.selected_backend;
        rt.selected_device = it->second.selected_device;
        rt.selected_device_type = it->second.selected_device_type;
        rt.use_backend_matmul = it->second.use_backend_matmul;
    }
}

static void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --audio <path> --segmentation-gguf <path> --output-dir <dir>\n"
        << "       [--num-speakers <int>] [--min-speakers <int>] [--max-speakers <int>]\n"
        << "       [--export-speaker-embeddings] [--embedding-gguf <path>] [--embedding-min-segment-duration-sec <float>]\n"
        << "       [--embedding-max-segments-per-speaker <int>]\n"
        << "       [--pipeline-dir <path>] [--device <auto|vulkan|cuda|gpu|igpu|cpu|<device-name>>]\n"
        << "       [--offline] [--progress] [--list-devices]\n";
}

static void write_text_file(const std::filesystem::path & path, const std::string & text) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    ofs << text;
}

static double round3(double x) {
    return std::round(x * 1000.0) / 1000.0;
}

static double round8(double x) {
    return std::round(x * 100000000.0) / 100000000.0;
}

static std::string fmt3(double x) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << x;
    return oss.str();
}

static std::string speaker_name(int idx) {
    std::ostringstream oss;
    oss << "SPEAKER_" << std::setfill('0') << std::setw(2) << std::max(0, idx);
    return oss.str();
}

static std::string now_timestamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[64] = {0};
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
    return std::string(buf);
}

static void write_segments_txt(const std::vector<diar_seg> & segs, const std::filesystem::path & out_path) {
    std::ostringstream oss;
    for (const auto & seg : segs) {
        oss << speaker_name(seg.speaker) << "\t" << fmt3(round3(seg.start_sec)) << "\t" << fmt3(round3(seg.end_sec)) << "\n";
    }
    write_text_file(out_path, oss.str());
}

static void write_segments_rttm(const std::vector<diar_seg> & segs, const std::filesystem::path & out_path, const std::string & uri) {
    std::ostringstream oss;
    for (const auto & seg : segs) {
        const double start = std::max(0.0, seg.start_sec);
        const double dur = std::max(0.0, seg.end_sec - seg.start_sec);
        oss << "SPEAKER " << uri
            << " 1 " << fmt3(round3(start))
            << " " << fmt3(round3(dur))
            << " <NA> <NA> " << speaker_name(seg.speaker) << " <NA> <NA>\n";
    }
    write_text_file(out_path, oss.str());
}

static std::vector<float> load_audio_mono_16k(const std::filesystem::path & audio_path) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, 16000);

    ma_decoder decoder;
    if (ma_decoder_init_file(audio_path.string().c_str(), &cfg, &decoder) != MA_SUCCESS) {
        throw std::runtime_error("failed to decode audio file with miniaudio: " + audio_path.string());
    }

    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS || frame_count == 0) {
        ma_decoder_uninit(&decoder);
        throw std::runtime_error("failed to read audio frame count: " + audio_path.string());
    }

    std::vector<float> pcm(static_cast<size_t>(frame_count));
    ma_uint64 frames_read = 0;
    if (ma_decoder_read_pcm_frames(&decoder, pcm.data(), frame_count, &frames_read) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        throw std::runtime_error("failed to read audio frames: " + audio_path.string());
    }
    ma_decoder_uninit(&decoder);

    pcm.resize(static_cast<size_t>(frames_read));
    return pcm;
}

static tensor_f32 load_tensor_as_f32(
    ggml_context * tctx,
    const std::string & tensor_name,
    const std::filesystem::path & src_path) {
    const ggml_tensor * t = ggml_get_tensor(tctx, tensor_name.c_str());
    if (t == nullptr) {
        throw std::runtime_error("tensor not found: " + tensor_name + " in " + src_path.string());
    }

    tensor_f32 out;
    out.ne0 = t->ne[0];
    out.ne1 = t->ne[1];
    out.ne2 = t->ne[2];
    out.ne3 = t->ne[3];

    const size_t n = ggml_nelements(t);
    out.data.resize(n);

    if (t->type == GGML_TYPE_F32) {
        const float * src = static_cast<const float *>(t->data);
        std::copy(src, src + n, out.data.begin());
        return out;
    }

    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = static_cast<const ggml_fp16_t *>(t->data);
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = ggml_fp16_to_fp32(src[i]);
        }
        return out;
    }

    throw std::runtime_error("unsupported tensor type for " + tensor_name + " in " + src_path.string());
}

static float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static void instance_norm_1d(feature_map & x, const tensor_f32 & w, const tensor_f32 & b, float eps = 1e-5f) {
    if (x.channels <= 0 || x.frames <= 0) {
        return;
    }
    if (static_cast<size_t>(x.channels) != w.data.size() || static_cast<size_t>(x.channels) != b.data.size()) {
        throw std::runtime_error("instance_norm_1d: channel mismatch");
    }

    const float inv_frames = 1.0f / static_cast<float>(x.frames);
    for (int c = 0; c < x.channels; ++c) {
        double mean = 0.0;
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(x.frames);
        for (int t = 0; t < x.frames; ++t) {
            mean += static_cast<double>(x.data[base + static_cast<size_t>(t)]);
        }
        mean *= static_cast<double>(inv_frames);

        double var = 0.0;
        for (int t = 0; t < x.frames; ++t) {
            const double d = static_cast<double>(x.data[base + static_cast<size_t>(t)]) - mean;
            var += d * d;
        }
        var *= static_cast<double>(inv_frames);

        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        const float gamma = w.data[static_cast<size_t>(c)];
        const float beta = b.data[static_cast<size_t>(c)];
        for (int t = 0; t < x.frames; ++t) {
            const size_t idx = base + static_cast<size_t>(t);
            const float normed = (x.data[idx] - static_cast<float>(mean)) * inv_std;
            x.data[idx] = normed * gamma + beta;
        }
    }
}

static void leaky_relu_inplace(feature_map & x, float slope = 0.01f) {
    for (float & v : x.data) {
        if (v < 0.0f) {
            v *= slope;
        }
    }
}

static feature_map maxpool1d(const feature_map & in, int kernel = 3, int stride = 3) {
    if (in.frames < kernel) {
        throw std::runtime_error("maxpool1d: input too short");
    }

    const int out_frames = (in.frames - kernel) / stride + 1;
    feature_map out;
    out.channels = in.channels;
    out.frames = out_frames;
    out.data.assign(static_cast<size_t>(out.channels) * static_cast<size_t>(out.frames), -std::numeric_limits<float>::infinity());

    for (int c = 0; c < in.channels; ++c) {
        for (int t = 0; t < out_frames; ++t) {
            float best = -std::numeric_limits<float>::infinity();
            const int t0 = t * stride;
            for (int k = 0; k < kernel; ++k) {
                const float v = in.at(c, t0 + k);
                if (v > best) {
                    best = v;
                }
            }
            out.at(c, t) = best;
        }
    }

    return out;
}

struct conv1d_pack_cache_entry {
    const tensor_f32 * w_ptr = nullptr;
    int in_ch = 0;
    int out_ch = 0;
    int kernel = 0;
    std::vector<float> packed_oc_k;
};

static const std::vector<float> & get_or_pack_conv1d_weights(
    const tensor_f32 & w,
    int in_ch,
    int out_ch,
    int kernel) {
    static thread_local std::vector<conv1d_pack_cache_entry> cache;

    for (const auto & entry : cache) {
        if (entry.w_ptr == &w &&
            entry.in_ch == in_ch &&
            entry.out_ch == out_ch &&
            entry.kernel == kernel) {
            return entry.packed_oc_k;
        }
    }

    conv1d_pack_cache_entry entry;
    entry.w_ptr = &w;
    entry.in_ch = in_ch;
    entry.out_ch = out_ch;
    entry.kernel = kernel;
    entry.packed_oc_k.assign(static_cast<size_t>(out_ch) * static_cast<size_t>(in_ch * kernel), 0.0f);

    const int w_kernel = static_cast<int>(w.ne0);
    const int w_in_ch = static_cast<int>(w.ne1);
    for (int oc = 0; oc < out_ch; ++oc) {
        float * dst = &entry.packed_oc_k[static_cast<size_t>(oc) * static_cast<size_t>(in_ch * kernel)];
        int kd = 0;
        for (int ic = 0; ic < in_ch; ++ic) {
            for (int k = 0; k < kernel; ++k) {
                const size_t widx = static_cast<size_t>(k) + static_cast<size_t>(w_kernel) * (
                    static_cast<size_t>(ic) + static_cast<size_t>(w_in_ch) * static_cast<size_t>(oc));
                dst[kd++] = w.data[widx];
            }
        }
    }

    cache.push_back(std::move(entry));
    return cache.back().packed_oc_k;
}

static feature_map conv1d_valid(
    const feature_map & in,
    const tensor_f32 & w,
    const tensor_f32 & b,
    int kernel,
    int stride,
    const diarization_runtime * rt) {
    const int in_ch = in.channels;
    const int out_ch = static_cast<int>(w.ne2);
    const int w_in_ch = static_cast<int>(w.ne1);
    const int w_kernel = static_cast<int>(w.ne0);

    if (w_in_ch != in_ch || w_kernel != kernel) {
        throw std::runtime_error("conv1d_valid: tensor dimensions mismatch");
    }
    if (static_cast<size_t>(out_ch) != b.data.size()) {
        throw std::runtime_error("conv1d_valid: bias dimension mismatch");
    }
    if (in.frames < kernel) {
        throw std::runtime_error("conv1d_valid: input too short");
    }

    const int out_frames = (in.frames - kernel) / stride + 1;

    feature_map out;
    out.channels = out_ch;
    out.frames = out_frames;
    out.data.assign(static_cast<size_t>(out_ch) * static_cast<size_t>(out_frames), 0.0f);

    const bool use_backend_conv1d =
        rt != nullptr &&
        rt->use_backend_matmul &&
        rt->backend != nullptr &&
        stride > 0;

    const bool use_native_conv1d_backend =
        use_backend_conv1d &&
        rt != nullptr &&
        (str_contains_ci(rt->selected_backend, "cuda") ||
         (allow_vulkan_native_conv1d() && str_contains_ci(rt->selected_backend, "vulkan")));

    if (use_native_conv1d_backend) {
        std::vector<float> y = conv1d_forward_backend_ggml(
            in,
            w.data,
            in_ch,
            out_ch,
            kernel,
            stride,
            0,
            rt,
            w.data.data(),
            w.data.size());
        if (y.size() != out.data.size()) {
            throw std::runtime_error("conv1d_valid: backend output shape mismatch");
        }

        for (int oc = 0; oc < out_ch; ++oc) {
            const float bias = b.data[static_cast<size_t>(oc)];
            const float * src = &y[static_cast<size_t>(oc) * static_cast<size_t>(out_frames)];
            for (int t = 0; t < out_frames; ++t) {
                out.at(oc, t) = src[t] + bias;
            }
        }
        return out;
    }

    if (use_backend_conv1d) {
        const int rows = out_frames;
        const int in_dim = in_ch * kernel;

        std::vector<float> xcol(static_cast<size_t>(rows) * static_cast<size_t>(in_dim), 0.0f);
        for (int t = 0; t < out_frames; ++t) {
            const int t0 = t * stride;
            float * dst = &xcol[static_cast<size_t>(t) * static_cast<size_t>(in_dim)];
            int kd = 0;
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel; ++k) {
                    dst[kd++] = in.at(ic, t0 + k);
                }
            }
        }

        const std::vector<float> & w_oc_k = get_or_pack_conv1d_weights(w, in_ch, out_ch, kernel);
        std::vector<float> y = matmul_rows_backend(rt, xcol, rows, in_dim, w_oc_k, out_ch);
        if (y.size() != out.data.size()) {
            throw std::runtime_error("conv1d_valid: backend output shape mismatch");
        }

        for (int t = 0; t < out_frames; ++t) {
            const float * src = &y[static_cast<size_t>(t) * static_cast<size_t>(out_ch)];
            for (int oc = 0; oc < out_ch; ++oc) {
                out.at(oc, t) = src[oc] + b.data[static_cast<size_t>(oc)];
            }
        }
        return out;
    }

    for (int oc = 0; oc < out_ch; ++oc) {
        const float bias = b.data[static_cast<size_t>(oc)];
        for (int t = 0; t < out_frames; ++t) {
            const int t0 = t * stride;
            double acc = static_cast<double>(bias);
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel; ++k) {
                    const size_t widx = static_cast<size_t>(k) + static_cast<size_t>(w_kernel) * (
                        static_cast<size_t>(ic) + static_cast<size_t>(w_in_ch) * static_cast<size_t>(oc));
                    acc += static_cast<double>(w.data[widx]) * static_cast<double>(in.at(ic, t0 + k));
                }
            }
            out.at(oc, t) = static_cast<float>(acc);
        }
    }

    return out;
}

static feature_map conv1d_sinc_stride10_window(
    const std::vector<float> & waveform,
    int frame_start,
    int frame_count,
    const segmentation_model & model,
    const diarization_runtime * rt) {
    const int kernel = 251;
    const int stride = 10;
    const int out_ch = 80;

    if (frame_start < 0 || frame_count <= 0) {
        throw std::runtime_error("conv1d_sinc_stride10_window: invalid frame range");
    }

    const int64_t first_sample = static_cast<int64_t>(frame_start) * static_cast<int64_t>(stride);
    const int64_t last_needed = first_sample + static_cast<int64_t>(frame_count - 1) * static_cast<int64_t>(stride) + static_cast<int64_t>(kernel);
    if (first_sample < 0 || last_needed > static_cast<int64_t>(waveform.size())) {
        throw std::runtime_error("conv1d_sinc_stride10_window: waveform range out of bounds");
    }

    feature_map out;
    out.channels = out_ch;
    out.frames = frame_count;
    out.data.assign(static_cast<size_t>(out_ch) * static_cast<size_t>(frame_count), 0.0f);

    const bool use_backend_sinc =
        rt != nullptr &&
        rt->use_backend_matmul &&
        rt->backend != nullptr &&
        static_cast<int>(model.sinc_filters.size()) == out_ch * kernel;

    const bool use_native_conv1d_backend =
        use_backend_sinc &&
        rt != nullptr &&
        (str_contains_ci(rt->selected_backend, "cuda") ||
         (allow_vulkan_native_conv1d() && str_contains_ci(rt->selected_backend, "vulkan")));

    if (use_native_conv1d_backend) {
        const int in_frames = (frame_count - 1) * stride + kernel;
        feature_map x;
        x.channels = 1;
        x.frames = in_frames;
        x.data.assign(static_cast<size_t>(in_frames), 0.0f);
        std::copy(
            waveform.begin() + static_cast<ptrdiff_t>(first_sample),
            waveform.begin() + static_cast<ptrdiff_t>(first_sample + in_frames),
            x.data.begin());

        std::vector<float> y = conv1d_forward_backend_ggml(
            x,
            model.sinc_filters,
            1,
            out_ch,
            kernel,
            stride,
            0,
            rt,
            model.sinc_filters.data(),
            model.sinc_filters.size());
        if (y.size() != out.data.size()) {
            throw std::runtime_error("conv1d_sinc_stride10: backend output shape mismatch");
        }
        for (int oc = 0; oc < out_ch; ++oc) {
            const float * src = &y[static_cast<size_t>(oc) * static_cast<size_t>(frame_count)];
            for (int t = 0; t < frame_count; ++t) {
                out.at(oc, t) = src[t];
            }
        }
        return out;
    }

    if (use_backend_sinc) {
        const int rows = frame_count;
        const int in_dim = kernel;

        std::vector<float> xcol(static_cast<size_t>(rows) * static_cast<size_t>(in_dim), 0.0f);
        for (int t = 0; t < frame_count; ++t) {
            const int64_t t0 = first_sample + static_cast<int64_t>(t) * static_cast<int64_t>(stride);
            float * dst = &xcol[static_cast<size_t>(t) * static_cast<size_t>(in_dim)];
            for (int k = 0; k < kernel; ++k) {
                dst[k] = waveform[static_cast<size_t>(t0 + static_cast<int64_t>(k))];
            }
        }

        std::vector<float> y = matmul_rows_backend(
            rt,
            xcol,
            rows,
            in_dim,
            model.sinc_filters,
            out_ch);
        if (y.size() != out.data.size()) {
            throw std::runtime_error("conv1d_sinc_stride10: backend output shape mismatch");
        }
        for (int t = 0; t < frame_count; ++t) {
            const float * src = &y[static_cast<size_t>(t) * static_cast<size_t>(out_ch)];
            for (int oc = 0; oc < out_ch; ++oc) {
                out.at(oc, t) = src[oc];
            }
        }
        return out;
    }

    for (int oc = 0; oc < out_ch; ++oc) {
        const size_t fbase = static_cast<size_t>(oc) * static_cast<size_t>(kernel);
        for (int t = 0; t < frame_count; ++t) {
            const int64_t t0 = first_sample + static_cast<int64_t>(t) * static_cast<int64_t>(stride);
            double acc = 0.0;
            for (int k = 0; k < kernel; ++k) {
                acc += static_cast<double>(model.sinc_filters[fbase + static_cast<size_t>(k)])
                    * static_cast<double>(waveform[static_cast<size_t>(t0 + static_cast<int64_t>(k))]);
            }
            out.at(oc, t) = static_cast<float>(acc);
        }
    }

    return out;
}

static feature_map conv1d_sinc_stride10(
    const std::vector<float> & waveform_normed,
    const segmentation_model & model,
    const diarization_runtime * rt) {
    const int kernel = 251;
    const int stride = 10;
    const int n = static_cast<int>(waveform_normed.size());
    if (n < kernel) {
        throw std::runtime_error("conv1d_sinc_stride10: input too short");
    }
    const int out_frames = (n - kernel) / stride + 1;
    return conv1d_sinc_stride10_window(waveform_normed, 0, out_frames, model, rt);
}

static bool gguf_has_tensor(ggml_context * tctx, const std::string & name) {
    return ggml_get_tensor(tctx, name.c_str()) != nullptr;
}

static int next_power_of_two(int x) {
    if (x <= 1) {
        return 1;
    }
    int p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

static float hz_to_mel(float hz) {
    return 1127.0f * std::log(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
}

static std::vector<float> build_hamming_window(int size) {
    if (size <= 0) {
        return {};
    }
    std::vector<float> w(static_cast<size_t>(size), 0.0f);
    if (size == 1) {
        w[0] = 1.0f;
        return w;
    }
    constexpr double kPi = 3.14159265358979323846;
    const double denom = static_cast<double>(size - 1);
    for (int i = 0; i < size; ++i) {
        const double phase = 2.0 * kPi * static_cast<double>(i) / denom;
        w[static_cast<size_t>(i)] = static_cast<float>(0.54 - 0.46 * std::cos(phase));
    }
    return w;
}

static std::vector<float> build_kaldi_mel_fbanks(
    int sample_rate,
    int fft_size,
    int num_mel_bins,
    float low_freq_hz = 20.0f,
    float high_freq_hz = 0.0f) {
    if (sample_rate <= 0 || fft_size <= 0 || num_mel_bins <= 0) {
        throw std::runtime_error("invalid fbank configuration");
    }

    const float nyquist = 0.5f * static_cast<float>(sample_rate);
    const float high = high_freq_hz > 0.0f ? high_freq_hz : (nyquist + high_freq_hz);
    if (low_freq_hz < 0.0f || high <= low_freq_hz || high > nyquist + 1e-4f) {
        throw std::runtime_error("invalid mel frequency range");
    }

    const int fft_bins = fft_size / 2 + 1;
    std::vector<float> fbanks(static_cast<size_t>(num_mel_bins) * static_cast<size_t>(fft_bins), 0.0f);

    const float mel_low = hz_to_mel(low_freq_hz);
    const float mel_high = hz_to_mel(high);
    const float mel_step = (mel_high - mel_low) / static_cast<float>(num_mel_bins + 1);

    std::vector<float> center_hz(static_cast<size_t>(num_mel_bins + 2), 0.0f);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        center_hz[static_cast<size_t>(i)] = mel_to_hz(mel_low + mel_step * static_cast<float>(i));
    }

    for (int m = 0; m < num_mel_bins; ++m) {
        const float left = center_hz[static_cast<size_t>(m)];
        const float center = center_hz[static_cast<size_t>(m + 1)];
        const float right = center_hz[static_cast<size_t>(m + 2)];

        for (int k = 0; k < fft_bins; ++k) {
            const float hz = static_cast<float>(k) * static_cast<float>(sample_rate) / static_cast<float>(fft_size);
            float w = 0.0f;
            if (hz > left && hz <= center) {
                w = (hz - left) / std::max(1e-20f, center - left);
            } else if (hz > center && hz < right) {
                w = (right - hz) / std::max(1e-20f, right - center);
            }
            fbanks[static_cast<size_t>(m) * static_cast<size_t>(fft_bins) + static_cast<size_t>(k)] = std::max(0.0f, w);
        }
    }

    return fbanks;
}

struct fft_twiddle_cache_entry {
    int n = 0;
    std::vector<std::complex<float>> wlen_by_stage;
};

static const std::vector<std::complex<float>> & get_fft_stage_wlen(int n) {
    static thread_local std::vector<fft_twiddle_cache_entry> cache;
    for (const auto & entry : cache) {
        if (entry.n == n) {
            return entry.wlen_by_stage;
        }
    }

    if (n <= 1 || (n & (n - 1)) != 0) {
        throw std::runtime_error("get_fft_stage_wlen: size must be power of two");
    }

    fft_twiddle_cache_entry entry;
    entry.n = n;
    int stages = 0;
    for (int len = 2; len <= n; len <<= 1) {
        stages++;
    }
    entry.wlen_by_stage.assign(static_cast<size_t>(stages), std::complex<float>(0.0f, 0.0f));

    constexpr float kPi = 3.14159265358979323846f;
    int stage = 0;
    for (int len = 2; len <= n; len <<= 1, ++stage) {
        const float ang = -2.0f * kPi / static_cast<float>(len);
        entry.wlen_by_stage[static_cast<size_t>(stage)] = std::complex<float>(std::cos(ang), std::sin(ang));
    }

    cache.push_back(std::move(entry));
    return cache.back().wlen_by_stage;
}

static void fft_inplace(std::vector<std::complex<float>> & a) {
    const int n = static_cast<int>(a.size());
    if (n <= 1) {
        return;
    }
    if ((n & (n - 1)) != 0) {
        throw std::runtime_error("fft_inplace: size must be power of two");
    }

    const auto & wlen_by_stage = get_fft_stage_wlen(n);

    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[static_cast<size_t>(i)], a[static_cast<size_t>(j)]);
        }
    }

    int stage = 0;
    for (int len = 2; len <= n; len <<= 1, ++stage) {
        const std::complex<float> wlen = wlen_by_stage[static_cast<size_t>(stage)];
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len / 2; ++k) {
                const std::complex<float> u = a[static_cast<size_t>(i + k)];
                const std::complex<float> v = a[static_cast<size_t>(i + k + len / 2)] * w;
                a[static_cast<size_t>(i + k)] = u + v;
                a[static_cast<size_t>(i + k + len / 2)] = u - v;
                w *= wlen;
            }
        }
    }
}

static std::vector<float> compute_fbank_features_core(
    const float * waveform,
    int n_samples,
    const embedding_model & model,
    int & out_num_frames,
    const diarization_runtime * rt) {
    if (model.sample_rate <= 0 || model.frame_length_samples <= 0 || model.frame_shift_samples <= 0) {
        throw std::runtime_error("compute_fbank_features: invalid model frontend parameters");
    }
    if (waveform == nullptr || n_samples < model.frame_length_samples) {
        throw std::runtime_error("compute_fbank_features: input chunk too short for fbank window");
    }
    if (model.hamming_window.size() != static_cast<size_t>(model.frame_length_samples)) {
        throw std::runtime_error("compute_fbank_features: invalid hamming window size");
    }

    const int num_frames = 1 + (n_samples - model.frame_length_samples) / model.frame_shift_samples;
    if (num_frames <= 0) {
        throw std::runtime_error("compute_fbank_features: no frames generated");
    }

    const int fft_bins = model.fft_size / 2 + 1;
    if (model.mel_fbanks.size() != static_cast<size_t>(model.num_mel_bins) * static_cast<size_t>(fft_bins)) {
        throw std::runtime_error("compute_fbank_features: invalid mel filterbank size");
    }

    std::vector<float> out(static_cast<size_t>(num_frames) * static_cast<size_t>(model.num_mel_bins), 0.0f);
    std::vector<float> frame(static_cast<size_t>(model.frame_length_samples), 0.0f);
    std::vector<std::complex<float>> fft_buf(static_cast<size_t>(model.fft_size), std::complex<float>(0.0f, 0.0f));
    std::vector<float> power(static_cast<size_t>(fft_bins), 0.0f);
    const bool use_backend_mel =
        rt != nullptr &&
        rt->use_backend_matmul &&
        rt->backend != nullptr;
    std::vector<float> power_all;
    if (use_backend_mel) {
        power_all.assign(static_cast<size_t>(num_frames) * static_cast<size_t>(fft_bins), 0.0f);
    }

    for (int t = 0; t < num_frames; ++t) {
        const int start = t * model.frame_shift_samples;
        for (int i = 0; i < model.frame_length_samples; ++i) {
            frame[static_cast<size_t>(i)] = waveform[start + i] * 32768.0f;
        }

        double mean = 0.0;
        for (float v : frame) {
            mean += static_cast<double>(v);
        }
        mean /= static_cast<double>(model.frame_length_samples);
        for (float & v : frame) {
            v -= static_cast<float>(mean);
        }

        for (int i = model.frame_length_samples - 1; i > 0; --i) {
            frame[static_cast<size_t>(i)] -= 0.97f * frame[static_cast<size_t>(i - 1)];
        }
        frame[0] -= 0.97f * frame[0];

        for (int i = 0; i < model.frame_length_samples; ++i) {
            frame[static_cast<size_t>(i)] *= model.hamming_window[static_cast<size_t>(i)];
        }

        std::fill(fft_buf.begin(), fft_buf.end(), std::complex<float>(0.0f, 0.0f));
        for (int i = 0; i < model.frame_length_samples; ++i) {
            fft_buf[static_cast<size_t>(i)] = std::complex<float>(frame[static_cast<size_t>(i)], 0.0f);
        }
        fft_inplace(fft_buf);

        for (int k = 0; k < fft_bins; ++k) {
            const std::complex<float> v = fft_buf[static_cast<size_t>(k)];
            power[static_cast<size_t>(k)] = std::norm(v);
        }

        if (use_backend_mel) {
            float * dst = &power_all[static_cast<size_t>(t) * static_cast<size_t>(fft_bins)];
            std::copy(power.begin(), power.end(), dst);
        } else {
            for (int m = 0; m < model.num_mel_bins; ++m) {
                double acc = 0.0;
                const size_t mbase = static_cast<size_t>(m) * static_cast<size_t>(fft_bins);
                for (int k = 0; k < fft_bins; ++k) {
                    acc += static_cast<double>(power[static_cast<size_t>(k)]) * static_cast<double>(model.mel_fbanks[mbase + static_cast<size_t>(k)]);
                }
                const float e = std::max(static_cast<float>(acc), std::numeric_limits<float>::epsilon());
                out[static_cast<size_t>(t) * static_cast<size_t>(model.num_mel_bins) + static_cast<size_t>(m)] = std::log(e);
            }
        }
    }

    if (use_backend_mel) {
        std::vector<float> mel = matmul_rows_backend(
            rt,
            power_all,
            num_frames,
            fft_bins,
            model.mel_fbanks,
            model.num_mel_bins);
        if (mel.size() != out.size()) {
            throw std::runtime_error("compute_fbank_features: backend mel shape mismatch");
        }
        for (size_t i = 0; i < mel.size(); ++i) {
            out[i] = std::log(std::max(mel[i], std::numeric_limits<float>::epsilon()));
        }
    }

    for (int m = 0; m < model.num_mel_bins; ++m) {
        double mean = 0.0;
        for (int t = 0; t < num_frames; ++t) {
            mean += static_cast<double>(out[static_cast<size_t>(t) * static_cast<size_t>(model.num_mel_bins) + static_cast<size_t>(m)]);
        }
        mean /= static_cast<double>(num_frames);
        for (int t = 0; t < num_frames; ++t) {
            const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(model.num_mel_bins) + static_cast<size_t>(m);
            out[idx] -= static_cast<float>(mean);
        }
    }

    out_num_frames = num_frames;
    return out;
}

static std::vector<float> compute_fbank_features(
    const std::vector<float> & waveform,
    const embedding_model & model,
    int & out_num_frames,
    const diarization_runtime * rt) {
    return compute_fbank_features_core(
        waveform.data(),
        static_cast<int>(waveform.size()),
        model,
        out_num_frames,
        rt);
}

static feature_map2d conv2d_forward(
    const feature_map2d & in,
    const conv2d_layer & conv,
    const diarization_runtime * rt,
    int padding = 1) {
    const int kw = static_cast<int>(conv.weight.ne0);
    const int kh = static_cast<int>(conv.weight.ne1);
    const int in_ch = static_cast<int>(conv.weight.ne2);
    const int out_ch = static_cast<int>(conv.weight.ne3);
    const int stride = std::max(1, conv.stride);

    if (in_ch != in.channels) {
        throw std::runtime_error("conv2d_forward: input channel mismatch");
    }
    if (kw <= 0 || kh <= 0 || out_ch <= 0) {
        throw std::runtime_error("conv2d_forward: invalid weight shape");
    }

    const int out_h = (in.height + 2 * padding - kh) / stride + 1;
    const int out_w = (in.width + 2 * padding - kw) / stride + 1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error("conv2d_forward: invalid output shape");
    }

    feature_map2d out;
    out.channels = out_ch;
    out.height = out_h;
    out.width = out_w;
    out.data.assign(static_cast<size_t>(out_ch) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w), 0.0f);

    const bool use_backend_conv2d_direct =
        rt != nullptr &&
        rt->use_backend_matmul &&
        rt->backend != nullptr &&
        rt->use_backend_conv2d_direct;
    if (use_backend_conv2d_direct) {
        try {
            std::vector<float> y = conv2d_forward_backend_direct(in, conv, padding, rt);
            if (y.size() == out.data.size()) {
                out.data = std::move(y);
                return out;
            }
        } catch (const std::exception & e) {
            if (rt->use_backend_conv2d_direct) {
                std::cerr << "[warn] disabling conv2d_direct backend path: " << e.what() << "\n";
                rt->use_backend_conv2d_direct = false;
                rt->conv2d_workspaces.clear();
            }
        }
    }

    const bool use_backend_conv =
        rt != nullptr &&
        rt->use_backend_matmul &&
        rt->backend != nullptr &&
        conv.kdim > 0 &&
        conv.weight_oc_k.size() == static_cast<size_t>(out_ch) * static_cast<size_t>(conv.kdim);

    if (use_backend_conv) {
        const int rows = out_h * out_w;
        std::vector<float> xcol(static_cast<size_t>(rows) * static_cast<size_t>(conv.kdim), 0.0f);
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                const int row = oh * out_w + ow;
                float * dst = &xcol[static_cast<size_t>(row) * static_cast<size_t>(conv.kdim)];
                int kd = 0;
                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int ky = 0; ky < kh; ++ky) {
                        const int ih = oh * stride + ky - padding;
                        for (int kx = 0; kx < kw; ++kx) {
                            const int iw = ow * stride + kx - padding;
                            dst[kd++] = (ih >= 0 && ih < in.height && iw >= 0 && iw < in.width) ? in.at(ic, ih, iw) : 0.0f;
                        }
                    }
                }
            }
        }

        std::vector<float> y = matmul_rows_backend(
            rt,
            xcol,
            rows,
            conv.kdim,
            conv.weight_oc_k,
            out_ch,
            true);
        if (y.size() != out.data.size()) {
            throw std::runtime_error("conv2d_forward: backend output shape mismatch");
        }
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                const int row = oh * out_w + ow;
                const float * src = &y[static_cast<size_t>(row) * static_cast<size_t>(out_ch)];
                for (int oc = 0; oc < out_ch; ++oc) {
                    out.at(oc, oh, ow) = src[oc];
                }
            }
        }
        return out;
    }

    for (int oc = 0; oc < out_ch; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double acc = 0.0;
                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int ky = 0; ky < kh; ++ky) {
                        const int ih = oh * stride + ky - padding;
                        if (ih < 0 || ih >= in.height) {
                            continue;
                        }
                        for (int kx = 0; kx < kw; ++kx) {
                            const int iw = ow * stride + kx - padding;
                            if (iw < 0 || iw >= in.width) {
                                continue;
                            }
                            const size_t widx =
                                static_cast<size_t>(kx) +
                                static_cast<size_t>(kw) * (
                                    static_cast<size_t>(ky) +
                                    static_cast<size_t>(kh) * (
                                        static_cast<size_t>(ic) +
                                        static_cast<size_t>(in_ch) * static_cast<size_t>(oc)));
                            acc += static_cast<double>(conv.weight.data[widx]) * static_cast<double>(in.at(ic, ih, iw));
                        }
                    }
                }
                out.at(oc, oh, ow) = static_cast<float>(acc);
            }
        }
    }

    return out;
}

static void batch_norm_2d_inplace(feature_map2d & x, const batch_norm_2d & bn) {
    if (x.channels <= 0 || x.height <= 0 || x.width <= 0) {
        return;
    }
    const size_t n = static_cast<size_t>(x.channels);
    if (bn.weight.size() != n || bn.bias.size() != n || bn.running_mean.size() != n || bn.running_var.size() != n) {
        throw std::runtime_error("batch_norm_2d_inplace: channel mismatch");
    }

    for (int c = 0; c < x.channels; ++c) {
        const float gamma = bn.weight[static_cast<size_t>(c)];
        const float beta = bn.bias[static_cast<size_t>(c)];
        const float mean = bn.running_mean[static_cast<size_t>(c)];
        const float var = bn.running_var[static_cast<size_t>(c)];
        const float inv_std = 1.0f / std::sqrt(var + bn.eps);
        for (int h = 0; h < x.height; ++h) {
            for (int w = 0; w < x.width; ++w) {
                float & v = x.at(c, h, w);
                v = (v - mean) * inv_std * gamma + beta;
            }
        }
    }
}

static void relu_inplace(feature_map2d & x) {
    for (float & v : x.data) {
        if (v < 0.0f) {
            v = 0.0f;
        }
    }
}

static void add_inplace(feature_map2d & dst, const feature_map2d & src) {
    if (dst.channels != src.channels || dst.height != src.height || dst.width != src.width) {
        throw std::runtime_error("add_inplace: shape mismatch");
    }
    for (size_t i = 0; i < dst.data.size(); ++i) {
        dst.data[i] += src.data[i];
    }
}

static void prepare_conv2d_layer(conv2d_layer & conv) {
    const int kw = static_cast<int>(conv.weight.ne0);
    const int kh = static_cast<int>(conv.weight.ne1);
    const int in_ch = static_cast<int>(conv.weight.ne2);
    const int out_ch = static_cast<int>(conv.weight.ne3);
    if (kw <= 0 || kh <= 0 || in_ch <= 0 || out_ch <= 0) {
        throw std::runtime_error("prepare_conv2d_layer: invalid conv weight shape");
    }
    const int kdim = kw * kh * in_ch;
    conv.kdim = kdim;
    conv.weight_oc_k.assign(static_cast<size_t>(out_ch) * static_cast<size_t>(kdim), 0.0f);

    for (int oc = 0; oc < out_ch; ++oc) {
        float * dst = &conv.weight_oc_k[static_cast<size_t>(oc) * static_cast<size_t>(kdim)];
        int kd = 0;
        for (int ic = 0; ic < in_ch; ++ic) {
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    const size_t widx =
                        static_cast<size_t>(kx) +
                        static_cast<size_t>(kw) * (
                            static_cast<size_t>(ky) +
                            static_cast<size_t>(kh) * (
                                static_cast<size_t>(ic) +
                                static_cast<size_t>(in_ch) * static_cast<size_t>(oc)));
                    dst[kd++] = conv.weight.data[widx];
                }
            }
        }
    }
}

static feature_map2d run_resnet_block(
    const feature_map2d & x,
    const resnet_block & b,
    const diarization_runtime * rt) {
    feature_map2d out = conv2d_forward(x, b.conv1, rt, 1);
    batch_norm_2d_inplace(out, b.bn1);
    relu_inplace(out);

    out = conv2d_forward(out, b.conv2, rt, 1);
    batch_norm_2d_inplace(out, b.bn2);

    feature_map2d shortcut = x;
    if (b.has_shortcut) {
        shortcut = conv2d_forward(x, b.shortcut_conv, rt, 0);
        batch_norm_2d_inplace(shortcut, b.shortcut_bn);
    }

    add_inplace(out, shortcut);
    relu_inplace(out);
    return out;
}

static std::vector<float> stats_pool_tstp(const feature_map2d & x) {
    if (x.channels <= 0 || x.height <= 0 || x.width <= 0) {
        throw std::runtime_error("stats_pool_tstp: invalid input shape");
    }
    const int D = x.channels * x.height;
    const int T = x.width;

    std::vector<float> stats(static_cast<size_t>(2 * D), 0.0f);
    for (int d = 0; d < D; ++d) {
        const int c = d / x.height;
        const int h = d % x.height;

        double mean = 0.0;
        for (int t = 0; t < T; ++t) {
            mean += static_cast<double>(x.at(c, h, t));
        }
        mean /= static_cast<double>(T);
        stats[static_cast<size_t>(d)] = static_cast<float>(mean);

        if (T <= 1) {
            stats[static_cast<size_t>(D + d)] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        double var = 0.0;
        for (int t = 0; t < T; ++t) {
            const double dv = static_cast<double>(x.at(c, h, t)) - mean;
            var += dv * dv;
        }
        var /= static_cast<double>(T - 1); // unbiased std (correction=1)
        stats[static_cast<size_t>(D + d)] = static_cast<float>(std::sqrt(std::max(0.0, var)));
    }
    return stats;
}

static std::vector<float> infer_embedding_samples(
    const embedding_model & model,
    const float * chunk_audio,
    int chunk_num_samples,
    const diarization_runtime * rt) {
    int num_frames = 0;
    std::vector<float> fbank = compute_fbank_features_core(chunk_audio, chunk_num_samples, model, num_frames, rt); // [T, F]

    feature_map2d x;
    x.channels = 1;
    x.height = model.num_mel_bins;
    x.width = num_frames;
    x.data.assign(static_cast<size_t>(x.channels) * static_cast<size_t>(x.height) * static_cast<size_t>(x.width), 0.0f);
    for (int t = 0; t < num_frames; ++t) {
        for (int f = 0; f < model.num_mel_bins; ++f) {
            x.at(0, f, t) = fbank[static_cast<size_t>(t) * static_cast<size_t>(model.num_mel_bins) + static_cast<size_t>(f)];
        }
    }

    x = conv2d_forward(x, model.conv1, rt, 1);
    batch_norm_2d_inplace(x, model.bn1);
    relu_inplace(x);

    for (const auto & stage : model.layers) {
        for (const auto & block : stage) {
            x = run_resnet_block(x, block, rt);
        }
    }

    std::vector<float> stats = stats_pool_tstp(x);
    const int in_dim = static_cast<int>(stats.size());
    if (static_cast<int>(model.seg1_w.ne0) != in_dim) {
        throw std::runtime_error("infer_embedding_chunk: seg_1 input dim mismatch");
    }
    std::vector<float> emb = linear_forward(
        stats,
        1,
        in_dim,
        model.seg1_w,
        model.seg1_b,
        rt);
    return emb;
}

static std::vector<float> infer_embedding_chunk(
    const embedding_model & model,
    const std::vector<float> & chunk_audio,
    const diarization_runtime * rt) {
    return infer_embedding_samples(
        model,
        chunk_audio.data(),
        static_cast<int>(chunk_audio.size()),
        rt);
}

static std::vector<float> infer_embedding_range(
    const embedding_model & model,
    const std::vector<float> & audio,
    int start_i,
    int end_i,
    const diarization_runtime * rt) {
    if (start_i < 0 || end_i < start_i || end_i > static_cast<int>(audio.size())) {
        throw std::runtime_error("infer_embedding_range: invalid audio range");
    }
    return infer_embedding_samples(
        model,
        audio.data() + static_cast<size_t>(start_i),
        end_i - start_i,
        rt);
}

static batch_norm_2d load_batch_norm_2d(
    ggml_context * tctx,
    const std::filesystem::path & src_path,
    const std::string & prefix) {
    batch_norm_2d bn;
    bn.weight = load_tensor_as_f32(tctx, prefix + ".weight", src_path).data;
    bn.bias = load_tensor_as_f32(tctx, prefix + ".bias", src_path).data;
    bn.running_mean = load_tensor_as_f32(tctx, prefix + ".running_mean", src_path).data;
    bn.running_var = load_tensor_as_f32(tctx, prefix + ".running_var", src_path).data;
    const size_t n = bn.weight.size();
    if (n == 0 || bn.bias.size() != n || bn.running_mean.size() != n || bn.running_var.size() != n) {
        throw std::runtime_error("invalid batch norm tensor shapes at " + prefix);
    }
    return bn;
}

static embedding_model load_embedding_model(const std::filesystem::path & gguf_path) {
    ggml_context * tctx = nullptr;
    gguf_init_params params = {};
    params.no_alloc = false;
    params.ctx = &tctx;

    gguf_context * gctx = gguf_init_from_file(gguf_path.string().c_str(), params);
    if (gctx == nullptr || tctx == nullptr) {
        if (gctx != nullptr) {
            gguf_free(gctx);
        }
        throw std::runtime_error("failed to load embedding GGUF: " + gguf_path.string());
    }

    auto load = [&](const std::string & name) {
        return load_tensor_as_f32(tctx, name, gguf_path);
    };

    embedding_model m;
    {
        const int key_sr = gguf_find_key(gctx, "pyannote.sample_rate");
        if (key_sr >= 0) {
            m.sample_rate = static_cast<int>(gguf_get_val_u32(gctx, key_sr));
        }

        const int key_arch = gguf_find_key(gctx, "pyannote.architecture.class");
        if (key_arch >= 0) {
            const char * s = gguf_get_val_str(gctx, key_arch);
            if (s != nullptr) {
                m.architecture_class = s;
            }
        }
    }

    m.frame_length_samples = static_cast<int>(std::llround(static_cast<double>(m.sample_rate) * 0.025));
    m.frame_shift_samples = static_cast<int>(std::llround(static_cast<double>(m.sample_rate) * 0.010));
    m.fft_size = next_power_of_two(m.frame_length_samples);
    m.min_num_samples = m.frame_length_samples;

    m.hamming_window = build_hamming_window(m.frame_length_samples);
    m.mel_fbanks = build_kaldi_mel_fbanks(
        m.sample_rate,
        m.fft_size,
        m.num_mel_bins,
        20.0f,
        0.0f);

    m.conv1.weight = load("pyannote.embedding.resnet.conv1.weight");
    m.conv1.stride = 1;
    prepare_conv2d_layer(m.conv1);
    m.bn1 = load_batch_norm_2d(tctx, gguf_path, "pyannote.embedding.resnet.bn1");

    const std::array<int, 4> blocks_per_stage = {3, 4, 6, 3};
    for (int stage = 0; stage < 4; ++stage) {
        m.layers[static_cast<size_t>(stage)].reserve(static_cast<size_t>(blocks_per_stage[static_cast<size_t>(stage)]));
        for (int bi = 0; bi < blocks_per_stage[static_cast<size_t>(stage)]; ++bi) {
            const std::string base =
                "pyannote.embedding.resnet.layer" + std::to_string(stage + 1) + "." + std::to_string(bi);

            resnet_block b;
            b.conv1.weight = load(base + ".conv1.weight");
            b.conv1.stride = (bi == 0 && stage > 0) ? 2 : 1;
            prepare_conv2d_layer(b.conv1);
            b.bn1 = load_batch_norm_2d(tctx, gguf_path, base + ".bn1");

            b.conv2.weight = load(base + ".conv2.weight");
            b.conv2.stride = 1;
            prepare_conv2d_layer(b.conv2);
            b.bn2 = load_batch_norm_2d(tctx, gguf_path, base + ".bn2");

            const std::string sc_conv = base + ".shortcut.0.weight";
            const std::string sc_bn = base + ".shortcut.1";
            b.has_shortcut = gguf_has_tensor(tctx, sc_conv);
            if (b.has_shortcut) {
                b.shortcut_conv.weight = load(sc_conv);
                b.shortcut_conv.stride = (bi == 0 && stage > 0) ? 2 : 1;
                prepare_conv2d_layer(b.shortcut_conv);
                b.shortcut_bn = load_batch_norm_2d(tctx, gguf_path, sc_bn);
            }

            m.layers[static_cast<size_t>(stage)].push_back(std::move(b));
        }
    }

    m.seg1_w = load("pyannote.embedding.resnet.seg_1.weight");
    m.seg1_b = load("pyannote.embedding.resnet.seg_1.bias");
    if (m.seg1_w.ne1 <= 0 || static_cast<size_t>(m.seg1_w.ne1) != m.seg1_b.data.size()) {
        throw std::runtime_error("invalid embedding seg_1 tensor dimensions");
    }
    m.embed_dim = static_cast<int>(m.seg1_w.ne1);

    gguf_free(gctx);
    ggml_free(tctx);
    return m;
}

static matmul_workspace * get_or_create_matmul_workspace(
    const diarization_runtime * rt,
    int rows,
    int in_dim,
    int out_dim,
    const float * weight_key_ptr,
    size_t weight_key_size) {
    if (rt == nullptr || rt->backend == nullptr) {
        throw std::runtime_error("get_or_create_matmul_workspace: runtime/backend is null");
    }

    for (const auto & uptr : rt->matmul_workspaces) {
        if (uptr != nullptr &&
            uptr->rows == rows &&
            uptr->in_dim == in_dim &&
            uptr->out_dim == out_dim &&
            uptr->weight_key_ptr == weight_key_ptr &&
            uptr->weight_key_size == weight_key_size) {
            return uptr.get();
        }
    }

    // Keep cache bounded to avoid VRAM growth from many dynamic shapes/weights.
    constexpr size_t kMaxWorkspaceCacheEntries = 64;
    if (rt->matmul_workspaces.size() >= kMaxWorkspaceCacheEntries) {
        if (rt->matmul_workspaces.front() != nullptr) {
            rt->matmul_workspaces.front()->clear();
        }
        rt->matmul_workspaces.erase(rt->matmul_workspaces.begin());
    }

    auto ws = std::make_unique<matmul_workspace>();
    ws->rows = rows;
    ws->in_dim = in_dim;
    ws->out_dim = out_dim;
    ws->weight_key_ptr = weight_key_ptr;
    ws->weight_key_size = weight_key_size;

    ggml_init_params params = {};
    params.mem_size = 16u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ws->ctx = ggml_init(params);
    if (ws->ctx == nullptr) {
        throw std::runtime_error("get_or_create_matmul_workspace: ggml_init failed");
    }

    ws->t_w = ggml_new_tensor_2d(ws->ctx, GGML_TYPE_F32, in_dim, out_dim);
    ws->t_x = ggml_new_tensor_2d(ws->ctx, GGML_TYPE_F32, in_dim, rows);
    ws->t_y = ggml_mul_mat(ws->ctx, ws->t_w, ws->t_x); // [out_dim, rows]

    ws->gf = ggml_new_graph(ws->ctx);
    ggml_build_forward_expand(ws->gf, ws->t_y);

    ws->buf = ggml_backend_alloc_ctx_tensors(ws->ctx, rt->backend);
    if (ws->buf == nullptr) {
        ws->clear();
        throw std::runtime_error("get_or_create_matmul_workspace: failed to allocate backend buffers");
    }

    matmul_workspace * ret = ws.get();
    rt->matmul_workspaces.push_back(std::move(ws));
    return ret;
}

static conv1d_workspace * get_or_create_conv1d_workspace(
    const diarization_runtime * rt,
    int in_frames,
    int in_ch,
    int out_ch,
    int kernel,
    int stride,
    int padding,
    const float * weight_key_ptr,
    size_t weight_key_size) {
    if (rt == nullptr || rt->backend == nullptr) {
        throw std::runtime_error("get_or_create_conv1d_workspace: runtime/backend is null");
    }

    for (const auto & uptr : rt->conv1d_workspaces) {
        if (uptr != nullptr &&
            uptr->in_frames == in_frames &&
            uptr->in_ch == in_ch &&
            uptr->out_ch == out_ch &&
            uptr->kernel == kernel &&
            uptr->stride == stride &&
            uptr->padding == padding &&
            uptr->weight_key_ptr == weight_key_ptr &&
            uptr->weight_key_size == weight_key_size) {
            return uptr.get();
        }
    }

    constexpr size_t kMaxConv1dWorkspaceCacheEntries = 128;
    if (rt->conv1d_workspaces.size() >= kMaxConv1dWorkspaceCacheEntries) {
        if (rt->conv1d_workspaces.front() != nullptr) {
            rt->conv1d_workspaces.front()->clear();
        }
        rt->conv1d_workspaces.erase(rt->conv1d_workspaces.begin());
    }

    auto ws = std::make_unique<conv1d_workspace>();
    ws->in_frames = in_frames;
    ws->in_ch = in_ch;
    ws->out_ch = out_ch;
    ws->kernel = kernel;
    ws->stride = stride;
    ws->padding = padding;
    ws->weight_key_ptr = weight_key_ptr;
    ws->weight_key_size = weight_key_size;

    ggml_init_params params = {};
    params.mem_size = 16u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ws->ctx = ggml_init(params);
    if (ws->ctx == nullptr) {
        throw std::runtime_error("get_or_create_conv1d_workspace: ggml_init failed");
    }

    ws->t_w = ggml_new_tensor_3d(ws->ctx, GGML_TYPE_F32, kernel, in_ch, out_ch);
    ws->t_x = ggml_new_tensor_3d(ws->ctx, GGML_TYPE_F32, in_frames, in_ch, 1);
    ws->t_y = ggml_conv_1d(ws->ctx, ws->t_w, ws->t_x, stride, padding, 1);

    ws->gf = ggml_new_graph(ws->ctx);
    ggml_build_forward_expand(ws->gf, ws->t_y);

    ws->buf = ggml_backend_alloc_ctx_tensors(ws->ctx, rt->backend);
    if (ws->buf == nullptr) {
        ws->clear();
        throw std::runtime_error("get_or_create_conv1d_workspace: failed to allocate backend buffers");
    }

    conv1d_workspace * ret = ws.get();
    rt->conv1d_workspaces.push_back(std::move(ws));
    return ret;
}

static conv2d_workspace * get_or_create_conv2d_workspace(
    const diarization_runtime * rt,
    int in_w,
    int in_h,
    int in_ch,
    int out_ch,
    int kw,
    int kh,
    int stride,
    int padding,
    const float * weight_key_ptr,
    size_t weight_key_size) {
    if (rt == nullptr || rt->backend == nullptr) {
        throw std::runtime_error("get_or_create_conv2d_workspace: runtime/backend is null");
    }

    for (const auto & uptr : rt->conv2d_workspaces) {
        if (uptr != nullptr &&
            uptr->in_w == in_w &&
            uptr->in_h == in_h &&
            uptr->in_ch == in_ch &&
            uptr->out_ch == out_ch &&
            uptr->kw == kw &&
            uptr->kh == kh &&
            uptr->stride == stride &&
            uptr->padding == padding &&
            uptr->weight_key_ptr == weight_key_ptr &&
            uptr->weight_key_size == weight_key_size) {
            return uptr.get();
        }
    }

    // Embedding stage uses many dynamic widths, keep this bounded.
    constexpr size_t kMaxConv2dWorkspaceCacheEntries = 96;
    if (rt->conv2d_workspaces.size() >= kMaxConv2dWorkspaceCacheEntries) {
        if (rt->conv2d_workspaces.front() != nullptr) {
            rt->conv2d_workspaces.front()->clear();
        }
        rt->conv2d_workspaces.erase(rt->conv2d_workspaces.begin());
    }

    auto ws = std::make_unique<conv2d_workspace>();
    ws->in_w = in_w;
    ws->in_h = in_h;
    ws->in_ch = in_ch;
    ws->out_ch = out_ch;
    ws->kw = kw;
    ws->kh = kh;
    ws->stride = stride;
    ws->padding = padding;
    ws->weight_key_ptr = weight_key_ptr;
    ws->weight_key_size = weight_key_size;

    ggml_init_params params = {};
    params.mem_size = 16u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ws->ctx = ggml_init(params);
    if (ws->ctx == nullptr) {
        throw std::runtime_error("get_or_create_conv2d_workspace: ggml_init failed");
    }

    ws->t_w = ggml_new_tensor_4d(ws->ctx, GGML_TYPE_F32, kw, kh, in_ch, out_ch);
    ws->t_x = ggml_new_tensor_4d(ws->ctx, GGML_TYPE_F32, in_w, in_h, in_ch, 1);
    ws->t_y = ggml_conv_2d_direct(ws->ctx, ws->t_w, ws->t_x, stride, stride, padding, padding, 1, 1);

    ws->gf = ggml_new_graph(ws->ctx);
    ggml_build_forward_expand(ws->gf, ws->t_y);

    ws->buf = ggml_backend_alloc_ctx_tensors(ws->ctx, rt->backend);
    if (ws->buf == nullptr) {
        ws->clear();
        throw std::runtime_error("get_or_create_conv2d_workspace: failed to allocate backend buffers");
    }

    conv2d_workspace * ret = ws.get();
    rt->conv2d_workspaces.push_back(std::move(ws));
    return ret;
}

static std::vector<float> conv1d_forward_backend_ggml(
    const feature_map & in,
    const std::vector<float> & w,
    int in_ch,
    int out_ch,
    int kernel,
    int stride,
    int padding,
    const diarization_runtime * rt,
    const float * weight_key_ptr,
    size_t weight_key_size) {
    if (rt == nullptr || rt->backend == nullptr) {
        throw std::runtime_error("conv1d_forward_backend_ggml: runtime/backend is null");
    }
    if (stride <= 0 || in_ch <= 0 || out_ch <= 0 || kernel <= 0) {
        throw std::runtime_error("conv1d_forward_backend_ggml: invalid parameters");
    }
    if (in.channels != in_ch) {
        throw std::runtime_error("conv1d_forward_backend_ggml: input channel mismatch");
    }
    if (static_cast<size_t>(in_ch) * static_cast<size_t>(in.frames) != in.data.size()) {
        throw std::runtime_error("conv1d_forward_backend_ggml: invalid input shape");
    }
    if (static_cast<size_t>(out_ch) * static_cast<size_t>(in_ch) * static_cast<size_t>(kernel) != w.size()) {
        throw std::runtime_error("conv1d_forward_backend_ggml: invalid weight shape");
    }

    const int out_frames = (in.frames + 2 * padding - kernel) / stride + 1;
    if (out_frames <= 0) {
        return {};
    }

    const size_t in_size = static_cast<size_t>(in_ch) * static_cast<size_t>(in.frames);
    const size_t out_size = static_cast<size_t>(out_ch) * static_cast<size_t>(out_frames);
    const size_t w_size = static_cast<size_t>(out_ch) * static_cast<size_t>(in_ch) * static_cast<size_t>(kernel);
    const size_t tensor_bytes_est = (in_size + out_size + w_size) * sizeof(float);
    constexpr size_t kMaxCachedTensorBytes = 96u * 1024u * 1024u;
    const bool do_cache = tensor_bytes_est <= kMaxCachedTensorBytes;

    if (do_cache) {
        conv1d_workspace * ws = get_or_create_conv1d_workspace(
            rt,
            in.frames,
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            weight_key_ptr,
            weight_key_size);
        if (ws == nullptr || ws->t_w == nullptr || ws->t_x == nullptr || ws->t_y == nullptr || ws->gf == nullptr) {
            throw std::runtime_error("conv1d_forward_backend_ggml: invalid workspace");
        }

        if (!ws->weight_uploaded || ws->cached_w_ptr != weight_key_ptr || ws->cached_w_size != weight_key_size) {
            ggml_backend_tensor_set(ws->t_w, w.data(), 0, w.size() * sizeof(float));
            ws->weight_uploaded = true;
            ws->cached_w_ptr = weight_key_ptr;
            ws->cached_w_size = weight_key_size;
        }

        ggml_backend_tensor_set(ws->t_x, in.data.data(), 0, in.data.size() * sizeof(float));
        const ggml_status st = ggml_backend_graph_compute(rt->backend, ws->gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("conv1d_forward_backend_ggml: graph compute failed");
        }

        std::vector<float> out(out_size, 0.0f);
        ggml_backend_tensor_get(ws->t_y, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_init_params params = {};
    params.mem_size = 16u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("conv1d_forward_backend_ggml: ggml_init failed");
    }

    ggml_backend_buffer_t buf = nullptr;
    try {
        ggml_tensor * t_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel, in_ch, out_ch);
        ggml_tensor * t_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in.frames, in_ch, 1);
        ggml_tensor * t_y = ggml_conv_1d(ctx, t_w, t_x, stride, padding, 1);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t_y);

        buf = ggml_backend_alloc_ctx_tensors(ctx, rt->backend);
        if (buf == nullptr) {
            throw std::runtime_error("conv1d_forward_backend_ggml: failed to allocate backend buffers");
        }

        ggml_backend_tensor_set(t_w, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(t_x, in.data.data(), 0, in.data.size() * sizeof(float));

        const ggml_status st = ggml_backend_graph_compute(rt->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("conv1d_forward_backend_ggml: graph compute failed");
        }

        std::vector<float> out(out_size, 0.0f);
        ggml_backend_tensor_get(t_y, out.data(), 0, out.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return out;
    } catch (...) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        throw;
    }
}

static std::vector<float> conv2d_forward_backend_direct(
    const feature_map2d & in,
    const conv2d_layer & conv,
    int padding,
    const diarization_runtime * rt) {
    if (rt == nullptr || rt->backend == nullptr) {
        throw std::runtime_error("conv2d_forward_backend_direct: runtime/backend is null");
    }

    const int kw = static_cast<int>(conv.weight.ne0);
    const int kh = static_cast<int>(conv.weight.ne1);
    const int in_ch = static_cast<int>(conv.weight.ne2);
    const int out_ch = static_cast<int>(conv.weight.ne3);
    const int stride = std::max(1, conv.stride);
    const int out_h = (in.height + 2 * padding - kh) / stride + 1;
    const int out_w = (in.width + 2 * padding - kw) / stride + 1;

    if (out_h <= 0 || out_w <= 0) {
        return {};
    }

    const size_t in_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height) * static_cast<size_t>(in_ch);
    const size_t out_size = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * static_cast<size_t>(out_ch);
    const size_t w_size = static_cast<size_t>(kw) * static_cast<size_t>(kh) * static_cast<size_t>(in_ch) * static_cast<size_t>(out_ch);
    if (in_size != in.data.size() || w_size != conv.weight.data.size()) {
        throw std::runtime_error("conv2d_forward_backend_direct: tensor shape mismatch");
    }

    const size_t tensor_bytes_est = (in_size + out_size + w_size) * sizeof(float);
    constexpr size_t kMaxCachedTensorBytes = 96u * 1024u * 1024u;
    const bool do_cache = tensor_bytes_est <= kMaxCachedTensorBytes;

    if (do_cache) {
        conv2d_workspace * ws = get_or_create_conv2d_workspace(
            rt,
            in.width,
            in.height,
            in_ch,
            out_ch,
            kw,
            kh,
            stride,
            padding,
            conv.weight.data.data(),
            conv.weight.data.size());
        if (ws == nullptr || ws->t_w == nullptr || ws->t_x == nullptr || ws->t_y == nullptr || ws->gf == nullptr) {
            throw std::runtime_error("conv2d_forward_backend_direct: invalid workspace");
        }

        if (!ws->weight_uploaded) {
            ggml_backend_tensor_set(ws->t_w, conv.weight.data.data(), 0, conv.weight.data.size() * sizeof(float));
            ws->weight_uploaded = true;
        }

        ggml_backend_tensor_set(ws->t_x, in.data.data(), 0, in.data.size() * sizeof(float));
        const ggml_status st = ggml_backend_graph_compute(rt->backend, ws->gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("conv2d_forward_backend_direct: graph compute failed");
        }

        std::vector<float> out(out_size, 0.0f);
        ggml_backend_tensor_get(ws->t_y, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_init_params params = {};
    params.mem_size = 16u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("conv2d_forward_backend_direct: ggml_init failed");
    }

    ggml_backend_buffer_t buf = nullptr;
    try {
        ggml_tensor * t_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, in_ch, out_ch);
        ggml_tensor * t_x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, in.width, in.height, in_ch, 1);
        ggml_tensor * t_y = ggml_conv_2d_direct(ctx, t_w, t_x, stride, stride, padding, padding, 1, 1);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t_y);

        buf = ggml_backend_alloc_ctx_tensors(ctx, rt->backend);
        if (buf == nullptr) {
            throw std::runtime_error("conv2d_forward_backend_direct: failed to allocate backend buffers");
        }

        ggml_backend_tensor_set(t_w, conv.weight.data.data(), 0, conv.weight.data.size() * sizeof(float));
        ggml_backend_tensor_set(t_x, in.data.data(), 0, in.data.size() * sizeof(float));

        const ggml_status st = ggml_backend_graph_compute(rt->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("conv2d_forward_backend_direct: graph compute failed");
        }

        std::vector<float> out(out_size, 0.0f);
        ggml_backend_tensor_get(t_y, out.data(), 0, out.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return out;
    } catch (...) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        throw;
    }
}

static std::vector<float> matmul_rows_backend(
    const diarization_runtime * rt,
    const std::vector<float> & in,
    int rows,
    int in_dim,
    const std::vector<float> & w,
    int out_dim,
    bool use_workspace_cache) {
    if (rt == nullptr || rt->backend == nullptr) {
        throw std::runtime_error("matmul_rows_backend: runtime/backend is null");
    }
    if (rows <= 0 || in_dim <= 0 || out_dim <= 0) {
        return {};
    }
    if (static_cast<size_t>(rows) * static_cast<size_t>(in_dim) != in.size()) {
        throw std::runtime_error("matmul_rows_backend: input shape mismatch");
    }
    if (static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim) != w.size()) {
        throw std::runtime_error("matmul_rows_backend: weight shape mismatch");
    }

    const size_t tensor_bytes_est =
        (static_cast<size_t>(rows) * static_cast<size_t>(in_dim) +
         static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim) +
         static_cast<size_t>(rows) * static_cast<size_t>(out_dim)) * sizeof(float);
    constexpr size_t kMaxCachedTensorBytes = 64u * 1024u * 1024u;
    const bool do_cache = use_workspace_cache && tensor_bytes_est <= kMaxCachedTensorBytes;

    if (do_cache) {
        matmul_workspace * ws = get_or_create_matmul_workspace(
            rt,
            rows,
            in_dim,
            out_dim,
            w.data(),
            w.size());
        if (ws == nullptr || ws->t_w == nullptr || ws->t_x == nullptr || ws->t_y == nullptr || ws->gf == nullptr) {
            throw std::runtime_error("matmul_rows_backend: invalid workspace");
        }

        if (!ws->weight_uploaded || ws->cached_w_ptr != w.data() || ws->cached_w_size != w.size()) {
            ggml_backend_tensor_set(ws->t_w, w.data(), 0, w.size() * sizeof(float));
            ws->weight_uploaded = true;
            ws->cached_w_ptr = w.data();
            ws->cached_w_size = w.size();
        }

        ggml_backend_tensor_set(ws->t_x, in.data(), 0, in.size() * sizeof(float));

        const ggml_status st = ggml_backend_graph_compute(rt->backend, ws->gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("matmul_rows_backend: graph compute failed");
        }

        std::vector<float> out(static_cast<size_t>(rows) * static_cast<size_t>(out_dim), 0.0f);
        ggml_backend_tensor_get(ws->t_y, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_init_params params = {};
    params.mem_size = 16u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("matmul_rows_backend: ggml_init failed");
    }

    ggml_backend_buffer_t buf = nullptr;
    try {
        ggml_tensor * t_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, out_dim);
        ggml_tensor * t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, rows);
        ggml_tensor * t_y = ggml_mul_mat(ctx, t_w, t_x); // [out_dim, rows]

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t_y);

        buf = ggml_backend_alloc_ctx_tensors(ctx, rt->backend);
        if (buf == nullptr) {
            throw std::runtime_error("matmul_rows_backend: failed to allocate backend buffers");
        }

        ggml_backend_tensor_set(t_w, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(t_x, in.data(), 0, in.size() * sizeof(float));

        const ggml_status st = ggml_backend_graph_compute(rt->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("matmul_rows_backend: graph compute failed");
        }

        std::vector<float> out(static_cast<size_t>(rows) * static_cast<size_t>(out_dim), 0.0f);
        ggml_backend_tensor_get(t_y, out.data(), 0, out.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return out;
    } catch (...) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        throw;
    }
}

static std::vector<float> linear_forward(
    const std::vector<float> & in,
    int rows,
    int in_dim,
    const tensor_f32 & w,
    const tensor_f32 & b,
    const diarization_runtime * rt) {
    const int w_in = static_cast<int>(w.ne0);
    const int w_out = static_cast<int>(w.ne1);
    if (w_in != in_dim || static_cast<size_t>(w_out) != b.data.size()) {
        throw std::runtime_error("linear_forward: tensor dimensions mismatch");
    }
    if (static_cast<size_t>(rows) * static_cast<size_t>(in_dim) != in.size()) {
        throw std::runtime_error("linear_forward: input size mismatch");
    }

    std::vector<float> out;
    if (rt != nullptr && rt->use_backend_matmul && rt->backend != nullptr) {
        out = matmul_rows_backend(rt, in, rows, in_dim, w.data, w_out);
        for (int r = 0; r < rows; ++r) {
            float * y = &out[static_cast<size_t>(r) * static_cast<size_t>(w_out)];
            for (int j = 0; j < w_out; ++j) {
                y[j] += b.data[static_cast<size_t>(j)];
            }
        }
        return out;
    }

    out.assign(static_cast<size_t>(rows) * static_cast<size_t>(w_out), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float * x = &in[static_cast<size_t>(r) * static_cast<size_t>(in_dim)];
        for (int j = 0; j < w_out; ++j) {
            double acc = static_cast<double>(b.data[static_cast<size_t>(j)]);
            const size_t wrow = static_cast<size_t>(j) * static_cast<size_t>(w_in);
            for (int i = 0; i < in_dim; ++i) {
                acc += static_cast<double>(w.data[wrow + static_cast<size_t>(i)]) * static_cast<double>(x[i]);
            }
            out[static_cast<size_t>(r) * static_cast<size_t>(w_out) + static_cast<size_t>(j)] = static_cast<float>(acc);
        }
    }

    return out;
}

static void leaky_relu_inplace(std::vector<float> & x, float slope = 0.01f) {
    for (float & v : x) {
        if (v < 0.0f) {
            v *= slope;
        }
    }
}

struct lstm_whh_hg_cache_entry {
    const lstm_direction * dir_ptr = nullptr;
    int H = 0;
    int G = 0;
    std::vector<float> packed_h_g; // [H, G]
};

static const std::vector<float> & get_or_pack_lstm_whh_hg(const lstm_direction & dir) {
    static thread_local std::vector<lstm_whh_hg_cache_entry> cache;

    const int H = dir.hidden_size;
    const int G = 4 * H;
    if (static_cast<int>(dir.w_hh.size()) != G * H) {
        throw std::runtime_error("get_or_pack_lstm_whh_hg: invalid w_hh tensor size");
    }

    for (const auto & entry : cache) {
        if (entry.dir_ptr == &dir && entry.H == H && entry.G == G) {
            return entry.packed_h_g;
        }
    }

    lstm_whh_hg_cache_entry entry;
    entry.dir_ptr = &dir;
    entry.H = H;
    entry.G = G;
    entry.packed_h_g.assign(static_cast<size_t>(H) * static_cast<size_t>(G), 0.0f);

    for (int j = 0; j < H; ++j) {
        float * dst = &entry.packed_h_g[static_cast<size_t>(j) * static_cast<size_t>(G)];
        for (int g = 0; g < G; ++g) {
            dst[g] = dir.w_hh[static_cast<size_t>(g) * static_cast<size_t>(H) + static_cast<size_t>(j)];
        }
    }

    cache.push_back(std::move(entry));
    return cache.back().packed_h_g;
}

static void lstm_accumulate_recurrent_hg(
    const std::vector<float> & whh_hg,
    int H,
    int G,
    const float * h_ptr,
    std::vector<float> & gates) {
    for (int j = 0; j < H; ++j) {
        const float hj = h_ptr[static_cast<size_t>(j)];
        const float * wjg = &whh_hg[static_cast<size_t>(j) * static_cast<size_t>(G)];

#if defined(__AVX2__) || defined(_M_AVX2)
        int g = 0;
        const __m256 h8 = _mm256_set1_ps(hj);
        for (; g + 8 <= G; g += 8) {
            const __m256 wv = _mm256_loadu_ps(wjg + g);
            const __m256 gv = _mm256_loadu_ps(gates.data() + g);
            const __m256 yv = _mm256_add_ps(gv, _mm256_mul_ps(wv, h8));
            _mm256_storeu_ps(gates.data() + g, yv);
        }
        for (; g < G; ++g) {
            gates[static_cast<size_t>(g)] += wjg[g] * hj;
        }
#else
        for (int g = 0; g < G; ++g) {
            gates[static_cast<size_t>(g)] += wjg[g] * hj;
        }
#endif
    }
}

static std::vector<float> lstm_direction_forward_from_xproj(
    const std::vector<float> & xproj,
    int time_steps,
    int xproj_stride,
    int xproj_offset,
    const lstm_direction & dir,
    bool reverse) {
    const int I = dir.input_size;
    const int H = dir.hidden_size;
    const int G = 4 * H;

    if (static_cast<int>(dir.w_ih.size()) != G * I ||
        static_cast<int>(dir.w_hh.size()) != G * H ||
        static_cast<int>(dir.b_ih.size()) != G ||
        static_cast<int>(dir.b_hh.size()) != G) {
        throw std::runtime_error("lstm_direction_forward: invalid LSTM tensor sizes");
    }

    std::vector<float> gate_bias(static_cast<size_t>(G), 0.0f);
    for (int g = 0; g < G; ++g) {
        gate_bias[static_cast<size_t>(g)] =
            dir.b_ih[static_cast<size_t>(g)] +
            dir.b_hh[static_cast<size_t>(g)];
    }
    const std::vector<float> & whh_hg = get_or_pack_lstm_whh_hg(dir);

    std::vector<float> out(static_cast<size_t>(time_steps) * static_cast<size_t>(H), 0.0f);
    std::vector<float> h(static_cast<size_t>(H), 0.0f);
    std::vector<float> c(static_cast<size_t>(H), 0.0f);
    std::vector<float> gates(static_cast<size_t>(G), 0.0f);
    if (xproj_stride < G || xproj_offset < 0 || xproj_offset + G > xproj_stride) {
        throw std::runtime_error("lstm_direction_forward_from_xproj: invalid xproj stride/offset");
    }
    if (xproj.size() != static_cast<size_t>(time_steps) * static_cast<size_t>(xproj_stride)) {
        throw std::runtime_error("lstm_direction_forward_from_xproj: xproj size mismatch");
    }

    for (int step = 0; step < time_steps; ++step) {
        const int t = reverse ? (time_steps - 1 - step) : step;
        const float * xp = &xproj[static_cast<size_t>(t) * static_cast<size_t>(xproj_stride) + static_cast<size_t>(xproj_offset)];
        const float * h_ptr = h.data();

        for (int g = 0; g < G; ++g) {
            gates[static_cast<size_t>(g)] = xp[g] + gate_bias[static_cast<size_t>(g)];
        }
        lstm_accumulate_recurrent_hg(whh_hg, H, G, h_ptr, gates);

        for (int j = 0; j < H; ++j) {
            const float i_gate = sigmoidf_stable(gates[static_cast<size_t>(j)]);
            const float f_gate = sigmoidf_stable(gates[static_cast<size_t>(H + j)]);
            const float g_gate = std::tanh(gates[static_cast<size_t>(2 * H + j)]);
            const float o_gate = sigmoidf_stable(gates[static_cast<size_t>(3 * H + j)]);

            c[static_cast<size_t>(j)] = f_gate * c[static_cast<size_t>(j)] + i_gate * g_gate;
            h[static_cast<size_t>(j)] = o_gate * std::tanh(c[static_cast<size_t>(j)]);
        }

        float * y = &out[static_cast<size_t>(t) * static_cast<size_t>(H)];
        std::copy(h.begin(), h.end(), y);
    }

    return out;
}

static std::vector<float> lstm_direction_forward(
    const std::vector<float> & in,
    int time_steps,
    const lstm_direction & dir,
    bool reverse,
    const diarization_runtime * rt) {
    const int I = dir.input_size;
    const int H = dir.hidden_size;
    const int G = 4 * H;

    if (static_cast<int>(in.size()) != time_steps * I) {
        throw std::runtime_error("lstm_direction_forward: input size mismatch");
    }

    std::vector<float> xproj(static_cast<size_t>(time_steps) * static_cast<size_t>(G), 0.0f);
    if (rt != nullptr && rt->use_backend_matmul && rt->backend != nullptr) {
        xproj = matmul_rows_backend(rt, in, time_steps, I, dir.w_ih, G);
    } else {
        for (int t = 0; t < time_steps; ++t) {
            const float * x = &in[static_cast<size_t>(t) * static_cast<size_t>(I)];
            float * xp = &xproj[static_cast<size_t>(t) * static_cast<size_t>(G)];
            for (int g = 0; g < G; ++g) {
                const float * wih = &dir.w_ih[static_cast<size_t>(g) * static_cast<size_t>(I)];
                double acc = 0.0;
                for (int i = 0; i < I; ++i) {
                    acc += static_cast<double>(wih[i]) * static_cast<double>(x[i]);
                }
                xp[g] = static_cast<float>(acc);
            }
        }
    }

    return lstm_direction_forward_from_xproj(xproj, time_steps, G, 0, dir, reverse);
}

struct lstm_wih_pair_cache_entry {
    const lstm_layer * layer_ptr = nullptr;
    int I = 0;
    int G = 0;
    std::vector<float> packed_2g_i; // [2G, I] = [fwd_wih; rev_wih]
};

static const std::vector<float> & get_or_pack_lstm_wih_pair(const lstm_layer & layer) {
    static thread_local std::vector<lstm_wih_pair_cache_entry> cache;

    const int I = layer.fwd.input_size;
    const int H = layer.fwd.hidden_size;
    const int G = 4 * H;
    if (layer.rev.input_size != I || layer.rev.hidden_size != H) {
        throw std::runtime_error("get_or_pack_lstm_wih_pair: direction shape mismatch");
    }

    for (const auto & entry : cache) {
        if (entry.layer_ptr == &layer && entry.I == I && entry.G == G) {
            return entry.packed_2g_i;
        }
    }

    lstm_wih_pair_cache_entry entry;
    entry.layer_ptr = &layer;
    entry.I = I;
    entry.G = G;
    entry.packed_2g_i.assign(static_cast<size_t>(2 * G) * static_cast<size_t>(I), 0.0f);

    std::copy(
        layer.fwd.w_ih.begin(),
        layer.fwd.w_ih.end(),
        entry.packed_2g_i.begin());
    std::copy(
        layer.rev.w_ih.begin(),
        layer.rev.w_ih.end(),
        entry.packed_2g_i.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(G) * static_cast<size_t>(I)));

    cache.push_back(std::move(entry));
    return cache.back().packed_2g_i;
}

static std::vector<float> lstm_bidir_forward(
    const std::vector<float> & in,
    int time_steps,
    const lstm_layer & layer,
    const diarization_runtime * rt) {
    const int H = layer.fwd.hidden_size;
    if (layer.rev.hidden_size != H || layer.fwd.input_size != layer.rev.input_size) {
        throw std::runtime_error("lstm_bidir_forward: direction mismatch");
    }

    std::vector<float> fwd;
    std::vector<float> rev;

    if (rt != nullptr && rt->use_backend_matmul && rt->backend != nullptr) {
        const int I = layer.fwd.input_size;
        const int G = 4 * H;
        const std::vector<float> & w2 = get_or_pack_lstm_wih_pair(layer); // [2G, I]
        std::vector<float> xproj2 = matmul_rows_backend(rt, in, time_steps, I, w2, 2 * G);
        fwd = lstm_direction_forward_from_xproj(xproj2, time_steps, 2 * G, 0,   layer.fwd, false);
        rev = lstm_direction_forward_from_xproj(xproj2, time_steps, 2 * G, G,   layer.rev, true);
    } else {
        fwd = lstm_direction_forward(in, time_steps, layer.fwd, false, rt);
        rev = lstm_direction_forward(in, time_steps, layer.rev, true, rt);
    }

    std::vector<float> out(static_cast<size_t>(time_steps) * static_cast<size_t>(2 * H), 0.0f);
    for (int t = 0; t < time_steps; ++t) {
        const float * f = &fwd[static_cast<size_t>(t) * static_cast<size_t>(H)];
        const float * r = &rev[static_cast<size_t>(t) * static_cast<size_t>(H)];
        float * y = &out[static_cast<size_t>(t) * static_cast<size_t>(2 * H)];
        std::copy(f, f + H, y);
        std::copy(r, r + H, y + H);
    }

    return out;
}

static std::vector<float> build_sinc_filters(const segmentation_model & model) {
    const int n_filters = 80;
    const int cutoff = n_filters / 2;
    const int kernel = 251;
    const int half_kernel = kernel / 2;

    if (static_cast<int>(model.sinc_low_hz.size()) != cutoff ||
        static_cast<int>(model.sinc_band_hz.size()) != cutoff ||
        static_cast<int>(model.sinc_window.size()) != half_kernel ||
        static_cast<int>(model.sinc_n.size()) != half_kernel) {
        throw std::runtime_error("build_sinc_filters: unexpected SincNet parameter shapes");
    }

    constexpr float min_low_hz = 50.0f;
    constexpr float min_band_hz = 50.0f;

    std::vector<float> filters(static_cast<size_t>(n_filters) * static_cast<size_t>(kernel), 0.0f);

    for (int i = 0; i < cutoff; ++i) {
        const float low = min_low_hz + std::fabs(model.sinc_low_hz[static_cast<size_t>(i)]);
        float high = low + min_band_hz + std::fabs(model.sinc_band_hz[static_cast<size_t>(i)]);
        high = std::clamp(high, min_low_hz, 0.5f * static_cast<float>(model.sample_rate));
        const float band = high - low;
        const float denom_band = std::max(1e-12f, 2.0f * band);

        std::array<float, 125> left_cos{};
        std::array<float, 125> left_sin{};

        for (int j = 0; j < half_kernel; ++j) {
            const float n = model.sinc_n[static_cast<size_t>(j)];
            const float denom = n / 2.0f;
            const float w = model.sinc_window[static_cast<size_t>(j)];

            left_cos[static_cast<size_t>(j)] = ((std::sin(high * n) - std::sin(low * n)) / denom) * w;
            left_sin[static_cast<size_t>(j)] = ((std::cos(low * n) - std::cos(high * n)) / denom) * w;
        }

        {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(kernel);
            for (int j = 0; j < half_kernel; ++j) {
                filters[base + static_cast<size_t>(j)] = left_cos[static_cast<size_t>(j)] / denom_band;
            }
            filters[base + static_cast<size_t>(half_kernel)] = (2.0f * band) / denom_band;
            for (int j = 0; j < half_kernel; ++j) {
                filters[base + static_cast<size_t>(half_kernel + 1 + j)] =
                    left_cos[static_cast<size_t>(half_kernel - 1 - j)] / denom_band;
            }
        }

        {
            const size_t base = static_cast<size_t>(cutoff + i) * static_cast<size_t>(kernel);
            for (int j = 0; j < half_kernel; ++j) {
                filters[base + static_cast<size_t>(j)] = left_sin[static_cast<size_t>(j)] / denom_band;
            }
            filters[base + static_cast<size_t>(half_kernel)] = 0.0f;
            for (int j = 0; j < half_kernel; ++j) {
                filters[base + static_cast<size_t>(half_kernel + 1 + j)] =
                    -left_sin[static_cast<size_t>(half_kernel - 1 - j)] / denom_band;
            }
        }
    }

    return filters;
}

static void compute_chunk_norm_affine(
    const std::vector<float> & chunk_audio,
    const segmentation_model & model,
    float & out_scale,
    float & out_bias) {
    double mean = 0.0;
    for (float v : chunk_audio) {
        mean += static_cast<double>(v);
    }
    mean /= static_cast<double>(chunk_audio.size());

    double var = 0.0;
    for (float v : chunk_audio) {
        const double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    var /= static_cast<double>(chunk_audio.size());

    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + 1e-5f);
    out_scale = inv_std * model.wav_norm_weight;
    out_bias = -static_cast<float>(mean) * inv_std * model.wav_norm_weight + model.wav_norm_bias;
}

static std::vector<uint8_t> infer_segmentation_chunk(
    const segmentation_model & model,
    const std::vector<float> & chunk_audio,
    const diarization_runtime * rt,
    const feature_map * raw_sinc_chunk = nullptr,
    const std::vector<float> * sinc_filter_sums = nullptr,
    segmentation_infer_profile * profile = nullptr) {
    if (chunk_audio.size() != 160000) {
        throw std::runtime_error("infer_segmentation_chunk expects 10s chunk at 16kHz (160000 samples)");
    }

    const auto t_norm0 = std::chrono::steady_clock::now();
    float norm_scale = 1.0f;
    float norm_bias = 0.0f;
    compute_chunk_norm_affine(chunk_audio, model, norm_scale, norm_bias);
    const auto t_norm1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->norm_affine_sec += std::chrono::duration<double>(t_norm1 - t_norm0).count();
    }

    const auto t_front0 = std::chrono::steady_clock::now();
    feature_map h;
    if (raw_sinc_chunk != nullptr) {
        if (sinc_filter_sums == nullptr || static_cast<int>(sinc_filter_sums->size()) != raw_sinc_chunk->channels) {
            throw std::runtime_error("infer_segmentation_chunk: missing or invalid sinc filter sums");
        }
        if (raw_sinc_chunk->channels != 80) {
            throw std::runtime_error("infer_segmentation_chunk: unexpected cached SincNet channel count");
        }
        h = *raw_sinc_chunk;
        for (int oc = 0; oc < h.channels; ++oc) {
            const float sum_f = (*sinc_filter_sums)[static_cast<size_t>(oc)];
            const size_t base = static_cast<size_t>(oc) * static_cast<size_t>(h.frames);
            for (int t = 0; t < h.frames; ++t) {
                const size_t idx = base + static_cast<size_t>(t);
                const float y = norm_scale * h.data[idx] + norm_bias * sum_f;
                h.data[idx] = std::fabs(y);
            }
        }
    } else {
        std::vector<float> x(chunk_audio.size(), 0.0f);
        for (size_t i = 0; i < chunk_audio.size(); ++i) {
            x[i] = chunk_audio[i] * norm_scale + norm_bias;
        }
        h = conv1d_sinc_stride10(x, model, rt);
        for (float & v : h.data) {
            v = std::fabs(v);
        }
    }
    const auto t_front1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->frontend_sec += std::chrono::duration<double>(t_front1 - t_front0).count();
    }

    const auto t_conv0 = std::chrono::steady_clock::now();
    h = maxpool1d(h, 3, 3);
    instance_norm_1d(h, model.norm0_w, model.norm0_b);
    leaky_relu_inplace(h, 0.01f);

    h = conv1d_valid(h, model.conv1_w, model.conv1_b, 5, 1, rt);
    h = maxpool1d(h, 3, 3);
    instance_norm_1d(h, model.norm1_w, model.norm1_b);
    leaky_relu_inplace(h, 0.01f);

    h = conv1d_valid(h, model.conv2_w, model.conv2_b, 5, 1, rt);
    h = maxpool1d(h, 3, 3);
    instance_norm_1d(h, model.norm2_w, model.norm2_b);
    leaky_relu_inplace(h, 0.01f);
    const auto t_conv1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->conv_stack_sec += std::chrono::duration<double>(t_conv1 - t_conv0).count();
    }

    if (h.channels != 60) {
        throw std::runtime_error("unexpected SincNet output channels");
    }

    const int T = h.frames;
    const auto t_seq0 = std::chrono::steady_clock::now();
    std::vector<float> seq(static_cast<size_t>(T) * 60u, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int f = 0; f < 60; ++f) {
            seq[static_cast<size_t>(t) * 60u + static_cast<size_t>(f)] = h.at(f, t);
        }
    }
    const auto t_seq1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->seq_pack_sec += std::chrono::duration<double>(t_seq1 - t_seq0).count();
    }

    const auto t_lstm0 = std::chrono::steady_clock::now();
    std::vector<float> cur = seq;
    for (int li = 0; li < 4; ++li) {
        cur = lstm_bidir_forward(cur, T, model.lstm[static_cast<size_t>(li)], rt);
    }
    const auto t_lstm1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->lstm_sec += std::chrono::duration<double>(t_lstm1 - t_lstm0).count();
    }

    const auto t_head0 = std::chrono::steady_clock::now();
    cur = linear_forward(cur, T, 256, model.linear0_w, model.linear0_b, rt);
    leaky_relu_inplace(cur, 0.01f);

    cur = linear_forward(cur, T, 128, model.linear1_w, model.linear1_b, rt);
    leaky_relu_inplace(cur, 0.01f);

    std::vector<float> logits = linear_forward(cur, T, 128, model.cls_w, model.cls_b, rt);
    const auto t_head1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->linear_head_sec += std::chrono::duration<double>(t_head1 - t_head0).count();
    }

    static const int pset_map[7][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
    };

    const auto t_dec0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> out(static_cast<size_t>(T) * 3u, 0);
    for (int t = 0; t < T; ++t) {
        const float * row = &logits[static_cast<size_t>(t) * 7u];
        float maxv = row[0];
        for (int i = 1; i < 7; ++i) {
            maxv = std::max(maxv, row[i]);
        }
        float sumexp = 0.0f;
        for (int i = 0; i < 7; ++i) {
            sumexp += std::exp(row[i] - maxv);
        }
        const float lse = maxv + std::log(std::max(sumexp, 1e-12f));

        int argm = 0;
        float best = row[0] - lse;
        for (int i = 1; i < 7; ++i) {
            const float v = row[i] - lse;
            if (v > best) {
                best = v;
                argm = i;
            }
        }
        out[static_cast<size_t>(t) * 3u + 0u] = static_cast<uint8_t>(pset_map[argm][0]);
        out[static_cast<size_t>(t) * 3u + 1u] = static_cast<uint8_t>(pset_map[argm][1]);
        out[static_cast<size_t>(t) * 3u + 2u] = static_cast<uint8_t>(pset_map[argm][2]);
    }
    const auto t_dec1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->decode_sec += std::chrono::duration<double>(t_dec1 - t_dec0).count();
        profile->chunks += 1;
    }

    return out;
}

static std::vector<float> aggregate_scores(
    const std::vector<std::vector<float>> & chunk_scores,
    int chunk_frames,
    int dim,
    double chunk_duration,
    double chunk_step,
    const sliding_window & frames,
    bool skip_average) {
    const int num_chunks = static_cast<int>(chunk_scores.size());
    if (num_chunks == 0) {
        return {};
    }

    const int num_frames = frames.closest_frame(
        0.0 + chunk_duration + static_cast<double>(num_chunks - 1) * chunk_step + 0.5 * frames.duration) + 1;

    std::vector<float> sum(static_cast<size_t>(num_frames) * static_cast<size_t>(dim), 0.0f);
    std::vector<float> cnt(static_cast<size_t>(num_frames) * static_cast<size_t>(dim), 0.0f);

    for (int c = 0; c < num_chunks; ++c) {
        const auto & sc = chunk_scores[static_cast<size_t>(c)];
        if (static_cast<int>(sc.size()) != chunk_frames * dim) {
            throw std::runtime_error("aggregate_scores: chunk score shape mismatch");
        }

        const double chunk_start = static_cast<double>(c) * chunk_step;
        const int start_frame = frames.closest_frame(chunk_start + 0.5 * frames.duration);

        for (int t = 0; t < chunk_frames; ++t) {
            const int out_t = start_frame + t;
            if (out_t < 0 || out_t >= num_frames) {
                continue;
            }
            for (int d = 0; d < dim; ++d) {
                const float v = sc[static_cast<size_t>(t) * static_cast<size_t>(dim) + static_cast<size_t>(d)];
                const size_t idx = static_cast<size_t>(out_t) * static_cast<size_t>(dim) + static_cast<size_t>(d);
                sum[idx] += v;
                cnt[idx] += 1.0f;
            }
        }
    }

    if (!skip_average) {
        for (size_t i = 0; i < sum.size(); ++i) {
            const float denom = std::max(cnt[i], 1e-12f);
            sum[i] /= denom;
        }
    }

    return sum;
}

static std::vector<diar_seg> binary_frames_to_segments(
    const std::vector<uint8_t> & binary,
    int num_frames,
    int num_speakers,
    const sliding_window & frames,
    double min_duration_off_sec) {
    std::vector<diar_seg> out;

    for (int sp = 0; sp < num_speakers; ++sp) {
        std::vector<std::pair<double, double>> intervals;
        intervals.reserve(static_cast<size_t>(num_frames));

        for (int t = 0; t < num_frames; ++t) {
            if (binary[static_cast<size_t>(t) * static_cast<size_t>(num_speakers) + static_cast<size_t>(sp)] == 0) {
                continue;
            }
            const double start = frames.start + static_cast<double>(t) * frames.step;
            const double end = start + frames.duration;
            intervals.emplace_back(start, end);
        }

        if (intervals.empty()) {
            continue;
        }

        std::sort(intervals.begin(), intervals.end());
        std::vector<std::pair<double, double>> merged;
        merged.reserve(intervals.size());
        merged.push_back(intervals[0]);

        for (size_t i = 1; i < intervals.size(); ++i) {
            auto & cur = merged.back();
            const auto & nxt = intervals[i];
            const double gap = nxt.first - cur.second;
            if (gap <= min_duration_off_sec + 1e-9) {
                cur.second = std::max(cur.second, nxt.second);
            } else {
                merged.push_back(nxt);
            }
        }

        for (const auto & iv : merged) {
            if (iv.second <= iv.first) {
                continue;
            }
            out.push_back({sp, iv.first, iv.second});
        }
    }

    std::sort(out.begin(), out.end(), [](const diar_seg & a, const diar_seg & b) {
        if (a.start_sec != b.start_sec) {
            return a.start_sec < b.start_sec;
        }
        if (a.end_sec != b.end_sec) {
            return a.end_sec < b.end_sec;
        }
        return a.speaker < b.speaker;
    });

    return out;
}

static void relabel_speakers_compact(std::vector<diar_seg> & regular, std::vector<diar_seg> & exclusive) {
    std::set<int> all_ids;
    for (const auto & s : regular) {
        all_ids.insert(s.speaker);
    }
    for (const auto & s : exclusive) {
        all_ids.insert(s.speaker);
    }
    if (all_ids.empty()) {
        return;
    }

    std::map<int, double> duration_by_id;
    const auto & source_for_duration = exclusive.empty() ? regular : exclusive;
    for (const auto & s : source_for_duration) {
        duration_by_id[s.speaker] += std::max(0.0, s.end_sec - s.start_sec);
    }

    std::vector<int> ordered_ids(all_ids.begin(), all_ids.end());
    std::sort(ordered_ids.begin(), ordered_ids.end(), [&](int a, int b) {
        const double da = duration_by_id.count(a) ? duration_by_id[a] : 0.0;
        const double db = duration_by_id.count(b) ? duration_by_id[b] : 0.0;
        if (da != db) {
            return da > db;
        }
        return a < b;
    });

    std::map<int, int> remap;
    int next_id = 0;
    for (int id : ordered_ids) {
        remap[id] = next_id++;
    }

    for (auto & s : regular) {
        s.speaker = remap[s.speaker];
    }
    for (auto & s : exclusive) {
        s.speaker = remap[s.speaker];
    }
}

static diarization_output diarize_from_segmentation(
    const segmentation_model & model,
    const std::vector<float> & audio,
    const diarization_runtime * rt,
    int num_speakers,
    int min_speakers,
    int max_speakers,
    double min_duration_off_sec) {
    constexpr int chunk_samples = 160000; // 10s
    constexpr int step_samples = 16000;   // 1s

    if (audio.empty()) {
        throw std::runtime_error("empty audio signal");
    }

    const auto diar_t0 = std::chrono::steady_clock::now();

    std::vector<std::vector<float>> chunk_signals;

    const int64_t n_samples = static_cast<int64_t>(audio.size());
    int64_t num_full_chunks = 0;
    if (n_samples >= chunk_samples) {
        num_full_chunks = (n_samples - chunk_samples) / step_samples + 1;
    }

    for (int64_t c = 0; c < num_full_chunks; ++c) {
        const int64_t start = c * step_samples;
        std::vector<float> chunk(static_cast<size_t>(chunk_samples), 0.0f);
        std::copy(
            audio.begin() + static_cast<ptrdiff_t>(start),
            audio.begin() + static_cast<ptrdiff_t>(start + chunk_samples),
            chunk.begin());
        chunk_signals.push_back(std::move(chunk));
    }

    const bool has_last_chunk = (n_samples < chunk_samples) || ((n_samples - chunk_samples) % step_samples > 0);
    if (has_last_chunk) {
        const int64_t start = num_full_chunks * step_samples;
        std::vector<float> chunk(static_cast<size_t>(chunk_samples), 0.0f);
        const int64_t remain = std::max<int64_t>(0, n_samples - start);
        if (remain > 0) {
            std::copy(
                audio.begin() + static_cast<ptrdiff_t>(start),
                audio.begin() + static_cast<ptrdiff_t>(start + remain),
                chunk.begin());
        }
        chunk_signals.push_back(std::move(chunk));
    }

    const int num_chunks = static_cast<int>(chunk_signals.size());
    if (num_chunks <= 0) {
        throw std::runtime_error("failed to construct audio chunks");
    }

    const auto diar_t1_build = std::chrono::steady_clock::now();

    std::vector<std::vector<uint8_t>> local_bin;
    local_bin.reserve(static_cast<size_t>(num_chunks));

    const int sinc_kernel = 251;
    const int sinc_stride = 10;
    const int sinc_out_ch = 80;
    const int sinc_frames_per_chunk = (chunk_samples - sinc_kernel) / sinc_stride + 1;
    const int sinc_step_frames = step_samples / sinc_stride;

    const bool use_sinc_rolling_cache =
        rt != nullptr &&
        rt->use_backend_matmul &&
        rt->backend != nullptr &&
        static_cast<int>(model.sinc_filters.size()) == sinc_out_ch * sinc_kernel &&
        sinc_step_frames > 0 &&
        sinc_frames_per_chunk > sinc_step_frames;

    std::vector<float> padded_audio;
    std::vector<float> sinc_filter_sums;
    feature_map raw_sinc_cache;

    if (use_sinc_rolling_cache) {
        const int64_t total_needed_samples =
            static_cast<int64_t>(num_chunks - 1) * static_cast<int64_t>(step_samples) + static_cast<int64_t>(chunk_samples);
        padded_audio.assign(static_cast<size_t>(total_needed_samples), 0.0f);
        const size_t ncopy = std::min(audio.size(), padded_audio.size());
        if (ncopy > 0) {
            std::copy(audio.begin(), audio.begin() + static_cast<ptrdiff_t>(ncopy), padded_audio.begin());
        }

        sinc_filter_sums.assign(static_cast<size_t>(sinc_out_ch), 0.0f);
        for (int oc = 0; oc < sinc_out_ch; ++oc) {
            const size_t base = static_cast<size_t>(oc) * static_cast<size_t>(sinc_kernel);
            double acc = 0.0;
            for (int k = 0; k < sinc_kernel; ++k) {
                acc += static_cast<double>(model.sinc_filters[base + static_cast<size_t>(k)]);
            }
            sinc_filter_sums[static_cast<size_t>(oc)] = static_cast<float>(acc);
        }

        raw_sinc_cache = conv1d_sinc_stride10_window(padded_audio, 0, sinc_frames_per_chunk, model, rt);
    }

    segmentation_infer_profile infer_profile;
    int chunk_frames = -1;
    const auto diar_t2_infer_start = std::chrono::steady_clock::now();
    for (int c = 0; c < num_chunks; ++c) {
        const auto & ch = chunk_signals[static_cast<size_t>(c)];
        std::vector<uint8_t> pred;
        if (use_sinc_rolling_cache) {
            pred = infer_segmentation_chunk(model, ch, rt, &raw_sinc_cache, &sinc_filter_sums, &infer_profile);
        } else {
            pred = infer_segmentation_chunk(model, ch, rt, nullptr, nullptr, &infer_profile);
        }
        if (pred.size() % 3u != 0u) {
            throw std::runtime_error("invalid segmentation output shape");
        }
        const int f = static_cast<int>(pred.size() / 3u);
        if (chunk_frames < 0) {
            chunk_frames = f;
        } else if (chunk_frames != f) {
            throw std::runtime_error("inconsistent chunk frame count");
        }
        local_bin.push_back(std::move(pred));

        if (use_sinc_rolling_cache && (c + 1) < num_chunks) {
            const int keep_frames = sinc_frames_per_chunk - sinc_step_frames;
            const int tail_start_frame = (c + 1) * sinc_step_frames + keep_frames;
            feature_map tail = conv1d_sinc_stride10_window(
                padded_audio,
                tail_start_frame,
                sinc_step_frames,
                model,
                rt);
            if (tail.channels != raw_sinc_cache.channels || tail.frames != sinc_step_frames) {
                throw std::runtime_error("invalid rolling SincNet tail shape");
            }

            for (int oc = 0; oc < raw_sinc_cache.channels; ++oc) {
                float * dst = &raw_sinc_cache.data[static_cast<size_t>(oc) * static_cast<size_t>(sinc_frames_per_chunk)];
                if (keep_frames > 0) {
                    std::memmove(
                        dst,
                        dst + static_cast<size_t>(sinc_step_frames),
                        static_cast<size_t>(keep_frames) * sizeof(float));
                }
                const float * src = &tail.data[static_cast<size_t>(oc) * static_cast<size_t>(sinc_step_frames)];
                std::copy(src, src + static_cast<size_t>(sinc_step_frames), dst + static_cast<size_t>(keep_frames));
            }
        }
    }

    if (chunk_frames <= 0) {
        throw std::runtime_error("no segmentation frames produced");
    }
    const auto diar_t3_infer_end = std::chrono::steady_clock::now();

    int global_limit = 8;
    if (num_speakers > 0) {
        global_limit = std::max(1, num_speakers);
    } else if (max_speakers > 0) {
        global_limit = std::max(1, max_speakers);
    }

    std::vector<std::array<int, 3>> local_to_global(static_cast<size_t>(num_chunks));
    for (auto & m : local_to_global) {
        m = {-1, -1, -1};
    }

    int global_count = 0;

    auto local_activity_count = [&](int c, int s) {
        int n = 0;
        const auto & a = local_bin[static_cast<size_t>(c)];
        for (int t = 0; t < chunk_frames; ++t) {
            n += static_cast<int>(a[static_cast<size_t>(t) * 3u + static_cast<size_t>(s)] != 0);
        }
        return n;
    };

    {
        std::array<int, 3> order = {0, 1, 2};
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return local_activity_count(0, a) > local_activity_count(0, b);
        });

        for (int si : order) {
            if (local_activity_count(0, si) <= 0) {
                continue;
            }
            if (global_count < global_limit) {
                local_to_global[0][static_cast<size_t>(si)] = global_count;
                global_count++;
            } else {
                local_to_global[0][static_cast<size_t>(si)] = std::max(0, global_limit - 1);
            }
        }

        if (global_count == 0) {
            local_to_global[0][0] = 0;
            global_count = 1;
        }
    }

    const double frame_center0 = 0.5 * model.frame_duration_sec;

    for (int c = 1; c < num_chunks; ++c) {
        const auto & prev = local_bin[static_cast<size_t>(c - 1)];
        const auto & cur = local_bin[static_cast<size_t>(c)];

        std::vector<uint8_t> prev_global_active(static_cast<size_t>(global_count) * static_cast<size_t>(chunk_frames), 0);
        for (int t = 0; t < chunk_frames; ++t) {
            for (int s = 0; s < 3; ++s) {
                if (prev[static_cast<size_t>(t) * 3u + static_cast<size_t>(s)] == 0) {
                    continue;
                }
                const int g = local_to_global[static_cast<size_t>(c - 1)][static_cast<size_t>(s)];
                if (g >= 0 && g < global_count) {
                    prev_global_active[static_cast<size_t>(g) * static_cast<size_t>(chunk_frames) + static_cast<size_t>(t)] = 1;
                }
            }
        }

        std::vector<float> score(static_cast<size_t>(3 * std::max(global_count, 1)), 0.0f);
        const double prev_start = static_cast<double>(c - 1) * 1.0;
        const double cur_start = static_cast<double>(c) * 1.0;

        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < chunk_frames; ++t) {
                if (cur[static_cast<size_t>(t) * 3u + static_cast<size_t>(s)] == 0) {
                    continue;
                }

                const double abs_center = cur_start + frame_center0 + static_cast<double>(t) * model.frame_step_sec;
                const double prev_rel = abs_center - prev_start;
                const int tp = static_cast<int>(std::llround((prev_rel - frame_center0) / model.frame_step_sec));
                if (tp < 0 || tp >= chunk_frames) {
                    continue;
                }

                for (int g = 0; g < global_count; ++g) {
                    const uint8_t active = prev_global_active[static_cast<size_t>(g) * static_cast<size_t>(chunk_frames) + static_cast<size_t>(tp)];
                    if (active) {
                        score[static_cast<size_t>(s) * static_cast<size_t>(std::max(global_count, 1)) + static_cast<size_t>(g)] += 1.0f;
                    }
                }
            }
        }

        std::array<int, 3> order = {0, 1, 2};
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return local_activity_count(c, a) > local_activity_count(c, b);
        });

        std::vector<uint8_t> used(static_cast<size_t>(global_count), 0);

        for (int s : order) {
            if (local_activity_count(c, s) <= 0) {
                local_to_global[static_cast<size_t>(c)][static_cast<size_t>(s)] = -1;
                continue;
            }

            int best_g = -1;
            float best_sc = -1.0f;
            for (int g = 0; g < global_count; ++g) {
                if (used[static_cast<size_t>(g)] != 0) {
                    continue;
                }
                const float sc = score[static_cast<size_t>(s) * static_cast<size_t>(std::max(global_count, 1)) + static_cast<size_t>(g)];
                if (sc > best_sc) {
                    best_sc = sc;
                    best_g = g;
                }
            }

            if (best_g >= 0) {
                local_to_global[static_cast<size_t>(c)][static_cast<size_t>(s)] = best_g;
                used[static_cast<size_t>(best_g)] = 1;
                continue;
            }

            if (global_count < global_limit) {
                const int ng = global_count;
                global_count++;
                used.resize(static_cast<size_t>(global_count), 0);
                used[static_cast<size_t>(ng)] = 1;
                local_to_global[static_cast<size_t>(c)][static_cast<size_t>(s)] = ng;
                continue;
            }

            best_g = 0;
            best_sc = -1.0f;
            for (int g = 0; g < global_count; ++g) {
                const float sc = score[static_cast<size_t>(s) * static_cast<size_t>(std::max(global_count, 1)) + static_cast<size_t>(g)];
                if (sc > best_sc) {
                    best_sc = sc;
                    best_g = g;
                }
            }
            local_to_global[static_cast<size_t>(c)][static_cast<size_t>(s)] = best_g;
        }
    }
    const auto diar_t4_mapping_end = std::chrono::steady_clock::now();

    std::vector<std::vector<float>> chunk_global(static_cast<size_t>(num_chunks));
    std::vector<std::vector<float>> chunk_count(static_cast<size_t>(num_chunks));
    for (int c = 0; c < num_chunks; ++c) {
        std::vector<float> g(static_cast<size_t>(chunk_frames) * static_cast<size_t>(global_count), 0.0f);
        std::vector<float> cnt(static_cast<size_t>(chunk_frames), 0.0f);

        const auto & loc = local_bin[static_cast<size_t>(c)];
        for (int t = 0; t < chunk_frames; ++t) {
            int csum = 0;
            for (int s = 0; s < 3; ++s) {
                const uint8_t v = loc[static_cast<size_t>(t) * 3u + static_cast<size_t>(s)];
                if (v == 0) {
                    continue;
                }
                csum += 1;
                const int gid = local_to_global[static_cast<size_t>(c)][static_cast<size_t>(s)];
                if (gid >= 0 && gid < global_count) {
                    g[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(gid)] = 1.0f;
                }
            }
            cnt[static_cast<size_t>(t)] = static_cast<float>(csum);
        }

        chunk_global[static_cast<size_t>(c)] = std::move(g);
        chunk_count[static_cast<size_t>(c)] = std::move(cnt);
    }

    const sliding_window frames = {
        0.0,
        model.frame_duration_sec,
        model.frame_step_sec,
    };

    std::vector<float> count_agg = aggregate_scores(
        chunk_count,
        chunk_frames,
        1,
        10.0,
        1.0,
        frames,
        false);

    const int total_frames = static_cast<int>(count_agg.size());
    std::vector<uint8_t> count_uint(static_cast<size_t>(total_frames), 0);

    for (int t = 0; t < total_frames; ++t) {
        int c = static_cast<int>(std::llround(count_agg[static_cast<size_t>(t)]));
        c = std::max(0, c);
        if (max_speakers > 0) {
            c = std::min(c, max_speakers);
        }
        if (num_speakers > 0) {
            c = std::min(c, num_speakers);
        }
        count_uint[static_cast<size_t>(t)] = static_cast<uint8_t>(c);
    }

    if (min_speakers > 0) {
        for (auto & c : count_uint) {
            c = static_cast<uint8_t>(std::max<int>(c, min_speakers));
        }
    }

    std::vector<float> act_agg = aggregate_scores(
        chunk_global,
        chunk_frames,
        global_count,
        10.0,
        1.0,
        frames,
        true);
    const auto diar_t5_agg_end = std::chrono::steady_clock::now();

    if (static_cast<int>(act_agg.size()) != total_frames * global_count) {
        throw std::runtime_error("activation aggregation shape mismatch");
    }

    std::vector<uint8_t> regular_bin(static_cast<size_t>(total_frames) * static_cast<size_t>(global_count), 0);
    std::vector<int> idx(static_cast<size_t>(global_count), 0);
    std::iota(idx.begin(), idx.end(), 0);

    for (int t = 0; t < total_frames; ++t) {
        const int c = std::min<int>(count_uint[static_cast<size_t>(t)], global_count);
        if (c <= 0) {
            continue;
        }

        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const float va = act_agg[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(a)];
            const float vb = act_agg[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(b)];
            if (va != vb) {
                return va > vb;
            }
            return a < b;
        });

        for (int k = 0; k < c; ++k) {
            regular_bin[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(idx[static_cast<size_t>(k)])] = 1;
        }
    }

    std::vector<uint8_t> excl_count = count_uint;
    for (auto & c : excl_count) {
        c = static_cast<uint8_t>(std::min<int>(c, 1));
    }

    std::vector<uint8_t> exclusive_bin(static_cast<size_t>(total_frames) * static_cast<size_t>(global_count), 0);
    for (int t = 0; t < total_frames; ++t) {
        const int c = std::min<int>(excl_count[static_cast<size_t>(t)], global_count);
        if (c <= 0) {
            continue;
        }

        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const float va = act_agg[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(a)];
            const float vb = act_agg[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(b)];
            if (va != vb) {
                return va > vb;
            }
            return a < b;
        });

        for (int k = 0; k < c; ++k) {
            exclusive_bin[static_cast<size_t>(t) * static_cast<size_t>(global_count) + static_cast<size_t>(idx[static_cast<size_t>(k)])] = 1;
        }
    }

    std::vector<diar_seg> regular = binary_frames_to_segments(
        regular_bin,
        total_frames,
        global_count,
        frames,
        min_duration_off_sec);

    std::vector<diar_seg> exclusive = binary_frames_to_segments(
        exclusive_bin,
        total_frames,
        global_count,
        frames,
        min_duration_off_sec);
    const auto diar_t6_bin_end = std::chrono::steady_clock::now();

    relabel_speakers_compact(regular, exclusive);

    std::set<int> spk;
    for (const auto & s : regular) {
        spk.insert(s.speaker);
    }

    diarization_output out;
    out.regular = std::move(regular);
    out.exclusive = std::move(exclusive);
    out.speaker_count = static_cast<int>(spk.size());
    out.num_chunks = num_chunks;
    out.num_frames_per_chunk = chunk_frames;
    out.elapsed_build_chunks_sec = std::chrono::duration<double>(diar_t1_build - diar_t0).count();
    out.elapsed_segmentation_infer_sec = std::chrono::duration<double>(diar_t3_infer_end - diar_t2_infer_start).count();
    out.elapsed_global_mapping_sec = std::chrono::duration<double>(diar_t4_mapping_end - diar_t3_infer_end).count();
    out.elapsed_aggregation_sec = std::chrono::duration<double>(diar_t5_agg_end - diar_t4_mapping_end).count();
    out.elapsed_binarize_segments_sec = std::chrono::duration<double>(diar_t6_bin_end - diar_t5_agg_end).count();
    out.infer_norm_affine_sec = infer_profile.norm_affine_sec;
    out.infer_frontend_sec = infer_profile.frontend_sec;
    out.infer_conv_stack_sec = infer_profile.conv_stack_sec;
    out.infer_seq_pack_sec = infer_profile.seq_pack_sec;
    out.infer_lstm_sec = infer_profile.lstm_sec;
    out.infer_linear_head_sec = infer_profile.linear_head_sec;
    out.infer_decode_sec = infer_profile.decode_sec;
    out.infer_profiled_chunks = infer_profile.chunks;
    return out;
}

static json extract_speaker_embeddings(
    const embedding_model & model,
    const std::vector<float> & audio,
    const diarization_output & diar,
    const diarization_runtime * rt,
    double min_segment_duration_sec,
    int max_segments_per_speaker) {
    if (audio.empty()) {
        throw std::runtime_error("extract_speaker_embeddings: empty audio");
    }

    const int emb_sr = model.sample_rate;
    const int min_num_samples = std::max(1, model.min_num_samples);
    const int max_per_speaker = std::max(1, max_segments_per_speaker);
    const int total_samples = static_cast<int>(audio.size());

    std::map<int, std::vector<std::pair<int, int>>> by_speaker;
    std::map<int, double> dur_by_speaker;
    std::map<int, int> seg_count_by_speaker;

    for (const auto & seg : diar.regular) {
        const int spk = seg.speaker;
        const double start = std::max(0.0, seg.start_sec);
        const double end = std::max(start, seg.end_sec);
        const double duration = end - start;
        if (duration < min_segment_duration_sec) {
            continue;
        }

        int start_i = static_cast<int>(std::llround(start * static_cast<double>(emb_sr)));
        int end_i = static_cast<int>(std::llround(end * static_cast<double>(emb_sr)));
        start_i = std::max(0, std::min(start_i, total_samples));
        end_i = std::max(start_i, std::min(end_i, total_samples));

        if (end_i - start_i < min_num_samples) {
            const double center = 0.5 * static_cast<double>(start_i + end_i);
            const int half = static_cast<int>(std::ceil(static_cast<double>(min_num_samples) / 2.0));
            start_i = std::max(0, static_cast<int>(center) - half);
            end_i = std::min(total_samples, start_i + min_num_samples);
            start_i = std::max(0, end_i - min_num_samples);
        }
        if (end_i - start_i < min_num_samples) {
            continue;
        }

        auto & ranges = by_speaker[spk];
        if (static_cast<int>(ranges.size()) >= max_per_speaker) {
            continue;
        }

        ranges.push_back({start_i, end_i});
        dur_by_speaker[spk] += static_cast<double>(end_i - start_i) / static_cast<double>(emb_sr);
        seg_count_by_speaker[spk] += 1;
    }

    std::map<std::pair<int, int>, std::vector<float>> embedding_cache;
    json speakers_payload = json::array();
    for (const auto & kv : by_speaker) {
        const int spk = kv.first;
        const auto & ranges = kv.second;
        if (ranges.empty()) {
            continue;
        }

        std::vector<float> accum(static_cast<size_t>(model.embed_dim), 0.0f);
        int emb_count = 0;

        for (const auto & r : ranges) {
            const std::pair<int, int> key = {r.first, r.second};
            auto emb_it = embedding_cache.find(key);
            if (emb_it == embedding_cache.end()) {
                std::vector<float> emb = infer_embedding_range(model, audio, r.first, r.second, rt);
                emb_it = embedding_cache.emplace(key, std::move(emb)).first;
            }

            const std::vector<float> & emb = emb_it->second;
            if (emb.size() != static_cast<size_t>(model.embed_dim)) {
                continue;
            }

            for (size_t i = 0; i < emb.size(); ++i) {
                accum[i] += emb[i];
            }
            emb_count += 1;
        }

        if (emb_count <= 0) {
            continue;
        }

        for (float & v : accum) {
            v /= static_cast<float>(emb_count);
        }

        double norm2 = 0.0;
        for (float v : accum) {
            norm2 += static_cast<double>(v) * static_cast<double>(v);
        }
        const double norm = std::sqrt(norm2);
        if (norm > 0.0) {
            for (float & v : accum) {
                v = static_cast<float>(static_cast<double>(v) / norm);
            }
        }

        json emb_json = json::array();
        for (float v : accum) {
            emb_json.push_back(round8(static_cast<double>(v)));
        }

        speakers_payload.push_back({
            {"speaker", speaker_name(spk)},
            {"embedding", emb_json},
            {"embedding_dim", static_cast<int64_t>(accum.size())},
            {"num_embedding_segments", static_cast<int64_t>(seg_count_by_speaker[spk])},
            {"embedding_duration_sec", round3(dur_by_speaker[spk])},
        });
    }

    return {
        {"enabled", true},
        {"status", "ok"},
        {"model", model.architecture_class},
        {"sample_rate", emb_sr},
        {"metric", "cosine"},
        {"embedding_dim", model.embed_dim},
        {"min_num_samples", min_num_samples},
        {"speakers", speakers_payload},
    };
}

static segmentation_model load_segmentation_model(const std::filesystem::path & gguf_path) {
    ggml_context * tctx = nullptr;
    gguf_init_params params = {};
    params.no_alloc = false;
    params.ctx = &tctx;

    gguf_context * gctx = gguf_init_from_file(gguf_path.string().c_str(), params);
    if (gctx == nullptr || tctx == nullptr) {
        if (gctx != nullptr) {
            gguf_free(gctx);
        }
        throw std::runtime_error("failed to load GGUF: " + gguf_path.string());
    }

    auto load = [&](const std::string & name) {
        return load_tensor_as_f32(tctx, name, gguf_path);
    };

    segmentation_model m;

    const int key_sr = gguf_find_key(gctx, "pyannote.sample_rate");
    if (key_sr >= 0) {
        m.sample_rate = static_cast<int>(gguf_get_val_u32(gctx, key_sr));
    }

    {
        const tensor_f32 w = load("pyannote.segmentation.sincnet.wav_norm1d.weight");
        const tensor_f32 b = load("pyannote.segmentation.sincnet.wav_norm1d.bias");
        if (w.data.size() != 1 || b.data.size() != 1) {
            throw std::runtime_error("invalid wav_norm tensor sizes");
        }
        m.wav_norm_weight = w.data[0];
        m.wav_norm_bias = b.data[0];
    }

    {
        const tensor_f32 low = load("pyannote.segmentation.sincnet.conv1d.0.filterbank.low_hz_");
        const tensor_f32 band = load("pyannote.segmentation.sincnet.conv1d.0.filterbank.band_hz_");
        const tensor_f32 window = load("pyannote.segmentation.sincnet.conv1d.0.filterbank.window_");
        const tensor_f32 n = load("pyannote.segmentation.sincnet.conv1d.0.filterbank.n_");

        m.sinc_low_hz = low.data;
        m.sinc_band_hz = band.data;
        m.sinc_window = window.data;
        m.sinc_n = n.data;
    }

    m.conv1_w = load("pyannote.segmentation.sincnet.conv1d.1.weight");
    m.conv1_b = load("pyannote.segmentation.sincnet.conv1d.1.bias");
    m.conv2_w = load("pyannote.segmentation.sincnet.conv1d.2.weight");
    m.conv2_b = load("pyannote.segmentation.sincnet.conv1d.2.bias");

    m.norm0_w = load("pyannote.segmentation.sincnet.norm1d.0.weight");
    m.norm0_b = load("pyannote.segmentation.sincnet.norm1d.0.bias");
    m.norm1_w = load("pyannote.segmentation.sincnet.norm1d.1.weight");
    m.norm1_b = load("pyannote.segmentation.sincnet.norm1d.1.bias");
    m.norm2_w = load("pyannote.segmentation.sincnet.norm1d.2.weight");
    m.norm2_b = load("pyannote.segmentation.sincnet.norm1d.2.bias");

    for (int li = 0; li < 4; ++li) {
        auto & layer = m.lstm[static_cast<size_t>(li)];
        const std::string base = "pyannote.segmentation.lstm.";

        auto load_dir = [&](lstm_direction & d, const std::string & suffix) {
            tensor_f32 wih = load(base + "weight_ih_l" + std::to_string(li) + suffix);
            tensor_f32 whh = load(base + "weight_hh_l" + std::to_string(li) + suffix);
            tensor_f32 bih = load(base + "bias_ih_l" + std::to_string(li) + suffix);
            tensor_f32 bhh = load(base + "bias_hh_l" + std::to_string(li) + suffix);

            d.input_size = static_cast<int>(wih.ne0);
            const int gates = static_cast<int>(wih.ne1);
            d.hidden_size = gates / 4;

            if (gates % 4 != 0 || static_cast<int>(whh.ne0) != d.hidden_size || static_cast<int>(whh.ne1) != gates) {
                throw std::runtime_error("invalid LSTM tensor dimensions");
            }

            d.w_ih = std::move(wih.data);
            d.w_hh = std::move(whh.data);
            d.b_ih = std::move(bih.data);
            d.b_hh = std::move(bhh.data);

            if (static_cast<int>(d.b_ih.size()) != gates || static_cast<int>(d.b_hh.size()) != gates) {
                throw std::runtime_error("invalid LSTM bias dimensions");
            }
        };

        load_dir(layer.fwd, "");
        load_dir(layer.rev, "_reverse");

        if (layer.fwd.hidden_size != 128 || layer.rev.hidden_size != 128) {
            throw std::runtime_error("unexpected LSTM hidden size");
        }
        if (li == 0) {
            if (layer.fwd.input_size != 60) {
                throw std::runtime_error("unexpected LSTM input size at layer 0");
            }
        } else if (layer.fwd.input_size != 256) {
            throw std::runtime_error("unexpected LSTM input size at deeper layer");
        }
    }

    m.linear0_w = load("pyannote.segmentation.linear.0.weight");
    m.linear0_b = load("pyannote.segmentation.linear.0.bias");
    m.linear1_w = load("pyannote.segmentation.linear.1.weight");
    m.linear1_b = load("pyannote.segmentation.linear.1.bias");
    m.cls_w = load("pyannote.segmentation.classifier.weight");
    m.cls_b = load("pyannote.segmentation.classifier.bias");

    m.sinc_filters = build_sinc_filters(m);

    gguf_free(gctx);
    ggml_free(tctx);
    return m;
}

int llama_pyannote_diarize_main(int argc, char ** argv) {
    std::filesystem::path audio_path;
    std::filesystem::path segmentation_gguf;
    std::filesystem::path embedding_gguf;
    std::filesystem::path output_dir = "diarization";
    std::string pipeline_dir;
    std::string device = "auto";

    int num_speakers = -1;
    int min_speakers = -1;
    int max_speakers = -1;

    bool offline = false;
    bool progress = false;
    bool list_devices = false;
    bool export_speaker_embeddings = false;
    double embedding_min_segment_duration_sec = 0.35;
    int embedding_max_segments_per_speaker = 64;
    double min_duration_off_sec = 0.0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--audio" && i + 1 < argc) {
            audio_path = argv[++i];
            continue;
        }
        if (arg == "--segmentation-gguf" && i + 1 < argc) {
            segmentation_gguf = argv[++i];
            continue;
        }
        if (arg == "--embedding-gguf" && i + 1 < argc) {
            embedding_gguf = argv[++i];
            continue;
        }
        if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
            continue;
        }
        if (arg == "--pipeline-dir" && i + 1 < argc) {
            pipeline_dir = argv[++i];
            continue;
        }
        if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
            continue;
        }
        if (arg == "--num-speakers" && i + 1 < argc) {
            num_speakers = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--min-speakers" && i + 1 < argc) {
            min_speakers = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--max-speakers" && i + 1 < argc) {
            max_speakers = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--offline") {
            offline = true;
            continue;
        }
        if (arg == "--progress") {
            progress = true;
            continue;
        }
        if (arg == "--list-devices") {
            list_devices = true;
            continue;
        }
        if (arg == "--export-speaker-embeddings") {
            export_speaker_embeddings = true;
            continue;
        }
        if (arg == "--embedding-min-segment-duration-sec" && i + 1 < argc) {
            embedding_min_segment_duration_sec = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--embedding-max-segments-per-speaker" && i + 1 < argc) {
            embedding_max_segments_per_speaker = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--min-duration-off" && i + 1 < argc) {
            min_duration_off_sec = std::stod(argv[++i]);
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        if (list_devices) {
            print_available_devices(std::cout);
            return 0;
        }

        if (audio_path.empty() || segmentation_gguf.empty()) {
            print_usage(argv[0]);
            return 2;
        }

        audio_path = std::filesystem::absolute(audio_path);
        segmentation_gguf = std::filesystem::absolute(segmentation_gguf);
        if (!embedding_gguf.empty()) {
            embedding_gguf = std::filesystem::absolute(embedding_gguf);
        }
        output_dir = std::filesystem::absolute(output_dir);

        if (!std::filesystem::exists(audio_path)) {
            throw std::runtime_error("audio file not found: " + audio_path.string());
        }
        if (!std::filesystem::exists(segmentation_gguf)) {
            throw std::runtime_error("segmentation GGUF not found: " + segmentation_gguf.string());
        }
        if (export_speaker_embeddings) {
            if (embedding_gguf.empty()) {
                throw std::runtime_error("missing required --embedding-gguf when --export-speaker-embeddings is enabled");
            }
            if (!std::filesystem::exists(embedding_gguf)) {
                throw std::runtime_error("embedding GGUF not found: " + embedding_gguf.string());
            }
        }

        diarization_runtime rt;
        init_runtime_backend(device, rt);
        std::cout << "[info] diarization runtime backend: " << rt.selected_backend
                  << " device: " << rt.selected_device
                  << " type: " << rt.selected_device_type
                  << " requested: " << rt.requested_device
                  << " matmul_accel: " << (rt.use_backend_matmul ? "on" : "off")
                  << "\n";

        {
        const auto t0 = std::chrono::steady_clock::now();

        std::cout << "[info] loading segmentation gguf: " << segmentation_gguf.string() << "\n";
        segmentation_model model = load_segmentation_model(segmentation_gguf);

        std::cout << "[info] decoding audio: " << audio_path.string() << "\n";
        std::vector<float> audio = load_audio_mono_16k(audio_path);

        std::cout << "[info] running native diarization\n";
        diarization_output diar = diarize_from_segmentation(
            model,
            audio,
            &rt,
            num_speakers,
            min_speakers,
            max_speakers,
            min_duration_off_sec);

        json speaker_embeddings = {
            {"enabled", export_speaker_embeddings},
            {"status", "disabled"},
            {"speakers", json::array()},
            {"embedding_min_segment_duration_sec", embedding_min_segment_duration_sec},
            {"embedding_max_segments_per_speaker", std::max(1, embedding_max_segments_per_speaker)},
            {"embedding_gguf", embedding_gguf.empty() ? "" : embedding_gguf.string()},
        };

        double embedding_elapsed_sec = 0.0;
        if (export_speaker_embeddings) {
            const auto emb_t0 = std::chrono::steady_clock::now();
            std::cout << "[info] loading embedding gguf: " << embedding_gguf.string() << "\n";
            embedding_model emb_model = load_embedding_model(embedding_gguf);
            std::cout << "[info] extracting native speaker embeddings\n";
            speaker_embeddings = extract_speaker_embeddings(
                emb_model,
                audio,
                diar,
                &rt,
                embedding_min_segment_duration_sec,
                embedding_max_segments_per_speaker);
            const auto emb_t1 = std::chrono::steady_clock::now();
            embedding_elapsed_sec = std::chrono::duration<double>(emb_t1 - emb_t0).count();
            speaker_embeddings["embedding_min_segment_duration_sec"] = embedding_min_segment_duration_sec;
            speaker_embeddings["embedding_max_segments_per_speaker"] = std::max(1, embedding_max_segments_per_speaker);
            speaker_embeddings["embedding_gguf"] = embedding_gguf.string();
            speaker_embeddings["elapsed_sec"] = round3(embedding_elapsed_sec);
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

        std::filesystem::create_directories(output_dir);

        const std::string ts = now_timestamp();
        const std::string stem = audio_path.stem().string() + "_pyannote_" + ts;

        const std::filesystem::path json_path = output_dir / (stem + ".json");
        const std::filesystem::path txt_path = output_dir / (stem + ".segments.txt");
        const std::filesystem::path rttm_path = output_dir / (stem + ".rttm");
        const std::filesystem::path ex_txt_path = output_dir / (stem + ".exclusive.segments.txt");
        const std::filesystem::path ex_rttm_path = output_dir / (stem + ".exclusive.rttm");

        {
            json regular = json::array();
            for (const auto & seg : diar.regular) {
                regular.push_back({
                    {"speaker", speaker_name(seg.speaker)},
                    {"start_sec", round3(seg.start_sec)},
                    {"end_sec", round3(seg.end_sec)},
                });
            }

            json exclusive = json::array();
            for (const auto & seg : diar.exclusive) {
                exclusive.push_back({
                    {"speaker", speaker_name(seg.speaker)},
                    {"start_sec", round3(seg.start_sec)},
                    {"end_sec", round3(seg.end_sec)},
                });
            }

            json payload = {
                {"audio_path", audio_path.string()},
                {"pipeline_dir", pipeline_dir.empty() ? "" : std::filesystem::absolute(pipeline_dir).string()},
                {"device", device},
                {"device_resolved", rt.selected_device},
                {"device_type", rt.selected_device_type},
                {"compute_backend", rt.selected_backend},
                {"gpu_matmul_acceleration", rt.use_backend_matmul},
                {"offline", offline},
                {"progress", progress},
                {"backend", "native_cpp"},
                {"elapsed_sec", round3(elapsed_sec)},
                {"num_segments", static_cast<int64_t>(diar.regular.size())},
                {"num_exclusive_segments", static_cast<int64_t>(diar.exclusive.size())},
                {"speaker_count", diar.speaker_count},
                {"num_chunks", diar.num_chunks},
                {"num_frames_per_chunk", diar.num_frames_per_chunk},
                {"stage_timings_sec", {
                    {"build_chunks", round3(diar.elapsed_build_chunks_sec)},
                    {"segmentation_infer", round3(diar.elapsed_segmentation_infer_sec)},
                    {"global_mapping", round3(diar.elapsed_global_mapping_sec)},
                    {"aggregation", round3(diar.elapsed_aggregation_sec)},
                    {"binarize_segments", round3(diar.elapsed_binarize_segments_sec)},
                    {"speaker_embeddings", round3(embedding_elapsed_sec)},
                    {"infer_profiled_chunks", diar.infer_profiled_chunks},
                    {"infer_norm_affine", round3(diar.infer_norm_affine_sec)},
                    {"infer_frontend", round3(diar.infer_frontend_sec)},
                    {"infer_conv_stack", round3(diar.infer_conv_stack_sec)},
                    {"infer_seq_pack", round3(diar.infer_seq_pack_sec)},
                    {"infer_lstm", round3(diar.infer_lstm_sec)},
                    {"infer_linear_head", round3(diar.infer_linear_head_sec)},
                    {"infer_decode", round3(diar.infer_decode_sec)},
                }},
                {"frame_step_sec", model.frame_step_sec},
                {"frame_duration_sec", model.frame_duration_sec},
                {"segmentation_gguf", segmentation_gguf.string()},
                {"regular_speaker_diarization", regular},
                {"exclusive_speaker_diarization", exclusive},
                {"speaker_embeddings", speaker_embeddings},
            };

            write_text_file(json_path, payload.dump(2));
            write_segments_txt(diar.regular, txt_path);
            write_segments_rttm(diar.regular, rttm_path, audio_path.stem().string());
            write_segments_txt(diar.exclusive, ex_txt_path);
            write_segments_rttm(diar.exclusive, ex_rttm_path, audio_path.stem().string());
        }

        std::cout << "[done] elapsed: " << std::fixed << std::setprecision(2) << elapsed_sec << "s\n";
        std::cout << "[done] json: " << json_path.string() << "\n";
        std::cout << "[done] txt:  " << txt_path.string() << "\n";
        std::cout << "[done] rttm: " << rttm_path.string() << "\n";
        std::cout << "[done] ex-txt:  " << ex_txt_path.string() << "\n";
        std::cout << "[done] ex-rttm: " << ex_rttm_path.string() << "\n";

        // Release large buffers before returning to the in-process caller.
        speaker_embeddings = json();
        diar.regular.clear();
        diar.exclusive.clear();
        std::vector<float>().swap(audio);
        {
            segmentation_model tmp_model;
            std::swap(model, tmp_model);
        }
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}

#ifndef LLAMA_PYANNOTE_NO_MAIN
int main(int argc, char ** argv) {
    return llama_pyannote_diarize_main(argc, argv);
}
#endif
