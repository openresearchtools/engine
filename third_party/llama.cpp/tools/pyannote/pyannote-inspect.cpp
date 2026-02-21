#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model <file.gguf> [--all-kv] [--all-tensors]\n"
        << "  --model <path>     GGUF file to inspect\n"
        << "  --all-kv           Print all KV entries (default: summary only)\n"
        << "  --all-tensors      Print all tensor entries (default: first 12)\n";
}

static std::string kv_to_string(const gguf_context * ctx, int64_t key_id) {
    const auto t = gguf_get_kv_type(ctx, key_id);
    switch (t) {
        case GGUF_TYPE_UINT8:   return std::to_string(gguf_get_val_u8(ctx, key_id));
        case GGUF_TYPE_INT8:    return std::to_string(gguf_get_val_i8(ctx, key_id));
        case GGUF_TYPE_UINT16:  return std::to_string(gguf_get_val_u16(ctx, key_id));
        case GGUF_TYPE_INT16:   return std::to_string(gguf_get_val_i16(ctx, key_id));
        case GGUF_TYPE_UINT32:  return std::to_string(gguf_get_val_u32(ctx, key_id));
        case GGUF_TYPE_INT32:   return std::to_string(gguf_get_val_i32(ctx, key_id));
        case GGUF_TYPE_FLOAT32: return std::to_string(gguf_get_val_f32(ctx, key_id));
        case GGUF_TYPE_UINT64:  return std::to_string(gguf_get_val_u64(ctx, key_id));
        case GGUF_TYPE_INT64:   return std::to_string(gguf_get_val_i64(ctx, key_id));
        case GGUF_TYPE_FLOAT64: return std::to_string(gguf_get_val_f64(ctx, key_id));
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, key_id) ? "true" : "false";
        case GGUF_TYPE_STRING: {
            const char * s = gguf_get_val_str(ctx, key_id);
            return s ? s : "";
        }
        case GGUF_TYPE_ARRAY: {
            const auto at = gguf_get_arr_type(ctx, key_id);
            const auto n = gguf_get_arr_n(ctx, key_id);
            return std::string("array<") + gguf_type_name(at) + ">[" + std::to_string(n) + "]";
        }
        case GGUF_TYPE_COUNT:
            break;
    }
    return "<unknown>";
}

int main(int argc, char ** argv) {
    std::string model_path;
    bool all_kv = false;
    bool all_tensors = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--all-kv") == 0) {
            all_kv = true;
            continue;
        }
        if (std::strcmp(argv[i], "--all-tensors") == 0) {
            all_tensors = true;
            continue;
        }
        print_usage(argv[0]);
        return 2;
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (ctx == nullptr) {
        std::cerr << "error: failed to load gguf: " << model_path << "\n";
        return 1;
    }

    const auto version = gguf_get_version(ctx);
    const auto n_kv = gguf_get_n_kv(ctx);
    const auto n_tensors = gguf_get_n_tensors(ctx);
    std::cout << "file: " << model_path << "\n";
    std::cout << "gguf_version: " << version << "\n";
    std::cout << "n_kv: " << n_kv << "\n";
    std::cout << "n_tensors: " << n_tensors << "\n";
    std::cout << "\n";

    auto print_key_if_exists = [&](const char * key) {
        const int64_t id = gguf_find_key(ctx, key);
        if (id >= 0) {
            std::cout << key << ": " << kv_to_string(ctx, id) << "\n";
        }
    };

    std::cout << "[pyannote summary]\n";
    print_key_if_exists("general.architecture");
    print_key_if_exists("pyannote.kind");
    print_key_if_exists("pyannote.architecture.class");
    print_key_if_exists("pyannote.architecture.module");
    print_key_if_exists("pyannote.sample_rate");
    print_key_if_exists("pyannote.num_channels");
    print_key_if_exists("pyannote.spec.problem");
    print_key_if_exists("pyannote.spec.resolution");
    print_key_if_exists("pyannote.spec.duration_sec");
    print_key_if_exists("pyannote.spec.num_classes");
    std::cout << "\n";

    std::cout << "[kv entries]\n";
    int64_t kv_to_print = all_kv ? n_kv : std::min<int64_t>(n_kv, 24);
    for (int64_t i = 0; i < kv_to_print; ++i) {
        const char * key = gguf_get_key(ctx, i);
        if (key == nullptr) {
            continue;
        }
        std::cout << i << ": " << key << " = " << kv_to_string(ctx, i) << "\n";
    }
    if (!all_kv && n_kv > kv_to_print) {
        std::cout << "... (" << (n_kv - kv_to_print) << " more, use --all-kv)\n";
    }
    std::cout << "\n";

    std::cout << "[tensors]\n";
    int64_t tensors_to_print = all_tensors ? n_tensors : std::min<int64_t>(n_tensors, 12);
    for (int64_t i = 0; i < tensors_to_print; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        const auto type = gguf_get_tensor_type(ctx, i);
        const auto size = gguf_get_tensor_size(ctx, i);
        std::cout
            << i
            << ": " << (name ? name : "<null>")
            << " | type=" << ggml_type_name(type)
            << " | size=" << size
            << " bytes\n";
    }
    if (!all_tensors && n_tensors > tensors_to_print) {
        std::cout << "... (" << (n_tensors - tensors_to_print) << " more, use --all-tensors)\n";
    }

    gguf_free(ctx);
    return 0;
}
