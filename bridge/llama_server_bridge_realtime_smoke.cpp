#include "llama_server_bridge.h"

#include "tools/realtime/backend-factory.h"
#include "tools/realtime/stream-manager.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

double parse_double(const char * text, const char * label) {
    char * end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || (end != nullptr && *end != '\0')) {
        throw std::invalid_argument(std::string("invalid numeric value for ") + label);
    }
    return value;
}

int32_t parse_bridge_backend_kind(const std::string & text) {
    if (text.empty() || text == "auto") {
        return 0;
    }
    const int32_t kind = llama_server_bridge_realtime_backend_kind_from_name(text.c_str());
    if (kind != 0) {
        return kind;
    }
    throw std::invalid_argument("unsupported backend kind: " + text);
}

void print_bridge_backends(void) {
    const int32_t count = llama_server_bridge_realtime_backend_count();
    for (int32_t i = 0; i < count; ++i) {
        const int32_t kind = llama_server_bridge_realtime_backend_kind_at(static_cast<size_t>(i));
        llama_server_bridge_realtime_backend_info info =
            llama_server_bridge_empty_realtime_backend_info();
        if (!llama_server_bridge_realtime_backend_get_info(kind, &info)) {
            continue;
        }
        std::cout
            << info.backend_kind
            << "\t"
            << (info.name != nullptr ? info.name : "")
            << "\tdefault_backend=" << (info.default_runtime_backend_name != nullptr ? info.default_runtime_backend_name : "")
            << "\tpreload=" << info.supports_model_preload
            << "\ttranscript=" << info.emits_transcript
            << "\tspeaker_spans=" << info.emits_speaker_spans
            << "\tsr=" << info.default_sample_rate_hz
            << "\tring=" << info.default_audio_ring_capacity_samples
            << "\tchannels=" << info.required_input_channels
            << "\n";
    }
}

llama::realtime::backend_kind to_native_backend_kind(const int32_t backend_kind) {
    if (backend_kind == LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_AUTO) {
        return llama::realtime::backend_kind::unknown;
    }
    const char * name = llama_server_bridge_realtime_backend_name(backend_kind);
    if (name == nullptr || name[0] == '\0') {
        throw std::invalid_argument("unsupported bridge backend kind");
    }
    const auto * descriptor = llama::realtime::find_backend_descriptor(std::string(name));
    if (descriptor == nullptr) {
        throw std::invalid_argument("unsupported native backend kind name");
    }
    return descriptor->kind;
}

std::vector<float> load_wav_mono_f32(const std::string & path, uint32_t & sample_rate_hz) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open WAV file: " + path);
    }

    auto read_u16 = [&](uint16_t & out) {
        char bytes[2];
        in.read(bytes, 2);
        if (!in) {
            throw std::runtime_error("unexpected EOF while reading WAV");
        }
        out = static_cast<uint16_t>(static_cast<unsigned char>(bytes[0]) | (static_cast<unsigned char>(bytes[1]) << 8));
    };
    auto read_u32 = [&](uint32_t & out) {
        char bytes[4];
        in.read(bytes, 4);
        if (!in) {
            throw std::runtime_error("unexpected EOF while reading WAV");
        }
        out = static_cast<uint32_t>(static_cast<unsigned char>(bytes[0])
            | (static_cast<unsigned char>(bytes[1]) << 8)
            | (static_cast<unsigned char>(bytes[2]) << 16)
            | (static_cast<unsigned char>(bytes[3]) << 24));
    };

    char riff[4];
    in.read(riff, 4);
    if (!in || std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("WAV missing RIFF header");
    }
    uint32_t riff_size = 0;
    read_u32(riff_size);
    (void) riff_size;
    char wave[4];
    in.read(wave, 4);
    if (!in || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("WAV missing WAVE header");
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> data_bytes;

    while (in) {
        char chunk_id[4];
        in.read(chunk_id, 4);
        if (!in) {
            break;
        }
        uint32_t chunk_size = 0;
        read_u32(chunk_size);
        const std::string id(chunk_id, 4);

        if (id == "fmt ") {
            read_u16(audio_format);
            read_u16(channels);
            read_u32(sample_rate);
            uint32_t byte_rate = 0;
            uint16_t block_align = 0;
            read_u32(byte_rate);
            read_u16(block_align);
            read_u16(bits_per_sample);
            (void) byte_rate;
            (void) block_align;
            if (chunk_size > 16) {
                in.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (id == "data") {
            data_bytes.resize(chunk_size);
            in.read(data_bytes.data(), chunk_size);
            if (!in) {
                throw std::runtime_error("failed to read WAV data chunk");
            }
        } else {
            in.seekg(chunk_size, std::ios::cur);
        }

        if ((chunk_size & 1u) != 0u) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (audio_format == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data_bytes.empty()) {
        throw std::runtime_error("incomplete WAV metadata");
    }

    sample_rate_hz = sample_rate;
    const size_t bytes_per_sample = bits_per_sample / 8;
    const size_t frame_size = bytes_per_sample * channels;
    if (frame_size == 0 || data_bytes.size() % frame_size != 0) {
        throw std::runtime_error("invalid WAV frame size");
    }

    const size_t frames = data_bytes.size() / frame_size;
    std::vector<float> out(frames, 0.0f);

    for (size_t i = 0; i < frames; ++i) {
        double accum = 0.0;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            const char * src = data_bytes.data() + static_cast<ptrdiff_t>(i * frame_size + ch * bytes_per_sample);
            float sample = 0.0f;
            if (audio_format == 1 && bits_per_sample == 16) {
                const int16_t v = static_cast<int16_t>(
                    static_cast<uint16_t>(static_cast<unsigned char>(src[0]) | (static_cast<unsigned char>(src[1]) << 8)));
                sample = static_cast<float>(v) / 32768.0f;
            } else if (audio_format == 1 && bits_per_sample == 32) {
                const int32_t v = static_cast<int32_t>(
                    static_cast<uint32_t>(static_cast<unsigned char>(src[0])
                    | (static_cast<unsigned char>(src[1]) << 8)
                    | (static_cast<unsigned char>(src[2]) << 16)
                    | (static_cast<unsigned char>(src[3]) << 24)));
                sample = static_cast<float>(v) / 2147483648.0f;
            } else if (audio_format == 3 && bits_per_sample == 32) {
                std::memcpy(&sample, src, sizeof(float));
            } else {
                throw std::runtime_error("unsupported WAV format for bridge realtime smoke");
            }
            accum += sample;
        }
        out[i] = static_cast<float>(accum / static_cast<double>(channels));
    }

    return out;
}

bool nearly_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

std::string json_escape(const std::string & text) {
    std::ostringstream out;
    for (const unsigned char ch : text) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

char * duplicate_cstr(const char * text) {
    if (text == nullptr) {
        return nullptr;
    }
    const size_t len = std::strlen(text);
    char * copy = static_cast<char *>(std::malloc(len + 1));
    if (copy == nullptr) {
        throw std::bad_alloc();
    }
    std::memcpy(copy, text, len);
    copy[len] = '\0';
    return copy;
}

llama_server_bridge_realtime_event clone_bridge_event(const llama_server_bridge_realtime_event & src) {
    llama_server_bridge_realtime_event ev = {};
    ev.type = src.type;
    ev.session_id = src.session_id;
    ev.begin_sec = src.begin_sec;
    ev.end_sec = src.end_sec;
    ev.speaker_id = src.speaker_id;
    ev.text = duplicate_cstr(src.text);
    ev.detail = duplicate_cstr(src.detail);
    return ev;
}

void free_cloned_bridge_events(std::vector<llama_server_bridge_realtime_event> & events) {
    for (auto & ev : events) {
        std::free(ev.text);
        std::free(ev.detail);
        ev.text = nullptr;
        ev.detail = nullptr;
    }
}

void write_bridge_events_json(
    const std::string & path,
    const std::string & backend_name,
    const llama_server_bridge_realtime_event * events,
    const size_t count) {

    namespace fs = std::filesystem;
    const fs::path out_path(path);
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open bridge events JSON for write: " + path);
    }

    out << "{\n";
    out << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
    out << "  \"num_events\": " << count << ",\n";
    out << "  \"events\": [\n";
    for (size_t i = 0; i < count; ++i) {
        out << "    {\n";
        out << "      \"type\": " << events[i].type << ",\n";
        out << "      \"speaker_id\": " << events[i].speaker_id << ",\n";
        out << "      \"begin_sec\": " << events[i].begin_sec << ",\n";
        out << "      \"end_sec\": " << events[i].end_sec << ",\n";
        out << "      \"text\": \"" << json_escape(events[i].text != nullptr ? events[i].text : "") << "\",\n";
        out << "      \"detail\": \"" << json_escape(events[i].detail != nullptr ? events[i].detail : "") << "\"\n";
        out << "    }";
        if (i + 1 < count) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void compare_events(
    const std::vector<llama::realtime::event> & direct,
    const llama_server_bridge_realtime_event * bridge_events,
    size_t bridge_count) {

    if (direct.size() != bridge_count) {
        throw std::runtime_error("bridge event count mismatch");
    }

    for (size_t i = 0; i < direct.size(); ++i) {
        if (static_cast<int32_t>(direct[i].type) != bridge_events[i].type) {
            throw std::runtime_error("bridge event type mismatch at index " + std::to_string(i));
        }
        if (direct[i].speaker_id != bridge_events[i].speaker_id) {
            throw std::runtime_error("bridge speaker mismatch at index " + std::to_string(i));
        }
        if (!nearly_equal(direct[i].begin_sec, bridge_events[i].begin_sec) ||
            !nearly_equal(direct[i].end_sec, bridge_events[i].end_sec)) {
            throw std::runtime_error("bridge event timing mismatch at index " + std::to_string(i));
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        std::string backend_kind_name = "auto";
        std::string backend_name = "Vulkan0";
        std::string audio_wav_path;
        std::string dump_events_json;
        double feed_ms = 100.0;
        size_t reuse_model_runs = 0;
        size_t reuse_create_runs = 0;
        bool bridge_only = false;
        bool list_backends = false;
        bool clear_model_cache = false;
        bool capture_debug = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--list-backends") {
                list_backends = true;
                continue;
            }
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
                continue;
            }
            if (arg == "--backend-kind" && i + 1 < argc) {
                backend_kind_name = argv[++i];
                continue;
            }
            if (arg == "--sortformer-gguf" && i + 1 < argc) {
                model_path = argv[++i];
                continue;
            }
            if (arg == "--backend" && i + 1 < argc) {
                backend_name = argv[++i];
                continue;
            }
            if (arg == "--audio-wav" && i + 1 < argc) {
                audio_wav_path = argv[++i];
                continue;
            }
            if (arg == "--dump-events-json" && i + 1 < argc) {
                dump_events_json = argv[++i];
                continue;
            }
            if (arg == "--feed-ms" && i + 1 < argc) {
                feed_ms = parse_double(argv[++i], "feed-ms");
                continue;
            }
            if (arg == "--reuse-model-runs" && i + 1 < argc) {
                reuse_model_runs = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
                continue;
            }
            if (arg == "--reuse-create-runs" && i + 1 < argc) {
                reuse_create_runs = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
                continue;
            }
            if (arg == "--bridge-only") {
                bridge_only = true;
                continue;
            }
            if (arg == "--clear-model-cache") {
                clear_model_cache = true;
                continue;
            }
            if (arg == "--capture-debug") {
                capture_debug = true;
                continue;
            }
            throw std::invalid_argument("unknown argument: " + arg);
        }

        if (list_backends) {
            print_bridge_backends();
            return 0;
        }

        if (model_path.empty() || audio_wav_path.empty()) {
            throw std::runtime_error("usage: --backend-kind <auto|backend-name> --model <path> --audio-wav <path> [--backend Vulkan0|CPU] [--feed-ms N]");
        }
        if (reuse_model_runs > 0 && reuse_create_runs > 0) {
            throw std::runtime_error("reuse-model-runs and reuse-create-runs are mutually exclusive");
        }
        if (reuse_create_runs > 0 && !bridge_only) {
            throw std::runtime_error("reuse-create-runs requires --bridge-only");
        }
        if (clear_model_cache) {
            llama_server_bridge_realtime_model_cache_clear();
        }

        uint32_t sample_rate_hz = 0;
        const auto audio = load_wav_mono_f32(audio_wav_path, sample_rate_hz);
        const size_t feed_samples = std::max<size_t>(1, static_cast<size_t>(std::llround(feed_ms * 0.001 * sample_rate_hz)));
        const int32_t backend_kind = parse_bridge_backend_kind(backend_kind_name);
        const std::string requested_backend_kind_name = backend_kind_name;
        if (backend_kind != 0) {
            backend_kind_name = llama_server_bridge_realtime_backend_name(backend_kind);
        }

        llama_server_bridge_realtime_params params =
            backend_kind != LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_AUTO
                ? llama_server_bridge_default_realtime_params_for_backend(backend_kind)
                : llama_server_bridge_default_realtime_params();
        params.model_path = model_path.c_str();
        params.backend_name = backend_name.c_str();
        params.expected_sample_rate_hz = sample_rate_hz;
        params.capture_debug = capture_debug ? 1u : 0u;
        llama::realtime::backend_model_params direct_backend_params;
        direct_backend_params.kind = to_native_backend_kind(backend_kind);
        direct_backend_params.model_path = model_path;
        direct_backend_params.backend_name = backend_name;
        std::shared_ptr<llama::realtime::loaded_backend_model> direct_loaded_model;
        llama_server_bridge_realtime_model * bridge_model = nullptr;
        double direct_model_load_sec = 0.0;
        double bridge_model_load_sec = 0.0;
        const bool reuse_model = reuse_model_runs > 0;
        const bool reuse_create = reuse_create_runs > 0;
        const size_t total_runs = reuse_model ? reuse_model_runs : (reuse_create ? reuse_create_runs : 1);

        if (reuse_model) {
            if (!bridge_only) {
                const auto t0 = std::chrono::steady_clock::now();
                direct_loaded_model = llama::realtime::load_backend_model(direct_backend_params);
                const auto t1 = std::chrono::steady_clock::now();
                direct_model_load_sec = std::chrono::duration<double>(t1 - t0).count();
            }

            const auto t2 = std::chrono::steady_clock::now();
            bridge_model = llama_server_bridge_realtime_model_create(&params);
            const auto t3 = std::chrono::steady_clock::now();
            bridge_model_load_sec = std::chrono::duration<double>(t3 - t2).count();
            if (bridge_model == nullptr) {
                throw std::runtime_error("failed to create bridge realtime model");
            }
            llama_server_bridge_realtime_backend_info model_info =
                llama_server_bridge_empty_realtime_backend_info();
            if (!llama_server_bridge_realtime_model_get_info(bridge_model, &model_info)) {
                throw std::runtime_error("failed to query bridge realtime model info");
            }
            const int32_t expected_model_backend_kind =
                backend_kind != 0
                    ? backend_kind
                    : llama_server_bridge_realtime_backend_kind_from_model_path(model_path.c_str());
            if (expected_model_backend_kind == 0) {
                throw std::runtime_error("failed to resolve expected backend kind for preloaded model");
            }
            if (model_info.backend_kind != expected_model_backend_kind) {
                throw std::runtime_error("bridge realtime model info backend kind mismatch");
            }
            if (model_info.default_runtime_backend_name == nullptr ||
                std::string(model_info.default_runtime_backend_name) != backend_name) {
                throw std::runtime_error("bridge realtime model info runtime backend mismatch");
            }
            if (model_info.default_sample_rate_hz != params.expected_sample_rate_hz) {
                throw std::runtime_error("bridge realtime model info sample rate mismatch");
            }
            const uint32_t expected_ring_capacity =
                params.audio_ring_capacity_samples != 0
                    ? params.audio_ring_capacity_samples
                    : llama_server_bridge_realtime_backend_default_audio_ring_capacity_samples(
                          expected_model_backend_kind);
            if (model_info.default_audio_ring_capacity_samples != expected_ring_capacity) {
                throw std::runtime_error("bridge realtime model info ring capacity mismatch");
            }
        }

        const int32_t cache_entries_before = llama_server_bridge_realtime_model_cache_entry_count();

        std::vector<llama_server_bridge_realtime_event> final_bridge_events;
        std::vector<double> run_elapsed_sec;
        run_elapsed_sec.reserve(total_runs);

        for (size_t run_index = 0; run_index < total_runs; ++run_index) {
            llama::realtime::stream_manager direct_manager;
            int64_t direct_session = 0;
            if (!bridge_only) {
                auto direct_backend = reuse_model
                    ? direct_loaded_model->create_backend(capture_debug)
                    : llama::realtime::create_backend(direct_backend_params, capture_debug);
                direct_session = direct_manager.create_session(std::move(direct_backend));
            }

            const auto run_begin = std::chrono::steady_clock::now();
            auto * bridge = reuse_model
                ? llama_server_bridge_realtime_create_from_model(bridge_model, capture_debug ? &params : nullptr)
                : llama_server_bridge_realtime_create(&params);
            if (bridge == nullptr) {
                throw std::runtime_error("failed to create bridge realtime session");
            }

            std::vector<llama_server_bridge_realtime_event> all_bridge_events;
            size_t offset = 0;
            while (offset < audio.size()) {
                const size_t n = std::min(feed_samples, audio.size() - offset);
                if (!bridge_only) {
                    direct_manager.push_audio(direct_session, audio.data() + offset, n, sample_rate_hz);
                }
                if (llama_server_bridge_realtime_push_audio_f32(bridge, audio.data() + offset, n, sample_rate_hz) != 0) {
                    throw std::runtime_error(llama_server_bridge_realtime_last_error(bridge));
                }

                llama_server_bridge_realtime_event * bridge_events = nullptr;
                size_t bridge_count = 0;
                if (llama_server_bridge_realtime_drain_events(bridge, &bridge_events, &bridge_count, 0) != 0) {
                    throw std::runtime_error(llama_server_bridge_realtime_last_error(bridge));
                }
                if (!bridge_only) {
                    const auto direct_events = direct_manager.drain_events(direct_session, 0);
                    compare_events(direct_events, bridge_events, bridge_count);
                }
                for (size_t i = 0; i < bridge_count; ++i) {
                    all_bridge_events.push_back(clone_bridge_event(bridge_events[i]));
                }
                llama_server_bridge_realtime_free_events(bridge_events, bridge_count);
                offset += n;
            }

            if (!bridge_only) {
                direct_manager.flush_session(direct_session);
            }
            if (llama_server_bridge_realtime_flush(bridge) != 0) {
                throw std::runtime_error(llama_server_bridge_realtime_last_error(bridge));
            }

            llama_server_bridge_realtime_event * bridge_events = nullptr;
            size_t bridge_count = 0;
            if (llama_server_bridge_realtime_drain_events(bridge, &bridge_events, &bridge_count, 0) != 0) {
                throw std::runtime_error(llama_server_bridge_realtime_last_error(bridge));
            }
            if (!bridge_only) {
                const auto direct_events = direct_manager.drain_events(direct_session, 0);
                compare_events(direct_events, bridge_events, bridge_count);
            }
            for (size_t i = 0; i < bridge_count; ++i) {
                all_bridge_events.push_back(clone_bridge_event(bridge_events[i]));
            }
            llama_server_bridge_realtime_free_events(bridge_events, bridge_count);

            const auto run_end = std::chrono::steady_clock::now();
            run_elapsed_sec.push_back(std::chrono::duration<double>(run_end - run_begin).count());

            if (run_index + 1 == total_runs) {
                final_bridge_events = std::move(all_bridge_events);
            } else {
                free_cloned_bridge_events(all_bridge_events);
            }

            llama_server_bridge_realtime_destroy(bridge);
            if (!bridge_only) {
                direct_manager.close_session(direct_session);
            }
        }

        const int32_t cache_entries_after = llama_server_bridge_realtime_model_cache_entry_count();

        if (!dump_events_json.empty()) {
            write_bridge_events_json(dump_events_json, backend_name, final_bridge_events.data(), final_bridge_events.size());
        }

        std::cout << "bridge realtime parity OK\n";
        std::cout << "  backend=" << backend_name << "\n";
        std::cout << "  backend_kind=" << (backend_kind != 0 ? backend_kind_name : requested_backend_kind_name) << "\n";
        std::cout << "  sample_rate_hz=" << sample_rate_hz << "\n";
        std::cout << "  feed_samples=" << feed_samples << "\n";
        std::cout << "  reuse_model=" << (reuse_model ? 1 : 0) << "\n";
        std::cout << "  reuse_create=" << (reuse_create ? 1 : 0) << "\n";
        std::cout << "  bridge_only=" << (bridge_only ? 1 : 0) << "\n";
        std::cout << "  model_cache_entries_before=" << cache_entries_before << "\n";
        std::cout << "  model_cache_entries_after=" << cache_entries_after << "\n";
        if (reuse_model) {
            if (!bridge_only) {
                std::cout << "  direct_model_load_sec=" << direct_model_load_sec << "\n";
            }
            std::cout << "  bridge_model_load_sec=" << bridge_model_load_sec << "\n";
        }
        for (size_t i = 0; i < run_elapsed_sec.size(); ++i) {
            std::cout << "  run[" << i << "]_elapsed_sec=" << run_elapsed_sec[i] << "\n";
        }
        std::cout << "  final_events=" << final_bridge_events.size() << "\n";
        for (size_t i = 0; i < final_bridge_events.size(); ++i) {
            std::cout
                << "  [" << i << "] type=" << final_bridge_events[i].type
                << " speaker=" << final_bridge_events[i].speaker_id
                << " begin=" << final_bridge_events[i].begin_sec
                << " end=" << final_bridge_events[i].end_sec
                << "\n";
        }

        if (bridge_model != nullptr) {
            llama_server_bridge_realtime_model_destroy(bridge_model);
        }
        free_cloned_bridge_events(final_bridge_events);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "bridge realtime smoke failed: " << e.what() << "\n";
        return 1;
    }
}
