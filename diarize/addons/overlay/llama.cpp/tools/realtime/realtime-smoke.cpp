#include "backend-factory.h"
#include "sortformer/sortformer-audio.h"
#include "sortformer/sortformer-backend.h"
#include "sortformer/sortformer-encoder.h"
#include "sortformer/sortformer-frontend.h"
#include "sortformer/sortformer-gguf.h"
#include "sortformer/sortformer-layer0.h"
#include "sortformer/sortformer-model.h"
#include "sortformer/sortformer-postnet.h"
#include "sortformer/sortformer-preencode.h"
#include "sortformer/sortformer-schema.h"
#include "sortformer/sortformer-state.h"
#include "sortformer/sortformer-streaming.h"
#include "stream-manager.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --backend-kind <sortformer> --model <path> [--simulate-seconds N] [--feed-ms N]\n"
        << "Loads realtime backend metadata and simulates the host-side streaming scheduler.\n"
        << "Other helpers:\n"
        << "  --list-backends\n"
        << "Optional native preencode parity mode:\n"
        << "  --backend <name> --features-bin <path> [--reference-bin <path>] [--dump-output-bin <path>]\n"
        << "Optional first-layer conformer parity mode:\n"
        << "  --backend <name> --layer0-ref-dir <path>\n"
        << "Optional full conformer encoder parity mode:\n"
        << "  --backend <name> --encoder-ref-dir <path>\n"
        << "Optional native features->encoder parity mode:\n"
        << "  --backend <name> --frontend-ref-dir <path>\n"
        << "Optional transformer/head parity mode:\n"
        << "  --backend <name> --full-ref-dir <path>\n"
        << "Optional transformer/head parity mode:\n"
        << "  --backend <name> --postnet-ref-dir <path>\n"
        << "Optional first real streaming-step parity mode:\n"
        << "  --backend <name> --stream-step-ref-dir <path>\n"
        << "Optional multi-step streaming-session parity mode:\n"
        << "  --backend <name> --stream-session-ref-dir <path>\n"
        << "Optional real audio-fed streaming-session parity mode:\n"
        << "  --backend <name> --audio-wav <path> --stream-audio-ref-dir <path> [--dump-events-json <path>]\n";
}

double parse_double(const char * text, const char * label) {
    char * end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || (end != nullptr && *end != '\0')) {
        throw std::invalid_argument(std::string("invalid numeric value for ") + label);
    }
    return value;
}

llama::realtime::backend_kind parse_backend_kind_or_throw(const std::string & text) {
    llama::realtime::backend_kind kind = llama::realtime::backend_kind::sortformer;
    if (!llama::realtime::parse_backend_kind_name(text, kind)) {
        throw std::invalid_argument("unsupported backend kind: " + text);
    }
    return kind;
}

void print_backends(void) {
    const size_t count = llama::realtime::backend_descriptor_count();
    for (size_t i = 0; i < count; ++i) {
        const auto * descriptor = llama::realtime::backend_descriptor_at(i);
        if (descriptor == nullptr || descriptor->name == nullptr) {
            continue;
        }
        std::cout
            << static_cast<int32_t>(descriptor->kind)
            << "\t"
            << descriptor->name
            << "\tdefault_backend="
            << (descriptor->default_runtime_backend_name != nullptr ? descriptor->default_runtime_backend_name : "")
            << "\tpreload="
            << (descriptor->supports_model_preload ? "1" : "0")
            << "\ttranscript="
            << (descriptor->emits_transcript ? "1" : "0")
            << "\tspeaker_spans="
            << (descriptor->emits_speaker_spans ? "1" : "0")
            << "\tsr="
            << descriptor->default_sample_rate_hz
            << "\tring="
            << descriptor->default_audio_ring_capacity_samples
            << "\tchannels="
            << descriptor->required_input_channels
            << "\n";
    }
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
    GGML_UNUSED(riff_size);
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
            GGML_UNUSED(byte_rate);
            GGML_UNUSED(block_align);
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
                throw std::runtime_error("unsupported WAV format for realtime smoke");
            }
            accum += sample;
        }
        out[i] = static_cast<float>(accum / static_cast<double>(channels));
    }

    return out;
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

std::string event_type_name(const llama::realtime::event_type type) {
    using llama::realtime::event_type;
    switch (type) {
        case event_type::backend_status: return "backend_status";
        case event_type::transcript_commit: return "transcript_commit";
        case event_type::speaker_span_commit: return "speaker_span_commit";
        case event_type::session_notice: return "session_notice";
        case event_type::backend_error: return "backend_error";
        default: return "unknown";
    }
}

void write_events_json(const std::string & path, const std::string & backend_name, const std::vector<llama::realtime::event> & events) {
    namespace fs = std::filesystem;
    const fs::path out_path(path);
    if (out_path.has_parent_path()) {
        fs::create_directories(out_path.parent_path());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open events JSON for write: " + path);
    }

    out << "{\n";
    out << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
    out << "  \"num_events\": " << events.size() << ",\n";
    out << "  \"events\": [\n";
    for (size_t i = 0; i < events.size(); ++i) {
        const auto & ev = events[i];
        out << "    {\n";
        out << "      \"type\": " << static_cast<int>(ev.type) << ",\n";
        out << "      \"type_name\": \"" << json_escape(event_type_name(ev.type)) << "\",\n";
        out << "      \"session_id\": " << ev.session_id << ",\n";
        out << "      \"speaker_id\": " << ev.speaker_id << ",\n";
        out << "      \"begin_sec\": " << ev.begin_sec << ",\n";
        out << "      \"end_sec\": " << ev.end_sec << ",\n";
        out << "      \"text\": \"" << json_escape(ev.text) << "\",\n";
        out << "      \"detail\": \"" << json_escape(ev.detail) << "\"\n";
        out << "    }";
        if (i + 1 < events.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        std::string backend_kind_name = "sortformer";
        std::string backend_name = "Vulkan0";
        std::string features_bin;
        std::string reference_bin;
        std::string dump_output_bin;
        std::string layer0_ref_dir;
        std::string encoder_ref_dir;
        std::string frontend_ref_dir;
        std::string full_ref_dir;
        std::string postnet_ref_dir;
        std::string stream_step_ref_dir;
        std::string stream_session_ref_dir;
        std::string stream_audio_ref_dir;
        std::string audio_wav_path;
        std::string dump_session_actual_dir;
        std::string dump_events_json;
        bool list_backends = false;
        double simulate_seconds = 6.0;
        double feed_ms = 100.0;

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
                backend_kind_name = "sortformer";
                continue;
            }
            if (arg == "--simulate-seconds" && i + 1 < argc) {
                simulate_seconds = parse_double(argv[++i], "simulate-seconds");
                continue;
            }
            if (arg == "--backend" && i + 1 < argc) {
                backend_name = argv[++i];
                continue;
            }
            if (arg == "--features-bin" && i + 1 < argc) {
                features_bin = argv[++i];
                continue;
            }
            if (arg == "--reference-bin" && i + 1 < argc) {
                reference_bin = argv[++i];
                continue;
            }
            if (arg == "--dump-output-bin" && i + 1 < argc) {
                dump_output_bin = argv[++i];
                continue;
            }
            if (arg == "--layer0-ref-dir" && i + 1 < argc) {
                layer0_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--encoder-ref-dir" && i + 1 < argc) {
                encoder_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--frontend-ref-dir" && i + 1 < argc) {
                frontend_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--full-ref-dir" && i + 1 < argc) {
                full_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--postnet-ref-dir" && i + 1 < argc) {
                postnet_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--stream-step-ref-dir" && i + 1 < argc) {
                stream_step_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--stream-session-ref-dir" && i + 1 < argc) {
                stream_session_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--stream-audio-ref-dir" && i + 1 < argc) {
                stream_audio_ref_dir = argv[++i];
                continue;
            }
            if (arg == "--audio-wav" && i + 1 < argc) {
                audio_wav_path = argv[++i];
                continue;
            }
            if (arg == "--dump-session-actual-dir" && i + 1 < argc) {
                dump_session_actual_dir = argv[++i];
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
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            throw std::invalid_argument("unknown argument: " + arg);
        }

        if (list_backends) {
            print_backends();
            return 0;
        }

        if (model_path.empty()) {
            print_usage(argv[0]);
            return 1;
        }

        const auto parsed_backend_kind = parse_backend_kind_or_throw(backend_kind_name);
        const char * normalized_backend_kind = llama::realtime::backend_kind_name(parsed_backend_kind);
        if (normalized_backend_kind == nullptr || parsed_backend_kind != llama::realtime::backend_kind::sortformer) {
            throw std::runtime_error("unsupported backend kind for realtime smoke");
        }
        const std::string & gguf_path = model_path;

        const auto meta = llama::realtime::load_sortformer_gguf(model_path);
        const auto validation = llama::realtime::validate_sortformer_gguf_tensors(model_path, meta);
        llama::realtime::sortformer_stream_state state(meta);

        std::cout << "Loaded realtime backend metadata:\n";
        std::cout << "  backend_kind=" << normalized_backend_kind << "\n";
        std::cout << "  " << llama::realtime::sortformer_metadata_summary(meta) << "\n";
        std::cout << "Validated Sortformer tensor catalog:\n";
        std::cout << "  " << llama::realtime::sortformer_validation_summary(validation) << "\n";
        if (!validation.missing.empty()) {
            std::cout << "  missing tensors:\n";
            for (const auto & name : validation.missing) {
                std::cout << "    " << name << "\n";
            }
        }
        if (!validation.unexpected.empty()) {
            std::cout << "  unexpected tensors:\n";
            for (const auto & name : validation.unexpected) {
                std::cout << "    " << name << "\n";
            }
        }
        if (!validation.missing.empty() || !validation.unexpected.empty()) {
            throw std::runtime_error("Sortformer tensor catalog validation failed");
        }

        if (!features_bin.empty()) {
            std::cout << "Running native Sortformer preencode:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  features_bin=" << features_bin << "\n";

            const auto features = llama::realtime::load_matrix_f32_bin(features_bin);
            const auto model = llama::realtime::sortformer_model::load_from_gguf(model_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";
            const auto out = llama::realtime::sortformer_run_preencode(model, features);

            std::cout << "  output_rows=" << out.rows << "\n";
            std::cout << "  output_cols=" << out.cols << "\n";

            if (!dump_output_bin.empty()) {
                llama::realtime::save_matrix_f32_bin(dump_output_bin, out);
                std::cout << "  dumped_output=" << dump_output_bin << "\n";
            }

            if (!reference_bin.empty()) {
                const auto ref = llama::realtime::load_matrix_f32_bin(reference_bin);
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(out, ref);
                const auto rmse = llama::realtime::sortformer_rmse(out, ref);
                std::cout << "  reference_bin=" << reference_bin << "\n";
                std::cout << "  max_abs_diff=" << max_abs << "\n";
                std::cout << "  rmse=" << rmse << "\n";
            }
            return 0;
        }

        if (!layer0_ref_dir.empty()) {
            std::cout << "Running native Sortformer layer0:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << layer0_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";
            const auto posenc_x = llama::realtime::load_matrix_f32_bin(layer0_ref_dir + "\\posenc_x.bin");
            const auto pos_emb = llama::realtime::load_matrix_f32_bin(layer0_ref_dir + "\\pos_emb.bin");
            const auto pad_mask = llama::realtime::load_matrix_f32_bin(layer0_ref_dir + "\\pad_mask.bin");
            const auto att_mask = llama::realtime::load_matrix_f32_bin(layer0_ref_dir + "\\att_mask.bin");
            const auto outputs = llama::realtime::sortformer_run_layer0(model, posenc_x, pos_emb, pad_mask, att_mask);

            const auto compare = [&](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const char * file_name) {
                const auto ref = llama::realtime::load_matrix_f32_bin(layer0_ref_dir + "\\" + file_name);
                std::cout << "  " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "  " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            compare("ff1_norm", outputs.ff1_norm, "layer0_ff1_norm.bin");
            compare("ff1_mm", outputs.ff1_mm, "layer0_ff1_mm.bin");
            compare("ff1_l1", outputs.ff1_l1, "layer0_ff1_l1.bin");
            compare("ff1_act", outputs.ff1_act, "layer0_ff1_act.bin");
            compare("ff1_out_mm", outputs.ff1_out_mm, "layer0_ff1_out_mm.bin");
            compare("ff1_out", outputs.ff1_out, "layer0_ff1_out.bin");
            compare("ff1_res", outputs.ff1_res, "layer0_ff1_res.bin");
            compare("att_norm", outputs.att_norm, "layer0_att_norm.bin");
            compare("matrix_ac_head0", outputs.matrix_ac_head0, "layer0_matrix_ac_head0.bin");
            compare("matrix_bd_head0", outputs.matrix_bd_head0, "layer0_matrix_bd_head0.bin");
            compare("scores_head0", outputs.scores_head0, "layer0_scores_head0.bin");
            compare("attn_head0", outputs.attn_head0, "layer0_attn_head0.bin");
            compare("att_value_head0", outputs.att_value_head0, "layer0_att_value_head0.bin");
            compare("att_x", outputs.att_x, "layer0_att_x.bin");
            compare("att_out", outputs.att_out, "layer0_att_out.bin");
            compare("att_res", outputs.att_res, "layer0_att_res.bin");
            compare("conv_norm", outputs.conv_norm, "layer0_conv_norm.bin");
            compare("conv_pw1", outputs.conv_pw1, "layer0_conv_pw1.bin");
            compare("conv_glu", outputs.conv_glu, "layer0_conv_glu.bin");
            compare("conv_dw", outputs.conv_dw, "layer0_conv_dw.bin");
            compare("conv_bn", outputs.conv_bn, "layer0_conv_bn.bin");
            compare("conv_act", outputs.conv_act, "layer0_conv_act.bin");
            compare("conv_pw2", outputs.conv_pw2, "layer0_conv_pw2.bin");
            compare("conv_out", outputs.conv_out, "layer0_conv_out.bin");
            compare("conv_res", outputs.conv_res, "layer0_conv_res.bin");
            compare("ff2_norm", outputs.ff2_norm, "layer0_ff2_norm.bin");
            compare("ff2_out", outputs.ff2_out, "layer0_ff2_out.bin");
            compare("ff2_res", outputs.ff2_res, "layer0_ff2_res.bin");
            compare("out", outputs.out, "layer0_out.bin");
            return 0;
        }

        if (!encoder_ref_dir.empty()) {
            std::cout << "Running native Sortformer encoder:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << encoder_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";
            const auto posenc_x = llama::realtime::load_matrix_f32_bin(encoder_ref_dir + "\\posenc_x.bin");
            const auto pos_emb = llama::realtime::load_matrix_f32_bin(encoder_ref_dir + "\\pos_emb.bin");
            const auto pad_mask = llama::realtime::load_matrix_f32_bin(encoder_ref_dir + "\\pad_mask.bin");
            const auto att_mask = llama::realtime::load_matrix_f32_bin(encoder_ref_dir + "\\att_mask.bin");
            const auto out = llama::realtime::sortformer_run_encoder(model, posenc_x, pos_emb, pad_mask, att_mask);
            const auto ref = llama::realtime::load_matrix_f32_bin(encoder_ref_dir + "\\encoder_out.bin");
            const auto max_abs = llama::realtime::sortformer_max_abs_diff(out, ref);
            const auto rmse = llama::realtime::sortformer_rmse(out, ref);
            std::cout << "  out: max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            return 0;
        }

        if (!frontend_ref_dir.empty()) {
            std::cout << "Running native Sortformer frontend encoder:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << frontend_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";
            const auto features = llama::realtime::load_matrix_f32_bin(frontend_ref_dir + "\\features.bin");
            const auto outputs = llama::realtime::sortformer_run_frontend_encoder(model, features);

            const auto compare = [&](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const char * file_name) {
                const auto ref = llama::realtime::load_matrix_f32_bin(frontend_ref_dir + "\\" + file_name);
                std::cout << "  " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "  " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            compare("preencode_out", outputs.preencode_out, "preencode_out.bin");
            compare("posenc_x", outputs.posenc_x, "posenc_x.bin");
            compare("pos_emb", outputs.pos_emb, "pos_emb.bin");
            compare("pad_mask", outputs.pad_mask, "pad_mask.bin");
            compare("att_mask", outputs.att_mask, "att_mask.bin");
            compare("encoder_out", outputs.encoder_out, "encoder_out.bin");
            return 0;
        }

        if (!full_ref_dir.empty()) {
            std::cout << "Running native Sortformer full features->preds:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << full_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";
            const auto features = llama::realtime::load_matrix_f32_bin(full_ref_dir + "\\features.bin");
            const auto frontend = llama::realtime::sortformer_run_frontend_encoder(model, features);
            const auto postnet = llama::realtime::sortformer_run_postnet(model, frontend.encoder_out, frontend.encoder_mask);

            const auto compare = [&](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const char * file_name) {
                const auto ref = llama::realtime::load_matrix_f32_bin(full_ref_dir + "\\" + file_name);
                std::cout << "  " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "  " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            compare("encoder_out", frontend.encoder_out, "fc_encoder_out.bin");
            compare("preds", postnet.preds, "preds.bin");
            return 0;
        }

        if (!postnet_ref_dir.empty()) {
            std::cout << "Running native Sortformer postnet:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << postnet_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";
            const auto fc_encoder_out = llama::realtime::load_matrix_f32_bin(postnet_ref_dir + "\\fc_encoder_out.bin");
            const auto encoder_mask = llama::realtime::load_matrix_f32_bin(postnet_ref_dir + "\\encoder_mask.bin");
            const auto outputs = llama::realtime::sortformer_run_postnet(model, fc_encoder_out, encoder_mask);

            const auto compare = [&](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const char * file_name) {
                const auto ref = llama::realtime::load_matrix_f32_bin(postnet_ref_dir + "\\" + file_name);
                std::cout << "  " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "  " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            compare("encoder_proj_out", outputs.encoder_proj_out, "encoder_proj_out.bin");
            compare("te_layer0_q", outputs.te_layer0_q, "te_layer0_q.bin");
            compare("te_layer0_k", outputs.te_layer0_k, "te_layer0_k.bin");
            compare("te_layer0_v", outputs.te_layer0_v, "te_layer0_v.bin");
            compare("te_layer0_scores_head0", outputs.te_layer0_scores_head0, "te_layer0_scores_head0.bin");
            compare("te_layer0_probs_head0", outputs.te_layer0_probs_head0, "te_layer0_probs_head0.bin");
            compare("te_layer0_context", outputs.te_layer0_context, "te_layer0_context.bin");
            compare("te_layer0_att_out", outputs.te_layer0_att_out, "te_layer0_att_out.bin");
            compare("te_layer0_att_res", outputs.te_layer0_att_res, "te_layer0_att_res.bin");
            compare("te_layer0_ln1", outputs.te_layer0_ln1, "te_layer0_ln1.bin");
            compare("te_layer0_ff_di", outputs.te_layer0_ff_di, "te_layer0_ff_di.bin");
            compare("te_layer0_ff_act", outputs.te_layer0_ff_act, "te_layer0_ff_act.bin");
            compare("te_layer0_ff_do", outputs.te_layer0_ff_do, "te_layer0_ff_do.bin");
            compare("te_layer0_out", outputs.te_layer0_out, "te_layer0_out.bin");
            compare("transformer_out", outputs.transformer_out, "transformer_out.bin");
            compare("head_hidden1", outputs.head_hidden1, "head_hidden1.bin");
            compare("head_hidden2", outputs.head_hidden2, "head_hidden2.bin");
            compare("head_hidden3", outputs.head_hidden3, "head_hidden3.bin");
            compare("head_logits", outputs.head_logits, "head_logits.bin");
            compare("preds", outputs.preds, "preds.bin");
            return 0;
        }

        if (!stream_step_ref_dir.empty()) {
            std::cout << "Running native Sortformer streaming step:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << stream_step_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";

            const auto chunk_features = llama::realtime::load_matrix_f32_bin(stream_step_ref_dir + "\\chunk_features.bin");
            const auto chunk_valid_rows_m = llama::realtime::load_matrix_f32_bin(stream_step_ref_dir + "\\chunk_valid_feature_rows.bin");
            const auto spkcache = llama::realtime::load_matrix_f32_bin(stream_step_ref_dir + "\\spkcache.bin");
            const auto fifo = llama::realtime::load_matrix_f32_bin(stream_step_ref_dir + "\\fifo.bin");
            if (chunk_valid_rows_m.rows != 1 || chunk_valid_rows_m.cols != 1) {
                throw std::runtime_error("chunk_valid_feature_rows.bin must be 1x1");
            }

            llama::realtime::sortformer_stream_cache_state cache_state;
            cache_state.spkcache = spkcache;
            cache_state.fifo = fifo;

            const auto outputs = llama::realtime::sortformer_run_stream_step(
                model,
                chunk_features,
                static_cast<uint32_t>(chunk_valid_rows_m.data[0]),
                cache_state);

            const auto compare = [&](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const char * file_name) {
                const auto ref = llama::realtime::load_matrix_f32_bin(stream_step_ref_dir + "\\" + file_name);
                std::cout << "  " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "  " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            compare("chunk_preencode", outputs.chunk_preencode, "chunk_preencode.bin");
            compare("concat_preencode", outputs.concat_preencode, "concat_preencode.bin");
            compare("encoder_proj_out", outputs.postnet.encoder_proj_out, "fc_encoder_out.bin");
            compare("encoder_mask", outputs.frontend.encoder_mask, "encoder_mask.bin");
            compare("preds_all", outputs.preds_all, "preds_all.bin");
            compare("chunk_preds", outputs.chunk_preds, "chunk_preds.bin");
            return 0;
        }

        if (!stream_session_ref_dir.empty()) {
            std::cout << "Running native Sortformer streaming session:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << stream_session_ref_dir << "\n";

            const auto model = llama::realtime::sortformer_model::load_from_gguf(gguf_path, backend_name);
            std::cout << "  actual_backend_name=" << ggml_backend_name(model.backend()) << "\n";

            namespace fs = std::filesystem;
            std::vector<fs::path> step_dirs;
            for (const auto & entry : fs::directory_iterator(stream_session_ref_dir)) {
                if (entry.is_directory() && entry.path().filename().string().rfind("step_", 0) == 0) {
                    step_dirs.push_back(entry.path());
                }
            }
            std::sort(step_dirs.begin(), step_dirs.end());
            if (step_dirs.empty()) {
                throw std::runtime_error("no step_* directories found in stream-session-ref-dir");
            }

            auto compare = [](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const std::string & ref_path) {
                const auto ref = llama::realtime::load_matrix_f32_bin(ref_path);
                std::cout << "    " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "    " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            auto state = llama::realtime::sortformer_make_stream_runtime_state(model.metadata());
            for (const auto & step_dir : step_dirs) {
                std::cout << "  step=" << step_dir.filename().string() << "\n";
                const auto chunk_features = llama::realtime::load_matrix_f32_bin((step_dir / "chunk_features.bin").string());
                const auto chunk_valid_rows_m = llama::realtime::load_matrix_f32_bin((step_dir / "chunk_valid_feature_rows.bin").string());
                const auto left_context_rows_m = llama::realtime::load_matrix_f32_bin((step_dir / "left_context_rows.bin").string());
                const auto right_context_rows_m = llama::realtime::load_matrix_f32_bin((step_dir / "right_context_rows.bin").string());
                if (chunk_valid_rows_m.rows != 1 || chunk_valid_rows_m.cols != 1 ||
                    left_context_rows_m.rows != 1 || left_context_rows_m.cols != 1 ||
                    right_context_rows_m.rows != 1 || right_context_rows_m.cols != 1) {
                    throw std::runtime_error("stream-session scalar matrix must be 1x1");
                }

                const auto outputs = llama::realtime::sortformer_streaming_update(
                    model,
                    chunk_features,
                    static_cast<uint32_t>(chunk_valid_rows_m.data[0]),
                    static_cast<uint32_t>(left_context_rows_m.data[0]),
                    static_cast<uint32_t>(right_context_rows_m.data[0]),
                    state);

                if (!dump_session_actual_dir.empty()) {
                    const auto dump_step_dir = fs::path(dump_session_actual_dir) / step_dir.filename();
                    fs::create_directories(dump_step_dir);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_preencode.bin").string(), outputs.chunk_preencode);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_core_preencode.bin").string(), outputs.chunk_core_preencode);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "concat_preencode.bin").string(), outputs.concat_preencode);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "encoder_proj_out.bin").string(), outputs.postnet.encoder_proj_out);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "preds_all.bin").string(), outputs.preds_all);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_preds.bin").string(), outputs.chunk_preds);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "spkcache_after.bin").string(), state.spkcache);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "spkcache_preds_after.bin").string(), state.spkcache_preds);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "fifo_after.bin").string(), state.fifo);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "fifo_preds_after.bin").string(), state.fifo_preds);

                    llama::realtime::sortformer_matrix_f32 mean_sil_dump;
                    mean_sil_dump.rows = 1;
                    mean_sil_dump.cols = static_cast<uint32_t>(state.mean_sil_emb.size());
                    mean_sil_dump.data = state.mean_sil_emb;
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "mean_sil_emb_after.bin").string(), mean_sil_dump);

                    llama::realtime::sortformer_matrix_f32 n_sil_dump;
                    n_sil_dump.rows = 1;
                    n_sil_dump.cols = 1;
                    n_sil_dump.data = {static_cast<float>(state.n_sil_frames)};
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "n_sil_frames_after.bin").string(), n_sil_dump);
                }

                compare("chunk_preencode", outputs.chunk_preencode, (step_dir / "chunk_preencode.bin").string());
                compare("chunk_core_preencode", outputs.chunk_core_preencode, (step_dir / "chunk_core_preencode.bin").string());
                compare("concat_preencode", outputs.concat_preencode, (step_dir / "concat_preencode.bin").string());
                compare("encoder_proj_out", outputs.postnet.encoder_proj_out, (step_dir / "encoder_proj_out.bin").string());
                compare("preds_all", outputs.preds_all, (step_dir / "preds_all.bin").string());
                compare("chunk_preds", outputs.chunk_preds, (step_dir / "chunk_preds.bin").string());
                compare("spkcache_after", state.spkcache, (step_dir / "spkcache_after.bin").string());
                compare("spkcache_preds_after", state.spkcache_preds, (step_dir / "spkcache_preds_after.bin").string());
                compare("fifo_after", state.fifo, (step_dir / "fifo_after.bin").string());
                compare("fifo_preds_after", state.fifo_preds, (step_dir / "fifo_preds_after.bin").string());

                llama::realtime::sortformer_matrix_f32 mean_sil_actual;
                mean_sil_actual.rows = 1;
                mean_sil_actual.cols = static_cast<uint32_t>(state.mean_sil_emb.size());
                mean_sil_actual.data = state.mean_sil_emb;
                compare("mean_sil_emb_after", mean_sil_actual, (step_dir / "mean_sil_emb_after.bin").string());

                llama::realtime::sortformer_matrix_f32 n_sil_actual;
                n_sil_actual.rows = 1;
                n_sil_actual.cols = 1;
                n_sil_actual.data = {static_cast<float>(state.n_sil_frames)};
                compare("n_sil_frames_after", n_sil_actual, (step_dir / "n_sil_frames_after.bin").string());
            }
            return 0;
        }

        if (!stream_audio_ref_dir.empty()) {
            if (audio_wav_path.empty()) {
                throw std::runtime_error("--stream-audio-ref-dir requires --audio-wav");
            }

            std::cout << "Running native Sortformer audio-fed streaming session:\n";
            std::cout << "  backend=" << backend_name << "\n";
            std::cout << "  ref_dir=" << stream_audio_ref_dir << "\n";
            std::cout << "  audio_wav=" << audio_wav_path << "\n";

            namespace fs = std::filesystem;
            std::vector<fs::path> step_dirs;
            for (const auto & entry : fs::directory_iterator(stream_audio_ref_dir)) {
                if (entry.is_directory() && entry.path().filename().string().rfind("step_", 0) == 0) {
                    step_dirs.push_back(entry.path());
                }
            }
            std::sort(step_dirs.begin(), step_dirs.end());
            if (step_dirs.empty()) {
                throw std::runtime_error("no step_* directories found in stream-audio-ref-dir");
            }

            uint32_t wav_sample_rate = 0;
            const auto audio = load_wav_mono_f32(audio_wav_path, wav_sample_rate);
            auto backend = std::make_unique<llama::realtime::sortformer_stream_backend>(gguf_path, backend_name, true);
            auto * backend_ptr = backend.get();
            llama::realtime::stream_manager manager;
            const int64_t session_id = manager.create_session(std::move(backend));
            std::cout << "  actual_backend_name=" << backend_ptr->backend_name() << "\n";

            const size_t feed_samples = std::max<size_t>(1, static_cast<size_t>((feed_ms / 1000.0) * wav_sample_rate));
            for (size_t offset = 0; offset < audio.size(); offset += feed_samples) {
                const size_t n = std::min(feed_samples, audio.size() - offset);
                manager.push_audio(session_id, audio.data() + static_cast<ptrdiff_t>(offset), n, wav_sample_rate);
            }
            manager.flush_session(session_id);
            const auto events = manager.drain_events(session_id, 0);
            std::cout << "  emitted_events=" << events.size() << "\n";
            if (!dump_events_json.empty()) {
                write_events_json(dump_events_json, backend_ptr->backend_name(), events);
                std::cout << "  dumped_events_json=" << dump_events_json << "\n";
            }

            const auto compare = [](const char * label, const llama::realtime::sortformer_matrix_f32 & actual, const std::string & ref_path) {
                const auto ref = llama::realtime::load_matrix_f32_bin(ref_path);
                std::cout << "    " << label << " shapes: actual=" << actual.rows << "x" << actual.cols << " ref=" << ref.rows << "x" << ref.cols << "\n";
                const auto max_abs = llama::realtime::sortformer_max_abs_diff(actual, ref);
                const auto rmse = llama::realtime::sortformer_rmse(actual, ref);
                std::cout << "    " << label << ": max_abs_diff=" << max_abs << " rmse=" << rmse << "\n";
            };

            const auto & debug_steps = backend_ptr->debug_steps();
            if (debug_steps.size() < step_dirs.size()) {
                throw std::runtime_error("audio-fed backend produced fewer steps than the reference");
            }

            for (size_t i = 0; i < step_dirs.size(); ++i) {
                const auto & step_dir = step_dirs[i];
                const auto & debug = debug_steps[i];
                std::cout << "  step=" << step_dir.filename().string() << "\n";

                if (!dump_session_actual_dir.empty()) {
                    const auto dump_step_dir = fs::path(dump_session_actual_dir) / step_dir.filename();
                    fs::create_directories(dump_step_dir);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_features.bin").string(), debug.chunk_features);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_preencode.bin").string(), debug.outputs.chunk_preencode);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_core_preencode.bin").string(), debug.outputs.chunk_core_preencode);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "concat_preencode.bin").string(), debug.outputs.concat_preencode);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "encoder_proj_out.bin").string(), debug.outputs.postnet.encoder_proj_out);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "preds_all.bin").string(), debug.outputs.preds_all);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "chunk_preds.bin").string(), debug.outputs.chunk_preds);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "spkcache_after.bin").string(), debug.state_after.spkcache);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "spkcache_preds_after.bin").string(), debug.state_after.spkcache_preds);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "fifo_after.bin").string(), debug.state_after.fifo);
                    llama::realtime::save_matrix_f32_bin((dump_step_dir / "fifo_preds_after.bin").string(), debug.state_after.fifo_preds);
                }

                compare("chunk_features", debug.chunk_features, (step_dir / "chunk_features.bin").string());
                compare("chunk_preencode", debug.outputs.chunk_preencode, (step_dir / "chunk_preencode.bin").string());
                compare("chunk_core_preencode", debug.outputs.chunk_core_preencode, (step_dir / "chunk_core_preencode.bin").string());
                compare("concat_preencode", debug.outputs.concat_preencode, (step_dir / "concat_preencode.bin").string());
                compare("encoder_proj_out", debug.outputs.postnet.encoder_proj_out, (step_dir / "encoder_proj_out.bin").string());
                compare("preds_all", debug.outputs.preds_all, (step_dir / "preds_all.bin").string());
                compare("chunk_preds", debug.outputs.chunk_preds, (step_dir / "chunk_preds.bin").string());
                compare("spkcache_after", debug.state_after.spkcache, (step_dir / "spkcache_after.bin").string());
                compare("spkcache_preds_after", debug.state_after.spkcache_preds, (step_dir / "spkcache_preds_after.bin").string());
                compare("fifo_after", debug.state_after.fifo, (step_dir / "fifo_after.bin").string());
                compare("fifo_preds_after", debug.state_after.fifo_preds, (step_dir / "fifo_preds_after.bin").string());

                llama::realtime::sortformer_matrix_f32 mean_sil_actual;
                mean_sil_actual.rows = 1;
                mean_sil_actual.cols = static_cast<uint32_t>(debug.state_after.mean_sil_emb.size());
                mean_sil_actual.data = debug.state_after.mean_sil_emb;
                compare("mean_sil_emb_after", mean_sil_actual, (step_dir / "mean_sil_emb_after.bin").string());

                llama::realtime::sortformer_matrix_f32 n_sil_actual;
                n_sil_actual.rows = 1;
                n_sil_actual.cols = 1;
                n_sil_actual.data = {static_cast<float>(debug.state_after.n_sil_frames)};
                compare("n_sil_frames_after", n_sil_actual, (step_dir / "n_sil_frames_after.bin").string());
            }

            manager.close_session(session_id);
            return 0;
        }

        const uint64_t total_samples = static_cast<uint64_t>(simulate_seconds * meta.sample_rate_hz);
        const uint64_t feed_samples = std::max<uint64_t>(1, static_cast<uint64_t>((feed_ms / 1000.0) * meta.sample_rate_hz));

        std::cout << "Simulating PCM feed:\n";
        std::cout << "  total_samples=" << total_samples << "\n";
        std::cout << "  feed_samples=" << feed_samples << "\n";

        uint64_t fed = 0;
        while (fed < total_samples) {
            const uint64_t n = std::min<uint64_t>(feed_samples, total_samples - fed);
            fed += n;
            state.set_available_pcm_samples(fed);

            while (state.has_ready_chunk()) {
                const auto req = state.next_chunk();
                std::cout
                    << "chunk[" << req.chunk_index << "]"
                    << " input_frames=[" << req.input_begin_frame << "," << req.nominal_input_end_frame << ")"
                    << " valid_input=" << req.valid_input_feature_frames
                    << " emit_frames=[" << req.emit_begin_frame << "," << req.nominal_emit_end_frame << ")"
                    << " avail_frames=" << req.available_feature_frames
                    << " lc_rows=" << req.left_context_rows
                    << " rc_rows=" << req.right_context_rows
                    << "\n";
                state.mark_chunk_complete();
            }
        }

        state.set_flushing(true);
        while (state.has_ready_chunk()) {
            const auto req = state.next_chunk();
            std::cout
                << "flush-chunk[" << req.chunk_index << "]"
                << " input_frames=[" << req.input_begin_frame << "," << req.nominal_input_end_frame << ")"
                << " valid_input=" << req.valid_input_feature_frames
                << " emit_frames=[" << req.emit_begin_frame << "," << req.nominal_emit_end_frame << ")"
                << " avail_frames=" << req.available_feature_frames
                << " lc_rows=" << req.left_context_rows
                << " rc_rows=" << req.right_context_rows
                << "\n";
            state.mark_chunk_complete();
        }

        std::cout << "Completed chunks: " << state.completed_chunks() << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
