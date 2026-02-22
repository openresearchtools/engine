#include "llama_server_bridge.h"

#include "common.h"
#include "server-common.h"
#include "server-context.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

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
    bool backend_acquired = false;
};

static std::mutex g_backend_mutex;
static int g_backend_refcount = 0;

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
        return true;
    }
    const std::string raw = trim_copy(devices_csv);
    if (raw.empty()) {
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
    p.main_gpu = 0;
    p.no_kv_offload = 0;
    p.mmproj_use_gpu = 1;
    p.cache_ram_mib = -1;
    p.seed = -1;
    p.ctx_shift = 1;
    p.kv_unified = 1;
    p.devices = nullptr;
    p.tensor_split = nullptr;
    p.split_mode = -1;
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

struct llama_server_bridge_json_result llama_server_bridge_empty_json_result(void) {
    llama_server_bridge_json_result out = {};
    out.ok = 0;
    out.status = 0;
    out.json = nullptr;
    out.error_json = nullptr;
    return out;
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

    bridge->params.n_ctx = std::max<int32_t>(0, params->n_ctx);
    bridge->params.n_batch = std::max<int32_t>(32, params->n_batch);
    bridge->params.n_ubatch = std::max<int32_t>(32, params->n_ubatch);
    bridge->params.n_parallel = std::max<int32_t>(1, params->n_parallel);
    bridge->params.n_gpu_layers = params->n_gpu_layers;
    bridge->params.main_gpu = std::max<int32_t>(0, params->main_gpu);
    bridge->params.no_kv_offload = params->no_kv_offload != 0;
    bridge->params.mmproj_use_gpu = params->mmproj_use_gpu != 0;
    bridge->params.no_mmproj = bridge->params.mmproj.path.empty();
    bridge->params.cache_ram_mib = params->cache_ram_mib;
    bridge->params.ctx_shift = params->ctx_shift != 0;
    bridge->params.kv_unified = params->kv_unified != 0;
    bridge->params.embedding = params->embedding != 0;

    if (params->pooling_type >= 0) {
        if (!is_valid_pooling_type(params->pooling_type)) {
            set_bridge_error(bridge.get(), "invalid pooling_type, expected -1..4");
            std::fprintf(stderr, "llama_server_bridge_create: invalid pooling_type=%d\n", params->pooling_type);
            return nullptr;
        }
        bridge->params.pooling_type = static_cast<enum llama_pooling_type>(params->pooling_type);
    }
    if (params->reranking != 0) {
        bridge->params.embedding = true;
        bridge->params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }
    if (bridge->params.embedding) {
        bridge->params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    std::string parse_error;
    if (!parse_split_mode(params->split_mode, &bridge->params.split_mode, parse_error)) {
        set_bridge_error(bridge.get(), parse_error);
        std::fprintf(stderr, "llama_server_bridge_create: split_mode parse failed: %s\n", parse_error.c_str());
        return nullptr;
    }
    if (!parse_tensor_split_csv(params->tensor_split, bridge->params.tensor_split, parse_error)) {
        set_bridge_error(bridge.get(), parse_error);
        std::fprintf(stderr, "llama_server_bridge_create: tensor_split parse failed: %s\n", parse_error.c_str());
        return nullptr;
    }

    if (params->n_threads > 0) {
        bridge->params.cpuparams.n_threads = params->n_threads;
    }
    if (params->n_threads_batch > 0) {
        bridge->params.cpuparams_batch.n_threads = params->n_threads_batch;
    } else if (params->n_threads > 0) {
        bridge->params.cpuparams_batch.n_threads = params->n_threads;
    }
    if (params->seed >= 0) {
        bridge->params.sampling.seed = (uint32_t) params->seed;
    }

    acquire_backend(bridge->params);
    bridge->backend_acquired = true;

    if (!parse_devices_csv(params->devices, bridge->params.devices, parse_error)) {
        set_bridge_error(bridge.get(), parse_error);
        std::fprintf(stderr, "llama_server_bridge_create: devices parse failed: %s\n", parse_error.c_str());
        release_backend();
        bridge->backend_acquired = false;
        return nullptr;
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

    bridge->loop_thread = std::thread([raw = bridge.get()]() {
        raw->ctx.start_loop();
    });

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

    return run_json_route(
        bridge,
        handler,
        "/v1/audio/transcriptions",
        req->body_json,
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
