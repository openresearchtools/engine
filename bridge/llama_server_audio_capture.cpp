#include "llama_server_audio_capture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
#include "webrtc/modules/audio_processing/include/audio_processing.h"
#endif

namespace fs = std::filesystem;

namespace {

constexpr uint32_t k_capture_sample_rate_hz = 48000;
constexpr uint32_t k_bridge_sample_rate_hz = 16000;
constexpr uint32_t k_bridge_channels = 1;
constexpr uint32_t k_capture_channels = 1;
constexpr size_t k_apm_frame_samples = k_capture_sample_rate_hz / 100;
constexpr uint32_t k_default_bridge_push_samples = 7680;

struct queued_event_record {
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

struct transcript_piece_record {
    uint64_t start_sample = 0;
    uint64_t end_sample = 0;
    std::string text;
};

static char * dup_cstr(const std::string & text) {
    char * out = static_cast<char *>(std::malloc(text.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

static void free_cstr(char * text) {
    if (text != nullptr) {
        std::free(text);
    }
}

static std::string trim_copy(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

static std::string sanitize_name(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        const bool keep = std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '-' || ch == '_';
        out.push_back(keep ? ch : '-');
    }
    while (!out.empty() && out.front() == '-') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out.empty() ? std::string("live-session") : out;
}

static std::string make_timestamped_session_name() {
    const auto now = std::chrono::system_clock::now();
    const auto as_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm = {};
#if defined(_WIN32)
    localtime_s(&local_tm, &as_time_t);
#else
    localtime_r(&as_time_t, &local_tm);
#endif
    std::ostringstream out;
    out << "live-";
    out << (local_tm.tm_year + 1900);
    out << '-';
    out.width(2);
    out.fill('0');
    out << (local_tm.tm_mon + 1);
    out << '-';
    out.width(2);
    out.fill('0');
    out << local_tm.tm_mday;
    out << '-';
    out.width(2);
    out.fill('0');
    out << local_tm.tm_hour;
    out << local_tm.tm_min;
    out << local_tm.tm_sec;
    return out.str();
}

static std::string format_timestamp_from_samples(uint64_t sample_index, uint32_t sample_rate_hz) {
    if (sample_rate_hz == 0) {
        sample_rate_hz = k_bridge_sample_rate_hz;
    }
    const uint64_t total_ms = (sample_index * 1000ull) / sample_rate_hz;
    const uint64_t hours = total_ms / 3600000ull;
    const uint64_t minutes = (total_ms % 3600000ull) / 60000ull;
    const uint64_t seconds = (total_ms % 60000ull) / 1000ull;
    const uint64_t millis = total_ms % 1000ull;

    std::ostringstream out;
    if (hours > 0) {
        out.width(2);
        out.fill('0');
        out << hours << ':';
    }
    out.width(2);
    out.fill('0');
    out << minutes << ':';
    out.width(2);
    out.fill('0');
    out << seconds << '.';
    out.width(3);
    out.fill('0');
    out << millis;
    return out.str();
}

static bool insert_unique_piece(std::vector<transcript_piece_record> & pieces, transcript_piece_record piece) {
    const auto duplicate = std::find_if(
        pieces.begin(),
        pieces.end(),
        [&](const transcript_piece_record & existing) {
            return existing.start_sample == piece.start_sample
                && existing.end_sample == piece.end_sample
                && existing.text == piece.text;
        });
    if (duplicate != pieces.end()) {
        return false;
    }
    pieces.push_back(std::move(piece));
    return true;
}

static std::string join_piece_texts(const std::vector<transcript_piece_record> & pieces) {
    std::ostringstream out;
    bool first = true;
    for (const auto & piece : pieces) {
        const std::string text = trim_copy(piece.text);
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

static std::string render_unassigned_markdown(const std::vector<transcript_piece_record> & pieces) {
    if (pieces.empty()) {
        return std::string();
    }
    const std::string plain = join_piece_texts(pieces);
    if (plain.empty()) {
        return std::string();
    }
    std::ostringstream out;
    out << "### UNASSIGNED [" << format_timestamp_from_samples(pieces.front().start_sample, k_bridge_sample_rate_hz)
        << " - " << format_timestamp_from_samples(pieces.back().end_sample, k_bridge_sample_rate_hz) << "]\n\n";
    out << plain << '\n';
    return out.str();
}

static std::string read_event_text(const char * text) {
    return text != nullptr ? std::string(text) : std::string();
}

static bool ensure_directory(const fs::path & path, std::string & error) {
    std::error_code ec;
    if (fs::create_directories(path, ec) || fs::exists(path, ec)) {
        return true;
    }
    error = "failed to create output directory '" + path.string() + "': " + ec.message();
    return false;
}

class wav_writer {
public:
    bool open(const fs::path & path, uint32_t sample_rate_hz, uint16_t channels, std::string & error);
    bool write_samples(const int16_t * samples, size_t sample_count, std::string & error);
    void close();
    ~wav_writer();

private:
    void write_u16(uint16_t value);
    void write_u32(uint32_t value);
    void write_header();
    void patch_header();

    fs::path path_;
    std::ofstream stream_;
    uint32_t sample_rate_hz_ = 0;
    uint16_t channels_ = 0;
    uint32_t bytes_written_ = 0;
};

class resampler_48k_to_16k {
public:
    bool init(std::string & error);
    void uninit();
    bool process(const int16_t * input, size_t input_frames, std::vector<int16_t> & output, std::string & error);
    ~resampler_48k_to_16k();

private:
    ma_resampler resampler_ = {};
    bool initialized_ = false;
};

#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
class webrtc_processor {
public:
    bool init(std::string & error);
    bool process_frame(const int16_t * input, size_t input_samples, std::array<int16_t, k_apm_frame_samples> & output, std::string & error);

private:
    struct apm_deleter {
        void operator()(webrtc::AudioProcessing * ptr) const {
            delete ptr;
        }
    };

    std::unique_ptr<webrtc::AudioProcessing, apm_deleter> apm_;
};
#endif

} // namespace

struct llama_server_audio_live {
    explicit llama_server_audio_live(const llama_server_audio_live_params & value)
        : params(value) {
    }

    llama_server_audio_live_params params = llama_server_audio_default_live_params();
    std::string output_dir;
    std::string session_name;
    std::string cleaned_wav_path;
    std::string transcript_path;
    std::string preview_path;

    std::mutex mutex;
    std::condition_variable cv;
    std::deque<queued_event_record> queued_events;
    std::deque<std::vector<int16_t>> capture_chunks;
    std::thread worker;
    std::atomic<bool> started{false};
    bool stop_requested = false;
    bool worker_finished = false;
    bool device_started = false;
    bool bridge_session_started = false;
    bool transcription_started = false;
    bool diarization_started = false;
    bool transcription_done = false;
    bool diarization_done = true;
    bool preview_written = false;
    bool final_written = false;
    bool backend_preview_seen = false;
    std::string last_error;
    uint64_t next_seq_no = 1;
    uint32_t next_notice_item_id = 1;

    ma_context context = {};
    ma_device device = {};
    bool context_initialized = false;
    bool device_initialized = false;
    ma_device_id selected_device_id = {};
    bool has_selected_device = false;

    llama_server_bridge_audio_session * session = nullptr;
    wav_writer cleaned_wav;
    resampler_48k_to_16k resampler;
#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
    webrtc_processor apm;
#endif
    bool webrtc_runtime_enabled = false;
    std::vector<int16_t> frame_buffer_48k;
    size_t frame_buffer_offset = 0;
    std::vector<int16_t> bridge_buffer_16k;
    size_t bridge_buffer_offset = 0;
    std::vector<transcript_piece_record> raw_pieces;
    std::string last_plain_transcript;
    std::string last_preview_markdown;
    std::string last_final_markdown;
};

namespace {

static void free_audio_event_fields(llama_server_bridge_audio_event & event);
static void free_output_paths_fields(llama_server_audio_output_paths & paths);
static std::string bridge_session_last_error(const llama_server_audio_live * live);
static void set_last_error(llama_server_audio_live * live, std::string message);
static void push_queued_event_locked(llama_server_audio_live * live, queued_event_record event);
static void push_notice_event(llama_server_audio_live * live, const std::string & text, const std::string & detail);
static void push_error_event(llama_server_audio_live * live, const std::string & text);
static bool write_text_file(const std::string & path, const std::string & text, std::string & error);
static bool initialize_output_files(llama_server_audio_live * live);
static bool update_preview_file(llama_server_audio_live * live, const std::string & markdown);
static bool update_transcript_file(llama_server_audio_live * live, const std::string & text);
static void compact_buffer_prefix(std::vector<int16_t> & buffer, size_t & offset);
static bool pump_bridge_events_nonblocking(llama_server_audio_live * live);
static bool push_bridge_samples(llama_server_audio_live * live, const int16_t * samples, size_t sample_count);
static bool flush_bridge_tail(llama_server_audio_live * live);
static bool append_resampled_samples(llama_server_audio_live * live, const std::vector<int16_t> & samples);
static bool process_capture_frame(llama_server_audio_live * live, const int16_t * samples, size_t sample_count);
static bool flush_remaining_capture_tail(llama_server_audio_live * live);
static bool process_capture_chunk(llama_server_audio_live * live, const std::vector<int16_t> & chunk);
static void cleanup_device(llama_server_audio_live * live);
static void cleanup_session(llama_server_audio_live * live);
static void audio_capture_data_callback(ma_device * device, void * output, const void * input, ma_uint32 frame_count);
static void audio_capture_notification_callback(const ma_device_notification * notification);
static bool resolve_capture_device(llama_server_audio_live * live, std::string & error);
static bool initialize_bridge_session(llama_server_audio_live * live, std::string & error);
static bool initialize_output_paths(llama_server_audio_live * live, std::string & error);
static bool initialize_audio_pipeline(llama_server_audio_live * live, std::string & error);
static bool finish_bridge_session(llama_server_audio_live * live);
static void worker_main(llama_server_audio_live * live);
static bool start_live_capture(llama_server_audio_live * live, std::string & error);
static void stop_live_capture(llama_server_audio_live * live);

} // namespace

bool wav_writer::open(const fs::path & path, uint32_t sample_rate_hz, uint16_t channels, std::string & error) {
    close();
    stream_.open(path, std::ios::binary | std::ios::trunc);
    if (!stream_.is_open()) {
        error = "failed to open cleaned wav output '" + path.string() + "'";
        return false;
    }
    path_ = path;
    sample_rate_hz_ = sample_rate_hz;
    channels_ = channels;
    bytes_written_ = 0;
    write_header();
    return true;
}

bool wav_writer::write_samples(const int16_t * samples, size_t sample_count, std::string & error) {
    if (!stream_.is_open() || samples == nullptr || sample_count == 0) {
        return true;
    }
    const size_t bytes = sample_count * sizeof(int16_t);
    stream_.write(reinterpret_cast<const char *>(samples), static_cast<std::streamsize>(bytes));
    if (!stream_) {
        error = "failed to write cleaned wav output '" + path_.string() + "'";
        return false;
    }
    bytes_written_ += static_cast<uint32_t>(bytes);
    return true;
}

void wav_writer::close() {
    if (!stream_.is_open()) {
        return;
    }
    patch_header();
    stream_.close();
}

wav_writer::~wav_writer() {
    close();
}

void wav_writer::write_u16(uint16_t value) {
    stream_.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void wav_writer::write_u32(uint32_t value) {
    stream_.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void wav_writer::write_header() {
    const uint32_t byte_rate = sample_rate_hz_ * channels_ * sizeof(int16_t);
    const uint16_t block_align = channels_ * sizeof(int16_t);

    stream_.write("RIFF", 4);
    write_u32(36);
    stream_.write("WAVE", 4);
    stream_.write("fmt ", 4);
    write_u32(16);
    write_u16(1);
    write_u16(channels_);
    write_u32(sample_rate_hz_);
    write_u32(byte_rate);
    write_u16(block_align);
    write_u16(16);
    stream_.write("data", 4);
    write_u32(0);
}

void wav_writer::patch_header() {
    if (!stream_.is_open()) {
        return;
    }
    stream_.seekp(4, std::ios::beg);
    write_u32(36 + bytes_written_);
    stream_.seekp(40, std::ios::beg);
    write_u32(bytes_written_);
    stream_.seekp(0, std::ios::end);
}

bool resampler_48k_to_16k::init(std::string & error) {
    if (initialized_) {
        return true;
    }
    ma_resampler_config config = ma_resampler_config_init(
        ma_format_s16,
        k_bridge_channels,
        k_capture_sample_rate_hz,
        k_bridge_sample_rate_hz,
        ma_resample_algorithm_linear);
    const ma_result result = ma_resampler_init(&config, nullptr, &resampler_);
    if (result != MA_SUCCESS) {
        error = "ma_resampler_init() failed";
        return false;
    }
    initialized_ = true;
    return true;
}

void resampler_48k_to_16k::uninit() {
    if (!initialized_) {
        return;
    }
    ma_resampler_uninit(&resampler_, nullptr);
    initialized_ = false;
}

bool resampler_48k_to_16k::process(
    const int16_t * input,
    size_t input_frames,
    std::vector<int16_t> & output,
    std::string & error) {
    if (!initialized_) {
        error = "resampler is not initialized";
        return false;
    }
    if (input_frames == 0) {
        output.clear();
        return true;
    }

    ma_uint64 in_frames = static_cast<ma_uint64>(input_frames);
    ma_uint64 out_frames = (in_frames * k_bridge_sample_rate_hz + k_capture_sample_rate_hz - 1)
        / k_capture_sample_rate_hz + 8;
    output.assign(static_cast<size_t>(out_frames), 0);
    const ma_result result = ma_resampler_process_pcm_frames(
        &resampler_,
        input,
        &in_frames,
        output.data(),
        &out_frames);
    if (result != MA_SUCCESS) {
        error = "ma_resampler_process_pcm_frames() failed";
        return false;
    }
    output.resize(static_cast<size_t>(out_frames));
    return true;
}

resampler_48k_to_16k::~resampler_48k_to_16k() {
    uninit();
}

#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
bool webrtc_processor::init(std::string & error) {
    if (apm_ != nullptr) {
        return true;
    }

    apm_.reset(webrtc::AudioProcessing::Create());
    if (!apm_) {
        error = "AudioProcessing::Create() failed";
        return false;
    }

    webrtc::ProcessingConfig config;
    config.input_stream() = webrtc::StreamConfig(k_capture_sample_rate_hz, 1, false);
    config.output_stream() = webrtc::StreamConfig(k_capture_sample_rate_hz, 1, false);
    config.reverse_input_stream() = webrtc::StreamConfig(k_capture_sample_rate_hz, 1, false);
    config.reverse_output_stream() = webrtc::StreamConfig(k_capture_sample_rate_hz, 1, false);

    if (apm_->Initialize(config) != webrtc::AudioProcessing::kNoError) {
        error = "AudioProcessing::Initialize() failed";
        apm_.reset();
        return false;
    }

    auto * hpf = apm_->high_pass_filter();
    auto * ns = apm_->noise_suppression();
    auto * gc = apm_->gain_control();

    if (hpf == nullptr || ns == nullptr || gc == nullptr) {
        error = "AudioProcessing component lookup failed";
        apm_.reset();
        return false;
    }

    hpf->Enable(true);
    ns->set_level(webrtc::NoiseSuppression::kModerate);
    ns->Enable(true);
    gc->set_mode(webrtc::GainControl::kAdaptiveDigital);
    gc->set_target_level_dbfs(3);
    gc->set_compression_gain_db(9);
    gc->enable_limiter(true);
    gc->Enable(true);
    return true;
}

bool webrtc_processor::process_frame(
    const int16_t * input,
    size_t input_samples,
    std::array<int16_t, k_apm_frame_samples> & output,
    std::string & error) {
    if (apm_ == nullptr) {
        std::copy_n(input, std::min(input_samples, output.size()), output.begin());
        if (input_samples < output.size()) {
            std::fill(output.begin() + static_cast<std::ptrdiff_t>(input_samples), output.end(), 0);
        }
        return true;
    }
    if (input == nullptr || input_samples != k_apm_frame_samples) {
        error = "WebRTC APM expects exactly 10 ms capture frames";
        return false;
    }

    std::array<float, k_apm_frame_samples> in_channel = {};
    std::array<float, k_apm_frame_samples> out_channel = {};
    for (size_t i = 0; i < input_samples; ++i) {
        in_channel[i] = static_cast<float>(input[i]) / 32768.0f;
    }

    const float * in_ptrs[1] = {in_channel.data()};
    float * out_ptrs[1] = {out_channel.data()};
    const webrtc::StreamConfig input_cfg(k_capture_sample_rate_hz, 1, false);
    const webrtc::StreamConfig output_cfg(k_capture_sample_rate_hz, 1, false);
    if (apm_->ProcessStream(in_ptrs, input_cfg, output_cfg, out_ptrs) != webrtc::AudioProcessing::kNoError) {
        error = "AudioProcessing::ProcessStream() failed";
        return false;
    }

    for (size_t i = 0; i < output.size(); ++i) {
        const float scaled = std::clamp(out_channel[i], -1.0f, 1.0f) * 32767.0f;
        output[i] = static_cast<int16_t>(scaled);
    }
    return true;
}
#endif

namespace {

static void free_audio_event_fields(llama_server_bridge_audio_event & event) {
    free_cstr(event.text);
    free_cstr(event.detail);
    event.text = nullptr;
    event.detail = nullptr;
}

static void free_output_paths_fields(llama_server_audio_output_paths & paths) {
    free_cstr(paths.output_dir);
    free_cstr(paths.cleaned_wav_path);
    free_cstr(paths.transcript_path);
    free_cstr(paths.preview_path);
    paths.output_dir = nullptr;
    paths.cleaned_wav_path = nullptr;
    paths.transcript_path = nullptr;
    paths.preview_path = nullptr;
}

static std::string bridge_session_last_error(const llama_server_audio_live * live) {
    if (live == nullptr || live->session == nullptr) {
        return std::string();
    }
    const char * error = llama_server_bridge_audio_session_last_error(live->session);
    return error != nullptr ? std::string(error) : std::string();
}

static void set_last_error(llama_server_audio_live * live, std::string message) {
    if (live == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(live->mutex);
    live->last_error = std::move(message);
    live->cv.notify_all();
}

static void push_queued_event_locked(llama_server_audio_live * live, queued_event_record event) {
    if (live == nullptr) {
        return;
    }
    if (event.seq_no == 0) {
        event.seq_no = live->next_seq_no++;
    } else if (event.seq_no >= live->next_seq_no) {
        live->next_seq_no = event.seq_no + 1;
    }
    if (live->params.event_queue_capacity > 0
        && live->queued_events.size() >= live->params.event_queue_capacity) {
        live->queued_events.pop_front();
    }
    live->queued_events.push_back(std::move(event));
    live->cv.notify_all();
}

static void push_notice_event(llama_server_audio_live * live, const std::string & text, const std::string & detail) {
    if (live == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(live->mutex);
    queued_event_record event = {};
    event.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE;
    event.item_id = live->next_notice_item_id++;
    event.text = text;
    event.detail = detail;
    push_queued_event_locked(live, std::move(event));
}

static void push_error_event(llama_server_audio_live * live, const std::string & text) {
    if (live == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(live->mutex);
    live->last_error = text;
    queued_event_record event = {};
    event.kind = LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR;
    event.item_id = live->next_notice_item_id++;
    event.text = text;
    push_queued_event_locked(live, std::move(event));
}

static bool write_text_file(const std::string & path, const std::string & text, std::string & error) {
    if (path.empty()) {
        return true;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        error = "failed to open output file '" + path + "'";
        return false;
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) {
        error = "failed to write output file '" + path + "'";
        return false;
    }
    return true;
}

static bool initialize_output_files(llama_server_audio_live * live) {
    if (live == nullptr) {
        return true;
    }

    std::string error;
    if (!live->transcript_path.empty() && !write_text_file(live->transcript_path, "", error)) {
        push_error_event(live, error);
        return false;
    }
    if (live->params.write_preview_file && !live->preview_path.empty()
        && !write_text_file(live->preview_path, "", error)) {
        push_error_event(live, error);
        return false;
    }
    return true;
}

static bool update_preview_file(llama_server_audio_live * live, const std::string & markdown) {
    if (live == nullptr || !live->params.write_preview_file || live->preview_path.empty()) {
        return true;
    }
    if (markdown == live->last_preview_markdown) {
        return true;
    }
    std::string error;
    if (!write_text_file(live->preview_path, markdown, error)) {
        push_error_event(live, error);
        return false;
    }
    live->last_preview_markdown = markdown;
    live->preview_written = true;
    return true;
}

static bool update_transcript_file(llama_server_audio_live * live, const std::string & text) {
    if (live == nullptr || live->transcript_path.empty()) {
        return true;
    }
    if (text == live->last_plain_transcript) {
        return true;
    }
    std::string error;
    if (!write_text_file(live->transcript_path, text, error)) {
        push_error_event(live, error);
        return false;
    }
    live->last_plain_transcript = text;
    return true;
}

static void compact_buffer_prefix(std::vector<int16_t> & buffer, size_t & offset) {
    if (offset == 0) {
        return;
    }
    if (offset >= buffer.size()) {
        buffer.clear();
        offset = 0;
        return;
    }
    if (offset >= 8192) {
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(offset));
        offset = 0;
    }
}

static bool pump_bridge_events_nonblocking(llama_server_audio_live * live) {
    if (live == nullptr || live->session == nullptr) {
        return true;
    }

    while (true) {
        const int32_t pending = llama_server_bridge_audio_session_wait_events(live->session, 0);
        if (pending < 0) {
            push_error_event(live, bridge_session_last_error(live));
            return false;
        }
        if (pending == 0) {
            break;
        }

        llama_server_bridge_audio_event * events = nullptr;
        size_t count = 0;
        if (llama_server_bridge_audio_session_drain_events(live->session, &events, &count, 0) != 0) {
            push_error_event(live, bridge_session_last_error(live));
            return false;
        }

        for (size_t i = 0; i < count; ++i) {
            const auto & event = events[i];
            if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT) {
                insert_unique_piece(
                    live->raw_pieces,
                    {event.start_sample, event.end_sample, read_event_text(event.text)});
                if (live->params.enable_diarization) {
                    if (!live->backend_preview_seen) {
                        update_preview_file(live, render_unassigned_markdown(live->raw_pieces));
                    }
                } else {
                    update_transcript_file(live, join_piece_texts(live->raw_pieces));
                }
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT) {
                const std::string markdown = read_event_text(event.text);
                live->backend_preview_seen = true;
                if ((event.flags & LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL) != 0u) {
                    live->last_final_markdown = markdown;
                    live->final_written = true;
                    std::string error;
                    if (!live->transcript_path.empty() && !write_text_file(live->transcript_path, markdown, error)) {
                        push_error_event(live, error);
                    }
                    update_preview_file(live, markdown);
                } else {
                    update_preview_file(live, markdown);
                }
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED) {
                live->transcription_done = true;
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED) {
                live->diarization_done = true;
            } else if (event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR
                || event.kind == LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR) {
                const std::string message = !read_event_text(event.text).empty()
                    ? read_event_text(event.text)
                    : read_event_text(event.detail);
                if (!message.empty()) {
                    set_last_error(live, message);
                }
            }

            queued_event_record cloned = {};
            cloned.seq_no = event.seq_no;
            cloned.kind = event.kind;
            cloned.flags = event.flags;
            cloned.start_sample = event.start_sample;
            cloned.end_sample = event.end_sample;
            cloned.speaker_id = event.speaker_id;
            cloned.item_id = event.item_id;
            cloned.text = read_event_text(event.text);
            cloned.detail = read_event_text(event.detail);
            {
                std::lock_guard<std::mutex> lock(live->mutex);
                push_queued_event_locked(live, std::move(cloned));
            }
        }

        if (events != nullptr && count > 0) {
            llama_server_bridge_audio_session_free_events(events, count);
        }
    }

    return true;
}

static bool push_bridge_samples(llama_server_audio_live * live, const int16_t * samples, size_t sample_count) {
    if (live == nullptr || live->session == nullptr || samples == nullptr || sample_count == 0) {
        return true;
    }
    if (llama_server_bridge_audio_session_push_audio(
            live->session,
            samples,
            sample_count,
            k_bridge_sample_rate_hz,
            k_bridge_channels,
            LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16) != 0) {
        push_error_event(live, bridge_session_last_error(live));
        return false;
    }
    return pump_bridge_events_nonblocking(live);
}

static bool flush_bridge_tail(llama_server_audio_live * live) {
    if (live == nullptr) {
        return true;
    }
    while (live->bridge_buffer_offset < live->bridge_buffer_16k.size()) {
        const size_t remaining = live->bridge_buffer_16k.size() - live->bridge_buffer_offset;
        if (!push_bridge_samples(live, live->bridge_buffer_16k.data() + live->bridge_buffer_offset, remaining)) {
            return false;
        }
        live->bridge_buffer_offset += remaining;
    }
    compact_buffer_prefix(live->bridge_buffer_16k, live->bridge_buffer_offset);
    return true;
}

static bool append_resampled_samples(llama_server_audio_live * live, const std::vector<int16_t> & samples) {
    if (live == nullptr || samples.empty()) {
        return true;
    }

    std::string error;
    if (live->params.write_clean_wav && !live->cleaned_wav.write_samples(samples.data(), samples.size(), error)) {
        push_error_event(live, error);
        return false;
    }

    live->bridge_buffer_16k.insert(live->bridge_buffer_16k.end(), samples.begin(), samples.end());
    while ((live->bridge_buffer_16k.size() - live->bridge_buffer_offset) >= live->params.bridge_push_samples) {
        if (!push_bridge_samples(
                live,
                live->bridge_buffer_16k.data() + live->bridge_buffer_offset,
                live->params.bridge_push_samples)) {
            return false;
        }
        live->bridge_buffer_offset += live->params.bridge_push_samples;
    }
    compact_buffer_prefix(live->bridge_buffer_16k, live->bridge_buffer_offset);
    return true;
}

static bool process_capture_frame(llama_server_audio_live * live, const int16_t * samples, size_t sample_count) {
    if (live == nullptr || samples == nullptr || sample_count == 0) {
        return true;
    }

    std::vector<int16_t> resampled;
    std::string error;
#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
    if (live->webrtc_runtime_enabled) {
        std::array<int16_t, k_apm_frame_samples> processed = {};
        if (!live->apm.process_frame(samples, sample_count, processed, error)) {
            push_error_event(live, error);
            return false;
        }
        if (!live->resampler.process(processed.data(), processed.size(), resampled, error)) {
            push_error_event(live, error);
            return false;
        }
    } else
#endif
    {
        if (!live->resampler.process(samples, sample_count, resampled, error)) {
            push_error_event(live, error);
            return false;
        }
    }
    return append_resampled_samples(live, resampled);
}

static bool flush_remaining_capture_tail(llama_server_audio_live * live) {
    if (live == nullptr) {
        return true;
    }
    const size_t remaining = live->frame_buffer_48k.size() - live->frame_buffer_offset;
    if (remaining == 0) {
        compact_buffer_prefix(live->frame_buffer_48k, live->frame_buffer_offset);
        return true;
    }

    std::vector<int16_t> tail(remaining);
    std::copy_n(live->frame_buffer_48k.data() + live->frame_buffer_offset, remaining, tail.data());
    live->frame_buffer_offset = live->frame_buffer_48k.size();
    compact_buffer_prefix(live->frame_buffer_48k, live->frame_buffer_offset);

    if (live->webrtc_runtime_enabled) {
#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
        std::array<int16_t, k_apm_frame_samples> padded = {};
        std::copy_n(tail.data(), std::min(tail.size(), padded.size()), padded.data());
        std::array<int16_t, k_apm_frame_samples> processed = {};
        std::string error;
        if (!live->apm.process_frame(padded.data(), padded.size(), processed, error)) {
            push_error_event(live, error);
            return false;
        }
        std::vector<int16_t> resampled;
        if (!live->resampler.process(processed.data(), processed.size(), resampled, error)) {
            push_error_event(live, error);
            return false;
        }
        const size_t wanted = (tail.size() * k_bridge_sample_rate_hz + k_capture_sample_rate_hz - 1)
            / k_capture_sample_rate_hz;
        if (resampled.size() > wanted) {
            resampled.resize(wanted);
        }
        return append_resampled_samples(live, resampled);
#endif
    }

    std::vector<int16_t> resampled;
    std::string error;
    if (!live->resampler.process(tail.data(), tail.size(), resampled, error)) {
        push_error_event(live, error);
        return false;
    }
    return append_resampled_samples(live, resampled);
}

static bool process_capture_chunk(llama_server_audio_live * live, const std::vector<int16_t> & chunk) {
    if (live == nullptr || chunk.empty()) {
        return true;
    }
    live->frame_buffer_48k.insert(live->frame_buffer_48k.end(), chunk.begin(), chunk.end());
    while ((live->frame_buffer_48k.size() - live->frame_buffer_offset) >= k_apm_frame_samples) {
        const int16_t * frame = live->frame_buffer_48k.data() + live->frame_buffer_offset;
        if (!process_capture_frame(live, frame, k_apm_frame_samples)) {
            return false;
        }
        live->frame_buffer_offset += k_apm_frame_samples;
    }
    compact_buffer_prefix(live->frame_buffer_48k, live->frame_buffer_offset);
    return true;
}

static void cleanup_device(llama_server_audio_live * live) {
    if (live == nullptr) {
        return;
    }
    if (live->device_started) {
        ma_device_stop(&live->device);
        live->device_started = false;
    }
    if (live->device_initialized) {
        ma_device_uninit(&live->device);
        live->device_initialized = false;
    }
    if (live->context_initialized) {
        ma_context_uninit(&live->context);
        live->context_initialized = false;
    }
}

static void cleanup_session(llama_server_audio_live * live) {
    if (live == nullptr) {
        return;
    }
    bool destroyed_session = false;
    if (live->session != nullptr) {
        llama_server_bridge_audio_session_destroy(live->session);
        live->session = nullptr;
        destroyed_session = true;
    }
    if (destroyed_session) {
        // Release cached realtime models while the DLL/runtime is still fully alive.
        llama_server_bridge_realtime_model_cache_clear();
    }
    live->bridge_session_started = false;
    live->transcription_started = false;
    live->diarization_started = false;
}

static void audio_capture_data_callback(ma_device * device, void * output, const void * input, ma_uint32 frame_count) {
    (void) output;
    if (device == nullptr || device->pUserData == nullptr || input == nullptr || frame_count == 0) {
        return;
    }
    auto * live = static_cast<llama_server_audio_live *>(device->pUserData);
    std::vector<int16_t> chunk(frame_count * k_capture_channels);
    std::memcpy(chunk.data(), input, chunk.size() * sizeof(int16_t));
    {
        std::lock_guard<std::mutex> lock(live->mutex);
        live->capture_chunks.push_back(std::move(chunk));
    }
    live->cv.notify_all();
}

static void audio_capture_notification_callback(const ma_device_notification * notification) {
    if (notification == nullptr || notification->pDevice == nullptr || notification->pDevice->pUserData == nullptr) {
        return;
    }
    auto * live = static_cast<llama_server_audio_live *>(notification->pDevice->pUserData);
    switch (notification->type) {
        case ma_device_notification_type_started:
            push_notice_event(live, "capture-started", "miniaudio capture device started");
            break;
        case ma_device_notification_type_stopped: {
            bool unexpected = false;
            {
                std::lock_guard<std::mutex> lock(live->mutex);
                unexpected = !live->stop_requested;
            }
            if (unexpected) {
                push_error_event(live, "capture device stopped unexpectedly");
            } else {
                push_notice_event(live, "capture-stopped", "miniaudio capture device stopped");
            }
            break;
        }
        case ma_device_notification_type_rerouted:
            push_notice_event(live, "capture-rerouted", "capture device rerouted");
            break;
        case ma_device_notification_type_interruption_began:
            push_notice_event(live, "capture-interrupted", "capture interruption began");
            break;
        case ma_device_notification_type_interruption_ended:
            push_notice_event(live, "capture-resumed", "capture interruption ended");
            break;
        default:
            break;
    }
}

static bool resolve_capture_device(llama_server_audio_live * live, std::string & error) {
    if (live == nullptr) {
        error = "capture live session is null";
        return false;
    }

    if (ma_context_init(nullptr, 0, nullptr, &live->context) != MA_SUCCESS) {
        error = "ma_context_init() failed";
        return false;
    }
    live->context_initialized = true;

    ma_device_info * playback_infos = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info * capture_infos = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&live->context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        error = "ma_context_get_devices() failed";
        return false;
    }

    const std::string desired_name = live->params.capture_device_name != nullptr
        ? trim_copy(live->params.capture_device_name)
        : std::string();

    if (!desired_name.empty() && live->params.capture_device_index >= 0) {
        error = "choose either capture_device_name or capture_device_index";
        return false;
    }
    if (capture_count == 0) {
        error = "no capture devices found";
        return false;
    }

    if (!desired_name.empty()) {
        for (ma_uint32 i = 0; i < capture_count; ++i) {
            if (desired_name == capture_infos[i].name) {
                live->selected_device_id = capture_infos[i].id;
                live->has_selected_device = true;
                return true;
            }
        }
        error = "capture device not found: '" + desired_name + "'";
        return false;
    }

    if (live->params.capture_device_index >= 0) {
        if (static_cast<ma_uint32>(live->params.capture_device_index) >= capture_count) {
            std::ostringstream out;
            out << "capture device index out of range: " << live->params.capture_device_index;
            error = out.str();
            return false;
        }
        live->selected_device_id = capture_infos[live->params.capture_device_index].id;
        live->has_selected_device = true;
        return true;
    }

    for (ma_uint32 i = 0; i < capture_count; ++i) {
        if (capture_infos[i].isDefault) {
            live->selected_device_id = capture_infos[i].id;
            live->has_selected_device = true;
            return true;
        }
    }

    live->selected_device_id = capture_infos[0].id;
    live->has_selected_device = true;
    return true;
}

static bool initialize_bridge_session(llama_server_audio_live * live, std::string & error) {
    if (live == nullptr) {
        error = "capture live session is null";
        return false;
    }

    auto session_params = live->params.session_params;
    if (session_params.expected_input_sample_rate_hz == 0) {
        session_params.expected_input_sample_rate_hz = k_bridge_sample_rate_hz;
    }
    if (session_params.expected_input_sample_rate_hz != k_bridge_sample_rate_hz) {
        error = "audio capture bridge session must run at 16 kHz";
        return false;
    }
    if (session_params.expected_input_channels == 0) {
        session_params.expected_input_channels = k_bridge_channels;
    }
    if (session_params.expected_input_channels != k_bridge_channels) {
        error = "audio capture bridge session expects mono input";
        return false;
    }
    if (live->params.event_queue_capacity > 0 && session_params.event_queue_capacity == 0) {
        session_params.event_queue_capacity = live->params.event_queue_capacity;
    }

    live->session = llama_server_bridge_audio_session_create(&session_params);
    if (live->session == nullptr) {
        error = "llama_server_bridge_audio_session_create() failed";
        return false;
    }
    live->bridge_session_started = true;

    if (live->params.enable_diarization) {
        auto diarization_params = live->params.diarization_params;
        if (diarization_params.expected_sample_rate_hz == 0) {
            diarization_params.expected_sample_rate_hz = k_bridge_sample_rate_hz;
        }
        if (diarization_params.expected_sample_rate_hz != k_bridge_sample_rate_hz) {
            error = "diarization backend expects 16 kHz input";
            return false;
        }
        if (llama_server_bridge_audio_session_start_diarization(live->session, &diarization_params) != 0) {
            error = bridge_session_last_error(live);
            return false;
        }
        live->diarization_started = true;
        live->diarization_done = false;
    }

    if (live->params.enable_transcription) {
        auto transcription_params = live->params.transcription_params;
        transcription_params.mode = LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE;
        if (transcription_params.realtime_params.expected_sample_rate_hz == 0) {
            transcription_params.realtime_params.expected_sample_rate_hz = k_bridge_sample_rate_hz;
        }
        if (transcription_params.realtime_params.expected_sample_rate_hz != k_bridge_sample_rate_hz) {
            error = "transcription backend expects 16 kHz input";
            return false;
        }
        if (llama_server_bridge_audio_session_start_transcription(live->session, &transcription_params) != 0) {
            error = bridge_session_last_error(live);
            return false;
        }
        live->transcription_started = true;
        live->transcription_done = false;
    } else {
        live->transcription_done = true;
    }

    return true;
}

static bool initialize_output_paths(llama_server_audio_live * live, std::string & error) {
    if (live == nullptr) {
        error = "capture live session is null";
        return false;
    }
    if (live->params.output_dir == nullptr || trim_copy(live->params.output_dir).empty()) {
        error = "output_dir is required";
        return false;
    }

    live->output_dir = live->params.output_dir;
    if (!ensure_directory(live->output_dir, error)) {
        return false;
    }

    live->session_name = sanitize_name(
        live->params.session_name != nullptr && !trim_copy(live->params.session_name).empty()
            ? trim_copy(live->params.session_name)
            : make_timestamped_session_name());

    const fs::path base = fs::path(live->output_dir) / live->session_name;
    live->cleaned_wav_path = base.string() + ".clean.wav";
    live->transcript_path = live->params.enable_diarization
        ? base.string() + ".transcript.md"
        : base.string() + ".transcript.txt";
    live->preview_path = live->params.enable_diarization
        ? base.string() + ".preview.md"
        : std::string();
    return true;
}

static bool initialize_audio_pipeline(llama_server_audio_live * live, std::string & error) {
    if (live == nullptr) {
        error = "capture live session is null";
        return false;
    }

    if (!live->resampler.init(error)) {
        return false;
    }

    live->webrtc_runtime_enabled = live->params.enable_webrtc != 0;
#if defined(LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
    if (live->webrtc_runtime_enabled && !live->apm.init(error)) {
        return false;
    }
#else
    if (live->webrtc_runtime_enabled) {
        push_notice_event(live, "webrtc-disabled", "WebRTC APM support was not compiled into this audio runtime");
        live->webrtc_runtime_enabled = false;
    }
#endif

    if (live->params.write_clean_wav) {
        if (!live->cleaned_wav.open(live->cleaned_wav_path, k_bridge_sample_rate_hz, k_bridge_channels, error)) {
            return false;
        }
    }

    if (!resolve_capture_device(live, error)) {
        return false;
    }

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_s16;
    config.capture.channels = k_capture_channels;
    if (live->has_selected_device) {
        config.capture.pDeviceID = &live->selected_device_id;
    }
    config.sampleRate = k_capture_sample_rate_hz;
    config.periodSizeInFrames = static_cast<ma_uint32>(k_apm_frame_samples);
    config.periods = 4;
    config.performanceProfile = ma_performance_profile_low_latency;
    config.dataCallback = audio_capture_data_callback;
    config.notificationCallback = audio_capture_notification_callback;
    config.pUserData = live;

    if (ma_device_init(&live->context, &config, &live->device) != MA_SUCCESS) {
        error = "ma_device_init() failed";
        return false;
    }
    live->device_initialized = true;
    return true;
}

static bool finish_bridge_session(llama_server_audio_live * live) {
    if (live == nullptr || live->session == nullptr) {
        return true;
    }

    if (!flush_remaining_capture_tail(live)) {
        return false;
    }
    if (!flush_bridge_tail(live)) {
        return false;
    }

    if (llama_server_bridge_audio_session_flush_audio(live->session) != 0) {
        push_error_event(live, bridge_session_last_error(live));
        return false;
    }
    if (!pump_bridge_events_nonblocking(live)) {
        return false;
    }

    if (live->diarization_started) {
        if (llama_server_bridge_audio_session_stop_diarization(live->session) != 0) {
            push_error_event(live, bridge_session_last_error(live));
            return false;
        }
        live->diarization_started = false;
    }

    while (!(live->transcription_done && live->diarization_done)) {
        const int32_t pending = llama_server_bridge_audio_session_wait_events(live->session, 100);
        if (pending < 0) {
            push_error_event(live, bridge_session_last_error(live));
            return false;
        }
        if (!pump_bridge_events_nonblocking(live)) {
            return false;
        }
    }

    if (!live->params.enable_diarization) {
        update_transcript_file(live, join_piece_texts(live->raw_pieces));
    } else if (!live->last_final_markdown.empty()) {
        update_preview_file(live, live->last_final_markdown);
    } else if (!live->raw_pieces.empty()) {
        const std::string fallback_markdown = render_unassigned_markdown(live->raw_pieces);
        std::string error;
        if (!live->transcript_path.empty() && !write_text_file(live->transcript_path, fallback_markdown, error)) {
            push_error_event(live, error);
            return false;
        }
        if (!update_preview_file(live, fallback_markdown)) {
            return false;
        }
    }
    return true;
}

static void worker_main(llama_server_audio_live * live) {
    if (live == nullptr) {
        return;
    }

    while (true) {
        std::vector<int16_t> chunk;
        {
            std::unique_lock<std::mutex> lock(live->mutex);
            live->cv.wait(lock, [&] {
                return live->stop_requested || !live->capture_chunks.empty();
            });
            if (!live->capture_chunks.empty()) {
                chunk = std::move(live->capture_chunks.front());
                live->capture_chunks.pop_front();
            } else if (live->stop_requested) {
                break;
            }
        }

        if (!chunk.empty() && !process_capture_chunk(live, chunk)) {
            break;
        }
        if (!pump_bridge_events_nonblocking(live)) {
            break;
        }
    }

    finish_bridge_session(live);
    live->cleaned_wav.close();

    {
        std::lock_guard<std::mutex> lock(live->mutex);
        live->worker_finished = true;
        live->cv.notify_all();
    }
}

static bool start_live_capture(llama_server_audio_live * live, std::string & error) {
    if (live == nullptr) {
        error = "capture live session is null";
        return false;
    }

    if (!live->params.enable_transcription) {
        error = "audio live capture currently requires enable_transcription=1";
        return false;
    }
    if (live->params.bridge_push_samples == 0) {
        live->params.bridge_push_samples = k_default_bridge_push_samples;
    }
    if (!initialize_output_paths(live, error)) {
        return false;
    }
    if (!initialize_output_files(live)) {
        error = live->last_error.empty() ? "failed to initialize output files" : live->last_error;
        return false;
    }
    if (!initialize_bridge_session(live, error)) {
        return false;
    }
    if (!initialize_audio_pipeline(live, error)) {
        return false;
    }

    live->worker = std::thread(worker_main, live);
    if (ma_device_start(&live->device) != MA_SUCCESS) {
        error = "ma_device_start() failed";
        return false;
    }
    live->device_started = true;
    live->started.store(true);

    push_notice_event(live, "output-dir", live->output_dir);
    if (live->params.write_clean_wav) {
        push_notice_event(live, "cleaned-wav", live->cleaned_wav_path);
    }
    if (live->params.enable_diarization) {
        push_notice_event(live, "preview-transcript", live->preview_path);
        push_notice_event(live, "final-transcript", live->transcript_path);
    } else {
        push_notice_event(live, "transcript", live->transcript_path);
    }
    return true;
}

static void stop_live_capture(llama_server_audio_live * live) {
    if (live == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(live->mutex);
        live->stop_requested = true;
    }
    live->cv.notify_all();

    if (live->device_started) {
        ma_device_stop(&live->device);
        live->device_started = false;
    }
    if (live->worker.joinable()) {
        live->worker.join();
    }

    cleanup_device(live);
    cleanup_session(live);
    live->cleaned_wav.close();
    live->started.store(false);
}

} // namespace

extern "C" {

struct llama_server_audio_live_params llama_server_audio_default_live_params(void) {
    struct llama_server_audio_live_params out = {};
    out.capture_device_index = -1;
    out.bridge_push_samples = k_default_bridge_push_samples;
    out.enable_webrtc = 1;
    out.enable_transcription = 1;
    out.enable_diarization = 0;
    out.write_clean_wav = 1;
    out.write_preview_file = 1;
    out.event_queue_capacity = 0;
    out.session_params = llama_server_bridge_default_audio_session_params();
    out.session_params.expected_input_sample_rate_hz = k_bridge_sample_rate_hz;
    out.session_params.expected_input_channels = k_bridge_channels;
    out.transcription_params = llama_server_bridge_default_audio_transcription_params();
    out.transcription_params.mode = LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE;
    out.transcription_params.realtime_params.expected_sample_rate_hz = k_bridge_sample_rate_hz;
    out.diarization_params = llama_server_bridge_default_realtime_params();
    out.diarization_params.expected_sample_rate_hz = k_bridge_sample_rate_hz;
    return out;
}

struct llama_server_audio_output_paths llama_server_audio_empty_output_paths(void) {
    struct llama_server_audio_output_paths out = {};
    return out;
}

int32_t llama_server_audio_list_capture_devices(
    struct llama_server_audio_capture_device_info ** out_devices,
    size_t * out_count) {
    if (out_devices == nullptr || out_count == nullptr) {
        return -1;
    }
    *out_devices = nullptr;
    *out_count = 0;

    ma_context context = {};
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) {
        return -1;
    }

    ma_device_info * playback_infos = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info * capture_infos = nullptr;
    ma_uint32 capture_count = 0;
    const ma_result result = ma_context_get_devices(
        &context,
        &playback_infos,
        &playback_count,
        &capture_infos,
        &capture_count);
    if (result != MA_SUCCESS) {
        ma_context_uninit(&context);
        return -1;
    }

    auto * devices = static_cast<llama_server_audio_capture_device_info *>(
        std::calloc(capture_count, sizeof(llama_server_audio_capture_device_info)));
    if (devices == nullptr && capture_count > 0) {
        ma_context_uninit(&context);
        return -1;
    }

    for (ma_uint32 i = 0; i < capture_count; ++i) {
        devices[i].index = static_cast<int32_t>(i);
        devices[i].is_default = capture_infos[i].isDefault ? 1 : 0;
        devices[i].name = dup_cstr(capture_infos[i].name);
        if (devices[i].name == nullptr) {
            for (ma_uint32 j = 0; j <= i; ++j) {
                free_cstr(devices[j].name);
            }
            std::free(devices);
            ma_context_uninit(&context);
            return -1;
        }
    }

    ma_context_uninit(&context);
    *out_devices = devices;
    *out_count = static_cast<size_t>(capture_count);
    return 0;
}

void llama_server_audio_free_capture_devices(
    struct llama_server_audio_capture_device_info * devices,
    size_t count) {
    if (devices == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free_cstr(devices[i].name);
        devices[i].name = nullptr;
    }
    std::free(devices);
}

struct llama_server_audio_live * llama_server_audio_live_create(
    const struct llama_server_audio_live_params * params) {
    try {
        const auto values = params != nullptr ? *params : llama_server_audio_default_live_params();
        return new llama_server_audio_live(values);
    } catch (...) {
        return nullptr;
    }
}

void llama_server_audio_live_destroy(struct llama_server_audio_live * live) {
    if (live == nullptr) {
        return;
    }
    stop_live_capture(live);
    delete live;
}

int32_t llama_server_audio_live_start(struct llama_server_audio_live * live) {
    if (live == nullptr) {
        return -1;
    }
    if (live->started.load()) {
        return 0;
    }

    std::string error;
    if (!start_live_capture(live, error)) {
        if (!error.empty()) {
            set_last_error(live, error);
            push_error_event(live, error);
        }
        stop_live_capture(live);
        return -1;
    }
    return 0;
}

int32_t llama_server_audio_live_stop(struct llama_server_audio_live * live) {
    if (live == nullptr) {
        return -1;
    }
    stop_live_capture(live);
    return 0;
}

int32_t llama_server_audio_live_wait_events(
    struct llama_server_audio_live * live,
    uint32_t timeout_ms) {
    if (live == nullptr) {
        return -1;
    }

    std::unique_lock<std::mutex> lock(live->mutex);
    if (live->queued_events.empty() && timeout_ms > 0) {
        live->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return !live->queued_events.empty() || !live->last_error.empty() || live->worker_finished;
        });
    }
    return static_cast<int32_t>(live->queued_events.size());
}

int32_t llama_server_audio_live_drain_events(
    struct llama_server_audio_live * live,
    struct llama_server_bridge_audio_event ** out_events,
    size_t * out_count,
    size_t max_events) {
    if (live == nullptr || out_events == nullptr || out_count == nullptr) {
        return -1;
    }
    *out_events = nullptr;
    *out_count = 0;

    std::deque<queued_event_record> drained;
    {
        std::lock_guard<std::mutex> lock(live->mutex);
        const size_t take = (max_events == 0 || max_events > live->queued_events.size())
            ? live->queued_events.size()
            : max_events;
        for (size_t i = 0; i < take; ++i) {
            drained.push_back(std::move(live->queued_events.front()));
            live->queued_events.pop_front();
        }
    }

    if (drained.empty()) {
        return 0;
    }

    auto * events = static_cast<llama_server_bridge_audio_event *>(
        std::calloc(drained.size(), sizeof(llama_server_bridge_audio_event)));
    if (events == nullptr) {
        return -1;
    }

    size_t produced = 0;
    for (auto & src : drained) {
        auto & dst = events[produced];
        dst.seq_no = src.seq_no;
        dst.kind = src.kind;
        dst.flags = src.flags;
        dst.start_sample = src.start_sample;
        dst.end_sample = src.end_sample;
        dst.speaker_id = src.speaker_id;
        dst.item_id = src.item_id;
        dst.text = dup_cstr(src.text);
        dst.detail = dup_cstr(src.detail);
        if ((dst.text == nullptr && !src.text.empty()) || (dst.detail == nullptr && !src.detail.empty())) {
            for (size_t i = 0; i <= produced; ++i) {
                free_audio_event_fields(events[i]);
            }
            std::free(events);
            return -1;
        }
        ++produced;
    }

    *out_events = events;
    *out_count = produced;
    return 0;
}

void llama_server_audio_live_free_events(
    struct llama_server_bridge_audio_event * events,
    size_t count) {
    if (events == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free_audio_event_fields(events[i]);
    }
    std::free(events);
}

int32_t llama_server_audio_live_get_output_paths(
    const struct llama_server_audio_live * live,
    struct llama_server_audio_output_paths * out_paths) {
    if (live == nullptr || out_paths == nullptr) {
        return -1;
    }
    *out_paths = llama_server_audio_empty_output_paths();
    out_paths->output_dir = dup_cstr(live->output_dir);
    out_paths->cleaned_wav_path = dup_cstr(live->cleaned_wav_path);
    out_paths->transcript_path = dup_cstr(live->transcript_path);
    out_paths->preview_path = dup_cstr(live->preview_path);
    if ((out_paths->output_dir == nullptr && !live->output_dir.empty())
        || (out_paths->cleaned_wav_path == nullptr && !live->cleaned_wav_path.empty())
        || (out_paths->transcript_path == nullptr && !live->transcript_path.empty())
        || (out_paths->preview_path == nullptr && !live->preview_path.empty())) {
        free_output_paths_fields(*out_paths);
        return -1;
    }
    return 0;
}

void llama_server_audio_output_paths_free(struct llama_server_audio_output_paths * paths) {
    if (paths == nullptr) {
        return;
    }
    free_output_paths_fields(*paths);
}

const char * llama_server_audio_live_last_error(const struct llama_server_audio_live * live) {
    if (live == nullptr) {
        return nullptr;
    }
    return live->last_error.empty() ? nullptr : live->last_error.c_str();
}

} // extern "C"
