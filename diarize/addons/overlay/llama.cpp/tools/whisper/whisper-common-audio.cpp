#define _USE_MATH_DEFINES

#include "common-whisper.h"

#include "whisper.h"

#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#include "miniaudio/miniaudio.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <algorithm>
#include <cstdio>
#include <fstream>

bool read_audio_data(
        const std::string & fname,
        std::vector<float> & pcmf32,
        std::vector<std::vector<float>> & pcmf32s,
        bool stereo) {
    std::vector<uint8_t> audio_data;

    ma_result result;
    ma_decoder_config decoder_config;
    ma_decoder decoder;

    decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE);

    if (fname == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        uint8_t buf[1024];
        while (true) {
            const size_t n = fread(buf, 1, sizeof(buf), stdin);
            if (n == 0) {
                break;
            }
            audio_data.insert(audio_data.end(), buf, buf + n);
        }

        result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder);
        if (result != MA_SUCCESS) {
            fprintf(stderr, "Error: failed to open audio data from stdin (%s)\n", ma_result_description(result));
            return false;
        }
    } else {
        result = ma_decoder_init_file(fname.c_str(), &decoder_config, &decoder);
        if (result != MA_SUCCESS) {
            fprintf(stderr, "error: failed to read audio data '%s' (%s)\n", fname.c_str(), ma_result_description(result));
            return false;
        }
    }

    ma_uint64 frame_count = 0;
    ma_uint64 frames_read = 0;

    if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
        fprintf(stderr, "error: failed to retrieve audio length (%s)\n", ma_result_description(result));
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32.resize(stereo ? frame_count * 2 : frame_count);
    if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
        fprintf(stderr, "error: failed to read audio frames (%s)\n", ma_result_description(result));
        ma_decoder_uninit(&decoder);
        return false;
    }
    ma_decoder_uninit(&decoder);

    if (stereo) {
        std::vector<float> stereo_data = pcmf32;
        pcmf32.resize(frame_count);

        for (uint64_t i = 0; i < frame_count; ++i) {
            pcmf32[i] = (stereo_data[2 * i] + stereo_data[2 * i + 1]);
        }

        pcmf32s.resize(2);
        pcmf32s[0].resize(frame_count);
        pcmf32s[1].resize(frame_count);
        for (uint64_t i = 0; i < frame_count; ++i) {
            pcmf32s[0][i] = stereo_data[2 * i];
            pcmf32s[1][i] = stereo_data[2 * i + 1];
        }
    }

    return true;
}

std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);
    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t * whisper_sample_rate) / 100)));
}

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id) {
    std::ofstream speak_file(path.c_str());
    if (speak_file.fail()) {
        fprintf(stderr, "%s: failed to open speak_file\n", __func__);
        return false;
    }

    speak_file.write(text.c_str(), text.size());
    speak_file.close();

    const int ret = system((command + " " + std::to_string(voice_id) + " " + path).c_str());
    if (ret != 0) {
        fprintf(stderr, "%s: failed to speak\n", __func__);
        return false;
    }

    return true;
}
