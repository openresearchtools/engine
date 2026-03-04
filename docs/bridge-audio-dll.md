# Bridge Audio DLL (`llama-server-bridge.dll`)

This document covers transcription and diarization calls through:
- `llama_server_bridge_audio_transcriptions` (JSON body)
- `llama_server_bridge_audio_transcriptions_raw` (raw bytes + metadata)

Reference header: `bridge/llama_server_bridge.h`.

## Simplest call

Use raw audio API for embedding apps (no base64 at your app boundary):

```c
#include "llama_server_bridge.h"

llama_server_bridge_params p = llama_server_bridge_default_params();
// model_path can be omitted for audio-only runtime
p.gpu = 1;

llama_server_bridge *bridge = llama_server_bridge_create(&p);

const char *metadata =
    "{"
    "\"mode\":\"transcript\","
    "\"custom\":\"auto\","
    "\"whisper_model\":\"C:/models/whisper.bin\","
    "\"diarization_models_dir\":\"C:/models/diarization\""
    "}";

llama_server_bridge_audio_raw_request req = llama_server_bridge_default_audio_raw_request();
req.audio_bytes = wav_or_mp3_bytes;
req.audio_bytes_len = wav_or_mp3_len;
req.audio_format = "mp3";
req.metadata_json = metadata;
req.ffmpeg_convert = 1;

llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
int rc = llama_server_bridge_audio_transcriptions_raw(bridge, &req, &out);

if (rc == 0 && out.ok == 1) {
    // out.json
}

llama_server_bridge_json_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## Device selection (repeat for this function)

Audio supports the same device selection model as other bridge endpoints.

- Set `params.gpu = <index>` for global runtime device selection.
- Do not set both `gpu` and `devices`.
- If no selector is set:
- Windows/Linux default to CPU-only.
- macOS defaults to first GPU.

Audio-specific behavior:
- If metadata does not explicitly set `whisper_gpu_device`, runtime injects selected GPU automatically when possible.
- If metadata does not explicitly set `diarization_device`, runtime injects selected bridge device name (for example `Vulkan1`) when possible.
- Metadata can also override with `gpu` or `device` fields.

Performance guidance:
- Keep whisper + diarization on one selected GPU when possible.
- Split/multi-device is mainly for fit constraints.

## Supported create parameters (all)

All bridge create fields are supported for audio:
- `model_path`, `mmproj_path`
- `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`
- `n_threads`, `n_threads_batch`
- `n_gpu_layers`, `main_gpu`, `gpu`
- `no_kv_offload`, `mmproj_use_gpu`, `cache_ram_mib`
- `seed`, `ctx_shift`, `kv_unified`
- `devices`, `tensor_split`, `split_mode`
- `embedding`, `reranking`, `pooling_type`

For audio-only usage, `model_path` may be omitted.

## Audio request APIs and parameters

## 1) JSON-body API

`llama_server_bridge_audio_request`:
- `body_json` (required)

Call:
- `llama_server_bridge_audio_transcriptions(bridge, &req, &out)`

Body must be a JSON object accepted by `/v1/audio/transcriptions` route.

## 2) Raw-bytes API

`llama_server_bridge_audio_raw_request`:
- `audio_bytes` (required)
- `audio_bytes_len` (required)
- `audio_format` (required, for example `wav`, `mp3`, `flac`)
- `metadata_json` (JSON object without `audio` field)
- `ffmpeg_convert` (`1` convert to WAV 16-bit mono 16k in RAM, `0` no convert)

Call:
- `llama_server_bridge_audio_transcriptions_raw(bridge, &req, &out)`

## Metadata keys supported by runtime

Core mode/source keys:
- `mode`: `subtitle`, `speech`, or `transcript`
- `custom`: mode-specific (`default`, `auto`, or numeric depending on mode)
- `whisper_model` or `whisper_model_path`
- `whisper_hf_repo` + `whisper_hf_file`
- `diarization_models_dir` or `diarization_model_dir`
- `diarization_hf_repo`

Device keys:
- `gpu` (index)
- `device` (string like `gpu:1` or `gpu=1`)
- `whisper_gpu_device`
- `whisper_no_gpu`
- `diarization_device`

Advanced whisper keys:
- `whisper_threads`
- `whisper_processors`
- `whisper_max_len`
- `whisper_audio_ctx`
- `whisper_best_of`
- `whisper_beam_size`
- `whisper_temperature`
- `whisper_language`
- `whisper_prompt`
- `whisper_translate`
- `whisper_no_fallback`
- `whisper_suppress_nst`
- `whisper_flash_attn`
- `whisper_offline`
- `whisper_word_time_offset_sec`
- `seconds_per_timeline_token`
- `source_audio_seconds`

Advanced diarization/alignment keys:
- `diarization_backend`
- `diarization_offline`
- `diarization_embedding_min_segment_duration_sec`
- `diarization_embedding_max_segments_per_speaker`
- `diarization_min_duration_off_sec`
- `speaker_seg_max_gap_sec`
- `speaker_seg_max_words`
- `speaker_seg_max_duration_sec`
- `speaker_seg_split_on_hard_break`
- `aligner_plda_sim_threshold`

## Full audio example (advanced metadata)

```c
const char *metadata =
    "{"
    "\"mode\":\"transcript\","
    "\"custom\":\"auto\","
    "\"whisper_model\":\"C:/models/whisper.bin\","
    "\"diarization_models_dir\":\"C:/models/diarization\","
    "\"gpu\":1,"
    "\"whisper_threads\":8,"
    "\"whisper_flash_attn\":true,"
    "\"diarization_backend\":\"native_cpp\","
    "\"diarization_embedding_min_segment_duration_sec\":0.8"
    "}";

llama_server_bridge_audio_raw_request req = llama_server_bridge_default_audio_raw_request();
req.audio_bytes = audio_bytes;
req.audio_bytes_len = audio_len;
req.audio_format = "mp3";
req.metadata_json = metadata;
req.ffmpeg_convert = 1;

llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
int rc = llama_server_bridge_audio_transcriptions_raw(bridge, &req, &out);
if (rc != 0 || out.ok == 0) {
    const char *last = llama_server_bridge_last_error(bridge);
    // inspect out.error_json and last
}
```
