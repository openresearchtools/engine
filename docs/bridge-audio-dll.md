# Bridge Audio DLL (`llama-server-bridge.dll`)

This document covers transcription and diarization calls through:
- `llama_server_bridge_audio_transcriptions` (JSON body)
- `llama_server_bridge_audio_transcriptions_raw` (raw bytes + metadata)
- `llama_server_bridge_audio_session_*` (shared PCM session with separate task control)

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

## Audio session API

The session API is the new bridge-level control surface for Rust orchestration:
- one shared audio ingress timeline
- separate `start_diarization(...)` and `start_transcription(...)`
- staged or simultaneous execution on the same buffered audio
- one unified event queue back to Rust

Current exported session calls:
- `llama_server_bridge_audio_session_create`
- `llama_server_bridge_audio_session_push_audio`
- `llama_server_bridge_audio_session_push_encoded`
- `llama_server_bridge_audio_session_flush_audio`
- `llama_server_bridge_audio_session_start_diarization`
- `llama_server_bridge_audio_session_stop_diarization`
- `llama_server_bridge_audio_session_start_transcription`
- `llama_server_bridge_audio_session_stop_transcription`
- `llama_server_bridge_audio_session_wait_events`
- `llama_server_bridge_audio_session_drain_events`
- `llama_server_bridge_audio_session_last_error`

### Session model

Control plane:
- diarization can run alone
- transcription can run alone
- both can run together
- either task can be started after audio has already been buffered

Data plane:
- audio is pushed once into the session
- the session keeps one shared PCM timeline
- late-start diarization replays the original buffered chunk boundaries into the existing realtime backend
- offline/final transcription runs from the buffered PCM after `flush_audio()`

### PCM ingress

`llama_server_bridge_audio_session_push_audio(...)` currently accepts:
- mono PCM only
- session sample rate only
- `F32` or `S16`

Default session format:
- `16000 Hz`
- `1 channel`

If you have encoded file bytes:
- use `llama_server_bridge_audio_session_push_encoded(...)`
- the bridge reuses shared FFmpeg conversion and pushes the normalized stream into the same session timeline

### Current event behavior

Diarization emits bridge-normalized commit events:
- diarization started/stopped
- speaker span commit
- backend status/error
- future realtime transcript commit support is reserved in the ABI

Transcription currently emits:
- transcription started
- `TRANSCRIPTION_WORD_COMMIT`
- `TRANSCRIPTION_PIECE_COMMIT`
- one final `TRANSCRIPTION_RESULT_JSON` event
- transcription stopped

The session transcription path forces the native audio route into `mode=timeline`, which returns:
- `transcript`
- `words`
- `whisper_pieces`
- `segments`

The bridge translates `words` and `whisper_pieces` into typed session events and still emits the final native JSON payload for debugging/replay. This keeps the current model logic intact while Rust takes ownership of downstream alignment/orchestration policy.

### Rust orchestration path

The `engine` crate now exposes a bridge-level runner for this ABI:

```powershell
engine bridge audio-session `
  --audio-file C:\path\clip.wav `
  --diarization-model-path C:\models\sortformer.gguf `
  --whisper-model C:\models\whisper.bin `
  --out C:\path\diarized_transcript.md `
  --timeline-json-out C:\path\timeline.json
```

Live raw PCM stdin with native realtime transcription and diarization:

```cmd
engine bridge audio-session ^
  --stdin-pcm-s16le ^
  --session-sample-rate 16000 ^
  --diarization-model-path C:\models\sortformer.gguf ^
  --transcription-realtime-model C:\models\voxtral.gguf ^
  --out C:\path\live_transcript.md ^
  < C:\path\clip.raw
```

`audio-session` now supports:
- file input with `--audio-file`
- live raw PCM stdin with `--stdin-pcm-s16le` or `--stdin-pcm-f32le`
- diarization only, transcription only, or combined
- staged execution with `--staged`
- offline Whisper route or native realtime transcription on the same Rust-side session path
- rolling committed updates:
  - when `--out` is set, the transcript file is rewritten during the run as committed text/turns arrive
  - when `--timeline-json-out` is set, the latest transcription JSON is rewritten during the run
  - for live stdin runs without `--out`, committed transcript updates are written to stdout incrementally

Useful flags:
- `--staged` to run diarization first and transcription second on the same buffered audio
- `--alignment-offset-ms` to apply a bounded global transcript-vs-diarization offset in Rust
- `--nearest-tolerance-ms` to control small nearest-span assignment fallback

The Rust assembler uses:
- speaker spans as the ownership truth source
- Whisper pieces as the primary transcript unit
- adjacency-only reassignment for `UNASSIGNED` text
- word timestamps only for local sentence-boundary cleanup

## Metadata keys supported by runtime

Core mode/source keys:
- `mode`: `subtitle`, `speech`, `transcript`, or `timeline`
- `custom`: mode-specific (`default`, `auto`, or numeric depending on mode)
- `whisper_model` or `whisper_model_path`
- `whisper_hf_repo` + `whisper_hf_file`
- `diarization_model_path`
- `diarization_models_dir` or `diarization_model_dir`

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
- `diarization_feed_ms`
- `speaker_seg_max_gap_sec`
- `speaker_seg_max_words`
- `speaker_seg_max_duration_sec`
- `speaker_seg_split_on_hard_break`

## Full audio example (advanced metadata)

```c
const char *metadata =
    "{"
    "\"mode\":\"transcript\","
    "\"custom\":\"auto\","
    "\"whisper_model\":\"C:/models/whisper.bin\","
    "\"diarization_model_path\":\"C:/models/diarization/sortformer.gguf\","
    "\"gpu\":1,"
    "\"whisper_threads\":8,"
    "\"whisper_flash_attn\":true,"
    "\"diarization_backend\":\"sortformer\","
    "\"diarization_feed_ms\":470,"
    "\"speaker_seg_split_on_hard_break\":true"
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
