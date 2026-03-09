# Bridge Audio DLL (`llama-server-bridge.dll`)

This document describes the current audio embedding surface exposed by [`llama-server-bridge.dll`](../bridge/llama_server_bridge.h).

It covers two different styles of use:

1. One-shot audio requests:
   - `llama_server_bridge_audio_transcriptions`
   - `llama_server_bridge_audio_transcriptions_raw`
2. Shared audio sessions:
   - `llama_server_bridge_audio_session_*`

For current ENGINE work, the session API is the preferred embedding path.

## Current audio architecture

Current native audio behavior is:

- diarization: native Sortformer GGUF only
- offline/file transcription: Whisper through the audio transcription route
- native realtime transcription: Voxtral through the realtime backend path
- orchestration: one shared session timeline with event-driven speaker/text assembly

Important consequences:

- diarization inputs must be native Sortformer GGUF files
- the session API supports staged or simultaneous diarization/transcription on the same buffered audio timeline

## Which API to use

Use the one-shot APIs when:

- you already have a whole audio file in memory
- you want one JSON result back
- you do not need rolling session events

Use the session API when:

- you want live PCM ingress
- you want diarization and transcription on one continuous session timeline
- you want staged processing on buffered audio
- you want rolling transcript/speaker events
- you want native realtime transcription backends such as Voxtral

## Core create parameters

All audio entrypoints use [`llama_server_bridge_params`](../bridge/llama_server_bridge.h) for bridge creation.

Important fields:

- `gpu`
- `devices`
- `tensor_split`
- `split_mode`
- `n_gpu_layers`
- `main_gpu`
- `n_threads`
- `n_threads_batch`
- `mmproj_use_gpu`

For audio-only use, `model_path` may be omitted.

## Device selection rules

Bridge-wide device selection:

- set `params.gpu = <index>` for a simple single-device selection
- or use `params.devices` / `params.tensor_split` / `params.split_mode`
- do not set both `gpu` and `devices`

Default behavior:

- Windows/Linux: CPU-only if no device selector is provided
- macOS: first GPU if no device selector is provided

Audio-specific behavior:

- Whisper/offline route follows the selected bridge device unless overridden by metadata
- native diarization uses the backend runtime name you pass in `realtime_params.backend_name`
- native realtime transcription uses the backend runtime name you pass in `realtime_params.backend_name`

Examples of backend runtime names:

- `CPU`
- `Vulkan0`
- `Vulkan1`
- `CUDA0`

## One-shot APIs

### 1) JSON body API

Function:

```c
int32_t llama_server_bridge_audio_transcriptions(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_audio_request * req,
    struct llama_server_bridge_json_result * out);
```

Request:

- `body_json` is required

Use this when your app already has the full request JSON body.

### 2) Raw bytes API

Function:

```c
int32_t llama_server_bridge_audio_transcriptions_raw(
    struct llama_server_bridge * bridge,
    const struct llama_server_bridge_audio_raw_request * req,
    struct llama_server_bridge_json_result * out);
```

Request fields:

- `audio_bytes`
- `audio_bytes_len`
- `audio_format`
- `metadata_json`
- `ffmpeg_convert`

`ffmpeg_convert = 1` means the bridge converts the input to WAV 16-bit mono 16 kHz in RAM before routing it.

### One-shot metadata keys

`metadata_json` is the same body shape accepted by the current audio transcription route, minus the `audio` field.

Common keys:

- `mode`: `speech`, `subtitle`, `transcript`, or `timeline`
- `custom`
- `whisper_model` or `whisper_model_path`
- `whisper_hf_repo` + `whisper_hf_file`
- `diarization_model_path`
- `diarization_models_dir`
- `diarization_device`
- `diarization_feed_ms`
- `whisper_gpu_device`
- `whisper_word_time_offset_sec`
- `seconds_per_timeline_token`
- `source_audio_seconds`

Current diarization rule:

- prefer `diarization_model_path` pointing to a Sortformer GGUF
- `diarization_models_dir` is accepted only as a directory of native Sortformer GGUF files

### Minimal one-shot diarized example

```c
#include "llama_server_bridge.h"

llama_server_bridge_params p = llama_server_bridge_default_params();
p.gpu = 0;

llama_server_bridge * bridge = llama_server_bridge_create(&p);

const char * metadata =
    "{"
    "\"mode\":\"transcript\","
    "\"custom\":\"auto\","
    "\"whisper_model\":\"./models/whisper.bin\","
    "\"diarization_model_path\":\"./models/diarization/sortformer.gguf\""
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
    /* out.json contains the route result */
}

llama_server_bridge_json_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## Session API

The session API is the current bridge-level embedding surface for continuous audio.

It gives you:

- one shared audio ingress timeline
- optional diarization task
- optional transcription task
- staged or simultaneous execution
- one event queue back out

### Session lifecycle

Typical order:

1. create session
2. optionally start diarization
3. optionally start transcription
4. push audio as PCM or encoded bytes
5. flush audio when the stream/file ends
6. wait for and drain events until completion
7. stop tasks if needed
8. destroy session

### Session creation

Use [`llama_server_bridge_audio_session_params`](../bridge/llama_server_bridge.h):

- `expected_input_sample_rate_hz`
- `expected_input_channels`
- `max_buffered_audio_samples`
- `event_queue_capacity`

Function:

```c
struct llama_server_bridge_audio_session * llama_server_bridge_audio_session_create(
    const struct llama_server_bridge_audio_session_params * params);
```

### Audio ingress

Raw PCM ingress:

```c
int32_t llama_server_bridge_audio_session_push_audio(
    struct llama_server_bridge_audio_session * session,
    const void * audio_bytes,
    size_t frame_count,
    uint32_t sample_rate_hz,
    uint32_t channels,
    int32_t sample_format);
```

Supported sample formats:

- `LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_F32`
- `LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16`

Encoded/file ingress:

```c
int32_t llama_server_bridge_audio_session_push_encoded(
    struct llama_server_bridge_audio_session * session,
    const uint8_t * audio_bytes,
    size_t audio_bytes_len,
    const char * audio_format);
```

Flush:

```c
int32_t llama_server_bridge_audio_session_flush_audio(
    struct llama_server_bridge_audio_session * session);
```

### Starting diarization

Diarization is a native realtime backend started with [`llama_server_bridge_realtime_params`](../bridge/llama_server_bridge.h).

Current native diarization backend:

- `LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER`

Important realtime params:

- `backend_kind`
- `model_path`
- `backend_name`
- `expected_sample_rate_hz`
- `audio_ring_capacity_samples`

Start call:

```c
int32_t llama_server_bridge_audio_session_start_diarization(
    struct llama_server_bridge_audio_session * session,
    const struct llama_server_bridge_realtime_params * params);
```

Stop call:

```c
int32_t llama_server_bridge_audio_session_stop_diarization(
    struct llama_server_bridge_audio_session * session);
```

### Starting transcription

Transcription uses [`llama_server_bridge_audio_transcription_params`](../bridge/llama_server_bridge.h).

This supports two modes:

- `LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE`
  - Whisper / route-style transcription
  - configured through `metadata_json`
- `LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE`
  - native realtime backend
  - currently used for Voxtral
  - configured through `realtime_params`

Start call:

```c
int32_t llama_server_bridge_audio_session_start_transcription(
    struct llama_server_bridge_audio_session * session,
    const struct llama_server_bridge_audio_transcription_params * params);
```

Stop call:

```c
int32_t llama_server_bridge_audio_session_stop_transcription(
    struct llama_server_bridge_audio_session * session);
```

### Waiting for events

Wait:

```c
int32_t llama_server_bridge_audio_session_wait_events(
    struct llama_server_bridge_audio_session * session,
    uint32_t timeout_ms);
```

Drain:

```c
int32_t llama_server_bridge_audio_session_drain_events(
    struct llama_server_bridge_audio_session * session,
    struct llama_server_bridge_audio_event ** out_events,
    size_t * out_count,
    size_t max_events);
```

Free:

```c
void llama_server_bridge_audio_session_free_events(
    struct llama_server_bridge_audio_event * events,
    size_t count);
```

Error accessor:

```c
const char * llama_server_bridge_audio_session_last_error(
    const struct llama_server_bridge_audio_session * session);
```

## Session event model

Session events are returned as [`llama_server_bridge_audio_event`](../bridge/llama_server_bridge.h):

- `seq_no`
- `kind`
- `flags`
- `start_sample`
- `end_sample`
- `speaker_id`
- `item_id`
- `text`
- `detail`

Current event kinds:

- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_NOTICE`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STARTED`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_STOPPED`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_SPAN_COMMIT`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_STATUS`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_DIARIZATION_BACKEND_ERROR`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STARTED`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_TRANSCRIPTION_STOPPED`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_STREAM_FLUSHED`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_ERROR`

Current flags:

- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FINAL`
- `LLAMA_SERVER_BRIDGE_AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY`

`FROM_BUFFER_REPLAY` matters for staged flows: if you start a task after buffering audio, the bridge can replay buffered audio into that task and mark the resulting events.

## Backend discovery helpers

Use these helpers if you want your app to discover native realtime backends dynamically:

- `llama_server_bridge_realtime_backend_count`
- `llama_server_bridge_realtime_backend_kind_at`
- `llama_server_bridge_realtime_backend_get_info`
- `llama_server_bridge_realtime_backend_name`
- `llama_server_bridge_realtime_backend_kind_from_name`
- `llama_server_bridge_realtime_backend_kind_from_model_path`
- `llama_server_bridge_realtime_backend_supports_model_preload`
- `llama_server_bridge_realtime_backend_emits_transcript`
- `llama_server_bridge_realtime_backend_emits_speaker_spans`
- `llama_server_bridge_realtime_backend_default_runtime_backend_name`
- `llama_server_bridge_realtime_backend_default_sample_rate_hz`
- `llama_server_bridge_realtime_backend_default_audio_ring_capacity_samples`
- `llama_server_bridge_realtime_backend_required_input_channels`

Current backend kinds:

- `AUTO`
- `SORTFORMER`
- `VOXTRAL`

## Recommended embedding patterns

### A) File transcription only (Whisper)

- create session
- start transcription in `OFFLINE_ROUTE` mode with Whisper metadata
- push encoded file bytes with `push_encoded`
- flush
- collect `TRANSCRIPTION_RESULT_JSON` and/or commit events

### B) File transcription + diarization (Whisper + Sortformer, staged or simultaneous)

- create session
- start Sortformer diarization
- either:
  - start Whisper immediately for simultaneous processing, or
  - start Whisper later for staged processing over buffered audio
- push encoded file bytes
- flush
- assemble transcript from commit events

### C) Live PCM transcription + diarization (Voxtral + Sortformer)

- create session
- start Sortformer diarization
- start transcription in `REALTIME_NATIVE` mode with Voxtral realtime params
- push PCM chunks as they arrive
- flush on stream end
- consume commit events continuously

## Full session example: live PCM with native Voxtral + native Sortformer

```c
#include "llama_server_bridge.h"

struct llama_server_bridge_audio_session_params s =
    llama_server_bridge_default_audio_session_params();
s.expected_input_sample_rate_hz = 16000;
s.expected_input_channels = 1;

struct llama_server_bridge_audio_session * session =
    llama_server_bridge_audio_session_create(&s);

struct llama_server_bridge_realtime_params diar =
    llama_server_bridge_default_realtime_params_for_backend(
        LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER);
diar.backend_kind = LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_SORTFORMER;
diar.model_path = "./models/diarization/sortformer.gguf";
diar.backend_name = "Vulkan0";
diar.expected_sample_rate_hz = 16000;

struct llama_server_bridge_audio_transcription_params tx =
    llama_server_bridge_default_audio_transcription_params();
tx.mode = LLAMA_SERVER_BRIDGE_AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE;
tx.realtime_params =
    llama_server_bridge_default_realtime_params_for_backend(
        LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_VOXTRAL);
tx.realtime_params.backend_kind = LLAMA_SERVER_BRIDGE_REALTIME_BACKEND_VOXTRAL;
tx.realtime_params.model_path = "./models/voxtral-mini-4b-realtime.gguf";
tx.realtime_params.backend_name = "Vulkan0";
tx.realtime_params.expected_sample_rate_hz = 16000;

llama_server_bridge_audio_session_start_diarization(session, &diar);
llama_server_bridge_audio_session_start_transcription(session, &tx);

while (have_more_pcm()) {
    const int16_t * pcm = get_pcm_chunk();
    size_t frames = get_pcm_chunk_frames();
    llama_server_bridge_audio_session_push_audio(
        session,
        pcm,
        frames,
        16000,
        1,
        LLAMA_SERVER_BRIDGE_AUDIO_SAMPLE_FORMAT_S16);

    if (llama_server_bridge_audio_session_wait_events(session, 10) == 0) {
        struct llama_server_bridge_audio_event * events = NULL;
        size_t count = 0;
        if (llama_server_bridge_audio_session_drain_events(session, &events, &count, 128) == 0) {
            for (size_t i = 0; i < count; ++i) {
                const struct llama_server_bridge_audio_event * ev = &events[i];
                /* handle diarization spans and transcription commits here */
            }
            llama_server_bridge_audio_session_free_events(events, count);
        }
    }
}

llama_server_bridge_audio_session_flush_audio(session);

/* keep draining until your app decides the session is finished */

llama_server_bridge_audio_session_destroy(session);
```

## Current diarization model requirement

The current bridge accepts native Sortformer GGUF files for diarization inputs.
