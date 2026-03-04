# GPU DLL Unification Plan (Full Matrix)

This document is the line-by-line matrix requested:
- each DLL entry point
- how it runs now
- how it must run in minimal mode
- which parameters it currently supports
- which parameters it must support

---

## 0) Global Default Contract (applies to all inference endpoints)

| ID | Rule | Current | Must |
|---|---|---|---|
| G01 | Universal GPU selector | `gpu` exists on bridge params and `--gpu` exists in CLI-style wrappers | `gpu=<list_devices index>` is the single minimal selector for all inference DLL paths |
| G02 | No GPU provided on Windows/Linux | CPU-only default (`devices="none"`) | Keep CPU-only default |
| G03 | No GPU provided on macOS/Metal | First GPU auto-select | Keep first GPU auto-select |
| G04 | GPU provided | Single-device path + `main_gpu=0` in single-device mode | Keep this as minimal behavior |
| G05 | Offload default with selected GPU | `n_gpu_layers=-1` unless caller overrides | Keep `-1` default if caller does not override |
| G06 | KV defaults | `kv_unified=1`, `no_kv_offload=0` by default | Keep as default unless caller overrides |
| G07 | Split defaults | `split_mode` defaults to none; no tensor split unless passed | Keep this |
| G08 | Override precedence | explicit caller values override defaults | Keep this |
| G09 | Index mapping | bridge avoids remapping through visibility env vars | Keep this |

---

## 1) DLL Entry Point Matrix (Every Export)

### 1.1 `llama-server-bridge.dll`

| ID | Entry Point | How It Runs Now | How It Must Run (Minimal) | Current Supported Params | Must Support Params |
|---|---|---|---|---|---|
| B01 | `llama_server_bridge_default_params()` | Returns initialized `llama_server_bridge_params` defaults; helper only | Keep helper; clearly documented as initializer (not endpoint) | none (returns struct) | none (returns struct) |
| B02 | `llama_server_bridge_default_chat_request()` | Returns default chat request struct | Keep | none (returns struct) | none |
| B03 | `llama_server_bridge_default_vlm_request()` | Returns default VLM request struct | Keep | none (returns struct) | none |
| B04 | `llama_server_bridge_empty_vlm_result()` | Returns zeroed result holder | Keep | none (returns struct) | none |
| B05 | `llama_server_bridge_default_embeddings_request()` | Returns default embeddings request struct | Keep | none (returns struct) | none |
| B06 | `llama_server_bridge_default_rerank_request()` | Returns default rerank request struct | Keep | none (returns struct) | none |
| B07 | `llama_server_bridge_default_audio_request()` | Returns default JSON-audio request struct | Keep | none (returns struct) | none |
| B08 | `llama_server_bridge_default_audio_raw_request()` | Returns default raw-audio request struct (`ffmpeg_convert=1`) | Keep | none (returns struct) | none |
| B09 | `llama_server_bridge_empty_json_result()` | Returns zeroed JSON result holder | Keep | none (returns struct) | none |
| B10 | `llama_server_bridge_create(params)` | Creates bridge runtime; applies defaults + validates `gpu/devices/split`; supports audio-only when model omitted | Keep; enforce global minimal contract for all tasks | `llama_server_bridge_params`: `model_path, mmproj_path, n_ctx, n_batch, n_ubatch, n_parallel, n_threads, n_threads_batch, n_gpu_layers, main_gpu, gpu, no_kv_offload, mmproj_use_gpu, cache_ram_mib, seed, ctx_shift, kv_unified, devices, tensor_split, split_mode, embedding, reranking, pooling_type` | Same list; same semantics; strict default behavior guarantee |
| B11 | `llama_server_bridge_destroy(bridge)` | Destroys context/routes/thread/backend | Keep | `bridge*` | `bridge*` |
| B12 | `llama_server_bridge_vlm_complete(bridge, req, out)` | VLM inference via in-process route; uses bridge runtime config | Keep; with selected `gpu`, text+mmproj default to same device unless explicit override | `llama_server_bridge_vlm_request`: `prompt, image_bytes, image_bytes_len, n_predict, id_slot, temperature, top_p, top_k, min_p, seed, repeat_last_n, repeat_penalty, presence_penalty, frequency_penalty, dry_multiplier, dry_allowed_length, dry_penalty_last_n` | Same list |
| B13 | `llama_server_bridge_chat_complete(bridge, req, out)` | Chat inference via in-process route | Keep; same `gpu` contract | `llama_server_bridge_chat_request`: `prompt, n_predict, id_slot, temperature, top_p, top_k, min_p, seed, repeat_last_n, repeat_penalty, presence_penalty, frequency_penalty, dry_multiplier, dry_allowed_length, dry_penalty_last_n` | Same list |
| B14 | `llama_server_bridge_embeddings(bridge, req, out)` | Embeddings endpoint call | Keep; same `gpu` contract | `llama_server_bridge_embeddings_request`: `body_json, oai_compat` | Same list |
| B15 | `llama_server_bridge_rerank(bridge, req, out)` | Rerank endpoint call | Keep; same `gpu` contract | `llama_server_bridge_rerank_request`: `body_json` | Same list |
| B16 | `llama_server_bridge_audio_transcriptions(bridge, req, out)` | JSON-body audio path (`/v1/audio/transcriptions`) | Keep; same `gpu` contract + same override precedence as raw path | `llama_server_bridge_audio_request`: `body_json` (body keys listed in Section 3) | Same body-key set and semantics |
| B17 | `llama_server_bridge_audio_transcriptions_raw(bridge, req, out)` | Raw bytes + metadata path; optional FFmpeg convert | Keep; same `gpu` contract + same key semantics as JSON path | `llama_server_bridge_audio_raw_request`: `audio_bytes, audio_bytes_len, audio_format, metadata_json, ffmpeg_convert` (metadata keys listed in Section 3) | Same list |
| B18 | `llama_server_bridge_result_free(out)` | Frees VLM/chat result strings | Keep | `llama_server_bridge_vlm_result*` | Same |
| B19 | `llama_server_bridge_json_result_free(out)` | Frees JSON result strings | Keep | `llama_server_bridge_json_result*` | Same |
| B20 | `llama_server_bridge_last_error(bridge)` | Returns bridge error text | Keep with clear/stable validation messages | `bridge*` | Same |
| B21 | `llama_server_bridge_list_devices(out_devices, out_count)` | Enumerates backend devices and indices | Keep as canonical source for `gpu` index | out pointers only | Same |
| B22 | `llama_server_bridge_free_devices(devices, count)` | Frees device enumeration array | Keep | device ptr + count | Same |

### 1.2 `pdfvlm.dll`

| ID | Entry Point | How It Runs Now | How It Must Run (Minimal) | Current Supported Params | Must Support Params |
|---|---|---|---|---|---|
| PV01 | `pdfvlm_run_from_argv(argc, argv, out_error)` | CLI-style parser inside DLL, routes PDF/image VLM conversion | Keep for compatibility only | argv flags: `--pdf, --image, --pdfium-lib/--pdfium-dll, --model, --mmproj, --out/--out-md, --out-dir, --pages, --scale, --oversample, --prompt, --n-predict, --n-ctx, --threads, --threads-batch, --batch-size, --parallel, --max-retries, --gpu, --devices, --n-gpu-layers, --main-gpu, --mmproj-use-gpu, --split-mode, --tensor-split` | Keep compatibility; plus add typed API (PV03-PV06) |
| PV02 | `pdfvlm_free_c_string(ptr)` | Frees error string from `pdfvlm_run_from_argv` | Keep | `char*` | Same |
| PV03 | **(to add)** `pdfvlm_default_params()` | Not present | Add typed initializer | n/a | returns typed params struct |
| PV04 | **(to add)** `pdfvlm_run_pdf(params, out)` | Not present | Add typed PDF execution API | n/a | typed params including `gpu/devices/split/offload/mmproj` + pdf input/output fields |
| PV05 | **(to add)** `pdfvlm_run_image(params, out)` | Not present | Add typed image execution API | n/a | typed params including `gpu/devices/split/offload/mmproj` + image input/output fields |
| PV06 | **(to add)** `pdfvlm_result_free(out)` | Not present | Add typed result free helper | n/a | typed result pointer |

### 1.3 `pdf.dll`

| ID | Entry Point | How It Runs Now | How It Must Run (Minimal) | Current Supported Params | Must Support Params |
|---|---|---|---|---|---|
| PD01 | `pdf_run_from_argv(argc, argv, out_error)` | Digital PDF extraction path, no model-inference GPU path | Keep | argv flags for extract flow (`--pdfium-lib`, extract args) | Same |
| PD02 | `pdf_free_c_string(ptr)` | Frees error string | Keep | `char*` | Same |

---

## 2) Runtime Parameter Matrix (`llama_server_bridge_params`)

| Param | Current Runtime / Default | Must Runtime (Minimal) | Currently Supported | Must Support |
|---|---|---|---|---|
| `model_path` | optional for audio-only, required for model tasks | same | yes | yes |
| `mmproj_path` | optional; used by VLM | if `gpu` set and no override, mmproj follows same selected device | yes | yes |
| `n_ctx` | default `32768` | same unless caller override | yes | yes |
| `n_batch` | default `2048` | same unless caller override | yes | yes |
| `n_ubatch` | default `2048` | same unless caller override | yes | yes |
| `n_parallel` | default `1` | same unless caller override | yes | yes |
| `n_threads` | default `8` | same unless caller override | yes | yes |
| `n_threads_batch` | default `8` | same unless caller override | yes | yes |
| `n_gpu_layers` | default `-1`; CPU path coerces effectively to CPU | if `gpu` set and caller does not override, keep `-1` | yes | yes |
| `main_gpu` | default `-1`; set to `0` in single-device mode | same | yes | yes |
| `gpu` | default `-1`; selects single device when set | universal minimal selector | yes | yes |
| `devices` | optional CSV, mutually exclusive with `gpu` | same; advanced override path | yes | yes |
| `split_mode` | defaults none | none unless explicitly set | yes | yes |
| `tensor_split` | optional CSV ratios | active only when explicitly set | yes | yes |
| `no_kv_offload` | default `0` | keep `0` unless explicit override | yes | yes |
| `kv_unified` | default `1` | keep `1` unless explicit override | yes | yes |
| `mmproj_use_gpu` | default `-1` auto | with selected GPU and mmproj present, defaults to same selected GPU | yes | yes |
| `cache_ram_mib` | default `-1` | same | yes | yes |
| `seed` | default `-1` | same | yes | yes |
| `ctx_shift` | default `1` | same | yes | yes |
| `embedding` | mode toggle | same | yes | yes |
| `reranking` | mode toggle | same | yes | yes |
| `pooling_type` | default `-1` | same | yes | yes |

---

## 3) Audio Transcription + Diarization Key Matrix (Every Key)

### 3.1 Routing / mode keys

| Key | Current Behavior | Must Minimal Behavior | Currently Supported | Must Support |
|---|---|---|---|---|
| `mode` | `subtitle`, `speech`, `transcript` | same | yes | yes |
| `custom` | mode-specific (`default/auto` or numeric depending on mode) | same | yes | yes |
| `gpu` | metadata override path exists | guaranteed parity across JSON and RAW endpoints | partial | yes |
| `device` | `gpu:N` style parse path exists | guaranteed parity across JSON and RAW endpoints | partial | yes |

### 3.2 Transcription source/backend keys

| Key | Current Behavior | Must Minimal Behavior | Currently Supported | Must Support |
|---|---|---|---|---|
| `transcription_backend` | `auto` mapped to inproc whisper | same | yes | yes |
| `whisper_model` | local whisper model path | same | yes | yes |
| `whisper_model_path` | alias for local whisper model path | same | yes | yes |
| `whisper_hf_repo` | HF source repo | same | yes | yes |
| `whisper_hf_file` | HF source file | same | yes | yes |
| `whisper_offline` | offline resolution flag | same | yes | yes |

### 3.3 Whisper runtime tuning keys

| Key | Current Behavior | Must Minimal Behavior | Currently Supported | Must Support |
|---|---|---|---|---|
| `whisper_threads` | controls whisper threads | same | yes | yes |
| `whisper_processors` | controls whisper processors | same | yes | yes |
| `whisper_max_len` | max token length option | same | yes | yes |
| `whisper_audio_ctx` | whisper audio context option | same | yes | yes |
| `whisper_best_of` | decoding option | same | yes | yes |
| `whisper_beam_size` | decoding option | same | yes | yes |
| `whisper_temperature` | decoding option | same | yes | yes |
| `whisper_translate` | translation flag | same | yes | yes |
| `whisper_no_gpu` | force CPU whisper | same | yes | yes |
| `whisper_gpu_device` | whisper GPU device index | same | yes | yes |
| `whisper_flash_attn` | flash attention on/off | same | yes | yes |
| `whisper_no_fallback` | no fallback option | same | yes | yes |
| `whisper_suppress_nst` | suppress non-speech tokens option | same | yes | yes |
| `whisper_word_time_offset_sec` | timestamp offset | same | yes | yes |
| `whisper_language` | language selection | same | yes | yes |
| `whisper_prompt` | whisper prompt | same | yes | yes |
| `seconds_per_timeline_token` | fallback timing scale | same | yes | yes |
| `source_audio_seconds` | source duration override | same | yes | yes |

### 3.4 Diarization source/backend keys

| Key | Current Behavior | Must Minimal Behavior | Currently Supported | Must Support |
|---|---|---|---|---|
| `diarization_backend` | `native_cpp` / `auto` | same | yes | yes |
| `diarization_models_dir` | local diarization model folder | same | yes | yes |
| `diarization_model_dir` | alias of models dir | same | yes | yes |
| `diarization_hf_repo` | HF diarization repo source | same | yes | yes |
| `diarization_device` | target device string for diarization runtime | same | yes | yes |
| `diarization_offline` | offline resolution flag | same | yes | yes |
| `diarization_embedding_min_segment_duration_sec` | diarization tuning | same | yes | yes |
| `diarization_embedding_max_segments_per_speaker` | diarization tuning | same | yes | yes |
| `diarization_min_duration_off_sec` | diarization tuning | same | yes | yes |
| `speaker_seg_max_gap_sec` | alignment segmentation tuning | same | yes | yes |
| `speaker_seg_max_words` | alignment segmentation tuning | same | yes | yes |
| `speaker_seg_max_duration_sec` | alignment segmentation tuning | same | yes | yes |
| `speaker_seg_split_on_hard_break` | alignment segmentation behavior | same | yes | yes |
| `aligner_plda_sim_threshold` | speaker alignment threshold | same | yes | yes |

### 3.5 Legacy compatibility keys

| Key | Current Behavior | Must Minimal Behavior | Currently Supported | Must Support |
|---|---|---|---|---|
| `enable_diarization` | legacy mode derivation path | keep compatibility | yes | yes |
| `transcript_format` | legacy mode derivation path | keep compatibility | yes | yes |
| `output_format` | legacy mode derivation path | keep compatibility | yes | yes |

---

## 4) Minimal Runtime Behavior Per Task (No Extra Flags)

| Task | Minimal Caller Input (Required) | Current Runtime | Must Runtime |
|---|---|---|---|
| Chat | `create(params)` + `chat_complete(req)`; optional `params.gpu` | Works | Keep |
| VLM | `create(params with model+mmproj)` + `vlm_complete(req)`; optional `params.gpu` | Works | Keep; ensure text+mmproj same selected GPU by default |
| Embeddings | `create(params)` + `embeddings(req)`; optional `params.gpu` | Works | Keep |
| Rerank | `create(params)` + `rerank(req)`; optional `params.gpu` | Works | Keep |
| Audio speech/subtitle | audio endpoint call + whisper source; optional `params.gpu` | Works | Keep; selected GPU applies unless explicit whisper override |
| Audio transcript+diarization | audio endpoint call + whisper source + diarization source; optional `params.gpu` | Works | Keep; selected GPU applies unless explicit whisper/diarization overrides |
| PDFVLM (current) | `pdfvlm_run_from_argv(...)` with required args; optional `--gpu` | Works | Keep compatibility |
| PDFVLM (target typed) | typed params struct + run function; optional `gpu` | Not available | Add |

---

## 5) Implementation Checklist (Tick Boxes)

### 5.1 API Shape

- [ ] Keep all current bridge exports stable.
- [ ] Add typed `pdfvlm` API exports without removing argv compatibility.
- [ ] Keep `pdf.dll` unchanged.

### 5.2 Default Wiring

- [ ] Enforce one `gpu` contract across chat/vlm/embed/rerank/audio/pdfvlm.
- [ ] Guarantee no implicit tensor split unless explicit split params are passed.
- [ ] Guarantee text+mmproj same selected GPU by default for VLM/PDFVLM.
- [ ] Guarantee KV unified + KV offload defaults unless explicit overrides.

### 5.3 Audio Parity

- [ ] Make JSON and RAW audio endpoints honor the same metadata key semantics.
- [ ] Guarantee `gpu` / `device` metadata aliases behave identically on both audio endpoints.
- [ ] Keep all advanced whisper/diarization keys accepted.

### 5.4 Validation

- [ ] Validate `gpu` index against `list_devices`.
- [ ] Reject invalid `gpu + devices` combinations.
- [ ] Keep clear error strings in `last_error` and JSON outputs.

### 5.5 Tests

- [ ] CPU-only default run (no `gpu`) passes for each task.
- [ ] Vulkan run on selected non-zero GPU passes for each task.
- [ ] CUDA run on selected non-zero GPU passes for each task.
- [ ] No unexpected compute spill to non-selected GPU in targeted tests.

