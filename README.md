# Openresearchtools-Engine

> **Early-stage repository:** this repo is still evolving quickly. The goal is to deliver a *single, embeddable, local AI runtime* that covers the common building blocks you usually end up wiring together from multiple projects.

Openresearchtools-Engine is a local AI runtime primarily based on Llama.CPP, that you can embed directly into an application.

It aims to unify chat, vision, embeddings, reranking, audio transcription/diarization, and PDF-to-Markdown in one native stack (Rust + C++), so you don’t have to glue together separate runtimes for each task.

## Implementation goals

* **Collapse multi-service pipelines into one embeddable engine** with a consistent API surface.
* **Keep deployment and runtime paths lightweight** (avoid heavy Python-first stacks in the inference/runtime layer).
* **Support true in-process integration**, not just process-spawn or HTTP-only approaches.
* **Make inference behavior controllable in production** with explicit GPU/CPU selection, offload controls, and multi-GPU knobs.
* **Handle both “easy” and “hard” PDFs** by supporting a fast digital-PDF path and a VLM-based conversion path for difficult layouts.

## What’s implemented so far

* In-process `llama-server-bridge` (no HTTP requirement for app embedding).
* Chat, VLM, embeddings, reranking.
* Audio transcription, plus an experimental high-quality Pyannote-based diarization path integrated into the llama runtime environment.
* PDF → Markdown:

  * fast native digital PDF path (`pdf.dll`)
  * VLM document-conversion path (`pdfvlm.dll`)
* GPU/CPU controls (single GPU, multi-GPU split, offload knobs).

## Project status and affiliation

* Some components are **experimental**, especially the diarization path.
* This project is an **independent engineering effort** and is **not affiliated with, sponsored by, or endorsed by** the upstream projects it builds on.
* References to third-party project names are for compatibility and attribution only.
* All third-party names and marks (including `llama.cpp`, `pyannote.audio`, `whisper.cpp`, `Docling`, `PDFium`, `FFmpeg`, and Qwen/Qwen3-VL) remain the property of their respective owners.

## How to embed it

* `engine.exe` is an example wrapper showing how to call functions.
* `engine` is not the embedding boundary; treat it as a reference CLI only.
* Production embedding target is native binaries:

  * `llama-server-bridge.dll`
  * `pdf.dll`
  * `pdfvlm.dll`

## Project layout

* `bridge/` native in-process bridge for llama runtime APIs.
* `engine/` Rust CLI wrapper.
* `pdf/` fast PDF extraction module.
* `pdfvlm/` PDF->image->VLM->Markdown module.
* `diarize/` patch assets for integrated audio stack.
* `build/` fetch/build scripts.
* `third_party/` runtime sources/binaries/licenses.

## CLI examples

### Device enumeration

Start here to confirm what the runtime can see (CPU/GPU devices) before you tune offload or multi-GPU splits.

```powershell
# Device enumeration
engine.exe list-devices
# Equivalent bridge form:
engine.exe bridge list-devices
```

### Common runtime flags (all bridge commands)

These runtime controls are available across `chat`, `vlm`, `audio`, `embed`, `rerank`, and `pdfvlm`:

* `--gpu <int>` single-device shortcut (device index from `list-devices`)
* `--devices <csv>` (for example `0`, `1`, `0,1`, or `none` for CPU-only)
* `--main-gpu <int>` index inside selected devices (default `-1`, auto in single-device mode)
* `--n-gpu-layers <int>` (`-1` = full offload where supported)
* `--mmproj-use-gpu <-1|0|1>` (`-1` auto, `1` GPU, `0` CPU)
* `--split-mode <none|layer|row>` and `--tensor-split <csv>` (multi-GPU split)
* `--threads <int>` and `--threads-batch <int>` (CPU compute thread controls)

Default behavior:
* Windows/Linux: if no GPU is specified (`--gpu`/`--devices` omitted), runtime is CPU-only
* macOS (Apple Silicon / Metal builds): if no GPU is specified, runtime selects the first GPU
* if `--gpu N` is specified, defaults become single-device on that GPU with full offload
* by default there is no tensor split; split only happens if `--split-mode` / `--tensor-split` is passed
* `--devices none` forces CPU-only on all platforms
* explicit flags always override defaults

Minimal runtime contract (DLL/embedded callers):
* pass only required task/model inputs plus optional device selector (`--gpu N` or `--devices ...`) when you want simple routing
* if a device selector is passed, model + KV cache + mmproj (auto mode) run on that selected device by default
* audio follows the same selected runtime device unless explicitly overridden with `--whisper-gpu-device` / `--diarization-device`
* all advanced flags are still supported; passing them overrides these defaults

### Chat

This runs a direct prompt-based chat request with full GPU offload on a single GPU. It's a good baseline test for "does the model run and is the GPU config correct?"

```powershell
# Chat with prompt (single GPU, full offload)
engine.exe chat `
  --model ".\models\model.gguf" `
  --prompt "Summarize key findings in 5 bullets." `
  --n-gpu-layers -1 `
  --main-gpu 0 `
  --n-ctx 50000 `
  --n-batch 1024 `
  --n-ubatch 1024 `
  --n-parallel 1 `
  --n-predict 10000
```

If you already have a Markdown file you want summarized, you can pass it directly. When you omit `--prompt`, the CLI uses a default summary prompt.

```powershell
# Chat from markdown only (uses default summary prompt)
engine.exe chat `
  --model ".\models\model.gguf" `
  --markdown ".\input.md" `
  --n-gpu-layers -1 `
  --main-gpu 0
```

To do targeted extraction or structured analysis, combine a prompt and a Markdown context file.

```powershell
# Chat with both prompt and markdown context
engine.exe chat `
  --model ".\models\model.gguf" `
  --prompt "Extract all statistical tests and p-values." `
  --markdown ".\input.md" `
  --n-gpu-layers -1 `
  --main-gpu 0
```

### Vision and VLM

Use `vlm` when you want to run a vision-language model over an image (including page renders) and produce Markdown or a prompt-driven description.

This example runs “image → Markdown” conversion with the default extraction prompt.

```powershell
# VLM markdown conversion (default prompt = markdown extraction)
engine.exe vlm `
  --model ".\models\vision.gguf" `
  --mmproj ".\models\mmproj.gguf" `
  --image ".\page.png" `
  --out ".\page.md" `
  --mmproj-use-gpu 1 `
  --n-gpu-layers -1 `
  --main-gpu 0
```

If you want the model to answer a specific question about an image, provide your own prompt.

```powershell
# VLM image chat (set your own prompt)
engine.exe vlm `
  --model ".\models\vision.gguf" `
  --mmproj ".\models\mmproj.gguf" `
  --image ".\image.png" `
  --prompt "Describe this image and summarize key elements." `
  --mmproj-use-gpu 0 `
  --n-gpu-layers -1 `
  --main-gpu 0
```

`--mmproj-use-gpu` controls where the vision projector runs:

* `-1` (default): auto-follow selected runtime device
* `1`: run mmproj on GPU
* `0`: run mmproj on CPU

### Multi‑GPU split

If a model is too large for one GPU, you can split across multiple devices. This example shows a layer split with an explicit tensor split ratio.

```powershell
# Multi-GPU split
engine.exe chat `
  --model ".\models\model.gguf" `
  --markdown ".\input.md" `
  --devices 0,1 `
  --split-mode layer `
  --tensor-split 0.6,0.4 `
  --n-gpu-layers -1 `
  --main-gpu 0
```

---

## Audio: transcription with and without diarization

Openresearchtools-Engine's audio path is designed for two common workflows:

* **Transcription without diarization** (single-speaker style output) using `--mode speech` or `--mode subtitle`.
* **Transcription with diarization** (speaker-aware transcript) using `--mode transcript` plus diarization models.

### Command shape

You can invoke the audio pipeline in either of these forms:

* `engine audio ...` or `engine bridge audio ...`
* Audio modes are always executed in audio-only runtime mode. A text `--model` GGUF is not required.
* If `--model` is provided on `engine audio`, it is ignored for compatibility.
* If `--gpu N` is set and no explicit `--whisper-gpu-device` / `--diarization-device` is provided, both follow the selected runtime device.
* If no device is set, audio follows the same defaults as the rest of the runtime (Windows/Linux CPU-only, macOS first GPU).

Output behavior:

* `--output-dir <dir>` writes the final file (`.srt` for `subtitle`, `.md` for `speech`/`transcript`) into that directory.
* If `--output-dir` is omitted, output defaults to the same directory as `--audio-file`.
* The pipeline keeps only the final output artifact.
* Audio input is normalized through FFmpeg conversion in RAM; supported input formats depend on the FFmpeg build you ship.

### `--mode` (main mode)

Current options are:

* `subtitle`
* `speech`
* `transcript`

### `--custom` (mode-specific behavior)

`--custom` is interpreted differently depending on the chosen mode:

* `subtitle`: `default` / `auto` **or** a positive number of seconds (float/int) to control windowing
* `speech`: `default` / `auto` **or** a positive number of seconds (float/int) to control windowing
* `transcript`: `default` / `auto` **or** a positive integer for a fixed speaker count

If `--custom` is omitted, the CLI uses `default`.

### Whisper model source (required)

Whisper is required for audio processing. Provide exactly one of:

* Local: `--whisper-model` (or `--whisper-model-path`)
* Hugging Face: `--whisper-hf-repo` + `--whisper-hf-file`

### Diarization sources and device (for `--mode transcript`)

When you want speaker-aware transcripts, provide diarization models in one of these ways:

* Local directory: `--diarization-models-dir <dir>`
* Hugging Face repo: `--diarization-hf-repo <repo>`

You can also optionally set:

* `--diarization-device <value>` (defaults to `auto`)

A repository of converted GGUF diarization models is available here:
[https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF](https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF)

### Advanced audio knobs (CLI flags or `--body-json`)

Advanced audio controls are available directly as `engine.exe audio` flags (including raw-bytes `--audio-file` runs), and can also be passed in request JSON via `--body-json`.

Whisper controls:

* `--whisper-threads`, `--whisper-processors`, `--whisper-max-len`, `--whisper-audio-ctx`
* `--whisper-best-of`, `--whisper-beam-size`, `--whisper-temperature`
* `--whisper-language`, `--whisper-prompt`, `--whisper-translate`
* `--whisper-no-fallback`, `--whisper-suppress-nst`
* `--whisper-no-gpu`, `--whisper-gpu-device`, `--whisper-flash-attn`, `--whisper-no-flash-attn`
* `--whisper-offline`
* `--whisper-word-time-offset-sec`

Diarization and alignment controls:

* `--diarization-backend` (`native_cpp`/`auto`)
* `--diarization-offline`
* `--diarization-embedding-min-segment-duration-sec`
* `--diarization-embedding-max-segments-per-speaker`
* `--diarization-min-duration-off-sec`
* `--speaker-seg-max-gap-sec`
* `--speaker-seg-max-words`
* `--speaker-seg-max-duration-sec`
* `--speaker-seg-split-on-hard-break`, `--speaker-seg-no-split-on-hard-break`
* `--aligner-plda-sim-threshold`

Pipeline/runtime controls:

* `--audio-only` (legacy compatibility flag; optional no-op)
* `--ffmpeg-convert`, `--no-ffmpeg-convert`
* `--transcription-backend`
* `--seconds-per-timeline-token`, `--source-audio-seconds`

### Custom Whisper model: timestamp alignment tuning (diarization)

If you use a custom Whisper model and diarized transcript speaker turns look shifted relative to words/subtitles, tune:

* `--whisper-word-time-offset-sec` (primary alignment control)
* `--source-audio-seconds` (optional timeline clamp)
* `--seconds-per-timeline-token` (fallback timing when word timestamps are sparse)

Recommended tuning flow:

* Start with `--whisper-word-time-offset-sec 0.73` (default behavior).
* Run a known audio sample in `--mode transcript` and inspect where speaker boundaries drift.
* Increase offset if words appear too early; decrease offset if words appear too late.
* Adjust in small steps (for example `0.05`-`0.15`) until speaker turns and transcript timing match.
* If needed, set `--source-audio-seconds` to the known audio duration to prevent end-of-file overshoot.

Example with a custom local Whisper model:

```powershell
engine.exe audio `
  --audio-file ".\meeting.mp3" `
  --output-dir ".\outputs" `
  --mode transcript `
  --custom auto `
  --whisper-model ".\models\my-custom-whisper.bin" `
  --diarization-models-dir ".\models\diarization" `
  --whisper-word-time-offset-sec 0.85 `
  --source-audio-seconds 1032.4
```

### Audio examples

This is a straightforward “speech mode” transcription run (no diarization). Use this when you just want clean text output and don’t need speaker separation.

```powershell
# speech mode, default custom
engine.exe audio `
  --audio-file ".\sample.mp3" `
  --output-dir ".\outputs" `
  --audio-format mp3 `
  --mode speech `
  --custom default `
  --whisper-model ".\models\whisper.bin"
```

This produces subtitle-style output, where you can control the window size via `--custom` (here, 4.5 seconds). It’s useful when you want timestamps/segments rather than one continuous paragraph.

```powershell
# subtitle mode, 4.5-second windowing via custom
engine.exe audio `
  --audio-file ".\sample.wav" `
  --output-dir ".\outputs" `
  --mode subtitle `
  --custom 4.5 `
  --whisper-model ".\models\whisper.bin"
```

This generates a speaker-aware transcript by enabling diarization. With `--custom auto`, the system estimates speaker count, and `--diarization-device` lets you choose where diarization runs (for example, CUDA, Vulkan, or auto).

```powershell
# transcript mode, auto speaker count, local diarization models
engine.exe audio `
  --audio-file ".\meeting.mp3" `
  --output-dir ".\outputs" `
  --mode transcript `
  --custom auto `
  --whisper-model ".\models\whisper.bin" `
  --diarization-models-dir ".\models\diarization" `
  --diarization-device auto
```

**Offline note:** if you want to run diarization fully offline with `--diarization-models-dir`, download *all required diarization model files* into that single folder (and keep the directory contents intact). The runtime expects everything it needs to be present locally in that directory.

This example also runs a diarized transcript, but forces a fixed speaker count (`--custom 3`) and pulls both Whisper and diarization models from Hugging Face.

```powershell
# transcript mode, fixed 3 speakers, diarization from HF
engine.exe audio `
  --audio-file ".\meeting.mp3" `
  --output-dir ".\outputs" `
  --mode transcript `
  --custom 3 `
  --whisper-hf-repo ggerganov/whisper.cpp `
  --whisper-hf-file ggml-tiny.en.bin `
  --diarization-hf-repo openresearchtools/speaker-diarization-community-1-GGUF `
  --diarization-device cuda
```

If you prefer a JSON request payload, you can still pass the same advanced controls through `--body-json`.

```powershell
# advanced audio knobs via body JSON
engine.exe audio --body-json ".\audio_request.json"
```

---

## PDF to Markdown

For PDFs, you generally have two paths:

* Use the **fast digital extractor** when the PDF has good text structure. (Important note, tables, formulas and any special layouts will be rendered in line, not great for complex tables, but extremely fast).
* Use the **VLM conversion** when the PDF is scanned, layout-heavy, or loses structure in digital extraction.

Fast digital PDF conversion:

```powershell
# Fast digital PDF conversion
engine.exe pdf extract --input ".\paper.pdf" --output ".\paper_fast.md" --overwrite
```

VLM PDF conversion (PDF → render → VLM → Markdown). Choose the option that matches how you ship the PDFium runtime library:

```powershell
# PDF VLM conversion
# Option A: pass library path each call
engine.exe pdfvlm `
  --pdf ".\paper.pdf" `
  --pdfium-lib ".\vendor\pdfium\pdfium.dll" `
  --model ".\models\vision.gguf" `
  --mmproj ".\models\mmproj.gguf" `
  --out ".\paper_vlm.md" `
  --threads 32 `
  --threads-batch 32 `
  --mmproj-use-gpu 1 `
  --n-gpu-layers -1 `
  --main-gpu 0

# Option B: bundled app - if PDFium is under vendor/pdfium next to engine(.exe), omit --pdfium-lib
engine.exe pdfvlm --pdf ".\paper.pdf" --model ".\models\vision.gguf" --mmproj ".\models\mmproj.gguf" --out ".\paper_vlm.md"

# Option C: set env var once (PDFIUM_DLL is still accepted for compatibility)
$env:PDFIUM_LIB=".\vendor\pdfium\pdfium.dll"
engine.exe pdfvlm --pdf ".\paper.pdf" --model ".\models\vision.gguf" --mmproj ".\models\mmproj.gguf" --out ".\paper_vlm.md"
```

### VLM model note (tested configuration)

For scientific PDF → Markdown conversion, we tested the Qwen3-VL GGUF release:
[https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main)

In our testing, we got **reasonably high quality results** with:

* `Qwen3VL-8B-Instruct-Q8_0.gguf` (fits a 16GB VRAM GPU)
* `mmproj-Qwen3VL-8B-Instruct-F16.gguf`

One important caveat: on **large, complex tables**, the model can occasionally make structural mistakes (for example, attributing a number to the wrong row or the wrong column). If you plan to extract data from tables, it’s strongly recommended to **inspect the original PDF and the tables themselves** before trusting downstream derived values.

We have **no affiliation with the Qwen team**. This is simply a personal observation after testing multiple models that fit within a 16GB VRAM GPU.

---

## Embeddings

`embed` is JSON-in/JSON-out.

- `--body-json` accepts either:
  - a JSON string payload, or
  - a path to a JSON file
- If `--out` is not set, response JSON is printed to console.
- If `--out` is set, response JSON is written to that file.

Batching (important):

- API-style batching: put multiple items in `"input": [...]` inside one JSON request.
- Markdown batching mode: use `--markdown` with:
  - `--batch-size <N>`
  - `--chunk-words <N>`
  - `--batches <N>`
- `--batch-size/--chunk-words/--batches` apply to `--markdown` mode (not `--body-json` mode).

API-style batched request (single call, many input rows):

```powershell
$payload = '{"input":["row 1 text","row 2 text","row 3 text","row 4 text"],"encoding_format":"float"}'

engine.exe embed `
  --model ".\models\embedding.gguf" `
  --body-json $payload `
  --out ".\embed_response.json"
```

Markdown batching example (multiple embedding calls generated by CLI):

```powershell
engine.exe embed `
  --model ".\models\embedding.gguf" `
  --markdown ".\corpus.md" `
  --batch-size 64 `
  --chunk-words 300 `
  --batches 10 `
  --out ".\embed_batched_response.json"
```

Minimal inline payload (no temp file):

```powershell
$payload = '{"input":["row one text","row two text","row three text"],"encoding_format":"float"}'

engine.exe embed `
  --model ".\models\embedding.gguf" `
  --body-json $payload
```

Save response to file:

```powershell
$payload = '{"input":["row one text","row two text","row three text"],"encoding_format":"float"}'

engine.exe embed `
  --model ".\models\embedding.gguf" `
  --body-json $payload `
  --out ".\embed_response.json"
```

File-based payload (if you prefer files):

```powershell
@'
{
  "input": ["a","b","c"],
  "encoding_format": "float"
}
'@ | Set-Content .\embed_request.json

engine.exe embed `
  --model ".\models\embedding.gguf" `
  --body-json ".\embed_request.json" `
  --out ".\embed_response.json"
```

---

## Reranking

`rerank` is also JSON-in/JSON-out.

Request shape:

```json
{
  "query": "text to match against",
  "documents": ["candidate 1", "candidate 2", "candidate 3"],
  "top_n": 2
}
```

Notes:

- `query` is required.
- `documents` is required (rerank needs candidates to score/sort).
- `top_n` is optional.
- `--body-json` accepts inline JSON string or a file path.
- Without `--out`, response JSON prints to console.

What `documents` means:

- `documents` is an array of plain text strings (sentences/paragraphs/chunks).
- They are candidate texts that will be ranked by relevance to `query`.
- `documents` is not a file-format field and not a file path list.
- If your source is a file (PDF/MD/TXT), extract/split text first, then pass those text chunks in `documents`.

Example payload with real candidate texts:

```json
{
  "query": "find statements about adverse effects",
  "documents": [
    "Mild headache was reported in 3% of patients.",
    "The model uses grouped-query attention and rotary embeddings.",
    "Nausea and dizziness were the most common side effects."
  ],
  "top_n": 2
}
```

Minimal inline payload (no temp file):

```powershell
$payload = '{"query":"table extraction quality","documents":["doc1","doc2","doc3"],"top_n":2}'

engine.exe rerank `
  --model ".\models\reranker.gguf" `
  --body-json $payload
```

Save response to file:

```powershell
$payload = '{"query":"find adverse effects","documents":["row A text","row B text","row C text"],"top_n":3}'

engine.exe rerank `
  --model ".\models\reranker.gguf" `
  --body-json $payload `
  --out ".\rerank_response.json"
```

File-based payload (if you prefer files):

```powershell
@'
{
  "query": "find rows about adverse effects",
  "documents": [
    "document row A",
    "document row B",
    "document row C"
  ],
  "top_n": 3
}
'@ | Set-Content .\rerank_request.json

engine.exe rerank `
  --model ".\models\reranker.gguf" `
  --body-json ".\rerank_request.json" `
  --out ".\rerank_response.json"
```

With multi-GPU split:

```powershell
$payload = '{"query":"find adverse effects","documents":["row A text","row B text","row C text"],"top_n":3}'

engine.exe rerank `
  --model ".\models\reranker.gguf" `
  --body-json $payload `
  --devices 0,1 `
  --split-mode layer `
  --tensor-split 0.6,0.4 `
  --n-gpu-layers -1
```

---

## Build docs

Build/fetch instructions are in:

* `build/README.md`

## Acknowledgments

Openresearchtools-Engine is possible because of the open work done by these projects. We are genuinely grateful to their maintainers and contributors. Without them, this project would not exist.

* `llama.cpp`: core model runtime, GPU offload controls, KV-cache behavior, multi-GPU split controls, and server-side inference lifecycle patterns used by the bridge and engine orchestration.
* `whisper.cpp`: transcription pipeline foundations, including audio-to-token flow, timestamp-oriented decoding behavior, and integration patterns for speech tasks.
* `pyannote.audio` and `WeSpeaker`: diarization lineage and reference ideas for segmentation/embedding-style speaker processing, plus speaker-turn reconstruction expectations used in the experimental diarization path.
* `Docling`: practical references for VLM document-conversion behavior, including page rendering/scaling heuristics and Markdown-oriented extraction expectations for PDF-to-Markdown workflows.
* `PDFium` and `pdfium-render`: PDF rasterization and page access primitives used for native page rendering/extraction in the PDF modules.
* `FFmpeg` (LGPL shared builds): audio normalization and format conversion path used when input media needs conversion to inference-friendly audio format.
* Rust ecosystem crates in `engine`, `pdf`, and `pdfvlm`: CLI plumbing, parsing, and runtime glue that make the native components usable as a cohesive application layer.

For full notices, license types, and source provenance:

* `third_party/licenses/README.md`
