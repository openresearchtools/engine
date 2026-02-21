# Third Party Notices

This repository contains runtime integration code and conversion/parity tooling that reference third-party projects.

## Runtime (shipped engine)

### 1) llama.cpp

- Role: Core C/C++ inference/runtime framework and server base.
- Source location in this repository layout: `../llama.cpp`
- Add-on integration files in this repository:
  - `../diarize/addons/overlay/llama.cpp/tools/CMakeLists.txt`
  - `../diarize/addons/overlay/llama.cpp/tools/server/*`
  - `../diarize/addons/overlay/llama.cpp/tools/pyannote/*`
  - `../diarize/addons/overlay/llama.cpp/tools/whisper/*`
- License: MIT
- License type: MIT
- License file: `llama.cpp-LICENSE.txt`

### 2) whisper.cpp

- Role: Native whisper transcription implementation integrated into `llama-server` in-process for audio flow.
- Source location in this repository layout: `../whisper.cpp`
- License: MIT
- License type: MIT
- License file: `whisper.cpp-LICENSE.txt`

### 3) pyannote.audio

- Role: reference/source for diarization pipeline structure and tensor naming/metadata semantics used by native C++ reimplementation.
- Native C++ runtime implementation is provided through overlayed pyannote integration under:
  - `../diarize/addons/overlay/llama.cpp/tools/pyannote/`
- Runtime endpoint path does not invoke Python.
- License: MIT
- License type: MIT
- License file: `pyannote.audio-LICENSE.txt`

### 4) WeSpeaker (transitive provenance via pyannote embedding stack)

- Role: upstream provenance for parts of embedding-model lineage referenced by pyannote embedding components.
- License: Apache-2.0
- License type: Apache-2.0
- License file: `WeSpeaker-LICENSE.txt`

### 5) docling (reference logic)

- Role: reference for VLM document-conversion behavior used by ENGINE `pdfvlm`.
- Source: `https://github.com/docling-project/docling`
- License type: MIT
- License file: `docling-LICENSE.txt`
- Attribution note: parts of VLM preprocessing logic were borrowed/adapted, including prompting, page-wise image rendering, scale/oversample flow, and Catmull-Rom (bicubic-style) downscale before inference.
- Borrowed/adapted scaling behavior used in `ENGINE/pdfvlm/src/pdf_to_markdown.rs` includes:
  - page-wise rasterization before VLM inference
  - target page scaling (`--scale`, default `2.0`)
  - temporary oversample (`--oversample`, default `1.5`)
  - Catmull-Rom (bicubic-style) resampling/downscale from temporary render to final target size

### 6) pdfium-render (Rust crate)

- Role: PDF rasterization binding used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file: `pdfium-render-LICENSE.md`

### 7) PDFium runtime binaries

- Runtime location: `../pdfium`
- License type: BSD-3-Clause + Apache-2.0 + additional third-party notices
- License file: `pdfium-LICENSE.txt`
- Binary source used in this project: `https://github.com/bblanchon/pdfium-binaries`
- Binary source license type: MIT
- Binary source license file: `pdfium-binaries-LICENSE.txt`
- Include corresponding PDFium/Chromium third-party notices in your final app distribution when required by your selected PDFium build.

### 8) FFmpeg runtime conversion (raw audio path)

- Role: in-memory audio normalization for bridge raw-audio transcription requests (convert to WAV 16-bit mono 16 kHz before endpoint call).
- Binary fetch source: `https://github.com/BtbN/FFmpeg-Builds`
- Binary fetch source license type: MIT
- Binary fetch source license file: `ffmpeg-builds-LICENSE.txt`
- Fetch script: `../build/download-ffmpeg-lgpl-win-x64.ps1`
- Expected runtime location: `../third_party/ffmpeg/`
- License type (intended build): LGPL (LGPL-only shared build)
- License file: `ffmpeg-LGPL-2.1.txt`
- Source/provenance note: `ffmpeg-SOURCE.txt`
- Include additional third-party notices shipped inside the downloaded FFmpeg package (if present) in your final distribution.

## Conversion / parity tooling (not required at runtime)

These packages/scripts are used by local conversion or parity scripts and are not required by the shipped native runtime inference path.

Tooling scripts:

- `../extras/convert_pyannote_checkpoint_to_gguf.py`
- `../extras/convert_pyannote_npz_to_gguf.py`
- `../extras/convert_pyannote_to_gguf.ps1`

Primary tooling dependencies with dedicated top-level license files:

- PyTorch (`torch`)
  - Role: read checkpoint tensors during conversion/parity workflows.
  - License type: BSD-3-Clause (with additional bundled third-party notices)
  - License files: `torch-LICENSE.txt`, `torch-NOTICE.txt`
- torchaudio
  - Role: audio/parity tooling dependency.
  - License type: BSD-2-Clause
  - License file: `torchaudio-LICENSE.txt`
- NumPy
  - Role: tensor and NPZ handling in conversion tooling.
  - License type: BSD-3-Clause (with bundled third-party license texts)
  - License file: `numpy-LICENSE.txt`

Full transitive tooling export:

- `tooling-full/`

## Runtime integration notes

- Audio patch/overlay mechanism for upstream sync is maintained in `../diarize/addons/overlay/llama.cpp/`.
- Bridge runtime integration code is maintained in `../bridge/`.
- PDF orchestration modules are in `../pdf/` and `../pdfvlm/`.
- Bridge raw-audio path supports in-memory conversion via FFmpeg when bridge is built with `LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON`.

## Dependency mapping (Rust workspace direct deps)

- `engine`: `serde_json`
- `pdf`: `anyhow`, `clap`, `once_cell`, `regex`, `walkdir`, `pdfium-render`
- `pdfvlm`: `pdfium-render`, `image`, `encoding_rs`

Direct crate license files copied into this folder:

- `serde_json` (MIT OR Apache-2.0): `serde_json-LICENSE-MIT.txt`, `serde_json-LICENSE-APACHE.txt`
- `anyhow` (MIT OR Apache-2.0): `anyhow-LICENSE-MIT.txt`, `anyhow-LICENSE-APACHE.txt`
- `clap` (MIT OR Apache-2.0): `clap-LICENSE-MIT.txt`, `clap-LICENSE-APACHE.txt`
- `once_cell` (MIT OR Apache-2.0): `once_cell-LICENSE-MIT.txt`, `once_cell-LICENSE-APACHE.txt`
- `regex` (MIT OR Apache-2.0): `regex-LICENSE-MIT.txt`, `regex-LICENSE-APACHE.txt`
- `walkdir` (Unlicense OR MIT): `walkdir-UNLICENSE.txt`, `walkdir-LICENSE-MIT.txt`, `walkdir-COPYING.txt`
- `image` (MIT OR Apache-2.0): `image-LICENSE-MIT.txt`, `image-LICENSE-APACHE.txt`
- `encoding_rs` (Apache-2.0 OR MIT + WHATWG text): `encoding_rs-LICENSE-MIT.txt`, `encoding_rs-LICENSE-APACHE.txt`, `encoding_rs-LICENSE-WHATWG.txt`
- `pdfium-render` (MIT OR Apache-2.0): `pdfium-render-LICENSE.md`
