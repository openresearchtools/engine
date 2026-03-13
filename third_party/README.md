# Third-Party Notices

This repository contains shipped runtime code, bundled runtime dependencies, native adaptations, and a small amount of repo-kept conversion/parity tooling.

Openresearchtools-Engine source code is licensed under the MIT License. Third-party dependencies and bundled components remain licensed under their respective original licenses.

This file is primarily a provenance map for source code and tooling that were
directly imported, adapted, or rewritten from named upstream projects.
Package-managed dependencies that are consumed in their normal published form
remain listed here at license-inventory level only; they are not broken down
by internal source-file lineage unless this repo directly imported/adapted
their source.

Where mixed-source files are involved, the file lists below are identigying  the main ENGINE files materially informed by a given
upstream project; they are not a claim that every listed line came verbatim
from that upstream.

## Bundle-facing license files

- Canonical consolidated bundle license file in the repo: `../LICENSES.md`
- Canonical third-party notice/provenance file in the repo: `README.md`
- Bundle-time rename for packaged apps: `README.md` is copied as `Third-Party-Notices.md`

Packaged applications now ship one overinclusive `LICENSES.md` plus one `Third-Party-Notices.md` at bundle root. The combined license file intentionally over-includes text so one file covers all supported bundle variants.

## Directly imported or adapted source in shipped runtime

### 1) llama.cpp

- Role: primary upstream C/C++ runtime base that the core ENGINE inference/server stack is built on.
- Upstream source snapshot kept in this repository: `third_party/llama.cpp`
- ENGINE code heavily builds on and extends that base through prepared-build patches plus repo integration layers.
- These ENGINE-added layers are where the native audio/session, In process bridge, Whisper.cpp, Sortformer, and Voxtral features are integrated on top of original llama cpp; they are not claims about upstream `llama.cpp` shipping those features by itself.
- 
- License type: MIT
- License file: `llama.cpp-LICENSE.txt`

### 1a) Additional licenses pulled into this app through the llama.cpp build

The `llama.cpp` CMake build used by this app aggregates extra third-party licenses into its generated `license.cpp`.

- `yhirose/cpp-httplib`
  - Upstream: <https://github.com/yhirose/cpp-httplib>
  - License type: MIT
  - License file: `cpp-httplib-LICENSE.txt`
- `nlohmann/json`
  - Upstream: <https://github.com/nlohmann/json>
  - License type: MIT
  - License file: `jsonhpp-LICENSE.txt`
- `google/boringssl`
  - Upstream: <https://github.com/google/boringssl>
  - License type: ISC-style / BoringSSL license
  - License file: `boringssl-LICENSE.txt`

Build note: BoringSSL is fetched by CMake in build profiles that enable `LLAMA_BUILD_BORINGSSL`.

### 1b) ggml

- Role: low-level tensor/runtime framework used by `llama.cpp`, `whisper.cpp`, and the native realtime subsystems.
- Upstream repository: `ggml-org/ggml`
- Upstream URL: <https://github.com/ggml-org/ggml>
- Source locations in this repository layout:
  - `third_party/llama.cpp/ggml`
  - `third_party/whisper.cpp/ggml`
- License type: MIT
- License file: `ggml-LICENSE.txt`

### 1c) miniaudio

- Role: embedded audio decode/capture helper used by the native whisper audio path.
- Upstream repository: `mackron/miniaudio`
- Upstream URL: <https://github.com/mackron/miniaudio>
- Source location in this repository layout:
  - `third_party/llama.cpp/vendor/miniaudio/miniaudio.h`
- Build-use location in this repository layout:
  - `diarize/addons/overlay/llama.cpp/tools/whisper/whisper-common-audio.cpp`
- License type: Public Domain OR MIT-0
- License file: `miniaudio-LICENSE.txt`

### 1d) Additional ggml CPU-component attributions

- YaRN reference implementation attribution inside ggml CPU rope path
  - Upstream: <https://github.com/jquesnelle/yarn>
  - License type: MIT
  - License file: `yarn-LICENSE.txt`
- llamafile SGEMM component used by ggml CPU backend
  - Upstream: <https://github.com/Mozilla-Ocho/llamafile>
  - License type: MIT
  - License file: `llamafile-sgemm-LICENSE.txt`
- KleidiAI source attribution used by ggml CPU backend when enabled
  - Upstream: <https://github.com/ARM-software/kleidiai>
  - License type: MIT
  - License file: `kleidiai-LICENSE.txt`

### 1e) Additional C/C++ source-attribution licenses

- `openvinotoolkit/openvino`
  - License type: Apache-2.0
  - License file: `openvino-LICENSE.txt`
- `ARM-software/optimized-routines`
  - License type: MIT OR Apache-2.0 WITH LLVM-exception
  - License file: `arm-optimized-routines-LICENSE.txt`
- `cmp-nct/ggllm.cpp`
  - License type: MIT
  - License file: `ggllm.cpp-LICENSE.txt`
- `ivanyu/string-algorithms`
  - License type: Public Domain / Unlicense text
  - License file: `string-algorithms-LICENSE.txt`
- `LostRuins/koboldcpp`
  - License type: MIT
  - License file: `koboldcpp-LICENSE.txt`
- `llvm/llvm-project`
  - License type: Apache-2.0 WITH LLVM-exception
  - License file: `llvm-project-LICENSE.TXT`

### 2) whisper.cpp

- Role: native Whisper transcription implementation integrated into the in-process audio flow.
- Source location in this repository layout: `third_party/whisper.cpp`
- License type: MIT
- License file: `whisper.cpp-LICENSE.txt`

### 3) voxtral-cpp

- Role: primary native `ggml` implementation base adapted for the current Voxtral realtime runtime, and the source origin for the repo-kept Voxtral GGUF conversion tooling.
- ENGINE-owned runtime files adapted from or heavily informed by `voxtral-cpp`:
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.h`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-backend.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-backend.h`
- ENGINE-kept conversion tooling adapted from `voxtral-cpp`:
  - `build/voxtral/convert_voxtral_to_gguf.py`
- Upstream reference points:
  - upstream runtime/math implementation in `voxtral-cpp/src/voxtral.cpp`
  - upstream GGUF conversion tooling in `voxtral-cpp/tools/convert_voxtral_to_gguf.py`
- Upstream reference: local source base mirrored from `voxtral-cpp`
- License type: MIT
- License file: `voxtral-cpp-LICENSE.txt`

### 4) NVIDIA NeMo / Sortformer references

- Role: reference/source for Sortformer archive semantics, config/tensor naming, and parity validation used by the native Sortformer conversion and validation flow only.
- The current Sortformer converter in `build/sortformer/convert_nemo_sortformer_to_gguf.py` is a repo-written ENGINE script, not a copied upstream converter.
- Its source basis was:
  - NVIDIA NeMo Sortformer archive/checkpoint structure
  - NeMo config/tensor naming semantics from the `.nemo` archive
  - NeMo parity/reference runs used during bring-up
- Main ENGINE files materially informed by NVIDIA NeMo / Sortformer references:
  - `build/sortformer/convert_nemo_sortformer_to_gguf.py`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-gguf.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-schema.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-postprocess.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/realtime-smoke.cpp`
- License type: Apache-2.0
- License file: `nvidia-nemo-LICENSE.txt`

### 4a) parakeet.cpp

- Role: external C++ Sortformer reference studied during native Sortformer bring-up, especially for conversion/runtime structure cross-checking.
- Main ENGINE files materially cross-checked or informed against `parakeet.cpp` Sortformer references:
  - `build/sortformer/convert_nemo_sortformer_to_gguf.py`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-model.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-streaming.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-frontend.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-encoder.cpp`
- Upstream: <https://github.com/frikallo/parakeet.cpp>
- License type: MIT
- License file: `parakeet-cpp-LICENSE.txt`

### 5) docling

- Role: reference logic for VLM document-conversion behavior used by `pdfvlm`.
- Upstream source: <https://github.com/docling-project/docling>
- License type: MIT
- License file: `docling-LICENSE.txt`

### 6) pdfium-render

- Role: PDF rasterization binding used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file: `pdfium-render-LICENSE.md`

### 7) PDFium runtime binaries

- Runtime location: `third_party/pdfium`
- Binary source used in this project: <https://github.com/bblanchon/pdfium-binaries>
- License type: BSD-3-Clause + Apache-2.0 + additional third-party notices
- License file: `pdfium-LICENSE.txt`
- Binary-source license type: MIT
- Binary-source license file: `pdfium-binaries-LICENSE.txt`

### 8) FFmpeg runtime conversion

- Role: in-memory audio normalization and decode/resample path for raw/file audio ingress.
- Windows/Linux binary fetch source: <https://github.com/BtbN/FFmpeg-Builds>
- Source-build reference for macOS arm64: <https://github.com/FFmpeg/FFmpeg>
- License type: LGPL 2.1-or-later intent for the shared-runtime builds staged by this repo
- License files:
  - `ffmpeg-builds-LICENSE.txt`
  - `ffmpeg-LGPL-2.1.txt`
  - `ffmpeg-SOURCE.txt`
  - `ffmpeg-SOURCE-windows-x64.txt`
  - `ffmpeg-SOURCE-ubuntu-x64.txt`
  - `ffmpeg-SOURCE-macos-arm64.txt`

### 9) NVIDIA CUDA runtime libraries

- Role: GPU acceleration runtime libraries used by CUDA backend bundles.
- License type: NVIDIA CUDA EULA
- License files:
  - `nvidia-cuda-EULA.txt`
  - `nvidia-cuda-runtime-NOTICE.txt`

## External reference and validation repositories (not linked into shipped runtime)

These repositories were used as behavior, validation, or benchmarking references during native bring-up. They are not bundled as runtime code by this repository.

### vLLM

- Role: realtime behavior and throughput reference for Voxtral evaluation.
- Main ENGINE files whose Voxtral behavior was materially cross-checked or informed against `vLLM` references:
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.h`
- Upstream: <https://github.com/vllm-project/vllm>
- License type: Apache-2.0
- License file: `vllm-LICENSE.txt`

### voxtral.c

- Role: streaming API/reference behavior input for Voxtral session semantics.
- Main ENGINE files materially informed by `voxtral.c` streaming/session ideas:
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-backend.cpp`
  - `bridge/llama_server_bridge.cpp`
  - `engine/src/llama_bridge.rs`
- Upstream: <https://github.com/antirez/voxtral.c>
- License type: MIT
- License file: `voxtral.c-LICENSE.txt`

### voxtral-mini-realtime-rs

- Role: external Rust/GGUF reference for Voxtral realtime integration patterns and session wiring study.
- Upstream: <https://github.com/TrevorS/voxtral-mini-realtime-rs>
- License type: Apache-2.0
- License file: `voxtral-mini-realtime-rs-LICENSE.txt`

### mlx-audio

- Role: external MLX-side reference for Mistral/Voxtral audio model behavior and implementation study.
- Upstream: <https://github.com/Blaizzy/mlx-audio>
- License type: MIT
- License file: `mlx-audio-LICENSE.txt`

## Directly imported or adapted repo-kept conversion / parity tooling (not required at runtime)

This section covers the Python dependency surface used by the repo-kept conversion/parity tooling already described above:

- `build/sortformer/convert_nemo_sortformer_to_gguf.py`
- `build/voxtral/convert_voxtral_to_gguf.py`

Primary direct Python dependencies used by those repo-kept converters:

- NumPy
  - Role: tensor/array handling in both the Sortformer converter and the repo-kept Voxtral GGUF converter
  - License type: BSD-3-Clause with bundled third-party notices
  - License file: `numpy-LICENSE.txt`
- PyTorch (`torch`)
  - Role: checkpoint tensor loading during Sortformer conversion/parity workflows
  - License type: BSD-3-Clause with additional notices
  - License files: `torch-LICENSE.txt`, `torch-NOTICE.txt`
- PyYAML
  - Role: NeMo archive config parsing during Sortformer conversion tooling
  - License type: MIT
  - License file: `PyYAML-LICENSE.txt`

Checked-in tooling license snapshot:

- `tooling-full/`

This folder is a repo-kept license snapshot for the current Python tooling stack. It is not part of the shipped runtime bundle.

## Runtime integration notes

- Audio patch/overlay mechanism for upstream sync is maintained in `diarize/addons/overlay/llama.cpp/`.
- Bridge runtime integration code is maintained in `bridge/`.
- PDF orchestration modules are in `pdf/` and `pdfvlm/`.
- Bridge raw-audio path supports in-memory conversion via FFmpeg when bridge is built with `LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON`.

## Package-managed dependency mapping (license inventory only)

### `serde_json`

- Role: JSON parsing/serialization in Rust runtime modules.
- License type: MIT OR Apache-2.0
- License files:
  - `serde_json-LICENSE-MIT.txt`
  - `serde_json-LICENSE-APACHE.txt`

### `anyhow`

- Role: error propagation/context in Rust runtime modules.
- License type: MIT OR Apache-2.0
- License files:
  - `anyhow-LICENSE-MIT.txt`
  - `anyhow-LICENSE-APACHE.txt`

### `clap`

- Role: command-line argument parsing for CLI binaries.
- License type: MIT OR Apache-2.0
- License files:
  - `clap-LICENSE-MIT.txt`
  - `clap-LICENSE-APACHE.txt`

### `once_cell`

- Role: one-time/lazy static initialization in runtime modules.
- License type: MIT OR Apache-2.0
- License files:
  - `once_cell-LICENSE-MIT.txt`
  - `once_cell-LICENSE-APACHE.txt`

### `regex`

- Role: regular-expression matching used by runtime text processing paths.
- License type: MIT OR Apache-2.0
- License files:
  - `regex-LICENSE-MIT.txt`
  - `regex-LICENSE-APACHE.txt`

### `walkdir`

- Role: filesystem traversal in runtime file-processing paths.
- License type: Unlicense OR MIT
- License files:
  - `walkdir-UNLICENSE.txt`
  - `walkdir-LICENSE-MIT.txt`
  - `walkdir-COPYING.txt`

### `image`

- Role: image buffer/format handling in runtime document/VLM processing paths.
- License type: MIT OR Apache-2.0
- License files:
  - `image-LICENSE-MIT.txt`
  - `image-LICENSE-APACHE.txt`

### `encoding_rs`

- Role: encoding conversion for text handling in runtime paths.
- License type: Apache-2.0 OR MIT plus WHATWG text
- License files:
  - `encoding_rs-LICENSE-MIT.txt`
  - `encoding_rs-LICENSE-APACHE.txt`
  - `encoding_rs-LICENSE-WHATWG.txt`

### `pdfium-render`

- Role: Rust binding layer to PDFium used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file: `pdfium-render-LICENSE.md`

## Rust transitive license export

- Full transitive Rust crate export (Windows target, non-dev graph): `rust-full/`
