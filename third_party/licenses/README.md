# Third Party Notices

This repository contains runtime integration code and conversion/parity tooling that reference third-party projects.

## Backend key-license bundles

- CUDA bundle key licenses: `LICENSES-cuda.txt`
- Vulkan bundle key licenses: `LICENSES-vulkan.txt`
- Legacy/default key bundle (full): `LICENSES.txt`

## Runtime (shipped engine)

### 1) llama.cpp

- Role: Core C/C++ inference/runtime framework and server base.
- Source location in this repository layout: [`third_party/llama.cpp`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp)
- Add-on integration files in this repository:
  - [`diarize/addons/overlay/llama.cpp/tools/CMakeLists.txt`](https://github.com/openresearchtools/engine/blob/main/diarize/addons/overlay/llama.cpp/tools/CMakeLists.txt)
  - [`diarize/addons/overlay/llama.cpp/tools/server/`](https://github.com/openresearchtools/engine/tree/main/diarize/addons/overlay/llama.cpp/tools/server/)
  - [`diarize/addons/overlay/llama.cpp/tools/pyannote/`](https://github.com/openresearchtools/engine/tree/main/diarize/addons/overlay/llama.cpp/tools/pyannote/)
  - [`diarize/addons/overlay/llama.cpp/tools/whisper/`](https://github.com/openresearchtools/engine/tree/main/diarize/addons/overlay/llama.cpp/tools/whisper/)
- License: MIT
- License type: MIT
- License file: `llama.cpp-LICENSE.txt`

### 1a) Additional licenses pulled into this app through llama.cpp build

The `llama.cpp` CMake build used by this app aggregates extra third-party licenses into its generated `license.cpp`.
For the current ENGINE Windows CUDA build, those additional upstream projects are:

- Repository: `yhirose/cpp-httplib`
  - Upstream: `https://github.com/yhirose/cpp-httplib`
  - License type: MIT
  - License file in this folder: `cpp-httplib-LICENSE.txt`
- Repository: `nlohmann/json`
  - Upstream: `https://github.com/nlohmann/json`
  - License type: MIT
  - License file in this folder: `jsonhpp-LICENSE.txt`
- Repository: `google/boringssl`
  - Upstream: `https://github.com/google/boringssl`
  - License type: ISC-style (BoringSSL license)
  - License file in this folder: `boringssl-LICENSE.txt`

Build-note: BoringSSL is fetched by CMake in this build profile (`LLAMA_BUILD_BORINGSSL=ON`) and linked into the shipped bridge/runtime binaries.

### 1b) ggml

- Role: low-level tensor/runtime framework used by `llama.cpp` and `whisper.cpp`.
- Upstream repository: `ggml-org/ggml`
- Upstream URL: `https://github.com/ggml-org/ggml`
- Source locations in this repository layout:
  - [`third_party/llama.cpp/ggml`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp/ggml)
  - [`third_party/whisper.cpp/ggml`](https://github.com/openresearchtools/engine/tree/main/third_party/whisper.cpp/ggml)
- License: MIT
- License type: MIT
- License file: `ggml-LICENSE.txt`

### 1c) miniaudio

- Role: embedded audio decode/capture helper used by the in-process whisper transcription path.
- Upstream repository: `mackron/miniaudio`
- Upstream URL: `https://github.com/mackron/miniaudio`
- Source location in this repository layout:
  - [`third_party/llama.cpp/vendor/miniaudio/miniaudio.h`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/vendor/miniaudio/miniaudio.h)
- Build-use locations in this repository layout:
  - [`third_party/llama.cpp/tools/whisper/whisper-common-audio.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/tools/whisper/whisper-common-audio.cpp)
  - [`third_party/llama.cpp/tools/pyannote/pyannote-diarize.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/tools/pyannote/pyannote-diarize.cpp)
- License: dual option (Public Domain / MIT No Attribution)
- License type: Public Domain (Unlicense-style) OR MIT-0
- License file: `miniaudio-LICENSE.txt`

### 1d) Additional ggml CPU-component attributions used by CPU backend variants

- YaRN reference implementation attribution inside ggml CPU rope path:
  - Source location: [`third_party/llama.cpp/ggml/src/ggml-cpu/ops.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/ggml/src/ggml-cpu/ops.cpp)
  - Upstream reference: `https://github.com/jquesnelle/yarn`
  - License type: MIT
  - License file: `yarn-LICENSE.txt`
- llamafile SGEMM component used by ggml CPU backend:
  - Source location: [`third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp)
  - Upstream reference: `https://github.com/Mozilla-Ocho/llamafile`
  - License type: MIT
  - License file: `llamafile-sgemm-LICENSE.txt`
- KleidiAI source attribution (used by ggml CPU backend when enabled):
  - Source locations:
    - [`third_party/llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp)
    - [`third_party/llama.cpp/ggml/src/ggml-cpu/kleidiai/kernels.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/ggml/src/ggml-cpu/kleidiai/kernels.cpp)
  - Upstream reference: `https://github.com/ARM-software/kleidiai`
  - License type: MIT
  - License file: `kleidiai-LICENSE.txt`

### 1e) Additional C/C++ source-attribution licenses (upstream originals)

- Repository: `openvinotoolkit/openvino`
  - Upstream: `https://github.com/openvinotoolkit/openvino`
  - Source-attribution locations:
    - [`third_party/llama.cpp/ggml/src/ggml-cpu/vec.h`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/ggml/src/ggml-cpu/vec.h)
    - [`third_party/whisper.cpp/ggml/src/ggml-cpu/vec.h`](https://github.com/openresearchtools/engine/blob/main/third_party/whisper.cpp/ggml/src/ggml-cpu/vec.h)
  - License type: Apache-2.0
  - License file in this folder: `openvino-LICENSE.txt`
- Repository: `ARM-software/optimized-routines`
  - Upstream: `https://github.com/ARM-software/optimized-routines`
  - Source-attribution locations:
    - [`third_party/llama.cpp/ggml/src/ggml-cpu/vec.h`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/ggml/src/ggml-cpu/vec.h)
    - [`third_party/whisper.cpp/ggml/src/ggml-cpu/vec.h`](https://github.com/openresearchtools/engine/blob/main/third_party/whisper.cpp/ggml/src/ggml-cpu/vec.h)
  - License type: MIT OR Apache-2.0 WITH LLVM-exception
  - License file in this folder: `arm-optimized-routines-LICENSE.txt`
- Repository: `cmp-nct/ggllm.cpp`
  - Upstream: `https://github.com/cmp-nct/ggllm.cpp`
  - Source-attribution location:
    - [`third_party/llama.cpp/src/llama-vocab.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/src/llama-vocab.cpp)
  - License type: MIT
  - License file in this folder: `ggllm.cpp-LICENSE.txt`
- Repository: `ivanyu/string-algorithms`
  - Upstream: `https://github.com/ivanyu/string-algorithms`
  - Source-attribution location:
    - [`third_party/llama.cpp/src/llama-sampler.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/src/llama-sampler.cpp)
  - License type: Public Domain (Unlicense text)
  - License file in this folder: `string-algorithms-LICENSE.txt`
- Repository: `LostRuins/koboldcpp`
  - Upstream: `https://github.com/LostRuins/koboldcpp`
  - Source-attribution location:
    - [`third_party/llama.cpp/src/llama-sampler.cpp`](https://github.com/openresearchtools/engine/blob/main/third_party/llama.cpp/src/llama-sampler.cpp)
  - License type: MIT (current upstream repository license)
  - License file in this folder: `koboldcpp-LICENSE.txt`
- Repository: `llvm/llvm-project`
  - Upstream: `https://github.com/llvm/llvm-project`
  - Source-attribution locations:
    - [`third_party/llama.cpp/ggml/src/ggml-sycl/`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp/ggml/src/ggml-sycl)
    - [`third_party/whisper.cpp/ggml/src/ggml-sycl/`](https://github.com/openresearchtools/engine/tree/main/third_party/whisper.cpp/ggml/src/ggml-sycl)
  - License type: Apache-2.0 WITH LLVM-exception
  - License file in this folder: `llvm-project-LICENSE.TXT`
  - Build note: SYCL backend sources are present in tree, but are not used in the current Windows CUDA release profile.

### 2) whisper.cpp

- Role: Native whisper transcription implementation integrated into `llama-server` in-process for audio flow.
- Source location in this repository layout: [`third_party/whisper.cpp`](https://github.com/openresearchtools/engine/tree/main/third_party/whisper.cpp)
- License: MIT
- License type: MIT
- License file: `whisper.cpp-LICENSE.txt`

### 3) pyannote.audio

- Role: reference/source for diarization pipeline structure and tensor naming/metadata semantics used by native C++ reimplementation.
- Native C++ runtime implementation is provided through overlayed pyannote integration under:
  - [`diarize/addons/overlay/llama.cpp/tools/pyannote/`](https://github.com/openresearchtools/engine/tree/main/diarize/addons/overlay/llama.cpp/tools/pyannote/)
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
- Borrowed/adapted scaling behavior used in [`engine/pdfvlm/src/pdf_to_markdown.rs`](https://github.com/openresearchtools/engine/blob/main/engine/pdfvlm/src/pdf_to_markdown.rs) includes:
  - page-wise rasterization before VLM inference
  - target page scaling (`--scale`, default `2.0`)
  - temporary oversample (`--oversample`, default `1.5`)
  - Catmull-Rom (bicubic-style) resampling/downscale from temporary render to final target size

### 6) pdfium-render (Rust crate)

- Role: PDF rasterization binding used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file: `pdfium-render-LICENSE.md`

### 7) PDFium runtime binaries

- Runtime location: [`third_party/pdfium`](https://github.com/openresearchtools/engine/tree/main/third_party/pdfium)
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
- Fetch script: [`build/download-ffmpeg-lgpl-win-x64.ps1`](https://github.com/openresearchtools/engine/blob/main/build/download-ffmpeg-lgpl-win-x64.ps1)
- Expected runtime location: [`third_party/ffmpeg`](https://github.com/openresearchtools/engine/tree/main/third_party/ffmpeg)
- License type (intended build): LGPL (LGPL-only shared build)
- License file: `ffmpeg-LGPL-2.1.txt`
- Source/provenance note: `ffmpeg-SOURCE.txt`

### 9) NVIDIA CUDA runtime libraries (Windows bundle)

- Role: GPU acceleration runtime libraries used by the CUDA backend in shipped ENGINE binaries.
- Typical shipped files in this project bundle:
  - `cublas64_13.dll`
  - `cublasLt64_13.dll`
- License/EULA type: NVIDIA CUDA EULA
- License files in this folder:
  - `nvidia-cuda-EULA.txt`
  - `nvidia-cuda-runtime-NOTICE.txt`
- Official EULA page: `https://docs.nvidia.com/cuda/eula/index.html`

## Conversion / parity tooling (not required at runtime)

These packages/scripts are used by local conversion or parity scripts and are not required by the shipped native runtime inference path.

Tooling scripts:

- [`extras/convert_pyannote_checkpoint_to_gguf.py`](https://github.com/openresearchtools/engine/blob/main/extras/convert_pyannote_checkpoint_to_gguf.py)
- [`extras/convert_pyannote_npz_to_gguf.py`](https://github.com/openresearchtools/engine/blob/main/extras/convert_pyannote_npz_to_gguf.py)
- [`extras/convert_pyannote_to_gguf.ps1`](https://github.com/openresearchtools/engine/blob/main/extras/convert_pyannote_to_gguf.ps1)

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

- [`tooling-full/`](https://github.com/openresearchtools/engine/tree/main/third_party/licenses/tooling-full/)

## Runtime integration notes

- Audio patch/overlay mechanism for upstream sync is maintained in [`diarize/addons/overlay/llama.cpp/`](https://github.com/openresearchtools/engine/tree/main/diarize/addons/overlay/llama.cpp/).
- Bridge runtime integration code is maintained in [`bridge/`](https://github.com/openresearchtools/engine/tree/main/bridge/).
- PDF orchestration modules are in [`pdf/`](https://github.com/openresearchtools/engine/tree/main/pdf/) and [`pdfvlm/`](https://github.com/openresearchtools/engine/tree/main/pdfvlm/).
- Bridge raw-audio path supports in-memory conversion via FFmpeg when bridge is built with `LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON`.

## Dependency mapping (Rust workspace direct deps)

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

- Role: command-line argument parsing for ENGINE CLI binaries.
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

- Role: filesystem directory traversal in runtime file-processing paths.
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
- License type: Apache-2.0 OR MIT (plus WHATWG text)
- License files:
  - `encoding_rs-LICENSE-MIT.txt`
  - `encoding_rs-LICENSE-APACHE.txt`
  - `encoding_rs-LICENSE-WHATWG.txt`

### `pdfium-render`

- Role: Rust binding layer to PDFium used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file:
  - `pdfium-render-LICENSE.md`

## Rust transitive license export (workspace)

- Full transitive Rust crate export (Windows target, non-dev graph): [`rust-full/`](https://github.com/openresearchtools/engine/tree/main/third_party/licenses/rust-full/)
