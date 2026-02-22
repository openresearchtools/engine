# Build Guide

All build/test outputs must stay outside this repo, under `..\ENGINEbuilds`.

## Prerequisites

- CMake on PATH (or pass `-CmakeExe`).
- Rust/Cargo on PATH (or pass `-CargoExe`).
- `third_party/llama.cpp` must be actual `llama.cpp` sources.
- `third_party/whisper.cpp` required at build time for the in-process audio integration.

## Fetch runtime dependencies (manual)

Windows local/manual fetch:

```powershell
# PDFium runtime
.\build\download-pdfium-win-x64.ps1

# FFmpeg LGPL shared runtime (optional)
.\build\download-ffmpeg-lgpl-win-x64.ps1
```

Default download destinations:

- `..\ENGINEbuilds\runtime-deps\pdfium\`
- `..\ENGINEbuilds\runtime-deps\ffmpeg\`

Both scripts now reject repo-internal destinations.

`build_full_stack_cuda.ps1` fetches PDFium automatically, and fetches FFmpeg automatically when `-EnableFfmpeg` is set.
Build source preparation is patch-only: repo `third_party/llama.cpp` snapshot + `0300` patch into `..\ENGINEbuilds\...` (no in-place overwrite of repo sources).

Windows x64 workflow fetch:

- Workflow: `.github/workflows/windows-x64.yml`
- Workflow selection fields:
  - `PDFIUM_RELEASE_TAG: latest`
  - `FFMPEG_RELEASE_API: https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest`
  - `FFMPEG_ASSET_PATTERN: *win64-lgpl-shared*.zip`
- FFmpeg source: `https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest` (asset pattern above)
- PDFium source: `https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest` (`pdfium-win-x64.tgz`)
- Bundle provenance file: `vendor/ffmpeg/ffmpeg-SOURCE.txt` (staged from `third_party/licenses/ffmpeg-SOURCE-windows-x64.txt` with fallback to `ffmpeg-SOURCE.txt`)

Ubuntu x64 workflow fetch:

- Workflow: `.github/workflows/ubuntu-x64.yml`
- Workflow pin fields:
  - `FFMPEG_RELEASE_API: https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest`
  - `FFMPEG_ASSET_NAME: ffmpeg-master-latest-linux64-lgpl-shared.tar.xz`
- FFmpeg source: `https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest`
- FFmpeg asset: `ffmpeg-master-latest-linux64-lgpl-shared.tar.xz`
- PDFium source: `https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest` (`pdfium-linux-x64.tgz`)

macOS arm64 workflow fetch/build:

- Workflow: `.github/workflows/macos-arm64.yml`
- Workflow pin fields:
  - `FFMPEG_TAG: n8.0.1`
  - `FFMPEG_SHA: 894da5ca7d742e4429ffb2af534fcda0103ef593`
- FFmpeg source-build from `https://github.com/FFmpeg/FFmpeg`
- Pinned tag/commit: `n8.0.1` / `894da5ca7d742e4429ffb2af534fcda0103ef593`
- PDFium source: `https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest` (`pdfium-mac-arm64.tgz`)

## Prepare external llama source (repo snapshot + patch)

Canonical prep path is cross-platform Python (`build/prepare_llama_source_from_patch.py`).
PowerShell wrapper (`build/prepare_llama_source_from_patch.ps1`) calls the same script.

```powershell
.\build\prepare_llama_source_from_patch.ps1 -Force
```

```bash
python3 ./build/prepare_llama_source_from_patch.py --force
```

Defaults:

- Prepared source: `..\ENGINEbuilds\sources\llama.cpp\`
- Repo source: `third_party/llama.cpp`
- Patch: `diarize/addons/patches/0300-llama-unified-audio.patch`

## One-command full backend flow (CUDA or Vulkan)

This builds patched llama+bridge for the selected backend, fetches runtime deps, then builds/stages engine bundle.
By default it does not build extra CLI binaries (only `engine.exe` is staged).
For `-Backend cuda`, CUDA runtime DLLs are staged into the bundle root by default.

```powershell
.\build\build_full_stack_cuda.ps1 `
  -Backend cuda `
  -CmakeConfig Release `
  -CargoProfile Release `
  -BuildWhisperCli $false `
  -EnableFfmpeg
```

Vulkan example:

```powershell
.\build\build_full_stack_cuda.ps1 `
  -Backend vulkan `
  -CmakeConfig Release `
  -CargoProfile Release `
  -BuildWhisperCli $false `
  -EnableFfmpeg
```

Optional FFmpeg source override in one-command flow (Windows script path):

```powershell
.\build\build_full_stack_cuda.ps1 `
  -EnableFfmpeg `
  -FfmpegReleaseApiUrl "https://api.github.com/repos/<owner>/<repo>/releases/latest" `
  -FfmpegAssetPattern "*win64-lgpl-shared*.zip"
```

Default outputs:

- `..\ENGINEbuilds\full-stack-<backend>\llama-build\`
- `..\ENGINEbuilds\full-stack-<backend>\cargo-target\`
- `..\ENGINEbuilds\full-stack-<backend>\bundle\`

## Step-by-step flow (manual)

### 1) Build patched llama + bridge (CUDA)

```powershell
.\build\build_bridge.ps1 `
  -Backend cuda `
  -Config Release `
  -BuildDir "..\ENGINEbuilds\llama\cmake-cuda-release" `
  -BuildLlamaServerCli:$false `
  -BuildPyannoteCli:$false `
  -EnableFfmpeg
```

### 2) Build Rust and stage bundle

```powershell
.\build\build_engine.ps1 `
  -Profile Release `
  -CmakeBuildDir "..\ENGINEbuilds\llama\cmake-cuda-release" `
  -CargoTargetDir "..\ENGINEbuilds\cargo-target" `
  -OutDir "..\ENGINEbuilds\bundle-release" `
  -LicenseProfile cuda `
  -StageCmakeRuntime $true `
  -StageFfmpegRuntime $true
```

## Outputs in bundle

- `engine.exe`
- `pdf.dll`
- `pdfvlm.dll`
- `vendor/pdfium/pdfium.dll` (if found)
- `llama-server-bridge.dll` and related llama/ggml runtime DLLs (if found, kept in bundle root)
- `cublas64_*.dll`, `cublasLt64_*.dll`, `cudart64_*.dll` (CUDA backend when CUDA runtime staging is enabled)
- `NVIDIA-CUDA-RUNTIME-NOTICE.txt` in bundle root (CUDA backend) with pointers to official CUDA licensing files under `licenses/third_party/`
- `vendor/ffmpeg/bin/*.dll` runtime files required by bridge audio conversion (if enabled)
- `vendor/pdfium/*` license/notice files copied from PDFium runtime source (with repo fallback files when needed)
- `vendor/ffmpeg/*` license/notice files copied from FFmpeg runtime source (with repo fallback files when needed)
- `LICENSE-ENGINE.txt` (project license)
- `licenses/LICENSES.txt` (key runtime/release license texts combined into one file)
- `licenses/THIRD_PARTY_NOTICES.md` (bundle-level notice index with pointers)
- `licenses/third_party/*` copied from repo `third_party/licenses` top-level curated files
  (tooling-only `torch`/`numpy`/`torchaudio` files are intentionally excluded)
- `licenses/rust-full/*` copied from repo `third_party/licenses/rust-full`

## GitHub Actions: Windows x64 (CUDA or Vulkan)

Workflow: `.github/workflows/windows-x64.yml`

Dispatch input `backend` supports:

- `cuda`
- `vulkan`

Artifacts:

- `engine-win-cuda-x64` (`engine-win-cuda-x64.zip`)
- `engine-win-vulkan-x64` (`engine-win-vulkan-x64.zip`)

Workflow dispatch examples:

```bash
gh workflow run windows-x64.yml --ref main -f backend=cuda
gh workflow run windows-x64.yml --ref main -f backend=vulkan
```

Windows bundle license/provenance notes:

- Key bundle license file is selected by backend profile:
  - CUDA: `third_party/licenses/LICENSES-cuda.txt`
  - Vulkan: `third_party/licenses/LICENSES-vulkan.txt`
  - Fallback: `third_party/licenses/LICENSES.txt`
- FFmpeg runtime files and license/provenance files are staged under `vendor/ffmpeg/`.
- `vendor/ffmpeg/ffmpeg-SOURCE.txt` is required and validated in workflow bundle checks.
- PDFium runtime and license/provenance files are staged under `vendor/pdfium/`.

## GitHub Actions: macOS arm64

Workflow: `.github/workflows/macos-arm64.yml`

Dispatch input `backend` supports:

- `metal`

Bundle artifact:

- `engine-macos-arm64-metal`
- Uploaded file: `engine-macos-arm64-metal.zip`

Workflow dispatch example:

```bash
gh workflow run macos-arm64.yml --ref main -f backend=metal
```

Fetch/build sources for macOS arm64 workflow:

- FFmpeg is built from source at pinned tag/commit (`n8.0.1` / `894da5ca7d742e4429ffb2af534fcda0103ef593`) from `https://github.com/FFmpeg/FFmpeg`.
- PDFium is fetched from `https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest` using asset `pdfium-mac-arm64.tgz` (latest release, not commit-pinned).

Reference FFmpeg build block (must match `.github/workflows/macos-arm64.yml`):

```bash
set -euo pipefail
ffmpeg_src="$BUILD_ROOT/sources/ffmpeg"
ffmpeg_out="$BUILD_ROOT/runtime-deps/ffmpeg"
rm -rf "$ffmpeg_src" "$ffmpeg_out"
mkdir -p "$(dirname "$ffmpeg_src")" "$ffmpeg_out"

git clone --depth 1 --branch "$FFMPEG_TAG" https://github.com/FFmpeg/FFmpeg "$ffmpeg_src"
actual_sha="$(git -C "$ffmpeg_src" rev-parse "HEAD^{commit}")"
if [[ "$actual_sha" != "$FFMPEG_SHA" ]]; then
  echo "Pinned SHA mismatch. Expected $FFMPEG_SHA, got $actual_sha"
  exit 1
fi

pushd "$ffmpeg_src"
./configure \
  --prefix="$ffmpeg_out" \
  --enable-shared \
  --disable-static \
  --disable-gpl \
  --disable-version3 \
  --disable-nonfree \
  --disable-autodetect \
  --disable-xlib \
  --disable-libxcb \
  --disable-libxcb-shm \
  --disable-libxcb-xfixes \
  --disable-libxcb-shape \
  --disable-vulkan \
  --disable-libplacebo \
  --enable-pic \
  --disable-programs \
  --disable-doc \
  --cc=clang \
  --arch=arm64 \
  --target-os=darwin
make -j"$(sysctl -n hw.ncpu)"
make install
popd
```

Bundle layout on macOS arm64:

- `engine`
- `libpdf.dylib`
- `libpdfvlm.dylib`
- `libllama-server-bridge*.dylib` and `libggml*.dylib` in bundle root
- `vendor/pdfium/libpdfium.dylib`
- `vendor/ffmpeg/lib/lib*.dylib`
- `licenses/*`

License staging note for macOS arm64:

- Key bundle file `licenses/LICENSES.txt` is selected from the metal profile (`third_party/licenses/LICENSES-metal.txt`), with fallback to `LICENSES-vulkan.txt` then `LICENSES.txt`.
- PDFium/FFmpeg license files are staged under `vendor/pdfium/*` and `vendor/ffmpeg/*` (no separate `licenses/pdfium` or `licenses/ffmpeg` folders).
- `vendor/ffmpeg/*` uses fallback repo-curated `ffmpeg-LGPL-2.1.txt` and backend-specific source notice files (`ffmpeg-SOURCE-*.txt`, staged as `vendor/ffmpeg/ffmpeg-SOURCE.txt`) when source installs do not ship license files in the install prefix.

## GitHub Actions: Ubuntu x64 Vulkan (CPU + Vulkan)

Workflow: `.github/workflows/ubuntu-x64.yml`

Dispatch behavior:

- No backend input is required.
- Build always includes both CPU and Vulkan support in one bundle.

Bundle artifact:

- `engine-ubuntu-x64-vulkan`
- Uploaded file: `engine-ubuntu-x64-vulkan.tar.gz`

Workflow dispatch example:

```bash
gh workflow run ubuntu-x64.yml --ref main
```

Bundle layout on Ubuntu x64:

- `engine`
- `libpdf.so`
- `libpdfvlm.so`
- `libllama-server-bridge*.so` and `libggml*.so` in bundle root
- `vendor/pdfium/libpdfium.so`
- `vendor/ffmpeg/lib/lib*.so*`
- `licenses/*`

License staging note for Ubuntu x64:

- Key bundle file `licenses/LICENSES.txt` is selected from `third_party/licenses/LICENSES-ubuntu-vulkan.txt`, with fallback to `LICENSES-vulkan.txt` then `LICENSES.txt`.
- PDFium/FFmpeg license files are staged under `vendor/pdfium/*` and `vendor/ffmpeg/*` (no separate `licenses/pdfium` or `licenses/ffmpeg` folders).
- FFmpeg in this workflow is fetched from `https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest` using asset `ffmpeg-master-latest-linux64-lgpl-shared.tar.xz`.

## GitHub Actions: Unified Release

Workflow: `.github/workflows/release-all.yml`

- Trigger is manual only (`workflow_dispatch`).
- Runs all build targets in parallel:
  - Windows x64 CUDA
  - Windows x64 Vulkan
  - macOS arm64 Metal
  - Ubuntu x64 Vulkan
- Creates or updates one GitHub Release page and uploads all assets.
- Runtime note used in release page:
  - Windows/Ubuntu artifacts include CPU runtime support.
  - macOS artifact is presented as Metal-focused (Apple Silicon unified-memory GPU path).
- Versioning:
  - If `tag` input is empty, auto-increments `v1.x` from repo tags (`v1.0`, `v1.1`, ... up to `v1.99`).
  - You can override with a custom tag via workflow input.
