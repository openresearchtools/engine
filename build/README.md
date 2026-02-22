# Build Guide

All build/test outputs must stay outside this repo, under `..\ENGINEbuilds`.

## Prerequisites

- CMake on PATH (or pass `-CmakeExe`).
- Rust/Cargo on PATH (or pass `-CargoExe`).
- `third_party/llama.cpp` must be actual `llama.cpp` sources.
- `third_party/whisper.cpp` required at build time for the in-process audio integration.

## Fetch runtime dependencies (manual)

```powershell
# PDFium runtime
.\build\download-pdfium-win-x64.ps1

# FFmpeg LGPL shared runtime (optional)
.\build\download-ffmpeg-lgpl-win-x64.ps1

# FFmpeg macOS arm64 example (set alternative release feed + asset pattern)
.\build\download-ffmpeg-lgpl-win-x64.ps1 `
  -ReleaseApiUrl "https://api.github.com/repos/<owner>/<repo>/releases/latest" `
  -AssetPattern "*macos*arm64*.zip"
```

Default download destinations:

- `..\ENGINEbuilds\runtime-deps\pdfium\`
- `..\ENGINEbuilds\runtime-deps\ffmpeg\`

Both scripts now reject repo-internal destinations.

`build_full_stack_cuda.ps1` fetches PDFium automatically, and fetches FFmpeg automatically when `-EnableFfmpeg` is set.
Build source preparation is patch-only: repo `third_party/llama.cpp` snapshot + `0300` patch into `..\ENGINEbuilds\...` (no in-place overwrite of repo sources).

## Prepare external llama source (repo snapshot + patch)

```powershell
.\build\prepare_llama_source_from_patch.ps1 -Force
```

Defaults:

- Prepared source: `..\ENGINEbuilds\sources\llama.cpp\`
- Repo source: `third_party/llama.cpp`
- Patch: `diarize/addons/patches/0300-llama-unified-audio.patch`

## One-command full backend flow (CUDA or Vulkan)

This builds patched llama+bridge for the selected backend, fetches runtime deps, then builds/stages engine bundle.
By default it does not build extra CLI binaries (only `engine.exe` is staged).

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

Optional FFmpeg source override in one-command flow:

```powershell
.\build\build_full_stack_cuda.ps1 `
  -EnableFfmpeg `
  -FfmpegReleaseApiUrl "https://api.github.com/repos/<owner>/<repo>/releases/latest" `
  -FfmpegAssetPattern "*macos*arm64*.zip"
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
- `pdfium.dll` (if found)
- `llama-server-bridge.dll` and related llama/ggml runtime DLLs (if found)
- FFmpeg runtime DLLs required by bridge audio conversion (if enabled)
- `LICENSES.txt` (key runtime/release license texts combined into one file)
- `THIRD_PARTY_NOTICES.md` (bundle-level notice index with pointers)
- `LICENSE-ENGINE.txt` (project license)
- `licenses/third_party/*` copied from repo `third_party/licenses` top-level curated files
  (tooling-only `torch`/`numpy`/`torchaudio` files are intentionally excluded)
- `licenses/pdfium/*` copied from fetched PDFium runtime
- `licenses/ffmpeg/*` copied from fetched FFmpeg runtime (when FFmpeg staging is enabled)
