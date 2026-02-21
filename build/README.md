# Build Guide

This folder contains fetch/build scripts for `ENGINE`.

## Prerequisites

- CMake available (use full path if not on PATH).
- Rust/Cargo.
- `third_party/llama.cpp` present.
- `third_party/whisper.cpp` present if building diarize/audio overlay path.

## Fetch runtime dependencies

```powershell
# PDFium runtime
.\build\download-pdfium-win-x64.ps1

# FFmpeg LGPL shared runtime (optional, for bridge raw-audio conversion)
.\build\download-ffmpeg-lgpl-win-x64.ps1
```

## Build bridge + llama runtime

```powershell
.\build\build_bridge.ps1 `
  -CmakeExe "C:\full\path\to\cmake.exe" `
  -LlamaCppDir ".\third_party\llama.cpp" `
  -BuildDir ".\out\llama-build"
```

Enable FFmpeg in bridge:

```powershell
.\build\build_bridge.ps1 `
  -CmakeExe "C:\full\path\to\cmake.exe" `
  -LlamaCppDir ".\third_party\llama.cpp" `
  -BuildDir ".\out\llama-build" `
  -EnableFfmpeg `
  -FfmpegRoot ".\third_party\ffmpeg"
```

## Build Rust artifacts and stage bundle

```powershell
.\build\build_engine.ps1 -Profile Release
```

Optional overrides:

```powershell
.\build\build_engine.ps1 `
  -Profile Release `
  -CargoExe "C:\full\path\to\cargo.exe" `
  -PdfiumDll ".\third_party\pdfium\bin\pdfium.dll" `
  -FfmpegBinDir ".\third_party\ffmpeg\bin" `
  -OutDir ".\out\bundle-release"
```

## Output

- `engine.exe`
- `pdf.dll`
- `pdfvlm.dll`
- `pdfium.dll` (if found)
- FFmpeg DLLs from `third_party/ffmpeg/bin` (if found)

Bundle output default:

- `out/bundle-release/` (or `out/bundle-debug/`)
