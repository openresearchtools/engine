# Diarize add-on pack

This folder contains the native audio patch layer to apply on top of `llama.cpp`.

## Contents

- `addons/patches/0001-whisper-pyannote-whispercpp.patch`
- `addons/overlay/llama.cpp/`
- `scripts/apply_llama_overlay.ps1`
- `scripts/build_unified_audio_stack_cuda.ps1`

## Expected source layout

- [`third_party/llama.cpp/`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp/)
- [`third_party/whisper.cpp/`](https://github.com/openresearchtools/engine/tree/main/third_party/whisper.cpp/)

## Apply overlay

```powershell
powershell -ExecutionPolicy Bypass -File .\diarize\scripts\apply_llama_overlay.ps1
```

If overlay files moved, pass explicit path:

```powershell
powershell -ExecutionPolicy Bypass -File .\diarize\scripts\apply_llama_overlay.ps1 `
  -OverlayRoot ".\diarize\addons\overlay\llama.cpp"
```

## Build unified CUDA audio stack

All build outputs are staged outside the repo by default:

- `..\ENGINEbuilds\audio\llama-cuda-release\`
- `..\ENGINEbuilds\audio\whisper-cuda-release\`

```powershell
powershell -ExecutionPolicy Bypass -File .\diarize\scripts\build_unified_audio_stack_cuda.ps1 `
  -Config Release `
  -ApplyOverlay $true `
  -BuildWhisperCli $true
```

## Licenses and notices

- [`third_party/licenses/README.md`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/README.md)
- [`third_party/licenses/llama.cpp-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/llama.cpp-LICENSE.txt)
- [`third_party/licenses/whisper.cpp-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/whisper.cpp-LICENSE.txt)
- [`third_party/licenses/pyannote.audio-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/pyannote.audio-LICENSE.txt)
- [`third_party/licenses/WeSpeaker-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/WeSpeaker-LICENSE.txt)
- [`third_party/licenses/tooling-full/`](https://github.com/openresearchtools/engine/tree/main/third_party/licenses/tooling-full/)
