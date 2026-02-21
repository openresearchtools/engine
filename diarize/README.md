# Diarize add-on pack

This folder contains the native audio patch layer to apply on top of `llama.cpp`.

## Contents

- `addons/patches/0001-whisper-pyannote-whispercpp.patch`
- `addons/overlay/llama.cpp/`
- `scripts/apply_llama_overlay.ps1`
- `scripts/build_unified_audio_stack_cuda.ps1`

## Expected source layout

- `../third_party/llama.cpp/`
- `../third_party/whisper.cpp/`

## Apply overlay

```powershell
powershell -ExecutionPolicy Bypass -File .\diarize\scripts\apply_llama_overlay.ps1
```

## Licenses and notices

- `../third_party/licenses/README.md`
- `../third_party/licenses/llama.cpp-LICENSE.txt`
- `../third_party/licenses/whisper.cpp-LICENSE.txt`
- `../third_party/licenses/pyannote.audio-LICENSE.txt`
- `../third_party/licenses/WeSpeaker-LICENSE.txt`
- `../third_party/licenses/tooling-full/`
