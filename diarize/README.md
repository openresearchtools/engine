# Diarize Add-on Pack

This folder is patch-only.

## Source Of Truth

- `addons/patches/0300-llama-unified-audio.patch`
- `addons/patches/0300-llama-unified-audio.meta.txt`

All llama/whisper integration changes are applied through `0300` in repo-source patch build flow.

## Build

Use the main build entrypoint:

```powershell
.\build\build_full_stack_cuda.ps1 -Backend cuda -PrepareLlamaSource $true
```

## Expected Source Layout

- [`third_party/llama.cpp/`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp/)
- [`third_party/whisper.cpp/`](https://github.com/openresearchtools/engine/tree/main/third_party/whisper.cpp/)

## Licenses And Notices

- [`third_party/licenses/README.md`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/README.md)
