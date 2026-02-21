# Add-on Layer

This folder contains the whisper+pyannote add-on layer intended to sit on top of a fresh upstream `llama.cpp`.

## Overlay

- `overlay/llama.cpp/` contains only the changed files.
- Apply it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/apply_llama_overlay.ps1 -LlamaRoot .\llama.cpp
```

## Patch (Reference)

- `patches/0001-whisper-pyannote-whispercpp.patch` is a reference diff for review/auditing.
