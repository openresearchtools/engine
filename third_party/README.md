# Third-party layout

This folder stores external source/runtime dependencies used by ENGINE.

## Structure

- `llama.cpp/`: upstream `llama.cpp` subtree in the `main` branch.
- `whisper.cpp/`: upstream `whisper.cpp` subtree in the `main` branch.
- `pdfium/`: PDFium runtime files.
  - Windows default runtime path: `pdfium/bin/pdfium.dll`
- `licenses/`: third-party license manifests and copies.

## Update flow

- `llama.cpp` and `whisper.cpp` are managed together with the subtree updater:
  - `extras/update_upstreams.ps1` pulls latest upstream `master` and applies a squashed commit.

## Notes

- `pdf` requires PDFium.
- `pdfvlm` requires PDFium only for PDF input mode.
- `bridge` features do not require PDFium.
