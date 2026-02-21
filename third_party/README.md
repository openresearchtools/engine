# Third-party layout

This folder stores external source/runtime dependencies used by ENGINE.

## Structure

- [`llama.cpp/`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp): upstream `llama.cpp` subtree in the `main` branch.
- [`whisper.cpp/`](https://github.com/openresearchtools/engine/tree/main/third_party/whisper.cpp): upstream `whisper.cpp` subtree in the `main` branch.
- [`pdfium/`](https://github.com/openresearchtools/engine/tree/main/third_party/pdfium): PDFium runtime files.
  - Windows default runtime path: [`pdfium/bin/pdfium.dll`](https://github.com/openresearchtools/engine/blob/main/third_party/pdfium/bin/pdfium.dll)
- [`licenses/`](https://github.com/openresearchtools/engine/tree/main/third_party/licenses): third-party license manifests and copies.

## Update flow

- `llama.cpp` and `whisper.cpp` are managed together with the subtree updater:
- [`extras/update_upstreams.ps1`](https://github.com/openresearchtools/engine/blob/main/extras/update_upstreams.ps1) pulls latest upstream `master` and applies a squashed commit.

## Notes

- `pdf` requires PDFium.
- `pdfvlm` requires PDFium only for PDF input mode.
- `bridge` features do not require PDFium.
