# Third-party layout

This folder stores external source/runtime dependencies used by ENGINE.

## Structure

- `llama.cpp/`: symlink or checkout of upstream llama.cpp source.
- `pdfium/`: PDFium runtime files.
  - Windows default runtime path: `pdfium/bin/pdfium.dll`
- `licenses/`: third-party license manifests and copies.

## Notes

- `pdf` requires PDFium.
- `pdfvlm` requires PDFium only for PDF input mode.
- `bridge` features do not require PDFium.
