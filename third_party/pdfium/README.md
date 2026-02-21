PDFium runtime location.

Default expected file on Windows:
- `third_party/pdfium/bin/pdfium.dll`

Build/stage behavior:
- `build/build_engine.ps1` copies this DLL into bundle output as `pdfium.dll` if present.

