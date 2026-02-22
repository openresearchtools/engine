# PDFium Runtime Placeholder

No PDFium binaries are stored in this repository.

Runtime files are fetched during build to:

- `..\ENGINEbuilds\runtime-deps\pdfium\`

Default fetch command:

```powershell
.\build\download-pdfium-win-x64.ps1
```

Default source (Windows x64 build):

- GitHub releases API: `https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest`
- Asset name: `pdfium-win-x64.tgz`

Final bundle behavior:

- `build/build_engine.ps1` copies `pdfium.dll` into the bundle when found.
- `build/build_engine.ps1` also copies PDFium license files into `bundle/vendor/pdfium/`.
