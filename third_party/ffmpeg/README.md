# FFmpeg Runtime Placeholder

No FFmpeg binaries are stored in this repository.

Runtime files are fetched during build to:

- `..\ENGINEbuilds\runtime-deps\ffmpeg\`

Default fetch command:

```powershell
.\build\download-ffmpeg-lgpl-win-x64.ps1
```

Default source (Windows x64 LGPL shared):

- GitHub releases API: `https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest`
- Asset pattern: `*win64-lgpl-shared*.zip`

macOS arm64 builds require an alternate release source and matching asset pattern, for example:

```powershell
.\build\download-ffmpeg-lgpl-win-x64.ps1 `
  -ReleaseApiUrl "https://api.github.com/repos/<owner>/<repo>/releases/latest" `
  -AssetPattern "*macos*arm64*.zip"
```

Final bundle behavior:

- `build/build_engine.ps1` copies required FFmpeg runtime libraries into the bundle.
- `build/build_engine.ps1` also copies FFmpeg license files into `bundle/licenses/ffmpeg/`.
