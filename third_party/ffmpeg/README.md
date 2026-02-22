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
- Workflow reference: `.github/workflows/windows-x64.yml`

Ubuntu x64 source (workflow fetch, LGPL shared):

- GitHub releases API: `https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest`
- Asset name: `ffmpeg-master-latest-linux64-lgpl-shared.tar.xz`
- Workflow reference: `.github/workflows/ubuntu-x64.yml`

macOS arm64 source (workflow build, LGPL shared):

- Source repository: `https://github.com/FFmpeg/FFmpeg`
- Pinned tag: `n8.0.1`
- Pinned commit: `894da5ca7d742e4429ffb2af534fcda0103ef593`
- Workflow reference: `.github/workflows/macos-arm64.yml`

Final bundle behavior:

- `build/build_engine.ps1` copies required FFmpeg runtime libraries into the bundle.
- `build/build_engine.ps1` also copies FFmpeg license files into `bundle/vendor/ffmpeg/`.
- Bundles stage backend-specific FFmpeg provenance as `bundle/vendor/ffmpeg/ffmpeg-SOURCE.txt`.
