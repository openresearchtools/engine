# Linux ARM64 Vulkan maintenance release

Run **Release Linux ARM64 Vulkan** (`linux-arm64-vulkan-release.yml`) with a new
tag and an existing source release (initially `v1.16`). It builds only the ARM64
CPU + Vulkan package on a native `ubuntu-24.04-arm` runner inside Ubuntu 24.04.
Windows, macOS, and Linux amd64 CUDA/Vulkan binaries are copied byte-for-byte from
the source release, checked against its manifest/GitHub digests, and published
with a fresh manifest and checksums. Existing workflows remain available unchanged.

The workflow defaults to a draft release so the Actions artifact can be tested
on ARM64 hardware before publication. Disable `draft` to publish immediately.
Use a new tag for each release; existing releases are never overwritten.

The shared Linux build/package scripts select native FFmpeg and PDFium assets
and Debian architecture. ARM64 uses a portable ARMv8 CPU fallback, with Vulkan
enabled; it does not require the build runner's SVE/SME CPU features. No engine
application code or vendored llama.cpp/whisper.cpp sources are changed.

All generated output, including container staging, downloads and local tests,
belongs in the sibling `../ENGINEbuilds` directory.

For a local container build on ARM64:

```sh
docker buildx build --platform linux/arm64 --target artifact \
  --build-arg PACKAGE_VERSION=1.17 \
  --output type=local,dest=../ENGINEbuilds/linux-arm64-vulkan \
  --file build/linux/engine-linux-arm64-vulkan.Dockerfile .
```

Install `engine-arm64.deb` with `sudo apt install ./engine-arm64.deb`, or extract
it with `dpkg-deb -x` into `../ENGINEbuilds` for testing without installation.
The executable is `opt/openresearchtools/engine/vulkan/example-cli` in the
extracted tree. Run `list-devices` first; pass `--gpu` explicitly for GPU tests.
