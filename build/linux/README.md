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
enabled; it does not require the build runner's SVE/SME CPU features. Engine
application code and vendored llama.cpp/whisper.cpp sources remain unchanged.

The existing `0300-llama-unified-audio.patch` carries the small Vulkan tile-size
fix from upstream [llama.cpp #27726](https://github.com/ggml-org/llama.cpp/pull/27726)
(commit `5e6a37cb115dc1074e274ac004373f5661909695`), adapted to the frozen snapshot.
Without it, the Adreno X1-45's 128-thread subgroups produce incorrect matrix
multiplication results and garbled Whisper/chat output. Only medium/large tiles
are clamped to 64; small tiles retain their original subgroup size. The patch
is applied by the existing source preparation script outside the repository.

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
