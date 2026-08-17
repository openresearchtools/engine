# Linux x86_64 Debian packages

Linux is distributed as two co-installable packages:

| Artifact | Debian `Package` | Backend | Runtime root |
|---|---|---|---|
| `engine-amd64.deb` | `openresearchtools-engine` | CPU + Vulkan | `/opt/openresearchtools/engine/vulkan` |
| `engine-amd64-cuda.deb` | `openresearchtools-engine-cuda` | CPU + CUDA | `/opt/openresearchtools/engine/cuda` |

The packages do not use `Conflicts` or `Replaces`. Applications can depend on
both packages and map a Vulkan/CUDA selector directly to the two runtime roots.
Installing a package also adds one unique convenience launcher:

- `/usr/bin/openresearchtools-engine-vulkan`
- `/usr/bin/openresearchtools-engine-cuda`

## Runtime contract

Each backend root is self-contained and has the same stable layout:

```text
/opt/openresearchtools/engine/<backend>/
├── engine-runtime.json
├── example-cli
├── libpdf.so
├── libpdfvlm.so
├── libllama-server-bridge.so*
├── libllama-server-audio.so*
├── libmulti-node-server.so*
├── libllama.so*
├── libggml*.so*
└── vendor/
    ├── ffmpeg/lib/
    ├── pdfium/
    └── cuda/                 # CUDA package only
```

`engine-runtime.json` is the installed-machine discovery contract. It records
the backend, Debian package, absolute root, executable, launcher, library
directories, key shared libraries, and capabilities. A client should consider a
backend installed only when this descriptor and executable exist. Shared
libraries should be opened by their absolute paths below the selected root.

All root ELF binaries and shared libraries have RUNPATH entries for `$ORIGIN`,
the FFmpeg directory, PDFium, and the private CUDA runtime directory. Vendor
libraries use `$ORIGIN`. This supports both direct CLI execution and absolute
`dlopen`/`LoadLibrary`-style embedding without a global `LD_LIBRARY_PATH`.

The Vulkan package depends on `libvulkan1` in addition to the common C/C++ and
OpenMP runtimes. The CUDA toolkit runtime libraries needed by ENGINE are staged
privately under `vendor/cuda`; `libcuda.so.1` remains a host NVIDIA driver
responsibility and is checked at runtime. This avoids a dependency on one
distribution's versioned NVIDIA driver package name.

Like the Windows CUDA bundle, the CUDA Debian package includes
`NVIDIA-CUDA-EULA.txt` and `NVIDIA-CUDA-RUNTIME-NOTICE.txt` at the runtime root
and beside the private CUDA libraries. Debian documentation copies are also
installed under `/usr/share/doc/openresearchtools-engine-cuda/`. Both packages
install their project and third-party notices below their own `/usr/share/doc`
directory.

## Container build

Docker Buildx and Podman are both supported. The commands below place all
container logs, intermediate outputs, and final packages under
`../ENGINEbuilds/linux-containers`:

```bash
./build/linux/container_build_debs.sh --version 1.0.0 --backend all
```

Build one backend when iterating:

```bash
./build/linux/container_build_debs.sh --version 1.0.0 --backend vulkan
./build/linux/container_build_debs.sh --version 1.0.0 --backend cuda
```

The final filenames are always `engine-amd64.deb` and `engine-amd64-cuda.deb`. The
version is stored in Debian metadata rather than encoded in the filename, which
makes the artifacts convenient for a stable APT repository path.

## Release policy

`.github/workflows/release-all.yml` builds only the two Linux packages. It does
not rebuild Windows or macOS. Set `carry_forward_release_tag` to an earlier
release containing the existing Windows/macOS runtimes and controller apps, or
leave it empty to select the immediately previous published release. Those
assets are downloaded, byte-checked after release assembly, and uploaded
unchanged alongside the newly built Linux packages.
