#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_engine_deb.sh --backend <vulkan|cuda> --version <debian-version> [--build-root <dir>]

Builds the patched ENGINE Linux x86_64 runtime and packages it as a .deb.
Every generated file is placed below the sibling ../ENGINEbuilds directory.
Run this directly in a prepared Ubuntu 24.04 build environment, or use
container_build_debs.sh for the supported container build.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

is_under() {
    case "$1/" in
        "$2"/*) return 0 ;;
        *) return 1 ;;
    esac
}

safe_reset_dir() {
    local target="$1"
    is_under "$target" "$builds_root" || die "refusing to reset path outside $builds_root: $target"
    rm -rf "$target"
    mkdir -p "$target"
}

resolve_release_asset() {
    local api="$1"
    local exact_name="$2"
    local fallback_contains="$3"
    python3 - "$api" "$exact_name" "$fallback_contains" <<'PY'
import json
import os
import sys
import urllib.request

api, exact_name, fallback_contains = sys.argv[1:]
headers = {"Accept": "application/vnd.github+json", "User-Agent": "OpenResearchTools-ENGINE-linux-build"}
token = os.environ.get("GH_TOKEN", "")
if token:
    headers["Authorization"] = f"Bearer {token}"
with urllib.request.urlopen(urllib.request.Request(api, headers=headers)) as response:
    release = json.load(response)
assets = release.get("assets", [])
for asset in assets:
    if asset.get("name") == exact_name:
        print(asset["browser_download_url"])
        raise SystemExit(0)
for asset in assets:
    if fallback_contains and fallback_contains in asset.get("name", ""):
        print(asset["browser_download_url"])
        raise SystemExit(0)
raise SystemExit(f"asset {exact_name!r} not found at {api}")
PY
}

copy_license_files() {
    local source_root="$1"
    local destination_root="$2"
    mkdir -p "$destination_root"
    while IFS= read -r -d '' source; do
        local relative="${source#"$source_root"/}"
        mkdir -p "$destination_root/$(dirname "$relative")"
        cp -L "$source" "$destination_root/$relative"
    done < <(find "$source_root" -type f \( \
        -iname '*LICENSE*' -o -iname '*LICENCE*' -o -iname '*COPYING*' -o \
        -iname '*COPYRIGHT*' -o -iname '*NOTICE*' -o -iname '*PATENTS*' -o \
        -iname '*EULA*' -o -iname '*SOURCE*' \) -print0)
}

backend=""
version=""
build_root=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="${2:-}"; shift 2 ;;
        --version) version="${2:-}"; shift 2 ;;
        --build-root) build_root="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$backend" == "vulkan" || "$backend" == "cuda" ]] || die "--backend must be vulkan or cuda"
[[ -n "$version" ]] || die "--version is required"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
builds_root="$(cd "$repo_root/.." && pwd -P)/ENGINEbuilds"
mkdir -p "$builds_root"
builds_root="$(cd "$builds_root" && pwd -P)"
if [[ -z "$build_root" ]]; then
    build_root="$builds_root/linux-$backend"
fi
build_root="$(realpath -m "$build_root")"
is_under "$build_root" "$builds_root" || die "build root must be under $builds_root"
mkdir -p "$build_root"
build_root="$(cd "$build_root" && pwd -P)"

for command in cmake ninja cargo python3 git curl tar patchelf dpkg-deb; do
    command -v "$command" >/dev/null || die "missing build command: $command"
done
if [[ "$backend" == "cuda" ]]; then
    command -v nvcc >/dev/null || die "CUDA build requested but nvcc is unavailable"
    [[ -s "$repo_root/third_party/licenses/nvidia-cuda-EULA.txt" ]] \
        || die "missing checked-in NVIDIA CUDA EULA"
    [[ -s "$repo_root/third_party/licenses/nvidia-cuda-runtime-NOTICE.txt" ]] \
        || die "missing checked-in NVIDIA CUDA runtime notice"
fi

sources_root="$build_root/sources"
deps_root="$build_root/runtime-deps"
llama_src="$sources_root/llama.cpp"
llama_build="$build_root/llama-build"
cargo_target="$build_root/cargo-target"
bundle="$build_root/bundle"
downloads="$build_root/downloads"
packages="$build_root/packages"
mkdir -p "$sources_root" "$deps_root" "$downloads" "$packages"

echo "Preparing patched llama.cpp source outside the repository"
python3 "$repo_root/build/prepare_llama_source_from_patch.py" \
    --repo-root "$repo_root" \
    --out-dir "$llama_src" \
    --force

ffmpeg_root="$deps_root/ffmpeg"
safe_reset_dir "$ffmpeg_root"
ffmpeg_url="$(resolve_release_asset \
    'https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest' \
    'ffmpeg-master-latest-linux64-lgpl-shared.tar.xz' \
    'linux64-lgpl-shared')"
ffmpeg_archive="$downloads/ffmpeg-linux64-lgpl-shared.tar.xz"
curl --retry 5 --fail --location "$ffmpeg_url" --output "$ffmpeg_archive"
tar -xJf "$ffmpeg_archive" -C "$ffmpeg_root" --strip-components=1
[[ -d "$ffmpeg_root/include" && -d "$ffmpeg_root/lib" ]] || die "invalid FFmpeg archive"

pdfium_root="$deps_root/pdfium"
safe_reset_dir "$pdfium_root"
pdfium_url="$(resolve_release_asset \
    'https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest' \
    'pdfium-linux-x64.tgz' \
    'pdfium-linux-x64')"
pdfium_archive="$downloads/pdfium-linux-x64.tgz"
curl --retry 5 --fail --location "$pdfium_url" --output "$pdfium_archive"
tar -xzf "$pdfium_archive" -C "$pdfium_root"
pdfium_lib="$(find "$pdfium_root" -type f -name 'libpdfium.so' -print -quit)"
[[ -n "$pdfium_lib" ]] || die "libpdfium.so not found after extraction"

webrtc_root="$deps_root/webrtc-audio-processing"
safe_reset_dir "$webrtc_root"
webrtc_src="$webrtc_root/src"
webrtc_build="$webrtc_root/build"
git clone --filter=blob:none https://github.com/cross-platform/webrtc-audio-processing.git "$webrtc_src"
git -C "$webrtc_src" checkout --detach 907015852bc78d8e3ac0e8fbb93c93e76110192a
meson setup "$webrtc_build" "$webrtc_src" --default-library=static --buildtype=release
meson compile -C "$webrtc_build"

webrtc_libs=()
while IFS= read -r -d '' library; do
    webrtc_libs+=("$library")
done < <(find "$webrtc_build" -type f -name '*.a' -print0)
[[ ${#webrtc_libs[@]} -gt 0 ]] || die "WebRTC AudioProcessing produced no static libraries"
webrtc_libraries="$(IFS=';'; echo "${webrtc_libs[*]}")"

safe_reset_dir "$llama_build"
cmake_args=(
    -S "$llama_src"
    -B "$llama_build"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    '-DCMAKE_INSTALL_RPATH=$ORIGIN'
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_MARKDOWN_BRIDGE=ON
    -DLLAMA_HTTPLIB=ON
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_BUILD_BORINGSSL=OFF
    -DLLAMA_BUILD_LIBRESSL=OFF
    -DLLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON
    -DLLAMA_SERVER_BRIDGE_FFMPEG_ROOT="$ffmpeg_root"
    -DLLAMA_SERVER_AUDIO_BUILD=ON
    -DLLAMA_SERVER_AUDIO_ENABLE_WEBRTC=ON
    -DLLAMA_SERVER_AUDIO_WEBRTC_ROOT="$webrtc_src"
    -DLLAMA_SERVER_AUDIO_WEBRTC_LIBRARIES="$webrtc_libraries"
    -DGGML_BACKEND_DL=ON
    -DGGML_CPU_ALL_VARIANTS=ON
    -DGGML_NATIVE=OFF
)
if [[ "$backend" == "vulkan" ]]; then
    cmake_args+=(-DGGML_VULKAN=ON)
else
    cmake_args+=(
        -DGGML_CUDA=ON
        -DCMAKE_CUDA_HOST_COMPILER="${CUDAHOSTCXX:-$(command -v g++)}"
    )
    if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
        cmake_args+=(-DCUDAToolkit_ROOT="$CUDAToolkit_ROOT")
    fi
fi
cmake "${cmake_args[@]}"
cmake --build "$llama_build" --parallel "${BUILD_JOBS:-$(nproc)}" \
    --target llama-server-bridge multi-node-server llama-server-audio

bridge_lib="$(find "$llama_build" -type f -name 'libllama-server-bridge.so*' -print -quit)"
[[ -n "$bridge_lib" ]] || die "bridge shared library was not built"
export LIBRARY_PATH="$(dirname "$bridge_lib"):${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$(dirname "$bridge_lib"):${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR="$cargo_target"
cargo build --release --manifest-path "$repo_root/Cargo.toml" -p pdf -p pdfvlm -p engine

safe_reset_dir "$bundle"
mkdir -p \
    "$bundle/vendor/pdfium" \
    "$bundle/vendor/ffmpeg/lib" \
    "$bundle/vendor/miniaudio" \
    "$bundle/vendor/webrtc-audio-processing"
cp "$cargo_target/release/example-cli" "$bundle/example-cli"
cp "$cargo_target/release/libpdf.so" "$bundle/libpdf.so"
cp "$cargo_target/release/libpdfvlm.so" "$bundle/libpdfvlm.so"

while IFS= read -r -d '' library; do
    cp -a "$library" "$bundle/$(basename "$library")"
done < <(find "$llama_build" \( -type f -o -type l \) \( \
    -name 'libmulti-node-server*.so*' -o \
    -name 'libllama-server-bridge*.so*' -o \
    -name 'libllama-server-audio*.so*' -o \
    -name 'libllama*.so*' -o \
    -name 'libggml*.so*' -o \
    -name 'libmtmd*.so*' \
\) -print0)

cp -L "$pdfium_lib" "$bundle/vendor/pdfium/libpdfium.so"
for component in avcodec avformat avutil swresample swscale; do
    found=0
    while IFS= read -r -d '' library; do
        cp -a "$library" "$bundle/vendor/ffmpeg/lib/$(basename "$library")"
        found=1
    done < <(find "$ffmpeg_root/lib" -maxdepth 1 \( -type f -o -type l \) -name "lib${component}*.so*" -print0)
    [[ $found -eq 1 ]] || die "FFmpeg runtime library lib$component was not found"
done

if [[ "$backend" == "cuda" ]]; then
    cuda_vendor="$bundle/vendor/cuda"
    mkdir -p "$cuda_vendor"
    cuda_search_roots=(
        "${CUDAToolkit_ROOT:-/usr/local/cuda}/lib64"
        "${CUDAToolkit_ROOT:-/usr/local/cuda}/targets/x86_64-linux/lib"
    )
    for component in cudart cublas cublasLt; do
        found=0
        for cuda_lib_root in "${cuda_search_roots[@]}"; do
            [[ -d "$cuda_lib_root" ]] || continue
            while IFS= read -r -d '' library; do
                cp -a "$library" "$cuda_vendor/$(basename "$library")"
                found=1
            done < <(find "$cuda_lib_root" -maxdepth 1 \( -type f -o -type l \) -name "lib${component}.so*" -print0)
        done
        [[ $found -eq 1 ]] || die "CUDA runtime library lib$component was not found"
    done
    cuda_eula="$repo_root/third_party/licenses/nvidia-cuda-EULA.txt"
    cuda_notice="$repo_root/third_party/licenses/nvidia-cuda-runtime-NOTICE.txt"
    [[ -s "$cuda_eula" ]] || die "missing NVIDIA CUDA EULA: $cuda_eula"
    [[ -s "$cuda_notice" ]] || die "missing NVIDIA CUDA runtime notice: $cuda_notice"
    # Match the Windows CUDA bundle contract: keep the redistributable terms at
    # runtime root as well as beside the privately shipped CUDA libraries.
    cp "$cuda_eula" "$bundle/NVIDIA-CUDA-EULA.txt"
    cp "$cuda_notice" "$bundle/NVIDIA-CUDA-RUNTIME-NOTICE.txt"
    cp "$cuda_eula" "$cuda_vendor/NVIDIA-CUDA-EULA.txt"
    cp "$cuda_notice" "$cuda_vendor/NVIDIA-CUDA-RUNTIME-NOTICE.txt"
fi

cp "$repo_root/LICENSE" "$bundle/LICENSE-ENGINE.txt"
cp "$repo_root/third_party/LICENSES.md" "$bundle/LICENSES.md"
cp "$repo_root/third_party/README.md" "$bundle/Third-Party-Notices.md"
copy_license_files "$pdfium_root" "$bundle/vendor/pdfium"
copy_license_files "$ffmpeg_root" "$bundle/vendor/ffmpeg"
copy_license_files "$webrtc_src" "$bundle/vendor/webrtc-audio-processing"
cp "$repo_root/third_party/licenses/pdfium-LICENSE.txt" "$bundle/vendor/pdfium/pdfium-LICENSE.txt"
cp "$repo_root/third_party/licenses/pdfium-binaries-LICENSE.txt" "$bundle/vendor/pdfium/pdfium-binaries-LICENSE.txt"
cp "$repo_root/third_party/licenses/ffmpeg-LGPL-2.1.txt" "$bundle/vendor/ffmpeg/ffmpeg-LGPL-2.1.txt"
cp "$repo_root/third_party/licenses/ffmpeg-SOURCE-ubuntu-x64.txt" "$bundle/vendor/ffmpeg/ffmpeg-SOURCE.txt"
cp "$repo_root/third_party/licenses/miniaudio-LICENSE.txt" "$bundle/vendor/miniaudio/miniaudio-LICENSE.txt"
cp "$repo_root/third_party/licenses/webrtc-audio-processing-LICENSE.txt" "$bundle/vendor/webrtc-audio-processing/webrtc-audio-processing-LICENSE.txt"

is_elf() {
    file -b "$1" | grep -q ELF
}

root_runpath='$ORIGIN:$ORIGIN/vendor/ffmpeg/lib:$ORIGIN/vendor/pdfium:$ORIGIN/vendor/cuda'
while IFS= read -r -d '' file_path; do
    [[ -L "$file_path" ]] && continue
    is_elf "$file_path" || continue
    patchelf --set-rpath "$root_runpath" "$file_path"
done < <(find "$bundle" -maxdepth 1 -type f -print0)
for vendor_dir in "$bundle/vendor/ffmpeg/lib" "$bundle/vendor/pdfium" "$bundle/vendor/cuda"; do
    [[ -d "$vendor_dir" ]] || continue
    while IFS= read -r -d '' file_path; do
        [[ -L "$file_path" ]] && continue
        is_elf "$file_path" || continue
        patchelf --set-rpath '$ORIGIN' "$file_path"
    done < <(find "$vendor_dir" -type f -print0)
done

while IFS= read -r -d '' file_path; do
    [[ -L "$file_path" ]] && continue
    is_elf "$file_path" || continue
    # Cargo needs the llama build directory while linking above, but package
    # validation must resolve against the staged bundle exclusively. Otherwise
    # LD_LIBRARY_PATH can select the temporary bridge library and incorrectly
    # report its vendored FFmpeg dependencies as missing.
    bundle_library_path="$bundle:$bundle/vendor/ffmpeg/lib:$bundle/vendor/pdfium"
    if [[ "$backend" == "cuda" ]]; then
        bundle_library_path="$bundle_library_path:$bundle/vendor/cuda"
    fi
    ldd_output="$(LD_LIBRARY_PATH="$bundle_library_path" ldd "$file_path" || true)"
    unresolved="$(grep -i 'not found' <<<"$ldd_output" || true)"
    if [[ "$backend" == "cuda" ]]; then
        # libcuda is supplied by the host NVIDIA driver, not the redistributable
        # CUDA toolkit. It is the only unresolved soname allowed in a CUDA image.
        unresolved="$(grep -vE '^[[:space:]]*libcuda\.so(\.1)?[[:space:]]+=>[[:space:]]+not found' <<<"$unresolved" || true)"
    fi
    if [[ -n "$unresolved" ]]; then
        echo "$ldd_output" >&2
        die "unresolved shared library dependency in $file_path"
    fi
done < <(find "$bundle" -type f -print0)

"$repo_root/build/linux/package_engine_deb.sh" \
    --backend "$backend" \
    --version "$version" \
    --bundle "$bundle" \
    --output-dir "$packages"
