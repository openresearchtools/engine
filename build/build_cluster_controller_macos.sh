#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
builds_root="$(cd "$repo_root/.." && pwd)/ENGINEbuilds"

python_user_bin="$(python3 -c 'import site; print(site.USER_BASE + "/bin")')"
export PATH="$python_user_bin:$PATH"

profile="${PROFILE:-Release}"
build_root="${BUILD_ROOT:-$builds_root/engine-controller-macos-metal}"
llama_src="${LLAMA_SRC:-$build_root/llama-src}"
llama_build="${LLAMA_BUILD:-$build_root/llama-build}"
cargo_target="${CARGO_TARGET_DIR:-$build_root/cargo-target}"
bundle_dir="${BUNDLE_DIR:-$build_root/bundle}"
runtime_dir="${RUNTIME_DIR:-$HOME/Library/Application Support/OpenResearchTools/engine}"
install_runtime="${INSTALL_RUNTIME:-0}"
jobs="${JOBS:-$(sysctl -n hw.logicalcpu)}"
ffmpeg_tag="${FFMPEG_TAG:-n8.0.1}"
ffmpeg_sha="${FFMPEG_SHA:-894da5ca7d742e4429ffb2af534fcda0103ef593}"
pdfium_release_api="${PDFIUM_RELEASE_API:-https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest}"
webrtc_audio_processing_ref="${WEBRTC_AUDIO_PROCESSING_REF:-907015852bc78d8e3ac0e8fbb93c93e76110192a}"

mkdir -p "$build_root" "$cargo_target"

ensure_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "Missing required command: $name" >&2
        exit 1
    fi
}

ensure_python_tool() {
    local bin_name="$1"
    local package_name="$2"
    if command -v "$bin_name" >/dev/null 2>&1; then
        return 0
    fi
    python3 -m pip install --user "$package_name"
    if ! command -v "$bin_name" >/dev/null 2>&1; then
        echo "Failed to install required tool: $bin_name" >&2
        exit 1
    fi
}

copy_license_files() {
    local src_root="$1"
    local dst_root="$2"

    [[ -d "$src_root" ]] || return 0
    mkdir -p "$dst_root"
    while IFS= read -r -d '' file; do
        local rel="${file#$src_root/}"
        mkdir -p "$dst_root/$(dirname "$rel")"
        cp -L "$file" "$dst_root/$rel"
    done < <(find "$src_root" -type f \( \
        -iname '*LICENSE*' -o \
        -iname '*LICENCE*' -o \
        -iname '*COPYING*' -o \
        -iname '*COPYRIGHT*' -o \
        -iname '*NOTICE*' -o \
        -iname '*PATENTS*' -o \
        -iname '*EULA*' -o \
        -iname '*SOURCE*' \
    \) -print0)
}

resolve_bundled_target() {
    local dep_base="$1"
    local bundle_root="$2"
    local ffmpeg_dir="$3"
    local pdfium_dir="$4"

    if [[ -f "$bundle_root/$dep_base" ]]; then
        echo "$bundle_root/$dep_base"
        return 0
    fi
    if [[ -f "$ffmpeg_dir/$dep_base" ]]; then
        echo "$ffmpeg_dir/$dep_base"
        return 0
    fi
    if [[ -f "$pdfium_dir/$dep_base" ]]; then
        echo "$pdfium_dir/$dep_base"
        return 0
    fi
    local dep_stem="${dep_base%.dylib}"
    local dir
    for dir in "$bundle_root" "$ffmpeg_dir" "$pdfium_dir"; do
        [[ -d "$dir" ]] || continue
        while IFS= read -r candidate; do
            [[ -n "$candidate" ]] || continue
            echo "$candidate"
            return 0
        done < <(find "$dir" -maxdepth 1 \( -type f -o -type l \) -name "${dep_stem}*.dylib" -print | LC_ALL=C sort)
    done
    return 1
}

to_loader_path() {
    local from_file="$1"
    local to_file="$2"
    python3 -c 'import os,sys; from_dir=os.path.dirname(os.path.abspath(sys.argv[1])); to_path=os.path.abspath(sys.argv[2]); rel=os.path.relpath(to_path, from_dir).replace("\\\\","/"); print(f"@loader_path/{rel}")' "$from_file" "$to_file"
}

collect_rpaths() {
    local file="$1"
    otool -l "$file" | awk '
        $1 == "cmd" && $2 == "LC_RPATH" { flag = 1; next }
        flag && $1 == "path" { print $2; flag = 0 }
    '
}

is_stale_build_rpath() {
    local value="$1"
    case "$value" in
        "$build_root"/*|"$llama_build"/*|"$cargo_target"/*|"$ffmpeg_out"/*|"$pdfium_root"/*|"$webrtc_root"/*|*"/ENGINEbuilds/"*)
            return 0
            ;;
    esac
    return 1
}

sync_macho_rpaths() {
    local file="$1"
    shift
    local -a desired=("$@")
    local -a current=()
    while IFS= read -r value; do
        [[ -n "$value" ]] || continue
        current+=("$value")
    done < <(collect_rpaths "$file")

    local -a stale=()
    local value
    for value in "${current[@]-}"; do
        if is_stale_build_rpath "$value"; then
            stale+=("$value")
        fi
    done

    local desired_value
    local stale_index=0
    for desired_value in "${desired[@]-}"; do
        local present=0
        for value in "${current[@]-}"; do
            if [[ "$value" == "$desired_value" ]]; then
                present=1
                break
            fi
        done
        if (( present )); then
            continue
        fi
        if (( stale_index < ${#stale[@]} )); then
            install_name_tool -rpath "${stale[$stale_index]}" "$desired_value" "$file"
            stale_index=$((stale_index + 1))
        else
            install_name_tool -add_rpath "$desired_value" "$file"
        fi
        current+=("$desired_value")
    done

    while IFS= read -r value; do
        [[ -n "$value" ]] || continue
        if is_stale_build_rpath "$value"; then
            install_name_tool -delete_rpath "$value" "$file"
        fi
    done < <(collect_rpaths "$file")
}

verify_no_build_rpaths() {
    local macho_files="$1"
    local file
    while IFS= read -r file; do
        [[ -n "$file" ]] || continue
        [[ -f "$file" ]] || continue
        while IFS= read -r value; do
            [[ -n "$value" ]] || continue
            if is_stale_build_rpath "$value"; then
                echo "Bundled Mach-O still contains a build-tree rpath: $file -> $value" >&2
                return 1
            fi
        done < <(collect_rpaths "$file")
    done < "$macho_files"
}

fixup_macho_paths() {
    local bundle_root="$1"
    local ffmpeg_dir="$2"
    local pdfium_dir="$3"
    local macho_files="$build_root/macho-files.txt"
    local dylib_files="$build_root/macho-dylib-files.txt"

    : > "$macho_files"
    : > "$dylib_files"

    echo "$bundle_root/example-cli" >> "$macho_files"
    echo "$bundle_root/Engine" >> "$macho_files"

    find "$bundle_root" -maxdepth 1 -type f -name "*.dylib" -print >> "$macho_files"
    find "$bundle_root" -maxdepth 1 -type f -name "*.dylib" -print >> "$dylib_files"
    find "$ffmpeg_dir" -type f -name "*.dylib" -print >> "$macho_files"
    find "$ffmpeg_dir" -type f -name "*.dylib" -print >> "$dylib_files"
    find "$pdfium_dir" -type f -name "*.dylib" -print >> "$macho_files"
    find "$pdfium_dir" -type f -name "*.dylib" -print >> "$dylib_files"

    sort -u "$macho_files" -o "$macho_files"
    sort -u "$dylib_files" -o "$dylib_files"

    while IFS= read -r file; do
        [[ -n "$file" ]] || continue
        [[ -f "$file" ]] || continue
        local base
        base="$(basename "$file")"
        install_name_tool -id "@loader_path/$base" "$file"
    done < "$dylib_files"

    while IFS= read -r file; do
        [[ -n "$file" ]] || continue
        [[ -f "$file" ]] || continue
        while IFS= read -r dep; do
            [[ -n "$dep" ]] || continue
            case "$dep" in
                /System/*|/usr/lib/*)
                    continue
                    ;;
            esac
            local dep_base
            dep_base="$(basename "$dep")"
            local target
            target="$(resolve_bundled_target "$dep_base" "$bundle_root" "$ffmpeg_dir" "$pdfium_dir" || true)"
            if [[ -n "$target" ]]; then
                local new_dep
                new_dep="$(to_loader_path "$file" "$target")"
                if [[ "$dep" != "$new_dep" ]]; then
                    install_name_tool -change "$dep" "$new_dep" "$file"
                fi
            fi
        done < <(otool -L "$file" | awk 'NR > 1 { print $1 }')

        sync_macho_rpaths "$file" "@loader_path"
    done < "$macho_files"

    verify_no_build_rpaths "$macho_files"
}

ensure_command git
ensure_command clang
ensure_command curl
ensure_command rsync
ensure_command make
ensure_command tar
ensure_python_tool cmake cmake
ensure_python_tool ninja ninja
ensure_python_tool meson meson

ffmpeg_src="$build_root/sources/ffmpeg"
ffmpeg_out="$build_root/runtime-deps/ffmpeg"
pdfium_root="$build_root/runtime-deps/pdfium"
webrtc_root="$build_root/runtime-deps/webrtc-audio-processing"
webrtc_src="$webrtc_root/src"
webrtc_build="$webrtc_root/build"

rm -rf "$ffmpeg_src" "$ffmpeg_out"
mkdir -p "$(dirname "$ffmpeg_src")" "$ffmpeg_out"
git clone --depth 1 --branch "$ffmpeg_tag" https://github.com/FFmpeg/FFmpeg "$ffmpeg_src"
actual_ffmpeg_sha="$(git -C "$ffmpeg_src" rev-parse 'HEAD^{commit}')"
if [[ "$actual_ffmpeg_sha" != "$ffmpeg_sha" ]]; then
    echo "Pinned FFmpeg SHA mismatch. Expected $ffmpeg_sha, got $actual_ffmpeg_sha" >&2
    exit 1
fi

pushd "$ffmpeg_src" >/dev/null
./configure \
    --prefix="$ffmpeg_out" \
    --enable-shared \
    --disable-static \
    --disable-gpl \
    --disable-version3 \
    --disable-nonfree \
    --disable-autodetect \
    --disable-xlib \
    --disable-libxcb \
    --disable-libxcb-shm \
    --disable-libxcb-xfixes \
    --disable-libxcb-shape \
    --disable-vulkan \
    --disable-libplacebo \
    --enable-pic \
    --disable-programs \
    --disable-doc \
    --cc=clang \
    --arch=arm64 \
    --target-os=darwin
make -j"$jobs"
make install
popd >/dev/null

rm -rf "$pdfium_root"
mkdir -p "$pdfium_root"
asset_url="$(PDFIUM_RELEASE_API="$pdfium_release_api" GH_TOKEN="${GH_TOKEN:-}" python3 - <<'PY'
import json
import os
import urllib.request

api = os.environ["PDFIUM_RELEASE_API"]
token = os.environ.get("GH_TOKEN", "")
headers = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "ENGINE-pdfium-fetch",
}
if token:
    headers["Authorization"] = f"Bearer {token}"

req = urllib.request.Request(api, headers=headers)
with urllib.request.urlopen(req) as resp:
    release = json.load(resp)

for asset in release.get("assets", []):
    if asset.get("name") == "pdfium-mac-arm64.tgz":
        print(asset.get("browser_download_url", ""))
        break
PY
)"
if [[ -z "$asset_url" ]]; then
    echo "Could not find asset pdfium-mac-arm64.tgz from $pdfium_release_api" >&2
    exit 1
fi
pdfium_archive="$build_root/pdfium-mac-arm64.tgz"
curl -L --fail "$asset_url" -o "$pdfium_archive"
tar -xzf "$pdfium_archive" -C "$pdfium_root"
pdfium_lib="$(find "$pdfium_root" -type f -name 'libpdfium.dylib' | head -n1 || true)"
if [[ -z "$pdfium_lib" ]]; then
    echo "libpdfium.dylib was not found under $pdfium_root" >&2
    exit 1
fi

rm -rf "$webrtc_root"
mkdir -p "$webrtc_root"
git clone --depth 1 https://github.com/cross-platform/webrtc-audio-processing.git "$webrtc_src"
git -C "$webrtc_src" fetch --depth 1 origin "$webrtc_audio_processing_ref"
git -C "$webrtc_src" checkout --force FETCH_HEAD
meson setup "$webrtc_build" "$webrtc_src" --default-library=static --buildtype=release
meson compile -C "$webrtc_build"

python3 "$repo_root/build/prepare_llama_source_from_patch.py" \
    --repo-root "$repo_root" \
    --out-dir "$llama_src" \
    --force

webrtc_libs=()
for candidate in \
    "$webrtc_root/build/webrtc/modules/audio_processing/libwebrtc_audio_processing.a" \
    "$webrtc_root/build/webrtc/modules/audio_processing/libwebrtc_audio_processing_privatearch.a" \
    "$webrtc_root/build/webrtc/modules/audio_coding/libaudio_coding.a" \
    "$webrtc_root/build/webrtc/common_audio/libcommon_audio.a" \
    "$webrtc_root/build/webrtc/common_audio/libcommon_audio_sse2.a" \
    "$webrtc_root/build/webrtc/system_wrappers/libsystem_wrappers.a" \
    "$webrtc_root/build/webrtc/base/liblibbase.a" \
    "$webrtc_root/build/webrtc/libwebrtc.a"; do
    [[ -f "$candidate" ]] && webrtc_libs+=("$candidate")
done
if [[ "${#webrtc_libs[@]}" -eq 0 ]]; then
    echo "WebRTC AudioProcessing static libraries not found under $webrtc_root" >&2
    exit 1
fi
webrtc_libs_cmake="$(printf '%s;' "${webrtc_libs[@]}")"
webrtc_libs_cmake="${webrtc_libs_cmake%;}"

rm -rf "$llama_build"
mkdir -p "$llama_build"
cmake -S "$llama_src" -B "$llama_build" -G Ninja \
    -DCMAKE_BUILD_TYPE="$profile" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_MARKDOWN_BRIDGE=ON \
    -DLLAMA_HTTPLIB=ON \
    -DGGML_RPC=ON \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_BORINGSSL=OFF \
    -DLLAMA_BUILD_LIBRESSL=OFF \
    -DLLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON \
    -DLLAMA_SERVER_BRIDGE_FFMPEG_ROOT="$ffmpeg_out" \
    -DLLAMA_SERVER_AUDIO_BUILD=ON \
    -DLLAMA_SERVER_AUDIO_ENABLE_WEBRTC=ON \
    -DLLAMA_SERVER_AUDIO_WEBRTC_ROOT="$webrtc_src" \
    -DLLAMA_SERVER_AUDIO_WEBRTC_LIBRARIES="$webrtc_libs_cmake" \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON
cmake --build "$llama_build" --parallel "$jobs" --target llama-server-bridge multi-node-server llama-server-audio

bridge_dir="$(dirname "$(find "$llama_build" -type f -name 'libllama-server-bridge.dylib' | head -n1)")"
if [[ -z "$bridge_dir" ]]; then
    echo "libllama-server-bridge.dylib not found after CMake build" >&2
    exit 1
fi

export LIBRARY_PATH="$bridge_dir:${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$bridge_dir:${DYLD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR="$cargo_target"

cargo build --release --jobs "$jobs" -p pdf -p pdfvlm -p engine -p clusterui

bundle_vendor="$bundle_dir/vendor"
bundle_pdfium="$bundle_vendor/pdfium"
bundle_ffmpeg_root="$bundle_vendor/ffmpeg"
bundle_ffmpeg="$bundle_ffmpeg_root/lib"
bundle_miniaudio="$bundle_vendor/miniaudio"
bundle_webrtc="$bundle_vendor/webrtc-audio-processing"
cargo_release="$cargo_target/release"
llama_lib_dir="$llama_build/bin"

rm -rf "$bundle_dir"
mkdir -p "$bundle_dir" "$bundle_pdfium" "$bundle_ffmpeg" "$bundle_miniaudio" "$bundle_webrtc"

cp "$cargo_release/example-cli" "$bundle_dir/example-cli"
cp "$cargo_release/Engine" "$bundle_dir/Engine"
cp "$cargo_release/libpdf.dylib" "$bundle_dir/libpdf.dylib"
cp "$cargo_release/libpdfvlm.dylib" "$bundle_dir/libpdfvlm.dylib"

while IFS= read -r -d '' lib; do
    cp -a "$lib" "$bundle_dir/$(basename "$lib")"
done < <(find "$llama_lib_dir" \( \
    -name 'libmulti-node-server*.dylib' -o \
    -name 'libllama-server-bridge*.dylib' -o \
    -name 'libllama-server-audio*.dylib' -o \
    -name 'libllama*.dylib' -o \
    -name 'libggml*.dylib' -o \
    -name 'libmtmd*.dylib' \
\) -print0)

cp -L "$pdfium_lib" "$bundle_pdfium/$(basename "$pdfium_lib")"

for name in avcodec avformat avutil swresample swscale; do
    found=0
    while IFS= read -r -d '' src; do
        cp -a "$src" "$bundle_ffmpeg/$(basename "$src")"
        found=1
    done < <(find "$ffmpeg_out/lib" -type f -name "lib${name}*.dylib" -print0)
    if [[ "$found" -eq 0 ]]; then
        echo "Missing FFmpeg runtime dylib for lib${name}" >&2
        exit 1
    fi
done

cp "$repo_root/LICENSE" "$bundle_dir/LICENSE-ENGINE.txt"
cp "$repo_root/third_party/LICENSES.md" "$bundle_dir/LICENSES.md"
cp "$repo_root/third_party/README.md" "$bundle_dir/Third-Party-Notices.md"
cp "$repo_root/third_party/licenses/miniaudio-LICENSE.txt" "$bundle_miniaudio/miniaudio-LICENSE.txt"
cp "$repo_root/third_party/licenses/webrtc-audio-processing-LICENSE.txt" "$bundle_webrtc/webrtc-audio-processing-LICENSE.txt"
cp "$repo_root/third_party/licenses/ffmpeg-LGPL-2.1.txt" "$bundle_ffmpeg_root/ffmpeg-LGPL-2.1.txt"
cp "$repo_root/third_party/licenses/pdfium-LICENSE.txt" "$bundle_pdfium/pdfium-LICENSE.txt"
cp "$repo_root/third_party/licenses/pdfium-binaries-LICENSE.txt" "$bundle_pdfium/pdfium-binaries-LICENSE.txt"

ffmpeg_source_notice="$repo_root/third_party/licenses/ffmpeg-SOURCE-macos-arm64.txt"
if [[ ! -f "$ffmpeg_source_notice" ]]; then
    ffmpeg_source_notice="$repo_root/third_party/licenses/ffmpeg-SOURCE.txt"
fi
if [[ -f "$ffmpeg_source_notice" ]]; then
    cp "$ffmpeg_source_notice" "$bundle_ffmpeg_root/ffmpeg-SOURCE.txt"
fi

copy_license_files "$pdfium_root" "$bundle_pdfium"
copy_license_files "$ffmpeg_out" "$bundle_ffmpeg_root"
copy_license_files "$webrtc_src" "$bundle_webrtc"

fixup_macho_paths "$bundle_dir" "$bundle_ffmpeg" "$bundle_pdfium"

if [[ "$install_runtime" == "1" ]]; then
    mkdir -p "$runtime_dir"
    rsync -a --delete \
        --exclude 'settings.json' \
        --exclude 'cluster-public-api.json' \
        "$bundle_dir/" "$runtime_dir/"
fi

echo "Engine controller build complete."
echo "Bundle:  $bundle_dir"
echo "Runtime: $runtime_dir"
echo "Parallel jobs: $jobs"
