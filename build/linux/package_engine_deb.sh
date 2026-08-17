#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: package_engine_deb.sh --backend <vulkan|cuda> --version <debian-version> --bundle <dir> [--output-dir <dir>]

Packages an already staged Linux x86_64 ENGINE bundle. The package staging tree
and resulting .deb are always written below the repository sibling ENGINEbuilds.
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

backend=""
version=""
bundle=""
output_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="${2:-}"; shift 2 ;;
        --version) version="${2:-}"; shift 2 ;;
        --bundle) bundle="${2:-}"; shift 2 ;;
        --output-dir) output_dir="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$backend" == "vulkan" || "$backend" == "cuda" ]] || die "--backend must be vulkan or cuda"
[[ -n "$version" ]] || die "--version is required"
[[ "$version" =~ ^[0-9][0-9A-Za-z.+:~-]*$ ]] || die "invalid Debian version: $version"
[[ -n "$bundle" ]] || die "--bundle is required"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
builds_root="$(cd "$repo_root/.." && pwd -P)/ENGINEbuilds"
mkdir -p "$builds_root"
builds_root="$(cd "$builds_root" && pwd -P)"
bundle="$(cd "$bundle" && pwd -P)"
if [[ -z "$output_dir" ]]; then
    output_dir="$builds_root/linux-$backend/packages"
fi
output_dir="$(realpath -m "$output_dir")"
is_under "$output_dir" "$builds_root" || die "output directory must be under $builds_root (got $output_dir)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd -P)"

is_under "$bundle" "$builds_root" || die "bundle must be under $builds_root (got $bundle)"

required=(
    example-cli
    libpdf.so
    libpdfvlm.so
    LICENSE-ENGINE.txt
    LICENSES.md
    Third-Party-Notices.md
    vendor/pdfium/libpdfium.so
)
for relative in "${required[@]}"; do
    [[ -f "$bundle/$relative" ]] || die "bundle is missing $relative"
done
[[ -x "$bundle/example-cli" ]] || die "bundle example-cli is not executable"
compgen -G "$bundle/libllama-server-bridge.so*" >/dev/null || die "bundle is missing libllama-server-bridge.so*"
compgen -G "$bundle/libllama-server-audio.so*" >/dev/null || die "bundle is missing libllama-server-audio.so*"
compgen -G "$bundle/libmulti-node-server.so*" >/dev/null || die "bundle is missing libmulti-node-server.so*"
# GGML_CPU_ALL_VARIANTS produces dispatchable modules such as
# libggml-cpu-x64.so and libggml-cpu-haswell.so instead of a generic module.
compgen -G "$bundle/libggml-cpu*.so*" >/dev/null || die "bundle is missing libggml-cpu*.so*"
compgen -G "$bundle/libggml-$backend.so*" >/dev/null || die "bundle is missing libggml-$backend.so*"

if [[ "$backend" == "vulkan" ]]; then
    package="openresearchtools-engine"
    artifact="engine-amd64.deb"
    depends="libc6, libstdc++6, libgcc-s1, libgomp1, libvulkan1"
    [[ ! -e "$bundle/vendor/cuda" ]] || die "Vulkan bundle unexpectedly contains vendor/cuda"
    ! compgen -G "$bundle/libggml-cuda.so*" >/dev/null \
        || die "Vulkan bundle unexpectedly contains libggml-cuda.so"
    ! compgen -G "$bundle/NVIDIA-CUDA-*" >/dev/null \
        || die "Vulkan bundle unexpectedly contains NVIDIA CUDA notices"
else
    package="openresearchtools-engine-cuda"
    artifact="engine-amd64-cuda.deb"
    depends="libc6, libstdc++6, libgcc-s1, libgomp1"
    [[ -d "$bundle/vendor/cuda" ]] || die "CUDA bundle is missing vendor/cuda"
    ! compgen -G "$bundle/libggml-vulkan.so*" >/dev/null \
        || die "CUDA bundle unexpectedly contains libggml-vulkan.so"
    for notice in NVIDIA-CUDA-EULA.txt NVIDIA-CUDA-RUNTIME-NOTICE.txt; do
        [[ -f "$bundle/$notice" ]] || die "CUDA bundle is missing root notice $notice"
        [[ -f "$bundle/vendor/cuda/$notice" ]] || die "CUDA bundle is missing vendor/cuda/$notice"
    done
    for component in cudart cublas cublasLt; do
        compgen -G "$bundle/vendor/cuda/lib${component}.so*" >/dev/null \
            || die "CUDA bundle is missing private lib${component}.so runtime"
    done
fi

package_root="$output_dir/${package}_${version}_amd64"
install_root="$package_root/opt/openresearchtools/engine/$backend"
launcher="$package_root/usr/bin/openresearchtools-engine-$backend"
doc_root="$package_root/usr/share/doc/$package"
rm -rf "$package_root"
mkdir -p "$install_root" "$package_root/DEBIAN" "$package_root/usr/bin" "$doc_root"
cp -a "$bundle/." "$install_root/"
cp "$bundle/LICENSE-ENGINE.txt" "$doc_root/copyright"
cp "$bundle/LICENSES.md" "$doc_root/LICENSES.md"
cp "$bundle/Third-Party-Notices.md" "$doc_root/Third-Party-Notices.md"
if [[ "$backend" == "cuda" ]]; then
    cp "$bundle/NVIDIA-CUDA-EULA.txt" "$doc_root/NVIDIA-CUDA-EULA.txt"
    cp "$bundle/NVIDIA-CUDA-RUNTIME-NOTICE.txt" "$doc_root/NVIDIA-CUDA-RUNTIME-NOTICE.txt"
fi

cat > "$install_root/engine-runtime.json" <<EOF
{
  "schema_version": 1,
  "backend": "$backend",
  "package": "$package",
  "package_version": "$version",
  "architecture": "amd64",
  "root": "/opt/openresearchtools/engine/$backend",
  "executable": "example-cli",
  "launcher": "/usr/bin/openresearchtools-engine-$backend",
  "library_directories": [".", "vendor/ffmpeg/lib", "vendor/pdfium"$(if [[ "$backend" == "cuda" ]]; then printf ', "vendor/cuda"'; fi)],
  "libraries": {
    "bridge": "libllama-server-bridge.so",
    "audio": "libllama-server-audio.so",
    "multi_node": "libmulti-node-server.so",
    "pdf": "libpdf.so",
    "pdf_vlm": "libpdfvlm.so"
  },
  "capabilities": ["cpu", "$backend", "audio", "pdf", "pdf_vlm", "embeddings", "reranking", "vlm"]
}
EOF

cat > "$launcher" <<EOF
#!/bin/sh
exec /opt/openresearchtools/engine/$backend/example-cli "\$@"
EOF
chmod 0755 "$launcher" "$install_root/example-cli"

installed_kib="$(du -sk "$install_root" | awk '{print $1}')"
cat > "$package_root/DEBIAN/control" <<EOF
Package: $package
Version: $version
Section: science
Priority: optional
Architecture: amd64
Depends: $depends
Installed-Size: $installed_kib
Maintainer: Open Research Tools <openresearchtools@users.noreply.github.com>
Homepage: https://github.com/openresearchtools/engine
Description: Open Research Tools embeddable ENGINE ($backend backend)
 CPU plus $backend runtime for local transcription, PDF processing, VLM,
 embeddings, reranking, and llama.cpp inference. This package installs into a
 backend-specific /opt path and can coexist with the other ENGINE backend.
EOF

find "$package_root" -type d -exec chmod 0755 {} +
find "$package_root" -type f -exec chmod 0644 {} +
chmod 0755 "$launcher" "$install_root/example-cli"
dpkg-deb --build --root-owner-group "$package_root" "$output_dir/$artifact"
dpkg-deb --info "$output_dir/$artifact" >/dev/null
dpkg-deb --contents "$output_dir/$artifact" \
    | grep -F "./opt/openresearchtools/engine/$backend/engine-runtime.json" >/dev/null
[[ "$(dpkg-deb --field "$output_dir/$artifact" Package)" == "$package" ]] \
    || die "packaged Debian identity does not match $package"
[[ "$(dpkg-deb --field "$output_dir/$artifact" Architecture)" == "amd64" ]] \
    || die "packaged Debian architecture is not amd64"
echo "$output_dir/$artifact"
