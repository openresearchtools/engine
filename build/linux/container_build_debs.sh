#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: container_build_debs.sh --version <debian-version> [--backend <vulkan|cuda|all>]

Builds engine-amd64.deb and/or engine-amd64-cuda.deb with Docker Buildx or Podman.
Podman storage, logs, and final outputs are kept below
../ENGINEbuilds/linux-containers.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

version=""
backend="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) version="${2:-}"; shift 2 ;;
        --backend) backend="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ -n "$version" ]] || die "--version is required"
[[ "$backend" == "vulkan" || "$backend" == "cuda" || "$backend" == "all" ]] || die "invalid backend: $backend"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
builds_root="$(cd "$repo_root/.." && pwd -P)/ENGINEbuilds"
container_root="$builds_root/linux-containers"
output_root="$container_root/output"
podman_tmp="$container_root/tmp"
podman_config="$container_root/xdg-config"
mkdir -p "$output_root" "$podman_tmp" "$podman_config"

# Keep container build scratch/configuration off the user's system drive. Podman
# already receives explicit root/runroot paths below; these cover the remaining
# temporary context and client configuration files.
export TMPDIR="$podman_tmp"
export XDG_CONFIG_HOME="$podman_config"

if command -v docker >/dev/null && docker buildx version >/dev/null 2>&1; then
    builder=docker
elif command -v podman >/dev/null; then
    builder=podman
else
    die "Docker Buildx or Podman is required"
fi

if [[ "$backend" == "all" ]]; then
    backends=(vulkan cuda)
else
    backends=("$backend")
fi

for selected in "${backends[@]}"; do
    destination="$output_root/$selected"
    rm -rf "$destination"
    mkdir -p "$destination"
    tag="openresearchtools-engine-linux-$selected:$version"
    common_args=(
        --platform linux/amd64
        --target artifact
        --build-arg "BACKEND=$selected"
        --build-arg "PACKAGE_VERSION=$version"
        --tag "$tag"
        --file "$repo_root/build/linux/engine-linux-x64.Dockerfile"
    )
    if [[ "$builder" == docker ]]; then
        docker buildx build \
            --progress plain \
            --output "type=local,dest=$destination" \
            "${common_args[@]}" \
            "$repo_root"
    else
        podman_root="$container_root/podman-storage"
        podman_runroot="$container_root/podman-runroot"
        mkdir -p "$podman_root" "$podman_runroot"
        podman --root "$podman_root" --runroot "$podman_runroot" build \
            --layers \
            --logfile "$container_root/$selected.log" \
            --output "type=local,dest=$destination" \
            "${common_args[@]}" \
            "$repo_root"
    fi
done

for artifact in "$output_root"/*/*.deb; do
    [[ -f "$artifact" ]] || continue
    cp "$artifact" "$output_root/$(basename "$artifact")"
done
sha256sum "$output_root"/*.deb > "$output_root/SHA256SUMS.txt"
echo "Packages: $output_root"
