#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builds_root="$(cd "$repo_root/.." && pwd)/ENGINEbuilds"
bundle_dir=""
cargo_target=""
target_triple=""
locked=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      bundle_dir="${2:-}"
      shift 2
      ;;
    --cargo-target-dir)
      cargo_target="${2:-}"
      shift 2
      ;;
    --target)
      target_triple="${2:-}"
      shift 2
      ;;
    --locked)
      locked=1
      shift
      ;;
    --no-locked)
      locked=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$bundle_dir" ]]; then
  bundle_dir="$builds_root/engine-controller-macos-arm64-bundle"
fi
if [[ -z "$cargo_target" ]]; then
  cargo_target="$builds_root/engine-controller-macos-arm64-target"
fi

bundle_dir="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$bundle_dir")"
cargo_target="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$cargo_target")"

repo_root_prefix="${repo_root%/}/"
if [[ "$bundle_dir" == "$repo_root" || "$bundle_dir" == "$repo_root_prefix"* ]]; then
  echo "BundleDir must be outside the repo: $bundle_dir" >&2
  exit 1
fi
if [[ "$cargo_target" == "$repo_root" || "$cargo_target" == "$repo_root_prefix"* ]]; then
  echo "CargoTargetDir must be outside the repo: $cargo_target" >&2
  exit 1
fi

rm -rf "$bundle_dir"
mkdir -p "$bundle_dir" "$cargo_target"

export CARGO_TARGET_DIR="$cargo_target"
build_args=(build --release -p clusterui --bin Engine)
if [[ "$locked" -eq 1 ]]; then
  build_args+=(--locked)
fi
if [[ -n "$target_triple" ]]; then
  build_args+=(--target "$target_triple")
fi

(
  cd "$repo_root"
  cargo "${build_args[@]}"
)

if [[ -n "$target_triple" ]]; then
  target_release="$cargo_target/$target_triple/release"
else
  target_release="$cargo_target/release"
fi

main_bin="$target_release/Engine"
[[ -f "$main_bin" ]] || { echo "Missing built controller binary: $main_bin" >&2; exit 1; }

app_name="Engine"
app_dir="$bundle_dir/${app_name}.app"
contents_dir="$app_dir/Contents"
macos_dir="$contents_dir/MacOS"
resources_dir="$contents_dir/Resources"
mkdir -p "$macos_dir" "$resources_dir"

cp "$main_bin" "$macos_dir/$app_name"
chmod +x "$macos_dir/$app_name" || true

icon_source_png="$repo_root/clusterui/assets/engine.png"
icon_name="Engine"
iconset_dir="$bundle_dir/${icon_name}.iconset"
icns_path="$resources_dir/${icon_name}.icns"

generate_icns() {
  local src_png="$1"
  local out_icns="$2"
  local out_iconset="$3"

  [[ -f "$src_png" ]] || {
    echo "Missing icon source PNG: $src_png" >&2
    return 1
  }
  command -v sips >/dev/null 2>&1 || {
    echo "Missing required command: sips" >&2
    return 1
  }
  command -v iconutil >/dev/null 2>&1 || {
    echo "Missing required command: iconutil" >&2
    return 1
  }

  rm -rf "$out_iconset"
  mkdir -p "$out_iconset"

  make_icon() {
    local pixels="$1"
    local name="$2"
    sips -z "$pixels" "$pixels" "$src_png" --out "$out_iconset/$name" >/dev/null
  }

  make_icon 16 "icon_16x16.png"
  make_icon 32 "icon_16x16@2x.png"
  make_icon 32 "icon_32x32.png"
  make_icon 64 "icon_32x32@2x.png"
  make_icon 128 "icon_128x128.png"
  make_icon 256 "icon_128x128@2x.png"
  make_icon 256 "icon_256x256.png"
  make_icon 512 "icon_256x256@2x.png"
  make_icon 512 "icon_512x512.png"
  make_icon 1024 "icon_512x512@2x.png"

  iconutil -c icns "$out_iconset" -o "$out_icns"
}

generate_icns "$icon_source_png" "$icns_path" "$iconset_dir"

app_version="$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$repo_root/clusterui/Cargo.toml")"
if [[ -z "$app_version" ]]; then
  app_version="0.0.0"
fi

cat > "$contents_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>${app_name}</string>
  <key>CFBundleDisplayName</key>
  <string>${app_name}</string>
  <key>CFBundleIdentifier</key>
  <string>com.openresearchtools.engine</string>
  <key>CFBundleVersion</key>
  <string>${app_version}</string>
  <key>CFBundleShortVersionString</key>
  <string>${app_version}</string>
  <key>CFBundleExecutable</key>
  <string>${app_name}</string>
  <key>CFBundleIconFile</key>
  <string>${icon_name}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$app_dir" >/dev/null 2>&1 || true
fi

dmg_name="Engine.dmg"
dmg_path="$bundle_dir/$dmg_name"
dmg_staging="$bundle_dir/.dmg-staging"
rm -rf "$dmg_staging" "$dmg_path"
mkdir -p "$dmg_staging"
cp -R "$app_dir" "$dmg_staging/"
ln -s /Applications "$dmg_staging/Applications"
if [[ -f "$icns_path" ]]; then
  cp "$icns_path" "$dmg_staging/.VolumeIcon.icns"
  if command -v SetFile >/dev/null 2>&1; then
    SetFile -a C "$dmg_staging" || true
  fi
fi

if command -v hdiutil >/dev/null 2>&1; then
  hdiutil create -volname "$app_name" -srcfolder "$dmg_staging" -ov -format UDZO "$dmg_path" >/dev/null
else
  echo "Warning: hdiutil not found; skipping DMG creation." >&2
fi

rm -rf "$iconset_dir"
rm -rf "$dmg_staging"
echo "Standalone controller ready: $dmg_path"
