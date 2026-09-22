#!/usr/bin/env python3
"""Add an ARM64 Vulkan package to byte-identical assets from an older release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for tag in (args.tag, args.source_tag):
        if not re.fullmatch(r"v[0-9]+\.[0-9]+(?:\.[0-9]+)?", tag):
            parser.error(f"invalid release tag: {tag}")
    if args.tag == args.source_tag:
        parser.error("source and destination releases must differ")
    builds_root = Path(__file__).resolve().parents[3] / "ENGINEbuilds"
    root = args.output_dir.resolve()
    if not root.is_relative_to(builds_root):
        parser.error(f"output directory must be under {builds_root}")
    repo = os.environ["GH_REPO"]
    source = json.loads(gh("api", f"repos/{repo}/releases/tags/{args.source_tag}"))
    if source["draft"]:
        parser.error("source release must be published")
    assets = root / "assets"
    carried = root / "carried"
    for path in (assets, carried):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    def download(name):
        if Path(name).name != name:
            raise ValueError(f"invalid asset name: {name}")
        gh("release", "download", args.source_tag, "--repo", repo,
           "--pattern", name, "--dir", str(carried))
        return carried / name

    old_manifest = json.loads(download("engine-manifest.json").read_text())
    old_entries = {entry["file_name"]: entry for entry in old_manifest["assets"]}
    entries = []
    copied_names = []
    for asset in source["assets"]:
        name = asset["name"]
        keep = (
            name in ("engine-amd64.deb", "engine-amd64-cuda.deb")
            or (name.endswith(".zip") and
                ("windows-x64" in name or "macos-arm64" in name))
            or name.endswith((".exe", ".dmg"))
        )
        if not keep:
            continue
        path = download(name)
        digest = sha256(path)
        if path.stat().st_size != asset["size"]:
            raise ValueError(f"incomplete download: {name}")
        expected = asset.get("digest")
        if expected and expected != f"sha256:{digest}":
            raise ValueError(f"GitHub asset digest mismatch: {name}")
        previous = old_entries.get(name)
        if previous:
            if previous["sha256"] != digest:
                raise ValueError(f"source manifest digest mismatch: {name}")
            entry = dict(previous)
            entry.update(url=f"https://github.com/{repo}/releases/download/{args.tag}/{name}",
                         carried_forward_from=args.source_tag)
            entries.append(entry)
        shutil.copy2(path, assets / name)
        if sha256(assets / name) != digest:
            raise ValueError(f"asset changed during copy: {name}")
        copied_names.append(name)

    required = {"linux-x64-vulkan", "linux-x64-cuda", "windows-x64-vulkan",
                "windows-x64-cuda", "macos-arm64-metal"}
    if not required.issubset({entry["id"] for entry in entries}):
        raise ValueError("source release is missing required runtime binaries/manifest entries")
    if not any(name.endswith(".dmg") for name in copied_names):
        raise ValueError("source release is missing the macOS controller disk image")

    package = args.package.resolve()
    for field, expected in (("Package", "openresearchtools-engine"),
                            ("Architecture", "arm64"), ("Version", args.tag[1:])):
        actual = subprocess.check_output(["dpkg-deb", "-f", str(package), field], text=True).strip()
        if actual != expected:
            raise ValueError(f"new package {field}: expected {expected}, got {actual}")
    shutil.copy2(package, assets / "engine-arm64.deb")
    entries.append({
        "id": "linux-arm64-vulkan", "platform": "linux-arm64", "backend": "vulkan",
        "archive": "deb", "file_name": "engine-arm64.deb",
        "url": f"https://github.com/{repo}/releases/download/{args.tag}/engine-arm64.deb",
        "sha256": sha256(assets / "engine-arm64.deb"),
    })
    manifest = {
        "schema_version": 1, "project": "Openresearchtools-Engine", "repository": repo,
        "tag": args.tag, "carried_forward_from": args.source_tag,
        "built_platforms": ["linux-arm64-vulkan"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_url": f"https://github.com/{repo}/releases/tag/{args.tag}",
        "assets": sorted(entries, key=lambda entry: entry["id"]),
    }
    (assets / "engine-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = [f"{sha256(path)}  {path.name}\n" for path in sorted(assets.iterdir())]
    (assets / "SHA256SUMS.txt").write_text("".join(checksums))
    base = f"https://github.com/{repo}/releases/download/{args.tag}"
    notes = f"""# Openresearchtools-Engine {args.tag}

Adds Linux ARM64 CPU + Vulkan support as `engine-arm64.deb`. No engine application
code changes. Windows, macOS, and Linux amd64 CUDA/Vulkan binaries are unchanged,
checksum-verified copies from `{args.source_tag}`; their embedded versions remain unchanged.

The new ARM64 build includes the narrowly scoped Vulkan subgroup-size correction
from [upstream llama.cpp #27726](https://github.com/ggml-org/llama.cpp/pull/27726),
applied through the existing build patch. This fixes incorrect matrix multiplication
on Adreno GPUs reporting 128-thread subgroups without updating the frozen engine.

## Linux ARM64 installation

For Ubuntu 24.04+ / Debian 13+ on ARM64, download
[`engine-arm64.deb`]({base}/engine-arm64.deb), then run:

```sh
sudo apt install ./engine-arm64.deb
openresearchtools-engine-vulkan list-devices
```

Runtime root: `/opt/openresearchtools/engine/vulkan`. Vulkan acceleration needs a
working host Vulkan driver. CPU fallback is included. There is no ARM64 CUDA build.

## Unchanged binaries from {args.source_tag}

"""
    notes += "".join(f"- [`{name}`]({base}/{name})\n" for name in sorted(copied_names))
    notes += "\n`engine-manifest.json` includes the new `linux-arm64-vulkan` runtime.\n"
    notes += "`SHA256SUMS.txt` covers all binaries and the manifest.\n"
    (root / "release-notes.md").write_text(notes)
    print(f"Prepared {len(copied_names)} unchanged assets plus ARM64 Vulkan in {assets}")


if __name__ == "__main__":
    main()
