#!/usr/bin/env python3
"""
Prepare an external llama.cpp source tree from repo snapshots + patch.

This script is the canonical cross-platform prep path used by both
Windows and macOS build flows.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_out = repo_root.parent / "ENGINEbuilds" / "sources" / "llama.cpp"

    parser = argparse.ArgumentParser(description="Prepare patched llama.cpp source tree")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-patch", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def is_subpath(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def copy_tree_replace(src: Path, dst: Path) -> None:
    if not src.is_dir():
        fail(f"Source directory not found: {src}")
    if src.resolve() == dst.resolve():
        fail(f"Refusing to copy a directory onto itself: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def overlay_tree(src_root: Path, dst_root: Path) -> None:
    if not src_root.is_dir():
        fail(f"Overlay directory not found: {src_root}")
    for src in sorted(src_root.rglob("*")):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ensure_bridge_hooks(cmake_path: Path) -> None:
    text = read_text(cmake_path)
    updated = text

    if "LLAMA_BUILD_MARKDOWN_BRIDGE" not in updated:
        pattern = r'(option\(LLAMA_BUILD_SERVER\s+"[^"]+"\s+\$\{LLAMA_STANDALONE\}\)\r?\n)'
        replacement = (
            r"\1"
            + 'option(LLAMA_BUILD_MARKDOWN_BRIDGE "llama: build markdown in-process bridge library" OFF)\n'
        )
        updated, count = re.subn(pattern, replacement, updated, count=1, flags=re.MULTILINE)
        if count == 0:
            fail(f"Failed to locate LLAMA_BUILD_SERVER option in {cmake_path}")

    if "add_subdirectory(MARKDOWN/bridge)" not in updated:
        block_pattern = (
            r"(if \(LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS\)\s*"
            r"add_subdirectory\(tools\)\s*endif\(\))"
        )
        block_replacement = (
            r"\1\n"
            r"if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS AND LLAMA_BUILD_MARKDOWN_BRIDGE)\n"
            r"    add_subdirectory(MARKDOWN/bridge)\n"
            r"endif()"
        )
        updated, count = re.subn(block_pattern, block_replacement, updated, count=1, flags=re.DOTALL)
        if count == 0:
            fail(f"Failed to locate tools block in {cmake_path}")

    if updated != text:
        cmake_path.write_text(updated, encoding="utf-8")


def run_git(out_dir: Path, args: list[str]) -> None:
    cmd = ["git", "-C", str(out_dir), *args]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        fail(f"Command failed: {' '.join(cmd)}")
    if result.stdout:
        print(result.stdout, end="")


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    builds_root = repo_root.parent / "ENGINEbuilds"

    if is_subpath(out_dir, repo_root):
        fail(f"OutDir must be outside the repo. Current: {out_dir}")
    if is_subpath(out_dir, repo_root / "third_party"):
        fail(f"OutDir must not target repo third_party. Current: {out_dir}")

    llama_repo = repo_root / "third_party" / "llama.cpp"
    whisper_repo = repo_root / "third_party" / "whisper.cpp"
    bridge_src = repo_root / "bridge"
    overlay_root = repo_root / "diarize" / "addons" / "overlay" / "llama.cpp" / "tools"
    patch_dir = repo_root / "diarize" / "addons" / "patches"
    patch_files = sorted(patch_dir.glob("*.patch"))

    if not llama_repo.is_dir():
        fail(f"Required repo source not found: {llama_repo}")
    if not whisper_repo.is_dir() or not (whisper_repo / "CMakeLists.txt").is_file():
        fallback_whisper_repo = llama_repo / "whisper.cpp"
        if fallback_whisper_repo.is_dir() and (fallback_whisper_repo / "CMakeLists.txt").is_file():
            whisper_repo = fallback_whisper_repo
        else:
            fail(f"Required repo source not found: {whisper_repo}")
    if not bridge_src.is_dir():
        fail(f"Bridge source dir not found: {bridge_src}")
    if not patch_dir.is_dir():
        fail(f"Required patch directory not found: {patch_dir}")
    if not patch_files:
        fail(f"No patch files found in: {patch_dir}")

    llama_cmake = llama_repo / "CMakeLists.txt"
    whisper_cmake = whisper_repo / "CMakeLists.txt"
    if not llama_cmake.is_file():
        fail(f"Missing llama.cpp CMakeLists.txt: {llama_cmake}")
    if not whisper_cmake.is_file():
        fail(f"Missing whisper.cpp CMakeLists.txt: {whisper_cmake}")

    llama_text = read_text(llama_cmake)
    if 'project("whisper.cpp"' in llama_text:
        fail(f"Expected llama.cpp sources at '{llama_repo}' but found whisper.cpp content.")
    if 'project("llama.cpp"' not in llama_text:
        fail(f"Unable to verify llama.cpp source root at '{llama_repo}'.")

    whisper_text = read_text(whisper_cmake)
    if 'project("whisper.cpp"' not in whisper_text:
        fail(f"Unable to verify whisper.cpp source root at '{whisper_repo}'.")

    if out_dir.exists():
        if not args.force:
            fail(f"OutDir already exists: {out_dir}. Re-run with --force to replace it.")
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    copy_tree_replace(llama_repo, out_dir)

    whisper_in_tree = out_dir / "whisper.cpp"
    copy_tree_replace(whisper_repo, whisper_in_tree)

    whisper_sibling = out_dir.parent / "whisper.cpp"
    if whisper_sibling.resolve() == whisper_repo.resolve():
        fail(f"Refusing to stage whisper sibling onto repo source tree: {whisper_sibling}")
    copy_tree_replace(whisper_repo, whisper_sibling)

    bridge_dst = out_dir / "MARKDOWN" / "bridge"
    copy_tree_replace(bridge_src, bridge_dst)

    ensure_bridge_hooks(out_dir / "CMakeLists.txt")

    status_count = 0
    if not args.skip_patch:
        run_git(out_dir, ["init"])
        for patch_file in patch_files:
            run_git(
                out_dir,
                [
                    "apply",
                    "--check",
                    "--ignore-space-change",
                    "--ignore-whitespace",
                    str(patch_file),
                ],
            )
            run_git(
                out_dir,
                [
                    "apply",
                    "--whitespace=nowarn",
                    "--ignore-space-change",
                    "--ignore-whitespace",
                    str(patch_file),
                ],
            )

        status = subprocess.run(
            ["git", "-C", str(out_dir), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        status_count = len([line for line in status.stdout.splitlines() if line.strip()])
        git_dir = out_dir / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

    overlay_tree(overlay_root, out_dir / "tools")

    print("Prepared llama source from repo snapshot:")
    print(f"  Source: {llama_repo}")
    print(f"  OutDir: {out_dir}")
    print(f"  Whisper sibling: {whisper_sibling}")
    print(f"  Bridge source staged at: {bridge_dst}")
    if args.skip_patch:
        print("  Patches: <skipped>")
    else:
        print("  Patches:")
        for patch_file in patch_files:
            print(f"    - {patch_file}")
    print(f"  Working tree entries after patch: {status_count}")
    print(f"  Build root hint: {builds_root}")


if __name__ == "__main__":
    main()
