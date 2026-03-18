#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


ALLOWED_METHODS = {
    zipfile.ZIP_STORED: "stored",
    zipfile.ZIP_DEFLATED: "deflated",
}

KNOWN_METHOD_NAMES = {
    zipfile.ZIP_STORED: "stored",
    zipfile.ZIP_DEFLATED: "deflated",
    getattr(zipfile, "ZIP_BZIP2", 12): "bzip2",
    getattr(zipfile, "ZIP_LZMA", 14): "lzma",
}


def method_name(method: int) -> str:
    return KNOWN_METHOD_NAMES.get(method, f"unknown({method})")


def verify_zip(path: Path) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path, "r") as archive:
        methods = sorted({info.compress_type for info in archive.infolist()})
        print(
            f"{path}: methods={', '.join(method_name(method) for method in methods) or '<empty>'}"
        )
        for info in archive.infolist():
            if info.compress_type not in ALLOWED_METHODS:
                issues.append(
                    f"{path}: entry '{info.filename}' uses unsupported zip method "
                    f"{method_name(info.compress_type)}"
                )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail if a runtime release zip uses compression methods outside stored/deflated."
    )
    parser.add_argument("zip_paths", nargs="+", help="Zip files to inspect")
    args = parser.parse_args()

    issues: list[str] = []
    for raw_path in args.zip_paths:
        path = Path(raw_path)
        if not path.is_file():
            issues.append(f"{path}: file not found")
            continue
        issues.extend(verify_zip(path))

    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
