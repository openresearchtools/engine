#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_local_gguf_module(repo_root: Path) -> None:
    gguf_py = repo_root / "third_party" / "llama.cpp" / "gguf-py"
    if not gguf_py.is_dir():
        raise FileNotFoundError(f"Local gguf-py package not found: {gguf_py}")
    sys.path.insert(0, str(gguf_py))


def _write_gguf(
    *,
    out_path: Path,
    arch: str,
    kv: dict[str, object],
    tensors: dict[str, np.ndarray],
) -> None:
    from gguf import GGUFWriter  # imported only after local path is injected

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(out_path), arch=arch)
    writer.add_uint32("general.file_type", 0)  # F32 payload
    writer.add_uint32("llama.block_count", 0)

    for key, val in kv.items():
        if isinstance(val, bool):
            writer.add_bool(key, val)
        elif isinstance(val, int):
            writer.add_int32(key, val)
        elif isinstance(val, float):
            writer.add_float32(key, float(val))
        else:
            writer.add_string(key, str(val))

    for name, arr in tensors.items():
        writer.add_tensor(name=name, tensor=arr.astype(np.float32, copy=False))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _convert_plda_npz(plda_npz: Path, out_path: Path) -> None:
    data = np.load(plda_npz)
    required = ["mu", "tr", "psi"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in {plda_npz}")

    mu = np.asarray(data["mu"], dtype=np.float32)
    tr = np.asarray(data["tr"], dtype=np.float32)
    psi = np.asarray(data["psi"], dtype=np.float32)

    if mu.ndim != 1 or psi.ndim != 1:
        raise ValueError("PLDA mu/psi must be rank-1 vectors")
    if tr.ndim != 2:
        raise ValueError("PLDA tr must be rank-2 matrix")
    if tr.shape[0] != tr.shape[1]:
        raise ValueError(f"PLDA tr must be square, got {tr.shape}")
    if mu.shape[0] != tr.shape[0] or psi.shape[0] != tr.shape[0]:
        raise ValueError(
            f"PLDA dimension mismatch: mu={mu.shape}, tr={tr.shape}, psi={psi.shape}"
        )

    _write_gguf(
        out_path=out_path,
        arch="pyannote-plda",
        kv={
            "general.name": "pyannote-community1-plda",
            "pyannote.kind": "plda",
            "pyannote.plda.dimension": int(mu.shape[0]),
            "pyannote.source.format": "npz",
            "pyannote.source.filename": plda_npz.name,
        },
        tensors={
            "pyannote.plda.mu": mu,
            "pyannote.plda.tr": tr,
            "pyannote.plda.psi": psi,
        },
    )


def _convert_xvec_transform_npz(xvec_npz: Path, out_path: Path) -> None:
    data = np.load(xvec_npz)
    required = ["mean1", "mean2", "lda"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in {xvec_npz}")

    mean1 = np.asarray(data["mean1"], dtype=np.float32)
    mean2 = np.asarray(data["mean2"], dtype=np.float32)
    lda = np.asarray(data["lda"], dtype=np.float32)

    if mean1.ndim != 1 or mean2.ndim != 1:
        raise ValueError("xvec means must be rank-1 vectors")
    if lda.ndim != 2:
        raise ValueError("xvec lda must be rank-2 matrix")
    if lda.shape[0] != mean1.shape[0] or lda.shape[1] != mean2.shape[0]:
        raise ValueError(
            f"xvec dimension mismatch: mean1={mean1.shape}, lda={lda.shape}, mean2={mean2.shape}"
        )

    _write_gguf(
        out_path=out_path,
        arch="pyannote-xvec-transform",
        kv={
            "general.name": "pyannote-community1-xvec-transform",
            "pyannote.kind": "xvec_transform",
            "pyannote.xvec.input_dim": int(mean1.shape[0]),
            "pyannote.xvec.output_dim": int(mean2.shape[0]),
            "pyannote.source.format": "npz",
            "pyannote.source.filename": xvec_npz.name,
        },
        tensors={
            "pyannote.xvec_transform.mean1": mean1,
            "pyannote.xvec_transform.mean2": mean2,
            "pyannote.xvec_transform.lda": lda,
        },
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert pyannote Community-1 PLDA/xvec npz files to GGUF."
    )
    p.add_argument(
        "--plda-npz",
        default="hf_models/pyannote-speaker-diarization-community-1/plda/plda.npz",
        help="Path to plda.npz",
    )
    p.add_argument(
        "--xvec-npz",
        default="hf_models/pyannote-speaker-diarization-community-1/plda/xvec_transform.npz",
        help="Path to xvec_transform.npz",
    )
    p.add_argument(
        "--plda-out",
        default="gguf/pyannote_plda_f32.gguf",
        help="Output GGUF path for PLDA tensors",
    )
    p.add_argument(
        "--xvec-out",
        default="gguf/pyannote_xvec_transform_f32.gguf",
        help="Output GGUF path for xvec transform tensors",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    _load_local_gguf_module(repo_root)

    plda_npz = Path(args.plda_npz).expanduser().resolve()
    xvec_npz = Path(args.xvec_npz).expanduser().resolve()
    plda_out = Path(args.plda_out).expanduser().resolve()
    xvec_out = Path(args.xvec_out).expanduser().resolve()

    if not plda_npz.is_file():
        raise FileNotFoundError(f"PLDA npz not found: {plda_npz}")
    if not xvec_npz.is_file():
        raise FileNotFoundError(f"xvec transform npz not found: {xvec_npz}")

    _convert_plda_npz(plda_npz, plda_out)
    _convert_xvec_transform_npz(xvec_npz, xvec_out)

    print(f"Wrote: {plda_out}")
    print(f"Wrote: {xvec_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
