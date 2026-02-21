#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch


LOG = logging.getLogger("convert_pyannote_checkpoint_to_gguf")


class _PyannoteProblem(Enum):
    MONO_LABEL_CLASSIFICATION = 1
    REPRESENTATION = 3


class _PyannoteResolution(Enum):
    FRAME = 1
    CHUNK = 2


class _PyannoteSpecifications:
    def __init__(self, *args: Any, **kwargs: Any):
        self.__dict__.update(kwargs)
        if args:
            self._args = args


_PyannoteProblem.__module__ = "pyannote.audio.core.task"
_PyannoteProblem.__qualname__ = "Problem"
_PyannoteResolution.__module__ = "pyannote.audio.core.task"
_PyannoteResolution.__qualname__ = "Resolution"
_PyannoteSpecifications.__module__ = "pyannote.audio.core.task"
_PyannoteSpecifications.__qualname__ = "Specifications"


def _load_local_gguf_module(repo_root: Path) -> None:
    gguf_py = repo_root / "third_party" / "llama.cpp" / "gguf-py"
    if not gguf_py.is_dir():
        raise FileNotFoundError(f"Local gguf-py package not found: {gguf_py}")
    sys.path.insert(0, str(gguf_py))


def _load_checkpoint_any(path: Path) -> Any:
    safe_types = [_PyannoteSpecifications, _PyannoteProblem, _PyannoteResolution]

    try:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is not None:
            with safe_globals(safe_types):
                return torch.load(
                    str(path),
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
        torch.serialization.add_safe_globals(safe_types)
        return torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except Exception as exc:
        LOG.debug("weights_only checkpoint load failed for %s: %s", path, exc)

    try:
        return torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert pyannote checkpoint folder (pytorch_model*.bin) to GGUF."
    )
    p.add_argument("model_dir", help="Model directory containing pytorch_model*.bin")
    p.add_argument("--outfile", required=True, help="Output GGUF path")
    p.add_argument(
        "--outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "auto"],
        help="Tensor output type. bf16 and auto are currently exported as f16.",
    )
    return p.parse_args()


def _resolve_ckpt_path(model_dir: Path) -> Path:
    parts = sorted(model_dir.glob("pytorch_model*.bin"))
    if not parts:
        raise FileNotFoundError(f"No pytorch_model*.bin found in {model_dir}")
    if len(parts) > 1:
        LOG.warning(
            "Multiple pytorch_model*.bin parts found; using first part '%s'",
            parts[0].name,
        )
    return parts[0]


def _get_kind(arch_class: str | None) -> str:
    key = (arch_class or "").lower()
    if key == "pyannet":
        return "segmentation"
    if key == "wespeakerresnet34":
        return "embedding"
    return "unknown"


def _as_numpy(t: torch.Tensor, outtype: str) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        if outtype == "f32":
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float16)
    return arr


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    repo_root = Path(__file__).resolve().parents[1]
    _load_local_gguf_module(repo_root)
    import gguf  # type: ignore
    from gguf import GGUFWriter  # type: ignore

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_path = Path(args.outfile).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    outtype = args.outtype
    if outtype in {"bf16", "auto"}:
        LOG.warning("outtype=%s currently exports tensors as f16 for pyannote stubs.", outtype)
        tensor_outtype = "f16"
    else:
        tensor_outtype = outtype

    ckpt_path = _resolve_ckpt_path(model_dir)
    LOG.info("loading checkpoint: %s", ckpt_path)
    checkpoint = _load_checkpoint_any(ckpt_path)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")

    pyannote_info = checkpoint.get("pyannote.audio")
    pyannote_arch: str | None = None
    pyannote_module: str | None = None
    pyannote_specs: Any = None
    if isinstance(pyannote_info, dict):
        arch_info = pyannote_info.get("architecture")
        if isinstance(arch_info, dict):
            cls_name = arch_info.get("class")
            mod_name = arch_info.get("module")
            pyannote_arch = cls_name if isinstance(cls_name, str) else None
            pyannote_module = mod_name if isinstance(mod_name, str) else None
        pyannote_specs = pyannote_info.get("specifications")

    pyannote_kind = _get_kind(pyannote_arch)
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        hyper_parameters = {}

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Could not locate a state_dict in {ckpt_path}")

    tensors: dict[str, np.ndarray] = {}
    prefix = "pyannote.embedding" if pyannote_kind == "embedding" else "pyannote.segmentation"
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if name.endswith(".num_batches_tracked"):
            continue
        tensors[f"{prefix}.{name}"] = _as_numpy(value, tensor_outtype)
    if not tensors:
        raise ValueError(f"No tensors found in pyannote checkpoint {ckpt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(out_path), arch="wavtokenizer-dec")
    file_type = gguf.LlamaFileType.ALL_F32 if tensor_outtype == "f32" else gguf.LlamaFileType.MOSTLY_F16
    writer.add_uint32("general.file_type", int(file_type))
    writer.add_uint32("llama.block_count", 0)
    writer.add_string("pyannote.kind", pyannote_kind)
    writer.add_string("pyannote.architecture.class", pyannote_arch or "unknown")
    writer.add_string("pyannote.architecture.module", pyannote_module or "unknown")
    writer.add_string("pyannote.source.format", "pytorch_model.bin")
    writer.add_string("pyannote.source.filename", ckpt_path.name)

    sample_rate = hyper_parameters.get("sample_rate")
    if sample_rate is not None:
        writer.add_uint32("pyannote.sample_rate", int(sample_rate))
    num_channels = hyper_parameters.get("num_channels")
    if num_channels is not None:
        writer.add_uint32("pyannote.num_channels", int(num_channels))

    if pyannote_specs is not None:
        problem = getattr(pyannote_specs, "problem", None)
        if problem is not None:
            writer.add_string("pyannote.spec.problem", str(getattr(problem, "name", problem)))
        resolution = getattr(pyannote_specs, "resolution", None)
        if resolution is not None:
            writer.add_string("pyannote.spec.resolution", str(getattr(resolution, "name", resolution)))
        duration = getattr(pyannote_specs, "duration", None)
        if duration is not None:
            writer.add_float32("pyannote.spec.duration_sec", float(duration))
        min_duration = getattr(pyannote_specs, "min_duration", None)
        if min_duration is not None:
            writer.add_float32("pyannote.spec.min_duration_sec", float(min_duration))
        powerset_max = getattr(pyannote_specs, "powerset_max_classes", None)
        if powerset_max is not None:
            writer.add_uint32("pyannote.spec.powerset_max_classes", int(powerset_max))
        perm_inv = getattr(pyannote_specs, "permutation_invariant", None)
        if perm_inv is not None:
            writer.add_bool("pyannote.spec.permutation_invariant", bool(perm_inv))
        warm_up = getattr(pyannote_specs, "warm_up", None)
        if isinstance(warm_up, (tuple, list)) and len(warm_up) == 2:
            writer.add_float32("pyannote.spec.warm_up_left_sec", float(warm_up[0]))
            writer.add_float32("pyannote.spec.warm_up_right_sec", float(warm_up[1]))
        classes = getattr(pyannote_specs, "classes", None)
        if isinstance(classes, (tuple, list)):
            writer.add_uint32("pyannote.spec.num_classes", int(len(classes)))

    for name in sorted(tensors.keys()):
        writer.add_tensor(name=name, tensor=tensors[name])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    LOG.info("wrote %s (%d tensors)", out_path, len(tensors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
