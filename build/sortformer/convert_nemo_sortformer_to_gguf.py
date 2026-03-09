#!/usr/bin/env python3
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml


TOKEN_MAP = {
    "preprocessor": "prep",
    "featurizer": "feat",
    "window": "win",
    "sortformer_modules": "mods",
    "encoder_proj": "ep",
    "first_hidden_to_hidden": "fh2h",
    "single_hidden_to_spks": "sh2s",
    "hidden_to_spks": "h2s",
    "encoder": "enc",
    "pre_encode": "pre",
    "batch_norm": "bn",
    "scale": "sc",
    "shift": "sh",
    "running_mean": "rm",
    "running_var": "rv",
    "depthwise_conv": "dw",
    "pointwise_conv1": "pw1",
    "pointwise_conv2": "pw2",
    "feed_forward1": "ff1",
    "feed_forward2": "ff2",
    "linear1": "l1",
    "linear2": "l2",
    "norm_conv": "nc",
    "norm_feed_forward1": "nff1",
    "norm_feed_forward2": "nff2",
    "norm_out": "no",
    "norm_self_att": "nsa",
    "self_attn": "att",
    "linear_k": "k",
    "linear_q": "q",
    "linear_v": "v",
    "linear_out": "o",
    "linear_pos": "p",
    "pos_bias_u": "pbu",
    "pos_bias_v": "pbv",
    "transformer_encoder": "te",
    "first_sub_layer": "sa",
    "second_sub_layer": "ff",
    "key_net": "k",
    "query_net": "q",
    "value_net": "v",
    "out_projection": "o",
    "layer_norm_1": "ln1",
    "layer_norm_2": "ln2",
    "dense_in": "di",
    "dense_out": "do",
    "weight": "w",
    "bias": "b",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_gguf():
    gguf_py = repo_root() / "third_party" / "llama.cpp" / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    import gguf  # type: ignore

    return gguf


def load_nemo_archive(model_path: Path):
    with tarfile.open(model_path, "r:*") as tf:
        config_text = tf.extractfile("model_config.yaml").read().decode("utf-8")
        ckpt_bytes = tf.extractfile("model_weights.ckpt").read()
    return yaml.safe_load(config_text), ckpt_bytes


def unwrap_state_dict(raw_obj):
    if isinstance(raw_obj, dict) and "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
        return raw_obj["state_dict"]
    if isinstance(raw_obj, dict):
        tensor_items = {k: v for k, v in raw_obj.items() if isinstance(v, torch.Tensor)}
        if tensor_items:
            return tensor_items
    raise RuntimeError("Could not locate a tensor state_dict inside model_weights.ckpt")


def add_scalar(writer, key: str, value):
    if isinstance(value, bool):
        writer.add_bool(key, value)
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if value >= 0:
            writer.add_uint32(key, value if value <= 0xFFFFFFFF else 0xFFFFFFFF)
        else:
            writer.add_int64(key, value)
        return
    if isinstance(value, float):
        writer.add_float32(key, value)
        return
    if isinstance(value, str) and value:
        writer.add_string(key, value)
        return


def add_array(writer, key: str, value):
    if not value:
        return
    if all(isinstance(x, bool) for x in value):
        writer.add_array(key, list(value))
        return
    if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        writer.add_array(key, [int(x) for x in value])
        return
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        writer.add_array(key, [float(x) if isinstance(x, float) else int(x) for x in value])
        return
    if all(isinstance(x, str) for x in value):
        writer.add_array(key, list(value))
        return


def flatten_config(writer, prefix: str, value):
    if isinstance(value, dict):
        for key, child in value.items():
            safe_key = str(key).replace("_target_", "target").replace(" ", "_")
            flatten_config(writer, f"{prefix}.{safe_key}", child)
        return
    if isinstance(value, (list, tuple)):
        add_array(writer, prefix, list(value))
        return
    add_scalar(writer, prefix, value)


def compact_tensor_name(name: str, prefix: str) -> str:
    parts = name.split(".")
    compact = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            compact.append(f"l{parts[i + 1]}")
            i += 2
            continue
        compact.append(TOKEN_MAP.get(part, part))
        i += 1

    if prefix:
        compact.insert(0, prefix)

    result = ".".join(compact)
    if len(result) >= 64:
        raise RuntimeError(f"compact tensor name is still too long ({len(result)}): {result}")
    return result


def top_level_groups(state_dict):
    groups = {}
    total_params = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        numel = int(tensor.numel())
        total_params += numel
        group = name.split(".", 1)[0]
        info = groups.setdefault(group, {"tensor_count": 0, "param_count": 0})
        info["tensor_count"] += 1
        info["param_count"] += numel
    return groups, total_params


def main():
    parser = argparse.ArgumentParser(description="Convert a NeMo Sortformer checkpoint to a compact native GGUF artifact.")
    parser.add_argument("--model", required=True, help="Path to the .nemo checkpoint")
    parser.add_argument("--out", required=True, help="Output GGUF path")
    parser.add_argument("--summary-json", help="Optional output JSON summary")
    parser.add_argument("--name", help="Optional model display name")
    parser.add_argument("--tensor-prefix", default="", help="Optional prefix to add to compact tensor names")
    parser.add_argument("--outtype", choices=["f32", "f16", "bf16"], default="f32", help="Weight storage type for floating tensors")
    args = parser.parse_args()

    gguf = import_gguf()

    model_path = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else None
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    config, ckpt_bytes = load_nemo_archive(model_path)
    raw_ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
    state_dict = unwrap_state_dict(raw_ckpt)

    model_name = args.name or model_path.stem
    writer = gguf.GGUFWriter(path=out_path, arch="sortformer")
    writer.add_name(model_name)
    writer.add_description("Native GGUF conversion of an NVIDIA NeMo Sortformer diarization checkpoint.")
    writer.add_string("sortformer.source.format", "nemo")
    writer.add_string("sortformer.source.path", str(model_path))
    writer.add_string("sortformer.tensor_name_scheme", "compact_v1")
    writer.add_string("sortformer.tensor_prefix", args.tensor_prefix)
    writer.add_string("sortformer.outtype", args.outtype)
    writer.add_tensor_data_layout("PyTorch")

    out_qtype = None
    if args.outtype == "f16":
        out_qtype = gguf.GGMLQuantizationType.F16
    elif args.outtype == "bf16":
        out_qtype = gguf.GGMLQuantizationType.BF16

    flatten_config(writer, "sortformer.config", config)

    groups, total_params = top_level_groups(state_dict)
    original_tensor_count = len([t for t in state_dict.values() if isinstance(t, torch.Tensor)])
    writer.add_uint32("sortformer.original_tensor_count", original_tensor_count)
    writer.add_uint64("sortformer.total_params", total_params)
    for group_name, info in sorted(groups.items()):
        writer.add_uint32(f"sortformer.group.{group_name}.tensor_count", int(info["tensor_count"]))
        writer.add_uint64(f"sortformer.group.{group_name}.param_count", int(info["param_count"]))

    kept = []
    skipped = []
    batch_norm_parts = {}
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            skipped.append({"name": name, "reason": "not_a_tensor"})
            continue
        if not tensor.is_floating_point():
            skipped.append({"name": name, "dtype": str(tensor.dtype).replace("torch.", ""), "reason": "non_floating"})
            continue

        arr = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
        gguf_name = compact_tensor_name(name, args.tensor_prefix)
        stored = np.asarray(arr, dtype=np.float32)
        if out_qtype is not None:
            stored = gguf.quants.quantize(stored, out_qtype)
        writer.add_tensor(gguf_name, stored, raw_dtype=out_qtype)
        kept.append({
            "name": name,
            "gguf_name": gguf_name,
            "shape": list(arr.shape),
            "numel": int(arr.size),
            "dtype": args.outtype,
        })
        if ".batch_norm." in name:
            base, suffix = name.rsplit(".", 1)
            batch_norm_parts.setdefault(base, {})[suffix] = arr

    derived = []
    for base, parts in sorted(batch_norm_parts.items()):
        required = {"weight", "bias", "running_mean", "running_var"}
        if set(parts.keys()) != required:
            missing = sorted(required.difference(parts.keys()))
            raise RuntimeError(f"incomplete batch_norm tensor set for {base}: missing {missing}")

        scale = parts["weight"] / np.sqrt(parts["running_var"] + 1.0e-5)
        shift = parts["bias"] - parts["running_mean"] * scale

        for suffix, arr in (("scale", scale.astype(np.float32, copy=False)), ("shift", shift.astype(np.float32, copy=False))):
            derived_name = f"{base}.{suffix}"
            gguf_name = compact_tensor_name(derived_name, args.tensor_prefix)
            stored = np.asarray(arr, dtype=np.float32)
            if out_qtype is not None:
                stored = gguf.quants.quantize(stored, out_qtype)
            writer.add_tensor(gguf_name, stored, raw_dtype=out_qtype)
            derived.append({
                "name": derived_name,
                "gguf_name": gguf_name,
                "shape": list(arr.shape),
                "numel": int(arr.size),
                "dtype": args.outtype,
                "derived": True,
            })

    all_kept = kept + derived
    writer.add_uint32("sortformer.tensor_count", len(all_kept))
    writer.add_uint32("sortformer.skipped_tensor_count", len(skipped))

    writer.write_header_to_file(path=out_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    summary = {
        "source_model": str(model_path),
        "output_gguf": str(out_path),
        "name": model_name,
        "outtype": args.outtype,
        "tensor_prefix": args.tensor_prefix,
        "tensor_name_scheme": "compact_v1",
        "original_tensor_count": original_tensor_count,
        "kept_tensor_count": len(all_kept),
        "derived_tensor_count": len(derived),
        "skipped_tensor_count": len(skipped),
        "total_params": total_params,
        "top_level_groups": groups,
        "kept_tensors": all_kept,
        "skipped_tensors": skipped,
    }

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output_gguf": str(out_path),
        "summary_json": "" if summary_path is None else str(summary_path),
        "kept_tensor_count": len(all_kept),
        "derived_tensor_count": len(derived),
        "skipped_tensor_count": len(skipped),
        "outtype": args.outtype,
        "tensor_name_scheme": "compact_v1",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
