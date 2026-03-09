# Common Runtime And Devices

This document is the shared reference for `llama-server-bridge.dll`.

## Simplest setup (recommended start)

1. Enumerate devices once.
2. Pick one device index (for example `gpu=1`).
3. Create one bridge instance with `llama_server_bridge_default_params()`.
4. Reuse that bridge for many requests.

```c
#include "llama_server_bridge.h"

llama_server_bridge_device_info *devices = NULL;
size_t device_count = 0;
if (llama_server_bridge_list_devices(&devices, &device_count) != 0) {
    // handle error
}

for (size_t i = 0; i < device_count; ++i) {
    // devices[i].index, devices[i].backend, devices[i].name, devices[i].description
}

llama_server_bridge_free_devices(devices, device_count);

llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/model.gguf";
p.gpu = 1; // choose a single enumerated device index

llama_server_bridge *bridge = llama_server_bridge_create(&p);
if (!bridge) {
    // create failed
}

// ... run requests ...

llama_server_bridge_destroy(bridge);
```

## Device routing defaults

These are current runtime rules:

- `gpu` and `devices` are mutually exclusive. Do not set both.
- If `gpu >= 0`, runtime uses single-device routing by that enumerated index.
- If `gpu` is set and `split_mode` was unset, runtime defaults to `split_mode=none`.
- If `gpu` is not set:
- Windows/Linux default is CPU-only (`devices=none`).
- macOS default is first available GPU.
- KV defaults are GPU-friendly:
- `kv_unified = 1` by default.
- `no_kv_offload = 0` by default (KV offload enabled where supported).
- For VLM, `mmproj_use_gpu = -1` means auto. With a selected GPU, mmproj follows the selected GPU by default.

## One-GPU vs split guidance

- Prefer one selected GPU (`gpu=N`) whenever model + KV fit in that GPU memory.
- Split (`split_mode=layer` or `split_mode=row`) is for fit-first, not speed-first.
- No split happens unless you explicitly pass split settings.

## Full bridge create parameters

All fields in `llama_server_bridge_params`:

- `model_path`: text model path. Required for chat/VLM/embeddings/rerank. Optional for audio-only mode.
- `mmproj_path`: vision projector path. Required for VLM workloads.
- `n_ctx`: default `32768`.
- `n_batch`: default `2048`.
- `n_ubatch`: default `2048`.
- `n_parallel`: default `1`.
- `n_threads`: default `8`.
- `n_threads_batch`: default `8`.
- `n_gpu_layers`: default `-1` (full offload where supported).
- `main_gpu`: default `-1`.
- `gpu`: default `-1` (unset). Set to list-devices index for single-device placement.
- `no_kv_offload`: default `0`.
- `mmproj_use_gpu`: default `-1` (auto), valid `-1/0/1`.
- `cache_ram_mib`: default `-1`.
- `seed`: default `-1`.
- `ctx_shift`: default `1`.
- `kv_unified`: default `1`.
- `devices`: CSV of device indices or names (advanced multi-device path).
- `tensor_split`: CSV ratios (advanced).
- `split_mode`: `-1` unset, `0` none, `1` layer, `2` row.
- `embedding`: default `0`; set `1` for embeddings mode.
- `reranking`: default `0`; set `1` for rerank mode.
- `pooling_type`: default `-1`.

## Exported lifecycle and utility APIs

- `llama_server_bridge_default_params`
- `llama_server_bridge_create`
- `llama_server_bridge_destroy`
- `llama_server_bridge_last_error`
- `llama_server_bridge_list_devices`
- `llama_server_bridge_free_devices`
- `llama_server_bridge_result_free`
- `llama_server_bridge_json_result_free`

## Error handling pattern

- Bridge call return code `0` means call path succeeded.
- Result structs still need `out.ok == 1`.
- On failure, inspect both:
- `out.error_json` (if provided)
- `llama_server_bridge_last_error(bridge)`

## Notes for host applications

- Always initialize structs with `*_default_*()` helpers.
- Keep one bridge instance alive for multiple requests to avoid repeated model load cost.
- Increase `n_parallel` only for concurrent requests.
