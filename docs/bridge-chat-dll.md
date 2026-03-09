# Bridge Chat DLL (`llama-server-bridge.dll`)

This document covers direct chat calls through:
- `llama_server_bridge_create`
- `llama_server_bridge_chat_complete`

Reference header: `bridge/llama_server_bridge.h`.

## Simplest call

```c
#include "llama_server_bridge.h"

llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/chat.gguf";
p.gpu = 1; // optional; remove for CPU default on Windows/Linux

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_chat_request req = llama_server_bridge_default_chat_request();
req.prompt = "Summarize this text in 5 bullets.";

llama_server_bridge_vlm_result out = llama_server_bridge_empty_vlm_result();
int rc = llama_server_bridge_chat_complete(bridge, &req, &out);

if (rc == 0 && out.ok == 1) {
    // out.text
}

llama_server_bridge_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## Device selection (repeat for this function)

- Enumerate with `llama_server_bridge_list_devices()`.
- Set `params.gpu = <enumerated index>` for minimal single-device routing.
- Do not set both `params.gpu` and `params.devices`.
- If no GPU is provided:
- Windows/Linux defaults to CPU-only.
- macOS defaults to first GPU.

Performance guidance:
- Single GPU is usually faster than split.
- Split is for out-of-memory fit cases.
- No tensor split happens unless you explicitly set split controls.

## Supported create parameters (all)

All fields are supported for chat bridge creation:

- `model_path`, `mmproj_path`
- `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`
- `n_threads`, `n_threads_batch`
- `n_gpu_layers`, `main_gpu`, `gpu`
- `no_kv_offload`, `mmproj_use_gpu`, `cache_ram_mib`
- `seed`, `ctx_shift`, `kv_unified`
- `devices`, `tensor_split`, `split_mode`
- `embedding`, `reranking`, `pooling_type`

Defaults (from `llama_server_bridge_default_params()`):
- `n_ctx=32768`, `n_batch=2048`, `n_ubatch=2048`, `n_parallel=1`
- `n_threads=8`, `n_threads_batch=8`
- `n_gpu_layers=-1`, `gpu=-1`, `main_gpu=-1`
- `no_kv_offload=0`, `kv_unified=1`
- `split_mode=0` (`none`)

## Chat request parameters (all)

`llama_server_bridge_chat_request` fields:
- `prompt` (required)
- `n_predict`
- `id_slot`
- `temperature`, `top_p`, `top_k`, `min_p`
- `seed`
- `repeat_last_n`, `repeat_penalty`
- `presence_penalty`, `frequency_penalty`
- `dry_multiplier`, `dry_allowed_length`, `dry_penalty_last_n`

Sentinel behavior in runtime:
- negative values typically mean "use route defaults" for optional sampling knobs.
- if `n_predict <= 0`, runtime falls back to internal default.

## Full chat example (advanced)

```c
llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/chat.gguf";
p.gpu = 1;
p.n_ctx = 50000;
p.n_batch = 1024;
p.n_ubatch = 1024;
p.n_parallel = 1;
p.n_threads = 8;
p.n_threads_batch = 8;
p.n_gpu_layers = -1;
p.main_gpu = -1;
p.no_kv_offload = 0;
p.kv_unified = 1;
p.split_mode = 0; // none

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_chat_request req = llama_server_bridge_default_chat_request();
req.prompt = "Explain the core findings in plain language.";
req.n_predict = 1200;
req.temperature = 0.2f;
req.top_p = 0.95f;
req.top_k = 40;
req.min_p = 0.05f;
req.repeat_last_n = 256;
req.repeat_penalty = 1.05f;
req.presence_penalty = 0.0f;
req.frequency_penalty = 0.0f;

llama_server_bridge_vlm_result out = llama_server_bridge_empty_vlm_result();
int rc = llama_server_bridge_chat_complete(bridge, &req, &out);
if (rc != 0 || out.ok == 0) {
    const char *last = llama_server_bridge_last_error(bridge);
    // out.error_json and last may both contain useful details
}

llama_server_bridge_result_free(&out);
llama_server_bridge_destroy(bridge);
```
