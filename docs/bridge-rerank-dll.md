# Bridge Rerank DLL (`llama-server-bridge.dll`)

This document covers direct rerank calls through:
- `llama_server_bridge_create`
- `llama_server_bridge_rerank`

Reference header: `bridge/llama_server_bridge.h`.

## Simplest call

```c
#include "llama_server_bridge.h"

llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/rerank.gguf";
p.reranking = 1;
p.gpu = 1;

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_rerank_request req = llama_server_bridge_default_rerank_request();
req.body_json =
    "{"
    "\"query\":\"what is this about?\","
    "\"documents\":[\"doc a\",\"doc b\"],"
    "\"top_n\":2"
    "}";

llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
int rc = llama_server_bridge_rerank(bridge, &req, &out);

if (rc == 0 && out.ok == 1) {
    // out.json
}

llama_server_bridge_json_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## Device selection (repeat for this function)

- Enumerate with `llama_server_bridge_list_devices()`.
- Set `params.gpu = <index>` for single-device routing.
- Do not set both `gpu` and `devices`.
- No selector defaults:
- Windows/Linux: CPU-only.
- macOS: first GPU.

Performance guidance:
- Keep rerank on one GPU when possible.
- Split only for fit when model/KV does not fit one GPU.

## Supported create parameters (all)

All bridge create fields are supported:
- `model_path`, `mmproj_path`
- `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`
- `n_threads`, `n_threads_batch`
- `n_gpu_layers`, `main_gpu`, `gpu`
- `no_kv_offload`, `mmproj_use_gpu`, `cache_ram_mib`
- `seed`, `ctx_shift`, `kv_unified`
- `devices`, `tensor_split`, `split_mode`
- `embedding`, `reranking`, `pooling_type`

Rerank mode requirements:
- set `reranking = 1`
- runtime also enables embedding mode internally for this path

## Rerank request parameters (all)

`llama_server_bridge_rerank_request` fields:
- `body_json` (required)

Expected JSON shape:
- `query`: string
- `documents` or `texts`: array of strings
- optional `top_n`

## Full rerank example (advanced)

```c
llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/rerank.gguf";
p.reranking = 1;
p.gpu = 1;
p.n_ctx = 8192;
p.n_batch = 2048;
p.n_ubatch = 2048;
p.n_parallel = 1;
p.n_threads = 8;
p.n_threads_batch = 8;
p.n_gpu_layers = -1;
p.no_kv_offload = 0;
p.kv_unified = 1;
p.split_mode = 0; // none

llama_server_bridge *bridge = llama_server_bridge_create(&p);

const char *body =
    "{"
    "\"query\":\"find passages about kv cache\","
    "\"documents\":["
    "\"doc one text\","
    "\"doc two text\","
    "\"doc three text\""
    "],"
    "\"top_n\":3"
    "}";

llama_server_bridge_rerank_request req = llama_server_bridge_default_rerank_request();
req.body_json = body;

llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
int rc = llama_server_bridge_rerank(bridge, &req, &out);
if (rc != 0 || out.ok == 0) {
    const char *last = llama_server_bridge_last_error(bridge);
    // inspect out.error_json and last
}

llama_server_bridge_json_result_free(&out);
llama_server_bridge_destroy(bridge);
```
