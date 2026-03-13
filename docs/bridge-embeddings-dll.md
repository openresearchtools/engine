# Bridge Embeddings DLL (`llama-server-bridge.dll`)

This document covers direct embeddings calls through:
- `llama_server_bridge_create`
- `llama_server_bridge_embeddings`

Reference header: `bridge/llama_server_bridge.h`.

## Simplest call

```c
#include "llama_server_bridge.h"

llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/embedding.gguf";
p.embedding = 1;
p.gpu = 1;

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_embeddings_request req = llama_server_bridge_default_embeddings_request();
req.body_json = "{\"input\":[\"hello world\"],\"encoding_format\":\"float\"}";
req.oai_compat = 1; // /v1/embeddings

llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
int rc = llama_server_bridge_embeddings(bridge, &req, &out);

if (rc == 0 && out.ok == 1) {
    // out.json
}

llama_server_bridge_json_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## Device selection (repeat for this function)

- Enumerate with `llama_server_bridge_list_devices()`.
- Set `params.gpu = <index>` for single-device placement.
- Do not set both `gpu` and `devices`.
- No selector defaults:
- Windows/Linux: CPU-only.
- macOS: first GPU.

Performance guidance:
- One GPU is usually faster than split.
- Split only if one GPU cannot fit model/KV.

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

Embeddings mode requirements:
- set `embedding = 1`
- keep `reranking = 0` for pure embeddings

## Embeddings request parameters (all)

`llama_server_bridge_embeddings_request` fields:
- `body_json` (required)
- `oai_compat`:
- `1` routes to `/v1/embeddings`
- `0` routes to `/embeddings`

Typical JSON body:
- `input`: string or array of strings
- optional OpenAI-compatible fields such as `encoding_format`

## Full embeddings example (advanced)

```c
llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/embedding.gguf";
p.embedding = 1;
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
    "\"input\":[\"first sentence\",\"second sentence\"],"
    "\"encoding_format\":\"float\""
    "}";

llama_server_bridge_embeddings_request req = llama_server_bridge_default_embeddings_request();
req.body_json = body;
req.oai_compat = 1;

llama_server_bridge_json_result out = llama_server_bridge_empty_json_result();
int rc = llama_server_bridge_embeddings(bridge, &req, &out);
if (rc != 0 || out.ok == 0) {
    const char *last = llama_server_bridge_last_error(bridge);
    // inspect out.error_json and last
}

llama_server_bridge_json_result_free(&out);
llama_server_bridge_destroy(bridge);
```
