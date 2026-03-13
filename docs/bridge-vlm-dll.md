# Bridge VLM DLL (`llama-server-bridge.dll`)

This document covers image VLM calls through:
- `llama_server_bridge_create`
- `llama_server_bridge_vlm_complete`

Reference header: `bridge/llama_server_bridge.h`.

## Simplest call

```c
#include "llama_server_bridge.h"

// Load whole model + mmproj on one GPU index from list-devices
llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/vision.gguf";
p.mmproj_path = "./models/mmproj.gguf";
p.gpu = 1;

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_vlm_request req = llama_server_bridge_default_vlm_request();
req.prompt = "Convert this image to markdown only.";
req.image_bytes = image_ptr;
req.image_bytes_len = image_size;

llama_server_bridge_vlm_result out = llama_server_bridge_empty_vlm_result();
int rc = llama_server_bridge_vlm_complete(bridge, &req, &out);

if (rc == 0 && out.ok == 1) {
    // out.text
}

llama_server_bridge_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## Device selection (repeat for this function)

- Enumerate devices with `llama_server_bridge_list_devices()`.
- Set `params.gpu = <index>` for the simplest single-device path.
- Do not set both `gpu` and `devices`.
- Defaults with no selector:
- Windows/Linux: CPU-only.
- macOS: first GPU.

VLM-specific default behavior:
- `mmproj_use_gpu=-1` (auto) means mmproj follows selected GPU when primary device is GPU.
- `n_gpu_layers=-1` default means full offload where supported.
- `split_mode` defaults to none unless explicitly set.

Performance guidance:
- Prefer one selected GPU (model + mmproj + KV together).
- Use split only when one GPU cannot fit model/KV.

## Supported create parameters (all)

All bridge create fields are supported for VLM:
- `model_path`, `mmproj_path`
- `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`
- `n_threads`, `n_threads_batch`
- `n_gpu_layers`, `main_gpu`, `gpu`
- `no_kv_offload`, `mmproj_use_gpu`, `cache_ram_mib`
- `seed`, `ctx_shift`, `kv_unified`
- `devices`, `tensor_split`, `split_mode`
- `embedding`, `reranking`, `pooling_type`

## VLM request parameters (all)

`llama_server_bridge_vlm_request` fields:
- `prompt` (required)
- `image_bytes` (required)
- `image_bytes_len` (required)
- `n_predict`
- `id_slot`
- `temperature`, `top_p`, `top_k`, `min_p`
- `seed`
- `repeat_last_n`, `repeat_penalty`
- `presence_penalty`, `frequency_penalty`
- `dry_multiplier`, `dry_allowed_length`, `dry_penalty_last_n`
- `reasoning` (`on`, `off`, `auto`, or null/unset)
- `reasoning_budget` (`-1`, `0`, positive integer, or unset sentinel)
- `reasoning_format` (`none`, `deepseek`, `deepseek-legacy`, or null/unset)

`llama_server_bridge_vlm_result` fields:
- `ok`, `truncated`, `stop`, `eos_reached`
- `n_decoded`, `n_prompt_tokens`, `n_tokens_cached`
- `prompt_ms`, `predicted_ms`
- `text`, `error_json`

Reasoning behavior:
- If `reasoning` is unset, no reasoning flags are sent.
- If `reasoning = "off"`, the bridge forces `reasoning_budget = 0`.
- If `reasoning = "on"` or `reasoning = "auto"` and no budget is supplied, the bridge sends `reasoning_budget = -1`.
- If `reasoning` is set and no format is supplied, the bridge sends `reasoning_format = "deepseek"`.
- `reasoning = "auto"` leaves template/runtime thinking behavior up to the model chat template.

## Full VLM example (advanced)

```c
llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/vision.gguf";
p.mmproj_path = "./models/mmproj.gguf";
p.gpu = 1;
p.n_ctx = 32768;
p.n_batch = 2048;
p.n_ubatch = 2048;
p.n_parallel = 1;
p.n_threads = 8;
p.n_threads_batch = 8;
p.n_gpu_layers = -1;
p.mmproj_use_gpu = -1; // auto
p.no_kv_offload = 0;
p.kv_unified = 1;
p.split_mode = 0; // none

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_vlm_request req = llama_server_bridge_default_vlm_request();
req.prompt = "Extract all visible text and return markdown only.";
req.image_bytes = image_ptr;
req.image_bytes_len = image_size;
req.n_predict = 5000;
req.temperature = 0.0f;
req.top_p = 1.0f;
req.top_k = -1;
req.min_p = -1.0f;
req.seed = -1;

llama_server_bridge_vlm_result out = llama_server_bridge_empty_vlm_result();
int rc = llama_server_bridge_vlm_complete(bridge, &req, &out);
if (rc != 0 || out.ok == 0) {
    const char *last = llama_server_bridge_last_error(bridge);
    // inspect out.error_json and last
}

llama_server_bridge_result_free(&out);
llama_server_bridge_destroy(bridge);
```

## VLM reasoning example

```c
llama_server_bridge_params p = llama_server_bridge_default_params();
p.model_path = "./models/vision.gguf";
p.mmproj_path = "./models/mmproj.gguf";
p.gpu = 0;

llama_server_bridge *bridge = llama_server_bridge_create(&p);

llama_server_bridge_vlm_request req = llama_server_bridge_default_vlm_request();
req.prompt = "Describe the image briefly, then identify the main object shown.";
req.image_bytes = image_ptr;
req.image_bytes_len = image_size;
req.n_predict = 1024;
req.reasoning = "off";
// reasoning_budget is automatically forced to 0 by the bridge when reasoning="off".
// reasoning_format defaults to "deepseek" when reasoning is set.

llama_server_bridge_vlm_result out = llama_server_bridge_empty_vlm_result();
int rc = llama_server_bridge_vlm_complete(bridge, &req, &out);

llama_server_bridge_result_free(&out);
llama_server_bridge_destroy(bridge);
```

To request visible reasoning output instead:

```c
req.reasoning = "on";
req.reasoning_budget = -1;
req.reasoning_format = "none";
```
