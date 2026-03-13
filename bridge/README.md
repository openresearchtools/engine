# Bridge (patchable layer)

`llama_server_bridge` is a thin C API over `llama.cpp` server internals:

- uses `server_context` + task queue directly.
- no HTTP transport.
- no base64 transport at app boundary for VLM and raw-audio bridge calls.
- audio endpoint compatibility currently serializes raw audio bytes to base64 inside bridge when routing through `/v1/audio/transcriptions`.
- multimodal request path uses raw image bytes (`task.cli_files`).
- embeddings and rerank are callable in-process through the same server task queue.
- GPU devices can be enumerated in-process (no HTTP) and selected explicitly.

## API stability intent

The Rust side calls only these C symbols:

- `llama_server_bridge_create`
- `llama_server_bridge_destroy`
- `llama_server_bridge_vlm_complete`
- `llama_server_bridge_embeddings`
- `llama_server_bridge_rerank`
- `llama_server_bridge_audio_transcriptions`
- `llama_server_bridge_audio_transcriptions_raw`
- `llama_server_bridge_result_free`
- `llama_server_bridge_last_error`
- `llama_server_bridge_list_devices`
- `llama_server_bridge_free_devices`
- plus default/empty struct helpers

As long as these signatures remain stable, Rust call sites do not change.

## Upstream patch footprint

Upstream touch points are:

- `llama.cpp/CMakeLists.txt`: add `LLAMA_BUILD_MARKDOWN_BRIDGE` option + `add_subdirectory(MARKDOWN/bridge)`.
- `llama.cpp/MARKDOWN/bridge/*`: bridge source and CMake target.
