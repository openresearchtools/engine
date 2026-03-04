# Engine DLL Embedding Docs

These docs describe direct DLL embedding for this repo's runtime.

Each function doc is written in the same order:
1. Simplest call (minimal usage)
2. Device selection for that function (`gpu` by enumerated index)
3. Full supported parameters
4. Full/advanced call example

## Document index

- [Common Runtime And Device Rules](./common-runtime-and-devices.md)
- [PDF DLL (`pdf.dll`)](./pdf-dll.md)
- [PDF VLM DLL (`pdfvlm.dll`)](./pdfvlm-dll.md)
- [Bridge Chat (`llama-server-bridge.dll`)](./bridge-chat-dll.md)
- [Bridge VLM (`llama-server-bridge.dll`)](./bridge-vlm-dll.md)
- [Bridge Embeddings (`llama-server-bridge.dll`)](./bridge-embeddings-dll.md)
- [Bridge Rerank (`llama-server-bridge.dll`)](./bridge-rerank-dll.md)
- [Bridge Audio / Transcription + Diarization (`llama-server-bridge.dll`)](./bridge-audio-dll.md)

## Practical performance rule

For all model workloads (chat, VLM, embeddings, rerank, audio):
- Single-GPU placement is usually faster than split.
- Only use multi-device split when model/KV does not fit one GPU.
- In this runtime, no split is used by default unless you explicitly pass split parameters.
