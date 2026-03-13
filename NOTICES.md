# Notices

Openresearchtools-Engine source code is licensed under the MIT License.

Third-party software used, adapted, or bundled by this repository remains under its original license terms.

Primary notice index:

* [third_party/licenses/README.md](third_party/licenses/README.md)

Bundled runtime/release license bundles:

* [third_party/licenses/LICENSES.txt](third_party/licenses/LICENSES.txt)
* [third_party/licenses/LICENSES-vulkan.txt](third_party/licenses/LICENSES-vulkan.txt)
* [third_party/licenses/LICENSES-metal.txt](third_party/licenses/LICENSES-metal.txt)
* [third_party/licenses/LICENSES-cuda.txt](third_party/licenses/LICENSES-cuda.txt)
* [third_party/licenses/LICENSES-ubuntu-vulkan.txt](third_party/licenses/LICENSES-ubuntu-vulkan.txt)

Top-level notice categories tracked in this repo:

* directly imported/adapted runtime code and bundled runtime binaries
* repo-kept native adaptations and overlays
* directly imported/adapted model conversion/parity tooling
* external reference and validation repositories used during bring-up
* package-managed dependency licenses tracked without source-lineage dissection

Current audio/runtime notice highlights:

* The core ENGINE runtime and in-process server stack are heavily based on `llama.cpp`, with ENGINE-added audio/session, Sortformer, and Voxtral integrations layered onto that base.
* `whisper.cpp` and `ggml` remain major native runtime dependencies in the current stack.
* The current Sortformer converter is a repo-written ENGINE script; NVIDIA NeMo references are used for Sortformer archive semantics, tensor/config mapping, and parity validation in that conversion flow.
* `parakeet.cpp` is tracked as an external C++ Sortformer reference that was studied during native Sortformer bring-up.
* Native Voxtral realtime runtime code is adapted from the MIT-licensed `voxtral-cpp` reference implementation, and the repo-kept Voxtral GGUF converter is adapted from `voxtral-cpp/tools/convert_voxtral_to_gguf.py`.
* `vLLM`, `voxtral.c`, `voxtral-mini-realtime-rs`, and `mlx-audio` are tracked as external reference/validation inputs for Voxtral behavior, benchmarking, and integration study.
