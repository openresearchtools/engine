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

* shipped runtime code and bundled runtime binaries
* repo-kept native adaptations and overlays
* repo-kept model conversion/parity tooling
* external reference and validation repositories used during bring-up

Current audio/runtime notice highlights:

* `llama.cpp`, `whisper.cpp`, and `ggml` remain the main native runtime base.
* Native Sortformer conversion/parity tooling is tracked against NVIDIA NeMo references.
* Native Voxtral realtime runtime code is adapted from the MIT-licensed `voxtral-cpp` reference implementation.
* `vLLM` and `voxtral.c` are tracked as external reference/validation inputs for Voxtral realtime behavior and benchmarking.
