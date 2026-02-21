# Extras (conversion tooling)

Tooling scripts for pyannote model conversion and GGUF packaging.

## Scripts

- `convert_pyannote_checkpoint_to_gguf.py`
- `convert_pyannote_npz_to_gguf.py`
- `convert_pyannote_to_gguf.ps1`
- `update_upstreams.ps1`
- `license_audit.ps1`
- `extract_cpp_licenses.ps1`

### Update third_party subtrees

From `extras` folder:

- `.\update_upstreams.ps1` updates both `third_party/llama.cpp` and `third_party/whisper.cpp`.
- `.\update_upstreams.ps1 -Commit` updates and commits both subtree updates.
- `.\update_upstreams.ps1 -Commit -Push` additionally pushes the commit.

These scripts are tooling-only and not part of runtime inference.

## Local dependencies

- [`third_party/llama.cpp/gguf-py/`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp/gguf-py/) for GGUF writer import
- Python environment with conversion dependencies

## Licenses and notices

- [`third_party/licenses/README.md`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/README.md)
- [`third_party/licenses/torch-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/torch-LICENSE.txt)
- [`third_party/licenses/torchaudio-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/torchaudio-LICENSE.txt)
- [`third_party/licenses/numpy-LICENSE.txt`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/numpy-LICENSE.txt)
- [`third_party/licenses/tooling-full/`](https://github.com/openresearchtools/engine/tree/main/third_party/licenses/tooling-full/)

### License audit

Run before release to enumerate files carrying embedded license markers:

```
.\extras\license_audit.ps1
```

Outputs:

- `../ENGINEbuilds/license-audits/license-audit-<timestamp>.csv`

Use `-NoThirdParty` if you want to exclude `third_party`.
Use `-OnlyFirstParty` to isolate only files outside `third_party`.
Use `-Strict -FailOnUnknown` for a stricter release check.
Use `-FailOnFirstParty` to fail if any first-party files contain license markers.

## C++ licenses snapshot

Run this to export upstream license notices for llama.cpp and whisper.cpp into:

- `third_party/licenses/cpp_licenses/llama.cpp/`
- `third_party/licenses/cpp_licenses/whisper.cpp/`

```
.\extras\extract_cpp_licenses.ps1
```

Use `-IncludeNested` to also copy nested `LICENSE`/`COPYING`/`NOTICE` files.
