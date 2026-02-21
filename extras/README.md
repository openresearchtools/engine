# Extras (conversion tooling)

Tooling scripts for pyannote model conversion and GGUF packaging.

## Scripts

- `convert_pyannote_checkpoint_to_gguf.py`
- `convert_pyannote_npz_to_gguf.py`
- `convert_pyannote_to_gguf.ps1`
- `update_upstreams.ps1`
- `release_inventory.ps1`

### Update third_party upstream sources

From `extras` folder:

- `.\update_upstreams.ps1` syncs both `third_party/llama.cpp` and `third_party/whisper.cpp` from upstream refs.
- `.\update_upstreams.ps1 -LlamaOnly` syncs only `third_party/llama.cpp`.
- `.\update_upstreams.ps1 -WhisperOnly` syncs only `third_party/whisper.cpp`.
- `.\update_upstreams.ps1 -LlamaRef <ref> -WhisperRef <ref>` pins branches/tags/commits.
- `.\update_upstreams.ps1 -Commit` stages and commits synced folders.
- `.\update_upstreams.ps1 -Commit -Push` also pushes.

These scripts are tooling-only and not part of runtime inference.

## Local dependencies

- [`third_party/llama.cpp/gguf-py/`](https://github.com/openresearchtools/engine/tree/main/third_party/llama.cpp/gguf-py/) for GGUF writer import
- Python environment with conversion dependencies

## Licenses and notices

- [`third_party/licenses/README.md`](https://github.com/openresearchtools/engine/blob/main/third_party/licenses/README.md)

## Release inventory tracking

Use this after packaging to capture exact shipped files, hashes, and PE dependency lists.

```
powershell -ExecutionPolicy Bypass -File .\extras\release_inventory.ps1 -ArtifactDir "..\ENGINEbuilds\my-release"
```

Output:

- `..\ENGINEbuilds\release-inventory\run-<timestamp>\artifact-files.csv`
- `..\ENGINEbuilds\release-inventory\run-<timestamp>\allowlist.txt`
- `..\ENGINEbuilds\release-inventory\run-<timestamp>\binary-dependents.txt`

Create baseline:

```
powershell -ExecutionPolicy Bypass -File .\extras\release_inventory.ps1 -ArtifactDir "..\ENGINEbuilds\my-release" -WriteBaseline -BaselineName "win-x64-release"
```

Compare against baseline:

```
powershell -ExecutionPolicy Bypass -File .\extras\release_inventory.ps1 -ArtifactDir "..\ENGINEbuilds\my-release" -CompareBaseline -BaselineName "win-x64-release"
```
