# Extras (conversion tooling)

Tooling scripts for pyannote model conversion and GGUF packaging.

## Scripts

- `convert_pyannote_checkpoint_to_gguf.py`
- `convert_pyannote_npz_to_gguf.py`
- `convert_pyannote_to_gguf.ps1`

These scripts are tooling-only and not part of runtime inference.

## Local dependencies

- `../third_party/llama.cpp/gguf-py/` for GGUF writer import
- Python environment with conversion dependencies

## Licenses and notices

- `../third_party/licenses/README.md`
- `../third_party/licenses/torch-LICENSE.txt`
- `../third_party/licenses/torchaudio-LICENSE.txt`
- `../third_party/licenses/numpy-LICENSE.txt`
- `../third_party/licenses/tooling-full/`
