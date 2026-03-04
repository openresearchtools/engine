# PDF VLM DLL (`pdfvlm.dll`)

`pdfvlm.dll` runs PDF/image to markdown using a VLM text model + mmproj model through the bridge runtime.

## Simplest call

Exports:

```c
// Exported by pdfvlm.dll
int32_t pdfvlm_run_from_argv(int32_t argc, const char * const *argv, char **out_error);
void pdfvlm_free_c_string(char *ptr);
```

Minimal PDF run on one selected GPU:

```c
const char *argv[] = {
    "pdf_to_markdown",
    "--pdf", "C:/docs/file.pdf",
    "--model", "C:/models/vision.gguf",
    "--mmproj", "C:/models/mmproj.gguf",
    "gpu=1"
};
char *err = NULL;
int rc = pdfvlm_run_from_argv((int32_t)(sizeof(argv) / sizeof(argv[0])), argv, &err);
if (rc != 0) {
    // handle err
}
if (err) {
    pdfvlm_free_c_string(err);
}
```

## Device selection (repeat for this function)

Use bridge-enumerated device indices.

- Preferred minimal selector: `--gpu <index>` or `gpu=<index>`.
- Advanced selector: `--devices <csv>`.
- Do not pass both `--gpu` and `--devices`.
- If `--gpu` is provided and split mode is not set, runtime forces single-device mode (`split_mode=none`).

Default behavior:
- Windows/Linux with no GPU selector: CPU-only default.
- macOS with no GPU selector: first GPU default.
- `n_gpu_layers` defaults to `-1`.
- `kv_unified=1` and KV offload are enabled by default in bridge runtime.
- `mmproj_use_gpu=-1` (auto) makes mmproj follow selected GPU by default.

Performance guidance:
- One GPU is usually faster than split.
- Only use split when model + KV cannot fit in one GPU.

## Full supported arguments

Input mode:
- `--pdf <path>` or `--image <path>` (one mode at a time).

Model/output:
- `--model <path>`
- `--mmproj <path>`
- `--out-md <path>` or `--out <path>`
- `--out-dir <path>`

PDF rendering:
- `--pdfium-lib <path>`
- `--pdfium-dll <path>` (alias)
- `--pages <list>` (for PDF mode)
- `--scale <float>`
- `--oversample <float>`

Generation/runtime:
- `--prompt <text>`
- `--n-predict <int>`
- `--n-ctx <int>`
- `--threads <int>`
- `--threads-batch <int>`
- `--batch-size <int>`
- `--parallel <int>` (used in PDF mode; image mode runs single slot)
- `--max-retries <int>`

Device/offload/split:
- `--gpu <index>`
- `gpu=<index>` or `--gpu=<index>`
- `--devices <csv>`
- `--n-gpu-layers <int>`
- `--main-gpu <int>`
- `--mmproj-use-gpu <-1|0|1>`
- `--split-mode <none|layer|row>`
- `--tensor-split <csv>`

## Full example (explicit split/offload control)

```c
const char *argv[] = {
    "pdf_to_markdown",
    "--pdf", "C:/docs/file.pdf",
    "--pdfium-lib", "C:/app/vendor/pdfium/pdfium.dll",
    "--pages", "1,3-5",
    "--model", "C:/models/vision.gguf",
    "--mmproj", "C:/models/mmproj.gguf",
    "--out", "C:/out/file.md",
    "--n-ctx", "32768",
    "--batch-size", "2048",
    "--parallel", "4",
    "--threads", "8",
    "--threads-batch", "8",
    "--gpu", "1",
    "--n-gpu-layers", "-1",
    "--mmproj-use-gpu", "-1",
    "--split-mode", "none"
};
char *err = NULL;
int rc = pdfvlm_run_from_argv((int32_t)(sizeof(argv) / sizeof(argv[0])), argv, &err);
if (rc != 0) {
    // handle err
}
if (err) {
    pdfvlm_free_c_string(err);
}
```
