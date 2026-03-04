# PDF DLL (`pdf.dll`)

`pdf.dll` is for native PDF text/markdown extraction only. It does not run model inference and does not use GPU selection.

## Simplest call

Call the argv-style export:

```c
// Exported by pdf.dll
int32_t pdf_run_from_argv(int32_t argc, const char * const *argv, char **out_error);
void pdf_free_c_string(char *ptr);
```

Minimal single-file extraction to stdout:

```c
const char *argv[] = {
    "pdf",
    "extract",
    "--input", "C:/docs/file.pdf"
};
char *err = NULL;
int rc = pdf_run_from_argv((int32_t)(sizeof(argv) / sizeof(argv[0])), argv, &err);
if (rc != 0) {
    // err contains message if available
}
if (err) {
    pdf_free_c_string(err);
}
```

## Device selection (repeat for this function)

Not applicable for `pdf.dll`.

- No `gpu` parameter is supported.
- No model or mmproj is used.

## Full supported arguments

`pdf_run_from_argv` supports the same arguments as `engine pdf ...`.

Global option:
- `--pdfium-lib <path>`: PDFium library file or folder containing it.

Subcommand:
- `extract`

`extract` options:
- `--input <path>`: single PDF file or directory of PDFs. Required.
- `--output <path>`: output file (single input) or output directory.
- `--non-recursive`: do not recurse when `--input` is a directory.
- `--password <password>`: optional PDF password.
- `--overwrite`: overwrite existing output.

Behavior notes:
- If input is one PDF and `--output` is omitted, extracted markdown is printed to stdout.
- If input is a directory, `--output` is required.
- Writes UTF-8 with BOM when writing files.

## Full example (directory extraction)

```c
const char *argv[] = {
    "pdf",
    "extract",
    "--pdfium-lib", "C:/app/vendor/pdfium/pdfium.dll",
    "--input", "C:/docs/in",
    "--output", "C:/docs/out",
    "--overwrite"
};
char *err = NULL;
int rc = pdf_run_from_argv((int32_t)(sizeof(argv) / sizeof(argv[0])), argv, &err);
if (rc != 0) {
    // handle err
}
if (err) {
    pdf_free_c_string(err);
}
```
