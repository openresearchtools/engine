use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::io;
use std::io::Cursor;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use crate::llama_bridge_ffi::{
    llama_server_bridge, llama_server_bridge_create, llama_server_bridge_default_params,
    llama_server_bridge_default_vlm_request, llama_server_bridge_destroy,
    llama_server_bridge_empty_vlm_result, llama_server_bridge_last_error,
    llama_server_bridge_params, llama_server_bridge_result_free, llama_server_bridge_vlm_complete,
    llama_server_bridge_vlm_request, llama_server_bridge_vlm_result,
};
use image::imageops::FilterType;
use image::ImageFormat;
use image::GenericImageView;
use pdfium_render::prelude::*;

const DEFAULT_PROMPT: &str = "Convert this page to markdown. Do not miss any text and only output the bare markdown! Any graphs or figures found convert to markdown table. If figure is image without details, describe what you see in the image. For tables, pay attention to whitespace: some cells may be intentionally empty, so keep empty and filled cells in the correct columns. Ensure correct assignment of column headings and subheadings for tables.";

#[derive(Debug, Clone)]
struct Args {
    pdf_path: PathBuf,
    pdfium_dll: PathBuf,
    model_path: PathBuf,
    mmproj_path: PathBuf,
    output_md: PathBuf,
    pages: Option<Vec<usize>>,
    scale: f32,
    oversample: f32,
    prompt: String,
    n_predict: i32,
    n_ctx_total: u32,
    n_threads: i32,
    batch_size: u32,
    parallel: usize,
    max_retries: usize,
}

#[derive(Debug, Clone)]
struct ImageArgs {
    image_path: PathBuf,
    model_path: PathBuf,
    mmproj_path: PathBuf,
    output_md: PathBuf,
    scale: f32,
    oversample: f32,
    prompt: String,
    n_predict: i32,
    n_ctx_total: u32,
    n_threads: i32,
    batch_size: u32,
    max_retries: usize,
}

#[derive(Debug)]
struct RenderedPage {
    page: u32,
    png_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Row {
    page: u32,
    seconds: f64,
    attempts: usize,
    chars: usize,
    markdown: String,
}

#[derive(Debug)]
struct VlmResponse {
    markdown: String,
    eos_reached: bool,
    truncated: bool,
}

#[derive(Debug)]
struct SharedBridge {
    ptr: *mut llama_server_bridge,
}

unsafe impl Send for SharedBridge {}
unsafe impl Sync for SharedBridge {}

impl Drop for SharedBridge {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                llama_server_bridge_destroy(self.ptr);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

impl SharedBridge {
    fn new(
        model_path: &Path,
        mmproj_path: &Path,
        n_ctx_total: u32,
        n_threads: i32,
        batch_size: u32,
        parallel: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        prepare_windows_bridge_runtime_paths();

        let c_model = CString::new(model_path.to_string_lossy().as_ref())?;
        let c_mmproj = CString::new(mmproj_path.to_string_lossy().as_ref())?;

        let mut params: llama_server_bridge_params = unsafe { llama_server_bridge_default_params() };
        params.model_path = c_model.as_ptr();
        params.mmproj_path = c_mmproj.as_ptr();
        params.n_ctx = n_ctx_total as i32;
        params.n_batch = batch_size as i32;
        params.n_ubatch = batch_size as i32;
        params.n_parallel = parallel as i32;
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
        params.n_gpu_layers = -1;
        params.main_gpu = 0;
        params.no_kv_offload = 0;
        params.mmproj_use_gpu = 1;
        params.cache_ram_mib = 0;
        params.seed = -1;
        params.ctx_shift = 1;
        params.kv_unified = 1;

        let ptr = unsafe { llama_server_bridge_create(&params) };
        if ptr.is_null() {
            return Err("llama_server_bridge_create() failed".into());
        }
        Ok(Self { ptr })
    }

    fn run_vlm(
        &self,
        prompt: &str,
        image_bytes: &[u8],
        n_predict: i32,
        use_retry_penalties: bool,
    ) -> Result<VlmResponse, Box<dyn std::error::Error>> {
        if self.ptr.is_null() {
            return Err("Bridge is null".into());
        }
        if image_bytes.is_empty() {
            return Err("Image bytes are empty".into());
        }

        let c_prompt = CString::new(prompt)?;
        let mut req: llama_server_bridge_vlm_request =
            unsafe { llama_server_bridge_default_vlm_request() };
        req.prompt = c_prompt.as_ptr();
        req.image_bytes = image_bytes.as_ptr();
        req.image_bytes_len = image_bytes.len();
        req.n_predict = n_predict;
        req.id_slot = -1;
        req.temperature = 0.0;
        req.top_p = 1.0;
        req.top_k = -1;
        req.min_p = -1.0;
        req.seed = -1;
        if use_retry_penalties {
            req.repeat_last_n = 256;
            req.repeat_penalty = 1.08;
            req.presence_penalty = 0.15;
            req.frequency_penalty = 0.10;
            req.dry_multiplier = 1.2;
            req.dry_allowed_length = 4;
            req.dry_penalty_last_n = 256;
        }

        let mut out: llama_server_bridge_vlm_result =
            unsafe { llama_server_bridge_empty_vlm_result() };
        let rc = unsafe { llama_server_bridge_vlm_complete(self.ptr, &req, &mut out) };

        let response = if rc != 0 || out.ok == 0 {
            let mut parts = Vec::new();
            if !out.error_json.is_null() {
                parts.push(c_ptr_to_string(out.error_json));
            }
            let last_error_ptr = unsafe { llama_server_bridge_last_error(self.ptr) };
            if !last_error_ptr.is_null() {
                parts.push(c_ptr_to_string(last_error_ptr));
            }
            let message = if parts.is_empty() {
                format!("Bridge inference failed (rc={rc})")
            } else {
                format!("Bridge inference failed (rc={rc}): {}", parts.join(" | "))
            };
            Err::<VlmResponse, Box<dyn std::error::Error>>(message.into())
        } else if out.text.is_null() {
            Err::<VlmResponse, Box<dyn std::error::Error>>("Bridge returned null output text".into())
        } else {
            Ok(VlmResponse {
                markdown: sanitize_model_markdown(&c_ptr_to_string(out.text)),
                eos_reached: out.eos_reached != 0,
                truncated: out.truncated != 0,
            })
        };

        unsafe { llama_server_bridge_result_free(&mut out) };
        response
    }
}

fn c_ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

fn sanitize_model_markdown(raw: &str) -> String {
    let normalized = raw.replace("\r\n", "\n");
    let mut lines: Vec<&str> = normalized.lines().collect();

    while matches!(lines.first(), Some(line) if line.trim().is_empty()) {
        lines.remove(0);
    }
    while matches!(lines.last(), Some(line) if line.trim().is_empty()) {
        lines.pop();
    }

    if lines.len() >= 2 {
        let first = lines[0].trim();
        let last = lines[lines.len() - 1].trim();
        if first.starts_with("```") && last == "```" {
            return lines[1..lines.len() - 1].join("\n").trim().to_string();
        }
    }

    normalized.trim().to_string()
}

fn usage() -> &'static str {
    "Usage:
  pdf_to_markdown --pdf <pdf_path> --model <model.gguf> --mmproj <mmproj.gguf> [options]
  pdf_to_markdown --image <image_path> --model <model.gguf> --mmproj <mmproj.gguf> [options]

Options:
  --pdf             Full path to input PDF.
  --pdfium-lib      Optional path to PDFium library file (e.g. pdfium.dll/libpdfium.dylib)
                    or a directory containing it.
  --pdfium-dll      Alias for --pdfium-lib (backward compatibility).
                    If omitted, tries PDFIUM_LIB / PDFIUM_DLL env vars, then relative app paths.
  --image           Full path to input image.
  --model           Full path to VLM model gguf.
  --mmproj          Full path to vision projector gguf.
  --out-md, --out   Full path to output markdown file.
  --out-dir         Output directory. Result file name is input PDF stem + .md.
  --pages           1-based page list, e.g. 1,5,7-9 (default: all pages).
  --scale           PDF render target scale (default: 2.0).
  --oversample      Temporary render multiplier before bicubic downscale (default: 1.5).
  --prompt          VLM prompt text.
  --n-predict       Max new tokens per page (default: 5000).
  --n-ctx           Total context size (default: 32768).
  --threads         CPU threads (default: 8).
  --batch-size      Eval chunk size (default: 2048).
  --parallel        Concurrent slots/pages (default: 4).
  --max-retries     Per-page retries for non-EOS/truncation/loop (default: 2).
  -h, --help        Show this help.

Output behavior:
  - Exactly one output markdown file is written.
  - No temporary image files are created.
  - Logs are printed to stdout.
"
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

#[cfg(windows)]
fn prepare_windows_bridge_runtime_paths() {
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::iter;
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use std::sync::Once;

    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x00001000;
    const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x00000400;

    unsafe extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut c_void;
        fn SetDllDirectoryW(path_name: *const u16) -> i32;
    }

    fn as_wide_null(path: &Path) -> Vec<u16> {
        path.as_os_str()
            .encode_wide()
            .chain(iter::once(0))
            .collect()
    }

    fn candidate_dirs() -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(exe_path) = env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                out.push(exe_dir.join("vendor").join("ffmpeg").join("bin"));
            }
        }
        if let Ok(cwd) = env::current_dir() {
            out.push(cwd.join("vendor").join("ffmpeg").join("bin"));
        }
        out
    }

    fn add_search_dir(path: &Path) {
        if !path.is_dir() {
            return;
        }
        let wide = as_wide_null(path);
        unsafe {
            let cookie = AddDllDirectory(wide.as_ptr());
            if cookie.is_null() {
                let _ = SetDllDirectoryW(wide.as_ptr());
            }
        }
    }

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe {
            let _ = SetDefaultDllDirectories(
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
            );
        }

        let mut seen = HashSet::new();
        for dir in candidate_dirs() {
            let key = dir.to_string_lossy().to_string();
            if seen.insert(key) {
                add_search_dir(&dir);
            }
        }
    });
}

#[cfg(not(windows))]
fn prepare_windows_bridge_runtime_paths() {}

fn resolve_pdfium_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        Pdfium::pdfium_platform_library_name_at_path(path)
    } else {
        path.to_path_buf()
    }
}

fn default_pdfium_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    let mut push_candidates_for_root = |root: &Path| {
        out.push(resolve_pdfium_path(root));
        out.push(resolve_pdfium_path(&root.join("vendor").join("pdfium")));
        out.push(resolve_pdfium_path(
            &root.join("third_party").join("pdfium").join("bin"),
        ));
        out.push(resolve_pdfium_path(
            &root.join("third_party").join("pdfium"),
        ));
    };

    if let Ok(cwd) = env::current_dir() {
        push_candidates_for_root(&cwd);
    }
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            push_candidates_for_root(exe_dir);
        }
    }

    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for p in out {
        let key = p.to_string_lossy().to_string();
        if seen.insert(key) {
            unique.push(p);
        }
    }
    unique
}

fn scale_to_pixels(value: u32, scale: f32) -> u32 {
    ((value as f32) * scale).round().max(1.0) as u32
}

fn parse_image_args(argv: &[String]) -> Result<ImageArgs, String> {
    if argv.len() == 1 {
        return Err(usage().to_string());
    }

    let mut image_path: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut mmproj_path: Option<PathBuf> = None;
    let mut out_md: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut scale: f32 = 2.0;
    let mut oversample: f32 = 1.5;
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut n_predict: i32 = 5000;
    let mut n_ctx_total: u32 = 32768;
    let mut n_threads: i32 = 8;
    let mut batch_size: u32 = 2048;
    let mut max_retries: usize = 2;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--image" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--image requires a value".to_string());
                }
                image_path = Some(PathBuf::from(&argv[i]));
            }
            "--model" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--model requires a value".to_string());
                }
                model_path = Some(PathBuf::from(&argv[i]));
            }
            "--mmproj" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--mmproj requires a value".to_string());
                }
                mmproj_path = Some(PathBuf::from(&argv[i]));
            }
            "--out-md" | "--out" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--out-md/--out requires a value".to_string());
                }
                out_md = Some(PathBuf::from(&argv[i]));
            }
            "--out-dir" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--out-dir requires a value".to_string());
                }
                out_dir = Some(PathBuf::from(&argv[i]));
            }
            "--scale" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--scale requires a value".to_string());
                }
                scale = argv[i]
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid --scale value: {}", argv[i]))?;
            }
            "--oversample" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--oversample requires a value".to_string());
                }
                oversample = argv[i]
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid --oversample value: {}", argv[i]))?;
            }
            "--prompt" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--prompt requires a value".to_string());
                }
                prompt = argv[i].clone();
            }
            "--n-predict" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--n-predict requires a value".to_string());
                }
                n_predict = argv[i]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid --n-predict value: {}", argv[i]))?;
            }
            "--n-ctx" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--n-ctx requires a value".to_string());
                }
                n_ctx_total = argv[i]
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --n-ctx value: {}", argv[i]))?;
            }
            "--threads" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--threads requires a value".to_string());
                }
                n_threads = argv[i]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid --threads value: {}", argv[i]))?;
            }
            "--batch-size" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--batch-size requires a value".to_string());
                }
                batch_size = argv[i]
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --batch-size value: {}", argv[i]))?;
            }
            "--max-retries" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--max-retries requires a value".to_string());
                }
                max_retries = argv[i]
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --max-retries value: {}", argv[i]))?;
            }
            "--parallel" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--parallel requires a value".to_string());
                }
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("Unknown argument in --image mode: {other}\n\n{}", usage())),
        }
        i += 1;
    }

    if scale <= 0.0 {
        return Err("--scale must be > 0".to_string());
    }
    if oversample <= 0.0 {
        return Err("--oversample must be > 0".to_string());
    }
    if n_predict <= 0 {
        return Err("--n-predict must be > 0".to_string());
    }
    if n_ctx_total == 0 {
        return Err("--n-ctx must be > 0".to_string());
    }
    if n_threads <= 0 {
        return Err("--threads must be > 0".to_string());
    }
    if batch_size == 0 {
        return Err("--batch-size must be > 0".to_string());
    }

    let image_path = image_path.ok_or_else(|| "--image is required".to_string())?;
    let model_path = model_path.ok_or_else(|| "--model is required".to_string())?;
    let mmproj_path = mmproj_path.ok_or_else(|| "--mmproj is required".to_string())?;

    let output_md = if let Some(p) = out_md {
        p
    } else if let Some(dir) = out_dir {
        let stem = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        dir.join(format!("{stem}.md"))
    } else {
        image_path.with_extension("md")
    };

    Ok(ImageArgs {
        image_path,
        model_path,
        mmproj_path,
        output_md,
        scale,
        oversample,
        prompt,
        n_predict,
        n_ctx_total,
        n_threads,
        batch_size,
        max_retries,
    })
}

fn preprocess_image_to_png(
    image_path: &Path,
    scale: f32,
    oversample: f32,
) -> Result<(Vec<u8>, (u32, u32), (u32, u32), (u32, u32)), Box<dyn std::error::Error>> {
    if !image_path.exists() {
        return Err(format!("Image not found: {}", image_path.display()).into());
    }

    let image = image::open(image_path)?;
    let (orig_w, orig_h) = image.dimensions();
    let temp_w = scale_to_pixels(orig_w, scale * oversample);
    let temp_h = scale_to_pixels(orig_h, scale * oversample);
    let target_w = scale_to_pixels(orig_w, scale);
    let target_h = scale_to_pixels(orig_h, scale);

    let temp = image.resize_exact(temp_w, temp_h, FilterType::CatmullRom);
    let final_img = temp.resize_exact(target_w, target_h, FilterType::CatmullRom);
    let mut cursor = Cursor::new(Vec::new());
    final_img.write_to(&mut cursor, ImageFormat::Png)?;

    Ok((
        cursor.into_inner(),
        (orig_w, orig_h),
        (temp_w, temp_h),
        (target_w, target_h),
    ))
}

fn parse_pages(raw: &str) -> Result<Vec<usize>, String> {
    let mut set = BTreeSet::new();
    for part in raw.split(',') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((a, b)) = token.split_once('-') {
            let start: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("Invalid page range start: {a}"))?;
            let end: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("Invalid page range end: {b}"))?;
            if start == 0 || end == 0 || start > end {
                return Err(format!("Invalid page range: {token}"));
            }
            for p in start..=end {
                set.insert(p);
            }
        } else {
            let page: usize = token
                .parse()
                .map_err(|_| format!("Invalid page number: {token}"))?;
            if page == 0 {
                return Err("Page numbers are 1-based and must be >= 1".to_string());
            }
            set.insert(page);
        }
    }
    if set.is_empty() {
        return Err("No valid pages were parsed".to_string());
    }
    Ok(set.into_iter().collect())
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    if argv.len() == 1 {
        return Err(usage().to_string());
    }

    let mut pdf_path: Option<PathBuf> = None;
    let mut pdfium_dll: Option<PathBuf> = env::var("PDFIUM_LIB")
        .ok()
        .or_else(|| env::var("PDFIUM_DLL").ok())
        .map(PathBuf::from);
    let mut model_path: Option<PathBuf> = None;
    let mut mmproj_path: Option<PathBuf> = None;
    let mut out_md: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut pages: Option<Vec<usize>> = None;
    let mut scale: f32 = 2.0;
    let mut oversample: f32 = 1.5;
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut n_predict: i32 = 5000;
    let mut n_ctx_total: u32 = 32768;
    let mut n_threads: i32 = 8;
    let mut batch_size: u32 = 2048;
    let mut parallel: usize = 4;
    let mut max_retries: usize = 2;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdf" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--pdf requires a value".to_string());
                }
                pdf_path = Some(PathBuf::from(&argv[i]));
            }
            "--pdfium-lib" | "--pdfium-dll" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--pdfium-lib/--pdfium-dll requires a value".to_string());
                }
                pdfium_dll = Some(PathBuf::from(&argv[i]));
            }
            "--model" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--model requires a value".to_string());
                }
                model_path = Some(PathBuf::from(&argv[i]));
            }
            "--mmproj" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--mmproj requires a value".to_string());
                }
                mmproj_path = Some(PathBuf::from(&argv[i]));
            }
            "--out-md" | "--out" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--out-md/--out requires a value".to_string());
                }
                out_md = Some(PathBuf::from(&argv[i]));
            }
            "--out-dir" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--out-dir requires a value".to_string());
                }
                out_dir = Some(PathBuf::from(&argv[i]));
            }
            "--pages" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--pages requires a value".to_string());
                }
                pages = Some(parse_pages(&argv[i])?);
            }
            "--scale" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--scale requires a value".to_string());
                }
                scale = argv[i]
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid --scale value: {}", argv[i]))?;
            }
            "--oversample" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--oversample requires a value".to_string());
                }
                oversample = argv[i]
                    .parse::<f32>()
                    .map_err(|_| format!("Invalid --oversample value: {}", argv[i]))?;
            }
            "--prompt" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--prompt requires a value".to_string());
                }
                prompt = argv[i].clone();
            }
            "--n-predict" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--n-predict requires a value".to_string());
                }
                n_predict = argv[i]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid --n-predict value: {}", argv[i]))?;
            }
            "--n-ctx" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--n-ctx requires a value".to_string());
                }
                n_ctx_total = argv[i]
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --n-ctx value: {}", argv[i]))?;
            }
            "--threads" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--threads requires a value".to_string());
                }
                n_threads = argv[i]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid --threads value: {}", argv[i]))?;
            }
            "--batch-size" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--batch-size requires a value".to_string());
                }
                batch_size = argv[i]
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --batch-size value: {}", argv[i]))?;
            }
            "--parallel" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--parallel requires a value".to_string());
                }
                parallel = argv[i]
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --parallel value: {}", argv[i]))?;
            }
            "--max-retries" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--max-retries requires a value".to_string());
                }
                max_retries = argv[i]
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --max-retries value: {}", argv[i]))?;
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("Unknown argument: {other}\n\n{}", usage())),
        }
        i += 1;
    }

    if scale <= 0.0 {
        return Err("--scale must be > 0".to_string());
    }
    if oversample <= 0.0 {
        return Err("--oversample must be > 0".to_string());
    }
    if n_predict <= 0 {
        return Err("--n-predict must be > 0".to_string());
    }
    if n_ctx_total == 0 {
        return Err("--n-ctx must be > 0".to_string());
    }
    if n_threads <= 0 {
        return Err("--threads must be > 0".to_string());
    }
    if batch_size == 0 {
        return Err("--batch-size must be > 0".to_string());
    }
    if parallel == 0 {
        return Err("--parallel must be > 0".to_string());
    }

    let pdf_path = pdf_path.ok_or_else(|| "--pdf is required".to_string())?;
    let pdfium_dll = if let Some(p) = pdfium_dll {
        let resolved = resolve_pdfium_path(&p);
        if !resolved.exists() {
            return Err(format!(
                "PDFium library not found at explicit/env path: {}",
                resolved.display()
            ));
        }
        resolved
    } else if let Some(found) = default_pdfium_candidates().into_iter().find(|p| p.exists()) {
        found
    } else {
        return Err(
            "Could not locate a PDFium library. Provide --pdfium-lib, set PDFIUM_LIB/PDFIUM_DLL, or bundle PDFium under vendor/pdfium next to the app.".to_string(),
        );
    };
    let model_path = model_path.ok_or_else(|| "--model is required".to_string())?;
    let mmproj_path = mmproj_path.ok_or_else(|| "--mmproj is required".to_string())?;

    let output_md = if let Some(p) = out_md {
        p
    } else if let Some(dir) = out_dir {
        let stem = pdf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        dir.join(format!("{stem}.md"))
    } else {
        pdf_path.with_extension("md")
    };

    Ok(Args {
        pdf_path,
        pdfium_dll,
        model_path,
        mmproj_path,
        output_md,
        pages,
        scale,
        oversample,
        prompt,
        n_predict,
        n_ctx_total,
        n_threads,
        batch_size,
        parallel,
        max_retries,
    })
}

fn points_to_pixels(points: f32, scale: f32) -> i32 {
    (points * scale).round().max(1.0) as i32
}

fn render_pages_to_memory(args: &Args) -> Result<Vec<RenderedPage>, Box<dyn std::error::Error>> {
    if !args.pdfium_dll.exists() {
        return Err(format!("PDFium library not found: {}", args.pdfium_dll.display()).into());
    }
    if !args.pdf_path.exists() {
        return Err(format!("PDF not found: {}", args.pdf_path.display()).into());
    }

    let pdfium = Pdfium::new(Pdfium::bind_to_library(&args.pdfium_dll)?);
    let document = pdfium.load_pdf_from_file(&args.pdf_path, None)?;
    let total_pages = document.pages().len() as usize;

    let page_numbers = match &args.pages {
        Some(pages) => pages.clone(),
        None => (1..=total_pages).collect(),
    };

    let mut rendered = Vec::new();
    for page_number in page_numbers {
        if page_number == 0 || page_number > total_pages {
            return Err(format!(
                "Requested page {} is out of bounds. Document has {} pages.",
                page_number, total_pages
            )
            .into());
        }

        let page_index: u16 = (page_number - 1)
            .try_into()
            .map_err(|_| format!("Page index out of range for Pdfium: {}", page_number - 1))?;
        let page = document.pages().get(page_index)?;

        let width_pts = page.width().value;
        let height_pts = page.height().value;

        let target_w = points_to_pixels(width_pts, args.scale);
        let target_h = points_to_pixels(height_pts, args.scale);
        let temp_w = points_to_pixels(width_pts, args.scale * args.oversample);
        let temp_h = points_to_pixels(height_pts, args.scale * args.oversample);

        let render_config = PdfRenderConfig::new().set_fixed_size(temp_w, temp_h);
        let bitmap = page.render_with_config(&render_config)?;
        let temp_img = bitmap.as_image();
        let final_img = temp_img.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom);

        let mut cursor = Cursor::new(Vec::new());
        final_img.write_to(&mut cursor, ImageFormat::Png)?;
        let png_bytes = cursor.into_inner();

        println!(
            "rendered page {:04} in RAM (temp {}x{}, final {}x{}, bytes={})",
            page_number,
            temp_w,
            temp_h,
            target_w,
            target_h,
            png_bytes.len()
        );

        rendered.push(RenderedPage {
            page: page_number as u32,
            png_bytes,
        });
    }

    Ok(rendered)
}

fn tokenize_words_for_loop_detection(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn normalize_line_for_loop_detection(line: &str) -> String {
    let mut out = String::new();
    let mut prev_space = false;
    for ch in line.chars() {
        let mapped = if ch.is_alphanumeric() { ch } else { ' ' };
        if mapped.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            for lc in mapped.to_lowercase() {
                out.push(lc);
            }
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn detect_consecutive_line_loop(text: &str, min_words: usize, min_repeats: usize) -> bool {
    let mut prev = String::new();
    let mut reps = 0usize;

    for raw in text.lines() {
        let line = normalize_line_for_loop_detection(raw);
        if line.is_empty() {
            prev.clear();
            reps = 0;
            continue;
        }
        if tokenize_words_for_loop_detection(&line).len() < min_words {
            prev.clear();
            reps = 0;
            continue;
        }
        if line == prev {
            reps += 1;
        } else {
            prev = line;
            reps = 1;
        }
        if reps >= min_repeats {
            return true;
        }
    }
    false
}

fn detect_consecutive_word_span_loop(
    text: &str,
    min_words: usize,
    max_words: usize,
    min_repeats: usize,
) -> bool {
    let words = tokenize_words_for_loop_detection(text);
    let n = words.len();
    if n < min_words * min_repeats {
        return false;
    }
    let max_span = std::cmp::min(max_words, n / min_repeats);
    if max_span < min_words {
        return false;
    }

    for span in (min_words..=max_span).rev() {
        let limit = n.saturating_sub(span * min_repeats);
        let mut i = 0usize;
        while i <= limit {
            let phrase = &words[i..(i + span)];
            let mut reps = 1usize;
            let mut j = i + span;
            while j + span <= n && words[j..(j + span)] == *phrase {
                reps += 1;
                j += span;
            }
            if reps >= min_repeats {
                return true;
            }
            i += 1;
        }
    }
    false
}

fn has_loop(text: &str) -> bool {
    detect_consecutive_line_loop(text, 6, 7) || detect_consecutive_word_span_loop(text, 6, 24, 7)
}

fn run_image_to_markdown_cli_from_args(
    argv: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let args =
        parse_image_args(argv).map_err(|msg| io::Error::new(io::ErrorKind::InvalidInput, msg))?;

    if !args.model_path.exists() {
        return Err(format!("Model not found: {}", args.model_path.display()).into());
    }
    if !args.mmproj_path.exists() {
        return Err(format!("MMProj not found: {}", args.mmproj_path.display()).into());
    }

    if let Some(parent) = args.output_md.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    println!("starting conversion");
    println!("image: {}", args.image_path.display());
    println!("out_md: {}", args.output_md.display());
    println!("render: scale={} oversample={}", args.scale, args.oversample);
    println!(
        "inference: parallel=1 ctx={} batch={} n_predict={} retries={}",
        args.n_ctx_total, args.batch_size, args.n_predict, args.max_retries
    );

    let (png_bytes, (orig_w, orig_h), (temp_w, temp_h), (final_w, final_h)) =
        preprocess_image_to_png(&args.image_path, args.scale, args.oversample)?;
    println!(
        "rendered image in RAM (orig {}x{}, temp {}x{}, final {}x{}, bytes={})",
        orig_w,
        orig_h,
        temp_w,
        temp_h,
        final_w,
        final_h,
        png_bytes.len()
    );

    let bridge = SharedBridge::new(
        &args.model_path,
        &args.mmproj_path,
        args.n_ctx_total,
        args.n_threads,
        args.batch_size,
        1,
    )?;

    let started = Instant::now();
    let mut attempts = 0usize;
    let response = loop {
        let retry_attempt = attempts > 0;
        let result = bridge.run_vlm(&args.prompt, &png_bytes, args.n_predict, retry_attempt);
        attempts += 1;
        match result {
            Ok(r) => {
                let loop_detected = has_loop(&r.markdown);
                let bad = !r.eos_reached || r.truncated || loop_detected;
                if bad && attempts <= args.max_retries {
                    println!(
                        "image retry {}/{} (eos_reached={}, truncated={}, loop_detected={})",
                        attempts,
                        args.max_retries,
                        r.eos_reached,
                        r.truncated,
                        loop_detected
                    );
                    continue;
                }
                if bad {
                    return Err(format!(
                        "quality gate failed after {} attempts: eos_reached={}, truncated={}, loop_detected={}",
                        attempts, r.eos_reached, r.truncated, loop_detected
                    )
                    .into());
                }
                break r;
            }
            Err(err) => {
                if attempts <= args.max_retries {
                    println!(
                        "image retry {}/{} due to error: {}",
                        attempts, args.max_retries, err
                    );
                    continue;
                }
                return Err(format!(
                    "inference failed after {} attempts: {}",
                    attempts, err
                )
                .into());
            }
        }
    };

    let elapsed = started.elapsed().as_secs_f64();
    let retries = attempts.saturating_sub(1);

    let mut final_md = String::new();
    final_md.push_str("<--page1-->\n\n");
    final_md.push_str(&response.markdown);
    if !final_md.ends_with('\n') {
        final_md.push('\n');
    }
    fs::write(&args.output_md, final_md)?;

    println!("done");
    println!("items: 1");
    println!("total_seconds: {:.3}", elapsed);
    println!("avg_seconds_per_item: {:.3}", elapsed);
    println!("total_output_chars: {}", response.markdown.chars().count());
    println!("total_retries_used: {}", retries);
    println!("output: {}", args.output_md.display());

    Ok(())
}

pub fn run_pdf_to_markdown_cli_from_args(argv: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if has_flag(argv, "--image") {
        return run_image_to_markdown_cli_from_args(argv);
    }

    let args =
        parse_args(argv).map_err(|msg| io::Error::new(io::ErrorKind::InvalidInput, msg))?;

    if !args.model_path.exists() {
        return Err(format!("Model not found: {}", args.model_path.display()).into());
    }
    if !args.mmproj_path.exists() {
        return Err(format!("MMProj not found: {}", args.mmproj_path.display()).into());
    }

    if let Some(parent) = args.output_md.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    println!("starting conversion");
    println!("pdf: {}", args.pdf_path.display());
    println!("out_md: {}", args.output_md.display());
    println!("render: scale={} oversample={}", args.scale, args.oversample);
    println!(
        "inference: parallel={} ctx={} batch={} n_predict={} retries={}",
        args.parallel, args.n_ctx_total, args.batch_size, args.n_predict, args.max_retries
    );

    let rendered_pages = render_pages_to_memory(&args)?;
    if rendered_pages.is_empty() {
        return Err("No pages rendered".into());
    }

    let bridge = Arc::new(SharedBridge::new(
        &args.model_path,
        &args.mmproj_path,
        args.n_ctx_total,
        args.n_threads,
        args.batch_size,
        args.parallel,
    )?);

    let mut q = VecDeque::new();
    for p in rendered_pages {
        q.push_back(p);
    }
    let queue = Arc::new(Mutex::new(q));
    let rows: Arc<Mutex<Vec<Row>>> = Arc::new(Mutex::new(Vec::new()));
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let started_all = Instant::now();
    thread::scope(|scope| {
        for _ in 0..args.parallel {
            let queue_ref = Arc::clone(&queue);
            let rows_ref = Arc::clone(&rows);
            let errors_ref = Arc::clone(&errors);
            let bridge_ref = Arc::clone(&bridge);
            let prompt = args.prompt.clone();
            let n_predict = args.n_predict;
            let max_retries = args.max_retries;

            scope.spawn(move || loop {
                let maybe_page = {
                    let mut guard = queue_ref.lock().expect("queue lock poisoned");
                    guard.pop_front()
                };

                let page_item = match maybe_page {
                    Some(p) => p,
                    None => break,
                };

                let mut seconds = 0.0f64;
                let mut attempts = 0usize;

                let response = loop {
                    let retry_attempt = attempts > 0;
                    let started = Instant::now();
                    let result = bridge_ref.run_vlm(
                        &prompt,
                        &page_item.png_bytes,
                        n_predict,
                        retry_attempt,
                    );
                    seconds += started.elapsed().as_secs_f64();
                    attempts += 1;

                    match result {
                        Ok(r) => {
                            let loop_detected = has_loop(&r.markdown);
                            let bad = !r.eos_reached || r.truncated || loop_detected;
                            if bad && attempts <= max_retries {
                                println!(
                                    "page {:04} retry {}/{} (eos_reached={}, truncated={}, loop_detected={})",
                                    page_item.page,
                                    attempts,
                                    max_retries,
                                    r.eos_reached,
                                    r.truncated,
                                    loop_detected
                                );
                                continue;
                            }
                            if bad {
                                let mut guard = errors_ref.lock().expect("errors lock poisoned");
                                guard.push(format!(
                                    "quality gate failed for page {:04} after {} attempts: eos_reached={}, truncated={}, loop_detected={}",
                                    page_item.page,
                                    attempts,
                                    r.eos_reached,
                                    r.truncated,
                                    loop_detected
                                ));
                                break None;
                            }
                            break Some(r);
                        }
                        Err(err) => {
                            if attempts <= max_retries {
                                println!(
                                    "page {:04} retry {}/{} due to error: {}",
                                    page_item.page,
                                    attempts,
                                    max_retries,
                                    err
                                );
                                continue;
                            }
                            let mut guard = errors_ref.lock().expect("errors lock poisoned");
                            guard.push(format!(
                                "inference failed for page {:04} after {} attempts: {}",
                                page_item.page,
                                attempts,
                                err
                            ));
                            break None;
                        }
                    }
                };

                let response = match response {
                    Some(r) => r,
                    None => continue,
                };

                println!(
                    "page {:04} done ({:.3}s, attempts={})",
                    page_item.page, seconds, attempts
                );

                let mut guard = rows_ref.lock().expect("rows lock poisoned");
                guard.push(Row {
                    page: page_item.page,
                    seconds,
                    attempts,
                    chars: response.markdown.chars().count(),
                    markdown: response.markdown,
                });
            });
        }
    });

    let total_seconds = started_all.elapsed().as_secs_f64();

    let errors = errors.lock().expect("errors lock poisoned");
    if !errors.is_empty() {
        let mut msg = String::from("One or more page runs failed:\n");
        for e in errors.iter() {
            msg.push_str(" - ");
            msg.push_str(e);
            msg.push('\n');
        }
        return Err(msg.into());
    }
    drop(errors);

    let mut rows = {
        let guard = rows.lock().expect("rows lock poisoned");
        guard.clone()
    };
    rows.sort_by_key(|r| r.page);

    let mut final_md = String::new();
    for r in &rows {
        final_md.push_str(&format!("<--page{}-->\n\n", r.page));
        final_md.push_str(&r.markdown);
        if !final_md.ends_with('\n') {
            final_md.push('\n');
        }
        final_md.push('\n');
    }
    fs::write(&args.output_md, final_md)?;

    let total_chars: usize = rows.iter().map(|r| r.chars).sum();
    let total_attempts: usize = rows.iter().map(|r| r.attempts).sum();
    let total_retries = total_attempts.saturating_sub(rows.len());
    let avg_seconds = if rows.is_empty() {
        0.0
    } else {
        total_seconds / rows.len() as f64
    };

    println!("done");
    println!("pages: {}", rows.len());
    println!("total_seconds: {:.3}", total_seconds);
    println!("avg_seconds_per_page: {:.3}", avg_seconds);
    println!("total_output_chars: {}", total_chars);
    println!("total_retries_used: {}", total_retries);
    println!("output: {}", args.output_md.display());

    Ok(())
}
