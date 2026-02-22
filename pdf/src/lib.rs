mod core;

pub use core::*;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(name = "pdf")]
#[command(about = "Fast native PDF text extraction using PDFium (C++ core).")]
struct Cli {
    #[arg(
        long,
        global = true,
        value_name = "PATH",
        help = "Path to PDFium library file (e.g. pdfium.dll/libpdfium.dylib/libpdfium.so), or a folder containing it"
    )]
    pdfium_lib: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Extract(ExtractArgs),
}

#[derive(Args, Debug, Clone)]
struct ExtractArgs {
    #[arg(
        long,
        value_name = "PATH",
        help = "Single PDF file or directory of PDFs"
    )]
    input: PathBuf,

    #[arg(
        long,
        value_name = "PATH",
        help = "Output file (single input) or output directory"
    )]
    output: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = false,
        help = "Only scan the top level when input is a directory"
    )]
    non_recursive: bool,

    #[arg(long, value_name = "PASSWORD")]
    password: Option<String>,

    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

pub fn run_pdf_cli_from_args(argv: &[String]) -> Result<()> {
    let cli = Cli::try_parse_from(argv)
        .with_context(|| "invalid arguments for pdf".to_string())?;

    match cli.command {
        Commands::Extract(args) => run_extract(&cli.pdfium_lib, args),
    }
}

fn run_extract(pdfium_lib: &Option<PathBuf>, args: ExtractArgs) -> Result<()> {
    let recursive = !args.non_recursive;
    let input = args.input.canonicalize().with_context(|| {
        format!(
            "Unable to resolve input path: {}",
            clean_display_path(&args.input)
        )
    })?;

    let extractor = PdfiumTextExtractor::new(pdfium_lib.as_deref())?;
    let pdf_files = collect_pdf_files(&input, recursive)?;

    if input.is_dir() && args.output.is_none() {
        bail!("--output is required when extracting a directory");
    }

    if input.is_file() && pdf_files.len() == 1 && args.output.is_none() {
        let doc = extractor
            .extract_pdf_to_markdown(&pdf_files[0], args.password.as_deref())
            .with_context(|| format!("Failed to extract {}", clean_display_path(&pdf_files[0])))?;
        print!("{}", doc.text);
        return Ok(());
    }

    let output_base = args
        .output
        .as_ref()
        .map(|p| absolute_from_cwd(p))
        .transpose()?;

    let mut succeeded = 0usize;

    for pdf in &pdf_files {
        let doc = extractor
            .extract_pdf_to_markdown(pdf, args.password.as_deref())
            .with_context(|| format!("Failed to extract {}", clean_display_path(pdf)))?;

        let out_path = resolve_output_path(&input, pdf, output_base.as_deref())?;
        write_utf8_bom(&out_path, &doc.text, args.overwrite)?;
        println!(
            "Wrote {} (pages: {}, chars: {})",
            clean_display_path(&out_path),
            doc.pages,
            doc.chars
        );
        succeeded += 1;
    }

    println!("Done. Extracted {succeeded} PDF(s).");
    Ok(())
}

fn collect_pdf_files(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if !is_pdf(input) {
            bail!("Input file is not a PDF: {}", clean_display_path(input));
        }
        return Ok(vec![input.to_path_buf()]);
    }

    if !input.is_dir() {
        bail!("Input path must be a PDF file or a directory");
    }

    let walker = if recursive {
        WalkDir::new(input)
    } else {
        WalkDir::new(input).max_depth(1)
    };

    let mut files = Vec::new();
    for entry in walker {
        let entry = entry.with_context(|| {
            format!(
                "Error while scanning directory {}",
                clean_display_path(input)
            )
        })?;
        if entry.file_type().is_file() && is_pdf(entry.path()) {
            files.push(entry.path().to_path_buf());
        }
    }

    files.sort_unstable_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    Ok(files)
}

fn resolve_output_path(input_root: &Path, pdf: &Path, output: Option<&Path>) -> Result<PathBuf> {
    let Some(output_path) = output else {
        bail!("Missing output path");
    };

    if input_root.is_file() {
        if output_path.is_dir() {
            return Ok(output_path.join(filename_with_md_extension(pdf)?));
        }
        return Ok(output_path.to_path_buf());
    }

    Ok(output_path.join(relative_output_path(input_root, pdf)?))
}

fn relative_output_path(input_root: &Path, pdf: &Path) -> Result<PathBuf> {
    let relative = pdf.strip_prefix(input_root).with_context(|| {
        format!(
            "Could not compute relative path for {}",
            clean_display_path(pdf)
        )
    })?;
    let mut out = relative.to_path_buf();
    out.set_extension("md");
    Ok(out)
}

fn filename_with_md_extension(pdf: &Path) -> Result<PathBuf> {
    let stem = pdf
        .file_stem()
        .ok_or_else(|| anyhow!("Missing file stem for {}", clean_display_path(pdf)))?;
    let mut file = PathBuf::from(stem);
    file.set_extension("md");
    Ok(file)
}

fn sanitize_for_c_string(input: &str) -> String {
    input.replace('\0', " ")
}

unsafe fn argv_from_c(argc: i32, argv: *const *const c_char) -> Result<Vec<String>> {
    if argc < 0 {
        bail!("argc must be >= 0");
    }
    if argc > 0 && argv.is_null() {
        bail!("argv is null");
    }

    let mut out = Vec::with_capacity(argc as usize);
    for i in 0..(argc as isize) {
        let ptr_item = unsafe { *argv.offset(i) };
        if ptr_item.is_null() {
            bail!("argv[{i}] is null");
        }
        let s = unsafe { CStr::from_ptr(ptr_item) }
            .to_string_lossy()
            .into_owned();
        out.push(s);
    }
    Ok(out)
}

unsafe fn set_out_error(out_error: *mut *mut c_char, message: Option<&str>) {
    if out_error.is_null() {
        return;
    }
    unsafe {
        *out_error = ptr::null_mut();
    }
    if let Some(msg) = message {
        let safe = sanitize_for_c_string(msg);
        if let Ok(c) = CString::new(safe) {
            unsafe {
                *out_error = c.into_raw();
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pdf_run_from_argv(
    argc: i32,
    argv: *const *const c_char,
    out_error: *mut *mut c_char,
) -> i32 {
    unsafe {
        set_out_error(out_error, None);
    }

    let args = match unsafe { argv_from_c(argc, argv) } {
        Ok(v) => v,
        Err(e) => {
            unsafe {
                set_out_error(out_error, Some(&e.to_string()));
            }
            return -1;
        }
    };

    match run_pdf_cli_from_args(&args) {
        Ok(()) => 0,
        Err(e) => {
            unsafe {
                set_out_error(out_error, Some(&e.to_string()));
            }
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pdf_free_c_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}
