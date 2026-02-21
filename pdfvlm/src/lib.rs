pub mod llama_bridge_ffi;
pub mod pdf_to_markdown;

pub use pdf_to_markdown::run_pdf_to_markdown_cli_from_args;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

fn sanitize_for_c_string(input: &str) -> String {
    input.replace('\0', " ")
}

unsafe fn argv_from_c(argc: i32, argv: *const *const c_char) -> Result<Vec<String>, String> {
    if argc < 0 {
        return Err("argc must be >= 0".to_string());
    }
    if argc > 0 && argv.is_null() {
        return Err("argv is null".to_string());
    }

    let mut out = Vec::with_capacity(argc as usize);
    for i in 0..(argc as isize) {
        let ptr_item = unsafe { *argv.offset(i) };
        if ptr_item.is_null() {
            return Err(format!("argv[{i}] is null"));
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

#[no_mangle]
pub unsafe extern "C" fn pdfvlm_run_from_argv(
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
                set_out_error(out_error, Some(&e));
            }
            return -1;
        }
    };

    match run_pdf_to_markdown_cli_from_args(&args) {
        Ok(()) => 0,
        Err(e) => {
            unsafe {
                set_out_error(out_error, Some(&e.to_string()));
            }
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pdfvlm_free_c_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}

