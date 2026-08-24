//! Upstream dcm2niix C++ core (`../dcm2niix/console`) as a static library.
//!
//! **Parity reference only.** Product conversion is Rust/rlx (`dcm-convert`).
//! Build `cargo build -p dcm-sys` for the `dcm2niix-ffi` binary (differential checks).

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

extern "C" {
    fn dcm2niix_run(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

/// Run upstream dcm2niix with the same argv layout as the C++ `main`.
pub fn run<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let c_args: Vec<CString> = args
        .into_iter()
        .map(|s| CString::new(s.as_ref()).expect("argument contains interior NUL"))
        .collect();
    let mut argv: Vec<*mut c_char> = c_args
        .iter()
        .map(|s| s.as_ptr() as *mut c_char)
        .collect();
    argv.push(std::ptr::null_mut());
    unsafe { dcm2niix_run(c_args.len() as c_int, argv.as_mut_ptr()) }
}
