//! Upstream C++ dcm2niix — **parity reference only**.
//!
//! Build with `--features ffi`. The product converter is the Rust/rlx
//! `dcm2niix` binary; this wrapper exists so `dcm-parity` (and developers)
//! can differential-test against Chris Rorden's C++ core.

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    ExitCode::from(dcm_sys::run(env::args()) as u8)
}
