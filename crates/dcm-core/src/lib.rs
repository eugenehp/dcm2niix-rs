//! Shared primitives for dcm2niix-rs.
//!
//! Errors, 4×4 matrices / DICOM→NIfTI geometry helpers, exit codes, and the
//! version string reported by the CLI. No DICOM I/O lives here.

pub mod dicom_time;
pub mod error;
pub mod exit;
pub mod matrix;

pub use dicom_time::{dicom_time_to_sec, format_printf_g, format_printf_g_f64, snap_f32};
pub use error::{Error, Result};
pub use exit::Exit;
pub use matrix::Matrix4;

/// Version reported by `dcm2niix --version` / `-h`.
///
/// The date stamp tracks this port, not every C++ tag. The `kDCMvers` suffix
/// in C names the JPEG backends; here decode comes from `dicom-pixeldata`.
pub const VERSION: &str = "v1.0.20260822 dcm2niix-rs (JP2/HTJ2K:OpenJPEG) (JP-LS:CharLS)";
pub const VERSION_DATE: &str = "v1.0.20260822";
