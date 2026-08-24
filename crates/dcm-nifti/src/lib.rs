//! NIfTI-1 header and `.nii` / `.nii.gz` / `.nii.zst` writer.
//!
//! Layout matches `nifti1.h`. `vox_offset` is 352 (348-byte header + 4-byte
//! extender), same as dcm2niix `nii_saveNII`. Optional ecode-44 extension
//! blocks support NIfTI-MRS.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use dcm_core::error::{Error, Result};
use dcm_core::matrix::{mat44_to_quatern, Matrix4};
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;

pub const DT_UINT8: i16 = 2;
pub const DT_INT16: i16 = 4;
pub const DT_INT32: i16 = 8;
pub const DT_FLOAT32: i16 = 16;
pub const DT_COMPLEX64: i16 = 32;
pub const DT_FLOAT64: i16 = 64;
pub const DT_RGB24: i16 = 128;
pub const DT_UINT16: i16 = 512;

pub const NIFTI_XFORM_UNKNOWN: i16 = 0;
pub const NIFTI_XFORM_SCANNER_ANAT: i16 = 1;
pub const NIFTI_UNITS_MM: u8 = 2;
pub const NIFTI_UNITS_SEC: u8 = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nifti1Header {
    pub sizeof_hdr: i32,
    pub data_type: [u8; 10],
    pub db_name: [u8; 18],
    pub extents: i32,
    pub session_error: i16,
    pub regular: u8,
    pub dim_info: u8,
    pub dim: [i16; 8],
    pub intent_p1: f32,
    pub intent_p2: f32,
    pub intent_p3: f32,
    pub intent_code: i16,
    pub datatype: i16,
    pub bitpix: i16,
    pub slice_start: i16,
    pub pixdim: [f32; 8],
    pub vox_offset: f32,
    pub scl_slope: f32,
    pub scl_inter: f32,
    pub slice_end: i16,
    pub slice_code: u8,
    pub xyzt_units: u8,
    pub cal_max: f32,
    pub cal_min: f32,
    pub slice_duration: f32,
    pub toffset: f32,
    pub glmax: i32,
    pub glmin: i32,
    pub descrip: [u8; 80],
    pub aux_file: [u8; 24],
    pub qform_code: i16,
    pub sform_code: i16,
    pub quatern_b: f32,
    pub quatern_c: f32,
    pub quatern_d: f32,
    pub qoffset_x: f32,
    pub qoffset_y: f32,
    pub qoffset_z: f32,
    pub srow_x: [f32; 4],
    pub srow_y: [f32; 4],
    pub srow_z: [f32; 4],
    pub intent_name: [u8; 16],
    pub magic: [u8; 4],
}

impl Default for Nifti1Header {
    fn default() -> Self {
        let mut h = unsafe { std::mem::zeroed::<Nifti1Header>() };
        h.sizeof_hdr = 348;
        h.regular = 114;
        h.magic = *b"n+1\0";
        h.vox_offset = 352.0;
        h.pixdim[0] = 1.0;
        h.xyzt_units = NIFTI_UNITS_MM + NIFTI_UNITS_SEC;
        h
    }
}

impl Nifti1Header {
    pub fn set_descrip(&mut self, s: &str) {
        let b = s.as_bytes();
        let n = b.len().min(79);
        self.descrip[..n].copy_from_slice(&b[..n]);
        self.descrip[79] = 0;
    }

    pub fn set_aux(&mut self, s: &str) {
        let b = s.as_bytes();
        let n = b.len().min(23);
        self.aux_file[..n].copy_from_slice(&b[..n]);
    }

    pub fn set_sform(&mut self, m: &Matrix4) {
        for c in 0..4 {
            self.srow_x[c] = m.0[0][c] as f32;
            self.srow_y[c] = m.0[1][c] as f32;
            self.srow_z[c] = m.0[2][c] as f32;
        }
        let stored = Matrix4::from_rows([
            [
                self.srow_x[0] as f64,
                self.srow_x[1] as f64,
                self.srow_x[2] as f64,
                self.srow_x[3] as f64,
            ],
            [
                self.srow_y[0] as f64,
                self.srow_y[1] as f64,
                self.srow_y[2] as f64,
                self.srow_y[3] as f64,
            ],
            [
                self.srow_z[0] as f64,
                self.srow_z[1] as f64,
                self.srow_z[2] as f64,
                self.srow_z[3] as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let (qb, qc, qd, qx, qy, qz, _dx, _dy, _dz, qfac) = mat44_to_quatern(&stored);
        self.quatern_b = qb;
        self.quatern_c = qc;
        self.quatern_d = qd;
        self.qoffset_x = qx;
        self.qoffset_y = qy;
        self.qoffset_z = qz;
        self.pixdim[0] = qfac;
        self.sform_code = NIFTI_XFORM_SCANNER_ANAT;
        self.qform_code = NIFTI_XFORM_SCANNER_ANAT;
        // Preserve IEEE signed zeros — C++ mat44 / setQSForm keeps them.
    }

    pub fn as_bytes(&self) -> [u8; 348] {
        // Safety: Nifti1Header is repr(C), 348 bytes, no padding on common
        // targets (the official layout is packed by field sizes).
        debug_assert_eq!(std::mem::size_of::<Nifti1Header>(), 348);
        let mut out = [0u8; 348];
        let src = unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, 348)
        };
        out.copy_from_slice(src);
        out
    }

    pub fn nvox(&self) -> usize {
        let mut n = 1usize;
        let ndim = self.dim[0].max(1) as usize;
        for i in 1..=ndim.min(7) {
            n *= self.dim[i].max(1) as usize;
        }
        n
    }
}

pub fn write_nii(path: impl AsRef<Path>, hdr: &Nifti1Header, voxels: &[u8]) -> Result<()> {
    write_nii_with_ext(path, hdr, voxels, None)
}

/// Write NIfTI with an optional header extension block (e.g. NIfTI-MRS ecode 44 JSON).
pub fn write_nii_with_ext(
    path: impl AsRef<Path>,
    hdr: &Nifti1Header,
    voxels: &[u8],
    hdr_ext_json: Option<&str>,
) -> Result<()> {
    let path = path.as_ref();
    let gz = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gz"))
        .unwrap_or(false);
    if gz {
        write_nii_gz_with_ext(path, hdr, voxels, 6, hdr_ext_json)
    } else {
        write_nii_raw_with_ext(path, hdr, voxels, hdr_ext_json)
    }
}

fn extension_block(hdr_ext_json: Option<&str>) -> Result<(Vec<u8>, f32, Option<[u8; 16]>)> {
    let Some(json) = hdr_ext_json else {
        return Ok((vec![0, 0, 0, 0], 352.0, None));
    };
    let jlen = json.len();
    let mut esize = (8 + jlen + 15) & !15; // round up to multiple of 16
    if esize < 16 {
        esize = 16;
    }
    if esize > i32::MAX as usize {
        return Ok((vec![0, 0, 0, 0], 352.0, None));
    }
    let mut block = vec![0u8; esize + 4];
    block[0] = 1; // extension present
    block[4..8].copy_from_slice(&(esize as i32).to_le_bytes());
    block[8..12].copy_from_slice(&44i32.to_le_bytes()); // ecode 44 = NIfTI-MRS
    block[12..12 + jlen].copy_from_slice(json.as_bytes());
    let mut intent = [0u8; 16];
    let tag = b"mrs_v0_11";
    intent[..tag.len()].copy_from_slice(tag);
    Ok((block, 352.0 + esize as f32, Some(intent)))
}

fn write_header_and_data_ext<W: Write>(
    mut w: W,
    hdr: &Nifti1Header,
    voxels: &[u8],
    hdr_ext_json: Option<&str>,
) -> Result<()> {
    let (ext, vox_offset, intent) = extension_block(hdr_ext_json)?;
    let mut hdr = *hdr;
    hdr.vox_offset = vox_offset;
    if let Some(intent) = intent {
        hdr.intent_name = intent;
    }
    w.write_all(&hdr.as_bytes())
        .map_err(|e| Error::convert(format!("writing NIfTI header: {e}")))?;
    w.write_all(&ext)
        .map_err(|e| Error::convert(format!("writing NIfTI extender: {e}")))?;
    w.write_all(voxels)
        .map_err(|e| Error::convert(format!("writing NIfTI voxels: {e}")))?;
    w.flush()
        .map_err(|e| Error::convert(format!("flushing NIfTI: {e}")))?;
    Ok(())
}

fn write_nii_raw_with_ext(
    path: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    hdr_ext_json: Option<&str>,
) -> Result<()> {
    let f = File::create(path).map_err(|e| Error::io(path, e))?;
    // 1 MiB buffer: large volumes benefit; header is tiny either way.
    write_header_and_data_ext(BufWriter::with_capacity(1 << 20, f), hdr, voxels, hdr_ext_json)
}

pub fn write_nii_gz(path: &Path, hdr: &Nifti1Header, voxels: &[u8], level: u32) -> Result<()> {
    write_nii_gz_with_ext(path, hdr, voxels, level, None)
}

/// Write `.nii.zst` (C++ `-z s` / `myEnableZSTD`).
pub fn write_nii_zst(path: &Path, hdr: &Nifti1Header, voxels: &[u8], level: i32) -> Result<()> {
    let f = File::create(path).map_err(|e| Error::io(path, e))?;
    let mut enc = zstd::stream::write::Encoder::new(
        BufWriter::with_capacity(1 << 20, f),
        level.clamp(1, 22),
    )
    .map_err(|e| Error::convert(format!("zstd encoder: {e}")))?;
    write_header_and_data_ext(&mut enc, hdr, voxels, None)?;
    enc.finish()
        .map_err(|e| Error::convert(format!("zstd finish: {e}")))?;
    Ok(())
}

fn write_nii_gz_with_ext(
    path: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    level: u32,
    hdr_ext_json: Option<&str>,
) -> Result<()> {
    let f = File::create(path).map_err(|e| Error::io(path, e))?;
    let enc = GzEncoder::new(
        BufWriter::with_capacity(1 << 20, f),
        Compression::new(level.min(9)),
    );
    write_header_and_data_ext(enc, hdr, voxels, hdr_ext_json)
}

/// Pack `f32` samples to little-endian `i16` (round + clamp).
pub fn f32_voxels_to_i16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() * 2];
    out.par_chunks_mut(2)
        .zip(data.par_iter())
        .for_each(|(dst, &v)| {
            let n = v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            dst.copy_from_slice(&n.to_le_bytes());
        });
    out
}

/// Pack `f32` samples to little-endian `u16` (round + clamp).
pub fn f32_voxels_to_u16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() * 2];
    out.par_chunks_mut(2)
        .zip(data.par_iter())
        .for_each(|(dst, &v)| {
            let n = v.round().clamp(0.0, u16::MAX as f32) as u16;
            dst.copy_from_slice(&n.to_le_bytes());
        });
    out
}

/// Pack `f32` samples to `u8` (round + clamp).
pub fn f32_voxels_to_u8_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    out.par_iter_mut()
        .zip(data.par_iter())
        .for_each(|(dst, &v)| {
            *dst = v.round().clamp(0.0, 255.0) as u8;
        });
    out
}

/// Reinterpret `f32` samples as little-endian bytes (bulk copy on LE hosts).
pub fn f32_voxels_to_f32_bytes(data: &[f32]) -> Vec<u8> {
    // Native-endian host f32 → LE bytes without per-element allocation churn.
    let mut out = Vec::with_capacity(data.len() * 4);
    if cfg!(target_endian = "little") {
        // SAFETY: f32 and [u8;4] have the same size/align; we only read bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        out.extend_from_slice(bytes);
    } else {
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_348_bytes() {
        assert_eq!(std::mem::size_of::<Nifti1Header>(), 348);
    }
}
