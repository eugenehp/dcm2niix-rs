//! Unequal slice spacing → equidistant volume (`nii_saveNII3Deq`).

use dcm_core::error::Result;
use dcm_core::matrix::Matrix4;
use dcm_nifti::{Nifti1Header, DT_FLOAT32, DT_INT16, DT_UINT8, DT_UINT16};

/// Resample a 3D volume onto equal slice spacing. Returns `_Eq` header + voxels.
pub fn equalize_slices(
    hdr: &Nifti1Header,
    voxels: &[u8],
    slice_mm: &[f32],
) -> Result<Option<(Nifti1Header, Vec<u8>)>> {
    let n_vox_2d = (hdr.dim[1] as usize) * (hdr.dim[2] as usize);
    let in_slices = hdr.dim[3] as usize;
    if n_vox_2d < 1 || hdr.dim[0] != 3 || in_slices < 3 || slice_mm.len() < in_slices {
        return Ok(None);
    }
    if !matches!(
        hdr.datatype,
        DT_FLOAT32 | DT_UINT8 | DT_INT16 | DT_UINT16
    ) {
        eprintln!(
            "Only able to make equidistant slices from 8,16-bit integer or 32-bit float image data."
        );
        return Ok(None);
    }
    let mut mn = slice_mm[1] - slice_mm[0];
    for i in 1..in_slices {
        let dx = slice_mm[i] - slice_mm[i - 1];
        if dx < mn && dx.abs() > 1e-6 {
            mn = dx;
        }
    }
    if mn <= 0.0 {
        eprintln!("Unable to equalize slice distances: slice order not consistently ascending");
        return Ok(None);
    }
    let mut out_slices = (slice_mm[in_slices - 1] / mn).ceil() as i32 + 1;
    if out_slices > 2 * in_slices as i32 {
        out_slices = 2 * in_slices as i32;
    }
    if out_slices < 3 {
        return Ok(None);
    }
    let out_slices = out_slices as usize;
    mn = slice_mm[in_slices - 1] / (out_slices as f32 - 1.0);
    let bp = (hdr.bitpix as usize / 8).max(1);
    let mut out = vec![0u8; n_vox_2d * out_slices * bp];
    for s in 0..out_slices {
        let out_mm = s as f32 * mn;
        let mut low_idx = 0usize;
        while low_idx < in_slices - 2 && slice_mm[low_idx + 1] < out_mm {
            low_idx += 1;
        }
        let mut hi_idx = low_idx + 1;
        let (mut low_wt, mut hi_wt) = (1.0f32, 0.0f32);
        if out_mm <= slice_mm[0] {
            low_idx = 0;
            hi_idx = 0;
        } else if out_mm >= slice_mm[in_slices - 1] {
            low_idx = in_slices - 1;
            hi_idx = in_slices - 1;
        } else if low_idx != hi_idx {
            let d = slice_mm[hi_idx] - slice_mm[low_idx];
            let frac = (out_mm - slice_mm[low_idx]) / d;
            low_wt = 1.0 - frac;
            hi_wt = frac;
        }
        let low_off = low_idx * n_vox_2d * bp;
        let hi_off = hi_idx * n_vox_2d * bp;
        let out_off = s * n_vox_2d * bp;
        match hdr.datatype {
            DT_FLOAT32 => {
                for v in 0..n_vox_2d {
                    let lo = f32::from_le_bytes(
                        voxels[low_off + v * 4..low_off + v * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let hi = f32::from_le_bytes(
                        voxels[hi_off + v * 4..hi_off + v * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let val = lo * low_wt + hi * hi_wt;
                    out[out_off + v * 4..out_off + v * 4 + 4]
                        .copy_from_slice(&val.to_le_bytes());
                }
            }
            DT_UINT8 => {
                for v in 0..n_vox_2d {
                    let lo = voxels[low_off + v] as f32;
                    let hi = voxels[hi_off + v] as f32;
                    out[out_off + v] = (lo * low_wt + hi * hi_wt).round() as u8;
                }
            }
            _ => {
                // INT16 / UINT16
                for v in 0..n_vox_2d {
                    let lo = i16::from_le_bytes([
                        voxels[low_off + v * 2],
                        voxels[low_off + v * 2 + 1],
                    ]) as f32;
                    let hi = i16::from_le_bytes([
                        voxels[hi_off + v * 2],
                        voxels[hi_off + v * 2 + 1],
                    ]) as f32;
                    let val = (lo * low_wt + hi * hi_wt).round() as i16;
                    let b = val.to_le_bytes();
                    out[out_off + v * 2] = b[0];
                    out[out_off + v * 2 + 1] = b[1];
                }
            }
        }
    }
    let mut hdr_x = *hdr;
    hdr_x.dim[3] = out_slices as i16;
    hdr_x.pixdim[3] = mn;
    // Preserve orientation: scale slice column to new spacing.
    let old = hdr.pixdim[3].abs().max(1e-6);
    let scale = mn / old;
    hdr_x.srow_x[2] *= scale;
    hdr_x.srow_y[2] *= scale;
    hdr_x.srow_z[2] *= scale;
    let m = Matrix4::from_rows([
        [
            hdr_x.srow_x[0] as f64,
            hdr_x.srow_x[1] as f64,
            hdr_x.srow_x[2] as f64,
            hdr_x.srow_x[3] as f64,
        ],
        [
            hdr_x.srow_y[0] as f64,
            hdr_x.srow_y[1] as f64,
            hdr_x.srow_y[2] as f64,
            hdr_x.srow_y[3] as f64,
        ],
        [
            hdr_x.srow_z[0] as f64,
            hdr_x.srow_z[1] as f64,
            hdr_x.srow_z[2] as f64,
            hdr_x.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    hdr_x.set_sform(&m);
    Ok(Some((hdr_x, out)))
}
