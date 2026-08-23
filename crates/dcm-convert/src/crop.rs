//! Neck crop for 3D volumes (`nii_saveCrop`).

use dcm_core::error::{Error, Result};
use dcm_nifti::{Nifti1Header, DT_INT16, DT_UINT16};

/// Crop excess ventral/dorsal slices after ortho. Returns cropped header + voxels, or `None` if not applicable.
pub fn try_crop(hdr: &Nifti1Header, voxels: &[u8]) -> Result<Option<(Nifti1Header, Vec<u8>)>> {
    let n_vox_2d = (hdr.dim[1] as usize) * (hdr.dim[2] as usize);
    if n_vox_2d < 1
        || hdr.pixdim[3].abs() < 0.001
        || hdr.dim[0] != 3
        || hdr.dim[3] < 128
        || (hdr.datatype != DT_INT16 && hdr.datatype != DT_UINT16)
    {
        return Ok(None);
    }
    let slices = hdr.dim[3] as usize;
    if voxels.len() < n_vox_2d * slices * 2 {
        return Err(Error::convert("crop: voxel buffer too short"));
    }
    let mut slice_sums = vec![0.0f64; slices];
    let mut max_slice = 0.0f64;
    for i in 0..slices {
        let start = i * n_vox_2d;
        let mut sum = 0.0;
        for j in 0..n_vox_2d {
            let off = (start + j) * 2;
            let v = if hdr.datatype == DT_UINT16 {
                u16::from_le_bytes([voxels[off], voxels[off + 1]]) as f64
            } else {
                i16::from_le_bytes([voxels[off], voxels[off + 1]]) as f64
            };
            sum += v;
        }
        slice_sums[i] = sum;
        max_slice = max_slice.max(sum);
    }
    if max_slice <= 0.0 {
        return Ok(None);
    }
    smooth1d(&mut slice_sums);
    for s in &mut slice_sums {
        *s /= max_slice;
    }
    let k_thresh = 0.09;
    let mut dorsal = slices - 1;
    while dorsal >= 1 {
        if slice_sums[dorsal - 1] > k_thresh {
            break;
        }
        if dorsal == 1 {
            return Ok(None);
        }
        dorsal -= 1;
    }
    if dorsal <= 1 {
        return Ok(None);
    }
    let k_max_dv_mm = 169.0;
    let mut ventral = dorsal as i32 - (k_max_dv_mm / hdr.pixdim[3] as f64).round() as i32;
    if ventral < 0 {
        ventral = 0;
    }
    let ventral = ventral as usize;
    eprintln!(" Cropping from slice {ventral} to {dorsal} (of {slices})");
    let n_keep = dorsal - ventral + 1;
    let mut hdr_x = *hdr;
    hdr_x.dim[3] = n_keep as i16;
    hdr_x.srow_x[3] += hdr.srow_x[2] * ventral as f32;
    hdr_x.srow_y[3] += hdr.srow_y[2] * ventral as f32;
    hdr_x.srow_z[3] += hdr.srow_z[2] * ventral as f32;
    // Refresh qform from updated sform.
    use dcm_core::matrix::Matrix4;
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

    let mut out = vec![0u8; n_vox_2d * n_keep * 2];
    for s in 0..n_keep {
        let src = (s + ventral) * n_vox_2d * 2;
        let dst = s * n_vox_2d * 2;
        out[dst..dst + n_vox_2d * 2].copy_from_slice(&voxels[src..src + n_vox_2d * 2]);
    }
    Ok(Some((hdr_x, out)))
}

fn smooth1d(v: &mut [f64]) {
    if v.len() < 3 {
        return;
    }
    let mut prev = v[0];
    for i in 1..v.len() - 1 {
        let cur = v[i];
        v[i] = (prev + cur + v[i + 1]) / 3.0;
        prev = cur;
    }
}
