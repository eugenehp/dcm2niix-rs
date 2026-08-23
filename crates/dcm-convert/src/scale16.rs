//! Lossless 16-bit range maximization (`nii_scale16bit*`).

use dcm_nifti::{Nifti1Header, DT_INT16, DT_UINT16};
use rayon::prelude::*;

use crate::opts::Maximize16Bit;

/// Scale INT16/UINT16 volumes to use more of the 16-bit range (`-l y`).
pub fn maximize_16bit(hdr: &mut Nifti1Header, voxels: &mut [u8], mode: Maximize16Bit, verbose: i32) {
    if mode == Maximize16Bit::False || mode == Maximize16Bit::Raw {
        return;
    }
    match hdr.datatype {
        DT_INT16 => scale_signed(hdr, voxels, verbose),
        DT_UINT16 => scale_unsigned(hdr, voxels, verbose),
        _ => {}
    }
}

fn n_vox(hdr: &Nifti1Header) -> usize {
    let mut dim3to7 = 1usize;
    for i in 3..8 {
        if hdr.dim[i] > 1 {
            dim3to7 *= hdr.dim[i] as usize;
        }
    }
    (hdr.dim[1] as usize) * (hdr.dim[2] as usize) * dim3to7
}

fn scale_signed(hdr: &mut Nifti1Header, voxels: &mut [u8], verbose: i32) {
    let n = n_vox(hdr);
    if n == 0 || voxels.len() < n * 2 {
        return;
    }
    let (min16, max16) = voxels[..n * 2]
        .par_chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .fold(
            || (i16::MAX, i16::MIN),
            |(mn, mx), v| (mn.min(v), mx.max(v)),
        )
        .reduce(
            || (i16::MAX, i16::MIN),
            |(a_mn, a_mx), (b_mn, b_mx)| (a_mn.min(b_mn), a_mx.max(b_mx)),
        );
    let k_mx = 32000i32;
    let mut scale = if max16 == 0 {
        1
    } else {
        k_mx / max16 as i32
    };
    if (min16 as i32).abs() > max16 as i32 {
        scale = k_mx / (min16 as i32).abs().max(1);
    }
    if scale < 2 {
        if verbose > 0 {
            eprintln!("Sufficient 16-bit range: raw {min16}..{max16}");
        }
        return;
    }
    hdr.scl_slope /= scale as f32;
    voxels[..n * 2].par_chunks_exact_mut(2).for_each(|b| {
        let v = i16::from_le_bytes([b[0], b[1]]);
        let scaled = (v as i32 * scale).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let out = scaled.to_le_bytes();
        b[0] = out[0];
        b[1] = out[1];
    });
    eprintln!("Maximizing 16-bit range: raw {min16}..{max16} is{scale}");
    store_scale_factor(scale, hdr);
}

fn scale_unsigned(hdr: &mut Nifti1Header, voxels: &mut [u8], verbose: i32) {
    let n = n_vox(hdr);
    if n == 0 || voxels.len() < n * 2 {
        return;
    }
    let max16 = voxels[..n * 2]
        .par_chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .max()
        .unwrap_or(0);
    let k_mx = 64000i32;
    let scale = if max16 == 0 {
        1
    } else {
        k_mx / max16 as i32
    };
    if scale < 2 {
        if verbose > 0 {
            eprintln!("Sufficient unsigned 16-bit range: raw max {max16}");
        }
        return;
    }
    hdr.scl_slope /= scale as f32;
    voxels[..n * 2].par_chunks_exact_mut(2).for_each(|b| {
        let v = u16::from_le_bytes([b[0], b[1]]);
        let scaled = (v as i32 * scale).min(u16::MAX as i32) as u16;
        let out = scaled.to_le_bytes();
        b[0] = out[0];
        b[1] = out[1];
    });
    eprintln!("Maximizing 16-bit range: raw max {max16} is{scale}");
    store_scale_factor(scale, hdr);
}

fn store_scale_factor(scale: i32, hdr: &mut Nifti1Header) {
    // C++ stores factor in unused `glmin` when positive.
    if scale > 1 {
        hdr.glmin = scale;
    }
}
