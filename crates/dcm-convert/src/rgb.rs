//! RGB planar ↔ packed conversion (`nii_planar2rgb` / `nii_rgb2planar`).

use dcm_nifti::Nifti1Header;

/// Convert planar `RRR…GGG…BBB…` to packed `RGBRGB…` in-place (NIfTI default).
pub fn planar_to_rgb(img: &mut [u8], hdr: &Nifti1Header) {
    if hdr.datatype != dcm_nifti::DT_RGB24 {
        return;
    }
    let nx = hdr.dim[1].max(1) as usize;
    let ny = hdr.dim[2].max(1) as usize;
    let mut n_slice = 1usize;
    for i in 3..8 {
        let d = hdr.dim[i];
        if d > 1 {
            n_slice *= d as usize;
        }
    }
    let n_pix = nx * ny;
    let slice_bytes = n_pix * 3;
    let mut temp = vec![0u8; slice_bytes];
    for sl in 0..n_slice {
        let base = sl * slice_bytes;
        if base + slice_bytes > img.len() {
            break;
        }
        let r = &img[base..base + n_pix];
        let g = &img[base + n_pix..base + 2 * n_pix];
        let b = &img[base + 2 * n_pix..base + 3 * n_pix];
        for i in 0..n_pix {
            temp[i * 3] = r[i];
            temp[i * 3 + 1] = g[i];
            temp[i * 3 + 2] = b[i];
        }
        img[base..base + slice_bytes].copy_from_slice(&temp);
    }
}

/// Convert packed `RGBRGB…` to planar `RRR…GGG…BBB…` (Analyze / `-` rgb planar).
pub fn rgb_to_planar(img: &mut [u8], hdr: &Nifti1Header) {
    if hdr.datatype != dcm_nifti::DT_RGB24 {
        return;
    }
    let nx = hdr.dim[1].max(1) as usize;
    let ny = hdr.dim[2].max(1) as usize;
    let mut n_slice = 1usize;
    for i in 3..8 {
        let d = hdr.dim[i];
        if d > 1 {
            n_slice *= d as usize;
        }
    }
    let n_pix = nx * ny;
    let slice_bytes = n_pix * 3;
    let mut temp = vec![0u8; slice_bytes];
    for sl in 0..n_slice {
        let base = sl * slice_bytes;
        if base + slice_bytes > img.len() {
            break;
        }
        let src = &img[base..base + slice_bytes];
        for i in 0..n_pix {
            temp[i] = src[i * 3];
            temp[n_pix + i] = src[i * 3 + 1];
            temp[2 * n_pix + i] = src[i * 3 + 2];
        }
        img[base..base + slice_bytes].copy_from_slice(&temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcm_nifti::{Nifti1Header, DT_RGB24};

    #[test]
    fn planar_roundtrip() {
        let mut hdr = Nifti1Header::default();
        hdr.datatype = DT_RGB24;
        hdr.dim = [3, 2, 1, 1, 1, 1, 1, 1];
        // planar: R0 R1 | G0 G1 | B0 B1
        let mut img = vec![10, 11, 20, 21, 30, 31];
        planar_to_rgb(&mut img, &hdr);
        assert_eq!(img, vec![10, 20, 30, 11, 21, 31]);
        rgb_to_planar(&mut img, &hdr);
        assert_eq!(img, vec![10, 11, 20, 21, 30, 31]);
    }
}
