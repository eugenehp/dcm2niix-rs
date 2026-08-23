//! DICOM Overlay Data → `_ROI{n}` NIfTI (C++ `loadOverlay` / ROI export).

use std::path::PathBuf;

use dcm_core::error::Result;
use dcm_dicom::DicomImage;
use dcm_nifti::{Nifti1Header, DT_UINT8};

use crate::filename::create_filename;
use crate::geom::{apply_flip_y_sform, apply_flip_z_sform};
use crate::opts::{Compress, DcmOpts, SaveFormat};
use crate::ortho::nii_set_ortho_f32;
use crate::voxels::{flip_y_volume, flip_z_volume};

/// Unpack 1-bit OverlayData into one byte per voxel (0/1).
pub fn unpack_overlay_bits(bits: &[u8], nvox: usize) -> Vec<u8> {
    let mut img = vec![0u8; nvox];
    // C++ mask: {1, 2, 4, 8, 16, 32, 64, 128} — LSB first within each byte.
    for i in 0..nvox {
        let byt = i >> 3;
        let bit = i % 8;
        if byt < bits.len() && (bits[byt] & (1u8 << bit)) != 0 {
            img[i] = 1;
        }
    }
    img
}

fn u8_to_f32(v: &[u8]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn f32_to_u8_bin(v: &[f32]) -> Vec<u8> {
    v.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect()
}

/// Write `_ROI1`…`_ROI16` companions for series that carry Overlay Data.
pub fn write_overlay_rois(
    images: &[&DicomImage],
    hdr: &Nifti1Header,
    need_flip_z: bool,
    use_ortho: bool,
    opts: &DcmOpts,
) -> Result<Vec<PathBuf>> {
    if opts.save_format != SaveFormat::Nifti || opts.bids == crate::opts::BidsMode::Only {
        return Ok(vec![]);
    }
    if !images.iter().any(|d| d.is_has_overlay) {
        return Ok(vec![]);
    }
    let nx = hdr.dim[1].max(1) as usize;
    let ny = hdr.dim[2].max(1) as usize;
    let nz = hdr.dim[3].max(1) as usize;
    let stem = create_filename(images[0], opts)?;
    let mut written = Vec::new();

    for j in 0..16 {
        let present = images.iter().any(|d| d.overlays[j].is_some());
        if !present {
            continue;
        }
        let mut img = vec![0u8; nx * ny * nz];
        if images.len() == 1 {
            if let Some(bits) = images[0].overlays[j].as_ref() {
                let unpacked = unpack_overlay_bits(bits, nx * ny * nz);
                let n = unpacked.len().min(img.len());
                img[..n].copy_from_slice(&unpacked[..n]);
            }
        } else if images.len() == nz {
            for (i, d) in images.iter().enumerate() {
                if let Some(bits) = d.overlays[j].as_ref() {
                    let slice = unpack_overlay_bits(bits, nx * ny);
                    let off = i * nx * ny;
                    let n = slice.len().min(nx * ny);
                    img[off..off + n].copy_from_slice(&slice[..n]);
                }
            }
        } else {
            // C++ only handles nConvert==1 or nConvert==dim[3].
            continue;
        }

        let mut hdrr = *hdr;
        hdrr.dim[0] = 3;
        hdrr.dim[4] = 1;
        hdrr.bitpix = 8;
        hdrr.datatype = DT_UINT8;
        hdrr.scl_inter = 0.0;
        hdrr.scl_slope = 1.0;
        hdrr.vox_offset = 352.0;

        let mut vol_f = u8_to_f32(&img);
        let mut sform = dcm_core::matrix::Matrix4::from_rows([
            [
                hdrr.srow_x[0] as f64,
                hdrr.srow_x[1] as f64,
                hdrr.srow_x[2] as f64,
                hdrr.srow_x[3] as f64,
            ],
            [
                hdrr.srow_y[0] as f64,
                hdrr.srow_y[1] as f64,
                hdrr.srow_y[2] as f64,
                hdrr.srow_y[3] as f64,
            ],
            [
                hdrr.srow_z[0] as f64,
                hdrr.srow_z[1] as f64,
                hdrr.srow_z[2] as f64,
                hdrr.srow_z[3] as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        if need_flip_z && nz > 1 {
            vol_f = flip_z_volume(vol_f, nx, ny, nz, 1);
            apply_flip_z_sform(&mut sform, nz);
            hdrr.set_sform(&sform);
        }
        if use_ortho {
            vol_f = nii_set_ortho_f32(vol_f, &mut hdrr);
        } else if opts.flip_y {
            vol_f = flip_y_volume(vol_f, nx, ny, hdrr.dim[3] as usize, 1);
            apply_flip_y_sform(&mut sform, hdrr.dim[2] as usize);
            hdrr.set_sform(&sform);
        }
        let vol = f32_to_u8_bin(&vol_f);

        let roi_stem = PathBuf::from(format!("{}_ROI{}", stem.display(), j + 1));
        let ext = match opts.compress {
            Compress::None | Compress::Save3d => "nii",
            Compress::Gz | Compress::InternalGz => "nii.gz",
            Compress::Zstd => "nii.zst",
        };
        let path = crate::unique_path(&roi_stem, ext, opts.name_conflict)?;
        match opts.compress {
            Compress::None | Compress::Save3d => dcm_nifti::write_nii(&path, &hdrr, &vol)?,
            Compress::Gz | Compress::InternalGz => {
                dcm_nifti::write_nii_gz(&path, &hdrr, &vol, opts.gz_level as u32)?
            }
            Compress::Zstd => dcm_nifti::write_nii_zst(&path, &hdrr, &vol, opts.gz_level)?,
        }
        println!(
            "Convert {} DICOM as {} ({}x{}x{})",
            images.len(),
            path.display(),
            nx,
            ny,
            nz
        );
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_lsb_first() {
        let bits = [0b0000_0011u8];
        let v = unpack_overlay_bits(&bits, 8);
        assert_eq!(v, vec![1, 1, 0, 0, 0, 0, 0, 0]);
    }
}
