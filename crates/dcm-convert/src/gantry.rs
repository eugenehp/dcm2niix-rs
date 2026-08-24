//! CT gantry-tilt estimate and shear correction (`computeGantryTiltPrecise` / `nii_saveNII3Dtilt`).

use dcm_core::matrix::{cross, dot, norm, Matrix4};
use dcm_dicom::{DicomImage, Modality};
use dcm_nifti::{Nifti1Header, DT_FLOAT32, DT_INT16};

/// Estimate gantry tilt (degrees) from adjacent-slice IPP vs IOP normal (`newTilt`).
pub fn compute_gantry_tilt_precise(d1: &DicomImage, d2: &DicomImage, verbose: i32) -> f64 {
    let mut ret = f64::NAN;
    if d1.patient_position[1].is_nan() {
        return ret;
    }
    let mut slice_vector = [
        d2.patient_position[1] - d1.patient_position[1],
        d2.patient_position[2] - d1.patient_position[2],
        d2.patient_position[3] - d1.patient_position[3],
    ];
    let mut len = norm(slice_vector);
    if len.abs() < 1e-12 {
        slice_vector = [
            d1.patient_position_last[1] - d1.patient_position[1],
            d1.patient_position_last[2] - d1.patient_position[2],
            d1.patient_position_last[3] - d1.patient_position[3],
        ];
        len = norm(slice_vector);
        if len.abs() < 1e-12 {
            return ret;
        }
    }
    if slice_vector[0].is_nan() {
        return ret;
    }
    slice_vector = make_positive(slice_vector);
    let read = [d1.orient[1], d1.orient[2], d1.orient[3]];
    let phase = [d1.orient[4], d1.orient[5], d1.orient[6]];
    let slice90 = make_positive(cross(read, phase));
    let len90 = norm(slice90);
    if len90.abs() < 1e-12 {
        return ret;
    }
    let cos_x = dot(slice90, slice_vector) / (len * len90);
    let deg_x = cos_x.clamp(-1.0, 1.0).acos() * (180.0 / std::f64::consts::PI);
    if (cos_x - 1.0).abs() > 1e-4 {
        ret = deg_x;
    }
    if ret.abs() < 1e-6 && d1.gantry_tilt.abs() < 1e-6 {
        return 0.0;
    }
    let signv = cross(slice_vector, slice90);
    let sign = signv[0].abs().max(signv[1].abs()).max(signv[2].abs());
    if ret.abs() < 1e-4 {
        return 0.0;
    }
    if sign > 0.0 {
        ret = -ret;
    }
    if ret >= 0.0 {
        return 0.0;
    }
    if verbose > 0 || ret.is_nan() {
        eprintln!(
            "Gantry Tilt based on 0018,1120 {}, estimated from slice vector {}",
            d1.gantry_tilt, ret
        );
    } else {
        eprintln!(
            "Gantry Tilt based on 0018,1120 {}, estimated from slice vector {}",
            d1.gantry_tilt, ret
        );
    }
    let _ = slice90;
    ret
}

fn make_positive(mut v: [f64; 3]) -> [f64; 3] {
    // Match C++ makePositive: flip whole vector if first non-zero component is negative.
    for c in v {
        if c.abs() > 1e-12 {
            if c < 0.0 {
                v[0] = -v[0];
                v[1] = -v[1];
                v[2] = -v[2];
            }
            break;
        }
    }
    v
}

/// Apply issue697 shear cleanup on sform when tilt is present (before optional resample).
pub fn apply_gantry_tilt_sform(hdr: &mut Nifti1Header, gantry_tilt_deg: f64) {
    if gantry_tilt_deg.abs() < 1e-4 {
        return;
    }
    let theta = gantry_tilt_deg * std::f64::consts::PI / 180.0;
    let c = theta.cos();
    if c.abs() < 1e-12 {
        return;
    }
    hdr.srow_y[2] = 0.0;
    hdr.srow_z[2] = hdr.pixdim[3];
    let m = Matrix4::from_rows([
        [
            hdr.srow_x[0] as f64,
            hdr.srow_x[1] as f64,
            hdr.srow_x[2] as f64,
            hdr.srow_x[3] as f64,
        ],
        [
            hdr.srow_y[0] as f64,
            hdr.srow_y[1] as f64,
            hdr.srow_y[2] as f64,
            hdr.srow_y[3] as f64,
        ],
        [
            hdr.srow_z[0] as f64,
            hdr.srow_z[1] as f64,
            hdr.srow_z[2] as f64,
            hdr.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    hdr.set_sform(&m);
}

/// Resample volume to remove gantry shear; returns corrected header + voxels for `_Tilt` save.
pub fn correct_tilt(
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    gantry_tilt_deg: f64,
) -> Option<(Nifti1Header, Vec<u8>)> {
    if gantry_tilt_deg == 0.0 {
        return None;
    }
    let n_vox_2d_in = (hdr.dim[1] as usize) * (hdr.dim[2] as usize);
    if n_vox_2d_in < 1 || hdr.dim[0] != 3 || hdr.dim[3] < 3 {
        return None;
    }
    match hdr.datatype {
        DT_INT16 => correct_tilt_i16(hdr, voxels, d, gantry_tilt_deg),
        DT_FLOAT32 => correct_tilt_f32(hdr, voxels, d, gantry_tilt_deg),
        _ => {
            eprintln!("Only able to correct gantry tilt for 16-bit integer or 32-bit float data with at least 3 slices.");
            None
        }
    }
}

fn correct_tilt_i16(
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    gantry_tilt_deg: f64,
) -> Option<(Nifti1Header, Vec<u8>)> {
    eprintln!("Gantry Tilt Correction is new: please validate conversions");
    let mut gnt = (gantry_tilt_deg / (180.0 / std::f64::consts::PI)).tan() / hdr.pixdim[2] as f64;
    let _ = &mut gnt; // keep mut for clarity with C++ sign flips (disabled under newTilt)
    let gnt = gnt;
    let nx = hdr.dim[1] as usize;
    let ny_in = hdr.dim[2] as usize;
    let nz = hdr.dim[3] as usize;
    let n_in = nx * ny_in;
    let max_slice_mm = ((nz - 1) as f64) * hdr.pixdim[3].abs() as f64;
    let px_offset = (gnt.abs() * max_slice_mm).ceil() as i32;
    let mut hdr_out = *hdr;
    hdr_out.dim[2] = hdr.dim[2] + px_offset as i16;
    if gnt < 0.0 {
        adjust_origin_for_negative_tilt(&mut hdr_out, px_offset);
    }
    let ny = hdr_out.dim[2] as usize;
    let n_out = nx * ny;
    let has_pad = !d.pixel_padding_value.is_nan();
    let pad = if has_pad {
        d.pixel_padding_value.round() as i16
    } else {
        let mut mn = i16::MAX;
        for i in 0..(n_in * nz) {
            let v = i16::from_le_bytes([voxels[i * 2], voxels[i * 2 + 1]]);
            mn = mn.min(v);
        }
        mn
    };
    let mut out = vec![0u8; n_out * nz * 2];
    for i in 0..(n_out * nz) {
        let b = pad.to_le_bytes();
        out[i * 2] = b[0];
        out[i * 2 + 1] = b[1];
    }
    let is_seg = d.modality == Modality::Seg;
    for s in 0..nz {
        let mut slice_mm = s as f64 * hdr.pixdim[3] as f64;
        if gnt < 0.0 {
            slice_mm -= max_slice_mm;
        }
        let offset = gnt * slice_mm;
        let frac_hi = offset.ceil() - offset;
        let frac_lo = 1.0 - frac_hi;
        for r in 0..ny {
            let r_i = r as f64 - offset;
            if r_i >= 0.0 && r_i < ny_in as f64 {
                let r_lo = r_i.floor() as usize;
                let mut r_hi = r_lo + 1;
                if r_hi >= ny_in {
                    r_hi = r_lo;
                }
                let base_lo = r_lo * nx + s * n_in;
                let base_hi = r_hi * nx + s * n_in;
                let base_out = r * nx + s * n_out;
                for c in 0..nx {
                    let lo = i16::from_le_bytes([
                        voxels[(base_lo + c) * 2],
                        voxels[(base_lo + c) * 2 + 1],
                    ]);
                    let hi = i16::from_le_bytes([
                        voxels[(base_hi + c) * 2],
                        voxels[(base_hi + c) * 2 + 1],
                    ]);
                    let v = if is_seg || (has_pad && (lo == pad || hi == pad)) {
                        if frac_hi >= 0.5 {
                            hi
                        } else {
                            lo
                        }
                    } else {
                        (lo as f64 * frac_lo + hi as f64 * frac_hi).round() as i16
                    };
                    let b = v.to_le_bytes();
                    out[(base_out + c) * 2] = b[0];
                    out[(base_out + c) * 2 + 1] = b[1];
                }
            }
        }
    }
    deshear_sform(&mut hdr_out);
    Some((hdr_out, out))
}

fn correct_tilt_f32(
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    gantry_tilt_deg: f64,
) -> Option<(Nifti1Header, Vec<u8>)> {
    eprintln!("Gantry Tilt Correction is new: please validate conversions");
    let gnt = (gantry_tilt_deg / (180.0 / std::f64::consts::PI)).tan() / hdr.pixdim[2] as f64;
    let nx = hdr.dim[1] as usize;
    let ny_in = hdr.dim[2] as usize;
    let nz = hdr.dim[3] as usize;
    let n_in = nx * ny_in;
    let max_slice_mm = ((nz - 1) as f64) * hdr.pixdim[3].abs() as f64;
    let px_offset = (gnt.abs() * max_slice_mm).ceil() as i32;
    let mut hdr_out = *hdr;
    hdr_out.dim[2] = hdr.dim[2] + px_offset as i16;
    if gnt < 0.0 {
        adjust_origin_for_negative_tilt(&mut hdr_out, px_offset);
    }
    let ny = hdr_out.dim[2] as usize;
    let n_out = nx * ny;
    let has_pad = !d.pixel_padding_value.is_nan();
    let pad = if has_pad {
        d.pixel_padding_value as f32
    } else {
        let mut mn = f32::INFINITY;
        for i in 0..(n_in * nz) {
            let v = f32::from_le_bytes([
                voxels[i * 4],
                voxels[i * 4 + 1],
                voxels[i * 4 + 2],
                voxels[i * 4 + 3],
            ]);
            mn = mn.min(v);
        }
        mn
    };
    let mut out = vec![0u8; n_out * nz * 4];
    for i in 0..(n_out * nz) {
        out[i * 4..i * 4 + 4].copy_from_slice(&pad.to_le_bytes());
    }
    let is_seg = d.modality == Modality::Seg;
    for s in 0..nz {
        let mut slice_mm = s as f64 * hdr.pixdim[3] as f64;
        if gnt < 0.0 {
            slice_mm -= max_slice_mm;
        }
        let offset = gnt * slice_mm;
        let frac_hi = offset.ceil() - offset;
        let frac_lo = 1.0 - frac_hi;
        for r in 0..ny {
            let r_i = r as f64 - offset;
            if r_i >= 0.0 && r_i < ny_in as f64 {
                let r_lo = r_i.floor() as usize;
                let mut r_hi = r_lo + 1;
                if r_hi >= ny_in {
                    r_hi = r_lo;
                }
                let base_lo = r_lo * nx + s * n_in;
                let base_hi = r_hi * nx + s * n_in;
                let base_out = r * nx + s * n_out;
                for c in 0..nx {
                    let lo = f32::from_le_bytes(
                        voxels[(base_lo + c) * 4..(base_lo + c) * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let hi = f32::from_le_bytes(
                        voxels[(base_hi + c) * 4..(base_hi + c) * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let v = if is_seg || (has_pad && (lo == pad || hi == pad)) {
                        if frac_hi >= 0.5 {
                            hi
                        } else {
                            lo
                        }
                    } else {
                        (lo as f64 * frac_lo + hi as f64 * frac_hi).round() as f32
                    };
                    out[(base_out + c) * 4..(base_out + c) * 4 + 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    deshear_sform(&mut hdr_out);
    Some((hdr_out, out))
}

fn adjust_origin_for_negative_tilt(hdr: &mut Nifti1Header, px_offset: i32) {
    // Shift origin along row (Y) by px_offset * pixdim[2] in scanner space.
    let dy = px_offset as f32 * hdr.pixdim[2];
    // Approximate: add dy along the unit Y column of sform.
    let len = (hdr.srow_x[1].powi(2) + hdr.srow_y[1].powi(2) + hdr.srow_z[1].powi(2)).sqrt();
    if len > 1e-6 {
        let s = dy / len;
        hdr.srow_x[3] += hdr.srow_x[1] * s;
        hdr.srow_y[3] += hdr.srow_y[1] * s;
        hdr.srow_z[3] += hdr.srow_z[1] * s;
    }
    let m = Matrix4::from_rows([
        [
            hdr.srow_x[0] as f64,
            hdr.srow_x[1] as f64,
            hdr.srow_x[2] as f64,
            hdr.srow_x[3] as f64,
        ],
        [
            hdr.srow_y[0] as f64,
            hdr.srow_y[1] as f64,
            hdr.srow_y[2] as f64,
            hdr.srow_y[3] as f64,
        ],
        [
            hdr.srow_z[0] as f64,
            hdr.srow_z[1] as f64,
            hdr.srow_z[2] as f64,
            hdr.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    hdr.set_sform(&m);
}

fn deshear_sform(hdr: &mut Nifti1Header) {
    // After resampling, clear residual shear similar to issue697 cleanup.
    hdr.srow_y[2] = 0.0;
    hdr.srow_z[2] = hdr.pixdim[3];
    let m = Matrix4::from_rows([
        [
            hdr.srow_x[0] as f64,
            hdr.srow_x[1] as f64,
            hdr.srow_x[2] as f64,
            hdr.srow_x[3] as f64,
        ],
        [
            hdr.srow_y[0] as f64,
            hdr.srow_y[1] as f64,
            hdr.srow_y[2] as f64,
            hdr.srow_y[3] as f64,
        ],
        [
            hdr.srow_z[0] as f64,
            hdr.srow_z[1] as f64,
            hdr.srow_z[2] as f64,
            hdr.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    hdr.set_sform(&m);
}
