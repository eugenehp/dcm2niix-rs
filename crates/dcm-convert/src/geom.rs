//! Spatial transform: `headerDcm2Nii` / `set_nii_header_x` / `nii_flipY`.

use dcm_core::matrix::{cross, mat33_det, nifti_dicom2mat, normalise, snap_mat44, Matrix4};
use dcm_core::snap_f32;
use dcm_dicom::DicomImage;
use dcm_nifti::{Nifti1Header, DT_INT16, DT_UINT16, NIFTI_UNITS_MM, NIFTI_UNITS_SEC};

pub fn slice_normal(orient: &[f64; 7]) -> [f64; 3] {
    let col = normalise([orient[1], orient[2], orient[3]]);
    let row = normalise([orient[4], orient[5], orient[6]]);
    cross(col, row)
}

/// `verify_slice_dir`: flip the k column of R when last-slice IPP disagrees.
/// Returns `true` when slices must also be reordered (`nii_flipZ`).
pub fn verify_slice_dir(first: &DicomImage, last: &DicomImage, nz: usize, r: &mut Matrix4) -> bool {
    if nz < 2 {
        return false;
    }
    let mut isl = 0usize;
    if r.0[1][2].abs() >= r.0[0][2].abs() && r.0[1][2].abs() >= r.0[2][2].abs() {
        isl = 1;
    }
    if r.0[2][2].abs() >= r.0[0][2].abs() && r.0[2][2].abs() >= r.0[1][2].abs() {
        isl = 2;
    }
    let mut pos = f64::NAN;
    if !last.patient_position[isl + 1].is_nan() {
        pos = last.patient_position[isl + 1];
        if is_same_float(pos as f32, first.patient_position[isl + 1] as f32) {
            pos = f64::NAN;
        }
    }
    if pos.is_nan() && !first.patient_position_last[isl + 1].is_nan() {
        pos = first.patient_position_last[isl + 1];
        if is_same_float(pos as f32, first.patient_position[isl + 1] as f32) {
            pos = f64::NAN;
        }
    }
    if pos.is_nan() && !first.last_scan_loc.is_nan() {
        pos = first.last_scan_loc;
    }
    if pos.is_nan() {
        return false;
    }
    let m = snap_mat44(r);
    let z = (nz as f32) - 1.0;
    let pos1v = nifti_vect44mat44_mul_f32([0.0, 0.0, z, 1.0], &m);
    let origin = snap_f32(m.0[isl][3]);
    let pos1 = snap_f32(pos1v[isl] as f64);
    let flip = (snap_f32(pos) > origin) != (pos1 > origin);
    if flip {
        let m = snap_mat44(r);
        for i in 0..4 {
            r.0[i][2] = (-(m.0[i][2] as f32)) as f64;
        }
        *r = snap_mat44(r);
    }
    flip
}

fn is_same_float(a: f32, b: f32) -> bool {
    (a - b).abs() <= f32::EPSILON
}

/// C++ `isSameFloatGE`: 1e-4 absolute tolerance.
pub fn is_same_float_ge(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.0001
}

/// C++ `intersliceDistance`: Euclidean IPP gap in float (gantry tilt = 0).
pub fn interslice_distance(a: &DicomImage, b: &DicomImage) -> f32 {
    if a.patient_position[1].is_nan() || b.patient_position[1].is_nan() {
        return a.xyz_mm[3] as f32;
    }
    let dx = (a.patient_position[1] as f32) - (b.patient_position[1] as f32);
    let dy = (a.patient_position[2] as f32) - (b.patient_position[2] as f32);
    let dz = (a.patient_position[3] as f32) - (b.patient_position[3] as f32);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// `nifti_vect44mat44_mul` with float storage (C++ `nifti1_io_core.cpp`).
fn nifti_vect44mat44_mul_f32(v: [f32; 4], m: &Matrix4) -> [f32; 4] {
    let m = snap_mat44(m);
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let mut s = 0.0f32;
        for j in 0..4 {
            s += (m.0[i][j] as f32) * v[j];
        }
        out[i] = s;
    }
    out
}

/// `nii_flipZ` affine update: origin moves to (0, 0, nz-1), k column negated.
pub fn apply_flip_z_sform(sform: &mut Matrix4, nz: usize) {
    let m = snap_mat44(sform);
    let z = (nz as f32) - 1.0;
    let ox = (m.0[0][2] as f32) * z + (m.0[0][3] as f32);
    let oy = (m.0[1][2] as f32) * z + (m.0[1][3] as f32);
    let oz = (m.0[2][2] as f32) * z + (m.0[2][3] as f32);
    let a = [
        [m.0[0][0] as f32, m.0[0][1] as f32, m.0[0][2] as f32],
        [m.0[1][0] as f32, m.0[1][1] as f32, m.0[1][2] as f32],
        [m.0[2][0] as f32, m.0[2][1] as f32, m.0[2][2] as f32],
    ];
    let flip = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
    let mut s = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = a[i][0] * flip[0][j] + a[i][1] * flip[1][j] + a[i][2] * flip[2][j];
        }
    }
    *sform = snap_mat44(&Matrix4::from_rows([
        [s[0][0] as f64, s[0][1] as f64, s[0][2] as f64, ox as f64],
        [s[1][0] as f64, s[1][1] as f64, s[1][2] as f64, oy as f64],
        [s[2][0] as f64, s[2][1] as f64, s[2][2] as f64, oz as f64],
        [0.0, 0.0, 0.0, 1.0],
    ]));
}

/// Siemens mosaic affine (`set_nii_header_x`): mosaic origin + LPS→RAS +
/// optional k-axis flip when `SliceNormalVector` disagrees with IOP.
///
/// `mosaic_cols` / `mosaic_rows` are the **packed** mosaic image size
/// (before demosaic), matching `d.xyzDim` in C++.
pub fn apply_siemens_mosaic_sform(
    q: &mut Matrix4,
    d: &DicomImage,
    mosaic_cols: usize,
    mosaic_rows: usize,
    n_mosaic_slices: i32,
) {
    if n_mosaic_slices < 2 {
        *q = snap_mat44(&q.lps_to_ras_f32());
        return;
    }
    let n_row_col = (n_mosaic_slices as f64).sqrt().ceil();
    let factor_x = (mosaic_cols as f64 - (mosaic_cols as f64 / n_row_col)) / 2.0;
    let factor_y = (mosaic_rows as f64 - (mosaic_rows as f64 / n_row_col)) / 2.0;
    // C++ mat44 float: Q44.m[r][3] = (float)(m[r][0]*fx + m[r][1]*fy + m[r][3])
    let mut m = snap_mat44(q);
    for r in 0..3 {
        let t = (m.0[r][0] as f32) as f64 * factor_x
            + (m.0[r][1] as f32) as f64 * factor_y
            + (m.0[r][3] as f32) as f64;
        m.0[r][3] = t as f32 as f64;
    }
    m = snap_mat44(&m.lps_to_ras_f32());
    let sn = d.csa.image.slice_norm;
    if sn[1] != 0.0 || sn[2] != 0.0 || sn[3] != 0.0 {
        // C++ LOAD_MAT33 uses float orient / sliceNorm.
        let m33 = [
            [
                d.orient[1] as f32 as f64,
                d.orient[4] as f32 as f64,
                sn[1] as f32 as f64,
            ],
            [
                d.orient[2] as f32 as f64,
                d.orient[5] as f32 as f64,
                sn[2] as f32 as f64,
            ],
            [
                d.orient[3] as f32 as f64,
                d.orient[6] as f32 as f64,
                sn[3] as f32 as f64,
            ],
        ];
        if mat33_det(m33) < 0.0 {
            let det = Matrix4::from_rows([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            m = snap_mat44(&mat44_mul_f32(&m, &det));
        }
    }
    *q = m;
}

/// Float mat44 multiply matching C++ `nifti_mat44_mul`.
fn mat44_mul_f32(a: &Matrix4, b: &Matrix4) -> Matrix4 {
    let a = snap_mat44(a);
    let b = snap_mat44(b);
    let mut out = Matrix4::from_rows([[0.0; 4]; 4]);
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 {
                s += (a.0[i][k] as f32) * (b.0[k][j] as f32);
            }
            out.0[i][j] = s as f64;
        }
    }
    out
}

/// dcm2niix `nii_flipY` affine update: origin moves to (0, ny-1, 0).
pub fn apply_flip_y_sform(sform: &mut Matrix4, ny: usize) {
    let m = snap_mat44(sform);
    let y = (ny as f32) - 1.0;
    let v = nifti_vect44mat44_mul_f32([0.0, y, 0.0, 1.0], &m);
    // C++ nifti_mat33_mul(s, mFlipY) — must use the full 3-term product so
    // signed zeros match (`a*0 + b*(-1) + c*0` is not the same as `-b`).
    let a = [
        [m.0[0][0] as f32, m.0[0][1] as f32, m.0[0][2] as f32],
        [m.0[1][0] as f32, m.0[1][1] as f32, m.0[1][2] as f32],
        [m.0[2][0] as f32, m.0[2][1] as f32, m.0[2][2] as f32],
    ];
    let flip = [[1.0f32, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut s = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = a[i][0] * flip[0][j] + a[i][1] * flip[1][j] + a[i][2] * flip[2][j];
        }
    }
    *sform = snap_mat44(&Matrix4::from_rows([
        [s[0][0] as f64, s[0][1] as f64, s[0][2] as f64, v[0] as f64],
        [s[1][0] as f64, s[1][1] as f64, s[1][2] as f64, v[1] as f64],
        [s[2][0] as f64, s[2][1] as f64, s[2][2] as f64, v[2] as f64],
        [0.0, 0.0, 0.0, 1.0],
    ]));
}

/// Signed through-slice distance from `a` to `b` (C++ `intersliceDistanceSigned`).
pub fn interslice_distance_signed(a: &DicomImage, b: &DicomImage) -> f64 {
    let dv = [
        b.patient_position[1] - a.patient_position[1],
        b.patient_position[2] - a.patient_position[2],
        b.patient_position[3] - a.patient_position[3],
    ];
    let len = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
    if len < f64::EPSILON {
        return 0.0;
    }
    let read = normalise([a.orient[1], a.orient[2], a.orient[3]]);
    let phase = normalise([a.orient[4], a.orient[5], a.orient[6]]);
    let slice = cross(read, phase);
    let dot = slice[0] * dv[0] + slice[1] * dv[1] + slice[2] * dv[2];
    if dot < 0.0 {
        -len
    } else {
        len
    }
}

pub fn header_from_series(
    d: &DicomImage,
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    xyz_mm: [f64; 4],
) -> Nifti1Header {
    let mut h = Nifti1Header::default();
    if d.bits_allocated <= 8 {
        h.datatype = dcm_nifti::DT_UINT8;
        h.bitpix = 8;
    } else if d.samples_per_pixel == 3 {
        h.datatype = dcm_nifti::DT_RGB24;
        h.bitpix = 24;
    } else if d.is_float || d.bits_allocated == 32 {
        h.datatype = dcm_nifti::DT_FLOAT32;
        h.bitpix = 32;
    } else if d.is_signed || d.bits_stored < 16 {
        h.datatype = DT_INT16;
        h.bitpix = 16;
    } else {
        h.datatype = DT_UINT16;
        h.bitpix = 16;
    }
    if d.samples_per_pixel == 3 {
        // RGB ignores scl_* (NIfTI convention).
        h.scl_slope = 1.0;
        h.scl_inter = 0.0;
    } else {
        h.scl_slope = d.inten_scale;
        h.scl_inter = d.inten_intercept;
    }
    h.pixdim[1] = xyz_mm[1] as f32;
    h.pixdim[2] = xyz_mm[2] as f32;
    h.pixdim[3] = xyz_mm[3] as f32;
    // C++ stores TR as float first: `(float)TR / 1000.0f` → matches Ref pixdim[4].
    h.pixdim[4] = (d.tr as f32) / 1000.0;
    h.dim[1] = nx as i16;
    h.dim[2] = ny as i16;
    h.dim[3] = nz.max(1) as i16;
    h.dim[4] = nt.max(1) as i16;
    h.dim[5] = 1;
    h.dim[6] = 1;
    h.dim[7] = 1;
    h.dim[0] = if nt > 1 { 4 } else { 3 };
    h.xyzt_units = NIFTI_UNITS_MM + NIFTI_UNITS_SEC;
    let mut orient = d.orient;
    if !d.has_orientation() {
        orient[1] = 1.0;
        orient[5] = 1.0;
    }
    let q = snap_mat44(&nifti_dicom2mat(orient, d.patient_position, xyz_mm).lps_to_ras_f32());
    h.set_sform(&q);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcm_core::matrix::Matrix4;

    #[test]
    fn flip_y_moves_origin_to_last_row() {
        let mut m = Matrix4::identity();
        m.0[0][0] = 2.0;
        m.0[1][1] = 3.0;
        m.0[2][2] = 4.0;
        apply_flip_y_sform(&mut m, 10);
        // voxel (0,9,0) was origin after flip → translation = (0, 27, 0)
        assert!((m.0[1][3] - 27.0).abs() < 1e-5);
        assert!((m.0[1][1] + 3.0).abs() < 1e-5);
    }

    #[test]
    fn uih_t1_pre_ortho_sform() {
        use dcm_dicom::read_header;
        let dir = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/t1_gre_fsp_3d_sag__134917",
        );
        if !dir.is_dir() {
            return;
        }
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "dcm").unwrap_or(false))
            .collect();
        paths.sort();
        let a = read_header(&paths[0]).unwrap();
        let b = read_header(&paths[1]).unwrap();
        let dx = interslice_distance(&a, &b);
        let mut xyz = a.xyz_mm;
        xyz[3] = dx as f64;
        let q0 = nifti_dicom2mat(a.orient, a.patient_position, xyz);
        // clang `-ffp-contract=fast` nifti_dicom2mat z-column
        assert_eq!(q0.0[0][2] as f32, -0.99964922666549683);
        assert_eq!(q0.0[0][1] as f32, 0.01294383592903614);
        let mut q = q0;
        let _ = verify_slice_dir(&a, &b, 160, &mut q);
        q = snap_mat44(&q.lps_to_ras_f32());
        assert_eq!(q.0[0][0] as f32, 0.0022971127182245255);
    }
}
