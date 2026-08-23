//! Port of `nii_setOrtho` (`nii_ortho.cpp`): reorient 3D volumes to nearest
//! orthogonal alignment instead of a simple Y-flip.

use dcm_core::matrix::Matrix4;
use dcm_nifti::Nifti1Header;

type Mat33 = [[f64; 3]; 3];

fn mat_dot_mul33(a: Mat33, b: Mat33) -> Mat33 {
    let mut ret = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ret[i][j] = a[i][j] * b[j][i];
        }
    }
    ret
}

fn get_ortho_residual(orig: Mat33, transform: Mat33) -> f64 {
    let mat = mat_dot_mul33(orig, transform);
    mat.iter().flat_map(|r| r.iter()).sum()
}

fn get_best_orient(r: &Matrix4, flip: [i32; 3]) -> Mat33 {
    let orig = r.linear3();
    let mut best = 0.0f64;
    let mut ret = [[0.0; 3]; 3];
    let candidates: [Mat33; 6] = [
        [
            [flip[0] as f64, 0.0, 0.0],
            [0.0, flip[1] as f64, 0.0],
            [0.0, 0.0, flip[2] as f64],
        ],
        [
            [flip[0] as f64, 0.0, 0.0],
            [0.0, 0.0, flip[1] as f64],
            [flip[2] as f64, 0.0, 0.0],
        ],
        [
            [0.0, flip[0] as f64, 0.0],
            [flip[1] as f64, 0.0, 0.0],
            [0.0, 0.0, flip[2] as f64],
        ],
        [
            [0.0, flip[0] as f64, 0.0],
            [0.0, 0.0, flip[1] as f64],
            [flip[2] as f64, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, flip[0] as f64],
            [flip[1] as f64, 0.0, 0.0],
            [0.0, flip[2] as f64, 0.0],
        ],
        [
            [0.0, 0.0, flip[0] as f64],
            [0.0, flip[1] as f64, 0.0],
            [flip[2] as f64, 0.0, 0.0],
        ],
    ];
    for newmat in candidates {
        let newval = get_ortho_residual(orig, newmat);
        if newval > best {
            best = newval;
            ret = newmat;
        }
    }
    ret
}

fn is_mat44_canonical(r: &Matrix4) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                if r.0[i][j] <= 0.0 {
                    return false;
                }
            } else if r.0[i][j] != 0.0 {
                return false;
            }
        }
    }
    true
}

fn set_orient_vec(m: Mat33) -> [i32; 3] {
    let mut ret = [0i32; 3];
    for i in 0..3 {
        for j in 0..3 {
            if m[i][j] > 0.0 {
                ret[j] = (i + 1) as i32;
            }
            if m[i][j] < 0.0 {
                ret[j] = -((i + 1) as i32);
            }
        }
    }
    ret
}

fn sform_from_header(h: &Nifti1Header) -> Matrix4 {
    Matrix4::from_rows([
        [
            h.srow_x[0] as f64,
            h.srow_x[1] as f64,
            h.srow_x[2] as f64,
            h.srow_x[3] as f64,
        ],
        [
            h.srow_y[0] as f64,
            h.srow_y[1] as f64,
            h.srow_y[2] as f64,
            h.srow_y[3] as f64,
        ],
        [
            h.srow_z[0] as f64,
            h.srow_z[1] as f64,
            h.srow_z[2] as f64,
            h.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn mat33_mul_f32(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    // Match C++ `matMul33`: accumulate from +0.0 so signed-zero products
    // become +0 (`(+0) + (-0) == +0`), not `-0` from a bare sum of products.
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0f32;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

fn xyz2mm_f32(r: &Matrix4, v: [f32; 3]) -> [f32; 3] {
    let m = dcm_core::matrix::snap_mat44(r);
    [
        (m.0[0][0] as f32) * v[0]
            + (m.0[0][1] as f32) * v[1]
            + (m.0[0][2] as f32) * v[2]
            + (m.0[0][3] as f32),
        (m.0[1][0] as f32) * v[0]
            + (m.0[1][1] as f32) * v[1]
            + (m.0[1][2] as f32) * v[2]
            + (m.0[1][3] as f32),
        (m.0[2][0] as f32) * v[0]
            + (m.0[2][1] as f32) * v[1]
            + (m.0[2][2] as f32) * v[2]
            + (m.0[2][3] as f32),
    ]
}

fn min_corner_flip(h: &Nifti1Header) -> ([i32; 3], [f32; 3]) {
    let s = sform_from_header(h);
    let mut flip_vecs = [[0i32; 3]; 8];
    let mut corners = [[0.0f32; 3]; 8];
    for i in 0..8 {
        flip_vecs[i][0] = if i & 1 != 0 { -1 } else { 1 };
        flip_vecs[i][1] = if i & 2 != 0 { -1 } else { 1 };
        flip_vecs[i][2] = if i & 4 != 0 { -1 } else { 1 };
        let mut c = [0.0f32; 3];
        if flip_vecs[i][0] < 1 {
            c[0] = (h.dim[1] - 1) as f32;
        }
        if flip_vecs[i][1] < 1 {
            c[1] = (h.dim[2] - 1) as f32;
        }
        if flip_vecs[i][2] < 1 {
            c[2] = (h.dim[3] - 1) as f32;
        }
        corners[i] = xyz2mm_f32(&s, c);
    }
    let mut min = corners[0];
    for i in 1..8 {
        for j in 0..3 {
            if corners[i][j] < min[j] {
                min[j] = corners[i][j];
            }
        }
    }
    let dist = |c: [f32; 3]| -> f32 {
        ((c[0] - min[0]).powi(2) + (c[1] - min[1]).powi(2) + (c[2] - min[2]).powi(2)).sqrt()
    };
    let mut min_idx = 0usize;
    let mut min_dx = dist(corners[0]);
    for i in 1..8 {
        let dx = dist(corners[i]);
        if dx < min_dx {
            min_dx = dx;
            min_idx = i;
        }
    }
    (flip_vecs[min_idx], corners[min_idx])
}

/// Reorient `vol` to nearest orthogonal alignment; updates `hdr` dims/sform.
pub fn nii_set_ortho_f32(mut vol: Vec<f32>, hdr: &mut Nifti1Header) -> Vec<f32> {
    if hdr.dim[1] < 1 || hdr.dim[2] < 1 || hdr.dim[3] < 1 {
        return vol;
    }
    if hdr.sform_code == 0 && hdr.qform_code != 0 {
        // q-only: set_sform already populated both in our pipeline
    }
    if hdr.sform_code == 0 {
        return vol;
    }
    let s = sform_from_header(hdr);
    if is_mat44_canonical(&s) {
        return vol;
    }
    let (flip_v, min_mm) = min_corner_flip(hdr);
    let orient = get_best_orient(&s, flip_v);
    let orient_vec = set_orient_vec(orient);
    if orient_vec == [1, 2, 3] {
        return vol;
    }
    let nx = hdr.dim[1] as usize;
    let ny = hdr.dim[2] as usize;
    let nz = hdr.dim[3] as usize;
    let nt = (4..8).fold(1usize, |acc, v| {
        let d = hdr.dim[v];
        if d > 1 {
            acc * d as usize
        } else {
            acc
        }
    });
    let (new_vol, out_dim) = crate::voxels::reorient_volume(vol, nx, ny, nz, nt, orient_vec);
    vol = new_vol;
    let out_pix = [
        hdr.pixdim[orient_vec[0].unsigned_abs() as usize],
        hdr.pixdim[orient_vec[1].unsigned_abs() as usize],
        hdr.pixdim[orient_vec[2].unsigned_abs() as usize],
    ];
    hdr.dim[1] = out_dim[0] as i16;
    hdr.dim[2] = out_dim[1] as i16;
    hdr.dim[3] = out_dim[2] as i16;
    hdr.pixdim[1] = out_pix[0];
    hdr.pixdim[2] = out_pix[1];
    hdr.pixdim[3] = out_pix[2];
    let mat = [
        [hdr.srow_x[0], hdr.srow_x[1], hdr.srow_x[2]],
        [hdr.srow_y[0], hdr.srow_y[1], hdr.srow_y[2]],
        [hdr.srow_z[0], hdr.srow_z[1], hdr.srow_z[2]],
    ];
    let orient_f32 = [
        [
            orient[0][0] as f32,
            orient[0][1] as f32,
            orient[0][2] as f32,
        ],
        [
            orient[1][0] as f32,
            orient[1][1] as f32,
            orient[1][2] as f32,
        ],
        [
            orient[2][0] as f32,
            orient[2][1] as f32,
            orient[2][2] as f32,
        ],
    ];
    let new_linear = mat33_mul_f32(mat, orient_f32);
    let new_s = dcm_core::matrix::snap_mat44(&Matrix4::from_rows([
        [
            new_linear[0][0] as f64,
            new_linear[0][1] as f64,
            new_linear[0][2] as f64,
            min_mm[0] as f64,
        ],
        [
            new_linear[1][0] as f64,
            new_linear[1][1] as f64,
            new_linear[1][2] as f64,
            min_mm[1] as f64,
        ],
        [
            new_linear[2][0] as f64,
            new_linear[2][1] as f64,
            new_linear[2][2] as f64,
            min_mm[2] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]));
    hdr.set_sform(&new_s);
    vol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{apply_flip_z_sform, verify_slice_dir};
    use dcm_core::matrix::{nifti_dicom2mat, snap_mat44};
    use dcm_dicom::read_header;
    use dcm_nifti::Nifti1Header;

    #[test]
    fn uih_t1_ortho_vec() {
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
        let first = read_header(&paths[0]).unwrap();
        let second = read_header(&paths[1]).unwrap();
        let nx = first.columns as i16;
        let ny = first.rows as i16;
        let nz = paths.len() as i16;
        let mut hdr = Nifti1Header::default();
        hdr.dim[0] = 3;
        hdr.dim[1] = nx;
        hdr.dim[2] = ny;
        hdr.dim[3] = nz;
        hdr.pixdim[1] = first.xyz_mm[1] as f32;
        hdr.pixdim[2] = first.xyz_mm[2] as f32;
        hdr.pixdim[3] = first.xyz_mm[3] as f32;
        hdr.sform_code = 1;
        let mut q = nifti_dicom2mat(first.orient, first.patient_position, first.xyz_mm);
        let flip = verify_slice_dir(&first, &second, nz as usize, &mut q);
        q = snap_mat44(&q.lps_to_ras_f32());
        hdr.set_sform(&q);
        if flip {
            let mut sform = sform_from_header(&hdr);
            apply_flip_z_sform(&mut sform, nz as usize);
            hdr.set_sform(&sform);
        }
        let s = sform_from_header(&hdr);
        eprintln!("canonical={}", is_mat44_canonical(&s));
        let (flip_v, _) = min_corner_flip(&hdr);
        let orient = get_best_orient(&s, flip_v);
        let orient_vec = set_orient_vec(orient);
        eprintln!("orient_vec={orient_vec:?} flip_z={flip}");
    }
}
