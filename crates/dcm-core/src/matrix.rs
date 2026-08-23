//! 4x4 homogeneous matrices used for DICOM LPS → NIfTI RAS.
//!
//! Arithmetic is `f64`. NIfTI stores `f32` sform rows; rounding happens at
//! write time, matching dcm2niix filling `srow_*` from `mat44` floats.

use std::fmt;
use std::ops::{Index, IndexMut, Mul};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix4(pub [[f64; 4]; 4]);

impl Default for Matrix4 {
    fn default() -> Self {
        Matrix4::identity()
    }
}

impl Index<(usize, usize)> for Matrix4 {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.0[r][c]
    }
}

impl IndexMut<(usize, usize)> for Matrix4 {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.0[r][c]
    }
}

impl Matrix4 {
    pub const fn identity() -> Self {
        Matrix4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub const fn from_rows(rows: [[f64; 4]; 4]) -> Self {
        Matrix4(rows)
    }

    pub fn apply_point(&self, p: [f64; 3]) -> [f64; 3] {
        let x = self.0[0][0] * p[0] + self.0[0][1] * p[1] + self.0[0][2] * p[2] + self.0[0][3];
        let y = self.0[1][0] * p[0] + self.0[1][1] * p[1] + self.0[1][2] * p[2] + self.0[1][3];
        let z = self.0[2][0] * p[0] + self.0[2][1] * p[1] + self.0[2][2] * p[2] + self.0[2][3];
        [x, y, z]
    }

    pub fn mul_mat(&self, rhs: &Matrix4) -> Matrix4 {
        let mut out = Matrix4([[0.0; 4]; 4]);
        for i in 0..4 {
            for j in 0..4 {
                out.0[i][j] = (0..4).map(|k| self.0[i][k] * rhs.0[k][j]).sum();
            }
        }
        out
    }

    /// Negate the first two rows: DICOM LPS → NIfTI RAS.
    pub fn lps_to_ras(&self) -> Matrix4 {
        let mut out = *self;
        for c in 0..4 {
            out.0[0][c] = -out.0[0][c];
            out.0[1][c] = -out.0[1][c];
        }
        out
    }

    /// LPS→RAS through `mat44` float storage (C++ `set_nii_header_x`).
    pub fn lps_to_ras_f32(&self) -> Matrix4 {
        let m = snap_mat44(self);
        Matrix4::from_rows([
            [
                (-(m.0[0][0] as f32)) as f64,
                (-(m.0[0][1] as f32)) as f64,
                (-(m.0[0][2] as f32)) as f64,
                (-(m.0[0][3] as f32)) as f64,
            ],
            [
                (-(m.0[1][0] as f32)) as f64,
                (-(m.0[1][1] as f32)) as f64,
                (-(m.0[1][2] as f32)) as f64,
                (-(m.0[1][3] as f32)) as f64,
            ],
            [m.0[2][0], m.0[2][1], m.0[2][2], m.0[2][3]],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn linear3(&self) -> [[f64; 3]; 3] {
        [
            [self.0[0][0], self.0[0][1], self.0[0][2]],
            [self.0[1][0], self.0[1][1], self.0[1][2]],
            [self.0[2][0], self.0[2][1], self.0[2][2]],
        ]
    }

    pub fn set_linear3(&mut self, m: [[f64; 3]; 3]) {
        for r in 0..3 {
            for c in 0..3 {
                self.0[r][c] = m[r][c];
            }
        }
    }

    pub fn translation(&self) -> [f64; 3] {
        [self.0[0][3], self.0[1][3], self.0[2][3]]
    }

    pub fn set_translation(&mut self, t: [f64; 3]) {
        self.0[0][3] = t[0];
        self.0[1][3] = t[1];
        self.0[2][3] = t[2];
    }

    pub fn approx_eq(&self, other: &Matrix4, eps: f64) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps))
    }
}

impl Mul for Matrix4 {
    type Output = Matrix4;
    fn mul(self, rhs: Matrix4) -> Matrix4 {
        self.mul_mat(&rhs)
    }
}

impl fmt::Display for Matrix4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.4} {:.4} {:.4} {:.4}; {:.4} {:.4} {:.4} {:.4}; {:.4} {:.4} {:.4} {:.4}]",
            self.0[0][0],
            self.0[0][1],
            self.0[0][2],
            self.0[0][3],
            self.0[1][0],
            self.0[1][1],
            self.0[1][2],
            self.0[1][3],
            self.0[2][0],
            self.0[2][1],
            self.0[2][2],
            self.0[2][3],
        )
    }
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

pub fn normalise(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

pub fn mat33_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

pub fn mat33_det(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub fn mat33_transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// `nifti_dicom2mat`: IOP (1-indexed 6-vector), IPP, voxel sizes.
///
/// Matches C++ `nifti_dicom2mat` which stores a `mat33` of **float** after
/// every step (normalize / cross / transpose / scale).
pub fn nifti_dicom2mat(orient: [f64; 7], patient_position: [f64; 4], xyz_mm: [f64; 4]) -> Matrix4 {
    let orient: [f32; 7] = orient.map(|v| v as f32);
    let patient_position: [f32; 4] = patient_position.map(|v| v as f32);
    let xyz_mm: [f32; 4] = xyz_mm.map(|v| v as f32);
    let mut q = [
        [orient[1], orient[2], orient[3]],
        [orient[4], orient[5], orient[6]],
        [0.0f32, 0.0, 0.0],
    ];
    // C++ `double val = Q.m[0][0]*Q.m[0][0] + …` with clang `-ffp-contract=fast`
    // becomes FMADD chain: fma(c,c, fma(a,a, b*b)). Match that (plain f32
    // sum-of-squares is ~1 ULP off and cascades into sform after ortho).
    let mut val = q[0][2].mul_add(q[0][2], q[0][0].mul_add(q[0][0], q[0][1] * q[0][1])) as f64;
    if val > 0.0 {
        val = 1.0 / val.sqrt();
        let s = val as f32;
        q[0][0] *= s;
        q[0][1] *= s;
        q[0][2] *= s;
    } else {
        q[0] = [1.0, 0.0, 0.0];
    }
    val = q[1][2].mul_add(q[1][2], q[1][0].mul_add(q[1][0], q[1][1] * q[1][1])) as f64;
    if val > 0.0 {
        val = 1.0 / val.sqrt();
        let s = val as f32;
        q[1][0] *= s;
        q[1][1] *= s;
        q[1][2] *= s;
    } else {
        q[1] = [0.0, 1.0, 0.0];
    }
    // Cross product: C/clang contracts `a*b - c*d` to FMA on aarch64.
    q[2][0] = q[0][1].mul_add(q[1][2], -(q[0][2] * q[1][1]));
    q[2][1] = q[0][2].mul_add(q[1][0], -(q[0][0] * q[1][2]));
    q[2][2] = q[0][0].mul_add(q[1][1], -(q[0][1] * q[1][0]));
    let mut q = mat33_transpose_f32(q);
    if mat33_det_f32(q) < 0.0 {
        q[0][2] = -q[0][2];
        q[1][2] = -q[1][2];
        q[2][2] = -q[2][2];
    }
    let diag = [
        [xyz_mm[1], 0.0, 0.0],
        [0.0, xyz_mm[2], 0.0],
        [0.0, 0.0, xyz_mm[3]],
    ];
    let q = mat33_mul_f32(q, diag);
    Matrix4::from_rows([
        [
            q[0][0] as f64,
            q[0][1] as f64,
            q[0][2] as f64,
            patient_position[1] as f64,
        ],
        [
            q[1][0] as f64,
            q[1][1] as f64,
            q[1][2] as f64,
            patient_position[2] as f64,
        ],
        [
            q[2][0] as f64,
            q[2][1] as f64,
            q[2][2] as f64,
            patient_position[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn mat33_mul_f32(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // Non-contracted: Siemens mosaic sform matches Ref without FMA here.
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mat33_det_f32(m: [[f32; 3]; 3]) -> f32 {
    let r11 = m[0][0] as f64;
    let r12 = m[0][1] as f64;
    let r13 = m[0][2] as f64;
    let r21 = m[1][0] as f64;
    let r22 = m[1][1] as f64;
    let r23 = m[1][2] as f64;
    let r31 = m[2][0] as f64;
    let r32 = m[2][1] as f64;
    let r33 = m[2][2] as f64;
    (r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23
        - r31 * r22 * r13) as f32
}

fn mat33_transpose_f32(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Closest rotation via Higham polar factor (`nifti_mat33_polar`).
fn mat33_polar_f32(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut x = a;
    let mut gam = mat33_det_f32(x);
    while gam == 0.0 {
        gam = 0.00001 * (0.001 + mat33_rownorm_f32(x));
        x[0][0] += gam;
        x[1][1] += gam;
        x[2][2] += gam;
        gam = mat33_det_f32(x);
    }
    let mut dif = 1.0f32;
    let mut k = 0;
    let mut z;
    loop {
        let y = mat33_inv_f32(x);
        let (gam, gmi) = if dif > 0.3 {
            let alp = (mat33_rownorm_f32(x) * mat33_colnorm_f32(x)).sqrt();
            let bet = (mat33_rownorm_f32(y) * mat33_colnorm_f32(y)).sqrt();
            let gam = (bet / alp).sqrt();
            (gam, 1.0 / gam)
        } else {
            (1.0, 1.0)
        };
        z = [
            [
                0.5 * (gam * x[0][0] + gmi * y[0][0]),
                0.5 * (gam * x[0][1] + gmi * y[1][0]),
                0.5 * (gam * x[0][2] + gmi * y[2][0]),
            ],
            [
                0.5 * (gam * x[1][0] + gmi * y[0][1]),
                0.5 * (gam * x[1][1] + gmi * y[1][1]),
                0.5 * (gam * x[1][2] + gmi * y[2][1]),
            ],
            [
                0.5 * (gam * x[2][0] + gmi * y[0][2]),
                0.5 * (gam * x[2][1] + gmi * y[1][2]),
                0.5 * (gam * x[2][2] + gmi * y[2][2]),
            ],
        ];
        dif = (z[0][0] - x[0][0]).abs()
            + (z[0][1] - x[0][1]).abs()
            + (z[0][2] - x[0][2]).abs()
            + (z[1][0] - x[1][0]).abs()
            + (z[1][1] - x[1][1]).abs()
            + (z[1][2] - x[1][2]).abs()
            + (z[2][0] - x[2][0]).abs()
            + (z[2][1] - x[2][1]).abs()
            + (z[2][2] - x[2][2]).abs();
        k += 1;
        if k > 100 || dif < 3.0e-6 {
            break;
        }
        x = z;
    }
    z
}

fn mat33_rownorm_f32(a: [[f32; 3]; 3]) -> f32 {
    let r1 = a[0][0].abs() + a[0][1].abs() + a[0][2].abs();
    let r2 = a[1][0].abs() + a[1][1].abs() + a[1][2].abs();
    let r3 = a[2][0].abs() + a[2][1].abs() + a[2][2].abs();
    r1.max(r2).max(r3)
}

fn mat33_colnorm_f32(a: [[f32; 3]; 3]) -> f32 {
    let r1 = a[0][0].abs() + a[1][0].abs() + a[2][0].abs();
    let r2 = a[0][1].abs() + a[1][1].abs() + a[2][1].abs();
    let r3 = a[0][2].abs() + a[1][2].abs() + a[2][2].abs();
    r1.max(r2).max(r3)
}

fn mat33_inv_f32(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let r11 = m[0][0] as f64;
    let r12 = m[0][1] as f64;
    let r13 = m[0][2] as f64;
    let r21 = m[1][0] as f64;
    let r22 = m[1][1] as f64;
    let r23 = m[1][2] as f64;
    let r31 = m[2][0] as f64;
    let r32 = m[2][1] as f64;
    let r33 = m[2][2] as f64;
    let mut deti =
        r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23
            - r31 * r22 * r13;
    if deti != 0.0 {
        deti = 1.0 / deti;
    }
    [
        [
            (deti * (r22 * r33 - r32 * r23)) as f32,
            (deti * (-r12 * r33 + r32 * r13)) as f32,
            (deti * (r12 * r23 - r22 * r13)) as f32,
        ],
        [
            (deti * (-r21 * r33 + r31 * r23)) as f32,
            (deti * (r11 * r33 - r31 * r13)) as f32,
            (deti * (-r11 * r23 + r21 * r13)) as f32,
        ],
        [
            (deti * (r21 * r32 - r31 * r22)) as f32,
            (deti * (-r11 * r32 + r31 * r12)) as f32,
            (deti * (r11 * r22 - r21 * r12)) as f32,
        ],
    ]
}

/// Snap a Matrix4 through C++ `mat44` float storage.
pub fn snap_mat44(m: &Matrix4) -> Matrix4 {
    Matrix4::from_rows([
        [
            m.0[0][0] as f32 as f64,
            m.0[0][1] as f32 as f64,
            m.0[0][2] as f32 as f64,
            m.0[0][3] as f32 as f64,
        ],
        [
            m.0[1][0] as f32 as f64,
            m.0[1][1] as f32 as f64,
            m.0[1][2] as f32 as f64,
            m.0[1][3] as f32 as f64,
        ],
        [
            m.0[2][0] as f32 as f64,
            m.0[2][1] as f32 as f64,
            m.0[2][2] as f32 as f64,
            m.0[2][3] as f32 as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// NIfTI qform quaternion from an sform (`nifti_mat44_to_quatern`).
pub fn mat44_to_quatern(r: &Matrix4) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    // C++ mat44 is float; load through f32 then promote to f64 for the polar path.
    let r = snap_mat44(r);
    let qx = r.0[0][3] as f32;
    let qy = r.0[1][3] as f32;
    let qz = r.0[2][3] as f32;
    let mut r11 = r.0[0][0];
    let mut r12 = r.0[0][1];
    let mut r13 = r.0[0][2];
    let mut r21 = r.0[1][0];
    let mut r22 = r.0[1][1];
    let mut r23 = r.0[1][2];
    let mut r31 = r.0[2][0];
    let mut r32 = r.0[2][1];
    let mut r33 = r.0[2][2];
    let mut xd = (r11 * r11 + r21 * r21 + r31 * r31).sqrt();
    let mut yd = (r12 * r12 + r22 * r22 + r32 * r32).sqrt();
    let mut zd = (r13 * r13 + r23 * r23 + r33 * r33).sqrt();
    if xd == 0.0 {
        r11 = 1.0;
        r21 = 0.0;
        r31 = 0.0;
        xd = 1.0;
    }
    if yd == 0.0 {
        r22 = 1.0;
        r12 = 0.0;
        r32 = 0.0;
        yd = 1.0;
    }
    if zd == 0.0 {
        r33 = 1.0;
        r13 = 0.0;
        r23 = 0.0;
        zd = 1.0;
    }
    r11 /= xd;
    r21 /= xd;
    r31 /= xd;
    r12 /= yd;
    r22 /= yd;
    r32 /= yd;
    r13 /= zd;
    r23 /= zd;
    r33 /= zd;
    let dx = xd as f32;
    let dy = yd as f32;
    let dz = zd as f32;
    // Polar works on float mat33 (cast each element like C++ loading Q from doubles).
    let qf = [
        [r11 as f32, r12 as f32, r13 as f32],
        [r21 as f32, r22 as f32, r23 as f32],
        [r31 as f32, r32 as f32, r33 as f32],
    ];
    let p = mat33_polar_f32(qf);
    r11 = p[0][0] as f64;
    r12 = p[0][1] as f64;
    r13 = p[0][2] as f64;
    r21 = p[1][0] as f64;
    r22 = p[1][1] as f64;
    r23 = p[1][2] as f64;
    r31 = p[2][0] as f64;
    r32 = p[2][1] as f64;
    r33 = p[2][2] as f64;
    let det = r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23
        - r31 * r22 * r13;
    let qfac: f32;
    if det > 0.0 {
        qfac = 1.0;
    } else {
        qfac = -1.0;
        r13 = -r13;
        r23 = -r23;
        r33 = -r33;
    }
    let mut a = r11 + r22 + r33 + 1.0;
    let (qb, qc, qd);
    if a > 0.5 {
        a = 0.5 * a.sqrt();
        qb = 0.25 * (r32 - r23) / a;
        qc = 0.25 * (r13 - r31) / a;
        qd = 0.25 * (r21 - r12) / a;
    } else {
        xd = 1.0 + r11 - (r22 + r33);
        yd = 1.0 + r22 - (r11 + r33);
        zd = 1.0 + r33 - (r11 + r22);
        if xd > 1.0 {
            let b = 0.5 * xd.sqrt();
            qb = b;
            qc = 0.25 * (r12 + r21) / b;
            qd = 0.25 * (r13 + r31) / b;
            a = 0.25 * (r32 - r23) / b;
        } else if yd > 1.0 {
            let c = 0.5 * yd.sqrt();
            qc = c;
            qb = 0.25 * (r12 + r21) / c;
            qd = 0.25 * (r23 + r32) / c;
            a = 0.25 * (r13 - r31) / c;
        } else {
            let d = 0.5 * zd.max(0.0).sqrt();
            let d = if d == 0.0 { 1e-12 } else { d };
            qd = d;
            qb = 0.25 * (r13 + r31) / d;
            qc = 0.25 * (r23 + r32) / d;
            a = 0.25 * (r21 - r12) / d;
        }
        if a < 0.0 {
            return ((-qb) as f32, (-qc) as f32, (-qd) as f32, qx, qy, qz, dx, dy, dz, qfac);
        }
    }
    (qb as f32, qc as f32, qd as f32, qx, qy, qz, dx, dy, dz, qfac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lps_to_ras_negates_first_two_rows() {
        let m = Matrix4::from_rows([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let r = m.lps_to_ras();
        assert_eq!(r.0[0], [-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(r.0[1], [-5.0, -6.0, -7.0, -8.0]);
        assert_eq!(r.0[2], [9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn dicom2mat_axial_identity_scaled() {
        let mut orient = [0.0; 7];
        orient[1] = 1.0;
        orient[5] = 1.0;
        let mut pos = [0.0; 4];
        pos[1] = -90.0;
        pos[2] = -120.0;
        pos[3] = 40.0;
        let mut mm = [0.0; 4];
        mm[1] = 1.0;
        mm[2] = 2.0;
        mm[3] = 3.0;
        let m = nifti_dicom2mat(orient, pos, mm);
        assert!((m.0[0][0] - 1.0).abs() < 1e-9);
        assert!((m.0[1][1] - 2.0).abs() < 1e-9);
        assert!((m.0[2][2] - 3.0).abs() < 1e-9);
        assert!((m.0[0][3] + 90.0).abs() < 1e-9);
    }

    #[test]
    fn uih_ap_dicom2mat_matches_cpp() {
        let mut orient = [0.0; 7];
        orient[1] = atof_f32("0.999113142");
        orient[2] = atof_f32("0.0305822305");
        orient[3] = atof_f32("0.0289414488");
        orient[4] = atof_f32("-0.0305694081");
        orient[5] = atof_f32("0.999532282");
        orient[6] = atof_f32("-0.000885508256");
        let mut pos = [0.0; 4];
        pos[1] = atof_f32("-114.442863");
        pos[2] = atof_f32("-112.324295");
        pos[3] = atof_f32("-6.39442587");
        let mut mm = [0.0; 4];
        mm[1] = 3.5;
        mm[2] = 3.5;
        mm[3] = atof_f32("4.19999981");
        let m = nifti_dicom2mat(orient, pos, mm);
        assert_eq!(m.0[0][0] as f32, 3.49689603);
        assert_eq!(m.0[0][1] as f32, -0.106992923);
        assert_eq!(m.0[1][1] as f32, 3.49836278);
        assert_eq!(m.0[1][2] as f32, -1.86499705e-10);
    }

    #[test]
    fn uih_t1_dicom2mat_matches_clang_fma() {
        // Exact bits from dcm_qa_uih t1_gre_fsp_3d_sag 134431 slice 0.
        let bits = |u: u32| -> f64 { f32::from_bits(u) as f64 };
        let mut orient = [0.0; 7];
        for (i, u) in [
            0u32, 3147205416, 1065353013, 988234670, 1020531298, 989251393, 3212831212,
        ]
        .into_iter()
        .enumerate()
        {
            orient[i] = bits(u);
        }
        let mut pos = [0.0; 4];
        for (i, u) in [2143289344u32, 3266649257, 3269588860, 1123087571]
            .into_iter()
            .enumerate()
        {
            pos[i] = bits(u);
        }
        let mut mm = [0.0; 4];
        mm[1] = 0.5;
        mm[2] = 0.5;
        mm[3] = bits(0x3f7fffaa);
        let m = nifti_dicom2mat(orient, pos, mm);
        assert_eq!(m.0[0][2] as f32, -0.99964922666549683);
        assert_eq!(m.0[0][1] as f32, 0.01294383592903614);
    }

    fn atof_f32(s: &str) -> f64 {
        (s.parse::<f64>().unwrap() as f32) as f64
    }
}

