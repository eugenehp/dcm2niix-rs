//! Volume voxel ops via rlx (CPU by default; optional wgpu GPU).
//!
//! # Layout
//!
//! NIfTI `[t][z][y][x]` with `x` fastest. Integer DICOM samples ride in `f32`
//! (exact for every ≤16-bit stored value).
//!
//! Direct CPU flips / gathers are bit-identical to the rlx graph path on the
//! same axes (flip, or flip+transpose for ortho reorient).
//!
//! # Device selection
//!
//! | `DCM2NIIX_RLX_DEVICE` | Behavior |
//! | --- | --- |
//! | `auto` (default) | Direct CPU below ~8 MiB; with feature `gpu`, rlx/wgpu above |
//! | `cpu` | Always direct in-place host flips / gathers |
//! | `gpu` | Always compile through rlx (`Device::Gpu` when feature `gpu` is on) |

use rayon::prelude::*;
use rlx_tensor::{Device, Tensor};

/// Bytes above which GPU realize is considered (H2D/D2H otherwise dominates).
const GPU_MIN_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DevicePref {
    Cpu,
    Gpu,
    Auto,
}

fn device_pref() -> DevicePref {
    match std::env::var("DCM2NIIX_RLX_DEVICE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" | "host" => DevicePref::Cpu,
        "gpu" | "wgpu" => DevicePref::Gpu,
        _ => DevicePref::Auto,
    }
}

fn realize(t: Tensor, n_bytes: usize) -> Vec<f32> {
    let pref = device_pref();
    #[cfg(feature = "gpu")]
    {
        let want_gpu = match pref {
            DevicePref::Cpu => false,
            DevicePref::Gpu => true,
            DevicePref::Auto => n_bytes >= GPU_MIN_BYTES,
        };
        if want_gpu {
            return t.to_vec_on(Device::Gpu);
        }
        return t.to_vec_on(Device::Cpu);
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (pref, n_bytes);
        t.to_vec_on(Device::Cpu)
    }
}

/// Fast host flip of axis `y` (rows) — bit-identical to rlx `flip(2)`.
/// Swaps whole rows; parallel over slices when there are many.
fn flip_y_cpu(mut data: Vec<f32>, nx: usize, ny: usize, nz: usize, nt: usize) -> Vec<f32> {
    if ny <= 1 {
        return data;
    }
    let slice = nx * ny;
    let n_planes = nz * nt;
    if n_planes >= 4 {
        // Disjoint `[t][z]` planes — safe to mutate in parallel.
        data.par_chunks_mut(slice).for_each(|plane| {
            for y in 0..ny / 2 {
                let a = y * nx;
                let b = (ny - 1 - y) * nx;
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let (left, right) = plane[lo..hi + nx].split_at_mut(hi - lo);
                let rlen = right.len();
                left[..nx].swap_with_slice(&mut right[rlen - nx..]);
            }
        });
        return data;
    }
    let vol = slice * nz;
    for t in 0..nt {
        for z in 0..nz {
            let base = t * vol + z * slice;
            for y in 0..ny / 2 {
                let a = base + y * nx;
                let b = base + (ny - 1 - y) * nx;
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let (left, right) = data[lo..hi + nx].split_at_mut(hi - lo);
                let rlen = right.len();
                left[..nx].swap_with_slice(&mut right[rlen - nx..]);
            }
        }
    }
    data
}

/// Fast host flip of axis `z` — bit-identical to rlx `flip(1)`.
/// Swaps whole slices.
fn flip_z_cpu(mut data: Vec<f32>, nx: usize, ny: usize, nz: usize, nt: usize) -> Vec<f32> {
    if nz <= 1 {
        return data;
    }
    let slice = nx * ny;
    let vol = slice * nz;
    for t in 0..nt {
        let base = t * vol;
        for z in 0..nz / 2 {
            let a = base + z * slice;
            let b = base + (nz - 1 - z) * slice;
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let (left, right) = data[lo..hi + slice].split_at_mut(hi - lo);
            let rlen = right.len();
            left[..slice].swap_with_slice(&mut right[rlen - slice..]);
        }
    }
    data
}

fn use_direct_cpu(n: usize) -> bool {
    // Avoid rlx compile latency on typical series; GPU path starts at GPU_MIN_BYTES.
    match device_pref() {
        DevicePref::Gpu => false,
        DevicePref::Cpu => true,
        DevicePref::Auto => !cfg!(feature = "gpu") || n * 4 < GPU_MIN_BYTES,
    }
}

/// Flip `y` (rows) — dcm2niix `nii_flipY`.
pub fn flip_y_volume(data: Vec<f32>, nx: usize, ny: usize, nz: usize, nt: usize) -> Vec<f32> {
    let n = nx * ny * nz * nt;
    assert_eq!(data.len(), n, "volume length {} != {n}", data.len());
    if ny <= 1 {
        return data;
    }
    if use_direct_cpu(n) {
        return flip_y_cpu(data, nx, ny, nz, nt);
    }
    let t = Tensor::from_vec(data, [nt, nz, ny, nx]);
    realize(t.flip(2), n * 4)
}

/// Flip slice order — dcm2niix `nii_flipImgZ`.
pub fn flip_z_volume(data: Vec<f32>, nx: usize, ny: usize, nz: usize, nt: usize) -> Vec<f32> {
    let n = nx * ny * nz * nt;
    assert_eq!(data.len(), n, "volume length {} != {n}", data.len());
    if nz <= 1 {
        return data;
    }
    if use_direct_cpu(n) {
        return flip_z_cpu(data, nx, ny, nz, nt);
    }
    let t = Tensor::from_vec(data, [nt, nz, ny, nx]);
    realize(t.flip(1), n * 4)
}

/// Fuse Y and/or Z flips into one rlx graph (or one CPU pass).
pub fn flip_yz_volume(
    data: Vec<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    flip_y: bool,
    flip_z: bool,
) -> Vec<f32> {
    match (flip_y && ny > 1, flip_z && nz > 1) {
        (false, false) => data,
        (true, false) => flip_y_volume(data, nx, ny, nz, nt),
        (false, true) => flip_z_volume(data, nx, ny, nz, nt),
        (true, true) => {
            let n = nx * ny * nz * nt;
            assert_eq!(data.len(), n, "volume length {} != {n}", data.len());
            if use_direct_cpu(n) {
                let data = flip_z_cpu(data, nx, ny, nz, nt);
                return flip_y_cpu(data, nx, ny, nz, nt);
            }
            let t = Tensor::from_vec(data, [nt, nz, ny, nx]);
            realize(t.flip(1).flip(2), n * 4)
        }
    }
}

/// Ortho reorient (`nii_setOrtho` voxel gather): `orient_vec` is ±1/±2/±3 for
/// out-x/y/z source axes (1=x, 2=y, 3=z), matching C++ `setOrientVec`.
///
/// Returns `(out_vol, [out_nx, out_ny, out_nz])`. Layout stays `[t][z][y][x]`.
pub fn reorient_volume(
    data: Vec<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    orient_vec: [i32; 3],
) -> (Vec<f32>, [usize; 3]) {
    let n = nx * ny * nz * nt;
    assert_eq!(data.len(), n, "volume length {} != {n}", data.len());
    if orient_vec == [1, 2, 3] {
        return (data, [nx, ny, nz]);
    }
    if use_direct_cpu(n) {
        return reorient_cpu(data, nx, ny, nz, nt, orient_vec);
    }
    reorient_rlx(data, nx, ny, nz, nt, orient_vec)
}

fn ortho_offset_lut(dim: usize, step: isize) -> Vec<isize> {
    let mut lut = vec![0isize; dim.max(1)];
    if dim > 0 {
        lut[0] = if step > 0 {
            0
        } else {
            -step * (dim as isize - 1)
        };
        for i in 1..dim {
            lut[i] = lut[i - 1] + step;
        }
    }
    lut
}

/// Direct host gather — bit-identical to the historical `reorient_f32` path.
/// Parallel over volumes when `nt > 1`.
fn reorient_cpu(
    vol: Vec<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    orient_vec: [i32; 3],
) -> (Vec<f32>, [usize; 3]) {
    let in_dim = [nx, ny, nz];
    let mut out_dim = [0usize; 3];
    let mut out_inc = [0isize; 3];
    for i in 0..3 {
        let src = orient_vec[i].unsigned_abs() as usize;
        out_dim[i] = in_dim[src - 1];
        out_inc[i] = match src {
            1 => 1,
            2 => nx as isize,
            3 => (nx * ny) as isize,
            _ => 1,
        };
        if orient_vec[i] < 0 {
            out_inc[i] = -out_inc[i];
        }
    }
    let spatial = nx * ny * nz;
    let out_spatial = out_dim[0] * out_dim[1] * out_dim[2];
    let x_lut = ortho_offset_lut(out_dim[0], out_inc[0]);
    let y_lut = ortho_offset_lut(out_dim[1], out_inc[1]);
    let z_lut = ortho_offset_lut(out_dim[2], out_inc[2]);

    let mut out = vec![0.0f32; out_spatial * nt];
    if nt == 1 {
        reorient_one_vol(
            &vol[..spatial],
            &mut out[..out_spatial],
            &x_lut,
            &y_lut,
            &z_lut,
            out_dim,
        );
    } else {
        out.par_chunks_mut(out_spatial)
            .zip(vol.par_chunks(spatial))
            .for_each(|(dst, src)| {
                reorient_one_vol(src, dst, &x_lut, &y_lut, &z_lut, out_dim);
            });
    }
    (out, out_dim)
}

#[inline]
fn reorient_one_vol(
    src: &[f32],
    dst: &mut [f32],
    x_lut: &[isize],
    y_lut: &[isize],
    z_lut: &[isize],
    out_dim: [usize; 3],
) {
    let mut o = 0usize;
    for z in 0..out_dim[2] {
        let z_off = z_lut[z];
        for y in 0..out_dim[1] {
            let yz = y_lut[y] + z_off;
            for x in 0..out_dim[0] {
                dst[o] = src[(x_lut[x] + yz) as usize];
                o += 1;
            }
        }
    }
}

/// Flip negative source axes, then transpose — equivalent to [`reorient_cpu`].
fn reorient_rlx(
    data: Vec<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    orient_vec: [i32; 3],
) -> (Vec<f32>, [usize; 3]) {
    let n = nx * ny * nz * nt;
    // Tensor axes: 0=t, 1=z, 2=y, 3=x  (matches flip_*_volume).
    let in_axis_to_tensor = |a: usize| -> usize {
        match a {
            0 => 3, // x
            1 => 2, // y
            2 => 1, // z
            _ => unreachable!(),
        }
    };
    let mut t = Tensor::from_vec(data, [nt, nz, ny, nx]);
    for a in 0..3 {
        let src = (a + 1) as i32;
        if orient_vec.iter().any(|&v| v == -src) {
            t = t.flip(in_axis_to_tensor(a));
        }
    }
    let perm = [
        0,
        in_axis_to_tensor(orient_vec[2].unsigned_abs() as usize - 1),
        in_axis_to_tensor(orient_vec[1].unsigned_abs() as usize - 1),
        in_axis_to_tensor(orient_vec[0].unsigned_abs() as usize - 1),
    ];
    let out = realize(t.transpose(perm), n * 4);
    let out_dim = [
        match orient_vec[0].unsigned_abs() {
            1 => nx,
            2 => ny,
            _ => nz,
        },
        match orient_vec[1].unsigned_abs() {
            1 => nx,
            2 => ny,
            _ => nz,
        },
        match orient_vec[2].unsigned_abs() {
            1 => nx,
            2 => ny,
            _ => nz,
        },
    ];
    (out, out_dim)
}

/// Pack slice buffers into a contiguous `[t][z][y][x]` volume in parallel.
pub fn pack_slices(slices: &[Vec<f32>], slice_len: usize) -> Vec<f32> {
    let nt = slices.len();
    let mut vol = vec![0.0f32; slice_len * nt];
    vol.par_chunks_mut(slice_len)
        .zip(slices.par_iter())
        .for_each(|(dst, src)| {
            debug_assert_eq!(src.len(), slice_len);
            dst.copy_from_slice(src);
        });
    vol
}

/// Parallel decode into one volume; uniform `nx`×`ny` slices write in place (no `Vec<Vec<f32>>`).
pub fn decode_stack_slices(
    paths: &[&std::path::Path],
    mmaps: &dcm_dicom::MmapCache,
    nx: usize,
    ny: usize,
) -> dcm_core::error::Result<Vec<f32>> {
    use dcm_core::error::{Error, Result};
    use dcm_dicom::decode_pixels_prefetched;

    let slice_len = nx * ny;
    let n = paths.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut vol = vec![0.0f32; slice_len * n];
    let lens: Vec<Result<usize>> = vol
        .par_chunks_mut(slice_len)
        .zip(paths.par_iter())
        .map(|(dst, &path)| {
            let (pix, r, c) = decode_pixels_prefetched(path, mmaps)?;
            if r != ny || c != nx {
                return Err(Error::convert(format!(
                    "{}: slice is {c}x{r} but series is {nx}x{ny}",
                    path.display()
                )));
            }
            let len = pix.len();
            if len == slice_len {
                dst.copy_from_slice(&pix);
            } else if len < slice_len {
                dst[..len].copy_from_slice(&pix);
                dst[len..].fill(0.0);
            } else {
                dst.copy_from_slice(&pix[..slice_len]);
            }
            Ok(len)
        })
        .collect();
    let lens: Vec<usize> = lens.into_iter().collect::<Result<_>>()?;
    if lens.iter().all(|&l| l == slice_len) {
        return Ok(vol);
    }
    // Incomplete / variable slice sizes (rare): assemble with extend.
    let mut out = Vec::new();
    for &path in paths {
        let (pix, r, c) = decode_pixels_prefetched(path, mmaps)?;
        if r != ny || c != nx {
            return Err(Error::convert(format!(
                "{}: slice is {c}x{r} but series is {nx}x{ny}",
                path.display()
            )));
        }
        out.extend(pix);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_y_reverses_rows() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = flip_y_volume(src, 3, 2, 1, 1);
        assert_eq!(out, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn flip_z_reverses_slices() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let out = flip_z_volume(src, 2, 1, 2, 1);
        assert_eq!(out, vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn fused_yz_matches_sequential() {
        let src: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let a = flip_y_volume(flip_z_volume(src.clone(), 2, 3, 2, 2), 2, 3, 2, 2);
        let b = flip_yz_volume(src, 2, 3, 2, 2, true, true);
        assert_eq!(a, b);
    }

    #[test]
    fn cpu_matches_rlx_y() {
        let src: Vec<f32> = (0..60).map(|v| v as f32).collect();
        let cpu = flip_y_cpu(src.clone(), 5, 4, 3, 1);
        let t = Tensor::from_vec(src, [1, 3, 4, 5]);
        let rlx = t.flip(2).to_vec_on(Device::Cpu);
        assert_eq!(cpu, rlx);
    }

    #[test]
    fn reorient_cpu_matches_rlx() {
        let src: Vec<f32> = (0..3 * 4 * 2 * 2).map(|v| v as f32).collect();
        let cases: [[i32; 3]; 7] = [
            [1, 2, 3],
            [2, 1, 3],
            [-1, 2, 3],
            [1, -2, 3],
            [3, 1, 2],
            [-2, 3, 1],
            [2, -3, 1],
        ];
        for ov in cases {
            let (cpu, d_cpu) = reorient_cpu(src.clone(), 3, 4, 2, 2, ov);
            let (rlx, d_rlx) = reorient_rlx(src.clone(), 3, 4, 2, 2, ov);
            assert_eq!(d_cpu, d_rlx, "dims for {ov:?}");
            assert_eq!(cpu, rlx, "voxels for {ov:?}");
        }
    }

    #[test]
    fn pack_slices_preserves_order() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let out = pack_slices(&[a, b], 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
