//! DTI sidecars (`.bval` / `.bvec`) — port of `nii_saveDTI` + vendor bvec correction.

use dcm_core::format_printf_g_f64;
use dcm_core::matrix::{cross, dot, normalise};

use std::fs::File;
use std::io::Write;
use std::path::Path;

use dcm_core::error::{Error, Result};
use dcm_dicom::{DicomImage, Manufacturer};

use crate::geom::slice_normal;

/// Result of ADC detection + preferred volume order (non-ADC first, ADC last).
#[derive(Debug, Clone)]
pub struct DtiVolumePlan {
    /// Permutation of volume indices: desired output order.
    pub order: Vec<usize>,
    /// Number of trailing ADC / isotropic volumes after reorder.
    pub num_adc: usize,
    /// Volume indices that are ADC (in original order).
    pub adc_indices: Vec<usize>,
}

/// True when volume looks like Philips/GE ADC or isotropic (nonzero b, zero vector).
pub fn is_adc_volume(img: &DicomImage) -> bool {
    if img.b_value < 0.0 {
        return false;
    }
    let min_adc = if img.manufacturer == Manufacturer::Siemens {
        50.0
    } else {
        6.0
    };
    let vlen = img
        .diffusion_direction
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    img.b_value > min_adc && vlen <= f64::EPSILON
}

/// Plan volume reorder: keep DWI first (optionally sorted by b), ADC last.
pub fn plan_dti_volumes(volumes: &[&DicomImage], sort_by_bval: bool) -> DtiVolumePlan {
    let mut dwi = Vec::new();
    let mut adc = Vec::new();
    for (i, v) in volumes.iter().enumerate() {
        if is_adc_volume(v)
            && matches!(
                v.manufacturer,
                Manufacturer::Philips | Manufacturer::Ge
            )
        {
            adc.push(i);
        } else {
            dwi.push(i);
        }
    }
    let mut num_adc = adc.len();
    // C++ disables ADC removal for b=0+trace pairs / isotropic-only series.
    if num_adc == 1 && dwi.len() < 2 {
        eprintln!(
            "Note: this appears to be a b=0+trace DWI; ADC/trace removal has been disabled."
        );
        num_adc = 0;
        dwi.extend(adc.drain(..));
    } else if num_adc > 0 && dwi.len() < 2 {
        eprintln!("Warning: Isotropic DWI series, all bvecs are zero (issue 405)");
        num_adc = 0;
        dwi.extend(adc.drain(..));
    } else if num_adc > 0 {
        eprintln!(
            "Note: {num_adc} volumes appear to be ADC or trace images that will be removed to allow processing"
        );
    }
    if sort_by_bval {
        dwi.sort_by(|&a, &b| {
            volumes[a]
                .b_value
                .partial_cmp(&volumes[b].b_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut order = dwi;
    let adc_indices = adc.clone();
    order.extend(adc);
    // If we disabled removal, order is just all volumes (maybe sorted).
    if num_adc == 0 && order.len() != volumes.len() {
        order = (0..volumes.len()).collect();
        if sort_by_bval {
            order.sort_by(|&a, &b| {
                volumes[a]
                    .b_value
                    .partial_cmp(&volumes[b].b_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
    DtiVolumePlan {
        order,
        num_adc,
        adc_indices,
    }
}

/// Reorder 4D volume bytes by `order` (volume-major, `vol_bytes` each).
pub fn reorder_volume_bytes(bytes: &[u8], vol_bytes: usize, order: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for &i in order {
        let start = i * vol_bytes;
        let end = start + vol_bytes;
        if end <= bytes.len() {
            out.extend_from_slice(&bytes[start..end]);
        } else {
            out.resize(out.len() + vol_bytes, 0);
        }
    }
    out
}

/// Truncate trailing `num_adc` volumes from a 4D NIfTI buffer (C++ `removeADC`).
pub fn remove_trailing_adc(hdr: &mut dcm_nifti::Nifti1Header, bytes: &mut Vec<u8>, num_adc: usize) {
    if num_adc == 0 {
        return;
    }
    let nt = hdr.dim[4].max(1) as usize;
    if num_adc >= nt {
        return;
    }
    let nx = hdr.dim[1].max(1) as usize;
    let ny = hdr.dim[2].max(1) as usize;
    let nz = hdr.dim[3].max(1) as usize;
    let bp = (hdr.bitpix as usize / 8).max(1);
    let keep = nt - num_adc;
    let keep_bytes = nx * ny * nz * bp * keep;
    bytes.truncate(keep_bytes);
    hdr.dim[4] = keep as i16;
    if keep < 2 {
        hdr.dim[0] = 3;
        hdr.dim[4] = 1;
    }
}

/// Write `.bval` / `.bvec` next to a NIfTI stem when diffusion metadata varies.
pub fn save_dti_sidecars(
    nii_stem: &Path,
    volumes: &[&DicomImage],
    _bids_spacing: bool,
    flip_y: bool,
    verbose: i32,
) -> Result<()> {
    save_dti_sidecars_ex(nii_stem, volumes, _bids_spacing, flip_y, verbose, false)
}

/// Like [`save_dti_sidecars`] with optional b-value sorting (`isSortDTIbyBVal`).
pub fn save_dti_sidecars_ex(
    nii_stem: &Path,
    volumes: &[&DicomImage],
    _bids_spacing: bool,
    flip_y: bool,
    verbose: i32,
    sort_by_bval: bool,
) -> Result<()> {
    if volumes.is_empty() {
        return Ok(());
    }
    let has_spatial = !volumes[0].patient_position[1].is_nan();
    let has_dti_meta = volumes.iter().any(|v| {
        v.b_value >= 0.0
            && (v.b_value > 0.0 || v.diffusion_direction.iter().any(|&x| x.abs() > 1e-12))
    });
    if !has_spatial && !has_dti_meta {
        return Ok(());
    }
    let mut bvals = Vec::with_capacity(volumes.len());
    let mut bvecs = Vec::with_capacity(volumes.len());
    let min_adc = if volumes[0].manufacturer == Manufacturer::Siemens {
        50.0
    } else {
        6.0
    };
    for img in volumes {
        if img.b_value < 0.0 {
            return Ok(());
        }
        let vlen = img
            .diffusion_direction
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        // Philips/Siemens ADC / isotropic derived: non-zero b with zero vector.
        if img.b_value > min_adc && vlen <= f64::EPSILON && !img.is_derived {
            eprintln!(
                "Volume appears to be derived image ADC/Isotropic (non-zero b-value with zero vector length)"
            );
            continue;
        }
        bvals.push(img.b_value);
        bvecs.push(img.diffusion_direction);
    }
    if bvals.is_empty() {
        return Ok(());
    }
    if !should_write_dti(&bvals, &bvecs) {
        return Ok(());
    }
    if sort_by_bval {
        let mut order: Vec<usize> = (0..bvals.len()).collect();
        order.sort_by(|&a, &b| {
            bvals[a]
                .partial_cmp(&bvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let nb: Vec<f64> = order.iter().map(|&i| bvals[i]).collect();
        let nv: Vec<[f64; 3]> = order.iter().map(|&i| bvecs[i]).collect();
        bvals = nb;
        bvecs = nv;
    }
    let d0 = &volumes[0];
    let slice_dir = estimate_slice_dir(&d0.orient);
    ge_correct_bvecs(d0, slice_dir, &mut bvals, &mut bvecs, verbose);
    siemens_philips_correct_bvecs(d0, slice_dir, &mut bvecs, verbose);
    if !flip_y {
        for v in bvecs.iter_mut() {
            if v[1].abs() > f64::EPSILON {
                v[1] = -v[1];
            }
            if v[0].abs() > f64::EPSILON {
                v[0] = -v[0];
            }
        }
    }
    let sep = ' ';
    write_bval(&nii_stem.with_extension("bval"), &bvals, sep)?;
    write_bvec(&nii_stem.with_extension("bvec"), &bvecs, sep)?;
    Ok(())
}

fn should_write_dti(bvals: &[f64], bvecs: &[[f64; 3]]) -> bool {
    if bvals.len() < 2 {
        return false;
    }
    let min_bval_threshold = 6.0f64;
    let mut b_value_varies = bvals.windows(2).any(|w| w[0] != w[1]);
    let min_bval = bvals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_bval > min_bval_threshold {
        b_value_varies = true;
    }
    if bvecs
        .iter()
        .any(|v| v.iter().any(|&x| x.abs() > f64::EPSILON))
    {
        b_value_varies = true;
    }
    b_value_varies
}

fn estimate_slice_dir(orient: &[f64; 7]) -> i32 {
    let n = slice_normal(orient);
    if n[0].abs() >= n[1].abs() && n[0].abs() >= n[2].abs() {
        1
    } else if n[1].abs() >= n[2].abs() {
        2
    } else {
        3
    }
}

fn ge_correct_bvecs(
    d: &DicomImage,
    slice_dir: i32,
    bvals: &mut [f64],
    bvecs: &mut [[f64; 3]],
    verbose: i32,
) {
    if !matches!(d.manufacturer, Manufacturer::Ge | Manufacturer::Canon) {
        return;
    }
    if d.is_bvec_world_coordinates {
        return;
    }
    let col = match d.phase_encoding_rc {
        'C' => true,
        'R' => false,
        _ => {
            eprintln!("Unable to determine DTI gradients, 0018,1312 should be either R or C");
            return;
        }
    };
    let abs_sd = slice_dir.abs();
    let mut flp = if abs_sd == 1 {
        [1, 1, 0]
    } else if abs_sd == 2 {
        [0, 1, 1]
    } else {
        [0, 0, 1]
    };
    if slice_dir < 0 {
        flp[2] = 1 - flp[2];
    }
    if verbose > 0 || !col {
        eprintln!(
            "Saving {} DTI gradients. GE Reorienting {} : please validate. isCol={} sliceDir={} flp={} {} {}",
            bvecs.len(),
            d.protocol_name,
            col as i32,
            slice_dir,
            flp[0],
            flp[1],
            flp[2]
        );
    }
    let mut scaled_warn = false;
    for i in 0..bvecs.len() {
        let vlen = (bvecs[i][0] * bvecs[i][0]
            + bvecs[i][1] * bvecs[i][1]
            + bvecs[i][2] * bvecs[i][2])
            .sqrt();
        if bvals[i] <= f64::EPSILON || vlen <= f64::EPSILON {
            bvecs[i] = [0.0, 0.0, 0.0];
            continue;
        }
        if (0.03..0.97).contains(&vlen) {
            let b_temp = bvals[i] * (vlen * vlen);
            let b_new = if b_temp > 0.0 && b_temp < 5.0 {
                5.0
            } else {
                ((b_temp + 2.5) / 5.0).floor() * 5.0
            };
            let scale = if b_new == 0.0 {
                0.0
            } else {
                (bvals[i] / b_new).sqrt()
            };
            if !scaled_warn {
                eprintln!("GE BVal scaling (e.g. {} -> {} s/mm^2)", bvals[i], b_new);
                scaled_warn = true;
            }
            bvals[i] = b_new;
            bvecs[i][0] *= scale;
            bvecs[i][1] *= scale;
            bvecs[i][2] *= scale;
        }
        for v in 0..3 {
            if flp[v] == 1 {
                bvecs[i][v] = -bvecs[i][v];
            }
        }
        bvecs[i][1] = -bvecs[i][1];
        if !col {
            let swap = bvecs[i][0];
            bvecs[i][0] = bvecs[i][1];
            bvecs[i][1] = swap;
            bvecs[i][0] = -bvecs[i][0];
        }
    }
    for v in bvecs.iter_mut() {
        for c in v.iter_mut() {
            *c = -*c;
            if *c == -0.0 {
                *c = 0.0;
            }
        }
    }
}

fn siemens_philips_correct_bvecs(
    d: &DicomImage,
    slice_dir: i32,
    bvecs: &mut [[f64; 3]],
    verbose: i32,
) {
    // C++: skip unless manufacturer is in the Siemens/Philips family OR vectors are already world-space.
    let in_family = matches!(
        d.manufacturer,
        Manufacturer::Siemens
            | Manufacturer::Philips
            | Manufacturer::Uih
            | Manufacturer::Hitachi
            | Manufacturer::Toshiba
            | Manufacturer::Mediso
            | Manufacturer::Bruker
    );
    if !d.is_bvec_world_coordinates && !in_family {
        return;
    }
    if d.manufacturer == Manufacturer::Uih {
        for v in bvecs.iter_mut() {
            v[1] = -v[1];
            for c in v.iter_mut() {
                if *c == -0.0 {
                    *c = 0.0;
                }
            }
        }
        return;
    }
    let read = normalise([d.orient[1], d.orient[2], d.orient[3]]);
    let phase = normalise([d.orient[4], d.orient[5], d.orient[6]]);
    let slice = normalise(cross(read, phase));
    let min_bval_threshold = if d.manufacturer == Manufacturer::Siemens {
        50.0
    } else {
        6.0
    };
    for (i, v) in bvecs.iter_mut().enumerate() {
        let vlen = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if vlen <= f64::EPSILON {
            continue;
        }
        // ADC / isotropic derived volume: non-zero b with zero vector (handled upstream).
        let _ = (i, min_bval_threshold);
        let old = [v[0], v[1], v[2]];
        let mut neu = [dot(old, read), dot(old, phase), dot(old, slice)];
        neu = normalise(neu);
        v[0] = neu[0];
        v[1] = -neu[1];
        v[2] = neu[2];
        if slice_dir.abs() == 4 {
            v[1] = -v[1];
        }
        for c in v.iter_mut() {
            if *c == -0.0 {
                *c = 0.0;
            }
        }
    }
    if verbose > 0 {
        eprintln!("Saving {} DTI gradients. Validate vectors.", bvecs.len());
    }
}

fn fmt_g(v: f64) -> String {
    format_printf_g_f64(v)
}

fn write_bval(path: &Path, bvals: &[f64], sep: char) -> Result<()> {
    let mut fp = File::create(path).map_err(|e| Error::io(path, e))?;
    for (i, b) in bvals.iter().enumerate() {
        if i > 0 {
            write!(fp, "{sep}").map_err(|e| Error::io(path, e))?;
        }
        write!(fp, "{}", fmt_g(*b)).map_err(|e| Error::io(path, e))?;
    }
    writeln!(fp).map_err(|e| Error::io(path, e))?;
    Ok(())
}

fn write_bvec(path: &Path, bvecs: &[[f64; 3]], sep: char) -> Result<()> {
    let mut fp = File::create(path).map_err(|e| Error::io(path, e))?;
    for axis in 0..3 {
        for (i, v) in bvecs.iter().enumerate() {
            if i > 0 {
                write!(fp, "{sep}").map_err(|e| Error::io(path, e))?;
            }
            write!(fp, "{}", fmt_g(v[axis])).map_err(|e| Error::io(path, e))?;
        }
        writeln!(fp).map_err(|e| Error::io(path, e))?;
    }
    Ok(())
}
