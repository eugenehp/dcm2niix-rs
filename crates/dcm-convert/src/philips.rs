//! Philips intensity scaling (`PhilipsPrecise`).

use dcm_dicom::{DicomImage, Manufacturer};
use dcm_nifti::Nifti1Header;

fn philips_precise_val(pv: f32, rs: f32, ri: f32, ss: f32) -> f32 {
    if (rs * ss) == 0.0 {
        0.0
    } else {
        (pv * rs + ri) / (rs * ss)
    }
}

fn same_float(a: f32, b: f32) -> bool {
    (a - b).abs() <= f32::EPSILON
}

/// Apply Philips RS:RI:SS precise scaling to the NIfTI scl_* fields when `-p y`.
pub fn apply_philips_precise(
    d: &DicomImage,
    philips_precise: bool,
    hdr: &mut Nifti1Header,
    verbose: i32,
) {
    if d.manufacturer != Manufacturer::Philips {
        return;
    }
    if d.is_scale_varies_enh {
        return;
    }
    if d.inten_scale_philips == 0.0 {
        return;
    }
    let l0 = philips_precise_val(0.0, d.inten_scale, d.inten_intercept, d.inten_scale_philips);
    let l1 = philips_precise_val(1.0, d.inten_scale, d.inten_intercept, d.inten_scale_philips);
    let mut inten_scale_p = d.inten_scale;
    let mut inten_intercept_p = d.inten_intercept;
    if l0 != l1 {
        inten_intercept_p = l0;
        inten_scale_p = l1 - l0;
    }
    if same_float(d.inten_intercept, inten_intercept_p) && same_float(d.inten_scale, inten_scale_p)
    {
        return;
    }
    eprintln!(
        "Philips Scaling Values RS:RI:SS = {}:{}:{} (see PMC3998685)",
        d.inten_scale, d.inten_intercept, d.inten_scale_philips
    );
    if verbose > 0 {
        eprintln!(" D scl_slope:scl_inter = {}:{}", d.inten_scale, d.inten_intercept);
        eprintln!(" P scl_slope:scl_inter = {inten_scale_p}:{inten_intercept_p}");
    }
    if philips_precise {
        if verbose > 0 {
            eprintln!(" Using P values ('-p n ' for D values)");
        }
        hdr.scl_slope = inten_scale_p;
        hdr.scl_inter = inten_intercept_p;
    } else if verbose > 0 {
        eprintln!(" Using D values ('-p y ' for P values)");
    }
}
