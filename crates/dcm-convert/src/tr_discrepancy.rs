//! Issue 560: reconcile reported vs estimated inter-volume TR.
//!
//! When AcquisitionTime spacing implies a different TR than `(0018,0080)` /
//! `pixdim[4]`, preserve the reported value as `RepetitionTimeInversion`
//! (IR non-Philips) or `RepetitionTimeExcitation`, then adopt the measured TR.
//!
//! C++ only runs this inside the 2D-slice stacking path (`hdr0.dim[3] < 2`
//! before stacking). Enhanced multi-frame / mosaic volumes (e.g. UIH DTI)
//! skip it — enabling those would rewrite Ref TR via NumberOfAverages.

use dcm_core::dicom_time_to_sec;
use dcm_dicom::{DicomImage, Manufacturer, Modality};
use dcm_nifti::Nifti1Header;

use crate::opts::DcmOpts;

const TOLERANCE_SEC: f64 = 50.0 / 1000.0;

/// Adjust TR when measured volume spacing disagrees with the header TR.
///
/// `stacked_from_2d` is true when inputs were single-slice (non-mosaic) files
/// stacked into a volume — matching C++ `hdr0.dim[3] < 2`.
pub fn apply_issue_560_tr(
    d: &mut DicomImage,
    hdr: &mut Nifti1Header,
    volume_reps: &[&DicomImage],
    opts: &DcmOpts,
    stacked_from_2d: bool,
) {
    if volume_reps.len() < 2 {
        return;
    }
    // C++: PET always eligible inside the 2D stack block; otherwise require
    // `isForceOnsetTimes && manufacturer != GE`. Outside 2D stacking, skip
    // (enhanced MF / mosaic) — except keep legacy PET multi-frame onset path
    // when PET already has volume reps (single-frame-per-file PET dynamics).
    let force = opts.force_onset_times && d.manufacturer != Manufacturer::Ge;
    if d.modality == Modality::Pt {
        // PET: allow both 2D-stack and one-file-per-volume dynamics.
    } else if !stacked_from_2d || !force {
        return;
    }

    // `volume_reps` is already one image per 4D volume (convert pipeline).
    let n_vol = volume_reps.len() as i32;
    let span = acquisition_time_difference(&volume_reps[0], &volume_reps[n_vol as usize - 1]);
    if span <= 0.0 {
        return;
    }
    let mut tr = span / (n_vol as f64 - 1.0);
    let reported = hdr.pixdim[4] as f64;
    if (tr - reported).abs() <= TOLERANCE_SEC {
        return;
    }
    if d.number_of_averages > 1.0 {
        tr /= d.number_of_averages;
    }
    if (tr - reported).abs() <= TOLERANCE_SEC {
        return;
    }
    if reported > 0.0 {
        eprintln!(
            "Discrepancy between reported ({reported}s) and estimated ({tr}s) repetition time (issue 560)."
        );
    }
    if d.is_ir && d.manufacturer != Manufacturer::Philips {
        d.repetition_time_inversion = reported;
    } else if reported > 0.0 {
        d.repetition_time_excitation = reported;
    }
    d.tr = tr * 1000.0;
    hdr.pixdim[4] = tr as f32;
}

fn acquisition_time_difference(a: &DicomImage, b: &DicomImage) -> f64 {
    if !a.acquisition_date.is_empty()
        && !b.acquisition_date.is_empty()
        && a.acquisition_date != b.acquisition_date
    {
        return -1.0;
    }
    let ta = parse_time_sec(&a.acquisition_time);
    let tb = parse_time_sec(&b.acquisition_time);
    if ta < 0.0 || tb < 0.0 {
        return -1.0;
    }
    (tb - ta).abs()
}

fn parse_time_sec(s: &str) -> f64 {
    let digits: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let Ok(t) = digits.parse::<f64>() else {
        return -1.0;
    };
    if t < 0.0 {
        return -1.0;
    }
    dicom_time_to_sec(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_gate() {
        assert!((TOLERANCE_SEC - 0.05).abs() < 1e-12);
    }
}
