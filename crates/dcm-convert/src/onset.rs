//! PET / multiphase volume onset helpers (`FrameTimesStart`).
//!
//! Matches C++ around `isForceOnsetTimes` in `nii_dicom_batch.cpp`:
//! - PET: onset = AcquisitionTime − SeriesTime (when SeriesTime known).
//! - Non-PET (force on, non-GE): only fill `volumeOnsetTime` when inter-volume
//!   acquisition spacing **varies**; then use deltas vs first volume.

use dcm_core::dicom_time_to_sec;
use dcm_dicom::{DicomImage, Modality};

use crate::opts::DcmOpts;

/// Fill `volume_onset_times` / per-volume frame arrays on `d0` from the series.
pub fn fill_volume_onset_times(
    d0: &mut DicomImage,
    volume_reps: &[&DicomImage],
    opts: &DcmOpts,
) {
    if volume_reps.len() < 2 {
        // Single volume: FrameTimesStart only when decayFactor is set (PET).
        if d0.modality == Modality::Pt && d0.decay_factor > 0.0 {
            let series_sec = parse_time_sec(&d0.series_time);
            let acq_sec = parse_time_sec(&d0.acquisition_time);
            let mut t_start = 0.0;
            if series_sec >= 0.0 && acq_sec >= 0.0 {
                t_start = (acq_sec - series_sec).max(0.0);
            }
            d0.volume_onset_times = vec![t_start];
        }
        return;
    }
    let want = d0.modality == Modality::Pt
        || (opts.force_onset_times && d0.manufacturer != dcm_dicom::Manufacturer::Ge);
    if !want {
        return;
    }

    let series_sec = if d0.modality == Modality::Pt {
        parse_time_sec(&d0.series_time)
    } else {
        -1.0
    };

    let mut durations = Vec::with_capacity(volume_reps.len());
    let mut refs = Vec::with_capacity(volume_reps.len());
    let mut decays = Vec::with_capacity(volume_reps.len());

    for v in volume_reps {
        durations.push(v.frame_duration);
        refs.push(v.frame_reference_time);
        decays.push(v.decay_factor);
    }

    // PET with known SeriesTime: always populate FrameTimesStart.
    if series_sec >= 0.0 {
        let mut onsets = Vec::with_capacity(volume_reps.len());
        for v in volume_reps {
            let acq = parse_time_sec(&v.acquisition_time);
            let mut t = if acq >= 0.0 {
                acq - series_sec
            } else {
                -1.0
            };
            if t < 0.0 {
                t = 0.0;
            }
            onsets.push(t);
        }
        if onsets.first().copied().unwrap_or(-1.0) >= 0.0 {
            d0.volume_onset_times = onsets;
        }
    } else {
        // Non-PET (or PET without SeriesTime): only when TR between volumes varies.
        const TOLERANCE_SEC: f64 = 50.0 / 1000.0;
        let mut min_tr = f64::MAX;
        let mut max_tr = -1.0f64;
        let mut prev = &volume_reps[0];
        for v in volume_reps.iter().skip(1) {
            let tr_diff = acquisition_time_difference(prev, v);
            prev = v;
            if tr_diff <= 0.0 {
                continue;
            }
            min_tr = min_tr.min(tr_diff);
            max_tr = max_tr.max(tr_diff);
        }
        let tr_varies = max_tr > 0.0 && (max_tr - min_tr) > TOLERANCE_SEC;
        if tr_varies {
            eprintln!("Warning: Seconds between volumes varies");
            let mut onsets = Vec::with_capacity(volume_reps.len());
            let first = &volume_reps[0];
            for v in volume_reps {
                onsets.push(acquisition_time_difference(first, v).max(0.0));
            }
            d0.volume_onset_times = onsets;
            // Decay factors only for PET.
            if d0.modality != Modality::Pt {
                for d in &mut decays {
                    *d = -1.0;
                }
            }
        }
    }

    if durations.iter().any(|&d| d >= 0.0) {
        d0.frame_durations = durations;
    }
    if refs.iter().any(|&d| d >= 0.0) {
        d0.frame_reference_times = refs;
    }
    if decays.iter().any(|&d| d >= 0.0) {
        d0.decay_factors = decays;
    }
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

/// Seconds between acquisition times (C++ `acquisitionTimeDifference`).
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
    tb - ta
}
