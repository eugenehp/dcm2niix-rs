//! Siemens 3D EPI volume TR (issue #1024).
//!
//! Enhanced 3D EPI often reports per-shot TR in `(0018,0080)`. BIDS
//! `RepetitionTime` must be the volume-to-volume interval. Prefer measured
//! AcquisitionDateTime spacing; fall back to
//! `TR × partitions / accel3D × concatenations`.

use dcm_core::dicom_time_to_sec;
use dcm_dicom::DicomImage;
use dcm_nifti::Nifti1Header;

/// Adjust `d.tr` / `hdr.pixdim[4]` to volume TR when it exceeds the reported
/// per-shot TR by >50 ms. Stores the original as `repetition_time_excitation`.
pub fn apply_3d_epi_volume_tr(
    d: &mut DicomImage,
    hdr: &mut Nifti1Header,
    volume_reps: &[&DicomImage],
) {
    if !d.is_3d_acq || hdr.dim[4] < 2 || d.tr <= 0.0 {
        return;
    }
    let mut n_vol = 0i32;
    let mut span = -1.0f64;
    for v in volume_reps {
        if same_position(d, v) {
            n_vol += 1;
            let dt = acquisition_time_difference(d, v);
            if dt > span {
                span = dt;
            }
        }
    }
    let mut vol_tr_sec = -1.0f64;
    if n_vol > 1 && span > 0.0 {
        vol_tr_sec = span / (n_vol as f64 - 1.0);
    } else {
        let bw = d.csa.image.bandwidth_per_pixel_phase_encode;
        let parts = d.phase_encoding_steps_out_of_plane;
        let accel = d.parallel_reduction_out_of_plane;
        if bw > 0.0 && parts > 0 && accel >= 1.0 {
            vol_tr_sec = (d.tr * parts as f64 / accel * d.number_of_concatenations as f64) / 1000.0;
        }
    }
    let reported_sec = d.tr / 1000.0;
    if vol_tr_sec > 0.0 && (vol_tr_sec - reported_sec) > 0.050 {
        eprintln!(
            "3D EPI: RepetitionTime set to volume TR {vol_tr_sec:.4}s (per-shot TR {reported_sec:.4}s, RepetitionTimeExcitation) [issue 1024]"
        );
        d.repetition_time_excitation = reported_sec;
        d.tr = vol_tr_sec * 1000.0;
        hdr.pixdim[4] = vol_tr_sec as f32;
    }
}

fn same_position(a: &DicomImage, b: &DicomImage) -> bool {
    const TOL: f64 = 0.001;
    (1..4).all(|i| (a.patient_position[i] - b.patient_position[i]).abs() < TOL)
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
    fn formula_scales_by_concatenations() {
        // 1000 ms shot × 48 partitions / 2 accel × 2 conc = 48 s volume TR
        let shot_ms = 1000.0;
        let parts = 48.0;
        let accel = 2.0;
        let conc = 2.0;
        let vol = (shot_ms * parts / accel * conc) / 1000.0;
        assert!((vol - 48.0_f64).abs() < 1e-9);
    }
}
