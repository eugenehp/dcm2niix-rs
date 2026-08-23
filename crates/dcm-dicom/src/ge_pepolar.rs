//! GE ABCD / research pepolar EPI variants (`kGE_EPI_PEPOLAR_*`).
//!
//! Port of the pepolar detection and polarity flip in `nii_dicom.cpp`
//! (pulse sequence name `epi_pepolar` + `(0019,10B3)` userData12 + volume
//! index) and the extra Y-flip at save time in `nii_dicom_batch.cpp`.

/// `-1` = not EPI / unknown.
pub const GE_EPI_UNKNOWN: i32 = -1;
/// Multi-phase EPI.
pub const GE_EPI_EPI: i32 = 0;
/// BrainWave / epiRT.
pub const GE_EPI_EPIRT: i32 = 1;
/// Spin-echo EPI2 / diffusion.
pub const GE_EPI_EPI2: i32 = 2;
/// Pepolar forward (base after `epi_pepolar` name).
pub const GE_EPI_PEPOLAR_FWD: i32 = 3;
pub const GE_EPI_PEPOLAR_REV: i32 = 4;
pub const GE_EPI_PEPOLAR_REV_FWD: i32 = 5;
pub const GE_EPI_PEPOLAR_FWD_REV: i32 = 6;
pub const GE_EPI_PEPOLAR_REV_FWD_FLIP: i32 = 7;
pub const GE_EPI_PEPOLAR_FWD_REV_FLIP: i32 = 8;

/// GE phase polarity: unflipped.
pub const GE_PE_UNFLIPPED: i32 = 0;
/// GE phase polarity: flipped (C++ `kGE_PHASE_ENCODING_POLARITY_FLIPPED` = 4).
pub const GE_PE_FLIPPED: i32 = 4;

/// True when `epi_version_ge` is any pepolar research class.
pub fn is_pepolar(epi_version_ge: i32) -> bool {
    epi_version_ge >= GE_EPI_PEPOLAR_FWD
}

/// Volumes that need an extra image-space Y flip after the normal `-y` flip.
pub fn needs_extra_y_flip(epi_version_ge: i32) -> bool {
    matches!(
        epi_version_ge,
        GE_EPI_PEPOLAR_REV | GE_EPI_PEPOLAR_FWD_REV_FLIP | GE_EPI_PEPOLAR_REV_FWD_FLIP
    )
}

/// Refine pepolar class from `(0019,10B3)` and temporal volume index, then
/// flip PE polarity / bump series number for reverse volumes (issue 532).
pub fn finalize_pepolar(
    epi_version_ge: &mut i32,
    phase_encoding_ge: &mut i32,
    series_number: &mut i64,
    user_data_12: i32,
    volume_number: i32,
) {
    if *epi_version_ge == GE_EPI_PEPOLAR_FWD {
        if user_data_12 == 1 {
            *epi_version_ge = GE_EPI_PEPOLAR_REV;
        } else if user_data_12 == 2 {
            *epi_version_ge = GE_EPI_PEPOLAR_REV_FWD;
        } else if user_data_12 == 3 {
            *epi_version_ge = GE_EPI_PEPOLAR_FWD_REV;
        }
    }
    if *epi_version_ge == GE_EPI_PEPOLAR_REV_FWD && volume_number > 0 && volume_number % 2 == 1 {
        *epi_version_ge = GE_EPI_PEPOLAR_REV_FWD_FLIP;
    }
    if *epi_version_ge == GE_EPI_PEPOLAR_FWD_REV && volume_number > 0 && volume_number % 2 == 0 {
        *epi_version_ge = GE_EPI_PEPOLAR_FWD_REV_FLIP;
    }
    if matches!(
        *epi_version_ge,
        GE_EPI_PEPOLAR_REV | GE_EPI_PEPOLAR_FWD_REV_FLIP | GE_EPI_PEPOLAR_REV_FWD_FLIP
    ) {
        if *epi_version_ge != GE_EPI_PEPOLAR_REV {
            *series_number += 1000;
        }
        if *phase_encoding_ge == GE_PE_UNFLIPPED {
            *phase_encoding_ge = GE_PE_FLIPPED;
        } else if *phase_encoding_ge == GE_PE_FLIPPED {
            *phase_encoding_ge = GE_PE_UNFLIPPED;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_data_selects_rev() {
        let mut epi = GE_EPI_PEPOLAR_FWD;
        let mut pe = GE_PE_UNFLIPPED;
        let mut series = 7i64;
        finalize_pepolar(&mut epi, &mut pe, &mut series, 1, -1);
        assert_eq!(epi, GE_EPI_PEPOLAR_REV);
        assert_eq!(pe, GE_PE_FLIPPED);
        assert_eq!(series, 7);
    }

    #[test]
    fn fwd_rev_even_volume_flips_and_bumps_series() {
        let mut epi = GE_EPI_PEPOLAR_FWD;
        let mut pe = GE_PE_FLIPPED;
        let mut series = 5i64;
        finalize_pepolar(&mut epi, &mut pe, &mut series, 3, 2);
        assert_eq!(epi, GE_EPI_PEPOLAR_FWD_REV_FLIP);
        assert_eq!(pe, GE_PE_UNFLIPPED);
        assert_eq!(series, 1005);
    }
}
