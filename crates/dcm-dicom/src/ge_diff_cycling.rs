//! GE diffusion gradient cycling (issue 635 / #796).
//!
//! Premier / UHP / 7.0T product `epi2` can cycle diffusion encodings across
//! TRs for thermal management. Mode is inferred from model name +
//! `(0019,10B3)` / `(0019,10B6)` (override with `--diffCyclingModeGE`).

/// Unknown / not applicable.
pub const GE_DIFF_CYCLING_UNKNOWN: i32 = -1;
/// Non-cycling systems (MR750, Architect, …).
pub const GE_DIFF_CYCLING_OFF: i32 = 0;
/// Default cycling on Premier/UHP/7T (`UserData12 == 1`).
/// SliceTiming uses the same within-TR epi2 pattern as OFF (see `slice_timing`).
pub const GE_DIFF_CYCLING_ALLTR: i32 = 1;
/// 2-TR cycling block (`UserData12 == 2`).
pub const GE_DIFF_CYCLING_2TR: i32 = 2;
/// 3-TR cycling block (`UserData12 == 3`).
pub const GE_DIFF_CYCLING_3TR: i32 = 3;
/// Special OFF (ABCD / ADNI / HCP / UKB product patches).
pub const GE_DIFF_CYCLING_SPOFF: i32 = 100;

/// Detect cycling mode for GE `epi2` / internal EPI2.
///
/// `user_data_12` = `(0019,10B3)`, `user_data_15` = `(0019,10B6)`.
/// Returns `(diff_cycling_mode, tensor_file_ge)` where tensor file may be
/// inferred as 2/3 when UserData11 was 0 (C++ `tensorFileGE`).
pub fn detect_diff_cycling(
    manufacturers_model_name: &str,
    epi_version_ge: i32,
    internal_epi_version_ge: i32,
    user_data_11: i32,
    user_data_12: i32,
    user_data_15: f64,
) -> (i32, i32) {
    // Only GE diffusion epi2.
    if epi_version_ge != 2 && internal_epi_version_ge != 2 {
        return (GE_DIFF_CYCLING_UNKNOWN, user_data_11);
    }
    let model = manufacturers_model_name.to_ascii_lowercase();
    let cycling_system = model.contains("premier")
        || model.contains("uhp")
        || model.contains("7.0t")
        || model.contains("7t");
    if !cycling_system {
        return (GE_DIFF_CYCLING_OFF, user_data_11);
    }

    // Special OFF: ABCD / multi-site patches (issue 796).
    let premier = model.contains("premier");
    let uhp_or_7t = model.contains("uhp") || model.contains("7.0t") || model.contains("7t");
    if premier && (user_data_15 - 0.72).abs() < 1e-4 {
        return (GE_DIFF_CYCLING_SPOFF, user_data_11);
    }
    if uhp_or_7t && (0.5..=1.0).contains(&user_data_15) {
        return (GE_DIFF_CYCLING_SPOFF, user_data_11);
    }

    let mut tensor = user_data_11;
    if user_data_12 == 2 {
        if user_data_11 == 0 {
            tensor = 2;
        }
        return (GE_DIFF_CYCLING_2TR, tensor);
    }
    if user_data_12 == 3 {
        if user_data_11 == 0 {
            tensor = 3;
        }
        return (GE_DIFF_CYCLING_3TR, tensor);
    }
    // Default on cycling-capable systems.
    (GE_DIFF_CYCLING_ALLTR, tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_premier_is_off() {
        let (m, _) = detect_diff_cycling("DISCOVERY MR750", 2, 2, 0, 2, 0.0);
        assert_eq!(m, GE_DIFF_CYCLING_OFF);
    }

    #[test]
    fn premier_abcd_spoff() {
        let (m, _) = detect_diff_cycling("SIGNA Premier", 2, 2, 0, 1, 0.72);
        assert_eq!(m, GE_DIFF_CYCLING_SPOFF);
    }

    #[test]
    fn premier_2tr() {
        let (m, t) = detect_diff_cycling("SIGNA Premier", 2, 2, 0, 2, 0.0);
        assert_eq!(m, GE_DIFF_CYCLING_2TR);
        assert_eq!(t, 2);
    }

    #[test]
    fn uhp_3tr() {
        let (m, t) = detect_diff_cycling("SIGNA UHP", 2, 2, 0, 3, 0.0);
        assert_eq!(m, GE_DIFF_CYCLING_3TR);
        assert_eq!(t, 3);
    }
}
