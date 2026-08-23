//! Port of `checkSliceTiming` / `sliceTimeGE` / `readSoftwareVersionsGE`.

use dcm_dicom::DicomImage;

/// When volume-1 MosaicRefAcqTimes disagree with volume-2 (or exceed TR),
/// substitute volume-2 timings — matching `nii_dicom_batch.cpp` `checkSliceTiming`.
pub fn check_slice_timing(d0: &mut DicomImage, d1: &DicomImage) {
    let t0 = &d0.csa.image.slice_timing_ms;
    let t1 = &d1.csa.image.slice_timing_ms;
    if t0.len() < 2 || t1.len() < 2 || d0.tr <= 0.0 {
        return;
    }
    let n = t0.len().min(t1.len());
    let (min0, max0) = min_max(&t0[..n]);
    let (min1, max1) = min_max(&t1[..n]);
    let tr = d0.tr;
    let issue870 = !same_float_ge(max0 - min0, max1 - min1);

    if (min0 != max0) && (max0 <= tr) && !issue870 {
        return;
    }
    if min1 >= max1 || max1 >= tr {
        if max0 > tr {
            // Leave d0 as-is; BIDS may still emit raw times.
        }
        return;
    }

    let mut mb = 0i32;
    let mut adj = Vec::with_capacity(n);
    let shift = if min1 > 0.0 { min1 } else { 0.0 };
    for &t in t1.iter().take(n) {
        let v = t - shift;
        if same_float_ge(v, 0.0) {
            mb += 1;
        }
        adj.push(v);
    }
    d0.csa.image.slice_timing_ms = adj;
    let mut mb_factor = d1.csa.image.multi_band_factor;
    if mb > 1 && mb > mb_factor {
        mb_factor = mb;
    }
    d0.csa.image.multi_band_factor = mb_factor.max(1);
}

fn min_max(t: &[f64]) -> (f64, f64) {
    let mut mn = t[0];
    let mut mx = t[0];
    for &v in t {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

fn same_float_ge(a: f64, b: f64) -> bool {
    (a - b).abs() <= 0.00001 * a.abs().max(b.abs()).max(1.0)
}

/// Parsed GE `SoftwareVersions` (0018,1020) — C++ `readSoftwareVersionsGE`.
#[derive(Debug, Clone, Copy)]
pub struct GeSoftwareVersion {
    pub major: f64,
    pub is_27r3: bool,
}

pub fn read_software_versions_ge(software_versions: &str) -> GeSoftwareVersion {
    let mut sep = software_versions;
    if let Some(i) = software_versions.find("SIGNA_LX1") {
        sep = &software_versions[i + "SIGNA_LX1".len()..];
        sep = sep.trim_start_matches(|c: char| c == '.' || c == ':' || c.is_whitespace());
    } else if let Some(i) = software_versions.find("MR Software release") {
        sep = &software_versions[i + "MR Software release".len()..];
        sep = sep.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    } else if let Some(i) = software_versions.rfind('\\') {
        sep = &software_versions[i + 1..];
    }
    // RX27.0_R02_… or MR29.1_EA_…
    let bytes: Vec<u8> = sep.bytes().take(11).collect();
    let vs = String::from_utf8_lossy(&bytes);
    let mut major_i = 0i32;
    let mut minor_i = 0i32;
    let mut release_i = 0i32;
    // "%c%c%d.%d_%c%c%d"
    let chars: Vec<char> = vs.chars().collect();
    if chars.len() >= 7 {
        // skip two letter prefix
        let rest: String = chars[2..].iter().collect();
        if let Some((maj, rem)) = rest.split_once('.') {
            major_i = maj.parse().unwrap_or(0);
            let rem = rem.trim_start_matches(|c: char| c.is_ascii_digit());
            // after minor digit(s) comes _R02 or _EA
            let after_minor = if let Some(idx) = rem.find('_') {
                // rem starts with minor digits then _
                let (min_s, after) = rem.split_at(idx);
                // Actually rem is like "0_R02_1831" — first char(s) are minor
                let _ = min_s;
                after
            } else {
                ""
            };
            // Re-parse properly: "27.0_R02" style from after prefix
            let body: String = chars[2..].iter().collect();
            if let Some((left, right)) = body.split_once('_') {
                if let Some((maj, min)) = left.split_once('.') {
                    major_i = maj.parse().unwrap_or(0);
                    minor_i = min.parse().unwrap_or(0);
                }
                let right = right.trim_start_matches(|c: char| !c.is_ascii_alphanumeric());
                if right.starts_with("EA") {
                    release_i = 0;
                } else if let Some(rest) = right.strip_prefix('R') {
                    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    release_i = num.parse().unwrap_or(0);
                }
                let _ = after_minor;
            }
        }
    }
    let major = major_i as f64 + 0.1 * minor_i as f64;
    let is_27r3 = major >= 27.1 || (major_i == 27 && release_i >= 3);
    GeSoftwareVersion { major, is_27r3 }
}

/// C++ `sliceTimeGE` — times in **milliseconds** (TR and group_delay are ms).
pub fn slice_time_ge(
    nz: usize,
    tr_ms: f64,
    mb: i32,
    interleaved: bool,
    ge_major: f64,
    is_27r3: bool,
    group_delay_ms: f64,
    flip_z: bool,
) -> Vec<f64> {
    if nz < 2 || tr_ms <= 0.0 {
        return Vec::new();
    }
    let mb = mb.max(1);
    if mb > 1 && ge_major < 26.0 {
        eprintln!("Unable to determine slice times for early GE HyperBand.");
        return vec![-1.0];
    }
    let mut n_excitations = ((nz as f64) / (mb as f64)).ceil() as i32;
    if mb > 1 && !is_27r3 && (n_excitations % 2) == 0 {
        n_excitations += 1;
    }
    let gd = if group_delay_ms > 0.0 {
        group_delay_ms
    } else {
        0.0
    };
    let sec_per_slice = (tr_ms - gd) / (n_excitations as f64);
    let mut slice_timing = vec![0.0f64; n_excitations.max(1) as usize];
    if !interleaved {
        for i in 0..n_excitations as usize {
            slice_timing[i] = i as f64 * sec_per_slice;
        }
    } else {
        let n_odd = (n_excitations - 1) / 2;
        for i in 0..n_excitations as usize {
            if i % 2 == 0 {
                slice_timing[i] = (i / 2) as f64 * sec_per_slice;
            } else {
                slice_timing[i] = (n_odd as usize + ((i + 1) / 2)) as f64 * sec_per_slice;
            }
        }
        if mb > 1
            && is_27r3
            && interleaved
            && n_excitations > 2
            && (n_excitations % 2) == 0
        {
            let a = (n_excitations - 1) as usize;
            let b = (n_excitations - 3) as usize;
            slice_timing.swap(a, b);
        }
    }
    let mut t = vec![0.0; nz];
    for i in 0..nz {
        t[i] = slice_timing[i % n_excitations as usize];
    }
    if flip_z {
        t.reverse();
    }
    t
}

/// GE slice-timing rescue for 4D EPI — C++ `sliceTimingGE` + `sliceTimeGE`.
///
/// Returns empty when timing cannot be estimated; a single `-1` marks unsupported.
pub fn ge_rescue_slice_timing_ms(
    series_description: &str,
    nz: usize,
    tr_ms: f64,
    flip_z: bool,
    ge_slice_order: i32,
    mb: i32,
    group_delay_ms: f64,
    software_versions: &str,
    epi_version_ge: i32,
    internal_epi_version_ge: i32,
    ge_iopt: &str,
    diff_cycling_mode_ge: i32,
) -> Vec<f64> {
    if nz < 2 || tr_ms <= 0.0 {
        return Vec::new();
    }
    if software_versions.len() < 10 {
        eprintln!("Unable to determine GE Slice timing, invalid SoftwareVersions (0018,1020)");
        return vec![-1.0];
    }
    let mut ver = read_software_versions_ge(software_versions);
    let mut interleaved = if ge_slice_order >= 0 {
        ge_slice_order != 0
    } else {
        series_description
            .to_ascii_lowercase()
            .contains("interleaved")
    };

    // Gate by EPI class (BrainWave / Multi-Phase / Diffusion) — C++ `sliceTimingGE`.
    let iopt_u = ge_iopt.to_ascii_uppercase();
    let is_epi_rt = epi_version_ge == 1 || iopt_u.contains("FMRI");
    let is_epi_mph = epi_version_ge == 0 || iopt_u.contains("MPH");
    let is_diff =
        epi_version_ge == 2 || internal_epi_version_ge == 2 || iopt_u.contains("DIFF");
    // Pepolar (`>= 3`): warn then still estimate times with default interleaving
    // (C++ falls through after the warning; it does not `return`).
    if epi_version_ge >= 3 {
        eprintln!("GE ABCD pepolar research sequence handling is experimental");
    } else if is_epi_rt {
        interleaved = ge_slice_order != 0;
    } else if is_epi_mph {
        if group_delay_ms < -0.5 {
            eprintln!("SliceTiming Unsupported: GE Multi-Phase EPI with Variable Delays");
            return vec![-1.0];
        }
    } else if is_diff {
        // Diffusion epi2 (issue 635): interleaved within-TR pattern with is27r3=false.
        // OFF / SPOFF / 2TR / 3TR / ALLTR share that EPI timing (C++ still refuses
        // ALLTR and 2TR/3TR; we estimate within-TR times for all of them).
        // Gradient cycling still varies shot/encoding order across the cycling
        // block (eddy per-volume slspec) — BIDS SliceTiming is series-level, so
        // we emit the canonical within-TR times.
        match diff_cycling_mode_ge {
            0 | 100 => {
                ver.is_27r3 = false;
            }
            1 => {
                eprintln!(
                    "GE Diffusion:ALLTR-Cycling: estimating SliceTiming from within-TR EPI pattern (is27r3=false); diffusion encodings cycle across every TR"
                );
                ver.is_27r3 = false;
                interleaved = true;
            }
            2 | 3 => {
                let n = diff_cycling_mode_ge;
                eprintln!(
                    "GE Diffusion:{n}TR-Cycling: estimating SliceTiming from within-TR EPI pattern (is27r3=false); shot order varies across the {n}-volume cycling block"
                );
                ver.is_27r3 = false;
                interleaved = true;
            }
            _ => {
                eprintln!("Unable to compute slice times for GE Diffusion");
                return vec![-1.0];
            }
        }
    } else if ge_slice_order < 0 {
        eprintln!("Unable to compute slice times for this GE dataset");
        return vec![-1.0];
    }

    let gd = if group_delay_ms < -0.5 {
        0.0
    } else {
        group_delay_ms
    };
    // Diffusion epi2: issue 635 uses groupDelay=0 with the OFF/2TR/3TR/ALLTR pattern.
    let gd = if is_diff && matches!(diff_cycling_mode_ge, 0 | 100 | 1 | 2 | 3) {
        0.0
    } else {
        gd
    };
    let interleaved = if is_diff && matches!(diff_cycling_mode_ge, 0 | 100 | 1 | 2 | 3) {
        true
    } else {
        interleaved
    };
    slice_time_ge(
        nz,
        tr_ms,
        mb.max(1),
        interleaved,
        ver.major,
        ver.is_27r3,
        gd,
        flip_z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rx27_r02() {
        let v = read_software_versions_ge("27\\LX\\MR Software release:RX27.0_R02_1831.a");
        assert!((v.major - 27.0).abs() < 1e-6);
        assert!(!v.is_27r3);
    }

    #[test]
    fn parse_mr29_ea() {
        let v = read_software_versions_ge("28\\LX\\MR29.1_EA_2039.g");
        assert!((v.major - 29.1).abs() < 1e-6);
        assert!(v.is_27r3);
    }

    #[test]
    fn ge_2tr_matches_off_pattern() {
        let sw = "27\\LX\\MR Software release:RX27.0_R02_1831.a";
        let off = ge_rescue_slice_timing_ms(
            "epi2", 60, 2000.0, false, 1, 3, 0.0, sw, 2, 2, "DIFF", 0,
        );
        let t2 = ge_rescue_slice_timing_ms(
            "epi2", 60, 2000.0, false, 1, 3, 0.0, sw, 2, 2, "DIFF", 2,
        );
        assert_eq!(off.len(), 60);
        assert_eq!(t2.len(), 60);
        for (a, b) in off.iter().zip(t2.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        assert!(off.iter().any(|&t| t > 0.0));
    }

    #[test]
    fn ge_alltr_matches_off_pattern() {
        let sw = "27\\LX\\MR Software release:RX27.0_R02_1831.a";
        let off = ge_rescue_slice_timing_ms(
            "epi2", 60, 2000.0, false, 1, 3, 0.0, sw, 2, 2, "DIFF", 0,
        );
        let alltr = ge_rescue_slice_timing_ms(
            "epi2", 60, 2000.0, false, 1, 3, 0.0, sw, 2, 2, "DIFF", 1,
        );
        assert_eq!(off.len(), 60);
        assert_eq!(alltr.len(), 60);
        for (a, b) in off.iter().zip(alltr.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn slice_time_sequential_no_mb() {
        let t = slice_time_ge(4, 2000.0, 1, false, 28.0, true, 0.0, false);
        assert_eq!(t.len(), 4);
        assert!((t[0] - 0.0).abs() < 1e-6);
        assert!((t[1] - 500.0).abs() < 1e-6);
        assert!((t[3] - 1500.0).abs() < 1e-6);
    }
}
