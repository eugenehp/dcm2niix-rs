//! DICOM time helpers matching C++ `dicomTimeToSec` / `printf("%g")`.

/// Round-trip through `float` like C++ `mat44` / CSA storage.
#[inline]
pub fn snap_f32(v: f64) -> f64 {
    (v as f32) as f64
}

/// Port of `dicomTimeToSec` (`snprintf("%+013.5f")` + `sscanf("%3d%2d%lf")`).
pub fn dicom_time_to_sec(dicom_time: f64) -> f64 {
    let s = format!("{:+013.5}", dicom_time);
    parse_hhmmss_sscanf(&s).unwrap_or(-1.0)
}

/// `%3d%2d%lf` on the `%+013.5f` buffer (field widths include sign for `%3d`).
fn parse_hhmmss_sscanf(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut i = 0usize;
    let start_d = i;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    while i < bytes.len() && i - start_d < 3 && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start_d {
        return None;
    }
    let ahour: i32 = s[start_d..i].parse().ok()?;
    let start_m = i;
    while i < bytes.len() && i - start_m < 2 && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start_m {
        return None;
    }
    let amin: i32 = s[start_m..i].parse().ok()?;
    let asec: f64 = s[i..].parse().ok()?;
    Some((ahour * 3600) as f64 + (amin * 60) as f64 + asec)
}

/// Approximate C `printf("%g")` on a float value (6 significant digits).
pub fn format_printf_g(v: f32) -> String {
    format_printf_g_f64(v as f64)
}

pub fn format_printf_g_f64(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let v = snap_f32(v);
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let prec = 6i32;
    if exp < -4 || exp >= prec {
        let mant = v / 10f64.powi(exp);
        let digits = (prec - 1).max(0) as usize;
        let mut m = format!("{:.digits$}", mant.abs());
        m = m.trim_end_matches('0').trim_end_matches('.').to_string();
        let sign = if v < 0.0 { "-" } else { "" };
        format!("{sign}{m}e{exp:+03}")
    } else {
        let dec = (prec - exp - 1).max(0) as usize;
        let mut s = format!("{v:.dec$}");
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dicom_time_matches_cpp_hhmmss() {
        assert!((dicom_time_to_sec(134431.5625) - 49471.5625).abs() < 1e-6);
        assert!((dicom_time_to_sec(134428.640625) - 49468.640625).abs() < 0.001);
        // `%+013.5f` keeps five fractional digits (2.640625 → 2.64062).
        assert!((snap_f32(dicom_time_to_sec(2.640625)) - snap_f32(2.640625)).abs() < 1e-5);
    }

    #[test]
    fn slice_timing_ms_pipeline() {
        let raw = [134431.5625f64, 134428.640625, 134431.265625];
        let raw: Vec<f64> = raw.iter().map(|&t| snap_f32(t)).collect();
        let sec: Vec<f64> = raw.iter().map(|&t| snap_f32(dicom_time_to_sec(t))).collect();
        let min_t = sec.iter().copied().fold(f64::INFINITY, f64::min);
        let rel: Vec<f64> = sec.iter().map(|&t| snap_f32(t - min_t)).collect();
        let ms: Vec<f64> = rel
            .iter()
            .map(|&t| snap_f32(dicom_time_to_sec(t) * 1000.0))
            .collect();
        assert!((snap_f32(ms[1]) - 0.0).abs() < 0.02);
        assert!((snap_f32(ms[0]) - 2921.88).abs() < 0.02);
        assert_eq!(format_printf_g((ms[0] / 1000.0) as f32), "2.92188");
    }
}
