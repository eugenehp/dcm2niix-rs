//! Raw explicit-VR scan for UIH nested tags (IPP, slice times) before pixel data.

use dicom_core::Tag;
use dcm_core::{dicom_time_to_sec, snap_f32};

/// Nested `(0020,0032)` / `(0008,0032)` / `(0018,9073)` inside UIH private sequences.
#[derive(Debug, Default, Clone)]
pub struct UihNestedMeta {
    pub ipps: Vec<[f64; 3]>,
    pub acq_times: Vec<f64>,
    pub acquisition_duration: f64,
    pub last_scan_loc: f64,
}

pub fn scan_uih_nested(path: &std::path::Path) -> UihNestedMeta {
    let Ok(data) = std::fs::read(path) else {
        return UihNestedMeta::default();
    };
    let limit = data
        .windows(4)
        .position(|w| w == [0xE0, 0x7F, 0x10, 0x00])
        .unwrap_or(data.len());

    let mut out = UihNestedMeta::default();
    out.last_scan_loc = f64::NAN;

    for hit in scan_raw_hits(&data[..limit], Tag(0x0020, 0x0032), 512) {
        if let Some(v) = read_f64_triple(&data, &hit) {
            out.ipps.push(v);
        }
    }
    for hit in scan_raw_hits(&data[..limit], Tag(0x0008, 0x0032), 512) {
        if let Some(v) = read_tm_atof(&data, &hit) {
            out.acq_times.push(v);
        }
    }
    for hit in scan_raw_hits(&data[..limit], Tag(0x0018, 0x9073), 64) {
        if out.acquisition_duration > 0.0 {
            break;
        }
        if let Some(v) = read_first_f64(&data, &hit) {
            if v > 0.0 {
                out.acquisition_duration = v;
            }
        }
    }
    for hit in scan_raw_hits(&data[..limit], Tag(0x0020, 0x1041), 8) {
        if let Some(v) = read_first_f64(&data, &hit) {
            out.last_scan_loc = v;
        }
    }
    out
}

struct RawHit {
    vr: [u8; 2],
    voff: usize,
    len: usize,
}

fn scan_raw_hits(data: &[u8], tag: Tag, max: usize) -> Vec<RawHit> {
    let pattern = [
        tag.group() as u8,
        (tag.group() >> 8) as u8,
        tag.element() as u8,
        (tag.element() >> 8) as u8,
    ];
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 8 <= data.len() && out.len() < max {
        if data[i..i + 4] != pattern {
            i += 1;
            continue;
        }
        let vr = [data[i + 4], data[i + 5]];
        if !vr[0].is_ascii_alphabetic() || !vr[1].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let (len, voff) = if matches!(&vr, b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN") {
            if i + 12 > data.len() {
                i += 1;
                continue;
            }
            (
                u32::from_le_bytes([data[i + 8], data[i + 9], data[i + 10], data[i + 11]])
                    as usize,
                i + 12,
            )
        } else {
            (
                u16::from_le_bytes([data[i + 6], data[i + 7]]) as usize,
                i + 8,
            )
        };
        if len > 256 || voff + len > data.len() {
            i += 1;
            continue;
        }
        out.push(RawHit { vr, voff, len });
        i += 1;
    }
    out
}

fn read_f64_triple(data: &[u8], hit: &RawHit) -> Option<[f64; 3]> {
    let vals = read_f64_values(data, hit)?;
    if vals.len() >= 3 {
        Some([vals[0], vals[1], vals[2]])
    } else {
        None
    }
}

fn read_first_f64(data: &[u8], hit: &RawHit) -> Option<f64> {
    read_f64_values(data, hit).and_then(|v| v.first().copied())
}

fn read_tm_atof(data: &[u8], hit: &RawHit) -> Option<f64> {
    let s = std::str::from_utf8(&data[hit.voff..hit.voff + hit.len]).ok()?;
    let s = s.trim_matches('\0').trim();
    if s.is_empty() {
        return None;
    }
    s.parse().ok()
}

fn read_f64_values(data: &[u8], hit: &RawHit) -> Option<Vec<f64>> {
    let slice = &data[hit.voff..hit.voff + hit.len];
    if hit.vr == *b"FD" {
        if hit.len % 8 != 0 {
            return None;
        }
        let mut out = Vec::with_capacity(hit.len / 8);
        for chunk in slice.chunks_exact(8) {
            out.push(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                chunk[7],
            ]));
        }
        return Some(out);
    }
    if hit.vr == *b"FL" {
        if hit.len % 4 != 0 {
            return None;
        }
        let mut out = Vec::with_capacity(hit.len / 4);
        for chunk in slice.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
        }
        return Some(out);
    }
    let s = std::str::from_utf8(slice).ok()?;
    let s = s.trim_matches('\0');
    if s.is_empty() {
        return None;
    }
    Some(
        s.split('\\')
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .collect(),
    )
}

/// Port of `checkSliceTiming` HHMMSS path (values → ms, min-subtracted).
pub fn process_uih_slice_timing_ms(raw_hhmmss: &[f64]) -> Vec<f64> {
    if raw_hhmmss.len() < 2 {
        return Vec::new();
    }
    // C++ stores `acquisitionTime` / `sliceTiming` as `float`.
    let raw: Vec<f64> = raw_hhmmss.iter().map(|&t| snap_f32(t)).collect();
    let mut sec: Vec<f64> = raw
        .iter()
        .map(|&t| snap_f32(dicom_time_to_sec(t)))
        .collect();
    let mut min_t = sec[0];
    let mut max_t = min_t;
    for &t in &sec {
        if t < min_t {
            min_t = t;
        }
        if t > max_t {
            max_t = t;
        }
    }
    const NOON: f64 = 43_200.0;
    const MIDNIGHT: f64 = 86_400.0;
    if max_t - min_t > NOON {
        for t in sec.iter_mut() {
            if *t > NOON {
                *t = snap_f32(*t - MIDNIGHT);
            }
        }
        min_t = sec.iter().copied().fold(f64::INFINITY, f64::min);
    }
    for t in sec.iter_mut() {
        *t = snap_f32(*t - min_t);
    }
    // checkSliceTiming applies dicomTimeToSec again before scaling to ms.
    sec.into_iter()
        .map(|t| snap_f32(dicom_time_to_sec(t) * 1000.0))
        .collect()
}
