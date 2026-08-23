//! Siemens XA / CMRR physio extraction (`xaPhysioConvert` / `cmrrPhysioConvert`).
//!
//! Output: `<stem>_recording-<label>_physio.tsv.gz` + `.json` matching C++ /
//! BIDS physio (SamplingFrequency timeline; no TSV header row).

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use dcm_core::error::{Error, Result};
use dcm_dicom::{physio_payload, DicomImage};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

const MDH_TIC_MS: f64 = 2.5;

/// Detect physio payload in a DICOM private `(7FE1,1010)` blob.
pub fn convert_physio(d: &DicomImage, out_stem: &Path) -> Result<Vec<PathBuf>> {
    if !d.is_xa_physio && !d.is_cmrr_physio {
        return Ok(vec![]);
    }
    let Some(raw) = physio_payload(&d.path)? else {
        return Ok(vec![]);
    };
    if d.is_xa_physio {
        convert_xa_physio(&raw, out_stem)
    } else {
        convert_cmrr_physio(&raw, d.acquisition_number.max(1), out_stem)
    }
}

fn bids_label(chan: &str) -> Option<&'static str> {
    match chan {
        "PULS" => Some("cardiac"),
        "RESP" => Some("respiratory"),
        "ECG" => Some("ecg"),
        "EXT" => Some("external_trigger"),
        _ => None,
    }
}

fn convert_xa_physio(raw: &[u8], out_stem: &Path) -> Result<Vec<PathBuf>> {
    let xml = if raw.len() >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
        let mut dec = GzDecoder::new(raw);
        let mut s = String::new();
        dec.read_to_string(&mut s)
            .map_err(|e| Error::convert(format!("physio gunzip: {e}")))?;
        s
    } else {
        String::from_utf8_lossy(raw).into_owned()
    };
    let xml = if let Some(i) = xml.find("<Physio") {
        xml[i..].to_string()
    } else {
        xml
    };

    let vol_tics = parse_volume_tics(&xml);
    let mut written = Vec::new();
    for (stype, label) in [
        ("PULS", "cardiac"),
        ("RESP", "respiratory"),
        ("ECG", "ecg"),
        ("EXT", "external_trigger"),
    ] {
        let samples = extract_xa_pmu(&xml, stype);
        if samples.len() < 2 {
            continue;
        }
        let (hz, start_sec, values, triggers) = rasterize_physio(&samples, &vol_tics);
        if values.is_empty() {
            continue;
        }
        let paths = write_physio_stream(
            out_stem,
            label,
            &values,
            if vol_tics.is_empty() {
                None
            } else {
                Some(&triggers)
            },
            None,
            None,
            hz,
            start_sec,
        )?;
        written.extend(paths);
    }
    Ok(written)
}

fn parse_volume_tics(xml: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Volume ") {
        let tag = &rest[start..];
        let end = tag.find('>').unwrap_or(tag.len());
        let head = &tag[..end];
        if let Some(v) = attr_i64(head, "ACQUISITION_TIME_TICS") {
            out.push(v);
        }
        rest = &rest[start + 1..];
    }
    out
}

fn extract_xa_pmu(xml: &str, stype: &str) -> Vec<(i64, f64)> {
    let mut out = Vec::new();
    let needle = format!("TYPE=\"{stype}\"");
    let mut rest = xml;
    while let Some(i) = rest.find(&needle) {
        let after = &rest[i..];
        // Prefer TIME_TICS + DATA pairs inside this stream block.
        let block_end = after.find("</PhysioStream>").unwrap_or(after.len().min(200_000));
        let block = &after[..block_end];
        let mut tics = Vec::new();
        let mut data = Vec::new();
        for part in block.split('<') {
            if let Some(rest) = part.strip_prefix("TIME_TICS>") {
                let v = rest
                    .split('<')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .parse::<i64>()
                    .unwrap_or(-1);
                if v >= 0 {
                    tics.push(v);
                }
            } else if let Some(rest) = part.strip_prefix("DATA>") {
                let v = rest
                    .split('<')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .parse::<f64>()
                    .unwrap_or(f64::NAN);
                if v.is_finite() {
                    data.push(v);
                }
            }
        }
        let n = tics.len().min(data.len());
        for i in 0..n {
            out.push((tics[i], data[i]));
        }
        rest = &rest[i + needle.len()..];
    }
    out
}

fn attr_i64(tag: &str, name: &str) -> Option<i64> {
    let key = format!("{name}=\"");
    let i = tag.find(&key)?;
    let rest = &tag[i + key.len()..];
    let end = rest.find('"')?;
    rest[..end].parse().ok()
}

/// Uniform-rate grid from tic samples (C++ `physioBidsFillUniform` simplified).
fn rasterize_physio(
    samples: &[(i64, f64)],
    vol_tics: &[i64],
) -> (f64, f64, Vec<f64>, Vec<u8>) {
    if samples.len() < 2 {
        return (0.0, 0.0, Vec::new(), Vec::new());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by_key(|(t, _)| *t);
    let dts: Vec<i64> = sorted.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let mut dts_pos: Vec<i64> = dts.into_iter().filter(|&d| d > 0).collect();
    dts_pos.sort_unstable();
    let med = if dts_pos.is_empty() {
        1i64
    } else {
        dts_pos[dts_pos.len() / 2].max(1)
    };
    let dt_ms = med as f64 * MDH_TIC_MS;
    let hz = 1000.0 / dt_ms;
    let first = sorted[0].0;
    let last = sorted[sorted.len() - 1].0;
    let span = (last - first) as f64;
    let mut exp_n = ((span / med as f64).round() as usize) + 1;
    if exp_n < sorted.len() {
        exp_n = sorted.len();
    }
    let mut values = vec![f64::NAN; exp_n];
    for &(tic, v) in &sorted {
        let idx = (((tic - first) as f64 / med as f64).round() as isize).clamp(0, (exp_n - 1) as isize)
            as usize;
        values[idx] = v;
    }
    let mut triggers = vec![0u8; exp_n];
    for &vt in vol_tics {
        if vt < first || vt > last {
            continue;
        }
        let idx = (((vt - first) as f64 / med as f64).ceil() as isize)
            .clamp(0, (exp_n - 1) as isize) as usize;
        triggers[idx] = 1;
    }
    let start_sec = if let Some(&v0) = vol_tics.first() {
        let start_ms = (first - v0) as f64 * MDH_TIC_MS;
        (start_ms as i64) as f64 / 1000.0
    } else {
        0.0
    };
    (hz, start_sec, values, triggers)
}

/// CMRR VE11C multi-waveform blob: `acqu_num * 1024` bytes per slot.
fn convert_cmrr_physio(raw: &[u8], acqu_num: i32, out_stem: &Path) -> Result<Vec<PathBuf>> {
    let acqu = acqu_num.max(1) as usize;
    let wave_len = acqu.saturating_mul(1024);
    if wave_len < 1024 || raw.len() < wave_len || raw.len() % wave_len != 0 {
        eprintln!(
            "Warning: CMRR PMU payload size {} is not a multiple of (AcquisitionNumber={})*1024.",
            raw.len(),
            acqu
        );
        return Ok(vec![]);
    }
    let n_waves = raw.len() / wave_len;
    if !(1..=12).contains(&n_waves) {
        eprintln!("Warning: CMRR PMU payload reports {n_waves} waveforms (suspicious).");
        return Ok(vec![]);
    }

    let mut vol_tics: Vec<i64> = Vec::new();
    let mut streams: Vec<CmrrStream> = Vec::new();

    for w in 0..n_waves {
        let wave = &raw[w * wave_len..(w + 1) * wave_len];
        let data_len = u32::from_le_bytes([wave[0], wave[1], wave[2], wave[3]]) as usize;
        if data_len == 0 || data_len > wave_len.saturating_sub(1024) {
            eprintln!("Warning: CMRR waveform {w} header reports data_len={data_len} exceeding slot; skipping.");
            continue;
        }
        let body = String::from_utf8_lossy(&wave[1024..1024 + data_len]);
        let mut st = CmrrStream::default();
        st.dt_ms = MDH_TIC_MS;
        let mut stream_header_read = false;
        let mut prev_vol = String::new();
        for line in body.lines() {
            parse_cmrr_line(
                line.trim_end_matches(['\r', ' ', '\t']),
                &mut st,
                &mut vol_tics,
                &mut prev_vol,
                &mut stream_header_read,
            );
        }
        if st.chan == "ACQUISITION_INFO" {
            continue;
        }
        if st.label.is_none() || st.tics.len() < 2 {
            continue;
        }
        streams.push(st);
    }

    let mut written = Vec::new();
    for st in &streams {
        let Some(label) = st.label else {
            continue;
        };
        let n = st.tics.len();
        let span_ms = (st.tics[n - 1] - st.tics[0]) as f64 * MDH_TIC_MS;
        let dt_ms = if st.dt_ms > 0.0 && st.dt_ms.is_finite() {
            st.dt_ms
        } else {
            span_ms / (n - 1) as f64
        };
        if dt_ms <= 0.0 || !dt_ms.is_finite() {
            eprintln!("Warning: CMRR stream {} has non-positive sample interval; skipping.", st.chan);
            continue;
        }
        let samp_freq = 1000.0 / dt_ms;
        let pairs: Vec<(i64, f64)> = st
            .tics
            .iter()
            .zip(st.signal.iter())
            .map(|(&t, &v)| (t, v))
            .collect();
        let (hz, start_sec, values, triggers) = rasterize_with_dt(&pairs, dt_ms, &vol_tics);
        let hz = if hz > 0.0 { hz } else { samp_freq };

        let peak_label = if st.trigger_tics.is_empty() {
            None
        } else if st.chan == "EXT" {
            Some(format!("{label}_peak"))
        } else {
            Some(format!("{label}_trigger"))
        };
        let peaks = if let Some(ref _pl) = peak_label {
            Some(raster_triggers(
                pairs[0].0,
                pairs[pairs.len() - 1].0,
                dt_ms / MDH_TIC_MS,
                values.len(),
                &st.trigger_tics,
            ))
        } else {
            None
        };

        let paths = write_physio_stream(
            out_stem,
            label,
            &values,
            if vol_tics.is_empty() {
                None
            } else {
                Some(&triggers)
            },
            peak_label.as_deref(),
            peaks.as_deref(),
            hz,
            start_sec,
        )?;
        written.extend(paths);
    }
    if written.is_empty() {
        eprintln!(
            "Warning: CMRR PMU: no physio streams written — were sensors connected, or see any per-stream warnings above"
        );
    }
    Ok(written)
}

#[derive(Default)]
struct CmrrStream {
    chan: String,
    label: Option<&'static str>,
    tics: Vec<i64>,
    signal: Vec<f64>,
    trigger_tics: Vec<i64>,
    dt_ms: f64,
}

fn parse_cmrr_line(
    line: &str,
    st: &mut CmrrStream,
    vol_tics: &mut Vec<i64>,
    prev_vol: &mut String,
    stream_header_read: &mut bool,
) {
    if line.is_empty() {
        return;
    }
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 3 {
        return;
    }
    if toks[1] == "=" {
        if toks[0] == "LogDataType" {
            st.chan = toks[2].to_string();
            st.label = bids_label(toks[2]);
        } else if toks[0] == "SampleTime" {
            if let Ok(v) = toks[2].parse::<f64>() {
                st.dt_ms = MDH_TIC_MS * v;
            }
        }
        return;
    }
    if st.chan == "ACQUISITION_INFO" && toks.len() == 5 {
        if !*stream_header_read {
            *stream_header_read = true;
            return;
        }
        if toks[4] != "0" {
            return;
        }
        if toks[0] == prev_vol.as_str() {
            return;
        }
        *prev_vol = toks[0].to_string();
        if let Ok(tic) = toks[2].parse::<i64>() {
            if tic >= 0 {
                vol_tics.push(tic);
            }
        }
        return;
    }
    if let Some(label) = st.label {
        if toks[1] != st.chan {
            let _ = label;
            return;
        }
        let Ok(tic) = toks[0].parse::<i64>() else {
            return;
        };
        if tic < 0 {
            return;
        }
        if toks.len() >= 4 && toks[3].contains("_TRIGGER") {
            st.trigger_tics.push(tic);
            return;
        }
        let Ok(v) = toks[2].parse::<f64>() else {
            return;
        };
        if !v.is_finite() {
            return;
        }
        st.tics.push(tic);
        st.signal.push(v);
    }
}

fn rasterize_with_dt(
    samples: &[(i64, f64)],
    dt_ms: f64,
    vol_tics: &[i64],
) -> (f64, f64, Vec<f64>, Vec<u8>) {
    if samples.len() < 2 || dt_ms <= 0.0 || !dt_ms.is_finite() {
        return (0.0, 0.0, Vec::new(), Vec::new());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by_key(|(t, _)| *t);
    let dt_tics = dt_ms / MDH_TIC_MS;
    let first = sorted[0].0;
    let last = sorted[sorted.len() - 1].0;
    let span = (last - first) as f64;
    let mut exp_n = ((span / dt_tics).round() as usize) + 1;
    if exp_n < sorted.len() {
        exp_n = sorted.len();
    }
    let mut values = vec![f64::NAN; exp_n];
    for &(tic, v) in &sorted {
        let idx = (((tic - first) as f64 / dt_tics).round() as isize)
            .clamp(0, (exp_n - 1) as isize) as usize;
        values[idx] = v;
    }
    let triggers = raster_triggers(first, last, dt_tics, exp_n, vol_tics);
    let hz = 1000.0 / dt_ms;
    let start_sec = if let Some(&v0) = vol_tics.first() {
        let start_ms = (first - v0) as f64 * MDH_TIC_MS;
        (start_ms as i64) as f64 / 1000.0
    } else {
        0.0
    };
    (hz, start_sec, values, triggers)
}

fn raster_triggers(first: i64, last: i64, dt_tics: f64, exp_n: usize, tics: &[i64]) -> Vec<u8> {
    let mut u = vec![0u8; exp_n];
    if dt_tics <= 0.0 || exp_n == 0 {
        return u;
    }
    for &t in tics {
        if t < first || t > last {
            continue;
        }
        let idx = (((t - first) as f64 / dt_tics).ceil() as isize)
            .clamp(0, (exp_n - 1) as isize) as usize;
        u[idx] = 1;
    }
    u
}

#[allow(clippy::too_many_arguments)]
fn write_physio_stream(
    out_stem: &Path,
    label: &str,
    values: &[f64],
    triggers: Option<&[u8]>,
    peak_label: Option<&str>,
    peaks: Option<&[u8]>,
    hz: f64,
    start_sec: f64,
) -> Result<Vec<PathBuf>> {
    let base = format!("{}_recording-{}_physio", out_stem.display(), label);
    let tsv = PathBuf::from(format!("{base}.tsv.gz"));
    let json = PathBuf::from(format!("{base}.json"));

    // TSV: no header (C++ / BIDS SamplingFrequency form).
    let f = File::create(&tsv).map_err(|e| Error::io(&tsv, e))?;
    let mut enc = GzEncoder::new(f, Compression::default());
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() {
            write!(enc, "{v}").map_err(|e| Error::io(&tsv, e))?;
        } else {
            write!(enc, "n/a").map_err(|e| Error::io(&tsv, e))?;
        }
        if let Some(tr) = triggers {
            write!(enc, "\t{}", tr.get(i).copied().unwrap_or(0)).map_err(|e| Error::io(&tsv, e))?;
        }
        if let Some(pk) = peaks {
            write!(enc, "\t{}", pk.get(i).copied().unwrap_or(0)).map_err(|e| Error::io(&tsv, e))?;
        }
        writeln!(enc).map_err(|e| Error::io(&tsv, e))?;
    }
    enc.finish().map_err(|e| Error::io(&tsv, e))?;

    let mut cols = vec![format!("\"{label}\"")];
    if triggers.is_some() {
        cols.push("\"trigger\"".into());
    }
    if let Some(pl) = peak_label {
        cols.push(format!("\"{pl}\""));
    }
    let mut body = String::from("{\n");
    body.push_str("\t\"PhysioType\": \"generic\",\n");
    body.push_str(&format!("\t\"Columns\": [{}],\n", cols.join(", ")));
    body.push_str(&format!("\t\"SamplingFrequency\": {hz},\n"));
    body.push_str(&format!("\t\"StartTime\": {start_sec}"));
    if let Some(pl) = peak_label {
        body.push_str(",\n");
        body.push_str(&format!(
            "\t\"{pl}\": {{\n\t\t\"Description\": \"Firmware-detected physiological event peak (e.g. cardiac R-wave or respiratory peak): 1 at a sample the scanner flagged as a detected peak, 0 otherwise. Independent of the scanner-volume `trigger` column.\"\n\t}}\n"
        ));
    } else {
        body.push('\n');
    }
    body.push_str("}\n");
    std::fs::write(&json, body).map_err(|e| Error::io(&json, e))?;
    Ok(vec![tsv, json])
}
