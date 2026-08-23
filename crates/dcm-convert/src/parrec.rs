//! Philips PAR/REC reader (`nii_readParRec`): dims, angulation, DTI,
//! RI/RS, and echo×cardiac×type×dynamic volume packing.

use std::fs;
use std::path::{Path, PathBuf};

use dcm_core::error::{Error, Result};
use dcm_dicom::{DicomImage, Manufacturer, Modality};

/// True if path looks like a Philips PAR header.
pub fn is_par_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("par"))
        .unwrap_or(false)
}

/// Per-volume diffusion / contrast meta after packing.
#[derive(Debug, Clone, Default)]
pub struct ParDtiVol {
    pub b_value: f64,
    pub direction: [f64; 3],
}

/// Per-volume contrast flags for multi-output split (mag/phase/real/imag, TE).
#[derive(Debug, Clone, Default)]
pub struct ParVolumeMeta {
    pub te: f64,
    pub echo_num: i32,
    pub trigger_delay: f64,
    pub is_phase: bool,
    pub is_real: bool,
    pub is_imaginary: bool,
    pub b_value: f64,
    pub direction: [f64; 3],
}

#[derive(Clone)]
struct ParRow {
    cols: Vec<f64>,
    disk_idx: usize,
}

/// Parse a PAR file into a synthetic `DicomImage` + voxels + per-volume meta.
/// REC is expected beside the PAR with the same stem.
///
/// Volumes are packed in C++ order:
/// `dyn → grad/bval → echo → cardiac → ASL label → (seq) → image type`
/// and REC slices are reordered via `sliceOrder`.
pub fn read_par_rec(
    par_path: &Path,
) -> Result<(
    DicomImage,
    Vec<f32>,
    [usize; 4],
    Vec<ParDtiVol>,
    Vec<ParVolumeMeta>,
)> {
    let text = fs::read_to_string(par_path).map_err(|e| Error::io(par_path, e))?;
    eprintln!("Warning: dcm2niix PAR is not actively supported (hint: use dicm2nii)");

    let mut protocol = String::new();
    let mut par_vers = 40;
    let mut tr = 0.0f64;
    let mut nz_hdr = 0usize;
    let mut max_diffusion_values = 1i32;
    let mut max_gradient_orients = 1i32;
    let mut max_cardiac_phases = 1i32;
    let mut max_echoes = 1i32;
    let mut max_dynamics = 1i32;
    let mut max_mixes = 1i32;
    let mut max_labels = 1i32;
    let mut v3_bits = 16i32;
    let mut v3_xdim = 128usize;
    let mut v3_ydim = 128usize;
    let mut v3_thick = 2.0f64;
    let mut v3_gap = 0.0f64;
    let mut rows: Vec<ParRow> = Vec::new();
    let mut disk_idx = 0usize;

    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('#') {
            // Version: CLINICAL TRYOUT … V4.2
            if t.contains("TRYOUT") || t.contains("CLINICAL") {
                if let Some(v) = t.split_whitespace().last() {
                    let digits: String = v.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
                    if let Ok(f) = digits.parse::<f64>() {
                        par_vers = (f * 10.0).round() as i32;
                    }
                }
            }
            continue;
        }
        if t.starts_with('.') {
            let parts: Vec<&str> = t.trim_start_matches('.').split_whitespace().collect();
            if parts.len() >= 4 && parts[0] == "Protocol" && parts[1].starts_with("name") {
                if let Some(v) = t.split(':').nth(1) {
                    protocol = v.trim().to_string();
                }
            } else if parts.len() >= 4 && parts[0] == "Repetition" && parts[1] == "time" {
                tr = parts.last().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            } else if parts.len() >= 6 && parts[0] == "Max." && parts[1] == "number" && parts[3] == "slices" {
                nz_hdr = parts[5].parse().unwrap_or(0);
            } else if parts.len() >= 7 && parts[0] == "Max." && parts[3] == "diffusion" {
                max_diffusion_values = parts[6].parse().unwrap_or(1).max(1);
            } else if parts.len() >= 7 && parts[0] == "Max." && parts[3] == "gradient" {
                max_gradient_orients = parts[6].parse().unwrap_or(1).max(1);
            } else if parts.len() >= 7 && parts[0] == "Max." && parts[3] == "cardiac" {
                max_cardiac_phases = parts[6].parse().unwrap_or(1).max(1);
            } else if parts.len() >= 6 && parts[0] == "Max." && parts[3] == "echoes" {
                max_echoes = parts[5].parse().unwrap_or(1).max(1);
            } else if parts.len() >= 6 && parts[0] == "Max." && parts[3] == "dynamics" {
                max_dynamics = parts[5].parse().unwrap_or(1).max(1);
            } else if parts.len() >= 6 && parts[0] == "Max." && parts[3] == "mixes" {
                max_mixes = parts[5].parse().unwrap_or(1).max(1);
                if max_mixes > 1 {
                    eprintln!("Error: maxNumberOfMixes > 1. Please update this software to support these images");
                }
            } else if parts.len() >= 8 && parts[0] == "Number" && parts[2] == "label" {
                max_labels = parts[7].parse().unwrap_or(1).max(1);
            } else if parts.len() >= 7 && parts[0] == "Recon" && parts[1] == "resolution" {
                v3_xdim = parts[5].parse().unwrap_or(128);
                v3_ydim = parts[6].parse().unwrap_or(128);
            } else if parts.len() >= 9 && parts[1] == "pixel" && parts[2] == "size" {
                v3_bits = parts[8].parse().unwrap_or(16);
            } else if parts.len() >= 5 && parts[0] == "Slice" && parts[1] == "gap" {
                v3_gap = parts[4].parse().unwrap_or(0.0);
            } else if parts.len() >= 5 && parts[0] == "Slice" && parts[1] == "thickness" {
                v3_thick = parts[4].parse().unwrap_or(2.0);
            }
            continue;
        }

        // Image-information row: whitespace-separated floats.
        let cols: Vec<f64> = t
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if cols.len() < 7 || cols[0] as i32 == 0 {
            continue;
        }
        rows.push(ParRow {
            cols,
            disk_idx,
        });
        disk_idx += 1;
    }

    if par_vers < 20 {
        return Err(Error::bad_file(format!(
            "{}: PAR files should have CLINICAL TRYOUT version 2.0–4.2",
            par_path.display()
        )));
    }
    if rows.is_empty() {
        return Err(Error::bad_file(format!(
            "{}: no image-information rows",
            par_path.display()
        )));
    }

    // Column index helpers (0-based), matching C++ k* defines.
    let (k_bits, k_xdim, k_ydim, k_ri, k_rs, k_ss, k_ang_ap, k_ang_fh, k_ang_rl, k_pos_ap, k_pos_fh, k_pos_rl, k_thick, k_gap, k_ori, k_xmm, k_ymm, k_te, k_dyn_time, k_trig, k_bval, k_inv, k_bval_num, k_grad_num, k_v1, k_v2, k_v3, k_asl) =
        if par_vers < 40 {
            (
                usize::MAX, usize::MAX, usize::MAX, 7, 8, 9, 12, 13, 14, 15, 16, 17,
                usize::MAX, usize::MAX, 19, 22, 23, 24, 25, 26, 27, usize::MAX, usize::MAX,
                usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX,
            )
        } else {
            (
                7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 25, 28, 29, 30, 31, 32, 33,
                40, 41, 42, 47, 45, 46, 48,
            )
        };

    let num3d_expected = (max_gradient_orients
        * max_diffusion_values
        * max_labels
        * max_cardiac_phases
        * max_echoes
        * max_dynamics
        * max_mixes) as usize;

    let c0 = &rows[0].cols;
    let (nx, ny, bits, mut dx, mut dy, mut dz, mut ang_ap, mut ang_fh, mut ang_rl, mut pos_ap, mut pos_fh, mut pos_rl, mut slice_orient, mut te, mut ri, mut rs, mut ss) =
        if par_vers < 40 {
            (
                v3_xdim,
                v3_ydim,
                v3_bits,
                col(c0, k_xmm).unwrap_or(1.0),
                col(c0, k_ymm).unwrap_or(1.0),
                v3_thick + v3_gap,
                col(c0, k_ang_ap).unwrap_or(0.0),
                col(c0, k_ang_fh).unwrap_or(0.0),
                col(c0, k_ang_rl).unwrap_or(0.0),
                col(c0, k_pos_ap).unwrap_or(0.0),
                col(c0, k_pos_fh).unwrap_or(0.0),
                col(c0, k_pos_rl).unwrap_or(0.0),
                col(c0, k_ori).unwrap_or(1.0) as i32,
                col(c0, k_te).unwrap_or(0.0),
                col(c0, k_ri).unwrap_or(0.0),
                col(c0, k_rs).unwrap_or(1.0),
                col(c0, k_ss).unwrap_or(0.0),
            )
        } else {
            (
                col(c0, k_xdim).unwrap_or(0.0) as usize,
                col(c0, k_ydim).unwrap_or(0.0) as usize,
                col(c0, k_bits).unwrap_or(16.0) as i32,
                col(c0, k_xmm).unwrap_or(1.0),
                col(c0, k_ymm).unwrap_or(1.0),
                col(c0, k_thick).unwrap_or(0.0) + col(c0, k_gap).unwrap_or(0.0),
                col(c0, k_ang_ap).unwrap_or(0.0),
                col(c0, k_ang_fh).unwrap_or(0.0),
                col(c0, k_ang_rl).unwrap_or(0.0),
                col(c0, k_pos_ap).unwrap_or(0.0),
                col(c0, k_pos_fh).unwrap_or(0.0),
                col(c0, k_pos_rl).unwrap_or(0.0),
                col(c0, k_ori).unwrap_or(1.0) as i32,
                col(c0, k_te).unwrap_or(0.0),
                col(c0, k_ri).unwrap_or(0.0),
                col(c0, k_rs).unwrap_or(1.0),
                col(c0, k_ss).unwrap_or(0.0),
            )
        };

    let mut nz = if nz_hdr > 0 {
        nz_hdr
    } else {
        rows.iter()
            .map(|r| r.cols[0] as i32)
            .max()
            .unwrap_or(1)
            .max(1) as usize
    };
    if dx <= 0.0 {
        dx = 1.0;
    }
    if dy <= 0.0 {
        dy = 1.0;
    }
    if dz <= 0.0 {
        dz = 1.0;
    }
    if rs == 0.0 {
        rs = 1.0;
    }

    let mut max_echo_seen = 1i32;
    let mut max_cardiac_seen = 1i32;
    let mut max_vol = -1i32;
    let mut has_mag = false;
    let mut has_real = false;
    let mut has_imag = false;
    let mut has_phase = false;
    let mut adc_warning = false;
    let mut seq1 = -1i32;
    let mut max_seq = -1i32;
    let mut seq_warning = false;
    let mut min_slice = i32::MAX;
    let mut max_slice = 0i32;
    let mut pos_last = [pos_rl, pos_ap, pos_fh];
    let mut max_dyn_time = 0.0f64;
    let mut min_dyn_time = 999999.0f64;
    let mut max_dyn = 0i32;
    let mut min_dyn = i32::MAX;
    let mut te_fallback = te;

    // sorted_slot → disk_idx; size grows with max slice index seen.
    let mut slice_order: Vec<i32> = Vec::new();
    let mut vol_meta_by_vol: Vec<Option<ParVolumeMeta>> = Vec::new();

    for r in &rows {
        let c = &r.cols;
        let slice = c[0] as i32;
        let echo = c.get(1).copied().unwrap_or(1.0) as i32;
        let dyn_num = c.get(2).copied().unwrap_or(1.0) as i32;
        let cardiac = c.get(3).copied().unwrap_or(1.0) as i32;
        let image_type = c.get(4).copied().unwrap_or(0.0);
        let sequence = c.get(5).copied().unwrap_or(0.0) as i32;

        max_echo_seen = max_echo_seen.max(echo);
        max_cardiac_seen = max_cardiac_seen.max(cardiac);
        min_slice = min_slice.min(slice);
        if slice > max_slice {
            max_slice = slice;
            pos_last = [
                col(c, k_pos_rl).unwrap_or(pos_last[0]),
                col(c, k_pos_ap).unwrap_or(pos_last[1]),
                col(c, k_pos_fh).unwrap_or(pos_last[2]),
            ];
        }
        max_dyn = max_dyn.max(dyn_num);
        min_dyn = min_dyn.min(dyn_num);
        if let Some(dt) = col(c, k_dyn_time) {
            max_dyn_time = max_dyn_time.max(dt);
            min_dyn_time = min_dyn_time.min(dt);
        }

        let bval = col(c, k_bval).unwrap_or(0.0);
        let v1 = col(c, k_v1).unwrap_or(0.0);
        let v2 = col(c, k_v2).unwrap_or(0.0);
        let v3 = col(c, k_v3).unwrap_or(0.0);
        let is_adc = max_gradient_orients >= 2
            && bval > 50.0
            && v1.abs() < 1e-12
            && v2.abs() < 1e-12
            && v3.abs() < 1e-12;
        if is_adc {
            adc_warning = true;
        }

        // Volume index (C++ packing).
        let mut vol_step = max_dynamics as i32;
        let mut vol = dyn_num - 1;
        if max_diffusion_values > 1 {
            let grad = col(c, k_grad_num).unwrap_or(1.0) as i32 - 1;
            let grad = grad.max(0);
            let bnum = col(c, k_bval_num).unwrap_or(1.0) as i32 - 1;
            let bnum = bnum.max(0);
            if is_adc {
                vol += vol_step * max_diffusion_values * max_gradient_orients + bnum;
            } else {
                vol += vol_step * grad + bnum * max_gradient_orients;
            }
            vol_step *= (max_diffusion_values + 1) * max_gradient_orients;
        }
        vol += vol_step * (echo - 1);
        vol_step *= max_echoes;
        vol += vol_step * (cardiac - 1);
        vol_step *= max_cardiac_phases;
        let mut asl = col(c, k_asl).unwrap_or(1.0) as i32;
        if asl < 1 {
            asl = 1;
        }
        vol += vol_step * (asl - 1);
        vol_step *= max_labels;

        if seq1 < 0 {
            seq1 = sequence;
        }
        if sequence > max_seq {
            max_seq = sequence;
        }
        if sequence != seq1 {
            if !seq_warning {
                eprintln!(
                    "Warning: 'scanning sequence' column varies within a single file. This behavior is not described at the top of the header."
                );
                seq_warning = true;
            }
            vol += vol_step;
            // vol_step *= 2; // unused after this for type offset
        }

        let mut is_magnitude = (image_type - 0.0).abs() < 1e-6;
        let mut is_real = (image_type - 1.0).abs() < 1e-6;
        let mut is_imaginary = (image_type - 2.0).abs() < 1e-6;
        let mut is_phase = (image_type - 3.0).abs() < 1e-6;
        if (image_type - 18.0).abs() < 1e-6 {
            is_real = true;
        }
        if (image_type - 4.0).abs() < 1e-6 {
            is_phase = true;
        }
        if !(0.0..=3.0).contains(&image_type) && (image_type - 18.0).abs() > 1e-6 {
            // Unknown type → treat as real (C++ kludge).
            is_real = true;
            is_magnitude = false;
        }
        if is_magnitude {
            has_mag = true;
        }
        if is_real {
            has_real = true;
        }
        if is_imaginary {
            has_imag = true;
        }
        if is_phase {
            has_phase = true;
        }

        if is_real {
            vol += num3d_expected as i32;
        }
        if is_imaginary {
            vol += 2 * num3d_expected as i32;
        }
        if is_phase {
            vol += 3 * num3d_expected as i32;
        }
        max_vol = max_vol.max(vol);

        let row_te = col(c, k_te).unwrap_or(0.0);
        let te_vol = if row_te.abs() < 1e-12 {
            te_fallback
        } else {
            te_fallback = row_te;
            row_te
        };
        let trig = col(c, k_trig).unwrap_or(0.0);

        let sorted_slice = slice + vol * nz as i32; // 1-based in C++ before -1
        let slot = (sorted_slice - 1) as usize;
        if slice_order.len() <= slot {
            slice_order.resize(slot + 1, -1);
        }
        slice_order[slot] = r.disk_idx as i32;

        let vi = vol as usize;
        if vol_meta_by_vol.len() <= vi {
            vol_meta_by_vol.resize(vi + 1, None);
        }
        if vol_meta_by_vol[vi].is_none() {
            vol_meta_by_vol[vi] = Some(ParVolumeMeta {
                te: te_vol,
                echo_num: echo.max(1),
                trigger_delay: trig,
                is_phase,
                is_real,
                is_imaginary,
                b_value: bval,
                direction: [v1, v2, v3],
            });
        }

        // Keep geometry from first row; refresh RI/RS if needed.
        if let Some(v) = col(c, k_ri) {
            ri = v;
        }
        if let Some(v) = col(c, k_rs) {
            if v != 0.0 {
                rs = v;
            }
        }
        if let Some(v) = col(c, k_ss) {
            ss = v;
        }
        if let Some(v) = col(c, k_ang_ap) {
            ang_ap = v;
        }
        if let Some(v) = col(c, k_ang_fh) {
            ang_fh = v;
        }
        if let Some(v) = col(c, k_ang_rl) {
            ang_rl = v;
        }
        if let Some(v) = col(c, k_ori) {
            slice_orient = v as i32;
        }
        if te <= 0.0 && te_vol > 0.0 {
            te = te_vol;
        }
        let _ = (k_inv, has_mag); // silence
    }

    if (max_slice - min_slice + 1) as usize != nz && max_slice >= min_slice {
        let num_slice = (max_slice - min_slice + 1) as usize;
        eprintln!(
            "Warning: Expected {nz} slices, but found {num_slice} ({min_slice}..{max_slice}). {}",
            par_path.display()
        );
        nz = num_slice.max(1);
    }

    if max_echo_seen > 1 || max_cardiac_seen > 1 {
        eprintln!(
            "Warning: Multiple Echo ({max_echo_seen}) or Cardiac ({max_cardiac_seen}). Carefully inspect output"
        );
    }
    if adc_warning {
        eprintln!(
            "Warning: PAR/REC dataset includes derived (isotropic, ADC, etc) map(s) that could disrupt analysis. Please remove volume and ensure vectors are reported correctly"
        );
    }

    // Measured TR from dynamic times when no DTI.
    if max_dyn > min_dyn && max_dyn_time > min_dyn_time && max_diffusion_values <= 1 {
        let num_dyn = (max_dyn - min_dyn + 1) as f64;
        if num_dyn > 1.0 {
            let tr_ms = 1000.0 * (max_dyn_time - min_dyn_time) / (num_dyn - 1.0);
            if (tr_ms - tr).abs() > 0.005 {
                eprintln!(
                    "Warning: Reported TR={tr}ms, measured TR={tr_ms}ms (prospect. motion corr.?)"
                );
            }
            tr = tr_ms;
        }
    }

    // Slice spacing from positions when possible.
    if nz > 1 && min_slice == 1 && max_slice > min_slice {
        let dxp = pos_rl - pos_last[0];
        let dyp = pos_ap - pos_last[1];
        let dzp = pos_fh - pos_last[2];
        let slice_mm = (dxp * dxp + dyp * dyp + dzp * dzp).sqrt() / (max_slice - min_slice) as f64;
        if (slice_mm - dz).abs() > 1e-3 && slice_mm > 0.0 {
            eprintln!(
                "Warning: Distance between slices reported by slice gap+thick does not match estimate from slice positions (issue 273)."
            );
            dz = slice_mm;
        }
    }

    let nt = if max_vol >= 0 {
        (max_vol + 1) as usize
    } else {
        1
    };
    let n_slices_total = rows.len();
    if n_slices_total % nz != 0 {
        return Err(Error::bad_file(format!(
            "{}: Total number of slices ({n_slices_total}) not divisible by slices per 3D volume ({nz})",
            par_path.display()
        )));
    }
    // Prefer packed nt from volume indices; fall back to disk count.
    let nt = if nt * nz >= n_slices_total {
        nt
    } else {
        n_slices_total / nz
    };

    eprintln!(
        "Done reading PAR header version {:.1}, with {} slices",
        par_vers as f64 / 10.0,
        n_slices_total
    );

    let rec = par_path.with_extension("rec");
    let rec_alt = par_path.with_extension("REC");
    let rec_path = if rec.exists() {
        rec
    } else if rec_alt.exists() {
        rec_alt
    } else {
        return Err(Error::bad_file(format!(
            "{}: matching REC not found",
            par_path.display()
        )));
    };
    let bytes = fs::read(&rec_path).map_err(|e| Error::io(&rec_path, e))?;
    let slice_vox = nx * ny;
    let bpp = if bits <= 8 { 1 } else { 2 };
    let disk_slices = n_slices_total;
    if bytes.len() < disk_slices * slice_vox * bpp && bytes.len() < disk_slices * slice_vox * 4 {
        return Err(Error::bad_file(format!(
            "{}: REC too small for {nx}x{ny}x{disk_slices}",
            rec_path.display()
        )));
    }

    // Decode disk-order slices to f32.
    let mut disk_vol = Vec::with_capacity(disk_slices * slice_vox);
    if bpp == 1 && bytes.len() >= disk_slices * slice_vox {
        for &b in bytes.iter().take(disk_slices * slice_vox) {
            disk_vol.push((b as f32) * (rs as f32) + (ri as f32));
        }
    } else if bytes.len() >= disk_slices * slice_vox * 2 {
        for chunk in bytes.chunks_exact(2).take(disk_slices * slice_vox) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f32;
            disk_vol.push(s * (rs as f32) + (ri as f32));
        }
    } else {
        for chunk in bytes.chunks_exact(4).take(disk_slices * slice_vox) {
            disk_vol.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }

    // Reorder into sorted [vol][z] layout using slice_order (disk index per sorted slot).
    let out_slices = nt * nz;
    let mut vol = vec![0.0f32; out_slices * slice_vox];
    let use_order = slice_order.iter().any(|&v| v >= 0) && slice_order.len() >= out_slices.min(disk_slices);
    if use_order {
        for (sorted_i, &disk) in slice_order.iter().enumerate().take(out_slices) {
            if disk < 0 {
                continue;
            }
            let di = disk as usize;
            if di >= disk_slices {
                continue;
            }
            let src = di * slice_vox;
            let dst = sorted_i * slice_vox;
            vol[dst..dst + slice_vox].copy_from_slice(&disk_vol[src..src + slice_vox]);
        }
    } else {
        let n = (disk_slices * slice_vox).min(vol.len());
        vol[..n].copy_from_slice(&disk_vol[..n]);
    }

    let orient = angulation_to_orient(ang_rl, ang_ap, ang_fh, slice_orient);
    let patient_position = [0.0, pos_rl, pos_ap, pos_fh];

    let mut d = minimal_image(par_path);
    d.rows = ny;
    d.columns = nx;
    d.xyz_mm = [1.0, dx, dy, dz];
    d.spacing_between_slices = dz;
    d.slice_thickness = dz;
    d.tr = tr;
    d.te = te;
    d.orient = orient;
    d.patient_position = patient_position;
    d.patient_position_last = [0.0, pos_last[0], pos_last[1], pos_last[2]];
    d.protocol_name = protocol;
    d.series_description = d.protocol_name.clone();
    d.manufacturer = Manufacturer::Philips;
    d.modality = Modality::Mr;
    d.number_of_frames = (nz * nt) as i32;
    d.inten_scale = rs as f32;
    d.inten_intercept = ri as f32;
    d.inten_scale_philips = ss as f32;
    d.bits_allocated = if bits <= 8 { 8 } else { 16 };
    d.bits_stored = d.bits_allocated;
    d.is_float = true;
    d.is_has_phase = has_phase;
    d.is_has_real = has_real;
    d.is_has_imaginary = has_imag;
    if max_echoes > 1 || max_echo_seen > 1 {
        d.echo_number = max_echo_seen.max(1);
    }

    let vol_metas: Vec<ParVolumeMeta> = (0..nt)
        .map(|i| {
            vol_meta_by_vol
                .get(i)
                .and_then(|o| o.clone())
                .unwrap_or_default()
        })
        .collect();

    let dti: Vec<ParDtiVol> = if vol_metas.iter().any(|v| {
        v.b_value > 0.0 || v.direction.iter().any(|&x| x.abs() > 1e-12)
    }) {
        vol_metas
            .iter()
            .map(|v| ParDtiVol {
                b_value: v.b_value,
                direction: v.direction,
            })
            .collect()
    } else {
        Vec::new()
    };
    if let Some(first) = dti.first() {
        d.b_value = first.b_value;
        d.diffusion_direction = first.direction;
    }
    if let Some(m0) = vol_metas.first() {
        if m0.te > 0.0 {
            d.te = m0.te;
        }
        d.echo_number = m0.echo_num.max(1);
        d.trigger_delay_time = m0.trigger_delay;
        d.is_has_phase = m0.is_phase;
        d.is_has_real = m0.is_real;
        d.is_has_imaginary = m0.is_imaginary;
    }

    Ok((d, vol, [nx, ny, nz, nt], dti, vol_metas))
}

fn col(cols: &[f64], idx: usize) -> Option<f64> {
    if idx == usize::MAX {
        return None;
    }
    cols.get(idx).copied()
}

/// Xiangrui Li / dcm2niix PAR angulation → DICOM direction cosines (subset).
fn angulation_to_orient(ang_rl: f64, ang_ap: f64, ang_fh: f64, slice_orient: i32) -> [f64; 7] {
    let d2r = std::f64::consts::PI / 180.0;
    let (ca0, ca1, ca2) = (
        (ang_rl * d2r).cos(),
        (ang_ap * d2r).cos(),
        (ang_fh * d2r).cos(),
    );
    let (sa0, sa1, sa2) = (
        (ang_rl * d2r).sin(),
        (ang_ap * d2r).sin(),
        (ang_fh * d2r).sin(),
    );
    let rx = [[1.0, 0.0, 0.0], [0.0, ca0, -sa0], [0.0, sa0, ca0]];
    let ry = [[ca1, 0.0, sa1], [0.0, 1.0, 0.0], [-sa1, 0.0, ca1]];
    let rz = [[ca2, -sa2, 0.0], [sa2, ca2, 0.0], [0.0, 0.0, 1.0]];
    let mut r = mat_mul(mat_mul(rx, ry), rz);
    // slice_orient: 1=TRA, 2=SAG, 3=COR
    let ixyz = match slice_orient {
        2 => {
            for row in &mut r {
                row[0] = -row[0];
                row[2] = -row[2];
            }
            [1usize, 2, 0]
        }
        3 => {
            for row in &mut r {
                row[1] = -row[1];
                row[2] = -row[2];
            }
            [0usize, 2, 1]
        }
        _ => [0usize, 1, 2],
    };
    let mut orient = [0.0; 7];
    orient[1] = r[0][ixyz[0]];
    orient[2] = r[1][ixyz[0]];
    orient[3] = r[2][ixyz[0]];
    orient[4] = r[0][ixyz[1]];
    orient[5] = r[1][ixyz[1]];
    orient[6] = r[2][ixyz[1]];
    orient
}

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

pub(crate) fn minimal_image(path: &Path) -> DicomImage {
    use dcm_dicom::CsaMeta;
    DicomImage {
        path: PathBuf::from(path),
        series_uid: format!("PARREC.{}", path.display()),
        series_uid_crc: 1,
        instance_uid: String::new(),
        study_uid: String::new(),
        series_number: 1,
        instance_number: 1,
        acquisition_number: 1,
        echo_number: 1,
        rows: 0,
        columns: 0,
        bits_allocated: 16,
        bits_stored: 16,
        samples_per_pixel: 1,
        is_signed: true,
        is_float: false,
        xyz_mm: [1.0, 1.0, 1.0, 1.0],
        slice_thickness: 1.0,
        orient: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        patient_position: [0.0, 0.0, 0.0, 0.0],
        patient_position_last: [f64::NAN; 4],
        last_scan_loc: f64::NAN,
        acquisition_duration: 0.0,
        manufacturer: Manufacturer::Philips,
        modality: Modality::Mr,
        manufacturer_name: "Philips".into(),
        manufacturers_model_name: String::new(),
        institution_name: String::new(),
        institution_address: String::new(),
        institutional_department: String::new(),
        procedure_step_description: String::new(),
        station_name: String::new(),
        device_serial_number: String::new(),
        software_versions: String::new(),
        protocol_name: String::new(),
        series_description: String::new(),
        sequence_name: String::new(),
        pulse_sequence_name: String::new(),
        scanning_sequence: String::new(),
        sequence_variant: String::new(),
        scan_options: String::new(),
        image_type: "ORIGINAL".into(),
        image_comments: String::new(),
        coil_name: String::new(),
        coil_string: String::new(),
        transmit_coil_name: String::new(),
        patient_name: String::new(),
        patient_id: String::new(),
        patient_sex: String::new(),
        patient_age: String::new(),
        referring_physician_name: String::new(),
        patient_birth_date: String::new(),
        patient_weight: 0.0,
        patient_size: 0.0,
        accession_number: String::new(),
        study_id: String::new(),
        study_description: String::new(),
        study_date: String::new(),
        study_time: String::new(),
        series_time: String::new(),
        acquisition_date: String::new(),
        acquisition_time: String::new(),
        body_part: String::new(),
        tr: 0.0,
        te: 0.0,
        ti: 0.0,
        flip_angle: 0.0,
        field_strength: 0.0,
        pixel_bandwidth: 0.0,
        echo_train_length: 0,
        phase_encoding_rc: ' ',
        inten_scale: 1.0,
        inten_intercept: 0.0,
        inten_scale_philips: 0.0,
        is_scale_varies_enh: false,
        is_derived: false,
        is_localizer: false,
        number_of_frames: 1,
        imaging_frequency: 0.0,
        patient_position_label: String::new(),
        spacing_between_slices: 0.0,
        acquisition_matrix_pe: 0,
        phase_encoding_steps: 0,
        phase_encoding_steps_out_of_plane: 0,
        number_of_concatenations: 1,
        repetition_time_excitation: -1.0,
        repetition_time_inversion: 0.0,
        percent_phase_fov: 0.0,
        percent_sampling: 0.0,
        mra_acquisition_type: String::new(),
        b_value: -1.0,
        diffusion_direction: [0.0; 3],
        pe_direction_displayed: String::new(),
        number_of_averages: 0.0,
        is_3d_acq: false,
        is_epi: false,
        is_ir: false,
        accel_fact_pe: 0.0,
        internal_pulse_sequence_name: String::new(),
        shim_setting: [0.0; 3],
        prescan_reuse_string: String::new(),
        effective_echo_spacing_ge: 0.0,
        acquisition_duration_s: 0.0,
        phase_encoding_ge: -1,
        parallel_reduction_out_of_plane: 0.0,
        sar: 0.0,
        dwell_time_ns: 0.0,
        csa: CsaMeta::default(),
        is_mosaic: false,
        image_orientation_text: String::new(),
        is_mrs: false,
        is_mrs_ref: false,
        data_point_columns: 0,
        resonant_nucleus: String::new(),
        mrs_acq_type: 0,

        voi_phase_fov: 0.0,
        voi_readout_fov: 0.0,
        voi_thickness: 0.0,
        voi_center_lps: [0.0; 3],
        has_voi_center: false,
        voi_orient: [0.0; 7],
        number_of_k_space_trajectories: 0,
        spectral_width_hz: 0.0,
        is_xa: false,
        is_pmsct_rle1: false,
        is_bvec_world_coordinates: false,
        gantry_tilt: 0.0,
        study_uid_crc: 0,
        coil_crc: 0,
        date_time: 0.0,
        is_has_phase: false,
        is_has_real: false,
        is_has_imaginary: false,
        is_has_magnitude: false,
        is_no_rf: false,
        image_type_text: String::new(),
        is_deep_learning: false,
        deep_learning_text: String::new(),
        frequency_encoding_steps: 0,
        is_variable_flip_angle: false,
        parallel_acquisition_technique: String::new(),
        is_raw_data_storage: false,
        is_grayscale_softcopy_presentation_state: false,
        is_quadruped: false,
        convolution_kernel: String::new(),
        recon_filter_size: f64::NAN,
        pixel_padding_value: f64::NAN,
        is_xray: false,
        exposure_time_ms: 0.0,
        x_ray_tube_current: 0.0,
        is_xa_physio: false,
        is_cmrr_physio: false,
        physio_offset: -1,
        physio_bytes: 0,
        trigger_delay_time: 0.0,
        asl_flags: 0,
        post_label_delay: 0,
        labeling_orientation: String::new(),
        vascular_crushing: -1,
        vascular_crushing_venc: 0.0,
        duration_label_pulse_ge: -1,
        number_of_excitations: -1.0,
        number_of_arms: -1.0,
        number_of_points_per_arm: -1.0,
        group_delay: 0.0,
        ge_slice_order: -1,
        ge_iopt: String::new(),
        epi_version_ge: -1,
        internal_epi_version_ge: -1,
        ge_user_data_12: 0,
        temporal_position: -1,
        water_fat_shift: 0.0,
        partial_fourier_direction: 0,
        is_partial_fourier: false,
        velocity_encode_scale_ge: 1.0,
        max_echo_num_ge: -1,
        rwv_scale: 0.0,
        rwv_intercept: 0.0,
        mt_state: -1,
        spoiling: -1,
        interp_3d: -1,
        phase_number: -1,
        acquisition_contrast: 0,
        is_diffusion: false,
        is_multi_echo: false,
        is_real_is_phase_map_hz: false,
        raw_data_run_number: 0,
        is_has_overlay: false,
        overlays: Default::default(),
        rtia_timer_ge: 0.0,
        is_planar_rgb: false,
        diff_cycling_mode_ge: -1,
        diff_cycling_mode_ge_override: false,
        number_of_diffusion_direction_ge: -1,
        number_of_diffusion_t2_ge: -1,
        tensor_file_ge: 0,
        compressed_sensing_factor: 0.0,
        frame_duration: -1.0,
        frame_reference_time: -1.0,
        decay_factor: -1.0,
        deidentification_method: String::new(),
        deidentification_method_code_sequence: vec![],
        ecat_isotope_halflife: 0.0,
        ecat_dosage: 0.0,
        volume_onset_times: Vec::new(),
        frame_durations: Vec::new(),
        frame_reference_times: Vec::new(),
        decay_factors: Vec::new(),
        radiopharmaceutical: String::new(),
        tracer_radionuclide: String::new(),
        radionuclide_total_dose: 0.0,
        radionuclide_half_life: 0.0,
        radionuclide_positron_fraction: 0.0,
        radiopharmaceutical_specific_activity: 0.0,
        injected_volume: 0.0,
        scatter_fraction: 0.0,
        radiopharmaceutical_start_time: String::new(),
        decay_correction: String::new(),
        attenuation_correction_method: String::new(),
        randoms_correction_method: String::new(),
        scatter_correction_method: String::new(),
        reconstruction_method: String::new(),
        units_pt: String::new(),
        dose_calibration_factor: 0.0,
    }
}
