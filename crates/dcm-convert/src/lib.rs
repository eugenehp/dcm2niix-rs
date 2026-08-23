//! Conversion pipeline: scan → group series → assemble volumes → write NIfTI/BIDS.
//!
//! # Overview
//!
//! [`convert`] is the library entry point used by the `dcm2niix` CLI. It walks
//! an input tree, groups images into series, assembles 3D/4D volumes (including
//! mosaics and enhanced multi-frame), applies geometry fixes, and writes
//! NIfTI (or NRRD / MGH / JNIfTI) plus optional BIDS sidecars.
//!
//! # Voxel backends
//!
//! Row/slice flips default to a tight in-place CPU path. Larger volumes can
//! compile through `rlx-tensor` (feature `gpu` enables wgpu via `Device::Gpu`).
//! See [`voxels`] and `DCM2NIIX_RLX_DEVICE`.

pub mod bids_guess;
pub mod crop;
pub mod descrip;
pub mod dti;
pub mod ecat;
pub mod epi_tr;
pub mod filename;
pub mod foreign;
pub mod gantry;
pub mod geom;
pub mod ini;
pub mod jnifti;
pub mod mosaic;
pub mod mrs;
pub mod onset;
pub mod opts;
pub mod overlay;
pub mod ortho;
pub mod parrec;
pub mod philips;
pub mod physio;
pub mod jobs;
pub mod pigz;
pub mod reproin;
pub mod rgb;
pub mod scale16;
pub mod slice_eq;
pub mod slice_timing;
pub mod text;
pub mod tr_discrepancy;
pub mod voxels;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dcm_bids::{write_sidecar_ex, Anonymize};
use dcm_core::error::{Error, Result};
use dcm_core::exit::Exit;
use dcm_core::matrix::{dot, nifti_dicom2mat, snap_mat44, Matrix4};
use dcm_dicom::{
    collect_dicom_files, decode_opened_raw_f32, decode_pixels_prefetched, needs_extra_y_flip,
    open_prefetched, warmup_convert_cache, DicomImage, MmapCache, Modality,
};
use rayon::prelude::*;
use dcm_nifti::{
    f32_voxels_to_f32_bytes, f32_voxels_to_i16_bytes, f32_voxels_to_u16_bytes,
    f32_voxels_to_u8_bytes, Nifti1Header, DT_FLOAT32, DT_INT16, DT_INT32, DT_UINT16, DT_UINT8,
};

use crate::descrip::nifti_descrip;
use crate::dti::{
    plan_dti_volumes, remove_trailing_adc, reorder_volume_bytes, save_dti_sidecars_ex,
};
use crate::filename::create_filename;
use crate::foreign::{write_mgh, write_nrrd};
use crate::gantry::{
    apply_gantry_tilt_sform, compute_gantry_tilt_precise, correct_tilt,
};
use crate::jnifti::write_jnifti;
use crate::geom::{
    apply_flip_y_sform, apply_flip_z_sform, apply_siemens_mosaic_sform, header_from_series,
    interslice_distance, interslice_distance_signed, is_same_float_ge, slice_normal,
    verify_slice_dir,
};
use crate::mosaic::demosaic_f32;
use crate::opts::{BidsMode, Compress, DcmOpts, NameConflict, SaveFormat, StackMode};
use crate::ortho::nii_set_ortho_f32;
use crate::scale16::maximize_16bit;
use crate::slice_timing::{check_slice_timing, ge_rescue_slice_timing_ms};
use crate::voxels::{flip_y_volume, flip_yz_volume};

#[derive(Debug, Clone)]
struct SliceMeta {
    img: DicomImage,
    depth: f64,
}

#[derive(Debug)]
/// Summary returned by [`convert`].
pub struct ConvertReport {
    /// DICOM files discovered under the input path.
    pub n_found: usize,
    /// Series successfully written.
    pub n_converted: usize,
    /// Series skipped (filters, conflicts, or non-image).
    pub n_skipped: usize,
    /// Paths of primary outputs (NIfTI / foreign formats), not sidecars.
    pub outputs: Vec<PathBuf>,
    /// Suggested process exit status (success / partial / failure).
    pub exit: Exit,
}

/// Run a full conversion with the given options (CLI-equivalent entry point).
///
/// Errors map to CLI exit codes via [`dcm_core::exit::Exit`].
pub fn convert(opts: &DcmOpts) -> Result<ConvertReport> {
    // Cap all nested Rayon work (header scan, decode, multi-series, `-a y`).
    crate::jobs::install(|| convert_inner(opts))
}

fn convert_inner(opts: &DcmOpts) -> Result<ConvertReport> {
    let indir = Path::new(&opts.indir);
    if !indir.exists() {
        return Err(Error::bad_file(format!(
            "Error: input folder invalid: {}",
            indir.display()
        )));
    }
    if !opts.outdir.is_empty() {
        let out = Path::new(&opts.outdir);
        if !out.exists() {
            std::fs::create_dir_all(out).map_err(|e| Error::io(out, e))?;
        }
        let md = std::fs::metadata(out).map_err(|e| Error::io(out, e))?;
        if md.permissions().readonly() {
            return Err(Error::convert(format!(
                "You do not have write permissions for the directory {}",
                out.display()
            )));
        }
    }

    // `-a y`: convert each immediate subdirectory independently.
    if opts.one_dir_at_a_time && indir.is_dir() && !opts.single_file {
        return convert_one_dir_at_a_time(opts);
    }

    let files = if opts.single_file {
        vec![indir.to_path_buf()]
    } else {
        collect_dicom_files(indir, opts.dir_search_depth)?
    };

    // ECAT7 foreign format.
    if ecat::is_ecat7(indir)
        || files.iter().any(|f| ecat::is_ecat7(f))
    {
        let ecat_paths: Vec<PathBuf> = if ecat::is_ecat7(indir) {
            vec![indir.to_path_buf()]
        } else {
            files.iter().filter(|f| ecat::is_ecat7(f)).cloned().collect()
        };
        let mut outputs = Vec::new();
        let mut n_converted = 0;
        let mut n_failed = 0;
        for ep in &ecat_paths {
            match convert_ecat(ep, opts) {
                Ok(paths) => {
                    n_converted += 1;
                    outputs.extend(paths);
                }
                Err(e) => {
                    eprintln!("Error converting ECAT {}: {e}", ep.display());
                    n_failed += 1;
                }
            }
        }
        if !ecat_paths.is_empty() {
            return Ok(ConvertReport {
                n_found: ecat_paths.len(),
                n_converted,
                n_skipped: 0,
                outputs,
                exit: if n_failed > 0 {
                    Exit::Failure
                } else {
                    Exit::Success
                },
            });
        }
    }

    // PAR/REC foreign format (alongside DICOM scan).
    if parrec::is_par_file(indir)
        || files.iter().any(|f| parrec::is_par_file(f))
        || (indir.is_dir()
            && std::fs::read_dir(indir).ok().into_iter().flatten().flatten().any(|e| {
                parrec::is_par_file(&e.path())
            }))
    {
        let par_paths: Vec<PathBuf> = if parrec::is_par_file(indir) {
            vec![indir.to_path_buf()]
        } else {
            let mut ps: Vec<_> = files.iter().filter(|f| parrec::is_par_file(f)).cloned().collect();
            if ps.is_empty() && indir.is_dir() {
                if let Ok(rd) = std::fs::read_dir(indir) {
                    for e in rd.flatten() {
                        if parrec::is_par_file(&e.path()) {
                            ps.push(e.path());
                        }
                    }
                }
            }
            ps
        };
        let mut outputs = Vec::new();
        let mut n_converted = 0;
        let mut n_failed = 0;
        for par in &par_paths {
            match convert_par_rec(par, opts) {
                Ok(paths) => {
                    n_converted += 1;
                    outputs.extend(paths);
                }
                Err(e) => {
                    eprintln!("Error converting PAR {}: {e}", par.display());
                    n_failed += 1;
                }
            }
        }
        if !par_paths.is_empty() {
            return Ok(ConvertReport {
                n_found: par_paths.len(),
                n_converted,
                n_skipped: 0,
                outputs,
                exit: if n_failed > 0 {
                    Exit::Failure
                } else {
                    Exit::Success
                },
            });
        }
    }

    if opts.verbose > 0 {
        eprintln!("Found {} DICOM file(s)", files.len());
    }
    if opts.search_only == 1 {
        println!("Found {} DICOM file(s)", files.len());
        return Ok(ConvertReport {
            n_found: files.len(),
            n_converted: 0,
            n_skipped: 0,
            outputs: vec![],
            exit: if files.is_empty() {
                Exit::NoValidFilesFound
            } else {
                Exit::Success
            },
        });
    }
    if opts.search_only == 2 {
        for f in &files {
            println!("{}", f.display());
        }
        println!("Found {} DICOM file(s)", files.len());
        return Ok(ConvertReport {
            n_found: files.len(),
            n_converted: 0,
            n_skipped: 0,
            outputs: vec![],
            exit: Exit::Success,
        });
    }

    let (mmap_cache, parsed) = warmup_convert_cache(&files);
    let mmap_cache = Arc::new(mmap_cache);

    let mut images = Vec::new();
    for (f, parsed_img) in parsed {
        match parsed_img {
            Ok(d) => {
                if d.rows == 0 || d.columns == 0 {
                    continue;
                }
                if opts.ignore_derived && (d.is_derived || d.is_localizer) {
                    continue;
                }
                images.push(d);
            }
            Err(e) => {
                if opts.verbose > 0 {
                    eprintln!("skip {}: {e}", f.display());
                }
            }
        }
    }
    if images.is_empty() {
        return Ok(ConvertReport {
            n_found: files.len(),
            n_converted: 0,
            n_skipped: files.len(),
            outputs: vec![],
            exit: Exit::NoValidFilesFound,
        });
    }

    if opts.rename_not_convert {
        let renamed = text::rename_dicoms(&images, opts)?;
        println!("Converted {} DICOMs", renamed.len());
        return Ok(ConvertReport {
            n_found: files.len(),
            n_converted: renamed.len(),
            n_skipped: 0,
            outputs: renamed,
            exit: Exit::Success,
        });
    }

    let groups = group_series(images, opts);
    let mut outputs = Vec::new();
    let mut n_converted = 0;
    let mut n_failed = 0;
    let n_groups = groups.len();

    // Separate report-only / filtered listing (sequential) from real converts.
    let mut work: Vec<(usize, Vec<DicomImage>)> = Vec::new();
    for (gi, g) in groups.into_iter().enumerate() {
        if opts.progress > 0 {
            eprintln!("Progress: {}/{}", gi + 1, n_groups);
        }
        if opts.report_series_only {
            let mut d0 = g[0].clone();
            if opts.guess_bids_filename {
                bids_guess::set_bids(&mut d0, g.len(), opts.verbose);
            }
            let stem = create_filename(&d0, opts).unwrap_or_else(|_| PathBuf::from("?"));
            if d0.echo_number > 1 {
                println!(
                    "\t{}.{}\t{}",
                    d0.series_uid_crc,
                    d0.echo_number,
                    stem.display()
                );
            } else {
                println!("\t{}\t{}", d0.series_uid_crc, stem.display());
            }
            println!(" {}", g[0].path.display());
            n_converted += 1;
            continue;
        }
        if !opts.series_filter.is_empty() {
            let crc = g[0].series_uid_crc as f64;
            if !opts.series_filter.iter().any(|s| (*s - crc).abs() < 0.5) {
                continue;
            }
        }
        work.push((gi, g));
    }

    if work.len() <= 1 {
        for (_gi, g) in work {
            emit_series_warnings(&g);
            let series_number = g[0].series_number;
            match convert_series(g, opts, &mmap_cache) {
                Ok(paths) => {
                    n_converted += 1;
                    outputs.extend(paths);
                }
                Err(e) => {
                    eprintln!("Error converting series {series_number}: {e}");
                    n_failed += 1;
                }
            }
        }
    } else {
        // Independent series: convert in parallel (decode+write dominate runtime).
        let results: Vec<_> = work
            .into_par_iter()
            .map(|(_gi, g)| {
                emit_series_warnings(&g);
                let series_number = g[0].series_number;
                let cache = Arc::clone(&mmap_cache);
                match convert_series(g, opts, &cache) {
                    Ok(paths) => Ok(paths),
                    Err(e) => Err((series_number, e)),
                }
            })
            .collect();
        for r in results {
            match r {
                Ok(paths) => {
                    n_converted += 1;
                    outputs.extend(paths);
                }
                Err((series_number, e)) => {
                    eprintln!("Error converting series {series_number}: {e}");
                    n_failed += 1;
                }
            }
        }
    }

    let exit = if n_converted == 0 && n_failed == 0 {
        Exit::NoValidFilesFound
    } else if n_failed > 0 && n_converted > 0 {
        Exit::SomeOkSomeBad
    } else if n_failed > 0 {
        Exit::Failure
    } else {
        Exit::Success
    };
    Ok(ConvertReport {
        n_found: files.len(),
        n_converted,
        n_skipped: 0,
        outputs,
        exit,
    })
}

fn emit_series_warnings(g: &[DicomImage]) {
    if g[0].patient_position_label.len() < 3 {
        eprintln!("Warning: Patient Position (0018,5100) not specified (issue 642).");
    }
    if g[0].is_quadruped {
        eprintln!(
            "Warning: Anatomical Orientation Type (0010,2210) is QUADRUPED: rotate coordinates accordingly (issue 642)"
        );
    }
    if g[0].manufacturer == dcm_dicom::Manufacturer::Unknown {
        eprintln!(
            "Warning: Unable to determine manufacturer (0008,0070), so conversion is not tuned for vendor."
        );
    }
    if g[0].manufacturer == dcm_dicom::Manufacturer::Ge
        && g[0].max_echo_num_ge > 0
        && g[0].internal_epi_version_ge != 2
    {
        eprintln!(
            "Warning: GE sequence with {} echoes. See issue 359",
            g[0].max_echo_num_ge
        );
    }
}

fn convert_one_dir_at_a_time(opts: &DcmOpts) -> Result<ConvertReport> {
    let indir = Path::new(&opts.indir);
    let mut outputs = Vec::new();
    let mut n_found = 0;
    let mut n_converted = 0;
    let mut n_failed = 0;
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(indir)
        .map_err(|e| Error::io(indir, e))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subdirs.sort();
    if subdirs.is_empty() {
        let mut o = opts.clone();
        o.one_dir_at_a_time = false;
        return convert_inner(&o);
    }
    // Parallel over subdirs on the bounded job pool (caller already `jobs::install`).
    let results: Vec<_> = subdirs
        .par_iter()
        .map(|sub| {
            let mut o = opts.clone();
            o.one_dir_at_a_time = false;
            o.set_indir(&sub.to_string_lossy());
            // Nested convert_inner (not convert) to avoid re-enter install.
            match convert_inner(&o) {
                Ok(r) => Ok((sub.clone(), r)),
                Err(e) => Err((sub.clone(), e)),
            }
        })
        .collect();
    for r in results {
        match r {
            Ok((_sub, r)) => {
                n_found += r.n_found;
                n_converted += r.n_converted;
                n_failed += if matches!(r.exit, Exit::Failure | Exit::SomeOkSomeBad) {
                    1
                } else {
                    0
                };
                outputs.extend(r.outputs);
            }
            Err((sub, e)) => {
                eprintln!("Error converting {}: {e}", sub.display());
                n_failed += 1;
            }
        }
    }
    let exit = if n_converted == 0 && n_failed == 0 {
        Exit::NoValidFilesFound
    } else if n_failed > 0 && n_converted > 0 {
        Exit::SomeOkSomeBad
    } else if n_failed > 0 {
        Exit::Failure
    } else {
        Exit::Success
    };
    Ok(ConvertReport {
        n_found,
        n_converted,
        n_skipped: 0,
        outputs,
        exit,
    })
}

fn group_series(images: Vec<DicomImage>, opts: &DcmOpts) -> Vec<Vec<DicomImage>> {
    // Pre-bucket by series identity (C++ `isSameSet` rejects cross-bucket pairs),
    // then union-find within each bucket — O(Σ n_b²) instead of O(n²).
    let n = images.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    fn unite(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }
    let mut buckets: BTreeMap<(u8, u64), Vec<usize>> = BTreeMap::new();
    for (i, img) in images.iter().enumerate() {
        let key = if opts.stack == StackMode::ForceIgnoreUid {
            (1u8, img.series_number as u64)
        } else {
            (0u8, img.series_uid_crc as u64)
        };
        buckets.entry(key).or_default().push(i);
    }
    for idxs in buckets.values() {
        for a in 0..idxs.len() {
            for b in (a + 1)..idxs.len() {
                let i = idxs[a];
                let j = idxs[b];
                if is_same_set(&images[i], &images[j], opts) {
                    unite(&mut parent, i, j);
                }
            }
        }
    }
    // Move images into groups (no full-struct clone).
    let mut slots: Vec<Option<DicomImage>> = images.into_iter().map(Some).collect();
    let mut out: BTreeMap<usize, Vec<DicomImage>> = BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        out.entry(r)
            .or_default()
            .push(slots[i].take().expect("group slot"));
    }
    out.into_values().collect()
}

/// Port of C++ `isSameSet` using fields available on `DicomImage`.
fn is_same_set(d1: &DicomImage, d2: &DicomImage, opts: &DcmOpts) -> bool {
    if d1.manufacturer != d2.manufacturer {
        return false;
    }
    if d1.modality != d2.modality {
        return false;
    }
    if d1.is_derived != d2.is_derived {
        return false;
    }
    if d1.rows != d2.rows || d1.columns != d2.columns {
        return false;
    }
    if d1.is_has_phase != d2.is_has_phase
        || d1.is_has_real != d2.is_has_real
        || d1.is_has_imaginary != d2.is_has_imaginary
    {
        return false;
    }
    if d1.is_no_rf != d2.is_no_rf {
        return false; // do not stack RF-off (noise) with imaging volumes
    }
    if opts.stack == StackMode::ForceIgnoreUid {
        if d1.series_number != d2.series_number {
            return false;
        }
    } else if d1.series_uid_crc != d2.series_uid_crc {
        return false;
    }
    // Study date/time + Study Instance UID (C++ isSameStudyInstanceUID / isSameTime).
    if d1.study_uid_crc != 0
        && d2.study_uid_crc != 0
        && d1.study_uid_crc != d2.study_uid_crc
        && (d1.date_time - d2.date_time).abs() > 1e-3
        && !opts.force_stack_dce
    {
        return false;
    }
    let force_stack = matches!(opts.stack, StackMode::Yes | StackMode::ForceIgnoreUid)
        || (opts.stack == StackMode::Auto && (d1.modality == Modality::Ct || d1.is_xray));
    if force_stack {
        return true;
    }
    if d1.coil_crc != 0 && d2.coil_crc != 0 && d1.coil_crc != d2.coil_crc {
        if opts.force_stack_dce {
            // C++ stacks despite coil variation when forceStackDCE.
        } else {
            return false;
        }
    }
    if (d1.te - d2.te).abs() > 1e-4 || d1.echo_number != d2.echo_number {
        return false;
    }
    if (d1.tr - d2.tr).abs() > 1e-4 {
        return false;
    }
    if (d1.flip_angle - d2.flip_angle).abs() > 1e-4 {
        return false;
    }
    // Philips ASL / multiphase: different TriggerDelayTime → different series
    // (issue 384). ASL label/control shares series despite trigger (issue 533).
    // `--ignore_trigger_times` skips this check.
    if !opts.ignore_trigger_times
        && d1.manufacturer == dcm_dicom::Manufacturer::Philips
        && d1.asl_flags == dcm_dicom::ASL_FLAG_NONE
        && (d1.trigger_delay_time - d2.trigger_delay_time).abs() > 1e-4
    {
        return false;
    }
    if !d1.protocol_name.is_empty()
        && !d2.protocol_name.is_empty()
        && d1.protocol_name != d2.protocol_name
    {
        return false;
    }
    if d1.has_orientation() && d2.has_orientation() {
        for i in 1..7 {
            if !is_same_float_ge(d1.orient[i] as f32, d2.orient[i] as f32) {
                // `-i o`: keep stacking despite orientation variation.
                if opts.keep_direction_varies {
                    break;
                }
                return false;
            }
        }
    }
    true
}

fn convert_ecat(path: &Path, opts: &DcmOpts) -> Result<Vec<PathBuf>> {
    let (d, hdr, bytes) = ecat::read_ecat7(path)?;
    write_outputs(&d, &hdr, bytes, 1, opts)
}

fn convert_par_rec(par: &Path, opts: &DcmOpts) -> Result<Vec<PathBuf>> {
    let (d0, vol, [nx, ny, nz, nt], _dti, metas) = parrec::read_par_rec(par)?;
    let xyz = d0.xyz_mm;
    let bp = 4usize; // float32
    let vol_bytes = nx * ny * nz * bp;
    let raw = f32_voxels_to_f32_bytes(&vol);

    // Assign series IDs like enhanced MF when TE / phase / real / imag / trigger vary.
    let need_split = nt > 1
        && metas.len() >= 2
        && metas.iter().any(|m| {
            let m0 = &metas[0];
            (m.te - m0.te).abs() > 1e-6
                || m.is_phase != m0.is_phase
                || m.is_real != m0.is_real
                || m.is_imaginary != m0.is_imaginary
                || (m.trigger_delay - m0.trigger_delay).abs() > 1e-3
        });
    let grad_ids: Vec<usize> = if need_split {
        let mut ids = vec![0usize; nt];
        ids[0] = 1;
        let mut series = 1usize;
        for i in 1..nt {
            for j in 0..i {
                let a = &metas[i];
                let b = &metas[j];
                if (a.te - b.te).abs() < 1e-6
                    && a.is_phase == b.is_phase
                    && a.is_real == b.is_real
                    && a.is_imaginary == b.is_imaginary
                    && (a.trigger_delay - b.trigger_delay).abs() < 1e-3
                {
                    ids[i] = ids[j];
                    break;
                }
            }
            if ids[i] == 0 {
                series += 1;
                ids[i] = series;
            }
        }
        ids
    } else {
        vec![1; nt]
    };
    let max_series = grad_ids.iter().copied().max().unwrap_or(1);

    let mut all_paths = Vec::new();
    for s in 1..=max_series {
        let vol_idxs: Vec<usize> = grad_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| if id == s { Some(i) } else { None })
            .collect();
        if vol_idxs.is_empty() {
            continue;
        }
        let nt_s = vol_idxs.len();
        let mut d = d0.clone();
        if let Some(&vi) = vol_idxs.first() {
            let m = &metas[vi];
            if m.te > 0.0 {
                d.te = m.te;
            }
            d.echo_number = m.echo_num.max(1);
            d.trigger_delay_time = m.trigger_delay;
            d.is_has_phase = m.is_phase;
            d.is_has_real = m.is_real;
            d.is_has_imaginary = m.is_imaginary;
            d.b_value = m.b_value;
            d.diffusion_direction = m.direction;
        }
        if opts.guess_bids_filename {
            bids_guess::set_bids(&mut d, 1, opts.verbose);
        }

        let mut packed = Vec::with_capacity(nt_s * vol_bytes);
        for &vi in &vol_idxs {
            let start = vi * vol_bytes;
            if start + vol_bytes <= raw.len() {
                packed.extend_from_slice(&raw[start..start + vol_bytes]);
            }
        }
        let mut hdr = header_from_series(&d, nx, ny, nz, nt_s, xyz);
        let q = snap_mat44(&nifti_dicom2mat(d.orient, d.patient_position, xyz).lps_to_ras_f32());
        hdr.set_sform(&q);
        hdr.datatype = DT_FLOAT32;
        hdr.bitpix = 32;
        let mut packed_f32 = bytes_to_f32(&packed);
        if opts.flip_y {
            packed_f32 = flip_y_volume(packed_f32, nx, ny, nz, nt_s);
            let mut sform = Matrix4::from_rows([
                [
                    hdr.srow_x[0] as f64,
                    hdr.srow_x[1] as f64,
                    hdr.srow_x[2] as f64,
                    hdr.srow_x[3] as f64,
                ],
                [
                    hdr.srow_y[0] as f64,
                    hdr.srow_y[1] as f64,
                    hdr.srow_y[2] as f64,
                    hdr.srow_y[3] as f64,
                ],
                [
                    hdr.srow_z[0] as f64,
                    hdr.srow_z[1] as f64,
                    hdr.srow_z[2] as f64,
                    hdr.srow_z[3] as f64,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            apply_flip_y_sform(&mut sform, ny);
            hdr.set_sform(&sform);
        }
        if needs_extra_y_flip(d.epi_version_ge) {
            packed_f32 = flip_y_volume(packed_f32, nx, ny, nz, nt_s);
        }
        let out_bytes = f32_voxels_to_f32_bytes(&packed_f32);
        let mut paths = write_outputs(&d, &hdr, out_bytes, 1, opts)?;

        let series_dti: Vec<parrec::ParDtiVol> = vol_idxs
            .iter()
            .map(|&i| parrec::ParDtiVol {
                b_value: metas[i].b_value,
                direction: metas[i].direction,
            })
            .collect();
        if series_dti.len() >= 2
            && series_dti.iter().any(|v| {
                v.b_value > 0.0 || v.direction.iter().any(|&x| x.abs() > 1e-12)
            })
        {
            if let Some(stem) = paths.iter().find_map(|p| nii_stem_from_path(p)) {
                let vols: Vec<DicomImage> = series_dti
                    .iter()
                    .map(|v| {
                        let mut img = d.clone();
                        img.b_value = v.b_value;
                        img.diffusion_direction = v.direction;
                        img
                    })
                    .collect();
                let vol_refs: Vec<&DicomImage> = vols.iter().collect();
                save_dti_sidecars_ex(
                    &stem,
                    &vol_refs,
                    opts.bids == BidsMode::Yes || opts.bids == BidsMode::Only,
                    opts.flip_y,
                    opts.verbose,
                    opts.sort_dti_by_bval,
                )?;
                paths.push(stem.with_extension("bval"));
                paths.push(stem.with_extension("bvec"));
            }
        }
        all_paths.append(&mut paths);
    }
    Ok(all_paths)
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn convert_enhanced_multiframe(
    first: &DicomImage,
    opts: &DcmOpts,
    mmaps: &MmapCache,
) -> Result<Vec<PathBuf>> {
    let obj = open_prefetched(&first.path, mmaps)?;
    let mut frames = dcm_dicom::read_per_frame_geometry(&obj);
    // One open: geometry + pixels (large enhanced MF used to open twice).
    let (vol, rows, cols) = decode_opened_raw_f32(&first.path, &obj)?;
    let nf = first.number_of_frames.max(1) as usize;
    let frame_vox = rows * cols;
    if vol.len() < frame_vox * nf {
        return Err(Error::convert(format!(
            "{}: enhanced MF expected {} frames, got {} voxels",
            first.path.display(),
            nf,
            vol.len()
        )));
    }
    let mut vol = vol;
    vol.truncate(frame_vox * nf);
    if frames.len() == nf {
        dcm_dicom::sort_frames_by_dimension_index(&mut frames, &mut vol, frame_vox);
    }

    let (nz, nt) = if frames.len() >= 2 {
        dcm_dicom::infer_stack_dims(&frames, nf)
    } else {
        (1, nf)
    };

    let mut d0 = first.clone();
    if let Some(fg) = frames.first() {
        if fg.orient[1..].iter().any(|v| *v != 0.0) {
            d0.orient = fg.orient;
        }
        if fg.patient_position[1..].iter().any(|v| !v.is_nan()) {
            d0.patient_position = fg.patient_position;
        }
        if fg.te > 0.0 {
            d0.te = fg.te;
        }
        d0.is_has_phase = fg.is_phase;
        d0.is_has_real = fg.is_real;
        d0.is_has_imaginary = fg.is_imaginary;
        d0.trigger_delay_time = fg.trigger_delay;
        d0.inten_scale = fg.inten_scale;
        d0.inten_intercept = fg.inten_intercept;
    }
    let mut xyz = d0.xyz_mm;
    if frames.len() >= 2 && nz > 1 {
        let a = frames[0].patient_position;
        let b = frames[1].patient_position;
        let dist = ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2) + (a[3] - b[3]).powi(2)).sqrt();
        if dist > 0.0 {
            xyz[3] = dist;
        }
    }

    let contrasts = dcm_dicom::volume_contrasts(&frames, nz, nt);
    let need_split = dcm_dicom::scale_or_te_varies(&contrasts);
    let grad_ids = if need_split {
        if first.manufacturer == dcm_dicom::Manufacturer::Philips {
            eprintln!("Warning: Philips enhanced DICOMs (hint: export as classic DICOM)");
        }
        if opts.verbose > 0 {
            eprintln!("Parameters vary across 3D volumes packed in single DICOM file:");
            for (i, c) in contrasts.iter().enumerate() {
                eprintln!(
                    " {i} TE={} Slope={} Inter={} Phase={}",
                    c.te, c.inten_scale, c.inten_intercept, c.is_phase as i32
                );
            }
        }
        dcm_dicom::assign_grad_dyn_vol(&contrasts, first.asl_flags)
    } else {
        vec![1; nt]
    };
    let max_series = grad_ids.iter().copied().max().unwrap_or(1);

    let mut all_paths = Vec::new();
    for s in 1..=max_series {
        let vol_idxs: Vec<usize> = grad_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| if id == s { Some(i) } else { None })
            .collect();
        if vol_idxs.is_empty() {
            continue;
        }
        let nt_s = vol_idxs.len();
        let mut d = d0.clone();
        if let Some(&vi) = vol_idxs.first() {
            let c = &contrasts[vi];
            d.te = c.te;
            if c.tr > 0.0 {
                d.tr = c.tr;
            }
            d.is_has_phase = c.is_phase;
            d.is_has_real = c.is_real;
            d.is_has_imaginary = c.is_imaginary;
            d.trigger_delay_time = c.trigger_delay;
            d.echo_number = c.echo_num.max(1);
            d.inten_scale = c.inten_scale;
            d.inten_intercept = c.inten_intercept;
            if c.b_value >= 0.0 {
                d.b_value = c.b_value;
                d.diffusion_direction = c.diffusion_direction;
            }
        }
        if opts.guess_bids_filename {
            bids_guess::set_bids(&mut d, 1, opts.verbose);
        }

        let mut hdr = header_from_series(&d, cols, rows, nz, nt_s, xyz);
        let mut q = nifti_dicom2mat(d.orient, d.patient_position, xyz);
        if let Some(last) = frames.get(nz.saturating_sub(1)) {
            let mut last_img = d.clone();
            last_img.patient_position = last.patient_position;
            let _ = verify_slice_dir(&d, &last_img, nz, &mut q);
        }
        q = snap_mat44(&q.lps_to_ras_f32());
        hdr.set_sform(&q);
        philips::apply_philips_precise(&d, opts.philips_precise, &mut hdr, opts.verbose);

        // Extract matching volumes (volume-major after spatial stack).
        let vol_bytes = cols * rows * nz;
        let mut sub = Vec::with_capacity(vol_bytes * nt_s);
        for &vi in &vol_idxs {
            let start = vi * vol_bytes;
            let end = start + vol_bytes;
            if end <= vol.len() {
                sub.extend_from_slice(&vol[start..end]);
            }
        }
        let mut sub = sub;
        if opts.flip_y {
            sub = flip_y_volume(sub, cols, rows, nz, nt_s);
            let mut sform = Matrix4::from_rows([
                [
                    hdr.srow_x[0] as f64,
                    hdr.srow_x[1] as f64,
                    hdr.srow_x[2] as f64,
                    hdr.srow_x[3] as f64,
                ],
                [
                    hdr.srow_y[0] as f64,
                    hdr.srow_y[1] as f64,
                    hdr.srow_y[2] as f64,
                    hdr.srow_y[3] as f64,
                ],
                [
                    hdr.srow_z[0] as f64,
                    hdr.srow_z[1] as f64,
                    hdr.srow_z[2] as f64,
                    hdr.srow_z[3] as f64,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            apply_flip_y_sform(&mut sform, rows);
            hdr.set_sform(&sform);
        }
        if needs_extra_y_flip(d.epi_version_ge) {
            sub = flip_y_volume(sub, cols, rows, nz, nt_s);
        }
        let bytes = f32_voxels_to_f32_bytes(&sub);
        hdr.datatype = DT_FLOAT32;
        hdr.bitpix = 32;
        let mut paths = write_outputs(&d, &hdr, bytes, 1, opts)?;
        // DTI sidecars only for first (magnitude) series when diffusion present.
        if s == 1 && vol_idxs.iter().any(|&i| contrasts[i].b_value >= 0.0) {
            let volumes: Vec<DicomImage> = vol_idxs
                .iter()
                .map(|&i| {
                    let mut img = d.clone();
                    img.b_value = contrasts[i].b_value;
                    img.diffusion_direction = contrasts[i].diffusion_direction;
                    img
                })
                .collect();
            if let Some(stem) = paths.iter().find_map(|p| nii_stem_from_path(p)) {
                let vol_refs: Vec<&DicomImage> = volumes.iter().collect();
                let _ = save_dti_sidecars_ex(
                    &stem,
                    &vol_refs,
                    false,
                    opts.flip_y,
                    opts.verbose,
                    opts.sort_dti_by_bval,
                );
                paths.push(stem.with_extension("bval"));
                paths.push(stem.with_extension("bvec"));
            }
        }
        all_paths.append(&mut paths);
    }
    Ok(all_paths)
}

fn reorder_slices(slices: &mut Vec<SliceMeta>, order: &[usize]) {
    debug_assert_eq!(slices.len(), order.len());
    let mut pool: Vec<Option<SliceMeta>> = slices.drain(..).map(Some).collect();
    slices.reserve(order.len());
    for &i in order {
        slices.push(pool[i].take().expect("duplicate index in reorder_slices"));
    }
}

fn sort_slices_for_stack(slices: &mut Vec<SliceMeta>) {
    if slices.len() < 2 {
        return;
    }
    // PET: prefer FrameReferenceTime order when it varies (issue 577).
    if slices[0].img.modality == dcm_dicom::Modality::Pt {
        let frts: Vec<f64> = slices.iter().map(|s| s.img.frame_reference_time).collect();
        let varies = frts.iter().any(|&t| t >= 0.0)
            && frts
                .windows(2)
                .any(|w| w[0] >= 0.0 && w[1] >= 0.0 && (w[0] - w[1]).abs() > 1e-6);
        if varies {
            slices.sort_by(|a, b| {
                a.img
                    .frame_reference_time
                    .partial_cmp(&b.img.frame_reference_time)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.img.instance_number.cmp(&b.img.instance_number))
            });
            return;
        }
    }
    let n = slices.len();
    let mut by_instance: Vec<usize> = (0..n).collect();
    by_instance.sort_by_key(|&i| slices[i].img.instance_number);

    let mut by_depth: Vec<usize> = (0..n).collect();
    by_depth.sort_by(|&a, &b| {
        slices[a]
            .depth
            .partial_cmp(&slices[b].depth)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                slices[a]
                    .img
                    .instance_number
                    .cmp(&slices[b].img.instance_number),
            )
    });

    let reversed_instance: Vec<_> = by_instance
        .iter()
        .rev()
        .map(|&i| slices[i].img.instance_number)
        .collect();
    let depth_order: Vec<_> = by_depth
        .iter()
        .map(|&i| slices[i].img.instance_number)
        .collect();
    if depth_order != reversed_instance {
        reorder_slices(slices, &by_depth);
        return;
    }
    if slices[by_instance[0]].img.patient_position[1].is_nan() || by_instance.len() < 3 {
        reorder_slices(slices, &by_instance);
        return;
    }
    let dx0 = interslice_distance_signed(
        &slices[by_instance[0]].img,
        &slices[by_instance[1]].img,
    );
    let tol = dx0.abs() * 0.1 + 1e-4;
    let sequential = by_instance.windows(2).all(|w| {
        let dx = interslice_distance_signed(&slices[w[0]].img, &slices[w[1]].img);
        (dx - dx0).abs() <= tol
    });
    if sequential {
        reorder_slices(slices, &by_instance);
    } else {
        reorder_slices(slices, &by_depth);
    }
}

/// Issue 1009: warn when in-plane PixelSpacing differs across a series.
fn warn_pixel_spacing_varies(images: &[DicomImage]) {
    if images.len() < 2 {
        return;
    }
    let x0 = images[0].xyz_mm[1];
    let y0 = images[0].xyz_mm[2];
    if !x0.is_finite() || !y0.is_finite() || x0 <= 0.0 || y0 <= 0.0 {
        return;
    }
    for img in &images[1..] {
        let x = img.xyz_mm[1];
        let y = img.xyz_mm[2];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if (x - x0).abs() > 1e-5 || (y - y0).abs() > 1e-5 {
            eprintln!(
                " PixelSpacing (0028,0030) varies {y0}×{x0} != {y}×{x} (issue 1009)"
            );
            return;
        }
    }
}

fn convert_series(
    images: Vec<DicomImage>,
    opts: &DcmOpts,
    mmaps: &MmapCache,
) -> Result<Vec<PathBuf>> {
    warn_pixel_spacing_varies(&images);
    if images[0].is_xa_physio || images[0].is_cmrr_physio {
        let stem = create_filename(&images[0], opts)?;
        return physio::convert_physio(&images[0], &stem);
    }
    if mrs::is_mrs_series(&images) {
        return mrs::convert_mrs(images, opts, mmaps);
    }
    if images.len() == 1 && images[0].number_of_frames > 1 {
        return convert_enhanced_multiframe(&images[0], opts, mmaps);
    }
    let mosaic_slices = images[0].csa.image.mosaic_slices;
    if mosaic_slices > 1 || images[0].is_mosaic {
        return convert_mosaic_series(images, opts, mosaic_slices.max(1), mmaps);
    }
    // Keep one clone of the input-order first image (C++ header path); move the
    // rest into SliceMeta without cloning every file in the series.
    let first = images[0].clone();
    let mut slices: Vec<SliceMeta> = images
        .into_iter()
        .map(|img| {
            let n = slice_normal(&img.orient);
            let p = [
                img.patient_position[1],
                img.patient_position[2],
                img.patient_position[3],
            ];
            let depth = if p.iter().any(|v| v.is_nan()) {
                img.instance_number as f64
            } else {
                dot(p, n)
            };
            SliceMeta { img, depth }
        })
        .collect();
    sort_slices_for_stack(&mut slices);

    let mut unique_depths: Vec<f64> = Vec::new();
    for s in &slices {
        if unique_depths.iter().all(|d| (s.depth - d).abs() > 1e-4) {
            unique_depths.push(s.depth);
        }
    }
    unique_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nz_pos = unique_depths.len().max(1);
    let n = slices.len();
    let (nz, nt) = if n % nz_pos == 0 {
        (nz_pos, n / nz_pos)
    } else {
        (n, 1)
    };
    // 4D EPI: C++ `dcmSort` is instance-major (vol0 slices, then vol1, …).
    if nt > 1 {
        slices.sort_by_key(|s| s.img.instance_number);
    }

    let mut xyz_mm = first.xyz_mm;
    let orig_dz = xyz_mm[3] as f32;
    // C++ 3D path (`dim[4] < 2`): `xyzMM[3] = intersliceDistance(d0, d1)`;
    // pixdim[3] stays the DICOM spacing when `isSameFloatGE(dx, pixdim[3])`.
    let mut measured_dx = 0.0f32;
    if nz > 1 && nt == 1 && slices.len() >= 2 {
        measured_dx = interslice_distance(&slices[0].img, &slices[1].img);
        if measured_dx > 0.0 {
            xyz_mm[3] = measured_dx as f64;
        }
    } else if nz > 1 && unique_depths.len() > 1 {
        let span = (unique_depths.last().unwrap() - unique_depths.first().unwrap()).abs();
        xyz_mm[3] = span / (nz as f64 - 1.0);
    }

    let nx = first.columns;
    let ny = first.rows;
    let mut hdr = header_from_series(&first, nx, ny, nz, nt, xyz_mm);
    if nt == 1 && measured_dx > 0.0 && is_same_float_ge(measured_dx, orig_dz) {
        hdr.pixdim[3] = orig_dz;
    }

    // C++ `headerDcm2Nii2(d0, d1, …)` passes the 2nd sorted slice as `d2`.
    let verify_other = if slices.len() >= 2 {
        &slices[1].img
    } else {
        &slices.last().unwrap().img
    };

    let mut q = nifti_dicom2mat(first.orient, first.patient_position, xyz_mm);
    let mut need_flip_z = false;
    if nz > 1 {
        need_flip_z = verify_slice_dir(&first, verify_other, nz, &mut q);
    }
    q = snap_mat44(&q.lps_to_ras_f32());
    hdr.set_sform(&q);
    philips::apply_philips_precise(&first, opts.philips_precise, &mut hdr, opts.verbose);

    let slice_paths: Vec<&Path> = slices.iter().map(|s| s.img.path.as_path()).collect();
    let mut vol = crate::voxels::decode_stack_slices(&slice_paths, mmaps, nx, ny)?;
    if vol.len() != nx * ny * nz * nt {
        // Incomplete 4D: keep what we have as 3D of n slices.
        let n_have = slices.len();
        hdr.dim[3] = n_have as i16;
        hdr.dim[4] = 1;
        hdr.dim[0] = 3;
    }

    let mut sform = Matrix4::from_rows([
        [
            hdr.srow_x[0] as f64,
            hdr.srow_x[1] as f64,
            hdr.srow_x[2] as f64,
            hdr.srow_x[3] as f64,
        ],
        [
            hdr.srow_y[0] as f64,
            hdr.srow_y[1] as f64,
            hdr.srow_y[2] as f64,
            hdr.srow_y[3] as f64,
        ],
        [
            hdr.srow_z[0] as f64,
            hdr.srow_z[1] as f64,
            hdr.srow_z[2] as f64,
            hdr.srow_z[3] as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let nz_out = hdr.dim[3] as usize;
    let nt_out = hdr.dim[4].max(1) as usize;
    let use_ortho =
        opts.rotate_3d && first.is_3d_acq && !first.is_epi && nz_out > 1 && hdr.dim[0] < 4;
    let do_flip_z = need_flip_z && nz_out > 1;
    let do_flip_y = opts.flip_y && !use_ortho;
    let pepolar_y = needs_extra_y_flip(first.epi_version_ge);
    if do_flip_z || do_flip_y {
        vol = flip_yz_volume(vol, nx, ny, nz_out, nt_out, do_flip_y, do_flip_z);
        if do_flip_z {
            apply_flip_z_sform(&mut sform, nz_out);
        }
        if do_flip_y {
            apply_flip_y_sform(&mut sform, hdr.dim[2] as usize);
        }
        hdr.set_sform(&sform);
    }
    // Pepolar reverse volumes: image-space Y flip only (C++ `nii_flipImgY`).
    if pepolar_y {
        vol = flip_y_volume(vol, nx, ny, nz_out, nt_out);
    }
    if use_ortho {
        vol = nii_set_ortho_f32(vol, &mut hdr);
    }

    hdr.set_descrip(&nifti_descrip(&first));
    if !first.image_comments.is_empty() {
        hdr.set_aux(&first.image_comments);
    }
    hdr.slice_code = first.csa.image.slice_order;
    if first.phase_encoding_rc != ' ' {
        if first.phase_encoding_rc == 'R' {
            hdr.dim_info = (3 << 4) + (1 << 2) + 2;
        } else if first.phase_encoding_rc == 'C' {
            hdr.dim_info = (3 << 4) + (2 << 2) + 1;
        }
    }

    let slice_refs: Vec<&DicomImage> = slices.iter().map(|s| &s.img).collect();
    let (mut bytes, datatype, bitpix) = pack_voxels(&vol, &first);
    hdr.datatype = datatype;
    hdr.bitpix = bitpix;
    hdr.vox_offset = 352.0;
    apply_intensity_variance_and_rgb(
        &mut hdr,
        &mut bytes,
        &vol,
        &slice_refs,
        &first,
        nx,
        ny,
        opts,
    );

    // CT / X-ray gantry tilt: refine angle, fix sform shear, optionally emit `_Tilt`.
    let mut gantry = first.gantry_tilt;
    if (first.modality == Modality::Ct || first.is_xray || gantry > 0.0)
        && slices.len() >= 2
        && (nz_out > 1 || nt_out == 1)
    {
        let last = &slices.last().unwrap().img;
        let est = compute_gantry_tilt_precise(&first, last, opts.verbose);
        if !est.is_nan() {
            gantry = est;
        }
    }
    let mut bytes = bytes;
    let mut tilt_extra: Option<(PathBuf, Nifti1Header, Vec<u8>)> = None;
    if gantry.abs() > 1e-4 {
        eprintln!(
            "Note these images have gantry tilt of {gantry} degrees (manufacturer ID = {:?})",
            first.manufacturer
        );
        apply_gantry_tilt_sform(&mut hdr, gantry);
        if opts.tilt_correct {
            if let Some((th, tv)) = correct_tilt(&hdr, &bytes, &first, gantry) {
                tilt_extra = Some((
                    PathBuf::from(format!(
                        "{}_Tilt",
                        create_filename(&first, opts)?.display()
                    )),
                    th,
                    tv,
                ));
            }
        } else {
            eprintln!("Tilt correction skipped");
        }
    }

    let mut bids_dcm = slices
        .first()
        .map(|s| s.img.clone())
        .unwrap_or_else(|| first.clone());
    bids_dcm.gantry_tilt = gantry;
    if opts.diff_cycling_mode_ge >= 0 {
        bids_dcm.diff_cycling_mode_ge = opts.diff_cycling_mode_ge;
        bids_dcm.diff_cycling_mode_ge_override = true;
    }
    // GE diffusion cycling modes that invalidate computed slice timing
    // are handled inside `ge_rescue_slice_timing_ms`.
    if bids_dcm.manufacturer == dcm_dicom::Manufacturer::Ge && nt_out > 1 {
        bids_dcm.csa.image.slice_timing_ms = ge_rescue_slice_timing_ms(
            &bids_dcm.series_description,
            nz_out,
            bids_dcm.tr,
            need_flip_z,
            bids_dcm.ge_slice_order,
            bids_dcm.csa.image.multi_band_factor,
            bids_dcm.group_delay,
            &bids_dcm.software_versions,
            bids_dcm.epi_version_ge,
            bids_dcm.internal_epi_version_ge,
            &bids_dcm.ge_iopt,
            bids_dcm.diff_cycling_mode_ge,
        );
    }
    if opts.test_x0021x105e
        && bids_dcm.manufacturer == dcm_dicom::Manufacturer::Ge
        && nz_out > 1
    {
        test_ge_rtia_slice_timing(&slices, &bids_dcm, nz_out);
    }

    // Volume representatives for DTI / PET (one meta per 4D volume).
    let vol_reps = volume_representatives(&slices, nt_out, nz_out);
    onset::fill_volume_onset_times(&mut bids_dcm, &vol_reps, opts);
    let stacked_from_2d = !first.is_mosaic
        && first.number_of_frames <= 1
        && slices.len() > nt_out
        && nz_out > 1;
    tr_discrepancy::apply_issue_560_tr(&mut bids_dcm, &mut hdr, &vol_reps, opts, stacked_from_2d);
    epi_tr::apply_3d_epi_volume_tr(&mut bids_dcm, &mut hdr, &vol_reps);
    if opts.guess_bids_filename {
        bids_guess::set_bids(&mut bids_dcm, n, opts.verbose);
    }

    // DTI: reorder ADC to end, save `_ADC` companion, truncate main.
    let mut adc_paths = Vec::new();
    let mut dti_plan = None;
    if volumes_have_dti(&vol_reps) && nt_out > 1 {
        let plan = plan_dti_volumes(&vol_reps, opts.sort_dti_by_bval);
        let bp = (hdr.bitpix as usize / 8).max(1);
        let vol_bytes = nx * ny * nz_out * bp;
        if plan.order.iter().enumerate().any(|(i, &o)| i != o) || plan.num_adc > 0 {
            bytes = reorder_volume_bytes(&bytes, vol_bytes, &plan.order);
        }
        if plan.num_adc > 0 && !opts.ignore_derived {
            adc_paths = write_dti_adc_companion(&first, opts, &hdr, &bytes, n)?;
            remove_trailing_adc(&mut hdr, &mut bytes, plan.num_adc);
        } else if plan.num_adc > 0 && opts.ignore_derived {
            eprintln!(
                "Ignoring derived diffusion image(s). Better isotropic and ADC maps can be generated later processing."
            );
            remove_trailing_adc(&mut hdr, &mut bytes, plan.num_adc);
        }
        dti_plan = Some(plan);
    }

    // Variable inter-slice distance → `_Eq` equidistant companion (3D only).
    // Computed before write consumes `bytes`.
    let mut eq_extra: Option<(PathBuf, Nifti1Header, Vec<u8>)> = None;
    if nt_out == 1 && nz_out > 2 && slices.len() >= 3 {
        let dx0 = interslice_distance(&slices[0].img, &slices[1].img);
        let mut varies = false;
        for w in slices.windows(2) {
            let dx = interslice_distance(&w[0].img, &w[1].img);
            if (dx - dx0).abs() > 0.0001 {
                varies = true;
                break;
            }
        }
        if varies {
            eprintln!(
                "Warning: Interslice distance varies in this volume (incompatible with NIfTI format)."
            );
            let mut slice_mm = vec![0.0f32; slices.len()];
            for i in 1..slices.len() {
                slice_mm[i] = interslice_distance(&slices[0].img, &slices[i].img);
            }
            if let Some((eh, ev)) = slice_eq::equalize_slices(&hdr, &bytes, &slice_mm)? {
                let eq_stem = PathBuf::from(format!(
                    "{}_Eq",
                    create_filename(&first, opts)?.display()
                ));
                eq_extra = Some((eq_stem, eh, ev));
            }
        }
    }

    let mut paths = write_outputs_vols(&bids_dcm, &hdr, bytes, n, opts, Some(&vol_reps))?;
    paths.append(&mut adc_paths);
    {
        let mut rois = overlay::write_overlay_rois(
            &slice_refs,
            &hdr,
            need_flip_z,
            use_ortho,
            opts,
        )?;
        paths.append(&mut rois);
    }
    if let Some((tilt_stem, th, tv)) = tilt_extra {
        let mut tilt_paths = write_volume_files(&first, &tilt_stem, &th, &tv, n, opts, None)?;
        paths.append(&mut tilt_paths);
    }
    if let Some((eq_stem, eh, ev)) = eq_extra {
        let mut eq_paths = write_volume_files(&first, &eq_stem, &eh, &ev, n, opts, None)?;
        paths.append(&mut eq_paths);
    }
    if volumes_have_dti(&vol_reps) {
        if let Some(stem) = paths.iter().find_map(|p| nii_stem_from_path(p)) {
            let vols: Vec<&DicomImage> = if let Some(plan) = dti_plan {
                let keep = vol_reps.len().saturating_sub(plan.num_adc);
                plan.order[..keep].iter().map(|&i| vol_reps[i]).collect()
            } else {
                vol_reps
            };
            save_dti_sidecars_ex(
                &stem,
                &vols,
                opts.bids == BidsMode::Yes || opts.bids == BidsMode::Only,
                opts.flip_y,
                opts.verbose,
                false, // already ordered
            )?;
            paths.push(stem.with_extension("bval"));
            paths.push(stem.with_extension("bvec"));
        }
    }
    Ok(paths)
}

fn convert_mosaic_series(
    mut images: Vec<DicomImage>,
    opts: &DcmOpts,
    mosaic_slices: i32,
    mmaps: &MmapCache,
) -> Result<Vec<PathBuf>> {
    let is_uih = images[0].manufacturer == dcm_dicom::Manufacturer::Uih;
    images.sort_by_key(|i| i.instance_number);
    if images.len() >= 2 {
        let (head, tail) = images.split_at_mut(1);
        check_slice_timing(&mut head[0], &tail[0]);
    }
    let image_refs: Vec<&DicomImage> = images.iter().collect();
    let first = &images[0];
    let decoded: Result<Vec<_>> = images
        .par_iter()
        .enumerate()
        .map(|(i, img)| {
            let (pix, r, c) = decode_pixels_prefetched(&img.path, mmaps)?;
            let ms = if img.csa.image.mosaic_slices > 1 {
                img.csa.image.mosaic_slices
            } else {
                mosaic_slices
            };
            let (dem, dc, dr) = demosaic_f32(&pix, c, r, ms, is_uih);
            Ok((dem, dc, dr, ms as usize, i))
        })
        .collect();
    let mut vols: Vec<Vec<f32>> = Vec::new();
    let mut nx = 0usize;
    let mut ny = 0usize;
    let mut nz = mosaic_slices as usize;
    for (dem, dc, dr, ms, i) in decoded? {
        if nx == 0 {
            nx = dc;
            ny = dr;
            nz = ms;
        }
        if dc != nx || dr != ny {
            return Err(Error::convert(format!(
                "{}: mosaic size mismatch",
                images[i].path.display()
            )));
        }
        vols.push(dem);
    }

    let nt = vols.len();
    let slice_vol = nx * ny * nz;
    let mut vol = if vols.iter().all(|v| v.len() == slice_vol) && !vols.is_empty() {
        crate::voxels::pack_slices(&vols, slice_vol)
    } else {
        let mut vol: Vec<f32> = Vec::with_capacity(slice_vol * nt);
        for v in &vols {
            vol.extend_from_slice(v);
        }
        vol
    };

    let mut xyz_mm = first.xyz_mm;
    if first.spacing_between_slices > 0.0 {
        xyz_mm[3] = first.spacing_between_slices;
    }

    let mut hdr = header_from_series(first, nx, ny, nz, nt, xyz_mm);
    let mut q = nifti_dicom2mat(first.orient, first.patient_position, xyz_mm);
    if first.manufacturer == dcm_dicom::Manufacturer::Siemens && mosaic_slices > 1 {
        apply_siemens_mosaic_sform(&mut q, first, first.columns, first.rows, mosaic_slices);
    } else {
        if mosaic_slices > 1 && !first.patient_position[1].is_nan() {
            let other = images.get(1).unwrap_or(first);
            let _flip = verify_slice_dir(first, other, nz, &mut q);
        }
        q = snap_mat44(&q.lps_to_ras_f32());
    }
    hdr.set_sform(&q);
    hdr.set_descrip(&nifti_descrip(first));
    if !first.image_comments.is_empty() {
        hdr.set_aux(&first.image_comments);
    }
    hdr.slice_code = first.csa.image.slice_order;
    if first.phase_encoding_rc == 'R' {
        hdr.dim_info = (3 << 4) + (1 << 2) + 2;
    } else if first.phase_encoding_rc == 'C' {
        hdr.dim_info = (3 << 4) + (2 << 2) + 1;
    }

    if opts.flip_y {
        vol = flip_y_volume(vol, nx, ny, nz, nt);
        let mut sform = Matrix4::from_rows([
            [
                hdr.srow_x[0] as f64,
                hdr.srow_x[1] as f64,
                hdr.srow_x[2] as f64,
                hdr.srow_x[3] as f64,
            ],
            [
                hdr.srow_y[0] as f64,
                hdr.srow_y[1] as f64,
                hdr.srow_y[2] as f64,
                hdr.srow_y[3] as f64,
            ],
            [
                hdr.srow_z[0] as f64,
                hdr.srow_z[1] as f64,
                hdr.srow_z[2] as f64,
                hdr.srow_z[3] as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        apply_flip_y_sform(&mut sform, ny);
        hdr.set_sform(&sform);
    }
    if needs_extra_y_flip(first.epi_version_ge) {
        vol = flip_y_volume(vol, nx, ny, nz, nt);
    }

    let (mut bytes, datatype, bitpix) = pack_voxels(&vol, first);
    hdr.datatype = datatype;
    hdr.bitpix = bitpix;
    hdr.vox_offset = 352.0;
    apply_intensity_variance_and_rgb(
        &mut hdr,
        &mut bytes,
        &vol,
        &image_refs,
        first,
        nx,
        ny,
        opts,
    );

    let mut bids_dcm = images[0].clone();
    onset::fill_volume_onset_times(&mut bids_dcm, &image_refs, opts);
    // Enhanced multi-frame path: each input is already a volume (C++ skips
    // issue 560 when hdr0.dim[3] >= 2).
    tr_discrepancy::apply_issue_560_tr(&mut bids_dcm, &mut hdr, &image_refs, opts, false);
    epi_tr::apply_3d_epi_volume_tr(&mut bids_dcm, &mut hdr, &image_refs);
    if opts.guess_bids_filename {
        bids_guess::set_bids(&mut bids_dcm, images.len(), opts.verbose);
    }

    let mut adc_paths = Vec::new();
    let mut dti_plan = None;
    if volumes_have_dti(&image_refs) && nt > 1 {
        let plan = plan_dti_volumes(&image_refs, opts.sort_dti_by_bval);
        let bp = (hdr.bitpix as usize / 8).max(1);
        let vol_bytes = nx * ny * nz * bp;
        if plan.order.iter().enumerate().any(|(i, &o)| i != o) || plan.num_adc > 0 {
            bytes = reorder_volume_bytes(&bytes, vol_bytes, &plan.order);
        }
        if plan.num_adc > 0 && !opts.ignore_derived {
            adc_paths = write_dti_adc_companion(first, opts, &hdr, &bytes, images.len())?;
            remove_trailing_adc(&mut hdr, &mut bytes, plan.num_adc);
        } else if plan.num_adc > 0 && opts.ignore_derived {
            eprintln!(
                "Ignoring derived diffusion image(s). Better isotropic and ADC maps can be generated later processing."
            );
            remove_trailing_adc(&mut hdr, &mut bytes, plan.num_adc);
        }
        dti_plan = Some(plan);
    }

    let mut paths = write_outputs(&bids_dcm, &hdr, bytes, images.len(), opts)?;
    paths.append(&mut adc_paths);
    if volumes_have_dti(&image_refs) {
        if let Some(stem) = paths.iter().find_map(|p| nii_stem_from_path(p)) {
            let vols: Vec<&DicomImage> = if let Some(plan) = dti_plan {
                let keep = image_refs.len().saturating_sub(plan.num_adc);
                plan.order[..keep].iter().map(|&i| image_refs[i]).collect()
            } else {
                image_refs
            };
            save_dti_sidecars_ex(
                &stem,
                &vols,
                opts.bids == BidsMode::Yes || opts.bids == BidsMode::Only,
                opts.flip_y,
                opts.verbose,
                false,
            )?;
            paths.push(stem.with_extension("bval"));
            paths.push(stem.with_extension("bvec"));
        }
    }
    Ok(paths)
}

fn volumes_have_dti(volumes: &[&DicomImage]) -> bool {
    volumes.len() >= 2
        && volumes.iter().all(|v| v.b_value >= 0.0)
        && volumes
            .iter()
            .any(|v| v.b_value > 0.0 || v.diffusion_direction.iter().any(|&x| x.abs() > 1e-12))
}

/// Write `_ADC` companion before truncating ADC volumes from the main stack.
fn write_dti_adc_companion(
    first: &DicomImage,
    opts: &DcmOpts,
    hdr: &Nifti1Header,
    bytes: &[u8],
    n_dicom: usize,
) -> Result<Vec<PathBuf>> {
    let adc_stem = PathBuf::from(format!(
        "{}_ADC",
        create_filename(first, opts)?.display()
    ));
    write_volume_files(first, &adc_stem, hdr, bytes, n_dicom, opts, None)
}

/// One `DicomImage` per 4D volume (or per slice when 3D) — references only.
fn volume_representatives<'a>(
    slices: &'a [SliceMeta],
    nt_out: usize,
    nz_out: usize,
) -> Vec<&'a DicomImage> {
    let n = slices.len();
    if nt_out <= 1 || n == nt_out {
        slices.iter().map(|s| &s.img).collect()
    } else if n >= nz_out * nt_out {
        (0..nt_out).map(|t| &slices[t * nz_out].img).collect()
    } else {
        slices.iter().map(|s| &s.img).collect()
    }
}

fn nii_stem_from_path(path: &Path) -> Option<PathBuf> {
    let s = path.to_string_lossy();
    if let Some(st) = s.strip_suffix(".nii.gz") {
        Some(PathBuf::from(st))
    } else if let Some(st) = s.strip_suffix(".nii.zst") {
        Some(PathBuf::from(st))
    } else if s.ends_with(".nii") {
        Some(path.with_extension(""))
    } else {
        None
    }
}

fn write_outputs(
    first: &DicomImage,
    hdr: &Nifti1Header,
    bytes: Vec<u8>,
    n_dicom: usize,
    opts: &DcmOpts,
) -> Result<Vec<PathBuf>> {
    write_outputs_vols(first, hdr, bytes, n_dicom, opts, None)
}

fn write_outputs_vols(
    first: &DicomImage,
    hdr: &Nifti1Header,
    mut bytes: Vec<u8>,
    n_dicom: usize,
    opts: &DcmOpts,
    volumes: Option<&[&DicomImage]>,
) -> Result<Vec<PathBuf>> {
    let stem = create_filename(first, opts)?;
    let mut hdr = *hdr;
    maximize_16bit(&mut hdr, &mut bytes, opts.maximize_16bit, opts.verbose);

    if opts.crop {
        if let Some((ch, cv)) = crop::try_crop(&hdr, &bytes)? {
            let crop_stem = PathBuf::from(format!("{}_Crop", stem.display()));
            let mut crop_paths =
                write_volume_files(first, &crop_stem, &ch, &cv, n_dicom, opts, volumes)?;
            let mut main = write_volume_files(first, &stem, &hdr, &bytes, n_dicom, opts, volumes)?;
            main.append(&mut crop_paths);
            if opts.create_text {
                main.push(text::save_text(&stem, first, &hdr, &first.path)?);
            }
            return Ok(main);
        }
    }

    let mut written = write_volume_files(first, &stem, &hdr, &bytes, n_dicom, opts, volumes)?;
    if opts.create_text {
        written.push(text::save_text(&stem, first, &hdr, &first.path)?);
    }
    Ok(written)
}

fn write_volume_files(
    first: &DicomImage,
    stem: &Path,
    hdr: &Nifti1Header,
    bytes: &[u8],
    n_dicom: usize,
    opts: &DcmOpts,
    volumes: Option<&[&DicomImage]>,
) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    if opts.bids == BidsMode::Only {
        let json_path = unique_path(stem, "json", opts.name_conflict)?;
        let anon = match opts.anonymize {
            opts::AnonymizeBids::Yes => Anonymize::Full,
            opts::AnonymizeBids::No => Anonymize::None,
            opts::AnonymizeBids::PiiOnly => Anonymize::PiiOnly,
        };
        write_sidecar_ex(
            &json_path,
            first,
            hdr,
            anon,
            dcm_core::VERSION_DATE,
            opts.philips_precise,
            &opts.image_comments,
        )?;
        written.push(json_path);
        return Ok(written);
    }

    // Save3D: split 4D into separate 3D volumes.
    if opts.compress == Compress::Save3d && hdr.dim[4] > 1 {
        let nx = hdr.dim[1] as usize;
        let ny = hdr.dim[2] as usize;
        let nz = hdr.dim[3] as usize;
        let nt = hdr.dim[4] as usize;
        let bp = (hdr.bitpix as usize / 8).max(1);
        let vol_bytes = nx * ny * nz * bp;
        for t in 0..nt {
            let mut h3 = *hdr;
            h3.dim[0] = 3;
            h3.dim[4] = 1;
            let slice = &bytes[t * vol_bytes..(t + 1) * vol_bytes];
            let stem_t = PathBuf::from(format!("{}_{:04}", stem.display(), t + 1));
            let one = volumes.and_then(|v| v.get(t)).map(std::slice::from_ref);
            written.extend(write_single_format(
                first, &stem_t, &h3, slice, n_dicom, opts, one,
            )?);
        }
        return Ok(written);
    }

    written.extend(write_single_format(
        first, stem, hdr, bytes, n_dicom, opts, volumes,
    )?);
    Ok(written)
}

fn write_single_format(
    first: &DicomImage,
    stem: &Path,
    hdr: &Nifti1Header,
    bytes: &[u8],
    n_dicom: usize,
    opts: &DcmOpts,
    volumes: Option<&[&DicomImage]>,
) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    let mut hdr = *hdr;
    // Only copy when we must byte-swap; native endian writes from the slice.
    let swapped;
    let bytes: &[u8] = if !opts.save_native_endian {
        let mut owned = bytes.to_vec();
        swap_nifti_endian(&mut hdr, &mut owned);
        swapped = owned;
        &swapped
    } else {
        bytes
    };

    match opts.save_format {
        SaveFormat::Nrrd => {
            let path = write_nrrd(stem, &hdr, bytes, first, volumes, opts.compress)?;
            println!(
                "Convert {} DICOM as {} ({})",
                n_dicom,
                path.display(),
                dims(&hdr)
            );
            written.push(path);
        }
        SaveFormat::Mgh => {
            let path = write_mgh(stem, &hdr, bytes, first, opts.compress)?;
            println!(
                "Convert {} DICOM as {} ({})",
                n_dicom,
                path.display(),
                dims(&hdr)
            );
            written.push(path);
        }
        SaveFormat::Jnii | SaveFormat::Bnii => {
            let path = write_jnifti(
                stem,
                &hdr,
                bytes,
                matches!(opts.save_format, SaveFormat::Bnii),
                opts.compress,
            )?;
            println!(
                "Convert {} DICOM as {} ({})",
                n_dicom,
                path.display(),
                dims(&hdr)
            );
            written.push(path);
        }
        SaveFormat::Nifti => {
            let ext = match opts.compress {
                Compress::None | Compress::Save3d => "nii",
                Compress::Gz | Compress::InternalGz => "nii.gz",
                Compress::Zstd => "nii.zst",
            };
            let nii_path = unique_path(stem, ext, opts.name_conflict)?;
            let write_bids = opts.bids == BidsMode::Yes;
            let json_path = sidecar_path(&nii_path);
            let anon = match opts.anonymize {
                opts::AnonymizeBids::Yes => Anonymize::Full,
                opts::AnonymizeBids::No => Anonymize::None,
                opts::AnonymizeBids::PiiOnly => Anonymize::PiiOnly,
            };

            // Overlap BIDS JSON with NIfTI compress/write (JSON is CPU+small I/O).
            let mut bids_err: Option<dcm_core::error::Error> = None;
            std::thread::scope(|scope| {
                let bids_handle = write_bids.then(|| {
                    let jp = json_path.clone();
                    scope.spawn(move || {
                        write_sidecar_ex(
                            &jp,
                            first,
                            &hdr,
                            anon,
                            dcm_core::VERSION_DATE,
                            opts.philips_precise,
                            &opts.image_comments,
                        )
                    })
                });

                let nifti_res = match opts.compress {
                    Compress::None | Compress::Save3d => {
                        dcm_nifti::write_nii(&nii_path, &hdr, bytes)
                    }
                    Compress::InternalGz => {
                        dcm_nifti::write_nii_gz(&nii_path, &hdr, bytes, opts.gz_level as u32)
                    }
                    Compress::Gz => {
                        if !opts.pigz_path.is_empty() && opts.piped_gz {
                            pigz::write_nii_via_pigz_pipe(
                                &nii_path,
                                &hdr,
                                bytes,
                                Path::new(&opts.pigz_path),
                                opts.gz_level,
                                opts.verbose,
                            )
                        } else if !opts.pigz_path.is_empty() {
                            let raw = PathBuf::from(
                                nii_path
                                    .to_string_lossy()
                                    .strip_suffix(".gz")
                                    .unwrap_or(nii_path.to_string_lossy().as_ref()),
                            );
                            dcm_nifti::write_nii(&raw, &hdr, bytes).and_then(|_| {
                                pigz::pigz_file(
                                    &raw,
                                    Path::new(&opts.pigz_path),
                                    opts.gz_level,
                                    bytes.len(),
                                    opts.verbose,
                                )
                            })
                        } else {
                            dcm_nifti::write_nii_gz(
                                &nii_path,
                                &hdr,
                                bytes,
                                opts.gz_level as u32,
                            )
                        }
                    }
                    Compress::Zstd => {
                        dcm_nifti::write_nii_zst(&nii_path, &hdr, bytes, opts.gz_level)
                    }
                };

                if let Some(h) = bids_handle {
                    match h.join() {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => bids_err = Some(e),
                        Err(_) => {
                            bids_err = Some(Error::convert("BIDS sidecar thread panicked"));
                        }
                    }
                }
                nifti_res
            })?;
            if let Some(e) = bids_err {
                return Err(e);
            }

            println!(
                "Convert {} DICOM as {} ({})",
                n_dicom,
                nii_path.display(),
                dims(&hdr)
            );
            written.push(nii_path);
            if write_bids {
                written.push(json_path);
            }
        }
    }
    Ok(written)
}

fn swap_nifti_endian(hdr: &mut Nifti1Header, voxels: &mut [u8]) {
    // Mark opposite endian via sizeof_hdr convention used by nifti tools.
    hdr.sizeof_hdr = hdr.sizeof_hdr.swap_bytes();
    let step = match hdr.datatype {
        DT_INT16 | DT_UINT16 => 2,
        DT_FLOAT32 | DT_INT32 => 4,
        _ => 0,
    };
    if step > 1 {
        for chunk in voxels.chunks_exact_mut(step) {
            chunk.reverse();
        }
    }
}

fn sidecar_path(nii: &Path) -> PathBuf {
    let s = nii.to_string_lossy();
    if let Some(stem) = s.strip_suffix(".nii.gz") {
        PathBuf::from(format!("{stem}.json"))
    } else if let Some(stem) = s.strip_suffix(".nii.zst") {
        PathBuf::from(format!("{stem}.json"))
    } else {
        nii.with_extension("json")
    }
}

fn dims(h: &Nifti1Header) -> String {
    if h.dim[4] > 1 {
        format!("{}x{}x{}x{}", h.dim[1], h.dim[2], h.dim[3], h.dim[4])
    } else {
        format!("{}x{}x{}", h.dim[1], h.dim[2], h.dim[3])
    }
}

/// Undocumented `-j y`: compare CSA/calculated slice timing to GE RTIA `(0021,105E)`.
fn test_ge_rtia_slice_timing(slices: &[SliceMeta], d0: &DicomImage, nz: usize) {
    if slices.len() < nz * 2 {
        // Need at least volume-2 slice timings from RTIA (C++ uses vol index + nz).
        if d0.rtia_timer_ge <= 0.0 {
            println!("DICOM images do not report RTIA timer(0021,105E)");
        }
        return;
    }
    let mut rtia: Vec<f64> = Vec::with_capacity(nz);
    let mut mn = f64::INFINITY;
    for v in 0..nz {
        let t = slices[v + nz].img.rtia_timer_ge;
        if t < mn {
            mn = t;
        }
        rtia.push(t);
    }
    if mn < 0.0 || !mn.is_finite() {
        return;
    }
    if rtia.iter().all(|&t| t <= 0.0) {
        println!("DICOM images do not report RTIA timer(0021,105E)");
        return;
    }
    let calc = &d0.csa.image.slice_timing_ms;
    let mut mx_err = 0.0f64;
    let mut adj = Vec::with_capacity(nz);
    for v in 0..nz {
        let ms = (rtia[v] - mn) * 1000.0;
        adj.push(ms);
        if v < calc.len() {
            mx_err = mx_err.max((ms - calc[v]).abs());
        }
    }
    println!("Slice Timing Error between calculated and RTIA timer(0021,105E): {mx_err}ms");
    if mx_err < 1.0 {
        return;
    }
    for v in 0..nz {
        let c = calc.get(v).copied().unwrap_or(0.0);
        println!("\t{c}\t{}", adj[v]);
    }
}

fn pack_voxels(vol: &[f32], first: &DicomImage) -> (Vec<u8>, i16, i16) {
    if first.samples_per_pixel == 3 {
        let mut out = Vec::with_capacity(vol.len());
        for &v in vol {
            out.push(v.round().clamp(0.0, 255.0) as u8);
        }
        return (out, dcm_nifti::DT_RGB24, 24);
    }
    if first.is_float || first.bits_allocated == 32 {
        return (f32_voxels_to_f32_bytes(vol), DT_FLOAT32, 32);
    }
    if first.bits_allocated <= 8 {
        return (f32_voxels_to_u8_bytes(vol), DT_UINT8, 8);
    }
    if first.is_signed || first.bits_stored < 16 {
        return (f32_voxels_to_i16_bytes(vol), DT_INT16, 16);
    }
    (f32_voxels_to_u16_bytes(vol), DT_UINT16, 16)
}

fn apply_intensity_variance_and_rgb(
    hdr: &mut Nifti1Header,
    bytes: &mut Vec<u8>,
    vol: &[f32],
    images: &[&DicomImage],
    first: &DicomImage,
    nx: usize,
    ny: usize,
    opts: &DcmOpts,
) {
    if first.samples_per_pixel != 3 && !images.is_empty() {
        let slope0 = first.inten_scale;
        let inter0 = first.inten_intercept;
        let varies = images.iter().any(|s| {
            (s.inten_scale - slope0).abs() > 1e-6 || (s.inten_intercept - inter0).abs() > 1e-6
        });
        if varies {
            if opts.ignore_intensity_scaling {
                eprintln!(
                    "Warning: Variance of DICOM slope/intercept is being ignored due to use of the `-p o` option."
                );
            } else if hdr.datatype != DT_FLOAT32 {
                eprintln!("Saving as 32-bit float (DICOM slope/intercept varies).");
                let spp = first.samples_per_pixel.max(1) as usize;
                let n_per = nx * ny * spp;
                let mut fvol = Vec::with_capacity(vol.len());
                for (si, s) in images.iter().enumerate() {
                    let start = si * n_per;
                    let end = (start + n_per).min(vol.len());
                    let sl = s.inten_scale;
                    let ic = s.inten_intercept;
                    for &v in &vol[start..end] {
                        fvol.push(v * sl + ic);
                    }
                }
                while fvol.len() < vol.len() {
                    fvol.push(0.0);
                }
                *bytes = f32_voxels_to_f32_bytes(&fvol);
                hdr.datatype = DT_FLOAT32;
                hdr.bitpix = 32;
                hdr.scl_slope = 1.0;
                hdr.scl_inter = 0.0;
            }
        }
    }
    if hdr.datatype == dcm_nifti::DT_RGB24 {
        if opts.rgb_planar {
            rgb::rgb_to_planar(bytes, hdr);
        } else if first.is_planar_rgb {
            rgb::planar_to_rgb(bytes, hdr);
        }
    }
}

pub(crate) fn unique_path(stem: &Path, ext: &str, conflict: NameConflict) -> Result<PathBuf> {
    let p = stem.with_extension(ext);
    if !p.exists() {
        return Ok(p);
    }
    match conflict {
        NameConflict::Skip => Err(Error::convert(format!("skipping existing {}", p.display()))),
        NameConflict::Overwrite => Ok(p),
        NameConflict::AddSuffix => {
            let mut i = 1;
            loop {
                let cand = PathBuf::from(format!("{}_{i}.{ext}", stem.display()));
                if !cand.exists() {
                    return Ok(cand);
                }
                i += 1;
                if i > 999 {
                    return Err(Error::convert("too many name collisions"));
                }
            }
        }
    }
}
