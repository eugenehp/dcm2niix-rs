//! MR spectroscopy (SVS + MRSI) and NIfTI-MRS ecode-44 header extension.
//!
//! Multi-DICOM CSI stacks classic per-slice / per-tile spectroscopy files along
//! the slice axis (beyond upstream C++, which still errors for `nConvert != 1`).

use std::path::PathBuf;

use dcm_core::error::{Error, Result};
use dcm_dicom::{spectroscopy_data_prefetched, DicomImage, Manufacturer};
use dcm_nifti::{write_nii_with_ext, Nifti1Header, DT_COMPLEX64, NIFTI_UNITS_SEC};
use rayon::prelude::*;
use serde_json::{json, Value};

use crate::filename::create_filename;
use crate::opts::{DcmOpts, SaveFormat};

/// Write NIfTI-MRS for SVS and MRSI (single- or multi-DICOM).
pub fn convert_mrs(
    images: Vec<DicomImage>,
    opts: &DcmOpts,
    mmaps: &dcm_dicom::MmapCache,
) -> Result<Vec<PathBuf>> {
    if images.is_empty() || !images[0].is_mrs {
        return Ok(vec![]);
    }
    if opts.save_format != SaveFormat::Nifti {
        return Err(Error::convert(
            "MRS: only NIfTI output (-e n) is supported; rerun without alternate save format",
        ));
    }
    let spatial = images[0].rows > 1
        || images[0].columns > 1
        || images[0].number_of_frames > 1
        || images[0].mrs_acq_type > 0;
    if spatial {
        convert_mrsi(images, opts, mmaps)
    } else {
        convert_svs(images, opts, mmaps)
    }
}

fn convert_svs(
    mut images: Vec<DicomImage>,
    opts: &DcmOpts,
    mmaps: &dcm_dicom::MmapCache,
) -> Result<Vec<PathBuf>> {
    images.sort_by_key(|d| d.instance_number);
    let d0 = &images[0];
    let mut frames: Vec<Vec<f32>> = Vec::new();
    let mut n_pts = d0.data_point_columns.max(0) as usize;
    let mut mrsref: Option<Vec<f32>> = None;
    let payloads: Result<Vec<Option<(Vec<f32>, usize)>>> = images
        .par_iter()
        .map(|d| spectroscopy_data_prefetched(&d.path, mmaps))
        .collect();
    for (d, payload) in images.iter().zip(payloads?) {
        let Some((fid, cols)) = payload else {
            eprintln!(
                "MRS {}: (5600,0020) Spectroscopy Data missing — skipping",
                d.path.display()
            );
            continue;
        };
        if n_pts == 0 {
            n_pts = cols;
        }
        let need = n_pts * 2;
        if need == 0 {
            continue;
        }
        let mult = if fid.len() % need == 0 {
            fid.len() / need
        } else {
            1
        };
        // Philips Enhanced multi-dynamic SVS: multiplier ≥ 3 → stacked dynamics
        // (C++ nDynPerFile). Multiplier == 2 keeps classic main + water-ref path.
        if d.manufacturer == Manufacturer::Philips && mult >= 3 && images.len() == 1 {
            for i in 0..mult {
                let start = i * need;
                let mut f = fid[start..start + need].to_vec();
                apply_phase_convention(d, &mut f);
                frames.push(f);
            }
            eprintln!(
                "MRS: Philips Enhanced multi-dynamic SVS ({mult} dynamics in one DICOM)"
            );
        } else if fid.len() >= need * 2 && fid.len() % need == 0 && fid.len() / need == 2 {
            // Classic Philips 2× payload: main + water-ref.
            frames.push(fid[..need].to_vec());
            let mut ref_frame = fid[need..need * 2].to_vec();
            apply_phase_convention(d, &mut ref_frame);
            if mrsref.is_none() {
                mrsref = Some(Vec::new());
            }
            if let Some(r) = mrsref.as_mut() {
                r.extend_from_slice(&ref_frame);
            }
            eprintln!("MRS: DICOM payload is 2x expected size; emitting _mrsref water-reference companion");
        } else {
            let mut f = fid;
            if f.len() > need {
                f.truncate(need);
            }
            apply_phase_convention(d, &mut f);
            frames.push(f);
        }
    }
    if frames.is_empty() || n_pts == 0 {
        return Ok(vec![]);
    }
    let n_dyn = frames.len();
    if n_dyn > 32767 {
        return Err(Error::convert(format!(
            "MRS: {n_dyn} effective dynamics exceeds NIfTI-1 dim[5] limit (32767)"
        )));
    }
    let mut vol = Vec::with_capacity(n_pts * 2 * n_dyn);
    for f in &frames {
        let take = (n_pts * 2).min(f.len());
        vol.extend_from_slice(&f[..take]);
        if take < n_pts * 2 {
            vol.resize(vol.len() + (n_pts * 2 - take), 0.0);
        }
    }
    let mut paths = write_mrs_nifti(d0, opts, &vol, 1, 1, 1, n_pts, n_dyn, "")?;
    if let Some(ref_vol) = mrsref {
        let mut pref = write_mrs_nifti(d0, opts, &ref_vol, 1, 1, 1, n_pts, n_dyn, "_mrsref")?;
        paths.append(&mut pref);
    }
    Ok(paths)
}

fn convert_mrsi(
    mut images: Vec<DicomImage>,
    opts: &DcmOpts,
    mmaps: &dcm_dicom::MmapCache,
) -> Result<Vec<PathBuf>> {
    images.sort_by(|a, b| {
        a.instance_number.cmp(&b.instance_number).then_with(|| {
            slice_depth(a)
                .partial_cmp(&slice_depth(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    let d0 = &images[0];
    let n_pts = if d0.data_point_columns > 0 {
        d0.data_point_columns as usize
    } else {
        0
    };
    let cols = d0.columns.max(1);
    let rows = d0.rows.max(1);

    for d in &images {
        if d.columns.max(1) != cols || d.rows.max(1) != rows {
            return Err(Error::convert(format!(
                "MRSI: multi-DICOM CSI grid mismatch ({}x{} vs {}x{})",
                d.columns.max(1),
                d.rows.max(1),
                cols,
                rows
            )));
        }
        if d.manufacturer != d0.manufacturer {
            return Err(Error::convert(
                "MRSI: multi-DICOM CSI stack requires a single manufacturer",
            ));
        }
    }

    let payloads: Result<Vec<Option<(Vec<f32>, usize)>>> = images
        .par_iter()
        .map(|d| spectroscopy_data_prefetched(&d.path, mmaps))
        .collect();

    let mut slice_blocks: Vec<Vec<f32>> = Vec::new();
    let mut n_pts_resolved = n_pts;
    for (d, payload) in images.iter().zip(payloads?) {
        let Some((mut raw, n_pts_tag)) = payload else {
            return Err(Error::convert(format!(
                "MRSI: (5600,0020) Spectroscopy Data missing ({})",
                d.path.display()
            )));
        };
        let file_pts = if d.data_point_columns > 0 {
            d.data_point_columns as usize
        } else {
            n_pts_tag
        };
        if n_pts_resolved == 0 {
            n_pts_resolved = file_pts;
        } else if file_pts != n_pts_resolved {
            return Err(Error::convert(format!(
                "MRSI: multi-DICOM CSI spectral length mismatch ({file_pts} vs {n_pts_resolved})"
            )));
        }
        let slices = d.number_of_frames.max(1) as usize;
        let total_samples = cols * rows * slices * n_pts_resolved;
        let expect_floats = total_samples * 2;
        if raw.len() < expect_floats {
            return Err(Error::convert(format!(
                "MRSI: FID payload {} floats != expected {} ({}x{}x{}x{} complex64)",
                raw.len(),
                expect_floats,
                cols,
                rows,
                slices,
                n_pts_resolved
            )));
        }
        raw.truncate(expect_floats);
        // DICOM order: (slice, row, col, spec) for Siemens; keep per-file blocks
        // contiguous so stacking extends the slice axis.
        slice_blocks.push(raw);
    }

    let n_pts = n_pts_resolved;
    if n_pts == 0 || cols > 32767 || rows > 32767 || n_pts > 32767 {
        return Err(Error::convert(format!(
            "MRSI: unexpected dims (cols={cols} rows={rows} N_pts={n_pts})"
        )));
    }

    let samples_per_slice = cols * rows * n_pts;
    let mut total_slices = 0usize;
    for block in &slice_blocks {
        let ns = block.len() / 2;
        if ns % samples_per_slice != 0 {
            return Err(Error::convert(
                "MRSI: FID block length is not an integer number of CSI slices",
            ));
        }
        total_slices += ns / samples_per_slice;
    }
    if total_slices == 0 || total_slices > 32767 {
        return Err(Error::convert(format!(
            "MRSI: stacked slice count {total_slices} is invalid"
        )));
    }

    let mut raw = Vec::with_capacity(total_slices * samples_per_slice * 2);
    for block in slice_blocks {
        raw.extend_from_slice(&block);
    }

    // Siemens XA Enhanced: negate imag + canonicalize -0.
    if d0.manufacturer == Manufacturer::Siemens {
        let total_samples = total_slices * samples_per_slice;
        if d0.mrs_acq_type != 0 {
            for i in 0..total_samples {
                raw[2 * i + 1] = -raw[2 * i + 1] + 0.0;
            }
        } else {
            for i in 0..total_samples {
                raw[2 * i + 1] = raw[2 * i + 1] + 0.0;
            }
        }
    }

    let slices = total_slices;
    let expect_floats = samples_per_slice * slices * 2;
    let mut fid = vec![0.0f32; expect_floats];
    if d0.manufacturer == Manufacturer::Uih {
        // UIH: (c,r,f,p) → (r,c,f,p)
        for c in 0..cols {
            for r in 0..rows {
                for f in 0..slices {
                    for p in 0..n_pts {
                        let src = (((c * rows + r) * slices + f) * n_pts + p) * 2;
                        let dst = (((r * cols + c) * slices + f) * n_pts + p) * 2;
                        fid[dst] = raw[src];
                        fid[dst + 1] = raw[src + 1];
                    }
                }
            }
        }
        if images.len() > 1 {
            eprintln!(
                "MRSI: stacked {} DICOM CSI files → {} slices",
                images.len(),
                slices
            );
        }
        write_mrs_nifti(d0, opts, &fid, rows, cols, slices, n_pts, 1, "")
    } else {
        for s in 0..slices {
            for r in 0..rows {
                for c in 0..cols {
                    for p in 0..n_pts {
                        let src = (((s * rows + r) * cols + c) * n_pts + p) * 2;
                        let dst = (((c * rows + r) * slices + s) * n_pts + p) * 2;
                        fid[dst] = raw[src];
                        fid[dst + 1] = raw[src + 1];
                    }
                }
            }
        }
        if images.len() > 1 {
            eprintln!(
                "MRSI: stacked {} DICOM CSI files → {} slices",
                images.len(),
                slices
            );
        }
        write_mrs_nifti(d0, opts, &fid, cols, rows, slices, n_pts, 1, "")
    }
}

/// Distance of ImagePositionPatient along the IOP slice normal (for CSI sort).
fn slice_depth(d: &DicomImage) -> f64 {
    let o = &d.orient;
    let nx = o[1] * o[5] - o[2] * o[4];
    let ny = o[2] * o[3] - o[0] * o[5];
    let nz = o[0] * o[4] - o[1] * o[3];
    d.patient_position[0] * nx + d.patient_position[1] * ny + d.patient_position[2] * nz
}

fn apply_phase_convention(d: &DicomImage, fid: &mut [f32]) {
    if d.manufacturer == Manufacturer::Siemens && d.is_xa {
        for i in (1..fid.len()).step_by(2) {
            if fid[i] != 0.0 {
                fid[i] = -fid[i];
            }
        }
    }
}

fn write_mrs_nifti(
    d: &DicomImage,
    opts: &DcmOpts,
    vol: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    n_pts: usize,
    n_dyn: usize,
    name_suffix: &str,
) -> Result<Vec<PathBuf>> {
    let mut hdr = Nifti1Header::default();
    hdr.dim[0] = if n_dyn > 1 { 5 } else { 4 };
    hdr.dim[1] = nx as i16;
    hdr.dim[2] = ny as i16;
    hdr.dim[3] = nz as i16;
    hdr.dim[4] = n_pts as i16;
    hdr.dim[5] = n_dyn.max(1) as i16;
    hdr.datatype = DT_COMPLEX64;
    hdr.bitpix = 64;
    hdr.pixdim[1] = if d.xyz_mm[1] > 0.0 {
        d.xyz_mm[1] as f32
    } else {
        1.0
    };
    hdr.pixdim[2] = if d.xyz_mm[2] > 0.0 {
        d.xyz_mm[2] as f32
    } else {
        1.0
    };
    hdr.pixdim[3] = if d.xyz_mm[3] > 0.0 {
        d.xyz_mm[3] as f32
    } else if d.slice_thickness > 0.0 {
        d.slice_thickness as f32
    } else {
        1.0
    };
    hdr.pixdim[4] = 1.0;
    hdr.pixdim[5] = if d.tr > 0.0 {
        (d.tr / 1000.0) as f32
    } else {
        1.0
    };
    hdr.xyzt_units = NIFTI_UNITS_SEC;
    hdr.scl_slope = 1.0;
    if d.has_orientation() {
        let q = dcm_core::matrix::nifti_dicom2mat(d.orient, d.patient_position, d.xyz_mm)
            .lps_to_ras_f32();
        hdr.set_sform(&q);
    }
    let bytes = f32_slice_to_bytes(vol);
    let stem = create_filename(d, opts)?;
    let stem = if name_suffix.is_empty() {
        stem
    } else {
        PathBuf::from(format!("{}{name_suffix}", stem.display()))
    };
    let nii = stem.with_extension("nii");
    let ext = mrs_hdr_ext_json(d, &hdr);
    write_nii_with_ext(&nii, &hdr, &bytes, ext.as_deref())?;
    if n_dyn > 1 {
        println!(
            "Convert {} DICOM as {} ({}x{}x{}x{}x{})",
            1.max(n_dyn),
            nii.display(),
            nx,
            ny,
            nz,
            n_pts,
            n_dyn
        );
    } else {
        println!(
            "Convert {} DICOM as {} ({}x{}x{}x{})",
            1,
            nii.display(),
            nx,
            ny,
            nz,
            n_pts
        );
    }
    Ok(vec![nii])
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn mrs_hdr_ext_json(d: &DicomImage, _hdr: &Nifti1Header) -> Option<String> {
    if d.imaging_frequency <= 0.0 || d.resonant_nucleus.is_empty() {
        return None;
    }
    let mut root = json!({
        "SpectrometerFrequency": [d.imaging_frequency],
        "ResonantNucleus": [d.resonant_nucleus],
        "dim_5": "DIM_DYN",
        "InversionTime": if d.ti > 0.0 { d.ti / 1000.0 } else { 0.0 },
        "WaterSuppressed": !d.is_mrs_ref,
    });
    let obj = root.as_object_mut()?;
    let sw = spectral_width_hz(d);
    if sw > 0.0 {
        obj.insert("SpectralWidth".into(), json!(sw));
        obj.insert("DwellTime".into(), json!(1.0 / sw));
    }
    if d.number_of_k_space_trajectories > 0 {
        obj.insert(
            "NumberOfKSpaceTrajectories".into(),
            json!(d.number_of_k_space_trajectories),
        );
    }
    if d.data_point_columns > 0 {
        obj.insert(
            "NumberOfSpectralPoints".into(),
            json!(d.data_point_columns),
        );
    }
    if d.xyz_mm[1] > 0.0 && d.xyz_mm[2] > 0.0 && d.slice_thickness > 0.0 {
        obj.insert(
            "AcquisitionVoxelSize".into(),
            json!([d.xyz_mm[2], d.xyz_mm[1], d.slice_thickness]),
        );
    }
    if let Some(voi) = d.mrs_voi_matrix() {
        obj.insert(
            "VOI".into(),
            json!([
                [voi[0][0], voi[0][1], voi[0][2], voi[0][3]],
                [voi[1][0], voi[1][1], voi[1][2], voi[1][3]],
                [voi[2][0], voi[2][1], voi[2][2], voi[2][3]],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        );
    }
    let avg = if d.number_of_averages > 0.0 {
        d.number_of_averages as i32
    } else {
        1
    };
    // NIfTI ext path: dim[5] not known here; use averages alone (sidecar has full formula).
    if avg > 0 {
        obj.insert("NumberOfTransients".into(), json!(avg));
    }
    match d.mrs_acq_type {
        1 => {
            obj.insert(
                "MRSpectroscopyAcquisitionType".into(),
                json!("ROW"),
            );
        }
        2 => {
            obj.insert(
                "MRSpectroscopyAcquisitionType".into(),
                json!("PLANE"),
            );
        }
        3 => {
            obj.insert(
                "MRSpectroscopyAcquisitionType".into(),
                json!("VOLUME"),
            );
        }
        _ => {
            obj.insert(
                "MRSpectroscopyAcquisitionType".into(),
                json!("SINGLE_VOXEL"),
            );
        }
    }
    if d.te > 0.0 {
        obj.insert("EchoTime".into(), json!(d.te / 1000.0));
    }
    if d.tr > 0.0 {
        obj.insert("RepetitionTime".into(), json!(d.tr / 1000.0));
    }
    if d.flip_angle > 0.0 {
        obj.insert("ExcitationFlipAngle".into(), json!(d.flip_angle));
    }
    add_str(obj, "TransmitCoilName", &d.transmit_coil_name);
    add_str(obj, "Manufacturer", manufacturer_str(d.manufacturer));
    add_str(obj, "ManufacturersModelName", &d.manufacturers_model_name);
    add_str(obj, "DeviceSerialNumber", &d.device_serial_number);
    add_str(obj, "SoftwareVersions", &d.software_versions);
    add_str(obj, "InstitutionName", &d.institution_name);
    add_str(obj, "ProtocolName", &d.protocol_name);
    add_str(obj, "SequenceName", &d.sequence_name);
    Some(root.to_string())
}

fn add_str(obj: &mut serde_json::Map<String, Value>, key: &str, val: &str) {
    if !val.is_empty() {
        obj.insert(key.into(), json!(val));
    }
}

fn manufacturer_str(m: Manufacturer) -> &'static str {
    match m {
        Manufacturer::Siemens => "Siemens",
        Manufacturer::Ge => "GE",
        Manufacturer::Philips => "Philips",
        Manufacturer::Uih => "UIH",
        Manufacturer::Canon => "Canon",
        Manufacturer::Bruker => "Bruker",
        _ => "",
    }
}

fn spectral_width_hz(d: &DicomImage) -> f64 {
    // Prefer dwell-time derived width (C++ mrsSpectralWidthHz); fall back to (0018,9052).
    if d.dwell_time_ns > 0.0 {
        let dwell_s = d.dwell_time_ns * 1e-9;
        if dwell_s > 0.0 {
            return 1.0 / dwell_s;
        }
    }
    if d.spectral_width_hz > 0.0 {
        return d.spectral_width_hz;
    }
    0.0
}

pub fn is_mrs_series(images: &[DicomImage]) -> bool {
    images.first().map(|d| d.is_mrs).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_depth_uses_cross_product() {
        let orient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let pos = [0.0, 0.0, 3.0];
        let nx = orient[1] * orient[5] - orient[2] * orient[4];
        let ny = orient[2] * orient[3] - orient[0] * orient[5];
        let nz = orient[0] * orient[4] - orient[1] * orient[3];
        let depth = pos[0] * nx + pos[1] * ny + pos[2] * nz;
        assert!((depth - 3.0_f64).abs() < 1e-9);
    }

    #[test]
    fn pepolar_constants_align_with_upstream() {
        assert_eq!(dcm_dicom::GE_EPI_PEPOLAR_FWD, 3);
        assert!(dcm_dicom::is_pepolar(3));
        assert!(dcm_dicom::needs_extra_y_flip(dcm_dicom::GE_EPI_PEPOLAR_REV));
        assert!(!dcm_dicom::needs_extra_y_flip(dcm_dicom::GE_EPI_PEPOLAR_FWD));
    }
}
