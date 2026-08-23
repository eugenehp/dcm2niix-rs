//! Enhanced multi-frame helpers (Shared / Per-Frame Functional Groups).

use dicom_core::Tag;
use dicom_dictionary_std::tags;
use dicom_object::mem::InMemDicomObject;
use dicom_object::{DefaultDicomObject, FileDicomObject};
use dcm_core::snap_f32;

use super::DicomImage;

/// Per-frame geometry + contrast extracted from `(5200,9230)`.
#[derive(Debug, Clone)]
pub struct FrameGeom {
    pub patient_position: [f64; 4],
    pub orient: [f64; 7],
    pub b_value: f64,
    pub diffusion_direction: [f64; 3],
    pub te: f64,
    pub tr: f64,
    pub trigger_delay: f64,
    pub inten_scale: f32,
    pub inten_intercept: f32,
    pub inten_scale_philips: f32,
    pub is_phase: bool,
    pub is_real: bool,
    pub is_imaginary: bool,
    pub dimension_index: [i32; 8],
}

impl Default for FrameGeom {
    fn default() -> Self {
        Self {
            patient_position: [f64::NAN; 4],
            orient: [0.0; 7],
            b_value: -1.0,
            diffusion_direction: [0.0; 3],
            te: 0.0,
            tr: 0.0,
            trigger_delay: 0.0,
            inten_scale: 1.0,
            inten_intercept: 0.0,
            inten_scale_philips: 0.0,
            is_phase: false,
            is_real: false,
            is_imaginary: false,
            dimension_index: [0; 8],
        }
    }
}

/// Per-volume contrast summary used for `isScaleOrTEVaries` splitting.
#[derive(Debug, Clone)]
pub struct VolumeContrast {
    pub te: f64,
    pub tr: f64,
    pub trigger_delay: f64,
    pub inten_scale: f32,
    pub inten_intercept: f32,
    pub is_phase: bool,
    pub is_real: bool,
    pub is_imaginary: bool,
    pub b_value: f64,
    pub diffusion_direction: [f64; 3],
    pub echo_num: i32,
}

/// Read per-frame IPP / IOP / diffusion / contrast from enhanced functional groups.
pub fn read_per_frame_geometry(obj: &DefaultDicomObject) -> Vec<FrameGeom> {
    let Ok(elem) = obj.element(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE) else {
        return vec![];
    };
    let Some(items) = elem.items() else {
        return vec![];
    };
    let mut frames = Vec::with_capacity(items.len());
    for item in items {
        let mut fg = FrameGeom::default();
        // PlanePositionSequence (0020,9113) → ImagePositionPatient
        if let Some(ipp) = nested_f64s(item, Tag(0x0020, 0x9113), tags::IMAGE_POSITION_PATIENT) {
            if ipp.len() >= 3 {
                fg.patient_position[1] = snap_f32(ipp[0]);
                fg.patient_position[2] = snap_f32(ipp[1]);
                fg.patient_position[3] = snap_f32(ipp[2]);
            }
        }
        if let Some(iop) = nested_f64s(item, Tag(0x0020, 0x9116), tags::IMAGE_ORIENTATION_PATIENT)
        {
            if iop.len() >= 6 {
                for i in 0..6 {
                    fg.orient[i + 1] = snap_f32(iop[i]);
                }
            }
        }
        if let Some(b) = nested_f64(item, Tag(0x0018, 0x9117), Tag(0x0018, 0x9087)) {
            fg.b_value = b;
        }
        if let Some(v) = nested_f64s(item, Tag(0x0018, 0x9076), Tag(0x0018, 0x9089)) {
            if v.len() >= 3 {
                fg.diffusion_direction = [v[0], v[1], v[2]];
            }
        }
        if let Some(te) = nested_f64(item, Tag(0x0018, 0x9114), Tag(0x0018, 0x9082)) {
            fg.te = te;
        }
        // MR Timing / TR (0018,9112) → (0018,0080)
        if let Some(tr) = nested_f64(item, Tag(0x0018, 0x9112), tags::REPETITION_TIME) {
            fg.tr = tr;
        }
        // Trigger delay (0018,1060) or (0020,9153) nested
        if let Some(t) = nested_f64(item, Tag(0x0018, 0x9112), Tag(0x0018, 0x1060))
            .or_else(|| nested_f64(item, Tag(0x0018, 0x9112), Tag(0x0020, 0x9153)))
            .or_else(|| item_f64(item, Tag(0x0020, 0x9153)))
            .or_else(|| item_f64(item, Tag(0x0018, 0x1060)))
        {
            fg.trigger_delay = if t.abs() < 1e-6 { 0.0 } else { t };
        }
        // PixelValueTransformationSequence (0028,9145) → rescale
        if let Some(slope) = nested_f64(item, Tag(0x0028, 0x9145), tags::RESCALE_SLOPE) {
            fg.inten_scale = slope as f32;
        }
        if let Some(inter) = nested_f64(item, Tag(0x0028, 0x9145), tags::RESCALE_INTERCEPT) {
            fg.inten_intercept = inter as f32;
        }
        // Philips SS (2005,100E) if present in frame
        if let Some(ss) = nested_f64(item, Tag(0x0028, 0x9145), Tag(0x2005, 0x100e))
            .or_else(|| item_f64(item, Tag(0x2005, 0x100e)))
        {
            fg.inten_scale_philips = ss as f32;
        }
        // ComplexImageComponent (0008,9208) — frame or MR Image Frame Type SQ
        if let Some(cs) = nested_str(item, Tag(0x0018, 0x9226), Tag(0x0008, 0x9208))
            .or_else(|| item_str(item, Tag(0x0008, 0x9208)))
        {
            let u = cs.to_ascii_uppercase();
            if u.starts_with("PH") {
                fg.is_phase = true;
            } else if u.starts_with("RE") {
                fg.is_real = true;
            } else if u.starts_with("IM") {
                fg.is_imaginary = true;
            }
        }
        // ImageType in MRImageFrameTypeSequence may also signal PHASE/REAL
        if let Some(it) = nested_str(item, Tag(0x0018, 0x9226), tags::IMAGE_TYPE) {
            let u = it.to_ascii_uppercase();
            if u.contains("PHASE") {
                fg.is_phase = true;
            }
            if u.contains("REAL") {
                fg.is_real = true;
            }
            if u.contains("IMAGINARY") {
                fg.is_imaginary = true;
            }
        }
        if let Ok(div) = item.element(Tag(0x0020, 0x9157)) {
            if let Ok(vals) = div.to_multi_float64() {
                for (i, v) in vals.iter().take(8).enumerate() {
                    fg.dimension_index[i] = *v as i32;
                }
            }
        }
        frames.push(fg);
    }
    frames
}

/// Sort frames by DimensionIndexValues (lexicographic), falling back to input order.
pub fn sort_frames_by_dimension_index(frames: &mut Vec<FrameGeom>, voxels: &mut [f32], frame_vox: usize) {
    if frames.len() < 2 || frames.iter().all(|f| f.dimension_index.iter().all(|&v| v == 0)) {
        return;
    }
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&a, &b| frames[a].dimension_index.cmp(&frames[b].dimension_index));
    if order.iter().enumerate().all(|(i, &o)| i == o) {
        return;
    }
    let old_frames = frames.clone();
    let old_vox = voxels.to_vec();
    for (new_i, &old_i) in order.iter().enumerate() {
        frames[new_i] = old_frames[old_i].clone();
        let src = old_i * frame_vox;
        let dst = new_i * frame_vox;
        if src + frame_vox <= old_vox.len() && dst + frame_vox <= voxels.len() {
            voxels[dst..dst + frame_vox].copy_from_slice(&old_vox[src..src + frame_vox]);
        }
    }
}

/// Infer `(nz, nt)` from unique IPP positions; require `nf % nz == 0`.
pub fn infer_stack_dims(frames: &[FrameGeom], nf: usize) -> (usize, usize) {
    if frames.len() < 2 {
        return (1, nf.max(1));
    }
    let mut unique: Vec<[f64; 3]> = Vec::new();
    for fg in frames {
        let p = [fg.patient_position[1], fg.patient_position[2], fg.patient_position[3]];
        if p.iter().any(|v| v.is_nan()) {
            continue;
        }
        if unique.iter().all(|u| {
            ((u[0] - p[0]).powi(2) + (u[1] - p[1]).powi(2) + (u[2] - p[2]).powi(2)).sqrt() > 1e-3
        }) {
            unique.push(p);
        }
    }
    let nz = unique.len().max(1);
    if nf % nz == 0 {
        (nz, nf / nz)
    } else {
        // Fallback: consecutive IPP distance heuristic
        let a = frames[0].patient_position;
        let b = frames[1].patient_position;
        let dist = ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2) + (a[3] - b[3]).powi(2)).sqrt();
        if dist < 1e-3 {
            (1, nf)
        } else {
            (nf, 1)
        }
    }
}

/// One contrast summary per 4D volume (from first slice of each volume).
pub fn volume_contrasts(frames: &[FrameGeom], nz: usize, nt: usize) -> Vec<VolumeContrast> {
    let mut out = Vec::with_capacity(nt);
    for t in 0..nt {
        let idx = t * nz;
        let fg = frames.get(idx).cloned().unwrap_or_default();
        out.push(VolumeContrast {
            te: fg.te,
            tr: fg.tr,
            trigger_delay: fg.trigger_delay,
            inten_scale: fg.inten_scale,
            inten_intercept: fg.inten_intercept,
            is_phase: fg.is_phase,
            is_real: fg.is_real,
            is_imaginary: fg.is_imaginary,
            b_value: fg.b_value,
            diffusion_direction: fg.diffusion_direction,
            echo_num: 1,
        });
    }
    // Assign echo numbers by unique TE order of appearance.
    let mut echo = 1i32;
    let mut te_map: Vec<(f64, i32)> = Vec::new();
    for v in &mut out {
        if let Some((_, e)) = te_map.iter().find(|(t, _)| (*t - v.te).abs() < 1e-6) {
            v.echo_num = *e;
        } else if v.te > 0.0 {
            te_map.push((v.te, echo));
            v.echo_num = echo;
            echo += 1;
        }
    }
    out
}

/// True when TE / phase / real / imag / trigger vary across volumes (C++ `isScaleOrTEVaries`).
pub fn scale_or_te_varies(vols: &[VolumeContrast]) -> bool {
    if vols.len() < 2 {
        return false;
    }
    let v0 = &vols[0];
    vols.iter().any(|v| {
        (v.te - v0.te).abs() > 1e-6
            || v.is_phase != v0.is_phase
            || v.is_real != v0.is_real
            || v.is_imaginary != v0.is_imaginary
            || (v.trigger_delay - v0.trigger_delay).abs() > 1e-3
    })
}

/// Assign `gradDynVol`-style series IDs (1-based) matching C++ matching rules.
/// When `asl_flags != 0`, trigger-delay mismatches are ignored (issue 533).
pub fn assign_grad_dyn_vol(vols: &[VolumeContrast], asl_flags: u32) -> Vec<usize> {
    let mut ids = vec![0usize; vols.len()];
    if vols.is_empty() {
        return ids;
    }
    ids[0] = 1;
    let mut series = 1usize;
    for i in 1..vols.len() {
        for j in 0..i {
            let a = &vols[i];
            let b = &vols[j];
            let trig_ok =
                asl_flags != 0 || (a.trigger_delay - b.trigger_delay).abs() < 1e-3;
            if trig_ok
                && (a.inten_intercept - b.inten_intercept).abs() < 1e-6
                && (a.inten_scale - b.inten_scale).abs() < 1e-6
                && a.is_real == b.is_real
                && a.is_imaginary == b.is_imaginary
                && a.is_phase == b.is_phase
                && (a.te - b.te).abs() < 1e-6
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
}

fn nested_f64s(
    item: &InMemDicomObject,
    seq: Tag,
    leaf: Tag,
) -> Option<Vec<f64>> {
    let seq_e = item.element(seq).ok()?;
    let items = seq_e.items()?;
    let first = items.first()?;
    let e = first.element(leaf).ok()?;
    e.to_multi_float64().ok()
}

fn nested_f64(item: &InMemDicomObject, seq: Tag, leaf: Tag) -> Option<f64> {
    nested_f64s(item, seq, leaf)?.into_iter().next()
}

fn nested_str(item: &InMemDicomObject, seq: Tag, leaf: Tag) -> Option<String> {
    let seq_e = item.element(seq).ok()?;
    let items = seq_e.items()?;
    let first = items.first()?;
    let e = first.element(leaf).ok()?;
    e.to_str().ok().map(|s| s.trim().to_string())
}

fn item_f64(item: &InMemDicomObject, tag: Tag) -> Option<f64> {
    item.element(tag)
        .ok()?
        .to_multi_float64()
        .ok()?
        .into_iter()
        .next()
}

fn item_str(item: &InMemDicomObject, tag: Tag) -> Option<String> {
    item.element(tag)
        .ok()?
        .to_str()
        .ok()
        .map(|s| s.trim().to_string())
}

/// Expand a multi-frame header into one `DicomImage` per frame (geometry only).
pub fn expand_frames(base: &DicomImage, frames: &[FrameGeom]) -> Vec<DicomImage> {
    if frames.is_empty() {
        return vec![base.clone()];
    }
    frames
        .iter()
        .enumerate()
        .map(|(i, fg)| {
            let mut d = base.clone();
            d.instance_number = (i + 1) as i32;
            d.number_of_frames = 1;
            if !fg.patient_position[1].is_nan() && fg.patient_position[1] != 0.0
                || fg.patient_position[2] != 0.0
                || fg.patient_position[3] != 0.0
            {
                d.patient_position = fg.patient_position;
            }
            if fg.orient[1..].iter().any(|v| *v != 0.0) {
                d.orient = fg.orient;
            }
            if fg.b_value >= 0.0 {
                d.b_value = fg.b_value;
                d.diffusion_direction = fg.diffusion_direction;
            }
            if fg.te > 0.0 {
                d.te = fg.te;
            }
            d.is_has_phase = fg.is_phase;
            d.is_has_real = fg.is_real;
            d.is_has_imaginary = fg.is_imaginary;
            d.trigger_delay_time = fg.trigger_delay;
            d
        })
        .collect()
}

/// True when this object is enhanced multi-frame with per-frame groups.
pub fn is_enhanced_multiframe(obj: &FileDicomObject<InMemDicomObject>) -> bool {
    obj.element(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE).is_ok()
}
