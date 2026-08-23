//! ASL classification flags matching C++ `kASL_FLAG_*` (`nii_dicom.h`).

/// No ASL evidence.
pub const ASL_FLAG_NONE: u32 = 0;
/// GE 3D pCASL.
pub const ASL_FLAG_GE_3DPCASL: u32 = 1;
/// GE 3D continuous ASL.
pub const ASL_FLAG_GE_3DCASL: u32 = 2;
/// GE pseudo-continuous.
pub const ASL_FLAG_GE_PSEUDOCONTINUOUS: u32 = 4;
/// GE continuous.
pub const ASL_FLAG_GE_CONTINUOUS: u32 = 8;
/// Philips control volume.
pub const ASL_FLAG_PHILIPS_CONTROL: u32 = 16;
/// Philips label volume.
pub const ASL_FLAG_PHILIPS_LABEL: u32 = 32;
/// GE pulsed ASL.
pub const ASL_FLAG_GE_PULSED: u32 = 64;

/// Classify `(0018,9250)` ArterialSpinLabelingContrast CS.
pub fn flags_from_asl_contrast(cs: &str) -> u32 {
    let u = cs.to_ascii_uppercase();
    if u.contains("PSEUDOCONTINUOUS") {
        ASL_FLAG_GE_PSEUDOCONTINUOUS
    } else if u.contains("CONTINUOUS") {
        ASL_FLAG_GE_CONTINUOUS
    } else if u.contains("PULSED") {
        ASL_FLAG_GE_PULSED
    } else {
        ASL_FLAG_NONE
    }
}

/// Classify GE `(0043,10A3)` ASLContrastTechnique.
pub fn flags_from_ge_contrast_technique(cs: &str) -> u32 {
    flags_from_asl_contrast(cs)
}

/// Classify GE `(0043,10A4)` ASLLabelingTechnique LO.
pub fn flags_from_ge_labeling_technique(lo: &str) -> u32 {
    let l = lo.to_ascii_lowercase();
    if l.contains("3d pulsed continuous") {
        ASL_FLAG_GE_3DPCASL
    } else if l.contains("3d continuous") {
        ASL_FLAG_GE_3DCASL
    } else {
        ASL_FLAG_NONE
    }
}

/// Classify Philips `(2005,1429)` MRImageLabelType (`L`abel / `C`ontrol).
pub fn flags_from_philips_label_type(cs: &str) -> u32 {
    match cs.bytes().next().map(|b| b.to_ascii_uppercase()) {
        Some(b'L') => ASL_FLAG_PHILIPS_LABEL,
        Some(b'C') => ASL_FLAG_PHILIPS_CONTROL,
        _ => ASL_FLAG_NONE,
    }
}
