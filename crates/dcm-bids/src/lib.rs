//! BIDS sidecar JSON writer (upstream `nii_SaveBIDS` / `nii_SaveBIDSX` subset).
//!
//! Anonymisation (`-ba y/n/o`) matches dcm2niix: `y` strips dates and PII,
//! `n` keeps both, `o` strips PII but keeps acquisition timestamps.
//!
//! Richer dataset-level BIDS tooling belongs in sibling `bids-rs`; this crate
//! focuses on conversion-time sidecars (including Siemens/GE/Philips/UIH fields).

mod siemens;

use std::fs::File;
use std::io::Write;
use std::path::Path;

use dcm_core::error::{Error, Result};
use dcm_dicom::{DicomImage, Manufacturer, Modality};
use dcm_nifti::Nifti1Header;
use serde_json::{json, Map, Value};

pub use siemens::{write_siemens_sidecar, write_siemens_sidecar_ex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anonymize {
    /// Strip dates and patient PII (default `-ba y`).
    Full,
    /// Keep dates and PII (`-ba n`).
    None,
    /// Strip PII, keep timestamps (`-ba o`).
    PiiOnly,
}

/// Write a BIDS JSON sidecar next to a NIfTI stem.
///
/// `conversion_sw` becomes `ConversionSoftware` / version fields in the JSON.
/// `philips_float_scaling` mirrors CLI `-p y/n` (`UsePhilipsFloatNotDisplayScaling`).
pub fn write_sidecar(
    path: impl AsRef<Path>,
    dcm: &DicomImage,
    hdr: &Nifti1Header,
    anon: Anonymize,
    conversion_sw: &str,
) -> Result<()> {
    write_sidecar_ex(path, dcm, hdr, anon, conversion_sw, true, "")
}

/// Like [`write_sidecar`], with Philips `-p` and `-c` conversion comments.
pub fn write_sidecar_ex(
    path: impl AsRef<Path>,
    dcm: &DicomImage,
    hdr: &Nifti1Header,
    anon: Anonymize,
    conversion_sw: &str,
    philips_float_scaling: bool,
    conversion_comments: &str,
) -> Result<()> {
    // Full sequential writer (nii_SaveBIDSX order) for all major MR vendors.
    if matches!(
        dcm.manufacturer,
        Manufacturer::Siemens
            | Manufacturer::Ge
            | Manufacturer::Uih
            | Manufacturer::Philips
            | Manufacturer::Canon
            | Manufacturer::Toshiba
            | Manufacturer::Hitachi
            | Manufacturer::Bruker
    ) || dcm.modality == Modality::Mr
        || dcm.modality == Modality::Pt
        || dcm.modality == Modality::Ct
    {
        return write_siemens_sidecar_ex(
            path,
            dcm,
            hdr,
            anon,
            conversion_sw,
            philips_float_scaling,
            conversion_comments,
        );
    }
    write_sidecar_basic(path, dcm, hdr, anon, conversion_sw)
}

fn write_sidecar_basic(
    path: impl AsRef<Path>,
    dcm: &DicomImage,
    hdr: &Nifti1Header,
    anon: Anonymize,
    conversion_sw: &str,
) -> Result<()> {
    let path = path.as_ref();
    let mut obj = Map::new();

    if !dcm.modality.as_str().is_empty() {
        obj.insert("Modality".into(), json!(dcm.modality.as_str()));
    }
    if !dcm.manufacturer.as_str().is_empty() && dcm.manufacturer.as_str() != "UNKNOWN" {
        obj.insert("Manufacturer".into(), json!(dcm.manufacturer.as_str()));
    }
    insert_str(&mut obj, "ManufacturersModelName", &dcm.manufacturers_model_name);
    if anon != Anonymize::Full {
        insert_str(&mut obj, "InstitutionName", &dcm.institution_name);
    }
    insert_str(&mut obj, "DeviceSerialNumber", &dcm.station_name);
    insert_str(&mut obj, "SoftwareVersions", &dcm.software_versions);
    if !dcm.protocol_name.is_empty() {
        obj.insert("ProtocolName".into(), json!(sanitize(&dcm.protocol_name)));
    }
    insert_str(&mut obj, "SeriesDescription", &dcm.series_description);
    if dcm.series_number != 0 {
        obj.insert("SeriesNumber".into(), json!(dcm.series_number));
    }
    insert_str(&mut obj, "ScanningSequence", &dcm.scanning_sequence);
    insert_str(&mut obj, "SequenceVariant", &dcm.sequence_variant);
    insert_str(&mut obj, "ScanOptions", &dcm.scan_options);
    insert_str(&mut obj, "SequenceName", &dcm.sequence_name);
    insert_str(&mut obj, "ImageType", &dcm.image_type);
    if dcm.echo_number > 0 {
        obj.insert("EchoNumber".into(), json!(dcm.echo_number));
    }
    if dcm.tr > 0.0 {
        obj.insert("RepetitionTime".into(), json!(dcm.tr / 1000.0));
    }
    if dcm.te > 0.0 {
        obj.insert("EchoTime".into(), json!(dcm.te / 1000.0));
    }
    if dcm.ti > 0.0 {
        obj.insert("InversionTime".into(), json!(dcm.ti / 1000.0));
    }
    if dcm.flip_angle > 0.0 {
        obj.insert("FlipAngle".into(), json!(dcm.flip_angle));
    }
    if dcm.field_strength > 0.0 {
        obj.insert("MagneticFieldStrength".into(), json!(dcm.field_strength));
    }
    if dcm.pixel_bandwidth > 0.0 {
        obj.insert("PixelBandwidth".into(), json!(dcm.pixel_bandwidth));
    }
    if dcm.echo_train_length > 0 {
        obj.insert("EchoTrainLength".into(), json!(dcm.echo_train_length));
    }
    match dcm.phase_encoding_rc {
        'R' => {
            obj.insert("InPlanePhaseEncodingDirectionDICOM".into(), json!("ROW"));
        }
        'C' => {
            obj.insert("InPlanePhaseEncodingDirectionDICOM".into(), json!("COL"));
        }
        _ => {}
    }
    if hdr.dim[4] > 1 && dcm.tr > 0.0 {
        obj.insert("RepetitionTime".into(), json!(dcm.tr / 1000.0));
    }
    if dcm.frame_duration > 0.0 {
        obj.insert("FrameDuration".into(), json!(dcm.frame_duration / 1000.0));
    }
    if dcm.frame_reference_time >= 0.0 {
        obj.insert(
            "FrameReferenceTime".into(),
            json!(dcm.frame_reference_time / 1000.0),
        );
    }
    if !dcm.volume_onset_times.is_empty() && dcm.volume_onset_times[0] >= 0.0 {
        obj.insert(
            "FrameTimesStart".into(),
            json!(dcm.volume_onset_times.clone()),
        );
    }
    if !dcm.decay_factors.is_empty() && dcm.decay_factors[0] >= 0.0 {
        obj.insert(
            "DecayCorrectionFactor".into(),
            json!(dcm.decay_factors.clone()),
        );
    } else if dcm.decay_factor > 0.0 {
        obj.insert("DecayFactor".into(), json!([dcm.decay_factor]));
    }
    if !dcm.csa.bids_data_type.is_empty() && !dcm.csa.bids_entity_suffix.is_empty() {
        obj.insert(
            "BidsGuess".into(),
            json!([
                dcm.csa.bids_data_type.clone(),
                dcm.csa.bids_entity_suffix.clone()
            ]),
        );
    }
    obj.insert(
        "ConversionSoftware".into(),
        json!("dcm2niix"),
    );
    obj.insert("ConversionSoftwareVersion".into(), json!(conversion_sw));

    if anon == Anonymize::None {
        insert_str(&mut obj, "PatientName", &dcm.patient_name);
        insert_str(&mut obj, "PatientID", &dcm.patient_id);
        insert_str(&mut obj, "PatientSex", &dcm.patient_sex);
        insert_str(&mut obj, "PatientAge", &dcm.patient_age);
        insert_str(&mut obj, "PatientBirthDate", &dcm.patient_birth_date);
        insert_str(&mut obj, "AccessionNumber", &dcm.accession_number);
        insert_datetime(&mut obj, dcm);
    } else if anon == Anonymize::PiiOnly {
        insert_datetime(&mut obj, dcm);
    }

    let json = Value::Object(obj);
    let pretty = serde_json::to_string_pretty(&json)
        .map_err(|e| Error::convert(format!("BIDS JSON: {e}")))?;
    let mut f = File::create(path).map_err(|e| Error::io(path, e))?;
    f.write_all(pretty.as_bytes())
        .map_err(|e| Error::io(path, e))?;
    f.write_all(b"\n").map_err(|e| Error::io(path, e))?;
    Ok(())
}

fn insert_datetime(obj: &mut Map<String, Value>, dcm: &DicomImage) {
    let date = if !dcm.acquisition_date.is_empty() {
        &dcm.acquisition_date
    } else {
        &dcm.study_date
    };
    let time = if !dcm.acquisition_time.is_empty() {
        &dcm.acquisition_time
    } else {
        &dcm.study_time
    };
    if date.len() >= 8 && !time.is_empty() {
        let dt = format!(
            "{}-{}-{}T{}",
            &date[0..4],
            &date[4..6],
            &date[6..8],
            format_time(time)
        );
        obj.insert("AcquisitionTime".into(), json!(dt));
    }
}

fn format_time(t: &str) -> String {
    let digits: String = t.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
    if digits.len() >= 6 {
        format!("{}:{}:{}", &digits[0..2], &digits[2..4], &digits[4..])
    } else {
        digits
    }
}

fn insert_str(obj: &mut Map<String, Value>, key: &str, val: &str) {
    if !val.is_empty() {
        obj.insert(key.into(), json!(val));
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}
