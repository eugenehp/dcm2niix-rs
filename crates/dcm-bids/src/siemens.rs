//! Sequential Siemens BIDS sidecar matching `nii_SaveBIDSX` field order/format.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use dcm_core::error::{Error, Result};
use dcm_core::format_printf_g_f64;
use dcm_dicom::DicomImage;
use dcm_nifti::Nifti1Header;

use crate::Anonymize;

pub fn write_siemens_sidecar(
    path: impl AsRef<Path>,
    dcm: &DicomImage,
    hdr: &Nifti1Header,
    anon: Anonymize,
    conversion_version: &str,
) -> Result<()> {
    write_siemens_sidecar_ex(path, dcm, hdr, anon, conversion_version, true, "")
}

/// Like [`write_siemens_sidecar`], with Philips `-p` and `-c` conversion comments.
pub fn write_siemens_sidecar_ex(
    path: impl AsRef<Path>,
    dcm: &DicomImage,
    hdr: &Nifti1Header,
    anon: Anonymize,
    conversion_version: &str,
    philips_float_scaling: bool,
    conversion_comments: &str,
) -> Result<()> {
    let path = path.as_ref();
    let mut f = File::create(path).map_err(|e| Error::io(path, e))?;
    let io = |e: std::io::Error| Error::io(path, e);

    writeln!(f, "{{").map_err(io)?;

    if !dcm.modality.as_str().is_empty() {
        writeln!(f, "\t\"Modality\": \"{}\",", dcm.modality.as_str()).map_err(io)?;
    }
    if dcm.field_strength > 0.0 {
        writeln!(
            f,
            "\t\"MagneticFieldStrength\": {},",
            format_g(dcm.field_strength)
        )
        .map_err(io)?;
    }
    if dcm.imaging_frequency > 0.0 && dcm.imaging_frequency < 9_000_000.0 {
        writeln!(
            f,
            "\t\"ImagingFrequency\": {},",
            format_g_prec(dcm.imaging_frequency, 10)
        )
        .map_err(io)?;
    }
    if dcm.manufacturer != dcm_dicom::Manufacturer::Unknown {
        writeln!(
            f,
            "\t\"Manufacturer\": \"{}\",",
            dcm.manufacturer.as_str()
        )
        .map_err(io)?;
    }
    json_str(&mut f, "InternalPulseSequenceName", &dcm.internal_pulse_sequence_name)?;
    json_str(&mut f, "ManufacturersModelName", &dcm.manufacturers_model_name)?;
    json_str(&mut f, "InstitutionName", &dcm.institution_name)?;
    if dcm.institutional_department.is_empty() {
        writeln!(f, "\t\"InstitutionalDepartmentName\": \"None\",").map_err(io)?;
    } else {
        json_str(
            &mut f,
            "InstitutionalDepartmentName",
            &dcm.institutional_department,
        )?;
    }
    json_str(&mut f, "InstitutionAddress", &dcm.institution_address)?;
    json_str(&mut f, "DeviceSerialNumber", &dcm.device_serial_number)?;
    json_str(&mut f, "StationName", &dcm.station_name)?;
    // `-ba y`: strip UIDs+PII; `-ba o`: keep UIDs, strip PII; `-ba n`: keep both.
    if anon != Anonymize::Full {
        json_str(&mut f, "SeriesInstanceUID", &dcm.series_uid)?;
        json_str(&mut f, "StudyInstanceUID", &dcm.study_uid)?;
        json_str(&mut f, "StudyID", &dcm.study_id)?;
    }
    if anon == Anonymize::None {
        json_str(&mut f, "ReferringPhysicianName", &dcm.referring_physician_name)?;
        json_str(&mut f, "PatientName", &dcm.patient_name)?;
        json_str(&mut f, "PatientID", &dcm.patient_id)?;
        json_str(&mut f, "AccessionNumber", &dcm.accession_number)?;
        if dcm.patient_birth_date.len() == 8 {
            let y = &dcm.patient_birth_date[0..4];
            let m = &dcm.patient_birth_date[4..6];
            let day = &dcm.patient_birth_date[6..8];
            writeln!(f, "\t\"PatientBirthDate\": \"{y}-{m}-{day}\",").map_err(io)?;
        }
        if !dcm.patient_sex.is_empty() && dcm.patient_sex != "?" {
            json_str(&mut f, "PatientSex", &dcm.patient_sex)?;
        }
        // BIDS PatientAge is years (DICOM AS `nnnY`).
        let age = dcm.patient_age.trim();
        if age.len() > 1 && age.as_bytes().last() == Some(&b'Y') {
            if let Ok(y) = age[..age.len() - 1].trim().parse::<i32>() {
                writeln!(f, "\t\"PatientAge\": {y},").map_err(io)?;
            }
        }
        if dcm.patient_weight > 0.0 {
            writeln!(
                f,
                "\t\"PatientWeight\": {},",
                format_g(dcm.patient_weight)
            )
            .map_err(io)?;
        }
        if dcm.patient_size > 0.0 {
            writeln!(f, "\t\"PatientSize\": {},", format_g(dcm.patient_size)).map_err(io)?;
        }
    }
    if dcm.is_quadruped {
        writeln!(f, "\t\"Quadruped\": true,").map_err(io)?;
    }
    json_str(&mut f, "BodyPart", &dcm.body_part)?;
    json_str(&mut f, "PatientPosition", &dcm.patient_position_label)?;
    json_str(&mut f, "ProcedureStepDescription", &dcm.procedure_step_description)?;
    json_str(&mut f, "SoftwareVersions", &dcm.software_versions)?;
    if dcm.mra_acquisition_type.eq_ignore_ascii_case("2D") {
        writeln!(f, "\t\"MRAcquisitionType\": \"2D\",").map_err(io)?;
    } else if dcm.mra_acquisition_type.eq_ignore_ascii_case("3D") {
        writeln!(f, "\t\"MRAcquisitionType\": \"3D\",").map_err(io)?;
    }
    json_str(&mut f, "StudyDescription", &dcm.study_description)?;
    json_str(&mut f, "SeriesDescription", &dcm.series_description)?;
    json_str(&mut f, "ProtocolName", &dcm.protocol_name)?;
    if dcm.is_mrs {
        let mrs_scan = match dcm.mrs_acq_type {
            1 | 2 | 3 => "MRSI",
            0 if !dcm.resonant_nucleus.is_empty() => "SVS",
            _ => "Unlocalized MRS",
        };
        writeln!(f, "\t\"ScanningSequence\": \"{mrs_scan}\",").map_err(io)?;
    } else {
        json_str(&mut f, "ScanningSequence", &dcm.scanning_sequence)?;
    }
    json_str(&mut f, "SequenceVariant", &dcm.sequence_variant)?;
    if let Some(pst) = pulse_sequence_type(dcm) {
        writeln!(f, "\t\"PulseSequenceType\": \"{pst}\",").map_err(io)?;
    }
    json_str(&mut f, "ScanOptions", &dcm.scan_options)?;
    if dcm.sequence_name.is_empty() {
        // XA60: promote PulseSequenceName when SequenceName absent.
        json_str(&mut f, "SequenceName", &dcm.pulse_sequence_name)?;
    } else {
        json_str(&mut f, "SequenceName", &dcm.sequence_name)?;
        json_str(&mut f, "PulseSequenceName", &dcm.pulse_sequence_name)?;
    }
    write_image_type(&mut f, dcm)?;
    write_image_type_text(&mut f, dcm)?;

    let it = dcm.image_type.replace('\\', "_");
    let itt = &dcm.image_type_text;
    if it.contains("_DIS2D")
        || it.contains("_DIS3D")
        || itt.contains("_DIS2D")
        || itt.contains("_DIS3D")
    {
        writeln!(f, "\t\"NonlinearGradientCorrection\": true,").map_err(io)?;
    }
    if it.contains("_ND") || itt.contains("_ND") {
        writeln!(f, "\t\"NonlinearGradientCorrection\": false,").map_err(io)?;
    }
    if dcm.is_derived {
        writeln!(f, "\t\"RawImage\": false,").map_err(io)?;
    }
    json_str(&mut f, "DeidentificationMethod", &dcm.deidentification_method)?;
    if !dcm.deidentification_method_code_sequence.is_empty() {
        writeln!(f, "\t\"DeidentificationMethodCodeSequence\": [ ").map_err(io)?;
        let n = dcm.deidentification_method_code_sequence.len();
        for (i, cs) in dcm.deidentification_method_code_sequence.iter().enumerate() {
            writeln!(f, "\t  {{ ").map_err(io)?;
            writeln!(
                f,
                "\t\t\"CodeValue\": \"{}\",",
                escape_json(&cs.code_value)
            )
            .map_err(io)?;
            writeln!(
                f,
                "\t\t\"CodingSchemeDesignator\": \"{}\",",
                escape_json(&cs.coding_scheme_designator)
            )
            .map_err(io)?;
            writeln!(
                f,
                "\t\t\"CodingSchemeVersion\": \"{}\",",
                escape_json(&cs.coding_scheme_version)
            )
            .map_err(io)?;
            write!(
                f,
                "\t\t\"CodeMeaning\": \"{}\"\n",
                escape_json(&cs.code_meaning)
            )
            .map_err(io)?;
            if i + 1 < n {
                writeln!(f, "\t  }},").map_err(io)?;
            } else {
                writeln!(f, "\t  }}").map_err(io)?;
            }
        }
        writeln!(f, "\t],").map_err(io)?;
    }

    if dcm.series_number > 0 {
        writeln!(f, "\t\"SeriesNumber\": {},", dcm.series_number).map_err(io)?;
    }
    write_acquisition_time(&mut f, dcm, anon)?;
    if dcm.acquisition_number > 0 {
        writeln!(f, "\t\"AcquisitionNumber\": {},", dcm.acquisition_number).map_err(io)?;
    }
    // `-c` empty (`"\t"`) masks both ImageComments and ConversionComments (C++).
    let mask_comments = conversion_comments == "\t";
    if !mask_comments {
        json_str(&mut f, "ImageComments", &dcm.image_comments)?;
        json_str(&mut f, "ConversionComments", conversion_comments)?;
    }
    json_float(&mut f, "TriggerDelayTime", dcm.trigger_delay_time)?;

    if dcm.rwv_scale != 0.0 {
        writeln!(
            f,
            "\t\"PhilipsRWVSlope\": {},",
            format_g(dcm.rwv_scale)
        )
        .map_err(io)?;
        writeln!(
            f,
            "\t\"PhilipsRWVIntercept\": {},",
            format_g(dcm.rwv_intercept)
        )
        .map_err(io)?;
    }
    if !dcm.is_scale_varies_enh
        && dcm.inten_scale_philips != 0.0
        && dcm.manufacturer == dcm_dicom::Manufacturer::Philips
    {
        writeln!(
            f,
            "\t\"PhilipsRescaleSlope\": {},",
            format_g(dcm.inten_scale as f64)
        )
        .map_err(io)?;
        writeln!(
            f,
            "\t\"PhilipsRescaleIntercept\": {},",
            format_g(dcm.inten_intercept as f64)
        )
        .map_err(io)?;
        writeln!(
            f,
            "\t\"PhilipsScaleSlope\": {},",
            format_g(dcm.inten_scale_philips as f64)
        )
        .map_err(io)?;
        // Default `-p y` (precise float); value mirrors convert-time `philips_precise`.
        writeln!(
            f,
            "\t\"UsePhilipsFloatNotDisplayScaling\": {},",
            if philips_float_scaling { 1 } else { 0 }
        )
        .map_err(io)?;
    }

    // CT parameters (before MRI slice geometry).
    json_float(&mut f, "ExposureTime", dcm.exposure_time_ms / 1000.0)?;
    json_float(&mut f, "XRayTubeCurrent", dcm.x_ray_tube_current)?;
    if dcm.te > 0.0 && dcm.is_xray {
        writeln!(f, "\t\"XRayExposure\": {},", format_g(dcm.te)).map_err(io)?;
    }
    if !dcm.is_xray {
        json_float(&mut f, "SliceThickness", dcm.slice_thickness)?;
        json_float(&mut f, "SpacingBetweenSlices", dcm.spacing_between_slices)?;
    }
    json_float(&mut f, "SAR", dcm.sar)?;
    if dcm.number_of_averages > 1.0 {
        json_float(&mut f, "NumberOfAverages", dcm.number_of_averages)?;
    }
    if dcm.csa.series.averages_double > 1.0 {
        json_float(&mut f, "AveragesDouble", dcm.csa.series.averages_double)?;
    }
    let tp = dcm.csa.image.table_pos;
    if tp[0] > 0.0 {
        writeln!(
            f,
            "\t\"TablePosition\": [\n\t\t{},\n\t\t{},\n\t\t{}\t],",
            format_g(tp[1]),
            format_g(tp[2]),
            format_g(tp[3])
        )
        .map_err(io)?;
    }
    if dcm.te > 0.0 && !dcm.is_xray {
        // GE fieldmapHz: EchoTime1/EchoTime2 from velocityEncodeScale (issue 617).
        if dcm.manufacturer == dcm_dicom::Manufacturer::Ge
            && dcm.is_real_is_phase_map_hz
            && dcm.velocity_encode_scale_ge < 0.0
        {
            let te1 = dcm.te / 1000.0;
            let te2 = te1 - 1.0 / (2.0 * std::f64::consts::PI * dcm.velocity_encode_scale_ge);
            writeln!(f, "\t\"EchoTime1\": {},", format_g(te1)).map_err(io)?;
            writeln!(f, "\t\"EchoTime2\": {},", format_g(te2)).map_err(io)?;
        } else {
            writeln!(f, "\t\"EchoTime\": {},", format_g(dcm.te / 1000.0)).map_err(io)?;
        }
    }
    if dcm.echo_number > 1 || (dcm.is_multi_echo && dcm.echo_number > 0) {
        writeln!(f, "\t\"EchoNumber\": {},", dcm.echo_number).map_err(io)?;
    }
    json_float(&mut f, "RepetitionTime", dcm.tr / 1000.0)?;
    if dcm.repetition_time_excitation > 0.0 {
        writeln!(
            f,
            "\t\"RepetitionTimeExcitation\": {},",
            format_g(dcm.repetition_time_excitation)
        )
        .map_err(io)?;
    }
    if dcm.repetition_time_inversion > 0.0 {
        writeln!(
            f,
            "\t\"RepetitionTimeInversion\": {},",
            format_g(dcm.repetition_time_inversion)
        )
        .map_err(io)?;
    }
    if dcm.is_3d_acq
        && dcm.csa.image.bandwidth_per_pixel_phase_encode > 0.0
        && dcm.number_of_concatenations > 1
    {
        writeln!(
            f,
            "\t\"MultiEchoShots\": {},",
            dcm.number_of_concatenations
        )
        .map_err(io)?;
    }
    if dcm.ti > 0.0 {
        writeln!(f, "\t\"InversionTime\": {},", format_g(dcm.ti / 1000.0)).map_err(io)?;
    }
    json_float(&mut f, "FlipAngle", dcm.flip_angle)?;
    if dcm.is_variable_flip_angle {
        writeln!(f, "\t\"VariableFlipAngleFlag\": true,").map_err(io)?;
    }
    if dcm.phase_number > 0 {
        writeln!(f, "\t\"PhaseNumber\": {},", dcm.phase_number).map_err(io)?;
    }

    // MTState / Spoiling (0018,9020 / 0018,9016 / CSA / SequenceVariant).
    let mut mt = dcm.mt_state;
    if mt < 0 && dcm.csa.series.uc_mtc == 1 {
        mt = 1;
    }
    if mt == 0 {
        writeln!(f, "\t\"MTState\": false,").map_err(io)?;
    } else if mt > 0 {
        writeln!(f, "\t\"MTState\": true,").map_err(io)?;
    }
    let mut is_spoiled = dcm.spoiling > 0;
    if dcm.spoiling == 0 {
        writeln!(f, "\t\"SpoilingState\": false,").map_err(io)?;
    }
    let var = dcm.sequence_variant.as_str();
    if dcm.spoiling < 0
        && (var.starts_with("SP\\") || var == "SP" || var.contains("\\SP"))
    {
        is_spoiled = true;
    }
    if is_spoiled {
        writeln!(f, "\t\"SpoilingState\": true,").map_err(io)?;
    }
    match dcm.spoiling {
        1 => writeln!(f, "\t\"SpoilingType\": \"RF\",").map_err(io)?,
        2 => writeln!(f, "\t\"SpoilingType\": \"GRADIENT\",").map_err(io)?,
        3 => writeln!(f, "\t\"SpoilingType\": \"COMBINED\",").map_err(io)?,
        _ => {}
    }
    if dcm.interp_3d > 1 {
        writeln!(f, "\t\"Interpolation3D\": {},", dcm.interp_3d).map_err(io)?;
    }
    json_float(&mut f, "WaterFatShift", dcm.water_fat_shift)?;
    if dcm.manufacturer == dcm_dicom::Manufacturer::Philips
        && dcm.water_fat_shift != 0.0
        && dcm.imaging_frequency > 0.0
        && dcm.echo_train_length > 0
    {
        let recon_pe = if hdr.dim[1] == hdr.dim[2] && hdr.dim[2] > 0 {
            hdr.dim[2] as i32
        } else if dcm.phase_encoding_rc == 'C' {
            hdr.dim[2] as i32
        } else if dcm.phase_encoding_rc == 'R' {
            hdr.dim[1] as i32
        } else {
            dcm.acquisition_matrix_pe
        };
        if recon_pe > 1 {
            let actual_es = dcm.water_fat_shift
                / (dcm.imaging_frequency * 3.4 * (dcm.echo_train_length as f64 + 1.0));
            let total_ro = actual_es * dcm.echo_train_length as f64;
            writeln!(
                f,
                "\t\"EstimatedEffectiveEchoSpacing\": {},",
                format_g(total_ro / (recon_pe as f64 - 1.0))
            )
            .map_err(io)?;
            writeln!(
                f,
                "\t\"EstimatedTotalReadoutTime\": {},",
                format_g(total_ro)
            )
            .map_err(io)?;
        }
    }

    if dcm.manufacturer == dcm_dicom::Manufacturer::Ge {
        write_ge_fields(&mut f, dcm, hdr)?;
    }
    if dcm.manufacturer == dcm_dicom::Manufacturer::Uih {
        json_str(&mut f, "PhaseEncodingDirectionDisplayed", &dcm.pe_direction_displayed)?;
    }

    let s = &dcm.csa.series;
    if dcm.manufacturer == dcm_dicom::Manufacturer::Siemens {
    if s.partial_fourier > 0 {
        let pf = match s.partial_fourier {
            1 => 0.5,
            2 => 0.625,
            4 => 0.75,
            8 => 0.875,
            _ => 1.0,
        };
        if pf < 1.0 {
            writeln!(f, "\t\"PartialFourier\": {},", format_g(pf)).map_err(io)?;
        }
    }
    if s.interp > 0 {
        writeln!(f, "\t\"Interpolation2D\": 1,").map_err(io)?;
    }
    if (s.dif_bipolar == 1 || s.dif_bipolar == 2) && dcm.csa.image.num_dti > 0 {
        if s.dif_bipolar == 1 {
            writeln!(f, "\t\"DiffusionScheme\": \"Bipolar\",").map_err(io)?;
        } else {
            writeln!(f, "\t\"DiffusionScheme\": \"Monopolar\",").map_err(io)?;
        }
    }
    if s.base_resolution > 0 {
        writeln!(f, "\t\"BaseResolution\": {},", s.base_resolution).map_err(io)?;
    }
    if s.shim_setting[0] != 0.0 {
        write!(f, "\t\"ShimSetting\": [\n").map_err(io)?;
        for (i, v) in s.shim_setting.iter().enumerate() {
            if i != 0 {
                writeln!(f, ",").map_err(io)?;
            }
            write!(f, "\t\t{}", format_g(*v)).map_err(io)?;
        }
        writeln!(f, "\t],").map_err(io)?;
    }
    json_float(&mut f, "DelayTime", s.delay_time_s)?;
    json_float(&mut f, "TxRefAmp", s.tx_ref_amp)?;
    json_float(&mut f, "PhaseResolution", s.phase_resolution)?;
    json_float(&mut f, "PhaseOversampling", s.phase_oversampling)?;
    json_float(&mut f, "VendorReportedEchoSpacing", s.echo_spacing_us as f64 / 1_000_000.0)?;
    json_str(&mut f, "ReceiveCoilName", &s.coil_id)?;
    if s.coil_id.is_empty() {
        json_str(&mut f, "ReceiveCoilName", &dcm.coil_name)?;
    }
    json_str(&mut f, "ReceiveCoilActiveElements", &s.coil_string)?;
    if s.coil_string != dcm.coil_string {
        json_str(&mut f, "CoilString", &dcm.coil_string)?;
    }
    json_str(&mut f, "PulseSequenceDetails", &s.pulse_sequence_details)?;
    json_str(&mut f, "FmriExternalInfo", &s.fmri_external_info)?;
    json_str(&mut f, "WipMemBlock", &s.wip_mem_block)?;
    write_siemens_asl(&mut f, dcm, hdr)?;
    if s.ref_lines_pe > 0 {
        writeln!(f, "\t\"RefLinesPE\": {},", s.ref_lines_pe).map_err(io)?;
    }
    if s.combine_mode == 1 {
        writeln!(f, "\t\"CoilCombinationMethod\": \"Sum of Squares\",").map_err(io)?;
    } else if s.combine_mode == 2 {
        writeln!(f, "\t\"CoilCombinationMethod\": \"Adaptive Combine\",").map_err(io)?;
    }
    json_str(&mut f, "ConsistencyInfo", &s.consistency_info)?;
    if s.parallel_reduction_factor_in_plane > 0 {
        match s.pat_mode {
            1 => writeln!(f, "\t\"MatrixCoilMode\": \"SENSE\",").map_err(io)?,
            2 => writeln!(f, "\t\"MatrixCoilMode\": \"GRAPPA\",").map_err(io)?,
            _ => writeln!(f, "\t\"MatrixCoilMode\": \"None\",").map_err(io)?,
        }
    } else {
        writeln!(f, "\t\"MatrixCoilMode\": \"None\",").map_err(io)?;
    }
    if dcm.csa.image.multi_band_factor > 1 {
        writeln!(
            f,
            "\t\"MultibandAccelerationFactor\": {},",
            dcm.csa.image.multi_band_factor
        )
        .map_err(io)?;
    }
    } // Siemens-only CSA extras

    if dcm.manufacturer != dcm_dicom::Manufacturer::Ge {
    json_float(&mut f, "PercentPhaseFOV", dcm.percent_phase_fov)?;
    json_float(&mut f, "PercentSampling", dcm.percent_sampling)?;
    if dcm.echo_train_length > 1 && !dcm.is_3d_acq {
        writeln!(f, "\t\"EchoTrainLength\": {},", dcm.echo_train_length).map_err(io)?;
    }
    match dcm.partial_fourier_direction {
        1 => writeln!(f, "\t\"PartialFourierDirection\": \"PHASE\",").map_err(io)?,
        2 => writeln!(f, "\t\"PartialFourierDirection\": \"FREQUENCY\",").map_err(io)?,
        3 => writeln!(f, "\t\"PartialFourierDirection\": \"SLICE_SELECT\",").map_err(io)?,
        4 => writeln!(f, "\t\"PartialFourierDirection\": \"COMBINATION\",").map_err(io)?,
        _ => {}
    }
    if dcm.phase_encoding_steps > 0
        && dcm.is_partial_fourier
        && dcm.manufacturer == dcm_dicom::Manufacturer::Philips
    {
        // issue 377
        writeln!(f, "\t\"PartialFourierEnabled\": \"YES\",").map_err(io)?;
        writeln!(
            f,
            "\t\"PhaseEncodingStepsNoPartialFourier\": {},",
            dcm.phase_encoding_steps
        )
        .map_err(io)?;
    } else if dcm.phase_encoding_steps > 0 {
        writeln!(
            f,
            "\t\"PhaseEncodingSteps\": {},",
            dcm.phase_encoding_steps
        )
        .map_err(io)?;
    } else if s.phase_encoding_lines > 0 {
        writeln!(f, "\t\"PhaseEncodingSteps\": {},", s.phase_encoding_lines).map_err(io)?;
    }
    if dcm.frequency_encoding_steps > 0 {
        writeln!(
            f,
            "\t\"FrequencyEncodingSteps\": {},",
            dcm.frequency_encoding_steps
        )
        .map_err(io)?;
    }
    if dcm.phase_encoding_steps_out_of_plane > 0 {
        writeln!(
            f,
            "\t\"PhaseEncodingStepsOutOfPlane\": {},",
            dcm.phase_encoding_steps_out_of_plane
        )
        .map_err(io)?;
    }

    // Prefer DICOM AcquisitionMatrix PE lines; fall back to CSA.
    let pe_lines = if dcm.acquisition_matrix_pe > 0 {
        dcm.acquisition_matrix_pe
    } else if s.phase_encoding_lines > 0 {
        s.phase_encoding_lines
    } else {
        0
    };
    if pe_lines > 0 {
        writeln!(f, "\t\"AcquisitionMatrixPE\": {},", pe_lines).map_err(io)?;
    }
    let mut recon_pe = pe_lines;
    if dcm.manufacturer == dcm_dicom::Manufacturer::Uih {
        let mosaic = dcm.is_mosaic || dcm.csa.image.mosaic_slices > 1;
        if mosaic && hdr.dim[1] > 0 && hdr.dim[2] > 0 {
            if hdr.dim[1] == hdr.dim[2] {
                recon_pe = hdr.dim[2] as i32;
            } else if dcm.phase_encoding_rc == 'C' {
                recon_pe = hdr.dim[2] as i32;
            } else if dcm.phase_encoding_rc == 'R' {
                recon_pe = hdr.dim[1] as i32;
            }
        } else {
            let pre_w = dcm.columns as i32;
            let pre_h = dcm.rows as i32;
            if pre_w > 0 && pre_h > 0 {
                if pre_w == pre_h {
                    recon_pe = pre_w;
                } else if dcm.phase_encoding_rc == 'C' {
                    recon_pe = pre_h;
                } else if dcm.phase_encoding_rc == 'R' {
                    recon_pe = pre_w;
                }
            }
        }
    } else if hdr.dim[1] > 0 && hdr.dim[2] > 0 {
        if hdr.dim[1] == hdr.dim[2] {
            recon_pe = hdr.dim[2] as i32;
        } else if dcm.phase_encoding_rc == 'C' {
            recon_pe = hdr.dim[2] as i32;
        } else if dcm.phase_encoding_rc == 'R' {
            recon_pe = hdr.dim[1] as i32;
        }
    }
    if recon_pe > 0 {
        writeln!(f, "\t\"ReconMatrixPE\": {},", recon_pe).map_err(io)?;
    }

    let bw_pp = if dcm.csa.image.bandwidth_per_pixel_phase_encode > 0.0 {
        dcm.csa.image.bandwidth_per_pixel_phase_encode
    } else {
        0.0
    };
    json_float(&mut f, "BandwidthPerPixelPhaseEncode", bw_pp)?;

    let accel_pe = if s.parallel_reduction_factor_in_plane as f64 >= 1.0 {
        s.parallel_reduction_factor_in_plane as f64
    } else {
        dcm.accel_fact_pe
    };
    if accel_pe >= 1.0 {
        writeln!(
            f,
            "\t\"ParallelReductionFactorInPlane\": {},",
            format_g(accel_pe)
        )
        .map_err(io)?;
    }
    json_str(
        &mut f,
        "ParallelAcquisitionTechnique",
        &dcm.parallel_acquisition_technique,
    )?;
    let accel_oop = if dcm.parallel_reduction_out_of_plane >= 1.0 {
        dcm.parallel_reduction_out_of_plane
    } else {
        s.parallel_reduction_factor_out_of_plane as f64
    };
    if accel_oop >= 1.0 {
        writeln!(
            f,
            "\t\"ParallelReductionFactorOutOfPlane\": {},",
            format_g(accel_oop)
        )
        .map_err(io)?;
    }
    if dcm.compressed_sensing_factor > 1.0 {
        writeln!(
            f,
            "\t\"CompressedSensingFactor\": {},",
            format_g(dcm.compressed_sensing_factor)
        )
        .map_err(io)?;
    }
    if dcm.is_deep_learning {
        writeln!(f, "\t\"DeepLearning\": true,").map_err(io)?;
        json_str(&mut f, "DeepLearningDetails", &dcm.deep_learning_text)?;
    }

    let mut effective_echo_spacing = 0.0;
    if recon_pe > 0 && bw_pp > 0.0 {
        effective_echo_spacing = 1.0 / (bw_pp * recon_pe as f64);
        writeln!(
            f,
            "\t\"EffectiveEchoSpacing\": {},",
            format_g(effective_echo_spacing)
        )
        .map_err(io)?;
        let mut true_es_factor = 1.0;
        if accel_pe > 1.0 {
            true_es_factor /= accel_pe;
        }
        if s.phase_oversampling > 0.0 {
            true_es_factor *= 1.0 + s.phase_oversampling;
        }
        let derived = 1.0 / (bw_pp * true_es_factor * recon_pe as f64);
        writeln!(
            f,
            "\t\"DerivedVendorReportedEchoSpacing\": {},",
            format_g(derived)
        )
        .map_err(io)?;
        writeln!(
            f,
            "\t\"TotalReadoutTime\": {},",
            format_g(effective_echo_spacing * (recon_pe as f64 - 1.0))
        )
        .map_err(io)?;
    } else if dcm.manufacturer == dcm_dicom::Manufacturer::Uih && dcm.acquisition_duration > 0.0 {
        writeln!(
            f,
            "\t\"TotalReadoutTime\": {},",
            format_g(dcm.acquisition_duration / 1000.0)
        )
        .map_err(io)?;
    }
    let _ = effective_echo_spacing;
    json_float(&mut f, "PixelBandwidth", dcm.pixel_bandwidth)?;
    // C++: emit AcquisitionDuration for all vendors except UIH (UIH uses it as TotalReadoutTime).
    if dcm.manufacturer != dcm_dicom::Manufacturer::Uih {
        json_float(&mut f, "AcquisitionDuration", dcm.acquisition_duration)?;
    }
    if dcm.number_of_k_space_trajectories > 0 {
        writeln!(
            f,
            "\t\"NumberOfKSpaceTrajectories\": {},",
            dcm.number_of_k_space_trajectories
        )
        .map_err(io)?;
    }

    let dwell_ns = if dcm.csa.image.real_dwell_time_ns > 0.0 {
        dcm.csa.image.real_dwell_time_ns
    } else {
        dcm.dwell_time_ns
    };
    if dwell_ns > 0.0 {
        writeln!(f, "\t\"DwellTime\": {},", format_g(dwell_ns * 1e-9)).map_err(io)?;
    }

    let ph_pos = dcm.csa.image.phase_encoding_direction_positive;
    if dcm.manufacturer == dcm_dicom::Manufacturer::Uih && !dcm.is_3d_acq {
        if dcm.phase_encoding_rc == 'C' {
            writeln!(f, "\t\"PhaseEncodingAxis\": \"j\",").map_err(io)?;
        } else if dcm.phase_encoding_rc == 'R' {
            writeln!(f, "\t\"PhaseEncodingAxis\": \"i\",").map_err(io)?;
        }
    } else if dcm.manufacturer != dcm_dicom::Manufacturer::Uih {
        // Siemens / others: 3D normally skips PE (issue 849 SPACE/TSE); restore for
        // non-SE ETL>1 and for 3D EPI with bandwidth (issue 1024).
        let scan_u = dcm.scanning_sequence.to_ascii_uppercase();
        let mut skip_pe = dcm.is_3d_acq;
        if dcm.echo_train_length > 1 && !scan_u.contains("SE") {
            skip_pe = false;
        }
        if dcm.is_3d_acq
            && dcm.csa.image.bandwidth_per_pixel_phase_encode > 0.0
            && !scan_u.contains("SE")
        {
            skip_pe = false;
        }
        if !skip_pe {
            if (dcm.phase_encoding_rc == 'R' || dcm.phase_encoding_rc == 'C') && ph_pos < 0 {
                if dcm.phase_encoding_rc == 'C' {
                    writeln!(f, "\t\"PhaseEncodingAxis\": \"j\",").map_err(io)?;
                } else {
                    writeln!(f, "\t\"PhaseEncodingAxis\": \"i\",").map_err(io)?;
                }
            }
            if (dcm.phase_encoding_rc == 'R' || dcm.phase_encoding_rc == 'C') && ph_pos >= 0 {
                let axis = if dcm.phase_encoding_rc == 'C' { "j" } else { "i" };
                let mut suffix = String::new();
                if ph_pos == 0 && dcm.phase_encoding_rc != 'C' {
                    suffix.push('-');
                } else if dcm.phase_encoding_rc == 'C' && ph_pos == 1 {
                    suffix.push('-');
                }
                writeln!(f, "\t\"PhaseEncodingDirection\": \"{axis}{suffix}\",").map_err(io)?;
            }
        }
    }
    } // not GE

    let nz = hdr.dim[3] as usize;
    let st = &dcm.csa.image.slice_timing_ms;
    if nz > 1 && st.len() >= nz && st[0] >= 0.0 {
        write!(f, "\t\"SliceTiming\": [\n").map_err(io)?;
        for (i, t) in st.iter().take(nz).enumerate() {
            if i != 0 {
                writeln!(f, ",").map_err(io)?;
            }
            write!(
                f,
                "\t\t{}",
                format_printf_g_f64(((*t as f32) / 1000.0) as f64)
            )
            .map_err(io)?;
        }
        writeln!(f, "\t],").map_err(io)?;
    }

    if dcm.has_orientation() {
        write!(f, "\t\"ImageOrientationPatientDICOM\": [\n").map_err(io)?;
        for i in 1..7 {
            if i != 1 {
                writeln!(f, ",").map_err(io)?;
            }
            write!(f, "\t\t{}", format_g(dcm.orient[i])).map_err(io)?;
        }
        writeln!(f, "\t],").map_err(io)?;
    }
    json_str(&mut f, "ImageOrientationText", &dcm.image_orientation_text)?;
    match dcm.phase_encoding_rc {
        'C' => writeln!(f, "\t\"InPlanePhaseEncodingDirectionDICOM\": \"COL\",").map_err(io)?,
        'R' => writeln!(f, "\t\"InPlanePhaseEncodingDirectionDICOM\": \"ROW\",").map_err(io)?,
        _ => {}
    }

    // PET isotope / correction module (nii_SaveBIDSX).
    if dcm.modality == dcm_dicom::Modality::Pt
        || !dcm.radiopharmaceutical.is_empty()
        || dcm.radionuclide_total_dose > 0.0
    {
        json_str(&mut f, "TracerName", &dcm.radiopharmaceutical)?;
        json_str(&mut f, "TracerRadionuclide", &dcm.tracer_radionuclide)?;
        if dcm.radionuclide_positron_fraction > 0.0 {
            writeln!(
                f,
                "\t\"RadionuclidePositronFraction\": {},",
                format_g(dcm.radionuclide_positron_fraction)
            )
            .map_err(io)?;
        }
        if dcm.radionuclide_total_dose > 0.0 {
            writeln!(
                f,
                "\t\"InjectedRadioactivity\": {},",
                format_g(dcm.radionuclide_total_dose / 1.0e6)
            )
            .map_err(io)?;
            writeln!(f, "\t\"InjectedRadioactivityUnits\": \"MBq\",").map_err(io)?;
        }
        if dcm.radiopharmaceutical_specific_activity > 0.0 {
            writeln!(
                f,
                "\t\"MolarActivity\": {},",
                format_g(dcm.radiopharmaceutical_specific_activity)
            )
            .map_err(io)?;
            writeln!(f, "\t\"MolarActivityUnits\": \"Bq/umol\",").map_err(io)?;
        }
        json_float(&mut f, "InjectedVolume", dcm.injected_volume)?;
        json_float(&mut f, "RadionuclideHalfLife", dcm.radionuclide_half_life)?;
        json_float(&mut f, "DoseCalibrationFactor", dcm.dose_calibration_factor)?;
        json_float(&mut f, "IsotopeHalfLife", dcm.ecat_isotope_halflife)?;
        json_float(&mut f, "Dosage", dcm.ecat_dosage)?;
        json_str(&mut f, "DecayCorrection", &dcm.decay_correction)?;
        json_str(
            &mut f,
            "AttenuationCorrectionMethod",
            &dcm.attenuation_correction_method,
        )?;
        if !dcm.attenuation_correction_method.is_empty() {
            let token = dcm
                .attenuation_correction_method
                .split(',')
                .next()
                .unwrap_or("")
                .trim();
            if !token.is_empty() {
                writeln!(f, "\t\"AttenuationCorrection\": \"{token}\",").map_err(io)?;
            }
        }
        json_str(&mut f, "ReconstructionMethod", &dcm.reconstruction_method)?;
        json_str(&mut f, "RandomsCorrectionMethod", &dcm.randoms_correction_method)?;
        json_str(&mut f, "ScatterCorrectionMethod", &dcm.scatter_correction_method)?;
        emit_recon_method_name(&mut f, &dcm.reconstruction_method)?;
        json_str(&mut f, "ConvolutionKernel", &dcm.convolution_kernel)?;
        emit_recon_filter(&mut f, dcm)?;
        json_str(&mut f, "Units", &dcm.units_pt)?;
        // BIDS-PET types ScatterFraction as array (not scalar).
        if dcm.scatter_fraction > 0.0 {
            writeln!(
                f,
                "\t\"ScatterFraction\": [{}],",
                format_g(dcm.scatter_fraction)
            )
            .map_err(io)?;
        }
        emit_pet_series_times(&mut f, dcm)?;
    }

    if dcm.frame_duration > 0.0 && dcm.frame_durations.is_empty() {
        writeln!(
            f,
            "\t\"FrameDuration\": {},",
            format_g(dcm.frame_duration / 1000.0)
        )
        .map_err(io)?;
    }
    if !dcm.frame_durations.is_empty() && hdr.dim[4] > 1 {
        write!(f, "\t\"FrameDuration\": [\n").map_err(io)?;
        for (i, d) in dcm.frame_durations.iter().take(hdr.dim[4] as usize).enumerate() {
            if i != 0 {
                writeln!(f, ",").map_err(io)?;
            }
            let sec = if *d > 0.0 {
                d / 1000.0
            } else if dcm.tr > 0.0 {
                dcm.tr / 1000.0
            } else {
                0.0
            };
            write!(f, "\t\t{}", format_g(sec)).map_err(io)?;
        }
        writeln!(f, "\t],").map_err(io)?;
    }
    if !dcm.volume_onset_times.is_empty()
        && dcm.volume_onset_times[0] >= 0.0
        && (hdr.dim[4] > 1 || dcm.volume_onset_times.len() == 1)
    {
        write!(f, "\t\"FrameTimesStart\": [\n").map_err(io)?;
        for (i, t) in dcm
            .volume_onset_times
            .iter()
            .take(hdr.dim[4].max(1) as usize)
            .enumerate()
        {
            if i != 0 {
                writeln!(f, ",").map_err(io)?;
            }
            if *t < 0.0 {
                break;
            }
            write!(f, "\t\t{}", format_g(*t)).map_err(io)?;
        }
        writeln!(f, "\t],").map_err(io)?;
    } else if dcm.decay_factor > 0.0 && hdr.dim[4] <= 1 {
        // Issue 983 / BEP009: single-volume PET without precomputed onsets.
        let series_sec = parse_dicom_time_str_sec(&dcm.series_time);
        let acq_sec = parse_dicom_time_str_sec(&dcm.acquisition_time);
        let mut t_start = 0.0;
        if series_sec >= 0.0 && acq_sec >= 0.0 {
            t_start = (acq_sec - series_sec).max(0.0);
        }
        writeln!(
            f,
            "\t\"FrameTimesStart\": [\n\t\t{}\t],",
            format_g(t_start)
        )
        .map_err(io)?;
    }
    if !dcm.decay_factors.is_empty() && dcm.decay_factors[0] >= 0.0 {
        write!(f, "\t\"DecayCorrectionFactor\": [\n").map_err(io)?;
        for (i, d) in dcm.decay_factors.iter().take(hdr.dim[4] as usize).enumerate() {
            if i != 0 {
                writeln!(f, ",").map_err(io)?;
            }
            if *d < 0.0 {
                break;
            }
            write!(f, "\t\t{}", format_g(*d)).map_err(io)?;
        }
        writeln!(f, "\t],").map_err(io)?;
    } else if dcm.decay_factor > 0.0 {
        writeln!(f, "\t\"DecayFactor\": [\n\t\t{}\t],", format_g(dcm.decay_factor)).map_err(io)?;
    }
    if dcm.frame_reference_time >= 0.0 && dcm.frame_reference_times.is_empty() {
        writeln!(
            f,
            "\t\"FrameReferenceTime\": {},",
            format_g(dcm.frame_reference_time / 1000.0)
        )
        .map_err(io)?;
    }
    if dcm.frame_reference_times.len() > 1 {
        let varies = dcm.frame_reference_times.windows(2).any(|w| w[0] != w[1]);
        if varies {
            write!(f, "\t\"FrameReferenceTime\": [\n").map_err(io)?;
            for (i, t) in dcm
                .frame_reference_times
                .iter()
                .take(hdr.dim[4] as usize)
                .enumerate()
            {
                if i != 0 {
                    writeln!(f, ",").map_err(io)?;
                }
                if *t < 0.0 {
                    break;
                }
                write!(f, "\t\t{}", format_g(t / 1000.0)).map_err(io)?;
            }
            writeln!(f, "\t],").map_err(io)?;
        }
    }

    write_generic_asl(&mut f, dcm)?;

    if dcm.is_mrs {
        // BIDS-MRS extras also written into NIfTI extension; mirror in sidecar.
        if matches!(dcm.mrs_acq_type, 0) {
            // SVS keeps dim_5; MRSI omits (C++).
            writeln!(f, "\t\"dim_5\": \"DIM_DYN\",").map_err(io)?;
        }
        if dcm.data_point_columns > 0 {
            writeln!(
                f,
                "\t\"NumberOfSpectralPoints\": {},",
                dcm.data_point_columns
            )
            .map_err(io)?;
        }
        if dcm.xyz_mm[1] > 0.0 && dcm.xyz_mm[2] > 0.0 && dcm.slice_thickness > 0.0 {
            // F1 pixdim-mirror: x = xyz_mm[2], y = xyz_mm[1], z = slice_thickness.
            writeln!(
                f,
                "\t\"AcquisitionVoxelSize\": [{}, {}, {}],",
                format_g(dcm.xyz_mm[2]),
                format_g(dcm.xyz_mm[1]),
                format_g(dcm.slice_thickness)
            )
            .map_err(io)?;
        }
        if let Some(voi) = dcm.mrs_voi_matrix() {
            // %.17g for full double round-trip (C++ mrsVoiMatrix / BIDS-MRS).
            let g = |v: f64| format_g_prec(v, 17);
            writeln!(
                f,
                "\t\"VOI\": [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [0.0, 0.0, 0.0, 1.0]],",
                g(voi[0][0]),
                g(voi[0][1]),
                g(voi[0][2]),
                g(voi[0][3]),
                g(voi[1][0]),
                g(voi[1][1]),
                g(voi[1][2]),
                g(voi[1][3]),
                g(voi[2][0]),
                g(voi[2][1]),
                g(voi[2][2]),
                g(voi[2][3]),
            )
            .map_err(io)?;
        }
        let avg = if dcm.number_of_averages > 0.0 {
            dcm.number_of_averages as i32
        } else {
            1
        };
        let n_dyn = if hdr.dim[0] >= 5 && hdr.dim[5] > 0 {
            hdr.dim[5] as i32
        } else {
            1
        };
        let transients = avg * n_dyn;
        if transients > 0 {
            writeln!(f, "\t\"NumberOfTransients\": {transients},").map_err(io)?;
        }
        json_str(&mut f, "TransmitCoilName", &dcm.transmit_coil_name)?;
        writeln!(
            f,
            "\t\"WaterSuppressed\": {},",
            if dcm.is_mrs_ref { "false" } else { "true" }
        )
        .map_err(io)?;
        let mrs_type = match dcm.mrs_acq_type {
            1 => "ROW",
            2 => "PLANE",
            3 => "VOLUME",
            _ => "SINGLE_VOXEL",
        };
        writeln!(f, "\t\"MRSpectroscopyAcquisitionType\": \"{mrs_type}\",").map_err(io)?;
        if !dcm.resonant_nucleus.is_empty() {
            writeln!(
                f,
                "\t\"ResonantNucleus\": [\"{}\"],",
                escape_json(&dcm.resonant_nucleus)
            )
            .map_err(io)?;
        }
        if dcm.imaging_frequency > 0.0 {
            writeln!(
                f,
                "\t\"SpectrometerFrequency\": [{}],",
                format_g(dcm.imaging_frequency)
            )
            .map_err(io)?;
        }
    }

    if !dcm.csa.bids_data_type.is_empty() && !dcm.csa.bids_entity_suffix.is_empty() {
        writeln!(
            f,
            "\t\"BidsGuess\": [\"{}\",\"{}\"],",
            dcm.csa.bids_data_type, dcm.csa.bids_entity_suffix
        )
        .map_err(io)?;
    }

    writeln!(f, "\t\"ConversionSoftware\": \"dcm2niix\",").map_err(io)?;
    writeln!(
        f,
        "\t\"ConversionSoftwareVersion\": \"{conversion_version}\""
    )
    .map_err(io)?;
    writeln!(f, "}}").map_err(io)?;
    Ok(())
}

fn write_siemens_asl(f: &mut File, dcm: &DicomImage, hdr: &Nifti1Header) -> Result<()> {
    let s = &dcm.csa.series;
    let psd = s.pulse_sequence_details.as_str();
    let mut is_pcasl = false;
    let mut is_pasl = false;
    let mut n_pld = 0i32;
    // C++ stores RepetitionTimePreparation as TR in milliseconds for these sequences.
    let mut repetition_time_preparation = 0.0f64;

    if psd.contains("ep2d_asl") {
        json_float_nan(f, "LabelingDuration", s.labeling_duration_us / 1_000_000.0)?;
        json_float_nan(f, "PostLabelingDelay", s.post_labeling_delay_us / 1_000_000.0)?;
    }
    if psd.contains("ep2d_pcasl") || psd.contains("ep2d_pcasl_UI_PHC") {
        is_pcasl = true;
        repetition_time_preparation = dcm.tr;
        json_float_nan(f, "LabelingDistance", s.ad_free[1])?;
        json_float_nan(f, "PostLabelingDelay", s.ad_free[2] / 1_000_000.0)?;
        let num_rf = s.ad_free[3];
        json_float_nan(f, "NumRFBlocks", num_rf)?;
        if num_rf.is_finite() && num_rf > 0.0 {
            json_float_nan(f, "LabelingDuration", (0.92 * 20.0 * num_rf) / 1000.0)?;
        }
        json_float_nan(f, "RFGap", s.ad_free[4] / 1_000_000.0)?;
        json_float_nan(f, "MeanGzx10", s.ad_free[10])?;
        json_float_nan(f, "PhiAdjust", s.ad_free[11])?;
    }
    if psd.contains("tgse_pcasl") {
        is_pcasl = true;
        repetition_time_preparation = dcm.tr;
        json_float_nan(f, "LabelingDuration", s.ad_free[2] / 1_000_000.0)?;
        json_float_nan(f, "RFGap", s.ad_free[4] / 1_000_000.0)?;
        json_float_nan(f, "MeanGzx10", s.ad_free[10])?;
        json_float_nan(f, "T1", s.ad_free[12] / 1_000_000.0)?;
        json_float_nan(f, "NumRFBlocks", s.ad_free[3])?;
    }
    if psd.contains("ep2d_pasl") {
        is_pasl = true;
        writeln!(f, "\t\"PASLType\": \"PICORE\",").map_err(|e| Error::convert(e.to_string()))?;
        json_float_nan(f, "BolusDuration", s.al_ti[0] / 1_000_000.0)?;
        json_float_nan(f, "InversionTime", s.al_ti[2] / 1_000_000.0)?;
    }
    if psd.contains("tgse_pasl") {
        is_pasl = true;
        writeln!(f, "\t\"PASLType\": \"FAIR QII\",").map_err(|e| Error::convert(e.to_string()))?;
        json_float_nan(f, "BolusDuration", s.al_ti[0] / 1_000_000.0)?;
        json_float_nan(f, "InversionTime", s.al_ti[2] / 1_000_000.0)?;
    }
    if psd.contains("ep2d_fairest") {
        is_pasl = true;
        writeln!(f, "\t\"PASLType\": \"FAIR\",").map_err(|e| Error::convert(e.to_string()))?;
        json_float_nan(f, "PostInversionDelay", s.ad_free[2] / 1000.0)?;
        json_float_nan(f, "PostLabelingDelay", s.ad_free[4] / 1000.0)?;
    }

    let mut is_oxford = false;
    if psd.contains("to_ep2d_VEPCASL") {
        is_oxford = true;
        is_pcasl = true;
        repetition_time_preparation = dcm.tr;
        json_float_nan(f, "InversionTime", s.al_ti[2] / 1_000_000.0)?;
        json_float_nan(f, "BolusDuration", s.al_ti[0] / 1_000_000.0)?;
        json_float(f, "LabelingPulseFlipAngle", s.al_free[4])?;
        json_float(f, "LabelingPulseDuration", s.al_free[5] / 1_000_000.0)?;
        json_float(f, "TagRFSeparation", s.al_free[6] / 1_000_000.0)?;
        json_float_nan(f, "LabelingPulseAverageGradient", s.ad_free[0])?;
        json_float_nan(f, "LabelingPulseMaximumGradient", s.ad_free[1])?;
        json_float(f, "TagDuration", s.al_free[9] / 1000.0)?;
        json_float(f, "MaximumT1Opt", s.al_free[10] / 1000.0)?;
        let mut valid = true;
        for k in 11..31 {
            if !s.al_free[k].is_finite() || s.al_free[k] <= 0.0 {
                valid = false;
            }
            if valid {
                n_pld += 1;
            }
        }
        if n_pld > 0 {
            write!(f, "\t\"PostLabelingDelay\": [\n").map_err(|e| Error::convert(e.to_string()))?;
            for i in 0..n_pld as usize {
                if i != 0 {
                    writeln!(f, ",").map_err(|e| Error::convert(e.to_string()))?;
                }
                write!(f, "\t\t{}", format_g(s.al_free[i + 11] / 1000.0))
                    .map_err(|e| Error::convert(e.to_string()))?;
            }
            writeln!(f, "\t],").map_err(|e| Error::convert(e.to_string()))?;
        }
        for k in 3..11 {
            if s.ad_free[k].is_finite() {
                writeln!(
                    f,
                    "\t\"sWipMemBlockAdFree{k}\": {},",
                    format_g(s.ad_free[k])
                )
                .map_err(|e| Error::convert(e.to_string()))?;
            }
        }
    }
    if psd.contains("jw_tgse_VEPCASL") {
        is_pcasl = true;
        is_oxford = true;
        json_float(f, "TagRFFlipAngle", s.al_free[6])?;
        json_float(f, "TagRFDuration", s.al_free[7] / 1_000_000.0)?;
        json_float(f, "TagRFSeparation", s.al_free[8] / 1_000_000.0)?;
        json_float(f, "MaximumT1Opt", s.al_free[9] / 1000.0)?;
        json_float(f, "Tag0", s.al_free[10] / 1000.0)?;
        json_float(f, "Tag1", s.al_free[11] / 1000.0)?;
        json_float(f, "Tag2", s.al_free[12] / 1000.0)?;
        json_float(f, "Tag3", s.al_free[13] / 1000.0)?;
        let mut valid = true;
        n_pld = 0;
        for k in 30..38 {
            if !s.al_free[k].is_finite() || s.al_free[k] <= 0.0 {
                valid = false;
            }
            if valid {
                n_pld += 1;
            }
        }
        if n_pld > 0 {
            write!(f, "\t\"InitialPostLabelDelay\": [\n")
                .map_err(|e| Error::convert(e.to_string()))?;
            for i in 0..n_pld as usize {
                if i != 0 {
                    writeln!(f, ",").map_err(|e| Error::convert(e.to_string()))?;
                }
                write!(f, "\t\t{}", format_g(s.al_free[i + 30] / 1000.0))
                    .map_err(|e| Error::convert(e.to_string()))?;
            }
            writeln!(f, "\t],").map_err(|e| Error::convert(e.to_string()))?;
        }
    }
    if is_oxford && s.tag_plane_thickness > 0.0 {
        writeln!(
            f,
            "\t\"TagPlaneDThickness\": {},",
            format_g(s.tag_plane_thickness)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
        writeln!(
            f,
            "\t\"TagPlaneUlShape\": {},",
            format_g(s.tag_plane_ul_shape)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
        writeln!(
            f,
            "\t\"TagPlaneSPositionDTra\": {},",
            format_g(s.tag_plane_position_d_tra)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
        writeln!(
            f,
            "\t\"TagPlaneSNormalDTra\": {},",
            format_g(s.tag_plane_normal_d_tra)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    if is_pcasl {
        writeln!(f, "\t\"ArterialSpinLabelingType\": \"PCASL\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    if is_pasl {
        writeln!(f, "\t\"ArterialSpinLabelingType\": \"PASL\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    if (is_pasl || is_pcasl) && s.interp <= 0 {
        let z = if dcm.slice_thickness > 0.0 {
            dcm.slice_thickness
        } else {
            dcm.xyz_mm[3]
        };
        writeln!(
            f,
            "\t\"AcquisitionVoxelSize\": [\n\t\t{},\n\t\t{},\n\t\t{}\t],",
            format_g(dcm.xyz_mm[1]),
            format_g(dcm.xyz_mm[2]),
            format_g(z)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    if is_pasl || is_pcasl {
        // CSA `sAsl.ulSuppressionMode`: 1 = None, ≥2 = background suppression
        // (e.g. GRAY-WHITE). Completes upstream TODO for BIDS BackgroundSuppression.
        if s.ul_suppression_mode > 0 {
            let on = s.ul_suppression_mode >= 2;
            writeln!(
                f,
                "\t\"BackgroundSuppression\": {},",
                if on { "true" } else { "false" }
            )
            .map_err(|e| Error::convert(e.to_string()))?;
        }
        let mut max_echo = s.l_contrasts;
        if max_echo < 1 {
            max_echo = 1;
        }
        if n_pld < 1 {
            n_pld = 1;
        }
        let n_contrasts = (n_pld * max_echo) as i64;
        let nt = hdr.dim[4] as i64;
        if n_contrasts > 0 {
            if nt % (n_contrasts * 2) == 0 {
                let pairs = nt / n_contrasts;
                if pairs > 0 {
                    writeln!(f, "\t\"TotalAcquiredPairs\": {pairs},")
                        .map_err(|e| Error::convert(e.to_string()))?;
                }
            } else if (nt - 1) % (n_contrasts * 2) == 0 {
                writeln!(f, "\t\"M0Type\": \"Included\",")
                    .map_err(|e| Error::convert(e.to_string()))?;
                let pairs = (nt - 1) / n_contrasts;
                if pairs > 0 {
                    writeln!(f, "\t\"TotalAcquiredPairs\": {pairs},")
                        .map_err(|e| Error::convert(e.to_string()))?;
                }
            } else {
                eprintln!("Unable to determine M0Type");
            }
        }
    }
    // Match C++: TR in milliseconds for pCASL sequences that set the prep TR.
    if repetition_time_preparation > 0.0 {
        writeln!(
            f,
            "\t\"RepetitionTimePreparation\": {},",
            format_g(repetition_time_preparation)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    Ok(())
}

fn write_generic_asl(f: &mut File, dcm: &DicomImage) -> Result<()> {
    use dcm_dicom::{
        ASL_FLAG_GE_3DCASL, ASL_FLAG_GE_3DPCASL, ASL_FLAG_GE_CONTINUOUS,
        ASL_FLAG_GE_PSEUDOCONTINUOUS, ASL_FLAG_GE_PULSED,
    };
    if dcm.post_label_delay > 0 {
        json_float(f, "PostLabelingDelay", dcm.post_label_delay as f64 / 1000.0)?;
    }
    json_str(f, "LabelingOrientation", &dcm.labeling_orientation)?;
    if dcm.vascular_crushing == 1 {
        writeln!(f, "\t\"VascularCrushing\": true,").map_err(|e| Error::convert(e.to_string()))?;
    } else if dcm.vascular_crushing == 0 {
        writeln!(f, "\t\"VascularCrushing\": false,").map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.vascular_crushing_venc > 0.0 {
        json_float(f, "VascularCrushingVENC", dcm.vascular_crushing_venc)?;
    }
    if dcm.asl_flags & (ASL_FLAG_GE_CONTINUOUS | ASL_FLAG_GE_3DCASL) != 0 {
        writeln!(f, "\t\"ArterialSpinLabelingType\": \"CASL\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.asl_flags & (ASL_FLAG_GE_PSEUDOCONTINUOUS | ASL_FLAG_GE_3DPCASL) != 0 {
        writeln!(f, "\t\"ArterialSpinLabelingType\": \"PCASL\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.asl_flags & ASL_FLAG_GE_PULSED != 0 {
        writeln!(f, "\t\"ArterialSpinLabelingType\": \"PASL\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.duration_label_pulse_ge > 0 {
        json_float(
            f,
            "LabelingDuration",
            dcm.duration_label_pulse_ge as f64 / 1000.0,
        )?;
        json_float(f, "PostLabelingDelay", dcm.ti / 1000.0)?;
        json_float(f, "NumberOfPointsPerArm", dcm.number_of_points_per_arm)?;
        json_float(f, "NumberOfArms", dcm.number_of_arms)?;
    }
    if dcm.number_of_excitations > 1.0 {
        json_float(f, "NumberOfExcitations", dcm.number_of_excitations)?;
    }
    Ok(())
}

fn json_float_nan(f: &mut File, key: &str, val: f64) -> Result<()> {
    if !val.is_finite() || val == 0.0 {
        return Ok(());
    }
    writeln!(f, "\t\"{key}\": {},", format_g(val)).map_err(|e| Error::convert(e.to_string()))
}

fn write_ge_fields(f: &mut File, dcm: &DicomImage, hdr: &Nifti1Header) -> Result<()> {
    match dcm.phase_encoding_ge {
        0 => writeln!(f, "\t\"PhaseEncodingPolarityGE\": \"Unflipped\",")
            .map_err(|e| Error::convert(e.to_string()))?,
        4 => writeln!(f, "\t\"PhaseEncodingPolarityGE\": \"Flipped\",")
            .map_err(|e| Error::convert(e.to_string()))?,
        _ => {}
    }
    if dcm.shim_setting.iter().any(|v| *v != 0.0) {
        write!(f, "\t\"ShimSetting\": [\n").map_err(|e| Error::convert(e.to_string()))?;
        for (i, v) in dcm.shim_setting.iter().enumerate() {
            if i != 0 {
                writeln!(f, ",").map_err(|e| Error::convert(e.to_string()))?;
            }
            write!(f, "\t\t{}", format_g(*v)).map_err(|e| Error::convert(e.to_string()))?;
        }
        writeln!(f, "\t],").map_err(|e| Error::convert(e.to_string()))?;
    }
    json_str(f, "PrescanReuseString", &dcm.prescan_reuse_string)?;
    json_str(f, "CoilString", &dcm.coil_string)?;
    json_float(f, "PercentPhaseFOV", dcm.percent_phase_fov)?;
    json_float(f, "PercentSampling", dcm.percent_sampling)?;
    let pe_lines = dcm.acquisition_matrix_pe;
    if pe_lines > 0 {
        writeln!(f, "\t\"AcquisitionMatrixPE\": {pe_lines},")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    let mut recon_pe = pe_lines;
    if hdr.dim[1] == hdr.dim[2] && hdr.dim[2] > 0 {
        recon_pe = hdr.dim[2] as i32;
    } else if dcm.phase_encoding_rc == 'C' {
        recon_pe = hdr.dim[2] as i32;
    } else if dcm.phase_encoding_rc == 'R' {
        recon_pe = hdr.dim[1] as i32;
    }
    if recon_pe > 0 {
        writeln!(f, "\t\"ReconMatrixPE\": {recon_pe},").map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.accel_fact_pe > 0.0 {
        writeln!(
            f,
            "\t\"ParallelReductionFactorInPlane\": {},",
            format_g(dcm.accel_fact_pe)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.parallel_reduction_out_of_plane > 0.0 {
        writeln!(
            f,
            "\t\"ParallelReductionFactorOutOfPlane\": {},",
            format_g(dcm.parallel_reduction_out_of_plane)
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    let ees = dcm.effective_echo_spacing_ge / 1_000_000.0;
    if ees > 0.0 {
        writeln!(f, "\t\"EffectiveEchoSpacing\": {},", format_g(ees))
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    json_float(f, "AcquisitionDuration", dcm.acquisition_duration_s)?;
    if dcm.number_of_k_space_trajectories > 0 {
        writeln!(
            f,
            "\t\"NumberOfKSpaceTrajectories\": {},",
            dcm.number_of_k_space_trajectories
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    if ees > 0.0 && recon_pe > 1 {
        writeln!(
            f,
            "\t\"TotalReadoutTime\": {},",
            format_g(ees * (recon_pe as f64 - 1.0))
        )
        .map_err(|e| Error::convert(e.to_string()))?;
    }
    json_float(f, "PixelBandwidth", dcm.pixel_bandwidth)?;
    // GE diffusion epi2 fields (issue 635).
    if dcm.epi_version_ge == 2 || dcm.internal_epi_version_ge == 2 {
        if dcm.number_of_diffusion_direction_ge > 0 {
            writeln!(
                f,
                "\t\"NumberOfDiffusionDirectionGE\": {},",
                dcm.number_of_diffusion_direction_ge
            )
            .map_err(|e| Error::convert(e.to_string()))?;
        }
        if dcm.number_of_diffusion_t2_ge > 0 {
            writeln!(
                f,
                "\t\"NumberOfDiffusionT2GE\": {},",
                dcm.number_of_diffusion_t2_ge
            )
            .map_err(|e| Error::convert(e.to_string()))?;
        }
        if dcm.tensor_file_ge > 0 {
            writeln!(f, "\t\"TensorFileNumberGE\": {},", dcm.tensor_file_ge)
                .map_err(|e| Error::convert(e.to_string()))?;
        }
    }
    // GE DiffGradientCyclingGE (issue 635).
    if dcm.diff_cycling_mode_ge_override {
        writeln!(f, "\t\"DiffGradientCyclingGE\": \"OVERRIDE\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    if dcm.diff_cycling_mode_ge > 0 {
        let label = match dcm.diff_cycling_mode_ge {
            1 => "ALLTR",
            2 => "2TR",
            3 => "3TR",
            100 => "SPOFF",
            _ => "",
        };
        if !label.is_empty() {
            writeln!(f, "\t\"DiffGradientCyclingGE\": \"{label}\",")
                .map_err(|e| Error::convert(e.to_string()))?;
        }
    }
    let ph_pos = dcm.csa.image.phase_encoding_direction_positive;
    if (dcm.phase_encoding_rc == 'R' || dcm.phase_encoding_rc == 'C') && ph_pos >= 0 {
        let axis = if dcm.phase_encoding_rc == 'C' { "j" } else { "i" };
        let mut suffix = String::new();
        if ph_pos == 0 && dcm.phase_encoding_rc != 'C' {
            suffix.push('-');
        } else if dcm.phase_encoding_rc == 'C' && ph_pos == 1 {
            suffix.push('-');
        }
        writeln!(f, "\t\"PhaseEncodingDirection\": \"{axis}{suffix}\",")
            .map_err(|e| Error::convert(e.to_string()))?;
    }
    Ok(())
}

fn json_str(f: &mut File, key: &str, val: &str) -> Result<()> {
    if val.is_empty() {
        return Ok(());
    }
    writeln!(f, "\t\"{key}\": \"{}\",", escape_json(val))
        .map_err(|e| Error::convert(e.to_string()))
}

fn json_float(f: &mut File, key: &str, val: f64) -> Result<()> {
    if !val.is_finite() || val <= 0.0 {
        return Ok(());
    }
    writeln!(f, "\t\"{key}\": {},", format_g(val)).map_err(|e| Error::convert(e.to_string()))
}

fn parse_dicom_time_str_sec(s: &str) -> f64 {
    let digits: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let Ok(t) = digits.parse::<f64>() else {
        return -1.0;
    };
    if t < 0.0 {
        return -1.0;
    }
    dcm_core::dicom_time_to_sec(t)
}

fn escape_json(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn write_image_type(f: &mut File, dcm: &DicomImage) -> Result<()> {
    if dcm.image_type.is_empty() {
        return Ok(());
    }
    let parts: Vec<&str> = dcm
        .image_type
        .split(|c| c == '\\' || c == '_')
        .filter(|s| !s.is_empty())
        .collect();
    write!(f, "\t\"ImageType\": [").map_err(|e| Error::convert(e.to_string()))?;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            write!(f, ", ").map_err(|e| Error::convert(e.to_string()))?;
        }
        write!(f, "\"{p}\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    // Append missing magnitude/phase/real tokens (C++ nii_SaveBIDSX / issue 881).
    let joined = format!("_{}_", dcm.image_type.replace('\\', "_").to_ascii_uppercase());
    let is_hz = dcm.is_has_real && dcm.is_real_is_phase_map_hz;
    let mut wrote_mag = false;
    if !is_hz && dcm.is_has_magnitude && !joined.contains("_MAGNITUDE_") {
        write!(f, ", \"MAGNITUDE\"").map_err(|e| Error::convert(e.to_string()))?;
        wrote_mag = true;
    }
    // Legacy: mosaics / GE / UIH often omit explicit MAGNITUDE in (0008,0008).
    if !wrote_mag
        && !joined.contains("_MAGNITUDE_")
        && (dcm.is_mosaic
            || dcm.manufacturer == dcm_dicom::Manufacturer::Ge
            || dcm.manufacturer == dcm_dicom::Manufacturer::Uih)
    {
        write!(f, ", \"MAGNITUDE\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    if !is_hz && dcm.is_has_phase && !joined.contains("_PHASE_") {
        write!(f, ", \"PHASE\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    if !is_hz && dcm.is_has_real && !joined.contains("_REAL_") {
        write!(f, ", \"REAL\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    if !is_hz && dcm.is_has_imaginary && !joined.contains("_IMAGINARY_") {
        write!(f, ", \"IMAGINARY\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    if is_hz && !joined.contains("_FIELDMAPHZ_") {
        write!(f, ", \"FIELDMAPHZ\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    writeln!(f, "],").map_err(|e| Error::convert(e.to_string()))
}

fn write_image_type_text(f: &mut File, dcm: &DicomImage) -> Result<()> {
    if dcm.image_type_text.is_empty() {
        return Ok(());
    }
    let parts: Vec<&str> = dcm
        .image_type_text
        .split(|c| c == '\\' || c == '_')
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return Ok(());
    }
    write!(f, "\t\"ImageTypeText\": [").map_err(|e| Error::convert(e.to_string()))?;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            write!(f, ", ").map_err(|e| Error::convert(e.to_string()))?;
        }
        write!(f, "\"{p}\"").map_err(|e| Error::convert(e.to_string()))?;
    }
    writeln!(f, "],").map_err(|e| Error::convert(e.to_string()))
}

fn write_acquisition_time(f: &mut File, dcm: &DicomImage, anon: Anonymize) -> Result<()> {
    if dcm.is_3d_acq {
        return Ok(());
    }
    // PET emits AcquisitionTime in the PET SeriesTime block instead.
    if dcm.modality == dcm_dicom::Modality::Pt {
        return Ok(());
    }
    let time = if !dcm.acquisition_time.is_empty() {
        &dcm.acquisition_time
    } else {
        &dcm.series_time
    };
    let digits: String = time
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if digits.len() < 6 {
        return Ok(());
    }
    let hh: i32 = digits[0..2].parse().unwrap_or(0);
    let mm: i32 = digits[2..4].parse().unwrap_or(0);
    let sec: f64 = digits[4..].parse().unwrap_or(0.0);
    writeln!(f, "\t\"AcquisitionTime\": \"{hh:02}:{mm:02}:{sec:09.6}\",")
        .map_err(|e| Error::convert(e.to_string()))?;
    // `-ba y` strips dates; `-ba n` / `-ba o` keep AcquisitionDateTime (C++).
    if anon == Anonymize::Full {
        return Ok(());
    }
    let date_digits: String = dcm
        .acquisition_date
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect();
    if date_digits.len() < 8 {
        return Ok(());
    }
    let year: i32 = date_digits[0..4].parse().unwrap_or(0);
    let month: i32 = date_digits[4..6].parse().unwrap_or(0);
    let day: i32 = date_digits[6..8].parse().unwrap_or(0);
    let year_s = if (0..=9999).contains(&year) {
        format!("{year:04}")
    } else {
        format!("{year:+04}")
    };
    writeln!(
        f,
        "\t\"AcquisitionDateTime\": \"{year_s}-{month:02}-{day:02}T{hh:02}:{mm:02}:{sec:09.6}\","
    )
    .map_err(|e| Error::convert(e.to_string()))
}

/// Issue 983: split Siemens `"XYZ Gauss4.00"` into ReconFilterType + Size.
fn emit_recon_filter(f: &mut File, dcm: &DicomImage) -> Result<()> {
    let io = |e: std::io::Error| Error::convert(e.to_string());
    if dcm.recon_filter_size.is_finite() {
        writeln!(
            f,
            "\t\"ReconFilterSize\": {},",
            format_g(dcm.recon_filter_size)
        )
        .map_err(io)?;
        return Ok(());
    }
    let k = dcm.convolution_kernel.as_str();
    if k.is_empty() {
        return Ok(());
    }
    let mut split_at: Option<usize> = None;
    for (i, c) in k.char_indices() {
        if c.is_ascii_digit() || c == '.' {
            if split_at.is_none() {
                split_at = Some(i);
            }
        } else if c != ' ' {
            split_at = None; // letters after digits → not "<name><number>"
        }
    }
    let Some(at) = split_at.filter(|&a| a > 0) else {
        return Ok(());
    };
    let name = k[..at].trim_end();
    let size: f64 = k[at..].parse().unwrap_or(0.0);
    if !name.is_empty() {
        writeln!(f, "\t\"ReconFilterType\": \"{}\",", escape_json(name)).map_err(io)?;
    }
    if size > 0.0 {
        writeln!(f, "\t\"ReconFilterSize\": {},", format_g(size)).map_err(io)?;
    }
    Ok(())
}

/// PET `ReconMethodName` + subset/iteration parameters (issue 802).
fn emit_recon_method_name(f: &mut File, method: &str) -> Result<()> {
    let io = |e: std::io::Error| Error::convert(e.to_string());
    if method.is_empty() {
        return Ok(());
    }
    let mut name = String::new();
    let exact: &[(&str, &str)] = &[
        ("PSF+TOF3i21s", "Point-Spread Function + Time Of Flight"),
        ("PSF TOF 3D OSEM", "Point-Spread Function 3D Time Of Flight"),
        ("OP-OSEM", "Ordinary Poisson - Ordered Subset Expectation Maximization"),
        (
            "OSEM3D-OP-PSF",
            "Ordinary Poisson 3D Ordered Subset Expectation Maximization + Point-Spread Function",
        ),
        ("LOR-RAMLA", "Line Of Response - Row Action Maximum Likelihood"),
        ("3D-RAMLA", "3D Row Action Maximum Likelihood"),
        ("3DRP", "3DRP"),
        ("3D Kinahan-Rogers", "3D Kinahan-Rogers"),
    ];
    for (pat, label) in exact {
        if method.contains(pat) {
            name = (*label).to_string();
            break;
        }
    }
    if name.is_empty() {
        if method.contains("OSEM") {
            name.push_str("Ordered Subset Expectation Maximization ");
        } else if method.contains("OS") {
            name.push_str("Ordered Subset ");
        }
        if method.contains("LOR") {
            name.push_str("Line Of Response ");
        }
        if method.contains("RAMLA") {
            name.push_str("Row Action Maximum Likelihood ");
        }
        if method.contains("OP") {
            name.push_str("Ordinary Poisson ");
        }
        if method.contains("PSF") {
            name.push_str("Point-Spread Function modelling ");
        }
        if method.contains("TOF") || method.contains("TF") {
            name.push_str("Time Of Flight ");
        }
        if method.contains("VPHD-S") {
            name.push_str(
                "3D Ordered Subset Expectation Maximization with Point-Spread Function modelling ",
            );
        } else if method.contains("VPHD") {
            name.push_str("VUE Point HD ");
        }
        if method.contains("VPFXS") {
            name.push_str(
                "VUE Point HD using Time Of Flight with Point-Spread Function modelling ",
            );
        } else if method.contains("VPFX") {
            name.push_str("VUE Point HD using Time Of Flight ");
        }
        if method.contains("Q.Clear") {
            name.push_str("VUE Point HD with regularization (smoothing) ");
        }
        if method.contains("BLOB") {
            name.push_str("3D spherically symmetric basis function ");
        }
        if method.contains("FilteredBack")
            || method.contains("Filtered Back")
            || method.contains("Filtered Backprojection")
        {
            name.push_str("Filtered Back Projection ");
        }
        if method.contains("3DRP") {
            name.push_str("3D Kinahan-Rogers ");
        }
        while name.ends_with(' ') {
            name.pop();
        }
    }
    if !name.is_empty() {
        writeln!(f, "\t\"ReconMethodName\": \"{}\",", escape_json(&name)).map_err(io)?;
    }
    let s_end = method.ends_with('s');
    let mut iterations = 0i32;
    for i in 1..33 {
        let pat = if s_end {
            format!("{i}i")
        } else {
            format!("i{i}")
        };
        if method.contains(&pat) {
            iterations = i;
        }
    }
    let mut subsets = 0i32;
    for i in 1..32 {
        let pat = if s_end {
            format!("{i}s")
        } else {
            format!("s{i}")
        };
        if method.contains(&pat) {
            subsets = i;
        }
    }
    if subsets > 0 && iterations > 0 {
        writeln!(f, "\t\"ReconMethodParameterLabels\": [\"subsets\", \"iterations\"],")
            .map_err(io)?;
        writeln!(f, "\t\"ReconMethodParameterValues\": [").map_err(io)?;
        writeln!(f, "\t\t{subsets},").map_err(io)?;
        writeln!(f, "\t\t{iterations}\t],").map_err(io)?;
    }
    Ok(())
}

/// PET SeriesTime + AcquisitionTime + ScanStart (issue 983). Do not emit TimeZero.
fn emit_pet_series_times(f: &mut File, dcm: &DicomImage) -> Result<()> {
    let io = |e: std::io::Error| Error::convert(e.to_string());
    let has_series = !dcm.series_time.is_empty();
    let has_acq = !dcm.acquisition_time.is_empty();
    if !has_series && !has_acq {
        return Ok(());
    }
    if has_series {
        if let Some((hh, mm, ss)) = parse_hhmmss_int(&dcm.series_time) {
            writeln!(f, "\t\"SeriesTime\": \"{hh:02}:{mm:02}:{ss:02}\",").map_err(io)?;
        }
    }
    if has_acq {
        if let Some((hh, mm, sec)) = parse_hhmmss_frac(&dcm.acquisition_time) {
            writeln!(
                f,
                "\t\"AcquisitionTime\": \"{hh:02}:{mm:02}:{sec:09.6}\","
            )
            .map_err(io)?;
        }
    }
    writeln!(f, "\t\"ScanStart\": 0,").map_err(io)?;
    if !dcm.decay_correction.is_empty() {
        let corrected = dcm.decay_correction != "NONE";
        writeln!(
            f,
            "\t\"ImageDecayCorrected\": {},",
            if corrected { "true" } else { "false" }
        )
        .map_err(io)?;
        if corrected && dcm.decay_correction == "START" {
            writeln!(f, "\t\"ImageDecayCorrectionTime\": 0,").map_err(io)?;
        } else if corrected && dcm.decay_correction == "ADMIN" {
            let inj = parse_dicom_time_str_sec(&dcm.radiopharmaceutical_start_time);
            let t0 = if !dcm.series_time.is_empty() {
                parse_dicom_time_str_sec(&dcm.series_time)
            } else {
                parse_dicom_time_str_sec(&dcm.acquisition_time)
            };
            if inj >= 0.0 && t0 >= 0.0 {
                writeln!(
                    f,
                    "\t\"ImageDecayCorrectionTime\": {},",
                    format_g(inj - t0)
                )
                .map_err(io)?;
            }
        }
    }
    Ok(())
}

fn parse_hhmmss_int(s: &str) -> Option<(i32, i32, i32)> {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 6 {
        return None;
    }
    let t: i32 = digits[..6].parse().ok()?;
    Some((t / 10000, (t / 100) % 100, t % 100))
}

fn parse_hhmmss_frac(s: &str) -> Option<(i32, i32, f64)> {
    let digits: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if digits.len() < 6 {
        return None;
    }
    let hh: i32 = digits[0..2].parse().ok()?;
    let mm: i32 = digits[2..4].parse().ok()?;
    let sec: f64 = digits[4..].parse().ok()?;
    Some((hh, mm, sec))
}

fn pulse_sequence_type(dcm: &DicomImage) -> Option<&'static str> {
    let seq = dcm.scanning_sequence.as_str();
    let var = dcm.sequence_variant.as_str();
    let is_epi = seq.contains("EP");
    let is_gre = seq.contains("GR");
    let is_se = seq.contains("SE");
    let is_ir = seq.contains("IR");
    let is_mp = var.contains("MP");
    let is_sp = var.starts_with("SP\\") || var == "SP" || var.contains("\\SP");
    let is_mb = dcm.csa.image.multi_band_factor > 1;
    if is_epi {
        if is_mb && is_se {
            Some("Multiband Spin Echo EPI")
        } else if is_mb {
            Some("Multiband Gradient Echo EPI")
        } else if is_se {
            Some("Spin Echo EPI")
        } else {
            Some("Gradient Echo EPI")
        }
    } else if is_gre {
        if is_mp && is_ir {
            Some("MPRAGE")
        } else if is_sp {
            Some("Spoiled Gradient Echo")
        } else {
            Some("Gradient Echo")
        }
    } else if is_se {
        if is_ir {
            Some("Inversion Recovery Spin Echo")
        } else {
            Some("Spin Echo")
        }
    } else {
        None
    }
}

/// Approximate C `printf("%g")` (6 significant digits).
fn format_g(v: f64) -> String {
    format_g_prec(v, 6)
}

fn format_g_prec(v: f64, prec: usize) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let p = prec as i32;
    if exp < -4 || exp >= p {
        let mant = v / 10f64.powi(exp);
        let digits = (prec.saturating_sub(1)).max(0);
        let mut m = format!("{:.digits$}", mant.abs());
        m = m.trim_end_matches('0').trim_end_matches('.').to_string();
        let sign = if v < 0.0 { "-" } else { "" };
        format!("{sign}{m}e{exp:+03}")
    } else {
        let dec = (p - exp - 1).max(0) as usize;
        let s = format!("{v:.dec$}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}
