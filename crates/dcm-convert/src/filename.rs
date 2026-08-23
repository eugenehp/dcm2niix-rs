//! `-f` filename expansion (`nii_createFilename`).

use std::path::{Path, PathBuf};
use std::sync::Once;

use dcm_core::error::{Error, Result};
use dcm_dicom::{DicomImage, Manufacturer};

use crate::opts::DcmOpts;

static WARNED_HAZARDOUS_BIDS: Once = Once::new();

pub fn create_filename(dcm: &DicomImage, opts: &DcmOpts) -> Result<PathBuf> {
    let mut outdir = if opts.outdir.is_empty() {
        let p = Path::new(&opts.indir);
        if p.is_file() {
            p.parent()
                .map(|x| x.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            p.to_path_buf()
        }
    } else {
        PathBuf::from(&opts.outdir)
    };
    let expanded = expand(&opts.filename, dcm, opts);
    if opts.filename.contains('%')
        && (opts.filename.contains("%h") || opts.filename.contains("%H"))
    {
        WARNED_HAZARDOUS_BIDS.call_once(|| {
            eprintln!(
                "Warning: hazardous (%h) or reproin (%H) bids naming experimental"
            );
        });
    }
    let expanded = if opts.add_name_postfixes {
        format!("{}{}", expanded, name_postfixes(dcm))
    } else {
        expanded
    };
    let expanded = sanitize_path_components(&expanded);
    if expanded.is_empty() {
        return Err(Error::convert("empty output filename"));
    }
    outdir.push(expanded);
    if let Some(parent) = outdir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    // ReproIn `%H`: seed dataset_description.json / README / task JSON once.
    if opts.filename.contains("%H") {
        if let Some(study_root) = study_root_for_boilerplate(&outdir, dcm, opts) {
            let _ = crate::reproin::ensure_bids_boilerplate(
                &study_root,
                dcm.csa.bids_data_type == "func",
                &dcm.csa.bids_task,
            );
            let anon_full = matches!(opts.anonymize, crate::opts::AnonymizeBids::Yes);
            let _ = crate::reproin::append_provenance(&study_root, &outdir, dcm, anon_full);
        }
    }
    Ok(outdir)
}

fn study_root_for_boilerplate(
    out_stem: &Path,
    dcm: &DicomImage,
    opts: &DcmOpts,
) -> Option<PathBuf> {
    // Prefer the study path component under outdir (sub-… lives under study).
    let outdir = if opts.outdir.is_empty() {
        out_stem.parent()?.parent()?.to_path_buf()
    } else {
        PathBuf::from(&opts.outdir)
    };
    let study = if opts.is_bids_root {
        crate::reproin::sanitize_project_path(&opts.bids_root)
    } else {
        crate::reproin::build_study_path(dcm)
    };
    if study.is_empty() {
        return Some(outdir);
    }
    Some(outdir.join(study))
}

fn expand(fmt: &str, dcm: &DicomImage, opts: &DcmOpts) -> String {
    let mut out = String::new();
    let chars: Vec<char> = fmt.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%' && i + 1 < chars.len() {
            let raw = chars[i + 1];
            // `%h` (hazardous BidsGuess) vs `%H` (ReproIn) are case-sensitive.
            if raw == 'h' {
                out.push_str(&crate::bids_guess::expand_hazardous_h(dcm, opts));
                i += 2;
                continue;
            }
            if raw == 'H' {
                out.push_str(&crate::reproin::expand_reproin_h(dcm, opts));
                i += 2;
                continue;
            }
            let f = raw.to_ascii_uppercase();
            match f {
                'A' => out.push_str(&dcm.coil_name),
                'B' => out.push_str(&stem(&dcm.path)),
                'C' => out.push_str(&dcm.image_comments),
                'D' => out.push_str(&dcm.series_description),
                'E' => out.push_str(&dcm.echo_number.to_string()),
                'F' => out.push_str(&opts.indir_parent),
                'G' => out.push_str(&dcm.accession_number),
                'I' => out.push_str(&dcm.patient_id),
                'J' => out.push_str(&dcm.series_uid),
                'K' => out.push_str(&dcm.study_uid),
                'M' => out.push_str(dcm.manufacturer.as_str()),
                'N' => out.push_str(&dcm.patient_name),
                'O' => out.push_str(&dcm.instance_uid),
                'P' => out.push_str(if dcm.protocol_name.is_empty() {
                    &dcm.series_description
                } else {
                    &dcm.protocol_name
                }),
                'R' => out.push_str(&dcm.instance_number.to_string()),
                'S' => out.push_str(&dcm.series_number.to_string()),
                'T' => out.push_str(&datetime_stamp(dcm)),
                'U' => out.push_str(&dcm.acquisition_number.to_string()),
                'V' => out.push_str(dcm.manufacturer.as_str()),
                'X' => out.push_str(&dcm.study_id),
                'Z' => out.push_str(&dcm.sequence_name),
                '%' => out.push('%'),
                other => {
                    out.push('%');
                    out.push(other);
                }
            }
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// C++ `nii_createFilename` post-fixes (`isAddNamePostFixes`): echo / phase / real /
/// imaginary / trigger-delay disambiguators.
fn name_postfixes(dcm: &DicomImage) -> String {
    let mut s = String::new();
    if dcm.echo_number > 1 {
        s.push_str(&format!("_e{}", dcm.echo_number));
    }
    if dcm.is_has_phase {
        s.push_str("_ph");
    }
    if dcm.is_no_rf {
        s.push_str("_noRF");
    }
    if dcm.is_has_imaginary {
        s.push_str("_imaginary");
    } else if dcm.is_has_real && dcm.is_real_is_phase_map_hz {
        s.push_str("_fieldmaphz");
    } else if dcm.is_has_real {
        s.push_str("_real");
    }
    // Issue 336: GE uses TriggerTime for slice timing — skip postfix for GE.
    // Issue 533: ASL also skips trigger postfix.
    if dcm.asl_flags == dcm_dicom::ASL_FLAG_NONE
        && dcm.trigger_delay_time >= 1.0
        && dcm.manufacturer != Manufacturer::Ge
    {
        s.push_str(&format!("_t{}", dcm.trigger_delay_time.round() as i32));
    }
    // Philips XX_ / PS_ name clash avoidance (always, even without other postfixes).
    if dcm.is_raw_data_storage {
        s.push_str("_Raw");
    }
    if dcm.is_grayscale_softcopy_presentation_state {
        s.push_str("_PS");
    }
    s
}

fn datetime_stamp(dcm: &DicomImage) -> String {
    // C++ `%t` uses `dcm.dateTime` = studyDate*1e6 + studyTime, then extracts
    // HHMMSS from the time-of-day portion (studyTime when studyDate is set).
    let time = if !dcm.study_time.is_empty() {
        &dcm.study_time
    } else if !dcm.series_time.is_empty() {
        &dcm.series_time
    } else {
        &dcm.acquisition_time
    };
    let digits: String = time.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 6 {
        digits[..6].to_string()
    } else {
        format!("{digits:0>6}")
    }
}

fn sanitize_path_components(s: &str) -> String {
    s.split(['/', '\\'])
        .map(sanitize_component)
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join(std::path::MAIN_SEPARATOR_STR)
}

fn sanitize_component(s: &str) -> String {
    // Match dcm2niix: spaces and Windows-forbidden chars → `_`.
    let mut out = String::new();
    for c in s.chars() {
        if c.is_control() || " <>:\"|?*;$`".contains(c) || c == '/' || c == '\\' {
            out.push('_');
        } else {
            out.push(c);
        }
    }
    // Collapse redundant underscores.
    let mut collapsed = String::new();
    for c in out.chars() {
        if c == '_' && collapsed.ends_with('_') {
            continue;
        }
        collapsed.push(c);
    }
    collapsed
        .trim_matches(|c: char| c == '.' || c == '_' || c == ' ')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opts::DcmOpts;
    use dcm_dicom::{DicomImage, Manufacturer, Modality};
    use std::path::PathBuf;

    fn dummy() -> DicomImage {
        DicomImage {
            path: PathBuf::from("x"),
            series_uid: "1.2.3".into(),
            series_uid_crc: 1,
            instance_uid: "1.2.3.4".into(),
            study_uid: "1.2".into(),
            series_number: 7,
            instance_number: 1,
            acquisition_number: 1,
            echo_number: 1,
            rows: 4,
            columns: 4,
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
            manufacturer: Manufacturer::Siemens,
            modality: Modality::Mr,
            manufacturer_name: "SIEMENS".into(),
            manufacturers_model_name: "Prisma".into(),
            institution_name: String::new(),
            institution_address: String::new(),
            institutional_department: String::new(),
            procedure_step_description: String::new(),
            station_name: String::new(),
            device_serial_number: String::new(),
            software_versions: String::new(),
            protocol_name: "t1_mprage".into(),
            series_description: "T1w".into(),
            sequence_name: "*tfl3d1".into(),
            pulse_sequence_name: String::new(),
            scanning_sequence: "GR".into(),
            sequence_variant: "SP".into(),
            scan_options: String::new(),
            image_type: "ORIGINAL".into(),
            image_comments: String::new(),
            coil_name: "Head".into(),
            coil_string: String::new(),
            transmit_coil_name: String::new(),
            patient_name: "Anon".into(),
            patient_id: "1".into(),
            patient_sex: "O".into(),
            patient_age: String::new(),
            referring_physician_name: String::new(),
            patient_birth_date: String::new(),
            patient_weight: 0.0,
            patient_size: 0.0,
            accession_number: String::new(),
            study_id: "s1".into(),
            study_description: String::new(),
            study_date: "20200101".into(),
            study_time: "120000".into(),
            series_time: "120000".into(),
            acquisition_date: "20200101".into(),
            acquisition_time: "120000.00".into(),
            body_part: "BRAIN".into(),
            tr: 2000.0,
            te: 2.5,
            ti: 900.0,
            flip_angle: 8.0,
            field_strength: 3.0,
            pixel_bandwidth: 200.0,
            echo_train_length: 1,
            phase_encoding_rc: 'C',
            inten_scale: 1.0,
            inten_intercept: 0.0,
            inten_scale_philips: 0.0,
            is_scale_varies_enh: false,
            is_derived: false,
            is_localizer: false,
            number_of_frames: 1,
            imaging_frequency: 123.0,
            patient_position_label: "HFS".into(),
            spacing_between_slices: 1.0,
            acquisition_matrix_pe: 256,
            phase_encoding_steps: 0,
            phase_encoding_steps_out_of_plane: 0,
            number_of_concatenations: 1,
            repetition_time_excitation: -1.0,
            repetition_time_inversion: 0.0,
            percent_phase_fov: 100.0,
            percent_sampling: 100.0,
            mra_acquisition_type: "3D".into(),
            b_value: -1.0,
            diffusion_direction: [0.0; 3],
            pe_direction_displayed: String::new(),
            number_of_averages: 0.0,
            is_3d_acq: true,
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
            csa: Default::default(),
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

    #[test]
    fn expands_protocol_and_series() {
        let opts = DcmOpts::default();
        let s = expand("%p_%s", &dummy(), &opts);
        assert_eq!(s, "t1_mprage_7");
    }

    #[test]
    fn fieldmaphz_postfix_beats_real() {
        let mut d = dummy();
        d.is_has_real = true;
        d.is_real_is_phase_map_hz = true;
        assert_eq!(name_postfixes(&d), "_fieldmaphz");
        d.is_real_is_phase_map_hz = false;
        assert_eq!(name_postfixes(&d), "_real");
    }
}
