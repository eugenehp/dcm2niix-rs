//! NIfTI header description + acquisition time helpers (dcm2niix `headerDcm2Nii2`).

use dcm_dicom::DicomImage;

/// `d.acquisitionTime` as a float (HHMMSS.sss).
pub fn acquisition_time_float(dcm: &DicomImage) -> f64 {
    let digits: String = dcm
        .acquisition_time
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    digits.parse().unwrap_or(0.0)
}

/// `TE=…;Time=…;phase=…` string written to `descrip`.
pub fn nifti_descrip(dcm: &DicomImage) -> String {
    let mut txt = if dcm.modality == dcm_dicom::Modality::Mr {
        format!(
            "TE={};Time={:.3}",
            fmt_te(dcm.te),
            acquisition_time_float(dcm)
        )
    } else {
        format!("Time={:.3}", acquisition_time_float(dcm))
    };
    if dcm.csa.image.phase_encoding_direction_positive >= 0 {
        txt.push_str(&format!(
            ";phase={}",
            dcm.csa.image.phase_encoding_direction_positive
        ));
    }
    let mb = multiband_factor(&dcm.csa.image.slice_timing_ms);
    if mb > 1 {
        txt.push_str(&format!(";mb={mb}"));
    }
    txt
}

fn multiband_factor(times: &[f64]) -> i32 {
    if times.is_empty() {
        return 1;
    }
    let t0 = times[0];
    times.iter().filter(|t| (*t - t0).abs() < 1e-4).count() as i32
}

fn fmt_te(te: f64) -> String {
    // C `%.2g` (2 significant digits).
    if te == 0.0 {
        return "0".into();
    }
    let exp = te.abs().log10().floor() as i32;
    let scale = 10f64.powi(1 - exp);
    let rounded = (te * scale).round() / scale;
    if exp >= 2 || exp < -1 {
        let s = format!("{rounded:.1e}");
        // C `%.2g` prints `1.1e+02` not `1.1e2`.
        if let Some((mant, rest)) = s.split_once('e') {
            let sign = if rest.starts_with('-') { "-" } else { "+" };
            let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            format!("{mant}e{sign}{digits:0>2}")
        } else {
            s
        }
    } else if rounded == rounded.round() {
        format!("{}", rounded.round() as i64)
    } else {
        format!("{rounded}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn te_uses_two_significant_digits() {
        assert_eq!(fmt_te(30.0), "30");
        assert_eq!(fmt_te(95.4), "95");
        assert_eq!(fmt_te(111.2), "1.1e+02");
    }

    #[test]
    fn parses_acq_time() {
        let mut d = DicomImage {
            acquisition_time: "135931.837500".into(),
            modality: dcm_dicom::Modality::Mr,
            te: 30.0,
            csa: Default::default(),
            ..minimal()
        };
        assert!((acquisition_time_float(&d) - 135931.837).abs() < 0.001);
        assert_eq!(nifti_descrip(&d), "TE=30;Time=135931.837");
        d.csa.image.phase_encoding_direction_positive = 1;
        assert_eq!(nifti_descrip(&d), "TE=30;Time=135931.837;phase=1");
    }

    fn minimal() -> DicomImage {
        use dcm_dicom::{CsaMeta, Manufacturer, Modality};
        use std::path::PathBuf;
        DicomImage {
            path: PathBuf::from("x"),
            series_uid: String::new(),
            series_uid_crc: 0,
            instance_uid: String::new(),
            study_uid: String::new(),
            series_number: 0,
            instance_number: 0,
            acquisition_number: 0,
            echo_number: 0,
            rows: 0,
            columns: 0,
            bits_allocated: 16,
            bits_stored: 16,
            samples_per_pixel: 1,
            is_signed: true,
            is_float: false,
            xyz_mm: [1.0; 4],
            slice_thickness: 1.0,
            orient: [0.0; 7],
            patient_position: [0.0; 4],
            patient_position_last: [f64::NAN; 4],
            last_scan_loc: f64::NAN,
            acquisition_duration: 0.0,
            manufacturer: Manufacturer::Siemens,
            modality: Modality::Mr,
            manufacturer_name: String::new(),
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
            image_type: String::new(),
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
}
