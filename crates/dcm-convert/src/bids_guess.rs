//! BIDS filename heuristics (`setBids` / `setBidsSiemens` / Philips / GE / heuristics).
//!
//! Fills `CsaMeta.bids_data_type` / `bids_entity_suffix` / `bids_task` used by
//! `-f %h` and the `BidsGuess` JSON field (`isGuessBidsFilename`).

use dcm_dicom::{DicomImage, Manufacturer, Modality};

/// Populate BIDS guess fields on `d` (mutates `d.csa`).
pub fn set_bids(d: &mut DicomImage, n_convert: usize, verbose: i32) {
    d.csa.bids_data_type.clear();
    d.csa.bids_entity_suffix.clear();
    d.csa.bids_task.clear();

    if d.modality == Modality::Pt {
        d.csa.bids_data_type = "pet".into();
        d.csa.bids_entity_suffix = "_pet".into();
        return;
    }
    if d.modality == Modality::Ct || d.modality == Modality::Seg {
        let (dt, suf) = if d.modality == Modality::Seg {
            ("anat", "_seg")
        } else {
            ("ct", "_ct")
        };
        d.csa.bids_data_type = dt.into();
        d.csa.bids_entity_suffix = suf.into();
        return;
    }
    if d.modality != Modality::Mr && d.modality != Modality::Unknown {
        return;
    }

    match d.manufacturer {
        Manufacturer::Siemens => set_bids_siemens(d, n_convert),
        Manufacturer::Philips => set_bids_philips(d, n_convert),
        Manufacturer::Ge => set_bids_ge(d, n_convert),
        // UIH / others: C++ leaves BidsGuess empty unless heuristics refine a
        // vendor-filled value — do not invent datatype here.
        _ => {}
    }
    if d.csa.bids_data_type.is_empty() {
        set_bids_from_acquisition_contrast(d);
    }
    set_bids_heuristics(d);
    if verbose > 0 {
        eprintln!(
            "::autoBids: seriesDesc:'{}' seq:'{}' bidsData:'{}' bidsSuffix:'{}'",
            d.series_description,
            d.sequence_name,
            d.csa.bids_data_type,
            d.csa.bids_entity_suffix
        );
    }
}

fn seq_blob(d: &DicomImage) -> String {
    let mut s = String::new();
    s.push_str(&d.csa.series.pulse_sequence_details);
    s.push(' ');
    s.push_str(&d.sequence_name);
    s.push(' ');
    s.push_str(&d.internal_pulse_sequence_name);
    s.push(' ');
    s.push_str(&d.protocol_name);
    s.push(' ');
    s.push_str(&d.series_description);
    s.to_ascii_lowercase()
}

fn contains_ci(hay: &str, needle: &str) -> bool {
    hay.to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn pe_dir_label(d: &DicomImage) -> Option<&'static str> {
    let ph = d.csa.image.phase_encoding_direction_positive;
    if ph < 0 {
        return None;
    }
    match d.phase_encoding_rc {
        'C' => Some(if ph == 1 { "AP" } else { "PA" }),
        'R' => Some(if ph == 0 { "LR" } else { "RL" }),
        _ => None,
    }
}

fn finish_suffix(modality: &str, d: &DicomImage, is_dir: bool, is_echo: bool, run: bool) -> String {
    let mut suf = String::new();
    if let Some(dir) = pe_dir_label(d).filter(|_| is_dir) {
        suf.push_str(&format!("_dir-{dir}"));
    }
    if is_echo && d.echo_number > 1 {
        suf.push_str(&format!("_echo-{}", d.echo_number));
    }
    if d.is_has_phase {
        suf.push_str("_part-phase");
    } else if d.is_has_real {
        suf.push_str("_part-real");
    } else if d.is_has_imaginary {
        suf.push_str("_part-imag");
    }
    if run && d.series_number > 0 {
        suf.push_str(&format!("_run-{}", d.series_number));
    }
    suf.push('_');
    suf.push_str(modality);
    suf
}

fn set_bids_siemens(d: &mut DicomImage, n_convert: usize) {
    let blob = seq_blob(d);
    let img = d.image_type.to_ascii_uppercase();
    let desc = d.series_description.to_ascii_lowercase();
    let seq = d.sequence_name.to_ascii_lowercase();
    let pulse = d.internal_pulse_sequence_name.to_ascii_lowercase();

    let mut is_non_spatial = true;
    for i in 1..=6 {
        if d.orient[i] != 0.0 {
            is_non_spatial = false;
            break;
        }
    }

    let (dtype, modality, is_dir, is_echo, run) = if (d.rows < 2 && n_convert < 1) || d.is_localizer {
        ("discard", "localizer", false, false, false)
    } else if d.is_derived && img.contains("DIFFUSION") {
        ("discard", "derivedDWI", false, false, false)
    } else if is_non_spatial {
        ("discard", "nonspatial", false, false, false)
    } else if blob.contains("b1map") {
        ("fmap", "TB1TFL", false, false, true)
    } else if blob.contains("tfl") || blob.contains("mp2rage") || blob.contains("wip925") {
        let dual_ti = {
            let t0 = d.csa.series.al_ti[0];
            let t1 = d.csa.series.al_ti[1];
            t0.is_finite() && t1.is_finite() && t0 > 0.0 && t1 > 0.0
        };
        let inv = d.csa.series.l_inv_contrasts > 1;
        let mod_ = if img.contains("T1 MAP") {
            "T1map"
        } else if img.contains("_UNI") {
            "UNIT1"
        } else if dual_ti || inv || blob.contains("mp2rage") {
            "MP2RAGE"
        } else {
            "T1w"
        };
        ("anat", mod_, false, false, true)
    } else if d.b_value >= 0.0
        || blob.contains("_diff")
        || blob.contains("resolve")
        || blob.contains("ep2d_diff")
        || blob.contains("ep2d_stejskal")
    {
        let mod_ = if desc.contains("_sbref") {
            "sbref"
        } else {
            "dwi"
        };
        ("dwi", mod_, true, false, true)
    } else if blob.contains("asl")
        || blob.contains("pasl")
        || blob.contains("pcasl")
        || blob.contains("fairest")
    {
        let mod_ = if desc.contains("_m0") { "m0scan" } else { "asl" };
        ("perf", mod_, false, false, true)
    } else if contains_ci(&d.internal_pulse_sequence_name, "spcR")
        || blob.contains("tse_vfl")
        || blob.contains("\\space")
        || blob.contains("tse")
    {
        let mod_ = if blob.contains("spcir")
            || contains_ci(&d.sequence_name, "spcir")
            || contains_ci(&d.sequence_name, "tir")
            || contains_ci(&d.internal_pulse_sequence_name, "spcir")
        {
            "FLAIR"
        } else if blob.contains("tse2d") || seq.contains("tse2d") || pulse.contains("tse2d") {
            if d.te > 45.0 {
                "T2w"
            } else {
                "PDw"
            }
        } else {
            "T2w"
        };
        ("anat", mod_, false, false, true)
    } else if blob.contains("ep2d_ase") {
        ("", "oef_ase", false, false, false)
    } else if blob.contains("ep2d_se") {
        ("fmap", "epi", true, false, true)
    } else if blob.contains("gre_field_mapping") || blob.contains("field_mapping") {
        if d.is_has_phase {
            ("fmap", "phasediff", false, false, false)
        } else if d.echo_number > 1 {
            d.csa.bids_data_type = "fmap".into();
            d.csa.bids_entity_suffix =
                finish_suffix(&format!("magnitude{}", d.echo_number), d, false, false, false);
            return;
        } else {
            ("fmap", "magnitude", false, false, false)
        }
    } else if blob.contains("\\trufi")
        || seq.contains("fl3d1_ns")
        || seq.contains("fl2d1")
        || seq.contains("tfl2d1")
    {
        ("discard", "localizer", false, false, false)
    } else if seq.contains("fl3d1r") {
        ("anat", "angio", false, false, true)
    } else if blob.contains("fl3d_vibe") {
        let mod_ = if d.te > 45.0 {
            "T2starw"
        } else if d.flip_angle >= 15.0 {
            "T1w"
        } else {
            "PDw"
        };
        ("anat", mod_, false, false, true)
    } else if blob.contains("ep_seg_fid") {
        if desc.contains("mip") {
            ("discard", "mIP", false, false, false)
        } else if desc.contains("swi_images") {
            ("discard", "SWI_Images", false, false, false)
        } else {
            ("anat", "T2starw", false, false, true)
        }
    } else if blob.contains("epfid2d")
        || blob.contains("ep2d_bold")
        || blob.contains("ep2d_pace")
        || blob.contains("bold")
        || d.is_epi
    {
        let mod_ = if d.is_no_rf {
            "noRF"
        } else if desc.contains("_sbref") {
            "sbref"
        } else {
            "bold"
        };
        d.csa.bids_task = "rest".into();
        ("func", mod_, true, true, true)
    } else if blob.contains("gre") || blob.contains("fl3d") || blob.contains("mprage") {
        let mod_ = if d.echo_number > 1 || blob.contains("megre") {
            "MEGRE"
        } else if d.te > 45.0 {
            "T2starw"
        } else {
            "T1w"
        };
        ("anat", mod_, false, false, true)
    } else {
        let mod_ = if d.te > 50.0 {
            "T2w"
        } else if d.te > 0.0 {
            "T1w"
        } else {
            ""
        };
        if mod_.is_empty() {
            return;
        }
        ("anat", mod_, false, false, true)
    };

    if dtype.is_empty() {
        // modality-only (e.g. oef_ase) — leave datatype empty like C++.
        d.csa.bids_entity_suffix = finish_suffix(modality, d, is_dir, is_echo, run);
        return;
    }
    d.csa.bids_data_type = dtype.into();
    d.csa.bids_entity_suffix = finish_suffix(modality, d, is_dir, is_echo, run);
    if d.is_derived && dtype != "discard" && !img.contains("_UNI") {
        // Keep derived flag for non-UNIT1; heuristics may override.
    }
}

fn set_bids_philips(d: &mut DicomImage, n_convert: usize) {
    let seq_var = d.sequence_variant.clone();
    let scan = d.scanning_sequence.clone();
    let pulse = format!(
        "{} {}",
        d.sequence_name, d.internal_pulse_sequence_name
    );
    let img = d.image_type.to_ascii_uppercase();

    let mut is_report_echo = true;
    let mut is_dir = false;
    let mut is_add_run = true;
    let mut is_part = false;
    let mut modality = String::new();
    let mut dtype = String::new();

    if ((d.rows < 4 && n_convert < 4) && d.columns.max(d.rows) < 4) || d.is_localizer {
        dtype = "discard".into();
        modality = "localizer".into();
    } else if seq_var.contains("MP") {
        dtype = "anat".into();
        modality = "T1w".into();
    } else if d.is_diffusion && seq_var.contains("SK") && scan.contains("SE") {
        dtype = "dwi".into();
        modality = "dwi".into();
        is_dir = true;
    } else if img.contains("PERFUSION") || d.asl_flags != 0 {
        dtype = "perf".into();
        modality = "asl".into();
    } else if contains_ci(&pulse, "SEEPI")
        && !d.is_diffusion
        && seq_var.contains("SK")
        && scan.contains("SE")
    {
        is_add_run = false;
        dtype = "fmap".into();
        modality = "epi".into();
        is_dir = true;
        eprintln!(
            "Unable to estimate BIDS `_dir` for fmap epi as Philips DICOMs do not report phase encoding polarity"
        );
    } else if !d.is_diffusion && seq_var.contains("SK") && scan.contains("SE") {
        dtype = "anat".into();
        modality = match mr_weighting_guess(d, true, false) {
            2 => "T2w".into(),
            _ => "PDw".into(),
        };
    } else if seq_var.contains("SK") && scan.contains("IR") {
        dtype = "anat".into();
        modality = "FLAIR".into();
    } else if img.contains("PERFUSION")
        && contains_ci(&pulse, "FEEPI")
        && seq_var.contains("SK")
        && scan.contains("GR")
    {
        dtype = "perf".into();
        modality = "asl".into();
    } else if (d.raw_data_run_number >= 1 || contains_ci(&pulse, "FEEPI"))
        && seq_var.contains("SK")
        && scan.contains("GR")
    {
        dtype = "func".into();
        modality = "bold".into();
        is_dir = true;
        d.csa.bids_task = "rest".into();
    } else if seq_var.contains("SS") && scan.contains("GR") {
        dtype = "anat".into();
        modality = "T2starw".into();
        is_part = true;
    } else if d.is_real_is_phase_map_hz && seq_var.contains("SS") && scan.contains("RM") {
        is_report_echo = false;
        dtype = "fmap".into();
        modality = if d.is_has_real {
            "fieldmap".into()
        } else {
            "magnitude".into()
        };
    } else if seq_var.contains("SP") && scan.contains("GR") {
        eprintln!(
            "Unable to distinguish Philips fieldmaps: phase difference, two phase/magnitude, direct fieldmapping."
        );
        is_report_echo = false;
        dtype = "fmap".into();
        if d.is_has_phase {
            modality = "phasediff".into();
        } else if d.echo_train_length < 2 {
            modality = format!("magnitude{}", d.echo_number);
        }
        if d.echo_train_length > 1 {
            modality = if d.is_has_phase {
                format!("phase{}", d.echo_number)
            } else if d.is_has_imaginary {
                format!("imaginary{}", d.echo_number)
            } else if d.is_has_real {
                format!("real{}", d.echo_number)
            } else {
                format!("magnitude{}", d.echo_number)
            };
        }
    }

    if dtype.is_empty() {
        return;
    }

    // `_acq-` from SequenceVariant + ScanningSequence + pulse + ASL delay + SENSE/MB.
    let mut acq = String::from("_acq-");
    for ch in seq_var
        .chars()
        .chain(scan.chars())
        .chain(pulse.chars())
    {
        if ch.is_ascii_alphanumeric() {
            acq.push(ch);
        }
    }
    if d.trigger_delay_time > 1.0 {
        acq.push_str(&format!("t{}", d.trigger_delay_time.round() as i32));
    }
    if d.accel_fact_pe > 1.0 {
        acq.push_str(&format!("p{}", (10.0 * d.accel_fact_pe).round() as i32));
    }
    if d.csa.image.multi_band_factor > 1 {
        acq.push_str(&format!("m{}", d.csa.image.multi_band_factor));
    }

    let mut suf = acq;
    if is_add_run && d.series_number > 0 {
        suf.push_str(&format!("_run-{}", d.series_number));
    }
    if is_report_echo && (d.echo_number > 1 || (d.is_multi_echo && d.echo_number > 0)) {
        suf.push_str(&format!("_echo-{}", d.echo_number));
    }
    if is_part {
        if d.is_has_phase {
            suf.push_str("_part-phase");
        }
    }
    if let Some(dir) = pe_dir_label(d).filter(|_| is_dir) {
        // Insert dir before modality: C++ puts dir via finish path; keep simple.
        suf.push_str(&format!("_dir-{dir}"));
    }
    if !modality.is_empty() {
        suf.push('_');
        suf.push_str(&modality);
    }
    if d.is_derived {
        dtype = "derived".into();
    }
    d.csa.bids_data_type = dtype;
    d.csa.bids_entity_suffix = suf;
}

fn set_bids_ge(d: &mut DicomImage, n_convert: usize) {
    let mut seq_name = d.internal_pulse_sequence_name.clone();
    if seq_name.is_empty() {
        seq_name = d.sequence_name.clone();
    }
    let seq_u = seq_name.to_ascii_uppercase();
    let scan = d.scanning_sequence.as_str();
    let desc = d.series_description.as_str();
    let mut is_ep_se = scan.contains("EP\\SE");
    let mut is_ep_gr = scan.contains("EP\\GR");
    let mut is_gr = is_ep_gr || seq_u.contains("GRE");
    if contains_ci(&d.procedure_step_description, "Gradient Echo") {
        is_gr = true;
        if scan.contains("EP\\RM") {
            is_ep_gr = true;
        }
    }
    if contains_ci(&d.procedure_step_description, "Spin Echo") && scan.contains("EP\\RM") {
        is_ep_se = true;
    }

    let mut is_report_echo = true;
    let mut is_dir = false;
    let mut is_add_run = true;
    let mut is_part = false;
    let mut modality = String::new();
    let mut dtype = String::new();

    if (d.rows < 2 && n_convert < 4) || d.is_localizer || contains_ci(desc, "3 Plane Loc") {
        dtype = "discard".into();
        modality = "localizer".into();
        is_report_echo = false;
    } else if d.is_real_is_phase_map_hz
        || seq_u.contains("B0MAP")
        || contains_ci(&d.sequence_name, "3db0map")
    {
        is_report_echo = false;
        is_add_run = false;
        dtype = "fmap".into();
        modality = if d.is_real_is_phase_map_hz {
            "fieldmap".into()
        } else {
            "magnitude".into()
        };
    } else if d.is_multi_echo && !is_ep_gr && is_gr {
        dtype = "anat".into();
        modality = "T2starw".into();
        is_part = true;
    } else if seq_u == "EFGRE3D" {
        dtype = "anat".into();
        modality = "T1w".into();
    } else if seq_u.contains("3DRADIAL") && scan.contains("GR\\IR") {
        dtype = "anat".into();
        modality = "T1w".into();
    } else if seq_u.contains("FSE") {
        dtype = "anat".into();
        modality = if scan.contains("IR") {
            "FLAIR".into()
        } else {
            match mr_weighting_guess(d, true, false) {
                2 => "T2w".into(),
                _ => "PDw".into(),
            }
        };
    } else if d.is_diffusion || d.b_value >= 0.0 {
        dtype = "dwi".into();
        modality = "dwi".into();
        is_dir = true;
    } else if d.asl_flags != 0 || contains_ci(&seq_name, "asl") {
        dtype = "perf".into();
        modality = "asl".into();
    } else if is_ep_se && !d.is_diffusion {
        dtype = "fmap".into();
        modality = "epi".into();
        is_dir = true;
    } else if is_ep_gr || d.is_epi || contains_ci(&seq_name, "epi") {
        dtype = "func".into();
        modality = if contains_ci(desc, "sbref") {
            "sbref".into()
        } else {
            "bold".into()
        };
        is_dir = true;
        d.csa.bids_task = "rest".into();
    } else if contains_ci(desc, "flair") || scan.contains("IR") {
        dtype = "anat".into();
        modality = "FLAIR".into();
    } else {
        dtype = "anat".into();
        modality = match mr_weighting_guess(d, false, false) {
            2 => "T2w".into(),
            4 => "T2starw".into(),
            3 => "PDw".into(),
            _ => "T1w".into(),
        };
    }

    if dtype.is_empty() {
        return;
    }
    d.csa.bids_data_type = if d.is_derived {
        "derived".into()
    } else {
        dtype
    };
    d.csa.bids_entity_suffix = finish_suffix(
        &modality,
        d,
        is_dir,
        is_report_echo,
        is_add_run,
    );
    if is_part && d.is_has_phase && !d.csa.bids_entity_suffix.contains("_part-") {
        // finish_suffix already handles phase/real/imag from flags
    }
}

/// C++ `MRWeightingGuess` — returns kMRWeighting* codes.
fn mr_weighting_guess(d: &DicomImage, is_spin_echo: bool, is_variable_flip: bool) -> i32 {
    match d.acquisition_contrast {
        1 | 2 | 3 | 4 | 5 | 6 => return d.acquisition_contrast,
        _ => {}
    }
    if d.te <= 0.0 {
        return 0;
    }
    if is_variable_flip {
        return 0;
    }
    let te_sec = d.te / 1000.0;
    if is_spin_echo {
        return if te_sec >= 0.045 { 2 } else { 3 }; // T2 / PD
    }
    if d.tr <= 0.0 || d.field_strength <= 0.0 || d.flip_angle <= 0.0 {
        return 0;
    }
    let tr_sec = d.tr / 1000.0;
    let t2star_est = 0.050 / d.field_strength;
    if te_sec >= 0.5 * t2star_est {
        return 4; // T2starw
    }
    let t1_est = 0.8 * d.field_strength.powf(0.38);
    let ernst_deg = (-tr_sec / t1_est).exp().acos().to_degrees();
    if d.flip_angle >= 1.3 * ernst_deg {
        return 1; // T1
    }
    3 // PD default
}

fn set_bids_from_acquisition_contrast(d: &mut DicomImage) {
    if d.modality != Modality::Mr || !d.csa.bids_data_type.is_empty() {
        return;
    }
    let (dtype, modality) = match d.acquisition_contrast {
        1 => ("anat", "T1w"),
        2 => ("anat", "T2w"),
        3 => ("anat", "PDw"),
        4 => ("anat", "T2starw"),
        5 => ("anat", "FLAIR"),
        6 => ("anat", "T2w"),
        7 if d.is_diffusion => ("dwi", "dwi"),
        9 => ("anat", "angio"),
        _ => return,
    };
    d.csa.bids_data_type = dtype.into();
    if d.csa.bids_entity_suffix.is_empty() {
        d.csa.bids_entity_suffix = format!("_{modality}");
    } else if !d.csa.bids_entity_suffix.ends_with(modality) {
        d.csa.bids_entity_suffix.push('_');
        d.csa.bids_entity_suffix.push_str(modality);
    }
}

/// Vendor-agnostic refinements (`setBidsHeuristics`).
fn set_bids_heuristics(d: &mut DicomImage) {
    if d.modality != Modality::Mr {
        return;
    }
    if d.csa.bids_data_type.contains("discard") {
        return;
    }
    let name = format!("{} {}", d.protocol_name, d.series_description).to_ascii_lowercase();
    if name.is_empty() {
        return;
    }
    // Word-boundary DWI derivative tokens.
    let deriv = [
        ("colfa", "colFA", "dwi"),
        ("col_fa", "colFA", "dwi"),
        ("col-fa", "colFA", "dwi"),
        ("expadc", "expADC", "dwi"),
        ("exp_adc", "expADC", "dwi"),
        ("exp-adc", "expADC", "dwi"),
        ("tracew", "trace", "dwi"),
        ("trace", "trace", "dwi"),
        ("tensor", "TENSOR", "derived"),
        ("s0map", "S0map", "dwi"),
        ("s0_map", "S0map", "dwi"),
        ("s0-map", "S0map", "dwi"),
        ("fa", "FA", "dwi"),
        ("adc", "ADC", "dwi"),
    ];
    for (tok, suf, dt) in deriv {
        if find_token_bdy(&name, tok) {
            d.csa.bids_data_type = dt.into();
            d.csa.bids_entity_suffix = format!("_{suf}");
            break;
        }
    }
    // Task hints for func.
    if d.csa.bids_data_type == "func"
        && d.csa.bids_task.is_empty()
        && !d.csa.bids_entity_suffix.contains("_task-")
    {
        if let Some(i) = name.find("task-") {
            let start_ok = i == 0 || is_bids_boundary(name.as_bytes()[i - 1] as char);
            if start_ok {
                let rest = &name[i + 5..];
                let task: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric())
                    .collect();
                if !task.is_empty() {
                    d.csa.bids_task = task;
                }
            }
        }
        if d.csa.bids_task.is_empty() {
            for (label, patterns) in [
                ("rest", &["rest", "resting"][..]),
                ("nback", &["nback", "n-back"][..]),
                ("motor", &["motor", "finger"][..]),
                ("story", &["story"][..]),
            ] {
                if patterns.iter().any(|p| name.contains(p)) {
                    d.csa.bids_task = label.into();
                    break;
                }
            }
        }
        if d.csa.bids_task.is_empty() {
            d.csa.bids_task = "rest".into();
        }
    }
}

fn is_bids_boundary(c: char) -> bool {
    matches!(c, '\0' | '_' | '-' | ' ' | '/' | '\\' | '.' | ',')
}

fn find_token_bdy(hay: &str, token: &str) -> bool {
    let bytes = hay.as_bytes();
    let t = token.as_bytes();
    let mut i = 0;
    while i + t.len() <= bytes.len() {
        if &bytes[i..i + t.len()] == t {
            let start_ok = i == 0 || is_bids_boundary(bytes[i - 1] as char);
            let end_ok = i + t.len() == bytes.len()
                || is_bids_boundary(bytes[i + t.len()] as char);
            if start_ok && end_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Expand `-f %h` hazardous BIDS path (subject/session/datatype/entities).
pub fn expand_hazardous_h(dcm: &DicomImage, opts: &crate::opts::DcmOpts) -> String {
    use crate::reproin::sanitize_label;

    let mut sub = if !opts.bids_subject.is_empty() {
        sanitize_label(&opts.bids_subject)
    } else if !dcm.patient_id.is_empty() {
        sanitize_label(&dcm.patient_id)
    } else {
        "1".into()
    };
    if sub.is_empty() {
        sub = "1".into();
    }
    let mut ses = if !opts.bids_session.is_empty() {
        sanitize_label(&opts.bids_session)
    } else if !dcm.study_date.is_empty() && !dcm.study_time.is_empty() {
        let t: String = dcm.study_time.chars().filter(|c| c.is_ascii_digit()).take(6).collect();
        sanitize_label(&format!("{}T{t}", dcm.study_date))
    } else {
        "1".into()
    };
    if ses.is_empty() {
        ses = "1".into();
    }

    if dcm.csa.bids_data_type.is_empty() {
        let proto = if dcm.protocol_name.is_empty() {
            &dcm.series_description
        } else {
            &dcm.protocol_name
        };
        return format!("Unknown/{}_{}", dcm.series_number, proto);
    }

    let mut out = format!(
        "sub-{sub}/ses-{ses}/{}/sub-{sub}_ses-{ses}",
        dcm.csa.bids_data_type
    );
    if dcm.csa.bids_data_type == "func" {
        let task = if dcm.csa.bids_task.is_empty() {
            "rest"
        } else {
            dcm.csa.bids_task.as_str()
        };
        out.push_str(&format!("_task-{task}"));
    }
    out.push_str(&dcm.csa.bids_entity_suffix);
    out
}
