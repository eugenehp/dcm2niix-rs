//! Siemens CSA private-header parsing (image + series ASCII).
//!
//! Ports the subset of `readCSAImageHeader` / `siemensCsaAscii` needed for
//! mosaic demosaic and BIDS sidecars on `dcm_qa*`.

use dicom_core::Tag;
use dicom_object::DefaultDicomObject;

const CSA_IMAGE: Tag = Tag(0x0029, 0x1010);
const CSA_SERIES: Tag = Tag(0x0029, 0x1020);

/// Binary CSA image header (`(0029,1010)`).
#[derive(Debug, Clone)]
pub struct CsaImage {
    pub mosaic_slices: i32,
    /// Milliseconds; empty when unknown.
    pub slice_timing_ms: Vec<f64>,
    pub bandwidth_per_pixel_phase_encode: f64,
    pub phase_encoding_direction_positive: i32,
    pub slice_measurement_duration_ms: f64,
    /// CSA `SliceNormalVector` (1-indexed like dcm2niix: index 0 unused).
    pub slice_norm: [f64; 4],
    pub slice_order: u8,
    /// CSA `ImaRelTablePosition` (1-indexed, Z already negated like C++).
    pub table_pos: [f64; 4],
    /// Nanoseconds from CSA `RealDwellTime` if present.
    pub real_dwell_time_ns: f64,
    pub multi_band_factor: i32,
    /// Number of DTI directions parsed (`CSA.numDti`); gates DiffusionScheme.
    pub num_dti: i32,
    /// CSA `VoiPhaseFoV` (mm); `0` = absent.
    pub voi_phase_fov: f64,
    /// CSA `VoiReadoutFoV` (mm); `0` = absent.
    pub voi_readout_fov: f64,
    /// CSA `VoiThickness` (mm); `0` = absent.
    pub voi_thickness: f64,
    /// CSA `VoiPosition` center in patient LPS (mm).
    pub voi_center_lps: [f64; 3],
    /// True when `VoiPosition` / MidSlabPosition was present.
    pub has_voi_center: bool,
    /// CSA `ImageOrientationPatient` when public IOP is absent (6 values).
    pub image_orientation: Option<[f64; 6]>,
}

impl Default for CsaImage {
    fn default() -> Self {
        Self {
            mosaic_slices: 0,
            slice_timing_ms: Vec::new(),
            bandwidth_per_pixel_phase_encode: 0.0,
            phase_encoding_direction_positive: -1,
            slice_measurement_duration_ms: 0.0,
            slice_norm: [0.0; 4],
            slice_order: 0,
            table_pos: [0.0; 4],
            real_dwell_time_ns: 0.0,
            multi_band_factor: 1,
            num_dti: 0,
            voi_phase_fov: 0.0,
            voi_readout_fov: 0.0,
            voi_thickness: 0.0,
            voi_center_lps: [0.0; 3],
            has_voi_center: false,
            image_orientation: None,
        }
    }
}

/// ASCII Phoenix / ASCCONV block from `(0029,1020)`.
#[derive(Debug, Clone)]
pub struct CsaSeries {
    pub base_resolution: i32,
    pub echo_spacing_us: i32,
    pub parallel_reduction_factor_in_plane: i32,
    pub parallel_reduction_factor_out_of_plane: i32,
    pub ref_lines_pe: i32,
    pub phase_encoding_lines: i32,
    pub partial_fourier: i32,
    pub interp: i32,
    pub uc_mode: i32,
    pub exist_uc_image_numb: i32,
    pub delay_time_s: f64,
    pub tx_ref_amp: f64,
    pub phase_resolution: f64,
    pub phase_oversampling: f64,
    /// CSA `dAveragesDouble` (fractional; independent of `(0018,0083)`).
    pub averages_double: f64,
    pub shim_setting: [f64; 8],
    pub coil_id: String,
    pub coil_string: String,
    pub consistency_info: String,
    pub pulse_sequence_details: String,
    pub protocol_name: String,
    pub wip_mem_block: String,
    /// CSA region after ASCCONV END (`FmriExternalInfo`, `||`-delimited); often empty.
    pub fmri_external_info: String,
    /// CSA `ucCoilCombineMode` (1 = Sum of Squares, 2 = Adaptive Combine).
    pub combine_mode: i32,
    /// CSA `sPat.ucPATMode` (1 = SENSE, 2 = GRAPPA).
    pub pat_mode: i32,
    /// CSA `sAsl.sPostLabelingDelay[0]` (µs).
    pub post_labeling_delay_us: f64,
    /// CSA `sAsl.ulLabelingDuration` (µs).
    pub labeling_duration_us: f64,
    /// CSA `sWipMemBlock.alFree[*]` (up to 64).
    pub al_free: [f64; 64],
    /// CSA `sWipMemBlock.adFree[*]` (NaN when absent).
    pub ad_free: [f64; 64],
    /// CSA `alTI[*]` (NaN when absent).
    pub al_ti: [f64; 64],
    /// CSA `sRSatArray.asElm[1]` labelling-plane thickness (mm); 0 = absent.
    pub tag_plane_thickness: f64,
    pub tag_plane_ul_shape: f64,
    pub tag_plane_position_d_tra: f64,
    pub tag_plane_normal_d_tra: f64,
    /// CSA `lContrasts` (echo / contrast count).
    pub l_contrasts: i32,
    /// CSA `sSliceArray.lConc` (concatenations / 3D-EPI multi-echo shots).
    pub l_conc: i32,
    /// CSA `sPrepPulses.ucMTC` (`1` = MT on).
    pub uc_mtc: i32,
    /// CSA `sDiffusion.ulMode` / bipolar flag (`1` = bipolar).
    pub dif_bipolar: i32,
    /// CSA `sWipMemBlock.alTE[*]` (µs); NaN when absent.
    pub al_te: [f64; 8],
    /// CSA `lInvContrasts` (MP2RAGE).
    pub l_inv_contrasts: i32,
    /// CSA `sAsl.ulSuppressionMode` (`0` = off / absent).
    pub ul_suppression_mode: i32,
}

impl Default for CsaSeries {
    fn default() -> Self {
        Self {
            base_resolution: 0,
            echo_spacing_us: 0,
            parallel_reduction_factor_in_plane: 0,
            parallel_reduction_factor_out_of_plane: 0,
            ref_lines_pe: 0,
            phase_encoding_lines: 0,
            partial_fourier: 0,
            interp: 0,
            uc_mode: 0,
            exist_uc_image_numb: 0,
            delay_time_s: 0.0,
            tx_ref_amp: 0.0,
            phase_resolution: 0.0,
            phase_oversampling: 0.0,
            averages_double: 0.0,
            shim_setting: [0.0; 8],
            coil_id: String::new(),
            coil_string: String::new(),
            consistency_info: String::new(),
            pulse_sequence_details: String::new(),
            protocol_name: String::new(),
            wip_mem_block: String::new(),
            fmri_external_info: String::new(),
            combine_mode: 0,
            pat_mode: 0,
            post_labeling_delay_us: 0.0,
            labeling_duration_us: 0.0,
            al_free: [0.0; 64],
            ad_free: [f64::NAN; 64],
            al_ti: [f64::NAN; 64],
            tag_plane_thickness: 0.0,
            tag_plane_ul_shape: 0.0,
            tag_plane_position_d_tra: 0.0,
            tag_plane_normal_d_tra: 0.0,
            l_contrasts: 0,
            l_conc: 0,
            uc_mtc: 0,
            dif_bipolar: 0,
            al_te: [f64::NAN; 8],
            l_inv_contrasts: 0,
            ul_suppression_mode: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CsaMeta {
    pub image: CsaImage,
    pub series: CsaSeries,
    /// BIDS datatype guess (`anat`/`func`/`dwi`/…); filled by `setBids`.
    pub bids_data_type: String,
    /// BIDS entity suffix (`_acq-…_dir-AP_run-6_bold`).
    pub bids_entity_suffix: String,
    /// BIDS task label for func (`rest`, …).
    pub bids_task: String,
}

pub fn read_csa(obj: &DefaultDicomObject) -> CsaMeta {
    let mut meta = CsaMeta::default();
    if let Some(bytes) = element_bytes(obj, CSA_IMAGE) {
        meta.image = parse_csa_image(&bytes);
    }
    if let Some(bytes) = element_bytes(obj, CSA_SERIES) {
        meta.series = parse_csa_series(&bytes);
    }
    meta
}

fn element_bytes(obj: &DefaultDicomObject, tag: Tag) -> Option<Vec<u8>> {
    obj.element(tag)
        .ok()?
        .to_bytes()
        .ok()
        .map(|c| c.into_owned())
}

fn parse_csa_image(buff: &[u8]) -> CsaImage {
    let mut out = CsaImage::default();
    if buff.len() < 36 || &buff[0..4] != b"SV10" {
        return out;
    }
    let mut l_pos = 8usize;
    let ln_tag = read_i32_le(buff, l_pos) as i32;
    if ln_tag < 1 || ln_tag > 128 || buff.get(l_pos + 4) != Some(&77) {
        return out;
    }
    l_pos += 8;
    for _ in 0..ln_tag {
        if l_pos + 84 > buff.len() {
            break;
        }
        let name = read_name(&buff[l_pos..l_pos + 64]);
        let nitems = read_i32_le(buff, l_pos + 76) as i32;
        l_pos += 84;
        match name.as_str() {
            "NumberOfImagesInMosaic" => {
                out.mosaic_slices = csa_first_float(buff, l_pos, nitems).round() as i32;
            }
            "SliceNormalVector" if nitems > 2 => {
                let (vals, _) = csa_multi_float(buff, l_pos, nitems);
                if vals.len() >= 3 {
                    out.slice_norm[1] = vals[0] as f64;
                    out.slice_norm[2] = vals[1] as f64;
                    out.slice_norm[3] = vals[2] as f64;
                }
            }
            "MosaicRefAcqTimes" if nitems > 3 => {
                let (vals, n) = csa_multi_float(buff, l_pos, nitems);
                out.slice_timing_ms = vals.into_iter().take(n).map(|v| v as f64).collect();
                out.slice_order = infer_slice_order(&out.slice_timing_ms);
                if let Some(&t0) = out.slice_timing_ms.first() {
                    out.multi_band_factor = out
                        .slice_timing_ms
                        .iter()
                        .filter(|t| (*t - t0).abs() < 1e-4)
                        .count() as i32;
                }
            }
            "BandwidthPerPixelPhaseEncode" => {
                out.bandwidth_per_pixel_phase_encode = csa_first_float(buff, l_pos, nitems) as f64;
            }
            "PhaseEncodingDirectionPositive" => {
                out.phase_encoding_direction_positive =
                    csa_first_float(buff, l_pos, nitems).round() as i32;
            }
            "ImaRelTablePosition" => {
                let (vals, _) = csa_multi_float(buff, l_pos, nitems);
                if vals.len() >= 3 {
                    out.table_pos[0] = 1.0;
                    out.table_pos[1] = vals[0] as f64;
                    out.table_pos[2] = vals[1] as f64;
                    out.table_pos[3] = -(vals[2] as f64);
                }
            }
            "RealDwellTime" => {
                out.real_dwell_time_ns = csa_first_float(buff, l_pos, nitems) as f64;
            }
            "SliceMeasurementDuration" => {
                out.slice_measurement_duration_ms = csa_first_float(buff, l_pos, nitems) as f64;
            }
            "VoiPhaseFoV" => {
                let v = csa_first_f64(buff, l_pos, nitems);
                if v > 0.0 {
                    out.voi_phase_fov = v;
                }
            }
            "VoiReadoutFoV" => {
                let v = csa_first_f64(buff, l_pos, nitems);
                if v > 0.0 {
                    out.voi_readout_fov = v;
                }
            }
            "VoiThickness" => {
                let v = csa_first_f64(buff, l_pos, nitems);
                if v > 0.0 {
                    out.voi_thickness = v;
                }
            }
            "VoiPosition" if nitems >= 3 => {
                let vals = csa_multi_f64(buff, l_pos, nitems);
                if vals.len() >= 3 {
                    out.voi_center_lps = [vals[0], vals[1], vals[2]];
                    out.has_voi_center = true;
                }
            }
            "ImageOrientationPatient" if nitems >= 6 => {
                let vals = csa_multi_f64(buff, l_pos, nitems);
                if vals.len() >= 6 {
                    out.image_orientation =
                        Some([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]]);
                }
            }
            _ => {}
        }
        l_pos = skip_csa_items(buff, l_pos, nitems);
    }
    out
}

/// Port of `checkSliceTimes` → NIfTI `slice_code`.
fn infer_slice_order(times: &[f64]) -> u8 {
    let n = times.len();
    if n < 3 {
        return 0;
    }
    let mut t = times.to_vec();
    let min_t = t.iter().copied().fold(f64::INFINITY, f64::min);
    if min_t < 0.0 {
        for v in &mut t {
            *v -= min_t;
        }
    }
    let n_zero = t.iter().filter(|v| **v == 0.0).count();
    let min_idx = t
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    if n_zero >= 2 {
        return 0;
    }
    if min_idx == 1 {
        5
    } else if min_idx == n - 2 {
        6
    } else if min_idx == 0 && t[1] < t[2] {
        1
    } else if min_idx == 0 && t[1] > t[2] {
        3
    } else if min_idx == n - 1 && t[n - 3] > t[n - 2] {
        2
    } else if min_idx == n - 1 && t[n - 3] < t[n - 2] {
        4
    } else {
        0
    }
}

fn parse_csa_series(buff: &[u8]) -> CsaSeries {
    let mut out = CsaSeries::default();
    let start = phoenix_offset(buff);
    let buf = if start > 0 && start < buff.len() {
        &buff[start..]
    } else {
        buff
    };
    let begin = find_subslice(buf, b"### ASCCONV BEGIN").unwrap_or(buf);
    let ascii = if let Some(end) = begin
        .windows(b"### ASCCONV END".len())
        .position(|w| w == b"### ASCCONV END")
    {
        &begin[..end]
    } else {
        begin
    };
    // FmriExternalInfo is after AscConvEnd with `||` delimiters; C++ read path
    // is still commented out, so leave empty (emitting would be a no-op via json_str).

    out.phase_encoding_lines = read_key_i32(ascii, "sKSpace.lPhaseEncodingLines");
    out.exist_uc_image_numb = read_key_i32(ascii, "sSliceArray.ucImageNumb");
    out.uc_mode = read_key_i32_neg(ascii, "sSliceArray.ucMode");
    out.base_resolution = read_key_i32(ascii, "sKSpace.lBaseResolution");
    out.interp = read_key_i32(ascii, "sKSpace.uc2DInterpolation");
    out.uc_mtc = read_key_i32(ascii, "sPrepPulses.ucMTC");
    out.dif_bipolar = read_key_i32(ascii, "sDiffusion.dsScheme");
    if out.dif_bipolar == 0 {
        let rom = read_key_i32(ascii, "sWipMemBlock.ucReadOutMode");
        if (1..=2).contains(&rom) {
            out.dif_bipolar = 3 - rom; // CMRR: invert
        }
    }
    out.l_inv_contrasts = read_key_i32(ascii, "lInvContrasts");
    out.ul_suppression_mode = read_key_i32(ascii, "sAsl.ulSuppressionMode");
    for k in 0..8 {
        out.al_te[k] = read_key_f64(ascii, &format!("alTE[{k}]")).unwrap_or(f64::NAN);
    }
    out.partial_fourier = read_key_i32(ascii, "sKSpace.ucPhasePartialFourier");
    out.echo_spacing_us = read_key_i32(ascii, "sFastImaging.lEchoSpacing");
    out.parallel_reduction_factor_in_plane = read_key_i32(ascii, "sPat.lAccelFactPE");
    out.parallel_reduction_factor_out_of_plane = read_key_i32(ascii, "sPat.lAccelFact3D");
    out.ref_lines_pe = read_key_i32(ascii, "sPat.lRefLinesPE");
    out.delay_time_s = read_key_f64(ascii, "lDelayTimeInTR")
        .map(|v| v / 1_000_000.0)
        .unwrap_or(0.0);
    out.phase_resolution = read_key_f64(ascii, "sKSpace.dPhaseResolution").unwrap_or(0.0);
    out.phase_oversampling =
        read_key_f64(ascii, "sKSpace.dPhaseOversamplingForDialog").unwrap_or(0.0);
    out.averages_double = read_key_f64(ascii, "dAveragesDouble").unwrap_or(0.0);
    out.tx_ref_amp = read_key_f64(ascii, "sTXSPEC.asNucleusInfo[0].flReferenceAmplitude").unwrap_or(0.0);

    out.shim_setting[0] = read_key_f64(ascii, "sGRADSPEC.asGPAData[0].lOffsetX").unwrap_or(0.0);
    out.shim_setting[1] = read_key_f64(ascii, "sGRADSPEC.asGPAData[0].lOffsetY").unwrap_or(0.0);
    out.shim_setting[2] = read_key_f64(ascii, "sGRADSPEC.asGPAData[0].lOffsetZ").unwrap_or(0.0);
    if out.shim_setting[0] == 0.0 {
        out.shim_setting[0] = read_key_f64(ascii, "sGRADSPEC.lOffsetX").unwrap_or(0.0);
    }
    if out.shim_setting[1] == 0.0 {
        out.shim_setting[1] = read_key_f64(ascii, "sGRADSPEC.lOffsetY").unwrap_or(0.0);
    }
    if out.shim_setting[2] == 0.0 {
        out.shim_setting[2] = read_key_f64(ascii, "sGRADSPEC.lOffsetZ").unwrap_or(0.0);
    }
    out.shim_setting[3] = read_key_f64(ascii, "sGRADSPEC.alShimCurrent[0]").unwrap_or(0.0);
    out.shim_setting[4] = read_key_f64(ascii, "sGRADSPEC.alShimCurrent[1]").unwrap_or(0.0);
    out.shim_setting[5] = read_key_f64(ascii, "sGRADSPEC.alShimCurrent[2]").unwrap_or(0.0);
    out.shim_setting[6] = read_key_f64(ascii, "sGRADSPEC.alShimCurrent[3]").unwrap_or(0.0);
    out.shim_setting[7] = read_key_f64(ascii, "sGRADSPEC.alShimCurrent[4]").unwrap_or(0.0);

    out.coil_id = read_key_str(ascii, "sCoilElementID.tCoilID");
    out.consistency_info = read_key_str(ascii, "sProtConsistencyInfo.tMeasuredBaselineString");
    if out.consistency_info.is_empty() {
        out.consistency_info = read_key_str(ascii, "sProtConsistencyInfo.tBaselineString");
    }
    out.coil_string = read_key_str(ascii, "sCoilSelectMeas.sCoilStringForConversion");
    out.pulse_sequence_details = read_key_str(ascii, "tSequenceFileName");
    out.protocol_name = read_key_str(ascii, "tProtocolName");
    out.wip_mem_block = read_key_str(ascii, "sWipMemBlock.tFree");
    out.combine_mode = read_key_i32_neg(ascii, "ucCoilCombineMode");
    out.pat_mode = read_key_i32_neg(ascii, "sPat.ucPATMode");
    out.post_labeling_delay_us =
        read_key_f64(ascii, "sAsl.sPostLabelingDelay[0]").unwrap_or(0.0);
    out.labeling_duration_us =
        read_key_f64(ascii, "sAsl.ulLabelingDuration").unwrap_or(0.0);
    out.l_contrasts = read_key_i32(ascii, "lContrasts");
    // issue 1024: DZNE 3D-EPI encodes multi-echo shots as concatenations.
    out.l_conc = read_key_i32(ascii, "sSliceArray.lConc");
    // WIP free arrays (ASL / Oxford VEPCASL / FWF).
    for k in 0..64 {
        out.al_ti[k] = f64::NAN;
        out.ad_free[k] = f64::NAN;
        out.al_free[k] = 0.0;
    }
    if ascii.windows(b"alTI[".len()).any(|w| w == b"alTI[") {
        for k in 0..64 {
            out.al_ti[k] =
                read_key_f64(ascii, &format!("alTI[{k}]")).unwrap_or(f64::NAN);
        }
    }
    if ascii
        .windows(b"sWipMemBlock.alFree[".len())
        .any(|w| w == b"sWipMemBlock.alFree[")
    {
        for k in 0..64 {
            out.al_free[k] =
                read_key_f64(ascii, &format!("sWipMemBlock.alFree[{k}]")).unwrap_or(0.0);
        }
    }
    let ad_prefix = if ascii
        .windows(b"sWipMemBlock.adFree[".len())
        .any(|w| w == b"sWipMemBlock.adFree[")
    {
        "sWipMemBlock.adFree["
    } else if ascii
        .windows(b"sWiPMemBlock.adFree[".len())
        .any(|w| w == b"sWiPMemBlock.adFree[")
    {
        "sWiPMemBlock.adFree["
    } else {
        ""
    };
    if !ad_prefix.is_empty() {
        for k in 0..64 {
            out.ad_free[k] =
                read_key_f64(ascii, &format!("{ad_prefix}{k}]")).unwrap_or(f64::NAN);
        }
    }
    out.tag_plane_thickness =
        read_key_f64(ascii, "sRSatArray.asElm[1].dThickness").unwrap_or(0.0);
    if out.tag_plane_thickness > 0.0 {
        out.tag_plane_ul_shape =
            read_key_f64(ascii, "sRSatArray.asElm[1].ulShape").unwrap_or(0.0);
        out.tag_plane_position_d_tra =
            read_key_f64(ascii, "sRSatArray.asElm[1].sPosition.dTra").unwrap_or(0.0);
        out.tag_plane_normal_d_tra =
            read_key_f64(ascii, "sRSatArray.asElm[1].sNormal.dTra").unwrap_or(0.0);
    }
    out
}

fn phoenix_offset(buff: &[u8]) -> usize {
    if buff.len() < 36 || &buff[0..4] != b"SV10" {
        return 0;
    }
    let mut l_pos = 8usize;
    let ln_tag = read_i32_le(buff, l_pos) as i32;
    if ln_tag < 1 || buff.get(l_pos + 4) != Some(&77) {
        return 0;
    }
    l_pos += 8;
    for _ in 0..ln_tag {
        if l_pos + 84 > buff.len() {
            break;
        }
        let name = read_name(&buff[l_pos..l_pos + 64]);
        let nitems = read_i32_le(buff, l_pos + 76) as i32;
        l_pos += 84;
        if name == "MrPhoenixProtocol" {
            return l_pos;
        }
        l_pos = skip_csa_items(buff, l_pos, nitems);
    }
    0
}

fn skip_csa_items(buff: &[u8], mut l_pos: usize, nitems: i32) -> usize {
    for _ in 0..nitems.max(0) {
        if l_pos + 16 > buff.len() {
            break;
        }
        let xx2_len = read_i32_le(buff, l_pos + 4) as i32;
        l_pos += 16;
        l_pos += ((xx2_len + 3) / 4 * 4) as usize;
    }
    l_pos
}

fn csa_first_float(buff: &[u8], l_pos: usize, nitems: i32) -> f32 {
    csa_multi_float(buff, l_pos, nitems).0.into_iter().next().unwrap_or(0.0)
}

fn csa_first_f64(buff: &[u8], l_pos: usize, nitems: i32) -> f64 {
    csa_multi_f64(buff, l_pos, nitems).into_iter().next().unwrap_or(0.0)
}

fn csa_multi_f64(buff: &[u8], mut l_pos: usize, nitems: i32) -> Vec<f64> {
    let mut out = Vec::new();
    for _ in 0..nitems.max(0) {
        if l_pos + 16 > buff.len() {
            break;
        }
        let xx2_len = read_i32_le(buff, l_pos + 4) as i32;
        l_pos += 16;
        if xx2_len > 0 && l_pos + xx2_len as usize <= buff.len() {
            let s = std::str::from_utf8(&buff[l_pos..l_pos + xx2_len as usize])
                .unwrap_or("")
                .trim_matches('\0');
            if let Ok(v) = s.trim().parse::<f64>() {
                out.push(v);
            }
        }
        l_pos += ((xx2_len + 3) / 4 * 4) as usize;
    }
    out
}

fn csa_multi_float(buff: &[u8], mut l_pos: usize, nitems: i32) -> (Vec<f32>, usize) {
    let mut out = Vec::new();
    for _ in 0..nitems.max(0) {
        if l_pos + 16 > buff.len() {
            break;
        }
        let xx2_len = read_i32_le(buff, l_pos + 4) as i32;
        l_pos += 16;
        if xx2_len > 0 && l_pos + xx2_len as usize <= buff.len() {
            let s = std::str::from_utf8(&buff[l_pos..l_pos + xx2_len as usize])
                .unwrap_or("")
                .trim_matches('\0');
            if let Ok(v) = s.trim().parse::<f64>() {
                out.push(v as f32);
            }
        }
        l_pos += ((xx2_len + 3) / 4 * 4) as usize;
    }
    let n = out.len();
    (out, n)
}

fn read_name(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

fn read_i32_le(buf: &[u8], off: usize) -> i32 {
    if off + 4 > buf.len() {
        return 0;
    }
    i32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn find_subslice<'a>(hay: &'a [u8], needle: &[u8]) -> Option<&'a [u8]> {
    hay.windows(needle.len())
        .position(|w| w == needle)
        .map(|i| &hay[i..])
}

fn read_key_i32(data: &[u8], key: &str) -> i32 {
    read_key_i32_neg(data, key).max(0)
}

fn read_key_i32_neg(data: &[u8], key: &str) -> i32 {
    let Some(pos) = find_subslice(data, key.as_bytes()) else {
        return -1;
    };
    let rest = &pos[key.len()..];
    let mut ret = 0i32;
    let mut seen = false;
    for &b in rest {
        if b == b'\n' {
            break;
        }
        if b.is_ascii_digit() {
            ret = ret * 10 + (b - b'0') as i32;
            seen = true;
        }
    }
    if seen { ret } else { -1 }
}

fn read_key_f64(data: &[u8], key: &str) -> Option<f64> {
    let pos = find_subslice(data, key.as_bytes())?;
    let rest = &pos[key.len()..];
    let mut s = String::new();
    for &b in rest {
        if b == b'\n' {
            break;
        }
        if b.is_ascii_digit() || b == b'.' || b == b'-' {
            s.push(b as char);
        } else if !s.is_empty() {
            break;
        }
    }
    if s.is_empty() {
        None
    } else {
        s.parse().ok()
    }
}

fn read_key_str(data: &[u8], key: &str) -> String {
    // Match C++ `readKeyStr`: skip empty `""` pairs then copy until the next quote.
    let Some(pos) = find_subslice(data, key.as_bytes()) else {
        return String::new();
    };
    let rest = &pos[key.len()..];
    let mut out = String::new();
    let mut in_quote = false;
    for &b in rest {
        if b == b'\n' {
            break;
        }
        if b == b'"' {
            if !out.is_empty() {
                break;
            }
            in_quote = true;
            continue;
        }
        if in_quote {
            out.push(b as char);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_key_parses_hex_pat_mode() {
        let data = b"sPat.ucPATMode = 0x2\nucCoilCombineMode = 1\n";
        assert_eq!(read_key_i32_neg(data, "sPat.ucPATMode"), 2);
        assert_eq!(read_key_i32_neg(data, "ucCoilCombineMode"), 1);
    }

    #[test]
    fn read_key_str_skips_empty_quotes() {
        let data = br#"tSequenceFileName = ""%SiemensSeq%\ep2d_bold""
sCoilElementID.tCoilID = ""HeadMatrix""
"#;
        assert_eq!(
            read_key_str(data, "tSequenceFileName"),
            "%SiemensSeq%\\ep2d_bold"
        );
        assert_eq!(read_key_str(data, "sCoilElementID.tCoilID"), "HeadMatrix");
    }
}
