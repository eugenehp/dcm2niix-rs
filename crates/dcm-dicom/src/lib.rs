//! DICOM file metadata — Rust counterpart of upstream `TDICOMdata`
//! (fields this port actually consumes).
//!
//! Pixel decode uses `dicom-pixeldata`. Siemens CSA / mosaic live in `csa`;
//! enhanced multi-frame in `enhanced`; GE `(0025,101B)` protocol blocks in
//! `ge_protocol`.

mod asl;
mod csa;
mod enhanced;
mod ge_diff_cycling;
mod ge_pepolar;
mod ge_protocol;
mod pmsct;
mod uih_scan;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use asl::{
    flags_from_asl_contrast, flags_from_ge_contrast_technique, flags_from_ge_labeling_technique,
    flags_from_philips_label_type, ASL_FLAG_GE_3DCASL, ASL_FLAG_GE_3DPCASL,
    ASL_FLAG_GE_CONTINUOUS, ASL_FLAG_GE_PSEUDOCONTINUOUS, ASL_FLAG_GE_PULSED, ASL_FLAG_NONE,
    ASL_FLAG_PHILIPS_CONTROL, ASL_FLAG_PHILIPS_LABEL,
};
pub use csa::{CsaImage, CsaMeta, CsaSeries};
pub use enhanced::{
    assign_grad_dyn_vol, expand_frames, infer_stack_dims, is_enhanced_multiframe,
    read_per_frame_geometry, scale_or_te_varies, sort_frames_by_dimension_index, volume_contrasts,
    FrameGeom, VolumeContrast,
};
pub use ge_diff_cycling::{
    detect_diff_cycling, GE_DIFF_CYCLING_2TR, GE_DIFF_CYCLING_3TR, GE_DIFF_CYCLING_ALLTR,
    GE_DIFF_CYCLING_OFF, GE_DIFF_CYCLING_SPOFF, GE_DIFF_CYCLING_UNKNOWN,
};
pub use ge_pepolar::{
    finalize_pepolar, is_pepolar, needs_extra_y_flip, GE_EPI_EPI, GE_EPI_EPI2, GE_EPI_EPIRT,
    GE_EPI_PEPOLAR_FWD, GE_EPI_PEPOLAR_FWD_REV, GE_EPI_PEPOLAR_FWD_REV_FLIP, GE_EPI_PEPOLAR_REV,
    GE_EPI_PEPOLAR_REV_FWD, GE_EPI_PEPOLAR_REV_FWD_FLIP, GE_EPI_UNKNOWN, GE_PE_FLIPPED,
    GE_PE_UNFLIPPED,
};
pub use ge_protocol::{parse_ge_protocol_block, GeProtocolBlock};
pub use pmsct::decode_pmsct_rle1;

use crc32fast::Hasher;
use dicom_core::Tag;
use dicom_dictionary_std::tags;
use dicom_object::mem::InMemDicomObject;
use dicom_object::{DefaultDicomObject, FileDicomObject, OpenFileOptions, open_file};
use dicom_pixeldata::{ConvertOptions, ModalityLutOption, PixelDecoder, PixelRepresentation, VoiLutOption};
use dcm_core::error::{Error, Result};
use dcm_core::snap_f32;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Manufacturer {
    Unknown = 0,
    Siemens = 1,
    Ge = 2,
    Philips = 3,
    Toshiba = 4,
    Uih = 5,
    Bruker = 6,
    Hitachi = 7,
    Canon = 8,
    Mediso = 9,
    Hyperfine = 11,
}

impl Manufacturer {
    pub fn from_tag(s: &str) -> Self {
        let u = s.to_ascii_uppercase();
        if u.contains("SIEMENS") {
            Manufacturer::Siemens
        } else if u.contains("GE MEDICAL") || u == "GE" {
            Manufacturer::Ge
        } else if u.contains("PHILIPS") {
            Manufacturer::Philips
        } else if u.contains("TOSHIBA") || u.contains("CANON") {
            if u.contains("CANON") {
                Manufacturer::Canon
            } else {
                Manufacturer::Toshiba
            }
        } else if u.contains("UIH") || u.contains("UNITED IMAGING") {
            Manufacturer::Uih
        } else if u.contains("BRUKER") {
            Manufacturer::Bruker
        } else if u.contains("HITACHI") {
            Manufacturer::Hitachi
        } else if u.contains("MEDISO") {
            Manufacturer::Mediso
        } else if u.contains("HYPERFINE") {
            Manufacturer::Hyperfine
        } else {
            Manufacturer::Unknown
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Manufacturer::Siemens => "Siemens",
            Manufacturer::Ge => "GE",
            Manufacturer::Philips => "Philips",
            Manufacturer::Toshiba => "Toshiba",
            Manufacturer::Uih => "UIH",
            Manufacturer::Bruker => "Bruker",
            Manufacturer::Hitachi => "Hitachi",
            Manufacturer::Canon => "Canon",
            Manufacturer::Mediso => "Mediso",
            Manufacturer::Hyperfine => "Hyperfine",
            Manufacturer::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Unknown,
    Cr,
    Ct,
    Mr,
    Pt,
    Us,
    /// DICOM Segmentation Storage (`SEG`).
    Seg,
}

impl Modality {
    pub fn from_tag(s: &str) -> Self {
        match s.trim() {
            "CR" => Modality::Cr,
            "CT" => Modality::Ct,
            "MR" => Modality::Mr,
            "PT" => Modality::Pt,
            "US" => Modality::Us,
            "SEG" => Modality::Seg,
            _ => Modality::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Cr => "CR",
            Modality::Ct => "CT",
            Modality::Mr => "MR",
            Modality::Pt => "PT",
            Modality::Us => "US",
            Modality::Seg => "SEG",
            Modality::Unknown => "",
        }
    }
}

/// Per-file DICOM fields used for grouping, geometry, naming, and BIDS.
///
/// Roughly mirrors upstream `TDICOMdata` for the subset this port needs.
/// Pixel buffers are not stored here — decode on demand via
/// [`decode_pixels_raw_f32`].
/// One item from `(0012,0064)` DeidentificationMethodCodeSequence (issue 877).
#[derive(Debug, Clone, Default)]
pub struct DeIdCodeSequence {
    pub code_value: String,
    pub coding_scheme_designator: String,
    pub coding_scheme_version: String,
    pub code_meaning: String,
}

#[derive(Debug, Clone)]
pub struct DicomImage {
    pub path: PathBuf,
    pub series_uid: String,
    pub series_uid_crc: u32,
    pub instance_uid: String,
    pub study_uid: String,
    pub series_number: i64,
    pub instance_number: i32,
    pub acquisition_number: i32,
    pub echo_number: i32,
    pub rows: usize,
    pub columns: usize,
    pub bits_allocated: i32,
    pub bits_stored: i32,
    pub samples_per_pixel: i32,
    pub is_signed: bool,
    pub is_float: bool,
    /// PixelSpacing as [row, column]; xyz_mm is [unused, col, row, slice].
    pub xyz_mm: [f64; 4],
    pub slice_thickness: f64,
    /// ImageOrientationPatient as 1-indexed 6-vector (dcm2niix `orient[]`).
    pub orient: [f64; 7],
    /// ImagePositionPatient, 1-indexed.
    pub patient_position: [f64; 4],
    /// Last `(0020,0032)` in file — UIH mosaic slice direction.
    pub patient_position_last: [f64; 4],
    /// `(0020,1041)` when `(0020,0032)` is absent.
    pub last_scan_loc: f64,
    /// `(0018,9073)` — dcm2niix `acquisitionDuration` (UIH TotalReadoutTime = val/1000).
    pub acquisition_duration: f64,
    pub manufacturer: Manufacturer,
    pub modality: Modality,
    pub manufacturer_name: String,
    pub manufacturers_model_name: String,
    pub institution_name: String,
    pub institution_address: String,
    pub institutional_department: String,
    pub procedure_step_description: String,
    pub station_name: String,
    pub device_serial_number: String,
    pub software_versions: String,
    pub protocol_name: String,
    pub series_description: String,
    pub sequence_name: String,
    /// `(0018,9005)` / GE `(0019,109C)` PulseSequenceName.
    pub pulse_sequence_name: String,
    pub scanning_sequence: String,
    pub sequence_variant: String,
    pub scan_options: String,
    pub image_type: String,
    pub image_comments: String,
    pub coil_name: String,
    /// Siemens private coil string `(0051,100F)`, e.g. `T:HEA;HEP`.
    pub coil_string: String,
    /// `(0018,1251)` / MRTransmitCoilSequence TransmitCoilName (MRS).
    pub transmit_coil_name: String,
    pub patient_name: String,
    pub patient_id: String,
    pub patient_sex: String,
    pub patient_age: String,
    /// `(0008,0090)` ReferringPhysicianName.
    pub referring_physician_name: String,
    pub patient_birth_date: String,
    /// `(0010,1030)` kg; `0` = absent.
    pub patient_weight: f64,
    /// `(0010,1020)` m; `0` = absent.
    pub patient_size: f64,
    pub accession_number: String,
    pub study_id: String,
    pub study_description: String,
    pub study_date: String,
    pub study_time: String,
    pub series_time: String,
    pub acquisition_date: String,
    pub acquisition_time: String,
    pub body_part: String,
    pub tr: f64,
    pub te: f64,
    pub ti: f64,
    pub flip_angle: f64,
    pub field_strength: f64,
    pub pixel_bandwidth: f64,
    pub echo_train_length: i32,
    pub phase_encoding_rc: char,
    pub inten_scale: f32,
    pub inten_intercept: f32,
    /// Philips private ScaleSlope (SS).
    pub inten_scale_philips: f32,
    pub is_scale_varies_enh: bool,
    pub is_derived: bool,
    pub is_localizer: bool,
    pub number_of_frames: i32,
    pub imaging_frequency: f64,
    pub patient_position_label: String,
    pub spacing_between_slices: f64,
    pub acquisition_matrix_pe: i32,
    /// DICOM `(0018,9231)` / `(0018,0089)` phase encoding steps.
    pub phase_encoding_steps: i32,
    /// `(0018,9232)` MRAcquisitionPhaseEncodingStepsOutOfPlane (3D partitions).
    pub phase_encoding_steps_out_of_plane: i32,
    /// CSA `sSliceArray.lConc` / concatenations; default `1` (issue 1024).
    pub number_of_concatenations: i32,
    /// Per-shot TR in seconds when volume TR was computed (issue 1024); `<0` absent.
    pub repetition_time_excitation: f64,
    /// Reported TR preserved as BIDS `RepetitionTimeInversion` (issue 560); `0` = absent.
    pub repetition_time_inversion: f64,
    pub percent_phase_fov: f64,
    pub percent_sampling: f64,
    pub mra_acquisition_type: String,
    /// UIH `(0065,1009)` / standard diffusion b-value; `-1` when unknown.
    pub b_value: f64,
    /// UIH `(0065,1037)` gradient direction [x, y, z] in scanner coords.
    pub diffusion_direction: [f64; 3],
    /// UIH `(0065,1005)` phase-encoding direction label.
    pub pe_direction_displayed: String,
    pub number_of_averages: f64,
    pub is_3d_acq: bool,
    pub is_epi: bool,
    /// Inversion-recovery (`ScanningSequence` IR / `(0018,9009)` YES).
    pub is_ir: bool,
    /// UIH `(0065,100D)` in-plane parallel factor (e.g. `F:2S` → 2).
    pub accel_fact_pe: f64,
    /// GE `(0019,109E)`.
    pub internal_pulse_sequence_name: String,
    /// GE `(0043,1002/1003/1004)`.
    pub shim_setting: [f64; 3],
    pub prescan_reuse_string: String,
    /// GE `(0043,102C)` microseconds.
    pub effective_echo_spacing_ge: f64,
    /// GE `(0019,105A)` converted to seconds.
    pub acquisition_duration_s: f64,
    /// `-1` unknown, `0` unflipped, `4` flipped.
    pub phase_encoding_ge: i32,
    pub parallel_reduction_out_of_plane: f64,
    pub sar: f64,
    pub dwell_time_ns: f64,
    pub csa: CsaMeta,
    pub is_mosaic: bool,
    pub image_orientation_text: String,
    pub is_mrs: bool,
    /// Water-reference MRS (`WaterSuppressed: false` in BIDS).
    pub is_mrs_ref: bool,
    /// `(0028,9002)` SpectroscopyAcquisitionDataColumns.
    pub data_point_columns: i32,
    /// `(0018,9100)` e.g. `"1H"`.
    pub resonant_nucleus: String,
    /// `(0018,9200)` MRS acquisition type: 0=none/SVS, 1=ROW, 2=PLANE, 3=VOLUME.
    pub mrs_acq_type: i32,
    /// MRS VOI phase FoV (mm); CSA / VolumeLocalizationSequence.
    pub voi_phase_fov: f64,
    /// MRS VOI readout FoV (mm).
    pub voi_readout_fov: f64,
    /// MRS VOI thickness (mm); gate for BIDS `VOI` with `has_voi_center`.
    pub voi_thickness: f64,
    /// MRS VOI center in patient LPS (mm).
    pub voi_center_lps: [f64; 3],
    /// True when VOI center was explicitly populated (audit M8).
    pub has_voi_center: bool,
    /// Full-precision IOP for [`Self::mrs_voi_matrix`] (no `snap_f32`); falls
    /// back to [`Self::orient`] when unset. 1-indexed like `orient`.
    pub voi_orient: [f64; 7],
    /// `(0018,9093)` NumberOfKSpaceTrajectories (MRS).
    pub number_of_k_space_trajectories: i32,
    /// `(0018,9052)` SpectralWidth (Hz); `0` = absent.
    pub spectral_width_hz: f64,
    /// Siemens NumarisX / XA line (phase convention for MRS).
    pub is_xa: bool,
    /// PMSCT_RLE1 / Elscint private compression (`07a1,100a`).
    pub is_pmsct_rle1: bool,
    /// Diffusion vectors already in world/scanner coordinates (skip GE image-space remap).
    pub is_bvec_world_coordinates: bool,
    /// DICOM `(0018,1120)` gantry/detector tilt (degrees); refined by slice geometry for CT.
    pub gantry_tilt: f64,
    pub study_uid_crc: u32,
    pub coil_crc: u32,
    /// `(studyDate * 1e6) + studyTime` — C++ `dateTime`.
    pub date_time: f64,
    pub is_has_phase: bool,
    pub is_has_real: bool,
    pub is_has_imaginary: bool,
    /// ImageType contains MAGNITUDE (and not only phase/real/imag).
    pub is_has_magnitude: bool,
    /// Siemens RF-off (`is_no_rf`) from ImageTypeText `NOISE`.
    pub is_no_rf: bool,
    /// Siemens `(0021,1175)` ImageTypeText (`\` normalized to `_`).
    pub image_type_text: String,
    /// Deep-learning recon flag (Siemens DRB/DRG/DRS, Philips/GE private text).
    pub is_deep_learning: bool,
    /// Deep-learning details string (`(0021,1176)` / vendor private).
    pub deep_learning_text: String,
    /// `(0018,9231)` companion: frequency encoding steps (`(0018,9058)`).
    pub frequency_encoding_steps: i32,
    /// `(0018,1315)` VariableFlipAngleFlag.
    pub is_variable_flip_angle: bool,
    /// `(0018,9078)` ParallelAcquisitionTechnique (e.g. SENSE, GRAPPA).
    pub parallel_acquisition_technique: String,
    /// Media Storage SOP is Raw Data / Philips XX_ style non-image.
    pub is_raw_data_storage: bool,
    /// Grayscale Softcopy Presentation State (Philips PS_).
    pub is_grayscale_softcopy_presentation_state: bool,
    /// `(0010,2210)` AnatomicalOrientationType contains QUADRUPED (issue 642).
    pub is_quadruped: bool,
    /// `(0018,1210)` ConvolutionKernel.
    pub convolution_kernel: String,
    /// GE `(0009,108F)` / parsed kernel size; `NaN` when absent.
    pub recon_filter_size: f64,
    /// `(0028,0120)`; NaN when absent.
    pub pixel_padding_value: f64,
    pub is_xray: bool,
    /// `(0018,1150)` Exposure Time (ms).
    pub exposure_time_ms: f64,
    /// `(0018,1151)` X-Ray Tube Current (mA).
    pub x_ray_tube_current: f64,
    pub is_xa_physio: bool,
    pub is_cmrr_physio: bool,
    pub physio_offset: i64,
    pub physio_bytes: i32,
    /// Philips `(0020,9153)` TriggerDelayTime (ms). Ignored when `--ignore_trigger_times`.
    pub trigger_delay_time: f64,
    /// ASL classification bitflags (C++ `aslFlags`); 0 = none.
    pub asl_flags: u32,
    /// Public DICOM `(0018,9258)` ASLPulseTrainDuration / post-label delay (ms).
    pub post_label_delay: i32,
    /// `(0018,9255)` LabelingOrientation (ASL).
    pub labeling_orientation: String,
    /// `(0018,9259)` VascularCrushing CS; `-1` unknown, `0` false, `1` true.
    pub vascular_crushing: i32,
    /// `(0018,925A)` VascularCrushingVENC (cm/s); `0` = absent.
    pub vascular_crushing_venc: f64,
    /// GE `(0043,10A5)` DurationLabelPulse (ms); `-1` = unknown.
    pub duration_label_pulse_ge: i32,
    /// GE ASL spiral arms / points / excitations.
    pub number_of_excitations: f64,
    pub number_of_arms: f64,
    pub number_of_points_per_arm: f64,
    /// Group delay (ms); GE `(0043,107C)` and/or Protocol Block.
    pub group_delay: f64,
    /// GE Protocol Block slice order (`0` = interleaved hint when known); `-1` unknown.
    pub ge_slice_order: i32,
    /// GE Protocol Block `IOPT` string (FMRI / MPh / DIFF).
    pub ge_iopt: String,
    /// GE EPI class: `-1` unknown, `0` epi, `1` epiRT, `2` epi2, `≥3` pepolar.
    pub epi_version_ge: i32,
    /// GE internal EPI: `-1` unknown, `1` EPI, `2` EPI2.
    pub internal_epi_version_ge: i32,
    /// GE `(0019,10B3)` userData12 — pepolar direction mode.
    pub ge_user_data_12: i32,
    /// DICOM `(0020,0100)` Temporal Position Identifier (volume index).
    pub temporal_position: i32,
    /// Philips `(2001,1022)` WaterFatShift; `0` = absent.
    pub water_fat_shift: f64,
    /// `(0018,9036)` PartialFourierDirection: `0` unknown, `1` PHASE, `2` FREQUENCY, `3` SLICE_SELECT, `4` COMBINATION.
    pub partial_fourier_direction: i32,
    /// Partial Fourier used (`(0018,9081)`, ScanOptions `PFF`, Philips `(2001,1019)`).
    pub is_partial_fourier: bool,
    /// GE `(0019,10E2)` VelocityEncodeScale (fieldmapHz delta-TE); default `1.0`.
    pub velocity_encode_scale_ge: f64,
    /// GE `(0019,10A9)` MaxEchoNumGE; `-1` = absent (issue 359).
    pub max_echo_num_ge: i32,
    /// Philips Real World Value slope `(0040,9225)`; `0` = absent.
    pub rwv_scale: f64,
    /// Philips Real World Value intercept `(0040,9224)`.
    pub rwv_intercept: f64,
    /// `(0018,9020)` / CSA `ucMTC`: `-1` unknown, `0` false, `1` true.
    pub mt_state: i32,
    /// `(0018,9016)` spoiling: `-1` unknown, `0` none, `1` RF, `2` gradient, `3` both.
    pub spoiling: i32,
    /// Through-plane ZIP / interpolation factor (`>1` when set); `-1` unknown.
    pub interp_3d: i32,
    /// Philips `(2001,1008)` phase number; `-1` unknown.
    pub phase_number: i32,
    /// DICOM `(0008,9209)` AcquisitionContrast → `MRWeighting*` codes; `0` = unknown.
    pub acquisition_contrast: i32,
    /// True when diffusion gradients / b-values were parsed.
    pub is_diffusion: bool,
    /// Multi-echo series (echo train / CSA contrasts > 1).
    pub is_multi_echo: bool,
    /// GE/Philips direct fieldmap (Hz) intent.
    pub is_real_is_phase_map_hz: bool,
    /// Philips `(2005,1063)` fMRIStatusIndication / raw data run.
    pub raw_data_run_number: i32,
    /// DICOM Overlay Data present (even groups `0x6000..0x601E`).
    pub is_has_overlay: bool,
    /// Raw OverlayData bitstreams for up to 16 overlays (`None` = absent).
    pub overlays: [Option<Vec<u8>>; 16],
    /// GE `(0021,105E)` RTIA timer (seconds); used by undocumented `-j y`.
    pub rtia_timer_ge: f64,
    /// DICOM `(0028,0006)` Planar Configuration (1 = planar RGB).
    pub is_planar_rgb: bool,
    /// GE diffusion gradient cycling mode (`kGE_DIFF_CYCLING_*`); `-1` = unknown.
    pub diff_cycling_mode_ge: i32,
    /// True when `--diffCyclingModeGE` overrode detection (BIDS emits `OVERRIDE`).
    pub diff_cycling_mode_ge_override: bool,
    /// GE `(0019,10E0)` NumberOfDiffusionDirectionGE; `-1` = absent.
    pub number_of_diffusion_direction_ge: i32,
    /// GE `(0019,10DF)` NumberOfDiffusionT2GE; `-1` = absent.
    pub number_of_diffusion_t2_ge: i32,
    /// GE tensor file number (`UserData11` / inferred for 2TR/3TR); `0` = absent.
    pub tensor_file_ge: i32,
    /// GE `(0043,10B7)` CompressedSensingFactor; `0` = absent.
    pub compressed_sensing_factor: f64,
    /// `(0018,1242)` / enhanced frame duration (ms); `-1` unknown.
    pub frame_duration: f64,
    /// `(0054,1300)` Frame Reference Time (ms); `-1` unknown.
    pub frame_reference_time: f64,
    /// `(0018,9731)` / enhanced Decay Factor; `-1` unknown.
    pub decay_factor: f64,
    /// `(0012,0063)` DeidentificationMethod (multi-value joined with `_` or raw).
    pub deidentification_method: String,
    /// `(0012,0064)` DeidentificationMethodCodeSequence (≤10 items).
    pub deidentification_method_code_sequence: Vec<DeIdCodeSequence>,
    /// ECAT main-header isotope half-life (s); `0` = absent.
    pub ecat_isotope_halflife: f64,
    /// ECAT main-header dosage (Bq/cc); `0` = absent.
    pub ecat_dosage: f64,
    /// Per-volume onset times (sec) for BIDS `FrameTimesStart`.
    pub volume_onset_times: Vec<f64>,
    /// Per-volume frame durations (ms) for BIDS `FrameDuration` array.
    pub frame_durations: Vec<f64>,
    /// Per-volume frame reference times (ms).
    pub frame_reference_times: Vec<f64>,
    /// Per-volume decay correction factors.
    pub decay_factors: Vec<f64>,
    /// PET `(0018,0031)` Radiopharmaceutical / TracerName.
    pub radiopharmaceutical: String,
    /// PET TracerRadionuclide (CodeMeaning).
    pub tracer_radionuclide: String,
    /// `(0018,1074)` Bq.
    pub radionuclide_total_dose: f64,
    /// `(0018,1075)` seconds.
    pub radionuclide_half_life: f64,
    /// `(0018,1076)`.
    pub radionuclide_positron_fraction: f64,
    /// `(0018,1077)` Bq/umol → MolarActivity.
    pub radiopharmaceutical_specific_activity: f64,
    /// `(0018,1071)`.
    pub injected_volume: f64,
    /// `(0054,1323)` ScatterFraction (0..1); `0` = absent.
    pub scatter_fraction: f64,
    /// `(0018,1072)` TM string.
    pub radiopharmaceutical_start_time: String,
    /// `(0054,1102)` DecayCorrection.
    pub decay_correction: String,
    /// `(0054,1101)`.
    pub attenuation_correction_method: String,
    /// `(0054,1100)` Randoms Correction Method.
    pub randoms_correction_method: String,
    /// `(0054,1105)` Scatter Correction Method.
    pub scatter_correction_method: String,
    /// `(0054,1103)` Reconstruction Method.
    pub reconstruction_method: String,
    /// `(0054,1001)` Units for PET.
    pub units_pt: String,
    /// Dose calibration factor.
    pub dose_calibration_factor: f64,
}

impl DicomImage {
    pub fn has_orientation(&self) -> bool {
        self.orient[1..7].iter().any(|v| *v != 0.0)
    }

    /// BIDS / NIfTI-MRS `VOI` 4×4 (LPS→RAS). Port of C++ `mrsVoiMatrix`.
    /// Requires `voi_thickness > 0` and `has_voi_center` (audit M8).
    ///
    /// Uses [`Self::voi_orient`] (full f64) when set, else snapped [`Self::orient`].
    pub fn mrs_voi_matrix(&self) -> Option<[[f64; 4]; 4]> {
        if !(self.voi_thickness > 0.0 && self.has_voi_center) {
            return None;
        }
        let o = if self.voi_orient[1..7].iter().any(|v| *v != 0.0) {
            &self.voi_orient
        } else {
            &self.orient
        };
        let mut r1x = o[1];
        let mut r1y = o[2];
        let mut r1z = o[3];
        let mut r2x = o[4];
        let mut r2y = o[5];
        let mut r2z = o[6];
        let n1 = (r1x * r1x + r1y * r1y + r1z * r1z).sqrt();
        let n2 = (r2x * r2x + r2y * r2y + r2z * r2z).sqrt();
        if n1 > 0.001 {
            r1x /= n1;
            r1y /= n1;
            r1z /= n1;
        }
        if n2 > 0.001 {
            r2x /= n2;
            r2y /= n2;
            r2z /= n2;
        }
        let sn_x = r1y * r2z - r1z * r2y;
        let sn_y = r1z * r2x - r1x * r2z;
        let sn_z = r1x * r2y - r1y * r2x;
        let phase_fov = if self.voi_phase_fov > 0.0 {
            self.voi_phase_fov
        } else {
            self.voi_thickness
        };
        let read_fov = if self.voi_readout_fov > 0.0 {
            self.voi_readout_fov
        } else {
            self.voi_thickness
        };
        let t = self.voi_thickness;
        let cx = self.voi_center_lps[0];
        let cy = self.voi_center_lps[1];
        let cz = self.voi_center_lps[2];
        Some([
            [-r1x * read_fov, -r2x * phase_fov, -sn_x * t, -cx],
            [-r1y * read_fov, -r2y * phase_fov, -sn_y * t, -cy],
            [r1z * read_fov, r2z * phase_fov, sn_z * t, cz],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

/// `(0018,9126)` VolumeLocalizationSequence → VOI box (Enhanced MRS).
///
/// Per-item: slab[0]=thickness, slab[1]=phase FoV, slab[2]=readout FoV;
/// first MidSlabPosition is the VOI center; first two SlabOrientation vectors
/// fill `orient` / `voi_orient` when public IOP is still empty.
fn read_volume_localization(
    obj: &DefaultDicomObject,
    voi_phase_fov: &mut f64,
    voi_readout_fov: &mut f64,
    voi_thickness: &mut f64,
    voi_center_lps: &mut [f64; 3],
    has_voi_center: &mut bool,
    orient: &mut [f64; 7],
    voi_orient: &mut [f64; 7],
) {
    let Ok(elem) = obj.element(tags::VOLUME_LOCALIZATION_SEQUENCE) else {
        return;
    };
    let Some(items) = elem.items() else {
        return;
    };
    let mut slab_orient: [f64; 7] = [0.0; 7];
    let mut slab_orient_count = 0i32;
    for (slab_idx, item) in items.iter().enumerate() {
        if let Some(v) = item
            .element(tags::SLAB_THICKNESS)
            .ok()
            .and_then(|e| e.to_float64().ok())
        {
            if v > 0.0 {
                match slab_idx {
                    0 if *voi_thickness == 0.0 => *voi_thickness = v,
                    1 if *voi_phase_fov == 0.0 => *voi_phase_fov = v,
                    2 if *voi_readout_fov == 0.0 => *voi_readout_fov = v,
                    _ => {}
                }
            }
        }
        if !*has_voi_center {
            if let Ok(e) = item.element(tags::MID_SLAB_POSITION) {
                if let Ok(vals) = e.to_multi_float64() {
                    if vals.len() >= 3 {
                        *voi_center_lps = [vals[0], vals[1], vals[2]];
                        *has_voi_center = true;
                    }
                }
            }
        }
        if slab_orient_count < 2 {
            if let Ok(e) = item.element(tags::SLAB_ORIENTATION) {
                if let Ok(vals) = e.to_multi_float64() {
                    if vals.len() >= 3 {
                        let base = (slab_orient_count * 3) as usize;
                        slab_orient[base + 1] = vals[0];
                        slab_orient[base + 2] = vals[1];
                        slab_orient[base + 3] = vals[2];
                        slab_orient_count += 1;
                    }
                }
            }
        }
    }
    if slab_orient_count >= 2 {
        // Full precision for VOI; snapped copy for sform when IOP empty.
        if voi_orient[1] == 0.0 && voi_orient[2] == 0.0 && voi_orient[3] == 0.0 {
            voi_orient[1..7].copy_from_slice(&slab_orient[1..7]);
        }
        if orient[1] == 0.0 && orient[2] == 0.0 && orient[3] == 0.0 {
            for i in 1..7 {
                orient[i] = snap_f32(slab_orient[i]);
            }
        }
    }
}

fn err(path: &Path, e: impl std::fmt::Display) -> Error {
    Error::bad_file(format!("{}: {e}", path.display()))
}

fn item_text(item: &InMemDicomObject, tag: Tag) -> String {
    item.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Parse `(0012,0064)` DeidentificationMethodCodeSequence (issue 877; max 10).
fn read_deid_code_sequence(obj: &DefaultDicomObject) -> Vec<DeIdCodeSequence> {
    const MAX_DEID_CS: usize = 10;
    let Ok(elem) = obj.element(Tag(0x0012, 0x0064)) else {
        return Vec::new();
    };
    let Some(items) = elem.items() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items.iter().take(MAX_DEID_CS) {
        let code_value = item_text(item, tags::CODE_VALUE);
        let coding_scheme_designator = item_text(item, tags::CODING_SCHEME_DESIGNATOR);
        let coding_scheme_version = item_text(item, tags::CODING_SCHEME_VERSION);
        let code_meaning = item_text(item, tags::CODE_MEANING);
        // C++ increments on CodeMeaning; keep entries that have any field.
        if code_value.is_empty()
            && coding_scheme_designator.is_empty()
            && coding_scheme_version.is_empty()
            && code_meaning.is_empty()
        {
            continue;
        }
        out.push(DeIdCodeSequence {
            code_value,
            coding_scheme_designator,
            coding_scheme_version,
            code_meaning,
        });
    }
    out
}

fn text(obj: &DefaultDicomObject, tag: Tag) -> String {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn int(obj: &DefaultDicomObject, tag: Tag) -> Option<i32> {
    obj.element(tag).ok().and_then(|e| e.to_int::<i32>().ok())
}

fn f64s(obj: &DefaultDicomObject, tag: Tag) -> Option<Vec<f64>> {
    obj.element(tag).ok()?.to_multi_float64().ok()
}

fn first_f64(obj: &DefaultDicomObject, tag: Tag) -> Option<f64> {
    obj.element(tag).ok()?.to_float64().ok()
}

/// Split `(0008,002A)` `YYYYMMDDHHMMSS.frac` like dcm2niix end-of-parse fallback.
fn split_acquisition_datetime(dt: &str) -> (String, String) {
    let dt = dt.trim();
    if dt.len() >= 14 {
        (
            dt[..8].to_string(),
            dt[8..].trim_end_matches('\0').trim().to_string(),
        )
    } else {
        (String::new(), String::new())
    }
}

fn is_zero_float_str(s: &str) -> bool {
    s.trim().is_empty() || s.trim().parse::<f64>().unwrap_or(0.0).abs() < f64::EPSILON
}

/// UIH mosaics repeat `(0008,0032)` per tile; dcm2niix keeps the last value.
fn scan_raw_explicit_values(path: &Path, tag: Tag) -> Vec<String> {
    let Ok(data) = std::fs::read(path) else {
        return Vec::new();
    };
    let pattern = [
        tag.group() as u8,
        (tag.group() >> 8) as u8,
        tag.element() as u8,
        (tag.element() >> 8) as u8,
    ];
    let pixel = [0xE0u8, 0x7F, 0x10u8, 0x00];
    let limit = data
        .windows(4)
        .position(|w| w == pixel)
        .unwrap_or(data.len());
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 8 <= limit {
        if data[i..i + 4] != pattern {
            i += 1;
            continue;
        }
        let vr = &data[i + 4..i + 6];
        if !vr[0].is_ascii_alphabetic() || !vr[1].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let (len, voff) = if matches!(vr, b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN") {
            if i + 12 > limit {
                i += 1;
                continue;
            }
            (
                u32::from_le_bytes([data[i + 8], data[i + 9], data[i + 10], data[i + 11]])
                    as usize,
                i + 12,
            )
        } else {
            (
                u16::from_le_bytes([data[i + 6], data[i + 7]]) as usize,
                i + 8,
            )
        };
        if len > 64 || voff + len > limit {
            i += 1;
            continue;
        }
        if let Ok(s) = std::str::from_utf8(&data[voff..voff + len]) {
            let s = s.trim_matches('\0').trim();
            if tag == tags::ACQUISITION_TIME && is_valid_dicom_time(s) {
                out.push(s.to_string());
            } else if tag == tags::ACQUISITION_DATE && is_valid_dicom_date(s) {
                out.push(s.to_string());
            } else if tag == tags::ACQUISITION_DATE_TIME && is_valid_dicom_datetime(s) {
                out.push(s.to_string());
            }
        }
        i += 1;
    }
    out
}

fn is_valid_dicom_date(s: &str) -> bool {
    s.len() == 8 && s.bytes().all(|b| b.is_ascii_digit())
}

fn is_valid_dicom_time(s: &str) -> bool {
    s.len() >= 6
        && s.chars().take(6).all(|c| c.is_ascii_digit())
        && s.chars().skip(6).all(|c| c.is_ascii_digit() || c == '.')
}

fn is_valid_dicom_datetime(s: &str) -> bool {
    s.len() >= 14 && is_valid_dicom_date(&s[..8]) && is_valid_dicom_time(&s[8..])
}

/// Prefer last `(0008,0032)` when the file repeats the tag (UIH tiles / enhanced
/// frames). Matches dcm2niix last-wins overwrite during a sequential parse.
fn acquisition_date_time(path: &Path, obj: &DefaultDicomObject) -> (String, String) {
    let mut acquisition_date = text(obj, tags::ACQUISITION_DATE);
    let mut acquisition_time = text(obj, tags::ACQUISITION_TIME);
    let scanned_t = scan_raw_explicit_values(path, tags::ACQUISITION_TIME);
    if let Some(t) = scanned_t.last() {
        // Prefer last valid TM when multiple are present (C++ last-wins).
        if scanned_t.len() > 1 || is_zero_float_str(&acquisition_time) {
            acquisition_time = t.clone();
        }
    } else if is_zero_float_str(&acquisition_time) {
        // keep empty
    }
    if is_zero_float_str(&acquisition_date) {
        if let Some(d) = scan_raw_explicit_values(path, tags::ACQUISITION_DATE).pop() {
            acquisition_date = d;
        }
    }
    if is_zero_float_str(&acquisition_time) && is_zero_float_str(&acquisition_date) {
        if let Some(dt) = obj
            .element(tags::ACQUISITION_DATE_TIME)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.as_ref().to_string())
            .or_else(|| scan_raw_explicit_values(path, tags::ACQUISITION_DATE_TIME).pop())
        {
            if dt.len() > 13 {
                let (d, t) = split_acquisition_datetime(&dt);
                if is_zero_float_str(&acquisition_date) {
                    acquisition_date = d;
                }
                if is_zero_float_str(&acquisition_time) {
                    acquisition_time = t;
                }
            }
        }
    }
    (acquisition_date, acquisition_time)
}

/// UIH `(0065,100D)` e.g. `F:2S` → `2` (C++ `dcmStrDigitsDotOnlyKey(':', …)`).
fn parse_uih_accel(s: &str) -> f64 {
    let mut after_key = false;
    let mut digits = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            if after_key {
                digits.push(ch);
            }
        } else if ch == ':' {
            after_key = true;
            digits.clear();
        } else if !after_key {
            digits.clear();
        }
    }
    digits.parse().unwrap_or(0.0)
}

fn crc32(s: &str) -> u32 {
    let mut h = Hasher::new();
    h.update(s.as_bytes());
    h.finalize()
}

fn open_opts() -> OpenFileOptions {
    OpenFileOptions::new()
}

/// Full dataset parse (includes Pixel Data) for decode and enhanced MF.
pub fn open(path: &Path) -> Result<DefaultDicomObject> {
    if let Some(mmap) = try_mmap_path(path) {
        return open_opts()
            .from_reader(std::io::Cursor::new(mmap.as_ref()))
            .map_err(|e| err(path, e));
    }
    let meta = std::fs::metadata(path).map_err(|e| Error::io(path, e))?;
    if meta.len() > PREFETCH_MAX {
        return open_file(path).map_err(|e| err(path, e));
    }
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    open_opts()
        .from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| err(path, e))
}

/// Series-/convert-scoped mmap cache (path → mapped file bytes).
pub type MmapCache = HashMap<PathBuf, Arc<memmap2::Mmap>>;

const PREFETCH_MAX: u64 = 64 * 1024 * 1024;

fn try_mmap_path(path: &Path) -> Option<Arc<memmap2::Mmap>> {
    let file = std::fs::File::open(path).ok()?;
    let meta = file.metadata().ok()?;
    if meta.len() > PREFETCH_MAX {
        return None;
    }
    unsafe { memmap2::Mmap::map(&file).ok() }.map(Arc::new)
}

/// Header-only open; returns mmap bytes to cache when the file was mapped (≤64 MiB).
fn open_header_cached(path: &Path) -> Result<(DefaultDicomObject, Option<Arc<memmap2::Mmap>>)> {
    let opts = open_opts().read_until(tags::PIXEL_DATA);
    if let Some(mmap) = try_mmap_path(path) {
        let obj = opts
            .from_reader(std::io::Cursor::new(mmap.as_ref()))
            .map_err(|e| err(path, e))?;
        return Ok((obj, Some(mmap)));
    }
    let obj = opts.open_file(path).map_err(|e| err(path, e))?;
    Ok((obj, None))
}

/// Metadata-only parse: stop before `(7FE0,0010)` Pixel Data.
fn open_header_only(path: &Path) -> Result<DefaultDicomObject> {
    open_header_prefetched(path, &MmapCache::new())
}

fn open_header_prefetched(path: &Path, cache: &MmapCache) -> Result<DefaultDicomObject> {
    let opts = open_opts().read_until(tags::PIXEL_DATA);
    if let Some(mmap) = cache.get(path) {
        return opts
            .from_reader(std::io::Cursor::new(mmap.as_ref()))
            .map_err(|e| err(path, e));
    }
    open_header_cached(path).map(|(obj, _)| obj)
}

/// Warm mmap for slice paths (≤64 MiB each) before parallel decode / MRS load.
pub fn prefetch_mmaps(paths: &[&Path]) -> MmapCache {
    paths
        .par_iter()
        .filter_map(|&path| try_mmap_path(path).map(|m| (path.to_path_buf(), m)))
        .collect()
}

/// One parallel pass: mmap eligible files and parse headers (convert entry path).
pub fn warmup_convert_cache(paths: &[PathBuf]) -> (MmapCache, Vec<(PathBuf, Result<DicomImage>)>) {
    let rows: Vec<_> = paths
        .par_iter()
        .enumerate()
        .map(|(i, path)| {
            let parsed = open_header_cached(path).and_then(|(obj, mmap)| {
                parse_header(path, &obj).map(|img| (mmap, img))
            });
            (i, parsed)
        })
        .collect();
    let mut cache = MmapCache::new();
    let mut out = Vec::with_capacity(rows.len());
    for (i, parsed) in rows {
        let path = paths[i].clone();
        match parsed {
            Ok((mmap, img)) => {
                if let Some(m) = mmap {
                    cache.insert(path.clone(), m);
                }
                out.push((path, Ok(img)));
            }
            Err(e) => out.push((path, Err(e))),
        }
    }
    (cache, out)
}

/// Full DICOM parse reusing a convert-scoped mmap cache when present.
pub fn open_prefetched(path: &Path, cache: &MmapCache) -> Result<DefaultDicomObject> {
    if let Some(mmap) = cache.get(path) {
        open_opts()
            .from_reader(std::io::Cursor::new(mmap.as_ref()))
            .map_err(|e| err(path, e))
    } else {
        open(path)
    }
}

pub fn read_header(path: impl AsRef<Path>) -> Result<DicomImage> {
    let path = path.as_ref();
    let obj = open_header_only(path)?;
    parse_header(path, &obj)
}

/// Like [`read_header`] but reuses mmap entries from [`prefetch_mmaps`].
pub fn read_header_prefetched(path: impl AsRef<Path>, cache: &MmapCache) -> Result<DicomImage> {
    let path = path.as_ref();
    let obj = open_header_prefetched(path, cache)?;
    parse_header(path, &obj)
}

pub fn parse_header(path: &Path, obj: &DefaultDicomObject) -> Result<DicomImage> {
    let series_uid = text(obj, tags::SERIES_INSTANCE_UID);
    let manufacturer_name = text(obj, tags::MANUFACTURER);
    let image_type = text(obj, tags::IMAGE_TYPE);
    let series_description = text(obj, tags::SERIES_DESCRIPTION);
    let is_derived = image_type.to_ascii_uppercase().contains("DERIVED");
    let is_localizer = series_description.to_ascii_uppercase().contains("LOCALIZER")
        || image_type.to_ascii_uppercase().contains("LOCALIZER");
    let is_mosaic = image_type.to_ascii_uppercase().contains("MOSAIC");
    let img_type_u = image_type.to_ascii_uppercase();
    let is_has_phase = img_type_u.contains("PHASE");
    let is_has_real = img_type_u.contains("REAL");
    let is_has_imaginary = img_type_u.contains("IMAGINARY");
    let is_has_magnitude = img_type_u.contains("MAGNITUDE");

    let mut csa = csa::read_csa(obj);

    let mut b_value = -1.0f64;
    let mut diffusion_direction = [-1.0f64, 2.0f64, 2.0f64];
    let pe_direction_displayed = text(obj, Tag(0x0065, 0x1005));
    let mut accel_fact_pe = 0.0f64;
    if let Some(n) = first_f64(obj, Tag(0x0065, 0x1050)) {
        let ms = n.round() as i32;
        if ms > 1 {
            csa.image.mosaic_slices = ms;
        }
    }
    if let Some(b) = first_f64(obj, Tag(0x0065, 0x1009)) {
        b_value = b;
    }
    if let Some(g) = f64s(obj, Tag(0x0065, 0x1037)) {
        if g.len() >= 3 {
            diffusion_direction = [g[0], g[1], g[2]];
        }
    }
    if let Some(sn) = f64s(obj, Tag(0x0065, 0x1014)) {
        if sn.len() >= 3 {
            csa.image.slice_norm[1] = sn[0];
            csa.image.slice_norm[2] = sn[1];
            csa.image.slice_norm[3] = sn[2];
        }
    }
    if let Some(s) = obj
        .element(Tag(0x0065, 0x100D))
        .ok()
        .and_then(|e| e.to_str().ok())
    {
        accel_fact_pe = parse_uih_accel(s.trim());
    }

    let mut orient = [0.0; 7];
    let mut voi_orient = [0.0; 7];
    if let Some(iop) = f64s(obj, tags::IMAGE_ORIENTATION_PATIENT) {
        if iop.len() >= 6 {
            orient[1] = snap_f32(iop[0]);
            orient[2] = snap_f32(iop[1]);
            orient[3] = snap_f32(iop[2]);
            orient[4] = snap_f32(iop[3]);
            orient[5] = snap_f32(iop[4]);
            orient[6] = snap_f32(iop[5]);
            // Full DS precision for MRS VOI (C++ stores float; we keep f64).
            voi_orient[1] = iop[0];
            voi_orient[2] = iop[1];
            voi_orient[3] = iop[2];
            voi_orient[4] = iop[3];
            voi_orient[5] = iop[4];
            voi_orient[6] = iop[5];
        }
    }
    // Classic Siemens MRS: CSA ImageOrientationPatient when public IOP empty.
    if orient[1] == 0.0 && orient[2] == 0.0 && orient[3] == 0.0 {
        if let Some(iop) = csa.image.image_orientation {
            orient[1] = snap_f32(iop[0]);
            orient[2] = snap_f32(iop[1]);
            orient[3] = snap_f32(iop[2]);
            orient[4] = snap_f32(iop[3]);
            orient[5] = snap_f32(iop[4]);
            orient[6] = snap_f32(iop[5]);
            voi_orient[1] = iop[0];
            voi_orient[2] = iop[1];
            voi_orient[3] = iop[2];
            voi_orient[4] = iop[3];
            voi_orient[5] = iop[4];
            voi_orient[6] = iop[5];
        }
    } else if voi_orient[1] == 0.0 {
        // Public IOP present but voi_orient not set (shouldn't happen); copy CSA.
        if let Some(iop) = csa.image.image_orientation {
            voi_orient[1] = iop[0];
            voi_orient[2] = iop[1];
            voi_orient[3] = iop[2];
            voi_orient[4] = iop[3];
            voi_orient[5] = iop[4];
            voi_orient[6] = iop[5];
        }
    }
    let mut voi_phase_fov = csa.image.voi_phase_fov;
    let mut voi_readout_fov = csa.image.voi_readout_fov;
    let mut voi_thickness = csa.image.voi_thickness;
    let mut voi_center_lps = csa.image.voi_center_lps;
    let mut has_voi_center = csa.image.has_voi_center;
    read_volume_localization(
        obj,
        &mut voi_phase_fov,
        &mut voi_readout_fov,
        &mut voi_thickness,
        &mut voi_center_lps,
        &mut has_voi_center,
        &mut orient,
        &mut voi_orient,
    );

    let mut patient_position = [f64::NAN; 4];
    let mut patient_position_last = [f64::NAN; 4];
    let mut last_scan_loc = f64::NAN;
    let mut acquisition_duration = 0.0f64;
    if let Some(ipp) = f64s(obj, tags::IMAGE_POSITION_PATIENT) {
        if ipp.len() >= 3 {
            patient_position[1] = snap_f32(ipp[0]);
            patient_position[2] = snap_f32(ipp[1]);
            patient_position[3] = snap_f32(ipp[2]);
        }
    }
    if has_voi_center && patient_position[1].is_nan() {
        patient_position[1] = voi_center_lps[0];
        patient_position[2] = voi_center_lps[1];
        patient_position[3] = voi_center_lps[2];
    }
    if manufacturer_name.to_ascii_uppercase().contains("UIH")
        || manufacturer_name.to_ascii_uppercase().contains("UNITED IMAGING")
    {
        let nested = uih_scan::scan_uih_nested(path);
        if patient_position[1].is_nan() {
            if let Some(first) = nested.ipps.first() {
                patient_position[1] = snap_f32(first[0]);
                patient_position[2] = snap_f32(first[1]);
                patient_position[3] = snap_f32(first[2]);
            }
        }
        if let Some(last) = nested.ipps.last() {
            patient_position_last[1] = snap_f32(last[0]);
            patient_position_last[2] = snap_f32(last[1]);
            patient_position_last[3] = snap_f32(last[2]);
        }
        if nested.acquisition_duration > 0.0 {
            acquisition_duration = nested.acquisition_duration;
        }
        if !nested.acq_times.is_empty() {
            csa.image.slice_timing_ms =
                uih_scan::process_uih_slice_timing_ms(&nested.acq_times);
        }
        if !nested.last_scan_loc.is_nan() {
            last_scan_loc = nested.last_scan_loc;
        }
    }
    // Public DICOM (0018,9073) AcquisitionDuration (seconds).
    if let Some(ad) = first_f64(obj, Tag(0x0018, 0x9073)) {
        if ad > 0.0 {
            acquisition_duration = ad;
        }
    }

    let mut xyz_mm = [1.0; 4];
    if let Some(ps) = f64s(obj, tags::PIXEL_SPACING) {
        if ps.len() >= 2 {
            xyz_mm[2] = snap_f32(ps[0]); // row spacing → Y
            xyz_mm[1] = snap_f32(ps[1]); // column spacing → X
        }
    }
    // Classic Siemens MRS: CSA VoiPhase/Readout FoV → voxel size when PixelSpacing
    // is still the 1.0 sentinel (same gates as C++ readCSAforMRS).
    if voi_phase_fov > 0.0 && xyz_mm[1] <= 1.0 {
        xyz_mm[1] = voi_phase_fov;
    }
    if voi_readout_fov > 0.0 && xyz_mm[2] <= 1.0 {
        xyz_mm[2] = voi_readout_fov;
    }
    let mut slice_thickness = first_f64(obj, tags::SLICE_THICKNESS).unwrap_or(0.0);
    if voi_thickness > 0.0 && slice_thickness == 0.0 {
        slice_thickness = voi_thickness;
    }
    if let Some(sp) = first_f64(obj, tags::SPACING_BETWEEN_SLICES) {
        xyz_mm[3] = snap_f32(sp.abs());
    } else if slice_thickness > 0.0 {
        xyz_mm[3] = snap_f32(slice_thickness);
    }

    let bits_allocated = int(obj, tags::BITS_ALLOCATED).unwrap_or(16);
    let pixel_repr = int(obj, tags::PIXEL_REPRESENTATION).unwrap_or(0);
    let photometric = text(obj, tags::PHOTOMETRIC_INTERPRETATION);
    let is_float = photometric.contains("FLOAT")
        || bits_allocated == 32 && pixel_repr == 0 && int(obj, Tag(0x0028, 0x0103)).is_none();

    let phase = text(obj, tags::IN_PLANE_PHASE_ENCODING_DIRECTION);
    let phase_encoding_rc = match phase.as_str() {
        "ROW" => 'R',
        "COL" | "COLUMN" => 'C',
        _ => ' ',
    };

    let (acquisition_date, acquisition_time) = acquisition_date_time(path, obj);

    Ok({
        let mut d = DicomImage {
        path: path.to_path_buf(),
        series_uid_crc: crc32(&series_uid),
        series_uid,
        instance_uid: text(obj, tags::SOP_INSTANCE_UID),
        study_uid: text(obj, tags::STUDY_INSTANCE_UID),
        series_number: int(obj, tags::SERIES_NUMBER).unwrap_or(0) as i64,
        instance_number: int(obj, tags::INSTANCE_NUMBER).unwrap_or(0),
        acquisition_number: int(obj, tags::ACQUISITION_NUMBER).unwrap_or(0),
        echo_number: int(obj, tags::ECHO_NUMBERS).unwrap_or(1),
        rows: int(obj, tags::ROWS).unwrap_or(0).max(0) as usize,
        columns: int(obj, tags::COLUMNS).unwrap_or(0).max(0) as usize,
        bits_allocated,
        bits_stored: int(obj, tags::BITS_STORED).unwrap_or(bits_allocated),
        samples_per_pixel: int(obj, tags::SAMPLES_PER_PIXEL).unwrap_or(1),
        is_signed: pixel_repr == 1,
        is_float,
        xyz_mm,
        slice_thickness,
        orient,
        patient_position,
        patient_position_last,
        last_scan_loc,
        acquisition_duration,
        manufacturer: Manufacturer::from_tag(&manufacturer_name),
        modality: Modality::from_tag(&text(obj, tags::MODALITY)),
        manufacturer_name,
        manufacturers_model_name: text(obj, tags::MANUFACTURER_MODEL_NAME),
        institution_name: text(obj, tags::INSTITUTION_NAME),
        institution_address: text(obj, tags::INSTITUTION_ADDRESS),
        institutional_department: text(obj, tags::INSTITUTIONAL_DEPARTMENT_NAME),
        procedure_step_description: {
            let s = text(obj, Tag(0x0040, 0x0254));
            if !s.is_empty() {
                s
            } else {
                text(obj, Tag(0x0040, 0x0255))
            }
        },
        station_name: text(obj, tags::STATION_NAME),
        device_serial_number: text(obj, tags::DEVICE_SERIAL_NUMBER),
        software_versions: text(obj, tags::SOFTWARE_VERSIONS),
        protocol_name: text(obj, tags::PROTOCOL_NAME),
        series_description,
        sequence_name: text(obj, tags::SEQUENCE_NAME),
        pulse_sequence_name: {
            let mut s = text(obj, Tag(0x0018, 0x9005));
            if s.is_empty() {
                s = text(obj, Tag(0x0019, 0x109C));
            }
            s
        },
        scanning_sequence: text(obj, tags::SCANNING_SEQUENCE),
        sequence_variant: text(obj, tags::SEQUENCE_VARIANT),
        scan_options: text(obj, tags::SCAN_OPTIONS),
        image_type,
        image_comments: text(obj, tags::IMAGE_COMMENTS),
        coil_name: text(obj, tags::RECEIVE_COIL_NAME),
        coil_string: text(obj, Tag(0x0051, 0x100F)),
        transmit_coil_name: {
            let mut s = text(obj, Tag(0x0018, 0x1251));
            if s.is_empty() {
                // Nested in MRTransmitCoilSequence (0018,9049).
                if let Ok(elem) = obj.element(Tag(0x0018, 0x9049)) {
                    if let Some(items) = elem.items() {
                        for item in items {
                            if let Ok(e) = item.element(Tag(0x0018, 0x1251)) {
                                if let Ok(v) = e.to_str() {
                                    let t = v.trim().to_string();
                                    if !t.is_empty() {
                                        s = t;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            s
        },
        patient_name: text(obj, tags::PATIENT_NAME),
        patient_id: text(obj, tags::PATIENT_ID),
        patient_sex: text(obj, tags::PATIENT_SEX),
        patient_age: text(obj, tags::PATIENT_AGE),
        referring_physician_name: text(obj, tags::REFERRING_PHYSICIAN_NAME),
        patient_birth_date: text(obj, tags::PATIENT_BIRTH_DATE),
        patient_weight: first_f64(obj, tags::PATIENT_WEIGHT).unwrap_or(0.0),
        patient_size: first_f64(obj, tags::PATIENT_SIZE).unwrap_or(0.0),
        accession_number: text(obj, tags::ACCESSION_NUMBER),
        study_id: text(obj, tags::STUDY_ID),
        study_description: text(obj, tags::STUDY_DESCRIPTION),
        study_date: text(obj, tags::STUDY_DATE),
        study_time: text(obj, tags::STUDY_TIME),
        series_time: text(obj, tags::SERIES_TIME),
        acquisition_date,
        acquisition_time,
        body_part: text(obj, tags::BODY_PART_EXAMINED),
        tr: first_f64(obj, tags::REPETITION_TIME).unwrap_or(0.0),
        te: first_f64(obj, tags::ECHO_TIME).unwrap_or(0.0),
        ti: first_f64(obj, tags::INVERSION_TIME).unwrap_or(0.0),
        flip_angle: first_f64(obj, tags::FLIP_ANGLE).unwrap_or(0.0),
        field_strength: first_f64(obj, tags::MAGNETIC_FIELD_STRENGTH).unwrap_or(0.0),
        pixel_bandwidth: first_f64(obj, tags::PIXEL_BANDWIDTH).unwrap_or(0.0),
        echo_train_length: int(obj, tags::ECHO_TRAIN_LENGTH).unwrap_or(0),
        phase_encoding_rc,
        inten_scale: first_f64(obj, tags::RESCALE_SLOPE).unwrap_or(1.0) as f32,
        inten_intercept: first_f64(obj, tags::RESCALE_INTERCEPT).unwrap_or(0.0) as f32,
        inten_scale_philips: first_f64(obj, Tag(0x2005, 0x100E))
            .or_else(|| first_f64(obj, Tag(0x2005, 0x140A)))
            .unwrap_or(0.0) as f32,
        is_scale_varies_enh: false,
        is_derived,
        is_localizer,
        number_of_frames: int(obj, tags::NUMBER_OF_FRAMES).unwrap_or(1),
        imaging_frequency: first_f64(obj, tags::IMAGING_FREQUENCY).unwrap_or(0.0),
        patient_position_label: text(obj, tags::PATIENT_POSITION),
        spacing_between_slices: snap_f32(
            first_f64(obj, tags::SPACING_BETWEEN_SLICES).unwrap_or(0.0),
        ),
        acquisition_matrix_pe: {
            let v = f64s(obj, tags::ACQUISITION_MATRIX).unwrap_or_default();
            // dcm2niix: prefer phase-encoding slots [3] then [2] of (0018,1310).
            if v.len() >= 4 && v[3] > 0.0 {
                v[3] as i32
            } else if v.len() >= 3 && v[2] > 0.0 {
                v[2] as i32
            } else {
                0
            }
        },
        phase_encoding_steps: int(obj, Tag(0x0018, 0x9231))
            .or_else(|| int(obj, Tag(0x0018, 0x0089)))
            .unwrap_or(0),
        phase_encoding_steps_out_of_plane: int(obj, Tag(0x0018, 0x9232)).unwrap_or(0),
        number_of_concatenations: 1,
        repetition_time_excitation: -1.0,
        repetition_time_inversion: 0.0,
        percent_phase_fov: first_f64(obj, tags::PERCENT_PHASE_FIELD_OF_VIEW).unwrap_or(0.0),
        percent_sampling: first_f64(obj, tags::PERCENT_SAMPLING).unwrap_or(0.0),
        mra_acquisition_type: text(obj, tags::MR_ACQUISITION_TYPE),
        b_value,
        diffusion_direction,
        pe_direction_displayed,
        number_of_averages: first_f64(obj, tags::NUMBER_OF_AVERAGES).unwrap_or(0.0),
        is_3d_acq: {
            let t = text(obj, tags::MR_ACQUISITION_TYPE);
            t.len() >= 2 && t.as_bytes()[0] == b'3' && t.as_bytes()[1].to_ascii_uppercase() == b'D'
        },
        is_epi: {
            let sq = text(obj, tags::SCANNING_SEQUENCE);
            sq.to_ascii_uppercase().contains("EP")
        },
        is_ir: {
            let sq = text(obj, tags::SCANNING_SEQUENCE).to_ascii_uppercase();
            let ir_flag = text(obj, Tag(0x0018, 0x9009)).to_ascii_uppercase();
            sq.contains("IR") || ir_flag.starts_with('Y')
        },
        accel_fact_pe: {
            if accel_fact_pe > 0.0 {
                accel_fact_pe
            } else {
                f64s(obj, Tag(0x0043, 0x1083))
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0.0)
            }
        },
        internal_pulse_sequence_name: text(obj, Tag(0x0019, 0x109E)),
        shim_setting: [
            int(obj, Tag(0x0043, 0x1002)).unwrap_or(0) as f64,
            int(obj, Tag(0x0043, 0x1003)).unwrap_or(0) as f64,
            int(obj, Tag(0x0043, 0x1004)).unwrap_or(0) as f64,
        ],
        prescan_reuse_string: text(obj, Tag(0x0043, 0x1095)),
        effective_echo_spacing_ge: first_f64(obj, Tag(0x0043, 0x102C)).unwrap_or(0.0),
        acquisition_duration_s: {
            let us = first_f64(obj, Tag(0x0019, 0x105A)).unwrap_or(0.0);
            if us > 1000.0 {
                us / 1_000_000.0
            } else {
                us
            }
        },
        phase_encoding_ge: parse_ge_phase_polarity(obj),
        parallel_reduction_out_of_plane: f64s(obj, Tag(0x0043, 0x1083))
            .and_then(|v| v.get(1).copied())
            .unwrap_or(0.0),
        sar: first_f64(obj, tags::SAR).unwrap_or(0.0),
        dwell_time_ns: first_f64(obj, Tag(0x0019, 0x1018))
            .or_else(|| first_f64(obj, Tag(0x0021, 0x1142)))
            .unwrap_or(0.0),
        csa,
        is_mosaic,
        image_orientation_text: text(obj, Tag(0x0051, 0x100E)),
        is_mrs: {
            let sop = text(obj, tags::SOP_CLASS_UID);
            let img_type = text(obj, Tag(0x0008, 0x0008)).to_ascii_uppercase();
            sop.contains("1.2.840.10008.5.1.4.1.1.4.2")
                || (text(obj, tags::MODALITY) == "MR" && img_type.contains("SPECTROSCOPY"))
        },
        is_mrs_ref: {
            // Water-reference series naming (C++ `isMrsRef` for BIDS WaterSuppressed).
            let sd = text(obj, tags::SERIES_DESCRIPTION).to_ascii_uppercase();
            let pn = text(obj, tags::PROTOCOL_NAME).to_ascii_uppercase();
            sd.contains("WRSOFF")
                || sd.contains("NO_WATER_SUPPRESSION")
                || sd.contains("WATERREF")
                || pn.contains("WRSOFF")
                || pn.contains("NO_WATER_SUPPRESSION")
        },
        data_point_columns: int(obj, Tag(0x0028, 0x9002)).unwrap_or(0),
        resonant_nucleus: text(obj, Tag(0x0018, 0x9100)),
        mrs_acq_type: {
            let t = text(obj, Tag(0x0018, 0x9200)).to_ascii_uppercase();
            if t.contains("VOLUME") {
                3
            } else if t.contains("PLANE") {
                2
            } else if t.contains("ROW") {
                1
            } else {
                0
            }
        },
        voi_phase_fov,
        voi_readout_fov,
        voi_thickness,
        voi_center_lps,
        has_voi_center,
        voi_orient,
        number_of_k_space_trajectories: int(obj, Tag(0x0018, 0x9093)).unwrap_or(0),
        spectral_width_hz: first_f64(obj, Tag(0x0018, 0x9052)).unwrap_or(0.0),
        is_xa: {
            let sw = text(obj, tags::SOFTWARE_VERSIONS).to_ascii_uppercase();
            let model = text(obj, tags::MANUFACTURER_MODEL_NAME).to_ascii_uppercase();
            sw.contains("XA") || model.contains("XA") || text(obj, Tag(0x0008, 0x1090)).contains("XA")
        },
        is_pmsct_rle1: {
            // Elscint/Philips PMSCT_RLE1 marker in (07a1,100a) or compression scheme string.
            text(obj, Tag(0x07a1, 0x100a))
                .to_ascii_uppercase()
                .contains("PMSCT_RLE1")
                || text(obj, Tag(0x07a1, 0x1011))
                    .to_ascii_uppercase()
                    .contains("PMSCT_RLE1")
        },
        is_bvec_world_coordinates: false,
        gantry_tilt: first_f64(obj, Tag(0x0018, 0x1120)).unwrap_or(0.0),
        study_uid_crc: crc32(&text(obj, tags::STUDY_INSTANCE_UID)),
        coil_crc: 0, // filled after struct build from coil_name
        date_time: {
            let sd = text(obj, tags::STUDY_DATE);
            let st = text(obj, tags::STUDY_TIME);
            sd.parse::<f64>().unwrap_or(0.0) * 1_000_000.0 + st.parse::<f64>().unwrap_or(0.0)
        },
        is_has_phase,
        is_has_real,
        is_has_imaginary,
        is_has_magnitude,
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
        convolution_kernel: text(obj, tags::CONVOLUTION_KERNEL),
        recon_filter_size: f64::NAN,
        pixel_padding_value: first_f64(obj, Tag(0x0028, 0x0120)).unwrap_or(f64::NAN),
        is_xray: matches!(
            Modality::from_tag(&text(obj, tags::MODALITY)),
            Modality::Ct | Modality::Cr
        ),
        exposure_time_ms: first_f64(obj, Tag(0x0018, 0x1150))
            .or_else(|| int(obj, Tag(0x0018, 0x1150)).map(|v| v as f64))
            .unwrap_or(0.0),
        x_ray_tube_current: first_f64(obj, Tag(0x0018, 0x1151))
            .or_else(|| int(obj, Tag(0x0018, 0x1151)).map(|v| v as f64))
            .unwrap_or(0.0),
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
        deidentification_method: {
            let mut s = text(obj, Tag(0x0012, 0x0063));
            if s.contains('\\') {
                s = s.replace('\\', "_");
            }
            s
        },
        deidentification_method_code_sequence: read_deid_code_sequence(obj),
        ecat_isotope_halflife: 0.0,
        ecat_dosage: 0.0,
        volume_onset_times: Vec::new(),
        frame_durations: Vec::new(),
        frame_reference_times: Vec::new(),
        decay_factors: Vec::new(),
        radiopharmaceutical: text(obj, Tag(0x0018, 0x0031)),
        tracer_radionuclide: text(obj, Tag(0x0008, 0x0104)), // CodeMeaning when present
        radionuclide_total_dose: first_f64(obj, Tag(0x0018, 0x1074)).unwrap_or(0.0),
        radionuclide_half_life: first_f64(obj, Tag(0x0018, 0x1075)).unwrap_or(0.0),
        radionuclide_positron_fraction: first_f64(obj, Tag(0x0018, 0x1076)).unwrap_or(0.0),
        radiopharmaceutical_specific_activity: first_f64(obj, Tag(0x0018, 0x1077)).unwrap_or(0.0),
        injected_volume: first_f64(obj, Tag(0x0018, 0x1071)).unwrap_or(0.0),
        scatter_fraction: first_f64(obj, Tag(0x0054, 0x1323)).unwrap_or(0.0),
        radiopharmaceutical_start_time: text(obj, Tag(0x0018, 0x1072)),
        decay_correction: text(obj, Tag(0x0054, 0x1102)),
        attenuation_correction_method: text(obj, Tag(0x0054, 0x1101)),
        randoms_correction_method: text(obj, Tag(0x0054, 0x1100)),
        scatter_correction_method: text(obj, Tag(0x0054, 0x1105)),
        reconstruction_method: text(obj, Tag(0x0054, 0x1103)),
        units_pt: text(obj, Tag(0x0054, 0x1001)),
        dose_calibration_factor: first_f64(obj, Tag(0x0054, 0x1322)).unwrap_or(0.0),
        };
        // issue 1024: CSA concatenations + fill-if-missing 3D accel.
        if d.csa.series.l_conc > 1 {
            d.number_of_concatenations = d.csa.series.l_conc;
        }
        if d.parallel_reduction_out_of_plane < 1.0
            && d.csa.series.parallel_reduction_factor_out_of_plane > 0
        {
            d.parallel_reduction_out_of_plane =
                d.csa.series.parallel_reduction_factor_out_of_plane as f64;
        } else if d.manufacturer != Manufacturer::Siemens {
            if let Some(oop) = first_f64(obj, Tag(0x0018, 0x9155)) {
                if oop >= 1.0 {
                    d.parallel_reduction_out_of_plane = oop;
                }
            }
        }
        // Siemens ImageTypeText (0021,1175): full `_`-delimited NOISE → RF-off.
        {
            let mut itt = text(obj, Tag(0x0021, 0x1175));
            if itt.contains('\\') {
                itt = itt.replace('\\', "_");
            }
            d.image_type_text = itt.clone();
            d.is_no_rf = image_type_has_noise_token(&itt);
            // DeepReveal Boost/Gain/Sharp (dcm_qa_cs_dl).
            let it = d.image_type.replace('\\', "_");
            if d.manufacturer == Manufacturer::Siemens
                && (it.contains("_DRB_")
                    || it.contains("_DRG_")
                    || it.contains("_DRS_")
                    || itt.contains("_DRB_")
                    || itt.contains("_DRG_")
                    || itt.contains("_DRS_"))
            {
                d.is_deep_learning = true;
            }
            let dl = text(obj, Tag(0x0021, 0x1176));
            if !dl.is_empty() {
                d.deep_learning_text = dl;
                d.is_deep_learning = true;
            }
        }
        // (0018,9058) FrequencyEncodingSteps; (0018,1315) VariableFlipAngleFlag;
        // (0018,9078) ParallelAcquisitionTechnique.
        if let Some(n) = int(obj, Tag(0x0018, 0x9058)) {
            d.frequency_encoding_steps = n;
        }
        {
            let vf = text(obj, Tag(0x0018, 0x1315)).to_ascii_uppercase();
            if vf.starts_with('Y') {
                d.is_variable_flip_angle = true;
            }
        }
        d.parallel_acquisition_technique = text(obj, Tag(0x0018, 0x9078));
        // GE / Philips deep-learning private text.
        if d.manufacturer == Manufacturer::Ge {
            let dl = text(obj, Tag(0x0043, 0x10CA));
            if !dl.is_empty() {
                d.deep_learning_text = dl;
                d.is_deep_learning = true;
            }
        }
        if d.manufacturer == Manufacturer::Philips {
            let dl = text(obj, Tag(0x2005, 0x1110)).to_ascii_uppercase();
            if !dl.is_empty() && dl != "NONE" {
                d.deep_learning_text = text(obj, Tag(0x2005, 0x1110));
                d.is_deep_learning = true;
            }
        }
        detect_physio(path, obj, &mut d);
        apply_ge_derived_fields(obj, &mut d);
        parse_overlays(obj, &mut d);
        // Media Storage / SOP Class UID → Raw / PS / SEG / MRS routing flags.
        {
            let sop = text(obj, tags::SOP_CLASS_UID);
            let media = text(obj, Tag(0x0002, 0x0002));
            let uid = if !media.is_empty() { media } else { sop };
            if uid.contains("1.2.840.10008.5.1.4.1.1.66.4")
                || text(obj, tags::MODALITY).eq_ignore_ascii_case("SEG")
            {
                d.modality = Modality::Seg;
                d.is_derived = true;
            } else if uid.contains("1.2.840.10008.5.1.4.1.1.66")
                || uid.contains("1.3.46.670589.11.0.0.12.1")
                || uid.contains("1.3.46.670589.11.0.0.12.2")
                || uid.contains("1.3.46.670589.11.0.0.12.4")
                || uid.contains("1.3.12.2.1107.5.9.1")
            {
                // Skip marking Raw when already routed to MRS/physio.
                if !d.is_mrs && !d.is_xa_physio && !d.is_cmrr_physio {
                    d.is_raw_data_storage = true;
                    d.is_derived = true;
                }
            }
            if uid.contains("1.2.840.10008.5.1.4.1.1.11.1") {
                d.is_grayscale_softcopy_presentation_state = true;
                d.is_derived = true;
            }
        }
        if d.manufacturer == Manufacturer::Ge {
            if let Some(sz) = first_f64(obj, Tag(0x0009, 0x108F)) {
                d.recon_filter_size = sz;
            }
        }
        // (0010,2210) AnatomicalOrientationType → QUADRUPED (issue 642).
        {
            let aot = text(obj, Tag(0x0010, 0x2210)).to_ascii_uppercase();
            if aot.len() >= 9 && aot.contains("QUADRUPED") {
                d.is_quadruped = true;
            }
        }
        // Matlab DICOMANON can scramble UIDs (issue 383).
        if d.deidentification_method
            .to_ascii_uppercase()
            .contains("DICOMANON")
        {
            eprintln!(
                "Warning: Matlab DICOMANON can scramble SeriesInstanceUID (0020,000e) and remove crucial data (see issue 383)."
            );
        }
        // Philips TriggerDelayTime (0020,9153); GE uses 0018,1060 for slice timing.
        if d.manufacturer == Manufacturer::Philips {
            if let Some(t) = first_f64(obj, Tag(0x0020, 0x9153)) {
                d.trigger_delay_time = if t.abs() < 1e-6 { 0.0 } else { t };
            }
            let lbl = text(obj, Tag(0x2005, 0x1429));
            if !lbl.is_empty() {
                d.asl_flags |= flags_from_philips_label_type(&lbl);
            }
        }
        // ASL public + GE private tags.
        {
            let contrast = text(obj, Tag(0x0018, 0x9250));
            if !contrast.is_empty() {
                d.asl_flags |= flags_from_asl_contrast(&contrast);
            }
            if let Some(pld) = int(obj, Tag(0x0018, 0x9258)) {
                d.post_label_delay = pld;
            } else if let Ok(elem) = obj.element(Tag(0x0018, 0x9258)) {
                if let Ok(s) = elem.to_str() {
                    if let Ok(v) = s.trim().parse::<i32>() {
                        d.post_label_delay = v;
                    }
                }
            }
            // LabelingOrientation (0018,9255), VascularCrushing (0018,9259/925A).
            d.labeling_orientation = text(obj, Tag(0x0018, 0x9255));
            {
                let vc = text(obj, Tag(0x0018, 0x9259)).to_ascii_uppercase();
                if vc.starts_with('Y') {
                    d.vascular_crushing = 1;
                } else if vc.starts_with('N') {
                    d.vascular_crushing = 0;
                }
            }
            if let Some(v) = first_f64(obj, Tag(0x0018, 0x925A)) {
                if v > 0.0 {
                    d.vascular_crushing_venc = v;
                }
            }
        }
        // GE RTIA timer (0021,105E) — seconds.
        if d.manufacturer == Manufacturer::Ge {
            if let Some(t) = first_f64(obj, Tag(0x0021, 0x105E)) {
                d.rtia_timer_ge = t;
            }
            let ct = text(obj, Tag(0x0043, 0x10A3));
            if !ct.is_empty() {
                d.asl_flags |= flags_from_ge_contrast_technique(&ct);
            }
            let lt = text(obj, Tag(0x0043, 0x10A4));
            if !lt.is_empty() {
                d.asl_flags |= flags_from_ge_labeling_technique(&lt);
            }
            if let Some(v) = int(obj, Tag(0x0043, 0x10A5)) {
                d.duration_label_pulse_ge = v;
            } else if let Ok(elem) = obj.element(Tag(0x0043, 0x10A5)) {
                if let Ok(s) = elem.to_str() {
                    if let Ok(n) = s.trim().parse::<i32>() {
                        d.duration_label_pulse_ge = n;
                    }
                }
            }
            // (0027,1060/61/62) points/arms/excitations (FL).
            if let Some(v) = first_f64(obj, Tag(0x0027, 0x1060)) {
                d.number_of_points_per_arm = v;
            }
            if let Some(v) = first_f64(obj, Tag(0x0027, 0x1061)) {
                d.number_of_arms = v;
            }
            if let Some(v) = first_f64(obj, Tag(0x0027, 0x1062)) {
                d.number_of_excitations = v;
            }
            // Group delay (0043,107C) seconds → ms.
            if let Some(gd) = first_f64(obj, Tag(0x0043, 0x107C)) {
                d.group_delay = gd * 1000.0;
                if d.group_delay > 0.0 {
                    d.tr += d.group_delay;
                }
            }
            // Protocol Data Block (0025,101B).
            if let Ok(elem) = obj.element(Tag(0x0025, 0x101B)) {
                if let Ok(bytes) = elem.to_bytes() {
                    if let Some(pb) = parse_ge_protocol_block(&bytes) {
                        if pb.mb_accel > 1 {
                            d.csa.image.multi_band_factor =
                                d.csa.image.multi_band_factor.max(pb.mb_accel);
                        }
                        d.ge_slice_order = pb.slice_order;
                        d.ge_iopt = pb.iopt.clone();
                        let gd_ms = pb.group_delay_s * 1000.0;
                        if gd_ms > 0.0 {
                            if d.group_delay <= 0.0 {
                                d.tr += gd_ms;
                            }
                            d.group_delay = gd_ms;
                        } else if pb.group_delay_s < -0.5 {
                            d.group_delay = pb.group_delay_s;
                        }
                    }
                }
            }
            // Pulse sequence name (0019,109C) → epiVersionGE.
            let psn = text(obj, Tag(0x0019, 0x109C)).to_ascii_lowercase();
            if psn.contains("epi_pepolar") {
                d.epi_version_ge = GE_EPI_PEPOLAR_FWD;
            } else if psn.contains("epi2") {
                d.epi_version_ge = GE_EPI_EPI2;
            } else if psn.contains("epirt") {
                d.epi_version_ge = GE_EPI_EPIRT;
            } else if psn.contains("epi") {
                d.epi_version_ge = GE_EPI_EPI;
            }
            let ipn = d.internal_pulse_sequence_name.to_ascii_uppercase();
            // C++ only remaps EPI/EPI2 when not already a pepolar class.
            if d.epi_version_ge < GE_EPI_PEPOLAR_FWD {
                if ipn == "EPI" {
                    d.internal_epi_version_ge = 1;
                    if d.epi_version_ge != GE_EPI_EPIRT {
                        d.epi_version_ge = GE_EPI_EPI;
                    }
                } else if ipn == "EPI2" {
                    d.internal_epi_version_ge = 2;
                }
            } else if ipn == "EPI2" {
                d.internal_epi_version_ge = 2;
            } else if ipn == "EPI" {
                d.internal_epi_version_ge = 1;
            }
            // (0019,10B3) pepolar userData12 + (0020,0100) temporal volume.
            if let Some(v) = first_f64(obj, Tag(0x0019, 0x10B3)) {
                d.ge_user_data_12 = v.round() as i32;
            } else if let Some(v) = int(obj, Tag(0x0019, 0x10B3)) {
                d.ge_user_data_12 = v;
            }
            if let Some(v) = int(obj, tags::TEMPORAL_POSITION_IDENTIFIER) {
                d.temporal_position = v;
            }
            finalize_pepolar(
                &mut d.epi_version_ge,
                &mut d.phase_encoding_ge,
                &mut d.series_number,
                d.ge_user_data_12,
                d.temporal_position,
            );
            // GE diffusion gradient cycling (issue 635 / #796).
            {
                let user11 = first_f64(obj, Tag(0x0019, 0x10B2))
                    .map(|v| v.round() as i32)
                    .or_else(|| int(obj, Tag(0x0019, 0x10B2)))
                    .unwrap_or(0);
                let user15 = first_f64(obj, Tag(0x0019, 0x10B6)).unwrap_or(0.0);
                let (mode, tensor) = detect_diff_cycling(
                    &d.manufacturers_model_name,
                    d.epi_version_ge,
                    d.internal_epi_version_ge,
                    user11,
                    d.ge_user_data_12,
                    user15,
                );
                d.diff_cycling_mode_ge = mode;
                if tensor > 0 {
                    d.tensor_file_ge = tensor;
                }
                if let Some(v) = first_f64(obj, Tag(0x0019, 0x10E0)) {
                    d.number_of_diffusion_direction_ge = v.round() as i32;
                }
                if let Some(v) = first_f64(obj, Tag(0x0019, 0x10DF)) {
                    d.number_of_diffusion_t2_ge = v.round() as i32;
                }
                if let Some(v) = first_f64(obj, Tag(0x0019, 0x10E2)) {
                    d.velocity_encode_scale_ge = v;
                }
                if let Some(v) = first_f64(obj, Tag(0x0019, 0x10A9)) {
                    d.max_echo_num_ge = v.round() as i32;
                } else if let Some(v) = int(obj, Tag(0x0019, 0x10A9)) {
                    d.max_echo_num_ge = v;
                }
                // Issue 690 / 777: zero NumberOfDiffusionDirectionGE for non-DTI.
                // Image-level VasCollapseFlag (0043,1030): 16=DTI, 14=T2/b0.
                // Series-level (0021,105A): 16=DFAXDTI.
                let img_dir = int(obj, Tag(0x0043, 0x1030)).unwrap_or(0);
                let series_dir = int(obj, Tag(0x0021, 0x105A)).unwrap_or(0);
                if should_zero_ge_diffusion_directions(img_dir, series_dir) {
                    d.number_of_diffusion_direction_ge = 0;
                }
                // CompressedSensingParameters (0043,10B7) LO "factor\..." (issue 672).
                let cs = text(obj, Tag(0x0043, 0x10B7));
                if !cs.is_empty() {
                    let factor = cs
                        .split(|c| c == '\\' || c == '/')
                        .next()
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .unwrap_or(0.0);
                    if factor > 1.0 {
                        d.compressed_sensing_factor = factor;
                    }
                }
            }
            // ZIP through-plane interpolation (issue 373).
            if d.spacing_between_slices > 0.0 && d.xyz_mm[3] > 0.0 {
                let zip = (d.xyz_mm[3] / d.spacing_between_slices).round() as i32;
                if zip > 1 {
                    d.interp_3d = zip;
                }
            }
        }
        // MagnetizationTransfer (0018,9020) and Spoiling (0018,9016).
        {
            let mt = text(obj, Tag(0x0018, 0x9020)).to_ascii_uppercase();
            if mt.starts_with("OF") {
                d.mt_state = 1; // OFF_RESONANCE → true
            } else if mt.starts_with("ON") || mt.starts_with("NO") {
                d.mt_state = 0; // ON_RESONANCE / NONE → false
            }
            let sp = text(obj, Tag(0x0018, 0x9016)).to_ascii_uppercase();
            if !sp.is_empty() {
                let has_rf = sp.contains("RF");
                let has_gr = sp.contains("GRADIENT");
                if has_rf && has_gr {
                    d.spoiling = 3;
                } else if has_rf {
                    d.spoiling = 1;
                } else if has_gr {
                    d.spoiling = 2;
                } else if sp.contains("NONE") {
                    d.spoiling = 0;
                }
            }
        }
        // PartialFourierDirection (0018,9036) + PartialFourier (0018,9081) / ScanOptions PFF.
        {
            let pfd = text(obj, Tag(0x0018, 0x9036)).to_ascii_uppercase();
            if pfd.starts_with('P') {
                d.partial_fourier_direction = 1;
            } else if pfd.starts_with('F') {
                d.partial_fourier_direction = 2;
            } else if pfd.starts_with('S') {
                d.partial_fourier_direction = 3;
            } else if pfd.starts_with('C') {
                d.partial_fourier_direction = 4;
            }
            let pf = text(obj, Tag(0x0018, 0x9081)).to_ascii_uppercase();
            if pf.starts_with('Y') {
                d.is_partial_fourier = true;
            }
            if d.scan_options.to_ascii_uppercase().contains("PFF") {
                d.is_partial_fourier = true;
            }
        }
        // Philips WaterFatShift / PhaseNumber.
        if d.manufacturer == Manufacturer::Philips {
            if let Some(wfs) = first_f64(obj, Tag(0x2001, 0x1022)) {
                d.water_fat_shift = wfs;
            }
            if let Some(pn) = int(obj, Tag(0x2001, 0x1008)) {
                d.phase_number = pn;
            } else if let Ok(elem) = obj.element(Tag(0x2001, 0x1008)) {
                if let Ok(s) = elem.to_str() {
                    if let Ok(v) = s.trim().parse::<i32>() {
                        d.phase_number = v;
                    }
                }
            }
            // (2005,1063) fMRIStatusIndication.
            if let Some(v) = int(obj, Tag(0x2005, 0x1063)) {
                d.raw_data_run_number = v;
            }
            // (2001,1019) Partial Matrix Scanned → PartialFourier.
            let pms = text(obj, Tag(0x2001, 0x1019)).to_ascii_uppercase();
            if pms.starts_with('Y') {
                d.is_partial_fourier = true;
            }
            // Real World Value slope/intercept (0040,9225 / 0040,9224).
            if let Some(s) = first_f64(obj, Tag(0x0040, 0x9225)) {
                if s <= 1.0e38 {
                    d.rwv_scale = s;
                    if (d.inten_scale - 1.0).abs() < 1e-6 {
                        d.inten_scale = s as f32;
                    }
                }
            }
            if let Some(i) = first_f64(obj, Tag(0x0040, 0x9224)) {
                d.rwv_intercept = i;
                if d.inten_intercept.abs() < 1e-12 {
                    d.inten_intercept = i as f32;
                }
            }
        }
        // (0008,9209) AcquisitionContrast.
        {
            let ac = text(obj, Tag(0x0008, 0x9209)).to_ascii_uppercase();
            let (w, set_diff) = match ac.as_str() {
                "DIFFUSION" => (7, true),
                "PERFUSION" => (8, false),
                "FLUID_ATTENUATED" => (5, false),
                "PROTON_DENSITY" => (3, false),
                "T2_STAR" => (4, false),
                "T1" => (1, false),
                "T2" => (2, false),
                "STIR" => (6, false),
                "TOF" => (9, false),
                "FLOW_ENCODED" => (10, false),
                "TAGGING" => (11, false),
                "MIXED" => (12, false),
                "OTHER" => (13, false),
                _ => (0, false),
            };
            if w > 0 {
                d.acquisition_contrast = w;
            }
            if set_diff {
                d.is_diffusion = true;
            }
        }
        if d.b_value >= 0.0 {
            d.is_diffusion = true;
            d.csa.image.num_dti = d.csa.image.num_dti.max(1);
        }
        if d.echo_number > 1 || d.csa.series.l_contrasts > 1 {
            d.is_multi_echo = true;
        }
        // GE B0map / fieldmap Hz.
        if d.manufacturer == Manufacturer::Ge {
            let psn = text(obj, Tag(0x0019, 0x109C)).to_ascii_lowercase();
            if psn.contains("b0map") || d.internal_pulse_sequence_name.eq_ignore_ascii_case("B0map")
            {
                d.is_real_is_phase_map_hz = d.is_has_real;
            }
        }
        d.is_planar_rgb = int(obj, tags::PLANAR_CONFIGURATION).unwrap_or(0) == 1;
        if let Some(fd) = first_f64(obj, Tag(0x0018, 0x1242)) {
            d.frame_duration = fd;
        }
        if let Some(fr) = first_f64(obj, Tag(0x0054, 0x1300)) {
            d.frame_reference_time = fr;
        }
        if let Some(df) = first_f64(obj, Tag(0x0018, 0x9731))
            .or_else(|| first_f64(obj, Tag(0x0054, 0x1321)))
        {
            d.decay_factor = df;
        }
        d.coil_crc = crc32(&d.coil_name);
        d
    })
}

/// Read DICOM Overlay groups `0x6000..0x601E` (even) into `d.overlays`.
/// Matches C++ overlayStart / isHasOverlay rules (bits=1, origin 1/1, size match).
fn parse_overlays(obj: &DefaultDicomObject, d: &mut DicomImage) {
    let mut ok = true;
    let mut overlay_rows = 0i32;
    let mut overlay_cols = 0i32;
    for oi in 0..16usize {
        let group = 0x6000u16 + (oi as u16) * 2;
        if let Some(r) = int(obj, Tag(group, 0x0010)) {
            overlay_rows = r;
        }
        if let Some(c) = int(obj, Tag(group, 0x0011)) {
            overlay_cols = c;
        }
        if let Ok(elem) = obj.element(Tag(group, 0x0050)) {
            // SS×2 OverlayOrigin — only (1,1) supported.
            if let Ok(bytes) = elem.to_bytes() {
                if bytes.len() >= 4 {
                    let row = i16::from_le_bytes([bytes[0], bytes[1]]);
                    let col = i16::from_le_bytes([bytes[2], bytes[3]]);
                    if row != 1 || col != 1 {
                        eprintln!("Unsupported overlay origin {row}/{col}");
                        ok = false;
                    }
                }
            }
        }
        if let Some(bits) = int(obj, Tag(group, 0x0100)) {
            if bits != 1 {
                eprintln!(
                    "Illegal/Obsolete DICOM: Overlay Bits Allocated must be 1, not {bits}"
                );
                ok = false;
            }
        }
        if let Some(pos) = int(obj, Tag(group, 0x0102)) {
            if pos != 0 {
                eprintln!(
                    "Illegal/Obsolete DICOM: Overlay Bit Position shall be 0, not {pos}"
                );
                ok = false;
            }
        }
        if let Ok(elem) = obj.element(Tag(group, 0x3000)) {
            if let Ok(bytes) = elem.to_bytes() {
                d.overlays[oi] = Some(bytes.to_vec());
                d.is_has_overlay = true;
            }
        }
    }
    if d.is_has_overlay {
        if overlay_cols > 0 && d.columns != overlay_cols as usize {
            ok = false;
        }
        if overlay_rows > 0 && d.rows != overlay_rows as usize {
            ok = false;
        }
        if !ok {
            d.is_has_overlay = false;
            d.overlays = Default::default();
        }
    }
}

fn image_type_has_noise_token(image_type_text: &str) -> bool {
    // Match "NOISE" as a full `_`-delimited token (issue #1025); never a bare
    // substring like "NOISELESS".
    let s = image_type_text;
    let bytes = s.as_bytes();
    let mut start = 0;
    while let Some(rel) = s[start..].find("NOISE") {
        let i = start + rel;
        let before = if i == 0 { b'_' } else { bytes[i - 1] };
        let after = bytes.get(i + 5).copied().unwrap_or(b'_');
        if before == b'_' && after == b'_' {
            return true;
        }
        start = i + 1;
    }
    false
}

fn detect_physio(path: &Path, obj: &DefaultDicomObject, d: &mut DicomImage) {
    let Ok(elem) = obj.element(Tag(0x7FE1, 0x1010)) else {
        return;
    };
    let Ok(bytes) = elem.to_bytes() else {
        return;
    };
    if bytes.len() < 4 {
        return;
    }
    d.physio_bytes = bytes.len() as i32;
    d.physio_offset = 0;
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        d.is_xa_physio = true;
        return;
    }
    // CMRR VE11C: validate first waveform header (C++ sniff).
    if bytes.len() <= 1024 {
        return;
    }
    let data_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let fname_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if data_len < 16 || data_len > bytes.len() || !(4..=255).contains(&fname_len) {
        return;
    }
    if fname_len + 8 > bytes.len() {
        return;
    }
    let fname = String::from_utf8_lossy(&bytes[8..8 + fname_len]);
    let ok_name = fname.contains("_PULS.log")
        || fname.contains("_RESP.log")
        || fname.contains("_EXT.log")
        || fname.contains("_ECG.log")
        || fname.contains("_Info.log");
    if !ok_name {
        return;
    }
    let first = bytes[1024];
    if (0x20..=0x7e).contains(&first) {
        d.is_cmrr_physio = true;
    }
    let _ = path;
}

/// Read Siemens private `(7FE1,1010)` physio bytes from a DICOM file.
pub fn physio_payload(path: &Path) -> Result<Option<Vec<u8>>> {
    let obj = open(path)?;
    let Ok(elem) = obj.element(Tag(0x7FE1, 0x1010)) else {
        return Ok(None);
    };
    let Ok(bytes) = elem.to_bytes() else {
        return Ok(None);
    };
    Ok(Some(bytes.into_owned()))
}

/// Spectroscopy Data `(5600,0020)` as interleaved real/imag float32, plus
/// Data Point Columns `(0028,9002)`.
pub fn spectroscopy_data_from_object(
    obj: &DefaultDicomObject,
) -> Result<Option<(Vec<f32>, usize)>> {
    let cols = int(obj, Tag(0x0028, 0x9002)).unwrap_or(0).max(0) as usize;
    let Ok(elem) = obj.element(Tag(0x5600, 0x0020)) else {
        return Ok(None);
    };
    let Ok(bytes) = elem.to_bytes() else {
        return Ok(None);
    };
    if bytes.len() < 8 {
        return Ok(None);
    }
    let n_f32 = bytes.len() / 4;
    let mut out = Vec::with_capacity(n_f32);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let n_pts = if cols > 0 {
        cols
    } else {
        (n_f32 / 2).max(1)
    };
    Ok(Some((out, n_pts)))
}

/// Read spectroscopy payload from a DICOM file (full parse).
pub fn spectroscopy_data(path: &Path) -> Result<Option<(Vec<f32>, usize)>> {
    spectroscopy_data_from_object(&open(path)?)
}

/// Spectroscopy payload using a convert-scoped mmap cache from [`prefetch_mmaps`].
pub fn spectroscopy_data_prefetched(
    path: &Path,
    cache: &MmapCache,
) -> Result<Option<(Vec<f32>, usize)>> {
    spectroscopy_data_from_object(&open_prefetched(path, cache)?)
}

fn apply_ge_derived_fields(obj: &DefaultDicomObject, d: &mut DicomImage) {
    if d.manufacturer != Manufacturer::Ge {
        return;
    }
    if d.phase_encoding_ge == 0 {
        d.csa.image.phase_encoding_direction_positive = 1;
    } else if d.phase_encoding_ge == 4 {
        d.csa.image.phase_encoding_direction_positive = 0;
    }
    if d.coil_string.is_empty() {
        d.coil_string = d.coil_name.clone();
    }
    if d.procedure_step_description.is_empty() && d.scanning_sequence.contains("GR") {
        d.procedure_step_description = "Gradient Echo".into();
    }
    if let Some(vals) = f64s(obj, Tag(0x0043, 0x10B2)) {
        if vals.len() >= 3 {
            d.csa.image.table_pos[0] = 1.0;
            d.csa.image.table_pos[3] = vals[2];
        }
    } else if let Some(s) = obj
        .element(Tag(0x0043, 0x10B2))
        .ok()
        .and_then(|e| e.to_str().ok())
    {
        let parts: Vec<f64> = s
            .split(|c: char| c == '\\' || c == '\\' || c.is_whitespace())
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        if parts.len() >= 3 {
            d.csa.image.table_pos[0] = 1.0;
            d.csa.image.table_pos[3] = parts[2];
        }
    }
}

fn parse_ge_phase_polarity(obj: &DefaultDicomObject) -> i32 {
    let Some(raw) = obj
        .element(Tag(0x0043, 0x102A))
        .ok()
        .and_then(|e| e.to_bytes().ok())
    else {
        return -1;
    };
    if raw.len() < 32 {
        return -1;
    }
    let hdr_offset = u16::from_le_bytes([raw[24], raw[25]]) as usize;
    if raw.len() < hdr_offset + 0x40 {
        return -1;
    }
    let version = f32::from_le_bytes(
        raw[hdr_offset..hdr_offset + 4]
            .try_into()
            .unwrap_or([0; 4]),
    );
    let mut hdr = hdr_offset;
    if version >= 25.002 {
        hdr += 0x004c;
    }
    if raw.len() < hdr + 0x32 {
        return -1;
    }
    (u16::from_le_bytes([raw[hdr + 0x30], raw[hdr + 0x31]]) & 0x0004) as i32
}

/// Decode pixels as f32 without applying the VOI LUT (display window).
/// The modality LUT (rescale) is left to the NIfTI scl_slope/intercept.
pub fn decode_pixels_raw_f32(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let obj = open(path)?;
    decode_opened_raw_f32(path, &obj)
}

/// Decode pixels using a convert-scoped mmap cache from [`prefetch_mmaps`].
pub fn decode_pixels_prefetched(
    path: &Path,
    cache: &MmapCache,
) -> Result<(Vec<f32>, usize, usize)> {
    let obj = open_prefetched(path, cache)?;
    decode_opened_raw_f32(path, &obj)
}

/// Decode pixels from an already-opened DICOM object (avoids a second `open`).
pub fn decode_opened_raw_f32(
    path: &Path,
    obj: &DefaultDicomObject,
) -> Result<(Vec<f32>, usize, usize)> {
    // PMSCT_RLE1 private compression — bypass dicom-pixeldata.
    let is_pmsct = text(obj, Tag(0x07a1, 0x100a))
        .to_ascii_uppercase()
        .contains("PMSCT_RLE1")
        || text(obj, Tag(0x07a1, 0x1011))
            .to_ascii_uppercase()
            .contains("PMSCT_RLE1");
    if is_pmsct {
        return decode_pmsct_f32(path, obj);
    }
    decode_object_raw_f32(path, obj)
}

fn decode_pmsct_f32(path: &Path, obj: &DefaultDicomObject) -> Result<(Vec<f32>, usize, usize)> {
    let rows = int(obj, tags::ROWS).unwrap_or(0).max(0) as usize;
    let cols = int(obj, tags::COLUMNS).unwrap_or(0).max(0) as usize;
    let frames = int(obj, tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1) as usize;
    let n = rows * cols * frames;
    let Ok(elem) = obj.element(tags::PIXEL_DATA) else {
        return Err(err(path, "PMSCT_RLE1: missing PixelData"));
    };
    let Ok(bytes) = elem.to_bytes() else {
        return Err(err(path, "PMSCT_RLE1: cannot read PixelData bytes"));
    };
    let decoded = decode_pmsct_rle1(&bytes, n)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = u16::from_le_bytes([decoded[i * 2], decoded[i * 2 + 1]]);
        out.push(v as f32);
    }
    Ok((out, rows, cols))
}

pub fn decode_object_raw_f32(
    path: &Path,
    obj: &FileDicomObject<InMemDicomObject>,
) -> Result<(Vec<f32>, usize, usize)> {
    let ts = obj.meta().transfer_syntax();
    if ts.contains("1.2.840.10008.1.2.4.201")
        || ts.contains("1.2.840.10008.1.2.4.202")
        || ts.contains("1.2.840.10008.1.2.4.203")
    {
        // C++ maps HTJ2K UIDs to the JPEG2000 path (issue 897).
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "HTJ2K transfer syntax detected ({ts}); decoding via OpenJPEG (please validate)"
            );
        }
    }
    match obj.decode_pixel_data() {
        Ok(decoded) => decode_decoded_raw_f32(path, obj, decoded),
        Err(e) => {
            // Fallback for ancient JPEG Lossless SOF 0xC3 (1.2.840.10008.1.2.4.57/70)
            // when the primary adapter path fails on fragmented bitstreams.
            if let Some(v) = try_jpeg_lossless_fallback(path, obj) {
                return v;
            }
            Err(err(path, format!("decoding pixel data: {e}")))
        }
    }
}

fn decode_decoded_raw_f32(
    path: &Path,
    obj: &FileDicomObject<InMemDicomObject>,
    decoded: dicom_pixeldata::DecodedPixelData<'_>,
) -> Result<(Vec<f32>, usize, usize)> {
    let rows = decoded.rows() as usize;
    let cols = decoded.columns() as usize;
    let _ = obj; // slope/intercept live on the NIfTI header; voxels stay stored values

    // Fast path: monochrome 8/16-bit native samples → f32 (no f64, no LUT).
    // Matches `ModalityLutOption::None` semantics used by the fallback below.
    if let Some(stored) = mono_raw_to_f32(&decoded) {
        return Ok((stored, rows, cols));
    }

    let options = ConvertOptions::new()
        .with_modality_lut(ModalityLutOption::None)
        .with_voi_lut(VoiLutOption::Identity);
    let stored: Vec<f32> = decoded
        .to_vec_with_options(&options)
        .map_err(|e| err(path, format!("converting pixel data: {e}")))?;
    Ok((stored, rows, cols))
}

/// Parallel convert of decoded monochrome 8/16-bit samples to stored `f32`.
fn mono_raw_to_f32(decoded: &dicom_pixeldata::DecodedPixelData<'_>) -> Option<Vec<f32>> {
    if decoded.samples_per_pixel() != 1 {
        return None;
    }
    let data = decoded.data();
    match decoded.bits_allocated() {
        8 => {
            // Same as dicom-pixeldata `ModalityLutOption::None` for 8-bit:
            // treat bytes as unsigned (signed 8-bit DICOMs are vanishingly rare).
            let mut out = vec![0f32; data.len()];
            out.par_iter_mut()
                .zip(data.par_iter())
                .for_each(|(o, &b)| *o = b as f32);
            Some(out)
        }
        16 => {
            if data.len() % 2 != 0 {
                return None;
            }
            let n = data.len() / 2;
            let mut out = vec![0f32; n];
            match decoded.pixel_representation() {
                PixelRepresentation::Unsigned => {
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        *o = u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as f32;
                    });
                }
                PixelRepresentation::Signed => {
                    out.par_iter_mut().enumerate().for_each(|(i, o)| {
                        *o = i16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as f32;
                    });
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Direct jpeg-decoder path for Transfer Syntaxes 1.2.840.10008.1.2.4.57 / .70.
fn try_jpeg_lossless_fallback(
    path: &Path,
    obj: &FileDicomObject<InMemDicomObject>,
) -> Option<Result<(Vec<f32>, usize, usize)>> {
    let ts = obj.meta().transfer_syntax().to_string();
    if !ts.contains("1.2.840.10008.1.2.4.57") && !ts.contains("1.2.840.10008.1.2.4.70") {
        return None;
    }
    let rows = int(obj, tags::ROWS)?.max(0) as usize;
    let cols = int(obj, tags::COLUMNS)?.max(0) as usize;
    let Ok(elem) = obj.element(tags::PIXEL_DATA) else {
        return None;
    };
    let Ok(bytes) = elem.to_bytes() else {
        return None;
    };
    // Encapsulated pixel data: skip Basic Offset Table item and decode fragments.
    let jpeg = extract_jpeg_bitstream(&bytes)?;
    match decode_sof3_to_f32(&jpeg, rows, cols) {
        Ok(v) => Some(Ok(v)),
        Err(e) => Some(Err(err(path, e))),
    }
}

fn extract_jpeg_bitstream(encapsulated: &[u8]) -> Option<Vec<u8>> {
    // Upstream J2K/JPEG fragment guard: degenerate buffers must not be sniffed.
    if encapsulated.len() <= 8 {
        return None;
    }
    // Look for SOI marker FF D8 anywhere and take through EOI FF D9.
    let mut i = 0;
    while i + 1 < encapsulated.len() {
        if encapsulated[i] == 0xFF && encapsulated[i + 1] == 0xD8 {
            let mut j = i + 2;
            while j + 1 < encapsulated.len() {
                if encapsulated[j] == 0xFF && encapsulated[j + 1] == 0xD9 {
                    return Some(encapsulated[i..j + 2].to_vec());
                }
                j += 1;
            }
            return Some(encapsulated[i..].to_vec());
        }
        i += 1;
    }
    // Already a bare JPEG stream.
    if encapsulated.len() >= 2 && encapsulated[0] == 0xFF && encapsulated[1] == 0xD8 {
        return Some(encapsulated.to_vec());
    }
    None
}

fn decode_sof3_to_f32(jpeg: &[u8], rows: usize, cols: usize) -> std::result::Result<(Vec<f32>, usize, usize), String> {
    use jpeg_decoder::{Decoder, PixelFormat};
    let mut dec = Decoder::new(std::io::Cursor::new(jpeg));
    let pixels = dec.decode().map_err(|e| format!("JPEG lossless decode: {e}"))?;
    let info = dec.info().ok_or_else(|| "JPEG lossless: missing frame info".to_string())?;
    let w = info.width as usize;
    let h = info.height as usize;
    let out_rows = if rows > 0 { rows } else { h };
    let out_cols = if cols > 0 { cols } else { w };
    let stored: Vec<f32> = match info.pixel_format {
        PixelFormat::L8 => pixels.iter().map(|&p| p as f32).collect(),
        PixelFormat::L16 => pixels
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]) as f32)
            .collect(),
        PixelFormat::RGB24 | PixelFormat::CMYK32 => pixels.iter().map(|&p| p as f32).collect(),
    };
    if stored.len() < out_rows * out_cols && stored.len() != w * h {
        return Err(format!(
            "JPEG lossless size mismatch: got {} samples for {out_cols}x{out_rows}",
            stored.len()
        ));
    }
    Ok((stored, out_rows, out_cols.max(1)))
}

pub fn collect_dicom_files(root: &Path, max_depth: u32) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    if !root.is_dir() {
        return Err(Error::bad_file(format!(
            "input folder invalid: {}",
            root.display()
        )));
    }
    let mut candidates = Vec::new();
    collect_candidates(root, 0, max_depth, &mut candidates);
    // Validate in parallel: prefer a cheap DICM-preamble sniff so we do not
    // fully parse every file twice (collect + read_header).
    use rayon::prelude::*;
    let mut out: Vec<PathBuf> = candidates
        .into_par_iter()
        .filter(|p| looks_like_dicom(p))
        .collect();
    out.sort();
    Ok(out)
}

fn collect_candidates(dir: &Path, depth: u32, max_depth: u32, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if depth < max_depth {
                collect_candidates(&p, depth + 1, max_depth, out);
            }
        } else {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            if name == "DICOMDIR" || name.starts_with('.') {
                continue;
            }
            if likely_non_dicom_name(&name) {
                continue;
            }
            out.push(p);
        }
    }
}

fn likely_non_dicom_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    // Extension-based rejects (common BIDS / sidecar / image junk beside DICOMs).
    const SKIP: &[&str] = &[
        ".nii", ".nii.gz", ".gz", ".json", ".bval", ".bvec", ".txt", ".md", ".html",
        ".htm", ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".py", ".sh", ".csv", ".tsv",
        ".xlsx", ".xml", ".log", ".nrrd", ".nhdr", ".mgh", ".mgz", ".gif", ".svg",
        ".parquet", ".h5", ".hdf5", ".mat", ".npz", ".yml", ".yaml", ".toml", ".rs",
        ".c", ".cpp", ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".bat", ".ds_store",
    ];
    SKIP.iter().any(|s| lower.ends_with(s))
}

/// Fast accept: Part-10 `DICM` at byte 128. Otherwise header-only parse
/// (mmap when ≤64 MiB) for preamble-less / Implicit VR files.
fn looks_like_dicom(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 132];
    let Ok(n) = f.read(&mut buf) else {
        return false;
    };
    if n >= 132 && &buf[128..132] == b"DICM" {
        return true;
    }
    open_header_cached(path).is_ok()
}

/// Issue 690 / 777: non-DTI GE series must not report diffusion direction counts.
pub(crate) fn should_zero_ge_diffusion_directions(img_dir: i32, series_dir: i32) -> bool {
    (img_dir > 0 && img_dir != 16 && img_dir != 14)
        || (series_dir > 0 && series_dir != 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ge_diffusion_direction_zeroing() {
        assert!(!should_zero_ge_diffusion_directions(0, 0));
        assert!(!should_zero_ge_diffusion_directions(16, 0));
        assert!(!should_zero_ge_diffusion_directions(14, 0));
        assert!(!should_zero_ge_diffusion_directions(0, 16));
        assert!(should_zero_ge_diffusion_directions(15, 0));
        assert!(should_zero_ge_diffusion_directions(0, 15));
    }

    #[test]
    fn manufacturer_siemens() {
        assert_eq!(
            Manufacturer::from_tag("SIEMENS"),
            Manufacturer::Siemens
        );
    }

    #[test]
    fn uih_list_tags() {
        let p = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/dti_tra_dir16_AP_SaveBySlc__140028/00000001.dcm",
        );
        if !p.is_file() {
            return;
        }
        let obj = open(p).unwrap();
        for elem in obj.iter() {
            let tag = elem.header().tag;
            if tag.group() == 0x0065 || tag == tags::IMAGE_POSITION_PATIENT {
                eprintln!(
                    "{:04x},{:04x} {:?}",
                    tag.group(),
                    tag.element(),
                    elem.to_str().ok()
                );
            }
        }
    }

    #[test]
    fn uih_t1_meta() {
        let dir = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/t1_gre_fsp_3d_sag__134917",
        );
        if !dir.is_dir() {
            return;
        }
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "dcm").unwrap_or(false))
            .collect();
        paths.sort();
        let first = read_header(&paths[0]).unwrap();
        let last = read_header(paths.last().unwrap()).unwrap();
        eprintln!(
            "T1 first inst={} rows={} cols={} 3d={} epi={} ipp={:?}",
            first.instance_number,
            first.rows,
            first.columns,
            first.is_3d_acq,
            first.is_epi,
            &first.patient_position[1..4]
        );
        eprintln!(
            "T1 last inst={} ipp={:?}",
            last.instance_number,
            &last.patient_position[1..4]
        );
        assert!(first.is_3d_acq);
        assert!(!first.is_epi);
        let mut by_depth = paths.clone();
        by_depth.sort_by(|a, b| {
            let da = read_header(a).unwrap();
            let db = read_header(b).unwrap();
            let na = [
                da.orient[4] * da.orient[3] - da.orient[5] * da.orient[2],
                da.orient[5] * da.orient[1] - da.orient[3] * da.orient[4],
                da.orient[2] * da.orient[5] - da.orient[4] * da.orient[1],
            ];
            let nb = [
                db.orient[4] * db.orient[3] - db.orient[5] * db.orient[2],
                db.orient[5] * db.orient[1] - db.orient[3] * db.orient[4],
                db.orient[2] * db.orient[5] - db.orient[4] * db.orient[1],
            ];
            let pa = [
                da.patient_position[1],
                da.patient_position[2],
                da.patient_position[3],
            ];
            let pb = [
                db.patient_position[1],
                db.patient_position[2],
                db.patient_position[3],
            ];
            let da = pa[0] * na[0] + pa[1] * na[1] + pa[2] * na[2];
            let db = pb[0] * nb[0] + pb[1] * nb[1] + pb[2] * nb[2];
            da.partial_cmp(&db).unwrap()
        });
        let order_match = by_depth.iter().zip(paths.iter()).all(|(a, b)| a == b);
        eprintln!("T1 depth order matches instance order: {order_match}");
        eprintln!(
            "depth-first {:?} inst-first {:?}",
            by_depth[0].file_name(),
            paths[0].file_name()
        );
        eprintln!(
            "depth-last {:?} inst-last {:?}",
            by_depth.last().unwrap().file_name(),
            paths.last().unwrap().file_name()
        );
    }

    #[test]
    fn uih_ipp_present() {
        let p = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/dti_tra_dir16_AP_SaveBySlc__140028/00000001.dcm",
        );
        if !p.is_file() {
            return;
        }
        let obj = open(p).unwrap();
        let ipp = f64s(&obj, tags::IMAGE_POSITION_PATIENT);
        eprintln!("IPP from f64s: {ipp:?}");
        let d = read_header(p).unwrap();
        eprintln!("DicomImage ipp: {:?}", &d.patient_position[1..4]);
        eprintln!("DicomImage orient: {:?}", &d.orient[1..7]);
        eprintln!("DicomImage has_orientation: {}", d.has_orientation());
    }

    #[test]
    fn uih_series_bvals() {
        let dir = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/dti_tra_dir16_AP_SaveBySlc__140028",
        );
        if !dir.is_dir() {
            return;
        }
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "dcm").unwrap_or(false))
            .collect();
        paths.sort();
        for p in &paths {
            let d = read_header(p).unwrap();
            eprintln!(
                "{} mosaic={} b={} grad={:?} ipp={:?}",
                p.file_name().unwrap().to_string_lossy(),
                d.csa.image.mosaic_slices,
                d.b_value,
                d.diffusion_direction,
                &d.patient_position[1..4]
            );
        }
        assert!(paths.len() >= 17);
    }

    #[test]


    fn uih_dti_acq_time() {
        let p = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/dti_tra_dir16_AP_SaveBySlc__140028/00000001.dcm",
        );
        if !p.is_file() {
            return;
        }
        let d = read_header(p).unwrap();
        assert_eq!(d.acquisition_time, "135713.284000");
    }

    #[test]
    fn uih_dti_tags() {
        let root = std::path::Path::new("/Users/Shared/dcm_qa_uih/In");
        if !root.is_dir() {
            return;
        }
        let path = std::fs::read_dir(root)
            .unwrap()
            .flatten()
            .find_map(|e| {
                let p = e.path();
                if p.is_dir() {
                    std::fs::read_dir(&p).ok()?.flatten().find_map(|f| {
                        let fp = f.path();
                        if fp.extension().map(|x| x == "dcm").unwrap_or(false) {
                            Some(fp)
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            });
        let Some(path) = path else { return };
        let d = read_header(&path).unwrap();
        assert!(d.csa.image.mosaic_slices > 1, "mosaic_slices");
        assert!(d.b_value >= 0.0, "b_value");
    }

    #[test]
    fn uih_t1_ref_slice_layout() {
        let dir = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/In/DTI_134434/t1_gre_fsp_3d_sag__134917",
        );
        let nii = std::path::Path::new(
            "/Users/Shared/dcm_qa_uih/Ref/t1_gre_fsp_3d_sag_2_134431.nii",
        );
        if !dir.is_dir() || !nii.is_file() {
            return;
        }
        let raw = std::fs::read(nii).unwrap();
        let nx = i16::from_le_bytes([raw[42], raw[43]]) as usize;
        let ny = i16::from_le_bytes([raw[44], raw[45]]) as usize;
        let nz = i16::from_le_bytes([raw[46], raw[47]]) as usize;
        let off = 352usize;
        let vox: Vec<u16> = raw[off..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(vox.len(), nx * ny * nz);

        let dcm = dir.join("00000001.dcm");
        let (pix, rows, cols) = decode_pixels_raw_f32(&dcm).unwrap();
        assert_eq!((rows, cols), (512, 460));

        let slice_ref = |si: usize| -> Vec<u16> {
            let mut s = Vec::with_capacity(ny * nz);
            for y in 0..ny {
                for z in 0..nz {
                    s.push(vox[si + nx * (y + ny * z)]);
                }
            }
            s
        };
        let match_rows_cols = |r: &[u16], d: &[f32]| -> f64 {
            if r.len() != d.len() {
                return 0.0;
            }
            r.iter()
                .zip(d.iter())
                .filter(|(a, b)| **a as i32 == **b as i32)
                .count() as f64
                / r.len() as f64
        };
        // ref[0,:,:] row-major ny x nz
        let r0 = slice_ref(0);
        let r159 = slice_ref(159);
        let (pix160, _, _) = decode_pixels_raw_f32(&dir.join("00000160.dcm")).unwrap();
        let to_u16 = |d: &[f32]| d.iter().map(|v| *v as u16).collect::<Vec<_>>();
        let p1 = to_u16(&pix);
        let p160 = to_u16(&pix160);
        // dicom row-major rows x cols; ref slice is ny x nz (460 x 512)
        let match_tr = |r: &[u16], rows: usize, cols: usize, p: &[u16]| -> f64 {
            let mut m = 0usize;
            for y in 0..rows {
                for x in 0..cols {
                    let rv = r[y * cols + x];
                    let pv = p[y * cols + x];
                    if rv == pv {
                        m += 1;
                    }
                }
            }
            m as f64 / r.len() as f64
        };
        let match_rt = |r: &[u16], rows: usize, cols: usize, p: &[u16]| -> f64 {
            let mut m = 0usize;
            for y in 0..cols {
                for x in 0..rows {
                    let rv = r[y * rows + x];
                    let pv = p[x * cols + y];
                    if rv == pv {
                        m += 1;
                    }
                }
            }
            m as f64 / r.len() as f64
        };
        eprintln!("ref[0] vs inst1 row-major {}", match_tr(&r0, rows, cols, &p1));
        eprintln!("ref[0] vs inst1 transposed {}", match_rt(&r0, rows, cols, &p1));
        eprintln!("ref[159] vs inst160 row-major {}", match_tr(&r159, rows, cols, &p160));
        eprintln!("ref[159] vs inst160 transposed {}", match_rt(&r159, rows, cols, &p160));
        eprintln!("ref[0] vs inst160 transposed {}", match_rt(&r0, rows, cols, &p160));
        eprintln!("ref[159] vs inst1 transposed {}", match_rt(&r159, rows, cols, &p1));

        // brute: which ref slice best matches inst1 transposed?
        let mut best_si = 0;
        let mut best_m = 0.0;
        for si in 0..nx {
            let rs = slice_ref(si);
            let m = match_rt(&rs, rows, cols, &p1);
            if m > best_m {
                best_m = m;
                best_si = si;
            }
        }
        eprintln!("best ref slice for inst1 transposed: si={best_si} match={best_m}");
    }

    #[test]
    fn crc_is_stable() {
        assert_eq!(crc32("abc"), crc32("abc"));
        assert_ne!(crc32("abc"), crc32("abd"));
    }

    #[test]
    fn noise_token_is_delimited() {
        assert!(image_type_has_noise_token("ORIGINAL_PRIMARY_NOISE_NONE"));
        assert!(image_type_has_noise_token("NOISE"));
        assert!(image_type_has_noise_token("FOO_NOISE"));
        assert!(!image_type_has_noise_token("NOISELESS"));
        assert!(!image_type_has_noise_token("XNOISE"));
        assert!(!image_type_has_noise_token(""));
    }
}

