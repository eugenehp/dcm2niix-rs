//! CLI options mirroring upstream `TDCMopts`.
//!
//! Populated by `dcm-cli` from argv; library users can construct [`DcmOpts`]
//! directly. Defaults match `dcm2niix` where practical (`-ba y`, Y-flip on, …).

use std::path::Path;

/// Whether to write BIDS JSON sidecars (`-b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidsMode {
    /// Write `.json` alongside NIfTI (`-b y`, default).
    Yes,
    /// Skip sidecars (`-b n`).
    No,
    /// Sidecars only — no NIfTI (`-b o`).
    Only,
}

/// BIDS anonymisation (`-ba`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymizeBids {
    /// Strip dates and patient PII (default).
    Yes,
    /// Keep dates and PII.
    No,
    /// Strip PII, keep acquisition timestamps (`-ba o`).
    PiiOnly,
}

/// Series stacking policy (`-m`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackMode {
    No,
    Yes,
    /// Force stack ignoring Series Instance UID (`-m 2`).
    ForceIgnoreUid,
    Auto,
}

/// Output name conflict policy (`-w`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameConflict {
    Skip = 0,
    Overwrite = 1,
    AddSuffix = 2,
}

/// Voxel compression / 3D save mode (`-z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compress {
    None,
    Gz,
    InternalGz,
    /// Save as 3D volumes (`-z 3`).
    Save3d,
    /// Zstandard `.nii.zst` (`-z s`).
    Zstd,
}

/// 16-bit intensity remapping (`-y` / maximize range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maximize16Bit {
    False,
    True,
    Raw,
}

/// Foreign / alternate image format (`-e`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveFormat {
    Nifti,
    Nrrd,
    Mgh,
    Jnii,
    Bnii,
}

/// Full conversion options — see field docs and `dcm2niix -h`.
#[derive(Debug, Clone)]
pub struct DcmOpts {
    pub filename: String,
    pub outdir: String,
    pub indir: String,
    pub indir_parent: String,
    pub image_comments: String,
    pub bids_subject: String,
    pub bids_session: String,
    pub bids_root: String,
    pub is_bids_root: bool,
    pub single_file: bool,
    pub rename_not_convert: bool,
    pub one_dir_at_a_time: bool,
    pub stack: StackMode,
    pub force_stack_dce: bool,
    pub ignore_derived: bool,
    pub keep_direction_varies: bool,
    pub crop: bool,
    pub rotate_3d: bool,
    pub compress: Compress,
    pub gz_level: i32,
    pub dir_search_depth: u32,
    pub search_only: i32,
    pub verbose: i32,
    pub name_conflict: NameConflict,
    pub flip_y: bool,
    pub bids: BidsMode,
    pub anonymize: AnonymizeBids,
    pub series_filter: Vec<f64>,
    /// Philips precise float scaling (`-p y`, default true).
    pub philips_precise: bool,
    pub ignore_intensity_scaling: bool,
    pub maximize_16bit: Maximize16Bit,
    pub save_format: SaveFormat,
    pub ignore_trigger_times: bool,
    pub add_name_postfixes: bool,
    /// Progress reporting: 0=off, 1=series, 2=detailed (C++ `isProgress`).
    pub progress: i32,
    pub create_text: bool,
    pub diff_cycling_mode_ge: i32,
    pub save_native_endian: bool,
    /// Correct CT gantry tilt by resampling (`isTiltCorrect`, default true).
    pub tilt_correct: bool,
    /// Persist defaults to `~/.dcm2nii.ini` after parse (`-g y/o`).
    pub save_ini: bool,
    /// Undocumented `-j y`: compare GE slice timing vs `(0021,105E)`.
    pub test_x0021x105e: bool,
    /// Reorder DTI volumes by ascending b-value (C++ `isSortDTIbyBVal`).
    pub sort_dti_by_bval: bool,
    /// Save RGB as planar RRR…GGG…BBB… (Analyze); default packed RGBRGB… (NIfTI).
    pub rgb_planar: bool,
    /// Force volume onset / FrameTimesStart computation (C++ `isForceOnsetTimes`).
    pub force_onset_times: bool,
    /// Path to external `pigz` (empty = internal zlib / flate2).
    pub pigz_path: String,
    /// Pipe uncompressed NIfTI into pigz without a temp `.nii` (`-z o`).
    pub piped_gz: bool,
    /// Emit `BidsGuess` / honour `-f %h` (C++ `isGuessBidsFilename`, default true).
    pub guess_bids_filename: bool,
    /// `-n <0>`: report series CRC + path, do not convert.
    pub report_series_only: bool,
}

impl Default for DcmOpts {
    fn default() -> Self {
        Self {
            filename: "%f_%p_%t_%s".into(),
            outdir: String::new(),
            indir: String::new(),
            indir_parent: String::new(),
            image_comments: String::new(),
            bids_subject: String::new(),
            bids_session: String::new(),
            bids_root: String::new(),
            is_bids_root: false,
            single_file: false,
            rename_not_convert: false,
            one_dir_at_a_time: false,
            stack: StackMode::Auto,
            force_stack_dce: true,
            ignore_derived: false,
            keep_direction_varies: false,
            crop: false,
            rotate_3d: true,
            compress: Compress::None,
            // Fast gzip default when `-z i` / internal fallback; pigz uses its own level.
            gz_level: 1,
            dir_search_depth: 5,
            search_only: 0,
            verbose: 0,
            name_conflict: NameConflict::AddSuffix,
            flip_y: true,
            bids: BidsMode::Yes,
            anonymize: AnonymizeBids::Yes,
            series_filter: Vec::new(),
            philips_precise: true,
            ignore_intensity_scaling: false,
            maximize_16bit: Maximize16Bit::False,
            save_format: SaveFormat::Nifti,
            ignore_trigger_times: false,
            add_name_postfixes: true,
            progress: 0,
            create_text: false,
            diff_cycling_mode_ge: -1,
            save_native_endian: true,
            tilt_correct: true,
            save_ini: false,
            test_x0021x105e: false,
            sort_dti_by_bval: false,
            rgb_planar: false,
            force_onset_times: true,
            pigz_path: String::new(),
            piped_gz: false,
            guess_bids_filename: true,
            report_series_only: false,
        }
    }
}

impl DcmOpts {
    pub fn set_indir(&mut self, p: &str) {
        self.indir = p.to_string();
        self.indir_parent = folder_name(Path::new(p));
    }
}

pub fn folder_name(p: &Path) -> String {
    let p = if p.is_file() {
        p.parent().unwrap_or(p)
    } else {
        p
    };
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "dicom".into())
}
