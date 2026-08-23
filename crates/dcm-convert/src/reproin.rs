//! ReproIn (DBIC heuristic) one-pass filename emulation for `-f %H`.
//!
//! Port of dcm2niix's `console/reproin.cpp` / `reproin.h`. The Python ground
//! truth lives in heudiconv's reproin heuristic (`parse_series_spec` /
//! `infotodict`); see `REPROIN.md` in the C++ tree for design notes and known
//! limitations (B0FieldIdentifier/IntendedFor etc. require a second pass).

use dcm_dicom::DicomImage;

use crate::opts::DcmOpts;

/// Native path separator used when building ReproIn paths (matches the C++
/// `kReproinPathSep`: `\` on Windows, `/` elsewhere).
const PATH_SEP: char = std::path::MAIN_SEPARATOR;

/// Parsed ReproIn protocol-name spec (`struct TReproinSpec`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReproinSpec {
    /// `anat | func | fmap | dwi | beh`.
    pub datatype: String,
    /// `T1w`, `bold`, `epi`, `magnitude{,1,2}`, `phasediff`, `sbref`, `scout`, ...
    pub suffix: String,
    pub ses: String,
    pub task: String,
    pub acq: String,
    pub rec: String,
    pub dir: String,
    /// Numeric token, kept literal; zero-padded at format time.
    pub run: String,
    /// Unrecognised key-value tokens, joined with `_`.
    pub bids: String,
    pub is_sbref: bool,
    pub is_derived: bool,
    pub valid: bool,
}

/// Known datatypes per the ReproIn spec (`KNOWN_DATATYPES` in the heuristic).
fn known_datatype(s: &str) -> bool {
    matches!(s, "anat" | "func" | "fmap" | "dwi" | "beh")
}

/// `_delete_chars(value, "#!@$%^&.,:;_-")` — sanitises task/acq/etc. values
/// encountered while tokenising. Extended beyond heudiconv's set with
/// path-traversal-relevant characters (slashes, backslashes, whitespace,
/// control chars) so DICOM/CLI values cannot inject directory structure into
/// a BIDS filename.
fn sanitize_value(s: &str) -> String {
    const ILLEGAL: &str = "#!@$%^&.,:;_-/\\ \t\r\n";
    s.chars()
        .filter(|c| (*c as u32) >= 0x20 && !ILLEGAL.contains(*c))
        .collect()
}

/// Strip raw path separators and control characters from a study-path
/// source. Called before [`pathify`] expands underscores/spaces into the
/// *intended* path separators, so any `/` or `\` surviving into the output
/// came from us.
fn drop_raw_path_chars(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '/' && *c != '\\' && (*c as u32) >= 0x20)
        .collect()
}

/// Public path-safety scrub for any BIDS label arriving from outside the
/// reproin parser (e.g. `-bi`/`-bv` values from the command line). Strips
/// path separators, whitespace, control chars, and `.` (so `.`/`..` segments
/// can't form a traversal). Preserves case and digits.
pub fn sanitize_label(s: &str) -> String {
    s.chars()
        .filter(|c| {
            (*c as u32) >= 0x20 && !matches!(c, '/' | '\\' | '.' | ' ' | '\t' | '\r' | '\n')
        })
        .collect()
}

/// Walk the path-separator-joined output and drop any `.` or `..` segments.
/// Defends against directory traversal when a DICOM StudyDescription
/// contains dots (e.g. `../escape` after pathify becomes `escape`).
fn reject_dot_segments(s: &str) -> String {
    s.split(PATH_SEP)
        .filter(|seg| !seg.is_empty() && *seg != "." && *seg != "..")
        .collect::<Vec<_>>()
        .join(&PATH_SEP.to_string())
}

/// Drop leading `[A-Z]+:` (vendor prefix) and `WIP ` (Philips).
fn strip_prefix(s: &str) -> String {
    let n = s.chars().take_while(|c| c.is_ascii_uppercase()).count();
    let s = if n > 0 && s.as_bytes().get(n) == Some(&b':') {
        &s[n + 1..]
    } else {
        s
    };
    s.strip_prefix("WIP ").unwrap_or(s).to_string()
}

/// Drop `__custom` trailing comment.
fn strip_custom(s: &str) -> String {
    match s.find("__") {
        Some(pos) => s[..pos].to_string(),
        None => s.to_string(),
    }
}

/// Apply the same fixups `parse_series_spec` applies before tokenising.
fn spec_fixups(s: &str) -> String {
    let mut s = s.to_string();
    // anat_T1w -> anat-T1w (only as a prefix, to avoid corrupting other tokens).
    if s.starts_with("anat_T1w") {
        s.replace_range(4..5, "-");
    }
    // hardi_64 -> dwi_acq-hardi64
    if let Some(pos) = s.find("hardi_64") {
        s.replace_range(pos..pos + 8, "dwi_acq-hardi64");
    }
    // AAHead_Scout -> anat-scout
    if let Some(pos) = s.find("AAHead_Scout") {
        s.replace_range(pos..pos + 12, "anat-scout");
    }
    // bare leading "scout" (token) -> "anat-scout"
    if s == "scout" || s.starts_with("scout_") {
        s.replace_range(0..5, "anat-scout");
    }
    s
}

/// Slot known entities, accumulate everything else into bids leftovers.
fn assign_token(spec: &mut ReproinSpec, raw: &str, bids: &mut String) {
    let Some((key, value)) = raw.split_once('-') else {
        // Unknown bareword: append to bids leftovers.
        if !bids.is_empty() {
            bids.push('_');
        }
        bids.push_str(raw);
        return;
    };
    match key {
        "ses" => spec.ses = sanitize_value(value),
        // Don't sanitise digits; preserve literal value (zero-padded at
        // format time). Still scrub path separators/control chars.
        "run" => spec.run = drop_raw_path_chars(value),
        "task" => spec.task = sanitize_value(value),
        "acq" => spec.acq = sanitize_value(value),
        "rec" => spec.rec = sanitize_value(value),
        "dir" => spec.dir = sanitize_value(value),
        _ => {
            // Unknown key-value: keep verbatim in bids leftovers, but scrub
            // path separators/control chars to keep the segment safe.
            let safe = drop_raw_path_chars(raw);
            if !bids.is_empty() {
                bids.push('_');
            }
            bids.push_str(&safe);
        }
    }
}

/// Core parser. Returns `Some(spec)` if `text` looks like a valid ReproIn spec.
fn parse_string(text: &str) -> Option<ReproinSpec> {
    if text.is_empty() {
        return None;
    }
    let mut buf = text.trim().to_string();
    buf = strip_prefix(&buf);
    buf = strip_custom(&buf);
    buf = spec_fixups(&buf);
    buf = buf.trim().to_string();
    if buf.is_empty() {
        return None;
    }
    // Tokenise on '_' (strtok_r merges consecutive delimiters into one).
    let mut tokens = buf.split('_').filter(|s| !s.is_empty());
    let first = tokens.next()?;
    // First token: <datatype>[-<suffix>]
    let (datatype, suffix) = match first.split_once('-') {
        Some((d, s)) => (d, s),
        None => (first, ""),
    };
    if !known_datatype(datatype) {
        return None;
    }
    let mut spec = ReproinSpec {
        datatype: datatype.to_string(),
        ..Default::default()
    };
    if !suffix.is_empty() {
        spec.suffix = suffix.to_string();
    }
    let mut bids_buf = String::new();
    for tok in tokens {
        assign_token(&mut spec, tok, &mut bids_buf);
    }
    spec.bids = bids_buf;
    spec.valid = true;
    Some(spec)
}

/// Pull `ImageType[idx]` (the IOD-specific specialisation slot is index 2:
/// M/P/FMRI/MPR/DIFFUSION). `image_type` is `_`-joined in dcm2niix (the
/// original backslash separators are normalised at DICOM-read time).
fn image_type_slot(image_type: &str, idx: usize) -> String {
    if image_type.is_empty() {
        return String::new();
    }
    image_type.split('_').nth(idx).unwrap_or("").to_string()
}

/// Parse ProtocolName (preferred) or SeriesDescription (fallback) into a
/// spec. Suffix inference (when ProtocolName lacks it) consults
/// `dcm.image_type` / `dcm.series_description`.
///
/// Siemens RF-off (`is_no_rf`) forces the BIDS `_noRF` suffix for `func`
/// (issue #1025), matching C++ `reproinParseSpec`.
pub fn parse_spec(dcm: &DicomImage) -> Option<ReproinSpec> {
    let mut spec =
        parse_string(&dcm.protocol_name).or_else(|| parse_string(&dcm.series_description))?;

    // SBRef override (matches heuristic: series_description.endswith("_SBRef")).
    if dcm.series_description.len() >= 6 && dcm.series_description.ends_with("_SBRef") {
        spec.suffix = "sbref".to_string();
        spec.is_sbref = true;
    }

    // Image-type-driven suffix inference (only if ProtocolName lacked one).
    let iod = image_type_slot(&dcm.image_type, 2);
    if spec.suffix.is_empty() {
        match spec.datatype.as_str() {
            "func" => {
                // "_pace_" anywhere in ProtocolName/SeriesDescription wins
                // over the imageType-driven default (heuristic.py:510-518).
                let is_pace = dcm.protocol_name.contains("_pace_")
                    || dcm.series_description.contains("_pace_");
                spec.suffix = if is_pace {
                    "pace".to_string()
                } else if iod == "P" {
                    "phase".to_string()
                } else {
                    "bold".to_string()
                };
            }
            "fmap" => {
                spec.suffix = if iod == "P" {
                    "phasediff".to_string()
                } else if iod == "DIFFUSION" {
                    "epi".to_string()
                } else if iod == "M" {
                    if !spec.dir.is_empty() {
                        "epi".to_string()
                    } else {
                        "magnitude".to_string()
                    }
                } else {
                    // Unknown IOD: assume magnitude as a reasonable default.
                    "magnitude".to_string()
                };
            }
            "dwi" => spec.suffix = "dwi".to_string(),
            _ => {}
        }
    }

    // RF-off (noise): force `_noRF` for func even on canonical ReproIn protocols.
    if dcm.is_no_rf && spec.datatype == "func" {
        spec.suffix = "noRF".to_string();
    }
    spec.is_derived = is_derived(dcm, Some(&spec));
    Some(spec)
}

/// Combined classifier: DICOM `is_derived` + textual heuristics on
/// SeriesDescription/ProtocolName + parsed datatype-suffix == anat-scout.
pub fn is_derived(dcm: &DicomImage, spec: Option<&ReproinSpec>) -> bool {
    if dcm.is_derived {
        return true;
    }
    if let Some(spec) = spec {
        if spec.datatype == "anat" && spec.suffix == "scout" {
            return true;
        }
    }
    // Inspect both SeriesDescription and ProtocolName so a marker present in
    // either field routes to derivatives/scanner.
    for s in [&dcm.series_description, &dcm.protocol_name] {
        if !s.is_empty()
            && (s.contains("-scout")
                || s.contains("_ADC")
                || s.contains("_TRACEW")
                || s.contains("_TRACE")
                || s.contains("_FA")
                || s.contains("AAHead_Scout"))
        {
            return true;
        }
    }
    false
}

/// Replace `' '`, `'_'` with path separators; drop `'-'`. Caret `'^'` also
/// becomes a path separator (PatientName style `Last^First`).
fn pathify(s: &str) -> String {
    let mut out = String::new();
    let mut has_caret = false;
    for c in s.chars() {
        if c == '^' {
            out.push(PATH_SEP);
            has_caret = true;
        } else if !has_caret && (c == '_' || c == ' ') {
            out.push(PATH_SEP);
        } else if c == '-' {
            // drop
        } else {
            out.push(c);
        }
    }
    out
}

/// Port of heudiconv's `fixup_subjectid` (`heuristic.py:975`):
/// ```text
/// subjectid = subjectid.lower()
/// reg = re.match(r"sid0*(\d+)$", subjectid)
/// if not reg:
///     return re.sub("[-_]", "", subjectid)
/// return "sid%06d" % int(reg.groups()[0])
/// ```
pub fn fixup_subject_id(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let lower = input.to_lowercase();
    // ^sid<digits>$ — Python regex `sid0*(\d+)$` accepts any non-empty digit
    // run after "sid", including all-zero strings (regex backtracking).
    if let Some(rest) = lower.strip_prefix("sid") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            let n: u64 = rest.parse().unwrap_or(0);
            return format!("sid{n:06}");
        }
    }
    // Otherwise strip '-' and '_' (heudiconv); sanitize_label below also
    // drops path separators, dots, whitespace, controls.
    let stripped: String = lower.chars().filter(|&c| c != '-' && c != '_').collect();
    sanitize_label(&stripped)
}

/// Resolve effective session value. Returns an empty string when no session
/// is determined from any source — caller should omit the `ses-` segment.
///
/// Precedence: `spec.ses` (parsed `_ses-<X>` from ProtocolName) wins if
/// present; otherwise `cli_session` (from `-bv`) is used. The literal
/// markers `{date}` (Python-friendly) and `DATE` (Siemens X60) resolve
/// against `dcm.study_date` (`YYYYMMDD`).
pub fn resolve_session(spec: Option<&ReproinSpec>, dcm: &DicomImage, cli_session: &str) -> String {
    let src = match spec.map(|s| s.ses.as_str()).filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None if !cli_session.is_empty() => cli_session.to_string(),
        None => return String::new(),
    };
    if src == "{date}" || src == "DATE" {
        return dcm.study_date.clone();
    }
    // Sanitise: -bv from CLI bypasses assign_token's sanitize_value. Use the
    // public label scrub (drops path separators, dots, whitespace, controls)
    // so '.' segments cannot reach the filename composer.
    sanitize_label(&src)
}

/// In-place (functional) path-safety scrub for the `-br` project-subdirectory
/// value. Runs the same pipeline as [`build_study_path`] on a CLI-supplied
/// string: drops raw path separators / control characters, expands `_`/` `
/// into path separators, and rejects `.`/`..` segments so `-br ../escape` or
/// `-br /tmp/other` cannot write outside the chosen `-o` directory.
pub fn sanitize_project_path(s: &str) -> String {
    let s = drop_raw_path_chars(s);
    let s = pathify(&s);
    reject_dot_segments(&s)
}

/// Build the study path component (e.g. `BrainHealth/AgingBrain`) from
/// StudyDescription, falling back to PerformedProcedureStepDescription.
/// Spaces and underscores become path separators. Empty if no source field
/// is set.
pub fn build_study_path(dcm: &DicomImage) -> String {
    let src = if !dcm.study_description.is_empty() {
        dcm.study_description.as_str()
    } else if !dcm.procedure_step_description.is_empty() {
        dcm.procedure_step_description.as_str()
    } else {
        return String::new();
    };
    sanitize_project_path(src)
}

/// Decide the concrete suffix when emitting fmap GRE multi-echo magnitudes.
/// `echo_num` is 1-based DICOM EchoNumbers; `is_multi_echo` indicates the
/// series has >1 echo. For phasediff (P) we always emit a single phasediff
/// regardless of echo. For magnitude we emit `magnitude<N>` when multi-echo,
/// else `magnitude`.
fn resolve_suffix_for_fmap(spec: &ReproinSpec, echo_num: i32, is_multi_echo: bool) -> String {
    if spec.suffix == "magnitude" && is_multi_echo && echo_num >= 1 {
        format!("magnitude{echo_num}")
    } else {
        spec.suffix.clone()
    }
}

/// Construct the BIDS-style basename (everything after the study path):
/// `<subDir>/<datatype>/<sub>_<ses>[_task-..][_acq-..][_dir-..][_run-NN][_echo-N]_<suffix>`.
/// `bids_subject`/`bids_session` are bare values without the `sub-`/`ses-`
/// prefix. `echo_num` is `dcm.echo_number`; pass `<=1` for single-echo.
/// `is_multi_echo` toggles `_echo-N` injection and magnitude1/magnitude2
/// selection for GRE fmaps. Returns `None` if the spec (or subject) is
/// invalid.
pub fn build_filename(
    spec: &ReproinSpec,
    bids_subject: &str,
    bids_session: &str,
    echo_num: i32,
    is_multi_echo: bool,
) -> Option<String> {
    if !spec.valid || spec.datatype.is_empty() {
        return None;
    }
    if bids_subject.is_empty() {
        return None; // caller must resolve subject (PatientID fallback handled there)
    }
    // Empty bids_session means: omit ses- segment entirely (heudiconv
    // behaviour for studies without an explicit _ses- marker).
    let has_ses = !bids_session.is_empty();

    let sub_seg = format!("sub-{bids_subject}");
    let ses_seg = if has_ses {
        format!("ses-{bids_session}")
    } else {
        String::new()
    };

    let head = if spec.is_derived {
        if has_ses {
            format!(
                "derivatives{PATH_SEP}scanner{PATH_SEP}{sub_seg}{PATH_SEP}{ses_seg}{PATH_SEP}{}{PATH_SEP}{sub_seg}_{ses_seg}",
                spec.datatype
            )
        } else {
            format!(
                "derivatives{PATH_SEP}scanner{PATH_SEP}{sub_seg}{PATH_SEP}{}{PATH_SEP}{sub_seg}",
                spec.datatype
            )
        }
    } else if has_ses {
        format!(
            "{sub_seg}{PATH_SEP}{ses_seg}{PATH_SEP}{}{PATH_SEP}{sub_seg}_{ses_seg}",
            spec.datatype
        )
    } else {
        format!("{sub_seg}{PATH_SEP}{}{PATH_SEP}{sub_seg}", spec.datatype)
    };

    let mut tail = String::new();
    // BIDS only allows _task- on func/sbref. Order: task, acq, rec, dir, bids, run, echo, suffix.
    let is_func = spec.datatype == "func";
    if is_func || spec.is_sbref {
        if !spec.task.is_empty() {
            tail.push_str(&format!("_task-{}", spec.task));
        } else if is_func {
            // BIDS requires _task- on func; default to "rest".
            tail.push_str("_task-rest");
        }
    }
    if !spec.acq.is_empty() {
        tail.push_str(&format!("_acq-{}", spec.acq));
    }
    if !spec.rec.is_empty() {
        tail.push_str(&format!("_rec-{}", spec.rec));
    }
    if !spec.dir.is_empty() {
        tail.push_str(&format!("_dir-{}", spec.dir));
    }
    if !spec.bids.is_empty() {
        tail.push_str(&format!("_{}", spec.bids));
    }
    if !spec.run.is_empty() {
        // Zero-pad numeric run to 2 digits; fall back to literal otherwise.
        match spec.run.parse::<i32>() {
            Ok(run_i) if run_i > 0 => tail.push_str(&format!("_run-{run_i:02}")),
            _ => tail.push_str(&format!("_run-{}", spec.run)),
        }
    }
    // Echo handling for func / dwi multi-echo. For fmap GRE multi-echo we
    // fold the echo into the suffix (magnitude1/magnitude2) rather than
    // emitting a separate _echo-N entity.
    let fold_echo_into_suffix = spec.datatype == "fmap";
    if is_multi_echo && echo_num >= 1 && !fold_echo_into_suffix {
        tail.push_str(&format!("_echo-{echo_num}"));
    }
    // Resolve the suffix (handles fmap magnitude/magnitudeN selection).
    let suffix = if fold_echo_into_suffix {
        resolve_suffix_for_fmap(spec, echo_num, is_multi_echo)
    } else {
        spec.suffix.clone()
    };
    if !suffix.is_empty() {
        tail.push_str(&format!("_{suffix}"));
    }
    Some(format!("{head}{tail}"))
}

/// Build the full `%H` path (study path + filename), mirroring
/// `nii_dicom_batch.cpp`'s `isReproin` branch: resolves the study
/// subdirectory (from `opts.bids_root` when `opts.is_bids_root`, else from
/// `dcm` via [`build_study_path`]), resolves subject/session, and calls
/// [`build_filename`]. Falls back to the legacy `Unknown/<series>_<protocol>`
/// basename when the protocol/series description does not parse as a
/// ReproIn spec.
///
/// Side effects: [`ensure_bids_boilerplate`] is invoked from `create_filename`
/// when `-f` contains `%H`.
pub fn expand_reproin_h(dcm: &DicomImage, opts: &DcmOpts) -> String {
    let spec = parse_spec(dcm);

    let study_pth = if opts.is_bids_root {
        sanitize_project_path(&opts.bids_root)
    } else {
        build_study_path(dcm)
    };

    let mut out = String::new();
    if !study_pth.is_empty() {
        out.push_str(&study_pth);
        out.push(PATH_SEP);
    }

    if let Some(spec) = &spec {
        let subject_val = if !opts.bids_subject.is_empty() {
            sanitize_label(&opts.bids_subject)
        } else {
            fixup_subject_id(&dcm.patient_id)
        };
        let session_val = resolve_session(Some(spec), dcm, &opts.bids_session);
        // dcm_dicom::DicomImage has no isMultiEcho flag (computed upstream in
        // dcm2niix from cross-file comparison during series grouping); approximate
        // with echo_number > 1, matching the `(d.echoNum > 1) || ...` fallback
        // condition used elsewhere in nii_dicom_batch.cpp.
        let is_multi_echo = dcm.echo_number > 1;
        if let Some(name) = build_filename(
            spec,
            &subject_val,
            &session_val,
            dcm.echo_number,
            is_multi_echo,
        ) {
            out.push_str(&name);
            return out;
        }
    }
    // Fallback: legacy "Unknown/<seriesNum>_<protocol>" basename.
    out.push_str("Unknown");
    out.push(PATH_SEP);
    out.push_str(&dcm.series_number.to_string());
    out.push('_');
    out.push_str(&dcm.protocol_name);
    out
}

/// C++ `createDummyBidsBoilerplate` — README.md, dataset_description.json,
/// and optional `task-*_bold.json` under the study root (idempotent).
pub fn ensure_bids_boilerplate(
    study_root: &std::path::Path,
    is_func: bool,
    task_name: &str,
) -> std::io::Result<()> {
    use std::fs;
    use std::io::Write;
    fs::create_dir_all(study_root)?;
    let readme = study_root.join("README.md");
    if !readme.exists() {
        let mut f = fs::File::create(&readme)?;
        writeln!(
            f,
            "Generated using dcm2niix ({})\n\nDescribe your dataset here. This file was generated by dcm2niix in a single pass. Details like IntendedFor, Subject ID, Session and tasks are not defined.",
            dcm_core::VERSION
        )?;
    }
    let desc = study_root.join("dataset_description.json");
    if !desc.exists() {
        let mut f = fs::File::create(&desc)?;
        f.write_all(
            b"{\n    \"Name\": \"dcm2niix dummy dataset\",\n    \"Authors\": [\"Chris Rorden\", \"Alex Teghipco\"],\n    \"BIDSVersion\": \"1.8.0\"\n}\n",
        )?;
    }
    if !is_func {
        return Ok(());
    }
    let task = if task_name.is_empty() {
        "rest"
    } else {
        task_name
    };
    let task_path = study_root.join(format!("task-{task}_bold.json"));
    if !task_path.exists() {
        let mut f = fs::File::create(&task_path)?;
        if task == "rest" {
            writeln!(
                f,
                "{{\n\"TaskName\": \"{task}\",\n\"CogAtlasID\": \"https://www.cognitiveatlas.org/task/id/trm_4c8a834779883/\"\n}}"
            )?;
        } else {
            writeln!(f, "{{\n\"TaskName\": \"{task}\"\n}}")?;
        }
    }
    Ok(())
}

fn tsv_field(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\t' | '\n' | '\r' => ' ',
            _ => c,
        })
        .collect()
}

/// C++ `reproinAppendProvenance` — append one row to `.reproin_provenance.tsv`.
pub fn append_provenance(
    study_root: &std::path::Path,
    out_stem: &std::path::Path,
    dcm: &DicomImage,
    anonymize_full: bool,
) -> std::io::Result<()> {
    use std::fs::{self, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};

    fs::create_dir_all(study_root)?;
    let tsv_path = study_root.join(".reproin_provenance.tsv");
    let bak_path = study_root.join(".reproin_provenance.tsv.bak");
    if anonymize_full {
        let _ = fs::remove_file(&bak_path);
    }

    let emit_demo = !anonymize_full;
    let expected_tabs = if emit_demo { 10 } else { 5 };

    if tsv_path.exists() {
        let mut peek = String::new();
        fs::File::open(&tsv_path)?.read_to_string(&mut peek)?;
        let hdr = peek.lines().next().unwrap_or("");
        let tabs = hdr.chars().filter(|&c| c == '\t').count();
        if tabs != expected_tabs {
            if anonymize_full {
                fs::remove_file(&tsv_path)?;
            } else {
                let _ = fs::remove_file(&bak_path);
                fs::rename(&tsv_path, &bak_path)?;
            }
        }
    }

    let mut tp = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&tsv_path)?;
    let empty = tp.seek(SeekFrom::End(0))? == 0;
    if empty {
        if emit_demo {
            writeln!(
                tp,
                "StudyInstanceUID\tSeriesNumber\tProtocolName\tSeriesDescription\tStudyDescription\tOutputStem\tPatientAge\tPatientSex\tStudyDate\tStudyTime\tPatientID"
            )?;
        } else {
            writeln!(
                tp,
                "StudyInstanceUID\tSeriesNumber\tProtocolName\tSeriesDescription\tStudyDescription\tOutputStem"
            )?;
        }
    }

    let root = study_root.to_string_lossy();
    let stem_s = out_stem.to_string_lossy();
    let rel = if let Some(r) = stem_s.strip_prefix(root.as_ref()) {
        r.trim_start_matches(['/', '\\'])
    } else {
        out_stem
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(stem_s.as_ref())
    };

    let f1 = tsv_field(&dcm.study_uid);
    let f2 = tsv_field(&dcm.protocol_name);
    let f3 = tsv_field(&dcm.series_description);
    let f4 = tsv_field(&dcm.study_description);
    let f5 = tsv_field(rel);
    if emit_demo {
        let sex = dcm
            .patient_sex
            .chars()
            .next()
            .filter(|c| matches!(c, 'M' | 'F' | 'O'))
            .map(|c| c.to_string())
            .unwrap_or_default();
        writeln!(
            tp,
            "{f1}\t{}\t{f2}\t{f3}\t{f4}\t{f5}\t{}\t{sex}\t{}\t{}\t{}",
            dcm.series_number,
            tsv_field(&dcm.patient_age),
            tsv_field(&dcm.study_date),
            tsv_field(&dcm.study_time),
            tsv_field(&dcm.patient_id),
        )?;
    } else {
        writeln!(
            tp,
            "{f1}\t{}\t{f2}\t{f3}\t{f4}\t{f5}",
            dcm.series_number
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcm_dicom::{CsaMeta, Manufacturer, Modality};
    use std::path::PathBuf;

    fn minimal() -> DicomImage {
        DicomImage {
            path: PathBuf::from("x"),
            series_uid: String::new(),
            series_uid_crc: 0,
            instance_uid: String::new(),
            study_uid: String::new(),
            series_number: 7,
            instance_number: 0,
            acquisition_number: 0,
            echo_number: 1,
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

    #[test]
    fn sanitize_label_strips_path_and_control_chars_but_keeps_case_and_digits() {
        assert_eq!(sanitize_label("Aging01"), "Aging01");
        assert_eq!(sanitize_label("../escape"), "escape");
        assert_eq!(sanitize_label("a/b\\c.d e\tf"), "abcdef");
        assert_eq!(sanitize_label(""), "");
        // '-' and '_' are preserved (unlike the internal parse-time scrub).
        assert_eq!(sanitize_label("foo-bar_baz"), "foo-bar_baz");
    }

    #[test]
    fn fixup_subject_id_normalises_sid_pattern() {
        assert_eq!(fixup_subject_id("SID000123"), "sid000123");
        assert_eq!(fixup_subject_id("sid7"), "sid000007");
        assert_eq!(fixup_subject_id("sid0000"), "sid000000");
    }

    #[test]
    fn fixup_subject_id_strips_dashes_and_underscores_otherwise() {
        assert_eq!(fixup_subject_id("AB-CD_12"), "abcd12");
        assert_eq!(fixup_subject_id("sidewalk"), "sidewalk");
        assert_eq!(fixup_subject_id(""), "");
    }

    #[test]
    fn parse_spec_handles_synthetic_func_bold_protocol() {
        let dcm = DicomImage {
            protocol_name: "func_task-rest_run-01_bold".into(),
            ..minimal()
        };
        let spec = parse_spec(&dcm).expect("should parse as a valid reproin spec");
        assert_eq!(spec.datatype, "func");
        assert_eq!(spec.task, "rest");
        assert_eq!(spec.run, "01");
        // "bold" has no '-' so it isn't a recognised entity token; per the
        // ReproIn grammar (REPROIN.md) it flows into the `bids` leftover
        // bucket, while the suffix is independently inferred as "bold" (the
        // `func` default) since the first token ("func") carried no
        // explicit `-<suffix>`.
        assert_eq!(spec.bids, "bold");
        assert_eq!(spec.suffix, "bold");
        assert!(spec.valid);
        assert!(!spec.is_derived);
    }

    #[test]
    fn build_filename_zero_pads_run_and_orders_entities() {
        let spec = ReproinSpec {
            datatype: "func".into(),
            suffix: "bold".into(),
            task: "rest".into(),
            run: "1".into(),
            valid: true,
            ..Default::default()
        };
        let name = build_filename(&spec, "01", "", 1, false).unwrap();
        assert_eq!(
            name,
            format!(
                "sub-01{sep}func{sep}sub-01_task-rest_run-01_bold",
                sep = std::path::MAIN_SEPARATOR
            )
        );
    }

    #[test]
    fn parse_spec_infers_suffix_from_image_type_when_absent() {
        // No explicit suffix in the protocol name; imageType slot 2 == "P" -> phase.
        let dcm = DicomImage {
            protocol_name: "func_task-rest".into(),
            image_type: "ORIGINAL_PRIMARY_P_ND".into(),
            ..minimal()
        };
        let spec = parse_spec(&dcm).unwrap();
        assert_eq!(spec.suffix, "phase");
    }

    #[test]
    fn parse_spec_returns_none_for_unrecognised_protocol() {
        let dcm = DicomImage {
            protocol_name: "not_a_reproin_protocol".into(),
            series_description: "also not reproin".into(),
            ..minimal()
        };
        assert!(parse_spec(&dcm).is_none());
    }

    #[test]
    fn expand_reproin_h_builds_study_and_filename_path() {
        let dcm = DicomImage {
            protocol_name: "func_task-rest_run-01".into(),
            study_description: "BrainHealth_AgingBrain".into(),
            patient_id: "sid000123".into(),
            ..minimal()
        };
        let opts = DcmOpts::default();
        let sep = std::path::MAIN_SEPARATOR;
        let expected = format!(
            "BrainHealth{sep}AgingBrain{sep}sub-sid000123{sep}func{sep}sub-sid000123_task-rest_run-01_bold"
        );
        assert_eq!(expand_reproin_h(&dcm, &opts), expected);
    }

    #[test]
    fn expand_reproin_h_falls_back_to_unknown_when_unparseable() {
        let dcm = DicomImage {
            protocol_name: "totally_unrecognisable".into(),
            series_number: 42,
            ..minimal()
        };
        let opts = DcmOpts::default();
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(
            expand_reproin_h(&dcm, &opts),
            format!("Unknown{sep}42_totally_unrecognisable")
        );
    }

    #[test]
    fn build_study_path_prefers_study_description_and_converts_separators() {
        let sep = std::path::MAIN_SEPARATOR;
        let dcm = DicomImage {
            study_description: "BrainHealth AgingBrain".into(),
            ..minimal()
        };
        assert_eq!(
            build_study_path(&dcm),
            format!("BrainHealth{sep}AgingBrain")
        );
    }

    #[test]
    fn build_study_path_falls_back_to_procedure_step_description() {
        let sep = std::path::MAIN_SEPARATOR;
        let dcm = DicomImage {
            procedure_step_description: "Some Procedure".into(),
            ..minimal()
        };
        assert_eq!(build_study_path(&dcm), format!("Some{sep}Procedure"));
    }

    #[test]
    fn sanitize_project_path_rejects_dot_segments_and_neutralises_raw_slashes() {
        let sep = std::path::MAIN_SEPARATOR;
        // A ".." segment produced by underscore-expansion is dropped.
        assert_eq!(sanitize_project_path("foo_.._bar"), format!("foo{sep}bar"));
        // Raw '/' from a hostile CLI value is stripped *before* pathify
        // runs, so it can never introduce a path separator itself — the
        // string becomes an inert, single-segment label instead of climbing
        // out of the output directory.
        assert_eq!(sanitize_project_path("../escape"), "..escape");
    }
}
