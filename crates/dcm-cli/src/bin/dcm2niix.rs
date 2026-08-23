//! `dcm2niix` — Rust/rlx DICOM → NIfTI converter.
//!
//! # Features
//!
//! - **default** — pure Rust conversion path
//! - **`gpu`** — optional wgpu realize for large volume flips
//! - **`ffi`** — also builds `dcm2niix-ffi` (upstream C++ parity reference)
//!
//! Broader BIDS dataset tooling lives in sibling `bids-rs`.

use std::env;
use std::process::ExitCode;

use dcm_convert::convert;
use dcm_convert::opts::{
    AnonymizeBids, BidsMode, Compress, DcmOpts, Maximize16Bit, NameConflict, SaveFormat, StackMode,
};
use dcm_core::exit::Exit;

fn main() -> ExitCode {
    ExitCode::from(run(env::args()) as u8)
}

fn run<I, S>(args: I) -> i32
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let collected: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
    if collected.len() < 2 || collected.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return if collected.len() < 2 {
            Exit::Failure.code() as i32
        } else {
            Exit::Success.code() as i32
        };
    }
    if collected
        .iter()
        .any(|a| a == "--version" || (a == "-v" && collected.len() == 2))
    {
        println!("Chris Rorden's dcm2niiX  {}", dcm_core::VERSION);
        return Exit::ReportVersion.code() as i32;
    }
    let opts = match parse_args(&collected) {
        Ok(o) => o,
        Err(msg) if msg == "__xml__" => return Exit::Success.code() as i32,
        Err(msg) if msg.starts_with("__ok__:") => {
            println!("{}", msg.trim_start_matches("__ok__:"));
            return Exit::Success.code() as i32;
        }
        Err(msg) if msg.starts_with("__fail__:") => {
            eprintln!("{}", msg.trim_start_matches("__fail__:"));
            return Exit::Failure.code() as i32;
        }
        Err(msg) => {
            eprintln!("{msg}");
            return Exit::Failure.code() as i32;
        }
    };
    if opts.indir.is_empty() {
        eprintln!("usage: dcm2niix [options] <in_folder>");
        return Exit::Failure.code() as i32;
    }
    match convert(&opts) {
        Ok(r) => r.exit.code() as i32,
        Err(e) => {
            eprintln!("{e}");
            Exit::Failure.code() as i32
        }
    }
}

fn yn(s: &str) -> Option<bool> {
    match s.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('y') | Some('1') => Some(true),
        Some('n') | Some('0') => Some(false),
        _ => None,
    }
}

fn parse_args(args: &[String]) -> Result<DcmOpts, String> {
    let mut opts = DcmOpts::default();
    dcm_convert::ini::read_ini(&mut opts);
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--big-endian" {
            i += 1;
            if i < args.len() {
                opts.save_native_endian = yn(&args[i]) != Some(false);
            }
        } else if a == "--ignore_trigger_times" {
            opts.ignore_trigger_times = true;
        } else if a == "--terse" {
            opts.add_name_postfixes = false;
        } else if a == "--progress" {
            opts.progress = 1;
            if i + 1 < args.len() {
                let n = &args[i + 1];
                match n.chars().next().map(|c| c.to_ascii_lowercase()) {
                    Some('n') | Some('0') => {
                        opts.progress = 0;
                        i += 1;
                    }
                    Some('y') | Some('1') => {
                        opts.progress = 1;
                        i += 1;
                    }
                    Some('2') => {
                        opts.progress = 2;
                        i += 1;
                    }
                    _ => {}
                }
            }
        } else if a == "--xml" {
            print_xml();
            return Err("__xml__".into());
        } else if a == "--diffCyclingModeGE" {
            i += 1;
            if i < args.len() {
                opts.diff_cycling_mode_ge = args[i].parse().unwrap_or(-1);
            }
        } else if a == "--version" {
            // handled above
        } else if let Some(rest) = a.strip_prefix('-') {
            if rest.len() == 1 && rest.chars().next().unwrap().is_ascii_digit() {
                opts.gz_level = rest.parse().unwrap_or(1);
            } else if rest.len() >= 1 {
                let flag = rest.chars().next().unwrap();
                match flag {
                    'a' => {
                        i += 1;
                        if i < args.len() {
                            opts.one_dir_at_a_time = yn(&args[i]).unwrap_or(false);
                        }
                    }
                    'b' => {
                        // -b, -ba, -bi, -bv, -br
                        if rest == "ba" {
                            i += 1;
                            if i < args.len() {
                                opts.anonymize = match args[i].chars().next().map(|c| c.to_ascii_lowercase())
                                {
                                    Some('n') | Some('0') => AnonymizeBids::No,
                                    Some('o') => AnonymizeBids::PiiOnly,
                                    _ => AnonymizeBids::Yes,
                                };
                            }
                        } else if rest == "bi" {
                            i += 1;
                            if i < args.len() {
                                opts.bids_subject = args[i].clone();
                            }
                        } else if rest == "bv" {
                            i += 1;
                            if i < args.len() {
                                opts.bids_session = args[i].clone();
                            }
                        } else if rest == "br" {
                            i += 1;
                            if i < args.len() {
                                opts.is_bids_root = true;
                                opts.bids_root = if args[i] == "." {
                                    String::new()
                                } else {
                                    args[i].clone()
                                };
                            }
                        } else {
                            i += 1;
                            if i < args.len() {
                                opts.bids = match args[i].chars().next().map(|c| c.to_ascii_lowercase())
                                {
                                    Some('n') | Some('0') => BidsMode::No,
                                    Some('o') => BidsMode::Only,
                                    _ => BidsMode::Yes,
                                };
                            }
                        }
                    }
                    'c' => {
                        i += 1;
                        if i < args.len() {
                            opts.image_comments = if args[i].is_empty() {
                                "\t".into()
                            } else {
                                args[i].chars().take(24).collect()
                            };
                        }
                    }
                    'd' => {
                        i += 1;
                        if i < args.len() {
                            opts.dir_search_depth = args[i].parse().unwrap_or(5);
                        }
                    }
                    'e' => {
                        i += 1;
                        if i < args.len() {
                            opts.save_format = match args[i].chars().next().map(|c| c.to_ascii_lowercase())
                            {
                                Some('y') | Some('1') => SaveFormat::Nrrd,
                                Some('o') | Some('2') => SaveFormat::Mgh,
                                Some('j') | Some('3') => SaveFormat::Jnii,
                                Some('b') | Some('4') => SaveFormat::Bnii,
                                _ => SaveFormat::Nifti,
                            };
                        }
                    }
                    'f' => {
                        i += 1;
                        if i < args.len() {
                            opts.filename = args[i].clone();
                        }
                    }
                    'i' => {
                        i += 1;
                        if i < args.len() {
                            match args[i].chars().next().map(|c| c.to_ascii_lowercase()) {
                                Some('o') => {
                                    opts.keep_direction_varies = true;
                                    opts.ignore_derived = false;
                                }
                                Some('n') | Some('0') => opts.ignore_derived = false,
                                _ => opts.ignore_derived = true,
                            }
                        }
                    }
                    'j' => {
                        i += 1;
                        if i < args.len()
                            && matches!(
                                args[i].chars().next().map(|c| c.to_ascii_lowercase()),
                                Some('y') | Some('1')
                            )
                        {
                            opts.test_x0021x105e = true;
                            println!(
                                "undocumented '-j y' compares GE slice timing from 0021,105E"
                            );
                        }
                    }
                    'l' => {
                        i += 1;
                        if i < args.len() {
                            opts.maximize_16bit = match args[i]
                                .chars()
                                .next()
                                .map(|c| c.to_ascii_lowercase())
                            {
                                Some('o') => Maximize16Bit::Raw,
                                Some('n') | Some('0') => Maximize16Bit::False,
                                _ => Maximize16Bit::True,
                            };
                        }
                    }
                    'm' => {
                        i += 1;
                        if i < args.len() {
                            match args[i].as_str() {
                                "n" | "N" | "0" => opts.stack = StackMode::No,
                                "y" | "Y" | "1" => opts.stack = StackMode::Yes,
                                "2" => opts.stack = StackMode::ForceIgnoreUid,
                                "o" | "O" => opts.force_stack_dce = false,
                                _ => {}
                            }
                        }
                    }
                    'n' => {
                        i += 1;
                        if i < args.len() {
                            if let Ok(v) = args[i].parse::<f64>() {
                                if v < 0.0 {
                                    opts.report_series_only = true;
                                    opts.series_filter.clear();
                                } else {
                                    opts.series_filter.push(v);
                                }
                            }
                        }
                    }
                    'o' => {
                        i += 1;
                        if i < args.len() {
                            opts.outdir = args[i].clone();
                        }
                    }
                    'p' => {
                        i += 1;
                        if i < args.len() {
                            match args[i].chars().next().map(|c| c.to_ascii_lowercase()) {
                                Some('o') => opts.ignore_intensity_scaling = true,
                                Some('n') | Some('0') => opts.philips_precise = false,
                                _ => opts.philips_precise = true,
                            }
                        }
                    }
                    'q' => {
                        i += 1;
                        if i < args.len() {
                            opts.search_only = match args[i]
                                .chars()
                                .next()
                                .map(|c| c.to_ascii_lowercase())
                            {
                                Some('y') => 1,
                                Some('l') => 2,
                                _ => 0,
                            };
                        }
                    }
                    'g' => {
                        i += 1;
                        if i < args.len() {
                            match args[i].chars().next().map(|c| c.to_ascii_lowercase()) {
                                Some('y') | Some('1') => opts.save_ini = true,
                                Some('o') => {
                                    opts = DcmOpts::default();
                                    opts.save_ini = true;
                                    eprintln!("Defaults reset");
                                    i = 0; // re-read (C++ behaviour)
                                }
                                Some('i') => {
                                    opts = DcmOpts::default();
                                    eprintln!("Defaults ignored");
                                    i = 0;
                                }
                                _ => {}
                            }
                        }
                    }
                    'u' => {
                        return Err(check_up_to_date());
                    }
                    'r' => {
                        i += 1;
                        if i < args.len() {
                            opts.rename_not_convert = yn(&args[i]).unwrap_or(false);
                        }
                    }
                    's' => {
                        i += 1;
                        if i < args.len() {
                            opts.single_file = yn(&args[i]) != Some(false);
                        }
                    }
                    't' => {
                        i += 1;
                        if i < args.len() {
                            opts.create_text = yn(&args[i]).unwrap_or(false);
                        }
                    }
                    'v' => {
                        if i + 1 < args.len()
                            && !args[i + 1].starts_with('-')
                            && args[i + 1]
                                .chars()
                                .next()
                                .map(|c| c.is_ascii_alphanumeric())
                                .unwrap_or(false)
                        {
                            i += 1;
                            opts.verbose = match args[i].chars().next().map(|c| c.to_ascii_lowercase())
                            {
                                Some('n') | Some('0') => 0,
                                Some('h') | Some('2') => 2,
                                _ => 1,
                            };
                        } else {
                            opts.verbose += 1;
                        }
                    }
                    'w' => {
                        i += 1;
                        if i < args.len() {
                            opts.name_conflict = match args[i].chars().next() {
                                Some('0') => NameConflict::Skip,
                                Some('1') => NameConflict::Overwrite,
                                _ => NameConflict::AddSuffix,
                            };
                        }
                    }
                    'x' => {
                        i += 1;
                        if i < args.len() {
                            match args[i].chars().next().map(|c| c.to_ascii_lowercase()) {
                                Some('n') | Some('0') => opts.crop = false,
                                Some('i') => {
                                    opts.rotate_3d = false;
                                    opts.crop = false;
                                }
                                _ => opts.crop = true,
                            }
                        }
                    }
                    'y' => {
                        i += 1;
                        if i < args.len() {
                            opts.flip_y = yn(&args[i]).unwrap_or(true);
                        }
                    }
                    'z' => {
                        i += 1;
                        if i < args.len() {
                            opts.compress = match args[i].chars().next().map(|c| c.to_ascii_lowercase())
                            {
                                Some('i') => Compress::InternalGz,
                                Some('y') | Some('g') => Compress::Gz,
                                Some('o') => {
                                    opts.piped_gz = true;
                                    Compress::Gz
                                }
                                Some('3') => Compress::Save3d,
                                Some('s') => Compress::Zstd,
                                _ => Compress::None,
                            };
                        }
                    }
                    'h' => print_help(),
                    _ => {
                        // unknown short flag: consume optional arg if present
                    }
                }
            }
        } else {
            opts.set_indir(a);
        }
        i += 1;
    }
    if opts.save_ini {
        let _ = dcm_convert::ini::save_ini(&opts);
    }
    // Prefer external pigz for `-z y/g/o` (C++ `readFindPigz`); `-z i` forces internal.
    // When pigz is available, use the piped path by default (faster than write-then-pigz).
    if matches!(opts.compress, Compress::Gz) && opts.pigz_path.is_empty() {
        if let Some(p) = dcm_convert::pigz::find_pigz(args.first().map(|s| s.as_str())) {
            opts.pigz_path = p.to_string_lossy().into_owned();
            if !opts.piped_gz {
                opts.piped_gz = true;
            }
        } else if opts.piped_gz {
            eprintln!("Warning: pigz not found; falling back to internal gzip for -z o");
            opts.piped_gz = false;
        }
    }
    Ok(opts)
}

fn check_up_to_date() -> String {
    // Best-effort: compare local VERSION_DATE stamp to GitHub latest release tag.
    let local = dcm_core::VERSION_DATE;
    match std::process::Command::new("curl")
        .args([
            "-fsSL",
            "https://api.github.com/repos/rordenlab/dcm2niix/releases/latest",
        ])
        .output()
    {
        Ok(out) if out.status.success() => {
            let body = String::from_utf8_lossy(&out.stdout);
            if let Some(tag) = body
                .split("\"tag_name\"")
                .nth(1)
                .and_then(|s| s.split('"').nth(1))
            {
                if body.contains(local) || tag.contains(&local.replace("v1.0.", "")) {
                    return format!("__ok__:Good news: Your version is up to date: {local}");
                }
                return format!(
                    "__fail__:Error: your version ('{local}') is not the latest release ('{tag}')\n https://github.com/rordenlab/dcm2niix/releases"
                );
            }
            format!("__ok__:Unable to parse latest release; local version {local}")
        }
        _ => format!(
            "__fail__:Error: unable to check version (network/curl)\n local: {local}"
        ),
    }
}

fn print_xml() {
    println!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<executable>
<title>dcm2niix</title>
<description>DICOM importer</description>
  <parameters>
    At least one parameter
  </parameters>
</executable>"#
    );
}

fn print_help() {
    eprintln!(
        "Chris Rorden's dcm2niiX  {}\n\
Usage: dcm2niix [options] <in_folder>\n\
 Options:\n\
  -1..-9   : gz compression level (1=fastest .. 9=smallest)\n\
  -a y/n   : one directory at a time\n\
  -b y/n/o : BIDS sidecar y=yes, n=no, o=only\n\
  -ba y/n/o: anonymize BIDS\n\
  -bi <id> : BIDS subject id (no sub- prefix)\n\
  -bv <id> : BIDS session id (no ses- prefix)\n\
  -br <p>  : BIDS project root / ReproIn study path\n\
  -c <txt> : image comments (max 24)\n\
  -d <n>   : directory search depth\n\
  -e y/o/j/b : export NRRD / MGH / JNIfTI / BNIfTI\n\
  -f <fmt> : filename (%%p %%s %%t %%h %%H …)\n\
  -g y/n/o/i : write/ignore/reset ~/.dcm2nii.ini defaults\n\
  -h       : help\n\
  -i y/n/o : ignore derived/2D; o=keep varying directions\n\
  -l y/n/o : maximize 16-bit range\n\
  -m y/n/2/o: merge series / ignore UID / no DCE stack\n\
  -n <crc> : convert only series with this CRC (<0 = list series)\n\
  -o <dir> : output directory\n\
  -p y/n/o : Philips precise scaling\n\
  -q y/l   : search only (list counts / paths)\n\
  -r y/n   : rename instead of convert\n\
  -s y/n   : convert single file\n\
  -t y/n   : text private-tag dump\n\
  -u       : up-to-date check\n\
  -v n/y/h : verbosity\n\
  -w 0/1/2 : name conflict skip/overwrite/suffix\n\
  -x y/n/i : crop / no-ortho\n\
  -y y/n   : flip Y\n\
  -z y/i/n/o/g/3/s : gzip (y=pigz piped if found, i=internal, o=piped pigz, s=zstd, 3=3D)\n\
  --progress [y/n/2] : report conversion progress\n\
  --xml    : Slicer format features\n\
  --version\n\
BIDS JSON expansion: see ../bids-rs\n",
        dcm_core::VERSION
    );
}
