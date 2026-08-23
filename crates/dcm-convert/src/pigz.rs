//! External `pigz` discovery + compress helpers (`readFindPigz` / `pigz_File` / piped gz).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use dcm_core::error::{Error, Result};
use dcm_nifti::Nifti1Header;

/// Locate a `pigz` binary (PATH + common install dirs), mirroring C++ `readFindPigz`.
pub fn find_pigz(argv0: Option<&str>) -> Option<PathBuf> {
    const NAMES: &[&str] = &["pigz", "pigz_mricron", "pigz_afni"];
    const PATHS: &[&str] = &[
        "/usr/local/bin/",
        "/usr/bin/",
        "/opt/homebrew/bin/",
    ];
    // PATH lookup
    for name in NAMES {
        if let Ok(p) = which(name) {
            return Some(p);
        }
    }
    // Fixed paths
    for base in PATHS {
        for name in NAMES {
            let p = PathBuf::from(format!("{base}{name}"));
            if is_exe(&p) {
                return Some(p);
            }
        }
    }
    // Next to the executable
    if let Some(a0) = argv0 {
        if let Some(dir) = Path::new(a0).parent() {
            for name in NAMES {
                let p = dir.join(name);
                if is_exe(&p) {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn which(name: &str) -> std::io::Result<PathBuf> {
    let out = Command::new("which").arg(name).output()?;
    if !out.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not found",
        ));
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "empty",
        ));
    }
    Ok(PathBuf::from(s))
}

fn is_exe(p: &Path) -> bool {
    p.is_file()
        && std::fs::metadata(p)
            .map(|m| {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    m.permissions().mode() & 0o111 != 0
                }
                #[cfg(not(unix))]
                {
                    true
                }
            })
            .unwrap_or(false)
}

/// Compress an existing `.nii` → `.nii.gz` with pigz (C++ `pigz_File`).
pub fn pigz_file(nii_path: &Path, pigz: &Path, gz_level: i32, imgsz: usize, verbose: i32) -> Result<()> {
    let mut cmd = Command::new(pigz);
    cmd.arg("--no-time").arg("-n").arg("-f");
    if imgsz > 1_000_000 {
        cmd.arg("-b").arg("960");
    }
    if (1..12).contains(&gz_level) {
        cmd.arg(format!("-{gz_level}"));
    }
    cmd.arg(nii_path);
    if verbose > 1 {
        eprintln!("Compress: {pigz:?} {nii_path:?}");
    }
    let status = cmd
        .status()
        .map_err(|e| Error::convert(format!("pigz spawn: {e}")))?;
    if !status.success() {
        return Err(Error::convert(format!(
            "External compression failed: {pigz:?} {nii_path:?}"
        )));
    }
    Ok(())
}

/// Write NIfTI header+voxels piped into pigz stdout → `.nii.gz` (C++ `-z o`).
pub fn write_nii_via_pigz_pipe(
    gz_path: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    pigz: &Path,
    gz_level: i32,
    verbose: i32,
) -> Result<()> {
    let mut cmd = Command::new(pigz);
    cmd.arg("--no-time").arg("-n").arg("-f");
    if (1..12).contains(&gz_level) {
        cmd.arg(format!("-{gz_level}"));
    }
    // Redirect pigz stdout to the .nii.gz file
    let out = std::fs::File::create(gz_path).map_err(|e| Error::io(gz_path, e))?;
    cmd.stdin(Stdio::piped()).stdout(out).stderr(Stdio::null());
    if verbose > 0 {
        eprintln!(" Optimal piped gz will fail if pigz version < 2.3.4.");
        eprintln!("Compress: {pigz:?} > {}", gz_path.display());
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| Error::convert(format!("Unable to open pigz pipe: {e}")))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| Error::convert("Unable to open pigz pipe"))?;
        let mut hdr = *hdr;
        hdr.vox_offset = 352.0;
        stdin
            .write_all(&hdr.as_bytes())
            .map_err(|e| Error::convert(format!("pigz pipe header: {e}")))?;
        stdin
            .write_all(&[0u8; 4])
            .map_err(|e| Error::convert(format!("pigz pipe ext: {e}")))?;
        stdin
            .write_all(voxels)
            .map_err(|e| Error::convert(format!("pigz pipe voxels: {e}")))?;
    }
    let status = child
        .wait()
        .map_err(|e| Error::convert(format!("pigz wait: {e}")))?;
    if !status.success() {
        let _ = std::fs::remove_file(gz_path);
        return Err(Error::convert(format!(
            "Unable to write {} via pigz pipe",
            gz_path.display()
        )));
    }
    Ok(())
}
