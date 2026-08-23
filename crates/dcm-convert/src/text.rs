//! Text dump (`nii_saveText`) and DICOM rename (`-r y`).

use std::fs;
use std::path::{Path, PathBuf};

use dcm_core::error::{Error, Result};
use dcm_core::VERSION;
use dcm_dicom::DicomImage;
use dcm_nifti::Nifti1Header;

use crate::filename::create_filename;
use crate::opts::DcmOpts;

pub fn save_text(
    stem: &Path,
    d: &DicomImage,
    hdr: &Nifti1Header,
    dcm_name: &Path,
) -> Result<PathBuf> {
    let path = stem.with_extension("txt");
    let coil = crc32fast::hash(d.coil_name.as_bytes());
    let line = format!(
        "{}\tField Strength:\t{}\tProtocolName:\t{}\tScanningSequence00180020:\t{}\tTE:\t{}\tTR:\t{}\tSeriesNum:\t{}\tAcquNum:\t{}\tImageNum:\t{}\tImageComments:\t{}\tDateTime:\t{}\tName:\t{}\tConvVers:\t{}\tDoB:\t{}\tGender:\t{}\tAge:\t{}\tDimXYZT:\t{}\t{}\t{}\t{}\tCoil:\t{}\tEchoNum:\t{}\tOrient(6)\t{}\t{}\t{}\t{}\t{}\t{}\tbitsAllocated\t{}\tInputName\t{}\n",
        stem.display(),
        d.field_strength,
        d.protocol_name,
        d.scanning_sequence,
        d.te,
        d.tr,
        d.series_number,
        d.acquisition_number,
        d.instance_number,
        d.image_comments,
        format!("{}{}", d.study_date, d.study_time),
        d.patient_name,
        VERSION,
        d.patient_birth_date,
        d.patient_sex.chars().next().unwrap_or(' '),
        d.patient_age,
        hdr.dim[1],
        hdr.dim[2],
        hdr.dim[3],
        hdr.dim[4],
        coil,
        d.echo_number,
        d.orient[1],
        d.orient[2],
        d.orient[3],
        d.orient[4],
        d.orient[5],
        d.orient[6],
        d.bits_allocated,
        dcm_name.display(),
    );
    fs::write(&path, line).map_err(|e| Error::io(&path, e))?;
    Ok(path)
}

/// Copy each DICOM to `outdir` using the filename template (`-r y`).
pub fn rename_dicoms(images: &[DicomImage], opts: &DcmOpts) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for d in images {
        if opts.ignore_derived && (d.is_derived || d.is_localizer) {
            continue;
        }
        if d.instance_number <= 0 {
            continue;
        }
        let stem = create_filename(d, opts)?;
        let dest = if stem.extension().is_some() {
            stem
        } else {
            stem.with_extension("dcm")
        };
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        fs::copy(&d.path, &dest).map_err(|e| Error::io(&dest, e))?;
        if opts.verbose > 0 {
            eprintln!("Renaming {} -> {}", d.path.display(), dest.display());
        }
        out.push(dest);
    }
    Ok(out)
}
