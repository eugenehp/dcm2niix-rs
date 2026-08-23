//! NRRD / MGH export (`nii_saveNRRD` / `nii_saveMGH`).
//!
//! Used when `-e` selects a foreign format. NRRD can embed per-volume DWMRI
//! gradient keys when a multi-volume slice list is supplied.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use dcm_core::error::{Error, Result};
use dcm_dicom::{DicomImage, Manufacturer, Modality};
use dcm_nifti::{Nifti1Header, DT_FLOAT32, DT_INT16, DT_INT32, DT_UINT16, DT_UINT8};
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::opts::Compress;

fn io(path: &Path, e: std::io::Error) -> Error {
    Error::io(path, e)
}

/// Write `.nrrd` (or `.nhdr` + `.raw.gz` when gzip is requested).
pub fn write_nrrd(
    stem: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    volumes: Option<&[&DicomImage]>,
    compress: Compress,
) -> Result<PathBuf> {
    match write_nrrd_inner(stem, hdr, voxels, d, volumes, compress) {
        Ok(p) => Ok(p),
        Err(e) => {
            // Fail-closed: drop partial header / raw siblings (C++ audit).
            let gzip = matches!(compress, Compress::Gz | Compress::InternalGz);
            let hdr_path = if gzip {
                stem.with_extension("nhdr")
            } else {
                stem.with_extension("nrrd")
            };
            let _ = std::fs::remove_file(&hdr_path);
            if gzip {
                let _ = std::fs::remove_file(stem.with_extension("raw.gz"));
            }
            Err(e)
        }
    }
}

fn write_nrrd_inner(
    stem: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    volumes: Option<&[&DicomImage]>,
    compress: Compress,
) -> Result<PathBuf> {
    let n_dim = hdr.dim[0].max(1) as usize;
    let gzip = matches!(compress, Compress::Gz | Compress::InternalGz);
    let path = if gzip {
        stem.with_extension("nhdr")
    } else {
        stem.with_extension("nrrd")
    };
    let mut fp = File::create(&path).map_err(|e| io(&path, e))?;
    writeln!(fp, "NRRD0005").map_err(|e| io(&path, e))?;
    writeln!(fp, "# Complete NRRD file format specification at:").map_err(|e| io(&path, e))?;
    writeln!(fp, "# http://teem.sourceforge.net/nrrd/format.html").map_err(|e| io(&path, e))?;
    let type_s = match hdr.datatype {
        DT_UINT8 => "uint8",
        DT_INT16 => "int16",
        DT_UINT16 => "uint16",
        DT_FLOAT32 => "float",
        DT_INT32 => "int32",
        _ => {
            return Err(Error::convert(format!(
                "Unknown NRRD datatype {}",
                hdr.datatype
            )))
        }
    };
    writeln!(fp, "type: {type_s}").map_err(|e| io(&path, e))?;
    writeln!(fp, "dimension: {n_dim}").map_err(|e| io(&path, e))?;
    writeln!(fp, "space: right-anterior-superior").map_err(|e| io(&path, e))?;
    write!(fp, "sizes:").map_err(|e| io(&path, e))?;
    for i in 1..=n_dim {
        write!(fp, " {}", hdr.dim[i]).map_err(|e| io(&path, e))?;
    }
    writeln!(fp).map_err(|e| io(&path, e))?;
    writeln!(fp, "endian: little").map_err(|e| io(&path, e))?;
    if gzip {
        let raw = stem.with_extension("raw.gz");
        let base = raw
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "data.raw.gz".into());
        writeln!(fp, "encoding: gzip").map_err(|e| io(&path, e))?;
        writeln!(fp, "data file: {base}").map_err(|e| io(&path, e))?;
        let f = File::create(&raw).map_err(|e| io(&raw, e))?;
        let mut enc = GzEncoder::new(f, Compression::default());
        enc.write_all(voxels).map_err(|e| io(&raw, e))?;
        enc.finish().map_err(|e| io(&raw, e))?;
    } else {
        writeln!(fp, "encoding: raw").map_err(|e| io(&path, e))?;
    }
    writeln!(fp, "space units: \"mm\" \"mm\" \"mm\"").map_err(|e| io(&path, e))?;
    writeln!(
        fp,
        "space origin: ({},{},{})",
        hdr.srow_x[3], hdr.srow_y[3], hdr.srow_z[3]
    )
    .map_err(|e| io(&path, e))?;
    write!(fp, "space directions:").map_err(|e| io(&path, e))?;
    write!(
        fp,
        " ({},{},{}) ({},{},{}) ({},{},{})",
        hdr.srow_x[0],
        hdr.srow_y[0],
        hdr.srow_z[0],
        hdr.srow_x[1],
        hdr.srow_y[1],
        hdr.srow_z[1],
        hdr.srow_x[2],
        hdr.srow_y[2],
        hdr.srow_z[2]
    )
    .map_err(|e| io(&path, e))?;
    for _ in 3..n_dim {
        write!(fp, " none").map_err(|e| io(&path, e))?;
    }
    writeln!(fp).map_err(|e| io(&path, e))?;
    if n_dim < 4 {
        writeln!(fp, "centerings: cell cell cell").map_err(|e| io(&path, e))?;
    } else {
        writeln!(fp, "centerings: cell cell cell ???").map_err(|e| io(&path, e))?;
    }
    write!(fp, "kinds:").map_err(|e| io(&path, e))?;
    for i in 0..n_dim {
        if i < 3 {
            write!(fp, " space").map_err(|e| io(&path, e))?;
        } else {
            write!(fp, " list").map_err(|e| io(&path, e))?;
        }
    }
    writeln!(fp).map_err(|e| io(&path, e))?;
    match d.modality {
        Modality::Mr => writeln!(fp, "DICOM_0008_0060_Modality:=MR").map_err(|e| io(&path, e))?,
        Modality::Ct => writeln!(fp, "DICOM_0008_0060_Modality:=CT").map_err(|e| io(&path, e))?,
        _ => {}
    }
    // DWI: C++ `modality:=DWMRI` + per-volume gradients when present.
    let vols: &[&DicomImage] = match volumes {
        Some(v) => v,
        None => &[d],
    };
    let is_dwi = vols.iter().any(|v| v.is_diffusion || v.b_value >= 0.0);
    if is_dwi {
        writeln!(fp, "modality:=DWMRI").map_err(|e| io(&path, e))?;
        let b_max = vols
            .iter()
            .map(|v| v.b_value.max(0.0))
            .fold(0.0f64, f64::max);
        if b_max > 0.0 {
            writeln!(fp, "DWMRI_b-value:={b_max}").map_err(|e| io(&path, e))?;
        }
        let nt = hdr.dim[4].max(1) as usize;
        let n = vols.len().min(nt).max(1);
        for i in 0..n {
            let v = &vols[i.min(vols.len() - 1)];
            let g = v.diffusion_direction;
            writeln!(
                fp,
                "DWMRI_gradient_{i:04}:={} {} {}",
                g[0], g[1], g[2]
            )
            .map_err(|e| io(&path, e))?;
        }
    }
    match d.manufacturer {
        Manufacturer::Siemens => {
            writeln!(fp, "DICOM_0008_0070_Manufacturer:=SIEMENS").map_err(|e| io(&path, e))?
        }
        Manufacturer::Philips => writeln!(
            fp,
            "DICOM_0008_0070_Manufacturer:=Philips Medical Systems"
        )
        .map_err(|e| io(&path, e))?,
        Manufacturer::Ge => writeln!(fp, "DICOM_0008_0070_Manufacturer:=GE MEDICAL SYSTEMS")
            .map_err(|e| io(&path, e))?,
        _ => {}
    }
    if d.tr > 0.0 {
        writeln!(fp, "DICOM_0018_0080_RepetitionTime:={}", d.tr).map_err(|e| io(&path, e))?;
    }
    if d.te > 0.0 {
        writeln!(fp, "DICOM_0018_0081_EchoTime:={}", d.te).map_err(|e| io(&path, e))?;
    }
    if !gzip {
        writeln!(fp).map_err(|e| io(&path, e))?;
        fp.write_all(voxels).map_err(|e| io(&path, e))?;
    }
    Ok(path)
}

/// Packed MGH header — matches C++ `Tmgh` (284 bytes).
#[repr(C, packed)]
struct MghHeader {
    version: i32,
    width: i32,
    height: i32,
    depth: i32,
    nframes: i32,
    type_: i32,
    dof: i32,
    good_ras: i16,
    spacing_x: f32,
    spacing_y: f32,
    spacing_z: f32,
    xr: f32,
    xa: f32,
    xs: f32,
    yr: f32,
    ya: f32,
    ys: f32,
    zr: f32,
    za: f32,
    zs: f32,
    cr: f32,
    ca: f32,
    cs: f32,
    pad: [i16; 97],
}

#[repr(C, packed)]
struct MghFooter {
    tr: f32,
    flip: f32,
    te: f32,
    ti: f32,
}

/// Write `.mgh` / `.mgz` (always big-endian payload).
pub fn write_mgh(
    stem: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    compress: Compress,
) -> Result<PathBuf> {
    match write_mgh_inner(stem, hdr, voxels, d, compress) {
        Ok(p) => Ok(p),
        Err(e) => {
            // Fail-closed: drop partial .mgh / .mgz (C++ audit).
            let gzip = matches!(compress, Compress::Gz | Compress::InternalGz);
            let path = if gzip {
                stem.with_extension("mgz")
            } else {
                stem.with_extension("mgh")
            };
            let _ = std::fs::remove_file(&path);
            Err(e)
        }
    }
}

fn write_mgh_inner(
    stem: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    d: &DicomImage,
    compress: Compress,
) -> Result<PathBuf> {
    let mgh_type: i32 = match hdr.datatype {
        DT_UINT8 => 0,
        DT_INT32 => 1,
        DT_FLOAT32 => 3,
        DT_INT16 | DT_UINT16 => 4,
        _ => {
            return Err(Error::convert(format!(
                "MGH format does not support NIfTI datatype {}",
                hdr.datatype
            )))
        }
    };
    let mut xmm = hdr.pixdim[1];
    let mut ymm = hdr.pixdim[2];
    let mut zmm = hdr.pixdim[3];
    if xmm <= 0.0 {
        xmm = 1.0;
    }
    if ymm <= 0.0 {
        ymm = 1.0;
    }
    if zmm <= 0.0 {
        zmm = 1.0;
    }
    let vec = [
        hdr.dim[1] as f32 * 0.5,
        hdr.dim[2] as f32 * 0.5,
        hdr.dim[3] as f32 * 0.5,
    ];
    let mgh = MghHeader {
        version: 1i32.to_be(),
        width: (hdr.dim[1] as i32).to_be(),
        height: (hdr.dim[2] as i32).to_be(),
        depth: (hdr.dim[3] as i32).to_be(),
        nframes: (hdr.dim[4].max(1) as i32).to_be(),
        type_: mgh_type.to_be(),
        dof: 0i32.to_be(),
        good_ras: 1i16.to_be(),
        spacing_x: f32_be(xmm),
        spacing_y: f32_be(ymm),
        spacing_z: f32_be(zmm),
        xr: f32_be(hdr.srow_x[0] / xmm),
        xa: f32_be(hdr.srow_y[0] / xmm),
        xs: f32_be(hdr.srow_z[0] / xmm),
        yr: f32_be(hdr.srow_x[1] / ymm),
        ya: f32_be(hdr.srow_y[1] / ymm),
        ys: f32_be(hdr.srow_z[1] / ymm),
        zr: f32_be(hdr.srow_x[2] / zmm),
        za: f32_be(hdr.srow_y[2] / zmm),
        zs: f32_be(hdr.srow_z[2] / zmm),
        cr: f32_be(
            hdr.srow_x[0] * vec[0]
                + hdr.srow_x[1] * vec[1]
                + hdr.srow_x[2] * vec[2]
                + hdr.srow_x[3],
        ),
        ca: f32_be(
            hdr.srow_y[0] * vec[0]
                + hdr.srow_y[1] * vec[1]
                + hdr.srow_y[2] * vec[2]
                + hdr.srow_y[3],
        ),
        cs: f32_be(
            hdr.srow_z[0] * vec[0]
                + hdr.srow_z[1] * vec[1]
                + hdr.srow_z[2] * vec[2]
                + hdr.srow_z[3],
        ),
        pad: [0; 97],
    };
    let footer = MghFooter {
        tr: f32_be(d.tr as f32),
        flip: f32_be(d.flip_angle as f32),
        te: f32_be(d.te as f32),
        ti: f32_be(d.ti as f32),
    };
    let mut be_vox = voxels.to_vec();
    swap_voxel_endian(hdr.datatype, &mut be_vox);

    let gzip = matches!(compress, Compress::Gz | Compress::InternalGz);
    let path = if gzip {
        stem.with_extension("mgz")
    } else {
        stem.with_extension("mgh")
    };
    debug_assert_eq!(std::mem::size_of::<MghHeader>(), 284);
    let hdr_bytes = unsafe {
        std::slice::from_raw_parts(
            (&mgh as *const MghHeader) as *const u8,
            std::mem::size_of::<MghHeader>(),
        )
    };
    let foot_bytes = unsafe {
        std::slice::from_raw_parts(
            (&footer as *const MghFooter) as *const u8,
            std::mem::size_of::<MghFooter>(),
        )
    };
    if gzip {
        let f = File::create(&path).map_err(|e| io(&path, e))?;
        let mut enc = GzEncoder::new(f, Compression::default());
        enc.write_all(hdr_bytes).map_err(|e| io(&path, e))?;
        enc.write_all(&be_vox).map_err(|e| io(&path, e))?;
        enc.write_all(foot_bytes).map_err(|e| io(&path, e))?;
        enc.finish().map_err(|e| io(&path, e))?;
    } else {
        let mut f = File::create(&path).map_err(|e| io(&path, e))?;
        f.write_all(hdr_bytes).map_err(|e| io(&path, e))?;
        f.write_all(&be_vox).map_err(|e| io(&path, e))?;
        f.write_all(foot_bytes).map_err(|e| io(&path, e))?;
    }
    Ok(path)
}

fn swap_voxel_endian(datatype: i16, buf: &mut [u8]) {
    let step = match datatype {
        DT_INT16 | DT_UINT16 => 2,
        DT_INT32 | DT_FLOAT32 => 4,
        _ => return,
    };
    for chunk in buf.chunks_exact_mut(step) {
        chunk.reverse();
    }
}

fn f32_be(v: f32) -> f32 {
    f32::from_bits(v.to_bits().to_be())
}
