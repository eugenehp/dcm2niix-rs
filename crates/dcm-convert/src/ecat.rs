//! ECAT7 foreign import (`nii_foreign.cpp` / `readEcat7`).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use dcm_core::error::{Error, Result};
use dcm_core::matrix::Matrix4;
use dcm_dicom::{DicomImage, Manufacturer, Modality};
use dcm_nifti::{Nifti1Header, DT_FLOAT32, DT_INT16, DT_UINT8};

use crate::parrec::minimal_image;

const ECAT7_BYTE: i16 = 1;
const ECAT7_SUNI2: i16 = 6;
const ECAT7_SUNI4: i16 = 7;

fn io(path: &Path, e: std::io::Error) -> Error {
    Error::io(path, e)
}

fn swap2(buf: &mut [u8]) {
    for c in buf.chunks_exact_mut(2) {
        c.swap(0, 1);
    }
}
fn swap4(buf: &mut [u8]) {
    for c in buf.chunks_exact_mut(4) {
        c.swap(0, 3);
        c.swap(1, 2);
    }
}

fn clean_str(s: &str) -> String {
    s.chars()
        .map(|c| {
            let u = c as u32;
            if u < 32 || u == 127 || u == 255 {
                '\0'
            } else if matches!(c, ' ' | ',' | '^' | '/' | '\\' | '%' | '*') {
                '_'
            } else {
                c
            }
        })
        .take_while(|&c| c != '\0')
        .collect::<String>()
        .trim_matches('\0')
        .to_string()
}

/// Returns true when `path` looks like an ECAT7 MATRIX file.
pub fn is_ecat7(path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 6];
    if f.read_exact(&mut magic).is_err() {
        return false;
    }
    &magic == b"MATRIX"
}

/// Read ECAT7 → synthetic `DicomImage` + voxel bytes + NIfTI header scaffold.
pub fn read_ecat7(path: &Path) -> Result<(DicomImage, Nifti1Header, Vec<u8>)> {
    let mut f = File::open(path).map_err(|e| io(path, e))?;
    let mut mhdr = [0u8; 512];
    f.read_exact(&mut mhdr).map_err(|e| io(path, e))?;
    if &mhdr[0..6] != b"MATRIX" {
        return Err(Error::convert(format!(
            "Signature not 'MATRIX' (ECAT7): {}",
            path.display()
        )));
    }
    let file_type = u16::from_le_bytes([mhdr[50], mhdr[51]]);
    let swap = file_type > 255;
    let file_type = if swap {
        u16::from_be_bytes([mhdr[50], mhdr[51]])
    } else {
        file_type
    };
    if !(1..=14).contains(&file_type) {
        return Err(Error::convert(format!("Unknown ECAT file type {file_type}")));
    }
    let mut cal = f32::from_le_bytes(mhdr[144..148].try_into().unwrap());
    if swap {
        cal = f32::from_be_bytes(mhdr[144..148].try_into().unwrap());
    }

    // List header at block 1 (offset 512)
    f.seek(SeekFrom::Start(512)).map_err(|e| io(path, e))?;
    let mut lhdr = [0u8; 512];
    f.read_exact(&mut lhdr).map_err(|e| io(path, e))?;
    let mut lvals = [0i32; 128];
    for i in 0..128 {
        let mut b = [lhdr[i * 4], lhdr[i * 4 + 1], lhdr[i * 4 + 2], lhdr[i * 4 + 3]];
        if swap {
            b.reverse();
        }
        lvals[i] = i32::from_le_bytes(b);
    }
    // r[0][1] is at index 5 (hdr[4] then r[0][0], r[0][1]...)
    // packed: hdr[4] + r[31][4] → first image block = r[0][1]
    let img_start_block = lvals[5];
    let img_hdr_off = (img_start_block as u64 - 1) * 512;
    f.seek(SeekFrom::Start(img_hdr_off)).map_err(|e| io(path, e))?;
    let mut ihdr = [0u8; 512];
    f.read_exact(&mut ihdr).map_err(|e| io(path, e))?;

    let mut data_type = i16::from_le_bytes([ihdr[0], ihdr[1]]);
    let mut x_dim = i16::from_le_bytes([ihdr[4], ihdr[5]]);
    let mut y_dim = i16::from_le_bytes([ihdr[6], ihdr[7]]);
    let mut z_dim = i16::from_le_bytes([ihdr[8], ihdr[9]]);
    let mut scale = f32::from_le_bytes(ihdr[26..30].try_into().unwrap());
    let mut x_pix = f32::from_le_bytes(ihdr[34..38].try_into().unwrap());
    let mut y_pix = f32::from_le_bytes(ihdr[38..42].try_into().unwrap());
    let mut z_pix = f32::from_le_bytes(ihdr[42..46].try_into().unwrap());
    let mut frame_dur = i32::from_le_bytes(ihdr[46..50].try_into().unwrap());
    if swap {
        data_type = i16::from_be_bytes([ihdr[0], ihdr[1]]);
        x_dim = i16::from_be_bytes([ihdr[4], ihdr[5]]);
        y_dim = i16::from_be_bytes([ihdr[6], ihdr[7]]);
        z_dim = i16::from_be_bytes([ihdr[8], ihdr[9]]);
        scale = f32::from_be_bytes(ihdr[26..30].try_into().unwrap());
        x_pix = f32::from_be_bytes(ihdr[34..38].try_into().unwrap());
        y_pix = f32::from_be_bytes(ihdr[38..42].try_into().unwrap());
        z_pix = f32::from_be_bytes(ihdr[42..46].try_into().unwrap());
        frame_dur = i32::from_be_bytes(ihdr[46..50].try_into().unwrap());
    }
    if data_type != ECAT7_BYTE && data_type != ECAT7_SUNI2 && data_type != ECAT7_SUNI4 {
        return Err(Error::convert(format!(
            "Unknown or unsupported ECAT data type {data_type}"
        )));
    }

    // Collect volume offsets + per-volume scale factors.
    let mut offsets: Vec<u64> = Vec::new();
    let mut slopes: Vec<f32> = Vec::new();
    let mut list = lvals;
    let mut list_off = 512u64;
    loop {
        if list[0] + list[3] != 31 {
            break;
        }
        let n_mat = list[3];
        if n_mat < 1 {
            break;
        }
        for k in 0..n_mat as usize {
            let block = list[4 + k * 4 + 1];
            if block > 0 {
                offsets.push(block as u64);
                // Read that volume's image header for scale_factor (offset 26).
                let ih_off = (block as u64 - 1) * 512;
                let cur = f.stream_position().map_err(|e| io(path, e))?;
                f.seek(SeekFrom::Start(ih_off)).map_err(|e| io(path, e))?;
                let mut ih = [0u8; 512];
                if f.read_exact(&mut ih).is_ok() {
                    let mut s = f32::from_le_bytes(ih[26..30].try_into().unwrap());
                    if swap {
                        s = f32::from_be_bytes(ih[26..30].try_into().unwrap());
                    }
                    slopes.push(s);
                } else {
                    slopes.push(scale);
                }
                f.seek(SeekFrom::Start(cur)).map_err(|e| io(path, e))?;
            }
        }
        if list[0] > 0 {
            break;
        }
        let next_block = list[1];
        if next_block <= 0 {
            break;
        }
        list_off = (next_block as u64 - 1) * 512;
        f.seek(SeekFrom::Start(list_off)).map_err(|e| io(path, e))?;
        f.read_exact(&mut lhdr).map_err(|e| io(path, e))?;
        for i in 0..128 {
            let mut b = [lhdr[i * 4], lhdr[i * 4 + 1], lhdr[i * 4 + 2], lhdr[i * 4 + 3]];
            if swap {
                b.reverse();
            }
            list[i] = i32::from_le_bytes(b);
        }
        if offsets.len() > 16000 {
            break;
        }
    }
    if offsets.is_empty() {
        offsets.push(img_start_block as u64);
        slopes.push(scale);
    }
    let num_vol = offsets.len();
    let nx = x_dim.max(1) as usize;
    let ny = y_dim.max(1) as usize;
    let nz = z_dim.max(1) as usize;
    let scale_varies = slopes.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-12);
    let mut bytes_per_vox = match data_type {
        ECAT7_BYTE => 1,
        ECAT7_SUNI4 => 4,
        _ => 2,
    };
    let img = if scale_varies && bytes_per_vox == 2 {
        // Promote to float32 applying per-volume scale * calibration (C++ path).
        let n_vox = nx * ny * nz;
        let mut out = vec![0u8; n_vox * 4 * num_vol];
        let mut tmp = vec![0u8; n_vox * 2];
        for (v, &blk) in offsets.iter().enumerate() {
            f.seek(SeekFrom::Start(blk * 512)).map_err(|e| io(path, e))?;
            f.read_exact(&mut tmp).map_err(|e| io(path, e))?;
            if swap {
                swap2(&mut tmp);
            }
            let sl = slopes.get(v).copied().unwrap_or(scale) * cal;
            for i in 0..n_vox {
                let s = i16::from_le_bytes([tmp[i * 2], tmp[i * 2 + 1]]);
                let fval = (s as f32) * sl;
                let o = (v * n_vox + i) * 4;
                out[o..o + 4].copy_from_slice(&fval.to_le_bytes());
            }
        }
        bytes_per_vox = 4;
        scale = 1.0;
        cal = 1.0;
        out
    } else {
        if scale_varies {
            return Err(Error::convert(
                "ECAT scale factor varies between volumes (check for updates)",
            ));
        }
        let bytes_per_vol = nx * ny * nz * bytes_per_vox;
        let mut img = vec![0u8; bytes_per_vol * num_vol];
        for (v, &blk) in offsets.iter().enumerate() {
            f.seek(SeekFrom::Start(blk * 512)).map_err(|e| io(path, e))?;
            f.read_exact(&mut img[v * bytes_per_vol..(v + 1) * bytes_per_vol])
                .map_err(|e| io(path, e))?;
        }
        if swap && bytes_per_vox == 2 {
            swap2(&mut img);
        }
        if swap && bytes_per_vox == 4 {
            swap4(&mut img);
        }
        img
    };
    eprintln!("Warning: ECAT support VERY experimental (Spatial transforms unknown)");
    // Spatial matrix in image header (mtx[9] at ~offset 238 in ecat_img_hdr).
    let mut has_mtx = false;
    for i in 0..9 {
        let off = 238 + i * 4;
        if off + 4 <= ihdr.len() {
            let mut m = f32::from_le_bytes(ihdr[off..off + 4].try_into().unwrap());
            if swap {
                m = f32::from_be_bytes(ihdr[off..off + 4].try_into().unwrap());
            }
            if m != 0.0 {
                has_mtx = true;
            }
        }
    }
    if has_mtx {
        eprintln!(
            "Warning: ECAT volume appears to store spatial transformation matrix (please check for updates)"
        );
    }
    let mut gantry = f32::from_le_bytes(mhdr[90..94].try_into().unwrap_or([0; 4]));
    if swap {
        gantry = f32::from_be_bytes(mhdr[90..94].try_into().unwrap_or([0; 4]));
    }
    if gantry != 0.0 {
        eprintln!("Warning: ECAT gantry tilt not supported {gantry}");
    }

    let mut d = minimal_image(path);
    d.manufacturer = Manufacturer::Siemens;
    d.modality = Modality::Pt;
    let mut isotope_hl = f32::from_le_bytes(mhdr[74..78].try_into().unwrap_or([0; 4]));
    let mut dosage = f32::from_le_bytes(mhdr[474..478].try_into().unwrap_or([0; 4]));
    if swap {
        isotope_hl = f32::from_be_bytes(mhdr[74..78].try_into().unwrap_or([0; 4]));
        dosage = f32::from_be_bytes(mhdr[474..478].try_into().unwrap_or([0; 4]));
    }
    d.ecat_isotope_halflife = isotope_hl as f64;
    d.ecat_dosage = dosage as f64;
    d.image_comments = clean_str(std::str::from_utf8(&mhdr[66..74]).unwrap_or(""));
    d.radiopharmaceutical = clean_str(std::str::from_utf8(&mhdr[78..110]).unwrap_or(""));
    d.patient_name = clean_str(std::str::from_utf8(&mhdr[172..204]).unwrap_or(""));
    d.patient_id = clean_str(std::str::from_utf8(&mhdr[156..172]).unwrap_or(""));
    d.series_description = clean_str(std::str::from_utf8(&mhdr[294..326]).unwrap_or(""));
    d.protocol_name = clean_str(std::str::from_utf8(&mhdr[144..156]).unwrap_or(""));
    // note: study_type offset may differ; keep best-effort strings
    d.bits_allocated = (bytes_per_vox * 8) as i32;
    d.bits_stored = if bytes_per_vox == 2 { 15 } else { d.bits_allocated };
    d.is_signed = bytes_per_vox == 2;
    d.is_float = bytes_per_vox == 4;
    d.xyz_mm = [0.0, (x_pix * 10.0) as f64, (y_pix * 10.0) as f64, (z_pix * 10.0) as f64];
    d.tr = frame_dur as f64;
    d.rows = ny;
    d.columns = nx;
    d.number_of_frames = num_vol as i32;

    let mut hdr = Nifti1Header::default();
    hdr.dim[0] = if num_vol > 1 { 4 } else { 3 };
    hdr.dim[1] = nx as i16;
    hdr.dim[2] = ny as i16;
    hdr.dim[3] = nz as i16;
    hdr.dim[4] = num_vol.max(1) as i16;
    hdr.pixdim[1] = d.xyz_mm[1] as f32;
    hdr.pixdim[2] = d.xyz_mm[2] as f32;
    hdr.pixdim[3] = d.xyz_mm[3] as f32;
    hdr.pixdim[4] = (frame_dur as f32) / 1000.0;
    hdr.datatype = match bytes_per_vox {
        1 => DT_UINT8,
        4 => DT_FLOAT32,
        _ => DT_INT16,
    };
    hdr.bitpix = (bytes_per_vox * 8) as i16;
    hdr.scl_slope = scale * cal;
    hdr.vox_offset = 352.0;
    // SPM-like starting estimate
    let m = Matrix4::from_rows([
        [
            -hdr.pixdim[1] as f64,
            0.0,
            0.0,
            ((nx as f64 - 2.0) / 2.0) * d.xyz_mm[1],
        ],
        [
            0.0,
            -hdr.pixdim[2] as f64,
            0.0,
            ((ny as f64 - 2.0) / 2.0) * d.xyz_mm[2],
        ],
        [
            0.0,
            0.0,
            -hdr.pixdim[3] as f64,
            ((nz as f64 - 2.0) / 2.0) * d.xyz_mm[3],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    hdr.set_sform(&m);
    Ok((d, hdr, img))
}
