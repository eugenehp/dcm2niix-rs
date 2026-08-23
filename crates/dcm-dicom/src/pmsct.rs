//! Philips/Elscint PMSCT_RLE1 pixel decode (`nii_loadImgPMSCT_RLE1`).

use dcm_core::error::{Error, Result};

/// Decompress PMSCT_RLE1 payload into little-endian u16 samples.
pub fn decode_pmsct_rle1(compressed: &[u8], n_samples: usize) -> Result<Vec<u8>> {
    if compressed.len() < 66 {
        return Err(Error::convert(format!(
            "{} is not enough bytes for PMSCT_RLE1 compression",
            compressed.len()
        )));
    }
    let imgsz = n_samples * 2;
    if compressed.len() == imgsz {
        return Ok(compressed.to_vec());
    }
    // RLE pass: 0xA5 <repeat-1> <value>
    let mut temp = Vec::with_capacity(imgsz);
    let mut i = 0usize;
    while i < compressed.len() {
        if compressed[i] == 0xa5 {
            if i + 2 >= compressed.len() {
                break;
            }
            let mut repeat = compressed[i + 1] as usize + 1;
            let value = compressed[i + 2] as i8 as u8;
            while repeat > 0 {
                temp.push(value);
                repeat -= 1;
            }
            i += 3;
        } else {
            temp.push(compressed[i]);
            i += 1;
        }
    }
    // Delta pass: 0x5A <lo> <hi> absolute; else relative to prior.
    let mut out = vec![0u8; imgsz];
    let mut delta: u16 = 0;
    let mut o = 0usize;
    let n16 = imgsz / 2;
    let mut i = 0usize;
    while i < temp.len() && o < n16 {
        if temp[i] == 0x5a {
            if i + 2 >= temp.len() {
                break;
            }
            let value = u16::from_le_bytes([temp[i + 1], temp[i + 2]]);
            out[o * 2..o * 2 + 2].copy_from_slice(&value.to_le_bytes());
            delta = value;
            o += 1;
            i += 3;
        } else {
            let value = (temp[i] as u16).wrapping_add(delta);
            out[o * 2..o * 2 + 2].copy_from_slice(&value.to_le_bytes());
            delta = value;
            o += 1;
            i += 1;
        }
    }
    Ok(out)
}
