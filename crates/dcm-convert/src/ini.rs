//! Defaults file `~/.dcm2nii.ini` (`-g`).

use std::fs;
use std::path::PathBuf;

use dcm_core::error::{Error, Result};

use crate::opts::{BidsMode, Compress, DcmOpts, Maximize16Bit};

fn ini_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".dcm2nii.ini")
}

pub fn read_ini(opts: &mut DcmOpts) {
    let path = ini_path();
    let Ok(text) = fs::read_to_string(&path) else {
        return;
    };
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "isGZ" => {
                if v.trim() == "1" {
                    opts.compress = Compress::InternalGz;
                }
            }
            "isMaximize16BitRange" => {
                opts.maximize_16bit = match v.trim() {
                    "0" => Maximize16Bit::False,
                    "2" => Maximize16Bit::Raw,
                    _ => Maximize16Bit::True,
                };
            }
            "isBIDS" => {
                opts.bids = if v.trim() == "0" {
                    BidsMode::No
                } else {
                    BidsMode::Yes
                };
            }
            "filename" => opts.filename = v.trim().to_string(),
            _ => {}
        }
    }
}

pub fn save_ini(opts: &DcmOpts) -> Result<()> {
    let path = ini_path();
    let is_gz = matches!(opts.compress, Compress::Gz | Compress::InternalGz) as i32;
    let max16 = match opts.maximize_16bit {
        Maximize16Bit::False => 0,
        Maximize16Bit::True => 1,
        Maximize16Bit::Raw => 2,
    };
    let bids = match opts.bids {
        BidsMode::No => 0,
        _ => 1,
    };
    let body = format!(
        "isGZ={is_gz}\nisMaximize16BitRange={max16}\nisBIDS={bids}\nfilename={}\n",
        opts.filename
    );
    eprintln!("Saving defaults file {}", path.display());
    fs::write(&path, body).map_err(|e| Error::io(&path, e))?;
    Ok(())
}
