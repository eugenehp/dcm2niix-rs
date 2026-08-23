//! JNIfTI / BNIfTI export (`-e j` / `-e b`) — NeuroJSON layout matching C++
//! `nii_savejnii` / `nii_savebnii`.
//!
//! `.jnii` is pretty-printed JSON with structured `NIFTIHeader` + base64 (or
//! zlib+base64) voxel payload. `.bnii` is BJData/UBJSON with the same fields.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use dcm_core::error::{Error, Result};
use dcm_nifti::Nifti1Header;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use serde_json::{json, Map, Value};

use crate::opts::Compress;

fn io(path: &Path, e: std::io::Error) -> Error {
    Error::io(path, e)
}

fn cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn b64_encode(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (a << 16) | (b << 8) | c;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// NIfTI datatype / intent / slice-code / unit tables (C++ `nii_savejnii`).
struct TypeInfo {
    nifti_name: &'static str,
    jdata_name: &'static str,
    jdata_elem_len: u8,
    jdata_marker: u8,
}

fn lookup_datatype(dt: i16) -> TypeInfo {
    // Parallel to C++ datatypeid / datatypestr / jdatatypestr / jdataelemlen / jdatamarker.
    const TABLE: &[(i16, TypeInfo)] = &[
        (
            2,
            TypeInfo {
                nifti_name: "uint8",
                jdata_name: "uint8",
                jdata_elem_len: 1,
                jdata_marker: b'U',
            },
        ),
        (
            4,
            TypeInfo {
                nifti_name: "int16",
                jdata_name: "int16",
                jdata_elem_len: 1,
                jdata_marker: b'I',
            },
        ),
        (
            8,
            TypeInfo {
                nifti_name: "int32",
                jdata_name: "int32",
                jdata_elem_len: 1,
                jdata_marker: b'l',
            },
        ),
        (
            16,
            TypeInfo {
                nifti_name: "single",
                jdata_name: "single",
                jdata_elem_len: 1,
                jdata_marker: b'd',
            },
        ),
        (
            32,
            TypeInfo {
                nifti_name: "complex64",
                jdata_name: "double",
                jdata_elem_len: 2,
                jdata_marker: b'D',
            },
        ),
        (
            64,
            TypeInfo {
                nifti_name: "double",
                jdata_name: "double",
                jdata_elem_len: 1,
                jdata_marker: b'D',
            },
        ),
        (
            128,
            TypeInfo {
                nifti_name: "rgb24",
                jdata_name: "uint8",
                jdata_elem_len: 3,
                jdata_marker: b'U',
            },
        ),
        (
            256,
            TypeInfo {
                nifti_name: "int8",
                jdata_name: "int8",
                jdata_elem_len: 1,
                jdata_marker: b'i',
            },
        ),
        (
            512,
            TypeInfo {
                nifti_name: "uint16",
                jdata_name: "uint16",
                jdata_elem_len: 1,
                jdata_marker: b'u',
            },
        ),
        (
            768,
            TypeInfo {
                nifti_name: "uint32",
                jdata_name: "uint32",
                jdata_elem_len: 1,
                jdata_marker: b'm',
            },
        ),
        (
            1024,
            TypeInfo {
                nifti_name: "int64",
                jdata_name: "int64",
                jdata_elem_len: 1,
                jdata_marker: b'L',
            },
        ),
        (
            1280,
            TypeInfo {
                nifti_name: "uint64",
                jdata_name: "uint64",
                jdata_elem_len: 1,
                jdata_marker: b'M',
            },
        ),
        (
            2304,
            TypeInfo {
                nifti_name: "rgba32",
                jdata_name: "uint8",
                jdata_elem_len: 4,
                jdata_marker: b'U',
            },
        ),
    ];
    for (id, info) in TABLE {
        if *id == dt {
            return TypeInfo {
                nifti_name: info.nifti_name,
                jdata_name: info.jdata_name,
                jdata_elem_len: info.jdata_elem_len,
                jdata_marker: info.jdata_marker,
            };
        }
    }
    TypeInfo {
        nifti_name: "",
        jdata_name: "uint8",
        jdata_elem_len: 1,
        jdata_marker: b'U',
    }
}

fn slice_type_name(code: u8) -> &'static str {
    match code {
        1 => "seq+",
        2 => "seq-",
        3 => "alt+",
        4 => "alt-",
        5 => "alt2+",
        6 => "alt2-",
        _ => "",
    }
}

fn intent_name(code: i16) -> &'static str {
    match code {
        2 => "corr",
        3 => "ttest",
        4 => "ftest",
        5 => "zscore",
        6 => "chi2",
        7 => "beta",
        8 => "binomial",
        9 => "gamma",
        10 => "poisson",
        11 => "normal",
        12 => "ncftest",
        13 => "ncchi2",
        14 => "logistic",
        15 => "laplace",
        16 => "uniform",
        17 => "ncttest",
        18 => "weibull",
        19 => "chi",
        20 => "invgauss",
        21 => "extval",
        22 => "pvalue",
        23 => "logpvalue",
        24 => "log10pvalue",
        1001 => "estimate",
        1002 => "label",
        1003 => "neuronames",
        1004 => "matrix",
        1005 => "symmatrix",
        1006 => "dispvec",
        1007 => "vector",
        1008 => "point",
        1009 => "triangle",
        1010 => "quaternion",
        1011 => "unitless",
        2001 => "tseries",
        2002 => "elem",
        2003 => "rgb",
        2004 => "rgba",
        2005 => "shape",
        _ => "",
    }
}

fn unit_name(code: u8) -> &'static str {
    match code {
        1 => "m",
        2 => "mm",
        3 => "um",
        8 => "s",
        16 => "ms",
        24 => "us",
        32 => "hz",
        40 => "ppm",
        48 => "rad",
        _ => "",
    }
}

fn space_unit(xyzt: u8) -> u8 {
    xyzt & 0x07
}

fn time_unit(xyzt: u8) -> u8 {
    xyzt & 0x38
}

fn img_bytes(hdr: &Nifti1Header) -> usize {
    let mut n = 1usize;
    let ndim = hdr.dim[0].max(1) as usize;
    for i in 1..=ndim {
        let d = hdr.dim[i].max(1) as usize;
        n = n.saturating_mul(d);
    }
    let bp = (hdr.bitpix.max(8) as usize) / 8;
    n.saturating_mul(bp)
}

fn zlib_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)
        .map_err(|e| Error::convert(e.to_string()))?;
    enc.finish()
        .map_err(|e| Error::convert(e.to_string()))
}

fn want_zip(compress: Compress) -> bool {
    matches!(compress, Compress::Gz | Compress::InternalGz)
}

/// Write `.jnii` (JSON NeuroJSON) or `.bnii` (BJData/UBJSON).
pub fn write_jnifti(
    stem: &Path,
    hdr: &Nifti1Header,
    voxels: &[u8],
    binary: bool,
    compress: Compress,
) -> Result<PathBuf> {
    let total = img_bytes(hdr).min(voxels.len());
    let im = &voxels[..total];
    let mut ndim = hdr.dim[0].max(1) as usize;
    let mut dim: Vec<i32> = (1..=ndim).map(|i| hdr.dim[i].max(1) as i32).collect();
    let info = lookup_datatype(hdr.datatype);
    if info.jdata_elem_len > 1 {
        dim.push(info.jdata_elem_len as i32);
        ndim += 1;
    }
    if binary {
        write_bnii(stem, hdr, im, &dim, ndim, &info, want_zip(compress))
    } else {
        write_jnii(stem, hdr, im, &dim, &info, want_zip(compress))
    }
}

fn write_jnii(
    stem: &Path,
    hdr: &Nifti1Header,
    im: &[u8],
    dim: &[i32],
    info: &TypeInfo,
    zip: bool,
) -> Result<PathBuf> {
    let path = stem.with_extension("jnii");
    let ndim = hdr.dim[0].max(1) as usize;
    let voxel_size: Vec<f32> = (1..=ndim).map(|i| hdr.pixdim[i]).collect();
    let orient_x = if hdr.pixdim[0] != 0.0 { "l" } else { "r" };

    let mut unit = Map::new();
    unit.insert("L".into(), Value::String(unit_name(space_unit(hdr.xyzt_units)).into()));
    unit.insert("T".into(), Value::String(unit_name(time_unit(hdr.xyzt_units)).into()));
    let special = hdr.xyzt_units;
    if special >= 32 {
        if let s @ ("hz" | "ppm" | "rad") = unit_name(special) {
            if !s.is_empty() {
                unit.insert("Special".into(), Value::String(s.into()));
            }
        }
    }

    let mut jhdr = Map::new();
    jhdr.insert("NIIHeaderSize".into(), json!(hdr.sizeof_hdr));
    // Header Dim is spatial/time only (before jdata element-length split).
    let hdr_dim: Vec<Value> = (1..=ndim).map(|i| json!(hdr.dim[i].max(1) as i32)).collect();
    jhdr.insert("Dim".into(), Value::Array(hdr_dim));
    jhdr.insert("Param1".into(), json!(hdr.intent_p1));
    jhdr.insert("Param2".into(), json!(hdr.intent_p2));
    jhdr.insert("Param3".into(), json!(hdr.intent_p3));
    jhdr.insert("Intent".into(), json!(intent_name(hdr.intent_code)));
    jhdr.insert("DataType".into(), json!(info.nifti_name));
    jhdr.insert("BitDepth".into(), json!(hdr.bitpix));
    jhdr.insert("FirstSliceID".into(), json!(hdr.slice_start));
    jhdr.insert("VoxelSize".into(), json!(voxel_size));
    jhdr.insert(
        "Orientation".into(),
        json!({ "x": orient_x, "y": "a", "z": "s" }),
    );
    jhdr.insert("ScaleSlope".into(), json!(hdr.scl_slope));
    jhdr.insert("ScaleOffset".into(), json!(hdr.scl_inter));
    jhdr.insert("LastSliceID".into(), json!(hdr.slice_end));
    jhdr.insert("SliceType".into(), json!(slice_type_name(hdr.slice_code)));
    jhdr.insert("Unit".into(), Value::Object(unit));
    jhdr.insert("MaxIntensity".into(), json!(hdr.cal_max));
    jhdr.insert("MinIntensity".into(), json!(hdr.cal_min));
    jhdr.insert("SliceTime".into(), json!(hdr.slice_duration));
    jhdr.insert("TimeOffset".into(), json!(hdr.toffset));
    jhdr.insert("Description".into(), json!(cstr(&hdr.descrip)));
    jhdr.insert("AuxFile".into(), json!(cstr(&hdr.aux_file)));
    jhdr.insert("QForm".into(), json!(hdr.qform_code));
    jhdr.insert("SForm".into(), json!(hdr.sform_code));
    jhdr.insert(
        "Quatern".into(),
        json!({ "b": hdr.quatern_b, "c": hdr.quatern_c, "d": hdr.quatern_d }),
    );
    jhdr.insert(
        "QuaternOffset".into(),
        json!({ "x": hdr.qoffset_x, "y": hdr.qoffset_y, "z": hdr.qoffset_z }),
    );
    jhdr.insert(
        "Affine".into(),
        json!([hdr.srow_x, hdr.srow_y, hdr.srow_z]),
    );
    jhdr.insert("Name".into(), json!(cstr(&hdr.intent_name)));
    jhdr.insert("NIIFormat".into(), json!(cstr(&hdr.magic)));
    if hdr.vox_offset != 0.0 {
        jhdr.insert("NIIByteOffset".into(), json!(hdr.vox_offset));
    }
    if !cstr(&hdr.data_type).is_empty() {
        jhdr.insert("A75DataTypeName".into(), json!(cstr(&hdr.data_type)));
    }
    if !cstr(&hdr.db_name).is_empty() {
        jhdr.insert("A75DBName".into(), json!(cstr(&hdr.db_name)));
    }
    if hdr.extents != 0 {
        jhdr.insert("A75Extends".into(), json!(hdr.extents));
    }
    if hdr.session_error != 0 {
        jhdr.insert("A75SessionError".into(), json!(hdr.session_error));
    }
    if hdr.glmax != 0 {
        jhdr.insert("A75GlobalMax".into(), json!(hdr.glmax));
    }
    if hdr.glmin != 0 {
        jhdr.insert("A75GlobalMin".into(), json!(hdr.glmin));
    }

    let n_elem = if hdr.bitpix > 0 {
        im.len() / ((hdr.bitpix as usize) >> 3).max(1)
    } else {
        im.len()
    };
    let mut data = Map::new();
    data.insert("_ArrayType_".into(), json!(info.jdata_name));
    data.insert(
        "_ArraySize_".into(),
        Value::Array(dim.iter().map(|d| json!(*d)).collect()),
    );
    data.insert("_ArrayOrder_".into(), json!("c"));
    if zip {
        let compressed = zlib_compress(im)?;
        data.insert("_ArrayZipType_".into(), json!("zlib"));
        data.insert("_ArrayZipSize_".into(), json!(n_elem));
        data.insert("_ArrayZipData_".into(), json!(b64_encode(&compressed)));
    } else {
        data.insert("_ArrayZipType_".into(), json!("base64"));
        data.insert("_ArrayZipSize_".into(), json!(n_elem));
        data.insert("_ArrayZipData_".into(), json!(b64_encode(im)));
    }

    let root = json!({
        "_DataInfo_": {
            "JNIFTIVersion": "0.5",
            "Comment": "Created by dcm2niix and NeuroJSON (http://neurojson.org)",
            "AnnotationFormat": "https://github.com/NeuroJSON/jnifti/blob/master/JNIfTI_specification.md",
            "SerialFormat": "http://json.org",
            "Parser": {
                "Python": "https://pypi.org/project/jdata\thttps://pypi.org/project/bjdata",
                "MATLAB": "https://github.com/NeuroJSON/jnifty",
                "JavaScript": "https://github.com/NeuroJSON/jsdata"
            }
        },
        "NIFTIHeader": Value::Object(jhdr),
        "NIFTIData": Value::Object(data)
    });

    let mut fp = File::create(&path).map_err(|e| io(&path, e))?;
    serde_json::to_writer_pretty(&mut fp, &root).map_err(|e| Error::convert(e.to_string()))?;
    writeln!(fp).map_err(|e| io(&path, e))?;
    Ok(path)
}

/// Minimal BJData/UBJSON writer matching C++ `nii_savebnii` structure.
fn write_bnii(
    stem: &Path,
    hdr: &Nifti1Header,
    im: &[u8],
    dim: &[i32],
    ndim: usize,
    info: &TypeInfo,
    zip: bool,
) -> Result<PathBuf> {
    let path = stem.with_extension("bnii");
    let mut fp = File::create(&path).map_err(|e| io(&path, e))?;
    let mut w = UbjsonWriter::new(&mut fp);

    w.begin_object()?;
    w.key("_DataInfo_")?;
    w.begin_object()?;
    w.str_field("JNIFTIVersion", "0.5")?;
    w.str_field(
        "Comment",
        "Created by dcm2niix and NeuroJSON (http://neurojson.org)",
    )?;
    w.str_field(
        "AnnotationFormat",
        "https://github.com/NeuroJSON/jnifti/blob/master/JNIfTI_specification.md",
    )?;
    w.str_field(
        "SerialFormat",
        "https://github.com/NeuroJSON/bjdata/blob/master/Binary_JData_Specification.md",
    )?;
    w.key("Parser")?;
    w.begin_object()?;
    w.str_field(
        "Python",
        "https://pypi.org/project/jdata\thttps://pypi.org/project/bjdata",
    )?;
    w.str_field("MATLAB", "https://github.com/NeuroJSON/jnifty")?;
    w.str_field("JavaScript", "https://github.com/NeuroJSON/jsdata")?;
    w.end_object()?;
    w.end_object()?;

    w.key("NIFTIHeader")?;
    w.begin_object()?;
    w.key("NIIHeaderSize")?;
    w.i32(hdr.sizeof_hdr)?;
    w.key("Dim")?;
    w.typed_i16_array(&hdr.dim[1..=hdr.dim[0].max(1) as usize])?;
    w.key("Param1")?;
    w.f32(hdr.intent_p1)?;
    w.key("Param2")?;
    w.f32(hdr.intent_p2)?;
    w.key("Param3")?;
    w.f32(hdr.intent_p3)?;
    w.str_field("Intent", intent_name(hdr.intent_code))?;
    w.str_field("DataType", info.nifti_name)?;
    w.key("BitDepth")?;
    w.u8(hdr.bitpix as u8)?;
    w.key("FirstSliceID")?;
    w.i16(hdr.slice_start)?;
    w.key("VoxelSize")?;
    let vs: Vec<f32> = (1..=hdr.dim[0].max(1) as usize)
        .map(|i| hdr.pixdim[i])
        .collect();
    w.typed_f32_array(&vs)?;
    w.key("Orientation")?;
    w.begin_object()?;
    w.str_field("x", if hdr.pixdim[0] != 0.0 { "l" } else { "r" })?;
    w.str_field("y", "a")?;
    w.str_field("z", "s")?;
    w.end_object()?;
    w.key("ScaleSlope")?;
    w.f32(hdr.scl_slope)?;
    w.key("ScaleOffset")?;
    w.f32(hdr.scl_inter)?;
    w.key("LastSliceID")?;
    w.i16(hdr.slice_end)?;
    w.str_field("SliceType", slice_type_name(hdr.slice_code))?;
    w.key("Unit")?;
    w.begin_object()?;
    w.str_field("L", unit_name(space_unit(hdr.xyzt_units)))?;
    w.str_field("T", unit_name(time_unit(hdr.xyzt_units)))?;
    w.end_object()?;
    w.key("MaxIntensity")?;
    w.f32(hdr.cal_max)?;
    w.key("MinIntensity")?;
    w.f32(hdr.cal_min)?;
    w.key("SliceTime")?;
    w.f32(hdr.slice_duration)?;
    w.key("TimeOffset")?;
    w.f32(hdr.toffset)?;
    w.str_field("Description", &cstr(&hdr.descrip))?;
    w.str_field("AuxFile", &cstr(&hdr.aux_file))?;
    w.key("QForm")?;
    w.i16(hdr.qform_code)?;
    w.key("SForm")?;
    w.i16(hdr.sform_code)?;
    w.key("Quatern")?;
    w.begin_object()?;
    w.key("b")?;
    w.f32(hdr.quatern_b)?;
    w.key("c")?;
    w.f32(hdr.quatern_c)?;
    w.key("d")?;
    w.f32(hdr.quatern_d)?;
    w.end_object()?;
    w.key("QuaternOffset")?;
    w.begin_object()?;
    w.key("x")?;
    w.f32(hdr.qoffset_x)?;
    w.key("y")?;
    w.f32(hdr.qoffset_y)?;
    w.key("z")?;
    w.f32(hdr.qoffset_z)?;
    w.end_object()?;
    w.key("Affine")?;
    w.affine_3x4(&hdr.srow_x, &hdr.srow_y, &hdr.srow_z)?;
    w.str_field("Name", &cstr(&hdr.intent_name))?;
    w.str_field("NIIFormat", &cstr(&hdr.magic))?;
    w.key("NIIByteOffset")?;
    w.i32(hdr.vox_offset as i32)?;
    w.end_object()?;

    let n_elem = if hdr.bitpix > 0 {
        im.len() / ((hdr.bitpix as usize) >> 3).max(1)
    } else {
        im.len()
    };
    w.key("NIFTIData")?;
    w.begin_object()?;
    w.str_field("_ArrayType_", info.jdata_name)?;
    w.key("_ArraySize_")?;
    w.typed_i32_array(dim)?;
    w.str_field("_ArrayOrder_", "c")?;
    if zip {
        let compressed = zlib_compress(im)?;
        w.str_field("_ArrayZipType_", "zlib")?;
        w.key("_ArrayZipSize_")?;
        w.i32(n_elem as i32)?;
        w.key("_ArrayZipData_")?;
        w.raw_u8_array(&compressed)?;
    } else {
        w.key("_ArrayData_")?;
        w.typed_raw_array(info.jdata_marker, n_elem, im)?;
    }
    w.end_object()?;
    w.end_object()?;
    let _ = ndim; // dims already encoded
    Ok(path)
}

struct UbjsonWriter<'a, W: Write> {
    w: &'a mut W,
}

impl<'a, W: Write> UbjsonWriter<'a, W> {
    fn new(w: &'a mut W) -> Self {
        Self { w }
    }

    fn write_all(&mut self, b: &[u8]) -> Result<()> {
        self.w.write_all(b).map_err(|e| Error::convert(e.to_string()))
    }

    fn begin_object(&mut self) -> Result<()> {
        self.write_all(b"{")
    }

    fn end_object(&mut self) -> Result<()> {
        self.write_all(b"}")
    }

    fn key(&mut self, name: &str) -> Result<()> {
        // Optimized key: N + U/len + bytes (BJData / C++ template style).
        self.write_all(b"N")?;
        self.write_str_payload(name)
    }

    fn write_str_payload(&mut self, s: &str) -> Result<()> {
        let b = s.as_bytes();
        if b.len() < 256 {
            self.write_all(b"U")?;
            self.write_all(&[b.len() as u8])?;
        } else {
            self.write_all(b"l")?;
            self.write_all(&(b.len() as u32).to_le_bytes())?;
        }
        self.write_all(b)
    }

    fn str_field(&mut self, name: &str, val: &str) -> Result<()> {
        self.key(name)?;
        self.write_all(b"S")?;
        self.write_str_payload(val)
    }

    fn u8(&mut self, v: u8) -> Result<()> {
        self.write_all(b"U")?;
        self.write_all(&[v])
    }

    fn i16(&mut self, v: i16) -> Result<()> {
        self.write_all(b"I")?;
        self.write_all(&v.to_le_bytes())
    }

    fn i32(&mut self, v: i32) -> Result<()> {
        self.write_all(b"l")?;
        self.write_all(&v.to_le_bytes())
    }

    fn f32(&mut self, v: f32) -> Result<()> {
        self.write_all(b"d")?;
        self.write_all(&v.to_le_bytes())
    }

    fn typed_i16_array(&mut self, vals: &[i16]) -> Result<()> {
        // [$I#U n ...]
        self.write_all(b"[$I#U")?;
        self.write_all(&[vals.len() as u8])?;
        for v in vals {
            self.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn typed_i32_array(&mut self, vals: &[i32]) -> Result<()> {
        self.write_all(b"[$l#U")?;
        self.write_all(&[vals.len() as u8])?;
        for v in vals {
            self.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn typed_f32_array(&mut self, vals: &[f32]) -> Result<()> {
        self.write_all(b"[$d#U")?;
        self.write_all(&[vals.len() as u8])?;
        for v in vals {
            self.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn affine_3x4(&mut self, x: &[f32; 4], y: &[f32; 4], z: &[f32; 4]) -> Result<()> {
        // [$d#[$U#U\x02\x03\x04 then 12 floats — C++ optimized 2D array.
        self.write_all(b"[$d#[$U#U")?;
        self.write_all(&[2u8, 3u8, 4u8])?;
        for row in [x, y, z] {
            for v in row {
                self.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn raw_u8_array(&mut self, data: &[u8]) -> Result<()> {
        // [$U#l count bytes
        self.write_all(b"[$U#")?;
        let n = data.len() as u32;
        self.write_all(b"l")?;
        self.write_all(&n.to_le_bytes())?;
        self.write_all(data)
    }

    fn typed_raw_array(&mut self, marker: u8, n_elem: usize, data: &[u8]) -> Result<()> {
        self.write_all(b"[$")?;
        self.write_all(&[marker])?;
        self.write_all(b"#")?;
        let n = n_elem as u32;
        self.write_all(b"l")?;
        self.write_all(&n.to_le_bytes())?;
        self.write_all(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcm_nifti::{DT_UINT16, NIFTI_UNITS_MM, NIFTI_UNITS_SEC};

    #[test]
    fn jnii_has_neurojson_sections() {
        let mut hdr = Nifti1Header::default();
        hdr.dim[0] = 3;
        hdr.dim[1] = 2;
        hdr.dim[2] = 2;
        hdr.dim[3] = 2;
        hdr.datatype = DT_UINT16;
        hdr.bitpix = 16;
        hdr.pixdim[1] = 1.0;
        hdr.pixdim[2] = 1.0;
        hdr.pixdim[3] = 1.0;
        hdr.xyzt_units = NIFTI_UNITS_MM + NIFTI_UNITS_SEC;
        let voxels = vec![0u8; 16];
        let dir = tempfile_dir();
        let stem = dir.join("t");
        let path = write_jnifti(&stem, &hdr, &voxels, false, Compress::None).unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.contains("\"_DataInfo_\""));
        assert!(s.contains("\"NIFTIHeader\""));
        assert!(s.contains("\"NIFTIData\""));
        assert!(s.contains("\"_ArrayOrder_\""));
        assert!(s.contains("\"base64\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    fn tempfile_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("jnifti_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
