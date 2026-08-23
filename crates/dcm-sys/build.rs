use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let console = dcm2niix_console(&manifest)
        .canonicalize()
        .unwrap_or_else(|_| {
            panic!(
                "dcm2niix sources not found at {} — set DCM2NIIX_ROOT or clone \
                 https://github.com/rordenlab/dcm2niix next to dcm2niix-rs",
                dcm2niix_console(&manifest).display()
            )
        });

    println!("cargo:rerun-if-changed={}", console.display());
    for name in CORE_SOURCES {
        println!("cargo:rerun-if-changed={}", console.join(name).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("src/dcm2niix_ffi.cpp").display()
    );

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        .include(&console)
        .define("myEnableJNIFTI", None)
        .warnings(false);

    let jpegls = env::var("CARGO_FEATURE_JPEGLS").is_ok();
    let want_openjpeg = env::var("CARGO_FEATURE_OPENJPEG").is_ok();

    if jpegls {
        build.define("myEnableJPEGLS", None);
        for src in CHARLS_SOURCES {
            build.file(console.join(src));
        }
    }

    if want_openjpeg {
        if let Some(cfg) = probe_openjpeg() {
            build.include(cfg.include);
            for flag in cfg.cflags {
                if let Some(stripped) = flag.strip_prefix("-D") {
                    if let Some((k, v)) = stripped.split_once('=') {
                        build.define(k, Some(v));
                    } else {
                        build.define(stripped, None);
                    }
                } else if let Some(stripped) = flag.strip_prefix("-I") {
                    build.include(stripped);
                }
            }
            for lib in cfg.libs {
                println!("cargo:rustc-link-lib={lib}");
            }
            for dir in cfg.lib_dirs {
                println!("cargo:rustc-link-search=native={dir}");
            }
        } else {
            println!(
                "cargo:warning=openjpeg feature requested but libopenjp2 not found; \
                 JPEG2000 transfer syntaxes disabled (install openjpeg or disable feature)"
            );
            build.define("myDisableOpenJPEG", None);
        }
    } else {
        build.define("myDisableOpenJPEG", None);
    }

    for src in CORE_SOURCES {
        build.file(console.join(src));
    }

    build.file(manifest.join("src/dcm2niix_ffi.cpp"));
    build.compile("dcm2niix");

    link_cpp_runtime();
    link_platform();
}

struct OpenJpegCfg {
    include: String,
    cflags: Vec<String>,
    libs: Vec<String>,
    lib_dirs: Vec<String>,
}

fn probe_openjpeg() -> Option<OpenJpegCfg> {
    let out = Command::new("pkg-config")
        .args(["--cflags-only-I", "--libs-only-l", "--libs-only-L", "libopenjp2"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut include = String::new();
    let mut cflags = Vec::new();
    let mut libs = Vec::new();
    let mut lib_dirs = Vec::new();
    for token in text.split_whitespace() {
        if let Some(p) = token.strip_prefix("-I") {
            if include.is_empty() {
                include = p.to_string();
            }
            cflags.push(token.to_string());
        } else if let Some(p) = token.strip_prefix("-L") {
            lib_dirs.push(p.to_string());
        } else if let Some(p) = token.strip_prefix("-l") {
            libs.push(p.to_string());
        } else if token.starts_with("-D") {
            cflags.push(token.to_string());
        }
    }
    if libs.is_empty() {
        return None;
    }
    Some(OpenJpegCfg {
        include,
        cflags,
        libs,
        lib_dirs,
    })
}

fn dcm2niix_console(manifest: &Path) -> PathBuf {
    if let Ok(root) = env::var("DCM2NIIX_ROOT") {
        return PathBuf::from(root).join("console");
    }
    let workspace = manifest.parent().unwrap().parent().unwrap();
    workspace
        .parent()
        .unwrap()
        .join("dcm2niix/console")
}

const CORE_SOURCES: &[&str] = &[
    "nii_foreign.cpp",
    "nii_dicom.cpp",
    "jpg_0XC3.cpp",
    "ujpeg.cpp",
    "nifti1_io_core.cpp",
    "nii_ortho.cpp",
    "nii_dicom_batch.cpp",
    "reproin.cpp",
    "dicom_fragments.cpp",
    "base64.cpp",
    "cJSON.cpp",
];

const CHARLS_SOURCES: &[&str] = &[
    "charls/jpegls.cpp",
    "charls/jpegmarkersegment.cpp",
    "charls/interface.cpp",
    "charls/jpegstreamwriter.cpp",
    "charls/jpegstreamreader.cpp",
];

fn link_cpp_runtime() {
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn link_platform() {
    // Stack size for the final `dcm2niix` binary is set in dcm-cli/build.rs
    // (`cargo:rustc-link-arg-bin`); link args from this rlib do not reach bins.
    let _ = env::var("CARGO_CFG_TARGET_OS");
}
