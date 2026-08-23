//! Run sibling `dcm_qa*` regression corpora (upstream dcm2niix release gate).
//!
//! ```text
//! cargo build --release -p dcm-cli -p dcm-parity
//! ./target/release/dcm-parity --all --dcm2niix ./target/release/dcm2niix
//! ```
//!
//! Diffs `Ref/` vs `Out/` with `diff -br`, ignoring `ConversionSoftwareVersion`
//! (and `BidsGuess` for NIH). UIH uses `-f %p_%s_%t` and does not ignore BidsGuess.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_CORPORA: &[&str] = &["dcm_qa", "dcm_qa_nih", "dcm_qa_uih"];

fn main() -> anyhow::Result<()> {
    let mut corpora: Vec<PathBuf> = Vec::new();
    let mut dcm2niix: Option<PathBuf> = None;
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                i += 1;
                corpora.push(PathBuf::from(&args[i]));
            }
            "--dcm2niix" => {
                i += 1;
                dcm2niix = Some(PathBuf::from(&args[i]));
            }
            "--all" => {}
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
        i += 1;
    }

    let exe = dcm2niix.unwrap_or_else(|| {
        env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("dcm2niix")))
            .filter(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("target/release/dcm2niix"))
    });
    if !exe.is_file() {
        anyhow::bail!(
            "dcm2niix not found at {} — build with: cargo build --release -p dcm-cli",
            exe.display()
        );
    }

    if corpora.is_empty() {
        let shared = workspace_parent();
        for name in DEFAULT_CORPORA {
            let p = shared.join(name);
            if p.is_dir() {
                corpora.push(p);
            }
        }
    }
    if corpora.is_empty() {
        anyhow::bail!(
            "no corpora found — clone dcm_qa next to dcm2niix-rs or pass --corpus DIR"
        );
    }

    let path_env = path_with(&exe);
    let mut failed = 0usize;
    for corpus in &corpora {
        match run_corpus(corpus, &path_env) {
            Ok(()) => println!("PASS {}", corpus.display()),
            Err(e) => {
                eprintln!("FAIL {}: {e:#}", corpus.display());
                failed += 1;
            }
        }
    }
    if failed > 0 {
        anyhow::bail!("{failed}/{} corpora failed", corpora.len());
    }
    println!("All {} corpora passed.", corpora.len());
    Ok(())
}

fn run_corpus(corpus: &Path, path_env: &str) -> anyhow::Result<()> {
    let batch = corpus.join("batch.sh");
    if !batch.is_file() {
        anyhow::bail!("missing batch.sh");
    }
    println!("--- {}", corpus.display());
    let status = Command::new("bash")
        .arg(&batch)
        .env("PATH", path_env)
        .status()?;
    if !status.success() {
        anyhow::bail!("batch.sh exited with {status}");
    }
    Ok(())
}

fn path_with(exe: &Path) -> String {
    let dir = exe.parent().unwrap_or(Path::new("."));
    match env::var("PATH") {
        Ok(p) => format!("{}:{}", dir.display(), p),
        Err(_) => dir.display().to_string(),
    }
}

fn workspace_parent() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn print_usage() {
    eprintln!(
        "\
dcm-parity — dcm_qa* regression runner for the Rust/rlx `dcm2niix` binary

  Build first:
    cargo build --release -p dcm-cli -p dcm-parity

  Optional C++ reference (differential only):
    cargo build --release -p dcm-cli --features ffi

  --all              run dcm_qa, dcm_qa_nih, dcm_qa_uih (default when present)
  --corpus DIR       add a corpus (repeatable)
  --dcm2niix PATH    converter binary (default: sibling `dcm2niix`)
"
    );
}
