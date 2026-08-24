# dcm2niix-rs

Pure-Rust rewrite of [dcm2niix](https://github.com/rordenlab/dcm2niix): DICOM → NIfTI (+ BIDS sidecars).

The product converter is always Rust (`dcm-dicom` / `dcm-convert`). Voxel reorders use [rlx-tensor](https://crates.io/crates/rlx-tensor) from crates.io; an optional `gpu` feature can realize large graphs on **wgpu**. Upstream C++ is available only as `dcm2niix-ffi` for differential checks.

Broader BIDS dataset tooling lives in sibling [`bids-rs`](../bids-rs). This repo emits conversion sidecars (including BidsGuess / ReproIn); extend dataset-level coverage there.

> Research software — not a clinical device. Same disclaimer as upstream dcm2niix.

## Quick start

```bash
# rlx-tensor comes from crates.io (0.2.14). Optional ../rlx dev: uncomment [patch.crates-io] in Cargo.toml.
cargo build --release -p dcm-cli
./target/release/dcm2niix -h
./target/release/dcm2niix -b y -z n -f %p_%s -o /tmp/out /path/to/dicoms
```

## Features

| Cargo feature | Crate | Purpose |
| --- | --- | --- |
| *(default)* | `dcm-cli` | Pure Rust converter |
| `gpu` | `dcm-cli` / `dcm-convert` | Optional wgpu realize for large volume flips (`Device::Gpu`) |

C++ parity reference (workspace only, not on crates.io): `cargo build --release -p dcm-sys` → `dcm2niix-ffi`.

```bash
cargo build --release -p dcm-cli --features gpu
```

Environment (voxel backend):

| Variable | Values | Meaning |
| --- | --- | --- |
| `DCM2NIIX_RLX_DEVICE` | `auto` (default), `cpu`, `gpu` | Prefer direct CPU flips/reorient, force CPU, or force rlx/wgpu |
| `DCM2NIIX_JOBS` | positive int | Cap Rayon workers for decode / multi-series / `-a y` (default: min(CPUs, 8)) |

Typical series stay on a tight in-place CPU flip. GPU is only considered when the `gpu` feature is enabled and the volume is large (≥ ~8 MiB of `f32`).

Compress: default is uncompressed (`-z n`). For gzip, `-z y` prefers piped **pigz** when available; internal gzip defaults to level **1** (use `-6`…`-9` for smaller files).

## Parity / regression

Official gate — bit-compare against `dcm_qa*` reference trees (ignores `ConversionSoftwareVersion`; NIH also ignores `BidsGuess`):

```bash
cargo build --release -p dcm-cli -p dcm-parity
./target/release/dcm-parity --all --dcm2niix ./target/release/dcm2niix
```

Expect corpora as siblings of this repo (`../dcm_qa`, `../dcm_qa_nih`, `../dcm_qa_uih`) or pass `--corpus /path`.

**Status:** **v0.1.0** — all three `dcm_qa*` corpora pass at 100% on the Rust binary vs upstream reference outputs on the parity gate.

See [`CHANGELOG.md`](CHANGELOG.md) for the full release notes.

Beyond upstream stubs:

- GE Diffusion **2TR/3TR/ALLTR** SliceTiming (within-TR EPI pattern; ahead of C++ which still refuses cycling modes)
- ASL `BackgroundSuppression` from CSA `sAsl.ulSuppressionMode` (C++ still TODO)

## What is implemented

- Classic multi-slice + Siemens/UIH mosaic; GE EPI slice timing
- Enhanced multi-frame (per-frame functional groups)
- BIDS JSON (`-b` / `-ba` / BidsGuess `%h`); ReproIn `%H` + provenance TSV
- Physio (Siemens XA gzip-XML, CMRR blob)
- MRS SVS + MRSI including multi-DICOM CSI stacking (NIfTI-MRS); Philips multi-dynamic SVS
- GE pepolar research sequences (polarity / Y-flip); Siemens RF-off `_noRF`
- PAR/REC; NRRD / MGH / JNIfTI / BNIfTI export (`-e`)
- Compressed transfer syntaxes via `dicom-pixeldata` (native / CharLS / OpenJPEG)
- Parallel slice decode (rayon); CT gantry tilt companions

## Benchmarks

```bash
./scripts/bench_e2e.sh
cargo run --release -p dcm-convert --example flip_bench
cargo run --release -p dcm-convert --features gpu --example flip_bench
```

On QA-sized data, default CPU + direct flips outperform the wgpu build (transfer/startup dominate).

## Crates

| Crate | Role |
| --- | --- |
| `dcm-cli` | `dcm2niix` binary (`dcm-sys` builds optional `dcm2niix-ffi` locally) |
| `dcm-convert` | Scan → group → assemble → write; ReproIn, physio, PAR/REC, rlx flips / ortho |
| `dcm-dicom` | Headers, CSA/UIH, enhanced FG, pixel decode |
| `dcm-nifti` | NIfTI-1 (+ gzip / zstd) writer |
| `dcm-bids` | Core BIDS JSON sidecars |
| `dcm-core` | Errors, matrices, version string |
| `dcm-parity` | `dcm_qa*` regression harness |
| `dcm-sys` | Optional C++ static lib (parity reference only) |

## Architecture notes

1. **I/O bound** — most wall time is DICOM read/decode; flips are cheap on CPU for typical volumes.
2. **Precision** — stored samples ride in `f32` (exact for ≤16-bit integers); packing uses round-to-nearest with clamp.
3. **Affine** — LPS→RAS and Siemens mosaic sform follow upstream; avoid global FMA on all matrix multiplies (breaks Siemens QA).
4. **BIDS** — default `-ba y` omits PII; do not regress that on QA JSON.

## Continuous integration

- **`ci.yml`** — build + unit tests on every push/PR
- **`parity.yml`** — full `dcm_qa*` gate vs neurolabusc corpora

```bash
cargo test --workspace --exclude dcm-sys
cargo build --release -p dcm-cli -p dcm-parity
./target/release/dcm-parity --all
```

## License

BSD-2-Clause — see [`LICENSE`](LICENSE) (aligned with upstream dcm2niix).  
rlx is MIT OR Apache-2.0.

## Releasing

See [`RELEASE.md`](RELEASE.md) for version bumps, the parity gate, and GitHub release steps.
