# Release checklist

## Version locations

| File | Field |
| --- | --- |
| `Cargo.toml` | `[workspace.package] version` and `[workspace.dependencies]` path crate versions |
| `crates/dcm-core/src/lib.rs` | `VERSION`, `VERSION_DATE` (`v1.0.YYYYMMDD` stamp for `--version` / BIDS) |
| `CHANGELOG.md` | New `## [x.y.z] — YYYY-MM-DD` section; keep `## Unreleased` at top |

## Pre-release gate

Requires sibling checkouts:

- `../rlx` — `rlx-tensor` path dependency
- `../dcm_qa`, `../dcm_qa_nih`, `../dcm_qa_uih` — parity corpora (optional locally; CI clones them)

```bash
cargo test --workspace --exclude dcm-sys
cargo build --release -p dcm-cli -p dcm-parity
./target/release/dcm2niix --version
./target/release/dcm-parity --all --dcm2niix ./target/release/dcm2niix
```

All three corpora must **PASS**.

Optional smoke:

```bash
cargo build --release -p dcm-cli --features gpu
cargo build --release -p dcm-cli --features ffi   # needs ../dcm2niix for dcm-sys
```

## Tag and GitHub release

```bash
git tag -a v0.1.0 -m "dcm2niix-rs 0.1.0"
git push origin v0.1.0
```

Attach release notes from `CHANGELOG.md` `[0.1.0]` section.

## Build artifacts (local)

```bash
cargo build --release -p dcm-cli
# Binary: target/release/dcm2niix
```

For redistribution, ship `dcm2niix` plus a note that **rlx** must be present at build time (not a runtime dependency of the release binary).

## CI

- **`ci.yml`** — build + unit tests on every push/PR
- **`parity.yml`** — full `dcm_qa*` gate on `main` / `master` when converter crates change
