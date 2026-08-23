//! Micro-bench: flip_y / flip_z CPU vs rlx-CPU vs (optional) wgpu.
//!
//! ```bash
//! cargo run --release -p dcm-convert --example flip_bench
//! cargo run --release -p dcm-convert --features gpu --example flip_bench
//! ```

use std::time::Instant;

use dcm_convert::voxels::{flip_y_volume, flip_yz_volume, flip_z_volume};

fn median_ms(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(label: &str, iters: usize, warmup: usize, mut f: impl FnMut()) {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "{label:40}  median {:>8.3} ms  mean {:>8.3} ms  (n={iters})",
        median_ms(&samples),
        mean
    );
}

fn main() {
    let shapes = [
        ("small  64×64×36×2", 64, 64, 36, 2),
        ("medium 128×128×60×10", 128, 128, 60, 10),
        ("large  192×192×96×8", 192, 192, 96, 8), // ~108 MiB f32
    ];

    let device = std::env::var("DCM2NIIX_RLX_DEVICE").unwrap_or_else(|_| "auto".into());
    println!(
        "flip microbench  DCM2NIIX_RLX_DEVICE={device}  gpu_feature={}",
        cfg!(feature = "gpu")
    );
    println!();

    for (name, nx, ny, nz, nt) in shapes {
        let n = nx * ny * nz * nt;
        let mb = (n * 4) as f64 / (1024.0 * 1024.0);
        println!("── {name}  ({mb:.1} MiB f32) ──");
        let src: Vec<f32> = (0..n).map(|i| (i % 997) as f32).collect();

        // Force CPU direct path
        std::env::set_var("DCM2NIIX_RLX_DEVICE", "cpu");
        let iters = if mb < 50.0 { 40 } else { 8 };
        let warmup = if mb < 50.0 { 5 } else { 2 };
        {
            let s = src.clone();
            bench("flip_y (direct/CPU)", iters, warmup, || {
                let _ = flip_y_volume(s.clone(), nx, ny, nz, nt);
            });
        }
        {
            let s = src.clone();
            bench("flip_z (direct/CPU)", iters, warmup, || {
                let _ = flip_z_volume(s.clone(), nx, ny, nz, nt);
            });
        }
        {
            let s = src.clone();
            bench("flip_yz fused (direct/CPU)", iters, warmup, || {
                let _ = flip_yz_volume(s.clone(), nx, ny, nz, nt, true, true);
            });
        }

        // Force rlx path (GPU if feature+device says so)
        std::env::set_var("DCM2NIIX_RLX_DEVICE", "gpu");
        {
            let s = src.clone();
            bench("flip_y (rlx gpu|cpu)", iters.max(4).min(12), 1, || {
                let _ = flip_y_volume(s.clone(), nx, ny, nz, nt);
            });
        }
        std::env::set_var("DCM2NIIX_RLX_DEVICE", "cpu");
        // Still hits rlx when forced? DevicePref::Cpu uses direct path.
        // Use env auto + large only for gpu. For rlx-CPU explicitly we need
        // DevicePref::Gpu without feature → still to_vec_on(Cpu) via realize
        // when feature off. With feature off, Gpu pref disables direct and
        // realize uses Cpu.
        {
            let s = src.clone();
            std::env::set_var("DCM2NIIX_RLX_DEVICE", "gpu");
            bench("flip_yz fused (rlx)", iters.max(4).min(12), 1, || {
                let _ = flip_yz_volume(s.clone(), nx, ny, nz, nt, true, true);
            });
        }
        println!();
    }
}
