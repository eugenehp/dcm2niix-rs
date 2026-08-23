//! Bounded Rayon pool for convert / decode / multi-series work.
//!
//! Caps concurrency so large folders do not thrash disk or oversubscribe
//! cores. Override with `DCM2NIIX_JOBS` (positive integer).

use std::sync::OnceLock;

use rayon::ThreadPool;

/// Default max worker threads (clamp of available parallelism).
pub const DEFAULT_JOB_CAP: usize = 8;

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Effective job limit: `DCM2NIIX_JOBS` or `min(available, 8)`.
pub fn job_limit() -> usize {
    if let Ok(s) = std::env::var("DCM2NIIX_JOBS") {
        if let Ok(n) = s.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, DEFAULT_JOB_CAP)
}

fn pool() -> &'static ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(job_limit())
            .thread_name(|i| format!("dcm2niix-{i}"))
            .build()
            .expect("dcm2niix rayon pool")
    })
}

/// Run `f` on the bounded pool (nested Rayon work shares these threads).
pub fn install<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    pool().install(f)
}
