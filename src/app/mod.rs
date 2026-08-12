//! Top-level app orchestration: CLI dispatch and the first-run → server
//! lifecycle (spec Part V section 28).

mod serve;

use crate::cli::{Cli, Command};
use crate::error::Result;

pub fn run(cli: Cli) -> Result<()> {
    tracing::debug!(?cli, "tqf starting");
    let config = cli.build_config()?;

    match &cli.command {
        Some(Command::Sync { path }) => {
            tracing::warn!(?path, "tqf sync: not yet implemented (phase 42)");
            Ok(())
        }
        Some(Command::Unsync { path }) => {
            tracing::warn!(?path, "tqf unsync: not yet implemented (phase 42)");
            Ok(())
        }
        Some(Command::Status) => {
            println!("tqf status: not yet implemented (phase 2 server skeleton)");
            Ok(())
        }
        Some(Command::Doctor) => {
            println!("tqf doctor: not yet implemented (phase 36 receipt/index hardening)");
            Ok(())
        }
        Some(Command::Optimize) => run_optimize(),
        None => serve::start(&cli, config),
    }
}

/// `tqf optimize` (spec §3): today this is phase 10's exit-criteria
/// deliverable — a synthetic Metal bandwidth/GEMV microbenchmark proving
/// the backend device/queue/buffer/pipeline/timing plumbing works end to
/// end. The full hardware autotune (persisted machine profile, multiple
/// kernel specializations per spec §51/§77) is a later phase.
#[cfg(feature = "metal")]
fn run_optimize() -> Result<()> {
    println!("tqf optimize: running synthetic Metal bandwidth/GEMV harness (phase 10 baseline)...");
    let report = crate::bench::metal_synthetic::run_synthetic_bandwidth_gemv()?;
    println!("device: {}", report.device_name);
    println!(
        "bandwidth_copy: {} elements, {:.2} ms, {:.1} GB/s",
        report.bandwidth.elements,
        report.bandwidth.elapsed.as_secs_f64() * 1000.0,
        report.bandwidth.gigabytes_per_second
    );
    println!(
        "naive_gemv_f32: {}x{}, {:.2} ms, {:.2} GFLOP/s",
        report.gemv.rows,
        report.gemv.cols,
        report.gemv.elapsed.as_secs_f64() * 1000.0,
        report.gemv.gflops
    );
    println!(
        "(Full hardware autotune with a persisted machine profile and multiple kernel \
         specializations is a later phase, spec §51/§77 — this is the phase-10 plumbing \
         smoke test.)"
    );
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn run_optimize() -> Result<()> {
    println!("tqf optimize: not yet implemented for this backend (phase 10 baseline is Metal-only so far)");
    Ok(())
}
