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
        None => run_server_or_gui(cli, config),
    }
}

/// spec §98: headless mode runs the server exactly as before, blocking
/// the calling thread — unchanged from every earlier phase, so
/// existing headless behavior/tests are untouched. Non-headless *only*
/// changes behavior when compiled with the `gui` feature on macOS: the
/// server moves to a background thread and the real main thread is
/// handed to the compiled-in SwiftUI app, per spec §98's "transfer the
/// main thread to a C-callable Swift entrypoint."
fn run_server_or_gui(cli: Cli, config: crate::config::Config) -> Result<()> {
    if cli.headless {
        return serve::start(&cli, config);
    }
    #[cfg(all(target_os = "macos", feature = "gui"))]
    {
        run_with_gui(cli, config)
    }
    #[cfg(not(all(target_os = "macos", feature = "gui")))]
    {
        serve::start(&cli, config)
    }
}

/// Runs the server on a background OS thread (it owns its own Tokio
/// runtime, spec §25's async/thread model is unaffected) and hands the
/// real main thread to `gui::macos::launch`, which never returns until
/// the user quits the app. `base_url` uses the server's real default
/// bind address (spec's Ollama-compatible default port) — if the
/// default port was actually occupied and the server's own bind-with-
/// fallback logic picked a different one, the GUI's hardcoded default
/// won't match; resolving that requires threading the real bound
/// address back from `serve::start` to this caller, which is not
/// implemented yet (see the Phase 46 qualification doc).
#[cfg(all(target_os = "macos", feature = "gui"))]
fn run_with_gui(cli: Cli, config: crate::config::Config) -> Result<()> {
    let server_thread = std::thread::spawn(move || serve::start(&cli, config));
    let base_url = format!("http://127.0.0.1:{}", crate::server::bind::DEFAULT_PORT);
    crate::gui::macos::launch(&base_url);
    // The GUI returned (the user quit the app). The background server
    // thread keeps running for the rest of the process's life, exactly
    // like a headless server would; this function returning lets
    // `main` exit, which is the same "close the window, quit the app"
    // behavior every other macOS app has.
    let _ = server_thread;
    Ok(())
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
