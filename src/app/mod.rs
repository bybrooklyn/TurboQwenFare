//! Top-level app orchestration: CLI dispatch and the first-run → server
//! lifecycle (spec Part V section 28).

mod doctor;
mod open;
mod serve;
mod status;
mod sync;

use crate::cli::{Cli, Command};
use crate::error::Result;

pub fn run(cli: Cli) -> Result<std::process::ExitCode> {
    tracing::debug!(?cli, "tqf starting");

    // Speaking MCP means this process is a subprocess of a coding client
    // with a JSON-RPC transport on stdout. It has to short-circuit
    // everything else — including config parsing side effects and any
    // `println!` — before another code path can write to stdout and
    // corrupt the transport.
    if cli.mcp_stdio {
        return run_mcp_stdio().map(|()| std::process::ExitCode::SUCCESS);
    }

    let config = cli.build_config()?;

    match &cli.command {
        Some(Command::Sync { path }) => sync::run_sync(path).map(|()| ok()),
        Some(Command::Unsync { path }) => sync::run_unsync(path).map(|()| ok()),
        Some(Command::Status) => status::run(&config).map(|()| ok()),
        Some(Command::Doctor) => doctor::run(&config),
        Some(Command::Optimize) => run_optimize().map(|()| ok()),
        None => run_server_or_gui(cli, config).map(|()| ok()),
    }
}

fn ok() -> std::process::ExitCode {
    std::process::ExitCode::SUCCESS
}

/// Serves the Model Context Protocol over stdio (spec §95, §228).
///
/// The index is `None`: nothing persists one yet, and spec §44 explicitly
/// allows the server to work without an index — its data tools then
/// return ordinary informative results rather than protocol errors.
fn run_mcp_stdio() -> Result<()> {
    use std::io::{stdin, stdout, BufReader};
    crate::mcp::stdio::run_stdio_loop(None, BufReader::new(stdin().lock()), stdout().lock())?;
    Ok(())
}

/// spec §98: headless mode runs the server exactly as before, blocking
/// the calling thread — unchanged from every earlier phase, so
/// existing headless behavior/tests are untouched. Non-headless *only*
/// changes behavior when compiled with the `gui` feature on macOS: the
/// server moves to a background thread and the real main thread is
/// handed to the compiled-in SwiftUI app, per spec §98's "transfer the
/// main thread to a C-callable Swift entrypoint."
fn run_server_or_gui(cli: Cli, config: crate::config::Config) -> Result<()> {
    // `--open` needs the server up *and* its real bound address before it
    // can write a client config, so it runs the server on a background
    // thread and waits for the address rather than assuming the default
    // port.
    if let Some(client) = cli.open.clone() {
        let kind = open::parse_client(&client)?;
        return run_with_client(cli, config, kind);
    }
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

/// Runs the server on a background thread, waits until it actually
/// answers, then launches the coding client against its real address. The
/// process exits when the client does, matching spec §224's "server exits
/// with the tqf process".
fn run_with_client(
    cli: Cli,
    config: crate::config::Config,
    kind: crate::integrations::config::ClientKind,
) -> Result<()> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let server_cli = cli.clone();
    let server =
        std::thread::spawn(move || serve::start_reporting(&server_cli, config, Some(sender)));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let addr = runtime.block_on(async move {
        match receiver.await {
            Ok(addr) if open::wait_until_ready(addr).await => Some(addr),
            Ok(addr) => {
                tracing::error!(%addr, "the server bound but never became ready");
                None
            }
            // The sender was dropped: the server thread failed before
            // binding, and its own error is the useful one.
            Err(_) => None,
        }
    });

    let Some(addr) = addr else {
        return match server.join() {
            Ok(result) => result,
            Err(_) => Err(crate::error::SetupError::ClientLaunch(
                "the server thread panicked before binding".to_string(),
            )
            .into()),
        };
    };

    let result = open::launch(&cli, kind, addr);
    // The server thread has no shutdown handle here; the process exiting
    // is what stops it, which is the documented behavior.
    drop(server);
    result
}

/// `tqf optimize` (spec §3): today this is phase 10's exit-criteria
/// deliverable — a synthetic Metal bandwidth/GEMV microbenchmark proving
/// the backend device/queue/buffer/pipeline/timing plumbing works end to
/// end. The full hardware autotune (persisted machine profile, multiple
/// kernel specializations per spec §51/§77) is a later phase.
#[cfg(tqf_metal)]
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

#[cfg(not(tqf_metal))]
fn run_optimize() -> Result<()> {
    println!("tqf optimize: not yet implemented for this backend (phase 10 baseline is Metal-only so far)");
    Ok(())
}
