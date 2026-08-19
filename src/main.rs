//! tqf: TurboQwenFare entry point. One crate, one binary (spec Part IV).

mod app;
mod backend;
mod bench;
mod cli;
mod config;
mod context;
mod dev;
mod error;
mod experts;
mod format;
mod gui;
mod helper_model;
mod ids;
mod integrations;
mod io;
mod mcp;
mod memory;
mod metrics;
mod model;
mod retrieval;
mod runtime;
mod sampling;
mod server;
mod setup;
mod simd;
mod source;
mod tokenizer;
mod vision;

use clap::Parser;

fn main() -> std::process::ExitCode {
    init_tracing();

    let cli = cli::Cli::parse();

    match app::run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("tqf: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// `EnvFilter::from_default_env()` alone enables *nothing* when `RUST_LOG`
/// is unset, which silently suppressed every `tracing::warn!` in the crate
/// — including the port-fallback warning that explains why an Ollama
/// client cannot see the server. Default to `tqf=info` so operational
/// warnings are visible out of the box; `RUST_LOG` still overrides.
///
/// Logs go to stderr, not stdout. That is load-bearing rather than
/// stylistic: the MCP server (spec §95) speaks newline-delimited JSON-RPC
/// over stdout, and a single log line written there corrupts the
/// transport.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("tqf=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
