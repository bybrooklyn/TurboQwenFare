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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = cli::Cli::parse();

    match app::run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("tqf: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}
