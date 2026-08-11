//! Command surface (spec Part I, section 3). Plain `tqf` starts the server
//! (and the desktop GUI unless `--headless`); subcommands are diagnostic or
//! index-management actions.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::config::{parse_human_quantity, Config};
use crate::error::Result;

#[derive(Parser, Debug)]
#[command(
    name = "tqf",
    version,
    about = "TurboQwenFare: bounded-memory Qwen3.6-35B-A3B inference server"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Run without launching the desktop GUI.
    #[arg(long, global = true)]
    pub headless: bool,

    /// Live working-set memory budget, e.g. "4G". A hard contract, not an
    /// advisory cache size.
    #[arg(long, global = true, value_name = "SIZE")]
    pub memory: Option<String>,

    /// Logical context limit, e.g. "128K" or "1M".
    #[arg(long, global = true, value_name = "SIZE")]
    pub context: Option<String>,

    /// Load the vision encoder lazily on the first multimodal request.
    #[arg(long, global = true)]
    pub enable_vision: bool,

    /// Bind address; defaults to loopback only (spec Part IX section 74).
    #[arg(long, global = true, value_name = "HOST")]
    pub host: Option<String>,

    /// Import a local compatible Qwen3.6 Q4 checkpoint instead of the
    /// pinned canonical source.
    #[arg(long, global = true, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Launch a coding client wired to this server: opencode, claude, codex.
    #[arg(long, global = true, value_name = "CLIENT")]
    pub open: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Index a directory into TQIndex.
    Sync { path: PathBuf },
    /// Remove a directory from TQIndex.
    Unsync { path: PathBuf },
    /// Print current server/model/memory status.
    Status,
    /// Run environment and installation diagnostics.
    Doctor,
    /// Run hardware autotune.
    Optimize,
}

pub fn dispatch(cli: Cli) -> Result<()> {
    // Global flags are validated up front regardless of subcommand, so a
    // malformed --memory/--context never gets silently ignored just because
    // the chosen subcommand happens not to read it yet.
    let config = build_config(&cli)?;

    match cli.command {
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
        Some(Command::Optimize) => {
            println!("tqf optimize: not yet implemented (phase 3 hardware autotune)");
            Ok(())
        }
        None => {
            tracing::info!(?config, headless = cli.headless, "server start: not yet implemented (phase 2)");
            Ok(())
        }
    }
}

fn build_config(cli: &Cli) -> Result<Config> {
    Ok(Config {
        memory_budget_bytes: cli.memory.as_deref().map(parse_human_quantity).transpose()?,
        context_limit_tokens: cli.context.as_deref().map(parse_human_quantity).transpose()?,
        enable_vision: cli.enable_vision,
        host: cli.host.clone(),
    })
}
