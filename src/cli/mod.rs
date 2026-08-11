//! Command surface (spec Part I, section 3). This module only defines and
//! parses arguments; `app` owns what happens with them.

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

    /// Assume "yes" for setup prompts when there is no interactive
    /// terminal to ask (spec Part IX section 76).
    #[arg(long, global = true)]
    pub yes: bool,

    /// Allow binding to a non-loopback address without requiring an API
    /// key (spec Part IX section 74). Unsafe; opt-in only.
    #[arg(long, global = true)]
    pub insecure: bool,
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

impl Cli {
    pub fn build_config(&self) -> Result<Config> {
        Ok(Config {
            memory_budget_bytes: self
                .memory
                .as_deref()
                .map(parse_human_quantity)
                .transpose()?,
            context_limit_tokens: self
                .context
                .as_deref()
                .map(parse_human_quantity)
                .transpose()?,
            enable_vision: self.enable_vision,
            host: self.host.clone(),
        })
    }
}
