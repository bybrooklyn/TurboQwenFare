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
        Some(Command::Optimize) => {
            println!("tqf optimize: not yet implemented (phase 3 hardware autotune)");
            Ok(())
        }
        None => serve::start(&cli, config),
    }
}
