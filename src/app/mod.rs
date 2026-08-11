//! First-run lifecycle and top-level app orchestration (spec Part V, section 28).

use crate::cli::Cli;

pub fn run(cli: Cli) -> crate::error::Result<()> {
    tracing::info!(?cli, "tqf starting");
    crate::cli::dispatch(cli)
}
