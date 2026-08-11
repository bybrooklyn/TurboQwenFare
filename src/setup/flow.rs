//! First-run lifecycle (spec Part V section 28): detect hardware, load
//! persisted config/profile, validate the trusted receipt, and — if no
//! model is installed — ask before doing anything. Real download/repack
//! land in phases 4-8; today this only decides *whether* setup should
//! proceed, and never leaves half-written state on disk either way (spec
//! Part V: "an interrupted... conversion leaves a resumable partial
//! installation, never a model directory that appears valid").

use std::io::{IsTerminal, Write};

use crate::config::paths;
use crate::config::persisted::atomic_write_toml;
use crate::error::Result;
use crate::setup::hardware::{self, HardwareProfile};
use crate::setup::receipt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupOutcome {
    /// A valid trusted receipt already exists; safe to start the server.
    Ready,
    /// User was asked and said no.
    DeclinedByUser,
    /// No tty and no `--yes`: refuses to guess rather than either hanging
    /// on a prompt nobody can answer or silently starting a multi-gigabyte
    /// download.
    NonInteractiveConfirmationRequired,
    /// User agreed to proceed, but download/repack (phases 4-8) isn't
    /// built yet. No partial receipt is ever written for this outcome.
    ProceedNotYetImplemented,
}

pub struct FirstRunResult {
    pub outcome: SetupOutcome,
    pub hardware: HardwareProfile,
}

/// Runs the section-28 state machine up through the setup decision.
/// `assume_yes` mirrors `--yes`, letting non-interactive callers (CI,
/// scripts, `--headless` automation) opt into setup without a tty.
pub fn run(assume_yes: bool) -> Result<FirstRunResult> {
    paths::ensure_layout()?;

    let hardware = hardware::detect();
    atomic_write_toml(&paths::profile_path()?, &hardware)?;

    let receipts_dir = paths::receipts_dir()?;
    if receipt::load_trusted_receipt(&receipts_dir).is_some() {
        return Ok(FirstRunResult {
            outcome: SetupOutcome::Ready,
            hardware,
        });
    }

    let outcome = if std::io::stdin().is_terminal() {
        let proceed = prompt_yes_no(
            "Download and optimize the canonical Qwen3.6-35B-A3B Q4 model now? [Y/n] ",
        )?;
        if proceed {
            SetupOutcome::ProceedNotYetImplemented
        } else {
            SetupOutcome::DeclinedByUser
        }
    } else if assume_yes {
        SetupOutcome::ProceedNotYetImplemented
    } else {
        SetupOutcome::NonInteractiveConfirmationRequired
    };

    Ok(FirstRunResult { outcome, hardware })
}

fn prompt_yes_no(question: &str) -> Result<bool> {
    print!("{question}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let answer = line.trim().to_ascii_lowercase();
    Ok(answer.is_empty() || answer == "y" || answer == "yes")
}
