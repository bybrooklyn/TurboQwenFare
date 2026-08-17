//! First-run lifecycle (spec Part V section 28): detect hardware, load
//! persisted config/profile, validate the trusted receipt, and — if no
//! model is installed — ask before downloading. The caller drives the real
//! source and conversion transactions only after `ProceedInstall`; this
//! module decides *whether* setup should proceed and never leaves
//! half-written state on disk either way (spec
//! Part V: "an interrupted... conversion leaves a resumable partial
//! installation, never a model directory that appears valid").

use std::io::{IsTerminal, Write};

use crate::config::paths;
use crate::config::persisted::atomic_write_toml;
use crate::error::Result;
use crate::memory::MemoryBroker;
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
    /// User agreed to the canonical download/repack transaction. A trusted
    /// receipt is still written only after the converted container validates.
    ProceedInstall,
}

pub struct FirstRunResult {
    pub outcome: SetupOutcome,
    pub hardware: HardwareProfile,
}

/// Runs the section-28 state machine up through the setup decision.
/// `assume_yes` mirrors `--yes`, letting non-interactive callers (CI,
/// scripts, `--headless` automation) opt into setup without a tty.
pub fn run(assume_yes: bool, broker: &MemoryBroker) -> Result<FirstRunResult> {
    paths::ensure_layout()?;

    let profile_path = paths::profile_path()?;
    let mut hardware = hardware::detect();
    if let Ok(text) = std::fs::read_to_string(&profile_path) {
        if let Ok(previous) = toml::from_str::<HardwareProfile>(&text) {
            hardware.preserve_compatible_quick_tune(&previous);
        }
    }
    atomic_write_toml(&profile_path, &hardware)?;

    let receipts_dir = paths::receipts_dir()?;
    if receipt::load_trusted_receipt(&receipts_dir, broker).is_some() {
        return Ok(FirstRunResult {
            outcome: SetupOutcome::Ready,
            hardware,
        });
    }

    let outcome = if assume_yes {
        SetupOutcome::ProceedInstall
    } else if std::io::stdin().is_terminal() {
        let proceed = prompt_yes_no(
            "Download and optimize the canonical Qwen3.6-35B-A3B Q4 model now? [Y/n] ",
        )?;
        if proceed {
            SetupOutcome::ProceedInstall
        } else {
            SetupOutcome::DeclinedByUser
        }
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
