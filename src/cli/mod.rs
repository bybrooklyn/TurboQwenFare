//! Command surface (spec Part I, section 3). This module only defines and
//! parses arguments; `app` owns what happens with them.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::config::{parse_human_quantity, Config};
use crate::error::Result;

#[derive(Parser, Debug, Clone)]
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

    /// Bind port. Defaults to Ollama's 11434, falling back to 11435 if
    /// that is occupied (spec Part IX section 69). An explicitly
    /// requested port is never silently moved: if it is busy, that is an
    /// error, because a client was told to use it.
    #[arg(long, global = true, value_name = "PORT")]
    pub port: Option<u16>,

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

    /// Speak the Model Context Protocol over stdio instead of starting a
    /// server (spec §95, §228).
    ///
    /// Hidden rather than a listed subcommand: spec §3 fixes the
    /// user-facing command surface, and this is not something a person
    /// runs by hand — it is the command `--open` writes into a coding
    /// client's config so the client can launch it. Adding a visible row
    /// to a table users read, for a flag only another program uses,
    /// would make the product look more complicated than it is.
    #[arg(long, global = true, hide = true)]
    pub mcp_stdio: bool,
}

#[derive(Subcommand, Debug, Clone)]
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
        let memory_budget_bytes = self
            .memory
            .as_deref()
            .map(parse_human_quantity)
            .transpose()?;
        // Caught here rather than at startup: `--memory 1K` parsed
        // cleanly and the server booted, then failed on whatever
        // reservation happened to come first — a message about a cache
        // tile, nowhere near the flag responsible.
        if let Some(bytes) = memory_budget_bytes {
            if bytes < crate::config::MINIMUM_MEMORY_BUDGET_BYTES {
                return Err(crate::error::ConfigError::MemoryBudgetTooSmall {
                    given: crate::config::human_bytes(bytes),
                    floor: crate::config::human_bytes(crate::config::MINIMUM_MEMORY_BUDGET_BYTES),
                }
                .into());
            }
        }
        // Both of these used to surface only after the first-run setup
        // gate, so a typo in a path was reported as "no model installed
        // and no interactive terminal to confirm setup" — a message
        // about a different problem entirely. An invalid argument is
        // knowable before any of that runs.
        if let Some(path) = &self.model {
            if !path.exists() {
                return Err(crate::error::ConfigError::ModelPathMissing(
                    path.display().to_string(),
                )
                .into());
            }
            if path.is_dir() {
                return Err(crate::error::ConfigError::ModelPathNotAFile(
                    path.display().to_string(),
                )
                .into());
            }
        }
        if let Some(host) = &self.host {
            if host.parse::<std::net::IpAddr>().is_err() {
                return Err(crate::error::ConfigError::InvalidHost(host.clone()).into());
            }
        }
        Ok(Config {
            memory_budget_bytes,
            context_limit_tokens: self
                .context
                .as_deref()
                .map(parse_human_quantity)
                .transpose()?,
            enable_vision: self.enable_vision,
            host: self.host.clone(),
            port: self.port,
        })
    }
}

#[cfg(test)]
mod memory_floor_tests {
    use super::*;
    use clap::Parser;

    fn config_for(args: &[&str]) -> crate::error::Result<Config> {
        Cli::try_parse_from(args).unwrap().build_config()
    }

    /// `--memory 1K` used to parse cleanly and start a server, which then
    /// failed on whatever reservation came first — a message about a
    /// cache tile, nowhere near the flag that caused it. Spec §40 defines
    /// no profile below 2 GiB, so the flag is where it is refused.
    #[test]
    fn a_budget_below_the_experimental_floor_is_refused_at_the_flag() {
        for below in ["1K", "512M", "1G", "2047M"] {
            let error = config_for(&["tqf", "--memory", below])
                .expect_err(&format!("--memory {below} must be refused"));
            let message = error.to_string();
            assert!(message.contains("experimental floor"), "{message}");
            // The message has to name the way out, not just the problem.
            assert!(message.contains("2G"), "{message}");
            assert!(message.contains("4G"), "{message}");
        }
    }

    /// The floor itself is allowed: spec §40 calls 2 GiB experimental,
    /// not forbidden.
    #[test]
    fn the_floor_and_above_are_accepted() {
        for allowed in ["2G", "4G", "8G", "2048M"] {
            let config = config_for(&["tqf", "--memory", allowed])
                .unwrap_or_else(|e| panic!("--memory {allowed} must be accepted: {e}"));
            assert!(
                config.memory_budget_bytes.unwrap() >= crate::config::MINIMUM_MEMORY_BUDGET_BYTES
            );
        }
    }

    /// A bad path or host used to reach the first-run setup gate before
    /// anything looked at it, so the user was told "no model installed
    /// and no interactive terminal to confirm setup" — a real message
    /// about a different problem. Each argument is now refused where it
    /// was typed, naming the value.
    #[test]
    fn invalid_paths_and_hosts_are_named_rather_than_reported_as_setup_failures() {
        let missing = config_for(&["tqf", "--model", "/nonexistent/typo.gguf"])
            .expect_err("a nonexistent --model must be refused")
            .to_string();
        assert!(missing.contains("typo.gguf"), "{missing}");
        assert!(missing.contains("does not exist"), "{missing}");

        // A directory is a distinct mistake from a missing file — most
        // often a user pointing at the folder the checkpoint is in.
        let dir = std::env::temp_dir();
        let is_dir = config_for(&["tqf", "--model", dir.to_str().unwrap()])
            .expect_err("a directory --model must be refused")
            .to_string();
        assert!(is_dir.contains(".gguf"), "{is_dir}");

        let host = config_for(&["tqf", "--host", "not-an-ip"])
            .expect_err("a non-IP --host must be refused")
            .to_string();
        assert!(host.contains("not-an-ip"), "{host}");

        // And the valid forms still pass, including the non-loopback
        // bind that mints an API key.
        for ok in ["127.0.0.1", "0.0.0.0", "::1"] {
            config_for(&["tqf", "--host", ok])
                .unwrap_or_else(|e| panic!("--host {ok} must be accepted: {e}"));
        }
        let real_file = std::env::current_dir().unwrap().join("Cargo.toml");
        config_for(&["tqf", "--model", real_file.to_str().unwrap()])
            .expect("an existing file must be accepted here; format checks come later");
    }

    /// No `--memory` at all still means the 4 GiB default, which the
    /// floor check must not turn into an error.
    #[test]
    fn an_absent_flag_is_not_a_budget_of_zero() {
        assert_eq!(
            config_for(&["tqf"]).unwrap().memory_budget_bytes,
            None,
            "an unset flag must stay unset, not become 0 and trip the floor"
        );
    }
}
