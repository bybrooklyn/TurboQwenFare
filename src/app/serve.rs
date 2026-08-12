//! Drives the section-28 first-run flow through to a bound, running server.
//! Plain `tqf` today almost always stops at the setup prompt, honestly,
//! because model download/repack (phases 4-8) doesn't exist yet — this is
//! not a shortcut around that, it's the real state machine.

use std::sync::Arc;
use std::time::Instant;

use crate::cli::Cli;
use crate::config::persisted::PersistedConfig;
use crate::config::{paths, Config};
use crate::error::{Result, SetupError};
use crate::runtime::GenerationSlot;
use crate::server::{self, bind, AppState};
use crate::setup::flow::{self, SetupOutcome};
use crate::setup::hardware::HardwareProfile;
use crate::source::{self, local::LocalFileSource, manifest::FetchedArtifact, ModelSource};

/// Diagnostic-only escape hatch so the server skeleton can be started and
/// exercised manually before real model install exists (spec Part I
/// section 3: "Developer/debug controls may exist behind ... environment
/// flags"). Never documented in `--help`.
const DEV_SKIP_MODEL_CHECK_ENV: &str = "TQF_DEV_UNSAFE_SKIP_MODEL_CHECK";

pub fn start(cli: &Cli, cli_config: Config) -> Result<()> {
    let config_path = paths::config_path()?;
    let mut persisted = PersistedConfig::load(&config_path)?;
    apply_cli_overrides(&mut persisted, &cli_config);
    let config = effective_config(cli_config, &persisted);

    let first_run = flow::run(cli.yes)?;
    let runtime = tokio::runtime::Runtime::new()?;

    let model_installed = match first_run.outcome {
        SetupOutcome::Ready => true,
        SetupOutcome::DeclinedByUser => {
            persisted.setup_completed = true;
            persisted.save(&config_path)?;
            println!("tqf: setup declined; nothing more to do.");
            return Ok(());
        }
        SetupOutcome::NonInteractiveConfirmationRequired => {
            // Deliberately not persisted as "completed": nothing was
            // actually decided, so a later run should ask again.
            println!(
                "tqf: no model installed and no interactive terminal to confirm setup.\n\
                 Re-run with --yes to proceed non-interactively, or run `tqf` in a terminal."
            );
            return Err(SetupError::NonInteractiveConfirmationRequired.into());
        }
        SetupOutcome::ProceedNotYetImplemented => {
            // Phase 4 exists now: attempt an actual source fetch (local
            // --model import, or the pinned HF source) through the real
            // CLI. GGUF import/.tqf repack (phases 5-8) still don't exist,
            // so this always reports `model_installed = false` regardless
            // of fetch outcome — no receipt is written, matching
            // `SetupOutcome::ProceedNotYetImplemented`'s existing contract
            // that no partial receipt is ever written for this outcome.
            runtime.block_on(attempt_source_fetch(cli, &first_run.hardware))
        }
    };

    persisted.setup_completed = true;
    persisted.save(&config_path)?;

    if !model_installed && std::env::var(DEV_SKIP_MODEL_CHECK_ENV).as_deref() != Ok("1") {
        return Ok(());
    }
    if !model_installed {
        tracing::warn!(
            "{DEV_SKIP_MODEL_CHECK_ENV}=1: starting the server skeleton with no model \
             installed. Diagnostic use only."
        );
    }

    runtime.block_on(run_server(cli, config, model_installed))
}

/// Exercises the phase-4 source resolver/downloader through the real CLI:
/// `--model <path>` imports a local file (spec §29 experimental import
/// path); otherwise this attempts the pinned canonical HF source. Always
/// returns `false` for `model_installed` — GGUF parsing and `.tqf` repack
/// (phases 5-8) don't exist yet, so a successful fetch is not an installed
/// model, just verified bytes on disk ready for those later phases.
async fn attempt_source_fetch(cli: &Cli, hardware: &HardwareProfile) -> bool {
    let models_dir = match paths::models_dir() {
        Ok(dir) => dir,
        Err(err) => {
            println!("tqf: could not resolve model storage directory: {err}");
            return false;
        }
    };

    if let Some(model_path) = &cli.model {
        match LocalFileSource::open(model_path.clone()) {
            Ok(local_source) => {
                let dest_dir = source::default_dest_dir(&models_dir, local_source.metadata());
                report_fetch(
                    source::fetch_verified(
                        &local_source,
                        source::SourceOwnership::UserOwned,
                        &dest_dir,
                        source::FetchOptions::default(),
                    )
                    .await,
                );
            }
            Err(err) => println!(
                "tqf: could not open --model path {}: {err}",
                model_path.display()
            ),
        }
        return false;
    }

    println!(
        "tqf: no model installed. Fetching the pinned canonical checkpoint ({}, {} bytes) — \
         this can take a while.\nDetected hardware: {} {}, {} cores, backend={}.",
        source::pinned::LANGUAGE_CHECKPOINT_FILENAME,
        source::pinned::LANGUAGE_CHECKPOINT_SIZE_BYTES,
        hardware.os,
        hardware.arch,
        hardware.cpu_cores,
        hardware.backend,
    );

    match source::hf::HfRangeSource::resolve(
        source::pinned::REPO_ID,
        source::pinned::REVISION,
        source::pinned::LANGUAGE_CHECKPOINT_FILENAME,
        Some(source::pinned::LANGUAGE_CHECKPOINT_SHA256.to_string()),
    )
    .await
    {
        Ok(hf_source) => {
            let dest_dir = source::default_dest_dir(&models_dir, hf_source.metadata());
            report_fetch(
                source::fetch_verified(
                    &hf_source,
                    source::SourceOwnership::TqfManaged,
                    &dest_dir,
                    source::FetchOptions::default(),
                )
                .await,
            );
        }
        Err(err) => println!("tqf: could not resolve pinned source: {err}"),
    }

    false
}

fn report_fetch(result: Result<FetchedArtifact>) {
    match result {
        Ok(artifact) => println!(
            "tqf: fetched and verified {} ({} bytes, sha256={}).\n\
             GGUF import/.tqf repack isn't implemented yet (spec phases 5-8).",
            artifact.artifact_name, artifact.size_bytes, artifact.sha256
        ),
        Err(err) => println!("tqf: source fetch failed: {err}"),
    }
}

/// Only overwrites a persisted field when the CLI actually provided a
/// value, so a flag omitted on this run doesn't erase a value saved by an
/// earlier one (spec Part IX section 76: subsequent starts are
/// zero-question).
fn apply_cli_overrides(persisted: &mut PersistedConfig, cli_config: &Config) {
    if let Some(bytes) = cli_config.memory_budget_bytes {
        persisted.memory_budget_bytes = Some(bytes);
    }
    if let Some(tokens) = cli_config.context_limit_tokens {
        persisted.context_limit_tokens = Some(tokens);
    }
    if cli_config.enable_vision {
        persisted.enable_vision = true;
    }
    if let Some(host) = &cli_config.host {
        persisted.host = Some(host.clone());
    }
}

/// CLI flags win when present; otherwise fall back to what a previous run
/// persisted.
fn effective_config(cli_config: Config, persisted: &PersistedConfig) -> Config {
    Config {
        memory_budget_bytes: cli_config
            .memory_budget_bytes
            .or(persisted.memory_budget_bytes),
        context_limit_tokens: cli_config
            .context_limit_tokens
            .or(persisted.context_limit_tokens),
        enable_vision: cli_config.enable_vision || persisted.enable_vision,
        host: cli_config.host.or_else(|| persisted.host.clone()),
    }
}

async fn run_server(cli: &Cli, config: Config, model_installed: bool) -> Result<()> {
    let bound = bind::resolve_and_bind(config.host.as_deref(), cli.insecure).await?;
    tracing::info!(addr = %bound.addr, "tqf listening");
    println!("tqf listening on http://{}", bound.addr);

    let state = AppState {
        config: Arc::new(config),
        model_installed,
        generation_slot: GenerationSlot::new(),
        started_at: Instant::now(),
        api_key: bound.api_key.map(|k| Arc::from(k.as_str())),
    };

    server::serve(bound.listener, state).await?;
    Ok(())
}
