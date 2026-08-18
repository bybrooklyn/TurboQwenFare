//! Drives the section-28 first-run flow through to an installed, validated
//! container and, once available, a bound running server.
//! Plain `tqf` drives canonical source fetch, conversion, and receipt
//! finalization as one transaction. It refuses to start a generation server
//! until a real Qwen runtime can be constructed from that trusted container.

use std::sync::Arc;
use std::time::Instant;

use crate::cli::Cli;
use crate::config::persisted::PersistedConfig;
use crate::config::{paths, Config};
use crate::error::{Result, SetupError};
use crate::format::tqf::convert_canonical_gguf;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;
use crate::model::qwen36::weights::Qwen36WeightManifest;
use crate::runtime::{GenerationSlot, Qwen36Generator, Qwen36ResidentReferenceGenerator};
use crate::server::{self, bind, AppState};
use crate::setup::flow::{self, SetupOutcome};
use crate::setup::hardware::{self, HardwareProfile};
use crate::setup::receipt;
use crate::source::{self, local::LocalFileSource, manifest::FetchedArtifact, ModelSource};

/// Diagnostic-only escape hatch so the server skeleton can be started and
/// exercised manually before real model install exists (spec Part I
/// section 3: "Developer/debug controls may exist behind ... environment
/// flags"). Never documented in `--help`.
const DEV_SKIP_MODEL_CHECK_ENV: &str = "TQF_DEV_UNSAFE_SKIP_MODEL_CHECK";
/// Enables the deliberately high-memory Phase-14/15 reference profile. It
/// is for parity qualification and real protocol wiring, not normal use.
const DEV_RESIDENT_REFERENCE_ENV: &str = "TQF_DEV_RESIDENT_REFERENCE";
const DEV_RESIDENT_STREAMING_ENV: &str = "TQF_DEV_RESIDENT_STREAMING";
/// Enables the Phase-18 whole-expert streaming reference graph. It still
/// uses the same bounded graph as normal startup; this switch only makes the
/// developer qualification profile explicit in startup output.
const DEV_STREAMING_REFERENCE_ENV: &str = "TQF_DEV_STREAMING_REFERENCE";

pub fn start(cli: &Cli, cli_config: Config) -> Result<()> {
    let config_path = paths::config_path()?;
    let mut persisted = PersistedConfig::load(&config_path)?;
    apply_cli_overrides(&mut persisted, &cli_config);
    let config = effective_config(cli_config, &persisted);
    let memory_budget = Bytes(config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024));
    let setup_broker = MemoryBroker::new(memory_budget);

    let first_run = flow::run(cli.yes, &setup_broker)?;
    let mut hardware = first_run.hardware;
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
        SetupOutcome::ProceedInstall => {
            // Fetch verification, lossless conversion, validated reopen, and
            // receipt publication run in that order. Any failure returns
            // false and leaves either a resumable conversion transaction or
            // no receipt — never a false installed state.
            runtime.block_on(attempt_source_fetch(cli, &hardware, &setup_broker))?;
            true
        }
    };

    if model_installed && hardware.quick_tune.is_none() {
        match hardware::run_short_autotune(&mut hardware) {
            Ok(()) => {
                crate::config::persisted::atomic_write_toml(&paths::profile_path()?, &hardware)?;
                if let Some(tune) = &hardware.quick_tune {
                    println!(
                        "tqf: short hardware tune complete on {} ({:.1} GB/s copy, {:.2} GFLOP/s baseline GEMV).",
                        tune.device_name,
                        tune.bandwidth_gigabytes_per_second,
                        tune.naive_gemv_gflops
                    );
                }
            }
            Err(error) => {
                tracing::warn!(%error, "short hardware tune failed; using baseline settings")
            }
        }
    }

    persisted.setup_completed = true;
    persisted.save(&config_path)?;

    if model_installed {
        if std::env::var(DEV_STREAMING_REFERENCE_ENV).as_deref() == Ok("1") {
            let receipts_dir = paths::receipts_dir()?;
            let receipt =
                receipt::load_trusted_receipt(&receipts_dir, &setup_broker).ok_or_else(|| {
                    crate::error::ModelError::Unsupported(
                        "trusted receipt disappeared or failed validation before runtime load"
                            .to_string(),
                    )
                })?;
            let budget = config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024);
            let context = config.context_limit_tokens.unwrap_or(128 * 1024) as usize;
            let generator: Arc<dyn Qwen36Generator> =
                Arc::new(Qwen36ResidentReferenceGenerator::open_streaming(
                    &receipt.tqf_path,
                    &receipt.tokenizer_gguf_path,
                    Bytes(budget),
                    context,
                    Bytes(budget / 4),
                )?);
            println!(
                "tqf: starting the developer whole-expert streaming reference server over the \
                 bounded runtime."
            );
            return runtime.block_on(run_server(cli, config, true, Some(generator)));
        }
        if std::env::var(DEV_RESIDENT_STREAMING_ENV).as_deref() == Ok("1") {
            let receipts_dir = paths::receipts_dir()?;
            let receipt =
                receipt::load_trusted_receipt(&receipts_dir, &setup_broker).ok_or_else(|| {
                    crate::error::ModelError::Unsupported(
                        "trusted receipt disappeared or failed validation before resident \
                         streaming runtime load"
                            .to_string(),
                    )
                })?;
            let budget = config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024);
            let context = config.context_limit_tokens.unwrap_or(128 * 1024) as usize;
            let generator: Arc<dyn Qwen36Generator> =
                Arc::new(Qwen36ResidentReferenceGenerator::open_resident_streaming(
                    &receipt.tqf_path,
                    &receipt.tokenizer_gguf_path,
                    Bytes(budget),
                    context,
                    Bytes(budget / 4),
                )?);
            println!("tqf: starting the Phase 25 resident-core streaming reference server.");
            return runtime.block_on(run_server(cli, config, true, Some(generator)));
        }
        if std::env::var(DEV_RESIDENT_REFERENCE_ENV).as_deref() == Ok("1") {
            let receipts_dir = paths::receipts_dir()?;
            let receipt =
                receipt::load_trusted_receipt(&receipts_dir, &setup_broker).ok_or_else(|| {
                    crate::error::ModelError::Unsupported(
                        "trusted receipt disappeared or failed validation before runtime load"
                            .to_string(),
                    )
                })?;
            let budget = config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024);
            let context = config.context_limit_tokens.unwrap_or(128 * 1024) as usize;
            let generator: Arc<dyn Qwen36Generator> =
                Arc::new(Qwen36ResidentReferenceGenerator::open(
                    &receipt.tqf_path,
                    &receipt.tokenizer_gguf_path,
                    Bytes(budget),
                    context,
                )?);
            println!(
                "tqf: starting the high-memory resident reference server; this is a developer \
                 parity profile, not the normal bounded-memory runtime."
            );
            return runtime.block_on(run_server(cli, config, true, Some(generator)));
        }
        let receipts_dir = paths::receipts_dir()?;
        let receipt =
            receipt::load_trusted_receipt(&receipts_dir, &setup_broker).ok_or_else(|| {
                crate::error::ModelError::Unsupported(
                    "trusted receipt disappeared or failed validation before bounded runtime load"
                        .to_string(),
                )
            })?;
        let budget = config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024);
        let context = config.context_limit_tokens.unwrap_or(128 * 1024) as usize;
        let generator: Arc<dyn Qwen36Generator> =
            Arc::new(Qwen36ResidentReferenceGenerator::open_streaming(
                &receipt.tqf_path,
                &receipt.tokenizer_gguf_path,
                Bytes(budget),
                context,
                Bytes(budget / 4),
            )?);
        println!("tqf: starting bounded Qwen3.6 server.");
        return runtime.block_on(run_server(cli, config, true, Some(generator)));
    }
    if std::env::var(DEV_SKIP_MODEL_CHECK_ENV).as_deref() != Ok("1") {
        return Ok(());
    }
    if !model_installed {
        tracing::warn!(
            "{DEV_SKIP_MODEL_CHECK_ENV}=1: starting the server skeleton with no model \
             installed. Diagnostic use only."
        );
    }

    runtime.block_on(run_server(cli, config, model_installed, None))
}

/// Drives source verification and canonical GGUF conversion through the
/// actual CLI. `--model <path>` preserves user ownership; the default uses
/// the pinned checkpoint. Only `finish_install` writes a receipt.
async fn attempt_source_fetch(
    cli: &Cli,
    hardware: &HardwareProfile,
    broker: &MemoryBroker,
) -> Result<()> {
    let models_dir = paths::models_dir()?;

    if let Some(model_path) = &cli.model {
        let local_source = LocalFileSource::open(model_path.clone())?;
        let dest_dir = source::default_dest_dir(&models_dir, local_source.metadata());
        let artifact = source::fetch_verified(
            &local_source,
            source::SourceOwnership::UserOwned,
            &dest_dir,
            source::FetchOptions::default(),
            broker,
        )
        .await?;
        return finish_install(artifact, &models_dir, broker);
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

    let hf_source = source::hf::HfRangeSource::resolve(
        source::pinned::REPO_ID,
        source::pinned::REVISION,
        source::pinned::LANGUAGE_CHECKPOINT_FILENAME,
        Some(source::pinned::LANGUAGE_CHECKPOINT_SHA256.to_string()),
    )
    .await?;
    let dest_dir = source::default_dest_dir(&models_dir, hf_source.metadata());
    let mut artifact = source::fetch_verified(
        &hf_source,
        source::SourceOwnership::TqfManaged,
        &dest_dir,
        source::FetchOptions::default(),
        broker,
    )
    .await?;
    // The CDN ETag is useful for resume isolation, but the trusted receipt's
    // revision field is the immutable repository commit from the release
    // pin—not an opaque transport fingerprint.
    artifact.source_revision = Some(source::pinned::REVISION.to_string());
    artifact.save(&dest_dir)?;
    finish_install(artifact, &models_dir, broker)
}

fn finish_install(
    artifact: FetchedArtifact,
    models_dir: &std::path::Path,
    broker: &MemoryBroker,
) -> Result<()> {
    let source_path = std::path::PathBuf::from(&artifact.local_path);
    let destination = models_dir.join("qwen3.6-35b-a3b.tqf");
    let report = convert_canonical_gguf(&source_path, &artifact.sha256, &destination, broker)?;
    Qwen36WeightManifest::open_with_broker(&report.path, broker)?;
    let receipts_dir = paths::receipts_dir()?;
    receipt::write_trusted_receipt(
        &receipts_dir,
        &report,
        artifact.source_revision,
        source_path,
        broker,
    )?;
    println!(
        "tqf: installed and validated {} as {} ({} bytes, sha256={}).",
        artifact.artifact_name,
        report.path.display(),
        report.verified_output_bytes,
        artifact.sha256
    );
    Ok(())
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

async fn run_server(
    cli: &Cli,
    config: Config,
    model_installed: bool,
    generator: Option<Arc<dyn Qwen36Generator>>,
) -> Result<()> {
    let bound = bind::resolve_and_bind(config.host.as_deref(), cli.insecure).await?;
    tracing::info!(addr = %bound.addr, "tqf listening");
    println!("tqf listening on http://{}", bound.addr);

    let state = AppState {
        config: Arc::new(config),
        model_installed,
        generation_slot: GenerationSlot::new(),
        generator,
        started_at: Instant::now(),
        api_key: bound.api_key.map(|k| Arc::from(k.as_str())),
    };

    server::serve(bound.listener, state).await?;
    Ok(())
}
