//! Source resolver/downloader (spec Part XVI phase 4, §276): gets verified
//! model bytes onto local disk — either by resumable HTTP range download
//! from the pinned canonical source, or by accepting an already-local file
//! (`tqf --model <path>`) — without parsing GGUF (phase 5) or touching the
//! `.tqf` container (phase 6+). Setup consumes this module's verified
//! artifact as the conversion transaction's input.
//!
//! `setup` calls into `source`, never the reverse (spec §112 phase map
//! lists "Source resolver/downloader" as its own phase, distinct from
//! "Setup/global data").

pub mod checksum;
pub mod hf;
pub mod journal;
pub mod local;
pub mod manifest;
pub mod ollama;
pub mod pin_capture;
pub mod pinned;
pub mod retry;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SourceError};
use crate::ids::Bytes;
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};
use retry::{RetryOutcome, RetryPolicy};

/// One backend TQF can pull model bytes from. Exactly the three the spec
/// names (§276): pinned HF HTTP range source, local file, and (deferred) an
/// Ollama blob locator — not a generic "any model source" abstraction.
#[async_trait::async_trait]
pub trait ModelSource: Send + Sync {
    fn metadata(&self) -> &SourceMetadata;

    /// Reads exactly `len` bytes at `offset`. Implementations must error
    /// rather than short-read or silently substitute different bytes.
    async fn read_range(
        &self,
        offset: u64,
        len: u64,
    ) -> std::result::Result<bytes::Bytes, SourceError>;

    /// `Some(path)` when this source is already fully present on local disk
    /// — `fetch_verified` then hashes it in place instead of copying it
    /// through the `.part`/journal download machinery. `None` (the
    /// default) for sources that must actually be fetched.
    fn local_path(&self) -> Option<&Path> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct SourceMetadata {
    pub artifact_name: String,
    pub size_bytes: Option<u64>,
    /// Opaque revision fingerprint (the HF ETag for the HTTP source; `None`
    /// for local files, which have no revision concept).
    pub revision: Option<String>,
    /// Published hash to verify against when known ahead of time (the §13
    /// pin). `None` for arbitrary local imports, which are hash-agnostic.
    pub expected_sha256: Option<String>,
    /// Stable identifier for where this came from (an HF repo id, or
    /// `"local"` for a local file) — recorded in the journal/manifest so a
    /// journal is never silently reused across different sources.
    pub source_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceOwnership {
    /// TQF downloaded this itself; may delete it after successful use
    /// (spec §29/§127).
    TqfManaged,
    /// The user pointed TQF at an existing file (or it's Ollama-owned);
    /// TQF must never delete it (spec §29/§127).
    UserOwned,
}

#[derive(Debug, Clone)]
pub struct FetchOptions {
    pub chunk_size_bytes: u64,
    pub retry_policy: RetryPolicy,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            chunk_size_bytes: 8 * 1024 * 1024,
            retry_policy: RetryPolicy::default(),
        }
    }
}

/// The on-disk directory a `dest_dir` should point at for a given source:
/// nested by source id and revision so two different pins never collide in
/// the same flat directory (e.g. a `--model` import followed later by a
/// pinned HF fetch of a same-named file). `~/.tqf/models/sources/<source
/// id>/<revision-or-"unpinned">/`, per caller convention — this function
/// only computes the path, callers still create it via `fetch_verified`.
pub fn default_dest_dir(models_dir: &Path, metadata: &SourceMetadata) -> std::path::PathBuf {
    let revision_component = metadata.revision.as_deref().unwrap_or("unpinned");
    models_dir
        .join("sources")
        .join(sanitize_path_component(&metadata.source_id))
        .join(sanitize_path_component(revision_component))
}

fn sanitize_path_component(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Downloads/verifies one artifact, resuming from any prior partial state.
/// Produces verified bytes on disk (or in place, for local sources) plus a
/// manifest — does not parse GGUF or touch `.tqf` (phases 5/6/8).
pub async fn fetch_verified(
    source: &dyn ModelSource,
    ownership: SourceOwnership,
    dest_dir: &Path,
    options: FetchOptions,
    broker: &MemoryBroker,
) -> Result<manifest::FetchedArtifact> {
    std::fs::create_dir_all(dest_dir)?;
    let metadata = source.metadata().clone();

    if let Some(existing) = manifest::FetchedArtifact::load(dest_dir)? {
        if existing.artifact_name == metadata.artifact_name {
            let final_path = dest_dir.join(&metadata.artifact_name);
            if let Ok(file_meta) = std::fs::metadata(&final_path) {
                if file_meta.len() == existing.size_bytes {
                    tracing::info!(
                        artifact = %metadata.artifact_name,
                        "already fetched and verified; reusing manifest"
                    );
                    return Ok(existing);
                }
            }
        }
    }

    if let Some(local_path) = source.local_path() {
        return fetch_local_in_place(local_path, &metadata, ownership, dest_dir, broker);
    }

    fetch_remote(source, &metadata, ownership, dest_dir, options, broker).await
}

fn fetch_local_in_place(
    local_path: &Path,
    metadata: &SourceMetadata,
    ownership: SourceOwnership,
    dest_dir: &Path,
    broker: &MemoryBroker,
) -> Result<manifest::FetchedArtifact> {
    let file_meta = std::fs::metadata(local_path).map_err(SourceError::LocalSourceUnavailable)?;
    let sha256 = checksum::hex_digest_file(local_path, broker)?;

    let artifact = manifest::FetchedArtifact::new(
        metadata.artifact_name.clone(),
        None,
        None,
        local_path.display().to_string(),
        file_meta.len(),
        sha256,
        ownership,
    );
    artifact.save(dest_dir)?;
    Ok(artifact)
}

async fn fetch_remote(
    source: &dyn ModelSource,
    metadata: &SourceMetadata,
    ownership: SourceOwnership,
    dest_dir: &Path,
    options: FetchOptions,
    broker: &MemoryBroker,
) -> Result<manifest::FetchedArtifact> {
    let total_size = metadata
        .size_bytes
        .ok_or_else(|| SourceError::UnknownSize {
            artifact: metadata.artifact_name.clone(),
        })?;

    let part_path = dest_dir.join(format!("{}.part", metadata.artifact_name));
    let journal_path = dest_dir.join(format!("{}.journal", metadata.artifact_name));
    let final_path = dest_dir.join(&metadata.artifact_name);

    let expected_header = journal::HeaderFields {
        source_repo_id: metadata.source_id.clone(),
        source_revision: metadata.revision.clone().unwrap_or_default(),
        artifact_name: metadata.artifact_name.clone(),
        expected_size_bytes: total_size,
        expected_sha256: metadata.expected_sha256.clone(),
        chunk_size_bytes: options.chunk_size_bytes,
    };

    let (mut writer, verified) = match journal::read(&journal_path)? {
        Some(recovered) => {
            journal::validate_header(&recovered.header, &expected_header)?;
            tracing::info!(
                artifact = %metadata.artifact_name,
                verified_chunks = recovered.verified.len(),
                "resuming interrupted download from journal"
            );
            (
                journal::JournalWriter::open_append(&journal_path)?,
                recovered.verified,
            )
        }
        None => {
            preallocate(&part_path, total_size)?;
            let header_entry = journal::JournalEntry::Header {
                schema_version: 1,
                source_repo_id: expected_header.source_repo_id.clone(),
                source_revision: expected_header.source_revision.clone(),
                artifact_name: expected_header.artifact_name.clone(),
                expected_size_bytes: expected_header.expected_size_bytes,
                expected_sha256: expected_header.expected_sha256.clone(),
                chunk_size_bytes: expected_header.chunk_size_bytes,
                started_at_unix: unix_now(),
            };
            (
                journal::JournalWriter::create(&journal_path, header_entry)?,
                Vec::new(),
            )
        }
    };

    let already_verified: std::collections::HashSet<(u64, u64)> = verified.into_iter().collect();

    let mut offset = 0u64;
    while offset < total_size {
        let len = options.chunk_size_bytes.min(total_size - offset);
        if already_verified.contains(&(offset, len)) {
            offset += len;
            continue;
        }

        let chunk_lease = broker.reserve(
            MemoryOwner::IoStaging,
            MemoryClass::Transient,
            Bytes(len),
            64,
        )?;
        let bytes = fetch_chunk_with_retry(source, offset, len, &options.retry_policy).await?;
        if bytes.len() as u64 != len {
            return Err(SourceError::ShortRead {
                artifact: metadata.artifact_name.clone(),
                offset,
                expected: len,
                actual: bytes.len() as u64,
            }
            .into());
        }
        let chunk_hash = checksum::hex_digest(&bytes);

        write_chunk(&part_path, offset, bytes).await?;

        writer.append(&journal::JournalEntry::ChunkVerified {
            offset,
            len,
            sha256: chunk_hash,
            verified_at_unix: unix_now(),
        })?;
        drop(chunk_lease);

        offset += len;
    }

    // The real correctness gate: one streaming whole-file hash pass, not
    // "the per-chunk checks all happened to pass" (spec §126: "A completed
    // extent is never trusted solely because its bytes exist").
    let whole_file_sha256 = checksum::hex_digest_file(&part_path, broker)?;
    if let Some(expected) = &metadata.expected_sha256 {
        if &whole_file_sha256 != expected {
            return Err(SourceError::ChecksumMismatch {
                artifact: metadata.artifact_name.clone(),
                expected: expected.clone(),
                actual: whole_file_sha256,
            }
            .into());
        }
    }

    writer.append(&journal::JournalEntry::Finalized {
        whole_file_sha256: whole_file_sha256.clone(),
        finalized_at_unix: unix_now(),
    })?;
    drop(writer);

    std::fs::rename(&part_path, &final_path)?;
    if let Some(parent) = final_path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }

    let artifact = manifest::FetchedArtifact::new(
        metadata.artifact_name.clone(),
        Some(metadata.source_id.clone()),
        metadata.revision.clone(),
        final_path.display().to_string(),
        total_size,
        whole_file_sha256,
        ownership,
    );
    artifact.save(dest_dir)?;
    std::fs::remove_file(&journal_path).ok();

    Ok(artifact)
}

async fn fetch_chunk_with_retry(
    source: &dyn ModelSource,
    offset: u64,
    len: u64,
    policy: &RetryPolicy,
) -> Result<bytes::Bytes> {
    retry::retry_with_backoff(policy, "source_fetch_chunk", || async {
        source.read_range(offset, len).await.map_err(classify_retry)
    })
    .await
    .map_err(Into::into)
}

/// Connection errors/timeouts/429/5xx are worth retrying; anything else
/// (404, a server that ignores range requests, an ETag that changed
/// mid-download) won't be fixed by trying again — fail loud instead.
fn classify_retry(err: SourceError) -> RetryOutcome<SourceError> {
    let retryable = matches!(err, SourceError::Network(_))
        || matches!(&err, SourceError::HttpStatus { status, .. } if is_retryable_status(*status));
    if retryable {
        RetryOutcome::Retryable(err)
    } else {
        RetryOutcome::Terminal(err)
    }
}

fn is_retryable_status(status: u16) -> bool {
    status == 429 || (500..=599).contains(&status)
}

fn preallocate(path: &Path, size: u64) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    file.set_len(size)?;
    Ok(())
}

/// Writes one chunk at its exact offset via positional `pwrite` (spec §30:
/// "Target extents may be written with `pwrite` in platform-optimal
/// order"), fsyncing before returning — the caller only trusts a chunk as
/// verified (appends the journal line) after this completes.
async fn write_chunk(path: &Path, offset: u64, bytes: bytes::Bytes) -> Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        use std::os::unix::fs::FileExt;
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        file.write_all_at(&bytes, offset)?;
        file.sync_all()?;
        Ok(())
    })
    .await
    .map_err(|join_err| crate::error::InternalError {
        incident_id: incident_id(),
        message: format!("write_chunk task panicked: {join_err}"),
    })??;
    Ok(())
}

fn incident_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("source-{nanos:x}")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod path_tests {
    use super::*;

    fn metadata(source_id: &str, revision: Option<&str>) -> SourceMetadata {
        SourceMetadata {
            artifact_name: "model.gguf".to_string(),
            size_bytes: Some(1024),
            revision: revision.map(str::to_string),
            expected_sha256: None,
            source_id: source_id.to_string(),
        }
    }

    #[test]
    fn nests_by_sanitized_source_id_and_revision() {
        let dir = default_dest_dir(
            Path::new("/home/.tqf/models"),
            &metadata("ggml-org/Qwen3.6-35B-A3B-GGUF", Some("\"abc123\"")),
        );
        assert_eq!(
            dir,
            Path::new("/home/.tqf/models/sources/ggml-org_Qwen3.6-35B-A3B-GGUF/_abc123_")
        );
    }

    #[test]
    fn missing_revision_falls_back_to_unpinned() {
        let dir = default_dest_dir(Path::new("/home/.tqf/models"), &metadata("local", None));
        assert_eq!(dir, Path::new("/home/.tqf/models/sources/local/unpinned"));
    }

    #[test]
    fn distinct_sources_never_collide() {
        let a = default_dest_dir(
            Path::new("/home/.tqf/models"),
            &metadata("repo-a", Some("rev1")),
        );
        let b = default_dest_dir(
            Path::new("/home/.tqf/models"),
            &metadata("repo-b", Some("rev1")),
        );
        assert_ne!(a, b);
    }
}
