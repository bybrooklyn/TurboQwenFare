//! Streaming conversion transaction (spec Part XIV §126, phase 8):
//! resumable `.tqf` writes driven by a per-extent append-only journal, so
//! a kill mid-conversion resumes after re-verifying journaled extents that
//! already landed, without rewriting them. Mirrors `source::journal`'s NDJSON design (a torn
//! trailing line is crash recovery, not corruption) applied to output
//! *target* extents rather than *source* download chunks — this is a
//! deliberately separate journal from Phase 4's source-download journal
//! (spec §127: conversion and source-fetch are distinct concerns).
//!
//! "Conversion progress is reported as verified output bytes, not merely
//! downloaded input bytes" (§126): `ConversionTransaction::verified_bytes`
//! sums only extents/experts that have actually been journaled, i.e.
//! written to the `.partial` file *and* fsynced as a journal entry.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ContainerError, Result};
use crate::ids::{ExpertId, LayerId};

use super::records::TqfSectionKind;
use super::writer::{RecoveredExpert, RecoveredExtent, TqfHeaderInfo, TqfWriter};

/// State machine from spec §126. `WritingMetadataSkeleton` and
/// `VerifyingPayload` are not independently observable states in this
/// implementation (each `write_extent`/`write_expert` call is itself an
/// atomic write-then-journal step), so this enum names only the states a
/// caller can actually be in between calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionState {
    Absent,
    CreatingPartial,
    WritingExtents,
    WritingFinalTables,
    Installed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
enum JournalEntry {
    Header {
        schema_version: u32,
        final_path: String,
        source_sha256_hex: String,
        started_at_unix: u64,
    },
    ExtentVerified {
        extent: RecoveredExtent,
        verified_at_unix: u64,
    },
    ExpertVerified {
        expert: RecoveredExpert,
        verified_at_unix: u64,
    },
    Finalized {
        finalized_at_unix: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HeaderFields {
    final_path: String,
    source_sha256_hex: String,
}

struct RecoveredState {
    header: HeaderFields,
    extents: Vec<RecoveredExtent>,
    experts: Vec<RecoveredExpert>,
    finalized: bool,
}

/// `None` if no journal exists yet. `Err` if the journal exists but is
/// inconsistent in a way that must not be silently reinterpreted (missing
/// header, corrupt non-trailing line) — identical posture to
/// `source::journal::read`.
fn read_journal(path: &Path) -> Result<Option<RecoveredState>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return Ok(None);
    }

    let mut entries = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        match serde_json::from_str::<JournalEntry>(line) {
            Ok(entry) => entries.push(entry),
            Err(err) if i == lines.len() - 1 => {
                // Each append fsyncs the .partial payload write before
                // fsyncing its own journal line, so a crash can only ever
                // interrupt the *final* journal append.
                tracing::warn!(
                    path = %path.display(),
                    %err,
                    "discarding truncated trailing conversion-journal line (crash recovery)"
                );
                break;
            }
            Err(err) => {
                tracing::error!(
                    path = %path.display(),
                    %err,
                    "corrupt non-trailing conversion journal line"
                );
                return Err(ContainerError::MalformedRecord {
                    table: "conversion journal",
                }
                .into());
            }
        }
    }

    let Some(JournalEntry::Header {
        final_path,
        source_sha256_hex,
        ..
    }) = entries.first().cloned()
    else {
        return Err(ContainerError::MalformedRecord {
            table: "conversion journal (missing header)",
        }
        .into());
    };

    let mut extents = Vec::new();
    let mut experts = Vec::new();
    let mut finalized = false;
    for entry in &entries[1..] {
        match entry {
            JournalEntry::ExtentVerified { extent, .. } => extents.push(extent.clone()),
            JournalEntry::ExpertVerified { expert, .. } => experts.push(expert.clone()),
            JournalEntry::Finalized { .. } => finalized = true,
            JournalEntry::Header { .. } => {
                return Err(ContainerError::MalformedRecord {
                    table: "conversion journal (duplicate header)",
                }
                .into());
            }
        }
    }

    Ok(Some(RecoveredState {
        header: HeaderFields {
            final_path,
            source_sha256_hex,
        },
        extents,
        experts,
        finalized,
    }))
}

struct JournalWriter {
    file: File,
}

impl JournalWriter {
    fn create(path: &Path, header: JournalEntry) -> Result<Self> {
        debug_assert!(matches!(header, JournalEntry::Header { .. }));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = Self { file };
        writer.append(&header)?;
        Ok(writer)
    }

    fn open_append(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self { file })
    }

    fn append(&mut self, entry: &JournalEntry) -> Result<()> {
        let mut line =
            serde_json::to_string(entry).map_err(|_| ContainerError::MalformedRecord {
                table: "conversion journal (serialize)",
            })?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.sync_all()?;
        Ok(())
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn journal_path_for(final_path: &Path) -> PathBuf {
    let mut name = final_path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".convert.journal");
    final_path.with_file_name(name)
}

/// Outcome of `ConversionTransaction::begin`: either a live transaction to
/// drive extent-by-extent, or a signal that a prior run already reached
/// `Installed` (the final file exists, its journal just wasn't cleaned up
/// — e.g. a crash between `commit()` and journal removal).
pub enum BeginOutcome {
    Transaction(ConversionTransaction),
    AlreadyInstalled,
}

impl std::fmt::Debug for BeginOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeginOutcome::Transaction(_) => write!(f, "BeginOutcome::Transaction(..)"),
            BeginOutcome::AlreadyInstalled => write!(f, "BeginOutcome::AlreadyInstalled"),
        }
    }
}

pub struct ConversionTransaction {
    writer: TqfWriter,
    journal: JournalWriter,
    journal_path: PathBuf,
    state: ConversionState,
}

impl ConversionTransaction {
    /// Starts a fresh conversion, or resumes an interrupted one, for
    /// `final_path`. `source_sha256_hex` binds the journal to a specific
    /// source artifact — resuming with a different source is rejected
    /// rather than silently mixing extents from two conversions (same
    /// posture as `source::journal::validate_header`).
    pub fn begin(
        final_path: impl Into<PathBuf>,
        header: TqfHeaderInfo,
        source_sha256_hex: &str,
    ) -> Result<BeginOutcome> {
        let final_path = final_path.into();

        if final_path.exists() {
            return Ok(BeginOutcome::AlreadyInstalled);
        }

        let journal_path = journal_path_for(&final_path);
        let expected_header = HeaderFields {
            final_path: final_path.display().to_string(),
            source_sha256_hex: source_sha256_hex.to_string(),
        };

        match read_journal(&journal_path)? {
            Some(recovered) => {
                if recovered.header != expected_header {
                    return Err(ContainerError::MalformedRecord {
                        table: "conversion journal (header mismatch for this request)",
                    }
                    .into());
                }
                if recovered.finalized {
                    tracing::warn!(
                        path = %final_path.display(),
                        "conversion journal was finalized before the installed file appeared; retrying the atomic commit"
                    );
                }
                tracing::info!(
                    path = %final_path.display(),
                    verified_extents = recovered.extents.len(),
                    verified_experts = recovered.experts.len(),
                    "resuming interrupted .tqf conversion from journal"
                );
                let writer = TqfWriter::resume_partial(
                    &final_path,
                    header,
                    &recovered.extents,
                    &recovered.experts,
                )?;
                let journal = JournalWriter::open_append(&journal_path)?;
                Ok(BeginOutcome::Transaction(ConversionTransaction {
                    writer,
                    journal,
                    journal_path,
                    state: ConversionState::WritingExtents,
                }))
            }
            None => {
                let writer = TqfWriter::create_partial(&final_path, header)?;
                let journal = JournalWriter::create(
                    &journal_path,
                    JournalEntry::Header {
                        schema_version: 1,
                        final_path: expected_header.final_path,
                        source_sha256_hex: expected_header.source_sha256_hex,
                        started_at_unix: unix_now(),
                    },
                )?;
                Ok(BeginOutcome::Transaction(ConversionTransaction {
                    writer,
                    journal,
                    journal_path,
                    state: ConversionState::CreatingPartial,
                }))
            }
        }
    }

    pub fn state(&self) -> ConversionState {
        self.state
    }

    pub fn has_extent(&self, name: &str) -> bool {
        self.writer.has_extent(name)
    }

    pub fn has_expert(&self, layer: LayerId, expert: ExpertId) -> bool {
        self.writer.has_expert(layer, expert)
    }

    /// Writes one extent and durably journals it before returning — the
    /// two together are what "verified" means (spec §126: a completed
    /// extent's journal hash must validate, not just "its bytes exist").
    #[allow(clippy::too_many_arguments)]
    pub fn write_extent(
        &mut self,
        role_id: u32,
        name: &str,
        layer: Option<LayerId>,
        section_kind: TqfSectionKind,
        dims: &[u64],
        dtype_id: u32,
        quant_layout_id: u32,
        required_alignment: u32,
        data: &[u8],
    ) -> Result<()> {
        self.state = ConversionState::WritingExtents;
        let recovered = self.writer.write_extent(
            role_id,
            name,
            layer,
            section_kind,
            dims,
            dtype_id,
            quant_layout_id,
            required_alignment,
            data,
        )?;
        self.writer.sync_payload()?;
        self.journal.append(&JournalEntry::ExtentVerified {
            extent: recovered,
            verified_at_unix: unix_now(),
        })?;
        Ok(())
    }

    pub fn write_expert(
        &mut self,
        layer: LayerId,
        expert: ExpertId,
        quant_layout_id: u16,
        gate_up: &[u8],
        down: &[u8],
    ) -> Result<()> {
        self.state = ConversionState::WritingExtents;
        let recovered = self
            .writer
            .write_expert(layer, expert, quant_layout_id, gate_up, down)?;
        self.writer.sync_payload()?;
        self.journal.append(&JournalEntry::ExpertVerified {
            expert: recovered,
            verified_at_unix: unix_now(),
        })?;
        Ok(())
    }

    /// Journaled form of `TqfWriter::write_expert_parts`, used by canonical
    /// conversion to avoid a second gate+up allocation per routed expert.
    pub fn write_expert_parts(
        &mut self,
        layer: LayerId,
        expert: ExpertId,
        quant_layout_id: u16,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
    ) -> Result<()> {
        self.state = ConversionState::WritingExtents;
        let recovered =
            self.writer
                .write_expert_parts(layer, expert, quant_layout_id, gate, up, down)?;
        self.writer.sync_payload()?;
        self.journal.append(&JournalEntry::ExpertVerified {
            expert: recovered,
            verified_at_unix: unix_now(),
        })?;
        Ok(())
    }

    /// Finalizes metadata tables, atomically renames `.partial` to the
    /// real file (`TqfWriter::commit`), and only then removes the
    /// conversion journal — matching §126's
    /// `WritingFinalTables -> FsyncPartial -> AtomicRename -> Installed`
    /// tail (writing the trusted receipt is the caller's job, one layer
    /// up, once this returns `Ok`).
    pub fn finish(self) -> Result<()> {
        let ConversionTransaction {
            writer,
            mut journal,
            journal_path,
            ..
        } = self;

        // Commit writes and fsyncs the final tables and superblock before the
        // atomic rename. The journal must not claim finalization first: a
        // crash in that gap must remain an ordinary resumable transaction.
        writer.commit()?;
        journal.append(&JournalEntry::Finalized {
            finalized_at_unix: unix_now(),
        })?;
        std::fs::remove_file(&journal_path).ok();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tqf-conversion-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn header() -> TqfHeaderInfo {
        TqfHeaderInfo {
            backend_id: 1,
            feature_bits: 0,
            model_family_id: [0xAA; 16],
            source_sha256: [0xBB; 32],
            conversion_fingerprint: [0xCC; 32],
        }
    }

    fn extent_bytes(fill: u8, len: usize) -> Vec<u8> {
        vec![fill; len]
    }

    #[test]
    fn fresh_conversion_writes_and_installs() {
        let path = fixture_path("fresh.tqf");
        let outcome = ConversionTransaction::begin(&path, header(), "deadbeef").unwrap();
        let mut txn = match outcome {
            BeginOutcome::Transaction(t) => t,
            BeginOutcome::AlreadyInstalled => panic!("expected a fresh transaction"),
        };
        assert_eq!(txn.state(), ConversionState::CreatingPartial);

        txn.write_extent(
            1,
            "token_embedding",
            None,
            TqfSectionKind::Embeddings,
            &[16, 4],
            12,
            12,
            64,
            &extent_bytes(0xAB, 64),
        )
        .unwrap();
        txn.finish().unwrap();

        assert!(path.exists());
        assert!(!journal_path_for(&path).exists());

        let reader = super::super::TqfReader::open_validated(&path).unwrap();
        let extent = reader.tensor(1, None).unwrap();
        assert_eq!(
            reader.read_extent_bytes(extent).unwrap(),
            extent_bytes(0xAB, 64)
        );
    }

    #[test]
    fn kill_after_partial_write_resumes_without_rewriting_verified_extents() {
        let path = fixture_path("kill-resume.tqf");

        // First "run": write one extent, then simulate a crash by
        // dropping the transaction without calling finish().
        {
            let outcome = ConversionTransaction::begin(&path, header(), "cafef00d").unwrap();
            let mut txn = match outcome {
                BeginOutcome::Transaction(t) => t,
                BeginOutcome::AlreadyInstalled => panic!("unexpected"),
            };
            txn.write_extent(
                1,
                "token_embedding",
                None,
                TqfSectionKind::Embeddings,
                &[16, 4],
                12,
                12,
                64,
                &extent_bytes(0xAB, 64),
            )
            .unwrap();
            // Deliberately no finish(): journal has one verified extent,
            // no Finalized entry, and the .partial file (not the real
            // file) is left on disk.
        }
        assert!(!path.exists());
        assert!(journal_path_for(&path).exists());

        // Second "run": resume. The already-verified extent must be
        // visible via has_extent() without re-writing it, and a second
        // extent completes the conversion.
        let outcome = ConversionTransaction::begin(&path, header(), "cafef00d").unwrap();
        let mut txn = match outcome {
            BeginOutcome::Transaction(t) => t,
            BeginOutcome::AlreadyInstalled => panic!("expected a resumed transaction"),
        };
        assert!(txn.has_extent("token_embedding"));
        assert!(!txn.has_extent("layers.0.q_proj"));

        txn.write_extent(
            2,
            "layers.0.q_proj",
            Some(LayerId(0)),
            TqfSectionKind::ResidentCore,
            &[8, 4],
            12,
            12,
            64,
            &extent_bytes(0xCD, 32),
        )
        .unwrap();
        txn.finish().unwrap();

        assert!(path.exists());
        assert!(!journal_path_for(&path).exists());

        let reader = super::super::TqfReader::open_validated(&path).unwrap();
        let embed = reader.tensor(1, None).unwrap();
        assert_eq!(
            reader.read_extent_bytes(embed).unwrap(),
            extent_bytes(0xAB, 64)
        );
        let q_proj = reader.tensor(2, Some(LayerId(0))).unwrap();
        assert_eq!(
            reader.read_extent_bytes(q_proj).unwrap(),
            extent_bytes(0xCD, 32)
        );
    }

    #[test]
    fn finalized_journal_without_installed_file_retries_commit() {
        let path = fixture_path("finalized-before-rename.tqf");
        {
            let outcome = ConversionTransaction::begin(&path, header(), "source-a").unwrap();
            let mut txn = match outcome {
                BeginOutcome::Transaction(t) => t,
                BeginOutcome::AlreadyInstalled => panic!("unexpected"),
            };
            txn.write_extent(
                1,
                "token_embedding",
                None,
                TqfSectionKind::Embeddings,
                &[16, 4],
                12,
                12,
                64,
                &extent_bytes(0xAB, 64),
            )
            .unwrap();

            // Reproduce the ordering used by older binaries: the journal
            // reached Finalized, but the process died before commit/rename.
            txn.journal
                .append(&JournalEntry::Finalized {
                    finalized_at_unix: unix_now(),
                })
                .unwrap();
        }

        assert!(!path.exists());
        let outcome = ConversionTransaction::begin(&path, header(), "source-a").unwrap();
        let txn = match outcome {
            BeginOutcome::Transaction(t) => t,
            BeginOutcome::AlreadyInstalled => panic!("expected a resumed transaction"),
        };
        assert!(txn.has_extent("token_embedding"));
        txn.finish().unwrap();

        assert!(path.exists());
        assert!(!journal_path_for(&path).exists());
        super::super::TqfReader::open_validated(&path).unwrap();
    }

    #[test]
    fn resuming_with_a_different_source_hash_is_rejected() {
        let path = fixture_path("mismatched-source.tqf");
        {
            let outcome = ConversionTransaction::begin(&path, header(), "source-a").unwrap();
            let mut txn = match outcome {
                BeginOutcome::Transaction(t) => t,
                BeginOutcome::AlreadyInstalled => panic!("unexpected"),
            };
            txn.write_extent(
                1,
                "token_embedding",
                None,
                TqfSectionKind::Embeddings,
                &[16, 4],
                12,
                12,
                64,
                &extent_bytes(0xAB, 64),
            )
            .unwrap();
        }

        let err = ConversionTransaction::begin(&path, header(), "source-b").unwrap_err();
        assert!(err.to_string().contains("conversion journal"));
    }

    #[test]
    fn resume_rejects_corrupted_payload_claimed_by_the_journal() {
        let path = fixture_path("corrupt-resume.tqf");
        {
            let outcome = ConversionTransaction::begin(&path, header(), "source-a").unwrap();
            let mut txn = match outcome {
                BeginOutcome::Transaction(t) => t,
                BeginOutcome::AlreadyInstalled => panic!("unexpected"),
            };
            txn.write_extent(
                1,
                "token_embedding",
                None,
                TqfSectionKind::Embeddings,
                &[16, 4],
                12,
                12,
                64,
                &extent_bytes(0xAB, 64),
            )
            .unwrap();
        }

        use std::os::unix::fs::FileExt;
        let mut partial_name = path.file_name().unwrap().to_os_string();
        partial_name.push(".partial");
        let partial = path.with_file_name(partial_name);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(partial)
            .unwrap();
        file.write_all_at(&[0xEE], 4096).unwrap();
        file.sync_all().unwrap();

        let error = ConversionTransaction::begin(&path, header(), "source-a").unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn verified_bytes_are_reported_not_downloaded_bytes() {
        // "Conversion progress is reported as verified output bytes, not
        // merely downloaded input bytes" (§126): has_extent only reports
        // true once write_extent has both written *and* journaled the
        // extent, never speculatively.
        let path = fixture_path("verified-not-downloaded.tqf");
        let outcome = ConversionTransaction::begin(&path, header(), "hash").unwrap();
        let mut txn = match outcome {
            BeginOutcome::Transaction(t) => t,
            BeginOutcome::AlreadyInstalled => panic!("unexpected"),
        };
        assert!(!txn.has_extent("token_embedding"));
        txn.write_extent(
            1,
            "token_embedding",
            None,
            TqfSectionKind::Embeddings,
            &[16, 4],
            12,
            12,
            64,
            &extent_bytes(0xAB, 64),
        )
        .unwrap();
        assert!(txn.has_extent("token_embedding"));
        txn.finish().unwrap();
    }

    #[test]
    fn already_installed_final_file_short_circuits() {
        let path = fixture_path("already-installed.tqf");
        {
            let outcome = ConversionTransaction::begin(&path, header(), "hash").unwrap();
            let txn = match outcome {
                BeginOutcome::Transaction(t) => t,
                BeginOutcome::AlreadyInstalled => panic!("unexpected"),
            };
            txn.finish().unwrap();
        }
        assert!(path.exists());

        let outcome = ConversionTransaction::begin(&path, header(), "hash").unwrap();
        assert!(matches!(outcome, BeginOutcome::AlreadyInstalled));
    }
}
