//! Append-only NDJSON resume journal (spec Part XIV section 126, scoped
//! down to what Phase 4 owns: verified *source* bytes on local disk, not
//! `.tqf` target extents — that mapping is Phase 8's job).
//!
//! NDJSON rather than extending `config::persisted::atomic_write_toml`:
//! that helper rewrites the whole file per call, which is wrong for a
//! journal appended once per chunk across a multi-gigabyte download. A torn
//! last line from a crash mid-append is just a JSON-parse failure on
//! exactly the final line — trivially detectable and discardable, unlike a
//! torn binary record.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SourceError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum JournalEntry {
    Header {
        schema_version: u32,
        source_repo_id: String,
        source_revision: String,
        artifact_name: String,
        expected_size_bytes: u64,
        expected_sha256: Option<String>,
        chunk_size_bytes: u64,
        started_at_unix: u64,
    },
    ChunkVerified {
        offset: u64,
        len: u64,
        sha256: String,
        verified_at_unix: u64,
    },
    Finalized {
        whole_file_sha256: String,
        finalized_at_unix: u64,
    },
    Failed {
        offset: Option<u64>,
        error: String,
        failed_at_unix: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderFields {
    pub source_repo_id: String,
    pub source_revision: String,
    pub artifact_name: String,
    pub expected_size_bytes: u64,
    pub expected_sha256: Option<String>,
    pub chunk_size_bytes: u64,
}

#[derive(Debug)]
pub struct RecoveredState {
    pub header: HeaderFields,
    /// `(offset, len)`, in the order chunks were verified. Not assumed
    /// sorted or contiguous by callers — a chunk can only ever be skipped
    /// on resume if its exact `(offset, len)` reappears in a later request.
    pub verified: Vec<(u64, u64)>,
}

/// Compares a freshly-recovered journal header against the header the
/// current request would write, so a journal is never silently reused for
/// a different pin/artifact/chunk-size than it was created for.
pub fn validate_header(recovered: &HeaderFields, expected: &HeaderFields) -> Result<()> {
    if recovered != expected {
        return Err(SourceError::JournalCorrupt(format!(
            "journal header does not match this request: recovered {recovered:?}, expected {expected:?}"
        ))
        .into());
    }
    Ok(())
}

/// `None` if no journal exists yet (fresh start). `Err` if the journal
/// exists but is inconsistent in a way that must not be silently
/// reinterpreted (missing/misplaced header, corrupt non-trailing line).
pub fn read(path: &Path) -> Result<Option<RecoveredState>> {
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
                // Only the last line can be torn: a chunk write always
                // fsyncs the .part file before appending+fsyncing its
                // journal line, so a crash can only ever interrupt the
                // final append. Discard it rather than error.
                tracing::warn!(
                    path = %path.display(),
                    %err,
                    "discarding truncated trailing journal line (crash recovery)"
                );
                break;
            }
            Err(err) => {
                return Err(SourceError::JournalCorrupt(format!(
                    "malformed journal line {} in {}: {err}",
                    i + 1,
                    path.display()
                ))
                .into());
            }
        }
    }

    let Some(JournalEntry::Header {
        source_repo_id,
        source_revision,
        artifact_name,
        expected_size_bytes,
        expected_sha256,
        chunk_size_bytes,
        ..
    }) = entries.first().cloned()
    else {
        return Err(SourceError::JournalCorrupt(format!(
            "journal {} does not begin with a Header entry",
            path.display()
        ))
        .into());
    };

    let mut verified = Vec::new();
    for entry in &entries[1..] {
        if let JournalEntry::ChunkVerified { offset, len, .. } = entry {
            verified.push((*offset, *len));
        }
    }

    Ok(Some(RecoveredState {
        header: HeaderFields {
            source_repo_id,
            source_revision,
            artifact_name,
            expected_size_bytes,
            expected_sha256,
            chunk_size_bytes,
        },
        verified,
    }))
}

/// Holds the journal file open across chunk appends. Each `append` fsyncs
/// before returning, so a completed append is durable before the caller
/// trusts the corresponding chunk as verified.
pub struct JournalWriter {
    file: File,
}

impl JournalWriter {
    /// Truncates and starts a fresh journal with `header` as its first
    /// entry. Callers must only do this when `read()` returned `None`
    /// (fresh start) — resuming an existing journal uses `open_append`.
    pub fn create(path: &Path, header: JournalEntry) -> Result<Self> {
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

    pub fn open_append(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self { file })
    }

    pub fn append(&mut self, entry: &JournalEntry) -> Result<()> {
        let mut line = serde_json::to_string(entry)
            .map_err(|e| SourceError::JournalCorrupt(format!("failed to serialize entry: {e}")))?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn header(chunk_size: u64) -> JournalEntry {
        JournalEntry::Header {
            schema_version: 1,
            source_repo_id: "ggml-org/Qwen3.6-35B-A3B-GGUF".to_string(),
            source_revision: "deadbeef".to_string(),
            artifact_name: "model.gguf".to_string(),
            expected_size_bytes: 1024,
            expected_sha256: Some("abc123".to_string()),
            chunk_size_bytes: chunk_size,
            started_at_unix: 0,
        }
    }

    fn test_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tqf-journal-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn missing_journal_is_none() {
        let path = test_path("missing.journal");
        assert!(read(&path).unwrap().is_none());
    }

    #[test]
    fn round_trips_header_and_chunks() {
        let path = test_path("roundtrip.journal");
        let mut writer = JournalWriter::create(&path, header(256)).unwrap();
        writer
            .append(&JournalEntry::ChunkVerified {
                offset: 0,
                len: 256,
                sha256: "hash0".to_string(),
                verified_at_unix: 1,
            })
            .unwrap();
        writer
            .append(&JournalEntry::ChunkVerified {
                offset: 256,
                len: 256,
                sha256: "hash1".to_string(),
                verified_at_unix: 2,
            })
            .unwrap();

        let recovered = read(&path).unwrap().unwrap();
        assert_eq!(recovered.header.expected_size_bytes, 1024);
        assert_eq!(recovered.verified, vec![(0, 256), (256, 256)]);
    }

    #[test]
    fn tolerates_truncated_trailing_line() {
        let path = test_path("truncated.journal");
        let mut writer = JournalWriter::create(&path, header(256)).unwrap();
        writer
            .append(&JournalEntry::ChunkVerified {
                offset: 0,
                len: 256,
                sha256: "hash0".to_string(),
                verified_at_unix: 1,
            })
            .unwrap();
        // Simulate a crash mid-append: a syntactically broken final line.
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"event\":\"ChunkVeri").unwrap();

        let recovered = read(&path).unwrap().unwrap();
        assert_eq!(recovered.verified, vec![(0, 256)]);
    }

    #[test]
    fn corrupt_non_trailing_line_is_an_error() {
        let path = test_path("corrupt.journal");
        std::fs::write(
            &path,
            "not even close to json\n{\"event\":\"ChunkVerified\",\"offset\":0,\"len\":1,\"sha256\":\"x\",\"verified_at_unix\":0}\n",
        )
        .unwrap();
        let err = read(&path).unwrap_err();
        assert!(err.to_string().contains("malformed journal line"));
    }

    #[test]
    fn header_mismatch_is_rejected() {
        let path = test_path("mismatch.journal");
        JournalWriter::create(&path, header(256)).unwrap();
        let recovered = read(&path).unwrap().unwrap();

        let mut expected = recovered.header.clone();
        expected.chunk_size_bytes = 512; // different from what's on disk
        assert!(validate_header(&recovered.header, &expected).is_err());

        assert!(validate_header(&recovered.header, &recovered.header.clone()).is_ok());
    }
}
