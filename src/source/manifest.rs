//! `manifest.toml`: a staging record of one fetched/verified artifact,
//! written after `fetch_verified` succeeds. Not the trusted `ModelReceipt`
//! (spec Part V section 36) — full §125 provenance and receipt finalization
//! land with the `.tqf` container work in phases 6-8. This is deliberately
//! smaller: just enough for those later phases to find and trust the bytes
//! Phase 4 produced, and for `SourceOwnership` bookkeeping (spec §29/§127:
//! TQF must never delete a user-pointed source).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::persisted::atomic_write_toml;
use crate::error::Result;
use crate::source::SourceOwnership;

pub const MANIFEST_FILE_NAME: &str = "manifest.toml";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FetchedArtifact {
    pub schema_version: u32,
    pub artifact_name: String,
    pub source_repo_id: Option<String>,
    pub source_revision: Option<String>,
    pub local_path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub ownership: SourceOwnership,
    pub fetched_at_unix: u64,
}

impl FetchedArtifact {
    pub fn new(
        artifact_name: String,
        source_repo_id: Option<String>,
        source_revision: Option<String>,
        local_path: String,
        size_bytes: u64,
        sha256: String,
        ownership: SourceOwnership,
    ) -> Self {
        Self {
            schema_version: 1,
            artifact_name,
            source_repo_id,
            source_revision,
            local_path,
            size_bytes,
            sha256,
            ownership,
            fetched_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        atomic_write_toml(&dir.join(MANIFEST_FILE_NAME), self)
    }

    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join(MANIFEST_FILE_NAME);
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(toml::from_str(&text).ok()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_atomic_write() {
        let dir = std::env::temp_dir().join(format!("tqf-manifest-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let artifact = FetchedArtifact::new(
            "model.gguf".to_string(),
            Some("ggml-org/Qwen3.6-35B-A3B-GGUF".to_string()),
            Some("deadbeef".to_string()),
            dir.join("model.gguf").display().to_string(),
            1024,
            "abc123".to_string(),
            SourceOwnership::TqfManaged,
        );
        artifact.save(&dir).unwrap();

        let loaded = FetchedArtifact::load(&dir).unwrap().unwrap();
        assert_eq!(loaded, artifact);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_manifest_is_none() {
        let dir = std::env::temp_dir().join(format!("tqf-manifest-missing-{}", std::process::id()));
        assert!(FetchedArtifact::load(&dir).unwrap().is_none());
    }
}
