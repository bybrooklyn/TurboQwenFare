//! Machine-global persisted config (spec Part IX section 76): first-run
//! decisions are saved so subsequent starts are zero-question. Writes are
//! atomic (write to a sibling temp file, then rename) so a crash mid-save
//! can never corrupt the previously good config — the same transactional
//! discipline the spec requires of model installation (Part V section 28).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ConfigError, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PersistedConfig {
    pub memory_budget_bytes: Option<u64>,
    pub context_limit_tokens: Option<u64>,
    pub enable_vision: bool,
    pub host: Option<String>,
    /// First-run setup reached a terminal, intentional state (installed or
    /// explicitly declined) — not just "a process happened to run once."
    pub setup_completed: bool,
}

impl PersistedConfig {
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => match toml::from_str(&text) {
                Ok(parsed) => Ok(parsed),
                Err(err) => {
                    tracing::warn!(%err, path = %path.display(), "config.toml is corrupt; using defaults");
                    Ok(Self::default())
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err.into()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        atomic_write_toml(path, self)
    }
}

/// Serializes `value` as TOML and writes it via a sibling-temp-file-then-
/// rename, so a crash mid-write can never leave a truncated/corrupt file
/// where a good one used to be. Shared by `PersistedConfig` and the
/// hardware profile (spec Part V section 28 transactional discipline).
pub fn atomic_write_toml<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let text = toml::to_string_pretty(value).map_err(|e| ConfigError::Serialize(e.to_string()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("toml.tmp");
    std::fs::write(&tmp_path, text.as_bytes())?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_is_atomic() {
        let dir = std::env::temp_dir().join(format!("tqf-persisted-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");

        let cfg = PersistedConfig {
            memory_budget_bytes: Some(4 * 1024 * 1024 * 1024),
            context_limit_tokens: Some(131072),
            enable_vision: true,
            host: None,
            setup_completed: true,
        };
        cfg.save(&path).unwrap();
        assert!(!path.with_extension("toml.tmp").exists());
        assert_eq!(PersistedConfig::load(&path).unwrap(), cfg);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_is_defaults() {
        let path = std::env::temp_dir().join(format!("tqf-missing-{}.toml", std::process::id()));
        assert_eq!(
            PersistedConfig::load(&path).unwrap(),
            PersistedConfig::default()
        );
    }
}
