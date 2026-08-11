//! `~/.tqf` layout (spec Part V section 28; Part IX section 76): a single
//! machine-global root holding persisted config, the hardware profile,
//! trusted receipts, and (later) installed model data.

use std::path::PathBuf;

use crate::error::{ConfigError, Result};

/// Root directory. Overridable via `TQF_HOME`, primarily so tests and
/// `tqf doctor`-style tooling never touch a real user's home directory.
pub fn home_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("TQF_HOME") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var("HOME")
        .map_err(|_| ConfigError::Environment("$HOME is not set".to_string()))?;
    Ok(PathBuf::from(home).join(".tqf"))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(home_dir()?.join("config.toml"))
}

pub fn profile_path() -> Result<PathBuf> {
    Ok(home_dir()?.join("profile.toml"))
}

pub fn receipts_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join("receipts"))
}

pub fn models_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join("models"))
}

/// Creates the directory layout if missing. Idempotent; safe to call on
/// every startup.
pub fn ensure_layout() -> Result<PathBuf> {
    let home = home_dir()?;
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(receipts_dir()?)?;
    std::fs::create_dir_all(models_dir()?)?;
    Ok(home)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_tqf_home_override() {
        let tmp = std::env::temp_dir().join(format!("tqf-paths-test-{}", std::process::id()));
        // SAFETY: test-only, single-threaded within this test's scope; no
        // other test in this crate reads/writes TQF_HOME concurrently.
        unsafe {
            std::env::set_var("TQF_HOME", &tmp);
        }
        assert_eq!(home_dir().unwrap(), tmp);
        assert_eq!(config_path().unwrap(), tmp.join("config.toml"));
        unsafe {
            std::env::remove_var("TQF_HOME");
        }
    }
}
