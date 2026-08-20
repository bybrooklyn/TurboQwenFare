//! Which roots are synced (spec §218).
//!
//! Two files, each with one job:
//!
//! - `<root>/.tqf/project.toml` travels with the project and preserves the
//!   index UUID, so a root keeps its identity across compactions and can
//!   be recognized after a move.
//! - `$TQF_HOME/roots.toml` is the machine-global list the *server* reads
//!   at startup, because a server started anywhere has no other way to
//!   learn which roots exist.
//!
//! Spec §218 also lists `index.journal` and `lock`. Neither is written:
//! the journal belongs with the append-only generation model this
//! baseline does not implement (see `codec`), and a lock file without a
//! real cross-process locking protocol would be a file that looks like
//! mutual exclusion and is not.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::persisted::atomic_write_toml;
use crate::error::Result;

/// `<root>/.tqf/project.toml` (spec §218).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectFile {
    /// Hex of the index UUID, so the identity survives a rewrite.
    pub index_uuid: String,
    /// The root as it was when synced. Informational: the index's own
    /// root identity hash (device + inode) is what actually recognizes a
    /// moved or renamed root.
    pub root: String,
    pub last_synced_unix: u64,
    pub generation: u64,
}

pub fn project_file_path(root: &Path) -> PathBuf {
    root.join(".tqf").join("project.toml")
}

pub fn write_project_file(root: &Path, file: &ProjectFile) -> Result<()> {
    atomic_write_toml(&project_file_path(root), file)
}

pub fn read_project_file(root: &Path) -> Option<ProjectFile> {
    let text = std::fs::read_to_string(project_file_path(root)).ok()?;
    toml::from_str(&text).ok()
}

/// The machine-global registry of synced roots.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Registry {
    #[serde(default)]
    pub roots: Vec<String>,
}

fn registry_path() -> Result<PathBuf> {
    Ok(crate::config::paths::home_dir()?.join("roots.toml"))
}

pub fn load_registry() -> Registry {
    let Ok(path) = registry_path() else {
        return Registry::default();
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => toml::from_str(&text).unwrap_or_else(|error| {
            // A corrupt registry must not stop the server from starting:
            // it is a convenience list, and every entry is re-derivable by
            // syncing again.
            tracing::warn!(%error, path = %path.display(), "roots.toml is corrupt; ignoring it");
            Registry::default()
        }),
        Err(_) => Registry::default(),
    }
}

/// Adds `root` if absent. Idempotent, so re-syncing does not accumulate
/// duplicate entries.
pub fn register(root: &Path) -> Result<()> {
    let mut registry = load_registry();
    let entry = root.display().to_string();
    if !registry.roots.contains(&entry) {
        registry.roots.push(entry);
        registry.roots.sort();
    }
    atomic_write_toml(&registry_path()?, &registry)
}

/// Removes `root` if present. Returns whether it was registered.
pub fn deregister(root: &Path) -> Result<bool> {
    let mut registry = load_registry();
    let entry = root.display().to_string();
    let before = registry.roots.len();
    registry.roots.retain(|existing| existing != &entry);
    let removed = registry.roots.len() != before;
    if removed {
        atomic_write_toml(&registry_path()?, &registry)?;
    }
    Ok(removed)
}

/// Registered roots that still have a readable index, and the ones that
/// do not — reported separately so a stale entry is visible rather than
/// silently skipped.
pub struct ResolvedRoots {
    pub live: Vec<PathBuf>,
    pub stale: Vec<PathBuf>,
}

pub fn resolve() -> ResolvedRoots {
    let mut live = Vec::new();
    let mut stale = Vec::new();
    for entry in load_registry().roots {
        let root = PathBuf::from(entry);
        if super::index_path(&root).exists() {
            live.push(root);
        } else {
            stale.push(root);
        }
    }
    ResolvedRoots { live, stale }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TQF_HOME` is process-global, so these run under one lock rather
    /// than racing each other for it.
    fn with_home<T>(name: &str, body: impl FnOnce() -> T) -> T {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let home = std::env::temp_dir().join(format!("tqf-registry-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        let previous = std::env::var("TQF_HOME").ok();
        std::env::set_var("TQF_HOME", &home);

        let result = body();

        match previous {
            Some(value) => std::env::set_var("TQF_HOME", value),
            None => std::env::remove_var("TQF_HOME"),
        }
        std::fs::remove_dir_all(&home).ok();
        result
    }

    #[test]
    fn registering_is_idempotent_and_deregistering_reports_whether_it_mattered() {
        with_home("idempotent", || {
            let root = PathBuf::from("/projects/alpha");
            assert!(load_registry().roots.is_empty());

            register(&root).unwrap();
            register(&root).unwrap();
            assert_eq!(load_registry().roots, vec!["/projects/alpha".to_string()]);

            assert!(deregister(&root).unwrap(), "the first removal matters");
            assert!(
                !deregister(&root).unwrap(),
                "removing what is not there must report so, not claim success"
            );
            assert!(load_registry().roots.is_empty());
        });
    }

    #[test]
    fn a_corrupt_registry_is_ignored_rather_than_fatal() {
        with_home("corrupt", || {
            let path = registry_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"this is not toml {{{").unwrap();
            assert!(load_registry().roots.is_empty());
        });
    }

    #[test]
    fn a_registered_root_without_an_index_is_reported_stale_not_dropped() {
        with_home("stale", || {
            let missing = std::env::temp_dir().join("tqf-registry-nonexistent-root");
            register(&missing).unwrap();

            let resolved = resolve();
            assert!(resolved.live.is_empty());
            assert_eq!(resolved.stale, vec![missing]);
        });
    }

    #[test]
    fn a_project_file_round_trips() {
        let dir = std::env::temp_dir().join(format!("tqf-project-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = ProjectFile {
            index_uuid: "0123456789abcdef".to_string(),
            root: dir.display().to_string(),
            last_synced_unix: 1_760_000_000,
            generation: 4,
        };
        write_project_file(&dir, &file).unwrap();
        assert_eq!(read_project_file(&dir), Some(file));
        std::fs::remove_dir_all(&dir).ok();
    }
}
