//! `.tqi` — the project-local persisted index (spec §173-§177, §185).
//!
//! Spec §173 is LOCKED on the shape: first-party, local, incremental,
//! content-aware, no external vector database, living at
//! `<root>/.tqf/index.tqi` beside a `project.toml`.

pub mod codec;
pub mod loaded;
pub mod registry;
pub mod segments;
pub mod superblock;

/// The project-local index location fixed by spec §173:
/// `<root>/.tqf/index.tqi`, beside the `project.toml` a later phase adds.
pub fn index_path(root: &std::path::Path) -> std::path::PathBuf {
    root.join(".tqf").join("index.tqi")
}
