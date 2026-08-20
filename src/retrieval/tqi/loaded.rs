//! Turning persisted indexes back into searchable ones at startup.
//!
//! The point of persistence is that this costs a file read rather than a
//! re-walk: `LexicalIndex::from_parts` rebuilds from the stored postings
//! without re-reading or re-tokenizing a single source file.

use std::path::{Path, PathBuf};

use crate::retrieval::lexical::LexicalIndex;

use super::{codec, registry};

/// One synced root, loaded and searchable.
pub struct LoadedRoot {
    pub root: PathBuf,
    pub index_path: PathBuf,
    pub generation: u64,
    pub file_count: usize,
    pub term_count: usize,
    pub lexical: LexicalIndex,
}

/// What startup found, including what it could not load — a root whose
/// index is corrupt must be visible, not silently missing.
#[derive(Default)]
pub struct LoadedIndexes {
    pub roots: Vec<LoadedRoot>,
    /// `(root, why)` for registered roots that failed to load.
    pub failed: Vec<(PathBuf, String)>,
    /// Registered roots with no index file at all.
    pub stale: Vec<PathBuf>,
}

impl LoadedIndexes {
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Searches every loaded root, tagging each hit with the root it came
    /// from, and returns the best `top_k` overall.
    ///
    /// Scores are comparable across roots because they come from the same
    /// scorer, but not across *lanes* — the exact lane deliberately
    /// bypasses BM25 (spec §193's rule against comparing raw cross-lane
    /// scores), so it is reported separately rather than merged here.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchHit> {
        let mut hits: Vec<SearchHit> = Vec::new();
        for root in &self.roots {
            for (path, score) in root.lexical.search(query, top_k) {
                hits.push(SearchHit {
                    root: root.root.display().to_string(),
                    path,
                    score,
                });
            }
        }
        // Deterministic ordering: score descending, then root and path, so
        // equal scores do not reorder between runs.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.root.cmp(&b.root))
                .then_with(|| a.path.cmp(&b.path))
        });
        hits.truncate(top_k);
        hits
    }

    pub fn exact_lookup(&self, identifier: &str) -> Vec<SearchHit> {
        let mut hits = Vec::new();
        for root in &self.roots {
            for path in root.lexical.exact_lookup(identifier) {
                hits.push(SearchHit {
                    root: root.root.display().to_string(),
                    path: path.to_string(),
                    score: 1.0,
                });
            }
        }
        hits.sort_by(|a, b| a.root.cmp(&b.root).then_with(|| a.path.cmp(&b.path)));
        hits
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub root: String,
    pub path: String,
    pub score: f32,
}

/// Loads one root's index.
pub fn load_root(root: &Path) -> crate::error::Result<LoadedRoot> {
    let index_path = super::index_path(root);
    let loaded = codec::read(&index_path)?;

    // Chunk order is file order, which the writer establishes and the
    // reader preserves, so paths line up with chunk ids by construction.
    let paths: Vec<String> = loaded
        .contents
        .files
        .iter()
        .map(|file| {
            loaded
                .contents
                .paths
                .get(&file.id)
                .cloned()
                .unwrap_or_default()
        })
        .collect();

    let lexical = LexicalIndex::from_parts(
        loaded.contents.postings,
        loaded.contents.exact,
        &loaded.contents.chunk_lengths,
        &paths,
    );

    Ok(LoadedRoot {
        root: root.to_path_buf(),
        index_path,
        generation: loaded.superblock.latest_generation,
        file_count: loaded.contents.files.len(),
        term_count: lexical.term_count(),
        lexical,
    })
}

/// Loads every registered root. Never fails as a whole: a corrupt index
/// on one root must not stop a server from starting and serving the rest.
pub fn load_registered() -> LoadedIndexes {
    let resolved = registry::resolve();
    let mut out = LoadedIndexes {
        stale: resolved.stale,
        ..LoadedIndexes::default()
    };

    for root in resolved.live {
        match load_root(&root) {
            Ok(loaded) => out.roots.push(loaded),
            Err(error) => {
                tracing::warn!(root = %root.display(), %error, "failed to load index");
                out.failed.push((root, error.to_string()));
            }
        }
    }
    out
}
