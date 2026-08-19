//! Lexical index baseline (spec §185-186; Phase 36, spec §308): "Build
//! useful search without embeddings first. Exact symbol/path/BM25/graph
//! baselines provide fallback while helper models are unavailable."
//!
//! Real AST-based structural chunking/symbol records/program graph (spec
//! §82-84, §180-184) need a real parser, which Phase 35 explicitly does
//! not add (see that phase's qualification doc). This module instead
//! builds the two evidence lanes that don't need one: a custom BM25-ish
//! inverted index (spec §185 REFERENCE BASELINE) and an exact
//! identifier/path lookup (spec §83's "Exact" lane), both over whole-file
//! chunks — a real, useful search baseline, not a placeholder.

use std::collections::HashMap;

const DEFAULT_K1: f32 = 1.2;
const DEFAULT_B: f32 = 0.75;

/// One indexed document — a whole file, in the absence of real AST-based
/// sub-file chunking (spec §180's `ChunkRecord` sans `parent_symbol`,
/// which needs a symbol table this phase doesn't build).
#[derive(Debug, Clone)]
struct ChunkEntry {
    path: String,
    token_count: u32,
}

/// Spec §185's token streams: natural-language/identifier-whole tokens
/// plus identifier subtokens split on snake/camel/digit boundaries, all
/// lowercased for lexical matching. Case is preserved separately for the
/// exact-identifier lane.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for raw in split_raw_tokens(text) {
        if raw.is_empty() {
            continue;
        }
        tokens.push(raw.to_ascii_lowercase());
        let subtokens = split_identifier(raw);
        if subtokens.len() > 1 {
            for sub in subtokens {
                tokens.push(sub.to_ascii_lowercase());
            }
        }
    }
    tokens
}

/// Splits on anything that isn't alphanumeric or `_` — the raw
/// whitespace/punctuation tokenization pass before identifier-subtoken
/// splitting.
fn split_raw_tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .collect()
}

/// snake_case / camelCase / PascalCase / SCREAMING_CASE / digit-boundary
/// identifier splitting (spec §185).
fn split_identifier(raw: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = raw.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            continue;
        }
        let boundary = if current.is_empty() {
            false
        } else {
            let prev = *current.as_bytes().last().unwrap() as char;
            let prev_is_lower_or_digit = prev.is_lowercase() || prev.is_ascii_digit();
            let curr_is_upper = c.is_uppercase();
            let digit_transition = prev.is_ascii_digit() != c.is_ascii_digit();
            // camel/Pascal boundary: lower/digit -> upper.
            (prev_is_lower_or_digit && curr_is_upper)
                // Acronym boundary: "HTTPServer" -> "HTTP", "Server".
                || (prev.is_uppercase() && curr_is_upper && chars.get(i + 1).is_some_and(|n| n.is_lowercase()))
                || digit_transition
        };
        if boundary {
            parts.push(std::mem::take(&mut current));
        }
        current.push(c);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// The persistable form of a lexical index (spec §185): postings sorted
/// by chunk id, the exact identifier lane, per-chunk token counts for
/// BM25 length normalization, and the chunk paths those counts belong to.
///
/// Named rather than returned as a bare 4-tuple so `export` and
/// `from_parts` describe the same thing and callers destructure against a
/// documented order.
pub type PersistedLexical = (
    std::collections::BTreeMap<String, Vec<(u32, u32)>>,
    std::collections::BTreeMap<String, Vec<u32>>,
    Vec<u32>,
    Vec<String>,
);

pub struct LexicalIndex {
    postings: HashMap<String, Vec<(u32, u32)>>, // term -> [(chunk_id, term_frequency)]
    chunks: Vec<ChunkEntry>,
    avg_doc_len: f32,
    /// Exact-token lane (spec §83): case-preserved whole raw tokens,
    /// bypassing BM25 entirely — "Exact symbol lookup bypasses semantic
    /// ANN entirely" (spec §182), and by construction here bypasses BM25
    /// term-frequency scoring too.
    exact: HashMap<String, Vec<u32>>,
}

impl LexicalIndex {
    /// Builds the index from `(path, content)` pairs — real file content,
    /// not synthetic pre-tokenized fixtures, matching how a real scan
    /// would feed this (spec §185's "chunk text itself remains in the
    /// source file, not duplicated wholesale into the index").
    pub fn build(documents: &[(String, String)]) -> Self {
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut chunks = Vec::with_capacity(documents.len());
        let mut exact: HashMap<String, Vec<u32>> = HashMap::new();
        let mut total_tokens = 0u64;

        for (chunk_id, (path, content)) in documents.iter().enumerate() {
            let chunk_id = chunk_id as u32;
            let tokens = tokenize(content);
            total_tokens += tokens.len() as u64;
            chunks.push(ChunkEntry {
                path: path.clone(),
                token_count: tokens.len() as u32,
            });

            let mut term_frequency: HashMap<String, u32> = HashMap::new();
            for token in &tokens {
                *term_frequency.entry(token.clone()).or_insert(0) += 1;
            }
            for (term, tf) in term_frequency {
                postings.entry(term).or_default().push((chunk_id, tf));
            }

            for raw in split_raw_tokens(content) {
                let entry = exact.entry(raw.to_string()).or_default();
                if entry.last() != Some(&chunk_id) {
                    entry.push(chunk_id);
                }
            }
        }

        let avg_doc_len = if chunks.is_empty() {
            0.0
        } else {
            total_tokens as f32 / chunks.len() as f32
        };

        Self {
            postings,
            chunks,
            avg_doc_len,
            exact,
        }
    }

    /// The index in the form `.tqi` persists (spec §185): postings
    /// sorted by chunk id, the exact identifier lane, and per-chunk token
    /// counts, which BM25 needs for length normalization.
    ///
    /// Returned rather than serialized here so `retrieval::tqi` owns the
    /// on-disk layout and this module stays about scoring.
    pub fn export(&self) -> PersistedLexical {
        let mut postings: std::collections::BTreeMap<String, Vec<(u32, u32)>> =
            std::collections::BTreeMap::new();
        for (term, list) in &self.postings {
            let mut list = list.clone();
            // Spec §185: postings are sorted by chunk id. The in-memory
            // build appends in document order, which is the same thing
            // today, but persisting relies on it so it is made explicit.
            list.sort_unstable_by_key(|(chunk, _)| *chunk);
            postings.insert(term.clone(), list);
        }

        let mut exact: std::collections::BTreeMap<String, Vec<u32>> =
            std::collections::BTreeMap::new();
        for (identifier, list) in &self.exact {
            let mut list = list.clone();
            list.sort_unstable();
            exact.insert(identifier.clone(), list);
        }

        (
            postings,
            exact,
            self.chunks.iter().map(|c| c.token_count).collect(),
            self.chunks.iter().map(|c| c.path.clone()).collect(),
        )
    }

    /// Rebuilds a searchable index from persisted parts, without
    /// re-reading or re-tokenizing any source file — which is the whole
    /// point of persisting it.
    pub fn from_parts(
        postings: std::collections::BTreeMap<String, Vec<(u32, u32)>>,
        exact: std::collections::BTreeMap<String, Vec<u32>>,
        chunk_lengths: &[u32],
        paths: &[String],
    ) -> Self {
        let chunks: Vec<ChunkEntry> = paths
            .iter()
            .zip(chunk_lengths.iter())
            .map(|(path, token_count)| ChunkEntry {
                path: path.clone(),
                token_count: *token_count,
            })
            .collect();
        let total: u64 = chunks.iter().map(|c| c.token_count as u64).sum();
        let avg_doc_len = if chunks.is_empty() {
            0.0
        } else {
            total as f32 / chunks.len() as f32
        };

        Self {
            postings: postings.into_iter().collect(),
            chunks,
            avg_doc_len,
            exact: exact.into_iter().collect(),
        }
    }

    /// Distinct BM25 terms across the whole index — a cheap measure of
    /// how much vocabulary a scan actually produced, which is what
    /// `tqf sync` reports to show the index is real rather than empty.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    pub fn document_count(&self) -> usize {
        self.chunks.len()
    }

    /// BM25 reference scoring (spec §186), `k1=1.2, b=0.75` (spec's own
    /// stated non-sacred defaults).
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        self.search_with_params(query, top_k, DEFAULT_K1, DEFAULT_B)
    }

    pub fn search_with_params(
        &self,
        query: &str,
        top_k: usize,
        k1: f32,
        b: f32,
    ) -> Vec<(String, f32)> {
        let query_terms: Vec<String> = tokenize(query).into_iter().collect();
        let n = self.chunks.len() as f32;
        if n == 0.0 {
            return Vec::new();
        }
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in &query_terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(chunk_id, tf) in postings {
                let doc_len = self.chunks[chunk_id as usize].token_count as f32;
                let tf = tf as f32;
                let denom = tf + k1 * (1.0 - b + b * doc_len / self.avg_doc_len.max(1.0));
                let term_score = idf * (tf * (k1 + 1.0)) / denom.max(1e-6);
                *scores.entry(chunk_id).or_insert(0.0) += term_score;
            }
        }
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
            .into_iter()
            .map(|(id, score)| (self.chunks[id as usize].path.clone(), score))
            .collect()
    }

    /// Exact identifier/path lookup (spec §83's "Exact" evidence lane):
    /// case-sensitive, whole-token match, no ranking/relevance scoring.
    pub fn exact_lookup(&self, identifier: &str) -> Vec<&str> {
        self.exact
            .get(identifier)
            .map(|ids| {
                ids.iter()
                    .map(|&id| self.chunks[id as usize].path.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_subtokens_split_snake_camel_pascal_and_digit_boundaries() {
        assert_eq!(
            split_identifier("snake_case_name"),
            vec!["snake", "case", "name"]
        );
        assert_eq!(
            split_identifier("camelCaseName"),
            vec!["camel", "Case", "Name"]
        );
        assert_eq!(
            split_identifier("PascalCaseName"),
            vec!["Pascal", "Case", "Name"]
        );
        assert_eq!(
            split_identifier("SCREAMING_CASE"),
            vec!["SCREAMING", "CASE"]
        );
        assert_eq!(split_identifier("HTTPServer"), vec!["HTTP", "Server"]);
        assert_eq!(split_identifier("value2Count"), vec!["value", "2", "Count"]);
        assert_eq!(split_identifier("plain"), vec!["plain"]);
    }

    #[test]
    fn bm25_ranks_the_document_with_more_relevant_term_density_higher() {
        let docs = vec![
            (
                "cache.rs".to_string(),
                "struct ExpertCache; impl ExpertCache { fn evict(&mut self) {} fn evict_all(&mut self) {} }".to_string(),
            ),
            (
                "unrelated.rs".to_string(),
                "struct Widget; impl Widget { fn render(&self) {} }".to_string(),
            ),
        ];
        let index = LexicalIndex::build(&docs);
        let results = index.search("evict expert cache", 5);
        assert_eq!(results[0].0, "cache.rs");
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn exact_lookup_finds_the_precise_identifier_case_sensitively() {
        let docs = vec![
            ("a.rs".to_string(), "pub struct MemoryBroker;".to_string()),
            ("b.rs".to_string(), "let memorybroker = 1;".to_string()),
        ];
        let index = LexicalIndex::build(&docs);
        assert_eq!(index.exact_lookup("MemoryBroker"), vec!["a.rs"]);
        assert_eq!(index.exact_lookup("memorybroker"), vec!["b.rs"]);
        assert!(index.exact_lookup("NoSuchSymbol").is_empty());
    }

    #[test]
    fn identifier_subtoken_query_finds_a_document_that_never_contains_the_whole_query_string() {
        let docs = vec![(
            "cache.rs".to_string(),
            "pub struct WholeExpertLfuCache { capacity: usize }".to_string(),
        )];
        let index = LexicalIndex::build(&docs);
        // "expert cache" never appears verbatim, only inside the compound
        // identifier `WholeExpertLfuCache` — subtoken splitting is what
        // makes this findable via lexical search at all.
        let results = index.search("expert cache", 5);
        assert_eq!(results[0].0, "cache.rs");
    }

    #[test]
    fn empty_index_search_returns_no_results_without_panicking() {
        let index = LexicalIndex::build(&[]);
        assert!(index.search("anything", 5).is_empty());
        assert!(index.exact_lookup("anything").is_empty());
    }

    /// Phase 36's literal exit gate ("useful search without semantic
    /// model"), proven end to end on real data: scans this actual crate
    /// (reusing Phase 35's real scanner/classifier — not a synthetic
    /// fixture), builds a lexical index over every real Rust source file,
    /// and confirms genuine exact-name and concept queries surface the
    /// real files that actually define/discuss them.
    #[test]
    fn real_repo_index_answers_real_exact_and_concept_queries_correctly() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let report = crate::retrieval::scan::scan_root(root).unwrap();
        let documents: Vec<(String, String)> = report
            .files
            .iter()
            .filter(|f| f.classification.language == Some("Rust"))
            .filter_map(|f| {
                let contents = std::fs::read_to_string(root.join(&f.relative_path)).ok()?;
                Some((f.relative_path.clone(), contents))
            })
            .collect();
        assert!(
            documents.len() > 100,
            "expected most of this crate's real .rs files indexed"
        );
        let index = LexicalIndex::build(&documents);

        // Exact lane: a real, unambiguous type name.
        let exact_hits = index.exact_lookup("MemoryBroker");
        assert!(
            exact_hits.iter().any(|p| p.contains("memory")),
            "expected MemoryBroker's real definition file among exact hits: {exact_hits:?}"
        );

        // BM25 lane: a real struct name that appears nowhere as a literal
        // whole-file-content substring query but is findable via
        // identifier-subtoken splitting plus real term density.
        let bm25_hits = index.search("whole expert lfu cache eviction", 5);
        assert!(!bm25_hits.is_empty());
        assert!(
            bm25_hits.iter().any(|(path, _)| path.contains("experts")),
            "expected src/experts/mod.rs among top BM25 hits for an expert-cache query: {bm25_hits:?}"
        );

        // BM25 lane: this module's own concern, findable by concept terms
        // that appear throughout it (gitignore/glob/pattern), not just one
        // exact identifier.
        let ignore_hits = index.search("gitignore glob pattern matching", 5);
        assert!(
            ignore_hits.iter().any(|(path, _)| path.contains("retrieval/ignore")),
            "expected retrieval/ignore.rs among top hits for a gitignore-matching query: {ignore_hits:?}"
        );

        println!(
            "phase36_real_search documents={} memorybroker_exact_hits={} expert_cache_query_top1={:?} gitignore_query_top1={:?}",
            index.document_count(),
            exact_hits.len(),
            bm25_hits.first(),
            ignore_hits.first(),
        );
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    fn corpus() -> Vec<(String, String)> {
        vec![
            (
                "src/memory/mod.rs".to_string(),
                "pub struct MemoryBroker { budget: u64 } impl MemoryBroker { pub fn reserve() {} }"
                    .to_string(),
            ),
            (
                "src/experts/mod.rs".to_string(),
                "pub struct WholeExpertLfuCache { capacity: usize } fn evict_least_frequent() {}"
                    .to_string(),
            ),
            (
                "README.md".to_string(),
                "TurboQwenFare streams experts from SSD through a memory broker.".to_string(),
            ),
        ]
    }

    /// Persisting is only worth anything if what comes back searches the
    /// same. Compares ranked results, not internal structure.
    #[test]
    fn an_exported_index_rebuilds_into_one_that_searches_identically() {
        let original = LexicalIndex::build(&corpus());
        let (postings, exact, lengths, paths) = original.export();
        let rebuilt = LexicalIndex::from_parts(postings, exact, &lengths, &paths);

        assert_eq!(rebuilt.document_count(), original.document_count());
        assert_eq!(rebuilt.term_count(), original.term_count());

        for query in [
            "memory broker",
            "expert cache eviction",
            "streams experts from ssd",
            "capacity",
        ] {
            let before = original.search(query, 5);
            let after = rebuilt.search(query, 5);
            assert_eq!(
                before.iter().map(|(p, _)| p).collect::<Vec<_>>(),
                after.iter().map(|(p, _)| p).collect::<Vec<_>>(),
                "ranking changed for {query:?}"
            );
            for ((_, a), (_, b)) in before.iter().zip(after.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "score changed for {query:?}: {a} vs {b}"
                );
            }
        }

        // The exact lane must survive too — it is case-preserved and
        // bypasses BM25 entirely.
        assert_eq!(
            rebuilt.exact_lookup("MemoryBroker"),
            original.exact_lookup("MemoryBroker")
        );
    }

    /// Spec §185: postings are sorted by chunk id. The persisted form
    /// relies on it, so the export makes it explicit rather than assuming
    /// build order.
    #[test]
    fn exported_postings_are_sorted_by_chunk_id() {
        let (postings, exact, _, _) = LexicalIndex::build(&corpus()).export();
        for (term, list) in &postings {
            let ids: Vec<u32> = list.iter().map(|(chunk, _)| *chunk).collect();
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            assert_eq!(ids, sorted, "postings for {term:?} are not sorted");
        }
        for (identifier, list) in &exact {
            let mut sorted = list.clone();
            sorted.sort_unstable();
            assert_eq!(*list, sorted, "exact list for {identifier:?} is not sorted");
        }
    }

    #[test]
    fn an_empty_index_exports_and_rebuilds() {
        let original = LexicalIndex::build(&[]);
        let (postings, exact, lengths, paths) = original.export();
        let rebuilt = LexicalIndex::from_parts(postings, exact, &lengths, &paths);
        assert_eq!(rebuilt.document_count(), 0);
        assert!(rebuilt.search("anything", 5).is_empty());
    }
}
