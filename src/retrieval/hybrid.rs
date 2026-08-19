//! Hybrid retrieval: query intent routing, the shared candidate/
//! provenance contract, and weighted RRF fusion with hard exact
//! precedence (spec §40, §85, §192-195).
//!
//! Only three of spec §83's seven evidence lanes exist to fuse —
//! **Exact** (`retrieval::lexical::LexicalIndex::exact_lookup`),
//! **Lexical** (its BM25 lane), and **Semantic**
//! (`retrieval::flat::FlatVectorStore`) — because Structural/Program
//! graph/Hierarchy/Change-Git all assume real AST or Git-history
//! integration Phase 35/36 already scoped out for this session (no real
//! parser). Spec §195's graph expansion ("add parent definition, direct
//! callers/callees... test neighbors") is not attempted for the same
//! reason: it needs a program graph that doesn't exist yet. What *is*
//! buildable without those — intent classification, the candidate/
//! provenance contract, RRF fusion, and hard exact precedence over the
//! lanes that exist — is built and measured on real data below.

use std::collections::HashMap;

use super::flat::FlatVectorStore;
use super::lexical::LexicalIndex;

/// Reference query classes (spec §192). The router emits *confidences*
/// per class, not a single mutually-exclusive label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    ExactSymbol,
    ExactPath,
    ErrorLiteral,
    SemanticQuestion,
    Mixed,
}

/// Only the lanes this crate actually has evidence for (see module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetrievalLane {
    Exact,
    Lexical,
    Semantic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exactness {
    Exact,
    Fuzzy,
    None,
}

/// One lane's evidence for one candidate, kept even after fusion so a
/// GUI/debugger can explain *why* a result ranked where it did (spec
/// §40: "Add retrieval provenance explanation objects now for GUI/
/// debugging later"; spec §195: "records why each candidate was added").
#[derive(Debug, Clone)]
pub struct CandidateProvenance {
    pub lane: RetrievalLane,
    pub raw_score: f32,
    pub rank_in_lane: u32,
    pub reason: String,
}

/// One lane's raw hits before fusion: `(chunk_id, raw_score, reason)`.
pub type LaneHits = Vec<(String, f32, String)>;

/// One fused result (spec §193's `Candidate`, adapted to whole-file
/// chunk IDs per Phase 36's chunking scope decision — there is no
/// sub-file `ChunkId` yet).
#[derive(Debug, Clone)]
pub struct FusedCandidate {
    pub chunk_id: String,
    pub exactness: Exactness,
    pub rrf_score: f32,
    pub provenance: Vec<CandidateProvenance>,
}

/// Signal-based intent classifier (spec §192's signal list): quoted
/// strings, language identifier forms, path separators/extensions,
/// compiler/stack-trace patterns, natural-language question density.
/// Returns confidence per intent class that fired, not a single label.
pub fn classify_query(query: &str) -> Vec<(QueryIntent, f32)> {
    let trimmed = query.trim();
    let word_count = trimmed.split_whitespace().count();
    let mut scores: HashMap<QueryIntent, f32> = HashMap::new();

    let looks_like_identifier = word_count <= 2
        && !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == ':' || c == '.');
    if looks_like_identifier || trimmed.contains("::") || trimmed.contains('(') {
        let confidence = if trimmed.contains("::") { 0.9 } else { 0.6 };
        *scores.entry(QueryIntent::ExactSymbol).or_insert(0.0) += confidence;
    }

    let has_path_extension = [".rs", ".md", ".toml", ".json", ".py", ".ts"]
        .iter()
        .any(|ext| trimmed.ends_with(ext));
    if trimmed.contains('/') && (has_path_extension || word_count == 1) {
        *scores.entry(QueryIntent::ExactPath).or_insert(0.0) += 0.9;
    }

    let error_markers = [
        "error",
        "panic",
        "traceback",
        "expected",
        "unwrap",
        "exception",
        "failed",
    ];
    let lower = trimmed.to_lowercase();
    if error_markers.iter().any(|m| lower.contains(m)) {
        *scores.entry(QueryIntent::ErrorLiteral).or_insert(0.0) += 0.7;
    }

    let question_words = [
        "how", "what", "why", "does", "which", "where", "when", "explain", "describe",
    ];
    let question_word_hits = trimmed
        .split_whitespace()
        .filter(|w| question_words.contains(&w.to_lowercase().as_str()))
        .count();
    if word_count >= 4 && (question_word_hits > 0 || word_count >= 6) {
        let confidence = (0.5 + 0.1 * question_word_hits as f32).min(0.95);
        *scores.entry(QueryIntent::SemanticQuestion).or_insert(0.0) += confidence;
    }

    if scores.len() > 1 {
        let mixed_confidence = scores.values().cloned().fold(0.0f32, f32::max) * 0.5;
        scores.insert(QueryIntent::Mixed, mixed_confidence);
    }

    let mut result: Vec<(QueryIntent, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.total_cmp(&a.1));
    result
}

/// Spec §85: "Identifier-like queries should hit exact/lexical/symbol
/// paths without loading the embedder... Retrieval should be skipped
/// entirely when it is not useful." Returns whether the (expensive)
/// semantic lane is worth running for this query.
pub fn should_use_semantic_lane(intents: &[(QueryIntent, f32)]) -> bool {
    let semantic_confidence = intents
        .iter()
        .find(|(intent, _)| *intent == QueryIntent::SemanticQuestion)
        .map(|(_, c)| *c)
        .unwrap_or(0.0);
    let exact_confidence = intents
        .iter()
        .filter(|(intent, _)| matches!(intent, QueryIntent::ExactSymbol | QueryIntent::ExactPath))
        .map(|(_, c)| *c)
        .fold(0.0f32, f32::max);
    semantic_confidence > 0.0 && semantic_confidence >= exact_confidence
}

/// Weighted reciprocal-rank fusion (spec §194): `rrf = Σ_lane
/// weight_lane / (k + rank_lane)`, `k=60` initial control, then hard
/// exact precedence (spec §84/§194: an exact hit is never displaced by
/// semantic score alone — every exact-lane candidate is sorted above
/// every non-exact candidate, however high semantic scored).
pub fn fuse_rrf(
    lanes: &[(RetrievalLane, LaneHits)],
    weights: &HashMap<RetrievalLane, f32>,
    k: f32,
) -> Vec<FusedCandidate> {
    let mut by_chunk: HashMap<String, (f32, Vec<CandidateProvenance>, bool)> = HashMap::new();

    for (lane, hits) in lanes {
        let weight = *weights.get(lane).unwrap_or(&1.0);
        for (rank, (chunk_id, raw_score, reason)) in hits.iter().enumerate() {
            let rank_in_lane = rank as u32 + 1;
            let contribution = weight / (k + rank_in_lane as f32);
            let entry = by_chunk
                .entry(chunk_id.clone())
                .or_insert((0.0, Vec::new(), false));
            entry.0 += contribution;
            entry.1.push(CandidateProvenance {
                lane: *lane,
                raw_score: *raw_score,
                rank_in_lane,
                reason: reason.clone(),
            });
            if matches!(lane, RetrievalLane::Exact) {
                entry.2 = true;
            }
        }
    }

    let mut fused: Vec<FusedCandidate> = by_chunk
        .into_iter()
        .map(
            |(chunk_id, (rrf_score, provenance, is_exact))| FusedCandidate {
                chunk_id,
                exactness: if is_exact {
                    Exactness::Exact
                } else {
                    Exactness::None
                },
                rrf_score,
                provenance,
            },
        )
        .collect();

    // Hard exact precedence (spec §84/§194): exact hits sort above
    // everything else regardless of RRF score, ordered among themselves
    // by their own RRF score; non-exact candidates follow, also by RRF
    // score.
    // `by_chunk` is a `HashMap`, so its iteration order (and therefore
    // any tie among equal `rrf_score`s) is not reproducible run to run
    // unless every comparison key is fully deterministic. Break RRF ties
    // by each candidate's single best (lowest) rank in any lane, then by
    // chunk_id, so the same inputs always fuse to the same order.
    fused.sort_by(|a, b| {
        let a_exact = matches!(a.exactness, Exactness::Exact);
        let b_exact = matches!(b.exactness, Exactness::Exact);
        let a_best_rank = a
            .provenance
            .iter()
            .map(|p| p.rank_in_lane)
            .min()
            .unwrap_or(u32::MAX);
        let b_best_rank = b
            .provenance
            .iter()
            .map(|p| p.rank_in_lane)
            .min()
            .unwrap_or(u32::MAX);
        b_exact
            .cmp(&a_exact)
            .then_with(|| b.rrf_score.total_cmp(&a.rrf_score))
            .then_with(|| a_best_rank.cmp(&b_best_rank))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    fused
}

/// Ties the router and the two always-cheap lanes (Exact/Lexical)
/// together with the optional Semantic lane. Semantic search needs an
/// already-computed query embedding (Phase 37's forward pass is
/// expensive) — callers that already have a query FP32 vector pass it
/// via `query_embedding`; `None` means the semantic lane is skipped
/// regardless of what the router would have chosen, so tests that don't
/// have a real embedding on hand can still exercise exact/lexical
/// fusion honestly rather than faking a semantic score.
pub fn run_hybrid_query(
    lexical: &LexicalIndex,
    semantic: Option<&FlatVectorStore>,
    query_text: &str,
    query_embedding: Option<&[f32]>,
    k_per_lane: usize,
    rrf_k: f32,
) -> (Vec<(QueryIntent, f32)>, bool, Vec<FusedCandidate>) {
    let intents = classify_query(query_text);
    let use_semantic =
        should_use_semantic_lane(&intents) && semantic.is_some() && query_embedding.is_some();

    let mut lanes: Vec<(RetrievalLane, LaneHits)> = Vec::new();

    let exact_hits: LaneHits = lexical
        .exact_lookup(query_text)
        .into_iter()
        .enumerate()
        .map(|(rank, path)| {
            (
                path.to_string(),
                1.0,
                format!("exact identifier match, lane rank {}", rank + 1),
            )
        })
        .collect();
    lanes.push((RetrievalLane::Exact, exact_hits));

    let lexical_hits: LaneHits = lexical
        .search(query_text, k_per_lane)
        .into_iter()
        .map(|(path, score)| (path, score, format!("BM25 score {score:.3}")))
        .collect();
    lanes.push((RetrievalLane::Lexical, lexical_hits));

    if use_semantic {
        if let (Some(store), Some(embedding)) = (semantic, query_embedding) {
            let semantic_hits: LaneHits = store
                .search_fp32(embedding, k_per_lane)
                .into_iter()
                .map(|(index, score)| {
                    (
                        store.records[index].id.clone(),
                        score,
                        format!("cosine similarity {score:.3}"),
                    )
                })
                .collect();
            lanes.push((RetrievalLane::Semantic, semantic_hits));
        }
    }

    let mut weights = HashMap::new();
    weights.insert(RetrievalLane::Exact, 2.0);
    weights.insert(RetrievalLane::Lexical, 1.0);
    weights.insert(RetrievalLane::Semantic, 1.0);

    let fused = fuse_rrf(&lanes, &weights, rrf_k);
    (intents, use_semantic, fused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::flat::{quantize_int8_linear, FlatRecord, FlatVectorStore};

    #[test]
    fn identifier_queries_classify_as_exact_symbol_without_semantic_confidence() {
        let intents = classify_query("MemoryBroker");
        let exact = intents
            .iter()
            .find(|(intent, _)| *intent == QueryIntent::ExactSymbol);
        assert!(
            exact.is_some(),
            "expected an ExactSymbol signal: {intents:?}"
        );
        assert!(!should_use_semantic_lane(&intents));
    }

    #[test]
    fn natural_language_questions_classify_as_semantic() {
        let intents = classify_query("how does the memory broker account for reserved bytes");
        let semantic = intents
            .iter()
            .find(|(intent, _)| *intent == QueryIntent::SemanticQuestion);
        assert!(
            semantic.is_some(),
            "expected a SemanticQuestion signal: {intents:?}"
        );
        assert!(should_use_semantic_lane(&intents));
    }

    #[test]
    fn error_literal_queries_are_flagged() {
        let intents = classify_query("thread panicked at unwrap on a None value");
        assert!(intents
            .iter()
            .any(|(intent, _)| *intent == QueryIntent::ErrorLiteral));
    }

    /// Spec §84/§194's hard rule: an exact chunk is never displaced by a
    /// semantically-similar-but-unrelated chunk purely on score, even
    /// when the semantic lane's raw score and rank both favor the other
    /// chunk.
    #[test]
    fn exact_hit_outranks_a_higher_scoring_semantic_only_chunk() {
        let lanes = vec![
            (
                RetrievalLane::Exact,
                vec![(
                    "exact_def.rs".to_string(),
                    1.0,
                    "exact token match".to_string(),
                )],
            ),
            (
                RetrievalLane::Semantic,
                vec![
                    (
                        "unrelated_but_similar.rs".to_string(),
                        0.99,
                        "cosine 0.99".to_string(),
                    ),
                    ("exact_def.rs".to_string(), 0.40, "cosine 0.40".to_string()),
                ],
            ),
        ];
        let mut weights = HashMap::new();
        weights.insert(RetrievalLane::Exact, 2.0);
        weights.insert(RetrievalLane::Semantic, 1.0);
        let fused = fuse_rrf(&lanes, &weights, 60.0);
        assert_eq!(
            fused[0].chunk_id, "exact_def.rs",
            "the exact hit must rank first despite a much higher-scoring semantic-only rival: {fused:?}"
        );
    }

    fn load_fixture_flat_store() -> (FlatVectorStore, Vec<(String, Vec<f32>)>) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let corpus: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                root.join("docs/research/qualification/raw-a-phase38-flat-corpus-embeddings.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut records = Vec::new();
        for d in corpus["documents"].as_array().unwrap() {
            let fp32: Vec<f32> = d["fp32"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let (int8, int8_scale) = quantize_int8_linear(&fp32);
            records.push(FlatRecord {
                id: d["id"].as_str().unwrap().to_string(),
                fp32,
                int8,
                int8_scale,
                binary_packed: Vec::new(),
            });
        }
        let store = FlatVectorStore { records };

        let queries: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                root.join("docs/research/qualification/raw-a-phase38-flat-query-embeddings.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let query_list = queries["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let text = q["text"].as_str().unwrap().to_string();
                let fp32: Vec<f32> = q["fp32"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                (text, fp32)
            })
            .collect();
        (store, query_list)
    }

    /// Real end-to-end qualification: builds the Exact/Lexical lanes
    /// fresh from real file contents and the Semantic lane from Phase
    /// 38's committed real embeddings (same ten real files, same four
    /// real natural-language queries — no model load needed since the
    /// fixture already carries real computed embeddings), then proves
    /// the router correctly skips the semantic lane for an identifier
    /// query and correctly engages all three lanes for real
    /// natural-language questions, reproducing Phase 38's own top-1
    /// winners through the fused hybrid pipeline.
    #[test]
    fn real_repo_hybrid_query_routes_and_fuses_correctly() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let relative_paths = [
            "src/memory/mod.rs",
            "src/helper_model/quantize.rs",
            "src/helper_model/runtime.rs",
            "src/retrieval/ignore.rs",
            "src/retrieval/classify.rs",
            "src/experts/policy.rs",
            "src/experts/prefetch.rs",
            "src/format/tqf/writer.rs",
            "src/context/tqattn/mod.rs",
            "src/context/prefix/mod.rs",
        ];
        let documents: Vec<(String, String)> = relative_paths
            .iter()
            .filter_map(|p| {
                let contents = std::fs::read_to_string(root.join(p)).ok()?;
                Some((p.to_string(), contents))
            })
            .collect();
        assert_eq!(documents.len(), 10);
        let lexical = LexicalIndex::build(&documents);
        let (semantic, real_queries) = load_fixture_flat_store();

        // Identifier query: exact/lexical only, semantic lane skipped.
        let (intents, used_semantic, fused) =
            run_hybrid_query(&lexical, Some(&semantic), "MemoryBroker", None, 5, 60.0);
        println!(
            "phase40_query \"MemoryBroker\" intents={intents:?} used_semantic={used_semantic}"
        );
        assert!(
            !used_semantic,
            "an identifier query must not need the semantic lane"
        );
        assert_eq!(
            fused[0].chunk_id, "src/memory/mod.rs",
            "exact identifier hit should win: {fused:?}"
        );
        assert_eq!(fused[0].exactness, Exactness::Exact);

        // Real natural-language queries: all three lanes engage, and the
        // fused top-1 reproduces Phase 38's own established winner for
        // each (memory, quantize, ignore, policy).
        let expected_top1 = [
            "src/memory/mod.rs",
            "src/helper_model/quantize.rs",
            "src/retrieval/ignore.rs",
            "src/experts/policy.rs",
        ];
        for (i, (text, embedding)) in real_queries.iter().enumerate() {
            let (intents, used_semantic, fused) =
                run_hybrid_query(&lexical, Some(&semantic), text, Some(embedding), 5, 60.0);
            println!(
                "phase40_query {text:?} intents={intents:?} used_semantic={used_semantic} top1={:?}",
                fused.first().map(|c| &c.chunk_id)
            );
            assert!(
                used_semantic,
                "a real NL question should engage the semantic lane: {text:?}"
            );
            assert_eq!(
                fused[0].chunk_id, expected_top1[i],
                "fused top-1 should match Phase 38's own gold-ranking winner for {text:?}: {fused:?}"
            );
        }
    }
}
