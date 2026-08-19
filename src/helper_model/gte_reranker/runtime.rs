//! Ties the GTE reranker together behind one call: pair-tokenize
//! `(query, document)`, run the ModernBERT cross-encoder, masked-mean
//! pool, and classify to a single relevance logit (spec §43). Like
//! Phase 37's `PplxEmbedRuntime`, the weights lease is transient by
//! construction — dropping `GteRerankerRuntime` releases the
//! `MemoryOwner::HelperModel` broker reservation.

use std::path::Path;

use crate::error::Result;
use crate::helper_model::pooling::mean_pool;
use crate::memory::MemoryBroker;
use crate::tokenizer::TqfTokenizer;

use super::forward::{classify_pooled, encode_sequence};
use super::geometry::GteRerankerGeometry;
use super::weights::GteRerankerWeights;

pub struct GteRerankerRuntime {
    weights: GteRerankerWeights,
    tokenizer: TqfTokenizer,
}

/// Drops trailing `[PAD]` tokens the tokenizer's own baked-in
/// `Fixed(8000)` policy adds to every single-pair encode (see
/// `geometry.rs`'s `PAD_TOKEN_ID` doc comment). Keeps at least one
/// token so a pathological all-pad input doesn't produce an empty
/// sequence.
fn trim_trailing_pad(mut token_ids: Vec<u32>) -> Vec<u32> {
    while token_ids.len() > 1 && *token_ids.last().unwrap() == GteRerankerGeometry::PAD_TOKEN_ID {
        token_ids.pop();
    }
    token_ids
}

impl GteRerankerRuntime {
    pub fn load(tqf_path: &Path, tokenizer_path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let weights = GteRerankerWeights::load(tqf_path, broker)?;
        let tokenizer = TqfTokenizer::from_tokenizer_json_file(tokenizer_path)?;
        Ok(Self { weights, tokenizer })
    }

    /// The tokenizer's raw output, padding included — useful for
    /// comparing against an external oracle's own raw token IDs, not
    /// for feeding directly into `encode_sequence` (see `score`).
    pub fn encode_pair_tokens(&self, query: &str, document: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode_pair(query, document, true)
    }

    /// Scores one `(query, document)` pair. Higher is more relevant;
    /// the raw logit is not a probability (spec's own reference
    /// heuristic in §196 compares logits/margins, not calibrated
    /// probabilities).
    pub fn score(&self, query: &str, document: &str) -> Result<f32> {
        let token_ids = trim_trailing_pad(self.tokenizer.encode_pair(query, document, true)?);
        let hidden = encode_sequence(&self.weights, &token_ids);
        let pooled = mean_pool(&hidden);
        Ok(classify_pooled(&self.weights, &pooled))
    }

    /// spec §196's "rerank at most a bounded candidate count" — scores
    /// every `(query, candidate)` pair and returns `(index, score)`
    /// sorted descending by score.
    pub fn rerank(&self, query: &str, candidates: &[String]) -> Result<Vec<(usize, f32)>> {
        let mut scored = Vec::with_capacity(candidates.len());
        for (i, doc) in candidates.iter().enumerate() {
            scored.push((i, self.score(query, doc)?));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }
}
