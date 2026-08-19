//! Fixed graph for the pplx-embed-v1-0.6b helper model (spec §37, §86):
//! a dense, bidirectional Qwen3-architecture encoder used only for
//! `/v1/embeddings`. Deliberately not a generic model loader, matching
//! `model::qwen36::geometry`'s pattern — one hard-coded geometry per
//! supported checkpoint rather than a config-driven graph.
//!
//! Resolved from the live `perplexity-ai/pplx-embed-v1-0.6b` `config.json`
//! on 2026-08-18 (`model_type: "bidirectional_pplx_qwen3"`,
//! `architectures: ["PPLXQwen3Model"]`, built on `transformers.Qwen3Model`
//! with all layers' `is_causal` forced false and an OR'd bidirectional
//! attention mask — see the model's own `modeling.py`).

pub struct PplxEmbedGeometry;

impl PplxEmbedGeometry {
    pub const NUM_LAYERS: usize = 28;
    pub const HIDDEN_SIZE: usize = 1024;
    pub const NUM_HEADS: usize = 16;
    pub const NUM_KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const INTERMEDIATE_SIZE: usize = 3072;
    pub const VOCAB_SIZE: usize = 151936;
    pub const ROPE_THETA: f32 = 1_000_000.0;
    pub const RMS_NORM_EPS: f32 = 1e-6;
    /// Matryoshka base output dimension; callers may truncate the pooled
    /// embedding to any prefix length `<= EMBED_DIM` (spec §86: "Matryoshka
    /// representation learning" — truncation needs no extra weights).
    pub const EMBED_DIM: usize = 1024;

    pub const fn kv_group_size() -> usize {
        Self::NUM_HEADS / Self::NUM_KV_HEADS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_heads_divide_query_heads_evenly() {
        assert_eq!(
            PplxEmbedGeometry::NUM_HEADS % PplxEmbedGeometry::NUM_KV_HEADS,
            0
        );
        assert_eq!(PplxEmbedGeometry::kv_group_size(), 2);
    }

    #[test]
    fn head_dim_times_heads_matches_q_projection_width() {
        assert_eq!(
            PplxEmbedGeometry::NUM_HEADS * PplxEmbedGeometry::HEAD_DIM,
            2048
        );
        assert_eq!(
            PplxEmbedGeometry::NUM_KV_HEADS * PplxEmbedGeometry::HEAD_DIM,
            1024
        );
    }
}
