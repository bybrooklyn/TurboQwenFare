//! Fixed graph for the GTE reranker helper model (spec §43, §93):
//! `Alibaba-NLP/gte-reranker-modernbert-base`, a `ModernBertFor
//! SequenceClassification` cross-encoder. Resolved from the live
//! checkpoint's `config.json` and a byte-level read of its safetensors
//! header on 2026-08-19 (138 tensors, 22 layers, confirmed empirically
//! that layer 0's `attn_norm` tensor does not exist — see
//! `weights.rs`'s `AttnNorm: Option<LoadedTensor>` for how that's
//! carried through as `Identity`, not a missing-tensor error).

pub struct GteRerankerGeometry;

impl GteRerankerGeometry {
    pub const NUM_LAYERS: usize = 22;
    pub const HIDDEN_SIZE: usize = 768;
    pub const NUM_HEADS: usize = 12;
    pub const HEAD_DIM: usize = 64;
    pub const INTERMEDIATE_SIZE: usize = 1152;
    pub const VOCAB_SIZE: usize = 50368;
    /// The checkpoint's own `tokenizer.json` bakes in a `Fixed(8000)`
    /// padding/truncation policy (both the Python and Rust `tokenizers`
    /// libraries apply it on every `encode` call, not just batched
    /// ones) — a real, single (query, document) pair is padded to 8000
    /// tokens with this ID even when it needs 36. That padding exists
    /// only for batch-uniform tensor shapes and is meaningless for this
    /// runtime's one-pair-per-call path, so callers trim trailing pad
    /// tokens before running the (otherwise unmasked) forward pass —
    /// see `runtime.rs`'s `trim_trailing_pad`.
    pub const PAD_TOKEN_ID: u32 = 50283;
    pub const LAYER_NORM_EPS: f32 = 1e-5;

    /// Global (full, non-windowed) attention layers are every `n`th
    /// layer starting at 0: 0, 3, 6, 9, 12, 15, 18, 21. All others are
    /// local/sliding-window layers.
    pub const GLOBAL_ATTN_EVERY_N_LAYERS: usize = 3;
    pub const GLOBAL_ROPE_THETA: f32 = 160_000.0;
    pub const LOCAL_ROPE_THETA: f32 = 10_000.0;
    /// A local-layer token attends to `kv_idx` iff `|q_idx - kv_idx| <=
    /// 64` (a symmetric window of radius 64 — confirmed against the
    /// real `transformers` masking code, NOT `local_attention/2 + 1`
    /// despite `ModernBertAttention.__init__` computing
    /// `config.sliding_window + 1`; that `+1`'d value is only forwarded
    /// to the flash-attention-2 kernel's window-size convention and is
    /// never used to build the actual eager/SDPA mask).
    pub const LOCAL_WINDOW_RADIUS: usize = 64;

    pub const fn is_global_layer(layer: usize) -> bool {
        layer.is_multiple_of(Self::GLOBAL_ATTN_EVERY_N_LAYERS)
    }

    pub const fn rope_theta(layer: usize) -> f32 {
        if Self::is_global_layer(layer) {
            Self::GLOBAL_ROPE_THETA
        } else {
            Self::LOCAL_ROPE_THETA
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_layers_match_the_real_checkpoint_pattern() {
        let global: Vec<usize> = (0..GteRerankerGeometry::NUM_LAYERS)
            .filter(|&l| GteRerankerGeometry::is_global_layer(l))
            .collect();
        assert_eq!(global, vec![0, 3, 6, 9, 12, 15, 18, 21]);
    }

    #[test]
    fn head_dim_times_heads_matches_fused_qkv_third() {
        assert_eq!(
            GteRerankerGeometry::NUM_HEADS * GteRerankerGeometry::HEAD_DIM,
            GteRerankerGeometry::HIDDEN_SIZE
        );
    }
}
