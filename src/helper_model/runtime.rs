//! Ties the pplx-embed helper model together behind one call: tokenize,
//! run the bidirectional encoder, mean-pool, optionally MRL-truncate, and
//! quantize (spec §37). The weights lease is transient by construction —
//! `PplxEmbedRuntime` owns `PplxEmbedWeights` directly, so dropping the
//! runtime releases the `MemoryOwner::HelperModel` broker reservation
//! (spec §115 item 7: unload immediately after the operation completes;
//! spec invariant #6 in Part VI's memory design table: "Embedding query...
//! Expert bytes give way briefly to pplx model").

use std::path::Path;

use crate::error::Result;
use crate::memory::MemoryBroker;
use crate::tokenizer::TqfTokenizer;

use super::forward::encode_sequence;
use super::pooling::mean_pool;
use super::quantize::{
    mrl_truncate, quantize_binary_packed, quantize_binary_signed, quantize_int8_tanh,
};
use super::weights::PplxEmbedWeights;

pub struct PplxEmbedRuntime {
    weights: PplxEmbedWeights,
    tokenizer: TqfTokenizer,
}

#[derive(Debug, Clone)]
pub struct PplxEmbedding {
    pub fp32: Vec<f32>,
    pub int8: Vec<i8>,
    pub binary: Vec<f32>,
    pub ubinary: Vec<u8>,
}

impl PplxEmbedRuntime {
    /// Loads the converted `.tqf` container and tokenizer under `broker`.
    /// The caller controls the transient window: construct immediately
    /// before the embedding request(s) and drop immediately after, so the
    /// `HelperModel` reservation does not compete with expert residency
    /// any longer than necessary.
    pub fn load(tqf_path: &Path, tokenizer_path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let weights = PplxEmbedWeights::load(tqf_path, broker)?;
        let tokenizer = TqfTokenizer::from_tokenizer_json_file(tokenizer_path)?;
        Ok(Self { weights, tokenizer })
    }

    /// Tokenizes `text` exactly as `embed` does internally, exposed so
    /// callers (and qualification tests comparing against an independent
    /// tokenizer oracle) can check token IDs on their own.
    pub fn encode_tokens(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, true)
    }

    /// Encodes one text into the full pipeline of representations. MRL
    /// truncation is applied before quantization when `mrl_dim` is
    /// `Some(_)` and smaller than the base 1024-d embedding.
    pub fn embed(&self, text: &str, mrl_dim: Option<usize>) -> Result<PplxEmbedding> {
        self.embed_with_input_budget(text, mrl_dim, None)
    }

    /// As `embed`, but truncates the *input* token sequence to
    /// `max_input_tokens` before running the encoder. The reference
    /// forward pass here is an unoptimized scalar loop whose cost is
    /// dominated by input length (linear per-token projection cost
    /// dwarfs the model's own quadratic attention term until sequences
    /// reach several thousand tokens — see the Phase 38 qualification
    /// doc), so bulk-indexing real documents needs a bounded token
    /// budget to stay tractable; this is a resource control, not a
    /// change to model semantics (the checkpoint's own 32K context
    /// window is unaffected).
    pub fn embed_with_input_budget(
        &self,
        text: &str,
        mrl_dim: Option<usize>,
        max_input_tokens: Option<usize>,
    ) -> Result<PplxEmbedding> {
        let mut token_ids = self.tokenizer.encode(text, true)?;
        if let Some(max) = max_input_tokens {
            token_ids.truncate(max);
        }
        let hidden = encode_sequence(&self.weights, &token_ids);
        let pooled = mean_pool(&hidden);
        let truncated = match mrl_dim {
            Some(dim) if dim < pooled.len() => mrl_truncate(&pooled, dim),
            _ => pooled,
        };
        Ok(PplxEmbedding {
            int8: quantize_int8_tanh(&truncated),
            binary: quantize_binary_signed(&truncated),
            ubinary: quantize_binary_packed(&truncated),
            fp32: truncated,
        })
    }
}
