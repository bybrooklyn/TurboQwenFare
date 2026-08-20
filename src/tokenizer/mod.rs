//! Qwen tokenizer and chat/tool/thinking message template (spec Part V,
//! phase 9, §281). The tokenizer is built directly from GGUF-embedded
//! vocab/merges metadata (`gguf` submodule); `chat` renders TQF's internal
//! message representation into the token stream the model actually
//! consumes, independent of any particular client protocol (spec Part IX:
//! protocol-specific framing must never leak into the model core — chat
//! templating is squarely model-core, not `server`, territory).

pub mod chat;
pub mod gguf;
#[cfg(test)]
mod tests;

use tokenizers::Tokenizer as HfTokenizer;

use crate::error::{ModelError, Result};
use crate::format::gguf::GgufFile;

/// A built tokenizer plus the specific token IDs TQF's chat template and
/// sampling loop need, which live outside `tokenizers::Tokenizer` itself
/// (bos/eos ids are GGUF metadata scalars, not part of the HF tokenizer
/// type's own state).
#[derive(Debug)]
pub struct TqfTokenizer {
    inner: HfTokenizer,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

/// Per-generation state for [`TqfTokenizer::decode_step`]. One of these
/// belongs to each in-flight generation; sharing one across requests would
/// splice their outputs together.
#[derive(Debug, Default)]
pub struct DecodeStreamState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl TqfTokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        gguf::build_from_gguf(file)
    }

    /// Builds directly from a standalone HF `tokenizer.json`, for models
    /// distributed as safetensors rather than GGUF (e.g. the pplx-embed
    /// helper model, spec §37) where no GGUF vocab/merges metadata exists.
    pub fn from_tokenizer_json_file(path: &std::path::Path) -> Result<Self> {
        let inner =
            HfTokenizer::from_file(path).map_err(|e| ModelError::TokenizerBuild(e.to_string()))?;
        Ok(Self {
            inner,
            bos_token_id: None,
            eos_token_id: None,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| ModelError::TokenizerBuild(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Pair encoding (`[CLS] a [SEP] b [SEP]`-style, per the tokenizer's
    /// own `TemplateProcessing` post-processor) for cross-encoder inputs
    /// like the GTE reranker (spec §43/§93), where a single sequence
    /// input would not apply the pair template.
    pub fn encode_pair(&self, a: &str, b: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode((a, b), add_special_tokens)
            .map_err(|e| ModelError::TokenizerBuild(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| ModelError::TokenizerBuild(e.to_string()).into())
    }

    /// Feeds one newly generated token to an incremental decode and
    /// returns whatever complete text that token completed, or `None` if
    /// it only added part of a multi-byte codepoint.
    ///
    /// Spec §71 calls out UTF-8 boundaries as a real streaming bug class:
    /// a naive per-token `decode` emits U+FFFD replacement characters
    /// whenever a codepoint spans tokens, which for CJK and emoji is
    /// routine rather than exotic. This defers to `tokenizers`' own
    /// `step_decode_stream`, which re-decodes and withholds any result
    /// ending in a replacement character — the same primitive the
    /// upstream library's streaming API uses.
    ///
    /// It takes the state by reference rather than borrowing the
    /// tokenizer (as `DecodeStream` would), so callers can keep the
    /// tokenizer behind its `Mutex`.
    pub fn decode_step(&self, state: &mut DecodeStreamState, token: u32) -> Result<Option<String>> {
        tokenizers::tokenizer::step_decode_stream(
            &self.inner,
            vec![token],
            /* skip_special_tokens */ false,
            &mut state.ids,
            &mut state.prefix,
            &mut state.prefix_index,
        )
        .map_err(|e| ModelError::TokenizerBuild(e.to_string()).into())
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
