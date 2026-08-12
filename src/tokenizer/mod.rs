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

impl TqfTokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        gguf::build_from_gguf(file)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| ModelError::TokenizerBuild(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| ModelError::TokenizerBuild(e.to_string()).into())
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
