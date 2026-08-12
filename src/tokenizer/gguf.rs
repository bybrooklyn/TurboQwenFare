//! Builds a `tokenizers::Tokenizer` directly from GGUF-embedded
//! vocab/merges metadata (spec §281 phase 9: "Implement tokenizer using a
//! mature Rust tokenizer dependency or direct format integration if it
//! meets license/performance needs" — this is the "direct format
//! integration" branch: no separate `tokenizer.json` fetch, since the
//! GGUF checkpoint is the only artifact TQF downloads).
//!
//! The pinned canonical checkpoint's `tokenizer.ggml.model` is byte-level
//! BPE (GGUF/llama.cpp's convention for GPT-2-family tokenizers, which
//! Qwen's is) — this module only supports that family; an unrecognized
//! `tokenizer.ggml.model` value is a hard error rather than a silent
//! fallback (spec §115 invariant #2's posture, applied to tokenizer
//! metadata).

use std::collections::HashMap;

use tokenizers::models::bpe::{Merges, Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

use crate::error::{ModelError, Result};
use crate::format::gguf::{GgufFile, GgufValue};

use super::TqfTokenizer;

/// GGUF/llama.cpp's `tokenizer.ggml.model` value for byte-level BPE.
const SUPPORTED_MODEL: &str = "gpt2";

const KEY_MODEL: &str = "tokenizer.ggml.model";
const KEY_TOKENS: &str = "tokenizer.ggml.tokens";
const KEY_MERGES: &str = "tokenizer.ggml.merges";
const KEY_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
const KEY_BOS: &str = "tokenizer.ggml.bos_token_id";
const KEY_EOS: &str = "tokenizer.ggml.eos_token_id";

/// llama.cpp's `LLAMA_TOKEN_TYPE_CONTROL`: chat/role markers like
/// `<|im_start|>` that must never be split by BPE merges.
const TOKEN_TYPE_CONTROL: i64 = 3;

pub fn build_from_gguf(file: &GgufFile) -> Result<TqfTokenizer> {
    let metadata = file.tokenizer_metadata();

    let model_kind = metadata
        .get(KEY_MODEL)
        .and_then(|v| v.as_str())
        .ok_or(ModelError::TokenizerMetadataMissing(KEY_MODEL))?;
    if model_kind != SUPPORTED_MODEL {
        return Err(ModelError::UnsupportedTokenizerModel(model_kind.to_string()).into());
    }

    let tokens = string_array(&metadata, KEY_TOKENS)?;
    let merges_raw = string_array(&metadata, KEY_MERGES)?;
    let token_types = metadata.get(KEY_TOKEN_TYPE).and_then(|v| v.as_array());

    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(id, tok)| (tok.clone(), id as u32))
        .collect();

    let merges: Merges = merges_raw
        .iter()
        .map(|line| {
            let mut parts = line.splitn(2, ' ');
            let left = parts.next().unwrap_or_default().to_string();
            let right = parts.next().unwrap_or_default().to_string();
            (left, right)
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(false)
        .build()
        .map_err(|e| ModelError::TokenizerBuild(e.to_string()))?;

    let mut tokenizer = HfTokenizer::new(bpe);
    // `add_prefix_space=false, trim_offsets=true, use_regex=true`: the
    // vocab GGUF ships is already byte-level-mapped (GPT-2 convention), so
    // the pre-tokenizer only needs to apply the same byte mapping plus the
    // standard GPT-2 splitting regex, not add its own leading space.
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevel::default()));

    if let Some(types) = token_types {
        let added: Vec<AddedToken> = types
            .iter()
            .zip(tokens.iter())
            .filter_map(|(ty, tok)| {
                if ty.as_i64() == Some(TOKEN_TYPE_CONTROL) {
                    Some(AddedToken::from(tok.clone(), true))
                } else {
                    None
                }
            })
            .collect();
        if !added.is_empty() {
            tokenizer
                .add_special_tokens(added)
                .map_err(|e| ModelError::TokenizerBuild(e.to_string()))?;
        }
    }

    let bos_token_id = metadata
        .get(KEY_BOS)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let eos_token_id = metadata
        .get(KEY_EOS)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Ok(TqfTokenizer {
        inner: tokenizer,
        bos_token_id,
        eos_token_id,
    })
}

fn string_array(metadata: &HashMap<&str, &GgufValue>, key: &'static str) -> Result<Vec<String>> {
    let value = metadata
        .get(key)
        .ok_or(ModelError::TokenizerMetadataMissing(key))?;
    let items = value
        .as_array()
        .ok_or_else(|| ModelError::TokenizerBuild(format!("{key} is not a GGUF array")))?;
    items
        .iter()
        .map(|v| {
            v.as_str().map(|s| s.to_string()).ok_or_else(|| {
                ModelError::TokenizerBuild(format!("{key} array element is not a string")).into()
            })
        })
        .collect()
}
