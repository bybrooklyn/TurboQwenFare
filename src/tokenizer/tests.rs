//! Golden fixtures for the GGUF-driven tokenizer build (spec §281's
//! remaining list items not covered by `chat.rs`'s own template-shape
//! tests: Unicode/byte fallback, and a sanity check that a rendered chat
//! transcript actually round-trips through the real tokenizer).

use std::path::PathBuf;

use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use super::chat::{render, ChatMessage, ChatRole};
use super::TqfTokenizer;
use crate::format::gguf;

const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_I32: u32 = 5;
const TYPE_U32: u32 = 4;

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    write_str(out, key);
    out.extend_from_slice(&TYPE_STRING.to_le_bytes());
    write_str(out, value);
}

fn write_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    write_str(out, key);
    out.extend_from_slice(&TYPE_U32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_kv_string_array(out: &mut Vec<u8>, key: &str, values: &[String]) {
    write_str(out, key);
    out.extend_from_slice(&TYPE_ARRAY.to_le_bytes());
    out.extend_from_slice(&TYPE_STRING.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        write_str(out, v);
    }
}

fn write_kv_i32_array(out: &mut Vec<u8>, key: &str, values: &[i32]) {
    write_str(out, key);
    out.extend_from_slice(&TYPE_ARRAY.to_le_bytes());
    out.extend_from_slice(&TYPE_I32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Builds a synthetic GGUF fixture whose vocab is the full 256-symbol
/// byte-level alphabet (so *any* UTF-8 input can be losslessly encoded via
/// byte fallback even with zero merges) plus a handful of Qwen-style
/// control tokens.
fn build_gguf_fixture() -> Vec<u8> {
    let mut tokens: Vec<String> = ByteLevel::alphabet()
        .into_iter()
        .map(String::from)
        .collect();
    // Deterministic ordering so token ids are stable across test runs.
    tokens.sort();

    let control_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];
    let mut token_types = vec![1i32; tokens.len()]; // NORMAL
    for ct in control_tokens {
        tokens.push(ct.to_string());
        token_types.push(3); // CONTROL
    }
    let eos_id = tokens.iter().position(|t| t == "<|endoftext|>").unwrap() as u32;

    let mut kvs = Vec::new();
    write_kv_string(&mut kvs, "tokenizer.ggml.model", "gpt2");
    write_kv_string_array(&mut kvs, "tokenizer.ggml.tokens", &tokens);
    write_kv_string_array(&mut kvs, "tokenizer.ggml.merges", &[]); // no merges: pure byte fallback
    write_kv_i32_array(&mut kvs, "tokenizer.ggml.token_type", &token_types);
    write_kv_u32(&mut kvs, "tokenizer.ggml.bos_token_id", eos_id);
    write_kv_u32(&mut kvs, "tokenizer.ggml.eos_token_id", eos_id);

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&6u64.to_le_bytes()); // metadata_kv_count
    out.extend_from_slice(&kvs);
    out
}

fn fixture_path(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!("tqf-tokenizer-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Tests run concurrently and several call `build_tokenizer()`, which
    // shares this helper — a bare per-process directory isn't enough to
    // keep one test's write from racing another's read of "the same"
    // fixture file.
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("{unique}-{name}"))
}

fn build_tokenizer() -> TqfTokenizer {
    let bytes = build_gguf_fixture();
    let path = fixture_path("tokenizer.gguf");
    std::fs::write(&path, &bytes).unwrap();
    let gguf_file = gguf::open(&path).unwrap();
    TqfTokenizer::from_gguf(&gguf_file).unwrap()
}

#[test]
fn builds_from_gguf_metadata() {
    let tokenizer = build_tokenizer();
    assert!(tokenizer.vocab_size() >= 256);
    assert_eq!(
        tokenizer.token_to_id("<|im_start|>"),
        tokenizer.token_to_id("<|im_start|>")
    );
    assert!(tokenizer.token_to_id("<|im_start|>").is_some());
}

#[test]
fn rejects_unsupported_tokenizer_model() {
    let mut bytes = Vec::new();
    let mut kvs = Vec::new();
    write_kv_string(&mut kvs, "tokenizer.ggml.model", "spm");
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&kvs);

    let path = fixture_path("unsupported.gguf");
    std::fs::write(&path, &bytes).unwrap();
    let gguf_file = gguf::open(&path).unwrap();
    let err = TqfTokenizer::from_gguf(&gguf_file).unwrap_err();
    assert!(err.to_string().contains("spm"));
}

#[test]
fn unicode_and_byte_fallback_round_trips_exactly() {
    let tokenizer = build_tokenizer();

    for sample in [
        "hello world",
        "emoji: \u{1F980} crab",            // 🦀, multi-byte UTF-8
        "\u{4e2d}\u{6587}\u{6d4b}\u{8bd5}", // Chinese text
        "mixed: caf\u{e9} na\u{ef}ve",      // accented Latin-1 supplement chars
        "",
    ] {
        let ids = tokenizer.encode(sample, false).unwrap();
        let decoded = tokenizer.decode(&ids, false).unwrap();
        assert_eq!(decoded, sample, "round trip failed for {sample:?}");
    }
}

#[test]
fn rendered_chat_transcript_encodes_and_decodes_special_tokens() {
    let tokenizer = build_tokenizer();
    let messages = vec![
        ChatMessage::text(ChatRole::System, "You are helpful."),
        ChatMessage::text(ChatRole::User, "Hi \u{1F44B}"),
    ];
    let rendered = render(&messages, &[], true);

    let ids = tokenizer.encode(&rendered, true).unwrap();
    assert!(!ids.is_empty());

    // <|im_start|>/<|im_end|> were registered as special/control tokens,
    // so they must survive encoding as single token ids, not get shredded
    // into byte-fallback fragments.
    let im_start_id = tokenizer.token_to_id("<|im_start|>").unwrap();
    assert!(ids.contains(&im_start_id));

    let decoded = tokenizer.decode(&ids, false).unwrap();
    assert!(decoded.contains("You are helpful."));
    assert!(decoded.contains("Hi"));
}
