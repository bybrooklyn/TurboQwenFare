//! Real-checkpoint greedy-sequence qualification for the fixed Qwen3.6 graph.
//!
//! The independent runtime remains an offline research oracle. TQF consumes a
//! versioned token artifact and executes only its own tokenizer, container,
//! broker, cache, and model loop. This keeps external runtime code out of the
//! product while making Phase 15's 1/16/128/512-token gate reproducible.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{ModelError, Result};
use crate::format::gguf;
use crate::ids::Bytes;
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};
use crate::model::qwen36::geometry::Qwen36Geometry;
use crate::model::qwen36::runtime::Qwen36BoundedReferenceRuntime;
use crate::source::pinned;
use crate::tokenizer::TqfTokenizer;

pub const ORACLE_SCHEMA_VERSION: u32 = 1;
const FOUR_GIB: u64 = 4 * 1024 * 1024 * 1024;
const DEFAULT_EXPERT_CACHE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GreedyOracleArtifact {
    pub schema_version: u32,
    pub fixture_id: String,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub model_source_sha256: String,
    pub oracle_runtime: String,
    pub oracle_revision: String,
    pub oracle_parameters: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GreedyQualificationReport {
    pub fixture_id: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub decode_steps: usize,
    pub elapsed_milliseconds: u128,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub cache_resident_bytes: u64,
    pub raw_miss_bytes: u64,
    pub broker_reserved_bytes_after_run: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualificationRouteTrace {
    pub schema_version: u32,
    pub fixture_id: String,
    pub model_source_sha256: String,
    pub steps: Vec<QualificationRouteStep>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualificationRouteStep {
    pub decode_step: usize,
    pub input_token: u32,
    pub output_token: u32,
    pub layers: Vec<QualificationRouteLayer>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualificationRouteLayer {
    pub layer: u8,
    pub expert_ids: [u16; 8],
    pub weights: [f32; 8],
}

fn qualification_error(message: impl Into<String>) -> crate::error::TqfError {
    ModelError::Unsupported(format!("qualification artifact: {}", message.into())).into()
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| qualification_error(format!("failed to serialize trace: {error}")))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub fn load_oracle(path: &Path) -> Result<GreedyOracleArtifact> {
    let bytes = std::fs::read(path)?;
    let artifact: GreedyOracleArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| qualification_error(format!("invalid JSON: {error}")))?;
    validate_oracle(&artifact)?;
    Ok(artifact)
}

pub fn validate_oracle(artifact: &GreedyOracleArtifact) -> Result<()> {
    if artifact.schema_version != ORACLE_SCHEMA_VERSION {
        return Err(qualification_error(format!(
            "unsupported schema version {}",
            artifact.schema_version
        )));
    }
    if artifact.fixture_id.trim().is_empty() {
        return Err(qualification_error("fixture_id is empty"));
    }
    if artifact.prompt_tokens.is_empty() {
        return Err(qualification_error("prompt token stream is empty"));
    }
    if artifact.generated_tokens.is_empty() {
        return Err(qualification_error("generated token stream is empty"));
    }
    if !matches!(artifact.generated_tokens.len(), 1 | 16 | 128 | 512) {
        return Err(qualification_error(format!(
            "generated length {} is outside the mandatory 1/16/128/512 matrix",
            artifact.generated_tokens.len()
        )));
    }
    if artifact.model_source_sha256 != pinned::LANGUAGE_CHECKPOINT_SHA256 {
        return Err(qualification_error(format!(
            "model SHA-256 {} does not match pinned source {}",
            artifact.model_source_sha256,
            pinned::LANGUAGE_CHECKPOINT_SHA256
        )));
    }
    if artifact.oracle_runtime.trim().is_empty() || artifact.oracle_revision.trim().is_empty() {
        return Err(qualification_error(
            "oracle runtime and immutable revision are required",
        ));
    }
    if artifact
        .prompt_tokens
        .iter()
        .chain(&artifact.generated_tokens)
        .any(|token| *token as usize >= Qwen36Geometry::VOCAB_SIZE)
    {
        return Err(qualification_error("token ID exceeds canonical vocabulary"));
    }
    Ok(())
}

fn verify_greedy_sequence<F>(
    prompt_tokens: &[u32],
    expected_tokens: &[u32],
    mut decode: F,
) -> Result<usize>
where
    F: FnMut(u32) -> Result<u32>,
{
    if prompt_tokens.is_empty() || expected_tokens.is_empty() {
        return Err(qualification_error(
            "prompt and expected sequence must both be nonempty",
        ));
    }
    let mut next = 0;
    let mut decode_steps = 0;
    for &token in prompt_tokens {
        next = decode(token)?;
        decode_steps += 1;
    }
    for (index, &expected) in expected_tokens.iter().enumerate() {
        if next != expected {
            return Err(qualification_error(format!(
                "greedy divergence at generated token {index}: expected {expected}, got {next}"
            )));
        }
        if index + 1 != expected_tokens.len() {
            next = decode(next)?;
            decode_steps += 1;
        }
    }
    Ok(decode_steps)
}

pub fn qualify_oracle(
    tqf_path: &Path,
    tokenizer_gguf_path: &Path,
    artifact: &GreedyOracleArtifact,
) -> Result<GreedyQualificationReport> {
    validate_oracle(artifact)?;
    let broker = MemoryBroker::new(Bytes(FOUR_GIB));
    let tokenizer_source = gguf::open_with_broker(tokenizer_gguf_path, &broker)?;
    let tokenizer = TqfTokenizer::from_gguf(&tokenizer_source)?;
    let encoded = tokenizer.encode(&artifact.prompt, false)?;
    if encoded != artifact.prompt_tokens {
        return Err(qualification_error(format!(
            "prompt tokenization differs: artifact {:?}, TQF {:?}",
            artifact.prompt_tokens, encoded
        )));
    }

    let context_capacity = artifact
        .prompt_tokens
        .len()
        .checked_add(artifact.generated_tokens.len())
        .ok_or_else(|| qualification_error("context length overflow"))?;
    let mut runtime = Qwen36BoundedReferenceRuntime::open(
        tqf_path,
        broker.clone(),
        context_capacity,
        Bytes(DEFAULT_EXPERT_CACHE_BYTES),
    )?;
    let route_trace_path = std::env::var_os("TQF_QUALIFICATION_ROUTE_TRACE").map(PathBuf::from);
    let maximum_decode_steps = artifact
        .prompt_tokens
        .len()
        .checked_add(artifact.generated_tokens.len().saturating_sub(1))
        .ok_or_else(|| qualification_error("route trace step count overflow"))?;
    let route_trace_bytes = maximum_decode_steps
        .checked_mul(Qwen36Geometry::NUM_LAYERS)
        .and_then(|records| records.checked_mul(128))
        .ok_or_else(|| qualification_error("route trace reservation overflow"))?;
    let _route_trace_lease = route_trace_path
        .as_ref()
        .map(|_| {
            broker.reserve(
                MemoryOwner::Scratch,
                MemoryClass::Transient,
                Bytes(route_trace_bytes.max(1) as u64),
                64,
            )
        })
        .transpose()?;
    let mut route_steps = Vec::with_capacity(if route_trace_path.is_some() {
        maximum_decode_steps
    } else {
        0
    });
    let started = Instant::now();
    let mut last_top_logits = None;
    let show_progress = std::env::var("TQF_QUALIFICATION_PROGRESS").as_deref() == Ok("1");
    let mut completed_decode_steps = 0usize;
    let decode_steps = verify_greedy_sequence(
        &artifact.prompt_tokens,
        &artifact.generated_tokens,
        |input| {
            let decoded = runtime.decode_greedy(input)?;
            if decoded.diagnostics.per_layer_hashes.len() != Qwen36Geometry::NUM_LAYERS
                || decoded.diagnostics.router_trace.len() != Qwen36Geometry::NUM_LAYERS
            {
                return Err(qualification_error(
                    "decode did not emit forty layer hashes and router traces",
                ));
            }
            last_top_logits = Some(decoded.diagnostics.top_logits);
            completed_decode_steps += 1;
            if route_trace_path.is_some() {
                route_steps.push(QualificationRouteStep {
                    decode_step: completed_decode_steps,
                    input_token: input,
                    output_token: decoded.token,
                    layers: decoded
                        .diagnostics
                        .router_trace
                        .iter()
                        .map(|trace| QualificationRouteLayer {
                            layer: trace.layer.0,
                            expert_ids: trace.route.ids.map(|expert| expert.0),
                            weights: trace.route.weights,
                        })
                        .collect(),
                });
            }
            if show_progress {
                let cache = runtime.expert_cache_stats();
                println!(
                    "qualification_decode_step={} input_token={} output_token={} elapsed_ms={} cache_hits={} cache_misses={} raw_miss_bytes={}",
                    completed_decode_steps,
                    input,
                    decoded.token,
                    started.elapsed().as_millis(),
                    cache.hits,
                    cache.misses,
                    cache.raw_miss_bytes.0,
                );
            }
            Ok(decoded.token)
        },
    )
    .map_err(|error| {
        qualification_error(format!(
            "{error}; most recent TQF top logits: {:?}",
            last_top_logits.unwrap_or(
                [crate::runtime::LogitCandidate {
                    token: 0,
                    logit: f32::NEG_INFINITY,
                }; 4]
            )
        ))
    })?;
    let cache = runtime.expert_cache_stats();
    if let Some(path) = route_trace_path {
        write_json_atomic(
            &path,
            &QualificationRouteTrace {
                schema_version: 1,
                fixture_id: artifact.fixture_id.clone(),
                model_source_sha256: artifact.model_source_sha256.clone(),
                steps: route_steps,
            },
        )?;
    }
    Ok(GreedyQualificationReport {
        fixture_id: artifact.fixture_id.clone(),
        prompt_tokens: artifact.prompt_tokens.len(),
        generated_tokens: artifact.generated_tokens.len(),
        decode_steps,
        elapsed_milliseconds: started.elapsed().as_millis(),
        cache_hits: cache.hits,
        cache_misses: cache.misses,
        cache_evictions: cache.evictions,
        cache_resident_bytes: cache.resident_bytes.0,
        raw_miss_bytes: cache.raw_miss_bytes.0,
        broker_reserved_bytes_after_run: broker.snapshot().reserved.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(length: usize) -> GreedyOracleArtifact {
        GreedyOracleArtifact {
            schema_version: ORACLE_SCHEMA_VERSION,
            fixture_id: "unit".to_string(),
            prompt: "A".to_string(),
            prompt_tokens: vec![32],
            generated_tokens: vec![220; length],
            model_source_sha256: pinned::LANGUAGE_CHECKPOINT_SHA256.to_string(),
            oracle_runtime: "fixture".to_string(),
            oracle_revision: "0123456789abcdef".to_string(),
            oracle_parameters: vec!["greedy".to_string()],
        }
    }

    #[test]
    fn accepts_exactly_the_mandatory_sequence_lengths() {
        for length in [1, 16, 128, 512] {
            validate_oracle(&artifact(length)).unwrap();
        }
        for length in [2, 15, 17, 511, 513] {
            assert!(validate_oracle(&artifact(length)).is_err());
        }
    }

    #[test]
    fn sequence_comparison_feeds_each_greedy_result_back_into_decode() {
        let mut inputs = Vec::new();
        let steps = verify_greedy_sequence(&[10, 11], &[12, 13, 14], |input| {
            inputs.push(input);
            Ok(input + 1)
        })
        .unwrap();
        assert_eq!(steps, 4);
        assert_eq!(inputs, vec![10, 11, 12, 13]);
    }

    #[test]
    fn sequence_comparison_reports_the_first_divergence() {
        let error = verify_greedy_sequence(&[10], &[12], |input| Ok(input + 1)).unwrap_err();
        assert!(error.to_string().contains("generated token 0"));
        assert!(error.to_string().contains("expected 12, got 11"));
    }

    #[test]
    #[ignore = "requires the canonical GGUF, converted TQF, and an external oracle artifact"]
    fn canonical_greedy_oracle_matches() {
        let tqf = std::env::var("TQF_CANONICAL_TQF").expect("set TQF_CANONICAL_TQF");
        let gguf = std::env::var("TQF_CANONICAL_GGUF").expect("set TQF_CANONICAL_GGUF");
        let oracle = std::env::var("TQF_CANONICAL_ORACLE").expect("set TQF_CANONICAL_ORACLE");
        let artifact = load_oracle(Path::new(&oracle)).unwrap();
        let report = qualify_oracle(Path::new(&tqf), Path::new(&gguf), &artifact).unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
