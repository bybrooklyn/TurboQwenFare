//! Fixed-graph Phase-15 reference decode harness. This is a developer
//! correctness instrument, not a generic runtime abstraction: its loop is
//! statically tied to Qwen3.6's forty layers and emits the hashes, router
//! traces, greedy tokens, and wall-clock stage timings the taskbook requires.

use std::time::{Duration, Instant};

use crate::error::{ModelError, Result};
use crate::experts::RouterResult;
use crate::ids::LayerId;
use crate::model::qwen36::geometry::Qwen36Geometry;

#[derive(Debug, Clone, PartialEq)]
pub struct RouterTrace {
    pub layer: LayerId,
    pub route: RouterResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerHash {
    pub layer: LayerId,
    pub hash: [u8; 32],
}

#[derive(Debug, Clone, Default)]
pub struct DecodeTimings {
    pub embedding: Duration,
    pub layers: Vec<(LayerId, Duration)>,
    pub final_norm: Duration,
    pub lm_head: Duration,
    pub sampling: Duration,
}

#[derive(Debug, Clone)]
pub struct DecodeDiagnostics {
    pub per_layer_hashes: Vec<LayerHash>,
    pub router_trace: Vec<RouterTrace>,
    pub top_logits: [LogitCandidate; 4],
    pub timings: DecodeTimings,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitCandidate {
    pub token: u32,
    pub logit: f32,
}

pub fn top_logit_candidates(logits: &[f32]) -> [LogitCandidate; 4] {
    let mut top = [LogitCandidate {
        token: 0,
        logit: f32::NEG_INFINITY,
    }; 4];
    for (index, &logit) in logits.iter().enumerate() {
        let candidate = LogitCandidate {
            token: index as u32,
            logit,
        };
        if let Some(position) = top.iter().position(|current| logit > current.logit) {
            top[position..].rotate_right(1);
            top[position] = candidate;
        }
    }
    top
}

#[derive(Debug, Clone)]
pub struct DecodeToken {
    pub token: u32,
    pub diagnostics: DecodeDiagnostics,
}

/// Result returned by the concrete Qwen layer callback. A full attention
/// layer leaves `router` populated after its MoE tail, as does a GDN layer.
pub struct LayerStep {
    pub hidden: Vec<f32>,
    pub router: Option<RouterResult>,
}

fn require_hidden(stage: &'static str, values: &[f32]) -> Result<()> {
    if values.len() == Qwen36Geometry::HIDDEN_SIZE {
        Ok(())
    } else {
        Err(ModelError::Shape {
            tensor: stage,
            expected: Qwen36Geometry::HIDDEN_SIZE,
            actual: values.len(),
        }
        .into())
    }
}

fn hash_hidden(values: &[f32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Runs one token through an intentionally fixed forty-layer graph. Closures
/// are monomorphized at the call site, so this diagnostic harness does not put
/// a virtual generic-model interface in a production hot loop. The actual
/// checkpoint loader supplies the Qwen-specific projection/kernel closures.
pub fn decode_greedy<E, L, N, H>(
    input_token: u32,
    embedding: E,
    mut layer: L,
    final_norm: N,
    lm_head: H,
) -> Result<DecodeToken>
where
    E: FnOnce(u32) -> Result<Vec<f32>>,
    L: FnMut(LayerId, Vec<f32>) -> Result<LayerStep>,
    N: FnOnce(Vec<f32>) -> Result<Vec<f32>>,
    H: FnOnce(Vec<f32>) -> Result<Vec<f32>>,
{
    let mut timings = DecodeTimings::default();
    let start = Instant::now();
    let mut hidden = embedding(input_token)?;
    timings.embedding = start.elapsed();
    require_hidden("embedding output", &hidden)?;

    let mut per_layer_hashes = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
    let mut router_trace = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
    for layer_index in 0..Qwen36Geometry::NUM_LAYERS {
        let layer_id = LayerId(layer_index as u8);
        let start = Instant::now();
        let result = layer(layer_id, hidden)?;
        timings.layers.push((layer_id, start.elapsed()));
        require_hidden("layer output", &result.hidden)?;
        per_layer_hashes.push(LayerHash {
            layer: layer_id,
            hash: hash_hidden(&result.hidden),
        });
        if let Some(route) = result.router {
            router_trace.push(RouterTrace {
                layer: layer_id,
                route,
            });
        }
        hidden = result.hidden;
    }

    let start = Instant::now();
    hidden = final_norm(hidden)?;
    timings.final_norm = start.elapsed();
    require_hidden("final norm output", &hidden)?;

    let start = Instant::now();
    let logits = lm_head(hidden)?;
    timings.lm_head = start.elapsed();
    if logits.is_empty() {
        return Err(ModelError::Shape {
            tensor: "LM head logits",
            expected: 1,
            actual: 0,
        }
        .into());
    }
    let start = Instant::now();
    let top_logits = top_logit_candidates(&logits);
    let token = top_logits[0].token;
    timings.sampling = start.elapsed();

    Ok(DecodeToken {
        token,
        diagnostics: DecodeDiagnostics {
            per_layer_hashes,
            router_trace,
            top_logits,
            timings,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_token(input: u32) -> DecodeToken {
        decode_greedy(
            input,
            |_| Ok(vec![0.0; Qwen36Geometry::HIDDEN_SIZE]),
            |layer, mut hidden| {
                hidden[layer.0 as usize % Qwen36Geometry::HIDDEN_SIZE] += 1.0;
                Ok(LayerStep {
                    hidden,
                    router: (layer.0 % 4 == 3).then(|| RouterResult {
                        ids: [crate::ids::ExpertId(0); Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN],
                        weights: [1.0 / Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN as f32;
                            Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN],
                    }),
                })
            },
            Ok,
            |hidden| Ok(vec![hidden.iter().sum(), 1.0, -1.0]),
        )
        .unwrap()
    }

    #[test]
    fn fixed_decode_visits_all_forty_layers_and_emits_developer_data() {
        let result = reference_token(7);
        assert_eq!(result.token, 0);
        assert_eq!(result.diagnostics.per_layer_hashes.len(), 40);
        assert_eq!(result.diagnostics.timings.layers.len(), 40);
        assert_eq!(result.diagnostics.router_trace.len(), 10);
        assert_eq!(result.diagnostics.per_layer_hashes[0].layer, LayerId(0));
        assert_eq!(result.diagnostics.per_layer_hashes[39].layer, LayerId(39));
    }

    #[test]
    fn synthetic_harness_is_deterministic_across_five_hundred_twelve_tokens() {
        let tokens: Vec<u32> = (0..512).map(|input| reference_token(input).token).collect();
        assert_eq!(tokens, vec![0; 512]);
    }
}
