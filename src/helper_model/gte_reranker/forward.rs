//! ModernBERT cross-encoder forward pass for the GTE reranker (spec
//! §43/§93). Pre-norm residual blocks, alternating global/local
//! (sliding-window radius 64) bidirectional attention with per-layer
//! RoPE theta, GeGLU MLP, masked-mean pooling, and a
//! dense->GELU->LayerNorm->Linear classification head producing one
//! scalar relevance logit per `(query, document)` pair. See
//! `geometry.rs`'s doc comment for where every one of these choices was
//! cross-checked against the real `transformers` source and the real
//! checkpoint's tensor inventory.

use super::geometry::GteRerankerGeometry;
use super::weights::{GteLayerWeights, GteRerankerWeights};

const HIDDEN: usize = GteRerankerGeometry::HIDDEN_SIZE;
const HEADS: usize = GteRerankerGeometry::NUM_HEADS;
const HEAD_DIM: usize = GteRerankerGeometry::HEAD_DIM;
const INTERMEDIATE: usize = GteRerankerGeometry::INTERMEDIATE_SIZE;
const EPS: f32 = GteRerankerGeometry::LAYER_NORM_EPS;

/// Weight-only (no bias) LayerNorm — every norm in this checkpoint has
/// `norm_bias=false` (confirmed: no `.bias` tensor exists for any norm
/// in the real checkpoint).
fn layernorm(values: &[f32], weight: &[f32]) -> Vec<f32> {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
    let inv_std = 1.0 / (variance + EPS).sqrt();
    values
        .iter()
        .zip(weight)
        .map(|(v, w)| (v - mean) * inv_std * w)
        .collect()
}

fn matvec(weight: &[f32], out_dim: usize, in_dim: usize, input: &[f32]) -> Vec<f32> {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(input.len(), in_dim);
    let mut out = vec![0.0f32; out_dim];
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += row[i] * input[i];
        }
        *slot = acc;
    }
    out
}

/// Abramowitz & Stegun 7.1.26 rational approximation, max absolute
/// error ~1.5e-7 — accurate enough for f32 throughout. ModernBERT uses
/// *exact* erf-based GELU (`hidden_activation="gelu"` resolves to
/// `ACT2FN["gelu"]`, not the tanh approximation `gelu_new`/
/// `gelu_pytorch_tanh`), confirmed against the real `transformers`
/// activation mapping.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.3275911f32;
    let t = 1.0 / (1.0 + p * x);
    let poly = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;
    let y = 1.0 - poly * t * (-x * x).exp();
    sign * y
}

fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Standard rotate-half RoPE over the full `HEAD_DIM` (no partial
/// rotary factor), with a caller-supplied theta (global vs local layer
/// — spec §43's alternating attention pattern).
fn apply_rope(values: &mut [f32; HEAD_DIM], position: usize, theta: f32) {
    const HALF: usize = HEAD_DIM / 2;
    let mut rotated = *values;
    for i in 0..HALF {
        let inv_freq = theta.powf(-(2.0 * i as f32) / HEAD_DIM as f32);
        let angle = position as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let x1 = values[i];
        let x2 = values[i + HALF];
        rotated[i] = x1 * cos - x2 * sin;
        rotated[i + HALF] = x2 * cos + x1 * sin;
    }
    *values = rotated;
}

type Heads = [[f32; HEAD_DIM]; HEADS];

struct LayerActivations {
    q: Vec<Heads>,
    k: Vec<Heads>,
    v: Vec<Heads>,
}

fn project_qkv(
    layer: &GteLayerWeights,
    normed: &[f32],
    position: usize,
    theta: f32,
) -> (Heads, Heads, Heads) {
    // Fused Wqkv output is [Q(768) | K(768) | V(768)] — confirmed
    // against the real `qkv.view(..., 3, num_heads, head_dim)` reshape
    // order: the "3" (qkv) axis is the slowest-varying of the three,
    // so the first HIDDEN elements are all of Q, not interleaved per
    // head.
    let qkv = matvec(&layer.attn_wqkv.values, 3 * HIDDEN, HIDDEN, normed);
    let (q_flat, rest) = qkv.split_at(HIDDEN);
    let (k_flat, v_flat) = rest.split_at(HIDDEN);

    let mut q = [[0.0f32; HEAD_DIM]; HEADS];
    let mut k = [[0.0f32; HEAD_DIM]; HEADS];
    let mut v = [[0.0f32; HEAD_DIM]; HEADS];
    for h in 0..HEADS {
        let mut qh = [0.0f32; HEAD_DIM];
        qh.copy_from_slice(&q_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
        apply_rope(&mut qh, position, theta);
        q[h] = qh;

        let mut kh = [0.0f32; HEAD_DIM];
        kh.copy_from_slice(&k_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
        apply_rope(&mut kh, position, theta);
        k[h] = kh;

        v[h].copy_from_slice(&v_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
    }
    (q, k, v)
}

/// Bidirectional attention with an optional sliding-window radius
/// (`None` = global/full attention; `Some(r)` = local, `|q-kv| <= r`).
fn attention(acts: &LayerActivations, seq_len: usize, window_radius: Option<usize>) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut concat_out = vec![0.0f32; seq_len * HEADS * HEAD_DIM];

    for t in 0..seq_len {
        let lo = window_radius.map(|r| t.saturating_sub(r)).unwrap_or(0);
        let hi = window_radius
            .map(|r| (t + r).min(seq_len - 1))
            .unwrap_or(seq_len - 1);
        for h in 0..HEADS {
            let mut scores: Vec<f32> = (lo..=hi)
                .map(|s| {
                    let dot: f32 = acts.q[t][h]
                        .iter()
                        .zip(&acts.k[s][h])
                        .map(|(a, b)| a * b)
                        .sum();
                    dot * scale
                })
                .collect();
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let dest = &mut concat_out[(t * HEADS + h) * HEAD_DIM..(t * HEADS + h + 1) * HEAD_DIM];
            for (weight, s) in scores.iter().zip(lo..=hi) {
                let weight = weight / sum;
                for (d, value) in dest.iter_mut().zip(&acts.v[s][h]) {
                    *d += weight * value;
                }
            }
        }
    }
    concat_out
}

/// Runs the full 22-layer ModernBERT encoder over one already-tokenized
/// `[CLS] query [SEP] document [SEP]` sequence and returns the
/// post-`final_norm` hidden state for every position, ready for masked-
/// mean pooling. There is no padding in this reference path (one
/// unbatched sequence per call), so "masked" mean is a plain mean over
/// every returned position.
pub fn encode_sequence(weights: &GteRerankerWeights, token_ids: &[u32]) -> Vec<Vec<f32>> {
    let seq_len = token_ids.len();
    let mut hidden: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| {
            let row = id as usize * HIDDEN;
            let raw = &weights.token_embedding.values[row..row + HIDDEN];
            layernorm(raw, &weights.embedding_norm.values)
        })
        .collect();

    for (layer_index, layer) in weights.layers.iter().enumerate() {
        let theta = GteRerankerGeometry::rope_theta(layer_index);
        let window = if GteRerankerGeometry::is_global_layer(layer_index) {
            None
        } else {
            Some(GteRerankerGeometry::LOCAL_WINDOW_RADIUS)
        };

        let mut acts = LayerActivations {
            q: Vec::with_capacity(seq_len),
            k: Vec::with_capacity(seq_len),
            v: Vec::with_capacity(seq_len),
        };
        for (t, h) in hidden.iter().enumerate() {
            // Layer 0's attn_norm is Identity (confirmed against the
            // real checkpoint: no tensor exists for it) — the
            // embeddings LayerNorm already normalized this input.
            let normed = match &layer.attn_norm {
                Some(norm) => layernorm(h, &norm.values),
                None => h.clone(),
            };
            let (q, k, v) = project_qkv(layer, &normed, t, theta);
            acts.q.push(q);
            acts.k.push(k);
            acts.v.push(v);
        }

        let attn_concat = attention(&acts, seq_len, window);
        for t in 0..seq_len {
            let concat = &attn_concat[t * HEADS * HEAD_DIM..(t + 1) * HEADS * HEAD_DIM];
            let attn_out = matvec(&layer.attn_wo.values, HIDDEN, HEADS * HEAD_DIM, concat);
            for i in 0..HIDDEN {
                hidden[t][i] += attn_out[i];
            }
        }

        for h in hidden.iter_mut() {
            let normed2 = layernorm(h, &layer.mlp_norm.values);
            let wi_out = matvec(&layer.mlp_wi.values, 2 * INTERMEDIATE, HIDDEN, &normed2);
            let (input_half, gate_half) = wi_out.split_at(INTERMEDIATE);
            let act: Vec<f32> = input_half
                .iter()
                .zip(gate_half)
                .map(|(a, g)| gelu(*a) * g)
                .collect();
            let down = matvec(&layer.mlp_wo.values, HIDDEN, INTERMEDIATE, &act);
            for i in 0..HIDDEN {
                h[i] += down[i];
            }
        }
    }

    for h in hidden.iter_mut() {
        *h = layernorm(h, &weights.final_norm.values);
    }
    hidden
}

/// The classification head: `dense (no bias) -> exact GELU -> LayerNorm
/// (no bias) -> classifier (WITH bias, the checkpoint's only bias
/// tensor)`. Takes the already mean-pooled `HIDDEN`-length vector and
/// returns the single relevance logit.
pub fn classify_pooled(weights: &GteRerankerWeights, pooled: &[f32]) -> f32 {
    let dense_out = matvec(&weights.head_dense.values, HIDDEN, HIDDEN, pooled);
    let activated: Vec<f32> = dense_out.into_iter().map(gelu).collect();
    let normed = layernorm(&activated, &weights.head_norm.values);
    let dot: f32 = weights
        .classifier_weight
        .values
        .iter()
        .zip(&normed)
        .map(|(w, x)| w * x)
        .sum();
    dot + weights.classifier_bias.values[0]
}
