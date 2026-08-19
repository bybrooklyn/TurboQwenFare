//! Dense bidirectional Qwen3-architecture forward pass for the pplx-embed
//! helper model (spec §37/§86). Unlike `model::qwen36`'s decode loop this
//! has no KV cache and no causal mask: every call processes one whole
//! token sequence at once (matches the reference `modeling.py`'s
//! `bidirectional_mask_function`, which lets every position attend to
//! every other valid position), and there is nothing to persist between
//! calls — a helper-model request is a single forward pass, not a decode
//! loop.

use super::geometry::PplxEmbedGeometry;
use super::weights::{PplxEmbedWeights, PplxLayerWeights};

const HIDDEN: usize = PplxEmbedGeometry::HIDDEN_SIZE;
const HEADS: usize = PplxEmbedGeometry::NUM_HEADS;
const KV_HEADS: usize = PplxEmbedGeometry::NUM_KV_HEADS;
const HEAD_DIM: usize = PplxEmbedGeometry::HEAD_DIM;
const INTERMEDIATE: usize = PplxEmbedGeometry::INTERMEDIATE_SIZE;

fn rmsnorm(values: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    values
        .iter()
        .zip(weight)
        .map(|(v, w)| v * scale * w)
        .collect()
}

/// Row-major matvec: `weight` is `[out_dim, in_dim]`, `out[o] = sum_i
/// weight[o*in_dim+i] * input[i]`.
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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Standard Llama/Qwen3 "rotate half" RoPE applied in place to one
/// `HEAD_DIM`-length vector at absolute position `position`.
fn apply_rope(values: &mut [f32; HEAD_DIM], position: usize) {
    const HALF: usize = HEAD_DIM / 2;
    let mut rotated = *values;
    for i in 0..HALF {
        let inv_freq = PplxEmbedGeometry::ROPE_THETA.powf(-(2.0 * i as f32) / HEAD_DIM as f32);
        let angle = position as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let x1 = values[i];
        let x2 = values[i + HALF];
        rotated[i] = x1 * cos - x2 * sin;
        rotated[i + HALF] = x2 * cos + x1 * sin;
    }
    *values = rotated;
}

type QueryHeads = [[f32; HEAD_DIM]; HEADS];
type KvHeads = [[f32; HEAD_DIM]; KV_HEADS];

struct LayerActivations {
    /// `[seq_len][HEADS][HEAD_DIM]`
    q: Vec<QueryHeads>,
    /// `[seq_len][KV_HEADS][HEAD_DIM]`
    k: Vec<KvHeads>,
    v: Vec<KvHeads>,
}

fn project_qkv(
    layer: &PplxLayerWeights,
    normed: &[f32],
    position: usize,
) -> (QueryHeads, KvHeads, KvHeads) {
    let q_flat = matvec(&layer.q_proj.values, HEADS * HEAD_DIM, HIDDEN, normed);
    let k_flat = matvec(&layer.k_proj.values, KV_HEADS * HEAD_DIM, HIDDEN, normed);
    let v_flat = matvec(&layer.v_proj.values, KV_HEADS * HEAD_DIM, HIDDEN, normed);

    let mut q = [[0.0f32; HEAD_DIM]; HEADS];
    for h in 0..HEADS {
        let raw = &q_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let normed_head = rmsnorm(raw, &layer.q_norm.values, PplxEmbedGeometry::RMS_NORM_EPS);
        let mut head = [0.0f32; HEAD_DIM];
        head.copy_from_slice(&normed_head);
        apply_rope(&mut head, position);
        q[h] = head;
    }

    let mut k = [[0.0f32; HEAD_DIM]; KV_HEADS];
    let mut v = [[0.0f32; HEAD_DIM]; KV_HEADS];
    for h in 0..KV_HEADS {
        let raw_k = &k_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let normed_head = rmsnorm(raw_k, &layer.k_norm.values, PplxEmbedGeometry::RMS_NORM_EPS);
        let mut head = [0.0f32; HEAD_DIM];
        head.copy_from_slice(&normed_head);
        apply_rope(&mut head, position);
        k[h] = head;
        v[h].copy_from_slice(&v_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
    }

    (q, k, v)
}

fn bidirectional_attention(acts: &LayerActivations, seq_len: usize) -> Vec<f32> {
    let group = HEADS / KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut concat_out = vec![0.0f32; seq_len * HEADS * HEAD_DIM];

    for t in 0..seq_len {
        for h in 0..HEADS {
            let kv_head = h / group;
            let mut scores: Vec<f32> = acts
                .k
                .iter()
                .map(|k| {
                    let dot: f32 = acts.q[t][h]
                        .iter()
                        .zip(&k[kv_head])
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
            for (weight, v) in scores.iter().zip(&acts.v) {
                let weight = weight / sum;
                for (d, value) in dest.iter_mut().zip(&v[kv_head]) {
                    *d += weight * value;
                }
            }
        }
    }
    concat_out
}

/// Runs the full 28-layer bidirectional encoder over one token sequence
/// and returns the post-final-norm hidden state for every position
/// (`[seq_len][HIDDEN]`), ready for mean pooling.
pub fn encode_sequence(weights: &PplxEmbedWeights, token_ids: &[u32]) -> Vec<Vec<f32>> {
    let seq_len = token_ids.len();
    let mut hidden: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| {
            let row = id as usize * HIDDEN;
            weights.token_embedding.values[row..row + HIDDEN].to_vec()
        })
        .collect();

    for layer in &weights.layers {
        let mut acts = LayerActivations {
            q: Vec::with_capacity(seq_len),
            k: Vec::with_capacity(seq_len),
            v: Vec::with_capacity(seq_len),
        };
        for (t, h) in hidden.iter().enumerate() {
            let normed = rmsnorm(
                h,
                &layer.input_layernorm.values,
                PplxEmbedGeometry::RMS_NORM_EPS,
            );
            let (q, k, v) = project_qkv(layer, &normed, t);
            acts.q.push(q);
            acts.k.push(k);
            acts.v.push(v);
        }

        let attn_concat = bidirectional_attention(&acts, seq_len);
        for t in 0..seq_len {
            let concat = &attn_concat[t * HEADS * HEAD_DIM..(t + 1) * HEADS * HEAD_DIM];
            let attn_out = matvec(&layer.o_proj.values, HIDDEN, HEADS * HEAD_DIM, concat);
            for i in 0..HIDDEN {
                hidden[t][i] += attn_out[i];
            }
        }

        for h in hidden.iter_mut() {
            let normed2 = rmsnorm(
                h,
                &layer.post_attention_layernorm.values,
                PplxEmbedGeometry::RMS_NORM_EPS,
            );
            let gate = matvec(&layer.gate_proj.values, INTERMEDIATE, HIDDEN, &normed2);
            let up = matvec(&layer.up_proj.values, INTERMEDIATE, HIDDEN, &normed2);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = matvec(&layer.down_proj.values, HIDDEN, INTERMEDIATE, &act);
            for i in 0..HIDDEN {
                h[i] += down[i];
            }
        }
    }

    for h in hidden.iter_mut() {
        *h = rmsnorm(
            h,
            &weights.final_norm.values,
            PplxEmbedGeometry::RMS_NORM_EPS,
        );
    }
    hidden
}
