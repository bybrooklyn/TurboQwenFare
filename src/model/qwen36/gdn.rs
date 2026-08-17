//! Gated DeltaNet (spec §284, phase 12): the recurrent layer used by 30 of
//! Qwen3.6's 40 layers (spec §8/§11). Implemented in the exact stage order
//! spec §284 names:
//!
//! 1. the checkpoint's four projections (`project`): fused Q/K/V plus Z,
//!    alpha, and beta;
//! 2. causal depthwise conv "tail" over the concatenated q/k/v (`ConvTailState`);
//! 3. per-head q/k L2 normalization (`qk_norm`);
//! 4. the FP32 delta-rule recurrent update (`recurrent_step`);
//! 5. gated norm (`gated_norm`);
//! 6. output projection (`out_projection`).
//!
//! Item 7 ("fused input projection only after parity") is deliberately not
//! implemented here: it names a *later* optimization — collapsing the four
//! separate matmuls in `project` into one fused `in_proj_qkvz`-style matmul
//! — that spec §284 itself gates on this unfused version first proving
//! correct. `project`'s four-output contract doesn't change either way, so
//! that fusion can land later without touching this module's callers.
//!
//! The projection names and gate/decay formula follow the upstream Qwen3.5
//! reference implementation, which is the architecture Qwen3.6 exports to
//! GGUF. Numeric parity with an actual Qwen3.6 checkpoint remains a Phase-15
//! qualification item; this module deliberately has no invented fallback
//! parameterization.
//!
//! The standalone `project` oracle below accepts dense f32 fixtures so its
//! recurrence tests stay fast and deterministic. The fixed Qwen runtime
//! separately feeds the real checkpoint's mixed F32/Q8_0 projections through
//! `LoadedQwen36Tensor::matvec` before calling these same recurrent helpers.

use crate::backend::reference;
use crate::error::{ModelError, Result};
use crate::ids::{Bytes, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};
use crate::model::qwen36::geometry::Qwen36Geometry;
use crate::model::qwen36::weights::Qwen36Activation;

const KEY_HEADS: usize = Qwen36Geometry::GDN_KEY_HEADS; // 16
const VALUE_HEADS: usize = Qwen36Geometry::GDN_VALUE_HEADS; // 32
const KEY_HEAD_DIM: usize = Qwen36Geometry::GDN_KEY_HEAD_DIM; // 128
const VALUE_HEAD_DIM: usize = Qwen36Geometry::GDN_VALUE_HEAD_DIM; // 128
const KEY_DIM: usize = Qwen36Geometry::GDN_KEY_DIM; // 2048
const VALUE_DIM: usize = Qwen36Geometry::GDN_VALUE_DIM; // 4096
const HIDDEN_SIZE: usize = Qwen36Geometry::HIDDEN_SIZE;
const CONV_WIDTH: usize = Qwen36Geometry::GDN_CONV_WIDTH; // 4
const CONV_CHANNELS: usize = Qwen36Geometry::GDN_CONV_CHANNELS; // 8192 = KEY_DIM + HIDDEN_SIZE + VALUE_DIM

/// Output of stage 1 (`project`): the four independently-computed
/// checkpoint projections. `a` and `b` each have one value per value head;
/// they do not come from a synthetic packed gate matrix.
#[derive(Debug, Clone)]
pub struct GdnProjected {
    pub q: Vec<f32>, // [KEY_DIM]
    pub k: Vec<f32>, // [KEY_DIM]
    pub v: Vec<f32>, // [VALUE_DIM]
    pub a: Vec<f32>, // [VALUE_HEADS] decay-gate logits
    pub b: Vec<f32>, // [VALUE_HEADS] write-gate (beta) logits
    pub z: Vec<f32>, // [VALUE_DIM] output-gate logits
}

/// Per-value-head recurrence controls read from `ssm_alpha`, `ssm_beta`,
/// `ssm_a`, and `ssm_dt` respectively. Grouping them prevents the hot
/// recurrence API from obscuring their required checkpoint association.
pub struct GdnRecurrentParameters<'a> {
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    /// Canonical GGUF stores `-exp(source A_log)`, folded by llama.cpp's
    /// converter. It is already negative and must not be exponentiated again.
    pub a_neg_exp: &'a [f32],
    pub dt_bias: &'a [f32],
}

/// Row-major `[rows, cols]` weight times a length-`cols` vector.
fn dense_matvec(weight: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        rows * cols,
        "dense_matvec: weight shape mismatch"
    );
    assert_eq!(
        x.len(),
        cols,
        "dense_matvec: input length {} != cols {cols}",
        x.len()
    );
    (0..rows)
        .map(|r| {
            let row = &weight[r * cols..(r + 1) * cols];
            row.iter().zip(x).map(|(w, v)| w * v).sum()
        })
        .collect()
}

/// Stage 1 (spec §284 item 1): Qwen's four logical projection groups from
/// the hidden-state input. `qkv_weight` is `[KEY_DIM + KEY_DIM + VALUE_DIM,
/// HIDDEN_SIZE]`; `z_weight` is `[VALUE_DIM, HIDDEN_SIZE]`; and
/// `alpha_weight`/`beta_weight` are `[VALUE_HEADS, HIDDEN_SIZE]`, all
/// row-major. These directly match `attn_qkv`, `attn_gate`, `ssm_alpha`, and
/// `ssm_beta` in the canonical GGUF language tensors.
pub fn project(
    hidden: &[f32],
    qkv_weight: &[f32],
    z_weight: &[f32],
    alpha_weight: &[f32],
    beta_weight: &[f32],
) -> GdnProjected {
    assert_eq!(hidden.len(), HIDDEN_SIZE);
    let qkv = dense_matvec(qkv_weight, hidden, CONV_CHANNELS, HIDDEN_SIZE);
    let q = qkv[..KEY_DIM].to_vec();
    let k = qkv[KEY_DIM..KEY_DIM * 2].to_vec();
    let v = qkv[KEY_DIM * 2..].to_vec();
    let z = dense_matvec(z_weight, hidden, VALUE_DIM, HIDDEN_SIZE);
    let a = dense_matvec(alpha_weight, hidden, VALUE_HEADS, HIDDEN_SIZE);
    let b = dense_matvec(beta_weight, hidden, VALUE_HEADS, HIDDEN_SIZE);
    GdnProjected { q, k, v, a, b, z }
}

/// Stage 2 (spec §284 item 2): short causal depthwise conv over the
/// concatenated `(q, k, v)` — `CONV_CHANNELS` independent per-channel
/// `CONV_WIDTH`-tap causal filters, SiLU-activated (the standard Mamba/GDN
/// conv-tail activation). Streaming: holds the last `CONV_WIDTH - 1`
/// timesteps per channel so `step` can process one token at a time without
/// ever looking ahead.
#[derive(Debug, Clone)]
pub struct ConvTailState {
    /// `[CONV_CHANNELS, CONV_WIDTH - 1]`, oldest-to-newest per channel.
    history: Vec<f32>,
}

impl ConvTailState {
    pub fn new() -> Self {
        Self {
            history: vec![0.0; CONV_CHANNELS * (CONV_WIDTH - 1)],
        }
    }

    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|v| *v = 0.0);
    }

    /// `qkv` is one timestep's concatenated `[q; k; v]`, length
    /// `CONV_CHANNELS`. `weight` is `[CONV_CHANNELS, CONV_WIDTH]` row-major
    /// (tap 0 = oldest, tap `CONV_WIDTH - 1` = current timestep); `bias` is
    /// `[CONV_CHANNELS]`.
    pub fn step(&mut self, qkv: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        assert_eq!(qkv.len(), CONV_CHANNELS);
        assert_eq!(weight.len(), CONV_CHANNELS * CONV_WIDTH);
        assert_eq!(bias.len(), CONV_CHANNELS);

        let taps = CONV_WIDTH - 1;
        let mut out = vec![0.0f32; CONV_CHANNELS];
        for c in 0..CONV_CHANNELS {
            let w = &weight[c * CONV_WIDTH..(c + 1) * CONV_WIDTH];
            let hist = &self.history[c * taps..(c + 1) * taps];
            let mut acc = bias[c];
            for (t, &h) in hist.iter().enumerate() {
                acc += w[t] * h;
            }
            acc += w[taps] * qkv[c];
            // SiLU activation.
            out[c] = acc / (1.0 + (-acc).exp());
        }

        // Slide each channel's history window forward by one timestep.
        for c in 0..CONV_CHANNELS {
            let base = c * taps;
            for t in 0..taps.saturating_sub(1) {
                self.history[base + t] = self.history[base + t + 1];
            }
            if taps > 0 {
                self.history[base + taps - 1] = qkv[c];
            }
        }

        out
    }

    /// Writes the causal convolution result into caller-owned storage.  The
    /// fixed graph uses this so its output activation can be reserved by the
    /// memory broker before allocation.
    pub fn step_into(&mut self, qkv: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32]) {
        assert_eq!(qkv.len(), CONV_CHANNELS);
        assert_eq!(weight.len(), CONV_CHANNELS * CONV_WIDTH);
        assert_eq!(bias.len(), CONV_CHANNELS);
        assert_eq!(out.len(), CONV_CHANNELS);

        let taps = CONV_WIDTH - 1;
        for c in 0..CONV_CHANNELS {
            let weights = &weight[c * CONV_WIDTH..(c + 1) * CONV_WIDTH];
            let history = &self.history[c * taps..(c + 1) * taps];
            let mut acc = bias[c];
            for (tap, &value) in history.iter().enumerate() {
                acc += weights[tap] * value;
            }
            acc += weights[taps] * qkv[c];
            out[c] = acc / (1.0 + (-acc).exp());
        }
        for c in 0..CONV_CHANNELS {
            let base = c * taps;
            for tap in 0..taps.saturating_sub(1) {
                self.history[base + tap] = self.history[base + tap + 1];
            }
            if taps > 0 {
                self.history[base + taps - 1] = qkv[c];
            }
        }
    }

    /// Variant for the canonical GGUF tensor set, whose `ssm_conv1d` has no
    /// persisted bias tensor.  Keeping zero bias implicit avoids allocating a
    /// transient 8,192-element convenience vector in the model hot path.
    pub fn step_without_bias_into(&mut self, qkv: &[f32], weight: &[f32], out: &mut [f32]) {
        assert_eq!(qkv.len(), CONV_CHANNELS);
        assert_eq!(weight.len(), CONV_CHANNELS * CONV_WIDTH);
        assert_eq!(out.len(), CONV_CHANNELS);
        let taps = CONV_WIDTH - 1;
        for c in 0..CONV_CHANNELS {
            let weights = &weight[c * CONV_WIDTH..(c + 1) * CONV_WIDTH];
            let history = &self.history[c * taps..(c + 1) * taps];
            let mut acc = 0.0;
            for (tap, &value) in history.iter().enumerate() {
                acc += weights[tap] * value;
            }
            acc += weights[taps] * qkv[c];
            out[c] = acc / (1.0 + (-acc).exp());
        }
        for c in 0..CONV_CHANNELS {
            let base = c * taps;
            for tap in 0..taps.saturating_sub(1) {
                self.history[base + tap] = self.history[base + tap + 1];
            }
            if taps > 0 {
                self.history[base + taps - 1] = qkv[c];
            }
        }
    }
}

impl Default for ConvTailState {
    fn default() -> Self {
        Self::new()
    }
}

/// Stage 3 (spec §284 item 3): per-head L2 normalization of q and k. Unlike
/// the full-attention branch, Gated DeltaNet has no learned Q/K norm tensors;
/// the upstream recurrent kernel uses `l2norm(..., eps=1e-6)`.
pub fn qk_norm(q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), KEY_DIM);
    assert_eq!(k.len(), KEY_DIM);
    let l2norm = |values: &[f32]| {
        values
            .chunks_exact(KEY_HEAD_DIM)
            .flat_map(|head| {
                let norm = head.iter().map(|value| value * value).sum::<f32>().sqrt();
                let inv_norm = 1.0 / norm.max(1e-6);
                head.iter().map(move |value| value * inv_norm)
            })
            .collect()
    };
    let q_out = l2norm(q);
    let k_out = l2norm(k);
    (q_out, k_out)
}

/// In-place, broker-safe GDN q/k normalization for the fixed graph.  The
/// caller owns the activations already, so no new heap buffers are needed.
pub fn qk_norm_in_place(q: &mut Qwen36Activation, k: &mut Qwen36Activation) -> Result<()> {
    if q.values.len() != KEY_DIM || k.values.len() != KEY_DIM {
        return Err(ModelError::Shape {
            tensor: "GDN q/k normalization",
            expected: KEY_DIM,
            actual: q.values.len().min(k.values.len()),
        }
        .into());
    }
    for values in [&mut q.values, &mut k.values] {
        for head in values.chunks_exact_mut(KEY_HEAD_DIM) {
            let norm = head.iter().map(|value| value * value).sum::<f32>().sqrt();
            let inverse = 1.0 / norm.max(1e-6);
            for value in head {
                *value *= inverse;
            }
        }
    }
    Ok(())
}

/// Broadcasts `KEY_HEADS` per-head vectors of `head_dim` up to
/// `VALUE_HEADS` heads in canonical GGUF order. llama.cpp's Qwen3.5
/// converter reorders V/Z/alpha/beta/A/dt/out-projection heads from the HF
/// grouped layout to tiled order, so Q/K repeat as
/// `[K0..K15, K0..K15]`, not `[K0,K0,K1,K1,...]`.
fn repeat_heads(x: &[f32], head_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), KEY_HEADS * head_dim);
    let group = VALUE_HEADS / KEY_HEADS;
    let mut out = vec![0.0f32; VALUE_HEADS * head_dim];
    for g in 0..group {
        for h in 0..KEY_HEADS {
            let src = &x[h * head_dim..(h + 1) * head_dim];
            let dst_head = g * KEY_HEADS + h;
            out[dst_head * head_dim..(dst_head + 1) * head_dim].copy_from_slice(src);
        }
    }
    out
}

/// Per-layer Gated DeltaNet recurrent state (spec §284: "Store one
/// recurrent-state object per GDN layer with exact reset/snapshot APIs.").
/// `recurrent` is `[VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM]` f32 (~2 MiB,
/// spec §11), constant-size with context length.
#[derive(Debug)]
pub struct GdnState {
    layer: LayerId,
    broker: MemoryBroker,
    recurrent: Vec<f32>,
    conv: ConvTailState,
    // Must be dropped after both physical state vectors.
    _lease: MemoryLease,
}

impl GdnState {
    pub fn bytes() -> Bytes {
        Bytes(
            ((VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM) + (CONV_CHANNELS * (CONV_WIDTH - 1)))
                as u64
                * std::mem::size_of::<f32>() as u64,
        )
    }

    /// Reserves the complete recurrent and convolution state before either
    /// vector is allocated. A GDN state is fixed live model state, never an
    /// untracked convenience allocation.
    pub fn new(broker: &MemoryBroker, layer: LayerId) -> crate::error::Result<Self> {
        let lease = broker.reserve(MemoryOwner::GdnState, MemoryClass::Fixed, Self::bytes(), 64)?;
        Ok(Self {
            layer,
            broker: broker.clone(),
            recurrent: vec![0.0; VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM],
            conv: ConvTailState::new(),
            _lease: lease,
        })
    }

    pub fn reset(&mut self) {
        self.recurrent.iter_mut().for_each(|v| *v = 0.0);
        self.conv.reset();
    }

    pub fn conv_tail_mut(&mut self) -> &mut ConvTailState {
        &mut self.conv
    }

    /// Exact snapshot: an owned copy of the full state, restorable via
    /// `restore` — e.g. for prefix-cache save points (spec §66/§67).
    pub fn snapshot(&self) -> crate::error::Result<GdnState> {
        let mut snapshot = Self::new(&self.broker, self.layer)?;
        snapshot.recurrent.copy_from_slice(&self.recurrent);
        snapshot.conv = self.conv.clone();
        Ok(snapshot)
    }

    pub fn restore(&mut self, snapshot: &GdnState) {
        self.recurrent.copy_from_slice(&snapshot.recurrent);
        self.conv = snapshot.conv.clone();
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Stage 4 (spec §284 item 4): the FP32 delta-rule recurrent update for one
/// decoded timestep. `q`, `k` are `[KEY_DIM]` (pre-`repeat_heads`, already
/// through `qk_norm`); `v` is `[VALUE_DIM]`; every field of `parameters` is
/// `[VALUE_HEADS]`. Returns `y`, the `[VALUE_DIM]`
/// recurrent read-out, and mutates `state.recurrent` in place.
///
/// Per value head `h` with state matrix `S_h` (`[KEY_HEAD_DIM,
/// VALUE_HEAD_DIM]`):
/// ```text
/// g_h     = a_neg_exp_h * softplus(a_h + dt_bias_h)
///             where GGUF `a_neg_exp_h = -exp(source A_log_h)`
/// decay_h = exp(g_h)                          // state retention in (0,1)
/// beta_h  = sigmoid(b_h)                     // write strength in (0,1)
/// S_h     = decay_h * S_h                    // decay before reading the prediction
/// pred_h  = S_h^T @ k_h                      // value the decayed state predicts for k_h
/// delta_h = beta_h * (v_h - pred_h)          // delta rule's correction term
/// S_h     = S_h + outer(k_h, delta_h)
/// y_h     = S_h^T @ q_h                      // read out with the query
/// ```
pub fn recurrent_step(
    state: &mut GdnState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    parameters: GdnRecurrentParameters<'_>,
) -> Vec<f32> {
    assert_eq!(q.len(), KEY_DIM);
    assert_eq!(k.len(), KEY_DIM);
    assert_eq!(v.len(), VALUE_DIM);
    assert_eq!(parameters.alpha.len(), VALUE_HEADS);
    assert_eq!(parameters.beta.len(), VALUE_HEADS);
    assert_eq!(parameters.a_neg_exp.len(), VALUE_HEADS);
    assert_eq!(parameters.dt_bias.len(), VALUE_HEADS);

    let q_full = repeat_heads(q, KEY_HEAD_DIM);
    let k_full = repeat_heads(k, KEY_HEAD_DIM);
    let mut y = vec![0.0f32; VALUE_DIM];
    for h in 0..VALUE_HEADS {
        let decay =
            (parameters.a_neg_exp[h] * softplus(parameters.alpha[h] + parameters.dt_bias[h])).exp();
        let beta = sigmoid(parameters.beta[h]);

        let q_h = &q_full[h * KEY_HEAD_DIM..(h + 1) * KEY_HEAD_DIM];
        let k_h = &k_full[h * KEY_HEAD_DIM..(h + 1) * KEY_HEAD_DIM];
        let v_h = &v[h * VALUE_HEAD_DIM..(h + 1) * VALUE_HEAD_DIM];

        let s_base = h * KEY_HEAD_DIM * VALUE_HEAD_DIM;
        let s_h = &mut state.recurrent[s_base..s_base + KEY_HEAD_DIM * VALUE_HEAD_DIM];

        // The official recurrent kernel decays the state before computing
        // `kv_mem`; predicting from the old state changes every noninitial
        // token even if the final update applies the same decay factor.
        for value in s_h.iter_mut() {
            *value *= decay;
        }

        // pred = S_h^T @ k_h : pred[j] = sum_i S_h[i,j] * k_h[i]
        let mut pred = vec![0.0f32; VALUE_HEAD_DIM];
        for i in 0..KEY_HEAD_DIM {
            let k_i = k_h[i];
            if k_i == 0.0 {
                continue;
            }
            let row = &s_h[i * VALUE_HEAD_DIM..(i + 1) * VALUE_HEAD_DIM];
            for j in 0..VALUE_HEAD_DIM {
                pred[j] += row[j] * k_i;
            }
        }

        let mut delta = vec![0.0f32; VALUE_HEAD_DIM];
        for j in 0..VALUE_HEAD_DIM {
            delta[j] = beta * (v_h[j] - pred[j]);
        }

        // S_h = S_h + outer(k_h, delta); decay was applied before pred.
        for i in 0..KEY_HEAD_DIM {
            let k_i = k_h[i];
            let row = &mut s_h[i * VALUE_HEAD_DIM..(i + 1) * VALUE_HEAD_DIM];
            for j in 0..VALUE_HEAD_DIM {
                row[j] += k_i * delta[j];
            }
        }

        // y_h = S_h^T @ q_h
        let y_h = &mut y[h * VALUE_HEAD_DIM..(h + 1) * VALUE_HEAD_DIM];
        for i in 0..KEY_HEAD_DIM {
            // The upstream recurrent reference scales L2-normalized queries
            // by `key_head_dim^-0.5` before the state readout.
            let q_i = q_h[i] * (KEY_HEAD_DIM as f32).sqrt().recip();
            if q_i == 0.0 {
                continue;
            }
            let row = &s_h[i * VALUE_HEAD_DIM..(i + 1) * VALUE_HEAD_DIM];
            for j in 0..VALUE_HEAD_DIM {
                y_h[j] += row[j] * q_i;
            }
        }
    }

    y
}

/// Broker-accounted fixed-graph recurrence.  It avoids materializing the
/// duplicated 32-head Q/K vectors: after the canonical GGUF converter's
/// tiled value-head reorder, head `h` reads directly from key head `h % 16`.
pub fn recurrent_step_accounted(
    broker: &MemoryBroker,
    state: &mut GdnState,
    q: &Qwen36Activation,
    k: &Qwen36Activation,
    v: &Qwen36Activation,
    parameters: GdnRecurrentParameters<'_>,
) -> Result<Qwen36Activation> {
    if q.values.len() != KEY_DIM
        || k.values.len() != KEY_DIM
        || v.values.len() != VALUE_DIM
        || parameters.alpha.len() != VALUE_HEADS
        || parameters.beta.len() != VALUE_HEADS
        || parameters.a_neg_exp.len() != VALUE_HEADS
        || parameters.dt_bias.len() != VALUE_HEADS
    {
        return Err(ModelError::Shape {
            tensor: "GDN recurrent inputs",
            expected: VALUE_DIM,
            actual: v.values.len(),
        }
        .into());
    }

    let mut output = Qwen36Activation::zeros(broker, VALUE_DIM)?;
    for head in 0..VALUE_HEADS {
        let decay = (parameters.a_neg_exp[head]
            * softplus(parameters.alpha[head] + parameters.dt_bias[head]))
        .exp();
        let beta = sigmoid(parameters.beta[head]);
        let key_head = head % KEY_HEADS;
        let q_head = &q.values[key_head * KEY_HEAD_DIM..(key_head + 1) * KEY_HEAD_DIM];
        let k_head = &k.values[key_head * KEY_HEAD_DIM..(key_head + 1) * KEY_HEAD_DIM];
        let v_head = &v.values[head * VALUE_HEAD_DIM..(head + 1) * VALUE_HEAD_DIM];
        let base = head * KEY_HEAD_DIM * VALUE_HEAD_DIM;
        let state_head = &mut state.recurrent[base..base + KEY_HEAD_DIM * VALUE_HEAD_DIM];

        for value in state_head.iter_mut() {
            *value *= decay;
        }

        let mut prediction = [0.0f32; VALUE_HEAD_DIM];
        for key_index in 0..KEY_HEAD_DIM {
            let row = &state_head[key_index * VALUE_HEAD_DIM..(key_index + 1) * VALUE_HEAD_DIM];
            for value_index in 0..VALUE_HEAD_DIM {
                prediction[value_index] += row[value_index] * k_head[key_index];
            }
        }
        let mut delta = [0.0f32; VALUE_HEAD_DIM];
        for value_index in 0..VALUE_HEAD_DIM {
            delta[value_index] = beta * (v_head[value_index] - prediction[value_index]);
        }
        for key_index in 0..KEY_HEAD_DIM {
            let row = &mut state_head[key_index * VALUE_HEAD_DIM..(key_index + 1) * VALUE_HEAD_DIM];
            for value_index in 0..VALUE_HEAD_DIM {
                row[value_index] += k_head[key_index] * delta[value_index];
            }
        }
        let target = &mut output.values[head * VALUE_HEAD_DIM..(head + 1) * VALUE_HEAD_DIM];
        for key_index in 0..KEY_HEAD_DIM {
            let row = &state_head[key_index * VALUE_HEAD_DIM..(key_index + 1) * VALUE_HEAD_DIM];
            let scaled_query = q_head[key_index] * (KEY_HEAD_DIM as f32).sqrt().recip();
            for value_index in 0..VALUE_HEAD_DIM {
                target[value_index] += row[value_index] * scaled_query;
            }
        }
    }
    Ok(output)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

/// Stage 5 (spec §284 item 5): gated norm — RMSNorm the recurrent read-out,
/// then gate it elementwise with `SiLU(z)` (the standard Mamba2/Gated
/// DeltaNet "gated RMSNorm": normalize first, then apply the output gate).
pub fn gated_norm(y: &[f32], z: &[f32], weight: &[f32]) -> Vec<f32> {
    assert_eq!(y.len(), VALUE_DIM);
    assert_eq!(z.len(), VALUE_DIM);
    if weight.len() == VALUE_HEAD_DIM {
        let mut output = vec![0.0; VALUE_DIM];
        for head in 0..VALUE_HEADS {
            let input = &y[head * VALUE_HEAD_DIM..(head + 1) * VALUE_HEAD_DIM];
            let output_head = &mut output[head * VALUE_HEAD_DIM..(head + 1) * VALUE_HEAD_DIM];
            let inverse = 1.0
                / (input.iter().map(|value| value * value).sum::<f32>() / VALUE_HEAD_DIM as f32
                    + 1e-6)
                    .sqrt();
            for index in 0..VALUE_HEAD_DIM {
                output_head[index] = input[index]
                    * inverse
                    * weight[index]
                    * (z[head * VALUE_HEAD_DIM + index]
                        / (1.0 + (-z[head * VALUE_HEAD_DIM + index]).exp()));
            }
        }
        output
    } else {
        let normed = reference::rmsnorm(y, weight, 1, VALUE_DIM, 1e-6);
        let gate = reference::silu(z);
        normed.iter().zip(&gate).map(|(n, g)| n * g).collect()
    }
}

/// Canonical 128-element per-head gated norm, writing only to an activation
/// that was reserved before allocation.
pub fn gated_norm_accounted(
    broker: &MemoryBroker,
    y: &Qwen36Activation,
    z: &Qwen36Activation,
    weight: &[f32],
) -> Result<Qwen36Activation> {
    if y.values.len() != VALUE_DIM || z.values.len() != VALUE_DIM || weight.len() != VALUE_HEAD_DIM
    {
        return Err(ModelError::Shape {
            tensor: "GDN gated norm",
            expected: VALUE_HEAD_DIM,
            actual: weight.len(),
        }
        .into());
    }
    let mut output = Qwen36Activation::zeros(broker, VALUE_DIM)?;
    for head in 0..VALUE_HEADS {
        let input = &y.values[head * VALUE_HEAD_DIM..(head + 1) * VALUE_HEAD_DIM];
        let inverse = 1.0
            / (input.iter().map(|value| value * value).sum::<f32>() / VALUE_HEAD_DIM as f32 + 1e-6)
                .sqrt();
        for index in 0..VALUE_HEAD_DIM {
            let position = head * VALUE_HEAD_DIM + index;
            let gate = z.values[position] / (1.0 + (-z.values[position]).exp());
            output.values[position] = input[index] * inverse * weight[index] * gate;
        }
    }
    Ok(output)
}

/// Stage 6 (spec §284 item 6): output projection back to `HIDDEN_SIZE`.
/// `out_weight` is `[HIDDEN_SIZE, VALUE_DIM]` row-major.
pub fn out_projection(y_gated: &[f32], out_weight: &[f32]) -> Vec<f32> {
    dense_matvec(out_weight, y_gated, HIDDEN_SIZE, VALUE_DIM)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_weight(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        for i in 0..rows.min(cols) {
            w[i * cols + i] = 1.0;
        }
        w
    }

    fn broker() -> MemoryBroker {
        MemoryBroker::new(Bytes(32 * 1024 * 1024))
    }

    fn state() -> GdnState {
        GdnState::new(&broker(), LayerId(0)).unwrap()
    }

    fn recurrent_parameters() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.0; VALUE_HEADS],
            vec![0.0; VALUE_HEADS],
            vec![-1.0; VALUE_HEADS],
            vec![0.0; VALUE_HEADS],
        )
    }

    #[test]
    fn gdn_state_and_snapshot_are_broker_accounted() {
        let broker = broker();
        let state = GdnState::new(&broker, LayerId(0)).unwrap();
        assert_eq!(broker.snapshot().reserved, GdnState::bytes());
        let snapshot = state.snapshot().unwrap();
        assert_eq!(broker.snapshot().reserved, Bytes(GdnState::bytes().0 * 2));
        drop(snapshot);
        drop(state);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn project_uses_the_canonical_qkv_z_alpha_beta_groups() {
        let hidden = vec![1.0f32; HIDDEN_SIZE];
        let qkv_weight = vec![0.0f32; CONV_CHANNELS * HIDDEN_SIZE];
        let z_weight = vec![0.0f32; VALUE_DIM * HIDDEN_SIZE];
        let alpha_weight = identity_weight(VALUE_HEADS, HIDDEN_SIZE);
        let beta_weight = identity_weight(VALUE_HEADS, HIDDEN_SIZE);

        let projected = project(&hidden, &qkv_weight, &z_weight, &alpha_weight, &beta_weight);
        assert_eq!(projected.q.len(), KEY_DIM);
        assert_eq!(projected.k.len(), KEY_DIM);
        assert_eq!(projected.v.len(), VALUE_DIM);
        assert_eq!(projected.a.len(), VALUE_HEADS);
        assert_eq!(projected.b.len(), VALUE_HEADS);
        assert_eq!(projected.z.len(), VALUE_DIM);
        assert_eq!(projected.a[0], 1.0);
        assert_eq!(projected.b[0], 1.0);
    }

    #[test]
    fn conv_tail_current_tap_only_reduces_to_silu_of_input() {
        // Weight with all history taps zero and the current-timestep tap
        // = 1, bias = 0: step() must reduce to plain SiLU(input).
        let mut weight = vec![0.0f32; CONV_CHANNELS * CONV_WIDTH];
        for c in 0..CONV_CHANNELS {
            weight[c * CONV_WIDTH + (CONV_WIDTH - 1)] = 1.0;
        }
        let bias = vec![0.0f32; CONV_CHANNELS];
        let mut state = ConvTailState::new();

        let input: Vec<f32> = (0..CONV_CHANNELS)
            .map(|i| (i as f32 * 0.001) - 4.0)
            .collect();
        let out = state.step(&input, &weight, &bias);
        let expected = reference::silu(&input);
        for (o, e) in out.iter().zip(&expected) {
            assert!((o - e).abs() < 1e-5, "{o} vs {e}");
        }
    }

    #[test]
    fn conv_tail_history_shifts_across_steps() {
        let taps = CONV_WIDTH - 1;
        let mut weight = vec![0.0f32; CONV_CHANNELS * CONV_WIDTH];
        // Tap `taps - 1` = the most-recently-shifted-in history slot -> one
        // step later, this picks up the *previous* timestep's input.
        weight[taps - 1] = 1.0;
        let bias = vec![0.0f32; CONV_CHANNELS];
        let mut state = ConvTailState::new();

        let step1 = vec![5.0f32; CONV_CHANNELS];
        let step2 = vec![0.0f32; CONV_CHANNELS];
        state.step(&step1, &weight, &bias); // history's newest slot now holds 5.0
        let out2 = state.step(&step2, &weight, &bias);
        let expected = reference::silu(&[5.0f32])[0];
        assert!(
            (out2[0] - expected).abs() < 1e-5,
            "{} vs {expected}",
            out2[0]
        );
    }

    #[test]
    fn qk_norm_l2_normalizes_each_head_without_learned_weights() {
        let q: Vec<f32> = (0..KEY_DIM).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let k: Vec<f32> = (0..KEY_DIM).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let (q_out, k_out) = qk_norm(&q, &k);
        for head in q_out
            .chunks_exact(KEY_HEAD_DIM)
            .chain(k_out.chunks_exact(KEY_HEAD_DIM))
        {
            let squared_norm = head.iter().map(|value| value * value).sum::<f32>();
            assert!((squared_norm - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn repeat_heads_matches_gguf_tiled_value_head_order() {
        let mut x = vec![0.0f32; KEY_HEADS * KEY_HEAD_DIM];
        for h in 0..KEY_HEADS {
            x[h * KEY_HEAD_DIM] = h as f32; // tag each head with its index
        }
        let repeated = repeat_heads(&x, KEY_HEAD_DIM);
        assert_eq!(repeated.len(), VALUE_HEADS * KEY_HEAD_DIM);
        let group = VALUE_HEADS / KEY_HEADS;
        for g in 0..group {
            for h in 0..KEY_HEADS {
                let dst = g * KEY_HEADS + h;
                assert_eq!(repeated[dst * KEY_HEAD_DIM], h as f32);
            }
        }
    }

    #[test]
    fn recurrent_step_zero_beta_leaves_state_at_zero() {
        let mut state = state();
        let q = vec![1.0f32; KEY_DIM];
        let k = vec![1.0f32; KEY_DIM];
        let v = vec![1.0f32; VALUE_DIM];
        let (a, mut b, a_neg_exp, dt_bias) = recurrent_parameters();
        b.fill(-30.0); // sigmoid(-large) ~= 0.

        let y = recurrent_step(
            &mut state,
            &q,
            &k,
            &v,
            GdnRecurrentParameters {
                alpha: &a,
                beta: &b,
                a_neg_exp: &a_neg_exp,
                dt_bias: &dt_bias,
            },
        );
        assert!(y.iter().all(|&v| v.abs() < 1e-4));
        assert!(state.recurrent.iter().all(|&v| v.abs() < 1e-4));
    }

    #[test]
    fn recurrent_step_full_write_from_zero_state_is_a_clean_outer_product_readout() {
        let mut state = state();
        // beta ~= 1; starting state is zero so pred = 0 and delta = v,
        // giving S = outer(k, v). Decay does not affect an all-zero state.
        let (a, mut b, a_neg_exp, dt_bias) = recurrent_parameters();
        b.fill(30.0);

        let mut q = vec![0.0f32; KEY_DIM];
        let mut k = vec![0.0f32; KEY_DIM];
        let mut v = vec![0.0f32; VALUE_DIM];
        // Single active head (head 0) with simple values so the expected
        // read-out has a closed form: y = (k . q) * v.
        q[0] = 2.0;
        k[0] = 3.0;
        v[0] = 5.0;
        v[1] = 7.0;

        let y = recurrent_step(
            &mut state,
            &q,
            &k,
            &v,
            GdnRecurrentParameters {
                alpha: &a,
                beta: &b,
                a_neg_exp: &a_neg_exp,
                dt_bias: &dt_bias,
            },
        );
        let dot_kq = 3.0 * 2.0 / (KEY_HEAD_DIM as f32).sqrt();
        assert!((y[0] - dot_kq * 5.0).abs() < 1e-2, "y[0]={}", y[0]);
        assert!((y[1] - dot_kq * 7.0).abs() < 1e-2, "y[1]={}", y[1]);
        // In GGUF tiled order, value-head 16 is the second copy of key-head
        // zero. It has an independent V slice, left at zero here.
        let head1_base = KEY_HEADS * VALUE_HEAD_DIM;
        assert!(
            (y[head1_base]).abs() < 1e-4,
            "y[head1_base]={}",
            y[head1_base]
        );
    }

    #[test]
    fn recurrent_step_decays_state_before_delta_prediction() {
        let mut state = state();
        state.recurrent[0] = 2.0;
        let mut q = vec![0.0; KEY_DIM];
        let mut k = vec![0.0; KEY_DIM];
        let v = vec![0.0; VALUE_DIM];
        q[0] = 1.0;
        k[0] = 1.0;
        let (a, mut b, a_neg_exp, dt_bias) = recurrent_parameters();
        b.fill(30.0);
        let decay = (a_neg_exp[0] * softplus(a[0] + dt_bias[0])).exp();
        let beta = sigmoid(b[0]);
        let decayed = 2.0 * decay;
        let expected_state = decayed + beta * (0.0 - decayed);

        let y = recurrent_step(
            &mut state,
            &q,
            &k,
            &v,
            GdnRecurrentParameters {
                alpha: &a,
                beta: &b,
                a_neg_exp: &a_neg_exp,
                dt_bias: &dt_bias,
            },
        );

        assert!((state.recurrent[0] - expected_state).abs() < 1e-6);
        assert!((y[0] - expected_state / (KEY_HEAD_DIM as f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn gdn_state_snapshot_restore_round_trips() {
        let mut state = state();
        let q = vec![1.0f32; KEY_DIM];
        let k = vec![1.0f32; KEY_DIM];
        let v = vec![2.0f32; VALUE_DIM];
        let (a, mut b, a_neg_exp, dt_bias) = recurrent_parameters();
        b.fill(10.0);
        recurrent_step(
            &mut state,
            &q,
            &k,
            &v,
            GdnRecurrentParameters {
                alpha: &a,
                beta: &b,
                a_neg_exp: &a_neg_exp,
                dt_bias: &dt_bias,
            },
        );

        let snapshot = state.snapshot().unwrap();
        recurrent_step(
            &mut state,
            &q,
            &k,
            &v,
            GdnRecurrentParameters {
                alpha: &a,
                beta: &b,
                a_neg_exp: &a_neg_exp,
                dt_bias: &dt_bias,
            },
        ); // mutate further
        assert_ne!(state.recurrent, snapshot.recurrent);

        state.restore(&snapshot);
        assert_eq!(state.recurrent, snapshot.recurrent);
    }

    #[test]
    fn gdn_state_reset_zeroes_recurrent_and_conv_history() {
        let mut state = state();
        let q = vec![1.0f32; KEY_DIM];
        let k = vec![1.0f32; KEY_DIM];
        let v = vec![2.0f32; VALUE_DIM];
        let (a, mut b, a_neg_exp, dt_bias) = recurrent_parameters();
        b.fill(10.0);
        recurrent_step(
            &mut state,
            &q,
            &k,
            &v,
            GdnRecurrentParameters {
                alpha: &a,
                beta: &b,
                a_neg_exp: &a_neg_exp,
                dt_bias: &dt_bias,
            },
        );
        state.conv_tail_mut().step(
            &vec![1.0f32; CONV_CHANNELS],
            &vec![0.0f32; CONV_CHANNELS * CONV_WIDTH],
            &vec![0.0f32; CONV_CHANNELS],
        );

        state.reset();
        assert!(state.recurrent.iter().all(|&v| v == 0.0));
        assert!(state.conv.history.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gated_norm_zero_gate_zeroes_output_regardless_of_y() {
        let y = vec![3.0f32; VALUE_DIM];
        let z = vec![0.0f32; VALUE_DIM]; // silu(0) = 0
        let weight = vec![1.0f32; VALUE_DIM];
        let out = gated_norm(&y, &z, &weight);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn out_projection_matches_dense_matvec() {
        let y = vec![1.0f32; VALUE_DIM];
        let weight = identity_weight(HIDDEN_SIZE, VALUE_DIM);
        let out = out_projection(&y, &weight);
        assert_eq!(out.len(), HIDDEN_SIZE);
        // Identity-shaped weight -> out[i] = y[i] for i < HIDDEN_SIZE.
        assert_eq!(out[0], 1.0);
    }
}
