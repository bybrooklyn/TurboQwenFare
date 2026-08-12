//! Gated DeltaNet (spec §284, phase 12): the recurrent layer used by 30 of
//! Qwen3.6's 40 layers (spec §8/§11). Implemented in the exact stage order
//! spec §284 names:
//!
//! 1. four separate projections (`project`);
//! 2. causal depthwise conv "tail" over the concatenated q/k/v (`ConvTailState`);
//! 3. per-head q/k RMS normalization (`qk_norm`);
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
//! **Formula caveat**: the exact per-head decay/write-gate parameterization
//! (`a`/`b` -> `decay`/`beta` below) and gated-norm composition are written
//! to match the published Gated DeltaNet delta-rule formulation (state
//! decay + rank-1 delta write, read out by the query), not yet checked
//! bit-for-bit against the real Qwen3.6 checkpoint weights — that
//! requires the actual weight tensors and a qualification test (spec
//! §115 invariant #8), which is later phase-15 "end-to-end decode" work.
//! What's locked by spec §284 today is the *stage order* and the
//! per-layer state object's reset/snapshot contract; both are implemented
//! exactly.
//!
//! Projections here are plain dense f32 matmuls, not yet the Q4_K
//! `backend::metal::kernels`/`backend::reference` matmuls phase 11 built —
//! wiring real per-tensor quantized weights through those kernels is a
//! weight-loading integration task for a later phase. This keeps the
//! recurrence math (this phase's actual subject) decoupled from
//! quantization concerns and the unit tests below fast and deterministic.

use crate::backend::reference;
use crate::model::qwen36::geometry::Qwen36Geometry;

const KEY_HEADS: usize = Qwen36Geometry::GDN_KEY_HEADS; // 16
const VALUE_HEADS: usize = Qwen36Geometry::GDN_VALUE_HEADS; // 32
const KEY_HEAD_DIM: usize = Qwen36Geometry::GDN_KEY_HEAD_DIM; // 128
const VALUE_HEAD_DIM: usize = Qwen36Geometry::GDN_VALUE_HEAD_DIM; // 128
const KEY_DIM: usize = Qwen36Geometry::GDN_KEY_DIM; // 2048
const VALUE_DIM: usize = Qwen36Geometry::GDN_VALUE_DIM; // 4096
const HIDDEN_SIZE: usize = Qwen36Geometry::HIDDEN_SIZE;
const CONV_WIDTH: usize = Qwen36Geometry::GDN_CONV_WIDTH; // 4
const CONV_CHANNELS: usize = Qwen36Geometry::GDN_CONV_CHANNELS; // 8192 = KEY_DIM + HIDDEN_SIZE + VALUE_DIM

/// Per-head decay (`a`) and write-strength (`b`) gate logits are one scalar
/// per K/Q head (repeated to the 32 V heads the same way Q/K themselves are,
/// spec §11), not one per value dim — so `gate_proj`'s output packs
/// `KEY_HEADS` (a) + `KEY_HEADS` (b) + `VALUE_DIM` (z, the elementwise
/// output gate) columns.
pub const GATE_DIM: usize = KEY_HEADS + KEY_HEADS + VALUE_DIM;

/// Output of stage 1 (`project`): the four independently-computed
/// projections, gate logits already split into their (a, b, z) roles.
#[derive(Debug, Clone)]
pub struct GdnProjected {
    pub q: Vec<f32>, // [KEY_DIM]
    pub k: Vec<f32>, // [KEY_DIM]
    pub v: Vec<f32>, // [VALUE_DIM]
    pub a: Vec<f32>, // [KEY_HEADS] decay-gate logits
    pub b: Vec<f32>, // [KEY_HEADS] write-gate (beta) logits
    pub z: Vec<f32>, // [VALUE_DIM] output-gate logits
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

/// Stage 1 (spec §284 item 1): four separate projections from the layer's
/// hidden-state input. `q_weight`/`k_weight` are `[KEY_DIM, HIDDEN_SIZE]`,
/// `v_weight` is `[VALUE_DIM, HIDDEN_SIZE]`, `gate_weight` is
/// `[GATE_DIM, HIDDEN_SIZE]`, all row-major.
pub fn project(
    hidden: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    gate_weight: &[f32],
) -> GdnProjected {
    assert_eq!(hidden.len(), HIDDEN_SIZE);
    let q = dense_matvec(q_weight, hidden, KEY_DIM, HIDDEN_SIZE);
    let k = dense_matvec(k_weight, hidden, KEY_DIM, HIDDEN_SIZE);
    let v = dense_matvec(v_weight, hidden, VALUE_DIM, HIDDEN_SIZE);
    let gate = dense_matvec(gate_weight, hidden, GATE_DIM, HIDDEN_SIZE);
    let a = gate[0..KEY_HEADS].to_vec();
    let b = gate[KEY_HEADS..2 * KEY_HEADS].to_vec();
    let z = gate[2 * KEY_HEADS..].to_vec();
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
}

impl Default for ConvTailState {
    fn default() -> Self {
        Self::new()
    }
}

/// Stage 3 (spec §284 item 3): per-head RMS normalization of q and k
/// (reshaped into `KEY_HEADS` heads of `KEY_HEAD_DIM`), reusing the same
/// `backend::reference::rmsnorm` phase 11 built and tested for the Metal
/// kernel's CPU oracle.
pub fn qk_norm(q: &[f32], k: &[f32], q_weight: &[f32], k_weight: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), KEY_DIM);
    assert_eq!(k.len(), KEY_DIM);
    let eps = 1e-6;
    let q_out = reference::rmsnorm(q, q_weight, KEY_HEADS, KEY_HEAD_DIM, eps);
    let k_out = reference::rmsnorm(k, k_weight, KEY_HEADS, KEY_HEAD_DIM, eps);
    (q_out, k_out)
}

/// Broadcasts `KEY_HEADS` per-head vectors of `head_dim` up to
/// `VALUE_HEADS` heads via `repeat_interleave` (head `i` of `KEY_HEADS`
/// becomes heads `2i` and `2i+1` of `VALUE_HEADS`) — spec §11: "repeats Q/K
/// from 16 key heads to 32 value heads", the standard grouped-query
/// `repeat_kv` convention.
fn repeat_heads(x: &[f32], head_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), KEY_HEADS * head_dim);
    let group = VALUE_HEADS / KEY_HEADS;
    let mut out = vec![0.0f32; VALUE_HEADS * head_dim];
    for h in 0..KEY_HEADS {
        let src = &x[h * head_dim..(h + 1) * head_dim];
        for g in 0..group {
            let dst_head = h * group + g;
            out[dst_head * head_dim..(dst_head + 1) * head_dim].copy_from_slice(src);
        }
    }
    out
}

/// Per-layer Gated DeltaNet recurrent state (spec §284: "Store one
/// recurrent-state object per GDN layer with exact reset/snapshot APIs.").
/// `recurrent` is `[VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM]` f32 (~2 MiB,
/// spec §11), constant-size with context length.
#[derive(Debug, Clone)]
pub struct GdnState {
    recurrent: Vec<f32>,
    conv: ConvTailState,
}

impl GdnState {
    pub fn new() -> Self {
        Self {
            recurrent: vec![0.0; VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM],
            conv: ConvTailState::new(),
        }
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
    pub fn snapshot(&self) -> GdnState {
        self.clone()
    }

    pub fn restore(&mut self, snapshot: &GdnState) {
        self.recurrent.copy_from_slice(&snapshot.recurrent);
        self.conv = snapshot.conv.clone();
    }
}

impl Default for GdnState {
    fn default() -> Self {
        Self::new()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Stage 4 (spec §284 item 4): the FP32 delta-rule recurrent update for one
/// decoded timestep. `q`, `k` are `[KEY_DIM]` (pre-`repeat_heads`, already
/// through `qk_norm`); `v` is `[VALUE_DIM]`; `a`, `b` are the `[KEY_HEADS]`
/// decay/write-gate logits from `project`. Returns `y`, the `[VALUE_DIM]`
/// recurrent read-out, and mutates `state.recurrent` in place.
///
/// Per value head `h` with state matrix `S_h` (`[KEY_HEAD_DIM,
/// VALUE_HEAD_DIM]`):
/// ```text
/// decay_h = sigmoid(a_h)                     // state retention in (0,1)
/// beta_h  = sigmoid(b_h)                     // write strength in (0,1)
/// pred_h  = S_h^T @ k_h                      // value the current state predicts for k_h
/// delta_h = beta_h * (v_h - pred_h)          // delta rule's correction term
/// S_h     = decay_h * S_h + outer(k_h, delta_h)
/// y_h     = S_h^T @ q_h                      // read out with the query
/// ```
pub fn recurrent_step(
    state: &mut GdnState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    assert_eq!(q.len(), KEY_DIM);
    assert_eq!(k.len(), KEY_DIM);
    assert_eq!(v.len(), VALUE_DIM);
    assert_eq!(a.len(), KEY_HEADS);
    assert_eq!(b.len(), KEY_HEADS);

    let q_full = repeat_heads(q, KEY_HEAD_DIM);
    let k_full = repeat_heads(k, KEY_HEAD_DIM);
    let group = VALUE_HEADS / KEY_HEADS;

    let mut y = vec![0.0f32; VALUE_DIM];
    for h in 0..VALUE_HEADS {
        let decay = sigmoid(a[h / group]);
        let beta = sigmoid(b[h / group]);

        let q_h = &q_full[h * KEY_HEAD_DIM..(h + 1) * KEY_HEAD_DIM];
        let k_h = &k_full[h * KEY_HEAD_DIM..(h + 1) * KEY_HEAD_DIM];
        let v_h = &v[h * VALUE_HEAD_DIM..(h + 1) * VALUE_HEAD_DIM];

        let s_base = h * KEY_HEAD_DIM * VALUE_HEAD_DIM;
        let s_h = &mut state.recurrent[s_base..s_base + KEY_HEAD_DIM * VALUE_HEAD_DIM];

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

        // S_h = decay * S_h + outer(k_h, delta)
        for i in 0..KEY_HEAD_DIM {
            let k_i = k_h[i];
            let row = &mut s_h[i * VALUE_HEAD_DIM..(i + 1) * VALUE_HEAD_DIM];
            for j in 0..VALUE_HEAD_DIM {
                row[j] = decay * row[j] + k_i * delta[j];
            }
        }

        // y_h = S_h^T @ q_h
        let y_h = &mut y[h * VALUE_HEAD_DIM..(h + 1) * VALUE_HEAD_DIM];
        for i in 0..KEY_HEAD_DIM {
            let q_i = q_h[i];
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

/// Stage 5 (spec §284 item 5): gated norm — RMSNorm the recurrent read-out,
/// then gate it elementwise with `SiLU(z)` (the standard Mamba2/Gated
/// DeltaNet "gated RMSNorm": normalize first, then apply the output gate).
pub fn gated_norm(y: &[f32], z: &[f32], weight: &[f32]) -> Vec<f32> {
    assert_eq!(y.len(), VALUE_DIM);
    assert_eq!(z.len(), VALUE_DIM);
    let normed = reference::rmsnorm(y, weight, 1, VALUE_DIM, 1e-6);
    let gate = reference::silu(z);
    normed.iter().zip(&gate).map(|(n, g)| n * g).collect()
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

    #[test]
    fn project_splits_gate_into_a_b_z_in_order() {
        let hidden = vec![1.0f32; HIDDEN_SIZE];
        // Identity-shaped gate weight: gate[i] picks out hidden[i] for i < HIDDEN_SIZE.
        let gate_weight = identity_weight(GATE_DIM, HIDDEN_SIZE);
        let q_weight = vec![0.0f32; KEY_DIM * HIDDEN_SIZE];
        let k_weight = vec![0.0f32; KEY_DIM * HIDDEN_SIZE];
        let v_weight = vec![0.0f32; VALUE_DIM * HIDDEN_SIZE];

        let projected = project(&hidden, &q_weight, &k_weight, &v_weight, &gate_weight);
        assert_eq!(projected.a.len(), KEY_HEADS);
        assert_eq!(projected.b.len(), KEY_HEADS);
        assert_eq!(projected.z.len(), VALUE_DIM);
        // Identity weight -> gate[i] = hidden[i] = 1.0 for i < HIDDEN_SIZE (all of GATE_DIM here
        // since GATE_DIM < HIDDEN_SIZE is false in the real geometry; check the first entries only).
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
    fn qk_norm_matches_reference_rmsnorm_per_head() {
        let q: Vec<f32> = (0..KEY_DIM).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let k: Vec<f32> = (0..KEY_DIM).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let q_weight = vec![1.0f32; KEY_HEAD_DIM];
        let k_weight = vec![1.0f32; KEY_HEAD_DIM];

        let (q_out, k_out) = qk_norm(&q, &k, &q_weight, &k_weight);
        let expected_q = reference::rmsnorm(&q, &q_weight, KEY_HEADS, KEY_HEAD_DIM, 1e-6);
        let expected_k = reference::rmsnorm(&k, &k_weight, KEY_HEADS, KEY_HEAD_DIM, 1e-6);
        assert_eq!(q_out, expected_q);
        assert_eq!(k_out, expected_k);
    }

    #[test]
    fn repeat_heads_duplicates_each_head_consecutively() {
        let mut x = vec![0.0f32; KEY_HEADS * KEY_HEAD_DIM];
        for h in 0..KEY_HEADS {
            x[h * KEY_HEAD_DIM] = h as f32; // tag each head with its index
        }
        let repeated = repeat_heads(&x, KEY_HEAD_DIM);
        assert_eq!(repeated.len(), VALUE_HEADS * KEY_HEAD_DIM);
        let group = VALUE_HEADS / KEY_HEADS;
        for h in 0..KEY_HEADS {
            for g in 0..group {
                let dst = h * group + g;
                assert_eq!(repeated[dst * KEY_HEAD_DIM], h as f32);
            }
        }
    }

    #[test]
    fn recurrent_step_zero_decay_zero_beta_leaves_state_at_zero() {
        let mut state = GdnState::new();
        let q = vec![1.0f32; KEY_DIM];
        let k = vec![1.0f32; KEY_DIM];
        let v = vec![1.0f32; VALUE_DIM];
        // sigmoid(-large) ~= 0 for both decay and beta.
        let a = vec![-30.0f32; KEY_HEADS];
        let b = vec![-30.0f32; KEY_HEADS];

        let y = recurrent_step(&mut state, &q, &k, &v, &a, &b);
        assert!(y.iter().all(|&v| v.abs() < 1e-4));
        assert!(state.recurrent.iter().all(|&v| v.abs() < 1e-4));
    }

    #[test]
    fn recurrent_step_full_write_from_zero_state_is_a_clean_outer_product_readout() {
        let mut state = GdnState::new();
        // decay, beta ~= 1 (sigmoid(large) ~= 1); starting state is zero so
        // pred = 0 and delta = v exactly, giving S = outer(k, v).
        let a = vec![30.0f32; KEY_HEADS];
        let b = vec![30.0f32; KEY_HEADS];

        let mut q = vec![0.0f32; KEY_DIM];
        let mut k = vec![0.0f32; KEY_DIM];
        let mut v = vec![0.0f32; VALUE_DIM];
        // Single active head (head 0) with simple values so the expected
        // read-out has a closed form: y = (k . q) * v.
        q[0] = 2.0;
        k[0] = 3.0;
        v[0] = 5.0;
        v[1] = 7.0;

        let y = recurrent_step(&mut state, &q, &k, &v, &a, &b);
        let dot_kq = 3.0 * 2.0; // only index 0 nonzero in the KEY_HEAD_DIM=128 head slice
        assert!((y[0] - dot_kq * 5.0).abs() < 1e-2, "y[0]={}", y[0]);
        assert!((y[1] - dot_kq * 7.0).abs() < 1e-2, "y[1]={}", y[1]);
        // Value-head 1 (duplicated from key-head 0 via repeat_heads, group
        // = 2) shares q/k with head 0 but has its own independent V slice
        // (v[128..256], left at zero here) -- repeat_heads only broadcasts
        // q/k, never v. So head 1's read-out must be zero, not head 0's
        // value, confirming v is not accidentally shared across the
        // repeat-group.
        let head1_base = VALUE_HEAD_DIM;
        assert!(
            (y[head1_base]).abs() < 1e-4,
            "y[head1_base]={}",
            y[head1_base]
        );
    }

    #[test]
    fn gdn_state_snapshot_restore_round_trips() {
        let mut state = GdnState::new();
        let q = vec![1.0f32; KEY_DIM];
        let k = vec![1.0f32; KEY_DIM];
        let v = vec![2.0f32; VALUE_DIM];
        let a = vec![10.0f32; KEY_HEADS];
        let b = vec![10.0f32; KEY_HEADS];
        recurrent_step(&mut state, &q, &k, &v, &a, &b);

        let snapshot = state.snapshot();
        recurrent_step(&mut state, &q, &k, &v, &a, &b); // mutate further
        assert_ne!(state.recurrent, snapshot.recurrent);

        state.restore(&snapshot);
        assert_eq!(state.recurrent, snapshot.recurrent);
    }

    #[test]
    fn gdn_state_reset_zeroes_recurrent_and_conv_history() {
        let mut state = GdnState::new();
        let q = vec![1.0f32; KEY_DIM];
        let k = vec![1.0f32; KEY_DIM];
        let v = vec![2.0f32; VALUE_DIM];
        let a = vec![10.0f32; KEY_HEADS];
        let b = vec![10.0f32; KEY_HEADS];
        recurrent_step(&mut state, &q, &k, &v, &a, &b);
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
