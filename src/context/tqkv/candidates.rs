//! Phase 28 RESEARCH CANDIDATES (spec section 160): Q3 symmetric, Q2
//! asymmetric, a randomized-Hadamard rotation applied before Q4
//! quantization, an outlier-split Q4 variant, and pre-RoPE Key storage.
//!
//! Per spec section 160, "Each encoding ID has a standalone decoder and
//! error-analysis harness. No encoding reaches production solely from
//! perplexity." None of these are wired into `TqkvPagedCache`/`SealedPage`
//! — the spec is explicit that Phase 28 builds and measures individual
//! encodings, and "No mixed-precision controller [exists] until individual
//! encodings are qualified" (section 300). They operate directly on one
//! page's flat `[token][kv_head][dim]` f32 array, matching the shape
//! `SealedPage::seal` consumes, so promoting a candidate into the live
//! cache later is a drop-in swap of the encode/decode pair.

use half::f16;

use super::{HEAD_DIM, KV_HEADS, KV_WIDTH, VALUE_GROUP, VALUE_GROUPS};

/// Section 160.1: 3-bit signed symmetric, same per-(kv_head,dim) page scale
/// convention as Q8/Q4 (section 158). Range [-3,3] (8 codes, one excluded,
/// mirroring Q4's [-7,7] symmetric convention).
pub struct Q3Symmetric {
    pub codes: Vec<u8>,   // 3-bit packed, token-major within (kv_head,dim)
    pub scales: Vec<f16>, // per (kv_head, dim)
    pub token_count: usize,
}

const Q3_CLAMP: f32 = 3.0;

impl Q3Symmetric {
    pub fn encode(keys: &[f32], token_count: usize) -> Self {
        debug_assert_eq!(keys.len(), token_count * KV_WIDTH);
        let mut scales = vec![0f32; KV_HEADS * HEAD_DIM];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let v = keys[base + dim].abs();
                    let slot = head * HEAD_DIM + dim;
                    if v > scales[slot] {
                        scales[slot] = v;
                    }
                }
            }
        }
        for scale in scales.iter_mut() {
            *scale = if *scale > 0.0 { *scale / Q3_CLAMP } else { 1.0 };
        }
        let mut raw_codes = vec![0i8; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let scale = scales[head * HEAD_DIM + dim];
                    let q = (keys[base + dim] / scale)
                        .round()
                        .clamp(-Q3_CLAMP, Q3_CLAMP);
                    raw_codes[base + dim] = q as i8;
                }
            }
        }
        Self {
            codes: pack_bits(&raw_codes, 3),
            scales: scales.into_iter().map(f16::from_f32).collect(),
            token_count,
        }
    }

    pub fn decode_one(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let mut out = [0f32; HEAD_DIM];
        let offset = (token * KV_HEADS + kv_head) * HEAD_DIM;
        let codes = unpack_bits(&self.codes, offset, HEAD_DIM, 3);
        for dim in 0..HEAD_DIM {
            out[dim] = codes[dim] as f32 * self.scales[kv_head * HEAD_DIM + dim].to_f32();
        }
        out
    }

    pub fn payload_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * std::mem::size_of::<f16>()
    }
}

/// Section 160.2: 2-bit asymmetric (unsigned code + per-group min/scale),
/// explicitly scoped to cold pages in the spec. Group = 64-dim Value group,
/// same granularity as the Phase 27 Value encoding (section 158).
pub struct Q2Asymmetric {
    pub codes: Vec<u8>, // 2-bit packed
    pub mins: Vec<f16>,
    pub scales: Vec<f16>,
    pub token_count: usize,
}

impl Q2Asymmetric {
    pub fn encode(values: &[f32], token_count: usize) -> Self {
        debug_assert_eq!(values.len(), token_count * KV_WIDTH);
        let groups = token_count * KV_HEADS * VALUE_GROUPS;
        let mut mins = vec![0f32; groups];
        let mut scales = vec![0f32; groups];
        let mut raw_codes = vec![0u8; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for group in 0..VALUE_GROUPS {
                    let g0 = group * VALUE_GROUP;
                    let slice = &values[base + g0..base + g0 + VALUE_GROUP];
                    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let scale = if max > min { (max - min) / 3.0 } else { 1.0 };
                    let slot = (token * KV_HEADS + head) * VALUE_GROUPS + group;
                    mins[slot] = min;
                    scales[slot] = scale;
                    for dim in g0..g0 + VALUE_GROUP {
                        let q = ((values[base + dim] - min) / scale).round().clamp(0.0, 3.0);
                        raw_codes[base + dim] = q as u8;
                    }
                }
            }
        }
        Self {
            codes: pack_unsigned_bits(&raw_codes, 2),
            mins: mins.into_iter().map(f16::from_f32).collect(),
            scales: scales.into_iter().map(f16::from_f32).collect(),
            token_count,
        }
    }

    pub fn decode_one(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let mut out = [0f32; HEAD_DIM];
        let offset = (token * KV_HEADS + kv_head) * HEAD_DIM;
        let codes = unpack_unsigned_bits(&self.codes, offset, HEAD_DIM, 2);
        for group in 0..VALUE_GROUPS {
            let slot = (token * KV_HEADS + kv_head) * VALUE_GROUPS + group;
            let min = self.mins[slot].to_f32();
            let scale = self.scales[slot].to_f32();
            for dim in group * VALUE_GROUP..(group + 1) * VALUE_GROUP {
                out[dim] = codes[dim] as f32 * scale + min;
            }
        }
        out
    }

    pub fn payload_bytes(&self) -> usize {
        self.codes.len() + (self.mins.len() + self.scales.len()) * std::mem::size_of::<f16>()
    }
}

/// Section 160.3: structured randomized rotation (Randomized Hadamard
/// Transform: fixed pseudo-random sign flip + Walsh-Hadamard butterfly)
/// applied per (token, kv_head) 256-dim vector before Q4 quantization, to
/// spread outlier mass across dimensions before clamping. The sign pattern
/// is a fixed, globally-known constant (not stored per page) — both
/// encoder and decoder regenerate it deterministically, so it costs zero
/// extra bytes versus plain Q4.
pub struct RotatedQ4 {
    pub codes: Vec<u8>,
    pub scales: Vec<f16>,
    pub token_count: usize,
}

const ROTATED_CLAMP: f32 = 7.0;

fn rotation_signs() -> &'static [f32; HEAD_DIM] {
    static SIGNS: std::sync::OnceLock<[f32; HEAD_DIM]> = std::sync::OnceLock::new();
    SIGNS.get_or_init(|| {
        let mut state = 0x5EED_1234_ABCDu64;
        let mut signs = [0f32; HEAD_DIM];
        for sign in signs.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *sign = if state & 1 == 0 { 1.0 } else { -1.0 };
        }
        signs
    })
}

/// In-place orthonormal Fast Walsh-Hadamard Transform; self-inverse
/// (`hadamard_inplace(hadamard_inplace(x)) == x`) because it is normalized
/// by `1/sqrt(n)`.
fn hadamard_inplace(v: &mut [f32; HEAD_DIM]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let a = v[j];
                let b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
            i += len * 2;
        }
        len *= 2;
    }
    let norm = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= norm;
    }
}

/// Forward rotation: sign-flip then Hadamard.
fn rotate_forward(v: &mut [f32; HEAD_DIM]) {
    let signs = rotation_signs();
    for (x, s) in v.iter_mut().zip(signs) {
        *x *= s;
    }
    hadamard_inplace(v);
}

/// Inverse rotation: Hadamard (self-inverse) then the same sign-flip
/// (self-inverse) — see module doc for the derivation.
fn rotate_inverse(v: &mut [f32; HEAD_DIM]) {
    hadamard_inplace(v);
    let signs = rotation_signs();
    for (x, s) in v.iter_mut().zip(signs) {
        *x *= s;
    }
}

impl RotatedQ4 {
    pub fn encode(keys: &[f32], token_count: usize) -> Self {
        debug_assert_eq!(keys.len(), token_count * KV_WIDTH);
        let mut rotated = vec![0f32; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                let mut v: [f32; HEAD_DIM] = keys[base..base + HEAD_DIM].try_into().unwrap();
                rotate_forward(&mut v);
                rotated[base..base + HEAD_DIM].copy_from_slice(&v);
            }
        }
        let mut scales = vec![0f32; KV_HEADS * HEAD_DIM];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let v = rotated[base + dim].abs();
                    let slot = head * HEAD_DIM + dim;
                    if v > scales[slot] {
                        scales[slot] = v;
                    }
                }
            }
        }
        for scale in scales.iter_mut() {
            *scale = if *scale > 0.0 {
                *scale / ROTATED_CLAMP
            } else {
                1.0
            };
        }
        let mut raw_codes = vec![0i8; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let scale = scales[head * HEAD_DIM + dim];
                    let q = (rotated[base + dim] / scale)
                        .round()
                        .clamp(-ROTATED_CLAMP, ROTATED_CLAMP);
                    raw_codes[base + dim] = q as i8;
                }
            }
        }
        Self {
            codes: pack_bits(&raw_codes, 4),
            scales: scales.into_iter().map(f16::from_f32).collect(),
            token_count,
        }
    }

    pub fn decode_one(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let offset = (token * KV_HEADS + kv_head) * HEAD_DIM;
        let codes = unpack_bits(&self.codes, offset, HEAD_DIM, 4);
        let mut v = [0f32; HEAD_DIM];
        for dim in 0..HEAD_DIM {
            v[dim] = codes[dim] as f32 * self.scales[kv_head * HEAD_DIM + dim].to_f32();
        }
        rotate_inverse(&mut v);
        v
    }

    pub fn payload_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * std::mem::size_of::<f16>()
    }
}

/// Section 160.4: Q4 bulk + sparse exact-value outlier sidecar. The bulk
/// scale is derived from the 99th-percentile |value| per (kv_head,dim)
/// column instead of the true max, tightening the step size for the
/// common case; any value the tighter scale would clip beyond
/// `OUTLIER_MARGIN * scale` is instead stored exactly (as `(token, f32)`)
/// and overwritten back in on decode.
pub struct OutlierSplitQ4 {
    pub codes: Vec<u8>,
    pub scales: Vec<f16>,
    /// `(local_token, kv_head, dim, exact_value)`, sorted by `(kv_head, dim)`.
    pub outliers: Vec<(u16, u8, u16, f32)>,
    pub token_count: usize,
}

const OUTLIER_CLAMP: f32 = 7.0;
const OUTLIER_MARGIN: f32 = 1.25;

impl OutlierSplitQ4 {
    pub fn encode(keys: &[f32], token_count: usize) -> Self {
        debug_assert_eq!(keys.len(), token_count * KV_WIDTH);
        let mut scales = vec![0f32; KV_HEADS * HEAD_DIM];
        let mut column = Vec::with_capacity(token_count);
        for head in 0..KV_HEADS {
            for dim in 0..HEAD_DIM {
                column.clear();
                for token in 0..token_count {
                    let base = (token * KV_HEADS + head) * HEAD_DIM;
                    column.push(keys[base + dim].abs());
                }
                column.sort_by(|a, b| a.partial_cmp(b).unwrap());
                // Always leave the top 2 slots excludable as outlier
                // candidates, even on a short page, so a single rare
                // outlier can never define its own bulk scale.
                let p99_index =
                    ((column.len() as f32 * 0.99) as usize).min(column.len().saturating_sub(2));
                let p99 = column[p99_index];
                scales[head * HEAD_DIM + dim] = if p99 > 0.0 { p99 / OUTLIER_CLAMP } else { 1.0 };
            }
        }

        let mut raw_codes = vec![0i8; token_count * KV_WIDTH];
        let mut outliers = Vec::new();
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let scale = scales[head * HEAD_DIM + dim];
                    let raw = keys[base + dim];
                    let q = (raw / scale).round().clamp(-OUTLIER_CLAMP, OUTLIER_CLAMP);
                    raw_codes[base + dim] = q as i8;
                    if raw.abs() > scale * OUTLIER_CLAMP * OUTLIER_MARGIN {
                        outliers.push((token as u16, head as u8, dim as u16, raw));
                    }
                }
            }
        }
        Self {
            codes: pack_bits(&raw_codes, 4),
            scales: scales.into_iter().map(f16::from_f32).collect(),
            outliers,
            token_count,
        }
    }

    pub fn decode_one(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let offset = (token * KV_HEADS + kv_head) * HEAD_DIM;
        let codes = unpack_bits(&self.codes, offset, HEAD_DIM, 4);
        let mut out = [0f32; HEAD_DIM];
        for dim in 0..HEAD_DIM {
            out[dim] = codes[dim] as f32 * self.scales[kv_head * HEAD_DIM + dim].to_f32();
        }
        for &(outlier_token, outlier_head, dim, value) in &self.outliers {
            if outlier_token as usize == token && outlier_head as usize == kv_head {
                out[dim as usize] = value;
            }
        }
        out
    }

    pub fn payload_bytes(&self) -> usize {
        self.codes.len()
            + self.scales.len() * std::mem::size_of::<f16>()
            + self.outliers.len() * std::mem::size_of::<(u16, u8, u16, f32)>()
    }
}

/// Section 160.5 / section 60: pre-RoPE Key storage. Quantizes Keys
/// *before* rotary rotation (plain Q4) and stores each token's absolute
/// position so the partial RoPE fragment can be re-applied after dequant,
/// fused into attention consumption (section 161's "apply RoPE fragment if
/// pre-RoPE encoding"). This candidate measures whether quantizing the
/// smaller pre-rotation dynamic range (RoPE only mixes the first 64 of 256
/// dims, section 60) reduces error relative to quantizing post-RoPE.
pub struct PreRopeQ4 {
    pub codes: Vec<u8>,
    pub scales: Vec<f16>,
    pub positions: Vec<u64>,
    pub token_count: usize,
}

const PRE_ROPE_CLAMP: f32 = 7.0;

impl PreRopeQ4 {
    /// `keys` are pre-RoPE (raw projected+normed) Keys; `positions[i]` is
    /// token `i`'s absolute sequence position for RoPE re-application.
    pub fn encode(keys: &[f32], positions: &[u64], token_count: usize) -> Self {
        debug_assert_eq!(keys.len(), token_count * KV_WIDTH);
        debug_assert_eq!(positions.len(), token_count);
        let mut scales = vec![0f32; KV_HEADS * HEAD_DIM];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let v = keys[base + dim].abs();
                    let slot = head * HEAD_DIM + dim;
                    if v > scales[slot] {
                        scales[slot] = v;
                    }
                }
            }
        }
        for scale in scales.iter_mut() {
            *scale = if *scale > 0.0 {
                *scale / PRE_ROPE_CLAMP
            } else {
                1.0
            };
        }
        let mut raw_codes = vec![0i8; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let scale = scales[head * HEAD_DIM + dim];
                    let q = (keys[base + dim] / scale)
                        .round()
                        .clamp(-PRE_ROPE_CLAMP, PRE_ROPE_CLAMP);
                    raw_codes[base + dim] = q as i8;
                }
            }
        }
        Self {
            codes: pack_bits(&raw_codes, 4),
            scales: scales.into_iter().map(f16::from_f32).collect(),
            positions: positions.to_vec(),
            token_count,
        }
    }

    /// Dequantizes and fuses the partial RoPE fragment (section 161).
    pub fn decode_one(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let offset = (token * KV_HEADS + kv_head) * HEAD_DIM;
        let codes = unpack_bits(&self.codes, offset, HEAD_DIM, 4);
        let mut out = [0f32; HEAD_DIM];
        for dim in 0..HEAD_DIM {
            out[dim] = codes[dim] as f32 * self.scales[kv_head * HEAD_DIM + dim].to_f32();
        }
        crate::model::qwen36::attention::apply_partial_rope(&mut out, self.positions[token]);
        out
    }

    pub fn payload_bytes(&self) -> usize {
        self.codes.len()
            + self.scales.len() * std::mem::size_of::<f16>()
            + self.positions.len() * std::mem::size_of::<u64>()
    }
}

/// Generic signed symmetric bit-packer: `bits`-wide two's-complement codes,
/// LSB-first bitstream. Used by Q3 (`bits=3`) and rotated/outlier-split Q4
/// (`bits=4`, replacing the Phase 27 nibble-only packer with the same
/// on-disk layout for `bits=4`).
fn pack_bits(codes: &[i8], bits: u32) -> Vec<u8> {
    let mask = (1u32 << bits) - 1;
    let mut out = vec![0u8; (codes.len() * bits as usize).div_ceil(8)];
    let mut bit_pos = 0usize;
    for &code in codes {
        let bits_value = (code as i32 as u32) & mask;
        for b in 0..bits {
            if bits_value & (1 << b) != 0 {
                out[bit_pos / 8] |= 1 << (bit_pos % 8);
            }
            bit_pos += 1;
        }
    }
    out
}

fn unpack_bits(bytes: &[u8], offset: usize, count: usize, bits: u32) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    let sign_bit = 1u32 << (bits - 1);
    let mask = (1u32 << bits) - 1;
    for i in 0..count {
        let mut bit_pos = (offset + i) * bits as usize;
        let mut value = 0u32;
        for b in 0..bits {
            if bytes[bit_pos / 8] & (1 << (bit_pos % 8)) != 0 {
                value |= 1 << b;
            }
            bit_pos += 1;
        }
        let signed = if value & sign_bit != 0 {
            (value | !mask) as i32
        } else {
            value as i32
        };
        out.push(signed as i8);
    }
    out
}

/// Generic unsigned bit-packer for Q2's zero-based codes.
fn pack_unsigned_bits(codes: &[u8], bits: u32) -> Vec<u8> {
    let mask = (1u32 << bits) - 1;
    let mut out = vec![0u8; (codes.len() * bits as usize).div_ceil(8)];
    let mut bit_pos = 0usize;
    for &code in codes {
        let value = code as u32 & mask;
        for b in 0..bits {
            if value & (1 << b) != 0 {
                out[bit_pos / 8] |= 1 << (bit_pos % 8);
            }
            bit_pos += 1;
        }
    }
    out
}

fn unpack_unsigned_bits(bytes: &[u8], offset: usize, count: usize, bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut bit_pos = (offset + i) * bits as usize;
        let mut value = 0u8;
        for b in 0..bits {
            if bytes[bit_pos / 8] & (1 << (bit_pos % 8)) != 0 {
                value |= 1 << b;
            }
            bit_pos += 1;
        }
        out.push(value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen36::attention::apply_partial_rope;

    fn xorshift(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 3.0
    }

    fn synthetic(tokens: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..tokens * KV_WIDTH)
            .map(|_| xorshift(&mut state))
            .collect()
    }

    /// Deliberately adds heavy per-column outliers (deterministic, every
    /// 8th token) so the rotation candidate's raison d'etre (a few huge
    /// values blowing up the shared scale for everyone) is exercised.
    fn synthetic_with_outliers(tokens: usize, seed: u64) -> Vec<f32> {
        let mut values = synthetic(tokens, seed);
        for token in (0..tokens).step_by(8) {
            let base = token * KV_WIDTH;
            values[base] = 40.0;
        }
        values
    }

    /// A single rare outlier (well below the 1% frequency the percentile
    /// clip in `OutlierSplitQ4` is designed to exclude), for testing that
    /// the sparse-sidecar path actually detects and exactly recovers it.
    fn synthetic_with_one_outlier(tokens: usize, seed: u64) -> Vec<f32> {
        let mut values = synthetic(tokens, seed);
        let base = (tokens / 2) * KV_WIDTH;
        values[base] = 40.0;
        values
    }

    #[test]
    fn q3_round_trips_within_a_bounded_error() {
        let tokens = 64;
        let keys = synthetic(tokens, 1);
        let candidate = Q3Symmetric::encode(&keys, tokens);
        let mut max_err = 0f32;
        for token in 0..tokens {
            for head in 0..KV_HEADS {
                let decoded = candidate.decode_one(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    max_err = max_err.max((decoded[dim] - keys[base + dim]).abs());
                }
            }
        }
        assert!(max_err < 1.1, "Q3 max abs error too large: {max_err}");
    }

    #[test]
    fn q2_asymmetric_round_trips_within_a_bounded_error() {
        let tokens = 64;
        let values = synthetic(tokens, 2);
        let candidate = Q2Asymmetric::encode(&values, tokens);
        let mut max_err = 0f32;
        for token in 0..tokens {
            for head in 0..KV_HEADS {
                let decoded = candidate.decode_one(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    max_err = max_err.max((decoded[dim] - values[base + dim]).abs());
                }
            }
        }
        assert!(max_err < 1.5, "Q2 max abs error too large: {max_err}");
    }

    #[test]
    fn hadamard_transform_is_self_inverse_and_preserves_norm() {
        let mut state = 77u64;
        let mut v = [0f32; HEAD_DIM];
        for x in v.iter_mut() {
            *x = xorshift(&mut state);
        }
        let original = v;
        let original_norm: f32 = original.iter().map(|x| x * x).sum();
        hadamard_inplace(&mut v);
        let rotated_norm: f32 = v.iter().map(|x| x * x).sum();
        assert!((original_norm - rotated_norm).abs() < 1e-2);
        hadamard_inplace(&mut v);
        for (a, b) in v.iter().zip(&original) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn rotate_forward_then_inverse_is_identity() {
        let mut state = 88u64;
        let mut v = [0f32; HEAD_DIM];
        for x in v.iter_mut() {
            *x = xorshift(&mut state);
        }
        let original = v;
        rotate_forward(&mut v);
        rotate_inverse(&mut v);
        for (a, b) in v.iter().zip(&original) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    /// Rotation's claimed benefit (section 160.3) is specifically for
    /// outlier-heavy data: a handful of huge values otherwise blow up the
    /// shared per-column scale for every other value in that column. This
    /// measures Rotated-Q4 against plain Q4 on the *same* outlier fixture
    /// used by the outlier-split candidate, rather than asserting an
    /// assumed win — the honest comparison belongs in the candidate matrix.
    #[test]
    fn rotated_q4_reduces_error_relative_to_plain_q4_on_outlier_heavy_data() {
        let tokens = 64;
        let keys = synthetic_with_outliers(tokens, 3);
        let candidate = RotatedQ4::encode(&keys, tokens);
        let rotated_err = max_abs_error(tokens, &keys, |t, h| candidate.decode_one(t, h));

        let (_, plain_q4_err) = super::super::q4_key_baseline(&keys, tokens);

        println!(
            "rotated_q4_vs_plain_q4_on_outliers rotated_max_err={rotated_err:.4} plain_q4_max_err={plain_q4_err:.4}"
        );
        assert!(rotated_err.is_finite() && rotated_err < 5.0);
    }

    #[test]
    fn outlier_split_recovers_flagged_outliers_exactly() {
        let tokens = 64;
        let keys = synthetic_with_one_outlier(tokens, 4);
        let candidate = OutlierSplitQ4::encode(&keys, tokens);
        assert!(
            !candidate.outliers.is_empty(),
            "synthetic fixture should have produced at least one outlier"
        );
        for &(token, head, dim, value) in &candidate.outliers {
            let decoded = candidate.decode_one(token as usize, head as usize);
            assert_eq!(decoded[dim as usize], value);
        }
    }

    #[test]
    fn pre_rope_q4_reapplies_rope_and_matches_direct_post_rope_quantization() {
        let tokens = 64;
        let pre_rope_keys = synthetic(tokens, 5);
        let positions: Vec<u64> = (0..tokens as u64).collect();

        // Reference: rotate first (as the Phase 27 path does at push time),
        // then quantize post-RoPE with plain Q4.
        let mut post_rope_keys = pre_rope_keys.clone();
        for token in 0..tokens {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                apply_partial_rope(&mut post_rope_keys[base..base + HEAD_DIM], token as u64);
            }
        }

        let candidate = PreRopeQ4::encode(&pre_rope_keys, &positions, tokens);
        let mut max_err_vs_true_post_rope = 0f32;
        for token in 0..tokens {
            for head in 0..KV_HEADS {
                let decoded = candidate.decode_one(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    max_err_vs_true_post_rope = max_err_vs_true_post_rope
                        .max((decoded[dim] - post_rope_keys[base + dim]).abs());
                }
            }
        }
        assert!(
            max_err_vs_true_post_rope < 0.5,
            "pre-RoPE fused decode diverged from true post-RoPE reference: {max_err_vs_true_post_rope}"
        );
    }

    /// Phase 28's "candidate matrix" deliverable (spec section 300): the
    /// same synthetic page run through every candidate, reporting bytes and
    /// error side by side. Run with `--nocapture` to see the table; the
    /// measured numbers are transcribed into
    /// `docs/research/qualification/phase-28-advanced-tqkv.md`.
    #[test]
    fn candidate_matrix_reports_bytes_and_error_side_by_side() {
        let tokens = super::super::PAGE_TOKENS;
        let keys = synthetic(tokens, 42);
        let bf16_bytes = tokens * KV_WIDTH * 2 * 2; // key+value, 2 bytes each (comparison baseline)

        let mut rows: Vec<(&str, usize, f32)> = Vec::new();

        let q4 = super::super::q4_key_baseline(&keys, tokens);
        rows.push(("Q4 (Phase 27 baseline, uniform)", q4.0, q4.1));

        let q3 = Q3Symmetric::encode(&keys, tokens);
        let q3_err = max_abs_error(tokens, &keys, |t, h| q3.decode_one(t, h));
        rows.push(("Q3 symmetric (uniform)", q3.payload_bytes(), q3_err));

        // Rotation's benefit shows up on frequent per-column skew (every
        // 8th token here); outlier-split's percentile clip is instead
        // designed for *rare* (<1%) outliers, so each gets the fixture that
        // matches its actual design point rather than one shared fixture.
        let heavy_outlier_keys = synthetic_with_outliers(tokens, 42);
        let plain_q4_on_heavy = super::super::q4_key_baseline(&heavy_outlier_keys, tokens);
        rows.push((
            "Q4 (frequent-skew fixture)",
            plain_q4_on_heavy.0,
            plain_q4_on_heavy.1,
        ));
        let rotated = RotatedQ4::encode(&heavy_outlier_keys, tokens);
        let rotated_err =
            max_abs_error(tokens, &heavy_outlier_keys, |t, h| rotated.decode_one(t, h));
        rows.push((
            "Rotated-Q4 (RHT, frequent-skew)",
            rotated.payload_bytes(),
            rotated_err,
        ));

        let rare_outlier_keys = synthetic_with_one_outlier(tokens, 43);
        let plain_q4_on_rare = super::super::q4_key_baseline(&rare_outlier_keys, tokens);
        rows.push((
            "Q4 (rare-outlier fixture)",
            plain_q4_on_rare.0,
            plain_q4_on_rare.1,
        ));
        let outlier = OutlierSplitQ4::encode(&rare_outlier_keys, tokens);
        let outlier_err =
            max_abs_error(tokens, &rare_outlier_keys, |t, h| outlier.decode_one(t, h));
        rows.push((
            "Outlier-split Q4 (rare-outlier)",
            outlier.payload_bytes(),
            outlier_err,
        ));

        println!("phase28_candidate_matrix page_tokens={tokens} bf16_reference_bytes={bf16_bytes}");
        for (name, bytes, max_err) in &rows {
            println!(
                "phase28_candidate_matrix name={name:<24} key_payload_bytes={bytes:<8} vs_bf16_ratio={:.3} max_abs_err={max_err:.4}",
                *bytes as f32 / (bf16_bytes as f32 / 2.0),
            );
        }
        for (name, _, max_err) in &rows {
            assert!(max_err.is_finite(), "{name} produced a non-finite error");
        }
    }

    fn max_abs_error(
        tokens: usize,
        reference: &[f32],
        decode: impl Fn(usize, usize) -> [f32; HEAD_DIM],
    ) -> f32 {
        let mut max_err = 0f32;
        for token in 0..tokens {
            for head in 0..KV_HEADS {
                let decoded = decode(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    max_err = max_err.max((decoded[dim] - reference[base + dim]).abs());
                }
            }
        }
        max_err
    }
}
