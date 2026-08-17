//! Reference dequantizers for the GGML quant types the pinned canonical
//! checkpoints actually use (spec §279 "source Q4 decoder"): Q4_0 (MTP
//! checkpoint), Q4_K/Q6_K (language weights, "Q4_K_M"), Q8_0 (vision
//! projector). Faithful reimplementations of the public ggml/llama.cpp
//! block layouts — this is interop with an external wire format (same
//! posture as `format/gguf`), not copied proprietary code.
//!
//! Every function takes exactly one already-bounds-checked block's raw
//! bytes (`GgmlType::block_bytes()` long, sliced by the caller) and
//! returns its decoded values; callers own bounds/length validation (spec
//! §115 invariant #3) before calling in.

use crate::format::quant::GgmlType;

/// IEEE 754 binary16 -> f32 via direct bit manipulation. The independent
/// validation decoder (`format::quant::validate`) intentionally uses a
/// different (floating-point-arithmetic-based) method so a bug in one
/// conversion doesn't silently pass the other's cross-check.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    let f32_bits: u32 = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal half -> normalized f32: shift the fraction left
            // until its implicit leading bit lands, adjusting the f32
            // exponent to match.
            let mut e: i32 = -1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let exp32 = (127 - 15 - e) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFFu32 << 23) | (frac << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

pub const Q4_0_BLOCK_ELEMENTS: usize = 32;
pub const Q4_K_BLOCK_ELEMENTS: usize = 256;
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;

/// `block_q4_0`: `ggml_half d` (2 bytes) + 16 bytes of packed nibbles.
/// Symmetric affine: `value = (nibble - 8) * d`.
pub fn dequantize_q4_0(block: &[u8]) -> [f32; Q4_0_BLOCK_ELEMENTS] {
    debug_assert_eq!(block.len(), GgmlType::Q4_0.block_bytes() as usize);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    let mut out = [0f32; Q4_0_BLOCK_ELEMENTS];
    for j in 0..16 {
        let byte = qs[j];
        let x0 = (byte & 0x0F) as i32 - 8;
        let x1 = (byte >> 4) as i32 - 8;
        out[j] = x0 as f32 * d;
        out[j + 16] = x1 as f32 * d;
    }
    out
}

/// `block_q8_0`: `ggml_half d` (2 bytes) + 32 signed int8 values.
pub fn dequantize_q8_0(block: &[u8]) -> [f32; Q8_0_BLOCK_ELEMENTS] {
    debug_assert_eq!(block.len(), GgmlType::Q8_0.block_bytes() as usize);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..34];
    let mut out = [0f32; Q8_0_BLOCK_ELEMENTS];
    for (j, out_slot) in out.iter_mut().enumerate() {
        *out_slot = (qs[j] as i8) as f32 * d;
    }
    out
}

/// Recovers the 6-bit quantized scale/min pair for sub-block `j` (0..8)
/// from the 12-byte packed `scales` array, per ggml's `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `block_q4_K`: `ggml_half d`, `ggml_half dmin` (super-block scale/min
/// scales), 12 bytes of packed 6-bit per-sub-block scale/min pairs (8
/// sub-blocks of 32 elements each), 128 bytes of packed nibbles.
/// `value = d * scale[sub] * nibble - dmin * min[sub]`.
pub fn dequantize_q4_k(block: &[u8]) -> [f32; Q4_K_BLOCK_ELEMENTS] {
    debug_assert_eq!(block.len(), GgmlType::Q4K.block_bytes() as usize);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut out = [0f32; Q4_K_BLOCK_ELEMENTS];
    let mut is = 0usize;
    let mut out_idx = 0usize;
    let mut q_idx = 0usize;
    for _ in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let mm1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let mm2 = dmin * m2 as f32;

        for l in 0..32 {
            out[out_idx + l] = d1 * (qs[q_idx + l] & 0x0F) as f32 - mm1;
        }
        for l in 0..32 {
            out[out_idx + 32 + l] = d2 * (qs[q_idx + l] >> 4) as f32 - mm2;
        }
        out_idx += 64;
        q_idx += 32;
        is += 2;
    }
    out
}

/// `block_q6_K`: 128 bytes of low nibbles, 64 bytes of packed high two-bit
/// values, 16 signed per-16-value scales, then an f16 super-block scale.
/// This layout is used by the canonical Q4_K_M LM head and must not be
/// treated as Q4_K merely because both have 256-value blocks.
pub fn dequantize_q6_k(block: &[u8]) -> [f32; Q6_K_BLOCK_ELEMENTS] {
    debug_assert_eq!(block.len(), GgmlType::Q6K.block_bytes() as usize);
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut out = [0.0; Q6_K_BLOCK_ELEMENTS];

    // Matches ggml's `dequantize_row_q6_K`: two 128-value halves, each
    // arranged as four interleaved 32-value groups.
    for half in 0..2 {
        let ql = &ql[half * 64..(half + 1) * 64];
        let qh = &qh[half * 32..(half + 1) * 32];
        let scales = &scales[half * 8..(half + 1) * 8];
        let base = half * 128;
        for index in 0..32 {
            let scale_group = index / 16;
            let q1 = ((ql[index] & 0x0f) | ((qh[index] & 0x03) << 4)) as i8 - 32;
            let q2 = ((ql[index + 32] & 0x0f) | (((qh[index] >> 2) & 0x03) << 4)) as i8 - 32;
            let q3 = ((ql[index] >> 4) | (((qh[index] >> 4) & 0x03) << 4)) as i8 - 32;
            let q4 = ((ql[index + 32] >> 4) | (((qh[index] >> 6) & 0x03) << 4)) as i8 - 32;
            out[base + index] = d * scales[scale_group] as i8 as f32 * q1 as f32;
            out[base + index + 32] = d * scales[scale_group + 2] as i8 as f32 * q2 as f32;
            out[base + index + 64] = d * scales[scale_group + 4] as i8 as f32 * q3 as f32;
            out[base + index + 96] = d * scales[scale_group + 6] as i8 as f32 * q4 as f32;
        }
    }
    out
}

/// Dispatches on `GgmlType`, returning `None` for types this repacker does
/// not decode. The pinned checkpoints only use Q4_0/Q4_K/Q8_0 for
/// quantized tensors; everything else the importer touches (F32/F16/BF16
/// norms, biases, embeddings in some conversions) is already full/half
/// precision and needs no dequantization at all.
pub fn dequantize_block(ggml_type: GgmlType, block: &[u8]) -> Option<Vec<f32>> {
    match ggml_type {
        GgmlType::Q4_0 => Some(dequantize_q4_0(block).to_vec()),
        GgmlType::Q4K => Some(dequantize_q4_k(block).to_vec()),
        GgmlType::Q6K => Some(dequantize_q6_k(block).to_vec()),
        GgmlType::Q8_0 => Some(dequantize_q8_0(block).to_vec()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F16_ONE: u16 = 0x3C00;
    const F16_TWO: u16 = 0x4000;
    const F16_ZERO: u16 = 0x0000;

    #[test]
    fn f16_known_values_round_trip() {
        assert_eq!(f16_to_f32(F16_ONE), 1.0);
        assert_eq!(f16_to_f32(F16_TWO), 2.0);
        assert_eq!(f16_to_f32(F16_ZERO), 0.0);
        // -1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn dequantize_q4_0_known_block() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&F16_ONE.to_le_bytes());
        // qs[0] = 0x18 -> low nibble 0x8 (-> 0), high nibble 0x1 (-> -7).
        block[2] = 0x18;
        let out = dequantize_q4_0(&block);
        assert_eq!(out[0], 0.0); // low nibble of qs[0]
        assert_eq!(out[16], -7.0); // high nibble of qs[0]
                                   // Remaining qs bytes are zero -> nibble 0 -> value -8.
        assert_eq!(out[1], -8.0);
        assert_eq!(out[17], -8.0);
    }

    #[test]
    fn dequantize_q8_0_known_block() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&F16_TWO.to_le_bytes());
        block[2] = 5i8.to_le_bytes()[0];
        block[3] = (-3i8).to_le_bytes()[0];
        let out = dequantize_q8_0(&block);
        assert_eq!(out[0], 10.0);
        assert_eq!(out[1], -6.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn dequantize_q4_k_with_unit_scales_and_zero_min() {
        // d = 1.0, dmin = 0.0, scales crafted so every sub-block scale
        // decodes to 1 and every min decodes to (irrelevant, since
        // dmin=0) -> dequantized value should equal the raw nibble.
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&F16_ONE.to_le_bytes());
        block[2..4].copy_from_slice(&F16_ZERO.to_le_bytes());
        // scales[0..4] = 1 (j<4 branch: d=q[j]&63); scales[8..12] = 1
        // (j>=4 branch: d=(q[j+4]&0xF)|((q[j-4]>>6)<<4), top bits of
        // scales[0..4] are 0 so this reduces to q[j+4]&0xF).
        block[4..8].copy_from_slice(&[1, 1, 1, 1]);
        block[8..12].copy_from_slice(&[1, 1, 1, 1]);
        // qs[0] = 0x12 -> low nibble 2, high nibble 1.
        block[16] = 0x12;
        let out = dequantize_q4_k(&block);
        assert_eq!(out[0], 2.0); // first low-nibble group, element 0
        assert_eq!(out[32], 1.0); // first high-nibble group, element 0
        assert_eq!(out[1], 0.0); // qs[1] is zero
    }

    #[test]
    fn dequantize_q6_k_known_zero_low_bits() {
        let mut block = [0u8; 210];
        block[192..208].fill(1); // signed group scale 1
        block[208..210].copy_from_slice(&F16_ONE.to_le_bytes());
        // ql/qh all zero encodes signed quant value -32 in every position.
        let out = dequantize_q6_k(&block);
        assert!(out.iter().all(|&value| value == -32.0));
    }

    #[test]
    fn dispatch_returns_none_for_undecoded_types() {
        assert!(dequantize_block(GgmlType::F32, &[0u8; 4]).is_some() == false);
        assert!(dequantize_block(GgmlType::Q5K, &[0u8; 176]).is_none());
    }
}
