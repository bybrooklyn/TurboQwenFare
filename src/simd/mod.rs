//! CPU SIMD kernels for paths that do not warrant a GPU dispatch (spec Part
//! VII section 56). Phase 25 (spec §297): the M4 assault's first compute
//! lever is making the Qwen quantized dot products - the expert Q4_K
//! SwiGLU path, the Q6_K LM head, and the Q8_0 projections - use NEON
//! instead of scalar dequantize-and-multiply.
//!
//! Correctness discipline: the NEON kernels deliberately reproduce the
//! *exact integer lane sums* of the scalar reference (`q4_k_dot`,
//! `q6_k_dot`, `q8_0_dot` in `model::qwen36::weights`), then hand those
//! integers back to the same f32 combination code. All arithmetic below
//! is integer (vtrn + widen + int16 multiply + pairwise add), and integer
//! multiply/add is exact and associative, so identical lane sums plus
//! identical f32 op order means the SIMD path is bit-identical to the
//! scalar path - the 128-token greedy oracle cannot diverge because of
//! the SIMD switch. Differential fuzz tests enforce that identity.
//!
//! A/B control (spec invariant #10): `TQF_SIMD_Q4K=0` (and
//! `TQF_SIMD_Q6K=0`, `TQF_SIMD_Q8_0=0`) force the scalar baseline back
//! on; the SIMD paths are the default on aarch64.

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// Lane sums for one Q4_K block, exactly matching the scalar
    /// reference's stride-8 decomposition: `lane_sums[l]` accumulates
    /// `scale[subblock] * q8[subblock*32 + quarter*8 + l] * q4[same]`
    /// over all subblocks (8) and quarters (4). Each scalar lane has four
    /// elements at stride 8; `vtrn` pairs quarter q0 with q1 and quarter
    /// q2 with q3, int16 products are exact, and `vpaddl` folds the pairs
    /// into the same four-term integer sum.
    pub unsafe fn q4k_block_lane_sums(
        packed_values: &[u8; 128],
        packed_scales: &[u8; 12],
        q8_values: &[i8; 256],
    ) -> [i32; 8] {
        let mask = vdupq_n_u8(0x0F);
        let mut acc_even = vdupq_n_s32(0);
        let mut acc_odd = vdupq_n_s32(0);
        for subblock in 0..8 {
            let chunk = subblock / 2;
            let high = subblock % 2 == 1;
            let qs = vld1q_u8(packed_values.as_ptr().add(chunk * 32));
            let qs_hi = vld1q_u8(packed_values.as_ptr().add(chunk * 32 + 16));
            let (nibbles0, nibbles1) = if high {
                (vshrq_n_u8(qs, 4), vshrq_n_u8(qs_hi, 4))
            } else {
                (vandq_u8(qs, mask), vandq_u8(qs_hi, mask))
            };
            let q4_0 = vreinterpretq_s8_u8(nibbles0);
            let q4_1 = vreinterpretq_s8_u8(nibbles1);

            let q8_a = vld1q_s8(q8_values.as_ptr().add(subblock * 32));
            let q8_b = vld1q_s8(q8_values.as_ptr().add(subblock * 32 + 16));

            let a0 = vmovl_s8(vget_low_s8(q8_a)); // elements  0..7
            let a1 = vmovl_s8(vget_high_s8(q8_a)); // elements  8..15
            let b0 = vmovl_s8(vget_low_s8(q8_b)); // elements 16..23
            let b1 = vmovl_s8(vget_high_s8(q8_b)); // elements 24..31
            let c0 = vmovl_s8(vget_low_s8(q4_0));
            let c1 = vmovl_s8(vget_high_s8(q4_0));
            let d0 = vmovl_s8(vget_low_s8(q4_1));
            let d1 = vmovl_s8(vget_high_s8(q4_1));

            // TRN on 16-bit lanes pairs *even* positions (vtrn1) and
            // *odd* positions (vtrn2). Even scalar lanes come from
            // (a0,a1,b0,b1) even indices, odd lanes from odd indices.
            let p = vmulq_s16(vtrn1q_s16(a0, a1), vtrn1q_s16(c0, c1));
            let q = vmulq_s16(vtrn2q_s16(a0, a1), vtrn2q_s16(c0, c1));
            let r = vmulq_s16(vtrn1q_s16(b0, b1), vtrn1q_s16(d0, d1));
            let s = vmulq_s16(vtrn2q_s16(b0, b1), vtrn2q_s16(d0, d1));

            let scale = q4_k_scale(subblock, packed_scales) as i32;
            let even = vmulq_n_s32(vpaddlq_s16(vaddq_s16(p, r)), scale);
            let odd = vmulq_n_s32(vpaddlq_s16(vaddq_s16(q, s)), scale);
            acc_even = vaddq_s32(acc_even, even);
            acc_odd = vaddq_s32(acc_odd, odd);
        }
        let mut out = [0i32; 8];
        store_interleaved_32(out.as_mut_ptr(), acc_even, acc_odd);
        out
    }

    /// Lane sums for one Q6_K block, same contract. The scalar reference
    /// walks 16 groups of 16 contiguous elements; each scalar lane has
    /// exactly two elements at stride 8.
    pub unsafe fn q6k_block_lane_sums(
        ql: &[u8; 128],
        qh: &[u8; 64],
        scales: &[u8; 16],
        q8_values: &[i8; 256],
    ) -> [i32; 8] {
        let mut acc_even = vdupq_n_s32(0);
        let mut acc_odd = vdupq_n_s32(0);
        let mask = vdupq_n_u8(0x0F);
        let shift32 = vdupq_n_u8(32);
        for group in 0..16 {
            let half = group / 8;
            let group_in_half = group % 8;
            // Scalar unpack mapping: groups 0-1 read qh[0..32] bits 0-1,
            // groups 2-3 read qh[0..32] bits 2-3, groups 4-5 read
            // qh[0..32] bits 4-5, groups 6-7 read qh[0..32] bits 6-7;
            // groups 0-3 use low nibbles, groups 4-7 high nibbles.
            let low = vld1q_u8(ql.as_ptr().add(half * 64 + (group_in_half % 4) * 16));
            let high = vld1q_u8(qh.as_ptr().add(half * 32 + (group_in_half % 2) * 16));
            let bits = match (group_in_half / 2) * 2 {
                0 => vandq_u8(high, vdupq_n_u8(0x03)),
                2 => vandq_u8(vshrq_n_u8(high, 2), vdupq_n_u8(0x03)),
                4 => vandq_u8(vshrq_n_u8(high, 4), vdupq_n_u8(0x03)),
                _ => vandq_u8(vshrq_n_u8(high, 6), vdupq_n_u8(0x03)),
            };
            let nibbles = if group_in_half < 4 {
                vandq_u8(low, mask)
            } else {
                vshrq_n_u8(low, 4)
            };
            let q4 = vsubq_u8(vorrq_u8(nibbles, vshlq_n_u8(bits, 4)), shift32);
            let q4_s = vreinterpretq_s8_u8(q4);

            let q8_16 = vld1q_s8(q8_values.as_ptr().add(group * 16));
            let a0 = vmovl_s8(vget_low_s8(q8_16));
            let a1 = vmovl_s8(vget_high_s8(q8_16));
            let c0 = vmovl_s8(vget_low_s8(q4_s));
            let c1 = vmovl_s8(vget_high_s8(q4_s));

            // Even lanes via vtrn1 pairs, odd lanes via vtrn2 pairs
            // (TRN pairs even/odd *positions*), interleaved at the end.
            let p = vmulq_s16(vtrn1q_s16(a0, a1), vtrn1q_s16(c0, c1));
            let q = vmulq_s16(vtrn2q_s16(a0, a1), vtrn2q_s16(c0, c1));
            let scale = scales[group] as i8 as i32;
            acc_even = vaddq_s32(acc_even, vmulq_n_s32(vpaddlq_s16(p), scale));
            acc_odd = vaddq_s32(acc_odd, vmulq_n_s32(vpaddlq_s16(q), scale));
        }
        let mut out = [0i32; 8];
        store_interleaved_32(out.as_mut_ptr(), acc_even, acc_odd);
        out
    }

    /// Dot of one Q8_0 block (32 elements) against its quantized input,
    /// exact integer match to the scalar reference. Int16 lane products
    /// (max 127*127 = 16129) fit exactly; pairwise adds are exact.
    /// Takes a 34-byte slice so the hot row loop never copies a block
    /// into an array just to hand it to the kernel.
    pub unsafe fn q8_0_block_dot(weights: &[u8], q8_values: &[i8; 32]) -> i32 {
        debug_assert_eq!(weights.len(), 34);
        let q_a = vld1q_s8(q8_values.as_ptr());
        let q_b = vld1q_s8(q8_values.as_ptr().add(16));
        let w_a = vld1q_s8(weights.as_ptr().add(2) as *const i8);
        let w_b = vld1q_s8(weights.as_ptr().add(18) as *const i8);
        let p = vmulq_s16(vmovl_s8(vget_low_s8(q_a)), vmovl_s8(vget_low_s8(w_a)));
        let q = vmulq_s16(vmovl_s8(vget_high_s8(q_a)), vmovl_s8(vget_high_s8(w_a)));
        let r = vmulq_s16(vmovl_s8(vget_low_s8(q_b)), vmovl_s8(vget_low_s8(w_b)));
        let s = vmulq_s16(vmovl_s8(vget_high_s8(q_b)), vmovl_s8(vget_high_s8(w_b)));
        vaddvq_s32(vaddq_s32(
            vpaddlq_s16(vaddq_s16(p, q)),
            vpaddlq_s16(vaddq_s16(r, s)),
        ))
    }

    /// Stores `[a0, b0, a1, b1, a2, b2, a3, b3]` into eight int32 lanes
    /// using the well-defined 64-bit zip intrinsics.
    #[inline]
    unsafe fn store_interleaved_32(out: *mut i32, a: int32x4_t, b: int32x4_t) {
        let z01 = vzip1_s32(vget_low_s32(a), vget_low_s32(b));
        let z23 = vzip2_s32(vget_low_s32(a), vget_low_s32(b));
        let z45 = vzip1_s32(vget_high_s32(a), vget_high_s32(b));
        let z67 = vzip2_s32(vget_high_s32(a), vget_high_s32(b));
        vst1q_s32(out, vcombine_s32(z01, z23));
        vst1q_s32(out.add(4), vcombine_s32(z45, z67));
    }

    /// The per-subblock 6-bit scale, mirroring
    /// `weights::q4_k_scale_min`'s scale half.
    fn q4_k_scale(index: usize, packed: &[u8; 12]) -> u8 {
        if index < 4 {
            packed[index] & 63
        } else {
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4)
        }
    }
}

/// Whether the NEON kernels are compiled in and enabled
/// (`TQF_SIMD_Q4K`/`TQF_SIMD_Q6K`/`TQF_SIMD_Q8_0` disable them for A/B;
/// default on for aarch64 - spec invariant #10). Read exactly once per
/// process: the hot dot loops call these per row/block, and a per-call
/// `std::env::var` there costs real milliseconds per token.
#[cfg(target_arch = "aarch64")]
fn simd_enabled_once(var: &str) -> bool {
    match std::env::var(var).ok().as_deref() {
        Some(value) => {
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("true")
        }
        None => true,
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn simd_enabled_once(_var: &str) -> bool {
    false
}

pub fn q4k_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| simd_enabled_once("TQF_SIMD_Q4K"))
}

pub fn q6k_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| simd_enabled_once("TQF_SIMD_Q6K"))
}

pub fn q8_0_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| simd_enabled_once("TQF_SIMD_Q8_0"))
}

/// Lane sums for one Q4_K block. Returns `None` when the SIMD path is
/// disabled; the caller falls back to the scalar reference.
#[cfg(target_arch = "aarch64")]
pub fn q4k_block_lane_sums(
    packed_values: &[u8; 128],
    packed_scales: &[u8; 12],
    q8_values: &[i8; 256],
) -> Option<[i32; 8]> {
    if q4k_enabled() {
        Some(unsafe { neon::q4k_block_lane_sums(packed_values, packed_scales, q8_values) })
    } else {
        None
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q4k_block_lane_sums(
    _packed_values: &[u8; 128],
    _packed_scales: &[u8; 12],
    _q8_values: &[i8; 256],
) -> Option<[i32; 8]> {
    None
}

/// Lane sums for one Q6_K block, same contract.
#[cfg(target_arch = "aarch64")]
pub fn q6k_block_lane_sums(
    ql: &[u8; 128],
    qh: &[u8; 64],
    scales: &[u8; 16],
    q8_values: &[i8; 256],
) -> Option<[i32; 8]> {
    if q6k_enabled() {
        Some(unsafe { neon::q6k_block_lane_sums(ql, qh, scales, q8_values) })
    } else {
        None
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q6k_block_lane_sums(
    _ql: &[u8; 128],
    _qh: &[u8; 64],
    _scales: &[u8; 16],
    _q8_values: &[i8; 256],
) -> Option<[i32; 8]> {
    None
}

/// Dot of one Q8_0 block against its quantized input.
#[cfg(target_arch = "aarch64")]
pub fn q8_0_block_dot(weights: &[u8], q8_values: &[i8; 32]) -> Option<i32> {
    if q8_0_enabled() {
        Some(unsafe { neon::q8_0_block_dot(weights, q8_values) })
    } else {
        None
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q8_0_block_dot(_weights: &[u8], _q8_values: &[i8; 32]) -> Option<i32> {
    None
}

/// Whole-row Q8_0 dot: one kernel call per weight row instead of one per
/// 32-element block (the Phase 25 call-overhead fix). The f32
/// accumulation order matches the scalar reference exactly - `output +=
/// integer_dot * (block_scale * stored_activation_scale)` per block in
/// physical order - so this is bit-identical.
#[cfg(target_arch = "aarch64")]
pub fn q8_0_row_dot(row: &[u8], quantized: &[([i8; 32], f32)]) -> Option<f32> {
    if !q8_0_enabled() {
        return None;
    }
    let mut output = 0.0f32;
    for (block_index, (quantized, stored_activation_scale)) in quantized.iter().enumerate() {
        let weight = &row[block_index * 34..(block_index + 1) * 34];
        let weight_scale = f16_to_f32(u16::from_le_bytes([weight[0], weight[1]]));
        let integer_dot = unsafe { neon::q8_0_block_dot(weight, quantized) };
        output += integer_dot as f32 * (weight_scale * *stored_activation_scale);
    }
    Some(output)
}

/// f16 half-precision conversion, bit-identical to the shared
/// `format::quant::dequant` helper (duplicated here so the hot row loop
/// has no cross-module non-inlined call).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let fraction = (bits & 0x03FF) as u32;
    let value: u32 = match exponent {
        0 => {
            if fraction == 0 {
                sign << 31
            } else {
                // Subnormal: normalize via the mantissa trick.
                let mut mantissa = fraction;
                let mut exponent_adjust = 127 - 15 + 1;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    exponent_adjust -= 1;
                }
                (sign << 31) | (exponent_adjust << 23) | ((mantissa & 0x03FF) << 13)
            }
        }
        0x1F => (sign << 31) | 0x7F80_0000 | (fraction << 13),
        _ => (sign << 31) | ((exponent + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q8_0_row_dot(_row: &[u8], _quantized: &[([i8; 32], f32)]) -> Option<f32> {
    None
}

#[cfg(test)]
mod tests {
    fn xorshift(state: &mut u64) -> u8 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state & 0xFF) as u8
    }

    /// Scalar reference for the Q4_K lane sums, mirroring
    /// `weights::q4_k_dot`'s inner loop exactly.
    fn q4k_lane_sums_scalar(values: &[u8; 128], scales: &[u8; 12], q8: &[i8; 256]) -> [i32; 8] {
        let mut lane_sums = [0i32; 8];
        for subblock in 0..8 {
            let scale = if subblock < 4 {
                scales[subblock] & 63
            } else {
                (scales[subblock + 4] & 0x0f) | ((scales[subblock - 4] >> 6) << 4)
            } as i32;
            let chunk = subblock / 2;
            let high = subblock % 2 == 1;
            let source = &values[chunk * 32..(chunk + 1) * 32];
            for quarter in 0..4 {
                for lane in 0..8 {
                    let q4 = if high {
                        source[quarter * 8 + lane] >> 4
                    } else {
                        source[quarter * 8 + lane] & 0x0f
                    } as i32;
                    let index = subblock * 32 + quarter * 8 + lane;
                    lane_sums[lane] += scale * q8[index] as i32 * q4;
                }
            }
        }
        lane_sums
    }

    /// Scalar reference for the Q6_K lane sums, mirroring
    /// `weights::q6_k_dot`'s unpacking and group loop exactly.
    fn q6k_lane_sums_scalar(
        ql: &[u8; 128],
        qh: &[u8; 64],
        scales: &[u8; 16],
        q8: &[i8; 256],
    ) -> [i32; 8] {
        let mut unpacked = [0i8; 256];
        for half in 0..2 {
            let low = &ql[half * 64..(half + 1) * 64];
            let high = &qh[half * 32..(half + 1) * 32];
            let base = half * 128;
            for index in 0..32 {
                unpacked[base + index] =
                    ((low[index] & 0x0f) | ((high[index] & 0x03) << 4)) as i8 - 32;
                unpacked[base + index + 32] =
                    ((low[index + 32] & 0x0f) | (((high[index] >> 2) & 0x03) << 4)) as i8 - 32;
                unpacked[base + index + 64] =
                    ((low[index] >> 4) | (((high[index] >> 4) & 0x03) << 4)) as i8 - 32;
                unpacked[base + index + 96] =
                    ((low[index + 32] >> 4) | (((high[index] >> 6) & 0x03) << 4)) as i8 - 32;
            }
        }
        let mut lane_sums = [0i32; 8];
        for group in 0..16 {
            let scale = scales[group] as i8 as i32;
            let start = group * 16;
            for half in 0..2 {
                for lane in 0..8 {
                    let index = start + half * 8 + lane;
                    lane_sums[lane] += scale * q8[index] as i32 * unpacked[index] as i32;
                }
            }
        }
        lane_sums
    }

    /// Differential identity: the NEON lane sums must equal the scalar
    /// reference for randomized blocks. This is the guard that keeps the
    /// SIMD switch bit-identical and therefore oracle-safe.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_neon_lane_sums_match_scalar_reference() {
        use super::neon;
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _seed in 0..256 {
            let mut values = [0u8; 128];
            let mut scales = [0u8; 12];
            let mut q8 = [0i8; 256];
            for byte in values.iter_mut().chain(&mut scales) {
                *byte = xorshift(&mut state);
            }
            for value in q8.iter_mut() {
                *value = (xorshift(&mut state) as i32 - 127) as i8;
            }
            let expected = q4k_lane_sums_scalar(&values, &scales, &q8);
            let actual = unsafe { neon::q4k_block_lane_sums(&values, &scales, &q8) };
            assert_eq!(actual, expected, "NEON Q4_K lane sums diverged");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_neon_lane_sums_match_scalar_reference() {
        use super::neon;
        let mut state = 0xFEED_FACE_CAFE_BEEFu64;
        for _seed in 0..256 {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0u8; 16];
            let mut q8 = [0i8; 256];
            for byte in ql.iter_mut().chain(&mut qh).chain(&mut scales) {
                *byte = xorshift(&mut state);
            }
            for value in q8.iter_mut() {
                *value = (xorshift(&mut state) as i32 - 127) as i8;
            }
            let expected = q6k_lane_sums_scalar(&ql, &qh, &scales, &q8);
            let actual = unsafe { neon::q6k_block_lane_sums(&ql, &qh, &scales, &q8) };
            assert_eq!(actual, expected, "NEON Q6_K lane sums diverged");
        }
    }

    #[test]
    fn f16_conversion_matches_shared_dequant() {
        let mut state = 0x5EED_1234u64;
        for _ in 0..65536 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = (state & 0xFFFF) as u16;
            let expected = crate::format::quant::dequant::f16_to_f32(bits);
            let actual = super::f16_to_f32(bits);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f16 {bits:#06x} converted differently"
            );
        }
    }

    /// Phase 25 microbenchmark: a Qwen-scale Q8_0 GEMV (2048 x 8192,
    /// the GDN in-projection shape) through the row kernel, reporting
    /// effective GFLOPS so the M4 assault ledger can price this stage.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q8_0_row_dot_microbenchmark() {
        use std::time::Instant;
        let rows = 2048usize;
        let cols = 8192usize;
        let mut state = 0xABCD_EF01u64;
        let mut matrix = vec![0u8; rows * (cols / 32) * 34];
        let mut input = vec![0f32; cols];
        for byte in matrix.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state & 0xFF) as u8;
        }
        for value in input.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = ((state % 1000) as f32) / 1000.0;
        }
        let quantized: Vec<([i8; 32], f32)> = input
            .chunks_exact(32)
            .map(|chunk| {
                let maximum = chunk
                    .iter()
                    .fold(0.0f32, |current, value| current.max(value.abs()));
                let activation_scale = maximum / 127.0;
                let inverse = if activation_scale == 0.0 {
                    0.0
                } else {
                    activation_scale.recip()
                };
                let stored = crate::format::quant::dequant::f16_to_f32(
                    crate::format::quant::dequant::f32_to_f16(activation_scale),
                );
                let mut quantized = [0i8; 32];
                for (slot, activation) in quantized.iter_mut().zip(chunk) {
                    *slot = (activation * inverse).round().clamp(-128.0, 127.0) as i8;
                }
                (quantized, stored)
            })
            .collect();
        let start = Instant::now();
        let mut acc = 0.0f64;
        let block_bytes = cols / 32 * 34;
        for row in 0..rows {
            let row_bytes = &matrix[row * block_bytes..(row + 1) * block_bytes];
            let out = super::q8_0_row_dot(row_bytes, &quantized).unwrap();
            acc += out as f64;
        }
        let elapsed = start.elapsed();
        let flops = 2.0 * rows as f64 * cols as f64;
        println!(
            "q8_0_microbench rows={rows} cols={cols} ms={:.1} gflops={:.1} acc={acc:.0}",
            elapsed.as_secs_f64() * 1e3,
            flops / elapsed.as_secs_f64() / 1e9,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q8_0_neon_dot_matches_scalar_reference() {
        use super::neon;
        let mut state = 0x0BAD_5EED_DEAD_BEEFu64;
        for _seed in 0..256 {
            let mut weights = [0u8; 34];
            let mut q8 = [0i8; 32];
            for byte in weights.iter_mut() {
                *byte = xorshift(&mut state);
            }
            for value in q8.iter_mut() {
                *value = (xorshift(&mut state) as i32 - 127) as i8;
            }
            let expected: i32 = weights[2..34]
                .iter()
                .zip(q8)
                .map(|(&weight, value)| weight as i8 as i32 * value as i32)
                .sum();
            let actual = unsafe { neon::q8_0_block_dot(&weights[..], &q8) };
            assert_eq!(actual, expected);
        }
    }
}
