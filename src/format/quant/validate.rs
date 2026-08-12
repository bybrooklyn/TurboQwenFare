//! Independent validation decoder for the passthrough repacker (spec §279:
//! "Implement source Q4 decoder, TQF packing candidates and independent
//! validation decoder... Do not write Metal kernel assumptions into the
//! generic validation decoder; independence helps catch shared bugs.
//! Produce per-tensor mismatch report containing first block/row/value.").
//!
//! Every decode routine here is written independently of
//! `format::quant::dequant` (different iteration structure, and — for the
//! half-float conversion — different arithmetic strategy entirely) so a
//! shared bug between the packer's assumptions and the primary decoder
//! does not silently cancel out and pass validation.

use crate::error::{ContainerError, Result};
use crate::format::gguf::{GgufFile, TensorDescriptor};
use crate::format::quant::{dequant, GgmlType};

#[derive(Debug, Clone)]
pub struct QuantMismatchReport {
    pub tensor_name: String,
    pub first_block_index: u64,
    pub first_element_index: u64,
    pub row: u64,
    pub source_value: f32,
    pub repacked_value: f32,
    pub detail: String,
}

impl std::fmt::Display for QuantMismatchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quant repack mismatch in tensor {:?}: block {} element {} (row {}): \
             source={} repacked={} ({})",
            self.tensor_name,
            self.first_block_index,
            self.first_element_index,
            self.row,
            self.source_value,
            self.repacked_value,
            self.detail
        )
    }
}

/// IEEE 754 binary16 -> f32 via floating-point arithmetic rather than bit
/// composition (`dequant::f16_to_f32` builds the f32 bit pattern directly)
/// — a deliberately different method, not a wrapper around the primary one.
fn f16_to_f32_independent(bits: u16) -> f32 {
    let sign: f32 = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f32;

    if exp == 0x1F {
        return if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    if exp == 0 {
        if frac == 0.0 {
            return sign * 0.0;
        }
        // Subnormal half: value = frac/1024 * 2^-14.
        return sign * (frac / 1024.0) * 2f32.powi(-14);
    }
    // Normal half: value = (1 + frac/1024) * 2^(exp-15).
    sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
}

fn dequantize_q4_0_independent(block: &[u8]) -> [f32; dequant::Q4_0_BLOCK_ELEMENTS] {
    let d = f16_to_f32_independent(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    let mut out = [0f32; dequant::Q4_0_BLOCK_ELEMENTS];
    for (i, out_slot) in out.iter_mut().enumerate() {
        let byte = qs[i % 16];
        // Arithmetic (div/mod) nibble extraction instead of the primary
        // decoder's bit-shift/mask approach.
        let nibble = if i < 16 { byte % 16 } else { byte / 16 };
        *out_slot = (nibble as f32 - 8.0) * d;
    }
    out
}

fn dequantize_q8_0_independent(block: &[u8]) -> [f32; dequant::Q8_0_BLOCK_ELEMENTS] {
    let d = f16_to_f32_independent(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..34];
    let mut out = [0f32; dequant::Q8_0_BLOCK_ELEMENTS];
    for (i, out_slot) in out.iter_mut().enumerate() {
        let raw = qs[i] as i32;
        let signed = if raw >= 128 { raw - 256 } else { raw };
        *out_slot = signed as f32 * d;
    }
    out
}

fn dequantize_q4_k_independent(block: &[u8]) -> [f32; dequant::Q4_K_BLOCK_ELEMENTS] {
    let d = f16_to_f32_independent(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_independent(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut out = [0f32; dequant::Q4_K_BLOCK_ELEMENTS];
    // Iterate sub-blocks j = 0..8 directly (the primary decoder walks
    // pairs of sub-blocks per 64-element group); derive each sub-block's
    // own output/qs offsets from j rather than carrying running cursors.
    for j in 0..8usize {
        let (sc, m) = if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            (
                (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        };
        let dj = d * sc as f32;
        let mj = dmin * m as f32;
        let group = j / 2; // which 64-element group (0..4)
        let half = j % 2; // low-nibble half (0) or high-nibble half (1)
        let q_base = group * 32;
        let out_base = group * 64 + half * 32;
        for l in 0..32 {
            let byte = qs[q_base + l];
            let nibble = if half == 0 { byte & 0x0F } else { byte >> 4 };
            out[out_base + l] = dj * nibble as f32 - mj;
        }
    }
    out
}

/// Dispatches to the independent decoder for a given block, mirroring
/// `dequant::dequantize_block`'s type coverage.
fn dequantize_block_independent(ggml_type: GgmlType, block: &[u8]) -> Option<Vec<f32>> {
    match ggml_type {
        GgmlType::Q4_0 => Some(dequantize_q4_0_independent(block).to_vec()),
        GgmlType::Q4K => Some(dequantize_q4_k_independent(block).to_vec()),
        GgmlType::Q8_0 => Some(dequantize_q8_0_independent(block).to_vec()),
        _ => None,
    }
}

/// Validates that `repacked` (the passthrough-packed `.tqf` extent bytes
/// for `tensor`) losslessly represents the same values as the GGUF source,
/// re-reading the source tensor from `gguf` and comparing block-by-block.
/// Fails on the *first* mismatching block, returning a report with the
/// block index, absolute element index, derived row, and both values —
/// never a bare "some byte differed somewhere".
pub fn validate_tensor(gguf: &GgufFile, tensor: &TensorDescriptor, repacked: &[u8]) -> Result<()> {
    let ggml_type = tensor.ggml_type;
    let block_bytes = ggml_type.block_bytes() as usize;
    let block_elements = ggml_type.block_size() as usize;
    let row_width = tensor.dims.first().copied().unwrap_or(1).max(1);

    let mismatch = |first_block_index: u64,
                    first_element_index: u64,
                    source_value: f32,
                    repacked_value: f32,
                    detail: &str| {
        ContainerError::QuantMismatch(QuantMismatchReport {
            tensor_name: tensor.name.clone(),
            first_block_index,
            first_element_index,
            row: first_element_index / row_width,
            source_value,
            repacked_value,
            detail: detail.to_string(),
        })
        .into()
    };

    if repacked.len() as u64 != tensor.byte_size {
        return Err(mismatch(
            0,
            0,
            0.0,
            0.0,
            &format!(
                "repacked byte length {} does not match source byte size {}",
                repacked.len(),
                tensor.byte_size
            ),
        ));
    }

    let mut source_reader = gguf.quant_block_reader(tensor)?;
    let mut block_index = 0u64;
    let mut repacked_cursor = 0usize;

    while let Some(source_batch) = source_reader.next_batch()? {
        let n_blocks_in_batch = source_batch.len() / block_bytes;
        for b in 0..n_blocks_in_batch {
            let src_block = &source_batch[b * block_bytes..(b + 1) * block_bytes];
            let rep_block = &repacked[repacked_cursor..repacked_cursor + block_bytes];
            repacked_cursor += block_bytes;

            if src_block != rep_block {
                let element_base = block_index * block_elements as u64;
                if let (Some(src_vals), Some(rep_vals)) = (
                    dequantize_block_independent(ggml_type, src_block),
                    dequantize_block_independent(ggml_type, rep_block),
                ) {
                    for (lane, (sv, rv)) in src_vals.iter().zip(rep_vals.iter()).enumerate() {
                        if sv != rv {
                            return Err(mismatch(
                                block_index,
                                element_base + lane as u64,
                                *sv,
                                *rv,
                                "quant block bytes differ",
                            ));
                        }
                    }
                }
                return Err(mismatch(
                    block_index,
                    element_base,
                    f32::NAN,
                    f32::NAN,
                    "quant block bytes differ (no reference decoder for this type)",
                ));
            }

            // Bytes are identical — still cross-check the primary decoder
            // against the independent one on this block's own bytes, so a
            // shared-but-wrong assumption (e.g. both decoders silently
            // agreeing on a mis-shifted nibble) doesn't slip through only
            // because the passthrough packer never changed anything.
            if let (Some(primary), Some(independent)) = (
                dequant::dequantize_block(ggml_type, src_block),
                dequantize_block_independent(ggml_type, src_block),
            ) {
                let element_base = block_index * block_elements as u64;
                for (lane, (a, b)) in primary.iter().zip(independent.iter()).enumerate() {
                    if a != b {
                        return Err(mismatch(
                            block_index,
                            element_base + lane as u64,
                            *a,
                            *b,
                            "primary and independent decoders disagree",
                        ));
                    }
                }
            }

            block_index += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_f16_matches_primary_on_known_values() {
        for bits in [0x3C00u16, 0x4000, 0x0000, 0xBC00, 0x7C00, 0xFC00] {
            let a = dequant::f16_to_f32(bits);
            let b = f16_to_f32_independent(bits);
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert_eq!(a, b, "mismatch for bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn independent_decoders_agree_with_primary_on_arbitrary_blocks() {
        // A pseudo-random-looking but fixed byte pattern, not all zero, so
        // both nibble halves and multiple scale sub-blocks are exercised.
        let mut q4_0_block = [0u8; 18];
        for (i, b) in q4_0_block.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        assert_eq!(
            dequant::dequantize_q4_0(&q4_0_block).to_vec(),
            dequantize_q4_0_independent(&q4_0_block).to_vec()
        );

        let mut q8_0_block = [0u8; 34];
        for (i, b) in q8_0_block.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(53).wrapping_add(7);
        }
        assert_eq!(
            dequant::dequantize_q8_0(&q8_0_block).to_vec(),
            dequantize_q8_0_independent(&q8_0_block).to_vec()
        );

        let mut q4_k_block = [0u8; 144];
        for (i, b) in q4_k_block.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(19).wrapping_add(3);
        }
        assert_eq!(
            dequant::dequantize_q4_k(&q4_k_block).to_vec(),
            dequantize_q4_k_independent(&q4_k_block).to_vec()
        );
    }
}
