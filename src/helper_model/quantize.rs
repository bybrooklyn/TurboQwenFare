//! Compact-vector quantization matching the checkpoint's own
//! `st_quantize.FlexibleQuantizer` module (spec §37: "Validate embedding
//! output against official/reference runtime before compact vectors").
//! Reproduced exactly from the source `st_quantize.py` shipped alongside
//! `perplexity-ai/pplx-embed-v1-0.6b`:
//!
//! - `int8`: `clamp(round(tanh(x) * 127), -128, 127)` — a *soft* tanh
//!   squash, not a per-vector min/max scale, so decoding needs no stored
//!   scale (README: "natively produce unnormalized int8-quantized
//!   embeddings... compare via cosine similarity").
//! - `binary`: `sign(x)` as `+1.0`/`-1.0` floats.
//! - `ubinary`: the same sign bits packed 8-per-byte, MSB-first
//!   (`numpy.packbits` default `bitorder="big"`), matching the model's
//!   `PackedBinaryQuantizer`.

/// MRL truncation (spec §86: "Matryoshka representation learning"): the
/// embedding is trained so any prefix is itself a valid lower-dimensional
/// embedding, so truncation needs no extra weights or renormalization.
pub fn mrl_truncate(values: &[f32], dim: usize) -> Vec<f32> {
    let len = dim.min(values.len());
    values[..len].to_vec()
}

pub fn quantize_int8_tanh(values: &[f32]) -> Vec<i8> {
    values
        .iter()
        .map(|&x| {
            let soft = x.tanh();
            (soft * 127.0).round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

pub fn quantize_binary_signed(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
        .collect()
}

pub fn quantize_binary_packed(values: &[f32]) -> Vec<u8> {
    values
        .chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &x) in chunk.iter().enumerate() {
                if x >= 0.0 {
                    byte |= 1 << (7 - i);
                }
            }
            byte
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrl_truncate_takes_a_prefix() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(mrl_truncate(&values, 2), vec![1.0, 2.0]);
        assert_eq!(mrl_truncate(&values, 10), values);
    }

    #[test]
    fn int8_clamps_to_signed_byte_range() {
        let values = vec![100.0, -100.0, 0.0];
        let q = quantize_int8_tanh(&values);
        assert!(q[0] > 0);
        assert!(q[1] < 0);
        assert_eq!(q[2], 0);
    }

    #[test]
    fn packed_binary_is_msb_first() {
        // 8 non-negative values -> all bits set -> 0xFF.
        let values = vec![1.0; 8];
        assert_eq!(quantize_binary_packed(&values), vec![0xFFu8]);
        // First value negative clears the MSB only.
        let mut mixed = vec![1.0; 8];
        mixed[0] = -1.0;
        assert_eq!(quantize_binary_packed(&mixed), vec![0b0111_1111u8]);
    }
}
