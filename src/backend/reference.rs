//! Backend-agnostic CPU reference kernels (spec §283, phase 11: "Every
//! kernel has CPU/reference fixture and shape/alignment assertions.").
//!
//! These are the correctness oracle every GPU kernel (Metal today, CUDA
//! later — spec §48 "Metal and CUDA share high-level operations... not
//! kernel implementation") must match, exercised by each backend's own
//! parity tests. Q4_K dequantization is delegated to
//! `format::quant::dequant::dequantize_q4_k` — the same decoder the `.tqf`
//! importer already ships and tests — rather than a second reimplementation;
//! that module is distinct from `format::quant::validate`'s *independent*
//! decoder, whose whole purpose is to not share code with the primary
//! decode path.

use crate::format::quant::dequant::dequantize_q4_k;
use crate::format::quant::GgmlType;

/// `weights` is `rows` rows of `cols / 32` contiguous Q8_0 blocks
/// (34 bytes: f16 scale + 32 int8 quantized values), row-major. Accumulates
/// per element in row order exactly like the loaded-tensor CPU path
/// (`decode_values` then a row dot), so the GPU Q8 kernels' oracle matches
/// the live GDN/attention projection math, not just the quant format.
pub fn q8_gemv(weights: &[u8], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let block_bytes = GgmlType::Q8_0.block_bytes() as usize;
    let blocks_per_row = cols / 32;
    (0..rows)
        .map(|row| {
            let row_base = row * blocks_per_row * block_bytes;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let block = &weights[row_base + b * block_bytes..row_base + (b + 1) * block_bytes];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let vblock = &vector[b * 32..(b + 1) * 32];
                for j in 0..32 {
                    acc += d * (block[2 + j] as i8) as f32 * vblock[j];
                }
            }
            acc
        })
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    crate::format::quant::dequant::f16_to_f32(bits)
}

/// `weights` is `rows` rows of `cols / 256` contiguous Q4_K blocks,
/// row-major. Panics (via slice indexing) on malformed input — callers own
/// shape validation, same contract as the GPU kernels' host wrappers.
pub fn q4k_gemv(weights: &[u8], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let block_bytes = GgmlType::Q4K.block_bytes() as usize;
    let blocks_per_row = cols / 256;
    (0..rows)
        .map(|row| {
            let row_base = row * blocks_per_row * block_bytes;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let block = &weights[row_base + b * block_bytes..row_base + (b + 1) * block_bytes];
                let dequant = dequantize_q4_k(block);
                let vblock = &vector[b * 256..(b + 1) * 256];
                for i in 0..256 {
                    acc += dequant[i] * vblock[i];
                }
            }
            acc
        })
        .collect()
}

/// `mat` is `tokens` rows of `cols` f32, row-major. Returns `tokens` rows of
/// `rows` f32, row-major (`out[token * rows + row]`), matching the GPU
/// kernel's output layout.
pub fn q4k_gemm(weights: &[u8], mat: &[f32], tokens: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * rows];
    for token in 0..tokens {
        let vector = &mat[token * cols..(token + 1) * cols];
        let row_out = q4k_gemv(weights, vector, rows, cols);
        out[token * rows..(token + 1) * rows].copy_from_slice(&row_out);
    }
    out
}

/// `x` is `rows` rows of `cols` f32, row-major; `weight` is `cols` f32.
pub fn rmsnorm(x: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_slice = &x[row * cols..(row + 1) * cols];
        let mean_sq = row_slice.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..cols {
            out[row * cols + i] = row_slice[i] * inv_rms * weight[i];
        }
    }
    out
}

pub fn residual_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

pub fn sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_add_is_elementwise_sum() {
        assert_eq!(residual_add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
    }

    #[test]
    fn silu_zero_is_zero() {
        assert_eq!(silu(&[0.0])[0], 0.0);
    }

    #[test]
    fn sigmoid_zero_is_one_half() {
        assert_eq!(sigmoid(&[0.0])[0], 0.5);
    }

    #[test]
    fn rmsnorm_unit_weight_normalizes_to_unit_rms() {
        let x = vec![3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5)
        let weight = vec![1.0, 1.0];
        let out = rmsnorm(&x, &weight, 1, 2, 0.0);
        let out_rms = (out.iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!((out_rms - 1.0).abs() < 1e-5);
    }
}
