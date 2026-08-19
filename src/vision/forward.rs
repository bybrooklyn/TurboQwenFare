//! Vision-tower forward pass: CLIP-style ViT encoder + `qwen3vl_merger`
//! projector, following `tools/mtmd/models/qwen3vl.cpp`'s
//! `clip_graph_qwen3vl::build()` (real llama.cpp source, read but not
//! linked) exactly:
//!
//! 1. Two independent 16x16x3->1152 conv kernels are summed over the
//!    *same* still-image input (`clip_graph_qwen2vl::build_inp_with_
//!    temporal_merge`, reused unmodified by qwen3vl) — mathematically
//!    equivalent to one conv with the two kernels pre-summed, since
//!    convolution is linear in the kernel for a fixed input.
//! 2. Patches are immediately reordered from raster order into
//!    "2x2-merge-block-major" order (block traversal row-major over
//!    `(by, bx)`, within-block traversal row-major over `(dy, dx)`) —
//!    confirmed two independent ways: the real `build()`'s
//!    permute/reshape chain, and the real per-axis `positions` array
//!    construction used for M-RoPE (`case PROJECTOR_TYPE_QWEN3VL` in
//!    `clip.cpp`). Every subsequent step (bias add, position-embedding
//!    add, all 27 transformer layers) operates in this block-major
//!    order, not raster order.
//! 3. The learned absolute position table (native 48x48 grid) is
//!    bilinear-resized with `align_corners` (`resize_position_
//!    embeddings`, `ggml-cpu/ops.cpp`'s plain `GGML_SCALE_MODE_BILINEAR`
//!    branch) to the actual patch grid, reordered the same way, and
//!    added.
//! 4. 27 pre-norm transformer blocks: LN1 -> fused QKV -> per-head 2D
//!    vision M-RoPE (`GGML_ROPE_TYPE_VISION`, see `apply_vision_rope`'s
//!    doc comment for the exact frequency/pairing derivation from
//!    `ggml_mrope_cache_init`/`rotate_pairs`) -> full bidirectional
//!    attention (no mask, no GQA) -> output proj -> residual -> LN2 ->
//!    GELU MLP (up/down, both biased) -> residual.
//! 5. Post-LN, then every 4 consecutive (already block-major, hence
//!    already one spatial 2x2 group) hidden vectors concatenate into one
//!    4608-wide row -> FC1+bias -> exact GELU -> FC2+bias -> the
//!    2048-wide projected embedding per merged token.

use super::geometry::VisionGeometry;
use super::weights::{VisionLayerWeights, VisionWeights};

const HIDDEN: usize = VisionGeometry::HIDDEN;
const HEADS: usize = VisionGeometry::HEADS;
const HEAD_DIM: usize = VisionGeometry::HEAD_DIM;
const INTERMEDIATE: usize = VisionGeometry::INTERMEDIATE;
const EPS: f32 = VisionGeometry::LN_EPS;
const PATCH: usize = VisionGeometry::PATCH_SIZE;
const NATIVE_SIDE: usize = VisionGeometry::NATIVE_PATCHES_PER_SIDE;
const MERGE: usize = VisionGeometry::SPATIAL_MERGE;
const MERGED_HIDDEN: usize = VisionGeometry::MERGED_HIDDEN;
const ROPE_PAIRS: usize = VisionGeometry::ROPE_PAIRS_PER_AXIS;

fn layernorm(values: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
    let inv_std = 1.0 / (variance + EPS).sqrt();
    values
        .iter()
        .zip(weight)
        .zip(bias)
        .map(|((v, w), b)| (v - mean) * inv_std * w + b)
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

fn add_bias(values: &mut [f32], bias: &[f32]) {
    for (v, b) in values.iter_mut().zip(bias) {
        *v += b;
    }
}

/// Abramowitz & Stegun 7.1.26, same convention as
/// `helper_model::gte_reranker::forward::erf` — `clip.use_gelu = true`
/// in the real checkpoint's metadata selects exact erf-based GELU, not
/// the tanh approximation.
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

/// The combined (summed) patch-embedding conv kernel, `[oc][c][ky][kx]`
/// flat-indexed exactly as the GGUF-native `[16,16,3,1152]` (kx
/// fastest) layout decodes to: `flat = oc*3*16*16 + c*16*16 + ky*16 +
/// kx`.
fn combined_patch_kernel(weights: &VisionWeights) -> Vec<f32> {
    weights
        .patch_embed_weight0
        .values
        .iter()
        .zip(&weights.patch_embed_weight1.values)
        .map(|(a, b)| a + b)
        .collect()
}

/// `image`: normalized pixels, row-major `[height][width][channel=3]`,
/// already `(raw - IMAGE_MEAN) / IMAGE_STD`. Returns raster-order
/// (`py * grid_w + px`, `px` fastest) patch embeddings, pre-bias.
fn patch_conv(image: &[f32], image_w: usize, image_h: usize, kernel: &[f32]) -> Vec<Vec<f32>> {
    let grid_w = image_w / PATCH;
    let grid_h = image_h / PATCH;
    let mut out = vec![vec![0.0f32; HIDDEN]; grid_w * grid_h];
    for py in 0..grid_h {
        for px in 0..grid_w {
            let patch_out = &mut out[py * grid_w + px];
            for ky in 0..PATCH {
                for kx in 0..PATCH {
                    let iy = py * PATCH + ky;
                    let ix = px * PATCH + kx;
                    let pixel_base = (iy * image_w + ix) * 3;
                    for c in 0..3 {
                        let pixel = image[pixel_base + c];
                        if pixel == 0.0 {
                            continue;
                        }
                        let kernel_base = c * PATCH * PATCH + ky * PATCH + kx;
                        for oc in 0..HIDDEN {
                            patch_out[oc] += pixel * kernel[oc * 3 * PATCH * PATCH + kernel_base];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Bilinear, `align_corners`-style resize of the native `48x48` absolute
/// position table (`ggml-cpu/ops.cpp`'s `GGML_SCALE_MODE_BILINEAR`
/// branch with `GGML_SCALE_FLAG_ALIGN_CORNERS`: `pixel_offset = 0`,
/// `scale = (dst - 1) / (src - 1)`). Returns raster-order (`row * grid_w
/// + col`, matching `patch_conv`'s `py * grid_w + px`) vectors. The
/// native table's own position index is `row * NATIVE_SIDE + col`
/// (confirmed against `resize_position_embeddings`'s reshape/permute
/// chain into `(width, height, n_embd)` before `ggml_interpolate`).
fn resize_position_embeddings(table: &[f32], grid_w: usize, grid_h: usize) -> Vec<Vec<f32>> {
    let sample = |row: usize, col: usize, ch: usize| -> f32 {
        table[(row * NATIVE_SIDE + col) * HIDDEN + ch]
    };
    let sf_x = if grid_w > 1 {
        (grid_w - 1) as f32 / (NATIVE_SIDE - 1) as f32
    } else {
        0.0
    };
    let sf_y = if grid_h > 1 {
        (grid_h - 1) as f32 / (NATIVE_SIDE - 1) as f32
    } else {
        0.0
    };
    let mut out = vec![vec![0.0f32; HIDDEN]; grid_w * grid_h];
    for row in 0..grid_h {
        let y = if grid_h > 1 { row as f32 / sf_y } else { 0.0 };
        let mut y0 = y.floor() as i64;
        let mut y1 = y0 + 1;
        y0 = y0.clamp(0, NATIVE_SIDE as i64 - 1);
        y1 = y1.clamp(0, NATIVE_SIDE as i64 - 1);
        let dy = (y - y0 as f32).clamp(0.0, 1.0);
        for col in 0..grid_w {
            let x = if grid_w > 1 { col as f32 / sf_x } else { 0.0 };
            let mut x0 = x.floor() as i64;
            let mut x1 = x0 + 1;
            x0 = x0.clamp(0, NATIVE_SIDE as i64 - 1);
            x1 = x1.clamp(0, NATIVE_SIDE as i64 - 1);
            let dx = (x - x0 as f32).clamp(0.0, 1.0);

            let dest = &mut out[row * grid_w + col];
            for (ch, slot) in dest.iter_mut().enumerate() {
                let a = sample(y0 as usize, x0 as usize, ch);
                let b = sample(y0 as usize, x1 as usize, ch);
                let c = sample(y1 as usize, x0 as usize, ch);
                let d = sample(y1 as usize, x1 as usize, ch);
                *slot = a * (1.0 - dx) * (1.0 - dy)
                    + b * dx * (1.0 - dy)
                    + c * (1.0 - dx) * dy
                    + d * dx * dy;
            }
        }
    }
    out
}

/// Reorders raster-order (`py * grid_w + px`) vectors into 2x2-merge-
/// block-major order: block traversal row-major over `(by, bx)`,
/// within-block traversal row-major over `(dy, dx)`. Returns, alongside
/// the reordered vectors, the `(row, col)` grid coordinate each output
/// slot came from (needed for M-RoPE's per-position row/col inputs,
/// which are computed directly in this same block order — spec
/// `PROJECTOR_TYPE_QWEN3VL`'s `positions` construction in `clip.cpp`).
pub(crate) fn reorder_to_merge_blocks(
    raster: &[Vec<f32>],
    grid_w: usize,
    grid_h: usize,
) -> (Vec<Vec<f32>>, Vec<(usize, usize)>) {
    let mut values = Vec::with_capacity(raster.len());
    let mut coords = Vec::with_capacity(raster.len());
    for by in 0..grid_h / MERGE {
        for bx in 0..grid_w / MERGE {
            for dy in 0..MERGE {
                for dx in 0..MERGE {
                    let row = by * MERGE + dy;
                    let col = bx * MERGE + dx;
                    values.push(raster[row * grid_w + col].clone());
                    coords.push((row, col));
                }
            }
        }
    }
    (values, coords)
}

/// 2D vision M-RoPE (`GGML_ROPE_TYPE_VISION`). Derived from the real
/// `ggml_mrope_cache_init`/`rotate_pairs` in `ggml-cpu/ops.cpp`: the
/// first `ROPE_PAIRS` (18) index pairs `(j, j+36)` rotate by `row *
/// freq_base^(-j/18)`, the next 18 pairs `(18+j, 18+j+36)` rotate by
/// `col * freq_base^(-j/18)` — the *entire* 72-wide head is consumed by
/// this paired rotation (unlike text NEOX RoPE, vision has no
/// pass-through tail; confirmed by `is_vision` skipping that loop and
/// asserting `n_dims == ne0/2`).
fn apply_vision_rope(v: &mut [f32; HEAD_DIM], row: usize, col: usize) {
    let theta_scale = VisionGeometry::ROPE_FREQ_BASE.powf(-1.0 / ROPE_PAIRS as f32);
    let orig = *v;
    let mut freq = 1.0f32;
    for j in 0..ROPE_PAIRS {
        let angle = row as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = orig[j];
        let b = orig[j + 36];
        v[j] = a * cos - b * sin;
        v[j + 36] = a * sin + b * cos;
        freq *= theta_scale;
    }
    let mut freq = 1.0f32;
    for j in 0..ROPE_PAIRS {
        let idx = 18 + j;
        let angle = col as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = orig[idx];
        let b = orig[idx + 36];
        v[idx] = a * cos - b * sin;
        v[idx + 36] = a * sin + b * cos;
        freq *= theta_scale;
    }
}

type Heads = [[f32; HEAD_DIM]; HEADS];

fn project_qkv(
    layer: &VisionLayerWeights,
    normed: &[f32],
    row: usize,
    col: usize,
) -> (Heads, Heads, Heads) {
    let qkv = matvec(&layer.attn_qkv_weight.values, 3 * HIDDEN, HIDDEN, normed);
    let mut qkv = qkv;
    add_bias(&mut qkv, &layer.attn_qkv_bias.values);
    let (q_flat, rest) = qkv.split_at(HIDDEN);
    let (k_flat, v_flat) = rest.split_at(HIDDEN);

    let mut q = [[0.0f32; HEAD_DIM]; HEADS];
    let mut k = [[0.0f32; HEAD_DIM]; HEADS];
    let mut v = [[0.0f32; HEAD_DIM]; HEADS];
    for h in 0..HEADS {
        let mut qh = [0.0f32; HEAD_DIM];
        qh.copy_from_slice(&q_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
        apply_vision_rope(&mut qh, row, col);
        q[h] = qh;

        let mut kh = [0.0f32; HEAD_DIM];
        kh.copy_from_slice(&k_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
        apply_vision_rope(&mut kh, row, col);
        k[h] = kh;

        v[h].copy_from_slice(&v_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM]);
    }
    (q, k, v)
}

/// Full bidirectional attention (no mask, no GQA — `build_attn(...,
/// nullptr, kq_scale, il)` in `qwen3vl.cpp` passes no mask tensor).
fn attention(q: &[Heads], k: &[Heads], v: &[Heads], seq_len: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut concat_out = vec![0.0f32; seq_len * HIDDEN];
    for t in 0..seq_len {
        for h in 0..HEADS {
            let mut scores: Vec<f32> = (0..seq_len)
                .map(|s| {
                    let dot: f32 = q[t][h].iter().zip(&k[s][h]).map(|(a, b)| a * b).sum();
                    dot * scale
                })
                .collect();
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let dest = &mut concat_out[t * HIDDEN + h * HEAD_DIM..t * HIDDEN + (h + 1) * HEAD_DIM];
            for (weight, s) in scores.iter().zip(0..seq_len) {
                let weight = weight / sum;
                for (d, value) in dest.iter_mut().zip(&v[s][h]) {
                    *d += weight * value;
                }
            }
        }
    }
    concat_out
}

/// Runs the full 27-layer encoder + post-LN + merger over one image's
/// patches and returns one `PROJECTION_DIM`-wide row per merged
/// (2x2-block) token, in block-major order.
pub fn encode_image(
    weights: &VisionWeights,
    image: &[f32],
    image_w: usize,
    image_h: usize,
) -> Vec<Vec<f32>> {
    assert!(image_w.is_multiple_of(PATCH * MERGE) && image_h.is_multiple_of(PATCH * MERGE));
    let grid_w = image_w / PATCH;
    let grid_h = image_h / PATCH;

    let kernel = combined_patch_kernel(weights);
    let raster = patch_conv(image, image_w, image_h, &kernel);
    let (mut hidden, coords) = reorder_to_merge_blocks(&raster, grid_w, grid_h);
    for h in hidden.iter_mut() {
        add_bias(h, &weights.patch_embed_bias.values);
    }

    let resized_pos = resize_position_embeddings(&weights.position_embed.values, grid_w, grid_h);
    let (pos_blocked, _) = reorder_to_merge_blocks(&resized_pos, grid_w, grid_h);
    for (h, p) in hidden.iter_mut().zip(&pos_blocked) {
        for (a, b) in h.iter_mut().zip(p) {
            *a += b;
        }
    }

    let seq_len = hidden.len();

    for layer in &weights.layers {
        let normed: Vec<Vec<f32>> = hidden
            .iter()
            .map(|h| layernorm(h, &layer.ln1_weight.values, &layer.ln1_bias.values))
            .collect();

        let mut q = Vec::with_capacity(seq_len);
        let mut k = Vec::with_capacity(seq_len);
        let mut v = Vec::with_capacity(seq_len);
        for (t, n) in normed.iter().enumerate() {
            let (row, col) = coords[t];
            let (qh, kh, vh) = project_qkv(layer, n, row, col);
            q.push(qh);
            k.push(kh);
            v.push(vh);
        }

        let attn_concat = attention(&q, &k, &v, seq_len);
        for t in 0..seq_len {
            let concat = &attn_concat[t * HIDDEN..(t + 1) * HIDDEN];
            let mut attn_out = matvec(&layer.attn_out_weight.values, HIDDEN, HIDDEN, concat);
            add_bias(&mut attn_out, &layer.attn_out_bias.values);
            for i in 0..HIDDEN {
                hidden[t][i] += attn_out[i];
            }
        }

        for h in hidden.iter_mut() {
            let normed2 = layernorm(h, &layer.ln2_weight.values, &layer.ln2_bias.values);
            let mut up = matvec(&layer.ffn_up_weight.values, INTERMEDIATE, HIDDEN, &normed2);
            add_bias(&mut up, &layer.ffn_up_bias.values);
            let act: Vec<f32> = up.into_iter().map(gelu).collect();
            let mut down = matvec(&layer.ffn_down_weight.values, HIDDEN, INTERMEDIATE, &act);
            add_bias(&mut down, &layer.ffn_down_bias.values);
            for i in 0..HIDDEN {
                h[i] += down[i];
            }
        }
    }

    for h in hidden.iter_mut() {
        *h = layernorm(
            h,
            &weights.post_ln_weight.values,
            &weights.post_ln_bias.values,
        );
    }

    let n_merged = seq_len / (MERGE * MERGE);
    let mut merged_rows = Vec::with_capacity(n_merged);
    for m in 0..n_merged {
        let mut row = Vec::with_capacity(MERGED_HIDDEN);
        for sub in 0..MERGE * MERGE {
            row.extend_from_slice(&hidden[m * MERGE * MERGE + sub]);
        }
        merged_rows.push(row);
    }

    let mut out = Vec::with_capacity(n_merged);
    for row in &merged_rows {
        let mut fc1 = matvec(
            &weights.merger_fc1_weight.values,
            MERGED_HIDDEN,
            MERGED_HIDDEN,
            row,
        );
        add_bias(&mut fc1, &weights.merger_fc1_bias.values);
        let act: Vec<f32> = fc1.into_iter().map(gelu).collect();
        let mut fc2 = matvec(
            &weights.merger_fc2_weight.values,
            VisionGeometry::PROJECTION_DIM,
            MERGED_HIDDEN,
            &act,
        );
        add_bias(&mut fc2, &weights.merger_fc2_bias.values);
        out.push(fc2);
    }
    out
}
