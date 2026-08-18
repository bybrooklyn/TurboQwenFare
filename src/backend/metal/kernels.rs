//! Phase 11 reference Q4 kernels (spec §283, REFERENCE BASELINE): "slow-
//! clear" kernels first — Q4 GEMV, Q4 batched GEMM/prefill, RMSNorm,
//! elementwise residual/SiLU/sigmoid, and a simple LM-head path built on top
//! of the GEMV kernel (spec §12: the output head is a single dense
//! Q4-quantized `[vocab, hidden]` matrix, so "LM head" is not a distinct
//! kernel family, just that GEMV applied to the vocab-shaped weight).
//!
//! Every kernel here dequantizes-then-loops with no threadgroup staging,
//! fused dequant/load-width tricks, or specialization variants — spec §51's
//! kernel-family specialization table ("load width, group layout, nibble
//! unpack...") is a later optimization phase this reference implementation
//! exists to be validated against, not compete with. Host wrappers assert
//! input shapes/alignment before dispatch (spec §283); `backend::reference`
//! holds the CPU oracle each kernel's tests check parity against.

use metal_sys::{CompileOptions, Library, MTLSize};

use crate::backend::reference;
use crate::error::{BackendError, Result};
use crate::format::quant::dequant::Q4_K_BLOCK_ELEMENTS;
use crate::format::quant::GgmlType;

use super::buffer::BufferLease;
use super::context::MetalContext;
use super::pipeline::PipelineCache;

pub const Q4K_GEMV_FUNCTION: &str = "tqf_q4k_gemv";
pub const Q4K_GEMV_STAGED16_FUNCTION: &str = "tqf_q4k_gemv_staged16";
pub const Q4K_GEMM_FUNCTION: &str = "tqf_q4k_gemm";
pub const RMSNORM_FUNCTION: &str = "tqf_rmsnorm";
pub const RESIDUAL_ADD_FUNCTION: &str = "tqf_residual_add";
pub const SILU_FUNCTION: &str = "tqf_silu";
pub const SIGMOID_FUNCTION: &str = "tqf_sigmoid";

/// Threads per threadgroup for `tqf_rmsnorm`'s row reduction — must match
/// the `threadgroup float shared[256]` size hard-coded in the MSL source
/// below (a compile-time-sized threadgroup array can't take a runtime
/// specialization constant without more plumbing than this reference
/// kernel needs yet).
const RMSNORM_THREADGROUP_SIZE: u64 = 256;

pub const REFERENCE_KERNELS_MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Ports format::quant::dequant::dequantize_q4_k's block_q4_K layout exactly
// (ggml/llama.cpp wire format, spec Part XIV): ggml_half d (2B), ggml_half
// dmin (2B), 12B of packed 6-bit sub-block scale/min pairs, 128B of packed
// nibbles -> 144B/block, 256 elements/block. `out` must hold 256 floats.
inline void tqf_q4k_dequant_block(device const uchar* block, thread float* out) {
    half d_h = as_type<half>(ushort(ushort(block[0]) | (ushort(block[1]) << 8)));
    half dmin_h = as_type<half>(ushort(ushort(block[2]) | (ushort(block[3]) << 8)));
    float d = float(d_h);
    float dmin = float(dmin_h);
    device const uchar* scales = block + 4;
    device const uchar* qs = block + 16;

    uint out_idx = 0;
    uint q_idx = 0;
    uint is = 0;
    for (uint iter = 0; iter < 4; ++iter) {
        uchar sc1, m1, sc2, m2;
        if (is < 4) {
            sc1 = scales[is] & 63;
            m1 = scales[is + 4] & 63;
        } else {
            sc1 = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
            m1 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
        }
        uint is2 = is + 1;
        if (is2 < 4) {
            sc2 = scales[is2] & 63;
            m2 = scales[is2 + 4] & 63;
        } else {
            sc2 = (scales[is2 + 4] & 0x0F) | ((scales[is2 - 4] >> 6) << 4);
            m2 = (scales[is2 + 4] >> 4) | ((scales[is2] >> 6) << 4);
        }
        float d1 = d * float(sc1);
        float mm1 = dmin * float(m1);
        float d2 = d * float(sc2);
        float mm2 = dmin * float(m2);
        for (uint l = 0; l < 32; ++l) {
            uchar byte = qs[q_idx + l];
            out[out_idx + l] = d1 * float(byte & 0x0F) - mm1;
            out[out_idx + 32 + l] = d2 * float(byte >> 4) - mm2;
        }
        out_idx += 64;
        q_idx += 32;
        is += 2;
    }
}

// out[row] = dot(dequant(weights[row, :]), vec). `weights` is `rows` rows of
// `cols/256` contiguous 144-byte Q4_K blocks, row-major.
kernel void tqf_q4k_gemv(
    device const uchar* weights [[buffer(0)]],
    device const float* vec [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    uint blocks_per_row = cols / 256;
    device const uchar* row_base = weights + (ulong)row * (ulong)blocks_per_row * 144;
    float acc = 0.0f;
    thread float dequant[256];
    for (uint b = 0; b < blocks_per_row; ++b) {
        tqf_q4k_dequant_block(row_base + (ulong)b * 144, dequant);
        device const float* vblock = vec + (ulong)b * 256;
        for (uint i = 0; i < 256; ++i) {
            acc += dequant[i] * vblock[i];
        }
    }
    out[row] = acc;
}

// Phase 20 NVMAI-derived MoE phase-1 staged variant (spec §292): 16 rows
// per threadgroup, block bytes cooperatively staged into threadgroup memory
// once per block instead of every row-thread re-reading the same 144 bytes
// from device. Each of the 256 threads owns exactly one element of each
// block for all 16 rows, so the per-block dequant arithmetic is exactly the
// per-element mapping of `dequantize_q4_k` (sub-block `p = e/32`, l = e%32
// always on the sc1 side: `get_scale_min_k4(p)`, nibble = qs[e/2] bit
// (e%2)*4) — identical values, different summation order per row.
kernel void tqf_q4k_gemv_staged16(
    device const uchar* weights [[buffer(0)]],
    device const float* vec [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup uchar w_stage[16 * 144];
    threadgroup float v_stage[256];
    threadgroup float sums[16 * 256];
    uint blocks_per_row = cols / 256;
    uint row_base = gid * 16;
    uint row_count = min(16u, rows - row_base);
    if (row_base >= rows) return;

    device const uchar* strip_base = weights + (ulong)row_base * (ulong)blocks_per_row * 144;
    float acc[16];
    for (uint r = 0; r < 16; ++r) {
        acc[r] = 0.0f;
    }
    for (uint b = 0; b < blocks_per_row; ++b) {
        for (uint i = tid; i < 16 * 144; i += 256) {
            uint r = i / 144;
            if (r < row_count) {
                w_stage[i] = strip_base[(ulong)r * (ulong)blocks_per_row * 144 + (ulong)b * 144 + (i % 144)];
            }
        }
        if (tid < 256) {
            v_stage[tid] = vec[(ulong)b * 256 + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            uint p = tid / 32;
            // Pairs (2g, 2g+1) share qs bytes [32g, 32g+32): the even pair
            // owns the low nibble, the odd pair the high nibble.
            uint nib_byte = (p / 2) * 32 + (tid % 32);
            uint nib_shift = 4 * (p % 2);
            float v = v_stage[tid];
            for (uint r = 0; r < row_count; ++r) {
                threadgroup const uchar* w = w_stage + r * 144;
                half d_h = as_type<half>(ushort(ushort(w[0]) | (ushort(w[1]) << 8)));
                half dmin_h = as_type<half>(ushort(ushort(w[2]) | (ushort(w[3]) << 8)));
                // scale/min pair for sub-block p from the 12 staged bytes
                // (identical branch structure to get_scale_min_k4).
                uchar sc, mm;
                if (p < 4) {
                    sc = w[4 + p] & 63;
                    mm = w[4 + p + 4] & 63;
                } else {
                    sc = (w[4 + p + 4] & 0x0F) | ((w[4 + p - 4] >> 6) << 4);
                    mm = (w[4 + p + 4] >> 4) | ((w[4 + p] >> 6) << 4);
                }
                uint nib = (w[16 + nib_byte] >> nib_shift) & 0xF;
                float value = float(d_h) * float(sc) * float(nib) - float(dmin_h) * float(mm);
                acc[r] += v * value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Cross-thread reduction: each thread owns element `tid`'s partial
    // contribution to every row's dot product; sum the 256 per-row partials
    // in threadgroup memory (tree reduce).
    for (uint r = 0; r < row_count; ++r) {
        sums[r * 256 + tid] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        for (uint r = 0; r < row_count; ++r) {
            if (tid < stride) {
                sums[r * 256 + tid] += sums[r * 256 + tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint r = 0; r < row_count; ++r) {
            out[row_base + r] = sums[r * 256];
        }
    }
}

// Batched GEMV / prefill primitive: out[token, row] = dot(dequant(weights[row,:]), mat[token,:]).
// `mat` is `tokens` rows of `cols` f32, row-major; `out` is `tokens` rows of
// `rows` f32, row-major.
kernel void tqf_q4k_gemm(
    device const uchar* weights [[buffer(0)]],
    device const float* mat [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.x;
    uint token = gid.y;
    if (row >= rows || token >= tokens) return;
    uint blocks_per_row = cols / 256;
    device const uchar* row_base = weights + (ulong)row * (ulong)blocks_per_row * 144;
    device const float* vec = mat + (ulong)token * (ulong)cols;
    float acc = 0.0f;
    thread float dequant[256];
    for (uint b = 0; b < blocks_per_row; ++b) {
        tqf_q4k_dequant_block(row_base + (ulong)b * 144, dequant);
        device const float* vblock = vec + (ulong)b * 256;
        for (uint i = 0; i < 256; ++i) {
            acc += dequant[i] * vblock[i];
        }
    }
    out[(ulong)token * (ulong)rows + row] = acc;
}

// out[row,:] = x[row,:] * rsqrt(mean(x[row,:]^2) + eps) * weight[:]. One
// 256-thread threadgroup per row.
kernel void tqf_rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float shared_sq[256];
    device const float* row_ptr = x + (ulong)row * (ulong)cols;
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += 256) {
        float v = row_ptr[i];
        local_sum += v * v;
    }
    shared_sq[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq[tid] += shared_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_sq = shared_sq[0] / float(cols);
    float inv_rms = rsqrt(mean_sq + eps);
    device float* out_row = out + (ulong)row * (ulong)cols;
    for (uint i = tid; i < cols; i += 256) {
        out_row[i] = row_ptr[i] * inv_rms * weight[i];
    }
}

kernel void tqf_residual_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] + b[id];
}

kernel void tqf_silu(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float v = x[id];
    out[id] = v / (1.0f + exp(-v));
}

kernel void tqf_sigmoid(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float v = x[id];
    out[id] = 1.0f / (1.0f + exp(-v));
}
"#;

pub fn load_reference_kernel_library(ctx: &MetalContext) -> Result<Library> {
    let options = CompileOptions::new();
    ctx.device()
        .new_library_with_source(REFERENCE_KERNELS_MSL_SOURCE, &options)
        .map_err(|e| {
            BackendError::Gpu(format!(
                "failed to compile reference kernel MSL source: {e}"
            ))
            .into()
        })
}

fn f32_slice_to_bytes(data: &[f32]) -> &[u8] {
    // Safety: read-only reinterpretation of `&[f32]` as `&[u8]` for the
    // duration of this borrow — `f32` has no padding.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn blocks_per_row(cols: usize) -> usize {
    assert_eq!(
        cols % Q4_K_BLOCK_ELEMENTS,
        0,
        "cols {cols} must be a multiple of the Q4_K block size {Q4_K_BLOCK_ELEMENTS}"
    );
    cols / Q4_K_BLOCK_ELEMENTS
}

fn assert_q4k_weights_shape(weights: &[u8], rows: usize, cols: usize) {
    let block_bytes = GgmlType::Q4K.block_bytes() as usize;
    let expected = rows * blocks_per_row(cols) * block_bytes;
    assert_eq!(
        weights.len(),
        expected,
        "weights length {} does not match expected {expected} bytes for a {rows}x{cols} Q4_K matrix",
        weights.len()
    );
}

/// `weights` is `rows` rows of `cols/256` contiguous Q4_K blocks, row-major
/// (the `.tqf`-passthrough on-disk layout for a `TQF_QUANT_PASSTHROUGH_Q4_K`
/// extent). Returns `rows` f32 dot-product results.
pub fn q4k_gemv(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    weights: &[u8],
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    assert_q4k_weights_shape(weights, rows, cols);
    let weights_buf = ctx.allocate_buffer_with_data(weights, "q4k-gemv-weights");
    dispatch_q4k_gemv_buffer(
        ctx,
        library,
        pipelines,
        &weights_buf,
        vector,
        rows,
        cols,
        Q4K_GEMV_FUNCTION,
        false,
    )
}

/// Same computation as `q4k_gemv`, but against a weights buffer the caller
/// already uploaded (e.g. once, at expert-cache admission time) instead of
/// re-uploading `rows*cols` Q4_K bytes on every call. This is the primitive
/// Phase 20's GPU-resident expert path is built on: NVMAI-style throughput
/// gains are only real once the weight upload is amortized across many
/// decode steps rather than paid per matvec.
pub fn q4k_gemv_persistent_weights(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    weights_buf: &BufferLease,
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let block_bytes = GgmlType::Q4K.block_bytes() as usize;
    let expected = (rows * blocks_per_row(cols) * block_bytes) as u64;
    assert_eq!(
        weights_buf.length(),
        expected,
        "persistent weights buffer is {} bytes, expected {expected} for a {rows}x{cols} Q4_K matrix",
        weights_buf.length()
    );
    dispatch_q4k_gemv_buffer(
        ctx,
        library,
        pipelines,
        weights_buf,
        vector,
        rows,
        cols,
        Q4K_GEMV_FUNCTION,
        false,
    )
}

/// Phase 20 NVMAI-derived staged variant of `q4k_gemv_persistent_weights`
/// (spec §292, "MoE phase-1 16-row threadgroup staging"): same contract and
/// buffers, different kernel. Json parity is asserted functionally (the
/// per-element dequant is exactly `dequantize_q4_k`'s), and the two kernels
/// stay selectable for benchmark A/B (isolated microbenchmarks first; full
/// decode A/B decides the eventual default, per spec §1005).
pub fn q4k_gemv_persistent_weights_staged16(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    weights_buf: &BufferLease,
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let block_bytes = GgmlType::Q4K.block_bytes() as usize;
    let expected = (rows * blocks_per_row(cols) * block_bytes) as u64;
    assert_eq!(
        weights_buf.length(),
        expected,
        "persistent weights buffer is {} bytes, expected {expected} for a {rows}x{cols} Q4_K matrix",
        weights_buf.length()
    );
    dispatch_q4k_gemv_buffer(
        ctx,
        library,
        pipelines,
        weights_buf,
        vector,
        rows,
        cols,
        Q4K_GEMV_STAGED16_FUNCTION,
        true,
    )
}

fn dispatch_q4k_gemv_buffer(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    weights_buf: &BufferLease,
    vector: &[f32],
    rows: usize,
    cols: usize,
    function_name: &str,
    staged: bool,
) -> Result<Vec<f32>> {
    assert_eq!(
        vector.len(),
        cols,
        "vector length {} does not match cols {cols}",
        vector.len()
    );

    let vector_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(vector), "q4k-gemv-vector");
    let out_buf = ctx.allocate_buffer((rows * 4).max(4) as u64, "q4k-gemv-out");
    let cols_buf = ctx.allocate_buffer_with_data(&(cols as u32).to_le_bytes(), "q4k-gemv-cols");
    let rows_buf = ctx.allocate_buffer_with_data(&(rows as u32).to_le_bytes(), "q4k-gemv-rows");

    let pipeline = pipelines.get_or_compile(ctx.device(), library, function_name, "")?;
    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(vector_buf.metal_buffer()), 0);
    encoder.set_buffer(2, Some(out_buf.metal_buffer()), 0);
    encoder.set_buffer(3, Some(cols_buf.metal_buffer()), 0);
    encoder.set_buffer(4, Some(rows_buf.metal_buffer()), 0);
    let (threads_per_group, groups) = if staged {
        // One 256-thread threadgroup covers 16 rows; a partial last group is
        // guarded inside the kernel.
        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new((rows as u64).max(1).div_ceil(16), 1, 1);
        (threads_per_group, groups)
    } else {
        let threads_per_group = MTLSize::new(64, 1, 1);
        let groups = MTLSize::new((rows as u64).max(1).div_ceil(64), 1, 1);
        (threads_per_group, groups)
    };
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(bytes_to_f32_vec(out_buf.as_slice()))
}

/// `mat` is `tokens` rows of `cols` f32, row-major. Returns `tokens` rows of
/// `rows` f32, row-major (`out[token * rows + row]`) — the prefill-shaped
/// batched form of `q4k_gemv`.
pub fn q4k_gemm(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    weights: &[u8],
    mat: &[f32],
    tokens: usize,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    assert_q4k_weights_shape(weights, rows, cols);
    assert_eq!(
        mat.len(),
        tokens * cols,
        "mat length {} does not match tokens*cols {}",
        mat.len(),
        tokens * cols
    );

    let weights_buf = ctx.allocate_buffer_with_data(weights, "q4k-gemm-weights");
    let mat_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(mat), "q4k-gemm-mat");
    let out_buf = ctx.allocate_buffer((tokens * rows * 4).max(4) as u64, "q4k-gemm-out");
    let cols_buf = ctx.allocate_buffer_with_data(&(cols as u32).to_le_bytes(), "q4k-gemm-cols");
    let rows_buf = ctx.allocate_buffer_with_data(&(rows as u32).to_le_bytes(), "q4k-gemm-rows");
    let tokens_buf =
        ctx.allocate_buffer_with_data(&(tokens as u32).to_le_bytes(), "q4k-gemm-tokens");

    let pipeline = pipelines.get_or_compile(ctx.device(), library, Q4K_GEMM_FUNCTION, "")?;
    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(mat_buf.metal_buffer()), 0);
    encoder.set_buffer(2, Some(out_buf.metal_buffer()), 0);
    encoder.set_buffer(3, Some(cols_buf.metal_buffer()), 0);
    encoder.set_buffer(4, Some(rows_buf.metal_buffer()), 0);
    encoder.set_buffer(5, Some(tokens_buf.metal_buffer()), 0);
    let threads_per_group = MTLSize::new(32, 1, 1);
    let groups = MTLSize::new((rows as u64).max(1).div_ceil(32), tokens.max(1) as u64, 1);
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(bytes_to_f32_vec(out_buf.as_slice()))
}

/// `x` is `rows` rows of `cols` f32, row-major; `weight` is `cols` f32.
pub fn rmsnorm(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    x: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    assert_eq!(
        x.len(),
        rows * cols,
        "x length {} does not match rows*cols {}",
        x.len(),
        rows * cols
    );
    assert_eq!(
        weight.len(),
        cols,
        "weight length {} does not match cols {cols}",
        weight.len()
    );

    let x_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(x), "rmsnorm-x");
    let weight_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(weight), "rmsnorm-weight");
    let out_buf = ctx.allocate_buffer((rows * cols * 4).max(4) as u64, "rmsnorm-out");
    let cols_buf = ctx.allocate_buffer_with_data(&(cols as u32).to_le_bytes(), "rmsnorm-cols");
    let eps_buf = ctx.allocate_buffer_with_data(&eps.to_le_bytes(), "rmsnorm-eps");

    let pipeline = pipelines.get_or_compile(ctx.device(), library, RMSNORM_FUNCTION, "")?;
    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(x_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(weight_buf.metal_buffer()), 0);
    encoder.set_buffer(2, Some(out_buf.metal_buffer()), 0);
    encoder.set_buffer(3, Some(cols_buf.metal_buffer()), 0);
    encoder.set_buffer(4, Some(eps_buf.metal_buffer()), 0);
    let threads_per_group = MTLSize::new(RMSNORM_THREADGROUP_SIZE, 1, 1);
    let groups = MTLSize::new(rows.max(1) as u64, 1, 1);
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(bytes_to_f32_vec(out_buf.as_slice()))
}

fn dispatch_elementwise_binary(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    function_name: &str,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    assert_eq!(
        a.len(),
        b.len(),
        "elementwise operand lengths differ: {} vs {}",
        a.len(),
        b.len()
    );
    let n = a.len();
    let a_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(a), "elementwise-a");
    let b_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(b), "elementwise-b");
    let out_buf = ctx.allocate_buffer((n * 4).max(4) as u64, "elementwise-out");

    let pipeline = pipelines.get_or_compile(ctx.device(), library, function_name, "")?;
    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(b_buf.metal_buffer()), 0);
    encoder.set_buffer(2, Some(out_buf.metal_buffer()), 0);
    let threads_per_group = MTLSize::new(256, 1, 1);
    let groups = MTLSize::new((n as u64).max(1).div_ceil(256), 1, 1);
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(bytes_to_f32_vec(out_buf.as_slice()))
}

fn dispatch_elementwise_unary(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    function_name: &str,
    x: &[f32],
) -> Result<Vec<f32>> {
    let n = x.len();
    let x_buf = ctx.allocate_buffer_with_data(f32_slice_to_bytes(x), "elementwise-x");
    let out_buf = ctx.allocate_buffer((n * 4).max(4) as u64, "elementwise-out");

    let pipeline = pipelines.get_or_compile(ctx.device(), library, function_name, "")?;
    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(x_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(out_buf.metal_buffer()), 0);
    let threads_per_group = MTLSize::new(256, 1, 1);
    let groups = MTLSize::new((n as u64).max(1).div_ceil(256), 1, 1);
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(bytes_to_f32_vec(out_buf.as_slice()))
}

pub fn residual_add(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    dispatch_elementwise_binary(ctx, library, pipelines, RESIDUAL_ADD_FUNCTION, a, b)
}

pub fn silu(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    x: &[f32],
) -> Result<Vec<f32>> {
    dispatch_elementwise_unary(ctx, library, pipelines, SILU_FUNCTION, x)
}

pub fn sigmoid(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    x: &[f32],
) -> Result<Vec<f32>> {
    dispatch_elementwise_unary(ctx, library, pipelines, SIGMOID_FUNCTION, x)
}

/// Simple LM-head path (spec §283): the output head is one dense
/// `[vocab, hidden]` Q4_K matrix (spec §12), so this is `q4k_gemv` with the
/// vocab/hidden shape asserted rather than a distinct kernel family. Later
/// phases may fuse a max/top-k sampling path onto this (spec §51's LM-head
/// specialization row) without changing this function's contract.
pub fn lm_head_logits(
    ctx: &MetalContext,
    library: &Library,
    pipelines: &mut PipelineCache,
    weights: &[u8],
    hidden_state: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    q4k_gemv(
        ctx,
        library,
        pipelines,
        weights,
        hidden_state,
        vocab_size,
        hidden_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_or_skip() -> Option<MetalContext> {
        match MetalContext::init() {
            Ok(ctx) => Some(ctx),
            Err(_) => {
                eprintln!("skipping Metal test: no device available in this environment");
                None
            }
        }
    }

    /// Builds a synthetic Q4_K-encoded `[rows, cols]` weight matrix (real
    /// block bytes through the real block codec, not zeros — a simple LCG
    /// fills scales/nibbles so every sub-block scale/min/nibble path gets
    /// exercised) plus a pseudo-random f32 input vector.
    fn synthetic_q4_k_weights(rows: usize, cols: usize) -> Vec<u8> {
        let block_bytes = GgmlType::Q4K.block_bytes() as usize;
        let n_blocks = rows * (cols / Q4_K_BLOCK_ELEMENTS);
        let mut out = vec![0u8; n_blocks * block_bytes];
        let mut counter: u32 = 1;
        for block in out.chunks_exact_mut(block_bytes) {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
            for b in block[4..].iter_mut() {
                counter = counter.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *b = (counter >> 16) as u8;
            }
        }
        out
    }

    fn synthetic_weights_and_vector(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let weights = synthetic_q4_k_weights(rows, cols);
        let vector: Vec<f32> = (0..cols).map(|i| ((i % 11) as f32) * 0.3 - 1.5).collect();
        (weights, vector)
    }

    #[test]
    fn q4k_gemv_matches_cpu_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let (rows, cols) = (17, 512); // deliberately not a power of two, > 1 block/row
        let (weights, vector) = synthetic_weights_and_vector(rows, cols);

        let gpu = q4k_gemv(
            &ctx,
            &library,
            &mut pipelines,
            &weights,
            &vector,
            rows,
            cols,
        )
        .unwrap();
        let cpu = reference::q4k_gemv(&weights, &vector, rows, cols);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-2, "gpu={g} cpu={c}");
        }
    }

    #[test]
    fn staged16_one_hot_element_probe_matches_dequant_exactly() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let (rows, cols) = (1usize, 256usize);
        let (weights, _) = synthetic_weights_and_vector(rows, cols);
        let dequant = crate::format::quant::dequant::dequantize_q4_k(&weights);
        let weights_buf = ctx.allocate_buffer_with_data(&weights, "probe");

        for (k, expected) in dequant.iter().enumerate() {
            let mut vector = vec![0f32; 256];
            vector[k] = 1.0;
            let out = q4k_gemv_persistent_weights_staged16(
                &ctx,
                &library,
                &mut pipelines,
                &weights_buf,
                &vector,
                rows,
                cols,
            )
            .unwrap();
            assert!(
                (out[0] - expected).abs() < 1e-3,
                "element {k}: kernel sent {out:?} but one-hot dequant expects {expected}"
            );
        }
    }

    #[test]
    fn staged16_gemv_matches_cpu_reference_and_reference_kernel() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        // Deliberately awkward shapes: rows < 16 and rows % 16 != 0 both
        // exercise the partial-group guards, 3 blocks/row exercises the
        // multi-block barrier loop, and the last two are the real
        // Qwen3.6 routed-expert geometries (gate/up [512, 2048], down
        // [2048, 512]).
        for (rows, cols) in [
            (1, 256),
            (16, 256),
            (17, 512),
            (33, 768),
            (100, 1024),
            (512, 2048),
            (2048, 512),
        ] {
            let (weights, vector) = synthetic_weights_and_vector(rows, cols);
            let weights_buf = ctx.allocate_buffer_with_data(&weights, "staged16-weights");

            let staged = q4k_gemv_persistent_weights_staged16(
                &ctx,
                &library,
                &mut pipelines,
                &weights_buf,
                &vector,
                rows,
                cols,
            )
            .unwrap();
            let reference = q4k_gemv_persistent_weights(
                &ctx,
                &library,
                &mut pipelines,
                &weights_buf,
                &vector,
                rows,
                cols,
            )
            .unwrap();
            let cpu = reference::q4k_gemv(&weights, &vector, rows, cols);

            assert_eq!(staged.len(), cpu.len());
            for (s, c) in staged.iter().zip(&cpu) {
                // The staged kernel reduces partials in a tree over threads
                // rather than sequentially, so this is a relative-tolerance
                // check on ordinary FP non-associativity (the one-hot probe
                // above holds the per-element dequant exact).
                assert!(
                    (s - c).abs() < (1e-4 * c.abs()).max(1e-2),
                    "staged16={s} cpu={c} at {rows}x{cols}"
                );
            }
            for (s, r) in staged.iter().zip(&reference) {
                assert!(
                    (s - r).abs() < (1e-4 * r.abs()).max(1e-2),
                    "staged16={s} reference-kernel={r} at {rows}x{cols}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of the Q4_K block size")]
    fn q4k_gemv_rejects_misaligned_cols() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();
        let weights = vec![0u8; 1];
        let vector = vec![0f32; 100];
        let _ = q4k_gemv(&ctx, &library, &mut pipelines, &weights, &vector, 1, 100);
    }

    #[test]
    fn q4k_gemm_matches_cpu_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let (tokens, rows, cols) = (3, 9, 256);
        let (weights, _) = synthetic_weights_and_vector(rows, cols);
        let mat: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
            .collect();

        let gpu = q4k_gemm(
            &ctx,
            &library,
            &mut pipelines,
            &weights,
            &mat,
            tokens,
            rows,
            cols,
        )
        .unwrap();
        let cpu = reference::q4k_gemm(&weights, &mat, tokens, rows, cols);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-2, "gpu={g} cpu={c}");
        }
    }

    #[test]
    fn rmsnorm_matches_cpu_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let (rows, cols) = (5, 300); // > one 256-thread reduction pass
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32) * 0.1 - 0.8)
            .collect();
        let weight: Vec<f32> = (0..cols).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
        let eps = 1e-6;

        let gpu = rmsnorm(&ctx, &library, &mut pipelines, &x, &weight, rows, cols, eps).unwrap();
        let cpu = reference::rmsnorm(&x, &weight, rows, cols, eps);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-3, "gpu={g} cpu={c}");
        }
    }

    #[test]
    fn residual_add_matches_cpu_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..1000).map(|i| -(i as f32) * 0.25).collect();

        let gpu = residual_add(&ctx, &library, &mut pipelines, &a, &b).unwrap();
        let cpu = reference::residual_add(&a, &b);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn silu_matches_cpu_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let x: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) * 0.05).collect();
        let gpu = silu(&ctx, &library, &mut pipelines, &x).unwrap();
        let cpu = reference::silu(&x);
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-4, "gpu={g} cpu={c}");
        }
    }

    #[test]
    fn sigmoid_matches_cpu_reference() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let x: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) * 0.05).collect();
        let gpu = sigmoid(&ctx, &library, &mut pipelines, &x).unwrap();
        let cpu = reference::sigmoid(&x);
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-4, "gpu={g} cpu={c}");
        }
    }

    #[test]
    fn lm_head_logits_matches_gemv_over_vocab_shape() {
        let Some(ctx) = ctx_or_skip() else { return };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        // A tiny stand-in "vocab" (real VOCAB_SIZE=248,320 would make this
        // test slow); the point is exercising the shared code path and
        // shape contract, not the real geometry.
        let (vocab, hidden) = (23, 256);
        let (weights, hidden_state) = synthetic_weights_and_vector(vocab, hidden);

        let logits = lm_head_logits(
            &ctx,
            &library,
            &mut pipelines,
            &weights,
            &hidden_state,
            vocab,
            hidden,
        )
        .unwrap();
        let cpu = reference::q4k_gemv(&weights, &hidden_state, vocab, hidden);

        assert_eq!(logits.len(), vocab);
        for (g, c) in logits.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-2, "gpu={g} cpu={c}");
        }
    }
}
