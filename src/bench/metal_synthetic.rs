//! Synthetic Metal bandwidth/GEMV benchmark (spec §282, phase 10: "Create
//! synthetic bandwidth/GEMV executable mode under `tqf optimize`/developer
//! harness rather than a second binary."). Exercises real device/queue/
//! buffer/pipeline/dispatch/timing plumbing from `backend::metal` against
//! the reference kernels in `backend::metal::shaderlib` — not a real model
//! kernel (spec §283+ owns that); this is proof the Metal baseline
//! infrastructure runs compute end to end, plus a first throughput number
//! to sanity-check hardware against.

use std::time::Duration;

use metal_sys::MTLSize;

use crate::backend::metal::{shaderlib, GpuStopwatch, MetalContext, PipelineCache};
use crate::error::{BackendError, Result};

#[derive(Debug, Clone, Copy)]
pub struct BandwidthResult {
    pub elements: usize,
    pub elapsed: Duration,
    pub gigabytes_per_second: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct GemvResult {
    pub rows: usize,
    pub cols: usize,
    pub elapsed: Duration,
    pub gflops: f64,
}

#[derive(Debug, Clone)]
pub struct SyntheticReport {
    pub device_name: String,
    pub bandwidth: BandwidthResult,
    pub gemv: GemvResult,
}

const BANDWIDTH_ELEMENTS: usize = 16 * 1024 * 1024; // 64 MiB per buffer
const GEMV_ROWS: usize = 512;
const GEMV_COLS: usize = 2048; // matches Qwen3.6's hidden size (spec §117)

/// Runs both microbenchmarks once with fixed, small-but-nontrivial problem
/// sizes. A real tuning pass over multiple sizes/repeats/specializations
/// is spec §106's "Performance benchmark protocol" (later phases); this is
/// the developer-harness entry point phase 10's exit criteria names.
pub fn run_synthetic_bandwidth_gemv() -> Result<SyntheticReport> {
    let ctx = MetalContext::init()?;
    let library = shaderlib::load_baseline_library(&ctx)?;
    let mut pipelines = PipelineCache::new();

    let bandwidth = run_bandwidth_copy(&ctx, &library, &mut pipelines)?;
    let gemv = run_naive_gemv(&ctx, &library, &mut pipelines)?;

    Ok(SyntheticReport {
        device_name: ctx.device_name().to_string(),
        bandwidth,
        gemv,
    })
}

fn f32_to_bytes(data: &[f32]) -> &[u8] {
    // Safety: `f32` has no padding/alignment surprises relevant here — a
    // read-only reinterpretation of a `&[f32]` as `&[u8]` for the
    // duration of this borrow.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run_bandwidth_copy(
    ctx: &MetalContext,
    library: &metal_sys::Library,
    pipelines: &mut PipelineCache,
) -> Result<BandwidthResult> {
    let src_data: Vec<f32> = (0..BANDWIDTH_ELEMENTS).map(|i| i as f32).collect();
    let src = ctx.allocate_buffer_with_data(f32_to_bytes(&src_data), "bandwidth-src");
    let dst = ctx.allocate_buffer((BANDWIDTH_ELEMENTS * 4) as u64, "bandwidth-dst");

    let elapsed = {
        let pipeline = pipelines.get_or_compile(
            ctx.device(),
            library,
            shaderlib::BANDWIDTH_COPY_FUNCTION,
            "",
        )?;

        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), 0);
        encoder.set_buffer(1, Some(dst.metal_buffer()), 0);
        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new((BANDWIDTH_ELEMENTS as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(groups, threads_per_group);
        encoder.end_encoding();

        let stopwatch = GpuStopwatch::start();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        stopwatch.elapsed()
    };

    // A correctness check, not just a timing number: the copy must be
    // exact (spec §111 "definition of a valid optimization" applies the
    // same discipline even to a throwaway microbenchmark).
    let dst_data = bytes_to_f32(dst.as_slice());
    if dst_data != src_data {
        return Err(
            BackendError::Gpu("bandwidth_copy kernel produced incorrect output".into()).into(),
        );
    }

    let bytes_moved = 2 * BANDWIDTH_ELEMENTS * std::mem::size_of::<f32>(); // read + write
    let gigabytes_per_second = bytes_moved as f64 / elapsed.as_secs_f64() / 1e9;

    Ok(BandwidthResult {
        elements: BANDWIDTH_ELEMENTS,
        elapsed,
        gigabytes_per_second,
    })
}

fn run_naive_gemv(
    ctx: &MetalContext,
    library: &metal_sys::Library,
    pipelines: &mut PipelineCache,
) -> Result<GemvResult> {
    let matrix: Vec<f32> = (0..GEMV_ROWS * GEMV_COLS)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();
    let vector: Vec<f32> = (0..GEMV_COLS)
        .map(|i| ((i % 7) as f32) * 0.2 - 0.6)
        .collect();

    let matrix_buf = ctx.allocate_buffer_with_data(f32_to_bytes(&matrix), "gemv-matrix");
    let vector_buf = ctx.allocate_buffer_with_data(f32_to_bytes(&vector), "gemv-vector");
    let out_buf = ctx.allocate_buffer((GEMV_ROWS * 4) as u64, "gemv-out");
    let cols_buf =
        ctx.allocate_buffer_with_data(&(GEMV_COLS as u32).to_le_bytes(), "gemv-cols-const");

    let elapsed = {
        let pipeline = pipelines.get_or_compile(
            ctx.device(),
            library,
            shaderlib::NAIVE_GEMV_F32_FUNCTION,
            "",
        )?;

        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(matrix_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(vector_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(out_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(cols_buf.metal_buffer()), 0);
        let threads_per_group = MTLSize::new(64, 1, 1);
        let groups = MTLSize::new((GEMV_ROWS as u64).div_ceil(64), 1, 1);
        encoder.dispatch_thread_groups(groups, threads_per_group);
        encoder.end_encoding();

        let stopwatch = GpuStopwatch::start();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        stopwatch.elapsed()
    };

    let gpu_out = bytes_to_f32(out_buf.as_slice());
    for row in 0..GEMV_ROWS {
        let expected: f32 = matrix[row * GEMV_COLS..(row + 1) * GEMV_COLS]
            .iter()
            .zip(&vector)
            .map(|(a, b)| a * b)
            .sum();
        let actual = gpu_out[row];
        // Summation order (and possible FMA use) can differ between the
        // CPU reference and the GPU kernel, so this is a tolerance check,
        // not bit-exactness — unlike the lossless-repack validation in
        // `format::quant::validate`, this is a floating-point compute
        // kernel, not a lossless bit-for-bit repack.
        if (expected - actual).abs() > 1e-2 {
            return Err(BackendError::Gpu(format!(
                "naive_gemv_f32 row {row} mismatch: expected {expected}, got {actual}"
            ))
            .into());
        }
    }

    let elapsed_secs = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let flops = 2.0 * GEMV_ROWS as f64 * GEMV_COLS as f64; // one mul + one add per element
    let gflops = flops / elapsed_secs / 1e9;

    Ok(GemvResult {
        rows: GEMV_ROWS,
        cols: GEMV_COLS,
        elapsed,
        gflops,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_benchmark_runs_and_produces_plausible_numbers() {
        if MetalContext::init().is_err() {
            eprintln!("skipping Metal test: no device available in this environment");
            return;
        }
        // Device availability was just confirmed above, so any error from
        // here on is a genuine kernel/plumbing failure, not "no GPU" —
        // let it fail the test loudly via `unwrap()` rather than skip.
        let report = run_synthetic_bandwidth_gemv().unwrap();
        assert!(!report.device_name.is_empty());
        assert!(report.bandwidth.gigabytes_per_second > 0.0);
        assert!(report.gemv.gflops > 0.0);
        assert_eq!(report.gemv.rows, GEMV_ROWS);
        assert_eq!(report.gemv.cols, GEMV_COLS);
    }
}
