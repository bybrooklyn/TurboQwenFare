//! Baseline metallib loading and runtime MSL compilation (spec §282 phase
//! 10, §51 "Metal shader packaging and specialization": "Ship a known-good
//! baseline metallib so the binary can start. Also embed MSL source... so
//! first-run tuning can compile M4-specific function-constant variants.").
//!
//! There is no build-time `xcrun metal`/`metallib` step in this crate yet
//! (spec §230 "macOS single-binary build architecture" is a later phase),
//! so `load_baseline_library` currently always takes the runtime-source
//! path. The precompiled-bytes path is still structured as the *first*
//! attempt — not dead code, but exactly where a future `build.rs` step
//! that embeds a compiled `.metallib` via `include_bytes!` plugs in
//! without changing this function's callers.

use metal_sys::{CompileOptions, Library};

use crate::error::{BackendError, Result};

use super::context::MetalContext;

/// Reference kernels for the synthetic bandwidth/GEMV harness (spec §282:
/// "Create synthetic bandwidth/GEMV executable mode under `tqf
/// optimize`"). Deliberately plain F32, not the real Q4 kernel family
/// (spec §283, phase 11) — this phase only needs to prove the device/
/// queue/library/pipeline/dispatch/timing plumbing works end to end.
pub const BASELINE_MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Copies `src` to `dst` one element per thread — a bandwidth microbenchmark:
// throughput = 2 * n_elements * sizeof(float) bytes moved / elapsed time.
kernel void tqf_bandwidth_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];
}

// Naive row-per-thread GEMV: out[row] = sum_k matrix[row, k] * vec[k].
// `matrix` is row-major [rows, cols]. No threadgroup staging, no fused
// dequantization — the reference Q4 kernel family (spec §146, phase 11)
// replaces this once it exists; this is only a plumbing microbenchmark.
kernel void tqf_naive_gemv_f32(
    device const float* matrix [[buffer(0)]],
    device const float* vec [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint base = row * cols;
    for (uint k = 0; k < cols; ++k) {
        acc += matrix[base + k] * vec[k];
    }
    out[row] = acc;
}
"#;

pub const BANDWIDTH_COPY_FUNCTION: &str = "tqf_bandwidth_copy";
pub const NAIVE_GEMV_F32_FUNCTION: &str = "tqf_naive_gemv_f32";

/// Precompiled `.metallib` bytes, embedded at build time when a future
/// `build.rs` step produces one. `None` today — see module docs.
const EMBEDDED_METALLIB: Option<&[u8]> = None;

pub fn load_baseline_library(ctx: &MetalContext) -> Result<Library> {
    if let Some(bytes) = EMBEDDED_METALLIB {
        return ctx.device().new_library_with_data(bytes).map_err(|e| {
            BackendError::Gpu(format!("failed to load embedded metallib: {e}")).into()
        });
    }

    let options = CompileOptions::new();
    ctx.device()
        .new_library_with_source(BASELINE_MSL_SOURCE, &options)
        .map_err(|e| {
            BackendError::Gpu(format!("failed to compile baseline MSL source: {e}")).into()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_library_compiles_and_exposes_both_functions() {
        let Ok(ctx) = MetalContext::init() else {
            eprintln!("skipping Metal test: no device available in this environment");
            return;
        };
        let library = load_baseline_library(&ctx).expect("baseline MSL source must compile");
        assert!(library.get_function(BANDWIDTH_COPY_FUNCTION, None).is_ok());
        assert!(library.get_function(NAIVE_GEMV_F32_FUNCTION, None).is_ok());
    }
}
