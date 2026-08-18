//! GPU-resident MoE expert weights (Phase 20 foundation, spec §112 row 20).
//!
//! Uploads one routed/shared expert's gate/up/down Q4_K matrices to
//! broker-registered Metal buffers once, then reuses those buffers across
//! every decode step the expert stays cache-resident for, instead of
//! re-uploading `gate`/`up`/`down` bytes on each matvec the way a naive
//! `kernels::q4k_gemv(weights: &[u8], ...)` call would. This is exactly the
//! gap NVMAI's R9/R11 findings identify (see
//! `docs/research/upstream-precedent.md`): a GPU MoE path only wins once
//! weight upload is amortized, not paid per token.
//!
//! This type is deliberately not yet wired into the live decode loop —
//! `experts::mod` still computes MoE on `backend::reference` CPU kernels,
//! and wiring this in would double-count resident memory (CPU bytes in
//! `LoadedQwen36Expert` plus a GPU copy here) against the same 4G budget
//! without an eviction design that accounts for both. It is a tested,
//! broker-safe primitive: the next step is either making it the sole
//! backing store for a cache entry (dropping the CPU `Vec<u8>` copy, using
//! `BufferLease::as_slice`/`write` for the unified-memory path) or teaching
//! the cache to charge the GPU copy against the same reservation as its CPU
//! counterpart.

use metal_sys::Library;

use crate::error::Result;
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};

use super::buffer::BufferLease;
use super::context::MetalContext;
use super::kernels::{q4k_gemv_persistent_weights, silu};
use super::pipeline::PipelineCache;

pub struct GpuResidentExpert {
    gate: BufferLease,
    up: BufferLease,
    down: BufferLease,
    expert_width: usize,
    hidden: usize,
}

impl GpuResidentExpert {
    /// Uploads `gate`/`up` (`[expert_width, hidden]` Q4_K, row-major) and
    /// `down` (`[hidden, expert_width]` Q4_K, row-major) once. Each buffer
    /// gets its own broker reservation before its physical Metal allocation
    /// (spec invariant #4), released when the returned value is dropped.
    pub fn upload(
        ctx: &MetalContext,
        broker: &MemoryBroker,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        expert_width: usize,
        hidden: usize,
    ) -> Result<Self> {
        let gate = ctx.allocate_broker_buffer_with_data(
            broker,
            MemoryOwner::ExpertPinned,
            MemoryClass::Elastic,
            gate,
            "gpu-expert-gate",
        )?;
        let up = ctx.allocate_broker_buffer_with_data(
            broker,
            MemoryOwner::ExpertPinned,
            MemoryClass::Elastic,
            up,
            "gpu-expert-up",
        )?;
        let down = ctx.allocate_broker_buffer_with_data(
            broker,
            MemoryOwner::ExpertPinned,
            MemoryClass::Elastic,
            down,
            "gpu-expert-down",
        )?;
        Ok(Self {
            gate,
            up,
            down,
            expert_width,
            hidden,
        })
    }

    /// SwiGLU-style expert forward — gate/up GEMV, `SiLU(gate) * up`, down
    /// GEMV — mirroring `experts::ResidentExpert::forward`'s math exactly,
    /// but against the persistent GPU buffers from `upload` instead of CPU
    /// Q4_K bytes.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        library: &Library,
        pipelines: &mut PipelineCache,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let gate = q4k_gemv_persistent_weights(
            ctx,
            library,
            pipelines,
            &self.gate,
            input,
            self.expert_width,
            self.hidden,
        )?;
        let up = q4k_gemv_persistent_weights(
            ctx,
            library,
            pipelines,
            &self.up,
            input,
            self.expert_width,
            self.hidden,
        )?;
        let activated_gate = silu(ctx, library, pipelines, &gate)?;
        let hidden: Vec<f32> = activated_gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| gate * up)
            .collect();
        q4k_gemv_persistent_weights(
            ctx,
            library,
            pipelines,
            &self.down,
            &hidden,
            self.hidden,
            self.expert_width,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::kernels::load_reference_kernel_library;
    use crate::backend::reference;
    use crate::format::quant::dequant::Q4_K_BLOCK_ELEMENTS;
    use crate::format::quant::GgmlType;
    use crate::ids::Bytes;

    fn ctx_or_skip() -> Option<MetalContext> {
        match MetalContext::init() {
            Ok(ctx) => Some(ctx),
            Err(_) => {
                eprintln!("skipping Metal test: no device available in this environment");
                None
            }
        }
    }

    /// Same synthetic-but-real Q4_K block generator as `kernels::tests` (not
    /// shared across files - each suite's fixture stays self-contained).
    fn synthetic_q4_k_weights(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let block_bytes = GgmlType::Q4K.block_bytes() as usize;
        let n_blocks = rows * (cols / Q4_K_BLOCK_ELEMENTS);
        let mut out = vec![0u8; n_blocks * block_bytes];
        let mut counter = seed;
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

    fn cpu_expert_forward(
        gate_w: &[u8],
        up_w: &[u8],
        down_w: &[u8],
        input: &[f32],
        expert_width: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let gate = reference::q4k_gemv(gate_w, input, expert_width, hidden);
        let up = reference::q4k_gemv(up_w, input, expert_width, hidden);
        let activated: Vec<f32> = reference::silu(&gate)
            .into_iter()
            .zip(up)
            .map(|(g, u)| g * u)
            .collect();
        reference::q4k_gemv(down_w, &activated, hidden, expert_width)
    }

    #[test]
    fn gpu_resident_expert_matches_cpu_reference_and_accounts_broker_bytes() {
        let Some(ctx) = ctx_or_skip() else {
            return;
        };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let (expert_width, hidden) = (512, 256); // small multiples of the Q4_K block size
        let gate_w = synthetic_q4_k_weights(expert_width, hidden, 1);
        let up_w = synthetic_q4_k_weights(expert_width, hidden, 7);
        let down_w = synthetic_q4_k_weights(hidden, expert_width, 13);
        let input: Vec<f32> = (0..hidden).map(|i| ((i % 11) as f32) * 0.3 - 1.5).collect();

        let broker = MemoryBroker::new(Bytes(
            (gate_w.len() + up_w.len() + down_w.len()) as u64 + 4096,
        ));
        let gpu_expert =
            GpuResidentExpert::upload(&ctx, &broker, &gate_w, &up_w, &down_w, expert_width, hidden)
                .unwrap();
        assert_eq!(
            broker.snapshot().reserved,
            Bytes((gate_w.len() + up_w.len() + down_w.len()) as u64)
        );

        let gpu = gpu_expert
            .forward(&ctx, &library, &mut pipelines, &input)
            .unwrap();
        let cpu = cpu_expert_forward(&gate_w, &up_w, &down_w, &input, expert_width, hidden);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(&cpu) {
            // Three chained GEMVs over synthetic (unnormalized) data reach
            // large magnitudes, where FP32 summation-order differences
            // between the GPU and CPU reduction produce a larger absolute
            // gap than the single-GEMV kernel tests use; the actual
            // agreement is still tight in relative terms.
            let relative = (g - c).abs() / c.abs().max(1.0);
            assert!(relative < 1e-3, "gpu={g} cpu={c} relative={relative}");
        }

        drop(gpu_expert);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn upload_reuses_buffers_across_multiple_forward_calls_without_reuploading() {
        let Some(ctx) = ctx_or_skip() else {
            return;
        };
        let library = load_reference_kernel_library(&ctx).unwrap();
        let mut pipelines = PipelineCache::new();

        let (expert_width, hidden) = (256, 256);
        let gate_w = synthetic_q4_k_weights(expert_width, hidden, 2);
        let up_w = synthetic_q4_k_weights(expert_width, hidden, 3);
        let down_w = synthetic_q4_k_weights(hidden, expert_width, 5);
        let broker = MemoryBroker::new(Bytes(
            (gate_w.len() + up_w.len() + down_w.len()) as u64 + 4096,
        ));
        let gpu_expert =
            GpuResidentExpert::upload(&ctx, &broker, &gate_w, &up_w, &down_w, expert_width, hidden)
                .unwrap();
        let reserved_after_upload = broker.snapshot().reserved;

        for token in 0..3 {
            let input: Vec<f32> = (0..hidden)
                .map(|i| ((i + token) % 7) as f32 * 0.2 - 0.5)
                .collect();
            let output = gpu_expert
                .forward(&ctx, &library, &mut pipelines, &input)
                .unwrap();
            assert_eq!(output.len(), expert_width);
            // Repeated forward calls must not grow the broker reservation:
            // the whole point of a persistent-weights buffer is that only
            // the small per-call vector/output buffers are re-allocated.
            assert_eq!(broker.snapshot().reserved, reserved_after_upload);
        }
    }
}
