//! GPU-resident MoE expert weights (Phase 20, spec §112 row 20).
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
//! Wired into the live decode loop as an A/B-able path (`TQF_EXPERT_GPU_RESIDENT`,
//! `experts::WholeExpertLfuCache`; spec invariant #10): when a cache
//! admission happens with the flag on, the expert's Q4_K bytes are uploaded
//! once into these buffers and the CPU copy is dropped, so the same bytes
//! are charged to the memory broker exactly once ("sole backing store"
//! variant, spec §50's shared-buffer expert-slot shape). The cache holds the
//! shared execution state (`GpuExecutionState`: device, compiled kernel
//! library, pipeline cache) behind a mutex; `forward` runs against that
//! state.
//!
//! The GEMV dispatch is selectable via `TQF_EXPERT_GPU_KERNEL`
//! (`staged16`, the benchmark-selected default, or `reference` — the Phase
//! 20 NVMAI-derived 16-row threadgroup-staged kernel in
//! `backend::metal::kernels` vs the Phase 11 reference kernel). The staged
//! kernel is parity-tested against both the reference kernel and the CPU
//! dequant oracle, and the real-checkpoint A/B that selected it as default
//! is recorded in the Phase 20 qualification notes.

use metal_sys::Library;

use crate::error::Result;
use crate::ids::Bytes;
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};

use super::buffer::BufferLease;
use super::context::MetalContext;
use super::kernels::{
    load_reference_kernel_library, q4k_gemv_persistent_weights,
    q4k_gemv_persistent_weights_staged16, silu,
};
use super::pipeline::PipelineCache;

/// Which Phase 20 GEMV kernel the GPU-resident expert path dispatches.
/// Selected once per process from `TQF_EXPERT_GPU_KERNEL` (values:
/// `reference` or `staged16`). `staged16` is the default: the real-weight
/// canonical-checkpoint A/B measured a 2.07x per-forward wall-time win
/// (1.94 ms vs 4.01 ms, gate/up/down chain including readbacks) with
/// effectively exact parity against the CPU oracle, so the benchmark
/// selected it — the env switch keeps the reference kernel reachable for
/// regression A/B (spec §1005: microbenchmarks first, decode A/B before
/// flipping a default; the decode-loop A/B remains outstanding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKernelKind {
    Reference,
    Staged16,
}

impl GpuKernelKind {
    pub fn from_env() -> Self {
        match std::env::var("TQF_EXPERT_GPU_KERNEL").as_deref() {
            Ok("reference") => Self::Reference,
            _ => Self::Staged16,
        }
    }
}

/// The shared GPU execution state a `GpuResidentExpert::forward` needs:
/// device/queue, the compiled kernel library, and the per-process pipeline
/// cache. Created once (lazily, on first Phase 20 cache admission) and
/// shared process-wide through `experts::WholeExpertLfuCache` so decode
/// never re-compiles MSL or re-acquires a device.
pub struct GpuExecutionState {
    pub ctx: MetalContext,
    pub library: Library,
    pub pipelines: PipelineCache,
    pub kernel_kind: GpuKernelKind,
}

impl GpuExecutionState {
    pub fn init() -> Result<Self> {
        let ctx = MetalContext::init()?;
        let library = load_reference_kernel_library(&ctx)?;
        Ok(Self {
            ctx,
            library,
            pipelines: PipelineCache::new(),
            kernel_kind: GpuKernelKind::from_env(),
        })
    }
}

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

    /// Brokered bytes this expert occupies in GPU buffers — identical to
    /// the CPU payload it was uploaded from, so cache eviction accounting
    /// (`resident_bytes`, policy utility) is the same for both variants.
    pub fn stored_bytes(&self) -> Bytes {
        Bytes(self.gate.length() + self.up.length() + self.down.length())
    }

    /// SwiGLU-style expert forward — gate/up GEMV, `SiLU(gate) * up`, down
    /// GEMV — mirroring `experts::ResidentExpert::forward`'s math exactly,
    /// but against the persistent GPU buffers from `upload` instead of CPU
    /// Q4_K bytes. `state` is the shared execution state (its pipeline
    /// cache needs `&mut` for lazy kernel compilation). The GEMV kernel is
    /// selected by `state.kernel_kind` (see `GpuKernelKind`).
    pub fn forward(&self, state: &mut GpuExecutionState, input: &[f32]) -> Result<Vec<f32>> {
        let gate = Self::gemv_matrix(state, &self.gate, input, self.expert_width, self.hidden)?;
        let up = Self::gemv_matrix(state, &self.up, input, self.expert_width, self.hidden)?;
        let activated_gate = silu(&state.ctx, &state.library, &mut state.pipelines, &gate)?;
        let hidden: Vec<f32> = activated_gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| gate * up)
            .collect();
        Self::gemv_matrix(state, &self.down, &hidden, self.hidden, self.expert_width)
    }

    /// One Q4_K GEMV against a persistent weights buffer, dispatched on the
    /// kernel selected by `state.kernel_kind` (see `GpuKernelKind`).
    fn gemv_matrix(
        state: &mut GpuExecutionState,
        weights_buf: &BufferLease,
        vector: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        match state.kernel_kind {
            GpuKernelKind::Staged16 => q4k_gemv_persistent_weights_staged16(
                &state.ctx,
                &state.library,
                &mut state.pipelines,
                weights_buf,
                vector,
                rows,
                cols,
            ),
            GpuKernelKind::Reference => q4k_gemv_persistent_weights(
                &state.ctx,
                &state.library,
                &mut state.pipelines,
                weights_buf,
                vector,
                rows,
                cols,
            ),
        }
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

    fn state_or_skip() -> Option<GpuExecutionState> {
        match GpuExecutionState::init() {
            Ok(state) => Some(state),
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
        let Some(mut state) = state_or_skip() else {
            return;
        };

        let (expert_width, hidden) = (512, 256); // small multiples of the Q4_K block size
        let gate_w = synthetic_q4_k_weights(expert_width, hidden, 1);
        let up_w = synthetic_q4_k_weights(expert_width, hidden, 7);
        let down_w = synthetic_q4_k_weights(hidden, expert_width, 13);
        let input: Vec<f32> = (0..hidden).map(|i| ((i % 11) as f32) * 0.3 - 1.5).collect();

        let broker = MemoryBroker::new(Bytes(
            (gate_w.len() + up_w.len() + down_w.len()) as u64 + 4096,
        ));
        let gpu_expert = GpuResidentExpert::upload(
            &state.ctx,
            &broker,
            &gate_w,
            &up_w,
            &down_w,
            expert_width,
            hidden,
        )
        .unwrap();
        assert_eq!(
            broker.snapshot().reserved,
            Bytes((gate_w.len() + up_w.len() + down_w.len()) as u64)
        );

        let gpu = gpu_expert.forward(&mut state, &input).unwrap();
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
        let Some(mut state) = state_or_skip() else {
            return;
        };

        let (expert_width, hidden) = (256, 256);
        let gate_w = synthetic_q4_k_weights(expert_width, hidden, 2);
        let up_w = synthetic_q4_k_weights(expert_width, hidden, 3);
        let down_w = synthetic_q4_k_weights(hidden, expert_width, 5);
        let broker = MemoryBroker::new(Bytes(
            (gate_w.len() + up_w.len() + down_w.len()) as u64 + 4096,
        ));
        let gpu_expert = GpuResidentExpert::upload(
            &state.ctx,
            &broker,
            &gate_w,
            &up_w,
            &down_w,
            expert_width,
            hidden,
        )
        .unwrap();
        let reserved_after_upload = broker.snapshot().reserved;

        for token in 0..3 {
            let input: Vec<f32> = (0..hidden)
                .map(|i| ((i + token) % 7) as f32 * 0.2 - 0.5)
                .collect();
            let output = gpu_expert.forward(&mut state, &input).unwrap();
            assert_eq!(output.len(), expert_width);
            // Repeated forward calls must not grow the broker reservation:
            // the whole point of a persistent-weights buffer is that only
            // the small per-call vector/output buffers are re-allocated.
            assert_eq!(broker.snapshot().reserved, reserved_after_upload);
        }
    }

    #[test]
    fn stored_bytes_equals_the_uploaded_payload_size() {
        let Some(mut state) = state_or_skip() else {
            return;
        };
        let (expert_width, hidden) = (256, 256);
        let gate_w = synthetic_q4_k_weights(expert_width, hidden, 11);
        let up_w = synthetic_q4_k_weights(expert_width, hidden, 12);
        let down_w = synthetic_q4_k_weights(hidden, expert_width, 13);
        let broker = MemoryBroker::new(Bytes(
            (gate_w.len() + up_w.len() + down_w.len()) as u64 + 4096,
        ));
        let gpu_expert = GpuResidentExpert::upload(
            &state.ctx,
            &broker,
            &gate_w,
            &up_w,
            &down_w,
            expert_width,
            hidden,
        )
        .unwrap();
        assert_eq!(
            gpu_expert.stored_bytes(),
            Bytes((gate_w.len() + up_w.len() + down_w.len()) as u64)
        );
    }

    #[test]
    fn staged16_and_reference_kernels_agree_at_real_expert_shapes() {
        let Some(mut state) = state_or_skip() else {
            return;
        };

        // Real Qwen3.6 routed-expert geometry (LoadedQwen36Expert payload
        // sizes): gate/up [512, 2048] Q4_K and down [2048, 512] Q4_K.
        let (expert_width, hidden) = (512, 2048);
        let gate_w = synthetic_q4_k_weights(expert_width, hidden, 101);
        let up_w = synthetic_q4_k_weights(expert_width, hidden, 103);
        let down_w = synthetic_q4_k_weights(hidden, expert_width, 107);
        let input: Vec<f32> = (0..hidden).map(|i| ((i % 11) as f32) * 0.3 - 1.5).collect();
        let broker = MemoryBroker::new(Bytes(
            (gate_w.len() + up_w.len() + down_w.len()) as u64 + 4096,
        ));
        let gpu_expert = GpuResidentExpert::upload(
            &state.ctx,
            &broker,
            &gate_w,
            &up_w,
            &down_w,
            expert_width,
            hidden,
        )
        .unwrap();

        let mut results = Vec::new();
        for kind in [GpuKernelKind::Reference, GpuKernelKind::Staged16] {
            state.kernel_kind = kind;
            results.push((kind, gpu_expert.forward(&mut state, &input).unwrap()));
        }
        let cpu = cpu_expert_forward(&gate_w, &up_w, &down_w, &input, expert_width, hidden);
        // The down matrix is [hidden, expert_width], so a full forward
        // returns one value per down row = hidden.
        assert_eq!(cpu.len(), hidden);

        // Each kernel family must stay within a chained-forward tolerance of
        // the CPU oracle at real shapes. The chained level is deliberately
        // looser than the single-GEMV tolerances asserted in
        // `backend::metal::kernels` (which hold at these exact shapes):
        // synthetic weight data drives the down GEMV into heavy cancellation
        // (~99.8%: |dot| ~ 8e7 from 512 terms of ~1e8 each), and the
        // SiLU/product chain carries correlated summation-order differences
        // linearly instead of cancelling them. 5e-2 relative bounds the
        // observed 2.4e-2 worst case; real Q4_K weight/activation
        // distributions are far less adversarial, and the real-checkpoint
        // parity test is the meaningful quality gate.
        for (kind, gpu) in &results {
            for (g, c) in gpu.iter().zip(&cpu) {
                let relative = (g - c).abs() / c.abs().max(1.0);
                assert!(
                    relative < 5e-2,
                    "{kind:?} kernel vs CPU oracle at real expert shapes: gpu={g} cpu={c} relative={relative}"
                );
            }
        }
        let (_, reference) = &results[0];
        let (_, staged) = &results[1];
        for (r, s) in reference.iter().zip(staged) {
            let relative = (r - s).abs() / r.abs().max(1.0);
            assert!(
                relative < 5e-2,
                "kernel kinds disagree at real expert shapes: reference={r} staged16={s} relative={relative}"
            );
        }
    }
}
