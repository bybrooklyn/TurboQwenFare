# Phase 20 qualification: GPU-resident experts and the staged16 GEMV kernel

Spec §292 ("Phase 20 — NVMAI-derived Metal optimization ports") — MoE phase-1
16-row threadgroup staging, with the GPU-resident expert path (the first
half of Phase 20, spec §112 row 20) now wired into the live decode loop.

## What changed

1. **GPU-resident experts in the live loop** (`experts::WholeExpertLfuCache`):
   cache entries are now `ExpertValue::{Cpu, Gpu}`. With
   `TQF_EXPERT_GPU_RESIDENT=1` (or `set_gpu_enabled`, default `auto`), a cache
   admission uploads the expert's gate/up/down Q4_K bytes into persistent,
   broker-registered Metal buffers once and drops the CPU copy — the same
   bytes are charged to the memory broker exactly once ("sole backing store",
   spec §50). The streaming decode site calls `forward_expert`, which binds a
   planned expert to whichever backing store the cache chose, so Metal types
   never leak out of the cache module.
2. **`tqf_q4k_gemv_staged16`** (`backend::metal::kernels`): 16 rows per
   256-thread threadgroup. The 144-byte Q4_K blocks are cooperatively staged
   into threadgroup memory once per block (instead of every row-thread
   re-reading them from device), the 256-element input-vector block is staged
   alongside, each thread dequantizes exactly one element per block per row
   using the exact per-element mapping of `get_scale_min_k4` (sub-block
   `p = e/32`, nibble `qs[(p/2)*32 + e%32]`), and a threadgroup tree
   reduction sums the 256 per-element partials into row dots.
3. **Kernel selection**: `GpuExecutionState::kernel_kind`, from
   `TQF_EXPERT_GPU_KERNEL` (`staged16` default, `reference` selectable). Per
   spec §1005 the reference kernel stays reachable for A/B.

## Parity evidence

- `staged16_one_hot_element_probe_matches_dequant_exactly`: one-hot vectors
  force the kernel to return individual dequantized elements; every one of
  the 256 elements matches `dequantize_q4_k` to 1e-3 on synthetic blocks.
- `staged16_gemv_matches_cpu_reference_and_reference_kernel`: staged16 vs
  CPU oracle **and** vs the reference kernel at `(1,256)`, `(16,256)`,
  `(17,512)`, `(33,768)`, `(100,1024)`, and the real expert geometries
  `(512,2048)` and `(2048,512)` — partial-group guards (rows % 16 != 0,
  rows < 16) and multi-block barrier loops all exercised. Tolerance
  `max(1e-4·|x|, 1e-2)`.
- `staged16_and_reference_kernels_agree_at_real_expert_shapes`: chained
  SwiGLU forward (three GEMVs) at the real `512×2048 / 2048×512` geometry,
  both kernels within 5e-2 relative of the CPU oracle chain and of each
  other. On adversarial synthetic data the down GEMV hits ~99.8% cancellation
  (|dot| ≈ 8e7 from 512 terms of ≈1e8), which carries correlated
  summation-order differences linearly through the SiLU/product chain;
  worst observed 2.4e-2 relative, bounded at 5e-2. Single-GEMV tolerances at
  those exact shapes are the tight ones (1e-4) and hold.

Two parity bugs were found and fixed while writing the staged kernel, both
caught by the one-hot probe rather than by dot-level tolerances: a dropped
`+16` qs-region offset in the nibble byte index, and a missing cross-thread
reduction (each thread had been writing its own per-element partial directly
to the output row). The one-hot probe stays in the suite as the exactness
regression test.

## Real-checkpoint A/B (isolated microbenchmark)

`gpu_resident_expert_matches_cpu_forward_on_canonical_weights`
(`#[ignore]`, `TQF_CANONICAL_TQF=~/.tqf/models/qwen3.6-35b-a3b.tqf`, layer 0
expert 7's real Q4_K payload, 10 forwards each, wall time including all
readbacks and the SiLU/product steps):

| kernel      | per-forward wall time | vs CPU oracle (chained, max relative) |
|-------------|-----------------------|---------------------------------------|
| reference   | 4.009 ms              | 0.000000                              |
| staged16    | 1.940 ms              | 0.000000                              |

**2.07x per-forward win, parity effectively exact on real weights.** The
real Q4_K distribution is far better conditioned than the adversarial
synthetic fixture — both kernels agree with the CPU oracle chain to the
printed 6-decimal precision at 2048 outputs. On this measured result the
GPU-resident path's kernel default is `staged16` (BENCHMARK-SELECTED within
the opt-in GPU path), with `TQF_EXPERT_GPU_KERNEL=reference` kept as the A/B
switch.

## What this does not claim

- The GPU-resident path itself remains opt-in (`TQF_EXPERT_GPU_RESIDENT`);
  the CPU reference path is still the shipping default. A decode-loop A/B
  (spec §1005: "microbenchmark wins do not automatically survive contention
  and overlap") is the remaining step before considering a default flip.
- Per-expert forward wall time is not decode throughput: a decode step runs
  up to 320 expert selections across 40 layers with scheduling/overlap
  behavior the isolated microbenchmark does not model.
- Function-constant shape specialization and GDN four-way projection fusion
  (the other Phase 20 ports) are still outstanding.
- Nothing here changes the GPU path's broker accounting: uploads register
  broker reservations before allocation, per-call vector/output buffers
  remain small transient allocations, and eviction releases the reservations.
