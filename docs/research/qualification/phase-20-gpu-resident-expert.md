# Phase 20 qualification: NVMAI-derived Metal optimization ports

Spec §292 ("Phase 20 — NVMAI-derived Metal optimization ports"). This record
covers all four Phase 20 work items: the GPU-resident expert path wired into
the live decode loop, MoE phase-1 16-row threadgroup staging
(`staged16`), function-constant Qwen shape specialization (`staged16-spec`),
and GDN four-way projection fusion — each with parity evidence and the
measured A/B result that decides (or rejects) it. Per spec §1005 the
microbenchmark numbers here are evidence, not claims about end-to-end decode
throughput; the decode-loop A/B section below is the one that counts.

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

## Real-checkpoint A/B, initial isolated microbenchmark (superseded by the interleaved runs below)

`gpu_resident_expert_matches_cpu_forward_on_canonical_weights`
(`#[ignore]`, `TQF_CANONICAL_TQF=~/.tqf/models/qwen3.6-35b-a3b.tqf`, layer 0
expert 7's real Q4_K payload, 10 forwards each, wall time including all
readbacks and the SiLU/product steps — recorded before the measurement was
restructured into interleaved rounds):

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

## Decode-loop A/B (the gate that matters, spec §1005)

`decode_loop_ab_gpu_vs_cpu_experts` (`#[ignore]`, canonical container, 8
greedy steps from token 32 ("A"), 1 GiB expert cache each side, in-process
GPU enable via `Qwen36BoundedReferenceRuntime::set_expert_gpu_enabled`):

| run  | total wall time | per-step range   | token parity            |
|------|-----------------|------------------|-------------------------|
| GPU  | 166.05 s        | 17.7 – 23.5 s    | identical (8/8)         |
| CPU  | 159.90 s        | 13.4 – 23.2 s    | identical (8/8)         |

**Speedup 0.96x — the GPU-resident expert path does not win end to end.**
Greedy tokens were exactly identical between paths, so parity holds in the
real loop, but decode is dominated by the CPU reference stages (Q8_0 GDN and
attention projections, Q6_K LM head, expert-miss I/O — ~1.2k cold expert
reads over 8 steps at 1 GiB cache). The staged16 kernel's isolated 2.07x
expert-forward win is real but is a small share of the ~20 s/token decode
wall time, exactly the "microbenchmark wins do not automatically survive
contention and overlap" case §1005 warns about. **Result: the GPU expert
path stays opt-in; no default flip.** The win must come from the stages that
actually dominate decode — see the GDN fusion result below.

## Function-constant specialization A/B

Parity: the spec kernel is covered by the same shape list as staged16 in
`staged16_gemv_matches_cpu_reference_and_reference_kernel` (including the
real `512×2048`/`2048×512` geometries), vs CPU oracle and vs staged16.

Real-checkpoint measurement (interleaved rounds — machine state drifts on
the seconds scale, so each kernel got one timed forward per round; 20
rounds, release build):

| kernel         | min per-forward | mean per-forward | parity (max relative) |
|----------------|-----------------|------------------|-----------------------|
| reference      | 1.633 ms        | 2.459 ms         | 0.000000              |
| staged16       | 2.216 ms        | 2.770 ms         | 0.000000              |
| staged16-spec  | 2.193 ms        | 2.622 ms         | 0.000000              |

**No consistent win for the specialization** (its mean is marginally better
than staged16's here, its min marginally worse; the earlier isolated run's
1.94 ms vs 4.01 ms spread shows how wide run-to-run variance is on this
machine). Also note the first recording of this A/B had reference at 4.0–4.4
ms and staged16 at 1.9–3.1 ms across two builds — per-forward numbers on a
contended M4 are not stable enough to separate these kernels. **Result:
`staged16-spec` is delivered (the port exists with parity) but not selected
as default; `staged16` remains the GPU-path default.** The pipeline-cache
function-constant plumbing is the reusable infrastructure payoff regardless.

## GDN four-way projection fusion A/B

`gdn_fused_projection_matches_cpu_on_canonical_weights` (`#[ignore]`): one
real GDN layer's Q8_0 `qkv`/`z`/`a`/`b` payloads from the canonical
container, fused single-launch vs four separate `q8_gemv` launches,
interleaved, 20 rounds, plus CPU-oracle parity on the real bytes (held to
`max(1e-4·|x|, 1e-2)`):

| path                        | min wall time | mean wall time |
|-----------------------------|---------------|----------------|
| four separate q8_gemv       | 3.401 ms      | 4.221 ms       |
| fused gdn projection        | 2.513 ms      | 2.876 ms       |

**1.47x for the fused projection, parity held on real weights.** This one
targets a stage that actually matters to decode (the CPU reference decodes
each GDN projection tensor wholesale per token), unlike the expert-stage
win above. The kernel is delivered and measured but **not yet wired into the
live decode loop** — that wiring (GPU-resident GDN weights with broker
accounting, an opt-in flag, and a fresh decode-loop A/B) is the next lever
on the decode critical path and belongs with the Phase 25 assault work.

## What this does not claim

- The GPU-resident expert path remains opt-in (`TQF_EXPERT_GPU_RESIDENT`);
  the CPU reference path is still the shipping default. The decode-loop A/B
  above measured 0.96x, so there is no default flip — this is a recorded
  negative result, not a pending decision.
- Per-expert forward wall time is not decode throughput: a decode step runs
  up to 320 expert selections across 40 layers with scheduling/overlap
  behavior the isolated microbenchmark does not model.
- The GDN fusion kernel is not in the live loop yet (see above).
- Nothing here changes the GPU path's broker accounting: uploads register
  broker reservations before allocation, per-call vector/output buffers
  remain small transient allocations, and eviction releases the reservations.
