# Phase 25 M4 short-context assault: measured breakdown and the first three levers

Spec Phase 25 deliverable (spec §297; exit gate "15 is floor; retain
ledger of further headroom"). This record is the measured breakdown, the
three implemented levers, and the honest status of the floor. The floor
is **not closed**; the numbers below say exactly why.

## Levers implemented (all parity-preserving)

1. **Resident-core streaming profile** (`Qwen36ReferenceRuntime::
   open_streaming`, generator `open_resident_streaming`,
   `TQF_DEV_RESIDENT_STREAMING`). The canonical container's resident core
   is 2.13 GiB (measured from metadata: `canonical_tqf_resident_core_
   stored_bytes`); holding it resident removes the bounded runtime's
   per-token re-read of every tensor. This also fixed a stale resident
   path (`run_gdn` still used the naive rank-2 conv view; the Phase 16
   channel-major decoding now applies to both runtimes).
2. **NEON quantized dot kernels** (`src/simd/mod.rs`): Q4_K (experts),
   Q6_K (LM head), Q8_0 (GDN/attention projections), constructed to
   produce the **exact integer lane sums** of the scalar reference and
   to reuse the identical f32 combination, so the SIMD switch is
   bit-identical — differential fuzz tests (256 random blocks each)
   enforce it, and the 16-token raw-a-16 oracle still matches exactly.
   `TQF_SIMD_Q4K/Q6K/Q8_0=0` restores the scalar baseline (invariant
   #10).
3. **Activation-quantization hoisting**: the Q4_K/Q6_K/Q8_0 matvecs
   quantize the activation once per input chunk instead of once per
   weight row (bit-identical — the quantization is per-chunk), and the
   SIMD env decision is cached per process instead of read per block
   (248k env lookups/token on the LM head alone).

## Measured progression (16 greedy tokens, canonical container, CPU path)

| Step | Wall ms/token | Notes |
|---|---|---|
| Pre-session bounded baseline | 23,400 | 0.043 tok/s (recorded in canonical-source-manifest) |
| + resident core | 7,900 | 3x |
| + SIMD kernels | 5,800 | |
| + quant hoist + row kernels + env cache | 2,340 | **10x over baseline**, 0.43 tok/s |
| warm tokens (step 15) | 1,775 | 0.56 tok/s |

## Measured breakdown (warm decode)

| Component | ms/token |
|---|---|
| Demand expert I/O (instrumented in-cache) | **1,825** |
| Layers compute (attention + GDN + MoE, 40 layers) | ~400 |
| LM head (Q6_K, SIMD) | 112 |
| Total | ~2,340 |

Compute microbenchmarks: Q8_0 GEMV at Qwen scale (2048x8192) = 0.9 ms
(38 GFLOPS); the compute ceiling of the current single-threaded loop is
~0.5 s/token (~2 tok/s) before any parallelism.

## The I/O wall (root cause of the gap to 15 tok/s)

- Per-token expert demand is ~540 MB (8 experts x 40 layers x 1.69 MB);
  with the 1 GiB cache the real trace shows ~45-50% hit, so ~230-280
  MB/token must come from SSD.
- **The canonical container lives on an external flash drive reading at
  139 MB/s** (measured `dd`), so demand I/O alone costs ~1.8 s/token —
  78% of decode time. The same traffic on the base M4's internal SSD
  (~2.8 GB/s) would cost ~80 ms/token.
- Phase 22's tiling A/B showed admission granularity cannot shrink these
  bytes; Phase 23's prefetch replay showed aggressive speculation adds
  net traffic at working capacities. The bytes/token number is the
  binding constraint, and it is a cache-size + policy problem, not a
  kernel problem.

## Ledger of remaining headroom (in expected order of value)

1. **Parallel MoE compute**: 8 expert forwards per layer are serial on
   one thread; the M4 has 10 cores. Fanning the 8 GEMVs across a bounded
   worker pool (the Phase 19 read-fanout pattern, applied to compute)
   is the largest untouched lever (~5-8x on the 400 ms layer compute).
2. **Container placement on internal SSD**: the qualified M4 profile
   assumes internal-SSD streaming; on this machine that needs a 20 GB
   copy (user declined during this session; the numbers above document
   the 139 MB/s ceiling).
3. **Q6_K LM-head row loop** (112 ms): a whole-matrix head kernel with
   one call per 248k rows already exists at block level; the residual is
   per-row Rust overhead.
4. **Live prefetch A/B** (Phase 23): with demand reads at 139 MB/s the
   overlap window is large; the replay says try depth 4 and measure.
5. **Sustained/thermal behavior**: M4 passive cooling means the
   first-minute ranking may not hold (spec §53); the harness above is
   the instrument for that measurement.

## What this does not claim

15 tok/s is not reached and not claimed. The record is: 10x decode
speedup with exact token parity, a measured component breakdown, the
I/O-wall root cause with numbers, and the prioritized headroom list.
