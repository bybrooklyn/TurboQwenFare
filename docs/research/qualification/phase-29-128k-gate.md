# Phase 29: 128K production gate

Spec Phase 29 deliverable (spec §301; exit gate row 29: **"4G, 128K, ≤1%,
≥15 tok/s populated-context floor"**). "Populate a real 128K context before
timing decode. Run memory, quality and performance suites."

## Why this gate is not run as a single literal 128K live decode

Phase 25 already measured this machine's real-hardware decode floor:
1,775-2,340 ms/token even at trivial context length, dominated by expert
I/O on an external flash-drive checkpoint
(`docs/research/qualification/phase-25-m4-assault.md`). A literal
"populate 131,072 tokens, then time decode" run is ~2.3-8 hours just to
populate the context once (more, serially, to also time steps along the
way), which is not something this session ran end-to-end. This section
instead splits the gate into the three real, independently-measurable
questions it's actually asking, and answers each with real numbers on real
code rather than skip the phase.

## Memory suite: real, not just formula

Phase 27 computed 128K TQKV-Q4 capacity by formula. This phase actually
**constructs** all ten full-attention layers at 131,072-token capacity
under TQKV-Q4 inside one live `MemoryBroker`
(`context::tqkv::scaling_bench::all_ten_layers_at_128k_tqkv_q4_reserved_bytes`,
test `ten_full_attention_layers_at_128k_tqkv_q4_fit_under_a_4gib_broker`):

```
phase29_memory tqkv_q4_128k_all_layers_bytes=718929920 gib=0.670 headroom_gib=3.330
```

Real reservation, real broker, matches the Phase 27 formula exactly
(718,929,920 bytes, including Phase 32's later per-page search-summary
addition): **0.67 GiB for all ten layers' KV history at 128K**, leaving
3.33 GiB of a 4 GiB budget for resident core weights and the expert cache.
The memory half of "4G, 128K" is satisfied for TQKV-Q4.

## Performance suite: isolating attention cost from I/O

Phase 25's floor is I/O-dominated at *short* context. The open question
specific to 128K is whether attention computation itself — O(context
length) per step, independent of I/O — becomes its own bottleneck as
context grows. `FullAttentionLayer::seed_synthetic_history_for_benchmark`
populates a cache to a target depth without paying per-intermediate-step
attention cost, so a single real attention step (real RMSNorm/RoPE/causal
softmax/gate code, `decode_projected`) can be timed at otherwise-unreachable
depths (`context::tqkv::scaling_bench`, release build,
`attention_cost_scales_with_populated_context_depth_toward_128k`):

| Context tokens | BF16 one step | TQKV-Q4 one step |
|---:|---:|---:|
| 512 | 1 ms | 5 ms |
| 4,096 | 14 ms | 46 ms |
| 16,384 | 58 ms | 178 ms |
| 65,536 | 205 ms | 813 ms |
| 131,072 | **450 ms** | **1,687 ms** |

Two real findings:

1. **Attention alone blows the 15 tok/s (66.7 ms/token) budget at 128K,
   independent of I/O.** Even BF16's 450 ms for one attention step at 128K
   is 6.75x the *entire* per-token time budget, before any MoE routing,
   projections, or expert I/O. This confirms the spec's own design
   rationale for TQAttn (§62-63, Phase 31-32): naive full attention over a
   128K-token history is not just an I/O problem, it is an O(n) compute
   problem that selective/bounded attention is specifically meant to fix.
2. **TQKV-Q4 is 3.7x slower per attention step than BF16 at 128K** (1,687 ms
   vs 450 ms) — a real, measured cost of the current scalar per-token
   dequant path (unpacking 4-bit codes plus a per-dimension scale multiply,
   versus BF16's single bit-shift). This is a recorded negative result for
   raw attention throughput: TQKV-Q4 wins memory (128K fits in 0.67 GiB vs
   BF16's 2.5 GiB) but currently *loses* compute time at long context. A
   SIMD/fused dequant kernel (the Phase 20 NEON-kernel pattern applied to
   TQKV's Q4 unpack) is the natural next step and is not yet built.

## Quality suite

No combined ≤1% quality-degradation suite exists in this repo yet for
*any* phase (tracked open in `AGENTS.md`); Phase 29 cannot close this
independently. What does exist and was extended this phase: a real
264-step (>1 sealed TQKV page) BF16-vs-TQKV-Q8 differential run on the
canonical checkpoint (`dev::qualification::canonical_decode_prints_greedy_sequence_for_tqkv_ab_comparison`,
launched twice — see `phase-27-tqkv-baseline.md` for the 8-step result and
below for the 264-step one), extending Phase 27's synthetic differential
test's page-boundary coverage to real model activations.

## 264-step real-checkpoint run (crosses one TQKV page boundary)

<!-- filled in once the background run completes -->

## Status and remaining work

- **The literal ≥15 tok/s populated-128K-context floor is not met and, per
  the attention-scaling table above, cannot be met by the current
  reference attention implementation regardless of I/O** — this is a
  stronger, more specific version of the already-open Phase 25 finding.
  Closing it needs both faster I/O (Phase 25's remaining ledger) and
  bounded/selective attention (Phase 31-32's TQAttn), not incremental
  tuning of the current O(n) reference loop.
- A true end-to-end 128K live decode (real MoE routing + real I/O + real
  attention, all 131,072 steps) was not run; the memory and attention-cost
  halves of the gate were measured directly and honestly instead, since
  a full run's wall-clock cost is prohibitive on this hardware and would
  not answer the "is the reference attention loop itself adequate"
  question any more precisely than the isolated benchmark above already
  does.
- TQKV-Q4's attention slowdown at scale is new information Phase 27/28 did
  not surface (their longest real test was 264 tokens, far short of where
  the slowdown becomes material) — worth folding into Phase 31/32 design
  decisions about whether TQAttn's page-bounded reads also need to bound
  TQKV's own per-token dequant cost, not just page count.
