# Phase 31: 256K and TQAttn trigger

Spec Phase 31 deliverable (spec §303; exit gate row 31: "256K usable
within 4G and ≤1%"). §303's literal instruction: "Measure full TQKV
attention at 256K. If ≥15 tok/s floor is maintained with acceptable
TTFT/memory, keep full evaluation default; otherwise proceed with TQAttn."

This phase is a decision gate, not a new mechanism — it extends Phase 29's
measurement to 256K and applies the spec's own stated rule.

## Measured evidence

**Memory** (`context::tqkv::scaling_bench::ten_full_attention_layers_at_256k_capacity_check`,
real broker construction of all ten full-attention layers, not formula):

```
phase31_memory tqkv_q4_256k_bytes=1427374080 gib=1.329 bf16_256k_bytes=5368709120 gib=5.000 bf16_fits_4gib=false
```

TQKV-Q4 at 256K: **1.33 GiB** (including Phase 32's later per-page search
summary), comfortably under a 4 GiB budget. BF16 at
256K: **5.00 GiB**, which *exceeds* a 4 GiB budget outright — BF16 cannot
even be constructed at 256K under the 4G profile, independent of speed.
This matches spec §65's own profile table exactly: "256K / 4G Production
extension: Attempt full; allow TQAttn if required for throughput."

**Performance** (`context::tqkv::scaling_bench::attention_cost_scales_with_populated_context_depth_toward_128k`,
extended through 262,144 tokens, release build, real single-attention-step
timing per Phase 29's isolation methodology):

| Context tokens | BF16 one step | TQKV-Q4 one step |
|---:|---:|---:|
| 131,072 | 404 ms | 1,448 ms |
| 262,144 | **822 ms** | **3,331 ms** |

Scaling is close to linear in context length (404→822 ms roughly doubles
from 128K→256K for BF16; 1,448→3,331 ms for TQKV-Q4), consistent with the
O(n) reference attention loop's expected asymptotics.

## Applying the spec's decision rule

Spec §303: **"If ≥15 tok/s floor is maintained... keep full evaluation
default; otherwise proceed with TQAttn."**

15 tok/s = 66.7 ms/token budget for the *entire* decode step. At 256K,
attention computation *alone* (before any MoE routing, projections, or
expert I/O) already costs:

- BF16: 822 ms — **12.3x over budget**
- TQKV-Q4: 3,331 ms — **49.9x over budget**

The floor is not maintained, not close, and the gap widens with context
length (the ratio was already 6.75x/25.3x at 128K, Phase 29). Per the
spec's own literal rule, **this triggers Phase 32: proceed with TQAttn.**
Full attention cannot be kept as the 256K-and-beyond default on this
reference implementation regardless of I/O or memory improvements — the
per-step compute itself is the blocker, and only bounding the number of
tokens actually attended to (TQAttn's page-selective read) can close that
gap; further TQKV precision/kernel work alone cannot, since it reduces
memory and bytes moved but does not reduce the O(n) score/softmax/weighted-
sum arithmetic itself.

## Status and remaining work

- ≤1% quality at 256K is not measured (no combined quality suite exists
  yet in this repo, tracked open since Phase 15/28/29).
- "256K usable within 4G" is TRUE for TQKV-Q4 (memory) but FALSE for
  throughput without TQAttn — the exit gate is only partially closed by
  this phase alone, exactly as intended: §303 frames 256K as the decision
  point for whether Phase 32 is needed, not as a phase that must itself
  deliver a working 256K decode.
- Phase 32 (TQAttn) is the direct, spec-mandated next step; this
  measurement is its architectural justification, not a suggestion.
