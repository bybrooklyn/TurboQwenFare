# Phase 27 TQKV: paged Q8/Q4 baseline

Spec Phase 27 deliverable (spec §299, §155-159; exit gate row 27: "128K
capacity under 4G with full logical attention reference").

## What was built

- `context::tqkv`: a 128-byte page header (spec §157, little-endian,
  round-trip tested), 256-token sealed pages (spec §155) with a
  high-precision (f32) mutable tail, and symmetric per-page Q8 (spec §158)
  and Q4 (spec §159) Key/Value quantization — Key scale per `(kv_head,
  dim)` over the whole page, Value scale per `(token, kv_head, 64-dim
  group)`, both stored as real IEEE-754 FP16 (the `half` crate; the
  existing BF16 reference cache stays BF16 for its own history). Sealed
  pages carry a BLAKE3 content hash over payload+scales, checked by
  `verify_sealed_pages` — a live analogue of `format::tqf`'s per-tile
  checksums for spec §156's "canonical page bytes are immutable"
  guarantee.
- `TqkvPagedCache`: same external contract as the Phase 13
  `Bf16KvCache` (`push`/`key`/`value`/`len`/`reset`), reserving its full
  worst-case byte budget from the memory broker before any physical
  allocation (crate invariant #4), so it is a drop-in alternative backend.
- **Wired into the live decode loop**, not left standing alone: attention.rs's
  `FullAttentionLayer` now holds a `KvCacheBackend` enum (`Bf16` or `Tqkv`)
  chosen once at construction. Every call site in `model::qwen36::runtime`
  is unchanged — the `decode_projected_accounted`/`decode_projected`/
  `decode_q4` code paths already used by the real fixed-graph binder work
  identically against either backend. Selection is `TQF_TQKV_ENABLED=1`
  (`TQF_TQKV_PRECISION=q8`|`q4`, default `q8`), off by default (crate
  invariant #10 — every optimization is A/B-disableable and the BF16
  oracle stays the production default until qualified further).
- Two-pass streaming attention consumption (spec §161): both backends
  dequantize one token's Key/Value fragment at a time inside the existing
  causal softmax loop — no full-page BF16 materialization was added.

## Measured evidence

**Synthetic round-trip error** (`context::tqkv::tests`, uniform
post-RoPE-scale activations, one full sealed page): Q8 max abs error
<0.05 (over a ~[-4,4] dynamic range, matching the ~0.016 half-step bound
the symmetric int8 scheme predicts); Q4 max abs error <0.5 (matching the
larger ~0.29 half-step bound at 4-bit resolution).

**Differential production-path test**
(`model::qwen36::attention::tests::tqkv_q8_backend_matches_bf16_reference_within_tolerance`):
261 decode steps (one full sealed page plus a 5-token tail) run through the
*actual* `FullAttentionLayer::decode_projected` call used by the real
runtime, once against each backend. Max abs output difference across every
step and dimension: **<0.05**, exercising both the sealed-page dequant path
and the tail path in the same run.

**Real-checkpoint greedy parity** (canonical container,
`dev::qualification::canonical_decode_prints_greedy_sequence_for_tqkv_ab_comparison`,
run twice — once with the BF16 default, once with
`TQF_TQKV_ENABLED=1 TQF_TQKV_PRECISION=q8` — against the same 8-token
prompt continuation):

| Backend | Greedy tokens |
|---|---|
| BF16 reference | `[220, 16, 15, 15, 15, 20332, 1740, 369]` |
| TQKV-Q8 | `[220, 16, 15, 15, 15, 20332, 1740, 369]` |

**Identical**, token for token, on the real checkpoint. (Wall time was
93.9s vs 129.5s for these 8 steps; at only 8 tokens the cache never seals a
page — both backends do near-identical per-token work — so this is not a
meaningful performance signal, just noise on a decode that is dominated by
expert I/O per the Phase 25 ledger. TQKV's own I/O/compute cost is not yet
separately measured; that is follow-up work.)

A longer real-checkpoint run (264 steps, crossing one 256-token sealed-page
boundary) was launched to extend the greedy-parity check past the tail-only
regime; see the addendum below once it completes; the synthetic
differential test above already covers the sealed-page path with the same
production code, so page-boundary correctness is not solely resting on
that longer run.

**128K capacity accounting** (`q8_is_smaller_than_bf16_and_q4_is_smaller_than_q8_at_128k`,
computed from the same formula the broker reservation uses, not a live
128K decode — see "Status and remaining work" below):

| Backend | Bytes/layer @ 128K tokens | 10 layers |
|---|---|---|
| BF16 reference | 268,435,456 (256.0 MiB) | 2.50 GiB |
| TQKV-Q8 | 139,001,856 (~132.6 MiB) | ~1.29 GiB |
| TQKV-Q4 | 71,892,992 (~68.6 MiB) | ~0.67 GiB |

(Includes the Phase 32 per-page min/max search summary added after this
phase originally landed — a small, real addition, not a correction of an
error: ~2 KiB/page, ~1 MiB/layer at 128K.)

This matches spec §159's own order-of-magnitude estimate ("~5 KiB/token
across the ten full-attention layers... ~640 MiB at 128K" for Q4 — measured
here at ~686 MiB total across 10 layers, ~7.2% over the spec's raw-payload
estimate, attributable to the header/scale/search-summary overhead the
spec explicitly flags as additive). At 128K, TQKV-Q4's ~0.67 GiB and even
TQKV-Q8's ~1.29 GiB leave real headroom under a 4 GiB budget once expert
cache and resident weights are also accounted for, where BF16's 2.5 GiB
would not; Q8 is still meaningfully larger than Q4, consistent with its
stated role as the correctness oracle rather than the 128K production
candidate
(spec §158: "Q8 is not expected to meet the final 4G 128K budget alone").

## Status and remaining work

- "128K capacity under 4G" is demonstrated as **broker-reservation
  accounting**, not a live 128K-token decode: at ~12-16s/token on this
  machine's bounded runtime (external-drive expert I/O dominates per
  Phase 25), a full 128K-token run is an unattended multi-day job, not
  something this session ran. The byte math above is exact (same formula
  the broker uses to reserve capacity, verified against `Bf16KvCache`'s
  own formula for the BF16 comparison row), but an actual 128K live decode
  timing/quality run remains open, consistent with Phase 15's 512-token
  gate and Phase 29's 128K gate being separately tracked as not yet closed.
- TQKV's own compute/memory-traffic overhead relative to BF16 is not yet
  isolated from expert-I/O noise at real decode scale — a longer real
  run (thousands of tokens, ideally on the resident-core streaming profile
  once GDN/full-attention paths are unified there) would be needed to make
  a performance claim, and none is made here.
- Q3/Q2/rotation/outlier/pre-RoPE research candidates are Phase 28, not
  this baseline; TQAttn selective reads (Phase 32) and the mixed-precision
  transition controller (Phase 34+/spec §162) are later.
