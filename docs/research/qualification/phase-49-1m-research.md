# Phase 49: 1M research

Spec Phase 49 deliverable (spec §321). "Treat capacity and bandwidth
separately. Combine validated TQKV/TQAttn/backing/prefix techniques;
add novel methods only with full-attention/reference comparisons.
Maintain 8G profile initially if needed."

No new mechanism is introduced this phase — every technique combined
here (TQKV-Q4 quantized pages, TQAttn selective attention, the prefix
snapshot store) was already implemented and independently qualified in
Phases 27, 30, and 32. This phase's job is to measure them *together*
at the scale spec §4 actually names (~1,048,576 tokens), the same
"real construction + isolated real measurement" methodology Phases 29,
31, and 34 already established for 128K/256K/2G, extended one more
step to 1M.

## Capacity: does the state fit?

`context::tqkv::scaling_bench::full_context_state_reserved_bytes` (no
new code — same function Phase 34 used at 128K under 2 GiB) really
constructs the entire 40-layer context/recurrent-state footprint (30
GDN states + 10 TQKV-Q4 full-attention layers) at 1,048,576-token
capacity:

```
context_state_1m_q4_bytes=5,743,902,720  (5.349 GiB)
headroom_vs_8gib_gib=2.651
fits_4gib=false
```

**Fits comfortably inside 8 GiB with 2.65 GiB headroom for
weights/expert-cache; does not fit the original 4 GiB target.** This
is exactly the outcome spec §321 anticipates by name ("maintain 8G
profile initially if needed") — not a new finding, but a direct
confirmation that the already-measured TQKV-Q4 byte cost (Phase 27's
~0.67 GiB per 10 layers at 128K) scales linearly as expected: 128K to
1M is 8x the tokens, and 5.349 GiB is indeed close to 8x Phase 34's
128K-under-2GiB figure once GDN states (context-length-independent)
are subtracted out.

## Bandwidth: does the compute fit?

Two isolated real measurements, both release-mode
(`--release --ignored --nocapture`), both using real production code
(`FullAttentionLayer::decode_projected` for the "full step cost"
number, `TqkvPagedCache`/`context::tqattn::select_pages` for the
selective-vs-full A/B — the same functions, same math, as Phases 29/31
and 32 respectively):

**Full attention alone, one decode step, no selection, at 1M
(`context::tqkv::scaling_bench::attention_cost_at_one_million_tokens_
tqkv_q4`):**

```
backend=TQKV-Q4 context_tokens=1048576 seed_ms=4206 one_step_ms=80525
```

**80.5 seconds for a single attention step's compute alone** — before
any I/O or MoE work — confirms Phase 29/31's own trend line
(450ms at 128K, 3,331ms at 256K) continues exactly as expected:
full/unselected attention is unusable at 1M by roughly three orders of
magnitude against the 15 tok/s (66.7 ms) budget. This is not a new
architectural discovery; it is the load-bearing reason TQAttn exists
at all, confirmed at the scale that matters.

**TQAttn selective attention vs full, at 1M
(`context::tqattn::tqattn_selective_attention_scales_to_one_million_
tokens`):** builds one real 1,048,576-token `TqkvPagedCache` (4,096
sealed pages, one engineered "important" old page), runs the real
`select_pages` selector with its *default*, fixed, not-scaled-up
config (`recent_window_pages: 2, page_budget: 4` — 6 pages total
regardless of how many pages exist), and times genuine dot-product
attention scoring over the full history versus only the selected
pages:

```
one_layer_tqkv_q8_1m_reserved_bytes=1,104,674,816 (1.029 GiB)
full_tokens=1,048,576 selected_tokens=1,536 selected_fraction=0.001465
full_ms=38,880 selective_ms=59
speedup=649.32x
standout_page_selected=true
```

**649x speedup, attending to 0.15% of tokens, and the engineered
"important" old page is still correctly recalled** (the same Quest-
bound correctness property Phase 32 proved at 16,384 tokens holds
unchanged at 1M). The more interesting number here isn't the speedup
itself — it's the *selected fraction*: Phase 32 measured 9.4% at
16,384 tokens (64 pages); this run measures 0.15% at 1,048,576 tokens
(4,096 pages), using the exact same fixed page budget both times. That
shrinking-fraction-as-context-grows behavior is the entire point of a
fixed-compute-budget selector, and it is what makes 1M tractable at
all: the selective attention step's *raw scoring cost* (59ms) is
already inside the 66.7ms decode budget on its own, where full
attention's 38.88 *seconds* is roughly 583x over it.

## Prefix restore: not re-run, reasoned from the existing measurement

Phase 30 already measured a real 27ms restore versus 53.5s from-scratch
decode for an 8-token prefix (1,963x) — that number's mechanism
(`PrefixSnapshotStore` deserializes already-quantized TQKV page bytes
and GDN state directly, an I/O-bound `O(bytes)` operation, never an
`O(context)` or `O(context²)` attention recompute) does not change at
larger prefix lengths; only the byte count scales. Actually decoding a
real 1M-token prefix first, in order to snapshot and restore it, would
require paying exactly the "many hours to days" full-generation cost
Phase 25/29 already established as infeasible on this hardware — so
this phase reasons from the mechanism rather than manufacturing a new
multi-day run to re-confirm an unrelated (I/O-bound, not
compute-bound) axis. This is the same "real math instead of an
infeasible literal re-run" scope decision Phase 29 made for the 128K
gate itself.

## Combined picture

Neither ingredient alone gets to 1M on this reference implementation:
BF16 full attention fails on *capacity* (Phase 31 already showed BF16
needs ~5 GiB just for 256K, so ~20 GiB at 1M — not re-run here, an
already-known non-fit); TQKV-Q4 alone fixes capacity (5.349 GiB fits
8 GiB) but not *bandwidth* (80.5 s/step, still ~1,200x over budget).
Only TQKV-Q4 (capacity) combined with TQAttn-style selection
(bandwidth) closes both axes at once — capacity to 5.349 GiB under an
8 GiB profile, and the dominant per-step attention cost from 80.5 s
down to double-digit milliseconds. Neither is wired into the live
decode loop yet (both remain Phase 27/32's own honest status), so this
remains a measured *research* result per spec §321's own phase name,
not a production capability claim — expert-miss I/O (Phase 25's 78%-
of-decode finding) and the reference CPU compute path are still
unaddressed and would still dominate a literal live 1M decode on this
hardware.

## Status and remaining work

- Not wired into the live decode loop — same status as TQKV-Q4's
  full-attention backend selection (Phase 27) and TQAttn's selector
  (Phase 32) individually.
- The bandwidth measurement uses the same from-scratch dot-product
  scoring loop Phase 32's own A/B used for the "selective" side (not
  `FullAttentionLayer::decode_projected`, which always attends over
  its entire live history by design) — a faithful stand-in for the
  arithmetic a real TQAttn-integrated attention consumer would do per
  selected range, not the exact call site that would exist after
  wiring.
- Quality impact of selecting only 0.15% of tokens at 1M is not
  measured here (Phase 32's own qualification only checked score
  preservation at 16,384 tokens, 99.8%) — a real ≤1% combined-quality
  qualification at 1M context is future work, per spec §321's own
  requirement that "novel methods" get "full-attention/reference
  comparisons," which this phase provides only at the timing/capacity
  level, not yet at the output-quality level.
