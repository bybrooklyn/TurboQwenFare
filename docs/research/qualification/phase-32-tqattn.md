# Phase 32: TQAttn selective page attention

Spec Phase 32 deliverable (spec §304, §62-63, §164-166; exit gate row 32:
"Long-context page budget produces measured speedup within quality
limit"). Triggered directly by Phase 31's measurement: full attention
cannot stay the default beyond 128K-256K on this reference implementation
(12x-50x over the 15 tok/s budget from attention compute alone), so per
spec §303's own rule, this phase was required next.

## What was built

`context::tqattn`, the spec §164 REFERENCE BASELINE selector — explicitly
**not** the self-indexing Key search encodings of §63/§167, which §300
says to attempt only after this baseline is qualified:

- **Search summary**: extended `SealedPage` (Phase 27) with a per-(kv_head,
  dim) raw min/max over every token in the page — computed for free during
  the existing per-page scale scan, stored as new `search_bytes` in the
  page header/blob (previously always zero), included in the content hash,
  round-tripped through `to_bytes`/`from_bytes` and `context::prefix`
  persistence. `TqkvPagedCache::bytes_for_tokens` was updated to reserve
  the extra bytes — this was caught as a real broker-under-reservation bug
  during development (the byte formula silently didn't grow when the
  struct did) and fixed before landing.
- **Quest-style bound** (§164): `page_bound` computes
  `sum_i q_i>=0 ? q_i*k_max_i : q_i*k_min_i` per query head, maximized
  over all 16 query heads mapped to their KV head (GQA-aware), over the
  same full 256-dim post-RoPE Key representation the real attention score
  itself uses.
- **Selector** (`select_pages`): always includes the recent window (in
  pages, §166) and any caller-specified protected pages, scores the
  remaining pages via the bound, takes the top `page_budget`.
- **Uncertainty expansion** (§165): two of the spec's six listed triggers
  are implemented and tested — a tight score gap at the selection
  boundary, and total selected tokens below a configured minimum — either
  one grows the budget by one page and re-evaluates. See "Status" below
  for the four not implemented.

## Measured evidence

**Selector correctness** (`context::tqattn::tests`, synthetic caches with
an engineered "standout" page):

- `selector_always_includes_the_recent_window_and_protected_pages`: exact
  page-set inclusion check.
- `selector_finds_a_standout_old_page_via_the_quest_bound`: a page well
  outside the recent window, with Keys aligned to the query, is found and
  selected even under a page budget that excludes most of the 20-page
  history — the core Quest-recall claim, verified against a known ground
  truth rather than assumed.
- `uncertainty_expansion_grows_the_budget_when_the_boundary_gap_is_tight`:
  two near-tied high-scoring pages just past a budget-of-1 boundary are
  both captured once the gap-margin trigger is armed, instead of
  arbitrarily keeping only one.

**Full-attention A/B** (`selective_attention_over_chosen_pages_is_faster_than_full_attention`,
16,384-token synthetic context, 64 pages, real wall-clock, real dot-product
work over `TqkvPagedCache`'s production `key`/`value` accessors):

```
phase32_ab full_tokens=16384 selected_tokens=1536 full_ns=2806966250 selective_ns=261887291
speedup=10.72x standout_page_selected=true full_score=5254524.500 selective_score_partial=5243722.000
```

**10.72x wall-clock speedup**, attending to 1,536 of 16,384 tokens (9.4%)
via a page budget of 6 (2 recent-window + 4 selected). The standout page
was correctly recalled, and the selective-sum score (5,243,722) recovered
99.8% of the full-attention score (5,254,524.5) on this fixture — the
overwhelming majority of the "important" signal here comes from the one
standout page the selector found, with the remaining near-zero-mean noise
pages contributing little either way.

## Status and remaining work

- **Not implemented**: 4 of §165's 6 uncertainty triggers — query-norm/
  summary-statistics-outside-calibration-distribution (needs a calibration
  pass this phase doesn't build), protected-pages-already-consume-most-of-
  budget, a developer "force full attention" switch for A/B (the module
  itself supports this trivially by calling full attention directly, as
  the benchmark above does; no dedicated config flag exists yet), and
  page-summary-quantization-saturation detection.
- **Not implemented**: §166's *dynamic* recent window — `recent_window_pages`
  is a fixed config value here, not driven by "current context length,
  attention-stage wall time, TQAttn selection quality proxy, memory
  pressure, current request type."
- **Not implemented**: §163's `ContextFlags` provenance bitflags —
  `protected_pages` is accepted as a raw page-index list, not derived from
  request-parser-tracked system/tool/pinned provenance. That plumbing is
  request-parser/session territory (closer to Phase 44's automatic-RAG
  context budgeting) rather than TQAttn's own selection mechanism.
- **Not wired into the live decode loop.** `FullAttentionLayer`/
  `Qwen36BoundedReferenceRuntime` still always attend over their entire
  history; `select_pages` exists as a real, tested, benchmarked standalone
  mechanism operating directly on `TqkvPagedCache`, matching Phase 27/28's
  precedent of landing a mechanism before wiring it into the fixed decode
  graph. Live wiring needs a page-index-range attention consumer
  (`attend_over_tokens` in the A/B test is the shape of it) plumbed into
  `FullAttentionLayer::decode_projected*` behind its own A/B switch.
- ≤1% quality impact is not measured against a real quality suite (none
  exists yet in this repo, tracked open since Phase 15). The 99.8%
  selective-vs-full score recovery above is a synthetic-fixture proxy, not
  a qualification result.
- Self-indexing Key search encodings (§63, §167) remain explicitly
  deferred per §300, now that this baseline exists to compare future
  candidates against.
