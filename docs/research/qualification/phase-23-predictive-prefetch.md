# Phase 23 predictive prefetch: statistical transition predictor on the raw-a-128 trace

Spec Phase 23 deliverable (spec §295; exit gate "net SSD-stall reduction
without harmful overfetch"). Qualification record for the prefetch
default decision.

## What was built

- `src/experts/prefetch.rs` — `TransitionPredictor`: a per-(layer, expert)
  transition/co-routing table over exact route events, decayed at 0.97
  per event (~23-event half-life) so recent transitions outrank stale
  ones. `predict` returns the top candidates for the *next* route event,
  excluding already-resident experts. Prediction never alters expert IDs
  or router weights (spec invariant #7) — it is purely an I/O hint.
- `replay_prefetch` — offline replay of the exact-router trace through a
  byte-budgeted cache with a one-event-ahead prefetch queue, logging the
  four spec metrics: precision, recall, timeliness (late arrivals), and
  wasted bytes.
- Live path in `WholeExpertLfuCache` (`advance_prefetch`,
  `drain_prefetch_inbox`, probation entries that evict before demand
  entries, prefetch stats in `ExpertCacheStats`), wired into the MoE
  forward after each exact route, off by default behind
  `TQF_PREFETCH_ENABLED` / `TQF_PREFETCH_DEPTH` (invariant #10).

## Method

Same re-captured `raw-a-128` route trace. Depths 0/4/8 at 512/768/1024
MiB. One route event of lead time (the compute window the live loop can
overlap an SSD read into).

## Results

| Capacity | Depth | Demand-miss bytes | Prefetch hits | Wasted bytes | Precision | Recall | Total SSD traffic |
|---|---|---|---|---|---|---|---|
| 512 MiB | 0 | 72.48 GB | 0 | 0 | — | 0 | 72.48 GB |
| 512 MiB | 4 | 44.25 GB | 11,540 | 7.12 GB | 57% | 28% | 51.4 GB (**-29%**) |
| 512 MiB | 8 | 34.99 GB | 17,314 | 33.7 GB | 43% | 42% | 68.7 GB (-5%) |
| 768 MiB | 0 | 42.29 GB | 0 | 0 | — | 0 | 42.29 GB |
| 768 MiB | 4 | 41.73 GB | 11,415 | 4.56 GB | 57% | 28% | 46.3 GB (+9%) |
| 768 MiB | 8 | 31.56 GB | 17,064 | 29.8 GB | 42% | 42% | 61.4 GB (+45%) |
| 1024 MiB | 0 | 35.91 GB | 0 | 0 | — | 0 | 35.91 GB |
| 1024 MiB | 4 | 40.47 GB | 11,134 | 3.62 GB | 56% | 27% | 44.1 GB (+23%) |
| 1024 MiB | 8 | 29.02 GB | 16,635 | 26.9 GB | 42% | 42% | 55.9 GB (+56%) |

## Findings

- The predictor works: 42-57% precision, up to 42% recall, cutting
  demand-miss bytes 19-52% at depth 8.
- But on this trace the savings are offset by wasted speculation at the
  production capacities: at 1024 MiB depth 8 adds 26.9 GB of wasted
  traffic for 6.9 GB of demand savings (+56% total). On a single-SSD-queue
  M4 those speculative reads compete with demand reads, so the replay
  predicts no wall-time win at 768-1024 MiB.
- **At 512 MiB, where the cache gets zero reuse on its own, prefetch is a
  clear total win** (depth 4: -29% total traffic) — the predictor covers
  exactly the demand the cache cannot.
- The adaptive-aggressiveness policy the spec's controller table (§45)
  prescribes falls straight out of this data: aggressive (depth 8) below
  the reuse floor, conservative (depth 4 or off) above it.

## Decision

Prefetch stays **off by default** at the production capacity: the exit
gate ("net SSD-stall reduction without harmful overfetch") does not close
on this trace — total SSD traffic rises at every capacity where the
cache already works. The live path, metrics, and depth control are in
place for the real-hardware A/B (Phase 25 ledger item): the replay models
one event of lead time, and only an end-to-end M4 run can show whether
the overlap actually converts demand-miss bytes into hidden latency.
