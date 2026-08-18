# Phase 21 cache-policy selection: raw-a-128 route-trace replay

Spec Phase 21 deliverable ("global expert broker... 4G plan significantly
improves bytes/token vs simple baseline," §112 row 21, BENCHMARK-SELECTED).
This is the qualification record for the policy `WholeExpertLfuCache`
defaults to (`DEFAULT_CACHE_POLICY` in `src/experts/mod.rs`).

## Methodology

1. Re-ran the already-qualified `raw-a-128` fixture (128-token greedy decode
   against the pinned real Q4_K_M checkpoint, `docs/research/qualification/raw-a-128-tqf.json`)
   with `TQF_QUALIFICATION_ROUTE_TRACE` set, so `dev::qualification::qualify_oracle`
   recorded the *exact* router output (expert IDs + weights, real router
   logits from the real checkpoint, not synthetic) for all 40 layers at every
   one of the 128 decode steps - 5,120 route events, 2.3 MB of trace.
2. Replayed that trace offline through `experts::policy::replay_trace`
   (`experts::policy::tests::qualification_trace_replays_all_phase21_policy_candidates`)
   against three candidate policies (LRU, LFU, decayed-cost-aware,
   half-life 160 events) at four cache capacities (256/512/768/1024 MiB),
   all well inside the 4 GiB budget. This tool changes nothing about routing
   or computed results - it only simulates which bytes a given
   admission/eviction policy would keep resident, so it is safe to run
   without re-touching the real checkpoint.
3. Cache policy has no effect on correctness (routing/weights/output are
   identical under every policy - only I/O volume differs), so this does not
   invalidate the existing `raw-a-16-tqf.json`/`raw-a-128-tqf.json`
   correctness qualifications, which ran under whatever the cache defaulted
   to at the time.

## Results

| Policy | 256 MiB | 512 MiB | 768 MiB | 1024 MiB |
|---|---|---|---|---|
| LRU | 0 hits / 72.5 GB miss | 0 hits / 72.5 GB miss | **17,058 hits / 42.3 GB miss** | **20,666 hits / 35.9 GB miss** |
| LFU | 0 hits / 72.5 GB miss | 0 hits / 72.5 GB miss | 7,077 hits / 60.0 GB miss | 9,231 hits / 56.1 GB miss |
| Decayed-cost-aware | 0 hits / 72.5 GB miss | 0 hits / 72.5 GB miss | 13,910 hits / 47.9 GB miss | 19,766 hits / 37.5 GB miss |

(`raw_miss_bytes` totals over the same 5,120-route-event, 128-token run;
`hits`/`misses` sum to 40,960 route-events at every capacity.)

## Findings

- **Below ~768 MiB, no policy gets any reuse at all** (0 hits everywhere):
  the trace's expert working set per route doesn't overlap enough between
  consecutive tokens to survive at 256/512 MiB regardless of eviction order.
  The Phase 15-18 default expert-cache capacity
  (`DEFAULT_EXPERT_CACHE_BYTES` in `dev::qualification`, 256 MiB) sits
  exactly in this dead zone - it was never large enough to benefit from
  *any* cache policy, which is presumably why the LFU placeholder default
  was never revisited before now. Right-sizing this capacity within the 4G
  budget is a Phase 24/25 concern (hard broker enforcement, then the M4
  throughput floor), not resolved by this record.
- **LRU wins clearly once capacity clears that floor.** At 1024 MiB, LRU's
  35.9 GB raw-miss total is a 36% reduction versus LFU's 56.1 GB over the
  identical trace - a "significant bytes/token improvement," the Phase 21
  exit-gate language, with real router data rather than a synthetic
  fixture. Decayed-cost-aware trails LRU by a few percent at every capacity
  tested but comfortably beats LFU.
- This is consistent with short-horizon temporal locality in consecutive
  decode steps' routing (recently-used experts recur more than
  globally-frequent ones do) rather than a small fixed set of "hot"
  experts recurring across the whole sequence, which is the pattern LFU is
  suited to instead.

## Decision

`WholeExpertLfuCache`'s default policy (`DEFAULT_CACHE_POLICY`,
`src/experts/mod.rs`) is now `CachePolicyKind::Lru`, replacing the
unmeasured LFU placeholder default from Phase 15-18. LFU and
decayed-cost-aware remain selectable via `TQF_EXPERT_CACHE_POLICY`/
`WholeExpertLfuCache::set_policy` for further A/B (spec invariant #10). The
type keeps its historical name (`WholeExpertLfuCache`) since renaming is a
pure churn cost with no correctness/behavior effect and this record is the
authoritative source for "what the default actually is."

## What this does not close

- The expert-cache *capacity* itself (256 MiB in the qualification harness)
  was not re-tuned here - only the policy at a fixed set of candidate
  capacities. A follow-up should pick a production capacity from this same
  data (768 MiB-1 GiB is the first region with any signal) once it's
  weighed against the rest of the 4G budget.
- This replay is a single 128-token/one-fixture sample. A broader workload
  matrix (Phase 15's own still-open exit gate) would strengthen confidence
  that LRU generalizes past this one prompt/continuation shape.
- Parallel I/O (Phase 19) and this cache-policy change are orthogonal and
  both now default-on; their combined effect has not been separately
  measured end-to-end (only I/O fan-out has its own pending real-checkpoint
  benchmark, see `upstream-precedent.md`'s R9 checklist entry).
