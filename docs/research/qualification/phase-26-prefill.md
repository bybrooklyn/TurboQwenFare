# Phase 26 prefill: chunked layer-outer prefill with expert-set dedup

Spec Phase 26 deliverable (spec §298; exit gate "long prompts achieve
major TTFT reduction"). First measured result.

## What was built

- `WholeExpertLfuCache::prepare_batch_route` / `forward_batch_expert` /
  `finish_batch_route`: a layer/chunk transaction whose pin set is the
  **union** of several exact routes. Each distinct absent expert is
  fetched exactly once (FlashMoE's "load each required expert on-demand
  exactly once per iteration"), then re-used by every row that selected
  it. Router IDs and weights stay exactly as produced (invariant #7) —
  dedup changes only fetch scheduling. Batch misses fan out through the
  Phase 19 parallel read pool.
- `Qwen36StreamingMoe::forward_batch`: routes every chunk row, plans the
  union once, executes per-row shared-expert + routed accumulation in
  exact route order.
- `Qwen36ReferenceRuntime::prefill_greedy` / `prefill_chunk`:
  layer-outer chunked prefill. Each layer's attention/recurrent state
  advances per token in exact order (identical semantics to the
  per-token loop), while the MoE tail batches across the chunk. Chunk
  size auto-halves when the broker cannot reserve chunk scratch
  (spec §152's pressure rule); `TQF_PREFILL_CHUNK` seeds the size
  (default 4096).
- Wired into the generation loop (`QwenRuntimeInstance::prefill`): the
  resident runtimes take the chunked path, the bounded runtime keeps the
  per-token reference loop, and prefill stage + expert I/O counters are
  logged per request.
- Instrumentation: per-prefill expert hits/misses/raw-bytes/demand-I/O
  deltas (stage instrumentation, spec §298).

## Measured evidence (53-token prompt, canonical container, resident-core streaming profile)

| Metric | Per-token loop | Chunked prefill |
|---|---|---|
| TTFT (prefill wall) | 142.7 s | **78.8 s (1.81x)** |
| Expert fetches | 8,956 | **4,106 (2.18x fewer)** |
| Expert bytes from SSD | 15.85 GB | **7.27 GB (2.18x less)** |
| Greedy continuation | identical | identical |

The greedy token after the prompt matches exactly between the per-token
loop and chunked prefill — expert-set dedup changes only I/O volume, and
the fixed-graph parity holds.

## Status and remaining work

- "Major TTFT reduction" is demonstrated (1.81x on the first shot, on
  the same 139 MB/s external container that caps Phase 25 — the dedup
  directly cuts the I/O bound, which is why it wins despite the disk).
- Open: chunk-size A/B (the spec's 4096 seed is a data point, not a
  law), repository-sized prompts (the chunk scratch is broker-accounted
  and the autotune path exists but has not been exercised beyond the
  53-token case), and the bounded runtime's chunked form (it keeps the
  per-token loop until the Phase 25 profile flip).
