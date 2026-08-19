# Phase 41: adaptive ANN research

Spec Phase 41 deliverable (spec §41, §89-90, §313; exit gate implicit in
§89: "A custom ANN feature survives only if it improves a defined
Pareto frontier of recall/latency/RAM/index size/update cost"). "Only
now implement custom semantic partitions. Baselines are already
available, preventing an unmeasured bespoke index."

## Scope decision: steps 1-2 of 5

Spec §313's candidate development sequence has five steps. Only the
first two are attempted:

1. **static balanced semantic partitions** — built and measured;
2. **repo-hierarchy overlay** — built and measured (path-derived, no
   AST needed);
3. hot/cold partition residency — needs a live cache with real
   query/memory pressure over time to mean anything; nothing to measure
   offline.
4. local split/merge after edits — needs a real incremental-update
   workload; Phase 42 (live sync) is where edits start happening at all.
5. workload-adaptive routing — needs real query traffic patterns to
   adapt *to*; a single offline benchmark run has no workload to learn
   from.

Steps 3-5 all require live, time-extended traffic this session's
offline, fixture-based measurement methodology cannot produce honestly
— building them now would mean writing untested logic with no real
signal to validate against, the same trap Phase 28's TQKV candidates
and Phase 39's TQVec candidates avoided by staying standalone/
unwired until qualified. They are deferred to whenever Phase 42's live
sync work gives them something real to adapt to.

## What was built

`retrieval::adaptive`:

- **`SemanticPartitionIndex`** — a deterministic (seeded, no external
  randomness) balanced k-means (Lloyd's algorithm) over Phase 38's
  L2-normalized FP32 vectors, using cosine similarity for both
  assignment and centroid convergence. `search(query, nprobe, k)` scans
  only the `nprobe` nearest-centroid partitions exactly, returning both
  the results and how many vectors were actually scanned — the
  measurable half of the Pareto tradeoff spec §89 requires before any
  custom ANN feature can be kept.
- **`HierarchyOverlay`** — spec §90's first three hierarchy levels
  (repository/module/file; type/function need real AST, out of scope
  per Phase 35/36) derived purely from path structure: the first path
  segment after `src/` is the "module." Exposes a same-module bonus a
  future hybrid-fusion step could add as spec §194's "active-file/
  module proximity bonus."

## Measured evidence

**Real corpus, real recall/scan-fraction measurement, honest result.**
Reusing Phase 38's committed real embeddings (no model load needed):

```
phase41_hierarchy modules={"context": 2, "experts": 2, "format": 1, "helper_model": 2, "memory": 1, "retrieval": 2}
phase41_partitions k=2 nprobe=1 mean_recall@5=0.70 mean_scan_fraction=0.45
phase41_partitions k=3 nprobe=1 mean_recall@5=0.60 mean_scan_fraction=0.35
```

The hierarchy overlay correctly groups the real ten-file corpus into
six real modules purely from path structure — no AST needed for this
much of spec §90's hierarchy.

**The partitioning result is a real negative finding, not a bug or a
disappointing accident — and it is exactly the outcome spec §89
predicts:** "flat search... can be surprisingly competitive for normal
repository sizes." At `k=2` partitions, restricting search to the
single nearest partition (`nprobe=1`) only avoids scanning 55% of the
corpus while *losing* 30 recall points (0.70 vs Phase 38's 1.0 for the
same real INT8/FP32 baselines); at `k=3` the scan savings improve to
65% but recall drops further to 0.60. Neither point is a Pareto
improvement over Phase 38's exact flat search at this corpus scale —
the whole reason the flat baseline had to exist first (spec §89's own
sequencing rule).

## Status and remaining work

- **This does not mean static partitioning is a dead end** — spec §89's
  own framing is that flat search wins "for normal repository sizes,"
  and ten documents is nowhere near a size where an ANN structure's
  fixed overhead (imbalanced partitions, coarse centroid quality) would
  be amortized. The honest conclusion this phase supports is narrower:
  *at this measured scale, static partitioning is not worth it* — not
  "static partitioning never helps." Revisiting at a real
  hundreds-to-thousands-of-document corpus is future work, same caveat
  Phase 38/39/40 all recorded about their own small real corpus.
- No hot/cold residency, split/merge, or workload-adaptive routing —
  see the scope decision above.
- The repo-hierarchy overlay's `same_module_bonus` is unit-tested but
  not wired into or measured through `retrieval::hybrid`'s RRF fusion
  (spec §194's proximity bonus) — that integration is future work once
  there is a real "active file" signal source (spec §85 mentions
  "active file/symbol from client metadata," which needs the protocol/
  session layer this phase doesn't touch).
- Partition initialization is a simple deterministic reservoir pick,
  not k-means++; with only 10-20 points this doesn't matter, but would
  need revisiting at real scale.
