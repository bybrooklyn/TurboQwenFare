# Phase 40: hybrid retrieval

Spec Phase 40 deliverable (spec §40, §85, §192-195, §312). "Implement
query intent lanes, RRF baseline, hard exact precedence and bounded
graph expansion. Add retrieval provenance explanation objects now for
GUI/debugging later."

## Scope decision: three lanes, no graph expansion

Spec §83 lists seven evidence lanes. Only three exist to fuse this
phase — **Exact** (Phase 36's `LexicalIndex::exact_lookup`), **Lexical**
(its BM25 lane), and **Semantic** (Phase 38's `FlatVectorStore`) —
because Structural/Program graph/Hierarchy/Change-Git all assume real
AST or Git-history integration that Phase 35/36 already scoped out this
session (no real parser, that decision's rationale carries forward
unchanged). Spec §195's graph expansion ("add parent definition, direct
callers/callees... test neighbors") needs a program graph that doesn't
exist for the same reason, so it is not attempted here either — building
a fake "graph expansion" on top of nothing would produce the same
"regex pretending to be a symbol table" problem Phase 36 explicitly
rejected. What *is* buildable without a parser — intent classification,
the candidate/provenance contract, RRF fusion, and hard exact precedence
over the lanes that exist — is built and measured on real data.

## What was built

`retrieval::hybrid`:

- **`QueryIntent`** (spec §192): `ExactSymbol`, `ExactPath`,
  `ErrorLiteral`, `SemanticQuestion`, `Mixed`. `classify_query` is a
  signal-based router (identifier shape, `::`/`(` forms, path
  separators + extensions, compiler/panic vocabulary, question-word
  density) returning *confidences* per class rather than one label,
  per spec's explicit "not a single mutually-exclusive classifier."
- **`should_use_semantic_lane`** (spec §85: "Identifier-like queries
  should hit exact/lexical/symbol paths without loading the embedder"):
  compares semantic vs. exact/path confidence to decide whether the
  (expensive) semantic lane runs at all.
- **`Candidate`/`CandidateProvenance`/`FusedCandidate`** (spec §193 and
  §40's "provenance explanation objects... for GUI/debugging"): every
  fused result keeps which lane(s) found it, at what raw score and
  in-lane rank, and a human-readable reason string.
- **`fuse_rrf`** (spec §194): weighted `Σ_lane weight/(k+rank)`, `k=60`,
  followed by hard exact precedence — every exact-lane hit sorts above
  every non-exact candidate regardless of RRF score, matching spec §84's
  "a direct definition match is not allowed to lose... solely through
  semantic score." Verified directly with a synthetic case where the
  semantic lane's raw score and rank both favor a different chunk than
  the exact hit — the exact hit still wins.

## Measured evidence

**Real files, real fusion, real routing decisions — reusing Phase 38's
committed fixtures so no model load is needed.** Lexical/Exact lanes
are built fresh from the same ten real files Phase 38 used; the
Semantic lane reuses Phase 38's committed real FP32 embeddings, and the
four real natural-language queries reuse their committed real query
embeddings (no re-running the reference forward pass).

```
phase40_query "MemoryBroker" intents=[(ExactSymbol, 0.6)] used_semantic=false
phase40_query "how does the memory broker account for reserved bytes per owner"
  intents=[(SemanticQuestion, 0.7)] used_semantic=true top1="src/memory/mod.rs"
phase40_query "int8 and binary quantization of a pooled sentence embedding"
  intents=[(SemanticQuestion, 0.5)] used_semantic=true top1="src/helper_model/quantize.rs"
phase40_query "gitignore glob pattern matching for a repository file scanner"
  intents=[(SemanticQuestion, 0.5)] used_semantic=true top1="src/retrieval/ignore.rs"
phase40_query "eviction policy for a cache of mixture of experts weights"
  intents=[(SemanticQuestion, 0.5)] used_semantic=true top1="src/experts/policy.rs"
```

The identifier query `"MemoryBroker"` correctly routes around the
semantic lane and the fused top-1 is the exact hit
(`src/memory/mod.rs`), with `Exactness::Exact` set. All four real
natural-language questions correctly engage all three lanes and the
fused top-1 reproduces Phase 38's own independently-established gold
winner for each query, through the full router → lane → RRF → hard-
precedence pipeline rather than a single lane in isolation.

**A real bug found and fixed in the process, worth recording rather
than glossing over:** the first version of `fuse_rrf` broke RRF ties
using `HashMap` iteration order, which is not reproducible across
process runs. This surfaced as a genuinely flaky test — passing when
run alone, failing intermittently in the full suite — on exactly the
"int8/quantization" query above: `src/helper_model/quantize.rs` (best
Semantic rank, rank 1) and `src/helper_model/runtime.rs` (best Lexical
rank, rank 1) land on an *exact* RRF-score tie (`0.032522473` on both,
to seven significant figures) once weighted equally, because RRF only
looks at rank, and both chunks are each other's lane's #1 with the
weights (Lexical 1.0, Semantic 1.0) equal. This is not a fusion bug —
it is RRF's real, known blind spot when two lanes weight equally and
disagree about which chunk is best — but the tie-break used to resolve
it must be deterministic. Fixed by breaking ties on best (lowest)
per-candidate lane rank, then lexicographically by `chunk_id` — a fully
deterministic secondary key, deliberately *not* a raw-score comparison,
per spec §193's explicit rule that "fusion never compares raw BM25 and
cosine numbers directly as though they share a scale."

## Status and remaining work

- No Structural/Program graph/Hierarchy/Change-Git lanes, and no graph
  expansion — see the scope decision above; both need real AST/program-
  graph output this session doesn't have.
- The reranker invocation policy (spec §196, GTE) is not attempted —
  that is Phase 43's explicit scope.
- Lane weights (Exact 2.0, Lexical 1.0, Semantic 1.0) and `k=60` are
  spec's stated initial controls, not calibrated against a real
  benchmark; the RRF-tie finding above suggests equal Lexical/Semantic
  weighting is a real place future calibration work could matter, not
  just a theoretical concern.
- The four-query, ten-document corpus is the same small real corpus
  Phase 38/39 used — real, not synthetic, but too small to be a
  statistically meaningful retrieval-quality benchmark on its own.
- `ExactPath`-intent queries (e.g. a literal file path as the query) are
  classified correctly but have no dedicated path-match lane wired into
  `run_hybrid_query` yet — the Exact lane only does identifier-token
  lookup; matching a query string directly against known chunk paths is
  future work.
