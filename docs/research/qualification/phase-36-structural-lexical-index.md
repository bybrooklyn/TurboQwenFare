# Phase 36: structural/lexical index

Spec Phase 36 deliverable (spec §308, §83, §185-186; exit gate row 36:
"Useful search without semantic model"). "Build useful search without
embeddings first. Exact symbol/path/BM25/graph baselines provide fallback
while helper models are unavailable."

## Scope decision: no AST, so no structural chunking/symbols/graph

Real structural chunking (spec §82, §181), symbol records (spec §182),
and the program graph (spec §183-184) all assume real AST output.
Phase 35 already made the decision not to add a grammar-parsing
dependency this session, so those three pieces have no real foundation to
build on here and are not attempted — building them on top of the
Phase 35 keyword-fingerprint substitute (rather than a real parser) would
produce fake-looking "symbols" that are actually just regex matches,
which is worse than not building them at all. What Phase 36 *does* build
needs no AST: the exact and lexical evidence lanes (spec §83).

## What was built

`retrieval::lexical::LexicalIndex`:

- **Tokenization** (spec §185): natural-language/whole-identifier tokens
  plus identifier subtokens split on snake_case/camelCase/PascalCase/
  SCREAMING_CASE/digit boundaries (including acronym runs, `HTTPServer` →
  `HTTP`, `Server`), lowercased for lexical matching.
- **BM25 reference scoring** (spec §186): the exact formula, `k1=1.2,
  b=0.75` as the spec's own stated non-sacred starting controls, standard
  IDF with the `+1` smoothing term.
- **Exact lane** (spec §83, §182 "Exact symbol lookup bypasses semantic
  ANN entirely"): a case-sensitive whole-raw-token index, structurally
  separate from the lowercased BM25 postings so exact lookups never
  inherit BM25's fuzziness.
- **Whole-file chunking**: in the absence of real AST-based sub-file
  chunking, one chunk = one file (a `ChunkRecord`-lite without
  `parent_symbol`, which needs a symbol table this phase doesn't build).
  A real, complete, useful reference baseline — not a placeholder — for
  exactly the reason spec §180 gives: "chunk text itself remains in the
  source file, not duplicated wholesale into the index."

## Measured evidence

**The literal exit gate, proven on real data, not synthetic fixtures.**
`real_repo_index_answers_real_exact_and_concept_queries_correctly` scans
this actual crate (reusing Phase 35's real scanner/classifier), builds a
lexical index over all 113 real Rust source files, and runs genuine
queries:

```
phase36_real_search documents=113 memorybroker_exact_hits=30
  expert_cache_query_top1=Some(("src/experts/mod.rs", 15.980503))
  gitignore_query_top1=Some(("src/retrieval/ignore.rs", 23.573912))
```

- Exact lane: `MemoryBroker` correctly resolves across 30 real files that
  reference it, including its own definition.
- BM25 lane: the natural-language-ish query "whole expert lfu cache
  eviction" top-ranks `src/experts/mod.rs` — the real file implementing
  `WholeExpertLfuCache`'s eviction logic — even though "expert cache
  eviction" never appears as that literal substring anywhere; only
  identifier-subtoken splitting of `WholeExpertLfuCache` plus real term
  density across the file make this findable at all.
- BM25 lane: "gitignore glob pattern matching" top-ranks
  `src/retrieval/ignore.rs` — this session's own just-written gitignore
  matcher — over every other file in the repo.

This is "useful search without a semantic model" demonstrated
end-to-end on the same real, large, heterogeneous corpus Phase 35 used,
continuing that phase's validate-against-real-data methodology rather
than resting on synthetic unit fixtures alone.

## Status and remaining work

- No structural chunking, symbol records, or program graph — see the
  scope decision above; all three need real AST output.
- No hierarchy lane (repo→module→file→type→function, spec §83) — that
  also assumes a symbol table.
- No change/Git lane (spec §83) — not attempted this phase.
- Postings are an in-memory `HashMap`, not spec §185's on-disk
  delta-encoded/bitpacked format — fine for the real 113-file corpus this
  phase validated against, but not yet the persistent `.tqi` container
  spec §173-177 (Phase 39/TQIndex storage) describes; this phase proves
  the scoring/tokenization logic works on real data before that storage
  engineering is warranted.
- Hybrid fusion across lanes (spec §84's "fuse calibrated ranks/confidence
  instead of pretending all raw scores live on one scale") has nothing to
  fuse yet beyond exact+BM25, since structural/graph/semantic lanes don't
  exist; real multi-lane fusion is later work once more lanes land.
