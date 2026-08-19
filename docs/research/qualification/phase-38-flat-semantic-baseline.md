# Phase 38: flat semantic baseline

Spec Phase 38 deliverable (spec §38, §188-189, §310; exit gate row 38:
"Gold semantic recall/latency baseline recorded"). "Store full/INT8
reference vectors and exact SIMD search. This is the gold recall
baseline; preserve benchmark results forever."

## What was built

`retrieval::flat::FlatVectorStore`, built on top of Phase 37's
`helper_model::PplxEmbedRuntime`:

- **Full reference vector** (spec §188): one canonical L2-normalized
  1024-d FP32 embedding per document.
- **INT8 reference control** (spec §189 control 2): symmetric per-vector
  linear quantization (`scale = max(|x|)/127`), computed at index-build
  time from the normalized FP32 vector — deliberately distinct from
  Phase 37's model-native tanh-INT8 compact output, which has no stored
  scale by design (see that phase's doc). Search descales
  `sum(q_i * d_i) * scale_q * scale_d` back to the FP32 dot-product
  magnitude.
- **Native binary/Hamming control** (spec §189 control 3, "where
  available"): reuses the model's own sign-based packed binary output
  from Phase 37 rather than a separately learned binary head, since a
  real one is already available. Search ranks by ascending Hamming
  distance (XOR + popcount).
- **MRL prefix + renormalize** (spec §188): `mrl_prefix_renormalized`
  truncates the normalized embedding to a shorter prefix and
  re-normalizes, ready for a 256/512-d benchmark pass.
- **Exact search**: brute-force `search_fp32`/`search_int8`/
  `search_binary` over every record, each a plain auto-vectorizable
  scalar Rust loop — a REFERENCE BASELINE tier, matching how
  `model::qwen36`'s own full-attention started scalar before later
  phases added real NEON/Metal kernels. Hand-written SIMD intrinsics are
  only worth it once this baseline's cost is shown to matter.
- `recall_at_k`: standard top-k overlap recall, used to score every
  compact control against the FP32 gold ranking.

## Measured evidence

**Real corpus, real queries, not synthetic vectors.** Ten real files
from this crate's own source tree, chosen for semantic diversity across
four distinct concern clusters (memory broker, embedding quantization,
gitignore/file scanning, expert-cache eviction), each truncated to a
128-token input budget (a resource control on the Phase 37 reference
forward pass's cost, not a change to model semantics — see
`PplxEmbedRuntime::embed_with_input_budget`'s doc comment) and embedded
through the real Rust runtime built in Phase 37:

```
phase38_flat_build documents=10 max_input_tokens=128 elapsed_ms=278344
phase38_query "how does the memory broker account for reserved bytes per owner"
  fp32_top1="src/memory/mod.rs" expected_contains=true int8_recall@5=1 binary_recall@5=0.8
phase38_query "int8 and binary quantization of a pooled sentence embedding"
  fp32_top1="src/helper_model/quantize.rs" expected_contains=true int8_recall@5=1 binary_recall@5=0.8
phase38_query "gitignore glob pattern matching for a repository file scanner"
  fp32_top1="src/retrieval/ignore.rs" expected_contains=true int8_recall@5=1 binary_recall@5=0.8
phase38_query "eviction policy for a cache of mixture of experts weights"
  fp32_top1="src/experts/policy.rs" expected_contains=true int8_recall@5=1 binary_recall@5=1
phase38_summary mean_int8_recall@5=1 mean_binary_recall@5=0.85
```

All four real natural-language queries top-1-matched their intended
real file under the FP32 gold ranking. Linear INT8 recall@5 was
**perfect (1.0)** against that gold ranking on every query — the
reference INT8 control preserves ranking essentially losslessly at this
corpus scale. The native binary/Hamming control was meaningfully
lossier, as expected for 32x compression (1024 bits vs 1024×4 bytes):
**0.85 mean recall@5** (0.8/0.8/0.8/1.0 across the four queries) — still
strongly correlated with the FP32 ranking, establishing the real
recall-loss floor a future TQVec candidate (Phase 39) needs to beat with
a smarter encoding, not just match with brute compression.

## Status and remaining work

- **Corpus size and index-build latency are the open item, not
  correctness.** Building the flat store over 10 real files (128 tokens
  each) took 278 s on the naive scalar reference forward pass — this is
  Phase 37's known cost profile (linear per-token projection cost
  dominates until sequences reach several thousand tokens), not new to
  this phase. Nothing here benchmarks *query* latency at realistic
  repository scale (hundreds to thousands of documents); that requires
  either a faster forward pass (SIMD/batched, future work per Phase 37's
  own notes) or accepting index-build as a one-time offline cost, which
  is the real production shape but wasn't measured at that scale this
  phase.
- FP16 (spec's literal control 1 wording) is not implemented separately
  from FP32 — the checkpoint's own native output is F32 and this crate
  already carries F32 throughout its reference paths, so a distinct F16
  storage tier was judged not to add a meaningfully different "full
  precision" data point over what's already the gold ranking.
  Revisit if a real RAM-footprint comparison ever needs it.
- MRL-256/512 recall was not measured this phase — `mrl_prefix_renormalized`
  exists and is unit-tested, but no real-corpus recall run compares
  truncated-dimension recall against the full 1024-d gold ranking yet.
- The 10-file/4-query corpus is real but small; it demonstrates the
  pipeline and establishes a real (not synthetic) recall floor, but is
  not the "CoIR and RepoBench" benchmark suite spec §193 eventually
  wants, nor large enough to be a statistically strong recall estimate.
- No persistent storage format for the flat index (in-memory
  `Vec<FlatRecord>` only) — spec's `.tqi` container storage engineering
  (Part X) is later work, same scope decision Phase 36 made for the
  lexical index.
