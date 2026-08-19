# Phase 43: GTE reranker

Spec Phase 43 deliverable (spec §43, §93, §196, §315). "Implement
transient cross-encoder and ambiguity heuristic. Benchmark downstream
answer quality and TTFT, not reranker benchmark alone."

## What was built

`helper_model::gte_reranker` — a from-scratch Rust implementation of
`Alibaba-NLP/gte-reranker-modernbert-base` (a `ModernBertFor
SequenceClassification` cross-encoder, spec §93), a sibling of Phase
37's pplx-embed implementation but a genuinely different architecture
family:

- **LayerNorm, not RMSNorm** (weight-only, no bias anywhere except one
  tensor — see below).
- **Fused QKV** (`Wqkv: 768 -> 2304`), full MHA (no GQA, 12 heads x 64
  dims).
- **Alternating global/local attention**: every 3rd layer (0, 3, 6, 9,
  12, 15, 18, 21) is global (full bidirectional attention, RoPE
  θ=160000); every other layer is local (sliding window, radius 64,
  RoPE θ=10000).
- **Layer 0's `attn_norm` is `Identity`** — no tensor for it exists in
  the real checkpoint; the embeddings LayerNorm already normalizes
  that input.
- **GeGLU MLP** (`Wi: 768 -> 2304` fused, split into `input`/`gate`
  halves, `Wo(gelu(input) * gate)`), exact erf-based GELU (not the tanh
  approximation).
- **Masked-mean pooling**, then a `dense -> exact-GELU -> LayerNorm ->
  Linear` classification head producing one relevance logit.
  `classifier.bias` is the checkpoint's *only* bias tensor anywhere.

Every one of these facts was cross-checked against two independent real
sources before writing any Rust: the actual `transformers` library's
`modeling_modernbert.py` (fetched from GitHub, not recalled from
memory) and a byte-level read of the real checkpoint's safetensors
header (138 tensors; confirmed empirically that layer 0's `attn_norm`
tensor genuinely doesn't exist, and that `classifier.bias` is the only
bias tensor in the whole network) — not assumed from the architecture's
general reputation.

## Measured evidence, and a real bug that took real debugging to find

**Real checkpoint, real ONNX oracle, same validation methodology as
Phase 37.** The first end-to-end run against the real official ONNX
export failed by a wide margin (expected logit 1.7368, got -0.5462, a
2.28 absolute difference) and took an alarming 962 seconds for three
short query/document pairs — clearly something structural, not
numerical noise.

**Bisection, not guessing.** Rather than assume the model math was
wrong, the real forward pass was verified layer by layer against the
same ONNX graph's own internal node outputs (`onnx.load` +
appending intermediate tensor names as graph outputs, run through
`onnxruntime` — a second independent technique from Phase 37's oracle
comparison). A hand-traced replica of the encoder, fed the real 36
tokens for one pair, matched the oracle at **every** checkpoint —
post-embedding-norm, layer 0, layer 1, and the final logit itself
(1.7367942 vs 1.7367948) — proving the model implementation was
correct all along.

**The real bug was upstream of the model entirely.** The checkpoint's
own `tokenizer.json` bakes in a `Fixed(8000)` padding/truncation
policy, which *both* the Python and Rust `tokenizers` libraries apply
on every `encode()` call by default, not just batched ones — a real
36-token `(query, document)` pair silently became an 8000-token
sequence, 7,964 of them `[PAD]`. Nothing in this runtime's forward
pass or pooling ever masked them out (a deliberate simplification,
following Phase 37/38's precedent of "single unbatched sequence, no
padding needed" — which stopped being true the moment the tokenizer's
own config injects padding regardless of batching). `mean_pool`
silently averaged in those 7,964 meaningless positions, and the
unmasked global-attention layers let every real token attend to them
too — explaining both the wrong answer and the near-16-minute runtime
(quadratic attention cost at 8000 tokens for a query that only needed
36). Fixed by trimming trailing `[PAD]` tokens before running the
forward pass (`runtime::trim_trailing_pad`, using the checkpoint's real
`pad_token_id=50283`) — a resource-efficiency fix, not a masking
feature, since padding here exists only for batch-uniform tensor
shapes that this one-pair-per-call runtime never needs.

After the fix, all three real pairs match the ONNX oracle to ~1e-6, and
total runtime for the three pairs dropped from 962 s to 7.5 s:

```
gte_query "how does the memory broker account for reserved bytes per owner"
  doc "The memory broker is the single source of truth..." expected=1.7367948 actual=1.7367942 abs_diff=6.0e-7
gte_query "how does the memory broker account for reserved bytes per owner"
  doc "TurboQwenFare streams experts from SSD..." expected=1.3552905 actual=1.3552908 abs_diff=2.4e-7
gte_query "gitignore glob pattern matching for a repository file scanner"
  doc "A real (scoped) .gitignore/.tqfignore glob matcher..." expected=2.1585875 actual=2.1585894 abs_diff=1.9e-6
```

## Status and remaining work

- **The ambiguity heuristic (spec §196) is not implemented** — spec's
  reference trigger conditions (fused top score below a confidence
  threshold, small top-1/top-5 margin, semantic/exact lane disagreement,
  many near-ties, explicit model request) need a live hybrid-fusion
  pipeline producing real candidate scores to threshold against; Phase
  40's `retrieval::hybrid` produces exactly that shape of output, so
  wiring the heuristic there is the natural next step, not attempted
  this phase.
- **"Benchmark downstream answer quality and TTFT, not reranker
  benchmark alone"** — not measured. That needs a real end-to-end
  retrieval-augmented generation pipeline (hybrid fusion → rerank →
  context builder → live Qwen3.6 decode) this phase's scope doesn't
  reach; `GteRerankerRuntime::rerank` exists and is real, but nothing
  yet measures its effect on an actual downstream answer.
- No transient-lease coordination with a live expert cache (spec's
  memory design table: "reranker loaded... GTE unloaded before
  decode") — `GteRerankerRuntime` uses the same `MemoryOwner::
  HelperModel`/`MemoryClass::Transient` broker pattern Phase 37
  established, but there is no live decode loop running alongside it
  in this session to actually contend for memory with.
- The padding-trim fix is specific to this checkpoint's baked-in
  `Fixed(8000)` tokenizer config; a different reranker checkpoint might
  not have this quirk, or might have a different fixed length —
  `PAD_TOKEN_ID`/the trim logic is not generic across models, matching
  this crate's "fixed graph, not generic model support" convention
  throughout `helper_model`.
