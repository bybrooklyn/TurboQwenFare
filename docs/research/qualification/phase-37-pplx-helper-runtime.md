# Phase 37: pplx helper runtime

Spec Phase 37 deliverable (spec §37, §309; exit gate row 37: "Embedding
requests under 4G/2G contracts"). "Implement helper `.tqf` conversion and
transient broker lease. Validate embedding output against official/
reference runtime before compact vectors."

## What was built

`helper_model::` — a new top-level module, deliberately a sibling of
`model`/`runtime` rather than nested inside either, matching the spec's
own dependency-firewall table which lists "helper-model runtime" as a
distinct thing `retrieval` may depend on. The checkpoint is
`perplexity-ai/pplx-embed-v1-0.6b` (spec §86): a dense (non-MoE),
**bidirectional** Qwen3-architecture encoder (`bidirectional_pplx_qwen3`
/ `PPLXQwen3Model` — literally `transformers.Qwen3Model` with every
layer's `is_causal` forced `false` and an OR'd bidirectional attention
mask, per the checkpoint's own `modeling.py`), 28 layers, hidden 1024,
16 query / 8 KV heads, head_dim 128, RoPE θ=1e6, mean pooling, Matryoshka
(MRL) truncation, INT8/binary compact output.

- `helper_model::safetensors` — a minimal read-only safetensors parser
  (the source ships as `model.safetensors`, not GGUF; `format::gguf`
  doesn't apply). New `SafetensorsError`/`FormatError::Safetensors`
  variants added to the crate's error taxonomy (spec §119) rather than
  overloading `GgufError` for a format it doesn't describe.
- `helper_model::convert` — losslessly repacks the safetensors checkpoint
  into a `.tqf` container (F32 passthrough, `TQF_QUANT_PASSTHROUGH_F32`)
  using the simpler non-journaled `TqfWriter` path (`format::tqf::writer`'s
  own documented "synthetic-fixture roundtrip" tier — appropriate here
  since the checkpoint is ~2.2 GiB and converted once, not the 20 GiB
  canonical Qwen3.6 GGUF needing `ConversionTransaction`'s resumable
  journal). A new `PplxTensorRole` enum (`helper_model::roles`) keeps this
  model's tensor roles a separate on-disk contract from
  `dev::inventory::TensorRole`, which is Qwen3.6's own stable role list.
- `helper_model::weights`/`forward`/`pooling`/`quantize`/`runtime` — loads
  every tensor under a new `MemoryOwner::HelperModel` /
  `MemoryClass::Transient` broker reservation (spec §115 invariant #4;
  spec's memory design table item 7, "transient helper model while its
  current operation is executing"), then runs a from-scratch CPU forward
  pass: per-head Q/K RMSNorm, full (non-partial) rotate-half RoPE, GQA
  with 2x query/KV head grouping, bidirectional (unmasked) softmax
  attention over the whole sequence, SwiGLU MLP. Output is mean-pooled,
  optionally MRL-truncated, then quantized to INT8 (`round(clamp(tanh(x)
  * 127, -128, 127))`), signed binary, and packed ubinary — reproduced
  exactly from the checkpoint's own shipped `st_quantize.py`
  (`FlexibleQuantizer`), not reverse-engineered from output shape alone.
- `MemoryOwner::HelperModel` added to `memory::MemoryOwner`/
  `OwnerReserved` (was 9 owners, now 10) so helper-model residency is
  accounted separately from the decode-critical owners.
- `TqfTokenizer::from_tokenizer_json_file` added alongside the existing
  `from_gguf` constructor, for models (like this one) distributed with a
  standalone HF `tokenizer.json` rather than GGUF-embedded vocab/merges.

## Measured evidence

**Independent oracle, not self-consistency.** The checkpoint ships its
own official ONNX export (`onnx/model.onnx` + external data, ~2.4 GiB)
with `int8`/`binary` quantization heads baked in by the model authors —
run once via `onnxruntime` (Python, no `torch`/`transformers` needed) to
capture ground truth for two real sentences, independent of this crate's
own Rust implementation:

```
tqf-research-scratch/pplx-embed-inspect/run_oracle.py
  -> oracle_output.json {texts, token_ids, pooler_output,
                          pooler_output_int8, pooler_output_binary}
```

`real_checkpoint_matches_the_onnx_oracle` (`#[ignore]`, gated on
`TQF_PPLX_SAFETENSORS`/`TQF_PPLX_TOKENIZER`/`TQF_PPLX_ORACLE_JSON`, same
convention as the main model's `TQF_CANONICAL_TQF`-gated tests) converts
the real downloaded `model.safetensors` (SHA-256
`2c8d2f64f8268ccd5383b7f9bea8e660349aa6a151bd68a5a47f4c129f2a4974`, all
310 tensors, 2,384,199,680 verified output bytes) and runs the full
tokenize -> encode -> pool -> quantize pipeline through the real Rust
code path:

```
pplx_embed_convert extents=310 bytes=2384199680 sha256=2c8d2f64f8268ccd5383b7f9bea8e660349aa6a151bd68a5a47f4c129f2a4974
text 0 fp32 cosine similarity vs ONNX oracle: 0.9999999999957988
text 0 int8 mismatches (>1 off) out of 1024: 0
text 0 binary sign mismatches out of 1024: 0
text 1 fp32 cosine similarity vs ONNX oracle: 0.9999999999979525
text 1 int8 mismatches (>1 off) out of 1024: 0
text 1 binary sign mismatches out of 1024: 0
```

Token IDs match the oracle's tokenizer output exactly for both sentences
(asserted before any forward-pass comparison runs). Pooled FP32 cosine
similarity is ~1.0 (float-noise level, consistent with the same class of
ordinary non-associativity documented in
`raw-a-512-divergence-investigation.md` — a from-scratch naive-loop CPU
implementation accumulating in a different order than ONNX Runtime's
kernels — not a defect). **Zero** of 1024 INT8 dims differ by more than 1
quantization step, and **zero** binary sign bits differ, for both test
sentences.

## Status and remaining work

- No HTTP wiring: `POST /v1/embeddings` (spec line 1199, 1832) does not
  yet call `helper_model::PplxEmbedRuntime` — this phase delivers the
  runtime and its broker/qualification story, matching how Phase 20's
  kernels were "delivered as a kernel + microbenchmark, not yet wired
  into the live loop." Wiring the server route, plus the "broker may
  collapse expert residency almost completely during a semantic
  embedding/rerank operation" coordination the memory design table
  describes, is follow-on work once the endpoint's request/response
  shape (batching, `dimensions` truncation param, `encoding_format`) is
  designed.
- Only `pplx-embed-v1-0.6b` (independent query/document embedding) is
  implemented; `pplx-embed-context-v1` (RAG chunk embedding, same
  architecture family per the model card) is not.
- The forward pass is a naive triple-nested-loop CPU implementation with
  no SIMD path — fine for the short texts a `/v1/embeddings` request
  actually sends (both real oracle sentences ran in well under a second
  each inside a 17 s total test, most of which is the one-time 2.2 GiB
  conversion + weight load), but not benchmarked against a throughput
  target; a batched/SIMD path is future work if profiling shows it
  matters.
- Only INT8/binary/ubinary + FP32 are exposed; no per-request choice of
  MRL dimension is wired past the `mrl_dim: Option<usize>` parameter
  already on `PplxEmbedRuntime::embed`.
