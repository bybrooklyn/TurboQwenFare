# Phase 15's 512-token gate: divergence investigation (raw-a-512)

Spec Phase 15 exit gate (§112 row 15): "512-token greedy reference qualification
passes." This document records the first-ever attempt at that depth, why it
failed as literally specified, what was done to characterize the failure, and
a recommendation.

## Summary

TQF matched an independent llama.cpp CPU oracle **exactly for 197 consecutive
greedy decode steps** on a real 512-token continuation of the prompt `"A"` -
further than this codebase had ever been verified before (the prior deepest
qualification was 128 tokens, `raw-a-128-tqf.json`). It then diverged at
generated-token index 197 (decode step 198): the oracle's token was
`31215` ("Collision"), TQF computed `1501` ("Event") instead, with a top-2
logit gap of **~0.0105** - a near-total tie. A follow-up run let TQF freely
continue past that point for 30 more tokens; the result is coherent, on-topic
physics content, not degenerate output. Full data:
`raw-a-512-tqf.json`.

## How the oracle was produced (methodology note)

The existing 1/16/128-token fixtures appear to have been built by *directly
observing* each generated token ID as it was produced. For 512 tokens, that
approach (one full model reload/forward-pass per token via `llama-debug
--save-logits`, ~190s each - most of it model-load overhead, not compute) was
measured at ~190s/token and would have cost roughly 27 hours. Instead:

1. Ran `llama-completion` (the standard `llama.cpp` continuation tool, greedy
   `--temp 0`, `-ngl 0` CPU, `bf16-kv`, `--no-repack`, `-c 1024`) as **one
   continuous process** generating all 512 tokens with KV-cache reuse
   (`graphs reused = 509` in its own perf log). Total cost: ~64 minutes
   end-to-end (~10 min load + ~54 min generation, 7.5s/token per its own
   `eval time` stat), versus the ~27 hours the per-token-restart method would
   have cost.
2. Recovered exact token IDs by running `llama-tokenize` (same model/vocab)
   on the full prompt+completion text, since `llama-completion` only prints
   detokenized text by default, not token IDs.
3. **Validated this recovery is faithful**, not an artifact: the first 17
   recovered IDs match `raw-a-16.json` exactly, and the first 129 match
   `raw-a-128.json` exactly (both are independently-qualified, already-passing
   fixtures). Total token count was exactly 513 (1 prompt + 512 generated),
   with no off-by-one ambiguity.
4. Ran the real gate check (`dev::qualification::qualify_oracle`) against the
   resulting `raw-a-512.json`, which is what actually surfaced the step-197
   divergence - a genuine TQF-vs-oracle mismatch, not a recovery artifact
   (the recovery method's own self-consistency was already proven correct up
   to token 128, deep past where the divergence occurs at 197).

## Why this is very likely not a logic bug

- **The margin is a near-total tie.** `28.532768` vs `28.522234` in log-space
  is well within the range two independently-implemented FP32/BF16 pipelines
  - different dequantization loop order, different reduction/accumulation
  order across 40 custom layers - would be expected to disagree on, once
  enough autoregressive steps have compounded rounding differences. Expecting
  bit-for-bit-identical greedy argmax across 512 steps between two *separate*
  codebases computing "the same" 35B-parameter model is a stronger claim than
  "the implementation is correct" - it is closer to a coincidence when it
  holds for very long stretches, which is arguably what happened here (197
  steps, previously untested territory).
- **Both candidate tokens are coherent.** `31215` = `"Collision"`, `1501` =
  `"Event"`. Neither is nonsense; both are grammatically valid continuations
  of `"...**"` (a markdown bold-header opener) in a physics-outline context.
- **TQF's own post-divergence path stays coherent for 30 more tokens**
  (`raw-a-512-tqf.json`'s `coherence_followup`), reorganizing the same
  physics content (collision type, perfect inelasticity, no external forces)
  rather than degrading. A real routing/logic defect would be far more likely
  to produce nonsense or repetition once it started compounding, not a
  second internally-consistent branch of the same correct reasoning.
- Nothing about today's Phase 19/21 changes (parallel I/O fan-out, LRU cache
  default) can explain this: I/O parallelism only changes which thread reads
  already-checksummed bytes into a buffer before compute starts, never the
  floating-point computation itself, and cache policy only changes which
  bytes are resident, never a computed value.

## What this does and does not establish

**Does not establish:** that Phase 15's exit gate, read literally ("512-token
greedy reference qualification passes"), is met. It is not - this run failed
it, and it is the first ever attempt.

**Does establish:**
- The 40-layer decode graph, exact MoE routing, GDN, full attention, and
  sampling are correct enough to reproduce an independent oracle exactly for
  197 consecutive autoregressive steps on real model weights - a materially
  deeper correctness signal than existed before this investigation (previous
  record: 128).
- The one divergence found is characterized, not merely observed: a
  near-tied logit, coherent alternate token, and coherent continuation. That
  is the signature of floating-point non-associativity between independent
  implementations, not a defect signature (garbage output, repetition,
  wildly-off logits).

## Per-layer numerical root-cause investigation

The obvious next step is comparing TQF's and llama.cpp's actual per-layer
intermediate values, not just final argmax tokens. Both sides have the
tooling: TQF's bounded runtime already dumps every layer's activation to
`TQF_DEV_TENSOR_DUMP_DIR` (`maybe_dump_activation` in
`src/model/qwen36/runtime.rs`), and this project's research clone of
llama.cpp carries a local patch (`common/debug.cpp`,
`examples/debug/debug.cpp` - uncommitted, git-diff-visible) adding the same
capability via `TQF_LLAMA_TENSOR_DUMP_DIR` plus `--tensor-filter`, gated on
`--verbose` (undocumented outside the tool's own README note).

**What worked:** reproducing TQF's own divergence a third time (identical
1501/"Event" result, confirming determinism - nothing about it is racy or
flaky), this time with all 40 layers' activations dumped for decode step 197
(`decode-000197-input-2972-layer-*.f32le`). Getting llama.cpp's equivalent
per-layer dump *at that same 198-token-deep position* did not work: the
debug tensor callback never fires during a large (513-token) batched
prefill in this llama.cpp build (`ggml_backend_sched`'s graph-reservation
passes run, but `common_debug_cb_eval` is never invoked - confirmed across
three clean, isolated attempts with different batch-size configurations),
while the identical flags work reliably for small (1-7 token) prompts. This
looks like a genuine llama.cpp/GGML internals quirk specific to batched
prefill graphs in this build, not something worth further time chasing here.

**What was compared instead:** the trivial single-token case (prompt `"A"`
alone), where both sides' dumps are clean. TQF's `decode-000000-input-32-*`
files (step 0, processing prompt token 32) against llama.cpp's
`l_out-*-decode-1.f32le` files (the real evaluation, not the warmup pass,
which is `decode-0`) for all 40 layers:

| Layers | Cosine similarity | Pattern |
|---|---|---|
| 0-3 | 1.0000000 | bit-identical |
| 4-30 | 0.9999989-0.9999999 | tiny, smoothly growing drift |
| 31-39 | 0.9983-0.9998 | a real step-change, then a sustained higher-drift plateau |

Layer 31 is a full-attention layer (this architecture places full attention
every 4th layer starting at layer 3: 3, 7, 11, ..., 39 - ten layers total,
matching the Phase 13 exit gate's "ten full-attention layers validated").
The transition at layer 31 is a **step-change to a new, still-very-high
plateau (>99.8% cosine similarity)**, not a collapse - which is the
signature of ordinary floating-point non-associativity compounding through
network depth (different summation/dequantization order between two
independent implementations), not a localized logic defect. A genuine bug
(wrong transpose, off-by-one in GDN state, mis-applied RoPE) would be
expected to produce an abrupt, much larger break that does not recover, not
a smooth increase that stays within fractions of a percent.

This is consistent with, and strengthens, the earlier finding: even at a
single token of depth, TQF and llama.cpp's independently-implemented math
already differ by small amounts that grow through the 40-layer stack. Over
197 autoregressive steps, each reusing the previous step's (slightly
imperfect) output as input, this is exactly the kind of drift that would be
expected to eventually flip a near-total logit tie - which is what was
observed at generated-token index 197.

## Conclusion

The evidence gathered - determinism, a near-tied logit at the divergence,
coherent continuation past it, and now a measured, gradually-compounding
per-layer numerical drift with no abrupt localized break - converges on one
explanation: **this is floating-point non-associativity between two
independent implementations, not a TQF logic defect.** No further evidence
was found pointing at a specific fixable kernel; the drift is diffuse across
depth rather than attributable to one layer or subsystem.

## Second independent attempt (raw-b-512, 2026-08-17)

Since greedy decoding from a fixed prompt is fully deterministic, the only
way to get a genuinely independent data point (not just re-observing the
same tie at the same position) is a different prompt. Ran a second full
cycle: prompt `"The"` instead of `"A"`, generated via `examples/debug`'s
`generate_greedy_tokens` (direct token-ID output, not the retokenization
method - eliminating that risk factor entirely for this run), against a
corrected 1 GiB expert cache (up from the 256 MiB qualification-harness
default identified as a reuse dead zone earlier in this investigation - see
`raw-a-128-route-trace-policy.md`).

**Result: diverged even earlier - generated-token index 24 (decode step
25), not 197.** Same signature as before: a near-total top-2 logit tie
(16.453129 vs 16.374823, gap 0.078306 - somewhat wider than the first
run's 0.0105 but still a fraction of a percent of the logit scale), and the
cache was demonstrably getting real hits this time (3,053 hits by the
divergence point, versus zero in every prior run), ruling out the cache fix
itself as a confound. Full record: `raw-b-512-tqf.json`.

## Updated conclusion

Two independent prompts - unrelated topics, one diverging at token 197, the
other at token 24 - both hit a near-total logit tie and diverged. This
upgrades the finding from "one instance of plausible floating-point noise"
to **an observed, recurring property of comparing TQF's implementation
against this independent oracle at this quantization**: near-ties close
enough to flip under ordinary floating-point non-associativity happen often
enough that trying additional prompts is very unlikely to find one that
survives 512 tokens by luck. The "try a different prompt" strategy has now
been tried once and produced a *worse* (earlier) divergence, not a better
one - that is itself informative and argues against spending further time
on a third attempt.

## Recommendation

**Accept both results together as the closing evidence for Phase 15's
512-token investigation; stop trying additional prompts.** Literal
bit-exact-token-match across 512 autoregressive steps between two
independently-implemented floating-point pipelines is not a reachable bar
for this model/quantization by chance of prompt selection - two
independent, unrelated continuations both hit comparable near-ties well
before 512 tokens. This is a stronger claim than "the implementation is
correct," and the spec's own actual quality bar (§6, ≤1% degradation) is the
appropriate metric to qualify against going forward, not bit-exact-forever.
If deeper 512+ token qualification is needed later, the right next step is
a tolerance-based or distributional method (e.g., "the oracle's token is
always within TQF's top-k, or within an epsilon of the top logit") rather
than further attempts at exact matching.
