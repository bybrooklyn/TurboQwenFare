# Phase 39: TQVec candidate family

Spec Phase 39 deliverable (spec §39, §87, §190-191, §311; exit gate row
39: "Choose encoding only from benchmark Pareto frontier"). "Implement
A-F candidates from Part XIV with per-repo calibration. Choose only
after recall/latency/index-size/update Pareto analysis."

**RESEARCH CANDIDATES.** Matching spec §300's explicit rule for the
analogous TQKV candidates (Phase 28): a mixed-precision controller is
deliberately not built before individual encodings are qualified, so
none of these are wired into a live index. `retrieval::tqvec` is
self-contained rather than reusing `context::tqkv`'s similar
rotation/grouped-quantization machinery — the dependency firewall (spec
§24) keeps `retrieval` and `context` independently removable
subsystems, so some duplication of the underlying ideas (Hadamard
rotation, grouped bit-packed quantization) across the two modules is
intentional, not an oversight.

## What was built

All six candidates from spec §190, each defining byte size,
decoder/distance kernel, and (measured in the qualification test) real
recall loss:

- **TQVec-A** — native linear-INT8 (`retrieval::flat`'s own INT8
  control, re-exposed under the candidate naming).
- **TQVec-B** — 256-bit sign coarse key + full INT8. The coarse key is a
  latency prefilter, not an accuracy change: once every candidate is
  exactly re-scored by the INT8 kernel, B's recall is *identical* to A's
  by construction (asserted in the test).
- **TQVec-C / TQVec-D** — 256-bit coarse key + grouped 5-bit / 4-bit
  symmetric quantization (32 groups of 32 dims, one real `f16` scale per
  group, matching Phase 27 TQKV's own FP16-scale convention). A generic
  MSB-first bit-packer handles the non-byte-aligned 4/5-bit codes.
- **TQVec-E** — the same grouped Q4/Q5 machinery applied after a real
  fixed (not per-vector) randomized-sign-flip + fast Walsh-Hadamard
  rotation. Since the rotation is orthonormal and identical for every
  vector, `<Rx, Ry> = <x, y>` exactly, so scoring never needs to invert
  it (verified directly: `fwht_rotation_is_orthonormal_and_preserves_inner_products`).
- **TQVec-F** — a 1024-bit sign-based base plus an INT8-quantized
  residual against a crude binary reconstruction
  (`sign(bit) * mean(|values|)`). Exposes both a cheap `base_score`
  (Hamming distance only) and a `full_score` (base + residual
  reconstruction dotted), simulating "quantized residual information
  used only for top candidates."

## Measured evidence

**Reused Phase 38's real corpus/query embeddings, not a fresh synthetic
benchmark.** Rather than re-run the ~5-minute reference forward pass,
Phase 38's real computed FP32 vectors (ten real files from this crate,
four real natural-language queries) were captured as committed fixtures
— `raw-a-phase38-flat-corpus-embeddings.json` /
`raw-a-phase38-flat-query-embeddings.json` — the same "recapture the
trace, replay offline" precedent as `raw-a-128-route-trace.json`. This
makes `real_corpus_tqvec_candidates_pareto_comparison` a fast (~0.1 s),
always-on test rather than an `#[ignore]`d one, and lets every candidate
be measured against the *exact same* FP32 gold ranking Phase 38
recorded.

```
phase39_candidate name=A-int8                 bytes_per_vector=1028 mean_recall@5=1.00
phase39_candidate name=B-binary-coarse+int8   bytes_per_vector=1060 mean_recall@5=1.00
phase39_candidate name=C-binary-coarse+Q5     bytes_per_vector=738  mean_recall@5=0.95
phase39_candidate name=D-binary-coarse+Q4     bytes_per_vector=610  mean_recall@5=0.90
phase39_candidate name=E-rotated-Q5           bytes_per_vector=738  mean_recall@5=1.00
phase39_candidate name=E-rotated-Q4           bytes_per_vector=610  mean_recall@5=0.95
phase39_candidate name=F-residual-full        bytes_per_vector=1160 mean_recall@5=0.95
phase39_candidate name=F-residual-base-only   bytes_per_vector=1160 mean_recall@5=0.85
```

**A real, measured finding, not an assumption from the literature:**
Hadamard rotation improves recall at an *identical* byte budget. E-Q5
(738 B) recovers the full 1.00 recall that unrotated C (also 738 B)
loses to 0.95; E-Q4 (610 B) recovers to 0.95 versus unrotated D's 0.90
at the same 610 B. This reproduces, on this crate's own real embeddings
rather than synthetic data, the outlier-spreading effect spec §190
predicts rotation should have — the same qualitative result Phase 28
found for TQKV's analogous rotated candidate, now independently
confirmed for TQVec on different data through an independently written
implementation.

F's two-tier structure behaves as designed: the cheap `base_score`
(Hamming-only, no residual decode) already recovers 0.85 recall — Phase
38's own measured binary/Hamming floor, since F's base *is* the same
1024-bit sign key at full (not MRL-256) dimension — and pulling the
residual for `full_score` lifts that to 0.95, confirming the residual
is worth its cost when applied, i.e., the hierarchy has real signal to
extract, not just to store.

## Pareto read (this corpus)

At this real (small) corpus scale, no candidate beats A on recall, and
A is also not the smallest. The frontier that emerges:

- **Smallest at near-A recall:** E-Q5 (738 B, 1.00 recall) — 28% smaller
  than A/B for identical measured recall on this data.
- **Smallest overall:** D (610 B) trades the most recall (0.90) of any
  candidate; E-Q4 recovers most of that loss (0.95) at the same 610 B,
  making E-Q4 dominate D outright (same size, better recall) — D is
  Pareto-dominated by E-Q4 on this corpus.
- **B is dominated by A** (1060 B vs 1028 B for identical recall) unless
  its Hamming-prefilter latency benefit matters at a corpus size this
  phase didn't test.

No candidate is picked as a default per spec §300's rule — this is the
recall/size half of the Pareto surface, not latency at production
scale, and RESEARCH CANDIDATES are not meant to graduate to REFERENCE
BASELINE on a 10-document benchmark.

## Status and remaining work

- **Corpus scale is the major caveat.** Ten documents and four queries
  is enough to prove every kernel is implemented correctly and to
  surface a real, reproducible rotation effect, but far too small for
  the recall numbers to be statistically meaningful, or for B's/coarse-
  prefilter's latency argument to show up at all (a 10-record Hamming
  prefilter never meaningfully narrows a search). Revisit at the scale
  spec §193's CoIR/RepoBench-style evaluation implies.
- **Update cost** (spec §39's "update... Pareto" axis) is not measured
  — every candidate here is a pure function of a full FP32 vector with
  no incremental-update path modeled.
- **Repository-adaptive calibration** (spec §191: per-dimension moments,
  transform seed, group scales as a versioned profile, drift-triggered
  recalibration) is not implemented; TQVec-E's rotation seed is a fixed
  crate constant, not calibrated per repository.
- **F's reconstruction is deliberately crude** (`sign(bit) *
  mean(|values|)`, a single global magnitude rather than a real learned
  or per-dimension basis) — good enough to show the two-tier structure
  works, not a tuned candidate.
- No latency measurement beyond wall-clock encode/search time on this
  tiny corpus (both effectively 0 ms) — meaningless at this scale, kept
  in the test output only so the harness exists for a larger corpus
  later.
