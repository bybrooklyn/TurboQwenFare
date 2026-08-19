# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TurboQwenFare (`tqf`) is a local, bounded-memory inference server specialized around a single model,
Qwen3.6-35B-A3B Q4. It is an early-stage Rust skeleton (crate version `0.0.1`) implementing the phased
roadmap in `TurboQwenFare_Master_v2_All_Encompassing_Specification.md` — that file is the normative
spec and is much larger than the current code. Consult it before making architectural decisions;
grep its section headers (`grep -n '^#' TurboQwenFare_Master_v2_All_Encompassing_Specification.md`)
rather than reading it end to end.

The spec uses a decision-status vocabulary that controls how much latitude an implementer has:
- **LOCKED** — do not change without an explicit architecture revision.
- **REFERENCE BASELINE** — implement first; replace only after an A/B result proves a better path.
- **RESEARCH CANDIDATE** — deliberately experimental; failure is an acceptable, recordable outcome.
- **BENCHMARK-SELECTED** — multiple valid implementations exist; the benchmark winner becomes default.

Implementation coverage now extends through the Phase 18 reference/bounded baseline: BF16 virtual-GQA
full attention, exact MoE routing/shared/routed computation, a 40-layer decode graph, normalized
OpenAI streaming adapters, canonical download/conversion/receipt startup, and a whole-expert cache
with exact load plans. A pinned real Q4_K_M checkpoint passes source and installed-container
topology validation, release headless server probes, and exact 1-, 16-, and 128-token greedy comparisons
against a pinned external oracle.

Phases 19 and 21 are also implemented and qualified with measured real-hardware evidence; Phase 20 has
real foundational work but is not wired into the live decode loop yet:
- **Phase 19 (parallel expert I/O):** `src/io/mod.rs` (`ReadFanout`/`fetch_all`) fans independently
  reserved expert-cache misses across a bounded thread pool, wired as the default in
  `WholeExpertLfuCache::prepare_exact_route` (`src/experts/mod.rs`); the Phase 18 serial path stays
  selectable via `TQF_EXPERT_IO_FANOUT` for A/B. Measured on the real checkpoint: 29.5x wall-time
  reduction for one exact route's independent misses (107ms serial vs 3ms parallel).
- **Phase 20 (Metal perf ports):** all four ports delivered with measured A/B records:
  broker-registered Metal buffer allocation (`MetalContext::allocate_broker_buffer*`, closing a
  previously-documented gap); `backend::metal::expert::GpuResidentExpert` (uploads one expert's
  Q4_K weights to persistent GPU buffers once, reused across forward calls instead of
  re-uploading per matvec) wired into the live decode loop — `WholeExpertLfuCache` entries are
  `ExpertValue::{Cpu,Gpu}` (`TQF_EXPERT_GPU_RESIDENT`, sole-backing-store accounting) and the
  streaming site calls `forward_expert`; the NVMAI-style 16-row threadgroup-staged fused GEMV
  (`tqf_q4k_gemv_staged16`, `TQF_EXPERT_GPU_KERNEL`, default `staged16`) with one-hot
  per-element parity against the CPU dequant oracle plus single-GEMV and chained-forward parity
  tests; function-constant shape specialization (`tqf_q4k_gemv_staged16_spec`,
  `staged16-spec`, pipeline-cache `FunctionConstantValues` plumbing) — parity holds but the
  real-weight A/B shows no consistent win, so it is not the default; and GDN four-way
  projection fusion (`tqf_q8_gemv_fused_gdn` + `tqf_q8_gemv` baseline) with CPU-oracle parity
  on real Q8_0 GDN weights and a measured **1.47x** over four separate launches (2.88 ms vs
  4.22 ms) — delivered as a kernel + microbenchmark, not yet wired into the live loop.
  **Decode-loop A/B done (8 greedy steps, canonical container): GPU vs CPU expert paths produce
  identical tokens but 0.96x wall time — no end-to-end win, so the GPU path stays opt-in
  (recorded negative result; decode is dominated by CPU reference projection stages and
  expert-miss I/O).** Full record: `docs/research/qualification/phase-20-gpu-resident-expert.md`.
  Remaining: wiring the fused GDN projection into the live decode loop (GPU-resident GDN
  weights + accounting + a fresh decode A/B — Phase 25 assault work).
- **Phase 21 (global expert cache):** `WholeExpertLfuCache`'s eviction policy is now pluggable
  (`CachePolicyKind::{Lru,Lfu,DecayedCostAware}`, `TQF_EXPERT_CACHE_POLICY`/`set_policy`). A real
  128-token/40-layer route trace from the live checkpoint was captured and replayed offline
  (`experts::policy::replay_trace`); LRU measured ~36% fewer bytes fetched than LFU at cache
  capacities that get any reuse at all, and is now the default (`DEFAULT_CACHE_POLICY`). Full record:
  `docs/research/qualification/raw-a-128-route-trace-policy.md`.

This is not equivalent to closing every Phase 13-18/19-21 exit gate. Still open: Phase 15's broader
workload matrix and its literal 512-token exact-match requirement (two independent 512-token attempts
were made and both diverged on a near-tied logit around token 197 and token 24 respectively — see
`docs/research/qualification/raw-a-512-divergence-investigation.md` for the full root-cause writeup;
short version: this looks like ordinary floating-point non-associativity between TQF and the
independent oracle rather than a defect, but the literal exit gate as worded does not close), combined
<=1% quality qualification, plain GUI startup, and RTX 3070 Ti/CUDA qualification. Phase 20's kernel
fusion and live-loop wiring are still not wired. Check the current code, tests, and
`docs/research/canonical-source-manifest.md` before making a stronger status claim.

Phases 22-26 are now implemented and recorded; each has a measured qualification ledger in
`docs/research/qualification/`:

- **Phase 22 (tiled experts):** neuron-width tile layouts (64/128/256/mixed) with per-tile BLAKE3
  checksums (`format::tqf::tiling`, `TQF_EXPERT_TILE_NEURONS` conversion control, no format
  migration), tile-granular verified reads, and an O(1)-LRU tile replay simulator. The real
  128-token route-trace A/B shows tiling changes fetched bytes by ~0% while multiplying read
  syscalls 5-9x — **whole-expert admission stays the default (recorded negative result)**.
  `docs/research/qualification/phase-22-tiled-experts.md`.
- **Phase 23 (predictive prefetch):** `experts::prefetch` transition/co-routing predictor with
  decay, offline replay logging precision/recall/timeliness/wasted bytes, and a live opt-in path
  in the cache (probation entries, `TQF_PREFETCH_ENABLED`/`TQF_PREFETCH_DEPTH`). Replay: 42-57%
  precision, -19..52% demand-miss bytes at depth 8, but +45..56% total SSD traffic at the
  capacities where the cache already works (depth 4 wins only below the reuse floor) — **stays
  off by default pending a live A/B**. `docs/research/qualification/phase-23-predictive-prefetch.md`.
- **Phase 24 (hard 4G broker):** OS footprint sampler (`memory::os_sampler`, macOS `task_info`/
  Linux `/proc/self/statm`), per-owner reserved breakdown + peak tracking in the broker, a 200k-step
  adversarial churn test, and real-checkpoint OS qualification: peak RSS 1,777 MiB vs broker peak
  1,488 MiB (689 MiB measured overhead envelope) on the resident-core streaming profile.
  `docs/research/qualification/phase-24-4g-broker-certification.md`.
- **Phase 25 (M4 assault):** resident-core streaming profile wired (`TQF_DEV_RESIDENT_STREAMING`;
  2.13 GiB measured core) — fixes the bounded runtime's per-token re-reads and a stale resident-path
  conv bug; bit-identical NEON Q4_K/Q6_K/Q8_0 dot kernels (`simd/`, differential fuzz-tested,
  A/B controls `TQF_SIMD_Q4K/Q6K/Q8_0`); activation-quantization hoisting. Measured: 23.4 → 2.34
  s/token (10x) with exact 16-token oracle parity, but **78% of decode is demand I/O**: the
  container lives on an external flash drive reading 139 MB/s against ~230-280 MB/token of expert
  misses. The 15 tok/s floor is NOT closed; the ledger (parallel MoE compute, container placement,
  LM-head loop, live prefetch) is in `docs/research/qualification/phase-25-m4-assault.md`.
- **Phase 26 (prefill):** chunked layer-outer prefill with per-(layer, chunk) expert-set dedup
  (`WholeExpertLfuCache::prepare_batch_route`, `Qwen36StreamingMoe::forward_batch`,
  `Qwen36ReferenceRuntime::prefill_greedy` with broker-pressure chunk autotuning), wired into the
  generation loop. Measured 53-token prompt: **1.81x TTFT** (142.7 s → 78.8 s), 2.18x fewer expert
  fetches/bytes, identical greedy continuation. `docs/research/qualification/phase-26-prefill.md`.

The re-captured exact-router trace for these replays is committed at
`docs/research/qualification/raw-a-128-route-trace.json` (set `TQF_QUALIFICATION_ROUTE_TRACE` to it
to re-run the Phase 21-23 replay harnesses).

Phases 27-28 land the first long-context (TQKV) work:

- **Phase 27 (TQKV Q8/Q4 baseline):** `context::tqkv` — 256-token sealed
  pages with a 128-byte header (little-endian, BLAKE3-checksummed), a
  high-precision mutable tail, and symmetric per-page Q8/Q4 Key/Value
  quantization (real IEEE-754 FP16 scales via the `half` crate), wired live
  into `FullAttentionLayer` as a `KvCacheBackend::{Bf16,Tqkv}` choice
  (`TQF_TQKV_ENABLED`/`TQF_TQKV_PRECISION`, off by default). A 261-step
  differential test against the BF16 oracle through the real production
  attention code path stays under 0.05 max abs error across a sealed-page
  boundary; on the real checkpoint, BF16 and TQKV-Q8 produced **identical**
  8-token greedy continuations. 128K capacity accounting (broker-reservation
  math, not a live 128K decode): BF16 2.50 GiB / TQKV-Q8 ~1.29 GiB / TQKV-Q4
  ~0.67 GiB across the ten full-attention layers. A follow-up real
  264-step run found the two backends diverge at step 9 on a razor-thin
  logit gap (0.002-0.045 out of ~21 magnitude, same class of finding as
  `raw-a-512-divergence-investigation.md`); investigated and recorded, not
  a defect. `docs/research/qualification/phase-27-tqkv-baseline.md`.
- **Phase 28 (advanced TQKV candidates):** `context::tqkv::candidates` —
  standalone Q3 symmetric, Q2 asymmetric, Rotated-Q4 (randomized Hadamard
  transform), outlier-split Q4, and pre-RoPE Q4 encode/decode pairs, per
  spec §300's explicit instruction not to build a mixed-precision
  controller before individual encodings are qualified — **none of these
  are wired into the live cache**. Candidate-matrix measurement: rotation
  cuts max error 2.3x on frequent-skew synthetic data at no byte cost;
  outlier-split cuts max error 13.3x on rare-outlier data for +12 bytes;
  Q3 trades ~25% smaller payload for ~2.3x the Q4 baseline's error.
  `docs/research/qualification/phase-28-advanced-tqkv.md`.
- **Phase 29 (128K production gate):** a literal end-to-end 128K live decode
  is not run (Phase 25's own I/O-bound floor puts it at many hours to days
  on this hardware); instead the gate's three parts are measured directly.
  Memory: `context::tqkv::scaling_bench` really constructs all ten
  full-attention layers at 131,072-token TQKV-Q4 capacity inside one 4 GiB
  broker (0.67 GiB reserved, 3.33 GiB headroom). Performance: a new
  `FullAttentionLayer::seed_synthetic_history_for_benchmark` isolates real
  attention-step cost from I/O at deep, otherwise-unreachable context
  lengths — measured **450 ms/step (BF16) and 1,687 ms/step (TQKV-Q4) at
  128K tokens**, i.e. attention compute alone (before any I/O or MoE work)
  already exceeds the entire 15 tok/s (66.7 ms) budget by 6.75x-25x,
  confirming TQAttn-style selective attention (Phase 31-32) is
  architecturally necessary, not optional — and surfacing a new recorded
  negative result: TQKV-Q4 is 3.7x slower per attention step than BF16 at
  128K (scalar per-token dequant cost). `docs/research/qualification/phase-29-128k-gate.md`.
- **Phase 30 (prefix snapshot store):** `context::prefix::PrefixSnapshotStore`
  — content-addressed, refcounted, crash-safe (atomic-write) on-disk
  storage for TQKV pages (reusing their existing BLAKE3 content hash as the
  page-content ID, spec §66) and GDN recurrent state
  (`GdnState::to_bytes`/`from_bytes`), an exact BLAKE3 token-prefix hash
  (spec §67, v1 exact-match only), and LRU disk-quota eviction. Wired into
  `Qwen36BoundedReferenceRuntime::snapshot_session`/`restore_session`.
  Real restart reuse demonstrated: a snapshot survives dropping the store
  handle entirely and reopening a fresh one against the same directory,
  restoring byte-identical state. On the real checkpoint, restoring a
  real 8-token prefix from a snapshot took 27 ms versus 53.5 s to decode
  it from scratch (**1,963x**), with byte-identical continuation
  confirmed. `docs/research/qualification/phase-30-prefix-store.md`.
- **Phase 31 (256K/TQAttn trigger):** applies spec §303's own decision
  rule after extending Phase 29's isolated-attention-step measurement to
  262,144 tokens. Memory: TQKV-Q4 at 256K reserves 1.33 GiB (fits 4G); BF16
  at 256K needs 5.00 GiB (does not fit 4G at all, independent of speed).
  Performance: attention alone costs **822 ms/step (BF16)** and
  **3,331 ms/step (TQKV-Q4)** at 256K — 12.3x and 49.9x over the 15 tok/s
  budget before any I/O or MoE work. Per §303's literal rule this formally
  **triggers Phase 32 (TQAttn)** — full attention cannot stay the default
  beyond this point on the reference implementation.
  `docs/research/qualification/phase-31-256k-tqattn-trigger.md`.
- **Phase 32 (TQAttn):** `context::tqattn` — the spec §164 REFERENCE
  BASELINE Quest-style [R21] selector, not the self-indexing Key
  candidates of §63/§167 (§300 defers those until this baseline is
  qualified). Extended `SealedPage` with a per-(kv_head, dim) min/max
  search summary (caught and fixed a real broker-under-reservation bug in
  the process: the capacity formula hadn't grown with the struct).
  Implements the §164 selector (recent window + protected pages always
  included, remaining pages scored via the Quest bound) and 2 of §165's 6
  uncertainty-expansion triggers. Real full-attention A/B on a 16,384-token
  synthetic context: **10.72x wall-clock speedup** attending to 9.4% of
  tokens (page budget 6 of 64 pages), with an engineered "important" old
  page correctly recalled and 99.8% of the full-attention score preserved.
  Not yet wired into the live decode loop.
  `docs/research/qualification/phase-32-tqattn.md`.
- **Phase 33 (MTP):** the real MTP sidecar checkpoint
  (`source::pinned::MTP_FILENAME`, a separate ~1 GiB GGUF) is not
  installed and a full second forward-pass runtime for it is out of
  scope, so `runtime::mtp` implements what doesn't require the sidecar to
  exist: NVMAI-derived accept/reject verification semantics and
  statistics (consulted directly from the real
  `StreamingMTPDecoder`/`StreamingMTP.swift` reference implementation),
  an adaptive hysteresis controller that defaults off and only enables
  after a sustained positive rolling net-benefit window, and expert-union
  bandwidth accounting. Measured against the real committed 128-step
  route trace (not synthetic): **20.81% fewer expert bytes** for a
  verified boundary+draft pair versus fetching each independently,
  averaged across 5,080 real (layer, consecutive-step) pairs.
  `docs/research/qualification/phase-33-mtp.md`.
- **Phase 34 (2G profile):** spec §40's staged sequence, stage 2 first.
  Real construction of the *entire* 40-layer context/recurrent-state
  footprint (30 GDN states + 10 TQKV-Q4 full-attention layers at 128K)
  inside a real 2 GiB broker: **0.731 GiB used, 1.269 GiB headroom** for
  weights/expert-cache. Stage 1: a real 8-step decode under a 2 GiB
  broker with TQKV-Q4 and a 384 MiB expert cache produced **bit-identical**
  tokens to the established BF16-4GiB baseline, peak reservation 459 MiB.
  Stage 3 (15 tok/s) is not attempted since Phase 25/29 already establish
  the reference compute path can't close that floor even with more cache,
  and spec §40 itself says a 2G speed miss doesn't invalidate the 4G
  system. `docs/research/qualification/phase-34-2g-profile.md`.
- **Phase 35 (file catalog/classifier):** `retrieval::{ignore,classify,scan}`
  — a real (scoped) `.gitignore`/`.tqfignore` glob matcher, content-first
  classification (byte-sniff binary detection, a keyword-fingerprint
  language scorer substituting for real AST `parser_quality`, shebang/
  basename hints, generated/vendor detectors), and a symlink-cycle-safe,
  root-escape-safe filesystem scanner. Validated against this crate's own
  live 150-file source tree, not just synthetic fixtures — catching two
  real bugs in the process: `DirEntry::metadata()` doesn't follow
  symlinks (a directory symlink silently fell through to a failed file
  read instead of being walked), and doc-comment-heavy real Rust files
  were misclassified by generic keyword collisions in prose. All 112 of
  this crate's own real `.rs` files now classify correctly.
  `docs/research/qualification/phase-35-file-catalog-classifier.md`.
- **Phase 36 (structural/lexical index):** `retrieval::lexical` — no real
  AST exists (Phase 35's scope decision), so structural chunking/symbols/
  program graph are not attempted; instead a real BM25 reference index
  (spec's own `k1=1.2, b=0.75`) with snake/camel/Pascal/digit-boundary
  identifier subtoken splitting, plus a case-sensitive exact-identifier
  lane, both over whole-file chunks. Proven end to end on this crate's own
  real 113-file source tree (not synthetic fixtures): `MemoryBroker`
  exact-resolves across 30 real referencing files; the query "whole
  expert lfu cache eviction" (never appearing as that literal substring
  anywhere) top-ranks `src/experts/mod.rs` via identifier-subtoken
  splitting; "gitignore glob pattern matching" top-ranks
  `src/retrieval/ignore.rs`. `docs/research/qualification/phase-36-structural-lexical-index.md`.
- **Phase 37 (pplx helper runtime):** `helper_model::` — a new top-level
  module (sibling of `model`/`runtime`, matching the spec's dependency-
  firewall table) implementing `perplexity-ai/pplx-embed-v1-0.6b`: a
  dense, **bidirectional** Qwen3-architecture encoder (28 layers, hidden
  1024, 16Q/8KV heads, RoPE θ=1e6), distinct from the causal MoE Qwen3.6
  core. A minimal safetensors reader (new `SafetensorsError`/
  `FormatError::Safetensors`), lossless F32-passthrough `.tqf` conversion,
  a from-scratch CPU forward pass (per-head QK-RMSNorm, full rotate-half
  RoPE, GQA, bidirectional softmax attention, SwiGLU), mean pooling, MRL
  truncation, and INT8/binary/ubinary quantization reproduced exactly
  from the checkpoint's own shipped `st_quantize.py`. Loaded under a new
  `MemoryOwner::HelperModel`/`MemoryClass::Transient` broker reservation
  (spec's "transient helper model while its current operation is
  executing"). Validated against the checkpoint's own official ONNX
  export as an independent oracle (not self-consistency): on the real
  2.2 GiB checkpoint (310 tensors, SHA-256 verified), token IDs match
  exactly, pooled FP32 cosine similarity is ~1.0, and **zero** of 1024
  INT8 dims differ by more than one quantization step with **zero**
  binary sign-bit mismatches, across both real test sentences. Not yet
  wired into `POST /v1/embeddings` — this phase delivers the runtime and
  its qualification, not the HTTP route.
  `docs/research/qualification/phase-37-pplx-helper-runtime.md`.
- **Phase 38 (flat semantic baseline):** `retrieval::flat::FlatVectorStore`
  — the gold recall baseline every future approximate index is measured
  against (spec §189). Full L2-normalized FP32 reference vectors, a
  separate per-vector-scale linear INT8 control (distinct from Phase
  37's model-native tanh-INT8 compact output), the model's own native
  sign-based binary/Hamming output reused as the binary control, MRL
  prefix+renormalize, and brute-force scalar exact search (REFERENCE
  BASELINE tier — SIMD kernels are later work if this is ever shown to
  matter). Measured on ten real files from this crate's own source tree
  across four distinct semantic clusters, embedded through the real
  Phase 37 runtime: all four real natural-language queries top-1-matched
  their intended file under the FP32 gold ranking; linear INT8 recall@5
  was **perfect (1.0)**; the native binary/Hamming control's recall@5
  was **0.85** (0.8/0.8/0.8/1.0), establishing the real recall-loss floor
  a future TQVec candidate (Phase 39) needs to beat.
  `docs/research/qualification/phase-38-flat-semantic-baseline.md`.
- **Phase 39 (TQVec research):** `retrieval::tqvec` — RESEARCH CANDIDATES
  A-F from spec §190 (native INT8; binary-coarse+INT8; binary-coarse+
  grouped Q5/Q4 with real `f16` per-group scales; rotated Q4/Q5 via a
  real fixed randomized-sign-flip + fast Walsh-Hadamard transform;
  residual hierarchy), none wired into a live index per spec §300's
  rule. Measured on Phase 38's real corpus/query embeddings (committed
  as `raw-a-phase38-flat-*.json` fixtures so the benchmark reruns in
  ~0.1s without re-embedding): A/B both hit perfect recall@5 (1028/1060
  bytes); grouped Q5/Q4 without rotation lose recall (0.95/0.90 at
  738/610 bytes); **rotation measurably recovers that loss at the
  identical byte budget** (Q5: 1.00, Q4: 0.95) — a real, reproduced
  finding (Phase 28 found the same qualitative rotation effect for TQKV,
  independently confirmed here for TQVec on different real data). The
  residual hierarchy's cheap base-only score reproduces Phase 38's 0.85
  Hamming floor; adding the residual lifts it to 0.95.
  `docs/research/qualification/phase-39-tqvec-candidates.md`.
- **Phase 40 (hybrid retrieval):** `retrieval::hybrid` — query-intent
  routing (spec §192's `QueryIntent`, confidence-scored not
  mutually-exclusive), the `Candidate`/`CandidateProvenance` contract
  (spec §193, "provenance explanation objects... for GUI/debugging"),
  weighted RRF fusion (`k=60`) with hard exact precedence (spec §84/
  §194: proven directly — an exact hit beats a higher-scoring
  semantic-only rival). Fuses only the three lanes that exist (Exact/
  Lexical from Phase 36, Semantic from Phase 38) — Structural/Program
  graph/Hierarchy/Change-Git and spec §195's graph expansion all need
  real AST/program-graph output Phase 35/36 already scoped out.
  Measured end to end on Phase 38's real corpus/query fixtures (no
  model load needed): an identifier query correctly skips the semantic
  lane and returns the exact hit; all four real NL queries correctly
  engage all three lanes and reproduce Phase 38's own gold-ranking
  winners through the full pipeline. Found and fixed a real
  nondeterminism bug in the process — `HashMap`-order tie-breaking made
  a genuine RRF tie (two chunks, each the other lane's #1, weighted
  equally) flaky across runs; fixed with a deterministic rank-then-
  chunk_id tie-break, never a cross-lane raw-score comparison (spec
  §193's explicit rule).
  `docs/research/qualification/phase-40-hybrid-retrieval.md`.
- **Phase 41 (adaptive ANN research):** `retrieval::adaptive` — steps
  1-2 of spec §313's five-step candidate sequence: a deterministic
  balanced k-means `SemanticPartitionIndex` over Phase 38's real
  vectors, and a path-derived `HierarchyOverlay` (repository/module/
  file, no AST needed). Steps 3-5 (hot/cold residency, split/merge,
  workload-adaptive routing) need live query/update traffic this
  session's offline methodology can't produce honestly, deferred to
  Phase 42. Measured on the real Phase 38 corpus: the hierarchy overlay
  correctly groups all ten real files into six real modules; static
  partitioning is a genuine **negative result at this scale** — `k=2`
  partitions/`nprobe=1` loses 30 recall points (0.70) for only a 55%
  scan reduction, `k=3` drops to 0.60 recall for 65% — confirming spec
  §89's own prediction that flat search "can be surprisingly
  competitive for normal repository sizes." Not a dead end, just not
  measured as a win at this corpus size.
  `docs/research/qualification/phase-41-adaptive-ann-research.md`.

## Commands

```sh
cargo build                 # debug build (Metal backend by default on macOS)
cargo build --release       # release profile: LTO, codegen-units=1, panics unwind (not abort)
cargo run -- --headless     # run the server without launching the SwiftUI GUI
cargo test                  # run the full test suite
cargo test config::tests::parses_suffixed_sizes   # run a single test by path
cargo test -- --list        # list all test names without running them
cargo clippy
cargo fmt
```

Feature flags select the compute backend at compile time (exactly one is expected per build):
`--features metal` (default) or `--features cuda`. There is no generic/CPU-only inference backend.

There is no CI workflow configured yet (`.github/workflows` does not exist) despite the spec defining
one (section 262, "CI lane design", Lanes A–E) — don't assume CI enforces anything today.

## Architecture

### Single crate, module boundaries as a dependency firewall

This is deliberately **one Cargo crate, not a workspace** (spec §23, "One-crate source tree" —
LOCKED). Do not propose splitting it into internal crates. Logical boundaries are enforced by module
structure and a dependency firewall (spec §24) instead:

| From | May depend on | Must not depend on |
|---|---|---|
| `model`/`runtime` | `memory`, `backend`, `io`, `tokenizer`, `sampling` | `retrieval`, `gui`, `integrations` |
| `server` | `runtime`, retrieval facade, protocol types | `gui` |
| `retrieval` | memory broker, helper-model runtime, parsers, `simd` | `gui`, coding-client internals |
| `gui` | local control/server interfaces only | model internals |
| `integrations` | server/MCP launch/config helpers | model kernels |
| `backend` | platform FFI, shared tensor metadata | retrieval/product logic |

The inference core must remain valid with retrieval and the GUI entirely disabled — they are optional
consumers of the server/runtime, never the reverse.

### Request flow (spec §22)

```
Clients (OpenAI/Anthropic/Ollama/GUI/MCP) → Request Normalizer → optional TQIndex RAG
  → Session Scheduler → Qwen3.6 Execution Core (TQKV/TQAttn, Expert Runtime, Prefix Runtime)
  → Memory Broker → Metal/CUDA, CPU SIMD, SSD
```

Every protocol (OpenAI Chat Completions/Responses, Anthropic Messages, Ollama) is normalized at the
HTTP boundary into one internal `NormalizedRequest`/`Session` representation (`src/runtime/request.rs`,
`src/runtime/session.rs`) before touching the model loop — protocol-specific framing must never leak
into the model core.

### Why the design looks unusual

The model (Qwen3.6-35B-A3B, MoE, hybrid GatedDeltaNet/full-attention) is deliberately run out-of-core:
experts are streamed from SSD through a **memory broker** that is the single source of truth for what
may be resident (spec Part VI). `--memory` is a hard live working-set budget, not an advisory cache
size — allocations must be registered with the broker *before* they happen ("allocate, then report" is
explicitly prohibited, spec §115). This out-of-core/streaming design is why the crate has `format/tqf`
(a custom container format), `context/{tqkv,tqattn,prefix}` (custom KV representations for long
context), and `experts/` as first-class modules rather than loading weights into RAM/VRAM wholesale.

### Async/thread model (spec §25)

Tokio owns non-differentiating async work: HTTP, SSE, downloads, filesystem watching, MCP, process
integration. The token-critical decode loop is intended to run on dedicated threads/queues with
explicit GPU/I/O synchronization — do not casually move inference-loop logic onto the Tokio executor.
v1 maintains a single active decode request and queues others; batch-1 latency/throughput is the core
metric and must not regress silently for the sake of future batching.

### Error taxonomy (`src/error.rs`)

One crate-wide `TqfError` enum aggregates a per-subsystem error type (`ConfigError`, `SetupError`,
`ModelError`, `FormatError`, `MemoryError`, `IoError`, `BackendError`, `ContextError`,
`RetrievalError`, `ProtocolError`, `InternalError`). This structure is fixed by spec §119; when adding
a new fallible subsystem, add a variant here rather than using a generic/boxed error type.
`InternalError` is reserved for violated invariants (not user/environment errors) and always carries
an `incident_id` for log correlation.

### Setup / first-run (`src/setup/`)

Implements the state machine in spec §28: detect hardware → load config → validate a trusted model
receipt → if missing, ask the user to download/convert (or require `--yes` non-interactively) → start
the server. This must stay transactional: an interrupted download/conversion must leave a resumable
partial state, never a model directory that looks valid. No partial receipt is ever written for a
declined or not-yet-implemented setup path — see `SetupOutcome` in `src/setup/flow.rs`.

## Core invariants (spec §115, LOCKED — apply across the whole crate)

1. The hot inference path is model-specific to Qwen3.6-35B-A3B; generic-model abstractions must not
   appear in performance-critical loops unless they compile away completely.
2. All persisted integers in TQF-owned binary formats are little-endian; readers reject unsupported
   endianness rather than guessing.
3. File byte offsets/lengths are `u64` on disk and in validation code; convert to `usize` only after
   checked bounds validation against the mapped/read region.
4. Every large allocation is registered with the memory broker *before* physical allocation.
5. Every async I/O op owns/borrows a destination lease that outlives completion — a cache tile cannot
   be evicted while a read into it is pending.
6. A GPU command may only reference buffers whose broker leases outlive the command-completion event.
7. Exact expert routing always comes from Qwen's real router; predictors may schedule bytes, never
   alter selected expert IDs or weights.
8. Any approximate context/retrieval optimization needs a correctness fallback path and a
   qualification test.
9. Persistent writes use temp/journal + fsync/atomic-rename semantics wherever corruption would be
   expensive to recompute or could break correctness.
10. Every performance optimization must be disableable via a developer/debug control for A/B testing,
    but ordinary users never see a quality/performance-mode maze.

Newtypes (`LayerId`, `ExpertId`, `TileId`, `ContextPageId`, `Bytes`, `Tokens`, spec §116) are used
throughout to keep layer/expert/page IDs and byte/token counts from being accidentally mixed — follow
this pattern rather than passing around bare `u64`/`usize`.

## Contributor "do not do this" list (spec §114)

- Do not add generic Llama/other-model support "while we are here" — this is a single-model runtime.
- Do not introduce an external vector database because the custom `TQIndex`/`TQVec` retrieval design
  (spec Part X) is hard.
- Do not split the repository into a workspace of internal crates.
- Do not add a user-facing quality mode; quality policy is automatic and must stay within the
  ≤1% global degradation ceiling (spec §6).
- Do not let `retrieval` or `gui` become dependencies of the inference core.
- Do not silently allocate above `--memory`, or rely on unreported OS page cache to justify a memory
  claim.
- Do not claim a headline tok/s number from a repetitive/degenerate workload.
- Do not measure only GPU kernel time and call it decode time.
- Do not merge an approximate context optimization without the combined ≤1% quality qualification.
- Not an IDE, autonomous agent loop, shell orchestrator, patch/git-commit engine, generic
  training/quantization toolkit, generic tensor framework, or vector-database product (spec §2). Coding
  clients (Claude Code, Codex, opencode via `--open`) remain responsible for editing files, running
  commands, and agent loops — TQF only serves the model and retrieval.

## Product command surface (spec §3 — user-facing, must stay this simple)

```
tqf | tqf --headless | tqf --memory 8G | tqf --context 1M | tqf --enable-vision
tqf --host 0.0.0.0 | tqf --model ./compatible-qwen36-q4.gguf
tqf sync . | tqf unsync . | tqf --open {opencode,claude,codex}
tqf status | tqf doctor | tqf optimize
```

Hard acceptance contract (spec §4): 4 GiB default memory budget (2 GiB experimental), 128K initial /
~1M target context, ≤1% quality degradation, ≥15 tok/s sustained decode floor, base M4 (Metal) as the
primary reference backend and RTX 3070 Ti (CUDA) as the mandatory Linux reference, one binary /
one crate, no helper executables.
