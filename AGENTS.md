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
- **Phase 20 (Metal perf ports):** broker-registered Metal buffer allocation
  (`MetalContext::allocate_broker_buffer*`, closing a previously-documented gap) and
  `backend::metal::expert::GpuResidentExpert` (uploads one expert's Q4_K weights to persistent GPU
  buffers once, reused across forward calls instead of re-uploading per matvec), now wired into the
  live decode loop: `WholeExpertLfuCache` entries are `ExpertValue::{Cpu,Gpu}`
  (`TQF_EXPERT_GPU_RESIDENT`, sole-backing-store accounting) and the streaming site calls
  `forward_expert`. The NVMAI-style 16-row threadgroup-staged fused GEMV
  (`tqf_q4k_gemv_staged16`, `TQF_EXPERT_GPU_KERNEL`) exists with one-hot per-element parity
  against the CPU dequant oracle plus single-GEMV and chained-forward parity tests; on the real
  canonical checkpoint it measured **2.07x per-expert-forward wall time vs the reference kernel
  (1.94 ms vs 4.01 ms) with effectively exact parity**, and is benchmark-selected as the GPU
  path's default kernel. Full record: `docs/research/qualification/phase-20-gpu-resident-expert.md`.
  Remaining Phase 20 ports: function-constant shape specialization, GDN four-way projection
  fusion; a decode-loop A/B (not just the isolated microbenchmark) is still owed before any
  default-path flip, and the GPU path itself stays opt-in.
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
independent oracle rather than a defect, but the literal exit gate as worded does not close), OS-observed
4 GiB qualification, the >=15 tok/s floor, combined <=1% quality qualification, plain GUI startup, and
RTX 3070 Ti/CUDA qualification. Phase 20's kernel fusion and live-loop wiring, and Phase 22 and later,
are not started. Check the current code, tests, and `docs/research/canonical-source-manifest.md` before
making a stronger status claim.

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
