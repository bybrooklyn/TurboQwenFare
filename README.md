# TurboQwenFare

A local AI server that runs one large model well, in a small amount of memory.

You start it, point any Ollama- or OpenAI-compatible app at it, and use it. It runs
entirely on your machine — nothing is sent anywhere.

The model is Qwen3.6-35B-A3B. Normally a model that size needs far more memory than a
laptop has. TurboQwenFare works by keeping only the parts it needs in memory at any
moment and reading the rest from disk as it goes, so it fits in about 4 GB instead of
20+.

```sh
cargo install just     # one-time, if you don't have it
just build
just serve
```

That starts the server without downloading anything, so you can check that your apps
can talk to it. When you're ready for the real thing:

```sh
just serve-real        # downloads and prepares the model (~20 GB, takes a while)
```

## Using it

The server listens on `http://127.0.0.1:11434` — the same address Ollama uses. Most
apps that already work with Ollama or with OpenAI will work if you point them here and
pick the model `qwen3.6:35b`.

```sh
curl http://127.0.0.1:11434/api/chat -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6:35b",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

Both API styles are available at once, so you don't have to care which one your app
speaks:

| Style | What's available |
|---|---|
| Ollama | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` |
| OpenAI | `/v1/chat/completions`, `/v1/responses`, `/v1/models` |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |
| Built in | `/health`, `/tqf/status`, `/tqf/metrics`, `/tqf/memory`, `/tqf/context` |

## Commands

```
tqf                       start the server (and the desktop app, on macOS)
tqf --headless            start the server only
tqf --memory 8G           give it more memory to work with
tqf --context 1M          allow much longer conversations
tqf --port 11435          listen somewhere else
tqf --model ./mine.gguf   use a model file you already have

tqf sync .                index a folder so the model can search it
tqf unsync .              stop indexing a folder
tqf --open claude         launch a coding client wired to this server
tqf status                what's installed and what's running
tqf doctor                check this machine for problems
tqf optimize              measure this machine's speed
```

## Connecting a coding client

`tqf --open claude` (or `opencode`, or `codex`) starts the server, writes a temporary
config pointing that client at it, launches it, and deletes the config when it exits.
Your own client configuration is never touched.

## If something's wrong

Run `tqf doctor`. It checks the things that actually go wrong — no disk space, a
half-finished download, another program already using the port — and tells you what to
do about each one.

The most common surprise: if you already have Ollama running, it owns port 11434, so
TurboQwenFare moves to 11435 and says so at startup. Your apps will still be talking to
Ollama unless you point them at the new address.

## Working on it

```sh
just              list every task
just ci           format, lint, and test — what CI runs
just smoke-ollama check a running server end to end
just verify-real  the full acceptance, against the real checkpoint
```

---

# Technical details

Everything below is implementation detail. You don't need any of it to use the server.

The normative design document is
[`TurboQwenFare_Master_v2_All_Encompassing_Specification.md`](TurboQwenFare_Master_v2_All_Encompassing_Specification.md);
`§` references below point into it. [`AGENTS.md`](AGENTS.md) is the working guide for
contributors.

## Why it's built this way

Qwen3.6-35B-A3B is a mixture-of-experts model: only a small fraction of its weights are
used for any given token. That makes it a candidate for running *out of core* — keeping
a working set resident and streaming the rest from SSD — rather than loading 20 GB into
RAM.

That single decision produces most of the unusual structure in this repository:

- **`memory/`** — a broker that is the single source of truth for what may be resident.
  `--memory` is a hard live working-set budget, not an advisory cache size. Every large
  allocation registers with the broker *before* it happens; "allocate, then report" is
  explicitly prohibited (§115).
- **`format/tqf/`** — a custom container format, because the runtime needs to read one
  expert's weights without touching the rest of the file.
- **`experts/`** — a bounded cache with pluggable eviction, fed by parallel reads.
- **`context/{tqkv,tqattn,prefix}/`** — custom KV-cache representations, because a
  128K-token context in BF16 does not fit alongside everything else.

## Layout

One Cargo crate, not a workspace (§23). Boundaries are enforced by module structure and
a dependency firewall (§24) rather than by crate separation: `model`/`runtime` may not
depend on `retrieval` or `gui`, and the inference core must stay valid with both
entirely disabled.

```
Clients (OpenAI/Anthropic/Ollama/GUI/MCP)
  → Request Normalizer → optional retrieval → Session Scheduler
  → Qwen3.6 Execution Core (TQKV/TQAttn, Expert Runtime, Prefix Runtime)
  → Memory Broker → Metal/CUDA, CPU SIMD, SSD
```

Every protocol is normalized at the HTTP boundary into one internal
`NormalizedRequest`/`SamplingParams` before touching the model loop (§22, §153).
Protocol framing never reaches the decoder.

## Building

`cargo build` works on every supported platform. The compute backend is chosen at
compile time and resolves per target:

| Target | Backend | Notes |
|---|---|---|
| macOS | Metal | `metal-sys` is a macOS-only dependency |
| Linux / other | reference | portable scalar path; CUDA (§322-323) is not implemented |

The `metal` feature is in `default` but is a no-op off macOS, because `metal-sys` sits
in a `[target.'cfg(target_os = "macos")'.dependencies]` table and is absent from the
dependency graph elsewhere. `build.rs` collapses "feature enabled" and "target can use
it" into a single `tqf_metal` cfg, which is what the backend-conditional sites test.
`scripts/check-platform-backends.sh` guards this by inspecting the resolve graph per
target, and runs as part of `just ci`.

The SwiftUI GUI is opt-in (`cargo build --features gui`) because `build.rs` invokes a
Swift toolchain for it; a headless build environment needs no Swift.

## Binding and ports

Loopback on 11434 by default — Ollama's port, deliberately (§69). If it is occupied,
tqf identifies the occupant (another tqf, a real Ollama, or unknown) and falls back to
11435 with a printed explanation. An explicit `--port` is never silently relocated: a
busy explicit port is an error, because a client was told to use it.

Loopback is unauthenticated, matching Ollama's local convention. A non-loopback bind
mints an API key and requires it, unless `--insecure` is passed (§74). The router
enforces this by construction: everything that can generate, embed, or enumerate sits
behind the key, and only fixed-content liveness probes (`/`, `/api/version`, `/health`,
`/v1/tqf/metrics`) answer without one.

## Sampling and streaming

`sampling/` implements temperature, `top_k`, `top_p`, `min_p`, repetition/frequency/
presence penalties, stop sequences, and a seeded xoshiro256++ generator.

Greedy decode is a structurally separate path: `Sampler::Greedy` returns the argmax the
decode loop already computed for its diagnostics, and no floating-point operation
touches the logits on that path. This is what keeps every oracle-parity qualification
record valid — those compare TQF's greedy token sequence against an independent
runtime, so perturbing selection by one ULP would silently invalidate all of them.
`temperature == 0` selects it on an exact comparison, and it is the default.

Streaming emits per-token deltas from inside the `spawn_blocking` decode loop; only
decoded events cross back to the async side, so the decode loop never runs on the Tokio
executor (§25). `runtime::stream_decoder` handles the four hazards §71 names — UTF-8
codepoints split across tokens, `<think>` gating, partial `<tool_call>` blocks, and stop
sequences spanning chunk boundaries — and is model-free, so all four are unit-tested.

Ollama streams NDJSON (`application/x-ndjson`, one bare JSON object per line, no `data:`
prefix, no `[DONE]`) and terminates with a `"done": true` object. Clients stop on that
object rather than on stream close, so omitting it hangs them.

## Memory contract

4 GiB default, 2 GiB experimental (§4, §40). The broker tracks reservations per owner
class and per peak. `memory::os_sampler` samples the OS-observed footprint alongside the
broker's own accounting, because a configuration is not "4G certified" if steady-state
decode is 3.9G but admission spikes to 4.7G.

## Automation

Recipe groups mirror the spec's own CI lane design (§262).

| Recipe | Lane | Needs |
|---|---|---|
| `just ci` | A | nothing — no GPU, no model, no network |
| `just conformance` | A | nothing — protocol fixtures only |
| `just build-gui` | B | macOS + Swift toolchain |
| `just qual-*` | D | the real pinned checkpoint |
| `just verify-real` | D | the checkpoint; runs the whole acceptance as one command |
| `just qual-all` | E | everything |

`just conformance` runs the §260 fixtures alone. They are written from the
specification and never from the implementation (§331): a test derived from current
behavior cannot fail, which is how a passing test asserting `temperature: 0.7`
returns 400 came to encode a limitation as the requirement.

`just verify-real` is the only way to check the three things `just ci` structurally
cannot: greedy parity against the pinned oracle, and both smoke suites against real
generation instead of an honest `503`. It preflights and names exactly which
artifacts are missing. It does not cover §289's exit gate — an unmodified
third-party client completing a conversation — and says so, because no script can
assert that.

`just test` runs single-threaded: a few tests mutate process-global environment
variables, which is unsound under the default parallel harness.

Checkpoint paths for the qualification recipes live in an untracked `.env`
(`just env-template` writes a stub). `just dead-code` reports how much of the crate the
product surface does not yet reach — that number should go down, never up.

## Status

Three different things are worth distinguishing, because this repository has a lot of
work that is finished and measured but not yet reachable from the product:

- **Implemented** — the code exists and is tested.
- **Qualified** — measured against a real checkpoint or an independent oracle, with a
  record in [`docs/research/qualification/`](docs/research/qualification/).
- **Wired** — reachable from the CLI or an HTTP endpoint.

| Area | Implemented | Qualified | Wired | Notes |
|---|:---:|:---:|:---:|---|
| Container format (`.tqf`) | ✅ | ✅ | ✅ | Format major frozen at 1 |
| Expert streaming + cache | ✅ | ✅ | ✅ | LRU default after a measured A/B ([Phase 21](docs/research/qualification/raw-a-128-route-trace-policy.md)) |
| Parallel expert I/O | ✅ | ✅ | ✅ | 29.5x over serial on real misses ([Phase 19](docs/research/qualification/phase-20-gpu-resident-expert.md)) |
| Chunked prefill | ✅ | ✅ | ✅ | 1.81x TTFT ([Phase 26](docs/research/qualification/phase-26-prefill.md)) |
| NEON SIMD kernels | ✅ | ✅ | ✅ | aarch64 only; 10x decode ([Phase 25](docs/research/qualification/phase-25-m4-assault.md)) |
| Memory broker | ✅ | ✅ | ✅ | 4G certified ([Phase 24](docs/research/qualification/phase-24-4g-broker-certification.md)) |
| Sampling | ✅ | — | ✅ | Greedy parity is structural; oracle run pending |
| Incremental streaming | ✅ | — | ✅ | OpenAI SSE + Ollama NDJSON |
| OpenAI surface | ✅ | — | ✅ | Chat Completions, Responses, models |
| Ollama surface | ✅ | — | ✅ | Chat, generate, tags, show, ps, version |
| Anthropic surface | ✅ | — | ✅ | Messages + streaming + token counting |
| `tqf status` / `doctor` | ✅ | — | ✅ | |
| `tqf sync` / `unsync` | ✅ | — | ✅ | Builds a real index; nothing persists it yet |
| Native `/tqf/*` API | ✅ | — | ✅ | §211 status, memory, context, indexes |
| TQKV quantized KV | ✅ | ✅ | opt-in | `TQF_TQKV_ENABLED` ([Phase 27](docs/research/qualification/phase-27-tqkv-baseline.md)) |
| Prefix snapshot store | ✅ | ✅ | opt-in | 1,963x over recompute ([Phase 30](docs/research/qualification/phase-30-prefix-store.md)) |
| Predictive prefetch | ✅ | ✅ | opt-in | Off by default — measured net-negative traffic ([Phase 23](docs/research/qualification/phase-23-predictive-prefetch.md)) |
| GPU-resident experts | ✅ | ✅ | opt-in | Off by default — 0.96x, a recorded negative ([Phase 20](docs/research/qualification/phase-20-gpu-resident-expert.md)) |
| TQAttn selective attention | ✅ | ✅ | ❌ | 10.7x at 16K, 649x at 1M ([Phase 32](docs/research/qualification/phase-32-tqattn.md)) |
| Retrieval (scan/lexical/semantic/hybrid) | ✅ | ✅ | partial | `tqf sync` builds one; no persisted `.tqi` ([Phases 35-42](docs/research/qualification/phase-40-hybrid-retrieval.md)) |
| MCP server | ✅ | ✅ | ✅ | `tqf --mcp-stdio`, launched by `--open` ([Phase 44](docs/research/qualification/phase-44-automatic-rag-mcp.md)) |
| Client launchers (`--open`) | ✅ | ✅ | ✅ | [Phase 45](docs/research/qualification/phase-45-client-launchers.md) |
| Embedding model (pplx) | ✅ | ✅ | ❌ | No pinned artifact, so nothing to install ([Phase 37](docs/research/qualification/phase-37-pplx-helper-runtime.md)) |
| Reranker (GTE) | ✅ | ✅ | ❌ | [Phase 43](docs/research/qualification/phase-43-gte-reranker.md) |
| Vision encoder | ✅ | ✅ | ❌ | Oracle-validated; request path not wired ([Phase 48](docs/research/qualification/phase-48-vision-encoder.md)) |
| SwiftUI GUI | ✅ | — | opt-in | `--features gui`, macOS only ([Phase 46](docs/research/qualification/phase-46-swiftui-bridge.md)) |
| CUDA backend | ❌ | — | ❌ | §322-323 not attempted |
| MTP speculative decode | partial | ✅ | ❌ | Sidecar checkpoint not installed ([Phase 33](docs/research/qualification/phase-33-mtp.md)) |

Endpoints backed by unwired work return an explicit `501` naming what is missing, rather
than degrading silently.

Two acceptance targets from §4 are **not** met and are worth stating plainly: the
≥15 tok/s sustained decode floor (Phase 25 measured 2.34 s/token, I/O-bound on the test
hardware) and the ≤1% combined quality qualification. The 512-token exact-match gate
also does not close as literally worded — see
[the divergence investigation](docs/research/qualification/raw-a-512-divergence-investigation.md),
which characterizes it as ordinary floating-point non-associativity rather than a defect.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) — the latter carries the
adopted NVMAI attribution and a full dependency license inventory (277 runtime
dependencies, zero copyleft).
