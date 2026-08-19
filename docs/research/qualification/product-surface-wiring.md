# Product surface: making the server actually serve

Not a spec phase. This records the work that took an inference engine with
~30 qualified phases behind it and made it behave like the product spec §3
describes: a binary a real Ollama or OpenAI client can point at and use.

The starting condition, measured rather than assumed:

- `cargo build` **failed on Linux**. `default = ["metal"]` pulled `metal-sys`
  into the graph unconditionally, so `core-graphics-types` died with
  `error[E0455]: link kind 'framework' is only supported on Apple targets`.
  Only `--no-default-features` worked, documented nowhere.
- `cargo test --no-default-features` → **383 passed, 1 failed**.
- `src/server/ollama/mod.rs` was a **one-line doc comment**, as was
  `src/server/anthropic/mod.rs` and `src/sampling/mod.rs`. The server binds
  Ollama's port 11434 by default, so every Ollama client connected
  successfully and then 404'd on every `/api/*` call.
- `--open` was parsed and never referenced; `tqf status`/`doctor`/`sync`/
  `unsync` printed "not yet implemented"; `src/mcp/` had no entrypoint.
- No `justfile`, `README.md`, CI workflow, or toolchain pin.

## Measured evidence

### The Linux build failure and its fix

`metal-sys` moved into `[target.'cfg(target_os = "macos")'.dependencies]`.
Cargo features are not target-conditional, so `feature = "metal"` stays true
on Linux; `build.rs` therefore emits a single `tqf_metal` cfg meaning "the
feature is on *and* this target can use it", and the ~30 backend-conditional
sites test that instead.

`scripts/check-platform-backends.sh` verifies the wiring by inspecting the
**resolve graph** per target. This distinction is load-bearing:
`--filter-platform` narrows `resolve` but still reports every package the
manifest mentions, so grepping the `packages` list finds `metal` on Linux and
reports a false pass.

```
ok: metal-sys absent  for x86_64-unknown-linux-gnu
ok: metal-sys absent  for aarch64-unknown-linux-gnu
ok: metal-sys present for aarch64-apple-darwin
ok: metal-sys present for x86_64-apple-darwin
```

`compiled_backend()` had been reporting `"metal"` on Linux, writing a false
fact into the persisted `HardwareProfile`. It now reports `"reference"`.

### The failing test was a real defect

`src/memory/os_sampler.rs`'s Linux branch read `/proc/self/statm`, which has
no peak field, and returned `0` for peak RSS — so the
`resident_peak >= resident` invariant every caller relies on was **false on
every Linux run**. Phase 24's entire premise is that a configuration is not
"4G certified" if steady-state is 3.9G but admission spikes to 4.7G; a sampler
reporting peak as 0 cannot make that judgement.

Fixed by reading `/proc/self/status` (`VmRSS`/`VmSize`/`VmHWM`), with
`sysconf(_SC_PAGESIZE)` replacing a hardcoded 4096 in the `statm` fallback
(aarch64 Linux and several distributions ship 16K/64K pages, which would have
scaled every reading by 4-16x).

### Greedy parity is structural, not hoped for

Real sampling (temperature, `top_k`, `top_p`, `min_p`, repetition/frequency/
presence penalties, stop sequences, seeded xoshiro256++) had to land without
disturbing greedy decode, because every oracle-parity record compares TQF's
greedy token sequence against an independent runtime.

The guarantee is structural rather than statistical: `Sampler::Greedy` is a
distinct enum arm returning the argmax the decode loop **already computed** for
its diagnostics. No floating-point operation touches the logits on that path.
`decode_greedy` keeps its exact signature at both runtimes and delegates to a
new `decode_step(token, sampler, history)`, so all 27 checkpoint-gated
qualification tests are untouched.

Two supporting facts are now pinned by tests: `top_logit_candidates` uses a
strict `>`, so **the lower token index wins a tie** (a future refactor to `>=`
would silently change every greedy run), and `SamplingParams::default()` is now
`temperature: 0.0` rather than `1.0`, so an adapter that forgets to set sampling
gets greedy instead of quietly sampling.

Evidence that sampling is real rather than asserted: 200,000 draws from a
4-way distribution track its softmax probabilities to within 0.01 per token.

| token | softmax | observed (200k draws) |
|---|---|---|
| 0 | 0.6439 | within 0.01 |
| 1 | 0.2369 | within 0.01 |
| 2 | 0.0871 | within 0.01 |
| 3 | 0.0321 | within 0.01 |

**Not verified here:** `just qual-parity` against the real 20 GB checkpoint.
The identity is structural and unit-tested, but the end-to-end oracle
comparison needs hardware this session did not have.

### Streaming was not streaming

`stream_chat_completion` awaited the *entire* generation and then emitted the
whole text as one SSE event. Every property spec §71 asks for was vacuously
true and none were tested.

The decode loop now emits per-token deltas from inside `spawn_blocking`; only
decoded events cross back via `blocking_send`, so the loop never moves onto the
Tokio executor (§25). `generate()` and `generate_streaming()` share one loop,
because two loops eventually disagree and "the streamed answer differs from the
batch answer" is its own bug class.

`runtime::stream_decoder` owns the four hazards and, being model-free, is
testable for all of them:

| Hazard | Test |
|---|---|
| UTF-8 split across tokens | multibyte text at chunk sizes 1-12, no U+FFFD |
| `<think>` gating | close tag split across chunks at sizes 1-8 |
| Partial `<tool_call>` | never leaks as visible text; unterminated → `length` |
| Stop across chunks | matches at sizes 1-6; a non-completing prefix is released |

The differential test is the load-bearing one: streamed output equals the batch
parser's output across **4 chunk sizes × 6 shapes**.

### The Ollama surface

Two framing details break every client while `curl` still looks fine:

1. **NDJSON, not SSE** — `application/x-ndjson`, one bare JSON object per line,
   no `data:` prefix, no `[DONE]`.
2. **The terminal `"done": true` object** — clients stop on it, not on stream
   close. Omitting it hangs them.

`just smoke-ollama` against a live server, with no model installed:

```
18 passed, 0 failed, 7 skipped
```

The 7 skips are generation assertions that need a real checkpoint; each is
skipped explicitly rather than silently passing on a 503.

Parameters that cannot be honored are rejected rather than ignored (`raw`,
`template`, `context`, `suffix`, `format`, images). But Ollama ships
`mirostat: 0`, `tfs_z: 1.0`, `typical_p: 1.0` as *defaults*, so the no-op value
is accepted and only a real request for those strategies is refused —
rejecting the defaults would 400 half the ecosystem.

### The auth split

Everything that can generate, embed, or enumerate sits behind the API key.
Only fixed-content liveness probes answer without one, because clients call
them before they have anywhere to put a credential.

`liveness_is_unauthenticated_but_generation_and_inventory_are_not` asserts
both halves: `/`, `/api/version`, `/health` are 200 without a key; `/api/tags`,
`/api/ps`, `/api/chat`, `/api/generate`, `/api/show`, `/api/embed` are 401.
Merging the Ollama routes at the top level is the path of least resistance and
would have exposed generation unauthenticated on a `0.0.0.0` bind (§74).

### Defects found by running things, not reading them

Every one of these was invisible to the test suite as it stood:

- **`EnvFilter::from_default_env()` with `RUST_LOG` unset enables nothing**, so
  every `tracing::warn!` in the crate was suppressed — including the
  port-fallback warning that explains why an Ollama client cannot see the
  server. Now defaults to `tqf=info`, on stderr so the MCP transport stays
  clean.
- **`TQF_DEV_UNSAFE_SKIP_MODEL_CHECK=1` did not work.** The setup flow ran
  first and either demanded confirmation or, with `--yes`, started a 20 GB
  download before the skip was consulted.
- **`BoundServer.addr` reported the requested port, not the bound one**, so
  `--port 0` handed every downstream consumer an address nothing was listening
  on. Caught by a new test asserting the explicit-port rule.
- **`/v1/messages` returned OpenAI's nested error envelope** on the
  service-unavailable path. The validation path was already correct, so the
  test written from the code passed; a live probe found it.
- **The `phase-verify` commit gate over-matched.** Its regex scanned the whole
  command string, so an ordinary commit whose *body* mentioned "(Phase 20)"
  followed by any later colon was blocked as a phase commit. Now anchored to
  the subject line, verified against six cases. It also used `shasum`, which is
  macOS-only, so the gate would have failed on every Linux phase commit.
- **A test-fixture collision.** The new `--open` launch test named its fake-binary
  directory `tqf-open-{pid}-{n}` — exactly what `write_ephemeral_config`
  generates — so that function's cleanup deleted the test's own fixture. The
  same class of bug Phases 42 and 45 each found and fixed independently.

### Real command output

`tqf sync src` on this crate's own tree:

```
scanned /home/user/TurboQwenFare/src
  161 files, 1.7 MiB, 0 ignored, in 1422 ms
classified:     161  Rust
indexed:  161 files into 161 chunks, 10152 distinct lexical terms

This index was NOT retained. Index persistence — spec §218's project registry
and the `.tqi` container that would hold this on disk — is not implemented [...]
```

A real 4-message MCP session over stdio round-trips correctly, and stdout stays
pure JSON-RPC even at `RUST_LOG=trace`.

`tqf doctor` runs 9 checks and exits nonzero only on failure, so it is usable in
a script. A check that cannot determine an answer warns rather than passing.

### Coverage

| | before | after |
|---|---|---|
| Tests passing | 383 (1 failing) | 489 |
| `cargo build` on Linux | fails | works |
| `clippy -D warnings` | not run | clean |
| Ollama endpoints | 0 | 12 |
| Dead-code findings | 517 | 445 |

The dead-code figure is the honest measure of what remains: the residue
clusters in `helper_model` (86), `context` (73), `retrieval` (58), and `vision`
(46) — exactly the four areas the README's status table marks as implemented
and qualified but not wired.

## What was deliberately not done

Each of these needs an artifact or hardware that does not exist in this build.
A stub pretending otherwise would be worse than an honest `501`.

- **Embeddings.** `helper_model` works and is oracle-validated, but
  `source::pinned` names no pplx-embed artifact, so there is no checkpoint to
  acquire. `/v1/embeddings` and `/api/embed` return a 501 naming *that*.
- **Vision.** Phase 48's encoder is oracle-validated, but `prompt_tokens`
  rejects vision input and no multimodal content-part mapping exists. Every
  adapter rejects image content explicitly rather than dropping it.
- **Persisted retrieval index.** No `.tqi` writer exists (Phase 36's own scope
  boundary). `tqf sync` builds a real index and says it is not retained.
- **CUDA.** Spec §322-323 unattempted; no NVIDIA hardware here.
- **The ≥15 tok/s floor and the ≤1% quality gate.** Unchanged by this work;
  Phase 25 and the 512-token divergence investigation remain the record.
