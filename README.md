# TurboQwenFare (`tqf`)

A local, bounded-memory inference server specialized around a single model:
**Qwen3.6-35B-A3B Q4**. One binary, one crate, no helper executables.

`tqf` runs the model *out of core*: experts stream from SSD through a memory broker
that is the single source of truth for what may be resident. `--memory` is a hard
live working-set budget, not an advisory cache size.

The normative design document is
[`TurboQwenFare_Master_v2_All_Encompassing_Specification.md`](TurboQwenFare_Master_v2_All_Encompassing_Specification.md).
`AGENTS.md` carries the working guide for contributors (and coding agents).

## Quick start

```sh
cargo install just          # macOS: brew install just — see "Automation" below
just build
just serve                  # HTTP surface, no checkpoint required
```

`just serve` starts the server with **no model installed**, so every protocol
endpoint is reachable and generation answers with an honest `503` instead of
fabricated output. To run the real thing:

```sh
just serve-real             # downloads + converts the pinned checkpoint (~20 GB)
```

## Building

```sh
cargo build                 # debug
cargo build --release       # LTO, codegen-units=1; panics unwind, not abort
cargo test
```

`cargo build` works on every supported platform. The compute backend is selected at
compile time and **resolves per target**:

| Target | Default backend | Notes |
|---|---|---|
| macOS | Metal | `metal-sys` is a macOS-only dependency; the `metal` feature drives it |
| Linux / other | reference | portable scalar path; CUDA (spec phases 50-51) is not implemented |

The `metal` feature is on by default but is a no-op off macOS, because `metal-sys`
lives in a `[target.'cfg(target_os = "macos")'.dependencies]` table and so is not in
the dependency graph elsewhere. The build script collapses "feature enabled" and
"target can use it" into a single `tqf_metal` cfg, which is what the ~30
backend-conditional sites in the crate test.

The SwiftUI GUI is opt-in (`cargo build --features gui`) because it invokes a Swift
toolchain from `build.rs`; a headless build environment needs no Swift.

## Serving

Default bind is loopback on **port 11434** — Ollama's port, by design (spec §69).
If that port is occupied, `tqf` reports what is holding it and moves to 11435 rather
than failing mysteriously. An explicit `--port` is never silently moved.

| Surface | Endpoints |
|---|---|
| OpenAI | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/embeddings` |
| Ollama | `GET /`, `GET /api/version`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `POST /api/chat`, `POST /api/generate`, `POST /api/embed` |
| Native | `GET /health`, `GET /v1/tqf/metrics` |

Loopback is no-auth for Ollama-like convenience. Binding to a non-loopback address
mints an API key and requires it, unless you explicitly pass `--insecure` (spec §74).

## Command surface

```
tqf | tqf --headless | tqf --memory 8G | tqf --context 1M | tqf --enable-vision
tqf --host 0.0.0.0 | tqf --port 11434 | tqf --model ./compatible-qwen36-q4.gguf
tqf sync . | tqf unsync . | tqf --open {opencode,claude,codex}
tqf status | tqf doctor | tqf optimize
```

## Automation

Everything routine is a `just` recipe. `just` with no arguments lists them.

```sh
just ci             # fmt --check + clippy -D warnings + the full fast test suite
just serve          # dev server, no checkpoint
just smoke-ollama   # curl every Ollama endpoint against a running server
just dead-code      # how much of the crate the product surface does not reach
```

The recipe groups mirror the spec's own CI lane design (§262): `just ci` is Lane A
(portable, no hardware or model), `just build-gui` is Lane B, and the `just qual-*`
recipes are Lanes D and E.

The qualification recipes need the real pinned checkpoint. Point them at it once:

```sh
just env-template   # writes an untracked .env
$EDITOR .env
just qual-parity    # greedy parity against the pinned external oracle
```

`just test` runs single-threaded: a few tests still mutate process-global
environment variables, which is unsound under the default parallel harness.

## Status

The inference core, container format, expert streaming, long-context work
(TQKV/TQAttn), retrieval, MCP, vision encoding, and the helper models are
implemented and individually qualified — see
[`docs/research/qualification/`](docs/research/qualification/) for the measured
record behind each, including the negative results.

Not everything qualified as a library is wired into the product surface yet, and
`just dead-code` reports the honest size of that gap. Known-unwired: the vision
request path, the persisted retrieval index (`.tqi`), and the CUDA backend.
Endpoints that depend on those report an explicit `501` naming what is missing
rather than degrading silently.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) — the latter carries the
adopted NVMAI attribution and a full dependency license inventory.
