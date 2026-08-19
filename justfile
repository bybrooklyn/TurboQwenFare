# TurboQwenFare task runner.
#
#   Install `just` once:  cargo install just   (or: brew install just / apt install just)
#   Then:                 just                 (lists every recipe)
#
# Recipes are grouped to match the spec's own CI lane design (§262):
#   Lane A  portable fast     — `just ci`, no hardware or model required
#   Lane B  macOS compile     — `just build-gui`
#   Lane D  model qualification — `just qual-*`, needs the real checkpoint
#   Lane E  release           — `just qual-all`
#
# Checkpoint paths for the Lane D/E recipes live in an untracked `.env`
# (see `just env-template`), so nobody has to paste 20 GB paths by hand.

set dotenv-load := true
set positional-arguments := true

# Several tests deliberately mutate process-global environment with
# `unsafe env::set_var` (config::paths, format::tqf::tiling, io) which is
# unsound under cargo's default parallel test harness. Serialize them.
test_flags := "--test-threads=1"

# List every recipe.
default:
    @just --list --unsorted

# ---------------------------------------------------------------- Lane A

# Format the whole tree.
fmt:
    cargo fmt

# Fail if anything is unformatted.
fmt-check:
    cargo fmt --check

# `dead_code` is allowed here and tracked separately by `just dead-code`:
# large parts of the crate are qualified library code not yet wired into
# the product, and burying real signals in a gate nobody can pass helps
# no one.
#
# Clippy, denying warnings.
lint:
    cargo clippy --all-targets -- -D warnings -A dead_code

# This number should go down, never up.
#
# Count how much of the crate is unreachable from the product surface.
dead-code:
    @cargo clippy --all-targets 2>&1 | grep -cE "never used|never constructed|never read" || true

# Every fixture cites the section it encodes (spec §260, §331).
#
# Protocol conformance only — written from the spec, never from the code.
conformance:
    cargo test --bin tqf conformance -- --nocapture {{test_flags}}

# The full fast suite. No GPU, no checkpoint, no network.
test:
    cargo test -- {{test_flags}}

# Run one test (or a filter) with output.
test-one filter:
    cargo test {{filter}} -- --nocapture {{test_flags}}

# List every test without running any.
test-list:
    @cargo test -- --list

# Lane A: exactly what CI runs on every push.
ci: fmt-check lint test check-backends

# ---------------------------------------------------------------- build

build:
    cargo build

release:
    cargo build --release

# Proves the platform-conditional backend wiring without needing a macOS box.
check-backends:
    ./scripts/check-platform-backends.sh

# Requires macOS and a Swift toolchain; every other target builds
# headless-only, which is why `gui` is not a default feature.
#
# Lane B: the single-binary build with the SwiftUI GUI linked in.
build-gui:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "just build-gui: the SwiftUI GUI is macOS-only (spec §96-98)." >&2
        echo "This target builds headless; use 'just build'." >&2
        exit 1
    fi
    cargo build --features gui
    nm -gU target/debug/tqf | grep -q _tqf_launch_gui \
        && echo "linked: _tqf_launch_gui present in the binary"

# ---------------------------------------------------------------- run

# Starts the HTTP surface with no model installed, so every protocol
# endpoint is exercisable without the 20 GB checkpoint. Generation
# endpoints answer with an honest 503 rather than fake output.
#
# Development server (no checkpoint required).
serve host="127.0.0.1" port="11434":
    TQF_DEV_UNSAFE_SKIP_MODEL_CHECK=1 \
      cargo run -- --headless --host {{host}} --port {{port}}

# Downloads and converts the pinned checkpoint on first run (~20 GB),
# then serves from it.
#
# The real bounded Qwen3.6 server.
serve-real memory="4G" context="128K":
    cargo run --release -- --headless --yes --memory {{memory}} --context {{context}}

status:
    cargo run -- status

doctor:
    cargo run -- doctor

# ---------------------------------------------------------------- smoke

# Checks the wire shapes real clients depend on. Start a server first
# with `just serve` (or `just serve-real`) in another terminal.
#
# Curl every Ollama-compatible endpoint against a running server.
smoke-ollama base="http://127.0.0.1:11434":
    ./scripts/smoke-ollama.sh {{base}}

# Same for the OpenAI-compatible surface.
smoke-openai base="http://127.0.0.1:11434":
    ./scripts/smoke-openai.sh {{base}}

# ------------------------------------------------- Lane D: qualification
#
# These need the real pinned checkpoint. Set the paths in `.env`:
#   TQF_CANONICAL_GGUF=/path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf
#   TQF_CANONICAL_TQF=/path/to/qwen3.6-35b-a3b.tqf
#   TQF_CANONICAL_ORACLE=/path/to/oracle.json

# Write a .env template with every checkpoint variable these recipes read.
env-template:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -e .env ]]; then echo ".env already exists; not overwriting." >&2; exit 1; fi
    cat > .env <<'EOF'
    # Real pinned Qwen3.6 artifacts (Lane D/E). Untracked on purpose.
    TQF_CANONICAL_GGUF=
    TQF_CANONICAL_TQF=
    TQF_CANONICAL_ORACLE=
    TQF_QUALIFICATION_ROUTE_TRACE=docs/research/qualification/raw-a-128-route-trace.json
    # Helper models (Phases 37/43/48) — separate checkpoints.
    TQF_PPLX_SAFETENSORS=
    TQF_PPLX_TOKENIZER=
    TQF_GTE_SAFETENSORS=
    TQF_GTE_TOKENIZER=
    TQF_VISION_MMPROJ=
    EOF
    echo "wrote .env — fill in the paths you have."

# This is the guard on every sampling/decode change: it must stay
# bit-identical.
#
# Greedy-parity against the pinned external oracle.
qual-parity:
    cargo test --release -- --ignored --nocapture {{test_flags}} greedy

# Phase 24 OS-observed memory footprint against the broker's accounting.
qual-memory:
    cargo test --release -- --ignored --nocapture {{test_flags}} footprint

# Phase 27/29/31 TQKV context qualification.
qual-context:
    cargo test --release -- --ignored --nocapture {{test_flags}} tqkv

# This one needs no checkpoint — the route trace is committed in the repo.
#
# Expert cache/prefetch/tiling replays over the real 128-token route trace.
qual-experts:
    cargo test --release -- --ignored --nocapture {{test_flags}} experts::

# Preflight names exactly what is missing, then runs the greedy-parity
# guard, starts the real server, and smokes both surfaces against real
# generation. This is everything `just ci` cannot reach.
#
# End-to-end acceptance against the real pinned checkpoint.
verify-real:
    ./scripts/verify-real.sh

# Lane E: every checkpoint-gated test.
qual-all:
    cargo test --release -- --ignored --nocapture {{test_flags}}

# Long isolated-attention scaling benchmarks (Phases 29/31/49).
bench:
    cargo test --release -- --ignored --nocapture {{test_flags}} scaling_bench tqattn

# ---------------------------------------------------------------- misc

clean:
    cargo clean

# Everything a change should pass before it is pushed.
pre-push: fmt ci
