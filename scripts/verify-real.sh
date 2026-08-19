#!/usr/bin/env bash
# End-to-end acceptance against the real pinned checkpoint.
#
# Everything `just ci` cannot reach lives here, because it needs the 20 GB
# checkpoint: the greedy-parity guard on the sampling path, and the two
# smoke suites run against real generation rather than an honest 503.
#
# Runs a preflight first and names exactly what is missing, because the
# alternative — a cryptic failure forty minutes into a release build — is
# how checkpoint-gated work ends up never being run at all.
#
#   just verify-real
set -uo pipefail

cd "$(dirname "$0")/.."

PORT="${TQF_VERIFY_PORT:-11438}"
BASE="http://127.0.0.1:${PORT}"
SERVER_LOG="$(mktemp)"
SERVER_PID=""

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    rm -f "$SERVER_LOG"
}
trap cleanup EXIT

bold()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
ok()    { printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()   { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
warn()  { printf '  \033[33mwarn\033[0m %s\n' "$1"; }

# ------------------------------------------------------------- preflight

bold "preflight"

missing=0
need_env() {
    local var="$1" why="$2"
    local value="${!var:-}"
    if [ -z "$value" ]; then
        bad "$var is unset — $why"
        missing=$((missing + 1))
    elif [ ! -e "$value" ]; then
        bad "$var points at a path that does not exist: $value"
        missing=$((missing + 1))
    else
        ok "$var → $value"
    fi
}

need_env TQF_CANONICAL_TQF    "the converted .tqf container the decode tests load"
need_env TQF_CANONICAL_GGUF   "the verified source GGUF (tokenizer metadata)"
need_env TQF_CANONICAL_ORACLE "the pinned external oracle to compare greedy tokens against"

if [ "$missing" -gt 0 ]; then
    cat >&2 <<'EOF'

Cannot run the real-checkpoint acceptance without those paths.

  just env-template     # writes an untracked .env
  $EDITOR .env          # fill in the paths
  just verify-real

If you have the GGUF but no converted container yet, `just serve-real`
performs the conversion once and writes a trusted receipt.
EOF
    exit 2
fi

# ------------------------------------------------- 1. greedy parity guard

bold "1/3  greedy parity against the pinned oracle"
echo "      This is the guard on the sampling work: Sampler::Greedy must return"
echo "      the same tokens the pre-sampling decode loop did, bit for bit."

if cargo test --release -- --ignored --nocapture --test-threads=1 \
       canonical_greedy_oracle_matches 2>&1 | tee /tmp/tqf-parity.log \
       | grep -qE "test result: ok\. [1-9]"; then
    ok "greedy tokens match the oracle"
    PARITY=pass
elif grep -q "0 passed" /tmp/tqf-parity.log; then
    warn "the oracle test did not run — check TQF_CANONICAL_ORACLE"
    PARITY=skipped
else
    bad "greedy parity FAILED — the sampling path changed decode output"
    echo "      Full log: /tmp/tqf-parity.log"
    PARITY=fail
fi

# --------------------------------------------- 2. real server, real tokens

bold "2/3  starting the real bounded server"

cargo build --release 2>&1 | tail -1
./target/release/tqf --headless --yes --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# First run converts the checkpoint, which is slow; allow for it.
for _ in $(seq 1 3600); do
    curl -fsS "$BASE/health" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        bad "the server exited before becoming healthy"
        tail -20 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "$BASE/health" >/dev/null 2>&1; then
    bad "the server never became healthy (waited 1h)"
    tail -20 "$SERVER_LOG"
    exit 1
fi

if curl -fsS "$BASE/health" | grep -q '"model_installed":true'; then
    ok "server healthy on $BASE with a model installed"
else
    bad "server is healthy but reports no model installed — smoke would only test wiring"
    tail -20 "$SERVER_LOG"
    exit 1
fi

# ------------------------------------------------- 3. smoke, real output

bold "3/3  protocol smoke against real generation"

./scripts/smoke-ollama.sh "$BASE"; OLLAMA=$?
./scripts/smoke-openai.sh "$BASE"; OPENAI=$?

# ----------------------------------------------------------- the verdict

bold "verdict"

verdict=0
case "$PARITY" in
    pass)    ok   "greedy parity holds against the pinned oracle" ;;
    skipped) warn "greedy parity not verified (oracle absent)"; verdict=1 ;;
    fail)    bad  "greedy parity broken"; verdict=1 ;;
esac
[ "$OLLAMA" -eq 0 ] && ok "Ollama surface conforms against real generation" \
                    || { bad "Ollama smoke failed"; verdict=1; }
[ "$OPENAI" -eq 0 ] && ok "OpenAI surface conforms against real generation" \
                    || { bad "OpenAI smoke failed"; verdict=1; }

cat <<EOF

Still not covered by this script, because it needs a human:

  spec §289's exit gate — an unmodified third-party client completing a
  real conversation. Point one at $BASE and confirm:

    OLLAMA_HOST=127.0.0.1:$PORT ollama list
    OLLAMA_HOST=127.0.0.1:$PORT ollama run qwen3.6:35b "hello"

  or open the same URL in Open WebUI / Continue. curl passing while a real
  client fails is the specific outcome that gate exists to catch.
EOF

exit "$verdict"
