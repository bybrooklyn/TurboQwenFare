#!/usr/bin/env bash
# Curl the OpenAI-compatible surface of a running tqf (spec §70).
set -uo pipefail

BASE="${1:-http://127.0.0.1:11434}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0; skipped=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail + 1)); }
skip() { printf '  \033[33mskip\033[0m %s\n' "$1"; skipped=$((skipped + 1)); }
note() { printf '\n\033[1m%s\033[0m\n' "$1"; }
# -H is required: `curl -d` defaults to form encoding, which the server
# correctly rejects with 415. Real clients always send JSON.
JSON=(-H 'Content-Type: application/json')
body()   { curl -fsS --max-time 120 "${JSON[@]}" "$@" 2>/dev/null; }
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 60 "${JSON[@]}" "$@"; }

# The dev server (`just serve`) runs with no checkpoint, so generation
# endpoints answer 503 by design. That is a pass for "the endpoint exists
# and is wired correctly" and a skip for "generation works" — conflating
# the two would either hide real 404s or cry wolf on every dev run.
MODEL_INSTALLED="$(curl -fsS --max-time 10 "$BASE/health" 2>/dev/null \
    | grep -o '"model_installed":[a-z]*' | cut -d: -f2)"
if [ "$MODEL_INSTALLED" != "true" ]; then
    printf '\033[33mnote\033[0m no model installed: generation checks verify wiring (503), not output\n'
fi

# generation_status <description> <path> <payload>
generation_status() {
    local desc="$1" path="$2" payload="$3" code
    code="$(status "$BASE$path" -d "$payload")"
    if [ "$code" = "200" ]; then
        ok "$desc"
    elif [ "$code" = "503" ] && [ "$MODEL_INSTALLED" != "true" ]; then
        skip "$desc (503: no model installed)"
    else
        bad "$desc (got $code)"
    fi
}

printf '\033[1msmoke-openai\033[0m against %s\n' "$BASE"

note 'discovery'
body "$BASE/v1/models" | grep -q 'qwen3.6-35b-a3b' \
    && ok 'GET /v1/models lists the canonical model' \
    || bad 'GET /v1/models lists the canonical model'

note 'chat completions'
R="$TMP/chat.json"
body "$BASE/v1/chat/completions" -d '{
  "model":"qwen3.6-35b-a3b",
  "messages":[{"role":"user","content":"Say hi."}],
  "max_tokens":16}' > "$R"
if [ -s "$R" ]; then
    grep -q '"choices"' "$R" && ok 'returns choices'  || bad 'returns choices'
    grep -q '"created"' "$R" && ok 'carries created'  || bad 'carries created'
    grep -q '"usage"'   "$R" && ok 'carries usage'    || bad 'carries usage'
elif [ "$MODEL_INSTALLED" != "true" ]; then
    skip 'response shape (no model installed)'
else
    bad 'POST /v1/chat/completions returned a body'
fi

note 'sampling parameters real clients send (these all used to 400)'
generation_status 'temperature 0.8 / top_p 0.9' /v1/chat/completions \
  '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"temperature":0.8,"top_p":0.9,"max_tokens":16}'
generation_status 'max_tokens 512' /v1/chat/completions \
  '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":512}'
generation_status 'stop sequences' /v1/chat/completions \
  '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"stop":["\n\n"],"max_tokens":16}'
generation_status 'seed / top_k / min_p' /v1/chat/completions \
  '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"seed":7,"top_k":40,"min_p":0.05,"temperature":0.7,"max_tokens":16}'

note 'an Ollama-style tag is accepted here too'
generation_status 'model "qwen3.6:35b"' /v1/chat/completions \
  '{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'

note 'SSE streaming'
S="$TMP/stream.sse"
# No -f: an unready server answers this with a 503 and a JSON error body,
# which is the correct behavior (a streaming request that cannot run must
# fail with a status, not a 200 stream carrying an error event). -f would
# discard that body and report it as a missing response.
STREAM_CODE="$(curl -sS -N --max-time 300 -o "$S" -w '%{http_code}' \
  "${JSON[@]}" "$BASE/v1/chat/completions" -d '{
  "model":"qwen3.6-35b-a3b",
  "messages":[{"role":"user","content":"Count to five."}],
  "stream":true,"max_tokens":32}' 2>/dev/null)"

if [ "$MODEL_INSTALLED" != "true" ]; then
    # There is no stream to inspect the framing of. What matters is that
    # the failure arrived as a status an SDK's error path can see: a 200
    # here would tell every OpenAI client the call succeeded and hand the
    # error to its stream parser instead.
    if [ "$STREAM_CODE" = "503" ]; then
        ok "streaming with no model returns 503 (got $STREAM_CODE)"
    else
        bad "streaming with no model returns 503 (got $STREAM_CODE)"
    fi
    skip 'SSE framing (no model installed)'
else
    chunks=$(grep -c '^data: ' "$S" 2>/dev/null || echo 0)
    if [ "$chunks" -gt 3 ]; then
        ok "streamed $chunks SSE chunks (incremental)"
    else
        bad "streamed $chunks SSE chunks — generation is not incremental"
    fi
    grep -q '\[DONE\]' "$S" && ok 'terminated with [DONE]' || bad 'terminated with [DONE]'
fi

printf '\n\033[1m%d passed, %d failed, %d skipped\033[0m\n' "$pass" "$fail" "$skipped"
[ "$fail" -eq 0 ]
