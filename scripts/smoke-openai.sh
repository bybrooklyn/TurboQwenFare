#!/usr/bin/env bash
# Curl the OpenAI-compatible surface of a running tqf (spec §70).
set -uo pipefail

BASE="${1:-http://127.0.0.1:11434}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '\n\033[1m%s\033[0m\n' "$1"; }
body()   { curl -fsS --max-time 120 "$@" 2>/dev/null; }
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$@"; }

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
else
    bad 'POST /v1/chat/completions returned a body'
fi

note 'sampling parameters real clients send'
code="$(status "$BASE/v1/chat/completions" -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"temperature":0.8,"top_p":0.9,"max_tokens":16}')"
[ "$code" = "200" ] && ok "temperature 0.8 accepted (got $code)" || bad "temperature 0.8 accepted (got $code)"
code="$(status "$BASE/v1/chat/completions" -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":512}')"
[ "$code" = "200" ] && ok "max_tokens 512 accepted (got $code)" || bad "max_tokens 512 accepted (got $code)"

note 'an Ollama-style tag is accepted here too'
code="$(status "$BASE/v1/chat/completions" -d '{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}],"max_tokens":8}')"
[ "$code" = "200" ] && ok "model \"qwen3.6:35b\" accepted (got $code)" || bad "model \"qwen3.6:35b\" accepted (got $code)"

note 'SSE streaming'
S="$TMP/stream.sse"
curl -fsS -N --max-time 300 "$BASE/v1/chat/completions" -d '{
  "model":"qwen3.6-35b-a3b",
  "messages":[{"role":"user","content":"Count to five."}],
  "stream":true,"max_tokens":32}' > "$S" 2>/dev/null
chunks=$(grep -c '^data: ' "$S" 2>/dev/null || echo 0)
[ "$chunks" -gt 3 ] \
    && ok "streamed $chunks SSE chunks (incremental)" \
    || bad "streamed $chunks SSE chunks — generation is not incremental"
grep -q '\[DONE\]' "$S" && ok 'terminated with [DONE]' || bad 'terminated with [DONE]'

printf '\n\033[1m%d passed, %d failed\033[0m\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
