#!/usr/bin/env bash
# Curl the Ollama-compatible surface of a running tqf and assert the wire
# shapes real clients depend on (spec §73, §210).
#
# This is deliberately a real HTTP check rather than another in-process
# test: the failure mode this whole surface exists to prevent is "curl
# passes, Open WebUI still breaks", and only real framing catches it.
#
#   just serve            # in one terminal
#   just smoke-ollama     # in another
set -uo pipefail

BASE="${1:-http://127.0.0.1:11434}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# check <description> <command...>
check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$desc"; else bad "$desc"; fi
}

body()   { curl -fsS --max-time 120 "$@" 2>/dev/null; }
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$@"; }

printf '\033[1msmoke-ollama\033[0m against %s\n' "$BASE"

note 'liveness (must work with no auth — clients probe before credentials)'
if body "$BASE/" | grep -q 'Ollama is running'; then
    ok 'GET / returns "Ollama is running"'
else
    bad 'GET / returns "Ollama is running"'
fi
check 'HEAD / succeeds'                curl -fsS -I --max-time 30 "$BASE/"
if body "$BASE/api/version" | grep -q '"version"'; then
    ok 'GET /api/version returns a version'
else
    bad 'GET /api/version returns a version'
fi

note 'inventory'
for ep in /api/tags /api/ps; do
    if body "$BASE$ep" | grep -q '"models"'; then
        ok "GET $ep returns a models array"
    else
        bad "GET $ep returns a models array"
    fi
done
if body "$BASE/api/show" -d '{"model":"qwen3.6:35b"}' | grep -q '"details"'; then
    ok 'POST /api/show returns details for an Ollama-style tag'
else
    bad 'POST /api/show returns details for an Ollama-style tag'
fi

note 'model-management endpoints are an honest 501, not an anonymous 404'
code="$(status "$BASE/api/pull" -d '{"name":"llama3"}')"
[ "$code" = "501" ] && ok "POST /api/pull -> 501 (got $code)" \
                    || bad "POST /api/pull -> 501 (got $code)"

note 'generation: non-streaming'
NS="$TMP/nonstream.json"
body "$BASE/api/chat" -d '{
  "model":"qwen3.6:35b","stream":false,
  "messages":[{"role":"user","content":"Say hi in three words."}],
  "options":{"temperature":0,"num_predict":16}}' > "$NS"
if [ -s "$NS" ]; then
    check 'response is a single valid JSON object' python3 -c "import json,sys; json.load(open('$NS'))"
    grep -q '"done":true'  "$NS" && ok 'carries done:true'          || bad 'carries done:true'
    grep -q '"message"'    "$NS" && ok 'nests message{role,content}' || bad 'nests message{role,content}'
    grep -q '"created_at"' "$NS" && ok 'carries created_at'          || bad 'carries created_at'
    grep -q '"eval_count"' "$NS" && ok 'carries eval_count timings'  || bad 'carries eval_count timings'
else
    bad 'POST /api/chat (stream:false) returned a body'
fi

note 'generation: NDJSON streaming — the framing Ollama clients parse'
HDR="$TMP/headers.txt"; ND="$TMP/stream.ndjson"
curl -fsS -N --max-time 300 -D "$HDR" "$BASE/api/chat" -d '{
  "model":"qwen3.6:35b",
  "messages":[{"role":"user","content":"Count to five."}],
  "options":{"temperature":0,"num_predict":32}}' > "$ND" 2>/dev/null

if [ -s "$ND" ]; then
    grep -qi 'application/x-ndjson' "$HDR" \
        && ok 'content-type is application/x-ndjson' \
        || bad "content-type is application/x-ndjson (got: $(grep -i '^content-type' "$HDR" | tr -d '\r'))"

    lines=$(wc -l < "$ND")
    [ "$lines" -gt 1 ] \
        && ok "streamed $lines lines (incremental)" \
        || bad "streamed $lines line — generation is not incremental"

    grep -q '^data: '  "$ND" && bad 'SSE "data:" framing leaked into NDJSON' || ok 'no SSE data: prefix'
    grep -q '\[DONE\]' "$ND" && bad 'SSE [DONE] sentinel leaked into NDJSON' || ok 'no [DONE] sentinel'

    if python3 -c "
import json,sys
for n, line in enumerate(open('$ND'), 1):
    line = line.strip()
    if line:
        json.loads(line)
" 2>/dev/null; then
        ok 'every line is a bare JSON object'
    else
        bad 'every line is a bare JSON object'
    fi

    head -1 "$ND" | grep -q '"done":false' && ok 'first line has done:false' || bad 'first line has done:false'
    tail -1 "$ND" | grep -q '"done":true'  && ok 'terminal line has done:true (clients hang without it)' \
                                           || bad 'terminal line has done:true (clients hang without it)'
    tail -1 "$ND" | grep -q '"done_reason"' && ok 'terminal line has done_reason' || bad 'terminal line has done_reason'
else
    bad 'POST /api/chat (streaming) returned a body'
fi

note 'generate uses "response", not "message"'
if body "$BASE/api/generate" -d '{"model":"qwen3.6:35b","prompt":"Hello","stream":false,"options":{"num_predict":8}}' \
     | grep -q '"response"'; then
    ok 'POST /api/generate returns a response field'
else
    bad 'POST /api/generate returns a response field'
fi

note 'sampling knobs real clients send by default are accepted'
code="$(status "$BASE/api/chat" -d '{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"temperature":0.8,"top_p":0.9,"top_k":40,"num_predict":8}}')"
[ "$code" = "200" ] && ok "temperature 0.8 / top_p 0.9 / top_k 40 accepted (got $code)" \
                    || bad "temperature 0.8 / top_p 0.9 / top_k 40 accepted (got $code)"

printf '\n\033[1m%d passed, %d failed\033[0m\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
