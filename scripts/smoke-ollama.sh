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
skipped=0

ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail + 1)); }
skip() { printf '  \033[33mskip\033[0m %s\n' "$1"; skipped=$((skipped + 1)); }
note() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# check <description> <command...>
check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$desc"; else bad "$desc"; fi
}

# -H is required: `curl -d` defaults to form encoding, which the server
# correctly rejects with 415. Real clients always send JSON.
JSON=(-H 'Content-Type: application/json')
body()   { curl -fsS --max-time 120 "${JSON[@]}" "$@" 2>/dev/null; }
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 30 "${JSON[@]}" "$@"; }

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
show_code="$(status "$BASE/api/show" -d '{"model":"qwen3.6:35b"}')"
if [ "$show_code" = "200" ] && body "$BASE/api/show" -d '{"model":"qwen3.6:35b"}' | grep -q '"details"'; then
    ok 'POST /api/show returns details for an Ollama-style tag'
elif [ "$show_code" = "404" ] && [ "$MODEL_INSTALLED" != "true" ]; then
    skip 'POST /api/show (404: no model installed)'
else
    bad "POST /api/show returns details for an Ollama-style tag (got $show_code)"
fi
# An unknown model must be rejected, installed or not.
bogus="$(status "$BASE/api/show" -d '{"model":"llama3"}')"
[ "$bogus" = "400" ] && ok "an unknown model is rejected (got $bogus)" \
                     || bad "an unknown model is rejected (got $bogus)"

note 'model-management endpoints are an honest 501, not an anonymous 404'
code="$(status "$BASE/api/pull" -d '{"name":"llama3"}')"
[ "$code" = "501" ] && ok "POST /api/pull -> 501 (got $code)" \
                    || bad "POST /api/pull -> 501 (got $code)"

note 'generation: non-streaming'
NS="$TMP/nonstream.json"
chat_code="$(status "$BASE/api/chat" -d '{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"num_predict":8}}')"
if [ "$chat_code" = "503" ] && [ "$MODEL_INSTALLED" != "true" ]; then
    skip 'POST /api/chat (503: no model installed)'
    # The envelope must still be Ollama-shaped, flat {"error": "..."},
    # not OpenAI's nested object (spec §212).
    if curl -s "${JSON[@]}" "$BASE/api/chat" \
         -d '{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}]}' \
         | python3 -c 'import json,sys; e=json.load(sys.stdin)["error"]; sys.exit(0 if isinstance(e,str) else 1)'; then
        ok 'the 503 uses Ollama'"'"'s flat error envelope'
    else
        bad 'the 503 uses Ollama'"'"'s flat error envelope'
    fi
    : > "$NS"
else
body "$BASE/api/chat" -d '{
  "model":"qwen3.6:35b","stream":false,
  "messages":[{"role":"user","content":"Say hi in three words."}],
  "options":{"temperature":0,"num_predict":16}}' > "$NS"
fi
if [ -s "$NS" ]; then
    check 'response is a single valid JSON object' python3 -c "import json,sys; json.load(open('$NS'))"
    grep -q '"done":true'  "$NS" && ok 'carries done:true'          || bad 'carries done:true'
    grep -q '"message"'    "$NS" && ok 'nests message{role,content}' || bad 'nests message{role,content}'
    grep -q '"created_at"' "$NS" && ok 'carries created_at'          || bad 'carries created_at'
    grep -q '"eval_count"' "$NS" && ok 'carries eval_count timings'  || bad 'carries eval_count timings'
elif [ "$MODEL_INSTALLED" = "true" ]; then
    bad 'POST /api/chat (stream:false) returned a body'
fi

note 'generation: NDJSON streaming — the framing Ollama clients parse'
HDR="$TMP/headers.txt"; ND="$TMP/stream.ndjson"
curl -fsS -N --max-time 300 "${JSON[@]}" -D "$HDR" "$BASE/api/chat" -d '{
  "model":"qwen3.6:35b",
  "messages":[{"role":"user","content":"Count to five."}],
  "options":{"temperature":0,"num_predict":32}}' > "$ND" 2>/dev/null

if [ -s "$ND" ]; then
    grep -qi 'application/x-ndjson' "$HDR" \
        && ok 'content-type is application/x-ndjson' \
        || bad "content-type is application/x-ndjson (got: $(grep -i '^content-type' "$HDR" | tr -d '\r'))"

    lines=$(wc -l < "$ND")
    if [ "$MODEL_INSTALLED" != "true" ]; then
        skip "incremental line count (no model installed; saw $lines)"
    elif [ "$lines" -gt 1 ]; then
        ok "streamed $lines lines (incremental)"
    else
        bad "streamed $lines line — generation is not incremental"
    fi

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

    if [ "$MODEL_INSTALLED" != "true" ]; then
        if tail -1 "$ND" | grep -q '"error"'; then
            ok 'the stream carries an Ollama-shaped error line when no model is installed'
        else
            bad 'the stream carries an Ollama-shaped error line when no model is installed'
        fi
    else
        head -1 "$ND" | grep -q '"done":false' && ok 'first line has done:false' || bad 'first line has done:false'
        tail -1 "$ND" | grep -q '"done":true'  && ok 'terminal line has done:true (clients hang without it)' \
                                               || bad 'terminal line has done:true (clients hang without it)'
        tail -1 "$ND" | grep -q '"done_reason"' && ok 'terminal line has done_reason' || bad 'terminal line has done_reason'
    fi
else
    bad 'POST /api/chat (streaming) returned a body'
fi

note 'generate uses "response", not "message"'
gen_code="$(status "$BASE/api/generate" -d '{"model":"qwen3.6:35b","prompt":"Hello","stream":false,"options":{"num_predict":8}}')"
if [ "$gen_code" = "200" ]; then
    body "$BASE/api/generate" -d '{"model":"qwen3.6:35b","prompt":"Hello","stream":false,"options":{"num_predict":8}}' \
      | grep -q '"response"' && ok 'POST /api/generate returns a response field' \
                             || bad 'POST /api/generate returns a response field'
elif [ "$gen_code" = "503" ] && [ "$MODEL_INSTALLED" != "true" ]; then
    skip 'POST /api/generate (503: no model installed)'
else
    bad "POST /api/generate (got $gen_code)"
fi

note 'parameters that cannot be honored are rejected, not silently ignored'
for payload_desc in \
    'raw:{"model":"qwen3.6:35b","prompt":"x","raw":true}' \
    'context:{"model":"qwen3.6:35b","prompt":"x","context":[1,2,3]}' \
    'format:{"model":"qwen3.6:35b","prompt":"x","format":"json"}' \
    'mirostat:{"model":"qwen3.6:35b","prompt":"x","options":{"mirostat":2}}'
do
    what="${payload_desc%%:*}"; payload="${payload_desc#*:}"
    code="$(status "$BASE/api/generate" -d "$payload")"
    [ "$code" = "400" ] && ok "$what is rejected (got $code)" || bad "$what is rejected (got $code)"
done

note 'Ollama defaults that are no-ops are accepted, not rejected'
noop="$(status "$BASE/api/generate" -d '{"model":"qwen3.6:35b","prompt":"x","stream":false,"options":{"mirostat":0,"tfs_z":1.0,"typical_p":1.0}}')"
if [ "$noop" = "200" ] || { [ "$noop" = "503" ] && [ "$MODEL_INSTALLED" != "true" ]; }; then
    ok "mirostat:0 / tfs_z:1.0 / typical_p:1.0 accepted (got $noop)"
else
    bad "mirostat:0 / tfs_z:1.0 / typical_p:1.0 accepted (got $noop)"
fi

note 'sampling knobs real clients send by default are accepted'
generation_status 'temperature 0.8 / top_p 0.9 / top_k 40' /api/chat \
  '{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"temperature":0.8,"top_p":0.9,"top_k":40,"num_predict":8}}'
generation_status 'seed / repeat_penalty / stop' /api/chat \
  '{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"seed":42,"repeat_penalty":1.1,"stop":["\n\n"],"num_predict":8}}'
generation_status 'num_predict -1 (unbounded) and keep_alive' /api/chat \
  '{"model":"qwen3.6:35b","stream":false,"keep_alive":"5m","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":-1}}'


printf '\n\033[1m%d passed, %d failed, %d skipped\033[0m\n' "$pass" "$fail" "$skipped"
[ "$fail" -eq 0 ]
