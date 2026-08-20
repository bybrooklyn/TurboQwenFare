#!/usr/bin/env python3
"""Spec §289's exit gate: measure this server with unmodified third-party
clients, not with this project's own tests.

Every other check in `scripts/` is curl shaping a request the way we
believe a client does. This one imports the real SDKs and lets them shape
it, which is the only way to catch the class of bug where our idea of the
wire and the SDK's disagree — a 200 the SDK reads as success, an error
body its model validator rejects, a content type its parser chokes on.

    pip install openai ollama anthropic
    just smoke-clients                     # or: python3 scripts/smoke-clients.py URL

Runs against a server with or without a model installed. With no model,
the assertions are about how each SDK *reports* unavailability: it must
raise its own typed status error carrying our actionable message, not a
schema-validation or JSON-decode failure that buries it.
"""

import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"

GREEN, RED, YELLOW, BOLD, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
passed = failed = skipped = 0


def note(text):
    print(f"\n{BOLD}{text}{OFF}")


def ok(text):
    global passed
    passed += 1
    print(f"  {GREEN}ok{OFF}   {text}")


def bad(text):
    global failed
    failed += 1
    print(f"  {RED}FAIL{OFF} {text}")


def skip(text):
    global skipped
    skipped += 1
    print(f"  {YELLOW}skip{OFF} {text}")


def model_installed():
    import urllib.request, json
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        return json.load(r).get("model_installed") is True


INSTALLED = model_installed()
print(f"{BOLD}third-party client conformance{OFF}  base={BASE}  model_installed={INSTALLED}")


def unavailable(label, call, status_attr):
    """A call that cannot run must fail as a *status*, in the SDK's own
    error type. A ValidationError or JSONDecodeError here means the SDK
    got something it read as success and then could not parse — the user
    sees a schema dump instead of "run `tqf` to finish setup"."""
    try:
        call()
    except Exception as e:
        name = type(e).__name__
        if name in ("ValidationError", "JSONDecodeError"):
            bad(f"{label}: parse failure hides the reason ({name}: {str(e)[:80]})")
            return
        status = getattr(e, status_attr, None)
        if status == 503:
            ok(f"{label}: {name} carrying status 503")
        else:
            bad(f"{label}: {name} with status {status!r}, expected 503")
        if "no model installed" not in str(e):
            bad(f"{label}: the actionable message did not reach the client")
        return
    bad(f"{label}: returned successfully from a server with no model")


# ------------------------------------------------------------ ollama-python
note("ollama-python")
try:
    import ollama
except ImportError:
    skip("ollama not installed (pip install ollama)")
else:
    c = ollama.Client(host=BASE)
    try:
        c.list()
        ok("list() parses the model inventory")
    except Exception as e:
        bad(f"list(): {type(e).__name__}: {str(e)[:100]}")

    if INSTALLED:
        try:
            r = c.chat(model="qwen3.6:35b", messages=[{"role": "user", "content": "hi"}])
            ok("chat() validates against the SDK's own response model") if r.message.content \
                else bad("chat() returned an empty message")
        except Exception as e:
            bad(f"chat(): {type(e).__name__}: {str(e)[:100]}")
        try:
            chunks = list(c.chat(model="qwen3.6:35b",
                                 messages=[{"role": "user", "content": "Count to five."}],
                                 stream=True))
            ok(f"chat(stream=True) yielded {len(chunks)} chunks") if len(chunks) > 1 \
                else bad(f"chat(stream=True) yielded {len(chunks)} chunk — not incremental")
        except Exception as e:
            bad(f"chat(stream=True): {type(e).__name__}: {str(e)[:100]}")
    else:
        # Both framings, because they failed differently: non-streaming
        # raised a pydantic ValidationError on the missing `message`
        # field, streaming raised ResponseError with status -1.
        unavailable("chat()", lambda: c.chat(
            model="qwen3.6:35b", messages=[{"role": "user", "content": "hi"}]), "status_code")
        unavailable("chat(stream=True)", lambda: list(c.chat(
            model="qwen3.6:35b", messages=[{"role": "user", "content": "hi"}], stream=True)),
            "status_code")

# ------------------------------------------------------------ openai-python
note("openai-python")
try:
    import openai
except ImportError:
    skip("openai not installed (pip install openai)")
else:
    o = openai.OpenAI(base_url=f"{BASE}/v1", api_key="not-used-on-loopback")
    try:
        ids = [m.id for m in o.models.list()]
        ok(f"models.list() -> {ids}") if ids else bad("models.list() returned nothing")
    except Exception as e:
        bad(f"models.list(): {type(e).__name__}: {str(e)[:100]}")

    if INSTALLED:
        try:
            r = o.chat.completions.create(
                model="qwen3.6-35b-a3b", messages=[{"role": "user", "content": "hi"}])
            ok("chat.completions.create() validates") if r.choices[0].message.content \
                else bad("chat.completions.create() returned empty content")
        except Exception as e:
            bad(f"chat.completions.create(): {type(e).__name__}: {str(e)[:100]}")
        try:
            n = sum(1 for _ in o.chat.completions.create(
                model="qwen3.6-35b-a3b",
                messages=[{"role": "user", "content": "Count to five."}], stream=True))
            ok(f"streamed {n} chunks") if n > 1 else bad(f"streamed {n} chunk — not incremental")
        except Exception as e:
            bad(f"stream: {type(e).__name__}: {str(e)[:100]}")
    else:
        # Streaming used to answer 200 + SSE, and the SDK raised
        # JSONDecodeError trying to parse a plain-text `data:` payload.
        unavailable("chat.completions.create(stream=True)", lambda: list(
            o.chat.completions.create(model="qwen3.6-35b-a3b",
                                      messages=[{"role": "user", "content": "hi"}], stream=True)),
            "status_code")

# --------------------------------------------------------- anthropic-python
note("anthropic-python")
try:
    import anthropic
except ImportError:
    skip("anthropic not installed (pip install anthropic)")
else:
    a = anthropic.Anthropic(base_url=BASE, api_key="not-used-on-loopback")
    if INSTALLED:
        try:
            r = a.messages.create(model="qwen3.6-35b-a3b", max_tokens=64,
                                  messages=[{"role": "user", "content": "hi"}])
            ok("messages.create() validates") if r.content else bad("messages.create() empty")
        except Exception as e:
            bad(f"messages.create(): {type(e).__name__}: {str(e)[:100]}")
        try:
            with a.messages.stream(model="qwen3.6-35b-a3b", max_tokens=64,
                                   messages=[{"role": "user",
                                              "content": "Count to five."}]) as s:
                text = "".join(s.text_stream)
            ok("messages.stream() reassembled text") if text else bad("stream produced no text")
        except Exception as e:
            bad(f"messages.stream(): {type(e).__name__}: {str(e)[:100]}")
    else:
        unavailable("messages.create()", lambda: a.messages.create(
            model="qwen3.6-35b-a3b", max_tokens=16,
            messages=[{"role": "user", "content": "hi"}]), "status_code")

print(f"\n{BOLD}{passed} passed, {failed} failed, {skipped} skipped{OFF}")
sys.exit(1 if failed else 0)
