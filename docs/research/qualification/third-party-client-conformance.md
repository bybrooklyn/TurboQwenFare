# Third-party client conformance

Spec §289's exit gate says a "useful build" is measured against an
unmodified third-party client, not against this project's own tests. Until
now nothing did that. Every check in `scripts/` was curl shaping a request
the way we believed a client shapes it — which cannot catch the class of
bug where our idea of the wire and the SDK's disagree.

Running the real SDKs found two such bugs immediately, and corrected a
claim this session had already written into a commit message.

## What the SDKs did before the fix

Measured by replaying tqf's exact prior responses from a stub HTTP server
into the real, unmodified client libraries (`openai` 3.3.1, `ollama`,
`anthropic`), so the comparison is against observed client behavior rather
than reasoning about it.

| Client | Call | Before | After |
|---|---|---|---|
| ollama-python | `chat()` | `ValidationError: message Field required` | `ResponseError`, status 503 |
| ollama-python | `chat(stream=True)` | `ResponseError`, status **-1** | `ResponseError`, status 503 |
| openai-python | `chat.completions.create(stream=True)` | `JSONDecodeError: Expecting value: line 1 column 1` | `InternalServerError`, status 503 |
| anthropic-python | `messages.create()` | `APIStatusError` with the correct body | `InternalServerError`, status 503 |

The two rows that matter are the first and third. A user whose model was
not yet installed got a pydantic schema dump naming a missing `message`
field, or a JSON decode error at character 0 — in both cases the server's
actual, actionable sentence ("no model installed yet; run `tqf` to
complete first-run setup") was either buried inside a validation
`input_value` blob or discarded entirely.

The second row is subtler and would not have been caught by reading the
body: the message survived, but the SDK reported `status_code: -1`. A
client branching on the status — retry on 503, fail fast on 400 — matches
nothing.

Anthropic's SDK was already correct, because its protocol has a
first-class in-stream `error` event and the SDK raises on it. So the fix
repaired two of three surfaces and made the third consistent with them.

### A correction

The commit that made this change said clients would "wait on a connection
that will never produce another line." That is not what the measurement
shows. None of the three hung: the connection closed, and each raised —
just with the wrong error type, or an unusable status. The "hang" failure
mode is real but belongs to a different case (a stream that delivers
content lines and then ends without `done: true`), which is why the
adapter's terminal error line now carries `done: true`. The pre-stream
failure this fix addresses is a *misreported* error, not a hang.

## What it checks now

`scripts/smoke-clients.py` (`just smoke-clients`) drives all three real
SDKs against a running server. It works with or without a model installed,
because the two states test different things:

- **No model** — that unavailability is reported as the SDK's own typed
  status error carrying our message. A `ValidationError` or
  `JSONDecodeError` is a failure by definition here: it means the SDK read
  something as success and then could not parse it.
- **Model installed** — that responses validate against each SDK's own
  response model, and that streaming yields more than one chunk (a single
  chunk means the server buffered the whole generation and emitted it at
  the end, which is what tqf used to do).

Measured on a dev server with no checkpoint: **6 passed, 0 failed**.

The installed-model half has not been run here — this container has no
checkpoint. It is the interesting half, since it exercises response-model
validation and incremental streaming, and it runs on a machine with the
real model via `just smoke-clients`.

## Why this is not in Lane A

It needs three pip packages, and pinning their versions in CI would mean
this gate tests the pinned versions rather than what users actually have.
It belongs to the same manual/hardware tier as the checkpoint-gated
`qual-*` recipes. `just smoke` runs it alongside the two curl suites
against one running server.
