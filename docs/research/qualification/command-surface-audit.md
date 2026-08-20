# Command-surface audit

Spec §335 says a capability is complete when a request or an invocation
reaches it — "the check is a request or an invocation, never a file
listing." This is that check applied to spec §3's whole user-facing
surface and to the three protocol surfaces, by running each one and
reading what came back.

Everything below was found by invocation. None of it was visible in the
test suite, because in each case the suite exercised the same code with
inputs that avoided the defect — or, twice, asserted the defect as the
requirement.

## What was wrong

| Surface | Symptom | Cause |
|---|---|---|
| `POST /api/*` | `415` on the exact `curl -d '{...}'` in Ollama's README | `axum::Json` requires `application/json`; real Ollama does not |
| `/api/chat`, `/v1/chat/completions`, `/v1/messages` | `200` + an in-band error for a request that could never run | all three chose to stream before checking readiness |
| `tqf status` | `--memory 8G` reported "4.00 GiB (default)" | rendered the persisted config, ignored the flags |
| `tqf status` | "index persistence is not implemented" | true when written, stale two commits later |
| `tqf --open <client>` | client started in an empty temp directory | `current_dir(&written.dir)`, to resolve `CODEX_HOME="."` |
| `--memory 1K` | accepted; server started | no floor check; failure surfaced at the first reservation |
| `--model /typo.gguf` | "no model installed and no interactive terminal…" | validated after the first-run setup gate |
| `--host not-an-ip` | same message | `InvalidHost` existed but was only raised at bind |
| `tqf sync /typo` | "No such file or directory (os error 2)" | raw errno from inside the walk |
| `tqf --headless`, no model | the same condition printed twice, two wordings | a `println!` and an error whose Display also prints |

Two of these were guarded by tests asserting the broken behavior:
`chat_completions_streaming_returns_valid_sse_framing` asserted the `200`,
and the launcher test read its config through a bare relative path, which
only resolved because the client was being relocated. Both passed. Both
were written from the code rather than from what a client needs — spec
§331's rule, which is stated for conformance fixtures and applies just as
well here.

## What was already right

Worth recording, because an audit that only lists faults implies the rest
was unexamined:

- Port conflicts. An explicit `--port` that is taken fails with the
  address and the way out. The default port taken by another process
  falls back and says, in as many words, that Ollama clients pointed at
  11434 will not reach tqf.
- `tqf --open` honors spec §100's confirm-before-install gate: with the
  client absent and no terminal, it declines rather than installing.
- `tqf doctor` reports hardware, backend, data root, disk, receipt,
  container, tokenizer, port, and memory plan, and is explicit about
  which checks it cannot perform.
- `tqf unsync` on a never-synced path explains that, naming the index
  path it looked for.
- `--memory`/`--context` reject unparseable sizes with the accepted
  suffixes.

## Third-party clients

The protocol findings came from `scripts/smoke-clients.py`
(`just smoke-clients`), which drives the real `openai`, `ollama`, and
`anthropic` libraries. Full before/after measurements, including a
correction to a claim made in a commit message here, are in
`third-party-client-conformance.md`.

## Not covered

- Anything requiring the real checkpoint. Every generation path was
  exercised in its unready state only; `just smoke-clients` and
  `just qual-*` cover the rest on a machine that has the model.
- The GUI (`--open` aside), which needs macOS and a Swift toolchain.
- `tqf optimize`, which reports honestly that it is Metal-only and this
  is a Linux container.
