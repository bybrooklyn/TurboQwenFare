# Phase 44: automatic RAG + MCP

Spec Phase 44 deliverable (spec §44, §94-95, §316). "Build dynamic
context budget and read-only MCP tools. Ensure retrieval is optional
and server works normally without an index."

## What was built

**`retrieval::context_budget`** (spec §94): a pure function of Phase
40's `QueryIntent` router output — no model call, no I/O — that decides
both *whether* retrieval is worth running for a query and, if so, how
large an injection budget to allow. Narrow identifier/path lookups get
a small budget (400 tokens, 5 candidates); genuinely open-ended
natural-language questions scale up to a broad budget (6000 tokens, 16
candidates) matching spec's own example ("a simple symbol question may
need a few hundred... a cross-module architecture question may need a
much larger set"); an empty signal (nothing classified) skips retrieval
entirely, matching spec §85's "retrieval should be skipped entirely
when it is not useful."

**`mcp::`** — a real MCP (Model Context Protocol) server, implemented
against the actual specification (protocol version `2025-06-18`,
fetched live from `modelcontextprotocol.io` rather than recalled from
memory) rather than an approximation:

- `protocol.rs` — JSON-RPC 2.0 request/response/error types matching
  the real MCP wire format exactly (`initialize`'s
  `protocolVersion`/`capabilities`/`serverInfo` shape, `tools/list`'s
  `inputSchema`, `tools/call`'s `content`/`isError` result shape).
- `server.rs` — the transport-agnostic `handle_request`: `initialize`,
  `tools/list`, `tools/call`, and silently-dropped notifications (no
  response for `notifications/initialized`, per JSON-RPC 2.0 — a
  request with no `id` never gets a response).
- `stdio.rs` — the real stdio transport (spec §95: "Support both stdio
  and streamable HTTP MCP"; HTTP is not attempted this phase — see
  below): newline-delimited JSON in, newline-delimited JSON out, one
  malformed line reported as a parse error rather than killing the
  session.
- `tools.rs` — all seven read-only tools from spec §95's list
  (`tqf_search`, `tqf_symbol`, `tqf_references`, `tqf_callers`,
  `tqf_tests`, `tqf_file`, `tqf_repo_map`), backed by this session's
  real retrieval work: `tqf_search` runs Phase 40's hybrid RRF fusion
  (Exact+Lexical; no semantic lane, since an MCP tool call is exactly
  the case spec §85 says shouldn't pay to load the embedder), `tqf_
  symbol` uses Phase 36's exact lane, `tqf_file` reads from the
  in-memory indexed corpus, `tqf_repo_map` reuses Phase 41's path-
  derived module grouping.

## Scope decision: three tools honestly report a real capability gap

`tqf_references`, `tqf_callers`, and `tqf_tests` all need a real
program graph (calls/references/test-coverage edges from an AST) that
Phase 35/36/40 already decided not to build without a real parser
dependency — the same call Phase 36 made explicitly: building fake
"references" from regex/keyword matches would be worse than not
building them, since it would look authoritative while being wrong.
These three tools *are* wired into `tools/list` (so a client can
discover they exist and what they're supposed to do) but their
handlers return `isError: true` with the real reason rather than
fabricating an answer — a normal MCP tool-execution error, not a
protocol-level failure.

## Measured evidence

**Real wire-level session, not just unit-level function calls.**
`stdio_transport_handles_a_real_session` feeds a genuine newline-
delimited four-message session (`initialize` → `notifications/
initialized` → `tools/list` → `tools/call`) through the actual
`run_stdio_loop`, and confirms exactly three response lines come back
(the notification correctly produces none) with correct real content.

**"Server works normally without an index," proven directly, not just
claimed:** `server_works_normally_with_no_index_built` calls all four
data-touching tools with `IndexState: None` and confirms every one
returns an ordinary (`isError: false`) result with a clear "no index
built yet, run `tqf sync`" message — never a protocol-level error, and
never a Rust panic from unwrapping an absent index.

**Real answers from a real index:** `real_index_tools_return_correct_
real_answers` builds a real `IndexState` from three of this crate's
own real files and confirms `tqf_search`/`tqf_symbol`/`tqf_file`/
`tqf_repo_map` each return the actually-correct real answer (e.g.
`tqf_symbol` for `"MemoryBroker"` returns exactly `src/memory/mod.rs`,
`tqf_repo_map`'s summary genuinely contains `memory/`, `retrieval/`,
and `experts/`) through the identical `handle_request` path a real
client would use — not a separate, only-tested-in-isolation code path.

**The capability-gap tools are tested too, not just implemented:**
`graph_dependent_tools_honestly_report_the_capability_gap` confirms all
three graph-dependent tools return `isError: true` with an explanation
that actually mentions the real reason ("program graph"), so a
regression that silently started returning fabricated results (or
silently stopped explaining why) would fail the test.

## Status and remaining work

- **HTTP (streamable) MCP transport is not implemented** — spec §95
  asks for both stdio and HTTP "so `--open` can use whichever
  integration path the client supports best." `handle_request` is
  already transport-agnostic (the stdio loop is a thin wrapper around
  it), so adding an HTTP/SSE transport later doesn't need to touch any
  MCP semantics, but the actual `axum` route wiring is not attempted
  this phase.
- **`context_budget`'s thresholds/budgets are hand-picked**, not
  calibrated against any measured downstream-quality data — spec §94
  gives qualitative examples ("a few hundred or thousand tokens" vs "a
  much larger set"), not exact numbers, and this phase has no live
  decode loop to measure actual context-window cost/benefit against.
- **Not wired into an actual running server or `--open` client
  integration** — `mcp::` exists as a real, tested library surface but
  nothing in `src/main.rs`/`src/server/` spawns it yet against a real
  stdin/stdout process or HTTP listener. `IndexState` also has no
  real construction path from `tqf sync` yet (Phase 42's `SyncEngine`
  is the natural source, but wiring `SyncEngine -> IndexState` for a
  live server is not attempted this phase).
- `tqf_search` never uses the semantic lane (Phase 38's `FlatVectorStore`)
  even when one is available, since no persistent embedding service
  exists to call cheaply per MCP request — `IndexState.semantic` is
  plumbed through as `Option<&FlatVectorStore>` for exactly this reason,
  but nothing populates it yet.
