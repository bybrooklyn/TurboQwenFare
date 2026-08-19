# Phase 45: client launchers

Spec Phase 45 deliverable (spec §45, §99-100, §317). "Implement
OpenCode then Claude Code then Codex, each with real integration smoke
tests and no permanent config mutation. Missing-client install path
always confirms."

## What was built

`integrations::config` — ephemeral provider/MCP configuration
generation for all three clients from spec §99's table. Every
mechanism was confirmed against each client's real, live documentation
before writing any code (fetched during this phase, not recalled from
training data):

- **OpenCode**: `OPENCODE_CONFIG=<path>` env var loads a config file
  from an arbitrary path (confirmed: opencode.ai/docs/config).
- **Claude Code**: `ANTHROPIC_BASE_URL` redirects the gateway (spec's
  own citation) plus `--mcp-config <path>`, which loads an MCP config
  for one run without touching `.mcp.json` or `~/.claude.json`
  (confirmed: code.claude.com/docs/en/mcp).
- **Codex**: `CODEX_HOME=<dir>` redirects Codex's *entire* config
  directory so `$CODEX_HOME/config.toml` is never the user's real
  `~/.codex/config.toml`, with `[model_providers.tqf]` using
  `wire_api = "responses"` matching spec's own citation of the
  Responses wire API (confirmed via the OpenAI developer community and
  current Codex config documentation).

`integrations::launch` — the real process lifecycle: a read-only `PATH`
search (`find_binary_on_path`, executable-bit checked, never executes
anything), `ensure_client_available` (spec §100's confirmation gate —
offers the real official install recipe but never runs it itself, so
the actual `Command::new(installer).spawn()` stays an explicit,
visible step the caller owns), `write_ephemeral_config` (writes every
config file under a fresh, uniquely-named temp directory), and
`run_to_completion` (spawns the real child, waits, and deletes the
ephemeral directory via `Drop` on every exit path — not just the happy
one).

## Measured evidence, and a real bug found by testing with a real process

**Real spawn, not a mock.** Rather than assert the launch mechanics
"should" work, `real_spawn_observes_env_vars_and_cleans_up_on_exit`
spawns an actual child process (`sh`, standing in for a real AI client
CLI — deliberately *not* `opencode`/`claude`/`codex` themselves, since
automated tests must not launch real external coding agents) and
proves, from the child's own observed environment, that: the ephemeral
config file exists on disk before the child runs, the child's own
`$OPENCODE_CONFIG` really resolves to that file, the child can really
`cat` it from its own working directory, and the config directory is
really gone once the child exits and `WrittenConfig` drops.

**A real concurrency bug, found and fixed the same way as Phase 42's.**
The first version named each ephemeral directory `tqf-open-<pid>` —
but `cargo test` runs many tests in parallel *within one process*, so
every concurrent call shared the same PID and therefore the same
directory name. Two tests calling `write_ephemeral_config`
simultaneously raced: one's cleanup could delete the directory out
from under the other's still-running child. Reproduced deterministically
(5/5 failures before the fix, 5/5 passes after) and fixed the same way
as Phase 42's real-repo-tree race — a uniqueness fix (an atomic
counter alongside the PID), not a change to the launch logic itself,
which was already correct.

**"No permanent config mutation," checked directly, not just by
convention:** `no_client_config_ever_names_a_real_permanent_config_path`
asserts none of the three clients' generated config file paths ever
reference `.claude.json`, `.codex/config`, or an absolute path — every
one is a filename relative to a caller-owned ephemeral directory.

## Status and remaining work

- **Not wired to `tqf --open <client>` CLI parsing** — `integrations::`
  is a real, tested library surface, but nothing in `src/main.rs`/
  `src/cli` invokes it from an actual `--open` argument yet, and there
  is no live TQF HTTP server this session's work runs alongside to
  point `server_base_url` at for real.
- **"Ensure the server is running, synchronize the associated index if
  one is registered"** (the first two steps of spec §99's sequence) are
  not implemented — this phase covers config generation and process
  lifecycle only; the server-readiness check and index-sync trigger
  need the live server/sync wiring Phase 42/44 also left unattached to
  an actual running process.
- Codex's `[mcp_servers.tqf]` TOML table shape follows the general
  Codex config.toml MCP-server convention but was not independently
  re-verified against Codex's own MCP documentation the way the
  `model_providers`/`wire_api` fields were — worth a follow-up check
  before this ships against a real Codex install.
- `ensure_client_available`'s test coverage is necessarily
  environment-dependent (whether OpenCode/Codex happen to be installed
  on the machine running the tests) since it deliberately doesn't fake
  `find_binary_on_path`'s result — the tests assert the *contract*
  holds in whichever branch actually executes, rather than forcing one
  branch, to avoid mocking the one thing (a real PATH lookup) this
  function exists to do for real.
