# Phase 52: release hardening

Spec Phase 52 deliverable (spec §324). "Run full fuzz/fault/memory/
protocol/quality/performance/license/clean-machine suite. Freeze
`.tqf`/`.tqi` major versions for the release and document migration
guarantees."

As literally specified, this phase assumes infrastructure this session
does not have and cannot honestly fabricate: a configured CI system
(spec §262's Lanes A-E; CLAUDE.md's own status note already records
"There is no CI workflow configured yet... don't assume CI enforces
anything today"), a second clean machine/user account for spec §267's
release smoke test, a dedicated fuzzer harness (`cargo-fuzz`/AFL) run
for any real duration, and a release signing/artifact pipeline (spec
§266). None of those are attempted here as if they were. What this
phase does instead is the same honest, bounded, real-evidence pattern
every other phase this session used: pick the sub-items that are
genuinely tractable on one development machine in one session, do them
for real, and say plainly what is left.

## What was done

**Server fuzz/security tests (spec §261) — four new real tests,
`src/server/security_tests.rs`,** against the real running axum
server over raw TCP (not mocked parsing):

- `giant_message_body_is_rejected_not_hung_or_crashed` — a real 32 MB
  message body. Confirmed empirically (not assumed): axum's `Json`
  extractor's default 2 MB body limit rejects it fast enough that the
  test client's own `write_all` sometimes hits `BrokenPipe` before
  finishing the upload — the server closes the connection immediately
  rather than buffering an unbounded body, which is exactly the
  behavior spec §261 asks for. The test tolerates that race
  deliberately (a new `raw_request` helper that swallows write errors)
  rather than treating a fast rejection as a test bug.
- `invalid_utf8_body_is_rejected_and_server_stays_healthy` — a body
  with a raw `0xFF 0xFE` byte sequence inside a JSON string. Confirms
  both the 4xx rejection *and* that the single-generation-slot
  scheduler is not left wedged afterward (a follow-up `/health`
  request still returns 200).
- `api_key_gate_rejects_missing_or_wrong_bearer_and_accepts_the_right_
  one` — spec §268's "non-loopback API uses auth by default"
  (`src/server/auth.rs`'s `require_api_key`) had **zero test coverage
  before this phase**. Real requests against a real router built with
  a real `api_key: Some(...)` state: no header, wrong scheme
  (`Basic` instead of `Bearer`), wrong token, and the correct token —
  plus confirms `/health` stays reachable unauthenticated (spec §268
  gates the API, not liveness probing).
- `oversized_header_value_is_rejected_not_hung` — a 1 MB
  `Authorization` header value; confirms hyper's own header-size
  handling resolves in well under the test's 3-second bound rather
  than hanging.

Required a small, real test-harness change: `test_router` gained an
`api_key: Option<&str>` parameter (previously hardcoded to `None`, so
the auth middleware was structurally untestable) and `spawn_test_
server_with_api_key` was added — both call sites in the existing
`tests.rs` updated, zero behavior change to any pre-existing test.

**License (spec §266, §324's "license" item) — real gaps, fixed, not
just noted.** `Cargo.toml` already declared `license = "Apache-2.0"`
but the repository had **no `LICENSE` file at all** — a real,
concrete compliance gap for anyone building a release artifact from
this source. Fixed:

- `LICENSE`: the real, complete, unmodified Apache License 2.0 text.
- `NOTICE`: TQF's own copyright header, a pointer to the existing
  `swift/NOTICE.md`/`swift/NVMAI-LICENSE.txt` NVMAI attribution
  (already real, from Phase 46), and a **real, machine-generated**
  third-party dependency license inventory — `cargo license
  --avoid-dev-deps --avoid-build-deps` against the actual `Cargo.lock`
  (277 direct+transitive runtime dependencies, snapshotted
  2026-08-19). Not hand-curated: every license string in `NOTICE` is
  exactly what `cargo-license` reports for this exact dependency tree.
  **No copyleft dependency exists anywhere in the tree** — every
  license present is permissive (Apache-2.0, MIT, BSD-2/3-Clause, ISC,
  CC0-1.0, Unicode-3.0, CDLA-Permissive-2.0, BSL-1.0, MIT-0, Zlib, or a
  per-crate OR-choice among those).

## Format version freeze and migration guarantees (spec §324)

**`.tqf`:** `FORMAT_MAJOR = 1, FORMAT_MINOR = 0`
(`src/format/tqf/superblock.rs`) is the frozen major version for this
release. The reader's real compatibility check
(`Superblock::decode`, step 3 of spec §121's validation order)
already enforces the guarantee this freeze promises: any future
`.tqf` file whose `format_major != 1` is rejected outright
(`ContainerError::UnsupportedMajorVersion`) rather than partially
parsed — the container format's own forward-compatibility contract is
"a major-version bump means old readers refuse the file, cleanly,
rather than misinterpret it." `format_minor` is decoded and stored but
not currently gated on by the reader; the intended (not yet exercised,
since minor version has never incremented) guarantee is that
minor-version bumps stay additive/backward-compatible within major
version 1 — a genuinely untested claim until a real 1.1 extent kind
or section actually ships, honestly flagged rather than asserted as
proven.

**`.tqi`:** cannot be frozen because **it does not exist yet as a real
persisted format.** Phase 42's own qualification doc already recorded
this precisely: "No durable journal/generation-pointer commit — same
scope boundary Phase 36+ have kept pending a persisted `.tqi` format."
`retrieval::sync`'s live-index state today is in-memory only
(`FileTable`, `SyncEngine`). Freezing a version number for a format
with no on-disk byte layout, no reader, and no writer would be
theater, not a guarantee — so this phase records the honest status
(unfrozen because unbuilt) instead of inventing a version number to
freeze.

## What was not attempted, and why

- **CI lanes (spec §262, Lanes A-E):** no `.github/workflows` exists
  in this repository and none was added — configuring real CI is an
  infrastructure/hosting decision for the project owner, not something
  a single coding session should silently commit as a side effect of
  a "hardening" phase. The Lane A checks it would run
  (`cargo fmt --check`, `cargo clippy`, unit tests) are exactly the
  commands this session already runs by hand before every commit
  (documented in `AGENTS.md`'s own "Commands" section).
- **Clean-machine release smoke test (spec §267):** requires a second
  clean macOS user account/machine and a real packaged release
  artifact; neither exists in this environment. Not simulated.
- **Full protocol conformance fixture matrix (spec §260):** existing
  coverage in `src/server/tests.rs` (streaming SSE framing, tool
  calls, structured output, error shapes, cancellation, OpenAI
  Chat/Responses) and `src/server/{anthropic,ollama}` is real but was
  not audited item-by-item against spec §260's full golden-fixture
  list this phase; that audit is real, scoped follow-up work, not
  attempted here to avoid claiming a completeness this phase didn't
  actually check.
- **Dedicated fuzz harness (`cargo-fuzz`/AFL):** spec §261's items are
  covered here as deterministic, fast, always-run unit tests (the
  established convention in this codebase — see `src/format/gguf/
  tests.rs`'s "fuzz-target checklist" comment) rather than a
  continuously-running coverage-guided fuzzer, which needs real
  wall-clock budget and a `cargo-fuzz` nightly toolchain neither
  available nor appropriate to install for a one-off session run.
- **Performance regression gates (spec §263) and the optimization
  ledger (spec §264):** these are process/CI mechanisms (comparing a
  new run's numbers against a stored baseline on every PR) rather than
  a one-time measurement; every phase's qualification doc in this
  directory already records real numbers by hand, but no automated
  gate exists to compare future runs against them.
- **`tqf licenses` CLI subcommand / GUI licenses panel** (spec §266's
  "through `tqf licenses`/GUI"): the product command surface fixed by
  `AGENTS.md`/spec §3 does not list a `licenses` subcommand today;
  `NOTICE` satisfies "accessible alongside release" as a plain file,
  but wiring it to a CLI command or GUI panel is real, additional
  product-surface work not attempted here.
