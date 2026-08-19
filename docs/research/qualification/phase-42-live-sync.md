# Phase 42: live sync

Spec Phase 42 deliverable (spec §42, §91, §198-199, §314). "Connect
watcher to incremental generation transactions. Stress editor save
storms and watcher overflow. Search remains usable during deferred
semantic updates."

## Scope decision: semantics without durability

Spec §198's transaction diagram ends in a durable `index.journal`/
superblock-generation-pointer commit. That needs the persisted `.tqi`
index storage format every retrieval phase since Phase 36 has
explicitly deferred ("this phase proves the scoring/tokenization logic
works on real data before that storage engineering is warranted"). This
phase keeps the same boundary: the *semantics* of incremental sync —
content-hash change detection, embedding reuse, `semantic_pending`
deferral, watcher debounce/coalesce, overflow fallback to a full walk —
are built and genuinely stress-tested. Durable journal/fsync/
generation-pointer commit is not, because there is still no on-disk
index to commit to.

## What was built

`retrieval::sync`:

- **`full_correctness_walk`** (spec §198 steps 1-3): reuses Phase 35's
  real scanner/classifier, hashes every real Rust file's content with
  BLAKE3, and diffs against a `FileTable` to produce `new`/`changed`/
  `deleted`/`unchanged`.
- **`SyncEngine`**: rebuilds the cheap Lexical/Exact lane immediately
  over every live file (`apply_structural_lexical`) and marks every
  new/changed path `semantic_pending` without touching the expensive
  semantic lane — the mechanism behind spec §198's "structural/lexical
  changes can commit first... a later semantic delta fills them without
  making search unavailable." `process_pending_semantic` drains at most
  a caller-given `budget` of pending paths per call through the real
  Phase 37 embedding runtime, modeling spec §199's "pause/deprioritize
  semantic embedding under inference pressure" as an explicit budget
  parameter rather than a background thread this phase doesn't have
  anywhere real to run.
- **`DebouncedEventQueue`** (spec §199): pure, deterministic debounce/
  coalesce logic — every `record()` for a path just refreshes its
  timer; `drain_ready` emits each path once, however many times it was
  recorded, once its window elapses.
- **`BoundedEventSink`** + **`LiveWatcher`**: a fixed-capacity sink that
  deliberately *drops* events past capacity and latches an `overflowed`
  flag (spec §199: "on overflow/lost events, schedule a full
  correctness walk"), wired to a real `notify` (FSEvents on macOS,
  inotify on Linux) filesystem watcher — not a hand-rolled kqueue/
  inotify binding, added as a new dependency for exactly the kind of
  well-scoped OS integration this crate already borrows mature crates
  for (`tokio`, `axum`, `reqwest`).

## Measured evidence — and two real bugs found and fixed by testing it for real

**Editor save storm (deterministic):**
`debounce_queue_coalesces_a_repeated_burst_into_one_entry_per_path`
fires 500 rapid record events across 5 paths and confirms they coalesce
to exactly 5 pending entries, none ready before the debounce window
elapses, all 5 ready (each exactly once) after.

**Watcher overflow → full-walk recovery (deterministic):**
`overflowing_event_sink_triggers_a_correct_full_walk_fallback` pushes
50 events into a 5-capacity sink, confirms it both latches `overflowed`
and genuinely drops the other 45 (not silently growing), then proves
the actual recovery guarantee — a `full_correctness_walk` against 8
real newly-created files still detects all 8 correctly, even though the
watcher only ever captured 5 of the 50 raw hints. This is spec §199's
"watcher events are hints, not the source of truth" made concrete and
tested, not just asserted in a comment.

**Real change detection on real files:**
`real_repo_incremental_walk_detects_change_new_and_delete` seeds an
isolated snapshot with real content copied from three of this crate's
own files and proves the full new→changed→deleted lifecycle end to end
through the real walker and `SyncEngine`.

**"Search remains usable during deferred semantic updates," proven
directly:** `lexical_search_stays_usable_while_semantic_is_pending`
shows lexical/exact search over newly-changed content works
immediately with zero embeddings computed, and that a file's *previous*
committed semantic vector stays servable (stale, not deleted) across an
edit until re-embedding actually completes — not going blind for that
file in the interim.

**Real OS watcher, and two real bugs this testing found:**

1. **A genuine test-isolation race.** The first version of the
   real-repo walk test mutated files directly under this crate's own
   live `src/` tree. `cargo test` runs tests in parallel, and a second
   test doing the same thing concurrently raced it — each test's walk
   correctly reflected the filesystem at that instant, but the shared
   mutable state made results order- and timing-dependent. Not a logic
   bug: fixed by giving every scratch-mutating test its own
   `isolated_real_snapshot` (a private temp directory seeded with real
   file content, exclusively owned), not by changing the walk logic.
2. **A real FSEvents path-canonicalization bug**, caught by the
   `#[ignore]`d real-OS-watcher smoke test genuinely failing on first
   run rather than being asserted away: macOS's FSEvents backend
   reports *canonicalized* paths (`/private/var/folders/...`), not
   whatever form the caller passed to `watch()` (`/var/folders/...` —
   `TMPDIR` is itself a symlink on macOS). `LiveWatcher` was comparing
   raw paths against canonicalized ones via `strip_prefix`, so every
   real event was silently dropped — the watcher genuinely never saw
   anything, but no assertion caught it until the real-OS test was
   actually run and failed. A standalone `notify` probe program (not
   committed) confirmed the canonicalization mismatch directly:
   `watching (raw): "/var/folders/.../notify_probe_test"` vs.
   `GOT EVENT: ... paths: ["/private/var/folders/.../notify_probe_test"]`.
   Fixed by canonicalizing both the watch root and every incoming event
   path before comparing. After the fix,
   `real_os_watcher_delivers_a_real_file_write_event` passes in 0.06s.

## Status and remaining work

- No durable journal/fsync/generation-pointer commit — see the scope
  decision above; there is no persisted index yet for that to commit
  to.
- `process_pending_semantic`'s budget is caller-driven, not actually
  wired to a real decode loop's inference pressure signal — there is no
  live decode loop this session's retrieval work runs alongside.
- The real-OS-watcher test stays `#[ignore]`d despite passing reliably
  on this machine: filesystem event delivery timing is environment-
  dependent (sandboxes/CI containers can restrict or delay FSEvents/
  inotify), so it is kept as an opt-in proof of wiring rather than a
  required-to-pass CI gate, consistent with how this crate already
  gates other real-hardware/real-timing tests.
- `full_correctness_walk` recomputes a full BLAKE3 hash for every file
  on every walk (spec §198's "quick hash" prefilter step, e.g. mtime/
  size before a full content hash, is not implemented) — fine at this
  session's corpus sizes, a real optimization opportunity at repository
  scale.
