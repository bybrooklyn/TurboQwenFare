# `.tqi` index persistence

**Spec:** §173 (storage principles, LOCKED), §174 (superblock), §175
(generation model), §176 (stable IDs), §177 (file record), §185 (lexical
baseline), §211 (native status API), §218 (project-local layout).

**Commit:** this branch. **Machine:** Linux x86_64, 4 cores, debug build.

## What changed

`tqf sync` did a real walk and a real index build and then discarded it —
its own report said so. The index now persists to `<root>/.tqf/index.tqi`,
is registered so a server started anywhere can find it, and is served over
HTTP. This is the first time retrieval is reachable from the product at
all rather than only from tests.

## Measured, on this crate's own source tree

Not a synthetic fixture — the corpus is the repository this file lives in.

| | |
|---|---|
| Files scanned | 253 |
| Files indexed | 174 (Rust only, §307's scope decision) |
| Distinct lexical terms | 10,782 |
| Index size | 1,120,745 bytes (1.09 MiB) |
| `tqf sync .` wall time (debug) | 1,956 ms |
| Server start → first served index query | **81 ms** |

**24x**: rebuilding the index costs 1,956 ms; loading it costs 81 ms from
process start to a served HTTP query. That ratio is the entire argument
for persisting it, and it is measured rather than assumed.

### Where the sync time goes

An earlier revision of this document said 96% of sync was the scan. That
was wrong, and the way it was wrong is worth keeping: the sync report
printed its *total* elapsed time on the "scanned ..." line, where it read
as the scan's own cost. Two numbers that happened to be within 20 ms of
each other looked like confirmation. The report now prints each phase,
because a timer labelled as something it is not will send the next
optimization pass at the wrong target — as it did here.

The real breakdown, release build, this repository (259 files scanned,
175 indexed, 5.3 MiB):

| | before | after |
|---|---|---|
| scan | 165 ms | **19 ms** |
| walk | ~2 ms | 2 ms |
| index build | 109 ms | **~82 ms** |
| write | ~29 ms | ~35 ms |
| **total** | **513 ms** | **~141 ms** |

Four changes, each verified to produce identical output rather than
merely similar output:

**The tree was walked twice.** `tqf sync` called `scan_root`, then called
`full_correctness_walk`, which calls `scan_root` again. Passing the
existing scan in took 513 ms to 329 ms. A test pins the two entry points
to the same plan and contents.

**Classification rescanned each file once per marker.** Isolating the scan
put 94% of it in `classify` — 155 ms of 165 ms, against 4.6 ms of reading
and 1.8 ms of BLAKE3 hashing. `best_fingerprint` ran
`text.matches(marker)` for every marker of every one of twelve language
fingerprints over a 64 KiB sample: several megabytes of scanning per file.
One Aho-Corasick pass finds the same occurrences: **155 ms to 12 ms**.

Reproducing `str::matches` exactly is the part worth care. It is leftmost
and non-overlapping *per pattern* — two different markers may cover the
same bytes, but one marker's own matches never overlap each other. Plain
`find_iter` over the whole pattern set would let one marker's match
suppress another's and quietly lose counts, so this uses overlapping
search with a per-pattern end cursor. A differential test runs both
implementations over every file in this repository and requires the same
language, kind, and confidence.

**The index build split each document twice and allocated per token.**
BM25 wants lowercased tokens plus identifier subtokens; the exact lane
wants the same tokens case-preserved, and re-split the whole document to
get them. The per-document tally also cloned every token to key itself,
allocating once per token *occurrence*. Splitting once and consuming the
tokens: 109 ms to 92 ms. Verified through `export` — what actually reaches
disk — and through real query rankings, since a difference in
`avg_doc_len` alone would leave the bytes identical while every score
shifted.

**Identifier splitting allocated for tokens that cannot split.**
`split_identifier` allocates a `Vec<char>`, a `Vec<String>`, and a
`String` per part — for every raw token, including the majority whose
single part the caller then discards. Every boundary it recognizes needs
an underscore, an uppercase character, or a digit, so a byte scan for
those three decides the common case without allocating: on this
repository the tokenize pass went from ~78 ms to 51 ms. Its oracle is
deliberately separate from the one above, which compares two builds that
both run through `tokenize_raw` and so could not catch a change inside
it; a third test asserts `may_split` never skips a token that would
actually split, since a false negative there would silently drop
subtokens from the index and no search result would obviously reveal it.

`cargo test --release -- --ignored scan_cost_split` re-derives the scan
split rather than asking anyone to trust these numbers.

### How it scales, and why incremental sync is not built

A synthetic 5,000-file / 15.4 MiB Rust tree syncs in **776 ms** (scan 102,
walk 37, index 541, write 92). Extrapolating, a 50,000-file monorepo is
roughly 8 seconds.

For an explicit `tqf sync .` — a setup step a person runs once per
project — 8 seconds is fine. The case where it is not fine is a watcher
re-syncing after every file save, and that is the point: **nothing calls
sync repeatedly today.** Phase 42 built `LiveWatcher` against real
FSEvents/inotify, but it is not wired into `tqf sync`, and spec §3 fixes
the command surface at `tqf sync .` / `tqf unsync .` with no `--watch`.

So incremental sync would be an optimization for a caller that does not
exist. That is the same shape as the defect this whole branch is about —
a measured, qualified library nothing reaches — and it is why this is
recorded as a decision rather than left as an omission.

What would change the answer: wiring `LiveWatcher` into a real
continuous-sync path. That needs a spec decision about the command
surface first, not just an implementation.

If it is built, the target is the BM25 build (541 of 776 ms at 5,000
files), not the scan. Reading is 4.6 ms and hashing 1.8 ms for this
repository's whole tree, so change detection by exact content hash is
essentially free — there is no reason to reach for an mtime/size
heuristic and no correctness caveat to manage. The hard part is reusing
an unchanged file's postings: chunk ids are positional, so they shift
when a file is added ahead of them, and remapping means extracting one
document's terms from a term-keyed posting map — a forward index the
format does not store. The honest options are a new segment (kinds 7-11
are reserved) or an incremental `LexicalIndex` that adds and removes a
document while keeping `avg_doc_len` and every score identical to a full
rebuild. Spec §176's stable `FileId` and `next_chunk_id` already
anticipate the first half of that.

A prerequisite either way: `index()` passes `FileTable::default()`, which
is always empty, so every file is classified `new` on every sync and
Phase 42's change detection never actually runs.

### Reproducing these numbers

```sh
cargo test --release -- --ignored --nocapture scan_cost_split
cargo test --release -- --ignored --nocapture build_cost_split
TQF_BENCH_TREE=/path/to/large/tree \
  cargo test --release -- --ignored --nocapture build_cost_split
```

### Real queries against it

```
POST /tqf/index/search  {"query": "whole expert lfu cache eviction"}
  17.364  src/experts/mod.rs
  15.505  src/retrieval/lexical.rs
  15.284  src/experts/policy.rs

POST /tqf/index/search  {"query": "MemoryBroker", "exact": true}
  10 files
```

The top hit reproduces Phase 36's own finding through the persisted path:
that query appears nowhere as a literal substring, and still top-ranks the
expert cache via identifier-subtoken splitting. The difference now is that
no source file was read to answer it.

## Two defects found by running it

Both were found by exercising the real command, not by reading the diff.

1. **The scanner indexed its own output.** A real `tqf sync` reported four
   scanned files in a three-file tree: the walk descended into `.tqf/` and
   read `index.tqi` back as a source file. Every subsequent sync would
   have re-read its own growing binary. `.tqf` now skips unconditionally
   alongside `.git`.
2. **`FileRecord`'s declared size was 72 bytes where its encoder writes
   80**, and `decode` hardcoded `chunk_count` to 1. Caught by a round-trip
   test written before the encoder — the encoder and decoder are two
   independent transcriptions of one layout, and the failure mode is a
   field silently reading another field's bytes.

## What is deliberately not implemented

Recorded rather than left for a reader to discover, per §335.

- **Append-only generations.** §175 describes appending immutable segments
  per sync with periodic compaction. This baseline commits one whole
  generation per sync and replaces the file atomically — §173's compaction
  path used as the ordinary path. It is correct and cheap at the corpus
  sizes measured here (Phase 41 found flat search competitive at this
  scale). Appending needs the tombstone and overlay machinery §175
  describes; it is not written rather than written badly.
- **Five of §175's nine segment kinds.** Symbols and graph edges need a
  real AST (Phase 35/36 scoped it out); vectors and partitions need the
  helper embedding model, which is not installed; tombstones are
  meaningful only once generations append. They are absent rather than
  written empty, because an empty segment claims a capability exists and
  produced no rows.
- **`index.journal` and `lock`** (§218). The journal belongs with the
  append-only model above. A lock file without a real cross-process
  locking protocol would be a file that looks like mutual exclusion and is
  not.
- **The semantic lane.** `/tqf/indexes` reports `"lanes": ["lexical",
  "exact"]` per index rather than implying all three.
- **Incremental sync.** Each sync re-walks and rewrites. The persisted
  file records carry the same BLAKE3 content hash the walk uses for change
  detection, so the next step has what it needs, but nothing yet skips
  unchanged files.

## MCP

Phase 44 built the tool surface and qualified it against a hand-built
`IndexState`; nothing constructed one from a real index, so `--mcp-stdio`
passed `None` and every data tool answered "no index".

A coding client spawns the MCP server as a subprocess inside the project
it is working on, so the working directory selects the root — not a flag,
and not "every registered root", which would leak one project's file list
into another's tool results. The deepest containing registered root wins,
so a synced subproject inside a synced monorepo serves its own index.

`IndexState` no longer holds file contents. It held a `HashMap<String,
String>` of every indexed file's full text, which put the whole repository
in RAM for the process's life to serve occasional `tqf_file` calls — and
the persisted index does not store contents anyway, it stores postings.
It now holds the path list and reads on demand, which also means
`tqf_file` returns the file as it is now rather than as it was at index
time.

That made `tqf_file` a disk read driven by a tool argument, so it has two
independent guards: the path must be one the index contains, and the
resolved location must still be inside the root after symlink resolution.
Either would probably do. Spec §95 makes this surface read-only, and a
read-only surface that can be walked out of its root is not read-only in
any useful sense.

Measured by invocation, not by inspection (spec §335) — a real
`tqf --mcp-stdio` subprocess against a real synced index, four JSON-RPC
messages over stdin:

| Request | Result |
|---|---|
| `initialize` | `protocolVersion 2025-06-18` |
| `tqf_symbol` `MemoryBroker` | 6 real files, `src/model/qwen36/runtime.rs` first |
| `tqf_search` "whole expert lfu cache eviction" | `src/experts/mod.rs` (rrf 0.0164) |
| `tqf_file` `../../../etc/passwd` | refused |

The search query appears as that literal string in no file; it resolves
through identifier-subtoken splitting, the same result Phase 36 measured
directly against a freshly built index — now reproduced through a
persisted one, a subprocess boundary, and the JSON-RPC transport.

## Verification

```sh
just ci                       # 539 tests
tqf sync <path>               # writes <path>/.tqf/index.tqi, registers the root
tqf --headless                # loads every registered root at startup
curl localhost:11434/tqf/indexes
curl localhost:11434/tqf/index/search -H 'Content-Type: application/json' \
     -d '{"query":"...","top_k":5}'
tqf unsync <path>             # removes the index and the registration

# MCP, from inside a synced root:
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | tqf --mcp-stdio
```

33 tests cover the container and its wiring, including truncation at every
boundary, a flipped byte caught by checksum, a tampered generation table,
a refused future major version, atomic replacement leaving no temp file,
and an exported index rebuilding into one that returns identical rankings
and scores.
