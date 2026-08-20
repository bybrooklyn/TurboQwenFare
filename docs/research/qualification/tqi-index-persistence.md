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

### Where the sync time goes, and one scan that was paid for twice

Profiling the release build put **96% of sync in the scan** — 494 ms of
513 ms — with tokenizing, BM25 construction, and the container write
together under 20 ms. That reorders the obvious next step: reusing stored
postings for unchanged files, which is what incremental sync usually
means, is chasing the 4%.

It also exposed a plain defect. `tqf sync` called `scan_root`, then called
`full_correctness_walk`, which calls `scan_root` again — and `scan_root`
reads every file in the tree to classify it by content. The repository was
read twice per sync. Passing the existing scan into the walk
(`full_correctness_walk_of`) took the release build from **513 ms to
329 ms (1.56x)** with byte-identical output.

The remaining scan cost is a genuine content read: change detection hashes
file contents. Skipping it for unchanged files needs a stat-only
pre-filter on `byte_len`/`mtime_ns`, both of which the persisted
`FileRecord` already carries. That is a real optimization with a real
correctness caveat — mtime and size can miss a same-size edit inside one
timestamp tick — so under invariant #8 it needs the content-hash walk to
stay available as the fallback. `full_correctness_walk` is already exactly
that fallback, and is already named for it. Not implemented here.

A related gap worth naming: `index()` passes `FileTable::default()`, which
is always empty, so every file is classified `new` on every sync and
Phase 42's change-detection machinery never actually runs. Seeding the
table from the persisted `FileRecord`s is the prerequisite for any of the
above.

The index is 1.09 MiB for a 3.0 MiB corpus. It stores postings, the exact
identifier lane, and per-chunk token counts — not the source text, per
§185's rule that chunk text remains in the source file rather than being
duplicated wholesale into the index.

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
