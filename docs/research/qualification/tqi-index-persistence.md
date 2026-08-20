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
| `tqf sync .` wall time | 1,956 ms |
| Server start → first served index query | **81 ms** |

**24x**: rebuilding the index costs 1,956 ms; loading it costs 81 ms from
process start to a served HTTP query. That ratio is the entire argument
for persisting it, and it is measured rather than assumed.

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
- **MCP wiring.** Phase 44's tools take an `IndexState`; the server now
  has real indexes to give them, but they are not connected yet.

## Verification

```sh
just ci                       # 539 tests
tqf sync <path>               # writes <path>/.tqf/index.tqi, registers the root
tqf --headless                # loads every registered root at startup
curl localhost:11434/tqf/indexes
curl localhost:11434/tqf/index/search -H 'Content-Type: application/json' \
     -d '{"query":"...","top_k":5}'
tqf unsync <path>             # removes the index and the registration
```

33 tests cover the container and its wiring, including truncation at every
boundary, a flipped byte caught by checksum, a tampered generation table,
a refused future major version, atomic replacement leaving no temp file,
and an exported index rebuilding into one that returns identical rankings
and scores.
