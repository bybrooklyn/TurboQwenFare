# Phase 30: prefix snapshot store

Spec Phase 30 deliverable (spec §302, §66-67, §156; exit gate row 30:
"Repeated-prefix TTFT reduction; restart reuse").

## What was built

`context::prefix::PrefixSnapshotStore` — a real, on-disk, content-addressed
snapshot store, not an in-memory cache:

- **Page-content IDs** (spec §66 "immutable page IDs"): reuses each TQKV
  sealed page's existing BLAKE3 content hash (already computed at seal time,
  Phase 27) directly as its storage identity — no second hash. `SealedPage`
  gained `to_bytes`/`from_bytes` (exact layout the page header's own
  offsets already describe) so pages round-trip through disk unchanged.
- **GDN snapshot blob** (spec §66): `GdnState::to_bytes`/`from_bytes` —
  fixed-size little-endian serialization of the recurrent state and conv
  tail, content-addressed the same way as TQKV pages.
- **Exact token prefix hash** (spec §67, v1 exact-match only):
  `PrefixSnapshotStore::token_prefix_hash` — BLAKE3 over the exact
  little-endian input-token-ID sequence.
- **LRU disk quota**: a top-level index (prefix hash, bytes, last-used) and
  a **reference-counted** blob table — a page or GDN blob is only deleted
  once no live snapshot references it, so two snapshots sharing a system
  prompt prefix share the same on-disk bytes and evicting one never
  corrupts the other.
- **Crash-safe manifests**: every write (blobs, snapshot manifests, index,
  refcount table) goes through `atomic_write_bytes`/`atomic_write_toml` —
  temp file, `fsync`, atomic rename, parent-directory `fsync` (crate
  invariant #9), the same pattern already used for config/receipt/manifest
  persistence elsewhere in the crate.
- **Runtime wiring**: `Qwen36BoundedReferenceRuntime::snapshot_session`/
  `restore_session` capture/restore every layer's TQKV+GDN state in one
  call, keyed by the exact fed-token sequence. Full-attention layers
  running the BF16 backend contribute nothing to a snapshot — prefix dedup
  is inherently TQKV-specific (spec §66's premise is deduplicating *TQKV*
  pages), so this only does something useful with `TQF_TQKV_ENABLED=1`.

## Measured evidence

**Storage mechanics (synthetic state, real file I/O)** — six tests in
`context::prefix::tests`, all passing:

- `full_attention_capture_round_trips_through_store_and_restore`: a real
  TQKV-Q8 layer decoded past a page boundary, captured, stored, reloaded,
  and restored into a fresh layer — sealed page count, tail bytes, and
  position all match exactly.
- `dedup_shares_one_blob_across_two_snapshots_with_an_identical_page`:
  two snapshots referencing the same page content end up with one blob on
  disk and a refcount of 2.
- `lru_quota_evicts_the_least_recently_used_snapshot_first`: a quota
  sized for one real sealed page evicts the older snapshot and keeps the
  newer one, verified by attempting to load both.
- `restart_reuse_survives_dropping_and_reopening_the_store`: stores a
  snapshot, **drops the `PrefixSnapshotStore` handle entirely** (no
  in-memory state survives), opens a brand-new handle against the same
  directory, and loads the snapshot back byte-identical — the literal
  "demonstrate restart reuse" deliverable.
- `gdn_state_round_trips_through_the_store`: a real `GdnState` round-trips
  through content-addressed storage exactly.
- `token_prefix_hash_is_exact_and_order_sensitive`: confirms the hash is
  a true function of exact token order (spec §67's "longest exact
  token-prefix match only" — no fuzzy/semantic matching).

**Real-checkpoint TTFT reduction and restart reuse**
(`dev::qualification::canonical_prefix_snapshot_restore_reduces_repeat_prefix_time`,
`TQF_TQKV_ENABLED=1`):

<!-- filled in once the real-hardware run completes -->

## Status and remaining work

- v1 exact-match only, as spec §67 mandates for the first implementation;
  no fuzzy/structural "middle of prompt changed" reuse.
- Only `Qwen36BoundedReferenceRuntime` is wired; `Qwen36ReferenceRuntime`
  (resident/streaming) is not, and would need the same
  `snapshot_session`/`restore_session` pair added against its own layer
  collection.
- No automatic snapshot-point policy exists yet (spec §66 lists candidate
  checkpoint points — message/tool-result/document boundaries, adaptive
  intervals); snapshotting is currently something a caller invokes
  explicitly, not something the generation loop does on its own at request
  boundaries.
- Snapshot storage size is not yet counted against the 4G/2G memory
  budgets — it is on-disk, bounded by its own `quota_bytes`, independent of
  the broker's in-memory budget, matching the existing SSD-backing
  precedent (expert cache, TQKV pages) but not yet cross-referenced in one
  combined disk-footprint accounting.
