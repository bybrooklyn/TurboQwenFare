# Research ledger

Frozen research records backing spec Phase 0 ("research harvest and
canonical manifest," §272). These are point-in-time records, not live
documentation — code that needs current values reads
`src/source/pinned.rs` / `src/model/qwen36/geometry.rs`, which these files
back up with the how/when/source of each value.

- [`canonical-source-manifest.md`](canonical-source-manifest.md) — the
  pinned Qwen3.6 source revision, GGUF artifact hashes/sizes, license, and
  the config.json geometry cross-check.
- [`upstream-precedent.md`](upstream-precedent.md) — frozen TurboFieldfare
  and NVMAI commit references, the findings table, the "must not copy"
  list, and an experiments-to-reproduce checklist.

## Status against spec §272's task list

- [x] Pin official Qwen3.6 model/GGUF source revisions.
- [x] Record SHA-256 and licenses.
- [x] Freeze mined TurboFieldfare and NVMAI SHAs in a research ledger.
- [x] Build tensor-inventory generator against source metadata — see
      `src/dev/inventory.rs`; the generator itself is complete and tested,
      but only against a synthetic fixture (see that module's own caveat
      about real GGUF tensor names being unresolved until the actual
      20+ GB checkpoint is downloaded and run through Phase 5's reader).
- [x] Create derived architecture calculator tests for dimensions —
      `src/model/qwen36/geometry.rs`.
- [x] Record known NVMAI wins and dead ends as experiments-to-reproduce —
      see `upstream-precedent.md`.
