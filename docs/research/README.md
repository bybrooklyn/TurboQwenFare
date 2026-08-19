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
- [`oracles/`](oracles/) — versioned, token-only outputs from pinned external
  research runtimes. TQF consumes these as qualification inputs; external
  runtime code is never linked or shipped.
- [`qualification/product-surface-wiring.md`](qualification/product-surface-wiring.md) —
  not a spec phase: the record of making the qualified engine behave like the
  product spec §3 describes (the Linux build failure, the absent Ollama
  surface, real sampling and streaming, and the defects found by running
  things rather than reading them).
- [`qualification/`](qualification/) — immutable result records tying a TQF
  commit and canonical source fingerprint to an executed oracle fixture and
  its measured cache/broker evidence, including
  [`raw-a-128-route-trace-policy.md`](qualification/raw-a-128-route-trace-policy.md),
  the Phase 21 cache-policy benchmark-selection record (LRU vs LFU vs
  decayed-cost-aware on a real 128-token route trace), and
  [`raw-a-512-divergence-investigation.md`](qualification/raw-a-512-divergence-investigation.md),
  the Phase 15 512-token gate attempts: two independent prompts (197 and 24
  consecutive matched steps respectively, both new depth records for their
  runs) each diverged on a near-tied logit against the independent oracle.
  The gate does not close as literally specified, but the divergences are
  characterized as recurring, benign floating-point near-ties rather than a
  defect - see the investigation doc's conclusion and recommendation.

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
