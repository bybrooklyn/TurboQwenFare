# Upstream precedent: TurboFieldfare and NVMAI

Spec Phase 0 deliverable ("Clone/freeze mined TurboFieldfare and NVMAI SHAs
in research ledger," §272). This freezes the commit references spec §14-16
and the bibliography (§A5, R7-R17) already cite, and records that both
upstream repositories were confirmed to actually exist and match their
described contents as of 2026-08-11 — the spec's citations were not taken
on faith, they were spot-checked.

## TurboFieldfare (R7) — foundational precedent, not a Qwen fork

- **Repository:** [`drumih/turbo-fieldfare`](https://github.com/drumih/turbo-fieldfare)
- **License:** Apache-2.0 (permits adaptation with attribution, per spec §14)
- **Confirmed 2026-08-11:** real, active repository. Description: "Gemma 4
  26B-A4B inference in ~2 GB of RAM on any M-series MacBook" — matches
  spec §14's description exactly (out-of-core MoE on Apple Silicon,
  bounded-memory installation via remote byte-range repacking, Gemma 4
  target, not Qwen). Latest release at fetch time: 0.4.1.
- **What TQF takes from it (spec §14):** the *system principles* — bounded
  memory, explicit expert streaming, measurement-driven kernel work, a
  model-specific container format — not code specific to Gemma 4.

## NVMAI (R8) — direct Qwen3.6 donor

- **Repository:** [`Pummelchen/NVMAI`](https://github.com/Pummelchen/NVMAI)
- **Confirmed 2026-08-11:** real repository (does not appear in general
  web search indexing, likely low visibility/star count, but resolves
  directly). Description: "Run Qwen 3.6 35B A3B on Apple M1-M5 with low RAM
  usage using SSD/NVM streaming" — matches spec §15's description (Apache-2.0
  fork of TurboFieldfare, already implements the hybrid Qwen graph, GDN,
  Q4/6/8-bit paths, repacking, server support). Primary language: Swift.

### Frozen commit references (spec §A5, R9-R17)

These exact commit SHAs are already frozen in the spec text itself — this
table exists so they're findable from one place without re-deriving them
from the bibliography, not because they were independently re-mined here.

| Ref | Commit | Finding |
|---|---|---|
| R9 | [`4beb74f`](https://github.com/Pummelchen/NVMAI/commit/4beb74f4a28de6d4a3222d079dc5306cbd7a32c0) | Parallel expert `pread` |
| R10 | [`069aed6`](https://github.com/Pummelchen/NVMAI/commit/069aed6394777216a06a252e5d2d47a063e37ab1) | 64 cache slots + resident pin |
| R11 | [`5a7902b`](https://github.com/Pummelchen/NVMAI/commit/5a7902baa9cec83eed1372e1e0fec58228357f7c) | MoE phase-1 MSL rewrite |
| R12 | [`4ea208d`](https://github.com/Pummelchen/NVMAI/commit/4ea208d9b563523103f7fea59998f368337116c2) | 4096-token prefill chunk |
| R13 | [`159ff74`](https://github.com/Pummelchen/NVMAI/commit/159ff74825115ceb82f5904d0587db1ec2e82e5d) | Fused GDN input projections |
| R14 | [`2ddf68e`](https://github.com/Pummelchen/NVMAI/commit/2ddf68e48ea29ef60a082abba309b37ef6a64506) | Persistent KV + GDN snapshots |
| R15 | [`2c3c7b8`](https://github.com/Pummelchen/NVMAI/commit/2c3c7b8ccd8537f4d2d26ce03c66f304b1689012) | CPU MTP drafting (negative result) |
| R16 | [`7cc8b5e`](https://github.com/Pummelchen/NVMAI/commit/7cc8b5ea98fc788b87fea83941b8181196d521f5) | Targeted `F_RDADVISE` |
| R17 | production hardening commit (trusted receipts, schema/path/server/GPU failure hardening) — cited by spec but not yet re-fetched here. | |

None of these commits have been re-cloned/diffed into this repository yet
— that's real engineering work for the phases that actually port each
technique (expert I/O parallelism is Phase 19, MoE MSL kernels are Phase
20, etc.), not something Phase 0 needs to do speculatively.

### Findings → TQF actions (spec §15, reproduced for local reference)

| NVMAI finding | Measured effect | TQF action |
|---|---|---|
| Parallel expert pread | I/O wall ~41.2→30.9 ms/token; decode 9.98→12.80 tok/s (+28%) | Port concept immediately; autotune worker count |
| 64 cache slots + resident pin | ~10% decode gain vs 32+pin; 128 slots could regress from pressure | Keep pinning lesson, replace fixed per-layer cache with global broker |
| MoE phase-1 MSL rewrite | Stage 14.4→9.24 ms/token; byte-identical output | Adapt kernel, specialize harder for M4/Q4 |
| 4096-token prefill chunk | 1280-token prompt ~13.6s vs 43.3s at 128-token chunks | Start autotune around large MoE-aware chunks |
| Fused GDN input projections | Four Q4 GEMVs fused, stage reduction measured | Adapt and extend fusion |
| Targeted F_RDADVISE | ~10.6% gain in one M3 Q4 test; neutral elsewhere | Autotune per host; never universalize |
| Persistent KV + GDN snapshots | Demonstrates hybrid-state prefix restore | Replace monolithic snapshots with deduplicated TQKV page references |
| CPU MTP drafting | Output-head bandwidth erased hoped-for CPU advantage | Do not prioritize CPU draft path; keep GPU MTP benchmark-driven |
| Stage accounting | Corrected GPU budget exposed attention/routed-MoE as major costs | Build detailed timing from day one |

### What TQF must not copy (spec §16 — reproduced for local reference)

- Not a multi-target Swift package — TQF stays one Rust crate/one binary.
- Not fixed equal per-layer cache budgets — TQF uses a global byte-budgeted
  cache (spec Part VI).
- Not text-only scope — TQF supports lazy vision behind `--enable-vision`.
- Do not assume MTP is beneficial merely because the model was trained for
  it — benchmark it.
- Do not expose a wall of expert/cache knobs to normal users — autotune
  instead.
- Do not treat 256K as the final context target — TQF pursues ~1M.

### Experiments to reproduce (not yet started)

Recorded here as a checklist for whichever phase first has a working Metal
decode loop to benchmark against (earliest: Phase 15, "end-to-end decode"):

- [ ] Reproduce the parallel-pread I/O-wall improvement (R9) on the M4
      reference target before porting the technique as a default.
- [ ] Reproduce the 64-slot cache/pinning result (R10) against TQF's own
      global broker design, not NVMAI's per-layer allocation.
- [ ] Reproduce the MoE MSL stage-time reduction (R11) — note this needs
      Qwen3.6-specific kernel work anyway, so "reproduce" here means
      confirming the *technique* transfers, not reusing the kernel as-is.
- [ ] Confirm or refute the CPU MTP negative result (R15) on TQF's own
      broker/memory model before ruling out a CPU draft path permanently.
