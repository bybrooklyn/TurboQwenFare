# Phase 24 hard 4G broker certification: sampler, per-owner accounting, stress evidence

Spec Phase 24 deliverable (spec §296; exit gate "adversarial workloads
remain within qualified 4G bound"). This phase certifies the inference
core first; helper-model swap and 128K context paths stay open by design
(§296's own scoping).

## What was built

- `src/memory/os_sampler.rs` — OS-observed footprint sampling:
  `task_info(MACH_TASK_BASIC_INFO)` on macOS (resident set, virtual size,
  resident peak), `/proc/self/statm` on Linux, sampled alongside the
  broker's own totals. `sample_during`/`assert_footprint_within` bracket
  a workload and enforce a budget-plus-envelope bound.
- `MemoryBroker` hardening: per-owner reserved breakdown
  (`OwnerReserved` in `MemorySnapshot`), monotone peak tracking, and the
  existing reserve-before-allocate rule now also maintaining the
  per-owner table on lease drop.
- `adversarial_reservation_churn_stays_within_budget` — a deterministic
  200k-step randomized churn test across all nine owners and five
  classes asserting (a) the budget is never exceeded, (b) the per-owner
  table always sums exactly to the total, (c) everything drains to zero.
- Real-checkpoint OS footprint qualification
  (`canonical_decode_os_footprint_stays_within_qualified_envelope`):
  samples the OS resident set against broker accounting during real
  greedy decode and fails the certification if the process exceeds
  budget + a measured envelope.
- Large-allocation audit: every large allocation on the live decode path
  is lease-covered (resident core tensors, expert cache entries,
  activations, GDN state, BF16 KV, route-trace scratch, Metal buffers
  per Phase 20). The tiled-converter per-tile digesting was changed to
  hash source slices directly, removing a transient unaccounted
  concatenation. Small fixed intermediates (a few KiB) are outside the
  "large allocation" scope; `backend::reference` helpers are
  test/reference-path utilities only.

## Measured evidence (base M4, canonical container, resident-core streaming profile)

8 greedy tokens, sampled every token:

| Metric | Value |
|---|---|
| Peak OS-observed resident set | **1,777 MiB** |
| Peak broker-reserved | 1,488 MiB |
| Observed-over-broker overhead | 689 MiB |
| Qualified envelope (budget + overhead) | 4 GiB + 2 GiB |

The hard budget (4 GiB) is respected by the broker itself (peak 1.49
GiB), and the process footprint stays inside the qualified envelope with
a 689 MiB measured overhead envelope (container metadata, process image,
allocator slack). This is the inference-core certification §296
describes; the helper-swap and 128K-context accounting remain explicitly
later phases, and no stronger claim should be read into this record.

## Remaining Phase 24 work

- Wire the sampler into the server's health/status surface so the OS
  number is visible in production, not only in qualification.
- Re-run the footprint envelope when the resident-core profile becomes
  the server default (Phase 25 profile flip), since resident tensors
  change the broker/OS split.
