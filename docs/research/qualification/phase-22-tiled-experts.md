# Phase 22 tiled experts: whole vs 64/128/256/mixed A/B on the raw-a-128 route trace

Spec Phase 22 deliverable (spec §294, exit gate "A/B decides default;
format requires no migration"). Qualification record for the expert-cache
admission granularity decision.

## What was built

- `src/format/tqf/tiling.rs` — `NeuronWidth::{Whole,N64,N128,N256,Mixed128}`
  tile plans. Q4_K block geometry (256 columns x 1 row) fixes what is
  tileable: gate/up (neuron = row dim) tile at any of 64/128/256; down
  (neuron = column dim) can only tile at 256 or 512 without splitting
  blocks, so 64/128 tilings emit down at 256. `Whole` stays the Phase 6
  canonical two-record layout.
- `TqfWriter::write_expert_parts_tiled` + per-tile BLAKE3 digests stored
  after the whole-extent digest (flag `EXPERT_INDEX_FLAG_TILE_CHECKSUMS`
  on the expert index record). The disk bytes are identical to the
  canonical layout — only metadata gains granularity — so a tiled
  container is readable by the whole-expert path with **no format
  migration**.
- `TqfReader` validates every tile table as a contiguous exact partition
  at open, and `read_expert_tile_into` verifies a per-tile digest before
  any partial read (partial reads are refused on containers without the
  flag — a partially resident expert is only admissible on a container
  that vouches for every tile).
- `Qwen36WeightLoader::load_expert` accepts any valid partition;
  `load_expert_tile` loads one checksum-verified tile with its own broker
  lease.
- `src/experts/tiling.rs` — tile-granularity replay simulator (LRU,
  byte-budgeted, pin-all-8-per-route) with hit ratio, read syscalls,
  overread padding, and fetched-never-reused bytes.
- Converter A/B control: `TQF_EXPERT_TILE_NEURONS=64|128|256|mixed`
  (spec invariant #10); canonical conversion stays whole-region.

## Method

Re-captured the exact-router trace for the pinned `raw-a-128` fixture
(128 tokens, 40 layers, 5,120 route events, real router output,
`docs/research/qualification/raw-a-128-route-trace.json`), then replayed
it offline at 128/256/512/768/1024 MiB capacities x 5 tilings. This
changes nothing about routing or computed results — only residency
simulation.

## Results

| Capacity | Whole | N64 | N128 | N256 | Mixed128 |
|---|---|---|---|---|---|
| 768 MiB | 34,120 hits / 42.29 GB | 307,080 / 42.29 GB | 170,600 / 42.29 GB | 102,360 / 42.29 GB | 153,540 / 42.29 GB |
| 1024 MiB | 41,330 hits / 35.92 GB | 372,066 / 35.90 GB | 206,698 / 35.90 GB | 124,014 / 35.90 GB | 186,021 / 35.90 GB |

(hits counted per tile demand; miss bytes are what matters — they are
identical across every tiling at every capacity to three significant
digits. Read syscalls: whole = 1 per expert miss; N128 = 5x; N64 = 9x.)

## Findings

- **Tiling buys no bytes on this trace.** Demand is all-or-nothing per
  expert and uniform across an expert's tiles, so per-tile hit rate
  equals the whole-expert hit rate and missing-tile bytes sum to the
  whole-expert bytes. The only measured byte difference is 0.05% at
  1024 MiB/N64 — noise-level.
- **Tiling multiplies read syscalls 5-9x** (one read per missing tile)
  for the same bytes, which is exactly the Phase 22 rejection criterion:
  "partial caching that wins hit ratio but destroys I/O latency is
  rejected." Here it wins nothing and raises syscall count.
- The one structural benefit of tiling — sub-expert eviction granularity
  (keeping 3 of 10 tiles alive across a churn instead of discarding the
  whole expert) — exists but is too small to register on real route
  statistics at the capacities the 4 GiB plan can afford.
- Padding overread is structurally zero: every Qwen Q4_K tile size is
  4096-aligned.

## Decision

Whole-expert admission remains the cache unit and the production default.
The tile infrastructure (partitioned layouts, per-tile checksums,
tile-granular reads) is retained behind `TQF_EXPERT_TILE_NEURONS` for
future uses that actually need partial residency (co-routing-aware
physical layout §35, tile-aware policy research §42), but Phase 22's A/B
answer on the real trace is: **whole experts win; recorded negative
result for fine-grained tiling**.
