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
| R17 | [`19aafd8`](https://github.com/Pummelchen/NVMAI/commit/19aafd8fe2d99ca2e761c785b4a44f6bf119a79a) | Trusted receipts plus schema/path/server/GPU failure hardening |

### Phase 19–20 re-mining (2026-08-16)

NVMAI was cloned outside the product repository at commit
`fd8234bd53487c854b0047dc007d5d79d36580c3`. The frozen R9, R11, R13, and
R16 diffs were inspected directly rather than relying only on their commit
messages. This clone is research input only; TQF remains one Rust crate and
does not link or ship NVMAI.

- R9 parallelizes only independently reserved cache misses. Each `pread`
  writes to a distinct destination slot; bookkeeping publication is locked
  and happens after the read. The serial path remains available for A/B.
  Its reported experiment used interleaved fresh servers and 512 greedy
  tokens, and compared both I/O wall time and end-to-end decode.
- R16 changed a host-derived default after five interleaved 128-token rounds.
  It was beneficial on an M3/Q4 deployment and neutral on another host, which
  confirms TQF's `BENCHMARK-SELECTED` requirement: `F_RDADVISE` cannot be a
  universal compile-time default.
- R11 stages the shared 2,048-element activation in threadgroup memory and
  maps 16 phase-1 rows to a 512-thread group. The porting requirement is the
  measured technique and its parity test, not a blind copy of NVMAI's affine
  INT4 physical layout, which differs from canonical GGUF Q4_K.
- R13 fuses QKV/Z/alpha/beta by dispatching one concatenated row space while
  preserving the existing per-row math and operand order. Its own measurement
  improved the stage by 6.6% but did not move end-to-end throughput outside
  noise. TQF must therefore retain separate and fused paths and reject the
  fused candidate if its own end-to-end A/B does not win.

No NVMAI source has been copied into TQF at this point. Any later direct code
adaptation must carry the Apache-2.0 notice and prominent modification marker
required by the specification.

### NVMAI v3.9 re-mining (2026-08-17)

NVMAI moves fast (per the user, "being updated very often"); the clone at
`/Volumes/flash1/tqf-research/NVMAI` was fast-forwarded from `fd8234b` to
`ec10e6a` (tag `v3.9`, 48 commits, 2026-08-16 to 2026-08-17). Two of NVMAI's
own docs (`docs/v4-core-design.md`, `docs/cpu-coexecution-plan.md`) were read
in full rather than skimming commit messages; both self-report measurements
and, notably, retract an earlier wrong estimate in the same document more
than once - a "measure, then correct the estimate" discipline the TQF spec
also demands (invariant #10-adjacent). No NVMAI source was copied into TQF.

**Directly actionable for TQF's open work:**

- **Cache-size undersizing is a cliff, not a slope, once you stop trusting
  the OS page cache.** NVMAI measured two different regimes: with the page
  cache allowed to backstop misses, a *small* slot cache (16) beat a large
  one (128) by ~35% - the OS was quietly holding the real working set that
  "declared RAM" never counted. With page-cache backstop disabled
  (`NVMAI_BOUNDED_IO`, `F_NOCACHE`) - the regime TQF's own invariants
  already commit to (do-not-do-list: "do not rely on unreported OS page
  cache to justify a memory claim") - the ordering **inverts**: 16 slots
  cost 2.2x the throughput of 128 (8.73 vs 18.91 tok/s), because a miss with
  no cache behind it is a real device read with nothing to fall back on.
  This directly cross-validates and sharpens TQF's own Phase 21 finding
  (`raw-a-128-route-trace-policy.md`): the 256 MiB expert-cache capacity
  used in qualification gets *zero* reuse under any policy (LRU/LFU/decayed)
  - it is sitting exactly in the dead zone NVMAI's cliff describes. Picking
  a production cache capacity should treat "below the reuse floor" as a
  cliff to stay clear of, not a knob to shave for memory headroom.
- **Predictive expert prefetch (spec Phase 23) is structurally impossible
  as usually conceived, not just unproven.** NVMAI replayed a real
  383-token routing trace against a simulated per-layer LRU cache: any
  cache of >=16 slots is never evicted within one token, so an expert used
  at layer L in token N-1 is *still resident* at token N. The predictable
  set (previous token's experts) and the actual miss set are therefore
  disjoint by construction - **0.00% of misses were catchable by
  previous-token prediction, at both 16 and 128 slots.** The real 38%
  token-to-token expert-reuse figure is genuine but already fully captured
  by the cache itself; there is nothing left for a predictor to add across
  layers, because layer L+1's routing is not known until layer L's router
  actually runs. TQF should not plan Phase 23 as "predict from recent
  history"; if prefetch has value at all it would need a different
  information source (e.g. the router's own hidden state pre-argmax),
  which NVMAI did not explore.
- **KV/context buffers: reserve on demand, not for max-context up front.**
  Full-attention KV layers sized for NVMAI's full 262144-token ceiling cost
  **1.63x slower decode even on a 25-token conversation**, purely from
  mapping overhead destroying locality despite lazy touching keeping RSS
  low. Growing from an 8192-token start and doubling on demand brought this
  to 1.02x (byte-identical output). The mechanism - a huge reserved stride
  hurts locality even when its extra pages are never written - is
  architecture-independent and worth checking against TQF's own
  `context/tqkv` allocation strategy once that work starts.
- **A cheap, directly portable decode-loop reordering, once Phase 20 wires
  compute into the live loop:** dispatch the shared-expert MLP's GPU work
  as soon as its only dependency (`routedX`) is ready, rather than encoding
  it after the CPU blocks on the router round-trip. Pure submission-order
  change, no new synchronization, byte-identical output; NVMAI measured GPU
  idle time in that transition drop from 7.881 to 0.016 ms/token.

**Two negative results worth banking as "don't try this" before Phase 20
temptation strikes:**

- **CPU/GPU co-execution during decode measured 22.6% *slower*** (8-thread
  CPU expert-FFN kernel running concurrently with GPU decode), because CPU
  and GPU share one memory controller and one power budget on Apple
  Silicon - a "free idle core" is not free once it actually computes.
  NVMAI's own verdict: "there is no split ratio that wins." Relevant
  because TQF's spec architecture (memory broker, CPU SIMD as a first-class
  storage-chain participant) makes this specific idea plausible to propose
  later.
- **Consolidating Metal command buffers (fewer, larger submissions)
  measured *slower*** (-4.6% median, not the predicted +12-15%), because
  merging serializes encode-then-commit and loses CPU/GPU pipelining that
  splitting was buying. Reverted in NVMAI.

**Confirmed unchanged:** no follow-on work to R11 (MoE phase-1 MSL rewrite)
or R13 (fused GDN input projections) landed in this window - the fusion
techniques TQF still hasn't ported are exactly as NVMAI left them in the
2026-08-16 re-mining above. One useful calibration point did land: NVMAI's
own *already-fused* MoE kernels measured only ~13-15% of theoretical Apple
Silicon GPU peak (~600 GFLOP/s of ~4 TFLOP/s), from 4-bit dequant ALU cost
alone, confirmed via `xcrun xctrace` at max clocks (not a stall or thermal
artifact) - a realistic ceiling to calibrate TQF's own eventual fusion work
against rather than expecting near-peak throughput.

**Explicitly out of TQF's scope:** an extended Apple Neural Engine (ANE)
investigation arc (~15 commits) found ANE offload during decode is a clear
negative (same shared-memory-controller problem as CPU co-execution) but
ANE for *prefill* is a genuine, well-measured win (attention alone 15-19x
faster at width 256-1024, prefill is compute- not bandwidth-bound). This is
a CoreML/ANE-specific path with no Metal/CUDA analog and TQF's spec is
Metal/CUDA only - not actionable beyond the general caution that prefill
kernels can plausibly be several times more efficient before reaching for
exotic hardware. Also closed as dead ends (routine, not techniques worth
tracking further): 6-bit quantization (packing inefficiency, withdrawn
entirely), lossless weight-compression (a full 40-layer x 256-expert scan
found representative layers at 93% of their entropy limit - already
near-incompressible), and a fused-GPU-argmax greedy head (~3% slower, not
faster).

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

- [x] Port the parallel-pread technique (R9) itself: `src/io/mod.rs`
      (`ReadFanout`/`fetch_all`) fans independently reserved expert-cache
      misses across a bounded thread pool, reusing `TqfReader`'s existing
      `pread`-based positional reads (`FileExt::read_exact_at`, safe to call
      concurrently on `&self`). Wired as the default in
      `WholeExpertLfuCache::prepare_exact_route`
      (`src/experts/mod.rs`), with the Phase 18 serial path kept selectable
      via `TQF_EXPERT_IO_FANOUT`/`set_io_fanout` for A/B (spec invariant
      #10). Parity + worker-bounding + deterministic-first-error covered by
      `io::tests`; a real-checkpoint wall-time comparison exists as
      `model::qwen36::weights::tests::parallel_io_fanout_meaningfully_beats_serial_on_the_canonical_checkpoint`
      (`#[ignore]`, needs `TQF_CANONICAL_TQF`).
- [x] Ran that benchmark on the real checkpoint (2026-08-17): one exact
      route's worth of eight independent cold expert misses took 107ms
      serial versus 3ms with the default 4-worker parallel fan-out - a
      **29.5x** wall-time reduction. This is a narrower, more extreme metric
      than R9's own end-to-end figure (~41.2->30.9 ms/token I/O wall, a
      ~1.33x reduction folded into overall decode) because it isolates pure
      I/O wall time for one batch of misses rather than amortizing it across
      a full decode step; the gap is plausibly explained by per-syscall/seek
      latency on this specific drive being maskable by concurrent dispatch
      but not by strictly sequential reads. The parallel default is now
      qualified, not just implemented, on this reference machine - a
      different drive (e.g. a faster NVMe enclosure) would be expected to
      narrow this gap, since faster media has less per-request latency to
      hide behind concurrency in the first place.
- [ ] Reproduce the 64-slot cache/pinning result (R10) against TQF's own
      global broker design, not NVMAI's per-layer allocation.
- [ ] Reproduce the MoE MSL stage-time reduction (R11) — note this needs
      Qwen3.6-specific kernel work anyway, so "reproduce" here means
      confirming the *technique* transfers, not reusing the kernel as-is.
      **Foundation landed, fusion not started:** `backend::metal::expert::GpuResidentExpert`
      uploads one expert's gate/up/down Q4_K matrices to broker-registered
      persistent Metal buffers once (`MetalContext::allocate_broker_buffer*`,
      also new) and reuses them across forward calls instead of re-uploading
      per matvec — the prerequisite the R11 finding itself depends on
      ("a GPU MoE path only wins once weight upload is amortized"). It
      still dispatches the unfused reference `tqf_q4k_gemv` kernel three
      times per expert (no threadgroup-staged shared activation, no 16
      rows/512-thread group), is not wired into the live decode loop
      (`experts::mod` still calls `backend::reference`), and its GPU buffer
      is a second copy alongside the CPU `LoadedQwen36Expert` bytes rather
      than a unified-memory zero-copy view — so it does not yet double as
      the cache's sole resident storage. Parity-tested on real Metal
      hardware in `backend::metal::expert::tests`.
- [ ] Confirm or refute the CPU MTP negative result (R15) on TQF's own
      broker/memory model before ruling out a CPU draft path permanently.
- [ ] Check whether TQF's production expert-cache capacity (not just its
      Phase 21 policy) sits above the reuse floor the 2026-08-17 NVMAI
      re-mining found (`raw-a-128-route-trace-policy.md`'s 256 MiB
      qualification default is confirmed to sit in the zero-reuse dead
      zone at every policy tested; NVMAI's independent measurement puts
      undersizing below that floor at a 2.2x throughput cliff under
      genuinely bounded I/O, not a gentle slope).
- [x] Phase 23 predictive prefetch, previous-token variant: NVMAI's
      2026-08-17 replay against a real 383-token trace found 0.00% of
      cache misses catchable by previous-token prediction at any cache
      size >=16 slots (the predictable set and the miss set are disjoint
      by construction once the cache exceeds one token's expert count).
      TQF should not plan Phase 23 around this specific approach; a
      different information source (e.g. pre-argmax router hidden state)
      would be needed and has not been explored by either project.
