# TurboQwenFare Phases 22–27: Tiled Experts to TQKV

**A deep research survey of the repository's internal design records, current implementation state, and the published literature each phase builds on**

> Scope: Phases 22 (tiled experts), 23 (predictive prefetch), 24 (hard 4G broker certification), 25 (M4 short-context assault), 26 (prefill), and 27 (TQKV Q8/Q4 baseline) of `TurboQwenFare_Master_v2_All_Encompassing_Specification.md`. Every external citation below was re-checked against its primary source online (arXiv, Apple Developer Documentation, or the upstream repositories) during August 2026. Section numbers (§) refer to the spec; file paths refer to this repository.

---

## Abstract

Phases 22–27 form the second critical path of TurboQwenFare (`tqf`), a single-binary, single-crate Rust server specialized on **Qwen3.6-35B-A3B at Q4** under a hard **4 GiB** live-memory wall, a **128K** initial context, a **≤1%** quality ceiling, and a **≥15 tok/s** decode floor on a base **M4** MacBook Air (§4, §113). The band's job is to turn a correct but whole-expert, whole-KV baseline into a fast, provably-bounded system: finer-grained expert residency (22), speculative I/O (23), a certified memory wall (24), measured Apple-Silicon throughput (25), fast prefill (26), and compressed long-context KV (27).

Two findings organize this paper. First, **the enabling data already exists in the repo**: the `.tqf` container emits per-expert `ExpertTileRecord`s with `neuron_start`/`neuron_count` (§124, `src/format/tqf/records.rs`), so Phase 22 needs a runtime change, not a format migration. Second, **the published literature validates every mechanism the spec prescribes, and in several cases on almost exactly TQF's target**: FlashMoE was evaluated on Qwen3-30B-A3B (the same 35B-A3B family), MoEpic quantifies the same split-ratio sensitivity the spec asks Phase 22 to measure, and KVQuant/KIVI/TurboQuant supply the per-channel/pre-RoPE/rotation techniques the TQKV research ladder defers to. The paper reconstructs each phase's normative contract, its current code state, and its external evidence base, then draws out the cross-cutting design discipline (measure-before-ship, predictor-as-hint, risk-ladder ordering) and the concrete open experiments.

---

## 1. Context and method

### 1.1 The project and the band's place in it

TurboQwenFare exists because of a bet stated in §1: sparse MoE activation, a hybrid recurrent/full-attention architecture, SSD streaming, model-specific kernels, memory virtualization, and continuous measurement can together beat "load the model into RAM and run a generic graph." The hard acceptance contract (§4) is unusually specific:

| Contract | Value |
|---|---|
| Default live memory | **4 GiB** hard working-set budget (2 GiB experimental) |
| Initial / target context | **128K** / ~1M logical tokens |
| Quality ceiling | **≤1%** measured regression |
| Performance floor | **≥15 tok/s** sustained decode; optimize beyond |
| Primary / secondary platform | Base **M4** MacBook Air (Metal); RTX 3070 Ti (CUDA) |
| Implementation | One Rust crate, one binary |

The spec's own sequencing (§113) places this band second: *"Correct model → server → out-of-core experts → 4G broker → 15+ tok/s real decode → TQKV 128K → prefix reuse → 256K/TQAttn → …"*. Phases 0–18 delivered a correct reference; Phases 19–21 began the performance work (parallel I/O, Metal foundation, a global cache broker with a benchmark-selected LRU default). Phases 22–27 are where the *coordination* happens — tiling, prefetch, memory proof, kernel tuning, prefill, and KV compression are all independent mechanisms whose interactions, not their individual existence, determine whether the product contracts are met.

### 1.2 Implementation status (honest baseline)

Per `AGENTS.md` and the source tree, the state entering this band is:

- **≤ Phase 18:** reference/bounded baseline done; a pinned real Q4_K_M checkpoint passes 1-/16-/128-token greedy parity against a pinned external oracle.
- **Phase 19:** parallel `pread` fan-out implemented, default-on, serial path A/B-able; the real-checkpoint wall-time benchmark is still `#[ignore]`.
- **Phase 20:** foundation only — `backend::metal::expert::GpuResidentExpert` uploads one expert's gate/up/down to persistent Metal buffers, but is not wired into the live loop and still dispatches the unfused reference kernel.
- **Phase 21:** partially closed — `WholeExpertLfuCache` with a real-route-trace-selected **LRU** default (`docs/research/qualification/raw-a-128-route-trace-policy.md`).
- **Phases 22–27:** **not started** in runtime code. Tile metadata exists; no tile-level cache unit, predictor, OS-footprint certification, prefill scheduler, or TQKV store yet.

That last sentence is the point of §2.

### 1.3 Method

For each phase, this paper (a) quotes the taskbook definition (§294–§299) and the phase-map exit gate (§112), (b) reconstructs the normative design from the relevant spec sections, (c) reports what the code actually contains, and (d) summarizes the external literature, with numbers quoted from the primary sources. A "Synthesis" subsection then states what the combination implies for implementation.

---

## 2. The shared substrate: tile metadata already exists

Phase 22's instruction to "turn already-present tile metadata into runtime cache units" (§294) is literally satisfiable today. The `.tqf` writer emits, per expert, an `ExpertIndexRecord` plus an array of `ExpertTileRecord`s:

```rust
pub struct ExpertTileRecord {
    pub matrix: ExpertMatrix,   // GateUp | Down
    pub tile_id: TileId,
    pub neuron_start: u16,
    pub neuron_count: u16,
    pub relative_offset: u32,
    pub stored_bytes: u32,
    pub quant_layout_id: u16,
    pub flags: u16,
}
```

The Phase 6 writer deliberately stored **one whole-region tile per matrix** (a single 512-neuron `GateUp` region), because "Phase 6 only needs the metadata shape to exist, not multiple tile widths implemented" (`records.rs` doc comment). The neuron geometry that makes tiling meaningful is §117's LOCKED table: **256 routed experts per layer, 512 neurons wide, top-8 routing plus one shared expert**. An individual expert is therefore small and tileable (§7), and the format was designed so the runtime could investigate partial-expert residency "without a format rewrite" (§17).

Consequence: the entire band is **unblocked on format work**. It is blocked on the runtime learning to treat a *tile* — not a whole expert — as the unit of admission, lease accounting, and I/O.

---

## 3. Phase 22 — Tiled experts

### 3.1 The normative contract

> "Turn already-present tile metadata into runtime cache units. Start with 128-neuron metadata/layout. Compare whole/64/128/256/mixed. Measure syscall count and overread; partial caching that wins hit ratio but destroys I/O latency is rejected." (§294)

Exit gate (§112 row 22): "A/B decides default; format requires no migration."

The design context, from Part VI:

- The cache key is **layer + expert + tile**, capacity is **global** (§41); a hot tile in one layer may deserve tens of MB while a high-entropy layer gets little.
- Tile-aware policy is an explicit research-matrix row (§42): "partial expert residency — measure extra operations/fragmentation."
- The format supports "64, 128, 256, or another measured width," and "different matrices may use different tilings if kernels and I/O benefit" (§34).
- Co-routing-aware physical layout (§35) may reorder expert/tile extents so commonly co-routed combinations sit close together — but **only after** a benchmark shows a net end-to-end win *including cold-cache conditions*, because over-read bytes can erase latency gains.

### 3.2 The external evidence base (deep)

**MoEpic** — *"Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism"* (Yan, Liu, Xu, Huang; arXiv:2509.08342, Sep 2025) — is the closest published relative of Phase 22, and the spec cites it explicitly (§17, R24). Its full mechanism, read from the paper:

1. **Vertical expert split.** Each expert is divided into a *top* and *bottom* segment. MoEpic caches the **top segment of hot experts**, so more experts' hot portions fit under a fixed budget, raising hit rate; the bottom segment is prefetched on demand (or the full expert if it is a cache miss).
2. **Split-ratio sensitivity.** The split ratio θ (top / full expert) is the load-bearing knob. At θ=0 MoEpic degenerates to prefetch-only; at θ=1 to cache-only. In their Qwen1.5-MoE experiments the sweet spot was **θ≈0.4**, reducing latency 37.97%–48.67% versus those two endpoints. Overly small θ leaves a bottom segment too large to hide; overly large θ shrinks the cacheable expert count and kills hit rate.
3. **Priority cache (LCP).** MoEpic records activation frequency μ and activation interval ν (tokens since last activation) per expert, and scores cache priority **P = μ · ρ^(ν/ω)** with ρ=0.25 and window ω=128 — an explicit hybrid of LFU (long-tail distribution) and LRU (temporal locality). At 30 cached experts/layer this yielded **62.21% hit rate vs 60.68% LFU, 56.38% LRU, 49.69% random**.
4. **Adaptive configuration.** Because uniform per-layer budgets are suboptimal (hit rates vary ~5.5% across layers; head/tail layers predict worse), MoEpic solves per-layer budget + split ratio with a **divide-and-conquer fixed-point iteration**.
5. **Headline results.** ~50% GPU cost saved, and **37.51%–65.73% lower inference latency** than cache- and prefetch-based baselines.

The motivation numbers are the most transferable part of the paper: in their setup expert **loading latency was 165 ms vs 57 ms compute per token**, and existing prefetch approaches hid only **34.55%–49.45%** of loading latency. That is exactly the failure mode TQF's Phase 22 rejection criterion ("wins hit ratio but destroys I/O latency") is written to catch.

**FlashMoE** — *"Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for MoE Inference on Edge Devices"* (Kim et al.; arXiv:2601.17063, Jan 2026; §17, R23) — is the closest published result to TQF's *entire* deployment scenario: **SSD, not DRAM, is the backing store**, and it was evaluated on **Qwen3-30B-A3B**, the same model family as TQF's Qwen3.6-35B-A3B. Its full contribution:

- **File-level expert/non-expert separation.** Experts are stored as individual `.pt` files per (layer, expert); only the ~5–7% non-expert weights load at startup, giving **4× faster initial load than llama.cpp and 6.8× vs Fiddler/DAOP**.
- **Belady-approximating ML cache.** A tiny per-layer feed-forward net (3 layers, hidden 128, SiLU, **≈113 KB**) maps normalized recency (1/r) and frequency (f/max f) to an eviction score, trained with MSE loss against Belady-optimal eviction targets mined from routing traces. Training the whole pipeline took ~2 hours on an A100.
- **Why heuristics fail.** Against Belady's 86% hit rate, LRU got ~73% — 1.9× more I/O. LRU's evicted experts were reused **34.2% of the time within 5 steps** (Belady: 0.1%). LRU beat LFU on only ~56% of decisions.
- **Headline results.** Up to **51% higher hit rate than LFU** (21% over LRU), 22%/35% I/O reduction, **up to 2.6× speedup** over existing MoE offloading systems, and ~7% faster than LRU alone on Qwen3-30B-A3B. Expert loading was **>70% of decode time** (FFN ≈158 µs vs SSD load ≈3 ms), which is why a better policy wins so much.
- **Hardware.** A user-grade desktop (RTX 5070 Ti, PCIe 5.0, 7.4 GB/s NVMe), deliberately leaving only ~1 GB of system DRAM available — the same "bounded memory" discipline TQF encodes as a hard wall.

Two other works complete the map. **MoE-Infinity** (Xue et al., arXiv:2401.14361) established the batch-size-one "sparsity-aware expert cache" that traces activation and guides replacement+prefetch, reporting 3.1–16.7× per-token latency wins over vLLM/Ollama/DeepSpeed on DeepSeek/Mixtral. **"In-depth Analysis on Caching and Pre-fetching in MoE Offloading"** (Lin et al., arXiv:2511.05814, Nov 2025) provides the trace-level study of expert activation and LRU/LFU behavior, finding LFU optimizations over LRU and "huge potential" in speculative prefetching.

### 3.3 Synthesis

Phase 22's experiment is *pre-specified by MoEpic*: the spec's "compare whole/64/128/256/mixed" is the neuron-dimension analogue of MoEpic's split-ratio sweep, and the spec's rejection criterion is MoEpic's own central finding restated as an acceptance test. The correct first artifact is an **offline tile-granularity replay** of the already-captured 128-token route trace (`raw-a-128-route-trace-policy.md`), reporting hit ratio *versus* syscall count and over-read bytes at 64/128/256-neuron tilings. FlashMoE supplies the adjacent warning and opportunity: TQF's Phase 21 LRU default is a strong static baseline, but on Qwen3-30B-A3B FlashMoE's learned policy beat LRU by 7% end-to-end — so the spec's deferral of "light learned policy" (§42) is a *later* opportunity, not a Phase 22 requirement.

---

## 4. Phase 23 — Predictive prefetch

### 4.1 The normative contract

> "Implement statistical predictor first. Inputs are recent route transitions/co-routing only. Add hidden-state predictor only after a no-model predictor baseline exists. Log prefetch precision, recall, timeliness, and wasted bytes." (§295)

Exit gate (§112 row 23): "Net SSD-stall reduction without harmful overfetch."

The normative design is §45: prediction "must never alter the model's actual expert routing. It is purely an I/O scheduling hint." The baseline is "a near-free statistical predictor from recent route transitions and co-routing matrices"; a hidden-state predictor may follow only if it pays for itself. The online controller (§46) closes the loop with a fixed policy table:

| Measured predictor state | Default action |
|---|---|
| High precision, reads arrive before demand | Prefetch deeper; consider two-layer horizon |
| Good precision but late reads | Start earlier or raise I/O concurrency |
| Moderate precision | Next-layer-only prefetch |
| Poor precision / heavy overfetch | Disable speculative reads; demand-fill only |
| Thermal/storage throttling | Re-evaluate concurrency |

The hard invariant is §115.7: **predictors may schedule bytes, never alter selected expert IDs or weights.** This is what makes Phase 23 safe: its worst case is wasted SSD traffic, not wrong outputs.

### 4.2 The external evidence base (deep)

The literature converges on "prediction works, but only as an I/O hint, and it must be tuned":

- **MoEpic's speculative prefetcher** predicts next-layer experts by feeding the current layer's intermediate activation into the *next* layer's router (valid because residual connections make adjacent-layer activations highly similar — cosine similarity measured high for Qwen1.5-MoE and Mixtral). Its measured caveat is the same one the spec's table encodes: prediction is accurate in the model's middle but poor at head/tail layers, so budget must shift to those layers.
- **AdapMoE** (cited within MoEpic) reaches **>80% prediction accuracy** using the same activation-reuse trick, yet still hides under half the loading latency — the exact reason prefetch alone is insufficient and must be paired with caching (Phase 22) and tiling.
- **MoE-Infinity** couples its cache with expert-activation prediction and prefetching.
- **ST-MoE** — *"A Spatio-Temporal Expert Prefetching Framework"* (Zhao et al., arXiv:2606.15453, Jun 2026) — makes the correlation claim rigorous and directly relevant to the spec's "route transitions/co-routing" inputs: expert requests correlate strongly **across adjacent layers (spatial) and consecutive tokens (temporal)** within an application domain, so a lightweight runtime predictor that preserves routing can overlap expert loading with compute.
- **NVMAI R16** (recorded in `docs/research/upstream-precedent.md`) is the cautionary local datum: `F_RDADVISE` was beneficial on one M3 host and neutral on another, which is why the spec makes read-ahead host-dependent and never universal.

### 4.3 Synthesis

Phase 23 is the cheapest phase in the band to *start* and among the easiest to do badly. The spec's unusual insistence on logging **precision, recall, timeliness, and wasted bytes separately** is strictly more rigorous than most published systems (which report end-to-end speedup only), and it is the right call: on a passively cooled M4 with one SSD queue and unified memory, prefetch competes with demand reads, and "more reads" can worsen latency (§45's thermal row; R16). The correct posture is benchmark-on-the-target-M4, with the controller table as the only allowed adaptation.

---

## 5. Phase 24 — Hard 4G broker certification

### 5.1 The normative contract

> "Move every large allocation behind leases. Add OS sampler and stress scenarios. Fix hidden allocator spikes before claiming 4G. The 4G profile is not released until helper-model swap and 128K context paths are also accounted later; this phase certifies the inference core first." (§296)

Exit gate (§112 row 24): "Adversarial workloads remain within qualified 4G bound."

This is the phase that converts `--memory` from aspiration to provable contract. The broker is already "the law" (§37): all large allocations — resident weights, expert cache, KV/context, GDN state, scratch, I/O staging, transient helper weights, vision — are centrally accounted, and **"allocate, then report" is prohibited** (§115.4). The pressure algorithm (§38) is a fixed eight-step ladder ending in "return a configuration error; never silently exceed the user limit."

What Phase 24 adds is the *certification* half:

1. **Every large allocation behind a lease.** The code already has `MemoryBroker::reserve(owner, bytes, class)` with `MemoryClass`/`MemoryOwner` enums (`src/memory/mod.rs`), and the GGUF/TQF readers already require broker admission before allocating metadata (`format/gguf/reader.rs`, `format/tqf/reader.rs`). Phase 24 targets the long tail: Metal buffers, scratch arenas, transient staging.
2. **OS-observed accounting** (§132): sample physical footprint/task VM resident metrics, TQF's own reserved/committed totals, Metal heap/buffer totals, and — critically — *peak* during helper-model swaps, vision activation, and context-page transitions. The governing rule is blunt: *"A configuration is not '4G certified' because steady-state decode is 3.9G if loading/reranking spikes to 4.7G."*

### 5.2 The external evidence base

This phase has the least paper precedent in the band — it is a certification discipline, not a technique. The relevant external anchors are:

- **TurboFieldfare** (R7, Apache-2.0; `drumih/turbo-fieldfare`) is the founding precedent: "Gemma 4 26B-A4B inference in ~2 GB of RAM on any M-series MacBook," whose bounded-memory, streaming-installer principles motivated the broker (§14).
- **FlashMoE's methodology** is the closest published analogue of the *spirit* of Phase 24: it deliberately locked system DRAM down to ~1 GB to prove the system genuinely offloads to SSD rather than silently leaning on page cache. That is precisely the discipline §114 forbids TQF from skipping ("Do not silently allocate above `--memory` or rely on unreported OS page cache to make memory claims").
- **NVMAI R17** (production hardening: trusted receipts + schema/path/server/GPU failure hardening) is the provenance/validation precedent.

### 5.3 Synthesis

Phase 24 is a *gate*, not a feature, and it is the phase that most directly protects the project's one non-negotiable promise. Its deliverable is "tests and a sampler," but those are what make the 4G number mean anything. The spec deliberately scopes it to the inference core first, deferring helper-swap and 128K accounting to later phases. A practical implication for planning: the §132 stress scenarios (helper swap, vision activation, page transition, adversarial pressure) can be enumerated and stubbed as test fixtures now, before any of them is implementable.

---

## 6. Phase 25 — M4 short-context assault

### 6.1 The normative contract

> "Use the optimization ledger. Focus current critical path from measured breakdown, not intuition. Likely levers: expert miss bytes/latency; attention/GDN Q4 bandwidth; MoE phase kernels; head bandwidth; overlap/synchronization. Keep optimizing beyond 15 tok/s." (§297)

Exit gate (§112 row 25): "15 is floor; retain ledger of further headroom."

The governing philosophy is §5/§106: throughput is one metric among several (TTFT, prefill time, long-context stalls, tool-call turnaround), 15 tok/s is a floor never a stopping point, and the contributor rules (§114) forbid the two classic cheats — "Do not claim 15 tok/s from a repetitive counting workload" and "Do not measure only GPU kernel time and call it decode time."

The backend design (§48–§53) this phase executes:

- **Unified memory should eliminate gratuitous CPU→GPU copies** (§50): an expert slot is an aligned CPU allocation wrapped as `MTLBuffer` storageModeShared with `pread` filling the bytes the GPU consumes; stable slot addresses avoid per-miss pipeline rebuilds; scratch arenas are reused.
- **Metal I/O queues** (`MTLIOCommandQueue`/`MTLIOCommandBuffer`, R29) are an explicit benchmark candidate against parallel `pread`/read-ahead/shared-buffer — and §21 warns *not* to assume the newer API wins.
- **Kernel specialization** (§51): Q4 GEMV, GDN in-proj, MoE phase-1 (8/16/32 rows/threadgroup), MoE down, LM head, and attention each have function-constant specialization candidates.
- **M4 thermal adaptation** (§53): the base MacBook Air is passively cooled, so first-minute rankings may differ from sustained rankings; the runtime may switch among *already-qualified* variants.

The NVMAI ledger (`upstream-precedent.md`) gives concrete transfer targets: parallel expert `pread` moved decode 9.98→12.80 tok/s (+28%); 64 cache slots + resident pin gained ~10% over 32+pin (128 slots could regress); the MoE phase-1 MSL rewrite cut stage time 14.4→9.24 ms/token with byte-identical output; fused GDN projections gained +6.6% at the stage but **nothing end-to-end**.

### 6.2 The external evidence base (deep)

The single most important online fact for this phase is the **memory-bandwidth budget** of the target hardware. Apple's published figures (Apple Newsroom, Oct 2024; Apple support specs):

| Chip | Unified memory bandwidth | Max unified memory |
|---|---|---|
| **Base M4** (MacBook Air) | **120 GB/s** | 32 GB |
| M4 Pro | 273 GB/s | 64 GB |
| M4 Max | 410–546 GB/s | 128 GB |

Decode on Apple Silicon is memory-bandwidth-bound (this is also KVQuant's §2.2 framing, quoted below). Against 120 GB/s, the 15 tok/s floor for a 35B-A3B Q4 model is a *real* engineering target, not a gimme — which is exactly why the spec forbids measuring kernel time in isolation and demands end-to-end throughput. For calibration:

- **MLX-class runtimes** reach ~230–525 tok/s on an M4 Max for *small* text models (e.g. "Native LLM and MLLM Inference at Scale on Apple Silicon," arXiv:2601.19139, reports up to 525 tok/s on M4 Max).
- Community llama.cpp numbers on *large* MoE models are far lower (single-digit tok/s territory on constrained setups), consistent with bandwidth-bound behavior.

The `MTLIOCommandQueue`/`MTLIOCommandBuffer` APIs (Apple Developer Documentation, R29) are the concrete "load file data directly into Metal resources and synchronize I/O with GPU work" path the spec wants A/B'd against explicit `pread`.

### 6.3 Synthesis

Phase 25 is where the earlier infrastructure pays off and where the project's measurement discipline is most load-bearing. The phase's real product is the **optimization ledger** (§109) plus a measured per-stage breakdown on the target M4. The single highest-value concrete step is closing the still-open reproduce-me items from `upstream-precedent.md` — the parallel-I/O and MoE-phase-1 benchmarks are implemented but `#[ignore]` pending the canonical checkpoint, and without that measured baseline every later phase's claims float. The 120 GB/s bandwidth number should be pinned as the denominator in every Phase 25 cost model.

---

## 7. Phase 26 — Prefill

### 7.1 The normative contract

> "Implement chunk autotuning, expert-set dedup per chunk, and stage instrumentation. Include prompts larger than one chunk and repository-sized contexts." (§298)

Exit gate (§112 row 26): "Long prompts achieve major TTFT reduction."

The reference baseline (§152) is chunked prefill seeded at **4096 tokens on M4**, subject to memory feasibility, with this per-chunk/layer loop:

1. run the token mixer for the chunk;
2. route all chunk rows;
3. collect the **set of distinct experts required by that layer/chunk**;
4. fetch each distinct absent expert **once**;
5. execute expert work for all rows selecting that expert;
6. release chunk scratch before advancing if memory pressure requires.

Chunk size auto-reduces when context/scratch pressure would violate `--memory`. The tuning metric is explicitly **TTFT and total prefill wall, not chunk microkernel speed**.

### 7.2 The external evidence base (deep)

**NVMAI R12** is the direct internal precedent (`upstream-precedent.md`): a 1280-token prompt ran **~13.6 s at 4096-token chunks vs ~43.3 s at 128-token chunks** — a >3× TTFT improvement from chunk size alone, and the "major TTFT reduction" the exit gate names.

**FlashMoE's prefill** is the closest published description of the dedup step: it "identifies which experts are accessed across the entire token batch … loads each required expert file on-demand from the SSD **exactly once per inference iteration**," then redistributes outputs via indexed addition. It also reports a reassuring scaling fact for Phase 26: the number of distinct routed experts grows **sublinearly** with prompt length (47% of experts loaded at 32 tokens → 58% → 64% → 67% as length doubles 32→256), so long prompts do not linearly explode the expert set.

The broader serving literature agrees on the mechanism and adds the scheduling frame:

- **Chunked prefill** (vLLM, TensorRT-LLM, Sarathi-Serve) splits a large prefill into compute-sized chunks interleaved with decode steps to cut TTFT and raise utilization. TensorRT-LLM's guidance emphasizes dividing prefill into manageable chunks for better GPU utilization.
- **Splitwise** (Patel et al., arXiv:2311.18677) and **DistServe** established **prefill/decode disaggregation** — running the compute-bound prefill and memory-bound decode phases on separate resources — reporting 2–7× throughput improvements. TQF is single-machine and batch-1 by design (v1 has one active decode request, `AGENTS.md`), so disaggregation is *conceptually* relevant (prefill is compute-bound, decode is bandwidth-bound) but must be realized as **intra-machine phase separation** (scratch accounting, GPU/I-O overlap) rather than multi-node splitting.

### 7.3 Synthesis

Prefill is where TQF most sharply diverges from "load the model and run a graph." For an out-of-core MoE system, a compute-bound prefill is an **I/O scheduling problem**: dedupe the expert set, prefetch it, overlap it, and release scratch under broker pressure. The 4096 seed is a data point, not a law — the deliverable is a measurement harness that A/Bs chunk sizes on the M4, exactly mirroring the Phase 21 policy selection. FlashMoE's sublinear-expert-growth result and NVMAI's 3× chunk-size effect together suggest Phase 26 has unusually high and unusually low-risk headroom.

---

## 8. Phase 27 — TQKV Q8/Q4 baseline

### 8.1 The normative contract

> "Implement page store/tail lifecycle, Q8 then Q4, and fused/blocked attention. BF16 full cache remains the oracle at smaller contexts." (§299)

Exit gate (§112 row 27): "128K capacity under 4G with full logical attention reference."

TQKV is TQF's physical representation of full-attention K/V history, and it is the most completely pre-specified phase in the band (Part VIII + §154–§162):

- **Logical contract** (§154): `--context N` is logical; internally a hierarchy of immutable completed pages plus one mutable tail page per full-attention layer. At 128K/256K, **no token eviction** is baseline.
- **Page geometry** (§155): 256 tokens/page reference, A/B among 128/256/512/1024. 128K → 512 pages/layer; 1M → ~3907 pages/layer.
- **Lifecycle** (§156): mutable tail → seal (compute quant params/search summary/outliers) → resident compressed → optional precision promotion/demotion and SSD backing.
- **Q8 reference** (§158): **post-RoPE** keys, signed int8 per-channel (FP16 scale per `(kv_head, dim)`); values int8 per 64-dim token group. "The first compressed oracle beneath BF16."
- **Q4 reference** (§159): keys signed int4 per-channel (scale = max_abs/7); values int4 per 64-dim group. **Raw Q4 K+V ≈ 5 KiB/token across the ten full-attention layers; at 128K ≈ 640 MiB** before scale/metadata.
- **Fused consumption** (§161): never materialize a selected compressed page as BF16; online/blocked softmax over streamed dequantized fragments.
- **Precision transitions** (§162): only among *pre-qualified* transitions; a pressure callback "cannot invent a new lower precision."
- **Quality ceiling** (§6): no encoding ships on perplexity alone; the *combined* ≤1% ceiling applies across all optimizations.

### 8.2 The external evidence base (deep)

TQF's Q8/Q4 baseline is a deliberate, conservative subset of a mature literature. Each cited work was read in full:

**KVQuant** — *"Towards 10 Million Context Length LLM Inference with KV Cache Quantization"* (Hooper et al., UC Berkeley; arXiv:2401.18079, NeurIPS 2024; R19). The origin of TQF's Q8 key design, with four methods:

1. **Per-channel Key quantization** — share scale/zero-point along the channel dimension because Keys have outlier *channels*; combined with per-token Values this alone gave a **3.82 perplexity improvement** at 3-bit (LLaMA-7B, Wikitext-2).
2. **Pre-RoPE Key quantization** — quantize Keys *before* the rotary embedding because RoPE mixes channel pairs by position-dependent angles, destroying the per-channel structure (**+0.82 ppl improvement**).
3. **Non-uniform datatypes (nuqX)** — sensitivity-weighted (Fisher-information) k-means signposts derived offline per layer (**+0.29 ppl** over 3-bit uniform).
4. **Per-vector dense-and-sparse** — isolate ~1% of per-channel/per-token outliers into a separate sparse representation (**+0.19 ppl**).

Headline: **<0.1 perplexity degradation at 3-bit** on LLaMA/Llama-2/Llama-3/Mistral, a **4.8×** footprint reduction, 1M-token LLaMA-7B on one A100-80GB (10M on 8 GPUs), and ~1.7× CUDA kernel speedups. The spec's ordering (post-RoPE Q8 *reference* first, pre-RoPE as a research candidate §160.5) is a deliberate "validate the machinery first" decision, **not** a rejection of KVQuant's central finding.

**KIVI** — *"A Tuning-Free Asymmetric 2bit Quantization for KV Cache"* (Liu et al., arXiv:2402.02750, ICML 2024; R20). The asymmetric template: **per-channel Keys, per-token Values**, 2-bit, grouped residual quantization, hardware-friendly. TQF's value encodings (per-token 64-dim groups) are KIVI-shaped.

**TurboQuant** — *"Online Vector Quantization with Near-optimal Distortion Rate"* (Zandieh et al., Google Research/Google DeepMind/NYU; arXiv:2504.19874, ICLR 2026; R18). The geometric framing the spec adopts:

- **MSE stage:** randomly rotate input vectors (coordinates then follow a Beta distribution, near-independent in high dimension), then apply per-coordinate optimal Lloyd-Max scalar quantizers. Provably within a **≈2.7×** factor of the Shannon lower bound (≈1.45× at 1 bit).
- **Inner-product stage:** MSE quantizers are *biased* for inner products, so TurboQuant appends a **1-bit Quantized Johnson-Lindenstrauss (QJL)** transform on the residual, yielding an **unbiased** inner-product estimator with near-optimal error.
- **Results:** **absolute quality neutrality at 3.5 bits/channel**, marginal degradation at 2.5 bits; >5× KV compression with perfect needle-in-a-haystack retrieval. This motivates the spec's rotated-Q3/Q4 and "TurboQuant-inspired vector quantization" candidates (§160.3, §160.6).

**Quest** — *"Query-Aware Sparsity for Efficient Long-Context LLM Inference"* (Tang et al., MIT Han Lab; arXiv:2406.10774, ICML 2024; R21). The page-selection mechanism TQAttn borrows later (§62, §164):

- Per page, store per-dimension **min and max Key values**. For query Q, the optimistic per-page upper bound is **Σᵢ max(Qᵢ·mᵢ, Qᵢ·Mᵢ)**; select the top-K pages by this bound and run real attention only on them.
- **7.03× self-attention speedup, 2.23× end-to-end latency reduction** at 32K context; negligible accuracy loss on PG19/passkey/LongBench.
- The sparsity is layer-dependent: the **first two layers are <10% sparse** (so Quest skips them), later layers >90% sparse.

**Self-Indexing KVCache** — *"Predicting Sparse Attention from Compressed Keys"* (Yang et al., arXiv:2603.14224, AAAI 2026; R22). The unification idea at the far end of TQF's research program (§63):

- A **sign-based 1-bit vector quantization** whose sign codes serve *both* as the quantization representation *and* as the sparse-attention retrieval index — eliminating external indices and learned predictors. Codebook built by **one-pass sign-based clustering** (no iterative k-means).
- Optionally keeps **64 full-precision sink tokens**; fused into FlashAttention via custom LUT-GEMV and sparse-attention CUDA kernels.
- **Up to 5× KV memory reduction, 6.7× sparse-attention speedup, 2× end-to-end latency vs FlashAttention v2.**

### 8.3 The model-architecture fact that makes 4G-at-128K achievable

The arithmetic that decides Phase 27's feasibility is structural, and it is confirmed by online sources as well as the spec. **Qwen3.6-35B-A3B** is a hybrid model: 40 layers organized as **ten cycles of three Gated DeltaNet (recurrent) blocks followed by one full-attention block** (confirmed by Vast.ai model pages, OpenRouter, and a Hugging Face architecture overview; the spec's §117 LOCKED geometry gives 30 GDN layers + 10 full-attention layers, native context 262,144).

The consequence: **only 10 of 40 layers grow with sequence length.** The GDN layers hold constant-size FP32 recurrent state (§11), so the KV store scales as 10/40 of what a dense 40-layer attention model would need. The spec's §159 estimate (~640 MiB raw Q4 K+V at 128K) is precisely this fact applied to the geometry (2 KV heads × 256 head-dim × 10 layers). No single quantization trick carries Phase 27; the 4G exit gate is achievable because the model *only has ten attention layers to compress*, and the literature (KVQuant/KIVI/TurboQuant) shows 2–4-bit is enough for those ten with negligible loss.

> *A small flag for the Phase 0 manifest:* some online summaries describe Qwen3.6-35B-A3B as "512 experts, 10 routed + 1 shared," while the spec's §117 LOCKED geometry says **256 experts/layer, 8 routed + 1 shared, 512 width**. The spec is normative for this project, but the discrepancy is worth a one-line cross-check against the pinned official config (the §272 manifest already has this as an open item for unresolved real-GGUF tensor names).

### 8.4 Synthesis

Phase 27 is the most spec-complete and most literature-heavy phase of the band. Its ordering discipline — **BF16 oracle before Q8, Q8 before Q4, post-RoPE before pre-RoPE** — accepts a suboptimal first encoding to validate the page/seal/fused-decode machinery with low error, then treats KVQuant/KIVI/TurboQuant as a *menu of later upgrades*, each gated by the ≤1% ceiling and a real long-context suite. Quest and Self-Indexing KVCache are explicitly *later* (TQAttn, §164–§165) but their page-granularity requirements are why TQKV's page geometry is fixed the way it is. The one design risk worth surfacing: the spec's Q8 reference stores **post-RoPE** keys "for simplest correctness," which KVQuant's analysis says is measurably worse; the migration path (§160.5 pre-RoPE candidate + §161 fused partial-RoPE consumption) is defined, but implementers should build the Q8 pipeline knowing the pre-RoPE switch is a likely-required, not optional, follow-on before the ≤1% gate at 128K.

---

## 9. Cross-cutting themes

1. **Measurement is the product.** Every exit gate is a benchmark or a certification, and §114 exists to prevent the three self-deceptions (degenerate workloads, kernel-only timing, unqualified approximation). The Phase 21 route-trace replay is the house template: offline, reproducible, policy-independent, and strong enough to change a default.

2. **The broker and the tile are the two load-bearing abstractions.** Phases 22–24 are three applications of one idea — *make residency granular, make memory provable* — and Phases 25–27 are only safe on top of them. Tiling reduces over-read; the broker bounds the cost of being wrong.

3. **Prediction is a hint, never a correctness change.** §115.7 binds Phase 22's co-routing layout, Phase 23's prefetch, and Phase 26's expert-set dedup: predictors schedule bytes; the real router decides expert IDs and weights. This is what makes the whole band A/B-testable rather than a correctness risk.

4. **The literature validates the architecture without dictating it.** MoEpic proves partial-expert residency; FlashMoE proves SSD-offloaded MoE on the Qwen3-30B-A3B family; KVQuant/KIVI/TurboQuant prove 2–4-bit KV; Quest and Self-Indexing KVCache prove page-grained sparse attention. TQF's job is to *re-derive each result against Qwen3.6's specific geometry* (10 attention layers, GQA, partial 64-dim RoPE, 512-neuron experts) — the spec repeats this "re-measure, don't copy" caveat for nearly every citation.

5. **The ordering is a risk ladder, not a queue.** Q8-before-Q4, statistical-predictor-before-learned-predictor, BF16-oracle-before-compression, post-RoPE-before-pre-RoPE, inference-core-certification-before-helper-swap-accounting. Every ordering decision in this band defers the clever-but-risky version until a boring baseline exists to A/B against. That is the single most consistent design principle across all six phases.

6. **The hardware budget dominates everything.** The base M4's 120 GB/s memory bandwidth, not GPU flops, is the denominator for Phase 25; the 30-vs-40 layer structure, not quantization cleverness, is the denominator for Phase 27. Any cost model that does not lead with these numbers is wrong by construction.

---

## 10. Open questions and recommended next work

- **Close the Phase 19/20 benchmark debt first.** The parallel-I/O and MoE-phase-1 reproductions are implemented but `#[ignore]` pending the canonical checkpoint. Without that measured M4 baseline, Phases 22/25 have no reference to beat.
- **Phase 22's first experiment is pre-specified.** Emit 128-neuron tile records for one expert and replay the existing 128-token route trace against a tile-granularity cache simulator, reporting hit ratio *versus* syscall count and over-read bytes at 64/128/256-neuron tilings — MoEpic's split-ratio sweep restated in the neuron dimension.
- **Phase 23's baseline is near-free and its metrics are already named.** A route-transition/co-routing table plus the four logged metrics (precision, recall, timeliness, wasted bytes) is a small, high-value deliverable.
- **Phase 24's scenarios can be stubbed now.** The §132 sampler scenarios (helper swap, vision activation, page transition, adversarial pressure) can be enumerated as test fixtures before any of them is implementable.
- **Phase 27's Q8 path is externally well-lit.** KVQuant (SqueezeAILab) and KIVI have public reference code that can inform — but must not be linked into — the reference decoder, per the one-crate/no-shipped-external-code rule. Build the Q8 pipeline with the pre-RoPE migration (§160.5) planned from day one.
- **Pin the two open facts.** (a) The Qwen3.6 experts-per-layer discrepancy (256 vs 512 online) should be resolved against the official config during Phase 0's manifest cross-check. (b) The 120 GB/s base-M4 bandwidth should be recorded in the Phase 25 cost model as the fixed denominator.

---

## References

**Internal (this repository):**

- `TurboQwenFare_Master_v2_All_Encompassing_Specification.md` — §1, §4–§7, §14–§21, §34–§46, §48–§53, §57–§65, §106, §109, §112–§115, §117, §132, §152, §154–§165, §294–§299.
- `AGENTS.md` — implementation status, invariants, dependency firewall.
- `src/format/tqf/records.rs`, `writer.rs`, `reader.rs` — expert tile metadata (§124).
- `src/experts/mod.rs`, `src/experts/policy.rs` — `WholeExpertLfuCache`, Phase 21 LRU default.
- `src/memory/mod.rs` — memory broker.
- `src/backend/metal/expert.rs`, `kernels.rs`, `buffer.rs` — Phase 20 foundation and reference kernels.
- `docs/research/upstream-precedent.md` — NVMAI/TurboFieldfare findings and reproduce-me checklist.
- `docs/research/qualification/raw-a-128-route-trace-policy.md` — Phase 21 cache-policy selection.

**External (all re-verified online, August 2026):**

- R18 — TurboQuant: "Online Vector Quantization with Near-optimal Distortion Rate," arXiv:2504.19874 (Google Research/DeepMind/NYU, ICLR 2026).
- R19 — KVQuant: "Towards 10 Million Context Length LLM Inference with KV Cache Quantization," arXiv:2401.18079 (UC Berkeley, NeurIPS 2024).
- R20 — KIVI: "A Tuning-Free Asymmetric 2bit Quantization for KV Cache," arXiv:2402.02750 (ICML 2024).
- R21 — Quest: "Query-Aware Sparsity for Efficient Long-Context LLM Inference," arXiv:2406.10774 (MIT Han Lab, ICML 2024).
- R22 — Self-Indexing KVCache: "Predicting Sparse Attention from Compressed Keys," arXiv:2603.14224 (AAAI 2026).
- R23 — FlashMoE: "Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for MoE Inference on Edge Devices," arXiv:2601.17063 (Jan 2026).
- R24 — MoEpic: "Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism," arXiv:2509.08342 (Sep 2025).
- R29 — Apple Developer Documentation: `MTLIOCommandQueue` / `MTLIOCommandBuffer` (Metal resource loading).
- MoE-Infinity: "Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache," arXiv:2401.14361 (Xue et al., 2024).
- "In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading," arXiv:2511.05814 (Lin et al., Nov 2025).
- ST-MoE: "A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference," arXiv:2606.15453 (Jun 2026).
- Splitwise: "Efficient Generative LLM Inference Using Phase Splitting," arXiv:2311.18677 (Patel et al., ISCA 2024).
- Chunked prefill: vLLM optimization guide; TensorRT-LLM chunked-prefill documentation; Sarathi-Serve, arXiv:2403.02310.
- "Native LLM and MLLM Inference at Scale on Apple Silicon," arXiv:2601.19139 (Jan 2026) — MLX throughput on M4 Max.
- Apple Newsroom, "Apple introduces M4 Pro and M4 Max" (Oct 2024) and Apple support spec pages — M4/M4 Pro/M4 Max memory bandwidth.
- Qwen3.6-35B-A3B architecture — Vast.ai model page, OpenRouter model page, Hugging Face architecture overview (hybrid 3:1 Gated DeltaNet / Gated Attention, 40 layers).
