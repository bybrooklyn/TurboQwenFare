**TURBOQWENFARE**

Master Research, Architecture,  
and Implementation Specification — **Master v2**

A complete design and build document for a Qwen3.6-35B-A3B Q4 inference server optimized for bounded memory, huge context, Apple Silicon, NVIDIA, and first-party retrieval.

| **Field**           | **Locked value**                                                             |
|---------------------|------------------------------------------------------------------------------|
| Product             | TurboQwenFare                                                                |
| Binary              | tqf                                                                          |
| Primary model       | Qwen3.6-35B-A3B                                                              |
| Weights             | Q4; canonical Q4_K_M source, lossless repack first                           |
| Default live memory | 4 GiB hard working-set budget                                                |
| Experimental memory | 2 GiB hard-wall research profile                                             |
| Initial context     | 128K logical tokens                                                          |
| Long-context target | ~1M logical tokens                                                           |
| Performance floor   | 15 decoded tok/s on qualified reference workloads; optimize beyond the floor |
| Primary platform    | Base M4 MacBook Air, 16 GiB / 256 GB                                         |
| Secondary platform  | Linux + NVIDIA; RTX 3070 Ti is a mandatory reference                         |
| Implementation      | One Rust crate, one distributed tqf binary                                   |
| License             | Apache-2.0                                                                   |


> **Document purpose:** This is not a pitch deck or a loose idea list. It is the master design record intended to let a new contributor understand why TurboQwenFare exists, reproduce the research logic, implement the system, benchmark it honestly, and know which constraints are locked versus experimental.


**Status: NORMATIVE MASTER ARCHITECTURE + IMPLEMENTATION CONTRACT**

Research sources were rechecked against current primary/official material on August 10, 2026. Derived memory and bandwidth calculations are explicitly labeled as calculations rather than upstream claims.

# Document control

| **Item**           | **Value**                                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------------------------|
| Revision           | Master v2 / August 10, 2026                                                                                           |
| Model decision     | Qwen3.6-35B-A3B is final for this runtime                                                                             |
| Decision policy    | Benchmark-driven implementation details; product locks require explicit change                                        |
| Quality ceiling    | No runtime/context/retrieval optimization may cause \>1% measured model-quality regression                            |
| Performance policy | 15 tok/s is minimum acceptance, never an optimization stopping point                                                  |
| Scope guard        | TQF is an AI inference server. Retrieval and coding-client integration are server capabilities, not an agent harness. |

# Decision-status vocabulary

Every nontrivial implementation statement in Master v2 belongs to one of four statuses. The status controls what an implementation agent may change without reopening product design.

| Status | Meaning | Change rule |
|---|---|---|
| **LOCKED** | Product/architecture contract already decided. | Do not change without an explicit architecture revision. |
| **REFERENCE BASELINE** | Concrete implementation required so correctness/performance can be measured. | Implement first; replace only after an A/B result proves a better path. |
| **RESEARCH CANDIDATE** | Deliberately experimental mechanism. | May fail; record the result in the optimization ledger. |
| **BENCHMARK-SELECTED** | Multiple valid implementations are specified. | The qualified benchmark winner becomes the machine/profile default. |

A phrase such as “benchmark this” is not permission to leave the subsystem unspecified. Master v2 always defines a reference baseline, the candidate variants, the metrics, and the selection/fallback rule.

# How to use this document

A new contributor should read Parts I–III first: purpose, hard locks, model anatomy, feasibility, and prior art. Those sections explain why the runtime looks unusual.

Runtime implementers should then use Parts IV–VIII as the normative subsystem specification: model format, memory broker, expert streaming, backends, context, and scheduling.

Product/server implementers should use Parts IX–XI for APIs, setup, GUI, retrieval, MCP, and integration behavior.

Before merging an optimization, consult the benchmark, quality, and optimization-ledger rules in Part XII. “It seems faster” is not an acceptance criterion.

The phased implementation plan in Part XIII is ordered to establish correctness baselines before black-magic optimizations. Do not skip the reference paths; without them, the ≤1% quality guarantee is untestable.

# Master contents

| **Part** | **Subject**                                                        |
|----------|--------------------------------------------------------------------|
| I        | Mission, Product Contract, and Scope                               |
| II       | Qwen3.6 Anatomy and Feasibility Mathematics                        |
| III      | Research Landscape and Prior Art                                   |
| IV       | Top-Level Runtime Architecture and Source Organization             |
| V        | Model Acquisition, Conversion, and the .tqf Container              |
| VI       | Memory Virtualization, Expert Streaming, and Scheduling            |
| VII      | Apple Metal and NVIDIA CUDA Backends                               |
| VIII     | Long Context: TQKV, TQAttn, Prefix State, and MTP                  |
| IX       | Server, Compatibility APIs, Setup, and Product UX                  |
| X        | TQIndex: First-Party Retrieval, RAG, and TQVec                     |
| XI       | SwiftUI GUI, Integrations, Security, and Operations                |
| XII      | Correctness, Quality, Performance, Research Methodology, and Risks |
| XIII     | Detailed Implementation Roadmap                                    |
| XIV      | Normative Implementation Contracts                                 |
| XV       | Testing, CI, Fault Injection, Release, and Operations               |
| XVI      | Phase-Level Engineering Taskbook                                    |
| A        | Appendices and Reference Material                                  |

**PART I**

**Mission, Product Contract, and Scope**

What TQF is, what it refuses to become, and the constraints every subsystem must respect.

# 1. Product thesis

TurboQwenFare is a local AI inference server specialized around Qwen3.6-35B-A3B Q4. Its external standard is Ollama-level simplicity; its internal standard is a hardware-aware research inference engine. The central bet is that sparse MoE activation, Qwen3.6’s hybrid recurrent/full-attention design, SSD streaming, model-specific kernels, aggressive memory virtualization, and continuous measurement can produce a user experience that conventional “load the model into RAM and run a generic graph” runtimes cannot.

The product must therefore optimize the entire path from model bytes on SSD to accepted output tokens: conversion layout, resident-core placement, expert cache admission, asynchronous reads, Metal/CUDA kernels, KV representation, long-context page selection, prefix reuse, request scheduling, and transient retrieval helpers. No single trick is expected to deliver the project goals. TQF is deliberately a system of coordinated optimizations.


> **The outside must remain boring.** A normal user should be able to run `tqf`, accept one model-download prompt, and receive a local server. The fact that the runtime may be moving expert tiles between SSD and unified memory, changing context precision, learning route statistics, and switching kernels under thermal pressure is implementation detail.


# 2. Non-goals and scope firewall

TurboQwenFare must not become a coding-agent harness merely because coding is an important benchmark and retrieval is useful. The runtime’s identity is “AI inference server.” Code-aware indexing is a retrieval capability; \`--open codex\` is compatibility glue. The model client remains responsible for editing files, running commands, applying patches, managing Git, and conducting agent loops.

- Not an IDE or editor.

- Not an autonomous agent loop.

- Not a shell/terminal orchestrator.

- Not a patch or Git-commit engine.

- Not a generic model-training or quantization toolkit.

- Not a generic tensor framework.

- Not a universal model runtime in v1.

- Not a vector-database product.

- Not a Python environment or plugin host.

This firewall should be reflected in source dependencies: core inference modules may not depend on retrieval, GUI, or client integration code. Optional layers consume the server/runtime; they do not infect it.

# 3. User-facing simplicity contract

**Intended normal command surface**


```text
tqf

tqf --headless

tqf --memory 8G

tqf --context 1M

tqf --enable-vision

tqf --host 0.0.0.0

tqf --model ./compatible-qwen36-q4.gguf

tqf sync .

tqf unsync .

tqf --open opencode

tqf --open claude

tqf --open codex

tqf status

tqf doctor

tqf optimize
```


Ordinary users must not have to understand GGUF variants, expert-cache slots, page-cache pressure, Metal command buffers, KV precision, embedding dimensions, ANN graph parameters, or MTP draft counts. Developer/debug controls may exist behind diagnostic commands or environment flags, but the product interface must prefer automatic policy.

# 4. Hard acceptance contract

| **Constraint**      | **Requirement** | **Interpretation**                                                             |
|---------------------|-----------------|--------------------------------------------------------------------------------|
| Model               | Qwen3.6-35B-A3B | The production hot path may hard-code model shapes.                            |
| Weights             | Q4              | Initial conversion is lossless with respect to the source quantized values.    |
| Memory              | 4 GiB default   | Global live working-set budget, not “VRAM only.”                               |
| Experimental memory | 2 GiB           | A genuine hard wall; no hidden helper-model overflow.                          |
| Context             | 128K initial    | Logical context grows dynamically; no fixed 128K reservation on startup.       |
| Long context        | ~1M target      | 8 GiB may be required initially; API semantics remain stable.                  |
| Quality             | ≤1% degradation | Measured over coding, long-context, tool use, reasoning, and retrieval suites. |
| Speed               | ≥15 tok/s floor | Sustained decode floor; maximum attainable throughput remains the objective.   |
| Apple               | Base M4         | Primary implementation/reference backend.                                      |
| NVIDIA              | RTX 3070 Ti     | Mandatory Linux reference; ≥6 GiB capable NVIDIA GPUs are target class.        |
| Distribution        | One binary      | One Rust crate, no helper executables.                                         |

# 5. Performance philosophy

The project must not accidentally optimize for a benchmark that has unusually repetitive expert routing. NVMAI’s Qwen measurements demonstrate that decode rate can vary strongly by workload because expert-cache locality varies. Consequently, TQF’s headline numbers must include realistic coding and general workloads, and every performance report must show context length, memory budget, cache state, generation length, and hardware.

The optimization objective is user-perceived inference speed, not one scalar. Decode throughput matters, but time-to-first-token, prefill time, long-context stalls, tool-call turnaround, and retrieval latency also determine whether the server feels fast. A 28 tok/s configuration with 2-second TTFT can be preferable to a 31 tok/s configuration that stalls for 14 seconds before generation.

# 6. Quality philosophy

The ≤1% quality ceiling applies to the whole runtime stack. Exact weight-layout transformations should normally be bit/value preserving. Context quantization, sparse attention, retrieval-driven context building, MTP, and other approximate mechanisms must be qualified separately and in combination. The budget is not permission to spend 1% on every subsystem independently.

- Exact/lossless optimizations: prefer deterministic greedy-token parity and layerwise tensor checks.

- Numerically reordered but semantically exact kernels: use strict logit/intermediate tolerances.

- KV compression and sparse attention: measure long-context retrieval, coding, reasoning, perplexity/logit drift, and tool behavior.

- Retrieval: evaluate recall, ranking, downstream task success, and hallucination/false-context failure modes.

- Vision: qualify multimodal behavior separately because text-only mode intentionally keeps vision unloaded.

**PART II**

**Qwen3.6 Anatomy and Feasibility Mathematics**

Exact architecture facts, derived tensor sizes, and why this particular MoE model is suited to out-of-core inference.

# 7. Why Qwen3.6-35B-A3B

Qwen’s official model card describes Qwen3.6-35B-A3B as a 35B-total, 3B-activated causal language model with a vision encoder. The language model uses 40 layers in a repeating 3:1 pattern: three Gated DeltaNet + MoE layers followed by one gated full-attention + MoE layer, repeated ten times. It has hidden size 2048, 256 routed experts, top-8 routing plus one shared expert, expert intermediate width 512, and native context 262,144 with documented extension to roughly 1,010,000 tokens. \[R1\]\[R2\]

This combination is unusually favorable to TQF. Sparse routed experts make the majority of MoE parameters streamable instead of continuously resident. Only ten layers require conventional growing K/V history. The other thirty layers use recurrent Gated DeltaNet state whose main state size is independent of context length. Qwen also ships an MTP component and can be served text-only without loading the vision encoder, directly supporting TQF’s lazy-vision design. \[R1\]

# 8. Canonical language architecture

| **Property**         | **Value**                   | **Why it matters**                                                              |
|----------------------|-----------------------------|---------------------------------------------------------------------------------|
| Total parameters     | 35B                         | Large enough to be genuinely capable; too large for naive low-memory residency. |
| Activated parameters | ~3B/token                   | Sparse compute enables out-of-core expert streaming.                            |
| Hidden size          | 2048                        | Allows highly specialized fixed-shape kernels.                                  |
| Layers               | 40                          | Scheduling loop is fixed and hard-codeable.                                     |
| Layer pattern        | 30 GDN + 10 full attention  | Long-context KV grows in only 25% of layers.                                    |
| GDN heads            | 32 V / 16 QK, dim 128       | Recurrent state is predictable and constant-size with context.                  |
| Full attention       | 16 Q / 2 KV, dim 256        | Low KV-head count makes long context much cheaper than classic MHA.             |
| RoPE dim             | 64                          | Only a sub-dimension is rotary, opening pre-RoPE/compressed-key opportunities.  |
| Experts              | 256/layer                   | Large cold expert pool; strong need for cache policy.                           |
| Top-K                | 8 routed + 1 shared         | 320 routed expert selections per decoded token across 40 layers.                |
| Expert width         | 512                         | Each routed expert is relatively small and tileable.                            |
| Vocab                | 248,320                     | Large output head is a bandwidth-sensitive hot tensor.                          |
| MTP                  | 1 hidden layer in config    | Useful only if accepted-token throughput wins in real tests.                    |
| GDN state dtype      | float32 in reference config | Recurrent-state fidelity must be handled carefully.                             |

# 9. Exact routed-expert weight math

The Transformers implementation represents each routed expert with gate, up, and down projections. With D=2048 and expert intermediate F=512, one expert contains 2048×512 gate weights, 2048×512 up weights, and 512×2048 down weights: 3,145,728 weights total. \[R3\]

**Derived expert-size calculation**


```text
expert_weights = 3 × 2048 × 512
               = 3,145,728 weights
raw_Q4_payload = 3,145,728 × 4 / 8
               = 1,572,864 bytes
               = 1.500 MiB per expert (before Q4 metadata)
```


A token selects 8 routed experts in each of 40 layers, or 320 routed-expert selections. If every selection missed and the on-disk Q4 layout were an ideal raw four bits/weight, routed payload would be 480 MiB/token. NVMAI’s Q4 slot accounting is around 1.55 MiB/expert in its format, implying about 496 MiB/token under 100% miss behavior. That translates to roughly 7.27 GiB/s of expert bytes at 15 tok/s, 9.69 GiB/s at 20 tok/s, and 14.53 GiB/s at 30 tok/s. \[R8–R11\]


> **Interpretation:** The bandwidth calculation is an upper-bound planning model, not a claim that TQF will read that many bytes. Expert hits, partial residency, coalesced reads, predictor overlap, shared-expert compute, OS/storage caching, and MTP can all reduce effective bytes per accepted output token. The optimization target should explicitly track SSD bytes/accepted token.


# 10. Full-attention KV math

Only ten language layers use conventional full attention. Each has two KV heads of dimension 256. For both K and V, BF16 storage therefore requires 10 × 2 × 2 × 256 × 2 bytes = 20 KiB per context token. This is the central long-context capacity calculation.

| **Logical context** | **BF16 KV payload** | **Approx. raw Q4 payload** | **Planning implication**                                               |
|---------------------|---------------------|----------------------------|------------------------------------------------------------------------|
| 4K                  | 0.078 GiB           | ~0.020 GiB                 | Easy.                                                                  |
| 8K                  | 0.156 GiB           | ~0.039 GiB                 | Easy; large expert cache remains.                                      |
| 32K                 | 0.625 GiB           | ~0.156 GiB                 | Still comfortable in 4G.                                               |
| 128K                | 2.50 GiB            | ~0.625 GiB                 | Compression is mandatory for a useful expert cache.                    |
| 256K                | 5.00 GiB            | ~1.25 GiB                  | Low-bit TQKV required in 4G.                                           |
| 1.01M               | ~19.26 GiB          | ~4.82 GiB                  | Compression alone may fit 8G, but attention bandwidth requires TQAttn. |

Raw Q4 is not the final TQKV size because scales, outlier metadata, search signatures, page metadata, and mixed-precision policies add overhead. Conversely, TQKV is allowed to use 2–3-bit cold pages, random rotations, pre-RoPE storage, and richer SSD backing, so the average in-memory bit rate may be lower than four bits while staying inside the ≤1% quality limit.

# 11. Gated DeltaNet state math

The reference recurrent implementation repeats Q/K from 16 key heads to 32 value heads and maintains a state shaped approximately \[32, 128, 128\] in float32 for each GDN layer. That is about 2 MiB per GDN layer, or roughly 60 MiB across 30 layers, plus a relatively small convolution tail. This state is constant-size with context length. \[R2\]\[R3\]

This is one of the most important structural reasons TQF can pursue huge logical context. A classic 40-layer Transformer would grow KV in all 40 layers; Qwen3.6 grows conventional KV in ten. The recurrent state still requires precision, snapshotting, and efficient updates, but it does not scale from hundreds of megabytes to tens of gigabytes as the prompt grows.

# 12. Other high-value tensors

The 248,320×2048 token embedding table and untied output head are each approximately 508.6M weights. At raw four-bit payload they are each roughly 242.5 MiB before quantization metadata. Their runtime behavior differs: only the current input-token row needs to be gathered from the embedding table during decode, while the entire output head participates in next-token logits. Therefore the embedding table is a strong candidate for nonresident/random-row access, while the LM head is a high-value hot tensor and kernel-optimization target.

The dense GDN projection weights, ten full-attention blocks, shared experts, routers, norms, and output head collectively form the always-used core. Exact converted tensor accounting must replace rough planning estimates before the memory broker is finalized. TQF must ship a developer command that prints exact resident candidates by tensor, bytes, quant type, and access frequency.

# 13. Canonical checkpoint set

The current ggml-org Qwen3.6 GGUF repository exposes a Q4_K_M language checkpoint of 20.4 GB, a Q4_0 MTP artifact of 1.06 GB, and a Q8_0 vision projector of 614 MB. The Q4_K_M file publishes SHA-256 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7. TQF should pin an immutable source revision rather than following a moving “main.” \[R4\]


> **Pinned-source rule:** The exact source commit/hash used by a TQF release is part of the release contract. `tqf update` may offer a newly qualified model revision; startup must never silently swap model bytes under an existing benchmark/correctness profile.


**PART III**

**Research Landscape and Prior Art**

What has already been proven, what has failed, what is worth stealing under Apache-2.0, and where TQF must invent new work.

# 14. TurboFieldfare: foundational systems precedent

TurboFieldfare demonstrates the core out-of-core MoE idea on Apple Silicon: keep a shared/resident core in memory, stream only selected routed experts from SSD, maintain a bounded expert cache, overlap shared-expert compute with expert reads, and perform bounded-memory model installation by repacking remote byte ranges directly rather than staging an entire source checkpoint. Its public measurements span an 8 GB M2 Air at roughly 5.1–6.3 tok/s and a 24 GB M5 Pro at roughly 31–35 tok/s for its Gemma 4 target. \[R7\]

TQF should treat TurboFieldfare as permanent upstream research input. The specific model differs, but the system principles—bounded memory, explicit expert streaming, measurement-driven kernel work, and a model-specific format—are directly aligned. Because TurboFieldfare is Apache-2.0, implementation may be adapted with correct attribution rather than re-created from memory.

# 15. NVMAI: direct Qwen3.6 donor and negative-results database

NVMAI is a focused Apache-2.0 fork of TurboFieldfare for Qwen3.6-35B-A3B. It is particularly valuable because it has already implemented the hybrid Qwen graph, Gated DeltaNet, Q4/6/8-bit paths, model repacking, server support, and extensive optimization instrumentation. TQF should mine both code and experiment history, while replacing architectural choices that conflict with the global 4 GiB budget. \[R8\]

| **NVMAI finding**             | **Measured effect / conclusion**                                               | **TQF action**                                                                       |
|-------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Parallel expert pread         | Q4 M3 I/O wall about 41.2→30.9 ms/token; decode 9.98→12.80 tok/s (+28%).       | Port concept immediately; autotune worker count. \[R9\]                              |
| 64 cache slots + resident pin | ~10% decode gain vs 32+pin; 128 slots could regress from pressure.             | Keep pinning lesson, replace fixed per-layer cache with global broker. \[R10\]       |
| MoE phase-1 MSL rewrite       | Stage 14.4→9.24 ms/token; byte-identical deterministic output.                 | Adapt kernel, then specialize harder for M4/Q4. \[R11\]                              |
| 4096-token prefill chunk      | 1280-token prompt ~13.6s vs 43.3s for 128-token chunks; ~13% faster than 1024. | Start autotune around large MoE-aware chunks. \[R12\]                                |
| Fused GDN input projections   | Four Q4 GEMVs fused; stage reduction measured, exact math preserved.           | Adapt and extend fusion. \[R13\]                                                     |
| Targeted F_RDADVISE           | ~10.6% gain in one M3 Q4 test; neutral elsewhere.                              | Autotune per host; never universalize. \[R16\]                                       |
| Persistent KV + GDN snapshots | Demonstrates hybrid-state prefix restore.                                      | Replace monolithic snapshot payloads with deduplicated TQKV page references. \[R14\] |
| CPU MTP drafting              | Output-head bandwidth erased hoped-for CPU advantage.                          | Do not prioritize CPU draft path; keep GPU MTP benchmark-driven. \[R15\]             |
| Stage accounting              | Corrected GPU budget exposed attention and routed MoE as major costs.          | Build detailed timing from day one. \[R8\]                                           |

# 16. What TQF must not copy from NVMAI

NVMAI’s larger expert-cache configurations allocate slots per layer and can consume several gigabytes just for routed expert cache. That design is reasonable on a 24 GB host but conflicts with TQF’s rule that the whole active system should fit in 4 GiB by default. TQF therefore takes the measured lesson—expert residency is enormously valuable—but implements a global byte-budgeted cache that can move capacity toward layers and expert tiles with the highest marginal value.

- Do not inherit a multi-target Swift package architecture; TQF remains one Rust crate/one binary.

- Do not inherit fixed equal per-layer cache budgets.

- Do not inherit text-only scope; TQF supports lazy vision behind \`--enable-vision\`.

- Do not assume MTP is beneficial merely because the model was trained for it.

- Do not expose a wall of expert/cache knobs to normal users; autotune and self-tune instead.

- Do not treat 256K as the final context target; TQF explicitly pursues ~1M.

# 17. MoE caching/offload research

FlashMoE is directly relevant because it treats SSD I/O, not arithmetic, as the key bottleneck in memory-constrained MoE inference. Its paper reports a lightweight ML-based cache policy that combines reuse signals and can improve cache hit rate by up to 51% over LRU/LFU in its evaluated setups, with up to 2.6× system speedup. TQF should use this as evidence that a learned/adaptive admission policy is worth benchmarking, not as permission to copy results to Qwen. \[R23\]

MoEpic provides complementary evidence for partial-expert residency: split experts into segments, cache hot portions, predict next-layer use, and allocate per-layer cache budget adaptively. Its evaluated systems report sizable latency reductions versus its baselines. TQF’s tiled \`.tqf\` format is intentionally designed so the runtime can investigate this class of partial-expert strategy without a format rewrite. \[R24\]

# 18. Long-context KV compression research

TurboQuant is especially aligned with TQF because it treats high-dimensional vectors geometrically rather than as arbitrary scalar blocks. It uses random rotation plus quantization and provides an inner-product-oriented formulation. The paper reports quality neutrality around 3.5 bits/channel and only marginal degradation around 2.5 bits/channel in its evaluated KV-cache tasks. This motivates TQKV’s research into rotated, low-bit, inner-product-preserving key representations. \[R18\]

KVQuant and KIVI show that Keys and Values should not automatically use identical quantization rules. KVQuant emphasizes per-channel Keys, pre-RoPE Key quantization, non-uniform datatypes, and explicit outlier handling; KIVI similarly finds per-channel Keys and per-token Values effective at two bits in its tested models. These techniques are starting hypotheses. Qwen3.6-specific sensitivity must be re-measured because it uses only ten full-attention layers, GQA, partial rotary dimensions, and a different model distribution. \[R19\]\[R20\]

# 19. Long-context sparse attention research

Quest establishes the useful separation between retaining a KV cache and reading every KV page for every query. It scores pages from Key statistics and the current Query, then loads only the most critical pages. This is conceptually close to TQAttn: preserve logical context, always include recent/protected content, and reduce memory movement rather than permanently deleting history. \[R21\]

Self-Indexing KVCache pushes the idea further by designing compressed Key representations that also act as the sparse-attention index. TQF should explicitly investigate whether TQKV Key pages can provide cheap search signatures so a separate million-token ANN structure is unnecessary. \[R22\]

# 20. Vector-index research for TQIndex

Quake and SPFresh are relevant not because TQF should clone either system, but because they demonstrate two requirements of a live code/document index: partitions should adapt to skewed query/update workloads, and local edits should trigger local repair rather than periodic global rebuilds. Quake uses workload-adaptive hierarchical partitioning and dynamic query parameters; SPFresh uses localized in-place partition repair. TQIndex should combine those principles with something generic ANN systems do not possess: repository hierarchy and exact program structure. \[R25\]\[R26\]

CoIR and RepoBench provide external evaluation baselines for code retrieval and repository-scale code completion/retrieval. TQIndex should also add project-specific tests over real repositories because generic benchmark recall does not reveal whether the retriever found the exact implementation, caller, test, configuration, or recent diff an agent actually needed. \[R27\]\[R28\]

# 21. Hardware I/O research

Apple Metal exposes I/O command queues capable of loading file data into Metal resources and synchronizing I/O with GPU work. TQF should benchmark Metal I/O against explicit parallel \`pread\`, read-ahead, and shared-buffer paths on the base M4 rather than assuming the newer API wins. \[R29\]

On NVIDIA, GPUDirect Storage provides an explicit direct DMA path between storage and GPU memory on supported systems, but NVIDIA’s current design guidance lists Quadro/Data Center class GDS-capable GPUs, not consumer GeForce as the baseline. The RTX 3070 Ti path therefore must be designed around explicit asynchronous SSD reads into pinned host staging plus \`cudaMemcpyAsync\`; GDS is an optional fast path on qualifying machines. \[R30\]

**PART IV**

**Top-Level Runtime Architecture and Source Organization**

One crate, one binary, strict dependency direction, explicit execution ownership, and a scheduler that understands Qwen rather than generic graph nodes.

# 22. System decomposition

**Logical architecture**


```text
CLIENTS

OpenAI / Anthropic / Ollama / GUI / MCP / integrations

│

▼

Request Normalizer

│

optional TQIndex RAG

│

▼

Session Scheduler

│

▼

Qwen3.6 Execution Core

┌──────────────┼──────────────┐

▼ ▼ ▼

TQKV/TQAttn Expert Runtime Prefix Runtime

└──────────────┼──────────────┘

▼

Memory Broker

┌───────────┼───────────┐

▼ ▼ ▼

Metal/CUDA CPU SIMD SSD
```


The core runtime owns Qwen execution and memory. Server code normalizes protocols into one internal request representation. Retrieval may augment an input, but the model core remains valid with retrieval entirely disabled. The GUI talks to local server/control endpoints rather than maintaining a second inference implementation in Swift.

# 23. One-crate source tree

The project is intentionally not split into a Cargo workspace. Logical boundaries live as Rust modules and folders inside one crate. This avoids internal package-management overhead while still allowing clear unsafe/FFI boundaries, feature-gated platform code, and unit testing.

**Normative module layout**


```text
src/

main.rs

build.rs

app/

cli/

config/

setup/

format/{tqf,gguf,quant}/

model/qwen36/

runtime/

memory/

experts/

context/{tqkv,tqattn,prefix}/

io/

backend/{metal,cuda}/

simd/

tokenizer/

sampling/

vision/

server/{openai,anthropic,ollama,tqf_api}/

retrieval/

mcp/

integrations/

gui/macos/

metrics/

bench/

dev/
```


# 24. Dependency firewall

| **From**      | **May depend on**                                  | **Must not depend on**       |
|---------------|----------------------------------------------------|------------------------------|
| model/runtime | memory, backend, IO, tokenizer, sampling           | retrieval, GUI, integrations |
| server        | runtime, retrieval facade, protocols               | GUI                          |
| retrieval     | memory broker, helper-model runtime, parsers, SIMD | GUI, coding-client internals |
| GUI           | local control/server interfaces                    | model internals              |
| integrations  | server/MCP launch/config helpers                   | model kernels                |
| backend       | platform FFI, shared tensor metadata               | retrieval/product logic      |

# 25. Async and thread model

Tokio should handle non-differentiating asynchronous control work: HTTP, SSE transport, downloads, filesystem watching, MCP, process integration, and background maintenance. The token-critical inference loop must use dedicated threads/queues and explicit GPU/I/O synchronization. The goal is deterministic control over when router results become visible to the CPU, when expert reads begin, which buffers they fill, and which command buffer consumes them.

The runtime should maintain a single active decode request in v1 and queue others, with clean cancellation. Later continuous batching may share the loaded model across sessions, but batch-1 latency/throughput remains the core performance objective and must not regress silently.

# 26. Internal request/session model

**Illustrative internal API (not final Rust syntax)**


```text
struct NormalizedRequest {

protocol: ProtocolFlavor,

messages: Vec<Message>,

tools: Vec<ToolDefinition>,

sampling: SamplingParams,

logical_context_limit: usize,

retrieval: RetrievalPolicy,

vision: Vec<VisionInput>,

stream: bool,

}

struct Session {

id: SessionId,

token_history: TokenStore,

context: ContextState,

prefix: Option<PrefixHandle>,

cancellation: CancellationToken,

}
```


Protocol-specific semantics are normalized at the boundary. OpenAI Responses events, Chat Completions chunks, Anthropic Messages events, and Ollama streams should all consume one generation event stream internally. This avoids protocol handling inside the model loop.

# 27. Unsafe policy

Early research may use \`unsafe\` aggressively in the places that genuinely need it: Metal/CUDA FFI, aligned/mapped storage, SIMD, zero-copy buffer wrapping, Swift bridging, and direct I/O. As paths stabilize, the project should move unsafety behind small audited modules and safe ownership APIs. The objective is not a cosmetically low unsafe-line count; it is a comprehensible safety boundary without sacrificing performance.

**PART V**

**Model Acquisition, Conversion, and the .tqf Container**

How a user gets from one binary to a verified hardware-optimized Qwen installation without knowing anything about model formats.

# 28. First-run state machine

**First-run lifecycle**


```text
START

│

├─ detect OS / CPU / GPU / memory / storage

├─ load global config and hardware profile

├─ validate trusted model receipt

│ └─ missing/invalid → model setup

│

├─ model setup asks: Download and optimize? [Y/n]

│ ├─ n → exit

│ └─ y → resolve pinned source → download/range-read → verify → repack → finalize

│

├─ short hardware autotune

├─ start server

└─ on macOS desktop, launch SwiftUI unless --headless
```


The setup process must be transactional. An interrupted network transfer or conversion leaves a resumable partial installation, never a model directory that appears valid. The final \`.tqf\` becomes visible to the runtime only after all required extents, metadata, and checksums validate and the final atomic rename/receipt write succeeds.

# 29. Source resolution and pinning

A release embeds exact expected source IDs and hashes. The canonical path is a pinned Q4 language checkpoint plus matching MTP data; vision is lazy/optional. Experimental \`--model\` imports may accept compatible Qwen3.6 Q4 files, but they do not inherit the canonical performance guarantee until separately qualified.

If TQF itself downloads a temporary source artifact solely for conversion, it may delete that temporary source after successful verified conversion. If the user pointed TQF at an existing file or an Ollama-owned blob, TQF must not delete the source. Ownership is explicit, not inferred from filename.

# 30. Streaming remote repack

The ideal installer never materializes a complete source GGUF and complete target \`.tqf\` simultaneously. When HTTP range access and source metadata allow it, TQF should fetch bounded source ranges, verify them, transform Q4 block layout if necessary, and write directly to final target extents. This follows the bounded-storage spirit proven by TurboFieldfare while adapting the container to Qwen and TQF’s tile layout. \[R7\]

- Download/range requests are resumable and independently checkable.

- Conversion scratch memory is bounded through the same memory-budget philosophy, although setup may use a separate explicit cap from live inference.

- Target extents may be written with \`pwrite\` in platform-optimal order.

- A small journal tracks completed verified extents.

- Finalization verifies manifest invariants, source identity, critical hashes, target table checksums, and section boundaries.

# 31. \`.tqf\` format goals

\`.tqf\` is not a general model exchange format. It is an execution container for TurboQwenFare. Its job is to let the runtime locate and consume the exact bytes needed by Qwen’s fixed graph with minimal parsing, minimal pointer chasing, predictable alignment, and direct compatibility with expert streaming and platform-specific Q4 kernels.

- Single file per major model/helper artifact.

- Platform/backend-specific physical layout is allowed.

- Strict model/source fingerprinting.

- Explicit tensor/extents rather than arbitrary framework object graphs.

- Expert tiles addressable independently.

- Resident core logically separated from cold expert store.

- Optional duplicated tiny layouts supported.

- Versioned, checksummed, forward-migration aware.

- Lossless source-Q4 semantics in the first production format.

# 32. Proposed \`.tqf\` physical structure

**Conceptual \`.tqf\` layout**


```text
+------------------------------+ 0

| Superblock |

| magic / version / backend |

| model ID / source hash |

+------------------------------+

| Architecture metadata |

| quant schema / tokenizer |

+------------------------------+

| Resident core extent(s) |

+------------------------------+

| Embedding / LM-head extents |

+------------------------------+

| Routed expert store |

| layer → expert → matrix/tile |

+------------------------------+

| MTP data |

+------------------------------+

| Optional duplicate layouts |

+------------------------------+

| Extent/index tables |

+------------------------------+

| Checksum table |

+------------------------------+

| Footer / table root hash |

+------------------------------+
```


# 33. Superblock and versioning

| **Field**                  | **Purpose**                                                                       |
|----------------------------|-----------------------------------------------------------------------------------|
| Magic                      | Reject non-TQF input without probing.                                             |
| Format major/minor         | Major incompatibility triggers reconversion; minor supports controlled migration. |
| Backend/layout tag         | Apple Metal, CUDA Ampere, later variants.                                         |
| Model family ID            | Qwen3.6-35B-A3B production architecture.                                          |
| Source revision/hash       | Reproducibility and trusted-install validation.                                   |
| Quant schema               | Source Q4 semantics and physical TQF Q4 packing version.                          |
| Extent-table offset/length | Constant-time navigation.                                                         |
| Checksums/root hash        | Corruption detection.                                                             |
| Feature bits               | MTP, vision linkage, tiled experts, duplicate layouts, etc.                       |

# 34. Expert tile addressability

The format should support tile IDs even before partial caching becomes the default. This prevents the runtime from being locked into whole-expert fetches. A routed expert can expose gate/up/down matrices as independently addressable regions or fused tile groups; tile granularity can be 64, 128, 256, or another measured width. Different matrices may use different tilings if kernels and I/O benefit.

**Illustrative expert-extent metadata**


```text
ExpertExtentKey {

layer: u8,

expert: u16,

matrix: Gate | Up | Down | FusedGateUp,

tile: u16,

}

ExtentRecord {

file_offset: u64,

stored_bytes: u32,

logical_shape: Shape,

quant_layout: QuantLayoutId,

alignment: u32,

checksum: ...

}
```


# 35. Co-routing-aware physical layout

TQF may reorder expert extents based on a shipped calibration corpus or machine-local setup trace so commonly co-routed expert/tile combinations sit closer together. The point is not to “train” the model; it is to reduce random I/O operations and create opportunities for one larger read to satisfy several predicted misses. This optimization must be compared against deterministic canonical ordering because extra bytes from over-reading can erase latency gains.


> **Format discipline:** Do not make a 20+ GB final conversion depend on an unproven layout theory. The converter should be capable of deterministic canonical output first. Co-routing layout becomes enabled only after the benchmark harness demonstrates a net end-to-end win on the target M4, including cold-cache conditions.


# 36. Trusted receipt and startup validation

A successfully installed model receives a compact trusted receipt containing source identity, expected file length, format version, architecture fingerprint, conversion implementation version, and target root checksum. Startup performs cheap metadata/receipt checks and selective probes. Full 20+ GB hashing is reserved for invalid/missing receipts, explicit \`tqf doctor\`, or suspected corruption. NVMAI’s hardening work supports this design direction. \[R17\]

**PART VI**

**Memory Virtualization, Expert Streaming, and Scheduling**

The core of TQF: a global budget manager, adaptive expert residency, explicit I/O, predictive overlap, and an online controller.

# 37. The memory broker is the law

\`--memory\` is a hard runtime contract. It is not an advisory cache size. All large allocations—resident weights, expert cache, KV/context state, GDN state, scratch, I/O staging, transient embedding/reranker weights, and active vision weights—must be accounted by one central broker. The implementation must include enough OS/backend overhead reserve that “4G” does not routinely produce a 5G process footprint through untracked framework allocations.

**Illustrative broker vocabulary**


```text
enum MemoryClass { Fixed, Protected, Elastic, Transient, Backing }

enum MemoryOwner {

Core, GdnState, ContextHot, ContextCold,

ExpertPinned, ExpertProbation, IoStaging, Scratch,

Embedder, Reranker, Vision, ServerReserve, GuiReserve,

}

let lease = broker.reserve(owner, bytes, class)?;
```


# 38. Broker pressure algorithm

1.  Reject impossible arithmetic before allocation: requested context + mandatory model core + protected runtime reserve must fit some validated plan.

2.  Reclaim completed scratch and expired transient buffers.

3.  Shrink low-value expert probationary cache using byte-value ranking.

4.  Reduce speculative prefetch staging/concurrency if staging pressure is high.

5.  Demote eligible TQKV pages to lower validated precision or move richer backing to SSD.

6.  Unload helper models immediately after embedding/reranking completes.

7.  If vision is active, shrink elastic expert/context acceleration before violating protected correctness state.

8.  Retry reservation; if no valid plan exists, return a configuration error. Never silently exceed the user limit.

# 39. Example 4 GiB live plans

| **Workload**    | **Core+GDN** | **Context**       | **Expert cache**    | **Transient/scratch/I/O** | **Notes**                                    |
|-----------------|--------------|-------------------|---------------------|---------------------------|----------------------------------------------|
| Short chat      | ~1.1–1.3G    | small             | largest share       | ~0.2–0.4G                 | Maximize expert residency.                   |
| 128K coding     | ~1.1–1.3G    | ~0.4–0.8G target  | ~1.4–2.0G           | ~0.2–0.4G                 | Exact sizes benchmark-derived.               |
| Embedding query | same         | session protected | temporarily reduced | embedder loaded           | Expert bytes give way briefly to pplx model. |
| Reranking       | same         | session protected | temporarily reduced | reranker loaded           | GTE unloaded before decode.                  |
| Vision request  | same         | context retained  | reduced             | vision active             | If impossible, offer larger memory budget.   |

These are planning envelopes, not hard-coded quotas. The runtime should continuously solve a resource-allocation problem from measured value per byte. “Expert cache = 1.6 GiB forever” is intentionally not the design.

# 40. Experimental 2 GiB plan

The 2 GiB profile is a deliberate moonshot. It must remain a genuine global wall even while retrieval helper models are used. The broker may therefore collapse expert residency almost completely during a semantic embedding/rerank operation, unload the helper, and rebuild the high-value expert working set before generation. The profile can use more aggressive context compression, smaller prefill chunks, smaller staging, and more SSD traffic, but not a hidden memory escape hatch.

The research acceptance sequence is staged: first prove correct Q4 generation under 2 GiB; then 128K logical context with ≤1% quality degradation; then attack decode toward the same 15 tok/s floor. A failure to hit the final speed goal does not invalidate the production 4G system, but the 2G work may produce useful cache and compression techniques that feed back into 4G.

# 41. Global adaptive expert cache

The cache key is layer+expert+tile, not “slot N in layer L.” Capacity is globally allocated. A frequently reused expert tile in one layer may deserve tens of megabytes while a high-entropy layer receives little protected residency. This is the principal architectural divergence from fixed per-layer cache designs.

Admission and retention should estimate expected avoided stall per byte. Candidate signals include decayed frequency, recent reuse distance, transition probability from previous token/layer, co-routing, actual measured read latency, tile size, predictor confidence, and whether an extent can be coalesced with another pending read.

**Conceptual cache value; final cost model is benchmark-driven**


```text
value(tile) ≈ P(reuse soon) × expected_miss_latency_saved × critical_path_factor
              ────────────────────────────────────────────────────────────────
                                   resident_bytes
```


# 42. Cache policy research matrix

| **Policy**             | **Purpose**                           | **Required comparison**                        |
|------------------------|---------------------------------------|------------------------------------------------|
| LRU                    | Recency baseline                      | Simple control.                                |
| LFU                    | Frequency baseline                    | TurboFieldfare/NVMAI-like baseline.            |
| Decayed LFU            | Track workload shift                  | Avoid permanent historic hotness.              |
| TinyLFU-like admission | Protect cache from one-off misses     | Measure metadata cost.                         |
| Transition-aware       | Exploit token-to-token route patterns | Compare precision/overfetch.                   |
| Co-routing-aware       | Keep groups likely used together      | Measure random-I/O reduction.                  |
| Cost-aware             | Prefer expensive misses               | Needs per-extent latency model.                |
| Tile-aware             | Partial expert residency              | Measure extra operations/fragmentation.        |
| Light learned policy   | FlashMoE-inspired                     | Only if its own compute/memory cost pays back. |

# 43. Parallel miss filling

For each layer, the router produces top-8 expert IDs. Cache planning reserves destination slots/tiles before issuing reads. Independent misses may then use parallel explicit reads into distinct buffers. The bookkeeping critical section should be small; the read itself should not hold a global cache lock. NVMAI’s measured +28% example makes this an early implementation priority. \[R9\]

**Decode miss-fill pipeline**


```text
router → top8

│

▼

cache plan

┌──────┴──────┐

hits misses

│

reserve independent slots

┌────┬────┬────┐

▼ ▼ ▼ ▼

read read read read (bounded concurrency)

└────┴────┴────┘

│

Metal-visible data
```


# 44. Shared-expert overlap

The shared expert is always active and should remain resident. After router results are available, TQF should launch any shared-expert work and cache-hit routed work that can proceed while miss reads are in flight. The scheduler should reason in critical-path time: compute that does not shorten the I/O stall is still useful if it occupies otherwise idle GPU cycles without contending destructively for unified-memory bandwidth.

# 45. Predictive prefetch

Prediction must never alter the model’s actual expert routing. It is purely an I/O scheduling hint. TQF should begin with a near-free statistical predictor from recent route transitions and co-routing matrices. A tiny hidden-state predictor may be added if it improves prefetch precision enough to justify compute and memory. The online controller reduces prefetch depth when wrong predictions waste SSD bandwidth.

| **Measured predictor state**               | **Default action**                                               |
|--------------------------------------------|------------------------------------------------------------------|
| High precision, reads arrive before demand | Prefetch deeper; consider two-layer horizon.                     |
| Good precision but late reads              | Start earlier or raise I/O concurrency.                          |
| Moderate precision                         | Next-layer-only prefetch.                                        |
| Poor precision / heavy overfetch           | Disable speculative reads; demand-fill only.                     |
| Thermal/storage throttling                 | Re-evaluate concurrency because “more reads” may worsen latency. |

# 46. Online performance controller

A self-tuning controller observes decode rate, TTFT, per-stage GPU time, SSD bytes/token, read latency, cache hit rate, prefetch accuracy, context-attention cost, and sustained performance. It can modify safe scheduling/resource parameters online. This is deliberately closer to an OS resource scheduler than a static inference configuration.

- Expert-cache partitioning and tile promotion/demotion.

- I/O worker count and request batching/coalescing.

- Read-ahead strategy and prefetch depth.

- TQKV precision distribution within validated bounds.

- TQAttn page budget and recent exact window.

- Background index/embedding priority.

- Kernel variant when sustained thermal behavior makes a different variant faster.

# 47. SSD write discipline

Inference should be read-heavy and write-light. Route events, performance counters, and filesystem-change storms are aggregated in RAM. Periodic compact atomic checkpoints persist only useful summaries. Prefix/index/KV backing writes should be batched and deduplicated where practical. Do not turn a high-throughput read-streaming engine into an SSD-write-amplification machine.

**PART VII**

**Apple Metal and NVIDIA CUDA Backends**

Native backends that share semantics but do not force each other into a lowest-common-denominator abstraction.

# 48. Backend interface philosophy

Metal and CUDA share high-level operations and scheduler semantics, not kernel implementation. The common layer defines Qwen operations, buffer leases, timing events, and dependencies. Each backend may choose radically different memory paths, launch geometry, compilation, and synchronization if it preserves the execution contract.

**Illustrative semantic boundary**


```text
trait Backend {

fn q4_gemv(...);

fn gdn_decode_step(...);

fn full_attention_decode(...);

fn moe_gate_up(...);

fn moe_down_accumulate(...);

fn lm_head(...);

fn tqkv_attention(...);

fn event/timing/synchronize(...);

}
```


# 49. macOS/Metal ownership

Rust owns inference. Use mature Objective-C/Metal bindings for device/queue/resource management, with explicit low-level FFI where bindings become limiting. Native MSL implements performance kernels. Naga may be used for utility/reference shaders or translation experiments, but production kernels have no obligation to be expressible through WGSL/Naga.

# 50. Metal resource strategy

Apple unified memory should eliminate gratuitous CPU→GPU copies. A baseline expert slot can be an aligned CPU allocation wrapped as \`MTLBuffer\` storageModeShared, with \`pread\` filling the same bytes the GPU consumes. TQF should also benchmark Metal I/O queues, which Apple explicitly provides for filesystem-to-resource loading and synchronization with GPU work. \[R29\]

- Resident core may be mapped/read-only and pinned or copied into a stable Metal-visible allocation based on measured behavior.

- Expert slots should use stable addresses so the GPU pipeline does not rebuild resources every miss.

- Scratch arenas should be reused; no token-loop allocation churn.

- Pipeline-state objects are compiled/cached once per specialization.

- Command-buffer errors are fatal to the request and propagated; never continue with corrupt outputs.

# 51. Metal shader packaging and specialization

Ship a known-good baseline metallib so the binary can start. Also embed MSL source or specialization templates so first-run tuning can compile M4-specific function-constant variants. Persist chosen pipeline configurations in the machine profile. If the OS/GPU materially changes, invalidate only affected measurements rather than rerunning every benchmark.

| **Kernel family** | **Initial specializations to test**                                     |
|-------------------|-------------------------------------------------------------------------|
| Q4 GEMV           | load width, group layout, rows/TG, nibble unpack, source-vs-TQF packing |
| GDN in-proj       | fused 12,352-row projection, function constants, shared activation      |
| MoE phase 1       | 8/16/32 rows/TG, threadgroup-staged x, gate/up fusion                   |
| MoE down          | accumulation geometry, top-K fusion, tile layout                        |
| LM head           | Q4 bandwidth, fused max/top-k sampling path                             |
| Attention         | QKV fusion, partial RoPE, GQA layout, TQKV fused decode                 |
| TQVec             | binary Hamming/popcount, low-bit dot/cosine refinement                  |

# 52. NVMAI kernel lessons to carry forward

NVMAI’s experiments show that model-specific specialization can be worth real bandwidth. Its MoE phase-1 threadgroup staging materially improved that stage, and its fused GDN input projections eliminated small launches while retaining exact output. It also measured that some seemingly plausible threadgroup/staging changes made no end-to-end difference or regressed. TQF’s kernel work should therefore use isolated microbenchmarks and full decode A/B tests; microbenchmark wins do not automatically survive contention and overlap. \[R11\]\[R13\]

# 53. M4 thermal adaptation

The base MacBook Air is passively cooled. First-minute kernel rankings may differ from sustained rankings. TQF should watch actual stage latency/throughput over long runs and may switch among already-qualified variants when sustained load changes the fastest strategy. This is performance adaptation, not a “battery saver” mode. If two strategies are essentially tied, prefer the one with less power/memory/SSD traffic.

# 54. CUDA baseline for RTX 3070 Ti

The GeForce production path assumes explicit SSD reads into a bounded pinned-host staging pool followed by \`cudaMemcpyAsync\` into VRAM cache slots. CUDA streams overlap read completion, H→D transfer, and compute. TQF must not require GPUDirect Storage for the RTX 3070 Ti because NVIDIA’s GDS suitability guidance targets GPUDirect-capable Quadro/Data Center GPUs. \[R30\]

**NVIDIA pipeline objective**


```text
NVMe read (N+2) ────────────────┐

Pinned host → H2D (N+1) ───────┼─ overlapped

CUDA compute expert N ──────────┘
```


# 55. CUDA kernel distribution

The Linux binary should not require an installed CUDA toolkit at runtime. Embed PTX and/or architecture-targeted cubins/fatbins as appropriate, use the installed NVIDIA driver/runtime interfaces, and cache JIT results if PTX specialization is necessary. Ampere is the minimum design baseline; later architecture-specific kernels may be selected by compute capability.

# 56. CPU SIMD role

CPU SIMD is used where it creates overlap rather than competing destructively with GPU bandwidth: tokenizer hot paths, index binary/vector search, cache-policy computations, hashing, route statistics, query routing, and SSD scheduling. On Apple unified memory, large CPU inference GEMVs may simply steal bandwidth from Metal. NVMAI’s CPU MTP-draft result is a strong warning against assuming idle cores equal free compute. \[R15\]

**PART VIII**

**Long Context: TQKV, TQAttn, Prefix State, and MTP**

Making 128K normal, 256K practical, and ~1M usable without pretending memory capacity is the only problem.

# 57. Long-context design principles

TQF presents a logical context window to API clients. The internal representation is allowed to change silently—precision, page residency, indexing, SSD backing, recent exact window—provided logical behavior remains within the ≤1% quality ceiling. “Compress” does not mean “truncate.” At 128K and 256K, no token eviction is part of the baseline design.

- Grow state dynamically; do not reserve the maximum context at startup.

- Quantize before considering deletion.

- Keep system/developer/tool/project-critical pages protected.

- Separate capacity from bandwidth: a 1M cache that fits but must be scanned every token is still too slow.

- Measure long-context speed at an already-populated context, not only short prompts with a large configured limit.

# 58. TQKV goals

TQKV is TQF’s physical representation of full-attention K/V history. It should support mixed precision and fused attention consumption so compressed pages are not expanded into giant FP16 buffers. Keys and Values may use asymmetric methods. The format should support pre-RoPE Key representations where Qwen-specific correctness/efficiency benefits are demonstrated.

# 59. TQKV page schema

**Conceptual context page**


```text
TQKVPage {

token_start, token_count,

key_encoding, value_encoding,

key_payload, value_payload,

scale_or_codebook_metadata,

optional_outlier_payload,

search_signature,

precision_class,

protection_flags,

backing_page_id,

}
```


Candidate page sizes such as 128, 256, 512, and 1024 tokens must be measured. Smaller pages improve selective attention granularity and precision adaptation but increase metadata, index work, and I/O operations. Larger pages improve sequential access but force over-reading at 1M.

# 60. Key/Value asymmetric compression

Initial TQKV experiments should include per-channel Key quantization, per-token Value quantization, structured/random rotations, explicit outlier sidecars, and low-bit cold pages inspired by KVQuant/KIVI/TurboQuant. Qwen3.6 has only a 64-dimensional rotary subspace within 256-dimensional full-attention heads, so pre-RoPE Key storage plus fused partial rotation during attention deserves dedicated investigation. \[R18–R20\]

# 61. Mixed precision policy

**Illustrative internal precision ladder**


```text
recent / critical → Q8/Q6-like

warm → Q4-like

cold → Q3-like

very cold at 1M → Q2/Q3 search representation + richer SSD backing
```


These are internal encoding classes, not user quality modes. A page can be demoted only if the validated error model permits it. Promotion cannot magically restore information discarded by lower precision; therefore 1M designs may retain richer compressed backing pages on SSD so an important old page can be fetched at higher fidelity when selected.

# 62. TQAttn: context preserved, reads selected

TQAttn addresses the bandwidth problem at huge context. It always includes a dynamically sized recent window and protected pages, then scores older pages for the current query. Selected pages receive the real attention computation; unselected pages remain stored and may be selected by a future query. This follows the core insight of Quest without binding TQF to Quest’s exact page statistic. \[R21\]

**TQAttn conceptual flow**


```text
1M logical context

│

├─ recent exact window ─────────────── ALWAYS

├─ protected instruction/tool pages ─ ALWAYS

└─ older TQKV pages

│

cheap scoring / self-index

│

top relevant pages

│

full attention compute
```


# 63. Self-indexing compressed keys

TQF should investigate a Key encoding that can answer two questions from essentially the same bytes: “is this page likely important for the current query?” and “what are the actual approximate Key values needed for attention?” Self-Indexing KVCache provides precedent for this unified compression/retrieval idea. A TQF version may use binary sign signatures, MRL-like multiresolution representations, rotated low-bit vectors, or page summary bounds, but must be evaluated on Qwen’s partial-RoPE GQA geometry. \[R22\]

# 64. Protected anchors

The request parser knows message roles and, for retrieval-enabled sessions, context provenance. TQAttn can therefore enforce hard page inclusion for system/developer instructions, tool schemas, explicit project instructions, current user content, and explicitly pinned retrieved material. Structural importance is a correctness signal, not merely a recency score.

# 65. Context profiles

| **Profile** | **Memory target**    | **Attention strategy**                                                                              | **Quality target**                 |
|-------------|----------------------|-----------------------------------------------------------------------------------------------------|------------------------------------|
| 128K / 4G   | Default              | Full logical attention baseline; TQKV compression; production may use proven transparent shortcuts. | ≤1% regression; strive much lower. |
| 256K / 4G   | Production extension | Attempt full; allow TQAttn if required for throughput.                                              | ≤1%.                               |
| ~1M / 8G    | Advanced             | TQKV + TQAttn + SSD backing + dynamic exact window.                                                 | ≤1%; ≥15 tok/s floor.              |
| 128K / 2G   | Experimental         | More aggressive TQKV + tiny expert cache; no hidden memory overrun.                                 | ≤1%; eventually ≥15 tok/s.         |

# 66. Prefix snapshots

Exact prefix reuse is especially powerful in coding and tool workflows where system prompts, tool schemas, repository maps, and prior conversation prefixes repeat. A Qwen3.6 snapshot needs both conventional full-attention KV state and Gated DeltaNet recurrent/convolution state. NVMAI proves this hybrid-state restore path is implementable. \[R14\]

TQF improves storage by deduplicating TQKV pages. Snapshots reference immutable page IDs plus a GDN state checkpoint rather than serializing complete K/V history every time. Candidate checkpoint points include message boundaries, tool-result boundaries, retrieved-document/file boundaries, and adaptive periodic intervals.

# 67. Prefix cache matching

v1 uses the longest exact token-prefix match only. This is safe, deterministic, and compatible across clients. The server may canonicalize semantically irrelevant serialization details that it controls—for example stable tool-schema ordering—only if doing so cannot change model semantics. Structural “middle of prompt changed” reuse is a separate research project and should not complicate the first implementation.

# 68. MTP policy

Qwen3.6 ships MTP and official serving examples expose speculative configurations, but NVMAI measurements show MTP can reduce throughput under some cache/I/O conditions. Therefore TQF implements MTP as another schedule candidate. The controller turns it on only when accepted tokens/second increases under the current context, cache state, and hardware. \[R1\]\[R8\]

Measure accepted tokens per verification, unique expert bytes touched, union of routed experts across draft tokens, LM-head cost, and SSD bytes per accepted output token. A draft mechanism that predicts multiple tokens but causes substantially more unique expert fetches can lose despite fewer target iterations.

**PART IX**

**Server, Compatibility APIs, Setup, and Product UX**

TQF is an inference server first. The protocol layer should make existing local-model tools work without teaching users a new ecosystem.

# 69. Local server contract

Default bind behavior follows Ollama’s local model: loopback on port 11434 with no local authentication requirement. Ollama’s current documentation states the local API is available at localhost:11434 and local access requires no authentication. \[R31\] If 11434 is already occupied, TQF should detect an existing TQF instance or move to a clearly reported fallback such as 11435 rather than failing mysteriously.

# 70. OpenAI-compatible surface

From the first usable model milestone, TQF should expose current OpenAI-style compatibility including Responses, Chat Completions, Embeddings, model listing, tools, structured output where supported, multimodal content when vision is enabled, and streaming. OpenAI’s current API reference includes Responses, Chat Completions, Embeddings, and explicit streaming event surfaces; TQF should implement the practical subset required by real clients while remaining strict about what it claims. \[R32\]

| **Method** | **Initial path**     | **Notes**                                                                       |
|------------|----------------------|---------------------------------------------------------------------------------|
| GET        | /v1/models           | Expose canonical model ID and capabilities.                                     |
| POST       | /v1/responses        | Primary modern compatibility path; required for current Codex custom providers. |
| POST       | /v1/chat/completions | Broad ecosystem compatibility.                                                  |
| POST       | /v1/embeddings       | Serve pplx embedding model or explicitly selected embedding capability.         |

# 71. Streaming correctness

SSE/event streaming is not “just flush strings.” TQF must test UTF-8 boundaries, incremental deltas, stop matching across chunk boundaries, client cancellation, backpressure, queue disconnects, and exactly-once event delivery. NVMAI fixed real double-delivery/late-stream bugs in this area; TQF should begin with regression tests that make those classes of bugs impossible to reintroduce.

# 72. Anthropic compatibility

Expose an Anthropic Messages-compatible facade sufficient for Claude Code and gateway-style clients. Claude Code’s gateway documentation explicitly supports redirecting the Anthropic endpoint through \`ANTHROPIC_BASE_URL\`. \`tqf --open claude\` can therefore launch Claude Code with temporary environment/auth values pointed at TQF without permanently modifying the user’s configuration. \[R34\]

# 73. Ollama compatibility

Expose the high-value Ollama endpoints required by local-model applications, such as \`/api/chat\`, \`/api/generate\`, \`/api/tags\`, \`/api/show\`, \`/api/ps\`, and \`/api/embed\`. TQF does not need to reproduce every model-build feature in Ollama; the goal is practical drop-in serving compatibility. \[R31\]

# 74. Non-loopback security

Loopback can remain no-auth for Ollama-like convenience. Binding to \`0.0.0.0\` or another non-loopback address changes the threat model because model generation, retrieval APIs, and potentially repository MCP tools become reachable from the network. TQF should generate/require an API key automatically for non-loopback exposure unless the user explicitly selects an unsafe no-auth override.

# 75. Request queue and cancellation

v1 runs one active generation at a time and queues additional requests. Client disconnect or explicit cancel should stop future token work at a safe command-buffer boundary and release transient resources. GPU work already submitted may not be cancelable immediately, so the state machine must distinguish “request canceled” from “GPU command can be physically aborted.”

# 76. Setup/configuration UX

Configuration lives in machine-global TQF storage with project-local retrieval metadata where relevant, but normal users should not need to edit TOML. First-run decisions are persisted so subsequent starts are zero-question unless a request is impossible. An invalid combination such as \`--memory 4G --context 1M --enable-vision\` must produce an exact memory explanation and offer a sufficient budget interactively; it must never silently change requested semantics or exceed 4G.

# 77. Hardware profile

A machine profile stores measured kernel variants, I/O strategy, SSD concurrency, prefill chunk, cache-policy parameters, and context-kernel choices. It is keyed by hardware/OS/backend fingerprints. Initial setup uses a short benchmark; \`tqf optimize\` can run a deeper matrix, and safe parameters can continue to refine from real workloads.

**PART X**

**TQIndex: First-Party Retrieval, RAG, and TQVec**

A general local retrieval service with code-aware structure—without turning TurboQwenFare into a coding harness or wrapping a generic vector database.

# 78. Retrieval mission

TQIndex is a first-party local index for code repositories, documentation collections, research folders, and other text-like project data. Code receives richer AST/program-graph treatment, but the index is not limited to code. The key product action is deliberately simple: \`tqf sync \<path\>\`. Once registered, the index performs full correctness walks, incremental reuse, and live filesystem updates automatically.


> **Scope guard:** TQIndex retrieves context. It does not edit files, run tests, commit code, or manage an agent loop. Those actions remain the responsibility of the client using TQF.


# 79. Content-first classification

Path names are almost meaningless as content classifiers. A valid software repository can live in any directory. TQF must therefore identify file type from bytes and syntax, using extension/path only as weak priors. A repository under \`~/gaysex/meridian\` is still Rust/WGSL/etc. if the content parses that way; a \`.rs\` file full of PNG bytes is not Rust.

**File classification pipeline**


```text
file

├─ bounded byte sniff: magic / NUL / UTF-8 / controls / entropy

├─ text vs binary/asset

├─ cheap syntax fingerprint → candidate languages

├─ shebang/modeline hints

├─ parse top candidate grammars

├─ parse-quality/confidence score

└─ content kind + language + generated/vendor probabilities
```


# 80. File content taxonomy

| **Axis**              | **Examples**                                                                              | **Behavior**                                                  |
|-----------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Kind                  | Code, Configuration, StructuredData, Documentation, PlainText, Binary, Asset, UnknownText | Determines parser/chunking/retrieval lanes.                   |
| Language              | Rust, C++, Python, TypeScript, Swift, Go, WGSL, …                                         | Determined by content-first classifier.                       |
| Generated probability | autogenerated source, lock files, minified output                                         | Downrank/exclude by default without erasing language.         |
| Vendor probability    | vendored deps, build outputs                                                              | Usually exclude/downrank; override with .tqfignore/config.    |
| Project role          | source, test, docs, config, build, assets                                                 | Derived from structure/content, not parent folder name alone. |

# 81. Parser strategy

Embed a broad Tree-sitter/equivalent grammar set and retain a robust fallback. Candidate language fingerprints prevent trying every grammar on every text file. Parser success—coverage, error-node ratio, structural plausibility—has much more weight than extension. Unknown valid text is never discarded: it still receives lexical/semantic indexing even when no AST is trustworthy.

- Initial broad set should include Rust, C/C++, Python, JS/TS/JSX/TSX, Go, Swift, Java, C#, Lua, Bash, Ruby/PHP where practical, WGSL/GLSL, HTML/CSS, JSON/TOML/YAML, Markdown, CMake and common build/config syntaxes.

- Directory symlinks are followed only when the resolved target stays within the indexed root; cycle detection is mandatory.

- \`.gitignore\` is a strong default. \`.tqfignore\` can extend or explicitly re-include appropriate content.

- Large/generated/vendor blobs should be detected before expensive parsing/embedding.

# 82. Structural chunking

Code is chunked on language structure rather than arbitrary byte windows. Units include modules, types, traits/interfaces, impl blocks, functions/methods, macros, constants, and meaningful nested blocks. Very large functions can be subchunked, but child chunks retain parent symbol/signature context so semantic embedding does not lose their identity.

Documents chunk by headings/sections/paragraphs/code blocks. Configuration chunks by logical sections/objects. The objective is to make each chunk independently useful to a retriever while preserving enough parent metadata to reconstruct context.

# 83. Index evidence lanes

| **Lane**      | **Evidence**                                          | **Typical query**                           |
|---------------|-------------------------------------------------------|---------------------------------------------|
| Exact         | symbol/path/identifier/literal                        | “Where is ExpertCache::evict?”              |
| Lexical       | BM25-like tokens, identifier splits, trigrams, errors | Compiler error or exact phrase.             |
| Structural    | AST, scopes, definitions                              | “Which type implements this trait?”         |
| Program graph | calls, refs, imports, types, tests, configs           | “What calls this and which test covers it?” |
| Semantic      | pplx embeddings                                       | Conceptual/natural-language question.       |
| Hierarchy     | repo→module→file→type→function                        | Architecture navigation.                    |
| Change/Git    | working tree, recent diffs/commits                    | “What changed this behavior?”               |

# 84. Structural authority rule

Exact known structural facts outrank neural similarity. A direct definition match is not allowed to lose because a README paragraph receives a slightly higher cosine score. Evidence types have different semantics; TQIndex must fuse calibrated ranks/confidence instead of pretending all raw scores live on one scale.

# 85. Query router

A lightweight router classifies query intent from tokens and current session metadata. Identifier-like queries should hit exact/lexical/symbol paths without loading the embedder. Natural-language architecture questions can launch semantic + structural/hierarchy lanes in parallel. Stack traces/errors combine lexical exactness with nearby symbol and change-graph evidence. Retrieval should be skipped entirely when it is not useful.

**Hybrid retrieval flow**


```text
query

├─ exact/symbol lane

├─ lexical lane

├─ hierarchy/AST lane

├─ program graph lane

├─ semantic lane (only if useful)

└─ change/Git lane

↓

rank/evidence fusion

↓

local graph expansion

↓

optional GTE reranker

↓

context builder
```


# 86. Embedding model: pplx-embed-v1-0.6b

The chosen embedding model is \`perplexity-ai/pplx-embed-v1-0.6b\`. Its model card specifies 1024-dimensional output, 32K context, Matryoshka representation learning, and native INT8/binary embedding support. It is intended for independent query/document embedding and is MIT-licensed. \[R5\]

TQIndex should preserve the information from the full 1024-dimensional representation while exploiting 256/512-dimensional MRL prefixes for coarse routing where measured. “Use 512 dimensions” therefore means “possibly use the first 512 for a cheap stage,” not permanently throw away the remaining information without evidence.

# 87. TQVec research program

TQVec is the custom compact vector representation. It is not predefined as INT8. The index should retain a flat full-precision/INT8 correctness baseline, then benchmark multiple encodings: native binary, INT8, rotated 4/5/6-bit vectors, binary coarse key + residual refinement, MRL-prefix hierarchies, and repository-adaptive codebooks/rotations. The winning scheme must be measured for recall, reranker/downstream quality, latency, RAM, and update cost.

**Target TQVec storage hierarchy**


```text
HOT RAM

exact structures + partition metadata + binary/MRL routing keys

│

▼

COLD MMAP

compact full TQVec + adjacency + partitions

│

▼

SSD / rare refinement

optional richer payloads
```


# 88. Repository-adaptive vector packing

TQIndex may derive local vector quantization parameters from a repository’s own embedding distribution: rotations, scales, compact codebooks, partition centroids. This is index compression, not neural model training. The benefit hypothesis is that a codebase’s embedding distribution is narrower/more structured than a universal web corpus, allowing better recall per bit. The cost is profile complexity and update stability; benchmark it rather than assuming it.

# 89. ANN development path

Start with SIMD flat search because it is the gold correctness baseline and can be surprisingly competitive for normal repository sizes. Then implement the intended TQF-native system: repository-hierarchical adaptive partitions overlaid with semantic partitions. Compare against HNSW and DiskANN-style baselines. A custom ANN feature survives only if it improves a defined Pareto frontier of recall/latency/RAM/index size/update cost.

# 90. Adaptive partition architecture

The advanced index should maintain two overlapping topologies. One is program/document hierarchy: repository, module, file, type, function, section. The second is semantic partition hierarchy. The query router may enter from either side. Strong seeds can then expand through graph neighbors. Partition split/merge and hot-RAM placement adapt to update/query workload, inspired by Quake and SPFresh principles without copying their generic assumptions. \[R25\]\[R26\]

# 91. Incremental sync and live repair

**Incremental synchronization**


```text
startup / tqf sync path

full filesystem correctness walk

│

├─ unchanged hash → reuse parse/vector/index records

├─ changed file → reparse → diff chunks/symbols

├─ new chunk → embed + insert

└─ deleted chunk → remove

│

local graph repair

local partition repair

│

live filesystem watcher
```


“Full on startup” means a complete correctness scan of the tree and index relationship, not re-embedding every unchanged chunk. Exact/lexical/structural retrieval should become usable before semantic embedding catches up; semantic work can finish progressively in the background and pause during latency-sensitive decode.

# 92. Git/change awareness

Git is optional metadata, not a prerequisite. When present, TQIndex may index the working-tree delta and lightweight recent history/diffs so queries about regressions or recent architectural changes can use actual change evidence. It should not embed the entire Git history by default. Unsaved editor overlays may later be supplied by clients/MCP as temporary RAM-only versions of files.

# 93. Reranker: gte-reranker-modernbert-base

The chosen reranker is \`Alibaba-NLP/gte-reranker-modernbert-base\`, a 149M text reranker with maximum input length 8192 and Apache-2.0 licensing. Its model card reports CoIR average 79.99 for the reranker. \[R6\] It should be loaded transiently only when candidate ambiguity justifies the memory/latency cost.

An exact symbol lookup does not load a 149M reranker. Broad semantic candidates with close fused scores might. During reranking the memory broker shrinks Qwen’s elastic expert cache, loads the reranker, scores a small candidate set, unloads it, and returns memory to Qwen before decode.

# 94. Automatic RAG and context budget

Automatic RAG is enabled when a selected index is active, but retrieval is not mandatory for every prompt. The system estimates whether external local context is useful and chooses a dynamic injection budget. A simple symbol question may need a few hundred or thousand tokens; a cross-module architecture question may need a much larger set plus graph expansion. Qwen may also invoke \`tqf_search\` during a tool-capable workflow if pre-generation retrieval was insufficient.

The GUI should show a subtle indicator such as “Used 7 project references,” expandable to evidence details. \`--no-rag\` disables automatic injection without disabling direct index APIs or MCP.

# 95. Retrieval APIs and MCP

**Initial retrieval surfaces**


```text
POST /api/index/search

POST /api/index/sync

GET /api/index/status

GET /api/index/list

MCP tools:

tqf_search

tqf_symbol

tqf_references

tqf_callers

tqf_tests

tqf_file

tqf_repo_map
```


Support both stdio and streamable HTTP MCP so \`--open\` can use whichever integration path the client supports best. Retrieval tools remain read-only initially. File edits/execution belong to the client harness.

**PART XI**

**SwiftUI GUI, Integrations, Security, and Operations**

A polished native face on macOS, zero-friction client launchers, and operational behavior that stays safe and predictable.

# 96. macOS SwiftUI architecture

The macOS GUI should directly adapt selected Apache-2.0 TurboFieldfare/NVMAI SwiftUI source rather than visually imitating it. SwiftUI source is compiled at build time and linked into the same \`tqf\` Mach-O executable. Rust remains the inference/server owner. The Swift layer is intentionally thin and consumes localhost/control endpoints for conversation, setup, metrics, index status, and configuration.

This satisfies the one-binary rule while preserving the native interaction quality that motivated the UI choice. Adapted files must retain Apache attribution/modification notices where required.

# 97. Brook’s™ UI direction

Keep the strongest source characteristics: conversation-first layout, restrained materials, native typography, polished install/load states, bottom composer, status HUD, and optional inspector. Improve information hierarchy, responsiveness, and identity. The normal view should remain simple; a single advanced inspector exposes the full systems-engineering cockpit for development and power users.

| **Inspector group** | **Live data**                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------------|
| Performance         | tok/s, TTFT, prefill tok/s, p50/p95 token latency, GPU time, SSD throughput                     |
| Memory              | budget, core, experts, context, scratch, helper model, vision                                   |
| Experts             | byte hit rate, selection hits, misses, predictor precision, prefetch timeliness                 |
| Context             | logical tokens, TQKV effective bits, recent window, TQAttn pages, protected pages, prefix reuse |
| Retrieval           | files, symbols, sync progress, search lanes, graph expansion, reranker use                      |
| Server              | port, protocols, queue depth, active client/integration                                         |

# 98. GUI process/event loop bridge

The single binary must coordinate Tokio/server threads with the macOS application main thread correctly. One practical approach is for Rust startup to initialize configuration/runtime/server worker threads, then transfer the main thread to a C-callable Swift entrypoint that creates \`NSApplication\`/SwiftUI hosting. Headless mode skips the Swift entrypoint entirely. No Swift inference runtime is duplicated.

# 99. \`--open\` integrations

\`tqf --open \<client\>\` is convenience glue: ensure the server is running, synchronize the associated index if one is registered, construct ephemeral provider/MCP configuration, launch the client as a child, and remove the temporary environment/config when it exits. The client’s permanent configuration remains untouched.

| **Client**  | **Compatibility basis**                                                              | **TQF strategy**                                                      |
|-------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| OpenCode    | Supports custom OpenAI-compatible providers with explicit baseURL/model map. \[R33\] | Generate ephemeral provider/MCP config and launch child.              |
| Claude Code | Supports gateway redirection via ANTHROPIC_BASE_URL. \[R34\]                         | Set temporary gateway/auth env + MCP config.                          |
| Codex       | Current provider registry supports custom base_url and Responses wire API. \[R35\]   | Generate ephemeral provider config targeting \`/v1/responses\` + MCP. |

# 100. Missing-client installation

If a requested client binary is absent, ask permission. TQF may use a known official installation recipe or a detected package manager, but it must not silently install external software. The goal is Ollama-easy, not magical system mutation.

# 101. Privacy defaults

- No telemetry/analytics by default.

- No prompt/repository upload.

- Network access is required only for explicit model/helper downloads or user-configured external behavior.

- Project indexes and prefix/context caches remain local.

- Logs must avoid dumping complete prompts/source by default; diagnostics can record hashes/IDs and timing.

# 102. Failure handling

GPU command-buffer failure, invalid model layout, failed expert read, corrupted index extent, or context-page checksum error must fail loudly and locally. Never continue producing text from known-corrupt state. Recovery paths should reset only the affected request/session when safe; model/global corruption may require unload/revalidation. Configuration and index metadata writes are atomic.

# 103. Disk budget and cleanup

Machine-global auxiliary data should remain roughly within the Q4 model plus about 5 GB of support/cache data, while project indexes are separately compact/bounded. Under pressure, reclaim old logs, stale prefix snapshots, rebuildable acceleration data, and optional high-fidelity cold KV backing before touching canonical model or current correctness state. Helper models count as support data and should be repacked compactly.

**PART XII**

**Correctness, Quality, Performance, Research Methodology, and Risks**

How to prove that black magic is real, detect when it lies, and keep the project from optimizing itself into a benchmark fraud.

# 104. Reference implementation strategy

Every optimized subsystem needs a simpler reference path. During early development, a large-memory correctness mode may keep more weights resident and use straightforward BF16/FP32 intermediates. That path is not the product; it is the oracle that allows TQF to validate Q4 repacking, layer outputs, routing, GDN recurrence, KV cache behavior, MTP, and long-context approximations.

- Tokenizer golden vectors against official/reference tooling.

- Tensor-by-tensor converted Q4 value checks.

- Layerwise intermediate checks for GDN and full attention.

- Router top-K IDs and weights.

- GDN recurrent-state snapshots.

- Greedy output parity for exact paths.

- Sampling statistical sanity tests for non-greedy paths.

# 105. Quality qualification suite

| **Area**          | **Required signals**                                   | **Failure rule**                                                  |
|-------------------|--------------------------------------------------------|-------------------------------------------------------------------|
| Coding            | repo tasks, generation correctness, tool arguments     | \>1% aggregate relative degradation rejects candidate.            |
| Long context      | needle/passkey, document/code retrieval, synthesis     | Critical retrieval regressions can reject even if aggregate \<1%. |
| Reasoning         | math/science/general reasoning                         | Track relative score and output drift.                            |
| Tool use          | tool choice + arguments + structured output            | Aim for zero measurable regression.                               |
| Perplexity/logits | reference distributions/intermediate errors            | Used to diagnose silent degradation.                              |
| Retrieval         | Recall@k/NDCG/downstream solve rate                    | Separate from model-quality budget; retrieval must show net gain. |
| Vision            | multimodal benchmarks and text regression when enabled | Qualify independently.                                            |

# 106. Performance benchmark protocol

Every published number must state hardware, OS, TQF commit, model source hash, \`.tqf\` format/profile, memory budget, logical context configured and populated, prompt/generation lengths, temperature/sampling, filesystem-cache condition where relevant, and whether the run was first/cold or warmed. “Peak 40 tok/s” without workload/context/cache state is not useful.

| **Metric**                    | **Why it exists**                                           |
|-------------------------------|-------------------------------------------------------------|
| Decode tok/s                  | Primary sustained output speed.                             |
| Prefill tok/s / TTFT          | Long-context usability.                                     |
| p50/p95 token latency         | Stall/jitter visibility.                                    |
| Peak working set              | Memory-contract compliance.                                 |
| SSD bytes/accepted token      | Core MoE streaming efficiency metric.                       |
| Expert byte hit rate          | More meaningful than selection hit if tiles differ in size. |
| Prefetch precision/timeliness | Separates accurate-but-late predictor from useful prefetch. |
| GPU stage times               | Find actual kernel bottleneck.                              |
| I/O wait and overlap          | Find hidden critical-path stalls.                           |
| Quality delta                 | Prevents speed wins that damage behavior.                   |

# 107. Workload set

- Real coding generation: Rust and other languages; 512+ output tokens.

- Repository QA with TQIndex disabled and enabled.

- Refactor/planning/tool-call prompts with realistic non-repetitive output.

- General prose and reasoning.

- Highly repetitive synthetic prompt only as a locality-envelope diagnostic, never headline.

- Context sweeps at 4K, 8K, 32K, 128K, 256K, and ~1M with context already populated.

- 2G/4G/8G memory profiles separately.

# 108. Retrieval benchmark protocol

TQIndex must be compared against flat semantic search, lexical BM25-like search, BM25+embeddings, HNSW, DiskANN-style baselines, and embedding+reranker baselines. Use CoIR and RepoBench where applicable, plus real repository tests that ask for definitions, callers, tests, configs, architecture, regressions, and cross-file dependencies. \[R27\]\[R28\]

Every novel index feature receives an ablation: structure, exact identifiers, graph expansion, Git, adaptive routing, TQVec encoding, semantic partitions, reranker, and workload adaptation. A sophisticated feature that does not improve the Pareto frontier is removed.

# 109. Optimization ledger

**Required experiment record**


```text
Experiment ID

Hypothesis

Baseline commit/profile

Hardware + thermal state

Workload + context

Memory budget

Implementation change

Decode / TTFT / stage times

SSD bytes/token

Quality delta

Statistical confidence / repeats

Verdict: keep / reject / investigate

Notes / known interactions
```


The ledger is also where negative results from NVMAI/TurboFieldfare are imported. This prevents future contributors from spending weeks rediscovering that a plausible kernel staging scheme or CPU MTP path already failed under a comparable bandwidth regime.

# 110. Major technical risks

| **Risk**                                   | **Why it matters**                                                   | **Mitigation / kill criterion**                                                                                                                 |
|--------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| M4 SSD bandwidth/locality insufficient     | 15+ coding tok/s may remain I/O-bound.                               | Global/partial cache, predictor, co-routing layout, Q4 packing. If 4G cannot hit floor after evidence-backed techniques, report limit honestly. |
| 128K/1M attention bandwidth                | Capacity can fit while per-token scan is too slow.                   | TQKV + TQAttn + self-indexed pages; strict ≤1% qualification.                                                                                   |
| Q4 kernel bottleneck                       | Compute/memory bandwidth may dominate even with perfect expert hits. | Model-specific MSL, fused stages, autotune; stage accounting.                                                                                   |
| Helper-model memory pressure               | RAG can starve expert cache.                                         | Transient load via broker; no concurrent residency unless value proven.                                                                         |
| Custom ANN complexity                      | Research could become a side project with weak payoff.               | Flat baseline + incremental milestones; every feature needs measurable win.                                                                     |
| One-crate size                             | Module coupling could become messy.                                  | Strict dependency firewall and internal ownership APIs; no need for package boundaries.                                                         |
| SwiftUI/Rust integration                   | Main-thread/lifecycle complexity.                                    | Keep Swift thin, server-backed, no Swift inference ownership.                                                                                   |
| Quality drift from multiple approximations | Each “tiny” error can compound.                                      | Combined qualification suite; ≤1% applies to final stack, not per feature.                                                                      |

# 111. Definition of a valid optimization

9.  It has a written hypothesis and baseline.

10. It passes correctness tests appropriate to whether the transformation is exact or approximate.

11. It improves at least one meaningful end-to-end metric on a representative workload.

12. It does not cause \>1% measured quality degradation in the qualified stack.

13. It does not secretly violate the requested memory budget or shift costs to unreported page cache/VRAM.

14. It has no unacceptable worst-case regression or includes a controller that disables it when harmful.

15. It is recorded in the optimization ledger so future work can reproduce or reject it.

**PART XIII**

**Detailed Implementation Roadmap**

A dependency-aware build sequence with concrete outputs and exit gates. Correctness first; optimized black magic follows only when there is a reference to beat.

# 112. Phase map

| **\#** | **Phase**                          | **Primary output**                                                                                                                                                    | **Exit gate**                                                                      |
|--------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 0      | Research harvest and source ledger | Freeze exact Qwen source; clone/mine TurboFieldfare+NVMAI; record reusable Apache files, optimization commits, failed experiments; create tensor inventory generator. | A versioned research ledger and exact source/checksum manifest exist.              |
| 1      | Single-crate skeleton              | Create Cargo project, module tree, error/log/config infrastructure, feature gating for macOS/Linux, src/build.rs.                                                     | \`cargo build --release\`; \`tqf --help\`; no workspace.                           |
| 2      | Server skeleton                    | Tokio/Axum control plane; \`/health\`, \`/v1/models\`, stubs for Responses/Chat; cancellation/event interfaces.                                                       | Server starts on loopback and protocol tests pass.                                 |
| 3      | Setup/global data                  | \`~/.tqf\`, config/receipts/profile state machine, hardware detection, interactive/noninteractive setup.                                                              | Missing-model path prompts correctly and exits/continues transactionally.          |
| 4      | Source resolver/downloader         | Pinned HF source, resumable range requests, local import, ownership tracking.                                                                                         | Interrupted transfer resumes; checksums verified.                                  |
| 5      | GGUF import reader                 | Read only Qwen-relevant GGUF metadata/tensors/quant blocks/tokenizer; strict bounds checks/fuzzing.                                                                   | Can enumerate canonical checkpoint exactly.                                        |
| 6      | \`.tqf\` v1 container              | Superblock, extent table, expert tiles, versioning, checksums, trusted receipt.                                                                                       | Round-trip arbitrary canonical tensors with no inference.                          |
| 7      | Lossless Q4 repack                 | Source-compatible then TQF-native physical block candidates; validation decoder.                                                                                      | Every represented Q4 value matches source.                                         |
| 8      | Streaming resumable conversion     | Direct range→target extents, journal, atomic finalize, source cleanup policy.                                                                                         | No full duplicate source needed for canonical install where remote layout permits. |
| 9      | Tokenizer/chat template            | Qwen tokenizer and message/tool/thinking formatting.                                                                                                                  | Golden token streams match reference.                                              |
| 10     | Metal baseline infrastructure      | Device, queues, shared buffers, pipeline cache, baseline metallib/MSL compile, timings.                                                                               | Microbench framework runs on base M4.                                              |
| 11     | Reference Q4 kernels               | Q4 GEMV/GEMM, RMSNorm, elementwise, sampling/head reference implementations.                                                                                          | Tensor unit tests vs CPU/reference.                                                |
| 12     | GDN implementation                 | Conv state, projections, recurrent update, gated norm, output projection; float32 state correctness.                                                                  | Per-layer intermediates and recurrent state match reference tolerances.            |
| 13     | Full attention implementation      | QKV, partial RoPE, GQA, gated attention output, BF16 reference KV.                                                                                                    | Ten full-attention layers validated.                                               |
| 14     | MoE implementation                 | Router top8, shared expert, routed gate/up/down, accumulation. Start resident for correctness.                                                                        | Full layer parity; router IDs/weights match.                                       |
| 15     | End-to-end correct decode          | Embedding, 40-layer loop, LM head, sampling; deterministic sequence tests.                                                                                            | 512-token greedy reference qualification passes.                                   |
| 16     | Live OpenAI server                 | Real Chat/Responses streaming and tool call parsing from correct runtime.                                                                                             | OpenAI-compatible client produces valid streamed output.                           |
| 17     | Canonical autoinstall              | Plain \`tqf\` performs download/repack/tune/start.                                                                                                                    | Fresh-machine smoke test succeeds without manual model knowledge.                  |
| 18     | Out-of-core experts                | Resident core + Q4 routed expert streaming, simple LFU baseline.                                                                                                      | Correct output while expert pool stays on SSD.                                     |
| 19     | Parallel I/O/read-ahead            | Parallel \`pread\`, F_RDADVISE/Metal I/O candidates, bounded slots, instrumentation.                                                                                  | M4 I/O strategy selected by measured end-to-end result.                            |
| 20     | Metal performance ports            | Adapt NVMAI MoE phase-1 and GDN fusion; function-constant specialization; Q4 packing A/B.                                                                             | Meaningful stage/end-to-end wins with parity.                                      |
| 21     | Global expert broker               | Replace fixed per-layer cache with byte-budgeted global admission; online stats.                                                                                      | 4G plan significantly improves bytes/token vs simple baseline.                     |
| 22     | Tiled/partial experts              | Enable matrix tiles, mixed tile widths, optional bundles/co-routing.                                                                                                  | A/B decides default; format requires no migration.                                 |
| 23     | Predictive prefetch                | Statistical predictor then optional hidden predictor; adaptive aggressiveness.                                                                                        | Net SSD-stall reduction without harmful overfetch.                                 |
| 24     | Hard 4G memory broker              | All large allocations registered; OS footprint accounting; helper-model leases.                                                                                       | Adversarial workloads remain within qualified 4G bound.                            |
| 25     | Short-context M4 assault           | Iterate scheduler/kernel/cache/I/O until real workloads hit and exceed 15 tok/s.                                                                                      | 15 is floor; retain ledger of further headroom.                                    |
| 26     | Prefill optimization               | 4096-ish chunk A/B, expert reuse across rows, prefix setup.                                                                                                           | Long prompts achieve major TTFT reduction.                                         |
| 27     | TQKV baseline                      | Paged Q8/Q4 KV, fused compressed attention consumption.                                                                                                               | 128K capacity under 4G with full logical attention reference.                      |
| 28     | Advanced TQKV                      | Q3/Q2 cold pages, rotations, outliers, pre-RoPE keys, mixed precision.                                                                                                | ≤1% quality at 128K; memory improves materially.                                   |
| 29     | 128K production gate               | Run full speed/quality/memory suite on base M4.                                                                                                                       | 4G, 128K, ≤1%, ≥15 tok/s populated-context floor.                                  |
| 30     | Prefix snapshot store              | Dedup TQKV page references + GDN state; persistent bounded LRU.                                                                                                       | Repeated-prefix TTFT reduction; restart reuse.                                     |
| 31     | 256K and TQAttn trigger            | Scale TQKV; test full attention; introduce selective page attention if floor fails.                                                                                   | 256K usable within 4G and ≤1%.                                                     |
| 32     | TQAttn/self-index keys             | Page signatures, protected/recent inclusion, query-aware selection, SSD richer backing.                                                                               | Long-context page budget produces measured speedup within quality limit.           |
| 33     | MTP                                | Implement MTP, accepted-token accounting, expert-union bandwidth metrics.                                                                                             | Controller enables only when beneficial.                                           |
| 34     | 2G experimental                    | Aggressive broker/TQKV/cache path; helper-model swapping.                                                                                                             | Correct 128K ≤2G; then attack speed.                                               |
| 35     | File catalog/classifier            | Full scan, symlink/ignore, byte sniff, content-first language classification.                                                                                         | Misleading extensions/paths test suite passes.                                     |
| 36     | Structural/lexical index           | AST chunks, symbols, program graph, hierarchy, BM25-ish/exact indexes.                                                                                                | Useful search without semantic model.                                              |
| 37     | pplx helper runtime                | Repack/load pplx-embed transiently via broker; batching.                                                                                                              | Embedding requests under 4G/2G contracts.                                          |
| 38     | Flat semantic baseline             | Full 1024-d representations + MRL prefixes; exact SIMD search.                                                                                                        | Gold semantic recall/latency baseline recorded.                                    |
| 39     | TQVec research                     | Binary/INT8/rotated 4–6bit/residual/repo-adaptive variants.                                                                                                           | Choose encoding only from benchmark Pareto frontier.                               |
| 40     | Hybrid query/fusion                | Intent router, multi-lane candidates, calibrated fusion, graph expansion.                                                                                             | Beats lexical/semantic-only baselines.                                             |
| 41     | Adaptive ANN partitions            | Repo hierarchy + semantic partitions; local split/merge/update.                                                                                                       | Compare to flat/HNSW/DiskANN-style; keep only if win.                              |
| 42     | Live incremental sync              | Full correctness scan + reuse, watcher, local repairs, background priority.                                                                                           | Edits become searchable quickly without rebuild.                                   |
| 43     | GTE reranker                       | Transient cross-encoder on ambiguous candidates.                                                                                                                      | Net retrieval/downstream win at acceptable TTFT.                                   |
| 44     | Automatic RAG/MCP                  | Dynamic retrieval budget, model-search tool, index APIs, stdio+HTTP MCP.                                                                                              | General apps and coding clients can consume TQIndex.                               |
| 45     | OpenCode/Claude/Codex launchers    | Ephemeral provider/MCP config, install prompts, child lifecycle.                                                                                                      | \`tqf --open ...\` works without permanent config mutation.                        |
| 46     | SwiftUI import/bridge              | Adapt Apache UI, link Swift into same binary, server-backed view model.                                                                                               | Plain \`tqf\` opens native UI; \`--headless\` does not.                            |
| 47     | Brook’s UI pass                    | Inspector, setup polish, index/context/perf observability, conversation UX.                                                                                           | Simple default, deep optional cockpit.                                             |
| 48     | Vision                             | Lazy projector/vision artifact, multimodal protocol mapping, memory broker behavior.                                                                                  | Image request works; text-only memory unchanged.                                   |
| 49     | ~1M context research               | TQKV/TQAttn/SSD backing/promotion/hierarchical search and new techniques.                                                                                             | ≤1% and ≥15 tok/s floor on target 8G profile.                                      |
| 50     | CUDA format/backend                | CUDA-specific \`.tqf\`, Q4/GDN/attention/MoE/TQKV kernels, pinned staging/streams.                                                                                    | Reference parity on NVIDIA.                                                        |
| 51     | RTX 3070 Ti gate                   | Optimize/validate Linux 3070 Ti, 6GB-class support envelope.                                                                                                          | Mandatory reference works and meets certified floors.                              |
| 52     | Release hardening                  | Fuzz parsers, crash/recovery tests, API conformance, license bundle, installer/resume torture.                                                                        | Release candidate survives fault injection and complete qualification suite.       |

# 113. Critical path to first useful build

The first usable milestone does not require TQKV, TQIndex, SwiftUI, or CUDA. It requires one binary that can install/repack the canonical model, run correct Qwen3.6 Q4 text decode on M4, and expose a streaming OpenAI-compatible server. The next critical path is SSD expert streaming + global 4G broker + M4 performance. Long context and retrieval then build on a correct fast core instead of masking basic inference bugs.

**Recommended high-level sequencing**


```text
Correct model → server → out-of-core experts → 4G broker → 15+ tok/s real decode

→ TQKV 128K → prefix reuse → 256K/TQAttn → TQIndex/RAG → 1M → CUDA
```


# 114. Contributor “do not do this” list

- Do not add generic Llama support “while we are here.”

- Do not introduce an external vector database because custom TQIndex is difficult.

- Do not split the repository into a workspace of internal crates.

- Do not add a user-facing quality mode; policy is automatic and quality must remain within the global limit.

- Do not claim 15 tok/s from a repetitive counting workload as the product result.

- Do not merge an approximate context optimization without the combined ≤1% qualification.

- Do not measure only GPU kernel time and call it decode time.

- Do not let retrieval or GUI become dependencies of the inference core.

- Do not silently allocate above \`--memory\` or rely on unreported OS page cache to make memory claims.

- Do not optimize a microbenchmark without measuring the end-to-end pipeline afterward.



**PART XIV**

**Normative Implementation Contracts**

This part exists so an implementation agent can build the first correct system without inventing hidden architecture. The earlier parts explain the design; this part defines the concrete reference implementation, wire layouts, state machines, APIs, ownership rules, and fallback behavior. Unless a subsection explicitly says **RESEARCH CANDIDATE** or **BENCHMARK-SELECTED**, the baseline below is normative.

# 115. Core implementation invariants

**LOCKED.** These invariants apply across the crate.

1. The hot inference path is model-specific to Qwen3.6-35B-A3B. Generic-model abstractions must not appear inside performance-critical loops unless they compile away completely.
2. All persisted integers in TQF-owned binary formats are little-endian. Readers must reject unsupported endianness rather than guessing.
3. All byte offsets and lengths that address files are `u64` on disk and in validation code. Conversions to `usize` occur only after checked bounds validation against the mapped/read region.
4. Every large allocation is registered with the memory broker before physical allocation. “Allocate, then report” is prohibited.
5. Every asynchronous I/O operation owns or borrows a destination lease whose lifetime is guaranteed through completion. A cache tile cannot be evicted while a read into it is pending.
6. A GPU command may only reference buffers whose broker leases outlive the command-completion event.
7. Exact expert routing always comes from Qwen's real router. Predictors may schedule bytes, never alter selected expert IDs or weights.
8. Any approximate context/retrieval optimization has a correctness fallback path and a qualification test.
9. Persistent writes use temp/journal + fsync/atomic-rename semantics where corruption would require expensive recomputation or could break correctness.
10. Every performance optimization must be disableable by a developer/debug control for A/B testing, but ordinary users do not see a quality/performance-mode maze.

# 116. Foundational Rust types

**REFERENCE BASELINE.** Use strongly typed newtypes for identifiers and byte counts so layer/expert/page IDs are not accidentally mixed.

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LayerId(pub u8);        // 0..39

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ExpertId(pub u16);      // 0..255

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TileId(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ContextPageId(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Bytes(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Tokens(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    GatedDeltaNet,
    FullAttention,
}
```

The canonical layer-kind table is compiled from the official 3-linear/1-full pattern and also verified against the installed model manifest. A mismatch is a fatal architecture error.

# 117. Canonical Qwen3.6 geometry

**LOCKED from the pinned official checkpoint.** The importer must validate these fields before conversion. [R2][R3]

| Parameter | Value |
|---|---:|
| Vocabulary | 248,320 |
| Hidden size `D` | 2,048 |
| Layers | 40 |
| Full-attention interval | every 4th layer |
| GDN layers | 30 |
| Full-attention layers | 10 |
| Full attention heads | 16 |
| Full KV heads | 2 |
| Full head dimension | 256 |
| Rotary fraction | 0.25 |
| Rotary sub-dimension | 64 |
| GDN key heads | 16 |
| GDN value heads | 32 |
| GDN key head dimension | 128 |
| GDN value head dimension | 128 |
| GDN key dimension | 2,048 |
| GDN value dimension | 4,096 |
| GDN convolution channels | 8,192 |
| GDN convolution width | 4 |
| Experts/layer | 256 |
| Routed experts/token/layer | 8 |
| Routed expert width | 512 |
| Shared expert width | 512 |
| MTP hidden layers | 1 |
| Native context | 262,144 |
| GDN recurrent-state dtype | FP32 |

The full-attention projection shapes are therefore, following the official configuration and Transformers reference implementation: [R2][R3]

```text
q_proj: [8192, 2048]   // query + output gate, 16 * 256 * 2
k_proj: [ 512, 2048]   // 2 * 256
v_proj: [ 512, 2048]
o_proj: [2048, 4096]   // 16 * 256 -> 2048
q_norm: [256]
k_norm: [256]
```

The Gated DeltaNet projection shapes are: [R2][R3]

```text
in_proj_qkv: [8192, 2048]  // key 2048 + query 2048 + value 4096
in_proj_z:   [4096, 2048]
in_proj_a:   [  32, 2048]
in_proj_b:   [  32, 2048]
conv1d:      [8192, 1, 4]   // depthwise
A_log:       [32]
dt_bias:     [32]
gated_norm:  [128]
out_proj:    [2048, 4096]
```

Each routed expert logically contains:

```text
gate: [512, 2048]
up:   [512, 2048]
down: [2048, 512]
```

and the source Transformers representation fuses `gate` and `up` into `[1024, 2048]` per expert. TQF may preserve or rearrange that fusion depending on the backend layout.

# 118. Canonical tensor-inventory generator

**REFERENCE BASELINE.** Do not maintain the production tensor inventory entirely by hand. Phase 0 must produce a generator that reads the canonical source metadata and emits a checked artifact such as:

```text
dev/generated/qwen36_tensor_inventory.json
```

Each row contains:

```rust
pub struct TensorInventoryEntry {
    pub canonical_name: String,
    pub logical_role: TensorRole,
    pub layer: Option<LayerId>,
    pub shape: Vec<u64>,
    pub source_quant: SourceQuant,
    pub source_bytes: u64,
    pub tqf_section: TqfSectionKind,
    pub residency: ResidencyClass,
    pub consumer: KernelConsumer,
    pub required_alignment: u32,
}
```

The generator must fail if the source checkpoint contains a production-language tensor that cannot be classified. Unknown vision tensors may be deferred only when `--enable-vision` is not part of the conversion; unknown text tensors are fatal.

The committed inventory is a reviewable golden record. CI regenerates it against the pinned source metadata and fails on drift.

# 119. Error taxonomy

**REFERENCE BASELINE.** Errors must preserve enough structure to distinguish user configuration, corrupted storage, incompatible models, transient I/O, resource exhaustion, GPU failure, and internal bugs.

```rust
pub enum TqfError {
    Config(ConfigError),
    Setup(SetupError),
    Model(ModelError),
    Format(FormatError),
    Memory(MemoryError),
    Io(IoError),
    Backend(BackendError),
    Context(ContextError),
    Retrieval(RetrievalError),
    Protocol(ProtocolError),
    Cancelled,
    Internal(InternalError),
}
```

Rules:

- User-caused errors return concise actionable CLI/API messages without backtraces by default.
- `Internal` indicates violated TQF invariants and should include a diagnostic incident ID in logs.
- GPU command failure is never translated into a normal model stop reason.
- A corrupted `.tqf` or index never triggers unsafe reads; readers bounds-check before mapping/dispatch.
- Memory budget failure includes `requested`, `available`, `owner`, and the minimum viable suggested configuration when known.

# 120. `.tqf` wire-format principles

**LOCKED.** `.tqf` is manually serialized, little-endian, and does not depend on Rust `repr(C)` memory layout for persistence. Rust structs may mirror wire records for convenience, but encode/decode functions write individual fields explicitly.

**REFERENCE BASELINE:**

- File alignment quantum: 4096 bytes.
- Superblock length: 4096 bytes exactly.
- Section starts: 4096-byte aligned.
- Whole routed-expert records: 4096-byte aligned.
- Tiles within an expert: at least 64-byte aligned; backend conversion may choose 128/256-byte alignment.
- Metadata tables: 64-byte record alignment where practical.
- Provenance hash: SHA-256.
- Internal fast integrity hash: BLAKE3-256 or an equivalently fast cryptographic hash selected at dependency review.
- No transparent compression around quantized weight payloads in v1; compressed data must remain directly addressable for I/O.

# 121. `.tqf` superblock v1

**REFERENCE BASELINE.** The first 4096 bytes use the following layout. Reserved bytes must be zero when written and ignored-but-preserved only by explicitly compatible migration tools.

| Offset | Size | Field |
|---:|---:|---|
| `0x000` | 8 | ASCII magic `TQFMODEL` |
| `0x008` | 2 | format major |
| `0x00A` | 2 | format minor |
| `0x00C` | 4 | superblock bytes, must be 4096 |
| `0x010` | 4 | endian marker `0x01020304` |
| `0x014` | 4 | backend/layout ID |
| `0x018` | 8 | feature bits |
| `0x020` | 16 | model-family UUID/ID |
| `0x030` | 32 | canonical source SHA-256 root/fingerprint |
| `0x050` | 32 | conversion-build fingerprint |
| `0x070` | 8 | file length |
| `0x078` | 8 | section table offset |
| `0x080` | 4 | section count |
| `0x084` | 4 | section-record bytes |
| `0x088` | 8 | extent table offset |
| `0x090` | 8 | extent count |
| `0x098` | 8 | string table offset |
| `0x0A0` | 8 | string table bytes |
| `0x0A8` | 8 | architecture-record offset |
| `0x0B0` | 8 | tokenizer-record offset |
| `0x0B8` | 8 | expert-index offset |
| `0x0C0` | 8 | expert-index bytes |
| `0x0C8` | 8 | checksum-table offset |
| `0x0D0` | 8 | checksum-table bytes |
| `0x0D8` | 32 | metadata-root BLAKE3 |
| `0x0F8` | 8 | creation Unix seconds |
| `0x100` | 8 | minimum reader capability bits |
| `0x108` | 3832 | reserved, zero |

Reader validation order:

1. Read only 4096 bytes.
2. Validate magic, superblock size, endian marker and supported major version.
3. Validate declared file length against `fstat`.
4. Checked-add/checked-multiply every table range; require it to fit inside file length.
5. Read metadata tables.
6. Validate metadata root hash.
7. Validate architecture fingerprint before any kernel is constructed.
8. Only then expose tensor/expert extents to runtime code.

# 122. Section records

Each section-table record is 64 bytes.

```text
u32 kind
u32 flags
u64 file_offset
u64 stored_bytes
u64 logical_bytes
u32 required_alignment
u32 checksum_index
u64 element_count_or_zero
u64 aux_offset_or_zero
u64 aux_count_or_zero
```

Initial section kinds:

```rust
pub enum TqfSectionKind {
    Architecture = 1,
    Tokenizer = 2,
    StringTable = 3,
    ResidentCore = 4,
    Embeddings = 5,
    LmHead = 6,
    RoutedExperts = 7,
    Mtp = 8,
    VisionLink = 9,
    DuplicateLayouts = 10,
    Extents = 11,
    ExpertIndex = 12,
    Checksums = 13,
    Provenance = 14,
}
```

Unknown optional section kinds may be ignored only when no required-capability bit references them. Unknown required sections are fatal.

# 123. Tensor extent record

**REFERENCE BASELINE:** 96 bytes. Tensor names live in the string table; no repeated variable-length names in hot metadata.

```text
u32 role_id
u32 flags
u64 name_string_offset
u64 file_offset
u64 stored_bytes
u64 logical_elements
u32 rank
u32 quant_layout_id
u32 dtype_id
u32 required_alignment
u64 dim0
u64 dim1
u64 dim2
u64 dim3
u32 checksum_index
u32 reserved
```

Ranks above four are not required for production text weights in v1. Importer-only metadata may use a separate generic shape representation; hot runtime metadata is intentionally fixed and compact.

# 124. Expert superextent and tile records

Each routed expert receives one 4096-aligned **superextent** so a whole-expert miss is one contiguous range whenever the chosen backend layout permits it. Tiles are subranges inside the superextent.

```rust
pub struct ExpertIndexRecord {
    pub layer: LayerId,
    pub expert: ExpertId,
    pub flags: u16,
    pub file_offset: u64,
    pub stored_bytes: u32,
    pub tile_first: u32,
    pub tile_count: u16,
    pub layout_id: u16,
    pub checksum_index: u32,
}

pub struct ExpertTileRecord {
    pub matrix: ExpertMatrix,      // GateUp or Down, later separate Gate/Up allowed
    pub tile_id: TileId,
    pub neuron_start: u16,
    pub neuron_count: u16,
    pub relative_offset: u32,
    pub stored_bytes: u32,
    pub quant_layout_id: u16,
    pub flags: u16,
}
```

**REFERENCE BASELINE tile layout:** expose 128-neuron intermediate tiles in metadata while allowing the first runtime to cache/fetch the entire expert. This avoids a format migration when partial caching is implemented. Gate+up tile `i` contains rows `[i*128 .. (i+1)*128)` from both projections; down tile `i` contains the matching input columns required for those intermediate neurons.

**BENCHMARK-SELECTED:** 64/128/256 and mixed gate-up/down tile widths.

# 125. Model-provenance record

The `.tqf` file stores enough provenance to prove what it was generated from, without embedding an arbitrary package-manager manifest.

Fields:

- source model repository ID;
- immutable revision/commit hash;
- source artifact name(s);
- source SHA-256 values;
- source model license identifier;
- converter semantic version;
- converter Git commit when available;
- conversion backend/layout ID;
- canonical tensor-inventory hash;
- hardware-independent conversion options.

Machine-specific tuning results do **not** belong in the 20+ GB model file unless they physically alter its layout. They live in the machine profile.

# 126. Conversion transaction state machine

**REFERENCE BASELINE.** Conversion is resumable per output extent.

```text
Absent
  │
  ▼
CreatingPartial
  │ create .partial + journal header
  ▼
WritingMetadataSkeleton
  │
  ▼
WritingExtents ─────── interruption ──────┐
  │                                       │
  │ journal marks verified completed extents
  ▼                                       │
VerifyingPayload <────────────────────────┘ resume
  │
  ▼
WritingFinalTables
  │
  ▼
FsyncPartial
  │
  ▼
AtomicRename
  │
  ▼
WritingTrustedReceipt
  │
  ▼
Installed
```

A journal entry is append-only until compaction/finalization and includes extent ID, expected source range/hash, target range, and completion hash. A completed extent is never trusted solely because its bytes exist; its journal hash must validate.

# 127. Conversion source-ownership rules

- If TQF downloads a temporary canonical source solely for conversion, successful conversion may delete the temporary source automatically.
- If source bytes belong to Ollama, a user path, or another external store, TQF never deletes them.
- Range-streamed remote conversion avoids creating the source file at all when feasible.
- Conversion failure leaves only `.partial` and journal data owned by TQF; `tqf doctor` can validate/resume/discard them.

# 128. Memory broker public contract

**REFERENCE BASELINE.** The memory broker uses RAII leases. A lease represents budget ownership, not necessarily a CPU pointer.

```rust
pub trait Reclaimable: Send + Sync {
    fn reclaimable_bytes(&self) -> Bytes;
    fn reclaim(&self, target: Bytes, reason: PressureReason) -> ReclaimResult;
}

pub struct MemoryLease {
    id: LeaseId,
    owner: MemoryOwner,
    class: MemoryClass,
    reserved: Bytes,
    committed: AtomicU64,
    broker: Arc<MemoryBrokerInner>,
}

impl MemoryBroker {
    pub fn reserve(
        &self,
        owner: MemoryOwner,
        class: MemoryClass,
        bytes: Bytes,
        alignment: u64,
    ) -> Result<MemoryLease, MemoryError>;

    pub async fn reserve_async(... ) -> Result<MemoryLease, MemoryError>;
    pub fn transfer(&self, lease: &mut MemoryLease, new_owner: MemoryOwner) -> Result<(), MemoryError>;
    pub fn snapshot(&self) -> MemorySnapshot;
}
```

Dropping a lease releases its reserved budget. Physical buffers whose deallocation is asynchronous retain the lease until the backend completion/deallocator confirms release.

# 129. Broker concurrency and deadlock rules

**LOCKED:**

1. The broker never calls a subsystem's reclaim callback while holding the broker's global accounting mutex.
2. Reservation performs accounting decision under a short lock, then invokes reclaimers in deterministic priority order outside the lock, then retries with a generation counter to detect races.
3. Reclaim callbacks may not recursively request protected memory from the same broker. They may release/demote resources only.
4. GPU resource destruction that can block waits on events outside the global broker lock.
5. Memory-owner stats use atomics or sharded counters; metrics scraping cannot block inference allocations.
6. At most one global pressure coordinator actively runs reclamation. Concurrent failed reservations join/await that pressure cycle rather than starting competing eviction storms.

# 130. Broker reservation algorithm

```text
reserve(owner, class, bytes):
    validate bytes/alignment
    if bytes > absolute_budget: fail ImpossibleRequest

    fast path:
        atomically/under short lock check free_reserved_budget
        if enough: debit owner; return lease

    slow path:
        enqueue request with priority and deadline
        become/join pressure coordinator
        coordinator computes shortage
        reclaim expired Transient
        reclaim Scratch
        shrink ExpertProbation by lowest value/byte
        reduce unused prefetch staging
        ask Context manager for permitted demotions
        unload inactive helper model
        if vision is elastic, shrink its accelerators
        retry reservations by priority

    if minimum protected footprint + request still cannot fit:
        fail with MemoryPlanError { requested, budget, minimum_required, suggestion }
```

The controller may alter elastic targets proactively before pressure occurs, but correctness never depends on predictive success.

# 131. Memory-owner priority order

Initial priorities, highest protection first:

1. backend/server safety reserve;
2. active Qwen mandatory resident core;
3. active GDN correctness state;
4. active-session logical context correctness state;
5. active output/sampling buffers;
6. in-flight I/O destinations referenced by pending commands;
7. transient helper model while its current operation is executing;
8. pinned high-value expert cache;
9. recent/hot context precision upgrades;
10. probationary expert cache;
11. speculative prefetch buffers;
12. background retrieval/index work;
13. inactive GUI caches/diagnostics.

This list does not imply static sizes. It defines what loses memory first.

# 132. OS-observed memory qualification

Internal accounting is necessary but insufficient. The 4G/2G qualification harness must sample OS-observed memory during adversarial scenarios.

On macOS record at least:

- physical footprint/task VM resident metrics;
- TQF internal reserved/committed totals;
- Metal heap/buffer totals known to TQF;
- peak during helper-model swaps;
- peak during vision activation;
- peak at context-page transitions.

On NVIDIA record:

- process host RSS;
- pinned host bytes;
- CUDA device allocated bytes;
- driver/context reserve separately;
- user-visible combined TQF budget measurement according to the declared accounting policy.

A configuration is not “4G certified” because steady-state decode is 3.9G if loading/reranking spikes to 4.7G.

# 133. Inference session state machine

**REFERENCE BASELINE.** One active generation v1.

```rust
pub enum SessionPhase {
    Queued,
    Preparing,
    Retrieving,
    Prefill,
    Decode,
    ToolBoundary,
    Finishing,
    CancelRequested,
    Failed,
    Complete,
}
```

Transitions:

```text
Queued -> Preparing
Preparing -> Retrieving? -> Prefill
Prefill -> Decode
Decode -> ToolBoundary -> Complete         // if model emits tool call/end turn
Decode -> Finishing -> Complete            // text completion
any active phase -> CancelRequested
CancelRequested -> Finishing/Complete      // after safe submitted work drains
any active phase -> Failed                 // nonrecoverable runtime/backend failure
```

Session owns:

- normalized request;
- tokenizer/chat state;
- context store handle;
- GDN state handle;
- prefix-snapshot ancestry;
- cancellation token;
- metrics span;
- output stream sender;
- sampling RNG state;
- optional retrieval provenance.

# 134. Per-token decode state machine

Each decoded token passes the following logical stages. Implementations may fuse command buffers but instrumentation must preserve equivalent stage accounting.

```text
TokenStart
  ↓
EmbeddingLookup
  ↓
for layer L = 0..39:
    InputNorm
      ↓
    TokenMixer(GDN or FullAttention)
      ↓
    Residual
      ↓
    PostAttentionNorm
      ↓
    RouterEncode
      ↓
    RouterReadbackBarrier
      ↓
    ExpertPlan
      ├─ cache hits ready
      ├─ shared expert compute may start
      └─ misses -> ExpertIoPending
      ↓
    RoutedPhase1 when required bytes ready
      ↓
    RoutedDownAccumulate
      ↓
    Shared+routed combine
      ↓
    Residual
  ↓
FinalNorm
  ↓
LmHead
  ↓
Sample/AcceptMtp
  ↓
ContextCommit
  ↓
TokenEnd
```

A layer's router barrier is a known CPU/GPU synchronization point because exact expert IDs drive storage reads. TQF's prefetch work exists largely to hide the consequences of this barrier; the scheduler must not pretend the dependency does not exist.

# 135. GPU/I/O overlap contract

**REFERENCE BASELINE:** maintain at most one previous layer's routed GPU work pending while computing the next layer's token mixer/router, as proven safe by command dependencies. More aggressive overlap is a research candidate.

At layer `L`:

1. submit token-mixer/tail/router work for `L`;
2. while it executes, prior layer `L-1` routed command may still complete;
3. wait only as required before reusing shared scratch/hidden state;
4. read router IDs for `L`;
5. create an immutable `ExpertPlan` reserving all destination cache entries;
6. launch shared expert and hit-only work that does not require misses;
7. submit bounded parallel reads for misses;
8. after relevant read completions, submit routed phase-1/down commands;
9. hold all leases until their GPU completion events fire.

No cache slot/tile can be reassigned between plan creation and final GPU completion.

# 136. `ExpertPlan` transaction

```rust
pub struct ExpertPlan {
    pub layer: LayerId,
    pub token_seq: u64,
    pub routed: [ExpertId; 8],
    pub weights: [f32; 8],
    pub bindings: [ExpertBinding; 8],
    pub misses: SmallVec<[MissRequest; 8]>,
    pub cache_generation: u64,
}

pub enum ExpertBinding {
    ResidentWhole { entry: CacheEntryId },
    ResidentTiles { entries: SmallVec<[CacheEntryId; 8]> },
    PendingWhole { reservation: CacheReservationId },
    PendingTiles { reservations: SmallVec<[CacheReservationId; 8]> },
}
```

Plan creation is atomic from the cache's perspective:

- every hit is pinned for the transaction;
- every miss destination is reserved;
- eviction candidates are marked unavailable;
- plan contains no raw pointers whose lifetime is not lease-backed.

If I/O fails, the entire plan fails and reserved entries return to a known empty state. Partial success must never make an unverified expert appear cache-valid.

# 137. Cache entry state machine

```rust
pub enum CacheEntryState {
    Empty,
    Reserved { plan: PlanId },
    Reading { io: IoRequestId },
    Ready { key: ExpertTileKey, generation: u64 },
    GpuPinned { key: ExpertTileKey, refs: u16 },
    EvictPending { key: ExpertTileKey },
}
```

Transitions are explicit. In particular, `Reading -> Ready` occurs only after the full requested byte range has been read and, when integrity checking is enabled for that extent, validated. `GpuPinned` may have multiple references if speculative/actual plans converge on the same cache entry.

# 138. Reference cache metadata

Start with metadata simple enough to validate:

```rust
pub struct CacheMeta {
    pub key: Option<ExpertTileKey>,
    pub state: CacheEntryState,
    pub bytes: u32,
    pub last_use_tick: u64,
    pub use_count: u32,
    pub decayed_frequency_q16: u32,
    pub measured_miss_ns_ema: u64,
    pub last_prefetch_tick: u64,
    pub prefetch_hits: u32,
    pub prefetch_waste: u32,
}
```

**REFERENCE BASELINE policy:** decayed-LFU admission/eviction with cost-per-byte weighting. Keep a trivial LRU and LFU implementation for controls.

Baseline score:

```text
reuse = 0.55 * normalized_decayed_frequency
      + 0.25 * recency_score
      + 0.20 * transition_probability

saved_ns = max(measured_miss_ns_ema, layer_default_miss_ns)

value = reuse * saved_ns / resident_bytes
```

The exact coefficients are **BENCHMARK-SELECTED** and must not become permanent magic constants without trace replay evidence.

# 139. Cache simulation before cache implementation changes

Every new policy must be testable against recorded routing/I/O traces without running the model. Trace format records:

```text
session class
model/profile fingerprint
token index
layer
selected expert IDs
routing weights
actual cache state before plan
miss byte ranges
measured completion latency
prefetch source/confidence
```

Trace replay outputs:

- byte hit ratio;
- expert-selection hit ratio;
- SSD bytes/token;
- estimated/unhidden I/O stall;
- admission churn;
- metadata memory;
- prefetch precision/recall/timeliness.

A cache policy is not promoted solely from simulation; simulation chooses candidates for real end-to-end A/B.

# 140. Parallel I/O worker design

**REFERENCE BASELINE on macOS:** a dedicated bounded I/O pool sized by the machine profile. Requests use positional reads and never mutate shared file offsets.

```rust
pub struct IoReadRequest {
    pub id: IoRequestId,
    pub file: ModelFileId,
    pub offset: u64,
    pub bytes: u32,
    pub destination: CacheReservationId,
    pub priority: IoPriority,
    pub deadline_hint: Option<Instant>,
}
```

Queues:

```text
DemandCritical   // exact current-layer misses
PredictedNear    // high-confidence next-layer prefetch
PredictedFar     // deeper speculative reads
Background       // index/model housekeeping; cannot starve inference
```

Demand reads always outrank speculative reads. If a demand request targets a byte range already in-flight speculatively, it promotes/joins that request rather than issuing a duplicate read.

# 141. I/O coalescing rules

The I/O scheduler MAY merge requests when:

- same model file;
- target extents are contiguous or separated by a small configurable gap;
- merged over-read bytes remain below the machine-profile threshold;
- destinations can receive the merged range without an extra full-copy penalty, or scatter from a bounded staging buffer remains cheaper than extra syscalls/latency.

The profiler measures:

```text
saved operation latency - extra bytes / measured bandwidth - scatter cost
```

Coalescing is disabled when SSD bandwidth, rather than operation latency, is the bottleneck.

# 142. Read-ahead and cache pollution

Read-ahead is **BENCHMARK-SELECTED** per machine and workload class. The controller records whether advised/prefetched pages actually become demanded. On memory pressure or poor precision, it reduces or disables read-ahead.

The resident core receives stronger page-residency protection than speculative expert ranges. A larger expert cache that causes the OS to evict/fault mandatory resident weights is a regression even if the logical expert hit rate rises.

# 143. Metal command-buffer structure

**REFERENCE BASELINE.** Preserve explicit command-buffer roles for timing and dependency reasoning:

```text
attn_or_gdn_cb      input norm + token mixer projections/recurrent/attention work
full_attn_cb        full softmax/value pass when separated for scheduling
router_tail_cb      output projection/residual/post norm/router
shared_cb           resident shared expert
routed_phase1_cb    routed gate/up/activation
routed_down_cb      down + weighted accumulate (may fuse with phase1 later)
head_cb             final norm/head/sampling helpers
```

A production optimization may fuse buffers, but must expose equivalent GPU timing labels in developer instrumentation and must show reduced end-to-end wall time.

# 144. Metal buffer ownership

Use `.storageModeShared` for expert buffers filled by CPU `pread` on Apple unified memory unless a measured alternative wins. Resident immutable weights may use a mapping/wrapping strategy optimized for page residency. Temporary GPU-only scratch may use private storage if copy cost is eliminated or hidden and memory accounting remains exact.

Every `MTLBuffer` wrapper stores:

```rust
pub struct MetalBufferLease {
    pub raw: ObjcMetalBuffer,
    pub memory: MemoryLease,
    pub bytes: u64,
    pub storage: MetalStorageClass,
    pub last_event: Option<GpuEventId>,
}
```

Deallocation does not release `MemoryLease` until outstanding events are complete.

# 145. MSL kernel ABI rules

Critical kernel argument indices are defined once in Rust and generated into/include-compatible MSL constants or verified by reflection tests. Do not hand-maintain two drifting argument-number lists.

Every performance kernel declares:

- expected quant-layout ID;
- exact logical shape(s);
- supported tile size(s);
- input/output scalar type;
- required buffer alignment;
- maximum threadgroup memory;
- deterministic accumulation expectation;
- reference-kernel name;
- timing metric label.

If a `.tqf` extent uses an incompatible quant-layout ID, pipeline construction fails before generation.

# 146. Q4 decode baseline

**REFERENCE BASELINE:** 4-bit affine/grouped source semantics with group size inherited from the canonical source. The kernel reads packed nibbles and scale/min-or-bias metadata directly and accumulates in FP32 where needed for numerical stability before producing BF16/FP16 hidden outputs according to the reference path.

No full row is dequantized into a persistent temporary buffer. A SIMD lane decodes the small block it immediately consumes.

Pseudo-body:

```text
for output row assigned to SIMD group:
    acc = 0f32
    for quant group g:
        load packed 4-bit block
        load group scale/(offset metadata)
        unpack lane-owned values into registers
        load activation fragment
        FMA into f32 partial
    SIMD reduce partials
    write output scalar
```

`TQF-Q4` packing candidates may reorder the packed block to remove shifts/gathers, but the validation decoder must prove the represented source values are unchanged.

# 147. GDN reference decode algorithm

For one token in a GDN layer:

1. RMS-normalize hidden state.
2. Compute fused or separate `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`.
3. Update the 4-wide depthwise convolution tail and apply SiLU to mixed QKV.
4. Split Q, K, V using exact dimensions 2048/2048/4096.
5. Reshape Q/K to key heads and V to 32 value heads; repeat Q/K as required to 32 value heads for recurrence semantics.
6. L2-normalize Q/K in FP32-equivalent semantics.
7. Compute `beta = sigmoid(b)`.
8. Compute `g = -exp(A_log) * softplus(a + dt_bias)` in FP32.
9. Decay/update each FP32 recurrent state matrix.
10. Produce core attention output.
11. Apply gated RMSNorm with `z`.
12. Project 4096 -> 2048.
13. Add residual.

The recurrent state per value head is 128x128 FP32, giving 2 MiB/layer for 32 heads; 30 layers are roughly 60 MiB before convolution tails/metadata. The implementation must preserve FP32 state unless a separate approximation passes the ≤1% global qualification.

# 148. Full-attention reference decode algorithm

For a full-attention layer:

1. RMS-normalize hidden state.
2. `q_proj` produces 8192 values, interpreted as 4096 query + 4096 sigmoid gate.
3. `k_proj` produces 512 values; `v_proj` produces 512.
4. Apply per-head Q/K RMSNorm.
5. Apply partial rotary position embedding to the first 64 dimensions of each 256-dimensional head.
6. Store K/V in the context representation.
7. Repeat/virtually broadcast two KV heads across 16 query heads; never physically duplicate full cache pages just to implement GQA.
8. Scale dot products by `256^-0.5 = 0.0625`.
9. Compute causal attention over permitted pages.
10. Concatenate 16x256 output.
11. Multiply elementwise by `sigmoid(gate)`.
12. `o_proj` 4096 -> 2048.
13. Add residual.

The BF16 full-KV reference uses 20 KiB/token across ten layers and is the correctness oracle for TQKV.

# 149. Router reference algorithm

Router input is the post-attention RMS-normalized hidden vector.

```text
logits = W_router[256,2048] * x
prob = softmax(logits in FP32)
(ids, weights) = topk(prob, 8)
weights /= sum(weights)
```

Tie behavior must match the reference implementation for deterministic tests. The initial reference can perform stable top-k on CPU for debugging; production performs router GEMV/top-k on GPU and reads back exactly eight IDs + weights.

# 150. Shared-expert reference algorithm

Shared expert is always resident:

```text
gate = gate_proj[512,2048] * x
up   = up_proj[512,2048]   * x
h    = silu(gate) * up
out  = down_proj[2048,512] * h
scale = sigmoid(shared_expert_gate[1,2048] * x)
shared_out = scale * out
```

Its compute should overlap demand expert I/O whenever doing so does not worsen the critical path through unified-memory contention.

# 151. Routed-expert reference algorithm

For each of 8 experts:

```text
gate/up = selected expert Q4 rows * x
h = silu(gate) * up
down = selected down projection * h
output += routing_weight * down
```

Production may process experts in fused batches and accumulate directly into one output buffer. It must preserve the routing weights and source Q4 semantics.

# 152. Prefill scheduler baseline

**REFERENCE BASELINE:** chunked prefill with a machine-profile default initially seeded at 4096 tokens on M4, subject to memory feasibility.

Within a chunk/layer:

1. run token mixer for the chunk;
2. route all chunk rows;
3. collect the set of distinct experts required by that layer/chunk;
4. fetch each distinct absent expert once;
5. execute expert work for all rows selecting that expert;
6. release chunk scratch before advancing if memory pressure requires.

Chunk size is reduced automatically when context/scratch pressure would violate `--memory`. The tuning metric is TTFT and total prefill wall, not chunk microkernel speed alone.

# 153. Sampling contract

Sampling must be independent of runtime optimization choices. Normalize protocol parameters into one internal `SamplingConfig`.

```rust
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<Vec<u32>>,
    pub max_output_tokens: u32,
}
```

Temperature 0 uses deterministic greedy selection. Developer parity tests always use greedy mode unless testing sampling itself.


# 154. Logical context representation

**LOCKED.** `--context N` is a logical capacity contract. The internal store is a hierarchy of immutable completed pages plus one mutable tail page per full-attention layer. GDN state is separate and constant-size.

```rust
pub struct ContextStore {
    pub logical_len: u32,
    pub max_logical_len: u32,
    pub full_layers: [FullLayerContext; 10],
    pub gdn: GdnStateStore,
    pub provenance: ContextProvenance,
    pub prefix_parent: Option<SnapshotId>,
}
```

At 128K/256K the baseline does not delete logical tokens. At ~1M, TQAttn may avoid evaluating unselected old pages, but those pages remain addressable state.

# 155. TQKV page geometry

**REFERENCE BASELINE:** 256 tokens/page for completed pages. The current tail page may contain 0..255 tokens in a high-precision append-friendly representation and is sealed/quantized when full.

Why 256 is the reference rather than a lock:

- 128K -> 512 pages/layer, small metadata burden;
- 1M -> ~3907 pages/layer, still searchable;
- one page is granular enough for selective attention;
- page data is large enough for efficient sequential SIMD/GPU access.

**BENCHMARK-SELECTED candidates:** 128, 256, 512, 1024. The page size is encoded per context-store version/profile; snapshots may not mix incompatible page formats without migration.

# 156. TQKV page lifecycle

```text
MutableTail(BF16/reference or high precision)
    │ fills to page size or memory-pressure seal
    ▼
Sealing
    │ calculate quant parameters/search summary/outliers
    ▼
ResidentCompressed
    │
    ├─ promote precision from richer backing if available
    ├─ demote to qualified lower precision
    ├─ move richer copy to SSD backing
    └─ snapshot references immutable page ID
```

Once a page is referenced by a persisted prefix snapshot, its canonical page bytes are immutable. Precision variants are addressed as derivatives of the canonical logical page ID.

# 157. TQKV page header

**REFERENCE BASELINE:** 128-byte in-memory/persisted backing header. The live GPU page may use a stripped device descriptor.

```text
u64 page_id
u32 token_start
u16 token_count
u8  layer_id
u8  kv_head_count            // 2
u16 head_dim                 // 256
u16 key_encoding
u16 value_encoding
u16 search_encoding
u16 flags
u32 key_payload_bytes
u32 value_payload_bytes
u32 quant_meta_bytes
u32 outlier_bytes
u32 search_bytes
u32 backing_generation
u64 key_payload_offset
u64 value_payload_offset
u64 quant_meta_offset
u64 outlier_offset
u64 search_offset
u8  content_hash[32]
reserved to 128 bytes
```

Offsets are relative to the page blob for disk-backed representations; in-memory descriptors may use buffer-relative offsets.

# 158. TQKV-Q8 reference encoding

**REFERENCE BASELINE correctness-compression step.** Before Q4/Q3 research, implement Q8 so the page machinery and fused decode path can be validated with low error.

Keys:

- store post-RoPE Key values initially for simplest correctness;
- signed int8 per channel over the page;
- one FP16 scale per `(kv_head, dimension)`;
- optional zero point is omitted for symmetric baseline;
- quantization is `round(clamp(k/scale, -127, 127))` with scale from max absolute value.

Values:

- signed int8 per token group;
- group width 64 dimensions;
- one FP16 scale per `(token, kv_head, group)`.

Q8 is not expected to meet the final 4G 128K budget alone; it is the first compressed oracle beneath BF16.

# 159. TQKV-Q4 reference candidate

**REFERENCE BASELINE for 128K capacity qualification after Q8.** This is a concrete first Q4 candidate, not a claim that it will survive the ≤1% gate.

Keys:

```text
for each page, layer, KV head, dimension:
    scale = max_abs(K[:,dim]) / 7
    q = signed int4(round(K/scale)), range [-7,7]
```

Store:

- packed signed 4-bit Keys, token-major within 32/64-dim GPU blocks;
- FP16 scale per head/channel per page;
- optional sparse outlier sidecar introduced only if the baseline fails quality.

Values:

```text
for each token, KV head, 64-dim group:
    scale = max_abs(V[group]) / 7
    q = signed int4(round(V/scale))
```

Store one FP16 scale/group/token.

Raw Q4 K+V payload is ~5 KiB/token across the ten full-attention layers. At 128K that is ~640 MiB before scale/metadata overhead. This is why 4G remains plausible even before more advanced compression.

# 160. TQKV low-bit research candidates

**RESEARCH CANDIDATES.** Implement as separate encoding IDs, never as hidden changes to Q4.

1. **Q3 symmetric** — 3-bit signed values + FP16 scale, packed in 32/64-value groups.
2. **Q2 asymmetric** — 2-bit values + per-group scale/zero; only for cold pages with richer backing.
3. **Rotated-Q3/Q4** — structured randomized/Hadamard-like rotation to reduce outliers before quantization.
4. **Outlier split** — low-bit bulk + sparse exact/FP16 outlier values and positions.
5. **Pre-RoPE Key encoding** — store unrotated keys and fuse 64-dim partial RoPE during attention consumption.
6. **TurboQuant-inspired vector quantization** — preserve inner products with a transform/codebook if end-to-end kernels remain cheap.

Each encoding ID has a standalone decoder and error-analysis harness. No encoding reaches production solely from perplexity; it must pass long-context coding/retrieval/tool suites.

# 161. Fused TQKV attention consumption

Production attention must not materialize an entire selected compressed page as BF16.

Kernel structure:

```text
for selected page:
    for query head:
        map to one of two KV heads
        stream quantized K blocks
        dequant/register-transform required K fragment
        apply RoPE fragment if pre-RoPE encoding
        accumulate q·k score
    online/blocked softmax
    stream quantized V blocks
    dequant fragment
    accumulate weighted V
```

Use numerically stable online softmax so large selected contexts do not require a giant logits array. A reference two-pass page implementation is acceptable before fusion.

# 162. TQKV precision transitions

The controller may only choose among **pre-qualified transitions**. Example profile table:

```text
128K balanced production profile:
    mutable/recent: BF16 or Q8
    sealed warm:    Q4
    old:            Q4

1M advanced profile candidate:
    recent:         Q8/Q6 candidate
    warm:           Q4
    cold:           Q3
    very cold:      Q2/Q3 search + richer disk backing
```

A memory-pressure callback cannot invent a new lower precision. It requests a transition from the context manager; the context manager chooses the lowest precision certified for the active qualification profile.

# 163. Context provenance and protection flags

Each page tracks provenance ranges independently from numeric K/V bytes.

```rust
bitflags! {
    pub struct ContextFlags: u32 {
        const SYSTEM      = 1 << 0;
        const DEVELOPER   = 1 << 1;
        const TOOL_SCHEMA = 1 << 2;
        const USER_RECENT = 1 << 3;
        const RETRIEVED   = 1 << 4;
        const PINNED      = 1 << 5;
        const TOOL_RESULT = 1 << 6;
        const REPO_RULES  = 1 << 7;
    }
}
```

A page may contain mixed provenance. If any token range is hard-protected, TQAttn must include the relevant page unless a later finer-grained subpage representation is implemented.

# 164. TQAttn reference selector

**REFERENCE BASELINE for selective 256K/1M attention:** Quest-style page upper-bound scoring over a compact post-RoPE Key summary. [R21]

For each completed page and KV head, maintain per-dimension `k_min` and `k_max` in a compact summary encoding. For a query vector `q`, an optimistic dot-product bound is:

```text
bound(q,page) = Σ_i q_i >= 0 ? q_i*k_max_i : q_i*k_min_i
```

With GQA, compute the bound for each query head using its mapped KV head and reduce page priority as the maximum or calibrated aggregate across heads.

Selector:

```text
selected = all recent-window pages
selected += all protected pages
score remaining pages cheaply
select highest scoring pages up to page/token budget
if uncertainty criterion triggers:
    expand budget
perform real attention only over selected pages
```

This is a **reference selector**, not the final claimed innovation. It establishes a quality/performance baseline against full attention.

# 165. TQAttn uncertainty fallback

Selective attention must fail toward more computation, not silent context loss.

Expand selected budget when any of the following occur:

- score gap at the selection boundary is below calibrated margin;
- query norm/summary statistics are outside the qualified calibration distribution;
- protected pages already consume most of the budget;
- quality-sentinel benchmark configuration requests full validation;
- selection returns fewer than the configured minimum historical tokens;
- page-summary quantization indicates saturation/outlier overflow.

Developer mode can force `full_attention` for A/B comparison at the same context.

# 166. Dynamic recent window

The recent exact/direct window is expressed in pages, not hard-coded tokens.

Controller inputs:

- current context length;
- attention-stage wall time;
- TQAttn selection quality proxy;
- memory pressure;
- current request type;
- selected historical page count.

Hard rules:

- never below the validated minimum recent window for the active profile;
- never shrink only to gain synthetic tok/s if the resulting profile has not passed ≤1%;
- increasing the recent window requires no quality requalification, only performance/memory feasibility.

# 167. Self-indexing Key research interface

A TQKV encoding may expose:

```rust
pub trait KeySearchEncoding {
    fn score_page(&self, query: &[f32], page: &SearchPageRef) -> f32;
    fn score_pages_batch(&self, query: &[f32], pages: &[SearchPageRef], out: &mut [f32]);
    fn false_negative_safety(&self) -> SearchSafetyClass;
}
```

Candidates include binary sign sketches, quantized min/max summaries, rotated low-bit summaries, and learned-free projections. Search metadata must remain far smaller than the underlying KV payload.

# 168. SSD KV backing format

For 1M advanced profiles, richer cold pages may live in a bounded global backing store:

```text
~/.tqf/runtime-cache/kv/<model-profile>/pages.dat
~/.tqf/runtime-cache/kv/<model-profile>/index.bin
```

Backing page record:

```text
logical_page_id
session/snapshot namespace hash
encoding ID
bytes
content hash
last_access_epoch
payload
```

The store is content-addressed where practical, append-first, and periodically compacted. Current-session required pages are protected from LRU deletion. The entire backing store counts against the global auxiliary disk budget.

# 169. Prefix snapshot identity

A prefix snapshot key is not a hash of raw JSON. It is a hash of the exact normalized **token stream and model/context semantics**.

```text
snapshot_key = BLAKE3(
    model_source_fingerprint ||
    tokenizer_template_version ||
    context_encoding_semantics_version ||
    token_ids[0..position]
)
```

Changing tool-schema canonicalization, tokenizer template, model revision, or context semantics invalidates incompatible snapshots automatically.

# 170. Prefix snapshot manifest

```rust
pub struct PrefixSnapshotManifest {
    pub version: u16,
    pub model_fingerprint: [u8; 32],
    pub token_hash: [u8; 32],
    pub position: u32,
    pub gdn_state_blob: BlobId,
    pub full_layer_pages: Vec<Vec<ContextPageId>>, // 10 logical layers
    pub tail_page_refs: Vec<Option<ContextPageId>>,
    pub created_at: u64,
    pub last_used: u64,
}
```

Snapshots reference deduplicated immutable page blobs. A crash while writing a snapshot cannot corrupt previously committed page blobs.

# 171. Prefix lookup algorithm

v1 uses longest exact token-prefix match.

Efficient implementation options:

- rolling chunk hashes every message boundary/periodic checkpoint;
- hash map from checkpoint hash -> candidate snapshot IDs;
- verify token length/final digest before restore.

Do not construct a giant token-by-token trie in RAM for every persisted session. A logical trie may exist in metadata, but lookup is checkpoint-hash driven.

# 172. MTP runtime contract

MTP is an optional acceleration candidate, not a separate model identity. TQF records:

```text
draft tokens proposed
accepted tokens
verification passes
unique routed experts in draft union
extra expert bytes
MTP head time
net accepted tok/s
```

The adaptive controller disables MTP when rolling net benefit is negative beyond hysteresis. Controller decisions are per machine/profile/context class and may adapt during a long session.

# 173. TQIndex storage principles

**LOCKED:** first-party, local, incremental, content-aware, no external vector database.

**REFERENCE BASELINE:** one project-local committed file plus a tiny journal:

```text
<root>/.tqf/
├── project.toml
├── index.tqi
└── index.journal
```

`index.tqi` is memory-mappable and generation-based. Updates append new immutable segments and a new commit record; compaction rewrites a fresh file atomically. This avoids fragile in-place pointer mutation while preserving fast incremental updates.

# 174. TQI superblock

First 4096 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | `TQFINDEX` |
| 8 | 2 | major |
| 10 | 2 | minor |
| 12 | 4 | superblock bytes |
| 16 | 16 | index UUID |
| 32 | 32 | root identity hash |
| 64 | 8 | latest committed generation |
| 72 | 8 | latest generation-table offset |
| 80 | 8 | latest generation-table bytes |
| 88 | 32 | generation-table hash |
| 120 | 8 | created at |
| 128 | 8 | last compacted at |
| 136 | ... | reserved |

The root identity hash uses normalized root device/path identity plus project UUID, but does not treat the human directory name as semantic evidence.

# 175. TQI generation model

Each sync creates a logical generation containing immutable segments:

```text
FileTableDelta
ChunkTableDelta
SymbolTableDelta
GraphEdgeDelta
LexicalTermDelta/Postings
VectorDelta
PartitionDelta
Tombstones
StatisticsDelta
```

The commit record references segment offsets and hashes. Startup finds the superblock's latest generation, validates the generation table, then builds lightweight overlay views across generations. When overlay count/fragmentation crosses a threshold, background compaction writes one new canonical generation.

# 176. Stable index IDs

Use persisted monotonic `u64` IDs, not hashes as primary keys.

```rust
FileId(u64)
ChunkId(u64)
SymbolId(u64)
EdgeId(u64)
VectorId(u64)
PartitionId(u64)
```

Hashes are used for identity/change detection.

- File rename can preserve `FileId` when detected via Git/inode/content evidence.
- A changed function normally receives a new chunk-content hash but may retain `ChunkId` if the structural node identity is confidently matched; otherwise tombstone+new ID is safer.
- Symbol IDs may persist through body-only changes when qualified name/signature identity remains stable.

# 177. File record

```rust
pub struct FileRecord {
    pub id: FileId,
    pub path_string: StringId,
    pub byte_len: u64,
    pub mtime_ns: u64,
    pub content_hash: [u8; 32],
    pub language: LanguageId,
    pub content_kind: ContentKind,
    pub confidence_q16: u16,
    pub generated_q16: u16,
    pub vendor_q16: u16,
    pub flags: FileFlags,
    pub first_chunk: u64,
    pub chunk_count: u32,
}
```

Path is metadata/hierarchy evidence only. Classification derives primarily from content/parser evidence.

# 178. Content-first classification algorithm

**REFERENCE BASELINE:**

```text
classify(file):
    read bounded sample: first 64 KiB; if large/ambiguous also middle+tail samples
    check known binary magic
    compute UTF-8 validity/NUL/control/entropy signals
    if confidently binary: return Binary/Asset

    detect shebang/modeline
    generate candidate languages from:
        syntax-token fingerprints
        shebang
        filename/extension weak prior
        known special basenames (Makefile/Dockerfile/etc.)

    try top parser candidates, normally <=3
    for each parse:
        measure parsed-byte coverage
        error-node ratio
        tree depth/node plausibility
        language-token consistency

    language = highest calibrated score
    kind = infer code/config/doc/structured/plain
    generated/vendor probabilities = independent detectors
```

Initial scoring reference:

```text
0.60 parser_quality
0.25 lexical/syntax fingerprint
0.10 metadata prior
0.05 shebang/modeline/known basename
```

These coefficients are calibration baselines. Parser success remains the strongest signal.

# 179. Parser-quality score

A reference normalized score:

```text
coverage      = parsed_non_error_bytes / sampled_or_file_bytes
error_penalty = min(1, error_nodes / max(1, total_nodes) * 8)
shape         = calibrated tree plausibility [0,1]
parser_quality = clamp(0.65*coverage + 0.20*shape + 0.15*(1-error_penalty), 0, 1)
```

For small text files, parse the entire file. For huge generated/minified files, parser sampling or early generated detection prevents pathological indexing cost.

# 180. Chunk record

```rust
pub struct ChunkRecord {
    pub id: ChunkId,
    pub file: FileId,
    pub parent_symbol: Option<SymbolId>,
    pub byte_start: u64,
    pub byte_end: u64,
    pub token_estimate: u32,
    pub kind: ChunkKind,
    pub content_hash: [u8; 32],
    pub embedding: Option<VectorId>,
    pub lexical_doc_len: u32,
    pub flags: ChunkFlags,
}
```

Chunk text itself remains in the source file, not duplicated wholesale into the index unless a small normalized/signature cache earns its storage cost. Retrieval validates the current content hash before returning a stale source span.

# 181. Structural chunking baseline

For code:

1. emit top-level module/type/function/trait/impl/macro/constant nodes;
2. include signature and leading docs/comments with the symbol chunk;
3. large symbol bodies over the embedding/token limit are split at nested statement/block boundaries while preserving `parent_symbol` and a signature prefix;
4. tiny adjacent declarations may be grouped only if semantic independence is poor and benchmark shows improvement;
5. never cut through a UTF-8 sequence or parser token.

For documents:

- heading section is primary chunk;
- large sections split by paragraphs/lists/code blocks;
- include heading breadcrumb in embedding input but not necessarily in returned raw span.

# 182. Symbol record

```rust
pub struct SymbolRecord {
    pub id: SymbolId,
    pub file: FileId,
    pub parent: Option<SymbolId>,
    pub name: StringId,
    pub qualified_name: StringId,
    pub signature: StringId,
    pub kind: SymbolKind,
    pub definition_chunk: ChunkId,
    pub byte_start: u64,
    pub byte_end: u64,
    pub visibility: VisibilityClass,
    pub language: LanguageId,
}
```

The index stores exact symbol names and normalized identifier tokens separately. Exact symbol lookup bypasses semantic ANN entirely.

# 183. Program graph edge schema

Initial edge types:

```rust
pub enum EdgeKind {
    Defines,
    Contains,
    Calls,
    References,
    Imports,
    Implements,
    Extends,
    TypeUses,
    Tests,
    Configures,
    GeneratedFrom,
    RecentChangeRelated,
}
```

Edges include confidence. Parser/compiler-quality exact relationships outrank heuristic references. Language frontends may start with syntactic references and grow more precise; the graph must never pretend a heuristic name match is a confirmed call edge.

# 184. Graph physical representation

Compacted generations use CSR-like adjacency grouped by source symbol/chunk and edge kind:

```text
node_index[node] -> edge_start/count
edge_targets[]
edge_kind[]
edge_confidence[]
```

Recent incremental generations keep small sorted delta arrays/overlays. Query traversal merges canonical + overlays. Compaction rebuilds canonical CSR.

# 185. Lexical index baseline

**REFERENCE BASELINE:** a custom BM25-ish inverted index plus exact identifier/path indexes.

Token streams:

- natural-language tokens;
- identifier whole token;
- identifier subtokens from snake/camel/digit boundaries;
- exact error/string literals where useful;
- path components as low-authority lexical metadata.

Postings are sorted by `ChunkId`, delta-encoded with variable integers or block bitpacking selected by benchmark. Each posting carries term frequency; document length lives in the chunk table.

Exact symbol/path maps use sorted/FST-like dictionaries or hash tables according to memory benchmark.

# 186. Lexical scoring baseline

BM25 reference:

```text
score(q,d) = Σ idf(t) * tf*(k1+1)/(tf + k1*(1-b+b*len/avg_len))
```

Initial `k1=1.2`, `b=0.75` are controls, not sacred constants. Identifier exact/prefix matches receive separately calibrated structural bonuses rather than distorting BM25's term statistics.

# 187. Embedding input canonicalization

`pplx-embed-v1-0.6b` receives a deterministic representation:

For code symbol chunk:

```text
<language> Rust
<path> src/runtime/cache.rs
<symbol> ExpertCache::evict(...)
<kind> method
<code>
...source chunk...
```

For documents:

```text
<path> docs/design.md
<headings> Runtime > Cache
<text>
...section...
```

Path is included as useful context but cannot determine content classification. Generated/vendor markers may be included or used only as ranking metadata depending on retrieval A/B.

# 188. Semantic vector normalization

Store one canonical full 1024-dimensional normalized embedding before TQVec conversion during indexing transaction. The transient FP/BF representation may be discarded after the compact representation is committed.

MRL prefixes at dimensions 256 and 512 may be generated by truncating the trained representation according to model semantics and re-normalizing as required by the embedding-model guidance/benchmark.

# 189. Flat semantic reference index

Before custom ANN, implement exact SIMD search over the compact/reference vector store.

Reference controls:

1. FP16 1024-d cosine/dot;
2. INT8 1024-d dot with per-vector scale if needed;
3. native binary embedding/Hamming candidate generation where available.

This flat search becomes the gold recall baseline for every approximate index.

# 190. TQVec candidate family

**RESEARCH CANDIDATES.** Every candidate must define byte size, decoder/distance kernel, MRL behavior, update cost, and recall loss.

### TQVec-A — native INT8 control

```text
1024 x int8 = 1024 bytes
+ scale/metadata
```

Simple SIMD dot-product baseline.

### TQVec-B — binary coarse + INT8 full

```text
256-d or 512-d sign/MRL key: 32/64 bytes
full INT8 vector: ~1024 bytes
```

Coarse Hamming filter, exact-ish INT8 refinement.

### TQVec-C — binary coarse + grouped Q5

```text
coarse 256-bit key:             32 B
1024 values x 5 bits:          640 B
32 groups x FP16 scale:         64 B
metadata/alignment:            ~16 B
------------------------------------
approx target:                 ~752 B/vector
```

### TQVec-D — binary coarse + grouped Q4

```text
coarse key:                     32 B
1024 values x 4 bits:          512 B
32 FP16 scales:                 64 B
metadata:                       ~16 B
------------------------------------
approx target:                 ~624 B/vector
```

### TQVec-E — rotated Q4/Q5

Apply a deterministic orthogonal/structured transform before grouped quantization to reduce outlier concentration. Store transform ID globally per repository/profile, not per vector.

### TQVec-F — residual hierarchy

Store a cheap MRL/binary base plus quantized residual information used only for top candidates.

The ambitious final choice may differ. These candidates make the research executable rather than aspirational.

# 191. Repository-adaptive TQVec calibration

A repository may build a tiny calibration profile from a bounded sample of canonical embeddings:

- per-dimension distribution moments;
- transform seed/ID;
- group scales/codebook parameters;
- quantization error statistics;
- MRL candidate-recall curves.

Calibration profile is versioned and included in the index generation. Incremental vectors use the same profile until a statistically significant distribution shift triggers a background recalibration/rebuild proposal. TQF does not silently rewrite the entire index during an interactive session without budget/transaction control.

# 192. Hybrid query classification

Reference query classes:

```rust
pub enum QueryIntent {
    ExactSymbol,
    ExactPath,
    ErrorLiteral,
    StructuralRelation,
    SemanticQuestion,
    ChangeHistory,
    GeneralDocument,
    Mixed,
}
```

Signals:

- quoted strings;
- language identifier forms (`::`, `.method`, function signature);
- path separators/extensions;
- compiler/stack-trace patterns;
- natural-language question density;
- active file/symbol from client metadata;
- Git/change words.

The router emits multiple lanes with confidence; it is not a single mutually-exclusive classifier.

# 193. Candidate-set contract

Every retrieval lane returns calibrated candidates in a shared structure:

```rust
pub struct Candidate {
    pub chunk: ChunkId,
    pub lane: RetrievalLane,
    pub raw_score: f32,
    pub rank: u32,
    pub exactness: Exactness,
    pub structural_confidence: f32,
    pub provenance: CandidateProvenance,
}
```

Fusion never compares raw BM25 and cosine numbers directly as though they share a scale.

# 194. Reference evidence fusion

**REFERENCE BASELINE:** weighted reciprocal-rank fusion with hard structural/exact precedence.

```text
rrf = Σ_lane weight_lane / (k + rank_lane)
```

Initial `k=60` control.

Then apply evidence bonuses/penalties:

```text
+ exact symbol definition bonus
+ exact path/literal bonus
+ confirmed graph relationship bonus
+ active-file/module proximity bonus
+ recent-change relevance bonus
- generated/vendor penalty
- stale/unverified span penalty
```

Hard rule: an exact requested symbol definition cannot be displaced by a semantically similar unrelated chunk solely through semantic score.

# 195. Graph expansion baseline

After fusion seeds, expand only a bounded local frontier.

```text
seed top 8-16
for each seed:
    add parent definition
    add direct callers/callees/references with confidence >= threshold
    add directly related tests
    add imported/implemented type neighbors when query intent suggests
limit total graph-added candidates
re-fuse evidence
```

Graph expansion records why each candidate was added so the GUI/metrics can explain retrieval behavior.

# 196. Reranker invocation policy

Load GTE only when an ambiguity heuristic fires. Reference heuristic:

- fused top score below confidence threshold; or
- top-1/top-5 score margin small; or
- semantic and exact/structural lanes strongly disagree; or
- broad semantic query returns many near-ties; or
- model explicitly requests deeper search.

Rerank at most a bounded candidate count initially (e.g. 24–48) and benchmark. TQF may shrink the Qwen expert cache to acquire helper-model memory, but unloads GTE before main decode resumes unless sustained multi-query use proves retention valuable within budget.

# 197. RAG context builder

Retrieval result selection is constrained by a dynamic token budget, not fixed top-N.

The builder:

1. groups overlapping chunks from the same file/symbol;
2. deduplicates near-identical generated/reference copies;
3. preserves exact definitions and project rules first;
4. allocates remaining budget by marginal evidence value/token;
5. includes concise provenance wrappers so Qwen can distinguish source files/doc sections;
6. avoids injecting low-confidence filler merely to consume budget.

Returned context provenance is stored so TQAttn can protect critical retrieved pages during the immediate generation.

# 198. Incremental sync transaction

```text
Full root walk
  ↓
compare path/stat/quick hash against FileTable
  ↓
for changed/new candidates compute full content hash
  ↓
parse/classify changed files
  ↓
compute structural chunks/symbols/edges
  ↓
reuse unchanged chunk embeddings by content+canonicalization hash
  ↓
embed only new/changed semantic chunks
  ↓
write generation delta to index.journal/index append area
  ↓
fsync data
  ↓
append generation commit record
  ↓
fsync
  ↓
atomically update superblock generation pointer
```

If embedding is deferred because Qwen is decoding, structural/lexical changes can commit first with `semantic_pending` flags. A later semantic delta fills them without making search unavailable.

# 199. Filesystem watcher behavior

Watcher events are hints, not the source of truth.

- debounce bursts in RAM;
- coalesce repeated writes/renames;
- on overflow/lost events, schedule a full correctness walk;
- never trust watcher event order as a transaction log;
- ignore `.tqf` internal writes to avoid recursion;
- pause/deprioritize semantic embedding under inference pressure;
- current exact/lexical metadata may update quickly even while embeddings wait.

# 200. Index compaction

Trigger when one or more thresholds are exceeded:

- generation count;
- tombstone ratio;
- postings/vector fragmentation;
- mmap segment count;
- measured query overhead.

Compaction writes `index.tqi.compact.partial`, validates it, fsyncs, then atomic-renames. Old committed index remains usable until replacement succeeds.

# 201. General-document behavior

Non-code roots use the same file/lexical/semantic infrastructure but omit unsupported program-graph edges. Markdown heading hierarchy, links, citations/references where recognizable, and structured configuration relationships become graph/hierarchy evidence. TQIndex must remain useful even when `Code% == 0`.


# 202. Internal protocol-normalized request model

All external APIs translate into one internal request representation before retrieval/tokenization. Protocol-specific code must not leak into the inference scheduler.

```rust
pub struct GenerateRequest {
    pub request_id: RequestId,
    pub model: ModelSelection,
    pub messages: Vec<NormalizedMessage>,
    pub tools: Vec<ToolDefinition>,
    pub response_format: ResponseFormat,
    pub sampling: SamplingConfig,
    pub reasoning: ReasoningRequest,
    pub stream: bool,
    pub retrieval: RetrievalRequest,
    pub vision_inputs: Vec<VisionInput>,
    pub client: ClientMetadata,
}

pub struct NormalizedMessage {
    pub role: MessageRole,
    pub parts: Vec<MessagePart>,
}

pub enum MessagePart {
    Text(String),
    Image(VisionInputId),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Reasoning(String),
}
```

Protocol adapters are responsible for rejecting fields whose semantics TQF cannot faithfully represent. Silently accepting unsupported behavior is worse than a clear compatibility error.

# 203. Model identifiers

Canonical public model ID:

```text
qwen3.6-35b-a3b
```

Optional aliases accepted by compatibility layers may include the source repository name or Ollama-like tags, but server responses should normalize to the canonical TQF ID unless the client requires echoing its requested alias.

Embedding model ID:

```text
pplx-embed-v1-0.6b
```

Reranker is initially internal and need not be exposed as a public generation model.

# 204. OpenAI Chat Completions contract

Initial supported request fields:

| Field | Behavior |
|---|---|
| `model` | canonical Qwen ID or recognized alias |
| `messages` | required; text/tool/vision parts as supported |
| `stream` | supported |
| `temperature` | supported |
| `top_p` | supported |
| `max_tokens` | compatibility alias |
| `max_completion_tokens` | supported |
| `stop` | string or array supported |
| `seed` | supported where deterministic sampling implementation permits |
| `tools` | function tools supported |
| `tool_choice` | auto/none/required/specific where Qwen chat formatting supports faithful mapping |
| `response_format` | JSON object/schema support implemented only when grammar/format enforcement is real |
| `n` | v1 supports only `1`; reject other values |
| `logprobs` | initially reject unless implemented faithfully |
| `frequency_penalty` / `presence_penalty` | reject or map only after explicit implementation; do not fake |

Unknown fields may be ignored only if OpenAI compatibility convention makes them explicitly optional and ignoring cannot change core semantics; otherwise return a structured 400 compatibility error.

# 205. Chat Completions streaming

SSE framing:

```text
data: {chat.completion.chunk JSON}\n\n
...
data: [DONE]\n\n
```

TQF maintains one UTF-8-safe incremental detokenizer. Stop-sequence matching operates across token/chunk boundaries before text is emitted. A byte sequence already sent cannot later be retracted because a stop matcher buffered too little; therefore the matcher retains the maximum required suffix before flushing.

Tests must include:

- multibyte UTF-8 split across token boundaries;
- stop string spanning 2+ tokens;
- empty deltas/tool-only deltas;
- tool JSON arriving over many tokens;
- client disconnect while GPU work is pending;
- exactly-once delivery after wakeups/backpressure.

# 206. OpenAI Responses contract

`/v1/responses` is the preferred modern surface and required for current Codex-style custom providers.

Initial internal mapping supports:

- `model`;
- `input` string or structured message items;
- `instructions` -> developer/system-style normalized guidance according to current protocol semantics;
- `tools`;
- `tool_choice`;
- `max_output_tokens`;
- `temperature`/sampling fields that have meaningful mapping;
- `stream`;
- text/JSON output configuration where faithfully enforceable.

TQF assigns response/item IDs locally. IDs do not claim OpenAI cloud provenance.

# 207. Responses streaming event sequence

Implement a deterministic event state machine rather than ad-hoc JSON writes. Reference event classes:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta ...
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

Tool-call events use the matching function-call item/delta forms expected by real clients. The exact compatibility set is frozen in protocol conformance fixtures derived from current public schemas; when schemas evolve, TQF updates deliberately rather than claiming blanket compatibility.

# 208. Embeddings API contract

`POST /v1/embeddings` routes to `pplx-embed-v1-0.6b`.

Requirements:

- string or array input;
- deterministic ordering;
- explicit model ID;
- float embedding output for compatibility even if the internal index stores TQVec;
- batching subject to memory broker;
- reject input exceeding helper model's qualified context rather than truncate silently.

Public embeddings do not expose the repository-adaptive TQVec representation; they return canonical model embeddings.

# 209. Anthropic Messages mapping

Anthropic facade translates:

```text
system            -> normalized system/developer guidance
messages          -> normalized messages
content text      -> Text
image blocks      -> vision input when enabled
assistant tool_use-> ToolCall
tool_result       -> ToolResult
tools             -> ToolDefinition
max_tokens        -> max output tokens
temperature/top_p -> sampling
```

Streaming adapter emits Anthropic-compatible message/content-block lifecycle events. Unknown Anthropic features are rejected with clear errors rather than partially emulated.

# 210. Ollama compatibility mapping

High-value endpoints:

```text
POST /api/chat
POST /api/generate
POST /api/embed
GET  /api/tags
POST /api/show
GET  /api/ps
```

Model-management/build endpoints that imply arbitrary model creation are not required. `/api/tags` can report the canonical Qwen model and embedding capability. `/api/ps` reports active model/runtime status in an Ollama-compatible shape as far as practical.

# 211. TQF-native status API

Native diagnostics are separate from compatibility namespaces.

Suggested endpoints:

```text
GET  /tqf/status
GET  /tqf/metrics
GET  /tqf/memory
GET  /tqf/context
GET  /tqf/indexes
POST /tqf/index/search
POST /tqf/index/sync
```

`/tqf/metrics` may expose JSON and/or Prometheus text via a separate content negotiation/path. Sensitive project paths should not be exposed to non-loopback clients without authentication/authorization.

# 212. HTTP error envelope

OpenAI surfaces return OpenAI-like errors; Anthropic surfaces return Anthropic-like errors. Internally map from structured `TqfError`.

Native reference:

```json
{
  "error": {
    "code": "memory_budget_too_small",
    "message": "1M context with vision requires at least 8 GiB on this profile",
    "retryable": false,
    "details": {"requested_bytes": 4294967296, "minimum_bytes": 8589934592}
  }
}
```

Never expose raw filesystem paths, backtraces, authorization tokens, or model source credentials in network error bodies.

# 213. Request queue semantics

v1 queue is FIFO with cancellation, except lightweight health/status/embedding/retrieval operations may have separate bounded queues when they do not violate memory/performance guarantees.

Generation queue entry records:

- arrival monotonic time;
- client connection handle;
- cancellation token;
- requested memory/context features;
- estimated preparation cost.

No starvation-inducing priority system in v1. Later interactive/background classes require explicit policy and tests.

# 214. Cancellation semantics

Cancellation is cooperative and safe:

1. set atomic/session cancellation flag;
2. tokenizer/retrieval/prefill loops stop at their next safe check;
3. GPU commands already submitted are allowed to complete unless backend supports safe cancel;
4. no new token/layer work is submitted after cancellation boundary;
5. in-flight cache I/O may be canceled if the OS/backend supports it cheaply, otherwise completes into reserved entries which are then made reusable;
6. output stream closes with protocol-appropriate termination if possible;
7. session resources release after pending GPU/I/O refs drain.

A canceled session cannot leave cache entries permanently `Reserved`/`GpuPinned`.

# 215. Network binding and auth

Defaults:

```text
host = 127.0.0.1
port = 11434 if free, otherwise detected TQF or reported fallback 11435
local auth = none
```

For non-loopback bind:

- generate an API key on first exposure unless one already exists;
- store key with restrictive filesystem permissions;
- require bearer auth on generation, retrieval and MCP HTTP surfaces;
- GUI clearly indicates network exposure;
- `--unsafe-no-auth` is an explicit scary override, not default behavior.

# 216. Setup state machine

```rust
pub enum SetupState {
    DetectingHardware,
    LoadingConfig,
    CheckingInstall,
    AwaitingDownloadConsent,
    ResolvingSource,
    Downloading,
    Converting,
    Verifying,
    QuickTuning,
    Ready,
    Failed,
}
```

Plain `tqf` runs this automatically. A GUI and CLI render the same underlying state/progress events.

# 217. Global data layout

Reference macOS/Linux logical layout:

```text
~/.tqf/
├── config.toml
├── auth.toml                # only if network auth generated
├── models/
│   ├── qwen3.6-35b-a3b/
│   │   ├── apple-metal-q4.tqf
│   │   └── receipt.toml
│   ├── pplx-embed-v1-0.6b/
│   └── gte-reranker-modernbert-base/
├── profiles/
│   └── <hardware-fingerprint>.toml/bin
├── runtime-cache/
│   ├── kv/
│   └── prefix/
├── logs/
└── notices/
```

Actual OS application-support paths may be selected for GUI-native packaging later, but CLI semantics expose one resolved data root. `TQF_HOME` may override it for portable/testing installs.

# 218. Project-local data layout

After `tqf sync PATH`:

```text
PATH/.tqf/
├── project.toml
├── index.tqi
├── index.journal
└── lock
```

`.tqf` should normally be added to ignore recommendations but TQF must not silently edit `.gitignore` unless the user asks. Project index can be rebuilt from source; project configuration preserves index UUID/root registration metadata.

# 219. Configuration schema

**REFERENCE BASELINE** `config.toml` keeps user-facing choices small.

```toml
version = 1
memory = "4G"
context = "128K"
vision = false
host = "127.0.0.1"
port = 11434
auto_rag = true

[model]
id = "qwen3.6-35b-a3b"
auto_update = false

[runtime]
auto_tune = true

[storage]
auxiliary_limit = "5G"
```

Machine-derived tuning knobs do **not** live here; they belong in the profile. Users should not see expert slots or Metal threadgroup sizes in normal config.

# 220. Hardware profile schema

Profile is invalidated by a compatibility fingerprint including:

- CPU/GPU family;
- Metal feature/OS version or NVIDIA architecture/driver capability;
- TQF build optimization ABI/version;
- model `.tqf` layout ID;
- SSD device identity/performance class where relevant.

Stored values may include:

```text
io backend + worker count
read-ahead policy
expert whole/tile cache defaults
Metal pipeline variant IDs
prefill chunk seed
TQKV kernel variant
TQAttn page kernel variant
thermal sustained alternatives
```

A profile is a cache of benchmark conclusions, not correctness-critical state. Deleting it causes safe retuning.

# 221. CLI parsing contract

Primary forms:

```text
tqf [server flags]
tqf sync <path>
tqf unsync <path>
tqf status
tqf doctor
tqf optimize
tqf licenses
tqf --open <opencode|claude|codex> [--gui]
```

Rules:

- `tqf` with no subcommand starts server and macOS GUI where supported.
- `--headless` suppresses GUI.
- explicit flags override persisted config for that invocation only unless a dedicated config command is later added.
- invalid memory/context values fail before expensive model loading.
- `--memory 2G` never silently becomes 4G.
- `--context` accepts friendly K/M suffixes and normalizes to token count.
- `--enable-vision` is the canonical vision flag; do not reintroduce `--vision` as a competing primary name.

# 222. `tqf sync` semantics

`sync PATH` means: make the registered index for `PATH` correct and as complete as current resource policy permits.

It performs:

1. root normalization/symlink safety check;
2. project registration if new;
3. full correctness walk;
4. structural/lexical delta commit;
5. semantic delta immediately when resources available, otherwise foreground progress unless server is actively serving and policy defers it;
6. watcher registration while a TQF server uses the root.

No `--full` distinction is needed for normal users because every sync is logically full-correctness with incremental reuse. Developer-only `--rebuild` may discard and recreate acceleration structures.

# 223. Index selection for normal server requests

TQF does not assume every cwd is a project.

Index selection priority:

1. explicit request/index ID from TQF-native client metadata;
2. `--root`/integration-selected registered root;
3. cwd exact registered root or descendant mapping if unambiguous;
4. no index.

Automatic RAG only runs when an active index is selected. Plain general chat remains normal inference.

# 224. `--open` process model

`--open` is an ephemeral child-process launcher.

```text
parse client
ensure server/runtime ready
select/register? existing synced project only; do not silently index unsynced root
incrementally sync selected root
construct temporary env/config
construct temporary MCP endpoint/config
spawn child inheriting terminal
forward signals
wait for child
remove temporary files/env scope
server exits with tqf process unless configured otherwise
```

No permanent client config mutation.

# 225. OpenCode integration contract

Prefer documented runtime environment/config injection. TQF creates in-memory or temporary configuration specifying:

- TQF OpenAI-compatible base URL;
- canonical Qwen model;
- temporary MCP search tools when index active;
- local API key placeholder/real key as required by client parser.

If OpenCode is absent, ask before installing using official recipe or recognized package manager.

# 226. Claude Code integration contract

Launch with temporary Anthropic gateway environment pointing at TQF, plus MCP configuration if available. TQF's Anthropic facade translates requests to the same normalized runtime. Do not modify the user's global Claude configuration.

# 227. Codex integration contract

Current Codex custom providers use the Responses wire API. [R35] TQF generates a temporary provider configuration with:

```text
base_url = http://127.0.0.1:<port>/v1
wire_api = responses
requires_openai_auth = false
```

plus MCP integration when an index is active. Temporary config location/environment must be chosen according to Codex's supported override mechanisms at implementation time and covered by an integration fixture.

# 228. MCP server contract

Support both stdio and streamable HTTP.

Initial read-only tools:

```text
tqf_search(query, path?, symbol?, limit?)
tqf_symbol(name, path?)
tqf_references(symbol_id/name, limit?)
tqf_callers(symbol_id/name, limit?)
tqf_tests(symbol_id/name, limit?)
tqf_file(path, range?)
tqf_repo_map(path?, depth?)
```

Tools return stable identifiers and concise provenance. They do not edit files or execute commands.

# 229. Retrieval security boundary

An index may contain private source/documents. Therefore:

- loopback APIs inherit local-user trust model;
- non-loopback retrieval requires auth;
- logs must not dump retrieved source by default;
- metrics expose counts/timings, not code text;
- `tqf status` may show registered paths locally;
- HTTP status returned remotely should redact paths unless authorized administrative endpoint is explicitly used.

# 230. macOS single-binary build architecture

**LOCKED product behavior:** one distributed `tqf` executable. SwiftUI source is built into that executable; it is not a sibling helper app.

`src/build.rs` on macOS:

1. locates the system `swiftc`/SDK toolchain used for supported build;
2. compiles Swift files in `src/gui/macos/` to object/module artifacts in `OUT_DIR`;
3. emits Cargo link-search/link-arg directives for those objects and required Apple frameworks;
4. compiles/embeds baseline Metal library resources or generates Rust byte arrays;
5. generates a small C header/Rust extern declaration for the bridge;
6. on Linux, skips Swift entirely.

Release CI verifies the resulting distribution contains one executable and no TQF-owned helper executable/dylib bundle.

# 231. Swift/Rust ABI

Keep FFI narrow.

Swift exports C-callable functions such as:

```swift
@_cdecl("tqf_gui_run")
public func tqf_gui_run(_ configPtr: UnsafePointer<CChar>?) -> Int32
```

Rust exposes only lifecycle/status helpers if localhost HTTP/events cannot cover a need. The GUI should primarily talk to the running Rust server over loopback so runtime structures never cross the FFI boundary.

No Rust object pointer is retained by SwiftUI unless wrapped in an explicitly lifetime-managed opaque handle.

# 232. macOS process/thread lifecycle

GUI mode:

1. Rust `main` performs argument parsing and required synchronous setup decisions.
2. Start the Tokio/control-plane server on a dedicated Rust thread/runtime.
3. Keep inference worker(s) on dedicated scheduler threads.
4. The original process main thread enters AppKit/SwiftUI via `tqf_gui_run`.
5. SwiftUI connects to loopback TQF status/generation APIs.
6. Closing the last primary TQF window requests graceful server shutdown by default, then exits process after active request cancellation/drain.

Headless mode never initializes AppKit/SwiftUI.

# 233. SwiftUI source adoption rules

Directly adapted TurboFieldfare/NVMAI Apache-2.0 Swift files: [R7][R8]

- retain relevant copyright/attribution notices;
- carry prominent modification notice;
- remove dependencies on their Swift inference model/runtime;
- replace view-model actions with TQF HTTP/event client state;
- preserve only useful UI behavior, not internal architecture.

The visual product evolves into TQF's own theme and inspector rather than remaining a rename.

# 234. GUI event channel

Generation content uses the same server streaming protocol as external clients where practical. Runtime metrics/setup/index state may use:

- lightweight polling for slow state; and/or
- a TQF-native local event stream/WebSocket/SSE endpoint.

The GUI must not access GPU buffers or memory broker directly. This keeps the frontend replaceable and prevents UI lifecycle from becoming inference correctness state.

# 235. NVIDIA backend semantic contract

CUDA backend implements the same model semantic operations but not the same physical buffer strategy.

Minimum abstractions:

```rust
trait Backend {
    type Buffer;
    type Event;
    type Stream;

    fn q4_gemv(...);
    fn gdn_decode(...);
    fn full_attention_decode(...);
    fn router(...);
    fn shared_expert(...);
    fn routed_moe(...);
    fn lm_head(...);
    fn tqkv_attention(...);
}
```

The trait must not force Metal command-buffer details or CUDA stream/event details into a fake universal API. It represents logical operations and dependencies.

# 236. CUDA storage pipeline

Reference Ampere/RTX path:

```text
SSD file
  ↓ async/parallel host read
pinned host staging lease
  ↓ cudaMemcpyAsync on transfer stream
VRAM cache reservation
  ↓ event
compute stream waits only when expert actually needed
  ↓
Q4 CUDA kernel
```

Host staging is bounded by memory broker and reused as a ring. Direct-storage/GDS is optional only on hardware where officially supported and benchmarked.

# 237. CUDA kernel distribution

The single binary may embed:

- PTX for forward compatibility where appropriate;
- cubins/SASS-targeted artifacts for qualified architectures when build pipeline supports it.

Runtime compiles/loads through installed NVIDIA driver APIs. CUDA toolkit is not a runtime user requirement.

# 238. CPU SIMD role

CPU SIMD is for:

- tokenizer/text preprocessing;
- TQVec/flat retrieval search;
- binary Hamming/popcount search;
- low-cost route predictors/cache statistics;
- hashing/checksums;
- selected setup/conversion transforms;
- I/O coordination.

Do not offload major Qwen decode matrix math to CPU on Apple merely because cores are idle; unified-memory bandwidth contention can reduce total throughput.


**PART XV**

**Testing, CI, Fault Injection, Release, and Operations**

The performance ambition makes testing stricter, not looser. TQF must be able to distinguish a faster kernel from a corrupted one, a real memory reduction from unreported page-cache use, and a retrieval improvement from benchmark leakage.

# 239. Test hierarchy

TQF uses five layers of validation:

1. **Pure unit tests** — parsers, quant decode, data structures, scoring, protocol conversions.
2. **Golden/reference tests** — Qwen tensors/tokenizer/intermediate outputs against trusted reference implementations.
3. **Subsystem integration tests** — `.tqf`, broker, I/O/cache, context, retrieval, server streaming.
4. **Hardware qualification** — M4 and RTX 3070 Ti performance/memory/correctness runs.
5. **End-to-end quality suites** — coding, long context, reasoning, tool use, retrieval, vision.

A performance patch cannot skip levels 1–3 because it “only changes a kernel.”

# 240. Deterministic golden corpus

The repository SHOULD include a small legally redistributable deterministic fixture set rather than model weights:

- tokenization/message fixtures;
- synthetic quantized matrices with known outputs;
- small GDN state-transition fixtures;
- attention K/V fixtures;
- router/top-k fixtures;
- `.tqf` metadata fixtures;
- `.tqi` miniature repositories;
- protocol request/stream fixtures.

Large real-model qualification is executed against a separately downloaded pinned checkpoint and records its source fingerprint in result artifacts.

# 241. Q4 repack qualification

For every source Q4 tensor:

1. decode source quant blocks using importer reference decoder;
2. repack to TQF layout;
3. decode TQF blocks using independent validation decoder;
4. compare represented quantized scalar values exactly;
5. compare GEMV outputs for random deterministic activation vectors;
6. check first/last/unaligned row cases;
7. fuzz block offsets/alignment.

A layout cannot be called “lossless” merely because model output looks similar.

# 242. Layerwise model qualification

For a bounded collection of deterministic token/prompt fixtures, compare:

- input RMSNorm output;
- GDN q/k/v/z/a/b projections;
- convolution state/tail;
- FP32 recurrent state after selected steps;
- GDN output;
- full-attention Q/K/V after norm/RoPE;
- attention output;
- router logits/top-8 IDs/weights;
- shared expert output;
- routed expert output;
- layer residual output;
- final logits.

Exact source-Q4 operation order should aim for greedy parity. Fused/reordered arithmetic uses predeclared absolute/relative error limits validated against downstream token behavior.

# 243. Greedy sequence qualification

Mandatory deterministic lengths:

```text
1 token
16 tokens
128 tokens
512 tokens
```

Workload fixtures include:

- prose;
- code generation;
- JSON/tool-call generation;
- repetitive stress case;
- prompts that route differently across experts.

Cross-process deterministic output is compared on semantic content/deltas, not protocol-generated IDs/timestamps.

# 244. Memory broker tests

Unit/integration scenarios:

- reserve/drop simple lease;
- alignment accounting;
- committed < reserved lazy backend;
- concurrent reservations;
- one pressure coordinator only;
- reclaim callback reentrancy guard;
- helper model swap under full expert cache;
- vision activation under 4G;
- context page demotion under pressure;
- impossible request returns suggestion;
- canceled GPU work delays lease release correctly;
- leaked lease detector in test builds.

A test-only broker can use a tiny 16–64 MiB budget to force pressure paths quickly.

# 245. Hard 4G/2G qualification scenarios

Measure OS-observed peak under:

1. cold startup/model load;
2. short decode at largest expert cache;
3. 128K populated context decode;
4. embedding query while context active;
5. rerank query while context active;
6. prefix snapshot save/restore;
7. `--enable-vision` image request;
8. cancellation during expert I/O;
9. long-running thermal decode;
10. background index sync while server is active.

A profile fails certification if any normal path exceeds its declared hard limit beyond the explicitly documented OS/driver accounting allowance.

# 246. `.tqf` parser fuzzing

Fuzz targets:

- superblock arbitrary bytes;
- integer overflow in offsets/counts;
- overlapping sections;
- truncated files;
- absurd table counts;
- bad string offsets;
- expert tile range outside superextent;
- unsupported quant layout;
- corrupted metadata hash;
- valid metadata with malicious huge logical dimensions.

Expected result is clean error, never panic, OOB map/read, or giant allocation before validation.

# 247. `.tqi` parser fuzzing

Fuzz:

- bad generation pointers;
- cyclic/overlapping segments;
- invalid IDs;
- corrupted postings lengths;
- graph edge target overflow;
- malformed string table;
- vector payload length mismatch;
- tombstone references to unknown generations;
- truncated latest generation with valid previous commit.

Reader must fall back to latest valid committed generation when safe and report repair requirement.

# 248. Conversion fault injection

Automated torture test interrupts conversion after every meaningful transaction stage:

```text
after partial creation
during metadata write
after N extents
mid extent write
before journal fsync
after journal fsync
before final table
before/after target fsync
before atomic rename
after rename before receipt
```

Restart must either resume safely or declare/recover the state without accepting corrupted model bytes.

# 249. Index sync fault injection

Simulate:

- process kill during generation append;
- disk full;
- permission failure;
- source file changes while being hashed;
- rename during parse;
- watcher overflow;
- embedding helper unload/cancel;
- compaction crash before/after rename.

The previous committed index remains queryable until a new generation is fully committed.

# 250. Expert-cache race tests

Stress with deterministic fake I/O delays:

- same expert requested by concurrent speculative and demand plan;
- eviction candidate becomes pinned before eviction commits;
- read fails after reservation;
- cancel while read pending;
- GPU pin survives cache-policy update;
- tile and whole-expert views overlap;
- generation counter rollover/large values;
- policy metadata updates while plan is immutable.

Use Loom or equivalent concurrency testing where practical for small state machines; use randomized high-iteration stress tests for OS/GPU paths.

# 251. I/O benchmark protocol

For each candidate backend:

- test exact expert-sized reads;
- tile-sized reads;
- 1–8 concurrent misses;
- contiguous/coalesced and random extents;
- cold filesystem cache;
- warm filesystem cache;
- sustained run to expose SSD throttling;
- concurrent GPU/CPU memory traffic.

Report:

```text
p50/p95 read latency
GiB/s
CPU utilization
bytes over-read
page-cache effect
end-to-end decode delta
```

A microbenchmark win does not become default until generation A/B confirms it.

# 252. M4 performance qualification protocol

Reference M4 Air runs should document:

- macOS version;
- TQF commit/profile;
- model source/hash;
- room/device thermal state where feasible;
- AC/battery status;
- memory pressure;
- context length and whether populated;
- generation length;
- prompt fixture hash;
- cold/warm model/file-cache condition.

Run order should interleave A/B candidates rather than run all A then all B, reducing thermal/time bias.

For meaningful optimizations use multiple rounds and report median plus dispersion, not best-of-N alone.

# 253. Sustained thermal benchmark

Because base M4 Air is passive, maintain at least one 20–30 minute sustained generation/prefill stress protocol. Record rolling:

- tok/s;
- GPU stage times;
- I/O stall;
- power/thermal indicators available without private APIs;
- controller kernel switches.

The profile may store a cold winner and sustained winner if runtime detection can choose safely.

# 254. NVIDIA performance protocol

RTX 3070 Ti qualification records:

- GPU model/VRAM;
- driver version;
- PCIe link capability;
- CPU/storage model;
- pinned host memory;
- CUDA device allocation;
- transfer/compute overlap;
- SSD cold/warm state.

Do not directly compare M4 unified-memory and RTX host+device numbers without describing topology.

# 255. Context quality suite

For each compression/selection profile compare to full/reference behavior on:

- passkey/needle retrieval at varied positions;
- multiple needles;
- distractor-heavy code/document context;
- repository symbol introduced very early and used late;
- system/tool instruction retention;
- long conversational consistency;
- code dependency/reference questions;
- JSON/tool-call correctness after long context;
- reasoning tasks whose evidence is distributed across context.

Report both task scores and failure examples. Aggregate ≤1% does not excuse catastrophic regressions on critical instruction/tool behavior; those receive separate guardrails.

# 256. Combined quality budget

Approximate components are qualified **together**. Test profiles include combinations:

```text
TQKV only
TQAttn only with high-precision KV
TQKV + TQAttn
TQKV + MTP
TQKV + TQAttn + MTP
RAG injection + compressed context
1M full advanced stack
```

Do not add independent 0.8% regressions and assume the total remains acceptable.

# 257. Retrieval evaluation

Metrics:

- Recall@K;
- MRR/nDCG where benchmark supports;
- exact symbol/path success;
- code retrieval benchmark score;
- downstream answer/task success;
- latency p50/p95;
- RAM;
- index bytes/chunk;
- update latency;
- reranker invocation rate/cost.

Baselines:

```text
exact/lexical only
BM25
flat FP16/INT8 semantic
HNSW baseline
DiskANN-style baseline when practical
BM25 + semantic fusion
fusion + graph
fusion + graph + reranker
TQF adaptive partition candidate
```

# 258. Retrieval ablation rules

Every novel feature needs an ablation:

- remove structure;
- remove semantic;
- remove lexical;
- remove graph expansion;
- remove Git metadata;
- replace TQVec with INT8;
- replace adaptive partitions with flat;
- disable reranker;
- disable workload adaptation.

SOTA-like complexity that does not improve the Pareto frontier is removed or remains experimental.

# 259. Content-classifier test corpus

Include deliberately misleading cases:

```text
~/gaysex/meridian/src/foo            // extensionless Rust
~/Documents/fuckAlchemist/main.py    // contains C++
~/not-code/repo/README                // prose
image.rs                             // actual PNG bytes
Dockerfile                           // extensionless config
Makefile                             // build language
script                               // Python shebang
minified.js                          // code/generated-like
vendor/generated.rs                  // Rust + generated signal
```

Expected classification is based on contents/parser evidence, never parent folder semantics.

# 260. Protocol conformance tests

Maintain golden request/response/stream fixtures for:

- OpenAI Chat;
- OpenAI Responses;
- Anthropic Messages;
- Ollama chat/generate/embed;
- tool calls;
- structured output;
- errors;
- cancellation/disconnect.

Integration CI runs representative official/open clients where licensing/automation permits, especially Codex/OpenCode-style provider configuration paths.

# 261. Server fuzz/security tests

Fuzz/limit:

- malformed JSON;
- giant arrays/messages/tool schemas;
- invalid UTF-8 at HTTP boundary;
- header abuse;
- chunked transfer edge cases;
- slow clients/backpressure;
- auth parsing;
- path traversal in model/index native APIs;
- MCP argument bounds.

Network-facing inputs never become file paths without root/permission validation.

# 262. CI lane design

### Lane A — portable fast

Every push/PR:

- `cargo fmt --check`;
- clippy with project lint policy;
- unit tests;
- parser/index fixtures;
- format roundtrip;
- protocol fixtures;
- no hardware/model required.

### Lane B — macOS compile/reference

- build one-binary SwiftUI/Metal target;
- Metal shader compile;
- small synthetic GPU kernel tests;
- FFI lifecycle smoke test.

### Lane C — Linux/NVIDIA compile

After CUDA phase begins:

- build driver integration;
- PTX/cubin load smoke;
- synthetic kernel correctness.

### Lane D — scheduled model qualification

On secured hardware with downloaded checkpoint:

- 512-token greedy parity;
- 4G memory suite subset;
- short performance regression suite;
- context smoke.

### Lane E — release qualification

Full M4/3070Ti/context/retrieval/quality/fault suite.

# 263. Performance regression gates

Do not fail normal PR CI on tiny noisy changes. Scheduled/reference hardware uses thresholds:

- >5% stable decode regression on core fixture: fail/investigate;
- >10% TTFT regression: fail/investigate;
- memory-budget violation: hard fail;
- SSD bytes/token large unexplained increase: fail/investigate;
- any quality-gate violation: hard fail.

Thresholds are tightened as benchmark variance becomes characterized.

# 264. Optimization ledger schema

Each experiment record should be machine-readable Markdown/TOML/JSON plus prose summary:

```text
experiment ID
date/commit
hypothesis
source inspiration
hardware/profile
baseline
variant
fixtures
memory/context
metrics
quality check
result
keep/reject/follow-up
```

Rejected experiments stay searchable. NVMAI/TurboFieldfare upstream mining entries can point directly to their commit/experiment source and TQF reproduction result.

# 265. Upstream research-mining workflow

TurboFieldfare and NVMAI are permanent research inputs.

Periodically:

1. compare upstream commits since last mined SHA;
2. classify runtime/kernel/I/O/setup/UI changes;
3. record promising changes in ledger;
4. port only after understanding assumptions;
5. reproduce on TQF M4 profile;
6. preserve Apache attribution for directly adapted code;
7. reject changes that conflict with 4G/128K architecture even if they win on 24G hosts.

# 266. Release artifact requirements

Release must include:

- one `tqf` executable per supported OS/arch artifact;
- Apache-2.0 LICENSE;
- NOTICE/third-party notices accessible alongside release and through `tqf licenses`/GUI;
- checksum/signature according to release policy;
- no bundled Qwen/helper weights;
- concise quick start;
- compatibility/support matrix.

# 267. Release smoke test from clean machine

On a clean supported M4 machine/user account:

```text
copy tqf binary
run tqf
accept model download
interrupt once and resume
finish conversion
launch GUI/server
send OpenAI request
sync a small project
do automatic RAG request
restart and verify trusted install/prefix/index reuse
```

No Xcode/Python/Homebrew dependency should be required for the downloaded release binary's normal runtime, beyond OS facilities and network access for model setup.

# 268. Security and privacy operational rules

- No telemetry by default.
- No prompt/repository upload.
- Model downloads use pinned HTTPS source and checksum verification.
- Tokens/API keys are redacted from logs.
- Non-loopback API uses auth by default.
- Project retrieval obeys registered-root boundaries.
- Symlinks cannot escape root during indexing by default.
- Model/index parsers treat files as hostile inputs.
- Automatic client install always asks.

# 269. Log levels and retention

Normal logs are compact:

```text
ERROR/WARN: actionable failures
INFO: startup/model/setup/request summaries
DEBUG: scheduler/cache/context details
TRACE: per-layer/per-I/O events, developer only
```

TRACE routing/kernel events remain RAM/ring-buffered or explicitly opted into; default logs must not produce huge SSD writes. Logs count against auxiliary storage and rotate automatically.

# 270. Metrics cardinality rules

Do not label metrics by arbitrary file path, symbol, expert ID or request ID in a way that creates enormous cardinality. Per-expert detailed stats live in developer snapshots, while production metrics aggregate by layer/cache class/workload class.

# 271. Crash diagnostics

On internal fatal errors, capture a bounded diagnostic report containing:

- TQF version/commit;
- model/profile fingerprint;
- platform;
- memory snapshot;
- current session phase;
- last scheduler stage;
- recent non-sensitive stage timing;
- GPU/backend error;
- no prompt/code text by default.

User may explicitly choose to share it; no automatic upload.

**PART XVI**

**Phase-Level Engineering Taskbook**

Part XIII gave the dependency map and exit gate. This taskbook adds the implementation work products so a coding agent does not have to invent the path inside each phase. File names are targets, not a ban on reasonable refactoring within the one-crate rule.

# 272. Phase 0 — research harvest and canonical manifest

**Primary files:** `src/dev/`, `docs/research/` or repository research records, generated inventory artifacts.

Tasks:

- Pin official Qwen3.6 model/config/tokenizer/MTP/vision source revisions.
- Record SHA-256 and licenses.
- Clone/freeze mined TurboFieldfare and NVMAI SHAs in research ledger.
- Build tensor-inventory generator against source metadata.
- Create derived architecture calculator tests for dimensions/parameter bytes.
- Record known NVMAI wins and dead ends as experiments-to-reproduce.

Tests:

- official config fields equal compile-time `Qwen36Geometry` constants;
- all 40 layer kinds match 3:1 pattern;
- generated inventory hash stable.

Do not proceed to format design with unresolved text-tensor names/shapes.

# 273. Phase 1 — crate skeleton

Create the `src/` module tree and enforce one package in CI.

Implement:

- `TqfError` skeleton;
- logging initialization;
- `Bytes/Tokens/LayerId/...` newtypes;
- platform compile gates;
- CLI parser with locked public flags;
- config path resolution.

Exit artifact: one release `tqf` that prints help/version on macOS/Linux.

# 274. Phase 2 — server skeleton

Implement:

- Tokio runtime/control plane;
- Axum router;
- `/health`;
- `/v1/models` placeholder;
- request IDs;
- cancellation token abstraction;
- SSE writer regression fixture.

No model code required yet. Establish protocol/test architecture now so inference later plugs into stable normalized requests.

# 275. Phase 3 — setup/global state

Implement `SetupState`, `~/.tqf`, atomic config, consent prompts and `--yes` policy.

Tests:

- no model -> Y continues/N exits;
- noninteractive missing confirmation fails cleanly;
- concurrent `tqf` setup uses an install lock and does not corrupt state;
- stale lock recovery policy is explicit.

# 276. Phase 4 — source resolver/downloader

Implement source abstraction:

```rust
trait ModelSource {
    fn metadata(&self) -> ...;
    async fn read_range(&self, offset, len) -> ...;
}
```

Backends:

- pinned HF HTTP range source;
- local file source;
- later Ollama blob locator.

Downloader has retry/backoff, ETag/revision checks, range validation and resume journal.

# 277. Phase 5 — Qwen-specific GGUF importer

Implement strict parser subset only after bounds validation.

Outputs:

- metadata map;
- tokenizer source data;
- tensor descriptors;
- source quant block iterator.

Fuzz now, before converter depends on it.

# 278. Phase 6 — `.tqf` v1 writer/reader

Implement superblock/section/extent/expert records from Part XIV.

Required APIs:

```rust
TqfWriter::create_partial(...)
TqfWriter::write_extent(...)
TqfWriter::commit(...)
TqfReader::open_validated(...)
TqfReader::tensor(role/layer)
TqfReader::expert(layer, expert)
```

Roundtrip synthetic fixtures and corruption tests.

# 279. Phase 7 — lossless Q4 repacker

Implement source Q4 decoder, TQF packing candidates and independent validation decoder.

Do not write Metal kernel assumptions into the generic validation decoder; independence helps catch shared bugs.

Produce per-tensor mismatch report containing first block/row/value.

# 280. Phase 8 — streaming conversion transaction

Integrate remote range reader and extent journal.

Simulate kill/resume continuously in tests. Conversion progress is reported as verified output bytes, not merely downloaded input bytes.

# 281. Phase 9 — tokenizer/chat semantics

Implement tokenizer using a mature Rust tokenizer dependency or direct format integration if it meets license/performance needs.

Golden fixtures:

- normal system/user/assistant;
- developer guidance;
- historical thinking if supported;
- tool definitions/results;
- vision placeholders;
- Unicode/byte fallback.

# 282. Phase 10 — Metal baseline infrastructure

Implement device/queue, buffer leases, pipeline cache, event timing, baseline metallib loading and optional runtime compilation.

Create synthetic bandwidth/GEMV executable mode under `tqf optimize`/developer harness rather than a second binary.

# 283. Phase 11 — reference Q4 kernels

Implement slow-clear kernels first:

- Q4 GEMV;
- Q4 batched GEMM/pre-fill primitive;
- RMSNorm;
- elementwise residual/SILU/sigmoid;
- simple LM-head path.

Every kernel has CPU/reference fixture and shape/alignment assertions.

# 284. Phase 12 — GDN

Implement in this order:

1. separate four projections;
2. conv tail;
3. q/k normalization;
4. FP32 recurrent update;
5. gated norm;
6. out projection;
7. fused input projection only after parity.

Store one recurrent-state object per GDN layer with exact reset/snapshot APIs.

# 285. Phase 13 — full attention

Start BF16 KV and full causal attention.

Implement virtual GQA mapping instead of physically duplicating two KV heads eight times. Validate partial 64-d RoPE explicitly.

# 286. Phase 14 — MoE resident correctness

For initial correctness, allow a high-memory development machine/profile to hold routed experts resident. Implement router/shared/routed computation before adding storage complexity.

This path remains a reference oracle after streaming exists.

# 287. Phase 15 — end-to-end decode

Connect embedding -> 40 layers -> norm/head -> sampler.

Required developer outputs:

- per-layer hash/tensor dump option;
- router trace option;
- greedy token log;
- stage timing framework even if not optimized.

Do not start aggressive cache work until 512-token reference sequence passes.

# 288. Phase 16 — real server

Wire normalized GenerateRequest to model session and streaming adapters. Tool-call parsing/formatting must be tested before `--open` integrations depend on it.

# 289. Phase 17 — canonical autoinstall

Combine setup/source/conversion/profile quick tune. A fresh user should now reach a working OpenAI server through plain `tqf` without knowing GGUF.

This is the first **useful build** milestone.

# 290. Phase 18 — out-of-core expert baseline

Mark resident core explicitly in format/runtime. Move routed expert pool to SSD. Implement whole-expert LFU cache first using exact plans/state machine.

Measure raw miss bytes/token before optimizing.

# 291. Phase 19 — parallel I/O and read-ahead

Implement demand queues, worker pool, in-flight dedup and benchmark selection among `pread`/read-ahead/Metal I/O candidates.

Reproduce NVMAI parallel-I/O result direction on TQF; if M4 differs, record it rather than forcing expected result.

# 292. Phase 20 — NVMAI-derived Metal optimization ports

Port/adapt one optimization at a time:

- MoE phase-1 16-row threadgroup staging;
- GDN four-way projection fusion;
- function-constant Qwen shape specialization;
- resident mapping/pinning strategy.

Each direct code adaptation includes Apache notice and a TQF A/B ledger entry.

# 293. Phase 21 — global cache broker

Replace per-layer fixed capacity with global cache entries.

Implement LRU/LFU/decayed cost-aware policies and trace replay. Do not add learned policy yet.

Exit evidence: same 4G memory produces less demand I/O/stall than fixed baseline on coding/prose traces.

# 294. Phase 22 — tiled experts

Turn already-present tile metadata into runtime cache units.

Start with 128-neuron metadata/layout. Compare whole/64/128/256/mixed. Measure syscall count and overread; partial caching that wins hit ratio but destroys I/O latency is rejected.

# 295. Phase 23 — predictive prefetch

Implement statistical predictor first. Inputs are recent route transitions/co-routing only. Add hidden-state predictor only after a no-model predictor baseline exists.

Log prefetch precision, recall, timeliness, and wasted bytes.

# 296. Phase 24 — hard 4G broker certification

Move every large allocation behind leases. Add OS sampler and stress scenarios. Fix hidden allocator spikes before claiming 4G.

The 4G profile is not released until helper-model swap and 128K context paths are also accounted later; this phase certifies the inference core first.

# 297. Phase 25 — M4 short-context assault

Use the optimization ledger. Focus current critical path from measured breakdown, not intuition.

Likely levers:

- expert miss bytes/latency;
- attention/GDN Q4 bandwidth;
- MoE phase kernels;
- head bandwidth;
- overlap/synchronization.

Keep optimizing beyond 15 tok/s.

# 298. Phase 26 — prefill

Implement chunk autotuning, expert-set dedup per chunk, and stage instrumentation. Include prompts larger than one chunk and repository-sized contexts.

# 299. Phase 27 — TQKV Q8/Q4 baseline

Implement page store/tail lifecycle, Q8 then Q4, and fused/blocked attention. BF16 full cache remains the oracle at smaller contexts.

# 300. Phase 28 — advanced TQKV

Run candidate matrix for Q3/Q2/rotation/outliers/pre-RoPE. No mixed-precision controller until individual encodings are qualified.

# 301. Phase 29 — 128K gate

Populate a real 128K context before timing decode. Run memory, quality and performance suites. This is the first production long-context milestone.

# 302. Phase 30 — prefix store

Implement page-content IDs, GDN snapshot blob, exact token prefix hash, LRU disk quota, crash-safe manifests. Demonstrate restart reuse.

# 303. Phase 31 — 256K/full-attention measurement

Measure full TQKV attention at 256K. If ≥15 tok/s floor is maintained with acceptable TTFT/memory, keep full evaluation default; otherwise proceed with TQAttn.

# 304. Phase 32 — TQAttn

Implement min/max page-bound reference selector, uncertainty expansion and full-attention A/B. Only after this baseline should self-indexing Key candidates be attempted.

# 305. Phase 33 — MTP

Implement official model semantics, accepted-token verification and bandwidth accounting. Controller default remains off until measured net positive on qualified workloads.

# 306. Phase 34 — 2G profile

Start by shrinking expert cache/staging while preserving correct output. Add context/helper swapping. 2G work cannot delay a stable 4G release indefinitely, but useful techniques feed back.

# 307. Phase 35 — file catalog/classifier

Implement scan/hash/classifier with misleading-path corpus immediately. Do not start semantic indexing before content IDs/classification are reliable.

# 308. Phase 36 — structural/lexical index

Build useful search without embeddings first. Exact symbol/path/BM25/graph baselines provide fallback while helper models are unavailable.

# 309. Phase 37 — pplx helper runtime

Implement helper `.tqf` conversion and transient broker lease. Validate embedding output against official/reference runtime before compact vectors.

# 310. Phase 38 — flat semantic baseline

Store full/INT8 reference vectors and exact SIMD search. This is the gold recall baseline; preserve benchmark results forever.

# 311. Phase 39 — TQVec

Implement A–F candidates from Part XIV with per-repo calibration. Choose only after recall/latency/index-size/update Pareto analysis.

# 312. Phase 40 — hybrid retrieval

Implement query intent lanes, RRF baseline, hard exact precedence and bounded graph expansion. Add retrieval provenance explanation objects now for GUI/debugging later.

# 313. Phase 41 — adaptive ANN research

Only now implement custom semantic partitions. Baselines are already available, preventing an unmeasured bespoke index.

Candidate development sequence:

1. static balanced semantic partitions;
2. repo-hierarchy overlay;
3. hot/cold partition residency;
4. local split/merge after edits;
5. workload-adaptive routing.

# 314. Phase 42 — live sync

Connect watcher to incremental generation transactions. Stress editor save storms and watcher overflow. Search remains usable during deferred semantic updates.

# 315. Phase 43 — GTE reranker

Implement transient cross-encoder and ambiguity heuristic. Benchmark downstream answer quality and TTFT, not reranker benchmark alone.

# 316. Phase 44 — automatic RAG + MCP

Build dynamic context budget and read-only MCP tools. Ensure retrieval is optional and server works normally without an index.

# 317. Phase 45 — client launchers

Implement OpenCode then Claude Code then Codex, each with real integration smoke tests and no permanent config mutation. Missing-client install path always confirms.

# 318. Phase 46 — SwiftUI bridge

Compile adopted UI into one binary, replace original app model with HTTP-backed TQF state, verify main-thread lifecycle and headless separation.

# 319. Phase 47 — TQF UI refinement

Create simple default conversation/setup experience and expandable engineering cockpit. The inspector consumes metrics; it must not change runtime policy directly except through supported configuration actions.

# 320. Phase 48 — vision

Repack/lazy-load vision artifact, protocol mapping, memory planner. Text-only startup and steady-state footprint must remain effectively unchanged when vision disabled.

# 321. Phase 49 — 1M research

Treat capacity and bandwidth separately. Combine validated TQKV/TQAttn/backing/prefix techniques; add novel methods only with full-attention/reference comparisons. Maintain 8G profile initially if needed.

# 322. Phase 50 — CUDA format/backend

Transcode logical Q4 values to CUDA layout. Implement resident core, streaming, Q4 kernels and context before tuning. Preserve the same high-level tests as Metal.

# 323. Phase 51 — RTX 3070 Ti qualification

Measure actual PCIe/SSD topology, tune pinned staging/cache, and establish 6GB-class minimum behavior. The 3070 Ti is mandatory, not an optional community report.

# 324. Phase 52 — release hardening

Run full fuzz/fault/memory/protocol/quality/performance/license/clean-machine suite. Freeze `.tqf`/`.tqi` major versions for the release and document migration guarantees.

# 325. Cross-phase rule: reference paths survive optimization

Do not delete every simple implementation as soon as a faster one appears. Keep compact reference paths for:

- Q4 decode/GEMV test;
- BF16/Q8 context at manageable lengths;
- full attention;
- flat vector search;
- LRU/LFU cache;
- MTP off;
- retrieval off.

Reference paths are how future black magic proves it did not quietly break the model.

# 326. Cross-phase rule: no hidden scope expansion

A phase may create helper utilities but cannot use its existence as justification for unrelated platform/model support. Examples:

- GGUF importer does not imply generic GGUF runtime.
- Tree-sitter parsers do not imply TQF becomes an IDE.
- MCP does not imply shell/edit tools.
- CUDA backend does not imply ROCm/Vulkan.
- embedding endpoint does not imply arbitrary embedding-model manager.

# 327. Cross-phase rule: every TODO is classified

Implementation TODOs should carry one class:

```text
CORRECTNESS
PERF
QUALITY
RESEARCH
PRODUCT
CLEANUP
```

A correctness TODO cannot be silently deferred behind performance work. Research TODOs do not block release unless tied to a locked acceptance gate.

# 328. Master v2 implementation handoff checklist

An implementation agent beginning work should be able to answer yes to all of these from this document:

- What exact model and shapes am I implementing? **Yes.**
- What is the first working server path? **Yes.**
- What is the `.tqf` header/table/extent baseline? **Yes.**
- How does conversion survive interruption? **Yes.**
- How are memory allocations owned/reclaimed? **Yes.**
- What is the exact decode/layer state machine? **Yes.**
- How are expert cache reservations protected from races? **Yes.**
- What is the first cache policy and how is it replaced? **Yes.**
- What TQKV format do I implement first? **Q8 then specified Q4.**
- What is the first TQAttn algorithm? **Min/max page bound + safe expansion.**
- How is prefix state keyed/persisted? **Yes.**
- What is the `.tqi` update model? **Log-structured committed generations.**
- How do I detect code if the path lies? **Content/parser-first classifier.**
- What semantic baselines/TQVec candidates exist? **Yes.**
- How does hybrid fusion work initially? **RRF + structural precedence.**
- Which API fields/protocols are supported? **Baseline defined.**
- How is SwiftUI kept in the same binary? **Build/ABI/lifecycle defined.**
- How do I prove 4G/15tok/s/≤1%? **Qualification protocols defined.**

Where this document still offers alternatives, it explicitly names the reference baseline and the benchmark that chooses the replacement. That is intentional: hardware research cannot be honestly specified as a fake certainty before measurement.

**PART A**

**Appendices and Reference Material**

Derived calculations, protocol examples, source bibliography, and practical implementation checklists.

# A1. Derived bandwidth and context calculator

| **Quantity**                   | **Formula**          | **Result**          |
|--------------------------------|----------------------|---------------------|
| Routed expert weights          | 3 × 2048 × 512       | 3,145,728 weights   |
| Raw Q4/expert                  | weights × 4/8        | 1.500 MiB           |
| Expert selections/token        | 8 × 40               | 320                 |
| Raw routed Q4/token            | 1.5 MiB × 320        | 480 MiB             |
| NVMAI-like 1.55MiB slots/token | 1.55 × 320           | 496 MiB             |
| At 15 tok/s, 100% miss         | 496 MiB × 15         | ~7.27 GiB/s         |
| At 30 tok/s, 100% miss         | 496 MiB × 30         | ~14.53 GiB/s        |
| Full KV BF16/token             | 10×K/V×2heads×256×2B | 20 KiB              |
| 128K BF16 full KV              | 20KiB × 131072       | 2.50 GiB            |
| 256K BF16 full KV              | 20KiB × 262144       | 5.00 GiB            |
| 1.01M BF16 full KV             | 20KiB × 1,010,000    | ~19.26 GiB          |
| GDN recurrent/layer            | 32×128×128×4B        | 2.00 MiB            |
| GDN recurrent all              | 2MiB × 30            | ~60 MiB + conv tail |

All rows are derived planning calculations from the official Qwen configuration and Transformers shapes unless attributed otherwise. Exact converted storage differs because quant metadata/alignment/layout add overhead; the runtime must generate exact byte inventories from the canonical \`.tqf\` file.

# A2. Memory broker invariants

- Every persistent/transient allocation above a small threshold must have a MemoryOwner tag.

- The broker maintains separate “logical reserved” and “physically committed” counters where a backend has lazy allocation semantics.

- A lease cannot outlive the resource owner without explicit transfer.

- Elastic memory must expose a reclaim callback or eviction interface.

- Context correctness state is protected unless the context subsystem itself has a validated lower-precision representation to transition into.

- Server/GUI/driver reserve is budgeted explicitly; memory claims are tested using OS-observed footprint in addition to internal counters.

- On NVIDIA, host+pinned+device bytes are combined for the user-facing budget.

# A3. Example API requests

**OpenAI Chat Completions compatibility example**


```text
curl http://127.0.0.1:11434/v1/chat/completions \

-H "Content-Type: application/json" \

-d '{

"model":"qwen3.6-35b-a3b",

"messages":[{"role":"user","content":"Explain the cache planner."}],

"stream":true

}'
```


**OpenAI Responses compatibility example**


```text
curl http://127.0.0.1:11434/v1/responses \

-H "Content-Type: application/json" \

-d '{

"model":"qwen3.6-35b-a3b",

"input":"Summarize the indexed project architecture.",

"stream":true

}'
```


**Ollama compatibility example**


```text
curl http://127.0.0.1:11434/api/chat \

-d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}]}'
```


# A4. Suggested internal metrics names

| **Metric**                        | **Type**  |
|-----------------------------------|-----------|
| tqf_decode_tokens_per_second      | gauge     |
| tqf_prefill_tokens_per_second     | gauge     |
| tqf_ttft_seconds                  | histogram |
| tqf_memory_bytes{owner}           | gauge     |
| tqf_expert_bytes_read_total       | counter   |
| tqf_expert_cache_byte_hit_ratio   | gauge     |
| tqf_expert_prefetch_precision     | gauge     |
| tqf_expert_prefetch_on_time_ratio | gauge     |
| tqf_gpu_stage_seconds{stage}      | histogram |
| tqf_io_wait_seconds               | histogram |
| tqf_tqkv_bytes                    | gauge     |
| tqf_tqkv_effective_bits           | gauge     |
| tqf_tqattn_pages_considered       | counter   |
| tqf_tqattn_pages_selected         | counter   |
| tqf_prefix_tokens_reused_total    | counter   |
| tqf_index_query_seconds{lane}     | histogram |
| tqf_index_candidates{lane}        | histogram |
| tqf_reranker_invocations_total    | counter   |

# A5. Research/source bibliography

| **ID / Source**                                               | **Use in TQF research**                                                               | **Link**                                                                                                                                                                                                                                      |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R1 — Qwen/Qwen3.6-35B-A3B model card                          | Official model architecture, context, MTP, text-only serving, coding positioning.     | [<u>https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md</u>](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)                                                                                                     |
| R2 — Qwen3.6 config.json                                      | Exact hidden size, layer types, GDN heads, FP32 state dtype, expert counts, vocab.    | [<u>https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json</u>](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)                                                                                                 |
| R3 — Hugging Face Transformers Qwen3.5/3.6 MoE implementation | Projection/state shapes and reference recurrence.                                     | [<u>https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py</u>](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) |
| R4 — ggml-org/Qwen3.6-35B-A3B-GGUF                            | Canonical GGUF artifacts, Q4/MTP/mmproj sizes and Q4 SHA-256.                         | [<u>https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF</u>](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF)                                                                                                                           |
| R5 — perplexity-ai/pplx-embed-v1-0.6b                         | 1024-d, 32K, MRL, native INT8/binary embedding model.                                 | [<u>https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b</u>](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)                                                                                                                     |
| R6 — Alibaba-NLP/gte-reranker-modernbert-base                 | 149M reranker, 8192 context, CoIR results, Apache-2.0.                                | [<u>https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base</u>](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)                                                                                                     |
| R7 — drumih/turbo-fieldfare                                   | Bounded-memory SSD expert streaming, streaming installer, Metal/Swift design.         | [<u>https://github.com/drumih/turbo-fieldfare</u>](https://github.com/drumih/turbo-fieldfare)                                                                                                                                                 |
| R8 — Pummelchen/NVMAI                                         | Qwen3.6-focused TurboFieldfare fork and benchmark/optimization history.               | [<u>https://github.com/Pummelchen/NVMAI</u>](https://github.com/Pummelchen/NVMAI)                                                                                                                                                             |
| R9 — NVMAI parallel expert I/O commit                         | Measured parallel pread improvement.                                                  | [<u>https://github.com/Pummelchen/NVMAI/commit/4beb74f4a28de6d4a3222d079dc5306cbd7a32c0</u>](https://github.com/Pummelchen/NVMAI/commit/4beb74f4a28de6d4a3222d079dc5306cbd7a32c0)                                                             |
| R10 — NVMAI cache/pinning commit                              | 64-slot + pinning measurements and memory-pressure result.                            | [<u>https://github.com/Pummelchen/NVMAI/commit/069aed6394777216a06a252e5d2d47a063e37ab1</u>](https://github.com/Pummelchen/NVMAI/commit/069aed6394777216a06a252e5d2d47a063e37ab1)                                                             |
| R11 — NVMAI Q4 MoE phase-1 commit                             | Threadgroup-staged activation and stage-time reduction.                               | [<u>https://github.com/Pummelchen/NVMAI/commit/5a7902baa9cec83eed1372e1e0fec58228357f7c</u>](https://github.com/Pummelchen/NVMAI/commit/5a7902baa9cec83eed1372e1e0fec58228357f7c)                                                             |
| R12 — NVMAI 4096-token prefill commit                         | MoE-aware prefill chunk measurement.                                                  | [<u>https://github.com/Pummelchen/NVMAI/commit/4ea208d9b563523103f7fea59998f368337116c2</u>](https://github.com/Pummelchen/NVMAI/commit/4ea208d9b563523103f7fea59998f368337116c2)                                                             |
| R13 — NVMAI fused GDN projection commit                       | Fused four-way GDN Q4 input projection.                                               | [<u>https://github.com/Pummelchen/NVMAI/commit/159ff74825115ceb82f5904d0587db1ec2e82e5d</u>](https://github.com/Pummelchen/NVMAI/commit/159ff74825115ceb82f5904d0587db1ec2e82e5d)                                                             |
| R14 — NVMAI persistent multi-prefix state commit              | KV + GDN snapshot implementation precedent.                                           | [<u>https://github.com/Pummelchen/NVMAI/commit/2ddf68e48ea29ef60a082abba309b37ef6a64506</u>](https://github.com/Pummelchen/NVMAI/commit/2ddf68e48ea29ef60a082abba309b37ef6a64506)                                                             |
| R15 — NVMAI CPU MTP draft feasibility commit                  | Unified-memory bandwidth negative result.                                             | [<u>https://github.com/Pummelchen/NVMAI/commit/2c3c7b8ccd8537f4d2d26ce03c66f304b1689012</u>](https://github.com/Pummelchen/NVMAI/commit/2c3c7b8ccd8537f4d2d26ce03c66f304b1689012)                                                             |
| R16 — NVMAI F_RDADVISE measurement                            | Host-dependent read-ahead benefit.                                                    | [<u>https://github.com/Pummelchen/NVMAI/commit/7cc8b5ea98fc788b87fea83941b8181196d521f5</u>](https://github.com/Pummelchen/NVMAI/commit/7cc8b5ea98fc788b87fea83941b8181196d521f5)                                                             |
| R17 — NVMAI production hardening commit                       | Trusted receipts, schema/path/server/GPU failure hardening.                           | [<u>https://github.com/Pummelchen/NVMAI/commit/19aafd8fe2d99ca2e761c785b4a44f6bf119a79a</u>](https://github.com/Pummelchen/NVMAI/commit/19aafd8fe2d99ca2e761c785b4a44f6bf119a79a)                                                             |
| R18 — TurboQuant                                              | Online vector quantization and KV/nearest-neighbor experiments.                       | [<u>https://arxiv.org/abs/2504.19874</u>](https://arxiv.org/abs/2504.19874)                                                                                                                                                                   |
| R19 — KVQuant                                                 | Low-bit K/V quantization, pre-RoPE keys, outliers.                                    | [<u>https://arxiv.org/abs/2401.18079</u>](https://arxiv.org/abs/2401.18079)                                                                                                                                                                   |
| R20 — KIVI                                                    | Tuning-free asymmetric 2-bit KV quantization.                                         | [<u>https://arxiv.org/abs/2402.02750</u>](https://arxiv.org/abs/2402.02750)                                                                                                                                                                   |
| R21 — Quest                                                   | Query-aware KV page sparsity for long-context attention.                              | [<u>https://arxiv.org/abs/2406.10774</u>](https://arxiv.org/abs/2406.10774)                                                                                                                                                                   |
| R22 — Self-Indexing KVCache                                   | Compressed keys as sparse-attention index.                                            | [<u>https://arxiv.org/abs/2603.14224</u>](https://arxiv.org/abs/2603.14224)                                                                                                                                                                   |
| R23 — FlashMoE                                                | SSD-offloaded MoE and adaptive cache-replacement research.                            | [<u>https://arxiv.org/abs/2601.17063</u>](https://arxiv.org/abs/2601.17063)                                                                                                                                                                   |
| R24 — MoEpic                                                  | Adaptive partial-expert split/cache/prefetch research.                                | [<u>https://arxiv.org/abs/2509.08342</u>](https://arxiv.org/abs/2509.08342)                                                                                                                                                                   |
| R25 — Quake                                                   | Workload-adaptive hierarchical vector indexing.                                       | [<u>https://arxiv.org/abs/2506.03437</u>](https://arxiv.org/abs/2506.03437)                                                                                                                                                                   |
| R26 — SPFresh                                                 | Localized in-place repair for dynamic vector indexes.                                 | [<u>https://arxiv.org/abs/2410.14452</u>](https://arxiv.org/abs/2410.14452)                                                                                                                                                                   |
| R27 — CoIR                                                    | Comprehensive code information retrieval benchmark.                                   | [<u>https://arxiv.org/abs/2407.02883</u>](https://arxiv.org/abs/2407.02883)                                                                                                                                                                   |
| R28 — RepoBench                                               | Repository-level retrieval/completion benchmark.                                      | [<u>https://arxiv.org/abs/2306.03091</u>](https://arxiv.org/abs/2306.03091)                                                                                                                                                                   |
| R29 — Apple Metal resource loading / WWDC22                   | Metal I/O queues and storage-to-resource loading.                                     | [<u>https://developer.apple.com/documentation/metal/resource-loading</u>](https://developer.apple.com/documentation/metal/resource-loading)                                                                                                   |
| R30 — NVIDIA GPUDirect Storage Design Guide                   | Explicit GDS path, pinned buffers, supported GPU suitability guidance.                | [<u>https://docs.nvidia.com/gpudirect-storage/design-guide/index.html</u>](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html)                                                                                                 |
| R31 — Ollama API documentation                                | localhost:11434 API, local no-auth behavior, compatibility surface.                   | [<u>https://docs.ollama.com/api/introduction</u>](https://docs.ollama.com/api/introduction)                                                                                                                                                   |
| R32 — OpenAI API reference                                    | Responses, Chat Completions, Embeddings, streaming interfaces used for compatibility. | [<u>https://developers.openai.com/api/reference</u>](https://developers.openai.com/api/reference)                                                                                                                                             |
| R33 — OpenCode provider documentation                         | Custom OpenAI-compatible provider baseURL/model configuration.                        | [<u>https://opencode.ai/v2/docs/providers</u>](https://opencode.ai/v2/docs/providers)                                                                                                                                                         |
| R34 — Anthropic Claude Code LLM gateway documentation         | ANTHROPIC_BASE_URL gateway configuration.                                             | [<u>https://docs.anthropic.com/en/docs/claude-code/llm-gateway</u>](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)                                                                                                               |
| R35 — OpenAI Codex model-provider registry                    | Custom base_url and Responses wire API in current Codex source.                       | [<u>https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs</u>](https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs)                                                                 |

# A6. Licensing/attribution checklist

- Project LICENSE: Apache License 2.0.

- NOTICE: include required notices from directly adapted Apache works.

- THIRD_PARTY_NOTICES.md: list model/helper licenses and source dependencies; model weights are not bundled in TQF release artifacts.

- Directly modified TurboFieldfare/NVMAI files carry prominent modification notices where required by Apache-2.0 section 4.

- GUI About → Open Source Licenses and \`tqf licenses\` expose notices to binary users.

- Pin exact dependency versions before release and collect their authoritative license/NOTICE files rather than relying on a manually written summary.

# A7. Final definition of done

TurboQwenFare is “done” only in the sense of a stable release when a fresh supported machine can receive one binary, install a pinned Qwen3.6 Q4 model automatically, serve existing OpenAI/Anthropic/Ollama-compatible clients, remain within its declared live memory budget, sustain the qualified performance floor on realistic workloads, provide 128K logical context by default with ≤1% measured quality loss, and expose optional retrieval with \`tqf sync .\` without becoming a coding harness. The optimization program continues after that release because 15 tok/s is explicitly a floor, not the project’s ambition.


> **Final principle:** TurboQwenFare should feel boringly simple from the outside and borderline unreasonable from the inside. The user gets `tqf`. The machine gets a model-specific Q4 execution appliance, SSD-streamed MoE, virtualized memory, huge-context compression/indexing, self-tuning scheduling, transient retrieval models, first-party hybrid search, and native hardware kernels.




---

**— End of TurboQwenFare Master v2 —**
