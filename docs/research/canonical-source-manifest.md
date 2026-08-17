# Canonical source manifest

Spec Phase 0 deliverable ("research harvest and canonical manifest," §272):
pin the official Qwen3.6 source revisions and record their hashes/licenses
before any format work depends on them. This file is that record — the
frozen counterpart to the live constants in `src/source/pinned.rs`.

**Note on ordering:** this manifest was produced *after* Phases 4-6
(source downloader, GGUF importer, `.tqf` container) had already landed,
not before, which is out of the order the spec recommends ("Do not proceed
to format design with unresolved text-tensor names/shapes," §272). The gap
is now closed: the canonical conversion log below now supplies the actual
language-GGUF tensor names and shapes used by `src/dev/inventory.rs`.

## Base model

- **Repository:** [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Architecture:** MoE with hybrid Gated-DeltaNet/full-attention (`qwen3_5_moe`
  in the Transformers architecture registry — the model class predates the
  3.6 version bump and was never renamed).
- **License:** `apache-2.0`
- **Modalities:** text/image/video, 262,144-token native context (extensible
  to ~1,010,000 per the model card; TQF's own 128K→~1M targets are
  independent of that upstream claim and still need TQF's own qualification,
  spec §6).

### Geometry cross-check (2026-08-11)

Every field in the spec's §117 "Canonical Qwen3.6 geometry" table (marked
**LOCKED**) was fetched live from `Qwen/Qwen3.6-35B-A3B/raw/main/config.json`
and compared. All matched exactly — no drift between the spec and the live
source as of this date:

| Field | Spec §117 | Live `config.json` |
|---|---|---|
| hidden_size | 2048 | 2048 |
| num_hidden_layers | 40 | 40 |
| full_attention_interval | every 4th layer | 4 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 2 | 2 |
| head_dim | 256 | 256 |
| partial_rotary_factor | 0.25 | 0.25 |
| linear_num_key_heads (GDN key heads) | 16 | 16 |
| linear_num_value_heads (GDN value heads) | 32 | 32 |
| linear_key_head_dim | 128 | 128 |
| linear_value_head_dim | 128 | 128 |
| linear_conv_kernel_dim | 4 | 4 |
| num_experts | 256 | 256 |
| num_experts_per_tok | 8 | 8 |
| moe_intermediate_size | 512 | 512 |
| shared_expert_intermediate_size | 512 | 512 |
| vocab_size | 248,320 | 248,320 |
| max_position_embeddings | 262,144 | 262,144 |

Additional fields observed in the live config not called out by the spec
table (recorded here for future phases, not yet consumed by any TQF code):
`rope_theta = 10000000`, `mrope_interleaved = true`, `mrope_section = [11, 11, 10]`
(multimodal RoPE sectioning — relevant once `--enable-vision` work begins,
spec Part XI).

This cross-check satisfies the Phase 0 test requirement "official config
fields equal compile-time `Qwen36Geometry` constants" — see
`src/model/qwen36/geometry.rs`, which encodes this same table as Rust
constants and is unit-tested against it.

## GGUF conversion (the actual TQF download source)

- **Repository:** [`ggml-org/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF)
- **Pinned commit:** `baec3ebee244827cda0f4557eafa8b28f7545fa6`
  - Fetched via `https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF`
    (`sha` field) on 2026-08-11, cross-checked with a second independent
    fetch of the same endpoint — both returned the identical 40-character
    hex value.
  - Repository `lastModified` at fetch time: `2026-07-16T18:27:09.000Z`.
- **License:** `apache-2.0` (inherited from the base model; GGUF conversion
  adds no separate license terms).
- **Conversion tooling:** auto-converted via `ggml-org/convert` per the
  repository's own description.

Per spec §13's pinned-source rule, `src/source/pinned.rs::REVISION` holds
this exact commit — never `"main"` — so a later `ggml-org` push can't
silently change model bytes under an existing benchmark/correctness
profile. Re-resolving this pin (e.g. for `tqf update`) means repeating the
fetch above and re-running the cross-checks, not just editing the constant.

### Pinned artifacts

Fetched via `https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF?blobs=true`
on 2026-08-11 (`siblings[].lfs.sha256`/`size`):

| Role | Filename | Size (bytes) | SHA-256 | Confidence |
|---|---|---:|---|---|
| Language checkpoint (Q4_K_M) | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | 20,419,565,568 | `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` | **High** — matches the value already published in spec §13 verbatim; three independent sources agree (spec text, HF API fetch #1, HF API fetch #2). |
| MTP draft (Q4_0) | `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` | 1,060,038,432 | `606fca331adcbfbdadc107512ce6a7161e84e1646ba0e0018256426f6296877f` | Medium — single live fetch, not independently re-verified. The downloader (`src/source/hf.rs`) still does its own whole-file SHA-256 check against this value at fetch time regardless, so a wrong digit here fails loudly (checksum mismatch) rather than silently. |
| Vision projector (Q8_0, mmproj) | `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` | 614,194,304 | `904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d` | Medium — same caveat as MTP above. |

The repository also carries `BF16` and `DFLASH` variants of each artifact
(full-precision and speculative-decode-oriented builds) that TQF does not
target — the canonical path per spec §13 is Q4_K_M language / Q4_0 MTP /
Q8_0 vision only.

### Canonical language tensor naming (verified 2026-08-12)

The upstream [`convert.log`](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/blob/main/convert.log)
was fetched directly and inspected without downloading the 20.4 GB GGUF.
Its language conversion lists 733 tensors. The names that resolve the
previous GDN/MoE ambiguity are:

- GDN layers: `blk.N.attn_qkv.weight`, `blk.N.attn_gate.weight`,
  `blk.N.ssm_a`, `blk.N.ssm_alpha.weight`, `blk.N.ssm_beta.weight`,
  `blk.N.ssm_conv1d.weight`, `blk.N.ssm_dt.bias`, `blk.N.ssm_norm.weight`,
  and `blk.N.ssm_out.weight`.
- Full-attention layers: `blk.N.attn_q`, `attn_k`, `attn_v`,
  `attn_output`, `attn_q_norm`, and `attn_k_norm`.
- MoE: `ffn_gate_inp` is the router; routed experts are
  `ffn_{gate,up,down}_exps`; shared-expert tensors are
  `ffn_gate_inp_shexp`, `ffn_gate_shexp`, `ffn_up_shexp`, and
  `ffn_down_shexp`.

The conversion also records the critical orientation/shape facts that are
not safe to infer: a GDN `attn_qkv` is `{2048,8192}`, `attn_gate` is
`{2048,4096}`, and the full-attention layer at block 3 has q/k/v shapes
`{2048,8192}`, `{2048,512}`, `{2048,512}` respectively. The inventory
recognizes these canonical names and unit-tests them; a future loader still
must validate these dimensions from the real downloaded descriptor before
kernel construction.

The companion [Transformers reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
resolves the GDN execution semantics: `attn_qkv` is split into Q/K/V,
`attn_gate` produces the SiLU output gate, `ssm_alpha` and `ssm_beta` each
produce 32 value-head scalars, and Q/K use per-head L2 normalization. The
official llama.cpp GGUF converter has already transformed `ssm_a` to
`-exp(source A_log)`, folded standard RMSNorm tensors to
`1 + source_weight`, and reordered GDN value heads from grouped to tiled
order. TQF therefore consumes those stored values directly: recurrent decay
is `exp(ssm_a * softplus(alpha + ssm_dt))`, standard RMSNorm multiplies by
the stored scale without adding another one, and value head `h` broadcasts
from key head `h % 16`. Numeric parity still requires the downloaded
checkpoint and token-level qualification.

### Q4_K_M tensor types (verified 2026-08-12 from the GGUF header)

Reading only the first 64 MiB of the official 20.4 GiB Q4_K_M file was enough
to parse all 733 descriptors. The language file is deliberately mixed-precision:

| GGML type | Count | Confirmed use |
|---|---:|---|
| `F32` | 301 | norms, router gates, GDN scalar/conv parameters |
| `Q8_0` | 310 | GDN and full-attention projections |
| `Q4_K` | 121 | token embedding and routed-expert matrices |
| `Q6_K` | 1 | `output.weight` LM head |

The runtime therefore must not use a Q4_K-only matvec dispatch. In particular,
the Q6_K output head is required before real greedy token generation can be
qualified.

### Phase 13–18 implementation boundary (2026-08-12)

The current reference graph binds these verified shapes and storage types into
BF16 virtual-GQA attention, Gated DeltaNet, exact top-8 MoE routing, Q6_K LM
head greedy decode, and the normalized OpenAI generator boundary. It is a
deliberately high-memory **resident-expert** correctness profile: the router,
shared expert, and all selected expert computations use the canonical weights
rather than synthetic callbacks. It emits per-layer hashes, router traces,
greedy tokens, and stage timings.

Conversion now writes every routed Q4_K expert as a checksum-backed
whole-expert superextent, and the runtime has an exact-router global LFU
reference cache with broker-first miss allocation and raw miss-byte accounting.
Normal installed-model startup selects the bounded graph: embeddings, LM head,
norms, and layer projections are leased only at their execution boundary;
GDN/attention state and the global expert cache persist. The developer-only
`TQF_DEV_STREAMING_REFERENCE=1` and `TQF_DEV_RESIDENT_REFERENCE=1` switches
remain for qualification and parity.

This is implementation evidence, not by itself a completed Phase 17 claim. The
real-checkpoint evidence recorded below closes the first-token and headless
server portions, while plain GUI startup and the full locked qualification
matrix remain separate gates.

### Independent canonical-token oracle (2026-08-16)

The pinned language GGUF above was downloaded through TQF's resumable source
transaction, passed its full-file SHA-256 gate, and was then evaluated with an
independent llama.cpp checkout pinned at
`4df29be4f4c3673f428170fda944a5b19f743bb8`. The reference command used CPU
execution, BF16 K/V, an eight-token context, disabled CPU weight repacking, and
saved the complete final-logit vector for the single-token prompt `A`.

- Prompt bytes: `A`
- Reference prompt token: `32`
- Greedy next token: `220`
- Winning logit: `11.4693` (runner-up token `13`, logit `10.7795`)
- Saved binary-logit SHA-256:
  `1f03a78e9a66ac317d1186271e967a05c652abf66388cb7bf771c5f1a807bed4`

The reference checkout has one loader-only local change: eager whole-file mmap
prefetch is disabled because `POSIX_MADV_WILLNEED` over a 20 GB USB-hosted
checkpoint blocked and thrashed a 16 GB machine. This does not change model
bytes, tokenizer behavior, graph construction, tensor kernels, or logits math.
llama.cpp remains a research/qualification oracle only; it is not copied,
linked, shipped, or added as a TQF dependency. Token `220` is the external
target for TQF's real-checkpoint single-token diagnostic. It does not replace
the Phase 15 exit gate's full 512-token greedy reference sequence.

### Canonical TQF installation and first-token parity (2026-08-16)

The same pinned source was converted through TQF's resumable, atomic
transaction and then reopened through the production topology validator before
the setup flow wrote a trusted receipt. The installed container is
20,409,912,064 bytes; its receipt records conversion fingerprint
`9a5bea2e7c4b2aa8d5faeaa6f1b71744f154f065bc066176268ddff9cbef989a` and
metadata root `14d108212b64b20f70fba209cbe642564d76c8562fb4035cb03cd83474b9fb34`.
The canonical inventory check accepted all 733 source tensor descriptors, and
the installed `.tqf` passed the complete fixed-graph role/shape validator.

Real-checkpoint execution exposed and fixed two storage-semantics mistakes
that synthetic fixtures had not exercised: `ssm_conv1d.weight` is rank-two
GGUF storage `{4, 8192}` consumed as channel-major depthwise weights, while
`ffn_gate_inp_shexp` is a rank-one `{2048}` vector whose result is a scalar dot
product, not a one-row matrix. Both paths now retain broker-before-allocation
accounting and have focused regression tests.

With a 4 GiB `MemoryBroker`, the release-only TQF diagnostic encoded `A` as
token `32` and produced greedy token `220` (`" "`), exactly matching the pinned
llama.cpp oracle above. Its first-token whole-expert-cache counters were 320
misses, 169 evictions, 267,190,272 resident bytes, and 566,231,040 raw miss
bytes. These are diagnostic counters, not a throughput result.

On 2026-08-17 the same pinned prompt was extended to the mandatory 16-token
length using `docs/research/oracles/raw-a-16.json`. TQF reproduced all 16
greedy llama.cpp token IDs exactly. Real layerwise comparison was required to
reach parity: GGML Q8_0/Q4_K/Q6_K matvecs quantize their activation operand to
Q8_0/Q8_K before the integer dot, and GDN Q/K L2 normalization is
`1 / max(sqrt(sum_squares), epsilon)`, not
`1 / sqrt(sum_squares + epsilon)`. These are checkpoint execution semantics,
not a token-specific correction.

The 16-token release run took 374,033 ms and recorded 5,120 expert misses,
4,969 evictions, zero hits, 267,190,272 resident expert bytes, and
9,059,696,640 raw miss bytes. This closes the raw-`A` 16-token correctness
fixture while demonstrating that the current reference cache is nowhere near
the Phase 19–25 performance target.

The same exact path passed the pinned raw-`A` 128-token oracle on 2026-08-17.
It took 1,244,716 ms and recorded 40,960 misses, 40,809 evictions, zero hits,
267,190,272 resident expert bytes, and 72,477,573,120 raw miss bytes. The
immutable result record is `docs/research/qualification/raw-a-128-tqf.json`.

The release setup path then validated the installed model, wrote the receipt,
completed its short hardware tune on the base Apple M4, and started the bounded
headless server at `127.0.0.1:11434`. `/health` reported version `0.0.1` with
`model_installed=true`; `/v1/models` reported installed model
`qwen3.6-35b-a3b`. A subsequent release invocation with only `tqf --headless`
reused the receipt and passed both probes again. The server was stopped cleanly
after each check.

Qualification boundary: the real bounded graph now closes the raw-`A` 1-,
16-, and 128-token deterministic lengths. It does not close Phase 15's broader
workload matrix or its 512-token length, the 512-token cache-ordering
gate, OS-observed 4 GiB qualification, the >=15 tok/s floor, the combined <=1%
quality gate, plain GUI startup, or RTX 3070 Ti/CUDA qualification. No claim in
this document should be read as closing those gates.

### `.src_sha` (upstream provenance, informational)

The GGUF repo carries a `.src_sha` file recording which upstream commit(s)
of the base model were converted:

```
DFLASH=f181eece646affea2c38b2765f1aaa01a9734ccd
PRIMARY=995ad96eacd98c81ed38be0c5b274b04031597b0
```

`PRIMARY` is the best candidate for the "source model repository ID;
immutable revision/commit hash" fields of the §125 model-provenance record
once Phase 7/8 actually populate one — not yet consumed by any TQF code.
Not independently re-verified beyond the single fetch that produced it.

## How to re-verify

```
curl -s https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF | jq .sha
curl -s 'https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF?blobs=true' \
  | jq '.siblings[] | select(.rfilename | test("Q4_K_M|mtp-.*Q4_0|mmproj.*Q8_0")) | {rfilename, size: .lfs.size, sha256: .lfs.sha256}'
```

If any value differs from this file, `src/source/pinned.rs` needs a
deliberate update (and a note here about when/why the pin moved) — not a
silent edit, per the pinned-source rule.
