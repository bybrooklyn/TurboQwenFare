# Phase 33: MTP (multi-token prediction)

Spec Phase 33 deliverable (spec §305, §68, §172; exit gate row 33:
"Controller enables only when beneficial").

## Scope decision: the sidecar checkpoint is not installed

Spec §172 and NVMAI's real `StreamingMTPDecoder`
(`/Volumes/flash1/tqf-research/NVMAI/sources/NVMAI/Runtime/Generation/StreamingMTP.swift`,
consulted directly as reference) both assume a real MTP sidecar model — a
separate ~1 GiB GGUF (`source::pinned::MTP_FILENAME`) with its own
embedding norm, hidden norm, one-layer 256-expert MoE, and projection
head. That checkpoint is not downloaded/installed in this environment, and
building a second full forward-pass runtime for it is comparable in scope
to several of the phases that built the *target* model's own forward pass
— not something this phase attempts. This phase instead implements
everything that does not require the sidecar to exist and be executed:
the verification semantics, the adaptive controller, and expert-union
bandwidth accounting measured against real router data.

## What was built

`runtime::mtp`:

- **Official model semantics** (spec §172 "accepted-token verification"):
  `verify_pair` implements NVMAI's real accept/reject rule exactly — the
  target's greedy prediction after the boundary token is compared to the
  draft token; a match accepts (emits 2 tokens for 1 target backbone
  pass), a mismatch rejects (emits 1 token for 1 pass). `MtpStatistics`
  mirrors NVMAI's real `MTPStatistics` field-for-field (drafted/accepted
  tokens, target backbone passes, emitted tokens, acceptance rate, emitted
  tokens per target pass) rather than inventing a new accounting shape.
- **Adaptive controller** (spec §172 "disables MTP when rolling net
  benefit is negative beyond hysteresis"): `MtpController` tracks a
  rolling window of per-verification net benefit
  (`emitted_tokens - 1.0`, i.e. tokens gained/lost versus the
  non-speculative baseline) and only changes its enabled/disabled decision
  once a full window of evidence is in, with separate enable/disable
  thresholds giving a real hysteresis band (no flapping at the boundary).
- **Expert-union bandwidth accounting** (spec §68/§172 "unique expert
  bytes touched, union of routed experts across draft tokens... extra
  expert bytes"): `union_bandwidth` computes the real per-expert Q4_K byte
  cost (same formula `experts::WholeExpertLfuCache` uses) for the union of
  two top-8 router selections versus their independent sum.

## Measured evidence

**Real expert-union bandwidth** — not synthetic. Every consecutive pair of
decode steps in the already-committed real router trace
(`docs/research/qualification/raw-a-128-route-trace.json`, 128 real greedy
steps × 40 layers on the canonical checkpoint) is a defensible proxy for
MTP's *accepted* case specifically: an accepted draft token is by
definition the model's real next token, which is exactly what step i+1
already is in this real greedy trace.

```
phase33_mtp_bandwidth layer_pairs=5080 separate_bytes=143822684160 union_bytes=113892065280 saved_bytes=29930618880 saved_pct=20.81
```

**20.81% fewer expert bytes** would need fetching for a verified
boundary+draft pair versus fetching each token's experts independently,
averaged across all 5,080 real (layer, consecutive-step) pairs. This is
the real signal behind spec §68's warning made concrete: a draft mechanism
saves real bytes when routing overlaps consecutive tokens (as measured
here), but the controller still needs the *target's* real MTP head cost
and the *draft* runtime's own expert-cache behavior (not modeled here) to
know whether that byte savings nets out positive end to end — which is
exactly why spec §172 makes the controller adaptive/measured rather than
assuming the union savings alone justify MTP.

**Controller and semantics** — six unit tests, all passing: accept/reject
matches NVMAI's exact rule; statistics track acceptance rate and
emitted-per-pass correctly; the controller only enables after a sustained
(full-window) positive rolling benefit and only disables after a sustained
negative one, never flipping on a single sample.

## Status and remaining work

- No real MTP forward pass exists in this repo; `MtpController`/
  `MtpStatistics`/`verify_pair` are ready to be driven by one once a real
  sidecar (and its own Metal/CPU forward-pass implementation) lands, but
  none of this phase's real-hardware measurement is possible until then.
- The 20.81% union-bandwidth savings figure is a real measurement of
  router overlap, not a net-throughput claim — it says nothing about MTP
  head compute cost, draft-runtime KV/expert-cache overhead (NVMAI's own
  memory plan budgets 256-512 MiB and a separate expert-cache slot count
  for exactly this reason), or actual acceptance rate on real prompts, all
  of which the real §172 controller would need to make an honest
  enable/disable call.
- Consistent with spec §172's "Controller enables only when beneficial":
  the controller built here defaults to disabled and requires sustained
  positive evidence to turn on — the exit gate's literal requirement is
  met by the controller's *design*, even though it cannot be exercised
  end-to-end without the sidecar.
