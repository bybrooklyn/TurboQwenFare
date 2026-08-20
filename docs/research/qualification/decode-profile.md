# Decode profiling

## Why

Spec §4 sets a 15 tok/s floor. Phase 25 measured 2.34 s/token on real
hardware and found 78% of it was demand expert I/O. Every number behind
that finding already existed in the code — `DecodeTimings` per step, and
the expert cache's own hit/miss/byte/stall counters — but nothing added
them up. Answering "where does the time go" meant re-deriving it from a
qualification document, which is a poor instrument for the iteration this
floor is going to need.

`runtime::profile::DecodeProfile` accumulates what is already measured.
It deliberately introduces no new measurement path: a profiler that
samples differently from the thing being optimized is how a 10x win on a
stage worth 3% of the total gets celebrated.

## What it reports

Per generation, once, at the end:

- **s/token and tok/s**, against the §4 floor by name.
- **Stage totals** — embedding, layers, final norm, LM head, sampling.
- **Unaccounted time**: the step's wall clock minus every stage timer.
  Reported rather than absorbed, because it is the part no timer covers
  and rounding it away would make the breakdown sum to 100% while
  describing less than all of it.
- **The five slowest layers.**
- **Expert cache**: hits, misses, hit rate, evictions, MiB demanded from
  disk, seconds stalled in those reads, MiB per token.
- **A verdict** — I/O bound or compute bound — because that is the
  decision the numbers exist to inform.

Two things it refuses to do, both from spec §114's contributor list. It
never reports GPU kernel time as decode time: the stage timings wrap
whole steps. And it never presents nested timers as siblings — layer time
already contains the expert work, so the report says so instead of adding
them.

Expert counters are lifetime totals on the cache, so they are reported as
a delta against a baseline taken at the first profiled step. The baseline
is the first *decode* step's reading, which keeps prefill's fetches out of
the decode numbers. Without a baseline the section reports nothing rather
than zero — a zero would read as "this run fetched nothing," the opposite
of the truth.

## Reachability

```sh
just profile-decode                 # real checkpoint; prints the summary
TQF_DECODE_PROFILE=1 tqf --headless # or set it directly
```

Off by default, and a separate control from `TQF_DEV_DECODE_DIAGNOSTICS`
(per-token hashes and router traces — a correctness instrument, and forty
layer hashes per token is not a form anyone reads a throughput question
out of). Invariant #10: disableable for A/B, invisible to ordinary users.

## What is verified, and where

The accumulator has six tests covering the parts that are easy to get
wrong and impossible to notice: per-layer time accumulating across steps
rather than being overwritten, unaccounted time being surfaced, the
expert delta subtracting its baseline, a missing baseline reporting
nothing instead of zero, the I/O-vs-compute verdict at Phase 25's own
measured 78%, and an empty profile not dividing by zero.

The wiring into the live decode loop is not exercised here: the resident
runtime needs the real `.tqf` container, which this container does not
have. It compiles, and its gate is checked, but the first real report
comes from `just profile-decode` on the machine with the checkpoint.
Stated plainly rather than implied, per §335 — this half is reachable but
unmeasured, and the phase that measures it is whoever next runs the
recipe.
