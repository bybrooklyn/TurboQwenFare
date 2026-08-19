# Phase 47: TQF UI refinement

Spec Phase 47 deliverable (spec §47, §319). "Create simple default
conversation/setup experience and expandable engineering cockpit. The
inspector consumes metrics; it must not change runtime policy directly
except through supported configuration actions."

## What was built

**A real new Rust metrics endpoint**, not fabricated data for the UI to
display: `GET /v1/tqf/metrics` (`src/server/tqf_api/mod.rs`) returns
real OS-sampled process memory (resident/virtual/peak-resident bytes)
plus the same real uptime/`model_installed` fields `/health` already
exposed. The sampling itself reuses Phase 24's real `task_info`/
`/proc/self/statm` sampler (`memory::os_sampler`), which previously
required a live `MemoryBroker` instance to call; added a new
broker-independent `sample_process_footprint()` so a plain HTTP
endpoint can call it without needing a model loaded.

**A real new SwiftUI cockpit**, building on Phase 46's working
compile/link pipeline:

- Adopted `MetricFormat.swift` verbatim from the real NVMAI source
  (POSIX-locale-fixed byte/rate/percent formatters — the same
  attribution convention Phase 46 established), made `public` since
  NVMAI's original was internal-only within its own app target.
- New `InspectorView.swift` — deliberately *not* a port of NVMAI's own
  much larger `InspectorView` (which surfaces per-kernel timing
  breakdowns like `cb1MillisecondsPerToken` that are specific to
  NVMAI's own in-process Metal runtime and have no TQF equivalent).
  TQF's inspector shows the real metrics TQF's own server actually has
  today: uptime, model-installed state, resident/peak/virtual memory.
- `TqfAppModel` gained real metrics polling (`startMetricsPolling`,
  a 2-second real HTTP poll loop calling the new endpoint) and a
  `showsInspector` toggle — spec's literal "simple default... and
  expandable... cockpit": `RootView`'s conversation pane is unchanged
  and always visible; toggling reveals `InspectorView` alongside it,
  changing only what's displayed.
- **Read-only by construction, not just by convention**: there is no
  "set metric" or "mutate policy" call anywhere in `TqfAppModel` or
  `InspectorView` — satisfying spec's "must not change runtime policy
  directly except through supported configuration actions" trivially,
  since no such actions exist yet to wire up (honestly noted below,
  not hidden).

## Measured evidence

**The metrics endpoint is tested for real, not asserted from
description.** `server::tests::tqf_metrics_reports_real_process_memory`
makes a genuine HTTP request against a genuinely running test server
and asserts the returned `resident_bytes` is a real positive number —
the test process's own actual resident memory, sampled live via
`task_info`, not a placeholder. A silently-failing sampler returning
zero or `None` would fail this test, not pass it by accident.

**Zero regressions, verified the same rigorous way as Phase 46**: a
from-scratch `cargo build --features gui` (after deleting both
`target/debug/tqf` and `swift/.build`) succeeds; `nm` confirms
`_tqf_launch_gui` is still present in the linked binary; `cargo test`
(408 passing, +1 over Phase 46's 407 for the new metrics test) and
`cargo test --features gui` (407 passing) both stay 100% clean; `swift
build -c release` compiles the extended package (now six real
TqfGUI-owned files plus the three adopted ones) with zero errors;
`cargo clippy`/`cargo fmt --check` report nothing new (the only
pre-existing `identity_op` warnings in `os_sampler.rs`'s own test code
predate this phase and are untouched).

## Status and remaining work

- **No visual/interactive verification** — same honest limitation as
  Phase 46: nothing in this environment can open a real display session
  to confirm the inspector panel actually slides in, the metrics
  actually update live on screen, or the layout looks right. The real,
  tested surface is: the Rust endpoint returns real data (proven by a
  real HTTP test), and the Swift package compiles cleanly against that
  data's real shape (`TqfMetrics`'s `Decodable` conformance matches
  `MetricsResponse`'s real JSON field names exactly).
- **"Supported configuration actions"** (spec's own phrase for what the
  inspector *is* allowed to trigger) don't exist yet in this build —
  there is no live decode loop or runtime-tunable policy to expose a
  safe action for, so the inspector is unconditionally read-only rather
  than read-only-with-a-documented-exception-list. Revisit once a real
  live-tunable knob exists (e.g. a cache-policy toggle) to wire one
  supported action through and prove the "except through" half of the
  spec sentence, not just the "must not" half.
- **No conversation "setup experience"** — spec also asks for a
  "simple default... setup experience," which needs the real `tqf`
  first-run/setup flow (`src/setup/`, spec §28) to have a GUI surface;
  not attempted this phase, which focused on the cockpit half of the
  sentence.
- Same NVMAI-adaptation scope boundary as Phase 46: `Mac/Components`/
  `Mac/Generation`/`Mac/Installation`'s remaining views stay unadapted,
  since they're coupled to NVMAI's own `AppModel` install/load state
  machine, which TQF's actual setup flow doesn't share.
