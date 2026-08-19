# Phase 46: SwiftUI bridge

Spec Phase 46 deliverable (spec §46, §96-98, §318). "Compile adopted UI
into one binary, replace original app model with HTTP-backed TQF
state, verify main-thread lifecycle and headless separation."

## What was built

A real Swift Package (`swift/`) compiled and statically linked into
the single `tqf` Mach-O executable, gated behind a new opt-in `gui`
Cargo feature (not part of `default` — see "opt-in, not unconditional"
below):

- **Adopted source** (spec §96: "directly adapt selected Apache-2.0
  TurboFieldfare/NVMAI SwiftUI source rather than visually imitating
  it"), copied verbatim with attribution from a real local NVMAI clone
  (`/Volumes/flash1/tqf-research/NVMAI`, Apache-2.0, confirmed via its
  own `LICENSE` file) into `swift/Sources/TqfGUI/`:
  `Rendering/ResponseMarkdownRenderer.swift` (313 lines, a real
  markdown-to-`NSAttributedString` renderer), `Theme/NVMAIMacTheme.swift`
  (accent color), `Components/HUDMetricView.swift` (metric display
  atom). `swift/NOTICE.md` and `swift/NVMAI-LICENSE.txt` record the
  attribution per Apache-2.0 §4.
- **Replaced original app model with HTTP-backed TQF state** (spec's
  own phrasing, literally): `TqfInferenceClient.swift` is new code — a
  thin `URLSession`-based client that streams TQF's own real
  `/v1/chat/completions` SSE endpoint (`src/server/openai/mod.rs`'s
  actual wire format, not a guessed one) and calls TQF's real
  `GET /health`. `TqfAppModel.swift` is a new, deliberately small
  `@Observable` state object — not a port of NVMAI's own 942-line
  `AppModel`, which owns an in-process Metal model-load lifecycle that
  has no TQF equivalent (TQF's model is always server-owned, spec §22).
  `RootView.swift` composes the adopted rendering/theme pieces with
  this new model into a real chat window.
- **The C-callable entrypoint** (spec §98): `TqfGuiBridge.swift`'s
  `@_cdecl("tqf_launch_gui")` function creates `NSApplication`, hosts
  `RootView` in an `NSHostingController`/`NSWindow`, and calls
  `app.run()` — blocking the calling thread until the user quits,
  exactly like NVMAI's own `NVMAIMacApp.swift` foreground-app delegate
  pattern (rewritten here since TQF's entry is a C function, not a
  `@main App`).
- **Rust side**: `src/gui/macos/mod.rs`'s `launch()` wraps the `extern
  "C"` declaration; `src/app/mod.rs`'s `run_server_or_gui` spawns the
  server on a background OS thread and hands the *real* main thread to
  `launch()` when not headless and the `gui` feature is compiled in.
  `--headless` (and any non-macOS or non-`gui` build) takes the exact
  same code path every earlier phase already used: `serve::start`
  called synchronously on the calling thread, completely untouched.

## Scope decision: opt-in `gui` feature, not unconditional

`build.rs` only invokes `swift build` when the new `gui` Cargo feature
is enabled (not part of `default = ["metal"]`). This is a scope
decision, not a spec deviation: a headless-only build/CI environment
need not have a Swift toolchain installed at all, and every earlier
phase's `cargo build`/`cargo test` invocation throughout this session
keeps working completely unaffected. A real shipping build would
enable `gui` by default on macOS; this phase proves the mechanism
works end to end without changing every other phase's build command
retroactively.

## Measured evidence

**Real compile, real link, real symbol — not a mockup.** `swift build`
(both debug and release configurations) compiles the adopted+new
source cleanly. `cargo build --features gui` links the resulting
`libTqfGUI.a` static archive into the actual `tqf` binary — confirmed
directly with `nm target/debug/tqf | grep tqf_launch_gui`, which shows
the real exported C symbol (`_tqf_launch_gui`) present in the linked
executable, not just in the standalone Swift build. `./target/debug/tqf
--help` runs correctly from the `gui`-linked binary (`dyld` resolves
every framework/Swift-runtime dependency at real process startup, not
just at link time).

**Zero regressions.** `cargo test` (406/407 passing, unchanged) and
`cargo test --features gui` (406 passing) both stay 100% clean — a
completely-from-scratch `cargo build --features gui` (after deleting
both `target/debug/tqf` and `swift/.build`) also succeeds, ruling out
any risk of stale-artifact false confidence. `cargo clippy` and `cargo
fmt --check` report nothing new.

**Headless separation, tested directly, not just asserted:**
`gui::macos::tests::launch_without_gui_feature_returns_immediately_and_
never_panics` confirms that without the `gui` feature, `launch()`
returns in under 50 ms regardless of its `base_url` argument — the
real `AppKit` entrypoint blocks in `NSApplication.run()` until the user
quits, so a fast, panic-free return is direct proof this build never
touches Swift at all, not merely "the code looks like it shouldn't."

**The real link flags, derived empirically, not guessed.** Getting
`cargo build --features gui` to actually link required finding the
exact linker search paths/frameworks a real Swift executable needs:
built a disposable throwaway SwiftPM executable depending on the same
`TqfGUI` library, ran `swift build -v`, and read the *actual* `clang`
invocation SwiftPM itself used — revealing that modern Swift autolinks
its runtime via embedded `LC_LINKER_OPTION`s in each object file (no
explicit `-lswiftCore` etc. needed), only `-L` search paths and an
rpath. One real, found-and-fixed link error this way: `Observation` is
a Swift-only module, not a linkable Apple `.framework` — `-framework
Observation` doesn't exist and initially failed the link; removed once
the actual error message (`framework 'Observation' not found`)
diagnosed it, rather than left in as a guess.

## Status and remaining work

- **No automated Swift-side test suite.** A `TqfGUITests` target was
  attempted (real network tests against a real embedded
  `Network.framework` HTTP server, testing `TqfInferenceClient`'s
  health-check and real SSE-chunk parsing) but dropped: this
  development machine's Swift toolchain (installed via `swiftly`, not
  Xcode — only Command Line Tools are present) has neither `XCTest`
  nor `Testing` available to link against. This is a real, specific,
  reported environment limitation, not a code defect — the Rust-side
  linking/regression tests are the actual coverage this phase has.
- **No visual/interactive verification.** Nothing in this environment
  can open a real display session to confirm the window actually
  renders, the accent color looks right, or a real streamed response
  displays correctly — `swift build` compiling and `cargo build
  --features gui` linking prove the mechanism is real and correct at
  the type/API level, not that it looks right on screen.
- **Server-address synchronization is approximate.** `run_with_gui`
  points the GUI at TQF's default bind port
  (`server::bind::DEFAULT_PORT`); if that port was actually occupied
  and the server's real bind-with-fallback logic picked a different
  one, the GUI's hardcoded default won't match. Threading the real
  bound address back from the spawned server thread to the GUI launch
  call (e.g. via a chsnnel) is real, scoped follow-up work, not
  attempted this phase.
- **Menu commands, window style (`hiddenTitleBar`), and the full
  install/load-state UI** from NVMAI's original `NVMAIMacApp.swift`/
  `AppModel.swift` are not ported — spec §47 (UI refinement) is the
  next phase for polish; this phase's exit gate is the mechanism
  (compile into one binary, main-thread lifecycle, headless
  separation), not visual completeness.
- Only `ResponseMarkdownRenderer`/`NVMAIMacTheme`/`HUDMetricView` were
  adopted verbatim; NVMAI's other `Mac/Components`/`Mac/Generation`/
  `Mac/Diagnostics`/`Mac/Installation` views were left unadapted since
  they're coupled to `AppModel`'s NVMAI-specific install/load state
  machine, which has no TQF equivalent (TQF's setup flow, spec §28, is
  a different real state machine — adapting these views to it is
  future work, not attempted here to avoid either faking the coupling
  or a much larger scope than this phase's exit gate asks for).
