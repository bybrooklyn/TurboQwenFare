---
name: phase-verify
description: Audit a just-finished TQF phase for real completion before it's committed — re-reads the diff, re-runs build/tests, checks the spec's literal exit gate, and checks/updates the phase's qualification doc. Use when a phase looks done and is about to be committed, when the user says "verify this phase", "is phase N actually done", "ready to commit phase N", or via /phase-verify. Pairs with the PreToolUse hook in .claude/settings.json that blocks `git commit` on a "Phase N: ..."-style message until this skill has produced a COMPLETE verdict.
---

# Phase Verify

A completion gate for TQF's phase-based workflow. Nothing gets committed as
"Phase N: ..." until this skill has actually re-derived, this run, that the phase's
literal exit gate is met — not recalled from earlier in the conversation, not
assumed because the code compiles.

This pairs with `.claude/hooks/phase-verify-gate.sh` (registered in
`.claude/settings.json`), which blocks `git commit` on any message matching
`Phase(s)? [0-9]+:` unless this skill produced a fresh COMPLETE verdict for exactly
what's currently staged. Running this skill is not optional ceremony — it's the only
way that hook opens.

## Quick Start

1. Determine which phase number is being audited:
   - Explicit: user says "verify phase 53" / `/phase-verify 53`.
   - Inferred: look at what's actually changed (`git status`, `git diff`) and match
     it against the next unclaimed row in the spec's phase map, or the phase number
     CLAUDE.md/recent commits imply is in flight.
   - If genuinely ambiguous, ask the user rather than guessing.
2. Run the full audit checklist below.
3. Write an explicit verdict. Only on COMPLETE, stage and write the sentinel.

## Audit checklist

Every step below must use evidence gathered *this run*. A claim from earlier in the
conversation, or "it worked when I checked before," does not satisfy any of these —
the diff may have changed since.

1. **Pull the real exit gate.** `grep -n '^| N ' "TurboQwenFare_Master_v2_All_Encompassing_Specification.md"`
   against the `# 112. Phase map` table (line ~1694) and quote that row's *Primary
   output* and *Exit gate* columns verbatim. Don't paraphrase the gate — the literal
   wording is what gets checked clause by clause in step 4.

2. **Read the actual diff.** `git status` and `git diff` (and `git diff --cached` if
   anything's already staged) — every changed file, not a summary of intent. Flag:
   changes unrelated to this phase's stated primary output; anything stubbed,
   `TODO`, `unimplemented!()`, or otherwise incomplete that's being presented as
   done; debug/scratch code left in.

3. **Re-run, don't recall.** `cargo build` (add `--release` if the phase is
   performance- or measurement-relevant), `cargo test`, `cargo clippy`. Capture the
   real output. A green run from earlier in this session doesn't count once the
   diff has changed — rerun after the tree is in its final state.

4. **Walk the exit gate clause by clause.** For each clause, name the concrete
   evidence — a specific test, a specific measured number, a specific file/line —
   that satisfies it. A gate demanding a number (tok/s, % quality degradation, an
   Nx speedup, a memory ceiling) requires an actual number captured in step 3/4,
   not an assertion that it's "probably" in range. If a clause can't be evidenced,
   it isn't met — full stop, regardless of how much surrounding work is finished.

5. **Confirm or write the qualification doc.**
   `docs/research/qualification/phase-N-<slug>.md` should exist (or be created as
   part of this phase) matching the shape every prior phase uses: opens `# Phase N:
   <title>`, cites the relevant spec section(s) and quotes the exit gate, then
   `## Scope decision`, `## What was built`, `## Measured evidence`, `## Status and
   remaining work`. The "Measured evidence" section must be the evidence actually
   just reproduced in step 3/4 — not older numbers copied forward, and not numbers
   that no longer match the current code.

6. **Apply real investigation discipline.** Any flaky test, ambiguous near-tied
   result, or failure that looks "probably environmental" does not get waved past.
   Either root-cause and resolve it, or write it up explicitly as an honest open
   gap — TQF's own convention (Phases 20, 22, 23, 41, the 512-token divergence
   writeup) is to record real negative results plainly rather than smooth them
   into an unqualified "done." A dismissed ambiguity is an INCOMPLETE verdict, not
   a footnote.

7. **Check the do-not-do-this list (spec §114) and the dependency firewall
   (§24).** No generic-model/Llama abstractions added "while here," no crate
   workspace split, no external vector DB, no user-facing quality mode, no
   `retrieval`/`gui` leaking into `model`/`runtime`, no silent over-`--memory`
   allocation. A phase that's technically functional but violates one of these is
   not COMPLETE as scoped.

8. **Check for stale-doc drift.** Confirm `CLAUDE.md`'s phase narrative doesn't
   already describe this phase in a way that conflicts with what's actually in the
   tree now (e.g. claiming something is wired into the live loop when this diff
   doesn't do that). If CLAUDE.md needs updating to reflect this phase, that update
   is part of what "done" means here.

## Verdict

State it explicitly: **COMPLETE** or **INCOMPLETE**. No middle state.

- **INCOMPLETE** — enumerate every unresolved item precisely (file:line where it
  applies). Do not `git add` anything, do not touch the sentinel file, tell the
  user exactly what's missing and why the gate hasn't cleared.
- **COMPLETE** — stage the intended files (`git add ...`), then write the sentinel
  the hook checks:

  ```sh
  git diff --cached | shasum -a 256 | cut -d' ' -f1 > .claude/.phase-verify-ok
  ```

  Report a short verdict summary to the user and note that `git commit -m "Phase
  N: ..."` will now pass the gate — for exactly what's currently staged.

## Sentinel / hook integration

- Sentinel path: `.claude/.phase-verify-ok` (gitignored — transient local state,
  never committed).
- Contents: the sha256 of `git diff --cached` at the moment verification passed,
  computed with `git diff --cached | shasum -a 256 | cut -d' ' -f1`.
- The hook (`.claude/hooks/phase-verify-gate.sh`) only inspects `git commit`
  invocations whose message matches `Phase(s)? [0-9]+:` — TQF's own commit
  convention. On a match it recomputes the same hash over the currently staged
  diff and compares it to the sentinel's contents.
- The sentinel is single-use: the hook deletes it after either an allow or a stale
  mismatch, so re-staging anything (or committing again later) requires a fresh
  `/phase-verify` pass. Never hand-write or copy forward a sentinel value — it
  must come from a real COMPLETE verdict reached this run.

## Failure modes

- The hook only gates commits whose `-m` message looks like `Phase N: ...` /
  `Phases N-M: ...`. Ordinary non-phase commits (a typo fix, an unrelated chore)
  are never blocked — that's intentional scope, not a gap. Don't widen the
  matcher to catch more commits without a real reason.
- If the gate denies a commit and the phase genuinely isn't ready, the fix is to
  finish or fix the actual gap and re-run `/phase-verify` — never to hand-edit or
  fabricate `.claude/.phase-verify-ok`. Bypassing the gate defeats the entire
  point of this skill.
- If staged content changes after a COMPLETE verdict (more files added, a fixup
  amended in), the hash will no longer match and the hook will correctly deny —
  re-run `/phase-verify` rather than treating that as a bug.
