#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash). Blocks `git commit` on a "Phase N: ..." /
# "Phases N-M: ..." message unless .claude/skills/phase-verify has produced a
# fresh COMPLETE verdict for exactly what's currently staged.
#
# Non-phase commits (typos, chores) are intentionally never gated. See
# .claude/skills/phase-verify/SKILL.md for the full mechanism and rationale.
set -euo pipefail

SENTINEL=".claude/.phase-verify-ok"

INPUT="$(cat)"
CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"

deny() {
  local reason="$1"
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}\n' \
    "$(printf '%s' "$reason" | jq -Rs .)"
  exit 0
}

# Only act on git commit invocations.
if ! printf '%s' "$CMD" | grep -qE 'git[[:space:]]+commit'; then
  exit 0
fi

# Only gate commits whose message *subject* follows the "Phase N: ..." /
# "Phases N-M: ..." convention. The pattern must therefore be anchored: it
# has to sit at the start of a line (the heredoc / `-F -` forms) or
# directly after `-m` (the inline form), with the colon right after the
# number.
#
# The earlier unanchored `Phase(s)? [0-9]+.*:` matched anywhere in the
# whole command string, so an ordinary commit whose *body* mentioned a
# phase in passing — "measured a negative (Phase 20)" followed by any
# later colon — was blocked as if it were a phase commit.
PHASE_SUBJECT='Phase(s)?[[:space:]]+[0-9]+([[:space:]]*-[[:space:]]*[0-9]+)?[[:space:]]*:'
if ! printf '%s' "$CMD" | grep -qE "^${PHASE_SUBJECT}" \
  && ! printf '%s' "$CMD" | grep -qE -- "-m[[:space:]]+[\"']?${PHASE_SUBJECT}"; then
  exit 0
fi

if [[ ! -f "$SENTINEL" ]]; then
  deny "No phase-verify sentinel found. Run the phase-verify skill (/phase-verify) and get a COMPLETE verdict before committing this phase."
fi

STORED_HASH="$(cat "$SENTINEL")"
# `shasum` ships with macOS, `sha256sum` with GNU coreutils on Linux.
# Using either keeps the gate working on both reference platforms.
if command -v shasum >/dev/null 2>&1; then
  CURRENT_HASH="$(git diff --cached | shasum -a 256 | cut -d' ' -f1)"
else
  CURRENT_HASH="$(git diff --cached | sha256sum | cut -d' ' -f1)"
fi

if [[ "$STORED_HASH" != "$CURRENT_HASH" ]]; then
  rm -f "$SENTINEL"
  deny "Staged changes differ from what phase-verify last checked. Re-run /phase-verify before committing."
fi

rm -f "$SENTINEL"
exit 0
