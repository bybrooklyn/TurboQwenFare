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

# Only gate commits whose message follows the "Phase N: ..." / "Phases N-M: ..."
# convention (matches both `-m "Phase N: ..."` and the `-m "$(cat <<'EOF' ...` form).
if ! printf '%s' "$CMD" | grep -qE 'Phase(s)?[[:space:]]+[0-9]+.*:'; then
  exit 0
fi

if [[ ! -f "$SENTINEL" ]]; then
  deny "No phase-verify sentinel found. Run the phase-verify skill (/phase-verify) and get a COMPLETE verdict before committing this phase."
fi

STORED_HASH="$(cat "$SENTINEL")"
CURRENT_HASH="$(git diff --cached | shasum -a 256 | cut -d' ' -f1)"

if [[ "$STORED_HASH" != "$CURRENT_HASH" ]]; then
  rm -f "$SENTINEL"
  deny "Staged changes differ from what phase-verify last checked. Re-run /phase-verify before committing."
fi

rm -f "$SENTINEL"
exit 0
