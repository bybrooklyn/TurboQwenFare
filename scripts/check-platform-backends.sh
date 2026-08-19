#!/usr/bin/env bash
# Guards the platform-conditional backend wiring: `metal-sys` must resolve
# for Apple targets and must be absent from the dependency graph
# everywhere else. Without this check a regression shows up only as a
# broken `cargo build` on somebody else's laptop.
#
# Note it inspects `resolve.nodes`, not the top-level `packages` list:
# `--filter-platform` narrows the resolve graph but still reports every
# package the manifest mentions, so grepping `packages` finds `metal` on
# Linux and reports a false pass.
set -euo pipefail

check() {
    local triple="$1" expected="$2"
    local found
    found="$(cargo metadata --format-version 1 --filter-platform "$triple" 2>/dev/null \
        | python3 -c "
import json, sys
resolve = json.load(sys.stdin)['resolve']['nodes']
print(sum(1 for node in resolve if '#metal@' in node['id']))
")"
    case "$expected" in
        absent)
            [ "$found" -eq 0 ] || { echo "FAIL: metal-sys resolved for $triple"; exit 1; }
            echo "ok: metal-sys absent for $triple"
            ;;
        present)
            [ "$found" -ge 1 ] || { echo "FAIL: metal-sys missing for $triple"; exit 1; }
            echo "ok: metal-sys present for $triple"
            ;;
    esac
}

check x86_64-unknown-linux-gnu  absent
check aarch64-unknown-linux-gnu absent
check aarch64-apple-darwin      present
check x86_64-apple-darwin       present
