#!/bin/sh
# Copy playground/generated/wllama into docs/public/playground-runtime and mark
# it ready for the VitePress playground page. Exit 1 if the Wasm tree is absent
# unless JET_PLAYGROUND_RUNTIME_OPTIONAL=1.
#
# Prefer generated/ when present so a fresh playground:wllama always refreshes
# docs/public. Fall back to an existing public tree (CI download artifact).
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
generated="$repo_root/playground/generated/wllama"
public_runtime="$repo_root/docs/public/playground-runtime"

mark_ready() {
  printf '%s\n' '{"ok":true}' >"$public_runtime/ready.json"
}

if [ -d "$generated" ] && [ -f "$generated/index.js" ]; then
  rm -rf "$public_runtime"
  mkdir -p "$(dirname -- "$public_runtime")"
  cp -R "$generated" "$public_runtime"
  mark_ready
  echo "Copied playground Wasm runtime to $public_runtime"
  exit 0
fi

if [ -d "$public_runtime" ] && [ -f "$public_runtime/index.js" ]; then
  mark_ready
  echo "Playground runtime ready at $public_runtime"
  exit 0
fi

if [ "${JET_PLAYGROUND_RUNTIME_OPTIONAL:-0}" = 1 ]; then
  rm -rf "$public_runtime"
  echo "warning: playground Wasm missing; /playground/ will show the in-page stub" >&2
  exit 0
fi

echo "error: playground Wasm missing (expected $generated or $public_runtime/index.js)" >&2
exit 1
