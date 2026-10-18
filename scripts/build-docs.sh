#!/bin/sh
# Prepare the playground Wasm runtime (when present) and build the VitePress site.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if [ "${JET_DOCS_REQUIRE_PLAYGROUND:-0}" = 1 ]; then
  sh "$repo_root/scripts/prepare-playground-runtime.sh"
else
  JET_PLAYGROUND_RUNTIME_OPTIONAL=1 sh "$repo_root/scripts/prepare-playground-runtime.sh"
fi

pnpm exec vitepress build docs
