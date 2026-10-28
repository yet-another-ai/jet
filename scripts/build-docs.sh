#!/bin/sh
# Build the documentation independently of the Hugging Face playground.
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
pnpm exec vitepress build docs
