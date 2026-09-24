#!/bin/sh
set -eu

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "package-metal: requires an Apple Silicon Mac" >&2
  exit 2
fi

repo=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$repo"
mise exec -- cargo build --release -p jet-cli --features metal,vision

target_dir=${CARGO_TARGET_DIR:-target}
case "$target_dir" in
  /*) ;;
  *) target_dir="$repo/$target_dir" ;;
esac
binary="$target_dir/release/jet"
if [ ! -f "$binary" ]; then
  echo "package-metal: release binary not found: $binary" >&2
  exit 1
fi

dependencies=$(otool -L "$binary")
for dependency in $(printf '%s\n' "$dependencies" | awk 'NR > 1 { print $1 }'); do
  case "$dependency" in
    /usr/lib/*|/System/Library/*) ;;
    *) echo "package-metal: unexpected dynamic dependency: $dependency" >&2; exit 1 ;;
  esac
done

mkdir -p dist
archive="$repo/dist/jet-metal-macos-arm64.tar.gz"
checksum="$archive.sha256"
if [ -e "$archive" ] || [ -e "$checksum" ]; then
  echo "package-metal: output already exists: $archive" >&2
  exit 1
fi

stage=$(mktemp -d "$repo/dist/.jet-metal.XXXXXX")
trap 'rm -rf "$stage"' EXIT HUP INT TERM
mkdir "$stage/jet-metal-macos-arm64"
cp "$binary" "$stage/jet-metal-macos-arm64/jet"
cp README.md "$stage/jet-metal-macos-arm64/README.md"
cp vendor/llama.cpp/LICENSE "$stage/jet-metal-macos-arm64/LLAMA-LICENSE"
tar -C "$stage" -czf "$archive" jet-metal-macos-arm64
(cd "$repo/dist" && shasum -a 256 "$(basename "$archive")") > "$checksum"
cat "$checksum"
