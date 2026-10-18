#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sysroot="$repo_root/target/emscripten-4.0.20-sysroot"

case "$(uname -m)" in
  arm64 | aarch64) default_image=emscripten/emsdk:4.0.20-arm64 ;;
  *) default_image=emscripten/emsdk:4.0.20 ;;
esac
image=${JET_EMSCRIPTEN_IMAGE:-$default_image}

mkdir -p "$sysroot"
docker run --rm \
  -v "$sysroot:/output" \
  --entrypoint sh \
  "$image" \
  -c 'cp -a /emsdk/upstream/emscripten/cache/sysroot/. /output/'

JET_EMSCRIPTEN_SYSROOT="$sysroot" \
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 -C panic=abort' \
cargo +nightly build \
  -p jet-llama-sys \
  --features external-link,webgpu,generate-bindings \
  --target wasm32-unknown-emscripten \
  -Z build-std=std,panic_abort \
  --release
