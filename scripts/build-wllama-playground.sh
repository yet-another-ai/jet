#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_dir="$repo_root/target/wllama-playground-src"
output_dir="$repo_root/playground/generated/wllama"
legacy_output_dir="$repo_root/playground/generated/jet-web"
patch_file="$repo_root/patches/wllama-jet-engine.patch"
llama_patch_file="$repo_root/patches/llama-webgpu-rmsnorm-workgroup.patch"
wllama_commit=dbc42c093dfd911360be12064ac8575fc629faa7

if [ ! -d "$source_dir/.git" ]; then
  git clone https://github.com/ngxson/wllama.git "$source_dir"
fi

git -C "$source_dir" fetch origin "$wllama_commit"
git -C "$source_dir" reset --hard "$wllama_commit"
git -C "$source_dir" clean -fdx
git -C "$source_dir" submodule update --init --recursive --force
git -C "$source_dir/llama.cpp" reset --hard
git -C "$source_dir/llama.cpp" clean -fdx
git -C "$source_dir" apply --check "$patch_file"
git -C "$source_dir" apply "$patch_file"
mkdir -p "$source_dir/jet-llama.cpp"
git -C "$repo_root/vendor/llama.cpp" archive HEAD | tar -x -C "$source_dir/jet-llama.cpp"
git -C "$source_dir/jet-llama.cpp" init -q
git -C "$source_dir/jet-llama.cpp" apply --unidiff-zero --check "$llama_patch_file"
git -C "$source_dir/jet-llama.cpp" apply --unidiff-zero "$llama_patch_file"

# The Emscripten bindings are generated locally and are not tracked.
sh "$repo_root/scripts/generate-emscripten-bindings.sh"

cd "$source_dir"
npm ci
export JET_REPO_ROOT="$repo_root"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 -C panic=abort" \
cargo +nightly build --manifest-path "$repo_root/Cargo.toml" \
  -p jet-web \
  --target wasm32-unknown-emscripten \
  -Z build-std=std,panic_abort \
  --release
SKIP_COMPAT=1 npm run build:wasm
npm run build

rm -rf "$output_dir" "$legacy_output_dir"
mkdir -p "$(dirname -- "$output_dir")"
cp -R "$source_dir/esm" "$output_dir"
