# Development

Jet pins llama.cpp as a Git submodule at commit
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (the v0.4.1 line). The default build is CPU-only and
disables OpenMP, BLAS, CUDA, Metal, Vulkan, HIP, SYCL, and RPC. The opt-in `vulkan` Cargo feature
enables only llama.cpp's Vulkan backend in addition to CPU fallback support.

Bindings matching that commit are checked into `jet-llama-sys`. Normal builds do not load
libclang. To deliberately regenerate them after changing the pin:

```sh
LIBCLANG_PATH="$(mise where http:llvm)/lib" \
  mise exec -- cargo check -p jet-llama-sys --features generate-bindings
```

The test model is `Qwen/Qwen3-0.6B-GGUF` revision
`23749fefcc72300e3a2ad315e1317431b06b590a`, file `Qwen3-0.6B-Q8_0.gguf`, with SHA-256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.

llguidance remains disabled. Bounded thinking uses llama.cpp's protocol-aware reasoning-budget
sampler and the model template's declared end markers; candidate scoring itself does not execute a
grammar.

Run the fast checks:

```sh
mise exec -- cargo fmt --check
mise exec -- cargo clippy --workspace --all-targets -- -D warnings
mise exec -- cargo test --workspace
```

After downloading the model, run ignored model and CLI tests:

```sh
JET_MODEL_PATH="$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  mise exec -- cargo test --workspace -- --ignored --test-threads=1
```

To compile and run the Vulkan smoke test, install the Vulkan loader and development headers,
SPIR-V headers, and `glslc`, then run:

```sh
JET_MODEL_PATH="$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  mise exec -- cargo test -p jet-engine --features vulkan \
  qwen_vulkan_smoke -- --ignored --nocapture
```

On NixOS, `nix shell` puts `glslc` on `PATH`, but CMake also needs the split header and loader
outputs. This self-contained invocation supplies them:

```sh
vulkan_headers="$(nix build --no-link --print-out-paths nixpkgs#vulkan-headers)"
vulkan_loader="$(nix build --no-link --print-out-paths nixpkgs#vulkan-loader)"
spirv_headers="$(nix build --no-link --print-out-paths nixpkgs#spirv-headers)"
CMAKE_PREFIX_PATH="$vulkan_headers:$vulkan_loader:$spirv_headers" \
CXXFLAGS="-I$spirv_headers/include" \
JET_MODEL_PATH="$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  nix shell nixpkgs#shaderc nixpkgs#vulkan-headers nixpkgs#vulkan-loader \
    nixpkgs#spirv-headers --command \
  mise exec -- cargo test -p jet-engine --features vulkan \
    qwen_vulkan_smoke -- --ignored --nocapture
```

On NixOS, the upstream prebuilt LLVM archive expects the host's zlib shared library. Regeneration
therefore needs a shell that exposes zlib in `LD_LIBRARY_PATH`. Regular builds do not need this.
