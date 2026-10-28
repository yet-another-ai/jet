# Jet WebGPU runtime

The independent [Jet Playground Space](https://huggingface.co/spaces/yet-another-ai/jet-playground)
owns the Vue UI, its frontend build, and the deployed runtime. Model artifacts
live in the corresponding Hugging Face Model repositories. The documentation
site links to the Space and builds without Wasm.

This directory remains the output location for Jet's **Wasm build pipeline**.
Rust crates, vendored llama.cpp, and Jet's wllama/WebGPU patches are maintained
here so browser and native scoring use the same implementation.

## Build

Install the root mise toolchain and start Docker. Run:

```sh
mise run playground:wllama
```

The script clones a pinned wllama revision, applies Jet patches, generates
Emscripten memory32 bindings on demand, and emits the complete runtime into
`playground/generated/wllama/` (gitignored). Bindings can also be regenerated
with `mise run playground:bindings`.

The standalone **Playground runtime** Actions workflow builds the same runtime
and retains the `playground-runtime` artifact for 14 days. CI runs it separately
from the documentation build and requires it in Gate.

## Update the Space

Clone the Space using SSH:

```sh
git clone git@hf.co:spaces/yet-another-ai/jet-playground
```

From Jet, export the complete generated tree and provenance into that checkout:

```sh
mise exec -- node scripts/export-playground-runtime.mjs ../jet-playground
```

In the Space checkout run `npm ci`, `npm run build`, and `npm run dev`.
Verify model download and a complete decision in a supported browser, then
commit the source and generated static files, then push the Space. Its static
SDK serves the prebuilt files directly, without hosted build credits.
COOP/COEP headers are configured in the Space README and local Vite server.
