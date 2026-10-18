# JET WebGPU playground runtime

The interactive playground UI lives in the VitePress docs theme
(`docs/.vitepress/theme/components/playground/`) and is served at
[`/playground/`](https://jet.yetanother.ai/playground/) as part of the
documentation site.

This directory holds the **Wasm build pipeline** for that page: scripts clone a
pinned wllama commit, apply Jet patches, and emit the Emscripten/WebGPU module
under `playground/generated/wllama` (gitignored). The docs build copies that
tree into `docs/public/playground-runtime/` so the page can load it.

## Build the Wasm module

Install the root mise toolchain. Docker must be running because wllama builds
llama.cpp with Emscripten in a pinned container.

```sh
mise install
mise run playground:dev
```

That builds Wasm, copies it to `docs/public/playground-runtime/` (with
`ready.json`), and starts VitePress. Or prepare an existing build and serve docs:

```sh
mise run playground:wllama
mise run playground:prepare
mise exec -- pnpm docs:dev
```

Production-style docs build (copies the runtime when present):

```sh
mise run playground:wllama
mise exec -- pnpm docs:build
mise exec -- pnpm docs:preview
```

Without a Wasm build, `pnpm docs:build` still succeeds; the playground page shows
an in-app stub. CI sets `JET_DOCS_REQUIRE_PLAYGROUND=1` so the Documentation
workflow fails if the module is missing.

## Bindings

`jet-llama-sys` uses bindings generated for Emscripten's memory32 ABI when
`webgpu` and `external-link` are enabled. Bindings are generated on demand and
are not tracked. Regenerate after changing the wrapper or vendored llama.cpp
headers:

```sh
mise run playground:bindings
```

## Cross-origin isolation

The Wasm module needs `SharedArrayBuffer` (atomics / imported memory). Locally,
VitePress sends COOP/COEP headers. On GitHub Pages, the playground page loads
`/playground/coi-serviceworker.js` (scoped to `/playground/`) to patch those
headers.

See the [user guide](https://jet.yetanother.ai/guide/playground) for browser
requirements and the decision pipeline overview.
