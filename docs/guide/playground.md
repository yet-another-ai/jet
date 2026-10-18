---
title: WebGPU playground
description: Run Jet's full decision pipeline in the browser with WebGPU and Wasm.
---

# WebGPU playground

The [interactive playground](/playground/) runs Jet's decision contract in a
compatible browser. It accepts a `DecisionRequest` and returns the same
`DecisionResponse` shape as the native engine.

Open the live app: [jet.yetanother.ai/playground/](/playground/)

The UI is a VitePress page (no sidebar) that mounts client-only Vue components
from the docs theme. The Wasm/wllama runtime is served from
`/playground-runtime/` on the same site.

## What it runs

One Emscripten module carries the browser pipeline:

```text
DecisionRequest
      │
      ▼
jet-engine + jet-decision + jet-core (Rust)
  validate + render + tokenize + schedule
      │
      ▼
Jet's vendored llama.cpp (C/C++)
  WebGPU inference + teacher-forced token log probabilities
      │
      ▼
jet-decision (the same Rust code as native Jet)
  length normalization + softmax + answer mapping + usage
      │
      ▼
DecisionResponse
```

The module calls the original `Engine::decide` path. wllama supplies browser
workers, model download/cache, HeapFS, and the TypeScript facade. It does not
implement a second scorer.

## Browser requirements

- A browser with WebGPU enabled
- Enough device and system memory for the GGUF model and Jet inference context
- A model URL that allows cross-origin browser downloads
- Cross-origin isolation (`SharedArrayBuffer`), because the Wasm module is built
  with atomics / imported memory for multithreaded execution

The docs site enables isolation in two ways:

- **Local `docs:dev` / `docs:preview`:** Vite sends
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`.
- **GitHub Pages:** Pages cannot set those headers, so the playground page loads
  a scoped [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)
  under `/playground/` (it may reload once to take control).

Normal VitePress client navigation to `/playground/` works; the page is part of
the docs SPA.

## Build locally

Install the root mise toolchain. Docker must be running because wllama builds
llama.cpp with Emscripten in a pinned container.

```sh
mise install
mise run playground:dev
```

That builds the Wasm module, copies it to `docs/public/playground-runtime/`
(including `ready.json`), and starts VitePress. To reuse an existing Wasm build
with a plain docs server:

```sh
mise run playground:prepare
mise exec -- pnpm docs:dev
```

Without a Wasm build, `pnpm docs:build` still succeeds and the playground page
shows an in-app stub. CI sets `JET_DOCS_REQUIRE_PLAYGROUND=1` so the docs
workflow fails if the module is missing.

Emscripten bindings stay on-demand and gitignored. Regenerate them after changing
the wrapper or vendored llama.cpp headers with `mise run playground:bindings`.

See [playground/README.md](https://github.com/yet-another-ai/jet/blob/main/playground/README.md)
for Wasm build notes.
