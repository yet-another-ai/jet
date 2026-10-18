---
title: WebGPU playground
description: Run Jet's full decision pipeline in the browser with WebGPU and Wasm.
---

# WebGPU playground

The [interactive playground](/playground/) runs Jet's decision contract in a
compatible browser. It accepts a `DecisionRequest` and returns the same
`DecisionResponse` shape as the native engine.

Open the live app: [jet.yetanother.ai/playground/](/playground/)

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
- **GitHub Pages:** Pages cannot set those headers, so `/playground/` loads a
  [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) shim that
  patches COOP/COEP for that path only.

## Build locally

Install the root mise toolchain. Docker must be running because wllama builds
llama.cpp with Emscripten in a pinned container.

```sh
mise install
mise run playground:dev
```

The first run clones the pinned wllama commit, applies the Jet patches, compiles
`jet-web` and llama.cpp into one WebGPU Wasm module, and starts Vite. Open the
printed URL, load the default Qwen3 GGUF, then select **Load & run**.

To embed the playground into a local docs build (same layout as production):

```sh
mise run playground:wllama
mise exec -- pnpm docs:build
mise exec -- pnpm docs:preview
```

Without a Wasm build, `pnpm docs:build` still succeeds and leaves a stub at
`/playground/`. CI sets `JET_DOCS_REQUIRE_PLAYGROUND=1` so the docs workflow
fails if the module is missing.

Emscripten bindings stay on-demand and gitignored. Regenerate them after changing
the wrapper or vendored llama.cpp headers with `mise run playground:bindings`.

See [playground/README.md](https://github.com/yet-another-ai/jet/blob/main/playground/README.md)
for the full developer notes.
