---
title: WebGPU playground
description: Run Jet's full decision pipeline in your browser on Hugging Face Spaces.
---

# WebGPU playground

**[Open Jet Playground on Hugging Face](https://huggingface.co/spaces/yet-another-ai/jet-playground)**

The playground is an independent Vue application hosted as a static Hugging Face
Space. It runs Jet Qwen3.5 0.8B Q8_0 on your device with WebGPU and Wasm.
The initial model download is about 812 MB; wllama caches it in browser storage.
Decision inputs stay in the browser and are not sent to an inference server.

## What it runs

The application accepts a `DecisionRequest` and returns the native
`DecisionResponse` shape. Its patched wllama bundle calls Jet's original
Rust `Engine::decide` path: prompt preparation, tokenization, llama.cpp candidate
scoring, length normalization, softmax, and answer mapping. There is no separate
JavaScript scorer.

## Browser requirements

- A current desktop Chrome or Edge with WebGPU enabled.
- Enough GPU and system memory for the model and inference context.
- Cross-origin isolation and `SharedArrayBuffer`. The Space sends COOP/COEP
  response headers directly; no service worker is needed.

If the embedded Space cannot enable isolation, use its **Open in a new tab**
link. Browser support and performance vary by device.

## Models

- [Jet Qwen3.5 0.8B](https://huggingface.co/yet-another-ai/Jet-Qwen3.5-0.8B):
  the default browser model, plus Transformers weights and its LoRA adapter.
- [Jet Qwen3.5 4B](https://huggingface.co/yet-another-ai/Jet-Qwen3.5-4B):
  weights, adapter and GGUF for native Jet. Its Q8_0 weights exceed this browser
  runtime's 4 GiB Wasm address space.

Model cards contain training methods, data attribution, archived evaluation
results, and limitations. These are text-only supervised LoRA adaptations.

## Development

UI source, its lockfile, local development instructions, and deployed runtime
live in the [Space repository](https://huggingface.co/spaces/yet-another-ai/jet-playground/tree/main).

Jet retains ownership of the Rust/Wasm build and llama.cpp patches. Run
`mise run playground:wllama` to build a new runtime, or download the
`playground-runtime` artifact from the standalone **Playground runtime** Actions
workflow. See [runtime build notes](https://github.com/yet-another-ai/jet/blob/main/playground/README.md).

Documentation builds only require Node and pnpm; they do not build or bundle the
playground runtime.
