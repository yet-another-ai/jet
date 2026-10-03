---
title: Getting started
description: Build Jet, download the example model, and run your first structured decision.
---

# Getting started

Jet scores a fixed set of answers with a local language model. This guide builds the CPU CLI, downloads the pinned example model, and runs a request from the repository.

## Prerequisites

Install [mise](https://mise.jdx.dev/) 2026.10.0 or newer and a platform toolchain capable of building Rust and the bundled llama.cpp.

## Build Jet

Clone the repository with its llama.cpp submodule, install the pinned tools, and build the release CLI:

```sh
git clone --recurse-submodules https://github.com/yet-another-ai/jet.git
cd jet
mise install
./scripts/download-test-model.sh
mise exec -- cargo build --release -p jet-cli
```

The download script installs the pinned Qwen3 0.6B GGUF model under `models/`.

## Run the example

```sh
./target/release/jet judge \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

`--output -` writes one JSON response per input line to standard output. The process keeps the model loaded while it consumes the input stream.

## Send your own request

Save this single-line object as `requests.jsonl`:

```json
{"state":{"ticket":"Login fails for all users"},"questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"},"owner":{"type":"choice","instructions":"Choose the owner","criteria":{"app":"Application team","infra":"Infrastructure team"}},"severity":{"type":"score","instructions":"Rate severity","criteria":["low","medium","high"]}}}
```

Run it through the same model:

```sh
./target/release/jet judge \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input requests.jsonl \
  --output results.jsonl
```

The names `urgent`, `owner`, and `severity` are preserved in the response. Choice results use the keys supplied in `criteria`; score results use a zero-based weighted index.

## Next steps

- Read [Requests and responses](./requests-and-responses) for every question and result shape.
- See [Build and advanced usage](../advanced) for GPU backends, performance controls, thinking, and multimodal input.
- Use the [Rust API](./rust-api) when Jet should run inside your process.
