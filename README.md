# Jet

Jet is a local CPU decisions engine. It scores a fixed set of structured candidates with
teacher forcing instead of generating free-form text.

The first release supports `noul`, `choice`, and zero-based `score` questions through a Rust API
and a JSONL command-line interface. It intentionally does not provide HTTP serving, GPU/NPU
execution, thinking mode, sampling, or free-text generation.

## Build

Initialize the pinned llama.cpp submodule and install the toolchain:

```sh
git submodule update --init --recursive
mise install
mise exec -- cargo build --workspace
```

LLVM is downloaded directly from the official LLVM GitHub release assets through mise's HTTP
backend. No conda environment is required.

Download the pinned Qwen3 test model:

```sh
./scripts/download-test-model.sh
```

## CLI

```sh
mise exec -- cargo run -p jet-cli -- judge \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --model-id qwen/qwen3-0.6b-q8_0 \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

Each non-empty input line is one request. The request deliberately has no `model` field:

```json
{"state":{"ticket":"Login fails"},"questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"},"owner":{"type":"choice","instructions":"Choose the owner","criteria":{"app":"Application team","infra":"Infrastructure team"}},"severity":{"type":"score","instructions":"Rate severity","criteria":["low","medium","high"]}}}
```

Successful lines contain only `model`, `answers`, and `usage`. Failed lines have the shape
`{"error":{"code":"...","message":"..."}}`; Jet preserves input order and exits non-zero if
any line fails.

See [docs/scoring.md](docs/scoring.md) for scoring semantics and
[docs/development.md](docs/development.md) for validation commands.
