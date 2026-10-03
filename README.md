# Jet

Jet makes local decisions with a language model. Give it some context and a fixed set of possible answers; it scores the candidates and returns a structured result instead of generating free-form text. Use it from a JSONL command-line interface or a Rust API.

Jet runs on CPU by default. Optional CUDA, Metal, and Vulkan backends can offload inference to a local GPU. With the `vision` feature and a compatible model, it can also answer questions about images.

## What you can do

- **Choose an answer:** Select a key from named candidates, such as a support team or next action.
- **Make a yes/no decision:** Use a `noul` question for a boolean result.
- **Rate an item:** Use a `score` question to select a zero-based index from an ordered scale.
- **Process a stream:** Send one JSON request per line and receive one JSON result per line, in order.
- **Use local models:** Run GGUF models on CPU or an available GPU, without an HTTP service.

Jet uses the model's chat template to present the context and candidate answers in natural language. The JSON request is an API format, not text passed directly to the model.

## Quick start

You'll need [mise](https://mise.jdx.dev/) 2026.10.0 or newer and a platform toolchain capable of building Rust and the bundled llama.cpp. The following steps build the CPU version and download the pinned Qwen3 example model:

```sh
git clone --recurse-submodules https://github.com/yet-another-ai/jet.git
cd jet
mise install
./scripts/download-test-model.sh
mise exec -- cargo build --release -p jet-cli
```

Run the included example requests:

```sh
./target/release/jet judge \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

`--output -` writes JSONL to standard output. Each successful response contains `model`, `answers`, and `usage`; a failed request produces an `error` object. Jet exits with a nonzero status if any line fails.

### Send your own request

Save this as `requests.jsonl` (one JSON object per line):

```json
{"state":{"ticket":"Login fails for all users"},"questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"},"owner":{"type":"choice","instructions":"Choose the owner","criteria":{"app":"Application team","infra":"Infrastructure team"}},"severity":{"type":"score","instructions":"Rate severity","criteria":["low","medium","high"]}}}
```

Then run:

```sh
./target/release/jet judge \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input requests.jsonl \
  --output results.jsonl
```

The question names (`urgent`, `owner`, `severity`) appear in the `answers` object. Choice answers use the keys you supplied in `criteria`; score answers use zero-based indexes.

## GPU support

| Backend | Build feature | Requirement |
| --- | --- | --- |
| CPU | Default | No GPU required |
| Metal | `metal` | Apple Silicon Mac and Xcode Command Line Tools |
| CUDA | `cuda` | NVIDIA driver and CUDA Toolkit |
| Vulkan | `vulkan` | Vulkan loader, development headers, SPIR-V headers, and `glslc` |

For example, to build and run on Metal:

```sh
mise exec -- cargo build --release -p jet-cli --features metal
./target/release/jet judge --backend metal \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input tests/fixtures/decisions.jsonl --output -
```

`--backend auto` is the default and selects an available backend compiled into the binary. Select a backend explicitly to require it; Jet reports an error if its feature or device is unavailable. See the [build and backend guide](docs/advanced.md) for CUDA and Vulkan commands, packaging, memory placement, and performance options.

## Images and other options

Build with `--features vision` to use `judge-multimodal` with a compatible GGUF model and visual projector. Requests can refer to local PNG or JPEG files or contain Base64 image data. See the [multimodal guide](docs/advanced.md#multimodal-decisions) for a complete request and command, and the [Doom demo](demos/doom/README.md) for an example that chooses game actions from frames.

Some models expose a supported reasoning protocol. `--thinking auto --thinking-tokens 256` enables a bounded reasoning stage before candidate scoring; the default is `--thinking disabled`. See the [thinking options](docs/advanced.md#cli) for details.

## Documentation

- [Documentation website](https://yet-another-ai.github.io/jet/)
- [Getting started](https://yet-another-ai.github.io/jet/guide/getting-started)
- [Build, GPU backends, CLI options, and multimodal usage](docs/advanced.md)
- [Rust API crates](crates/jet-engine/src/lib.rs)
- [Accuracy evaluation](tests/accuracy/README.md)
- [Qwen3.5-4B LoRA training and public datasets](training/README.md)
- [Technical report](paper/README.md)

## Scope

Jet is designed for structured decisions over a fixed candidate set. It does not provide general text generation, HTTP serving, or NPU execution. Multimodal decisions currently require disabled thinking.
