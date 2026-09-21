# Jet

Jet is a local decisions engine. It scores a fixed set of structured candidates with teacher
forcing instead of generating free-form text. CPU execution is the default; an optional Vulkan
backend offloads the model, KV cache, and supported operations to a local GPU.

Jet supports `noul`, `choice`, and zero-based `score` questions through a Rust API and a JSONL
command-line interface. It does not provide HTTP serving, NPU execution, or general free-text
generation. An optional bounded thinking stage can run before candidate scoring when the model's
embedded chat template exposes a supported reasoning protocol.

JSON is the external API, not the model prompt. Jet renders each question through a Tera template
into human-readable context, instruction, and semantic candidate sections. It scores those semantic
answers and maps the winner back to the original boolean, choice key, or score index.

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

### Vulkan

Vulkan is opt-in so ordinary CPU builds keep their existing dependencies. Install a Vulkan loader
and development headers, SPIR-V headers, and `glslc` (from shaderc or the Vulkan SDK), then build
with the `vulkan` feature and select the backend explicitly:

```sh
mise exec -- cargo run -p jet-cli --features vulkan -- judge \
  --backend vulkan \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

Model loading fails with a clear error if `--backend vulkan` is used without the feature or without
a usable Vulkan GPU. A successful run logs the selected `Vulkan0` device and the number of model
layers offloaded before emitting JSONL results.

Control weight placement with `--gpu-layers N` (Vulkan defaults to all layers). Layers beyond
that GPU allocation execute on CPU. For MoE models, `--cpu-moe-layers N` keeps the routed experts
in the first N transformer layers on CPU while leaving attention and shared experts eligible for
GPU execution. These are layer counts, not percentages of bytes. For example, this Q4 placement
keeps most weights on the development machine's 16 GiB Arc A770:

```sh
./target/release/jet judge \
  --backend vulkan \
  --model-path models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-id qwen/qwen3.6-35b-a3b-q4_k_m \
  --cpu-moe-layers 11 \
  --max-sequences 2 \
  --micro-batch 256 \
  --max-output-rows 256 \
  --threads 8 \
  --no-mmap \
  --thinking disabled \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

Explicit placement disables automatic GPU reassignment of CPU weight operations. CPU-resident
experts compute on CPU; this does not stream selected experts into GPU memory for each token.
The default preserves llama.cpp's automatic loading policy, which uses mmap when the backends
support it; the operating system caches mapped file pages in RAM.
The model file can live on an SSD, but disk paging is not equivalent to RAM-resident
inference. `--no-mmap` instead loads weights into allocated memory. The Rust equivalents are
`EngineConfig::gpu_layers`, `cpu_moe_layers`, and `use_mmap`.
CPU graph segments reuse a scorer-owned worker pool across requests and between scoring and
thinking contexts. Workers sleep between segments rather than being recreated at each CPU/GPU
transition. See the [profiling results](docs/performance.md) for the measured effect.
Placement must also leave space for caches and computation buffers; increasing context length,
sequence slots, or enabling thinking changes the memory requirement. See the
[Qwen3.6 measurements](docs/accuracy.md#qwen36-q4-cpuvulkan-offload) for the tested settings.

For hybrid models such as Qwen3.5, batched execution prefills each question once and reuses its
unchanged prefix while scoring candidate continuations serially. Two sequence slots are sufficient
for this path (`--max-sequences 2`). Cache cleanup resets sequence metadata without synchronously
zeroing the full KV and recurrent-state buffers.

## CLI

```sh
mise exec -- cargo run -p jet-cli -- judge \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --model-id qwen/qwen3-0.6b-q8_0 \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

Enable thinking when the model supports it, with at most 256 reasoning-body tokens per question:

```sh
mise exec -- cargo run -p jet-cli -- judge \
  --model-path model.gguf \
  --model-id local/model \
  --thinking auto \
  --thinking-tokens 256 \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

`--thinking disabled` uses the template's direct-answer path. `auto` falls back to direct scoring
for ordinary non-reasoning models. `required` rejects a template that does not expose a usable
reasoning end marker and final-answer continuation.

With the Vulkan backend, bounded thinking uses a separate single-sequence context that exposes one
logits row to the CPU sampler. Both contexts use the same configured weight placement. Once
thinking closes, batched candidate scoring runs in the main context, with full-vocabulary softmax
and target-token gathering on the backend holding the output weights.

Add `--timings /tmp/jet-timings.json` to write cumulative engine stage timings and workload counters
to a separate JSON file. The path must differ from the input and output paths; `-` is not supported.
This leaves response JSONL and `usage` unchanged. Collection is disabled by default. See the
[timing definitions](docs/development.md#stage-timings) when comparing model loading, prefill,
candidate scoring, and total scorer time.

Each non-empty input line is one request. The request deliberately has no `model` field:

```json
{"state":{"ticket":"Login fails"},"questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"},"owner":{"type":"choice","instructions":"Choose the owner","criteria":{"app":"Application team","infra":"Infrastructure team"}},"severity":{"type":"score","instructions":"Rate severity","criteria":["low","medium","high"]}}}
```

Successful lines contain only `model`, `answers`, and `usage`. Failed lines have the shape
`{"error":{"code":"...","message":"..."}}`; Jet preserves input order and exits non-zero if
any line fails.

See [docs/scoring.md](docs/scoring.md) for scoring semantics and
[docs/development.md](docs/development.md) for validation commands. The reproducible BoolQ/MMLU
accuracy workflow is documented in [tests/accuracy/README.md](tests/accuracy/README.md); downloaded
datasets and derived evaluation files are excluded from Git.
