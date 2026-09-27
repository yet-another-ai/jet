# Jet

Jet is a local decisions engine. It scores a fixed set of structured candidates with teacher
forcing instead of generating free-form text. The default CPU-only build runs on CPU. Optional
CUDA, Metal, and Vulkan backends offload the model, KV cache, and supported operations to a local
GPU.

Jet supports `noul`, `choice`, and zero-based `score` questions through a Rust API and a JSONL
command-line interface. It does not provide HTTP serving, NPU execution, or general free-text
generation. An optional bounded thinking stage can run before candidate scoring when the model's
embedded chat template exposes a supported reasoning protocol.

JSON is the external API, not the model prompt. Jet renders each question through a Tera template
into human-readable context, instruction, and semantic candidate sections. It scores those semantic
answers and maps the winner back to the original boolean, choice key, or score index.

## Build

The English technical report template lives in [paper/](paper/README.md). Install its
LaTeX toolchain with `mise install tinytex`, then run `mise run paper` to generate
`paper/build/main.pdf`.

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

### Backend selection

The CLI defaults to `--backend auto`. It selects the first available backend compiled into the
binary in this order: CUDA, Metal, Vulkan, CPU. Use `--backend cuda`, `--backend metal`,
`--backend vulkan`, or `--backend cpu` to require a specific backend. An explicit GPU selection
fails when that feature or a matching device is unavailable. `EngineConfig::auto` provides the same selection for Rust
callers; `EngineConfig::cpu`, `cuda`, `metal`, and `vulkan` select explicitly.

Build with CUDA and Vulkan together using `--features cuda,vulkan`. On a machine with both backends,
the `auto` choice uses CUDA. Model loading binds the selected GPU device explicitly, so a
Vulkan request cannot silently use a CUDA device. The initial CUDA integration selects one GPU.

### Metal

On macOS, build with Xcode Command Line Tools and the opt-in Metal feature. The initial target is
Apple Silicon; confirm the minimum macOS deployment version on the target Mac.

```sh
mise exec -- cargo build --release -p jet-cli --features metal,vision
./target/release/jet judge --backend metal \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input tests/fixtures/decisions.jsonl --output -
```

The build statically links llama.cpp's Metal backend and embeds its shader source. Metal compiles
that source at runtime, so the first model load can take longer. `--backend auto` uses Metal when
its feature and a device are available; `--backend cpu` forces CPU execution. Metal uses the same
`--gpu-layers`, `--cpu-moe-layers`, and `--no-mmap` options as the other GPU backends. Mac validation
commands are in [development.md](docs/development.md#metal-validation).

On an Apple Silicon Mac, `mise run package:metal` builds the `metal,vision` release CLI and
produces `dist/jet-metal-macos-arm64.tar.gz` with a SHA-256 file.

### CUDA

Install the CUDA Toolkit with `nvcc` and a compatible NVIDIA driver. The CUDA feature builds
llama.cpp's CUDA backend from the pinned submodule:

```sh
mise exec -- cargo run -p jet-cli --features cuda -- judge \
  --backend cuda \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --input tests/fixtures/decisions.jsonl \
  --output -
```

Set `JET_CUDA_ARCHITECTURES` to a CMake CUDA architecture list when building for a different GPU
or on a machine without a GPU, for example `86;89`. CMake discovers the Toolkit using its
standard `CUDAToolkit_ROOT`, `CUDA_PATH`, and `CUDACXX` inputs. To produce a Windows archive with
both GPU backends and the matching CUDA runtime DLLs, run:

```powershell
$env:CUDAToolkit_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0'
$env:CARGO_TARGET_DIR = 'E:\jet-target'
mise run package:cuda
```

The archive is written to `dist/jet-cuda-windows-x64.zip`. The task inspects the executable's
imports and bundles cuBLAS, cuBLASLt, cudart, and their required CUDA Toolkit DLLs. The CLI
delay-loads CUDA imports so a host without an NVIDIA driver can still select Vulkan or CPU.
Jet links CUDA Runtime statically, while bundling cudart for cuBLAS's runtime needs. The NVIDIA
driver must still be installed for CUDA execution. To
package CUDA without Vulkan, invoke `pwsh -File scripts/package-cuda.ps1 -Features cuda`.

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

Control weight placement with `--gpu-layers N` (GPU backends default to all layers). Layers beyond
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

CPU preparation reuses the native chat template and tokenizer buffer. With disabled thinking on
Vulkan hybrid models, a bounded worker prepares upcoming requests while scoring the current request;
`--no-preparation-pipeline` disables this overlap for comparisons. Scoring remains serial to
preserve recurrent-state behavior. `--batch-requests` controls the input chunk, not the native
inference batch.

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

### Multimodal decisions

Build with the `vision` feature and download the Qwen3.6 language model and its matching visual
projector. The downloader pins both files by revision and SHA-256:

```sh
./scripts/download-accuracy-models.sh --qwen36-vision
mise exec -- cargo run -p jet-cli --features vision -- judge-multimodal \
  --model-path models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --mmproj-path models/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf \
  --input - --output -
```

Send one JSON object per line. `state` is optional and defaults to `null`. `source` accepts Base64
image bytes or a local path; PNG and JPEG are supported. Paths are relative to the input JSONL
file's directory, or the current directory when reading stdin. The Rust `Engine::decide_multimodal`
API accepts image bytes directly.

```json
{"request_id":"frame-1","state":{"goal":"reach the exit"},"images":[{"id":"screen","source":{"type":"path","media_type":"image/png","path":"tests/fixtures/vision/red.png"}}],"questions":{"action":{"type":"choice","instructions":"Choose the best next action based on the image","criteria":{"left":"Move left","right":"Move right","wait":"Wait"}}}}
```

`judge-multimodal` keeps the model loaded and writes and flushes one response for each input line,
even while stdin remains open. A response echoes `request_id`, returns the existing answer types,
and reports total, text, and image input tokens. An invalid image fails its request; later lines
still run. The initial implementation requires disabled thinking. Image count, encoded bytes,
decoded pixels, and visual token limits can be set with the `--max-images`, `--max-image-bytes`,
`--max-image-pixels`, and `--image-max-tokens` options. Context occupancy counts visual embeddings
as well as text tokens. Use `--backend cuda` with `--features 'vision,cuda'` or `--backend vulkan`
with `--features 'vision,vulkan'` when the corresponding SDK
and device are available.

The [Doom demo](demos/doom/README.md) connects this interface to a running game: 160×100 frames,
nine discrete actions, and asynchronous inference capped at two requests per second. It uses
ViZDoom's bundled Freedoom2 assets by default and supports your own Doom IWAD.

See [docs/scoring.md](docs/scoring.md) for scoring semantics and
[docs/development.md](docs/development.md) for validation commands. The reproducible BoolQ/MMLU
accuracy workflow is documented in [tests/accuracy/README.md](tests/accuracy/README.md); downloaded
datasets and derived evaluation files are excluded from Git.
