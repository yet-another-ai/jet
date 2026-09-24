# Development

Jet pins llama.cpp as a Git submodule at commit
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (the v0.4.1 line). The default build is CPU-only and
disables OpenMP, BLAS, CUDA, Metal, Vulkan, HIP, SYCL, and RPC. The opt-in `cuda` and `vulkan`
Cargo features independently enable those llama.cpp backends. The CLI defaults to automatic
selection in CUDA, Vulkan, CPU order among compiled features with available devices.

Bindings matching that commit are checked into `jet-llama-sys`. Normal builds do not load
libclang. To deliberately regenerate them after changing the pin:

```sh
LIBCLANG_PATH="$(mise where http:llvm)/lib" \
  mise exec -- cargo check -p jet-llama-sys --features generate-bindings
```

The test model is `Qwen/Qwen3-0.6B-GGUF` revision
`23749fefcc72300e3a2ad315e1317431b06b590a`, file `Qwen3-0.6B-Q8_0.gguf`, with SHA-256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.

llguidance remains disabled. Bounded thinking uses llama.cpp's protocol-aware reasoning-budget
sampler and the model template's declared end markers; candidate scoring itself does not execute a
grammar.

Run the fast checks:

```sh
mise exec -- cargo fmt --check
mise exec -- cargo clippy --workspace --all-targets -- -D warnings
mise exec -- cargo test --workspace
```

After downloading the model, run ignored model and CLI tests:

```sh
JET_MODEL_PATH="$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  mise exec -- cargo test --workspace -- --ignored --test-threads=1
```

To compile and run the Vulkan smoke test, install the Vulkan loader and development headers,
SPIR-V headers, and `glslc`, then run:

```sh
JET_MODEL_PATH="$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  mise exec -- cargo test -p jet-engine --features vulkan \
  qwen_vulkan_smoke -- --ignored --nocapture
```

The Qwen3.5 regression test compares serial prefix reuse against independent reference scoring,
including candidate reordering, mixed continuation lengths, gather-width growth, and reuse across
questions with only two sequence slots:

```sh
JET_QWEN35_MODEL_PATH="$PWD/models/Qwen3.5-0.8B-Q8_0.gguf" \
  mise exec -- cargo test -p jet-engine --features vulkan \
  qwen35_hybrid_batch_matches_reference -- --ignored --nocapture
```

On NixOS, `nix shell` puts `glslc` on `PATH`, but CMake also needs the split header and loader
outputs. This self-contained invocation supplies them:

```sh
vulkan_headers="$(nix build --no-link --print-out-paths nixpkgs#vulkan-headers)"
vulkan_loader="$(nix build --no-link --print-out-paths nixpkgs#vulkan-loader)"
spirv_headers="$(nix build --no-link --print-out-paths nixpkgs#spirv-headers)"
CMAKE_PREFIX_PATH="$vulkan_headers:$vulkan_loader:$spirv_headers" \
CXXFLAGS="-I$spirv_headers/include" \
JET_MODEL_PATH="$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  nix shell nixpkgs#shaderc nixpkgs#vulkan-headers nixpkgs#vulkan-loader \
    nixpkgs#spirv-headers --command \
  mise exec -- cargo test -p jet-engine --features vulkan \
    qwen_vulkan_smoke -- --ignored --nocapture
```

On NixOS, the upstream prebuilt LLVM archive expects the host's zlib shared library. Regeneration
therefore needs a shell that exposes zlib in `LD_LIBRARY_PATH`. Regular builds do not need this.

On Windows/MSVC, install a Vulkan SDK and use a short Cargo target directory to avoid generated
shader paths exceeding compiler limits. PowerShell 7 example (adjust the SDK path):

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0'
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
$env:CARGO_TARGET_DIR = 'E:\jet-target'
mise exec -- cargo build --release -p jet-cli --features vulkan
```

For a CUDA build, install the CUDA Toolkit, then run:

```sh
JET_CUDA_ARCHITECTURES="86;89" mise exec -- cargo build -p jet-cli --features cuda
```

`JET_CUDA_ARCHITECTURES` forwards to CMake's `CMAKE_CUDA_ARCHITECTURES`; set it for a headless
build or a known deployment GPU. CMake discovers the compiler and libraries with `CUDACXX`,
`CUDAToolkit_ROOT`, or `CUDA_PATH`. The static GGML build links CUDA dependencies using the
resolved library paths in CMake's cache. To test automatic selection on a machine with both
toolchains, build with `--features cuda,vulkan` and run the same model once each with
`--backend auto`, `cuda`, `vulkan`, and `cpu`. Confirm the selected device in the native log.

For Windows distribution, set `CUDAToolkit_ROOT` and run `mise run package:cuda`. This builds
the release CLI with `cuda,vulkan`, scans ordinary and delay-loaded PE imports with `llvm-readobj`,
copies matching CUDA Toolkit DLLs and their transitive DLL dependencies beside `jet.exe`,
and writes `dist/jet-cuda-windows-x64.zip` with SHA-256 hashes. The task fails if a required CUDA
DLL is absent from the selected Toolkit. Use `pwsh -File scripts/package-cuda.ps1 -Features cuda`
for a CUDA-only binary; `-Features cuda,vulkan,vision` also includes vision. CUDA execution on the
target machine needs a compatible NVIDIA driver but does not need the CUDA Toolkit. The Windows
CLI delay-loads cuBLAS, cuBLASLt, and the CUDA driver so a missing NVIDIA driver does not prevent
`--backend auto`
from reaching Vulkan or CPU during startup. On Linux, the current CUDA build links the Toolkit
libraries statically, leaving the driver supplied by the host.
Rust applications embedding `jet-engine` on Windows must also delay-load their final executable's
CUDA imports if they need to start without an NVIDIA driver; the Jet CLI configures this itself.

With `JET_MODEL_PATH` set, run the ignored CUDA smoke test with
`mise exec -- cargo test -p jet-engine --features cuda qwen_cuda_smoke -- --ignored --nocapture`.
Use `JET_QWEN35_MODEL_PATH` and `qwen35_cuda_hybrid_batch_matches_reference` for the CUDA
hybrid/gather regression. Run GPU model tests serially to avoid memory interference.

The native build locates Visual Studio's `Release` library subdirectories and links `advapi32`.
No manual static-library copying or linker flags are required.

Preparation regressions use the same Qwen3.5 fixture:

```sh
JET_QWEN35_MODEL_PATH="$PWD/models/Qwen3.5-0.8B-Q8_0.gguf" \
  mise exec -- cargo test -p jet-engine --features vulkan preparation -- --ignored --test-threads=1
```

These compare cached/uncached rendering and tokenization, synchronous/pipelined preparation,
mixed request errors and order, and worker cancellation/reuse. The existing hybrid regression
above compares serial/reference scoring. Run GPU model tests serially to avoid memory and timing
interference.

## Stage timings

The CLI accepts `--timings PATH` to write one JSON document after processing its input, including
when individual request lines return errors. It preserves the response JSONL format. Use a
separate file path: `-` and paths identical to `--input` or `--output` are rejected. The Rust API
enables the same collection with `EngineConfig::collect_timings = true` and exposes cumulative
totals through `Engine::timings()`. Collection is disabled by default.

For example, after preparing the accuracy dataset and building a Vulkan release binary:

```sh
./target/release/jet judge \
  --backend vulkan \
  --model-path models/Qwen3.5-0.8B-Q8_0.gguf \
  --model-id qwen/qwen3.5-0.8b-q8_0 \
  --max-sequences 2 \
  --thinking disabled \
  --input tests/accuracy/generated/requests.jsonl \
  --output /tmp/jet-responses.jsonl \
  --timings /tmp/jet-timings.json
```

All duration fields are cumulative wall-clock milliseconds measured inside the native scorer.
They include host work and backend waits, rather than individual GPU kernel durations:

| Field | Scope |
| --- | --- |
| `load_ms` | Backend initialization, model loading, and inference-context setup. |
| `prepare_ms` | Native chat-template rendering, prompt/candidate tokenization, and scorer validation; excludes thinking and its cache resets. |
| `prepare_wait_ms` | Consumer wait for the bounded CPU preparation worker; zero on the synchronous path. |
| `thinking_ms` | Bounded reasoning generation, including its prompt decode and sampling; excludes explicit cache resets. |
| `prefill_ms` | Scoring-prompt decoding and reading the final prompt output, which scores candidate first tokens. |
| `candidate_ms` | Decoding and scoring candidate continuations after the prompt. |
| `cache_ms` | Explicit sequence reset, removal, and prefix-copy API calls. |
| `batch_ms` | Complete scorer-batch processing, including preparation, thinking, cache management, scoring, and bookkeeping; excludes loading. |

`batch_ms` is an enclosing total, so do not add it to the other scoring phases. With the preparation
pipeline, `prepare_ms` is worker time excluding queue backpressure and overlaps GPU scoring; phase
sums may exceed `batch_ms`. `prepare_wait_ms` is already included in `batch_ms`, not extra work.
Phase subtotals can also be smaller because worker setup, scheduling, and bookkeeping are included
only in the total. Deferred
recurrent-state copies execute during decoding and are charged to that decode phase, rather than
to `cache_ms`. Timing collection relies on the synchronization already needed to read scoring
outputs; it does not add a wait between prompt chunks.

These measurements exclude JSONL parsing and writing, outer Tera question rendering, and response
normalization. Measure process wall time separately when startup and complete request handling
matter; `load_ms + batch_ms` is not a replacement for it.

The document also contains these cumulative counters:

- `batches`: calls into the native scoring batch; input chunks without scoring jobs do not count.
- `jobs`: questions submitted to the scorer, including those that fail scorer validation.
- `prefill_count`: scoring prompt prefills, counting repeated prefills in reference mode or split waves.
- `prefill_tokens`: physical scoring-prompt tokens submitted for decoding, including frozen reasoning when enabled and repeated prefixes when needed.
- `candidate_tokens`: continuation tokens submitted for decoding. A successful candidate of length `N` requires `N - 1` such tokens because its first token is scored by the final prompt output.
- `decode_calls`: calls to llama.cpp decode across scoring and thinking.

The two token counters exclude thinking-generation decoding; that work is measured by
`thinking_ms` and contributes to `decode_calls`. These workload counters are separate from response
`usage`, which continues to count one original prompt per question, every scored candidate token,
and generated thinking tokens. Use identical input, model, execution mode, and batching settings
when comparing performance.

## Reproducible model benchmarks

For the Qwen3.6 CPU/Vulkan profile, measured GPU busy time, CPU thread-pool
diagnosis, and isolated optimization comparisons, see [performance.md](performance.md).

`scripts/benchmark-model.py` runs a built CLI against the prepared accuracy workload and saves
responses, accuracy reports, stage timings, process wall times, commands, and failure logs. It
defaults to automatic backend selection, disabled thinking, eight requests per input batch, and
two fresh processes. Set `--backend vulkan` to reproduce the Vulkan-only comparisons.
Choose a new output directory for each configuration:

```sh
python3 scripts/benchmark-model.py \
  --binary target/release/jet \
  --model-path models/Qwen3.5-2B-Q8_0.gguf \
  --model-id qwen/qwen3.5-2b-q8_0 \
  --output-dir tests/accuracy/generated/benchmark-2b \
  --runs 2 \
  --extra-args --max-sequences 2
```

Use `--limit 8 --runs 1` before `--extra-args` for a short smoke run. The limit selects the same
first rows from both requests and gold labels. `--input`, `--gold`, `--backend`, and
`--batch-requests` are explicit script options; other CLI settings follow `--extra-args`, which
must come last. The script stops after the first failed run and preserves its artifacts.

`provenance.json` records the Git revision and working-tree status, binary and dataset hashes,
model file sizes and modification times, and relevant environment overrides. A working-tree
patch, the harness, evaluator, and selected input/gold rows are also archived. Optional
`--model-provenance download.json` records supplied download metadata without independently
verifying it or rereading large model files. Split GGUF models must be passed as their first
shard; metadata for every shard is recorded.

`summary.json` reports whole-run averages and stage time per request. These are amortized costs,
not individual request latencies or P95: the CLI buffers output, including with
`--batch-requests 1`. There is no explicit warmup or cache eviction between runs, and wall time
includes loading and shutdown. Compare models with the same workload and settings.
