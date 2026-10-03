---
title: Rust API
description: Embed Jet with the jet-core protocol types and jet-engine inference API.
---

# Rust API

Jet separates its public protocol from model execution:

- `jet-core` contains request, question, answer, usage, and error types.
- `jet-engine` loads a GGUF model and evaluates single or batched requests.
- `jet-cli` wraps both crates in the JSONL command-line interface.

## Load an engine

Choose an automatic or explicit backend with `EngineConfig`, then load the model once:

```rust
use jet_core::{DecisionRequest, Result};
use jet_engine::{Engine, EngineConfig};

fn decide(request: DecisionRequest) -> Result<()> {
    let config = EngineConfig::auto(
        "models/Qwen3-0.6B-Q8_0.gguf",
        "qwen/qwen3-0.6b-q8_0",
    );
    let mut engine = Engine::load(config)?;
    let response = engine.decide(request)?;
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
```

`EngineConfig::cpu`, `cuda`, `metal`, and `vulkan` require a specific backend. GPU constructors still need the matching Cargo feature on `jet-engine`.

## Evaluate a batch

`Engine::decide_batch` accepts a slice and returns one result for every input request:

```rust
let results = engine.decide_batch(&requests);
for result in results {
    match result {
        Ok(response) => println!("{}", serde_json::to_string(&response)?),
        Err(error) => eprintln!("{error}"),
    }
}
```

The method keeps failures attached to their individual requests. Reuse the same `Engine` so the model and inference resources remain loaded.

## Configuration

The main tuning fields on `EngineConfig` correspond to CLI options:

| Rust field | CLI option | Purpose |
| --- | --- | --- |
| `gpu_layers` | `--gpu-layers` | Limit the transformer/output layers placed on a GPU. |
| `cpu_moe_layers` | `--cpu-moe-layers` | Keep routed experts from early MoE layers on CPU. |
| `use_mmap` | `--no-mmap` | Control memory-mapped model loading. |
| `context_tokens_per_sequence` | `--context-tokens` | Set the context capacity for each sequence. |
| `token_batch` | `--token-batch` | Set the logical token batch size. |
| `micro_batch` | `--micro-batch` | Set the physical evaluation batch size. |
| `max_sequences` | `--max-sequences` | Reserve concurrent native sequence slots. |
| `execution_mode` | `--execution` | Select batched or reference evaluation. |
| `thinking` | `--thinking*` | Configure the optional bounded reasoning stage. |
| `collect_timings` | `--timings` | Collect cumulative stage timings and counters. |

See [Build and advanced usage](../advanced) for backend requirements and how these controls affect hybrid CPU/GPU execution.

## Multimodal API

With the `vision` feature, set `EngineConfig::vision` using `VisionConfig::new(mmproj_path)` and call `Engine::decide_multimodal`. The Rust API accepts decoded image bytes in `ImageInput`; the CLI additionally supports file paths and Base64 sources. The same `Question` and `Answer` types are used for text and image decisions.
