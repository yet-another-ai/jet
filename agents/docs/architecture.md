# Architecture

Jet scores fixed candidate answers with a local language model. JSON is the
external API shape; the model sees natural-language prompts rendered from that
API, not raw JSON.

## Crate ownership

| Crate | Owns | Must not own |
| --- | --- | --- |
| `jet-core` | Protocol types, errors, score normalization | llama.cpp, I/O, CLI parsing |
| `jet-engine` | Model load, prompt export, teacher-forced scoring, backends, optional vision | CLI argument parsing, packaging archives |
| `jet-llama-sys` | FFI, C++ glue, and build of vendored llama.cpp | Product protocol or CLI UX |
| `jet-cli` | JSONL CLI, feature-gated multimodal entry points | Scoring math or FFI details |
| `xtask` | Repository packaging tasks | Runtime inference behavior |

Dependency direction flows toward `jet-core`. `jet-cli` and packaging tools may
depend on `jet-engine`; `jet-core` must remain usable without a loaded model.

## Protocol

- Requests and responses are serializable structured values (`DecisionRequest`,
  `DecisionResponse`, and related types in `jet-core`).
- Question kinds are `noul`, `choice`, and zero-based `score`.
- Failures use typed `JetError` / `ErrorCode` values rather than free-form
  panics at API boundaries.
- Prompt rendering (Tera templates in `jet-engine`) is an implementation detail
  of scoring; public docs describe the JSON API users exchange.

## Backends and features

- Default builds are CPU-only.
- Optional Cargo features enable CUDA, Metal, Vulkan, and vision. Callers must
  select an available backend; missing features or devices fail explicitly.
- CI validates CPU Clippy/tests and CPU packaging. GPU toolkits, model
  downloads, and training/runtime benchmarks stay out of the default gate unless
  a maintainer explicitly expands that contract.

## Native boundary

- llama.cpp enters through `jet-llama-sys` and the pinned `vendor/llama.cpp`
  submodule.
- `unsafe` FFI and C++ glue stay in `jet-llama-sys` / narrow engine adapters.
  Higher layers prefer safe Rust APIs and explicit error returns.
- Changing the submodule pin, FFI surface, or CPU portability flags
  (`JET_CPU_PORTABLE`) is an architecture-sensitive change; update owning docs
  and CI notes in the same PR when behavior changes.

## Documentation owners

| Concern | Owner |
| --- | --- |
| User-facing CLI and API behavior | `docs/` (VitePress) |
| Current crate and backend rules | this document |
| Implementation and review rules | `agents/docs/engineering-standards.md` |
| Durable decision rationale | `agents/docs/adr/` when a maintainer requests a record |
| CI matrix, caches, Gate | `docs/ci.md` |
