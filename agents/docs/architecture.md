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
| `jet-server` | Actix HTTP adapter, admission control, dedicated inference thread, server lifecycle | Scoring math or FFI details |
| `jet-cli` | JSONL CLI, `serve` configuration, feature-gated multimodal entry points | Scoring math or FFI details |
| `xtask` | Repository packaging tasks | Runtime inference behavior |

Dependency direction flows toward `jet-core`. `jet-cli` depends on `jet-server`
and `jet-engine`; `jet-server` depends on `jet-engine`. Packaging tools may
depend on `jet-engine`; `jet-core` must remain usable without a loaded model.

## Protocol

- Requests and responses are serializable structured values (`DecisionRequest`,
  `DecisionResponse`, and related types in `jet-core`).
- Question kinds are `noul`, `choice`, and zero-based `score`.
- Failures use typed `JetError` / `ErrorCode` values rather than free-form
  panics at API boundaries.
- Prompt rendering (Tera templates in `jet-engine`) is an implementation detail
  of scoring; public docs describe the JSON API users exchange.

## HTTP service

`jet serve` exposes the `noul`, `choice`, and `score` subset of OpenRouter
Decisions at `POST /api/alpha/decisions`. HTTP wire types live in `jet-core`;
validation and transport errors are adapted by `jet-server` without changing
the JSONL protocol or candidate scoring semantics. Each HTTP request names the
single loaded model. This is not a general OpenRouter proxy or chat API.

Vision builds can enable image input by configuring a matching projector with
`--mmproj-path`; this requires disabled thinking. Image input follows the
OpenRouter Decisions `state` array format: direct `image_url` entries contain
PNG/JPEG base64 data URLs and may specify `detail` as `auto`, `low`, or `high`.
The detail hint is accepted without changing processing. The adapter normalizes
image bytes before admission and preserves the order of text and image blocks
when invoking the engine. Base64 is not rendered into model prompts. Nested
image-shaped JSON objects retain ordinary text semantics. HTTP does not fetch
remote images or expose server file paths. Builds without vision, or servers
without a projector, explicitly reject image blocks.

The body limit defaults to 1 MiB for text-only serving and 16 MiB with a
projector, unless overridden. Per-image byte, pixel, and token limits and the
image count limit remain separate from the aggregate JSON body limit; base64
expansion counts toward that body limit. `jet-cli/vision` enables both engine
and server vision support.

The inference thread creates, owns, and destroys `Engine` on that thread.
Actix workers retain admission state and a bounded task sender, and
asynchronously await replies. Text and image tasks share this worker and its
admission controls; multimodal tasks use the existing visual scoring path.
The model and native contexts remain loaded between requests, and startup warmup
finishes before the service accepts traffic. HTTP worker count is independent
of inference parallelism; there is one inference execution thread.

Requests run immediately without a cross-request batching delay. Candidate
batching within an engine request remains available. Queueing is enabled by
default, with room for one waiting request; a full queue returns `429` without
waiting for admission. `--no-queue` admits at most one request, including the
handoff to the worker, and immediately rejects further requests with `429`.
It conflicts with an explicit `--queue-capacity` value.
Expired or abandoned work is skipped before inference starts. A `504` deadline
or cancelled HTTP handler does not interrupt an already-running native inference
call or release its admission slot before the call completes.

Successful HTTP responses add `inference_ms`, measured with a monotonic clock
around the individual engine decision call. This includes prompt preparation,
tokenization, scoring, normalization, and, for image requests, native image
decoding and vision encoding. It excludes queueing, HTTP parsing, base64
normalization, model loading, and network transit. It is not pure GPU time or
a cumulative `EngineTimings` counter. Responses also add `queue_ms`, measured
from accepted submission after validation and image normalization to the start
of engine inference. This includes the cross-thread handoff, so a small
positive value is possible with
`--no-queue`. Error responses contain neither timing field. See the
[HTTP server guide](../../docs/guide/http-server.md) for endpoint, authentication,
and compatibility details.

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
