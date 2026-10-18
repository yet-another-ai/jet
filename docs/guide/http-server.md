---
title: HTTP server
description: Serve local text and image decisions with Actix, a persistent model, optional bounded queueing, and per-request inference and queue timing.
---

# HTTP server

`jet serve` keeps one model loaded and serves decisions over HTTP. It uses
the basic request and response fields of the
[OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
and adds `inference_ms` and `queue_ms` to each successful response. It supports
`noul`, `choice`, and `score`, with optional PNG/JPEG input in vision builds;
it does not implement the rest of the OpenRouter platform.

## Start the server

The standard CPU CLI and GPU builds include server mode. After following
[Getting started](./getting-started), run:

```sh
./target/release/jet serve \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --model-id local/jet \
  --backend auto \
  --bind 127.0.0.1:8080
```

Jet loads and warms up the model before accepting traffic. A dedicated thread
owns the engine, model, and inference contexts for the lifetime of the server.
Actix handles HTTP asynchronously while that thread scores requests one at a time.

`serve` shares the model, backend, context, thread, execution, and thinking
options of `judge`. `--thinking disabled` is the default. GPU backends require
the matching build feature; see [Build and advanced usage](../advanced).

Server configuration:

| Option | Default | Meaning |
| --- | --- | --- |
| `--bind` | `127.0.0.1:8080` | Address on which to listen. |
| `--model-id` | `qwen/qwen3-0.6b-q8_0` | Model identifier clients must supply; examples here explicitly use `local/jet`. |
| `--queue-capacity` | `1` | Maximum waiting requests when queueing is enabled, excluding the running request. |
| `--no-queue` | Off | Reject new requests with `429` while another request is admitted or running. |
| `--request-timeout-ms` | `30000` | Deadline for queueing and inference after request validation. |
| `--http-workers` | `2` | Actix HTTP workers; does not increase inference concurrency. |
| `--max-body-bytes` | `1048576` (1 MiB), or `16777216` (16 MiB) with `--mmproj-path` | Maximum total JSON request body size in bytes; an explicit value overrides either default. |

The queue capacity, timeout, worker count, and body limit must be positive.
Queueing is enabled by default. `--no-queue` cannot be combined with an explicit
`--queue-capacity` value.

### Enable image input

Build with `vision` and the backend feature you need, then load a compatible
vision-capable GGUF and its matching `mmproj` projector. For example:

```sh
mise exec -- cargo build --locked --release -p jet-cli --features cuda,vision
./target/release/jet serve \
  --model-path models/Qwen3.5-27B-Q4_K_M.gguf \
  --model-id local/jet \
  --mmproj-path models/mmproj-Qwen3.5-27B.gguf \
  --backend cuda \
  --thinking disabled \
  --context-tokens 16384 \
  --max-sequences 2
```

Replace the model and projector paths with your files; projector filenames vary
by distributor. Vision adds projector weights, image embeddings, and working
buffers to memory usage. Verify the model, context size, and image workload fit
your hardware. The model and projector are loaded once at startup. Without
`--mmproj-path`, the server remains text-only even in a vision build.

These options are available only with the `vision` build feature. All image
limits must be positive; setting an image limit explicitly requires
`--mmproj-path`. Vision serving requires `--thinking disabled`; `auto` and
`required` are rejected before loading the model.

| Option | Default | Meaning |
| --- | --- | --- |
| `--mmproj-path` | Unset | Matching projector path; enables image requests. |
| `--max-images` | `4` | Maximum images in one request. |
| `--max-image-bytes` | `10485760` (10 MiB) | Maximum PNG/JPEG file bytes per image after base64 decoding. |
| `--max-image-pixels` | `16777216` | Maximum decoded pixels per image. |
| `--image-max-tokens` | `1024` | Vision token budget per image. |

The total request body limit is independent of the per-image limits. Base64
adds approximately one third to the file size, and JSON and text also count
toward `--max-body-bytes`. The default 16 MiB body limit therefore does not
permit four 10 MiB images in one request.

## Send a decision request

Save the following as `request.json`:

```json
{
  "model": "local/jet",
  "state": {"ticket": "Login fails for all users"},
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Should this incident page the on-call engineer?"
    },
    "owner": {
      "type": "choice",
      "instructions": "Choose the team that should own this ticket",
      "criteria": {"app": "Application team", "infra": "Infrastructure team"}
    },
    "severity": {
      "type": "score",
      "instructions": "Rate incident severity",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

Send it with `Content-Type: application/json`:

```sh
curl http://127.0.0.1:8080/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

On Windows, use `curl.exe` from PowerShell 7. Successful responses use the same
answer shapes and normalized weights as the [JSONL protocol](./requests-and-responses).
For example, the following values illustrate the response format; they are not
measured results:

```json
{
  "model": "local/jet",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.91},
    "owner": {
      "type": "choice",
      "choice": "infra",
      "probabilities": {"app": 0.08, "infra": 0.92}
    },
    "severity": {
      "type": "score",
      "score": 1.78,
      "probabilities": {"0": 0.03, "1": 0.16, "2": 0.81}
    }
  },
  "usage": {"input_tokens": 128, "output_tokens": 9},
  "inference_ms": 12.34,
  "queue_ms": 0.08
}
```

`inference_ms` is the elapsed wall-clock time, in milliseconds, around this
request's engine decision call, measured with a monotonic clock. It includes
prompt construction, tokenization, model scoring, and result normalization.
For image requests it also includes native image decoding and vision encoding.
It excludes model loading, HTTP parsing, base64 normalization, queueing,
response serialization, and network transit. It is neither pure GPU kernel time
nor the cumulative engine timing reported by the JSONL CLI's `--timings` option.

`queue_ms` is the elapsed time, in milliseconds, from accepted submission after
request validation and base64 normalization to the start of engine inference,
also measured with a monotonic clock. It includes waiting behind earlier requests and the handoff to
the inference thread. It excludes HTTP parsing and validation. Even with
`--no-queue`, thread handoff can produce a small positive `queue_ms`. The sum of
these fields excludes response serialization and network transit, so it is not
the complete client-observed latency. Error responses do not include either
timing field.

## Send an image decision request

With vision enabled, use the
[OpenRouter multimodal Decisions format](https://openrouter.ai/docs/guides/community/multimodal-decisions):
place image blocks alongside text in the top-level `state` array. Jet preserves
the order of those text and image blocks. For example, this Bash command sends
a local JPEG without giving the server filesystem access:

```bash
image_b64=$(base64 < signal.jpg | tr -d '\n')
curl http://127.0.0.1:8080/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "local/jet",
  "state": [
    "Identify the traffic light color in this image.",
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,${image_b64}",
        "detail": "auto"
      }
    }
  ],
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is the traffic light?",
      "criteria": {"red": "Red", "yellow": "Yellow", "green": "Green"}
    }
  }
}
JSON
```

Only `data:image/png;base64,...` and `data:image/jpeg;base64,...` URLs are
supported. Jet does not download remote URLs or read local paths supplied in
HTTP requests. `image_url.detail` may be omitted or set to `auto`, `low`, or
`high`; it is accepted for compatibility and does not change processing. Use
the server's image limits to configure vision processing.

Image blocks are recognized only as direct entries of the top-level `state`
array. Other entries retain their JSON value semantics. Image-shaped objects
nested inside ordinary JSON values are text, not visual input. Actual image
blocks are removed from text rendering so their base64 does not consume text
context. Plain text requests keep their existing behavior.

Malformed image blocks, unsupported media types or sources, invalid images,
image limits exceeded, or image requests sent to a text-only server return
`400`. The total body limit is checked separately and returns `413`. Success
responses keep the same `answers`, `usage`, `inference_ms`, and `queue_ms`
structure shown above.

## Compatibility

`model`, `state`, and `questions` are required. `model` must equal the server's
`--model-id`; a request cannot load a different model. Question names are
preserved in `answers`.

HTTP text values in `state`, `instructions`, and criteria must be strings,
objects, or arrays; scalar booleans and numbers are rejected at those positions.
Choice descriptions may also be `null`; Jet then uses that option's key as its
candidate text. Explicit descriptions and nested `null` values are preserved.
A `score` requires at least one criterion, and explicit `noul` criteria require
both `false` and `true`.
`session_id` and `user` are optional strings of at most 256 characters; `trace`
is an optional object. Metadata is accepted but is not persisted, reported to a
provider, or included in model prompts.

| Field or feature | Support |
| --- | --- |
| Text `noul`, `choice`, `score` | Supported using Jet's existing candidate scoring. |
| `session_id`, `trace`, `user` | Accepted as metadata; they do not affect scoring or create a conversation history. |
| `provider` | May be omitted, `null`, or `{}`; cloud routing options are rejected. |
| PNG/JPEG images in `state[]` | Supported as base64 data URLs when the server is built with `vision` and configured with `--mmproj-path`. |
| Remote image URLs, local image paths, other media types, `box` questions | Unsupported. |
| Chat completions, free-form generation, streaming | Unsupported. |
| Provider routing, billing, cloud model discovery | Unsupported. |

Optional OpenRouter response fields such as `id`, `provider`, `confidence`,
`legend`, and `usage.cost` are not synthesized. This endpoint exposes a local
Decisions subset, not a drop-in implementation of every OpenRouter API.

## Authentication and endpoints

The default listener is local to the machine. Set `JET_API_KEY` before starting
the server to require a bearer token on the Decisions and models routes:

```powershell
$env:JET_API_KEY = 'replace-with-your-secret'
./target/release/jet serve --model-path models/Qwen3-0.6B-Q8_0.gguf --model-id local/jet
```

Clients then send `Authorization: Bearer <key>`. With `JET_API_KEY` unset, those
routes accept requests without authentication. An explicitly empty value is a
startup error. The health routes remain unauthenticated.

| Method and route | Purpose |
| --- | --- |
| `POST /api/alpha/decisions` | Score a text or enabled image decision request. |
| `GET /api/v1/models` | Discover the single configured model identifier. |
| `GET /healthz` | Check that the HTTP process is responding. |
| `GET /readyz` | Check that the inference worker is available. |

The models response lists `architecture.input_modalities` as `["text", "image"]`
when a projector is configured, and `["text"]` otherwise.

Use a different `--bind` address when clients need network access. The listener
uses plain HTTP; terminate TLS at a reverse proxy for remote access.

## Queueing, deadlines, and errors

The inference worker starts an available request immediately. It never waits
to fill a cross-request batch. Candidate batching inside an individual engine
call remains enabled when using batched execution.

Queueing is enabled by default: one request can run while up to
`--queue-capacity` requests wait. A full queue returns `429` immediately.
Increasing this limit allows more requests to wait; it does not add model
execution parallelism and can increase tail latency.

To disable the waiting queue, start the server with `--no-queue`:

```sh
./target/release/jet serve \
  --model-path models/Qwen3-0.6B-Q8_0.gguf \
  --model-id local/jet \
  --no-queue
```

In this mode, one request can be admitted at a time. Further requests receive
`429` immediately, including during the handoff to the inference thread.

The request deadline covers queueing plus inference after validation, excluding
body upload, JSON parsing, and base64 normalization. Text and image requests
share the same queue and inference worker. Work whose deadline expires or whose
waiting handler is cancelled before execution is skipped. This does not guarantee
immediate detection of every client disconnect. An HTTP timeout or handler
cancellation does not interrupt a native inference call that has already
started: that call continues to occupy the worker until it completes, and its
result is discarded. With `--no-queue`, new requests continue to receive `429`
while that call is running.

Errors use an HTTP status code and a numeric `error.code`:

```json
{"error":{"code":429,"message":"inference queue is full"}}
```

| Status | Meaning |
| --- | --- |
| `400` | Invalid JSON, unsupported fields, invalid question, mismatched model, invalid or oversized image, or vision not enabled. |
| `401` | Missing or incorrect bearer token when authentication is configured. |
| `404` | Unknown route. |
| `405` | Unsupported method on a Decisions or models route. |
| `413` | Request body exceeds the configured limit. |
| `415` | Request is not sent with a supported JSON content type. |
| `429` | Inference queue is full, or another request is admitted or running with `--no-queue`. |
| `500` | Internal inference failure. |
| `503` | Inference worker is unavailable. |
| `504` | Request deadline elapsed while waiting or running. |

Body decoding runs before the authorization check, so malformed or oversized
requests can receive a body error before `401`.

## Measure latency

Measure client-observed latency as well as `inference_ms` and `queue_ms`.
Client latency also includes HTTP processing and network transit. Use the same model,
backend, inference parameters, and representative requests when comparing
direct Rust calls, a persistent JSONL process, and HTTP.

Record P50, P95, and P99 latency, queueing time, rejected requests, throughput,
and memory usage under your expected concurrency. Start with a single client
and increase concurrency to expose saturation. The server's small default
queue favors bounded waiting; it does not guarantee a latency target for every
model or machine. Keep-alive HTTP connections avoid reconnecting on each request.
