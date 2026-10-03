---
title: Requests and responses
description: The JSONL protocol for Jet decision requests, answer types, errors, and token usage.
---

# Requests and responses

The CLI accepts UTF-8 JSON Lines: every non-empty line is one complete request. Jet returns one line for each request, preserves input order, and exits with a nonzero status if any line fails.

## Request object

Each request contains two fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | any JSON value | Context shared by every question in the request. |
| `questions` | object | Named questions. Each key is preserved in `answers`. |

The request deliberately has no `model` field. The CLI loads one model for the input stream using `--model-path` and identifies it with `--model-id`.

## Question types

### `noul`

A `noul` question returns a boolean-style decision weight. Criteria are optional; when present, they give the two values semantic meaning.

```json
{
  "type": "noul",
  "instructions": "Should this incident page the on-call engineer?",
  "criteria": {
    "false": "Do not page",
    "true": "Page now"
  }
}
```

The response contains `noul`, the normalized weight assigned to the `true` candidate:

```json
{"type":"noul","noul":0.91}
```

### `choice`

A `choice` question maps stable keys to the semantic candidates shown to the model:

```json
{
  "type": "choice",
  "instructions": "Choose the team that should own this ticket",
  "criteria": {
    "app": "Application team",
    "infra": "Infrastructure team"
  }
}
```

The response includes the winning key and every normalized candidate weight:

```json
{"type":"choice","choice":"infra","probabilities":{"app":0.08,"infra":0.92}}
```

### `score`

A `score` question receives an ordered list. The result is the probability-weighted, zero-based index across that list.

```json
{
  "type": "score",
  "instructions": "Rate incident severity",
  "criteria": ["low", "medium", "high"]
}
```

```json
{"type":"score","score":1.78,"probabilities":{"0":0.03,"1":0.16,"2":0.81}}
```

## Successful response

```json
{
  "model": "qwen/qwen3-0.6b-q8_0",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.91},
    "owner": {
      "type": "choice",
      "choice": "infra",
      "probabilities": {"app": 0.08, "infra": 0.92}
    }
  },
  "usage": {"input_tokens": 81, "output_tokens": 4}
}
```

The probability fields are normalized decision weights derived from mean candidate token log-probabilities. They are useful for comparing the candidates in one question, but they are not calibrated probabilities of correctness. See [Candidate score normalization](../advanced#candidate-score-normalization) for the exact calculation.

## Error response

Invalid JSON, invalid request schemas, model failures, and other per-line failures have this shape:

```json
{"error":{"code":"invalid_request","message":"..."}}
```

A failed line does not stop later input lines from running. Jet still exits nonzero after processing the stream so batch jobs can detect partial failure.

## Inspect rendered prompts

`export-prompts` applies the model-independent Jet prompt templates without loading a model:

```sh
./target/release/jet export-prompts \
  --input requests.jsonl \
  --output -
```

This is useful when reviewing how state, instructions, and semantic candidates are presented before running inference.
