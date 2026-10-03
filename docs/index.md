---
layout: home
title: Jet — Local structured decisions
description: Score fixed candidate answers with a local language model and receive structured JSON results.

hero:
  name: Jet
  text: Local decisions, structured answers
  tagline: Give a local language model context and fixed candidates. Jet scores them and returns a result you can use directly.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/yet-another-ai/jet

features:
  - icon: ◎
    title: Fixed candidate scoring
    details: Choose an answer, make a yes/no decision, or rate an item without parsing free-form model output.
  - icon: ↯
    title: Local inference
    details: Run GGUF models on CPU, CUDA, Metal, or Vulkan without an HTTP service.
  - icon: ⇥
    title: Streaming JSONL
    details: Process one request per line and receive ordered, machine-readable results from the CLI.
  - icon: ◫
    title: Image decisions
    details: Use the optional vision feature to score the same structured questions against PNG or JPEG inputs.
  - icon: ◇
    title: Rust API
    details: Embed the decision engine directly with typed requests, responses, configuration, and error handling.
  - icon: ∿
    title: Predictable outputs
    details: Candidate keys and score indexes come from your schema, so downstream code does not need an output parser.
---

<div class="vp-doc" style="margin: 32px auto 0; max-width: 688px">

```json
{"state":{"ticket":"Login fails for all users"},"questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"},"owner":{"type":"choice","instructions":"Choose the owner","criteria":{"app":"Application team","infra":"Infrastructure team"}}}}
```

</div>
