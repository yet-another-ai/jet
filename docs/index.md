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
  - title: Fixed candidate scoring
    details: Choose an answer, make a yes/no decision, or rate an item without parsing free-form model output.
  - title: Local inference
    details: Run GGUF models on CPU, CUDA, Metal, or Vulkan without an HTTP service.
  - title: Streaming JSONL
    details: Process one request per line and receive ordered, machine-readable results from the CLI.
  - title: Image decisions
    details: Use the optional vision feature to score the same structured questions against PNG or JPEG inputs.
  - title: Rust API
    details: Embed the decision engine directly with typed requests, responses, configuration, and error handling.
  - title: Predictable outputs
    details: Candidate keys and score indexes come from your schema, so downstream code does not need an output parser.
---
