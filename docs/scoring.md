# Scoring semantics

Jet renders each question with the GGUF model's embedded Jinja chat template. Jet does not emulate
the template with handwritten prompt strings. Ordinary templates without reasoning support are
valid and use their normal assistant-generation prefix.

Thinking has three engine modes. `disabled` requests the template's direct-answer path and rejects
a reasoning template when disabling it cannot be verified. `auto` generates reasoning when the
template exposes a start/final transition and one or more end markers, otherwise it uses direct
scoring. `required` reports an unsupported-model error instead of falling back.

Thinking protocols come from llama.cpp's template parser. This covers tag protocols such as Qwen
and DeepSeek as well as channel protocols used by GPT-OSS, Kimi, and Gemma templates. Jet passes the
generated text back as `reasoning_content` and asks the same template for its content continuation,
so an end marker is never assumed to be the final-answer prefix. Unknown templates without enough
protocol metadata fall back only in `auto` mode.

The thinking token budget counts generated tokens inside the reasoning block. The protocol-aware
sampler may finish a partial UTF-8 character and add end-marker tokens after the budget is
exhausted. Thinking uses configurable temperature, top-k, top-p, and seed values. The resulting
trace is generated once per question and then frozen; all candidate scores are conditional on that
same trace.

The user message is canonical JSON containing `state`, `question.instructions`, every criterion,
and `allowed_labels`. Object keys are sorted, array order is retained, and text that resembles a
special token is JSON-escaped before tokenization.

Candidates are scored as continuations of the rendered assistant prefix:

- `noul`: the JSON scalars `false` and `true`;
- `choice`: each choice key encoded as a JSON string;
- `score`: zero-based JSON integer indices.

The score is the sum of each candidate token's log-probability. The first release does not score
an end-of-turn token and does not apply length normalization, temperature, or sampling. Therefore,
candidate token length affects raw sequence likelihood. Returned probabilities are normalized only
within the question's candidate set; they are not calibrated correctness probabilities.

The optimized executor prefills a question prefix, including its frozen reasoning when enabled,
once when all its candidates fit the configured sequence budget, copies that sequence's memory, and
scores candidate suffixes in native batches.
Oversized work is split into waves without truncating input. The reference execution mode scores
each candidate independently and exists for correctness comparisons.

`usage.input_tokens` counts the original rendered question prompt once. `usage.output_tokens`
includes generated thinking tokens, including protocol closure, plus every candidate token that was
scored.
