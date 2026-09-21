# Scoring semantics

Jet renders each question with the GGUF model's embedded Jinja chat template. The renderer sets
`enable_thinking=false` and rejects a model whose template does not expose a reliable non-thinking
switch. Jet does not emulate the template with handwritten prompt strings.

Some Qwen3 templates represent disabled thinking with an empty, already-closed
`<think>\n\n</think>` protocol marker. Jet accepts that marker because candidate scoring starts after
the close tag; it still rejects an open marker or any thinking content.

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

`usage.input_tokens` counts each question's unique rendered prompt once.
`usage.output_tokens` counts every candidate token that was scored.

The optimized executor prefills a question prefix once when all its candidates fit the configured
sequence budget, copies that sequence's memory, and scores candidate suffixes in native batches.
Oversized work is split into waves without truncating input. The reference execution mode scores
each candidate independently and exists for correctness comparisons.
