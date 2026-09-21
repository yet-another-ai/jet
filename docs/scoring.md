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

For Vulkan batched execution, generation and scoring use two contexts over the same model. The
single-sequence generation context copies its selected logits row to the CPU for the bounded
thinking sampler. The scoring context performs vocabulary softmax, target-token gather, and log on
the GPU and returns only a fixed-width target vector per output row. This avoids changing a live
sampler graph between thinking and scoring and does not duplicate model weights; the additional
state is one generation KV cache plus its compute buffers.

The JSONL request is only Jet's external API. Before tokenization, Jet uses Tera to render each
question as a human-readable user message with explicit `Context`, `Instruction`, and `Candidate
answers` sections. Structured objects are rendered with sorted keys, arrays retain their input
order, and text that resembles a special token is escaped before it reaches the model template.
The system and user messages define the same concise output contract: return one quoted candidate
exactly as listed, with no explanation, whitespace, or extra text. The user message does not append
an `Answer:` completion cue; the model's embedded chat template supplies the only assistant boundary.

Candidates are scored as continuations of the rendered assistant prefix:

- `noul`: the semantic false/true descriptions, defaulting to `"No"` and `"Yes"`;
- `choice`: each criterion's semantic value;
- `score`: each score level's semantic description.

Each semantic candidate is encoded as a quoted JSON string for an unambiguous answer boundary.
After scoring, Jet maps the winning semantic continuation back to the public API representation:
`false`/`true`, the original choice key, or the zero-based score index. Two candidates that render
to the same semantic continuation are rejected instead of being counted twice.

The score is the sum of each candidate token's log-probability. The first release does not score
an end-of-turn token and does not apply length normalization, temperature, or sampling. Therefore,
candidate token length affects raw sequence likelihood. Returned probabilities are normalized only
within the question's candidate set; they are not calibrated correctness probabilities.
The explicit output contract reduces probability mass assigned to unrelated chat continuations,
but cannot eliminate it. Jet's normalized result remains conditional on the configured candidate
set and must not be interpreted as the candidate's absolute probability over the full vocabulary.

For pure KV-cache models, the optimized executor prefills a question prefix, including its frozen
reasoning when enabled, once when all candidates fit the configured sequence budget, copies that
sequence's memory, and scores candidate suffixes in native batches. Hybrid and recurrent models
cannot safely advance multiple copied states in one llama.cpp decode wave, so Jet automatically
isolates their candidate groups. This repeats prefix work and reduces throughput but preserves
reference-equivalent scores. Oversized work is split into waves without truncating input. The
reference execution mode scores each candidate independently and exists for correctness
comparisons.

`usage.input_tokens` counts the original rendered question prompt once. `usage.output_tokens`
includes generated thinking tokens, including protocol closure, plus every candidate token that was
scored.
