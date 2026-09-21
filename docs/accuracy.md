# Accuracy baseline

The reproducible task-accuracy runs were recorded on 2026-09-21 and 2026-09-22. They used the pinned
`Qwen3-0.6B-Q8_0.gguf` test model (SHA-256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`), batched execution, the
model's embedded chat template, and disabled thinking.

The deterministic sample seed was `20260921`. The same examples were run before and after replacing
the legacy canonical-JSON/opaque-label prompt with the Tera-rendered instruction prompt and
semantic candidate scoring:

| Prompt contract | Dataset | Selection | Correct | Accuracy | Mean gold NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| Legacy JSON + output labels | BoolQ validation | 256, balanced by label | 128 | 50.00% | 2.0258 |
| Legacy JSON + output labels | MMLU test | 5 from each of 57 subjects | 63 / 285 | 22.11% | 5.8408 |
| Tera + semantic answers | BoolQ validation | same 256 | 143 | 55.86% | 0.8423 |
| Tera + semantic answers | MMLU test | same 285 | 109 / 285 | 38.25% | 3.2025 |

All 541 requests completed without an engine or input error in both runs. The semantic-answer CPU
run took approximately 3 minutes 40 seconds on the development machine; semantic candidates may
contain more tokens than the old single-letter labels.

The legacy results revealed a fatal output-label prior: every BoolQ example was predicted `true`,
while MMLU predicted `A` 283 times and `D` twice. After scoring the answer meanings and mapping them
back to API keys, MMLU predictions were distributed across `A/B/C/D` as `119/48/57/61` and accuracy
rose from 22.11% to 38.25%, above the 25% random baseline. This confirms that the old JSON/opaque
label prompt contract, rather than only model capability, caused the below-random MMLU result.

BoolQ reached 55.86%, above the balanced 50% random baseline, but predicted `false` 235 times and
`true` only 21 times. Qwen3-0.6B with thinking disabled therefore still has a substantial binary
answer prior on this task. Prompt-contract correctness, calibration, and benchmark capability
should remain separate acceptance criteria.

This is a Jet-specific zero-shot evaluation. Its Tera prompt, system instruction, lack of MMLU
few-shot demonstrations, and continuation-only candidate scoring differ from official benchmark
protocols, so the values must not be presented as official BoolQ or MMLU scores. A larger model,
label-permutation checks, candidate-prior calibration, and a separately reported bounded-thinking
run are still needed before accuracy gates are set.

Reproduce the run with:

```sh
./scripts/run-accuracy-eval.sh
```

The source data, generated requests, gold labels, raw responses, and machine-readable report are
stored below `tests/accuracy/data/` and `tests/accuracy/generated/`, both of which are ignored by
Git. Source revisions and SHA-256 digests are recorded by the generated manifest.
