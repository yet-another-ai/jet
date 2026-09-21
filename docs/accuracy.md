# Accuracy baseline

The reproducible task-accuracy runs were recorded on 2026-09-21 and 2026-09-22. The original runs
used the pinned `Qwen3-0.6B-Q8_0.gguf` test model (SHA-256
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

## Vulkan backend comparison

On 2026-09-22, the same 541 semantic-answer requests were run once through the release Vulkan
backend on an Intel Arc A770, with no explicit warm-up. The measurement includes process startup,
Vulkan initialization, model loading, and evaluation. The CPU timing is the previously recorded
run on the same development machine, so its value is approximate.

| Backend | Wall time | Requests/s | Overall | BoolQ | MMLU | Mean gold NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | ~220 s | ~2.46 | 252/541 (46.58%) | 143/256 (55.86%) | 109/285 (38.25%) | 2.0857 |
| Vulkan, CPU softmax | 70.86 s | 7.63 | 255/541 (47.13%) | 142/256 (55.47%) | 113/285 (39.65%) | 2.0978 |
| Vulkan, GPU softmax + target gather | 69.51 s | 7.78 | 255/541 (47.13%) | 142/256 (55.47%) | 113/285 (39.65%) | 2.0977 |

The optimized Vulkan path is about a **3.17x end-to-end speedup** and a **68.4% wall-time reduction**
versus the recorded CPU run. Moving vocabulary normalization to Vulkan improved the original GPU
run by only 1.9%, so logits normalization and host transfer were not the dominant bottleneck on
this workload. Gathering only target probabilities did not improve wall time beyond the full-row
GPU softmax experiment, but reduced its Vulkan compute buffer from 449.02 MiB to 317.39 MiB. Peak
host RSS for the final run was approximately 741 MiB. All measurements were cold process starts
with no explicit warm-up.

The target-gather run produced no top-1 changes relative to the full-row GPU softmax run across all
541 examples. The maximum absolute candidate-probability difference was `7.1e-8` and the mean was
`4.5e-9`.

## Disabled-thinking model comparison

On 2026-09-22, Qwen3.5-0.8B and Qwen3.5-2B were evaluated with the same 541 requests, Q8_0
quantization, embedded chat templates, correctness-safe hybrid execution, optimized Vulkan target
gather, and `--thinking disabled`. All model layers were offloaded to the same Intel Arc A770. Each
timing is a cold CLI process start and includes Vulkan initialization, model loading, and
evaluation, but not compilation or dataset preparation.

The Qwen3.5 artifacts are pinned independently from the Qwen3 test fixture:

| Model | GGUF source revision | SHA-256 | File size |
| --- | --- | --- | ---: |
| Qwen3.5-0.8B Q8_0 | `bartowski/Qwen_Qwen3.5-0.8B-GGUF` @ `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48` | `7182e2362766bb9569209bbc24cf1a4cdfbb8ab161babdb2080c84fa62c08c2f` | 835,325,024 bytes |
| Qwen3.5-2B Q8_0 | `bartowski/Qwen_Qwen3.5-2B-GGUF` @ `7d26695454df6de5fbcce2e58681e62dae06ce43` | `be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2` | 2,080,140,384 bytes |

| Model | Wall time | Requests/s | Overall | BoolQ | MMLU | Mean gold NLL | Peak host RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | 69.51 s | 7.78 | 255/541 (47.13%) | 142/256 (55.47%) | 113/285 (39.65%) | 2.0977 | ~741 MiB |
| Qwen3.5-0.8B Q8_0 | 244.08 s | 2.22 | 280/541 (51.76%) | 176/256 (68.75%) | 104/285 (36.49%) | 1.0740 | 964.4 MiB |
| Qwen3.5-2B Q8_0 | 316.05 s | 1.71 | 343/541 (63.40%) | 190/256 (74.22%) | 153/285 (53.68%) | 0.9052 | 2,151.1 MiB |

Relative to Qwen3-0.6B, Qwen3.5-0.8B took 3.51x as long and Qwen3.5-2B took 4.55x as long. The
Qwen3.5 tokenizer also produced 128,583 input tokens for this sample, versus 124,943 for Qwen3,
so the wall-time ratios are not pure per-token model-speed ratios. More importantly, Qwen3.5 is a
hybrid recurrent/attention architecture. Jet serializes its candidate groups for correctness,
whereas Qwen3 retains shared-prefix candidate fan-out and cross-question batching. These timings
therefore compare the current safe end-to-end implementations, not equivalent batch geometry.

Qwen3.5-0.8B gained 25 correct examples over Qwen3-0.6B (+4.62 percentage points overall), while
Qwen3.5-2B gained 88 (+16.27 points). The 2B model reached 53.68% on Jet's MMLU sample, close to the
55.3 MMLU-Pro result reported for non-thinking mode in the
[upstream model card](https://huggingface.co/Qwen/Qwen3.5-2B), but the values are not directly
comparable: the datasets, sample sizes, prompts, few-shot setup, and answer-scoring protocol differ.
These remain Jet-specific continuation-only semantic-answer results, not official Qwen benchmark
scores.

### Hybrid recurrent batching correctness

The first Qwen3.5 run exposed a silent correctness bug in Jet's original multi-sequence executor.
It copied one hybrid recurrent prefix to several candidate sequences and advanced those forks in a
single decode wave. On a 20-question diagnostic slice, 9 top-1 predictions differed from the
independent reference path, with a maximum candidate-probability difference near 1.0. This produced
the invalid MMLU figures 27.02% for 0.8B and 28.77% for 2B and mean gold NLL values above 55.

Jet now detects recurrent and hybrid models through llama.cpp and evaluates one candidate group per
wave. Pure KV-cache models keep the original shared-prefix batching. With the safe schedule, the
20-question slice had no top-1 differences from reference and a maximum probability difference of
`1.12e-7`. The final 2B run was also bit-identical across all 541 responses to a separate
`--max-sequences 2` safety run. The table above contains only the corrected results; the discarded
run is recorded here solely to explain the failure mode and the performance change.

Download the pinned Qwen3.5 files and reproduce either run with:

```sh
./scripts/download-accuracy-models.sh

JET_BACKEND=vulkan \
JET_MODEL_PATH=models/Qwen3.5-0.8B-Q8_0.gguf \
JET_MODEL_ID=qwen/qwen3.5-0.8b-q8_0 \
JET_ACCURACY_DIR=tests/accuracy/generated-vulkan-qwen35-08b \
  ./scripts/run-accuracy-eval.sh

JET_BACKEND=vulkan \
JET_MODEL_PATH=models/Qwen3.5-2B-Q8_0.gguf \
JET_MODEL_ID=qwen/qwen3.5-2b-q8_0 \
JET_ACCURACY_DIR=tests/accuracy/generated-vulkan-qwen35-2b \
  ./scripts/run-accuracy-eval.sh
```

## 256-token bounded-thinking comparison

The same Vulkan target-gather build was also run with `--thinking required --thinking-tokens 256`.
The quota is an upper bound: measured as additional output tokens relative to the disabled run,
reasoning plus protocol closure averaged 127.7 tokens, had a median of 129, and ranged from 64 to
129. The model therefore closed its reasoning naturally before reaching the configured maximum on
every example.

| Thinking | Wall time | Requests/s | Overall | BoolQ | MMLU | Mean gold NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disabled | 69.51 s | 7.78 | 255/541 (47.13%) | 142/256 (55.47%) | 113/285 (39.65%) | 2.0977 |
| Required, quota 256 | 644.23 s | 0.84 | 273/541 (50.46%) | 167/256 (65.23%) | 106/285 (37.19%) | 2.2883 |

Thinking improved overall accuracy by 18 examples, or 3.33 percentage points, but the effect was
not uniform. BoolQ gained 25 correct answers (+9.77 points), while MMLU lost 7 (-2.46 points).
Among 118 changed top-1 predictions, 58 moved from wrong to correct and 40 from correct to wrong;
the dataset splits were 39/14 for BoolQ and 19/26 for MMLU. Mean gold NLL worsened on both datasets
and mean confidence increased from 0.8033 to 0.8737, indicating stronger but less well-calibrated
predictions. End-to-end wall time increased by 9.27x.

The disabled-thinking CPU and Vulkan backend results are not bit-identical. Fourteen of 541 top-1
predictions changed: seven moved
from wrong to correct, four from correct to wrong, and three remained wrong. Across reported
candidate probabilities, the median absolute difference was 0.0042, the 95th percentile was
0.0667, and the maximum was 0.2998. Overall top-1 accuracy therefore did not decline in this run,
but the three-example increase is best treated as backend numerical variation, not as evidence that
Vulkan improves model quality. BoolQ lost one correct answer while MMLU gained four, and mean gold
NLL increased slightly.

Reproduce the Vulkan run with:

```sh
JET_BACKEND=vulkan \
JET_ACCURACY_DIR=tests/accuracy/generated-vulkan \
  ./scripts/run-accuracy-eval.sh
```

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
