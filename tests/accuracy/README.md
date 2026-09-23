# Accuracy evaluation

This evaluation maps public labelled datasets onto Jet's decision protocol:

- BoolQ validation examples become `noul` questions.
- MMLU test examples become `choice` questions with `A` through `D` candidates.

The source archives, extracted data, generated requests, gold labels, responses, and reports are
intentionally ignored by Git. The preparation script verifies the normalized BoolQ JSONL and the
MMLU archive with pinned SHA-256 digests before using them.

Run the reproducible default sample (256 balanced BoolQ examples and 5 examples from each of the
57 MMLU subjects):

```sh
./scripts/run-accuracy-eval.sh
```

The default remains the pinned Qwen3-0.6B test model. Download the two Qwen3.5 Q8_0 evaluation
models with:

```sh
./scripts/download-accuracy-models.sh
```

Download the pinned Qwen3.6-35B-A3B Q4_K_M text model separately (20.42 GB):

```sh
./scripts/download-accuracy-models.sh --qwen36
```

This selects the ggml-org conversion without MTP or vision-projector weights and verifies its
SHA-256 digest. For the development machine's 16 GiB Arc A770, use
`scripts/benchmark-model.py` with
`--extra-args --cpu-moe-layers 11 --max-sequences 2 --micro-batch 256 --max-output-rows 256 --threads 8 --no-mmap`.
For this specific GGUF, GPU weights occupy 14.10 GiB and CPU weights 4.91 GiB before caches and
compute buffers. The placement was chosen from measured memory residency and performance,
not a fixed RAM percentage. See the [Qwen3.6 measurements](../../docs/accuracy.md#qwen36-q4-cpuvulkan-offload),
[placement options](../../README.md#vulkan), and
[benchmark harness](../../docs/development.md#reproducible-model-benchmarks).

Run the same evaluation through the opt-in Vulkan backend with a separate output directory:

```sh
JET_BACKEND=vulkan \
JET_ACCURACY_DIR=tests/accuracy/generated-vulkan \
  ./scripts/run-accuracy-eval.sh
```

Select another model with `JET_MODEL_PATH` and set `JET_MODEL_ID` so the response metadata names
the evaluated model. For example:

```sh
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

This requires the Vulkan build dependencies described in
[the development guide](../../docs/development.md). `JET_BACKEND` defaults to `cpu`.
Jet automatically uses isolated candidate waves for Qwen3.5's hybrid recurrent state; no extra CLI
flag is required for the correctness-safe path.

Override sample sizes with `JET_BOOLQ_LIMIT` and `JET_MMLU_PER_SUBJECT`. For example, the complete
MMLU test set has unequal subject sizes, so use the preparation script directly if a sampling rule
other than a fixed count per subject is needed.

The primary metric is top-1 accuracy. Mean negative log-likelihood of the gold label is included as
a diagnostic. These results are not directly comparable to the official MMLU leaderboard because
Jet uses its own Tera-rendered zero-shot instruction prompt and semantic candidate scoring.
Pass matching `--requests` to `evaluate-accuracy.py` when evaluating MMLU: if multiple option
letters have identical answer text, any letter equivalent to the gold answer counts as correct.
The gold probability sums those equivalent labels. The report lists duplicate-answer questions.

See [the recorded baseline](../../docs/accuracy.md) for the latest checked-in result and its
interpretation.

## Sources and licenses

- [BoolQ](https://github.com/google-research-datasets/boolean-questions), CC BY-SA 3.0. The labelled
  validation split is retrieved through the Hugging Face dataset server because the original GCS
  link no longer permits anonymous downloads.
- [MMLU](https://github.com/hendrycks/test), MIT. The data archive is downloaded from the URL in the
  upstream repository.

Retain these attributions when redistributing derived fixtures.
