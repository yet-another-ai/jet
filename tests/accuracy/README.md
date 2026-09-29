# Accuracy evaluation

This evaluation maps public labelled datasets onto Jet's decision protocol:

- BoolQ validation examples become `noul` questions.
- MMLU test examples become `choice` questions with `A` through `D` candidates.

The source archives, extracted data, generated requests, gold labels, responses, and reports are
intentionally ignored by Git. The preparation script verifies the normalized BoolQ JSONL and the
MMLU archive with pinned SHA-256 digests before using them.

Run the reproducible default sample (256 balanced BoolQ examples and 5 examples from each of the
57 MMLU subjects) after downloading the Qwen3.5-0.8B Q8_0 model:

```sh
./scripts/download-accuracy-models.sh
./scripts/run-accuracy-eval.sh
```

The default is Qwen3.5-0.8B Q8_0. Download both Qwen3.5 Q8_0 evaluation
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

Download the pinned Qwen3.5-4B and 9B Q8_0 models (14.43 GB total) with:

```sh
./scripts/download-accuracy-models.sh --qwen35-large
```

The downloader verifies both files against their pinned SHA-256 digests.

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
[the development guide](../../docs/development.md). `JET_BACKEND` defaults to `auto`; set
`JET_GPU_FEATURES=cuda,vulkan` when running the script with both GPU backends compiled in.
Jet automatically uses isolated candidate waves for Qwen3.5's hybrid recurrent state; no extra CLI
flag is required for the correctness-safe path.

Override sample sizes with `JET_BOOLQ_LIMIT` and `JET_MMLU_PER_SUBJECT`. For the complete
14,042-question MMLU test set, prepare MMLU alone with:

```sh
python scripts/prepare-accuracy-data.py --boolq-limit 0 --mmlu-all --ascii-json \
  --output-dir tests/accuracy/generated-mmlu-full
```

Run `scripts/benchmark-model.py` with these `requests.jsonl` and `gold.jsonl`
files, a model path, and `--backend cuda` or `--backend vulkan`. The RTX 4090
full-set study used `--runs 1 --backend vulkan --batch-requests 8 --extra-args
--max-sequences 2 --micro-batch 256 --max-output-rows 256 --threads 8 --no-mmap`.

To run the four Qwen3.5 Arc A770 model configurations (0.8B, 2B, 4B, and 9B) on all 14,042
MMLU questions, use `bash scripts/run-a770-full-mmlu.sh`. It runs the models
sequentially, preserves the benchmark harness's responses, reports, timings,
and provenance under `tests/accuracy/generated/a770-*-mmlu-full-*`, and skips
only runs with 14,042 responses and no evaluation failures. The models use
full GPU placement. Set `JET_A770_RUN_TAG` to give a new campaign its own
output directories, and `JET_A770_BINARY` to select a validated executable.
The default run tag is the current UTC date. Pass model names to run a subset,
for example `bash scripts/run-a770-full-mmlu.sh 4b 9b` to extend an existing study.

For long runs, launch the benchmark as a detached systemd user service so that
interrupting a terminal or assistant task does not terminate the experiment.
`scripts/check-a770-mmlu.py` is a one-shot checker suitable for a 30-minute user
timer. It takes the completed 4B and 9B artifact directories, benchmark and timer
unit names, and a status-file path (see `--help`). It never calls a model or API.
Once both full runs pass validation, it imports their measurements into the paper's
JSON and A770 table, updates the accompanying evidence and protocol, regenerates
the chart, and builds the PDF. It stops its timer on completion or failure.
An interrupted run is retained and excluded; rerun it in a new directory instead
of combining partial-run timings. The generated PDF should receive visual review
before external circulation.

The September 29 continuation uses `jet-a770-9b-20260929.service` and
`jet-a770-check-20260929.timer`. Inspect them without invoking an assistant:

```sh
systemctl --user list-timers 'jet-a770*'
journalctl --user -u jet-a770-check-20260929.service -n 20
cat tests/accuracy/generated/a770-large-monitor-20260929/status.json
```

Build the executable from the current source before running: an older binary
rejected duplicate choice texts and small positive Vulkan log-probabilities
caused by floating-point rounding. Those failures invalidated an earlier
full-set attempt, which is excluded from the paper.
The generated JSONL uses LF on Linux; the earlier RTX 4090 Windows records
used CRLF, so byte hashes differ even when the JSON objects match.

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
