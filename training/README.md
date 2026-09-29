# JET LoRA training

This recipe trains **Qwen/Qwen3.5-4B with BF16 LoRA for text decisions**. It uses
JET's actual Rust prompt renderer and scores the quoted semantic answer. It does
not train JSON output, answer letters, reasoning traces, or a separate classifier
head. Images and ordinal `score` datasets are outside this first recipe.

Training uses PyTorch, Transformers and PEFT. Python, dependencies, the model
revision and public dataset files are pinned. No training metrics or artifacts
are uploaded. Generated data, caches, adapters and merged models are ignored by Git.

## Install and prepare

Run these commands from the repository root. The repository's `mise.toml` pins
`uv` to 0.12.13, and `mise.lock` records its resolved artifact. `uv` manages the
separate Python 3.12 environment under `training/.venv`. TinyTeX is excluded
from Windows tool activation because its mise installer supports Linux/macOS.

```sh
mise exec rust@1.98 -- cargo build -p jet-cli
mise exec -- uv sync --project training --locked --extra train
```

Data preparation and tokenization can run without CUDA. Training requires one
NVIDIA GPU with BF16 support. The locked PyTorch wheel uses CUDA 12.8 on Windows
and Linux. The starting recipe uses microbatch 1, gradient accumulation 16,
gradient checkpointing, rank 16 and a 2,048-token limit. Its peak memory and speed
must be measured on the real model; Unsloth memory estimates do not describe this
standard Transformers implementation. Optional DeltaNet acceleration kernels are
not installed by this recipe; the PyTorch reference path is slower.

On Windows:

```sh
mise exec -- uv run --directory training --locked -m jet_training.data --jet-binary ../target/debug/jet.exe --output-dir data/default
```

On Linux, use `--jet-binary ../target/debug/jet`. Use an executable built from the
current checkout: older executables do not have `export-prompts`.

If you already have frozen evaluation requests, add one or more
`--exclude-requests PATH` arguments. For the existing full-MMLU workload:

```sh
mise exec -- uv run --directory training --locked -m jet_training.data --jet-binary ../target/debug/jet.exe --output-dir data/default --exclude-requests ../tests/accuracy/generated-mmlu-full-ascii/requests.jsonl
```

Only run one of these preparation commands for a given output directory. Existing
nonempty outputs are rejected. Use `--limit-per-source 8 --output-dir data/smoke`
for a small fixture, or `--offline` to reproduce from checksum-verified cached
sources. Even a capped run scans complete held-out splits before sampling.

The default mixture uses all eligible BoolQ and ARC training examples, plus
10,000 SNLI training examples. Source files total approximately 27 MB; generated
JSONL files are larger because each record retains prompts and provenance.
The pinned default preparation produces 22,791 train, 10,708 validation and
16,640 test records before tokenizer length filtering.

| Source | Training | Checkpoint validation | Final test | License |
| --- | --- | --- | --- | --- |
| BoolQ | Official train | None | Official validation | CC BY-SA 3.0 |
| ARC Easy + Challenge | Official train | Official validation | Official test | CC BY-SA 4.0 |
| SNLI | Deterministic train subset | Official validation | Official test | CC BY-SA 4.0 |

BoolQ validation stays out of checkpoint selection to preserve JET's existing
benchmark. MMLU is not a training source. Invalid SNLI labels and ambiguous ARC
answers are dropped. Deduplication uses label-blind normalized content, reserves
test before validation before train, and runs before sampling. ARC question and
sorted answer text keys are shared across Easy and Challenge and external MMLU
requests. This catches exact normalized overlaps, not paraphrases or contamination
already present in the pretrained model. Training choice order is deterministically
permuted and the gold key follows the permutation.

Each split contains:

- `requests.jsonl` and `gold.jsonl`: inputs to the existing JET benchmark harness.
- `prompts.jsonl`: exact `jet export-prompts` output.
- `records.jsonl`: messages, all quoted candidates, the correct quoted target,
  source revision, original row index and content hashes.

`manifest.json` records all file hashes, source revisions, exporter binary hash,
selection counts, deduplication drops and external holdouts. A missing manifest
means preparation did not complete. Dataset attributions and individual parquet
hashes are retained in [datasets.lock.json](datasets.lock.json); retain attribution,
license notices and transformation information when redistributing derived data.

## Validate tokenization, then train

Paths in [configs/qwen35-4b-lora.toml](configs/qwen35-4b-lora.toml) are relative to
that config file. The model revision is immutable; training starts from the
original Hugging Face weights, not the local GGUF.

```sh
mise exec -- uv run --directory training --locked --extra train -m jet_training.train --config configs/qwen35-4b-lora.toml --dry-run --report data/default/tokenization.json
```

This downloads only tokenizer/configuration files. It validates every record and
reports retained examples, length drops and supervised token counts. The model's
chat template is applied with `enable_thinking=False`. Only the complete target
continuation has loss: prompt and padding labels are `-100`, no EOS is appended,
and unstable token boundaries fail. Overlength examples are dropped whole and
reported; targets are never silently truncated. A deterministic subset of up to
512 validation examples is used for checkpoint-loss monitoring. Full evaluation
files remain intact.

Start with a short hardware probe in a separate run directory:

```sh
mise exec -- uv run --directory training --locked --extra train -m jet_training.train --config configs/qwen35-4b-lora.toml --max-steps 2 --validation-max-samples 8 --output-dir runs/qwen35-4b-smoke
```

This command downloads the original model weights and performs real optimizer
steps. Check memory, finite loss and step time before a full run:

```sh
mise exec -- uv run --directory training --locked --extra train -m jet_training.train --config configs/qwen35-4b-lora.toml --micro-batch-size 4 --gradient-accumulation-steps 4
```

On the 24 GiB RTX 4090, this effective batch of 16 ran a two-step 4B probe and
the four longest training examples without exhausting GPU memory. The latter
reserved about 25.1 GB including CUDA allocator headroom, leaving little spare
VRAM for other GPU workloads. The conservative config defaults remain batch 1
and accumulation 16. Choose one batch setting for the entire run and repeat it
exactly when resuming.

Only language-model linear layers receive LoRA, including both ordinary attention
and DeltaNet projections. The vision tower is not loaded. Existing run directories
are rejected unless a complete checkpoint is explicitly resumed:

```sh
mise exec -- uv run --directory training --locked --extra train -m jet_training.train --config configs/qwen35-4b-lora.toml --micro-batch-size 4 --gradient-accumulation-steps 4 --resume-from-checkpoint runs/qwen35-4b-lora/checkpoint-100
```

Resume checks the config, model revision and data hashes. Checkpoints retain
optimizer/scheduler/RNG state. When resuming a limited run, repeat its original
`--max-steps`, `--validation-max-samples`, batch-size, accumulation and
`--output-dir` overrides; changing the
training horizon is rejected. A completed run contains the `adapter/` directory,
`run.json`, `tokenization.json`, `metrics.json`, and `completed.json`. Loss alone
does not establish better JET decisions; compare deployed models below.

## Merge, convert and evaluate

Merge the adapter into its pinned BF16 base before deployment, eliminating a
separate inference adapter path:

```sh
mise exec -- uv run --directory training --locked --extra train -m jet_training.export --config configs/qwen35-4b-lora.toml --adapter runs/qwen35-4b-lora/adapter --output-dir exports/qwen35-4b-merged
```

Install the converter extras and use the vendored conversion code:

```sh
mise exec -- uv sync --project training --locked --extra train --extra convert
mise exec -- uv run --directory training --locked --extra train --extra convert python ../vendor/llama.cpp/convert_hf_to_gguf.py exports/qwen35-4b-merged --outfile exports/qwen35-4b-lora-Q8_0.gguf --outtype q8_0 --no-nextn
```

Use a JET executable with the same GPU backend as the baseline. From the repository
root (replace binary and model paths as appropriate):

```sh
mise exec -- uv run --directory training --locked python ../scripts/benchmark-model.py --binary ../target/release/jet.exe --model-path exports/qwen35-4b-lora-Q8_0.gguf --model-id jet/qwen35-4b-lora-q8_0 --input data/default/test/requests.jsonl --gold data/default/test/gold.jsonl --output-dir runs/eval-lora --backend cuda --runs 2
```

Run the same command for the original Q8_0 model with a different output directory.
Also run the frozen full-MMLU regression workload and a separate real-business
holdout. The evaluator reports per-dataset accuracy, gold NLL, sum-of-squares Brier
score and 10-bin top-label ECE. Equivalent candidate texts count as one semantic
answer. Binary Brier uses both class probabilities (twice the single-positive-class
convention). Whole-run requests/second includes startup and model loading; it is
not warm latency or P95. Quality and latency improvements require measured results.

## Local checks

```sh
mise exec -- uv run --directory training --locked --extra train python -m unittest discover -s tests -v
mise exec -- uv run --directory training --locked ruff check jet_training tests
mise exec rust@1.98 -- cargo test -p jet-cli --test export_prompts -p jet-engine --lib
```

The Python suite covers split isolation, source integrity, label alignment, token
masking and checkpoint behavior. Small synthetic models test training mechanics;
they are not evidence of Qwen3.5-4B task quality. Public model/training references:
[Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[PEFT LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora).

The real BF16 hybrid-model/PEFT checkpoint tests are opt-in: set
`JET_RUN_TRAINING_RUNTIME=1` before running unittest. They use tiny randomly
initialized models on CUDA and do not download pretrained model weights.
