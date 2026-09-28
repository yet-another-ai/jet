# Evidence map for the report

Prepared against Jet `4ff0ab63018e79e5f4639aef97924f4e26bb0405`, with native
submodule `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. These identify the source
snapshot consulted, not every historical benchmark binary. The full-MMLU runs
below were performed on September 23--24, 2026, after the first draft.
Figures are rounded from the recorded experiments.

| Report claim/table | Source record | Conditions and interpretation |
| --- | --- | --- |
| Semantic prompt comparison | `docs/accuracy.md`, Accuracy baseline | Same 541 examples, Qwen3-0.6B Q8_0 CPU; prompt and candidate representation change together. |
| Small-model decision-quality results | `docs/accuracy.md`, Disabled-thinking model comparison | Arc A770, Q8_0, corrected hybrid execution before serial prefix reuse; fresh-process timing. |
| CPU/Vulkan and target gather | `docs/accuracy.md`, Vulkan backend comparison | CPU time approximate; CPU/Vulkan outputs differ. Target gather comparison is against full-row GPU softmax. |
| 2.18–2.23x prefix comparison | `docs/accuracy.md`, Vulkan serial-prefix reuse benchmark | One old run, two new runs/model; bundles prefix reuse and metadata resets. Local `vulkan-prefix-benchmark-20260922/summary.json` cross-checked. |
| 66.03% fewer prefill tokens | Same prefix section | 378,526 old tokens derived from schedule; 128,583 new tokens instrumented. Not a wall-time estimate. |
| 35B MoE placement/accuracy | `docs/accuracy.md`, Qwen3.6 Q4 CPU/Vulkan offload | 541 examples, 11 CPU expert layers, i9-13900K/64 GiB/16 GiB A770. |
| 7.36% worker improvement | `docs/performance.md`, Full 541-request validation | Single full run/version, byte-identical responses; separate 32-case probe supports direction. Local full-pool541 report cross-checked. |
| 4090 execution-ablation table | `docs/performance.md`, RTX 4090 preparation pipeline / Fixed 285-question MMLU comparison | 285 questions only, Vulkan full GPU, one final run/configuration, loading included; raw 4090 artifacts not present locally. |
| Full-MMLU model comparison | Local `tests/accuracy/generated/qwen35-*-4090-mmlu-full-20260924/` and `qwen36-4090-mmlu-full-fixed-20260923/` | Six complete 14,042-question RTX 4090 Vulkan runs, one/model, same binary/input/gold; each has 14,042 responses and zero evaluation failures. Summary below. |
| CPU/Vulkan 14 changed predictions | `docs/accuracy.md`, text after bounded-thinking comparison | Explicit counterexample to cross-backend bitwise/decision invariance. |
| Hybrid correctness and rejected batching | `docs/accuracy.md`, Hybrid recurrent batching correctness; `docs/performance.md`, Deferred independent-request experiment | Evidence for retaining serial recurrent continuations; not evidence of a universal native-backend bug. |
| Thinking quality/cost | `docs/accuracy.md`, 256-token bounded-thinking comparison | Separate sampled-generation condition, excluded from sampler-free claim. |
| Multimodal design | `README.md`, Multimodal decisions; `demos/doom/README.md`; `crates/jet-engine/src/llama/vision.rs` | Implemented interface/demo; no quantitative visual-quality claim. |

External references in `references.bib` were checked against their primary pages:
TypeSafe's announcement and the MMLU, BoolQ, and PagedAttention arXiv records.
The Jev discussion attributes design claims to its provider and does not infer
unpublished architecture or claim measured parity. Internal records are tracked here rather than cited as external literature.

Before external circulation: archive raw artifacts and
binary/source provenance; add matched-subset Jev and constrained-generation baselines,
repeated latency/throughput measurements, calibration, and visual evaluation.
An earlier author-supplied Qwen3.6-35B-A3B full-MMLU estimate lacked raw
responses and evaluation metadata. The manuscript uses the auditable local
full-set result of 11,601/14,042 (82.62%). The earlier estimate cannot be
reconciled as a repeat measurement under verified identical conditions.
Jev 1.13 full-MMLU accuracy is 89.06% as supplied by the author.
The 0.271 seconds/request for 4090 is 77.183 / 285, an amortized
complete-process cost, not a measured latency percentile.

The revised bibliography additionally cites the original Transformer paper,
Guo et al. on calibration, the EleutherAI evaluation harness, and llama.cpp.
Backend attribution was checked against `crates/jet-llama-sys/build.rs`, which
configures `GGML_CUDA`/`GGML_VULKAN` and links the upstream `ggml-cuda`/
`ggml-vulkan` libraries. JET's decision-specific orchestration is distinguished
from the upstream model runtime and GPU kernels. The report title is now
“JET: Justification Evaluation in Transformer”.

## Full-MMLU RTX 4090 runs

The local `tests/accuracy/generated/qwen35-full-mmlu-comparison-20260924.md`
lists the five Qwen3.5 runs and Qwen3.6 baseline. Each retained benchmark
directory contains `requests.jsonl`, `gold.jsonl`, `provenance.json`,
`summary.json`, `run-01/responses.jsonl`, and `run-01/report.json`.
These files are ignored by Git; the manuscript's aggregate data is preserved in
`figures/mmlu-results.json` and its full-MMLU table. All six runs used
binary SHA-256 `d0b01369a2c4fc08f2ba135ff9bed95a9d0bc0139f81da45d58cd7680f7de707`,
the same benchmark harness and evaluator hashes,
input SHA-256 `5393a9f78039744f999e7c403a0bac0fd7b26931f513e3f9e8dc8ef34d69cf7c`,
and gold SHA-256 `6f0e61960c7459f6738805a4c2e2f89b9287171d798760ea140f8b3bb57cf209`.
The evaluator accepts distinct option labels with identical answer text as
equivalent. Each run has one successful completion and no failed requests.
Filtering the Qwen3.6 full-run responses to the historical 285-question input
recovered 250/285 correct and all 285 of the same top-1 choices as the
separate subset run. This verifies that its higher subset percentage comes
from the question selection under those two measured runs.
`python scripts/prepare-accuracy-data.py --boolq-limit 0 --mmlu-all
--ascii-json --output-dir tests/accuracy/generated-mmlu-full-ascii` reproduced
both original input hashes from the pinned MMLU archive on September 28, 2026.

| Model | Correct / 14,042 | Accuracy | Wall (s) | Req./s |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B Q8_0 | 4,552 | 32.42% | 1,434.16 | 9.79 |
| Qwen3.5-2B Q8_0 | 6,818 | 48.55% | 1,443.47 | 9.73 |
| Qwen3.5-4B Q8_0 | 9,323 | 66.39% | 1,986.14 | 7.07 |
| Qwen3.5-9B Q8_0 | 10,344 | 73.66% | 2,466.26 | 5.69 |
| Qwen3.5-27B Q4_K_M | 11,948 | 85.09% | 6,353.75 | 2.21 |
| Qwen3.6-35B-A3B Q4_K_M | 11,601 | 82.62% | 3,370.71 | 4.17 |

The shared invocation is `python scripts/benchmark-model.py --binary
E:/jet-target/release/jet.exe --model-path models/<model>.gguf --model-id
<model-id> --input <full-MMLU-requests.jsonl> --gold
<full-MMLU-gold.jsonl> --output-dir <new-directory> --runs 1 --backend vulkan
--batch-requests 8 --extra-args --max-sequences 2 --micro-batch 256
--max-output-rows 256 --threads 8 --no-mmap`. Exact commands and source
revisions are in each `provenance.json`; Qwen3.5 model source revisions and
checksums are in the local `qwen35-model-sources-20260924.json` file.
The Qwen3.6 source revision and SHA-256 are pinned in
`scripts/download-accuracy-models.sh`.

The September 28 CUDA cross-check is retained in
`tests/accuracy/generated/qwen36-4090-cuda-mmlu-full-20260928/`.
It used the same request, gold, and evaluator hashes as the Vulkan full run,
but executable SHA-256
`fb8394921e604e8281cfae18b253d2b1608184c6f8e299896d4d6acff3057f83`
at source commit `81828a89754977561b590d27ac1bfb60931fa473`.
Its one successful run returned 14,042 responses, zero failures, and
11,595 correct (82.57%) in 2,829.36 seconds (4.96 requests/s).
There are 328 changed top-1 choices versus the older Vulkan binary's run;
the six-model table therefore keeps its common Vulkan binary, and the CUDA
wall-time difference is not presented as an isolated backend speedup.

## Larger-model and Jev comparison

- Jev 1.13: 89.06% on the complete MMLU test set, supplied by the user as a prior
  manual measurement. The user confirms a sampler-free decision interface with
  no configurable thinking or few-shot settings. This is the report author's
  own result, not attributed to TypeSafe. Raw responses and exact aggregation
  metadata have not been supplied; no matched-subset or paired comparison is
  claimed. Sampler-free behavior is treated as shared by both systems.
- Llama 3.1 70B/405B Instruct: 83.6/87.3, official Meta model card,
  English MMLU, 5-shot, `macro_avg/acc`. These are not the separate 0-shot
  CoT values (86.0/88.6) on that page.
- Qwen2.5-72B-Instruct / DeepSeek-V3: 85.3/88.5, DeepSeek's published chat-model
  comparison, MMLU exact match. The Qwen number is explicitly sourced to that
  comparison, not a new JET run. DeepSeek-V3 has 671B total / 37B activated
  parameters in the same source table.
- Best full-MMLU JET versus Jev score gap: 89.06 - 85.09 = 3.97 percentage points.
  Shared benchmark coverage does not establish identical prompting, scoring,
  aggregation, or statistical equivalence.

Self-authored benchmark/demo documents are no longer bibliography entries. The
small-model accuracy results are presented as valid decision-quality measurements;
execution optimizations preserve their outputs in the recorded equivalence checks.
Timing ablations remain separately identified so older timing configurations are
not mistaken for the optimized path.

The comparison also checks typed outputs directly against
`crates/jet-core/src/protocol.rs` and `crates/jet-engine/src/evaluator.rs`:
`noul` is the normalized true probability; `choice` is the argmax label; `score`
is the expected zero-based level index. Choice and score include probability
maps. This corrects the earlier draft's boolean/index output description.

## MMLU accuracy / throughput figure

- The user supplied Jev 1.13 full-MMLU accuracy of 89.06% and API throughput of
  2.86 req/s. Neither number is a provider-published result. API concurrency and
  detailed timing boundaries were not supplied.
- `figures/mmlu-results.json` contains the historical benchmark rows, six
  comparable Vulkan full-MMLU runs, and the later CUDA check, including repeated
  prefix runs and the response-equivalent GPU full-row-softmax configuration.
- CPU/A770 throughput covers mixed BoolQ/MMLU; RTX 4090 throughput covers MMLU
  alone. JET includes both subset and full-MMLU accuracy; Jev uses full MMLU.
- Figure 2 selects representative disabled-thinking configurations using
  `show_in_figure`; `figure_label` supplies reader-facing deployment names.
  Historical measurements remain in the data file. Each RTX 4090 full-MMLU
  point combines accuracy and throughput from the same run.
- Missing rates remain null. Where wall time is known but no rate is published,
  rate = 541 / complete-process seconds. Approximate CPU rate is marked as such.
- No throughput is imputed for external model-card results, invalid state-isolation
  experiments, or diagnostic slices. Those points are not part of the figure.
- Thinking improves BoolQ and combined accuracy in the recorded run but reduces
  MMLU accuracy. The report describes a task-dependent opportunity to exchange
  throughput for accuracy, not a universal MMLU improvement.

## Editorial revision

The editorial revision simplified prose and retained the original five displayed
equations and workflow diagram. This update adds a full-MMLU model table and
six measured RTX 4090 points to the accuracy--throughput figure. The 285-question
preparation study remains separate from the full-set model comparison.

## Token-to-decision algorithm

The Decision Method section retains the full mathematical chain: vocabulary
softmax, stable token log-probabilities, sequence likelihood and log-score,
candidate-set normalization, and output aggregation. All five displayed equations
from the original draft are retained. These definitions were checked against `target_log_probability` in
`crates/jet-engine/src/llama.rs`, device target-log-probability extraction in the
same file, `normalize_log_probabilities` in `crates/jet-core/src/math.rs`, and
answer construction in `crates/jet-engine/src/evaluator.rs`. Answer scoring uses
teacher-forced candidate tokens, full-vocabulary normalization, no temperature,
no end-of-turn score, and no length normalization. Optional reasoning is generated
once and becomes shared conditioning context. The numerical example illustrates
candidate normalization and is not an experimental measurement.
