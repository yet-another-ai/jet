# Evidence map for the report

Prepared against Jet `4ff0ab63018e79e5f4639aef97924f4e26bb0405`, with native
submodule `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. These identify the source
snapshot consulted, not every historical benchmark binary. The RTX 4090 full-MMLU
runs were performed on September 23--24, 2026; the Arc A770 runs followed on
September 28, and the RTX 4090 CUDA series finished on September 29.
Figures are rounded from the recorded experiments.

| Report claim/table | Source record | Conditions and interpretation |
| --- | --- | --- |
| Arc A770 full-MMLU model comparison | Local `tests/accuracy/generated/a770-*-mmlu-full-20260928-fixed/` | Two Qwen3.5 Q8_0 models, 14,042 questions each, full GPU placement, one fresh-process run/model, zero failures. |
| 2.18–2.23x prefix comparison | `docs/accuracy.md`, Vulkan serial-prefix reuse benchmark | One old run, two new runs/model; bundles prefix reuse and metadata resets. Local `vulkan-prefix-benchmark-20260922/summary.json` cross-checked. |
| 66.03% fewer prefill tokens | Same prefix section | 378,526 old tokens derived from schedule; 128,583 new tokens instrumented. Not a wall-time estimate. |
| Historical prototype preparation study (excluded from manuscript) | `docs/performance.md`, RTX 4090 preparation pipeline / Fixed 285-question MMLU comparison | 285 questions only, Vulkan full GPU, one final run/configuration, loading included; raw 4090 artifacts not present locally. |
| Full-MMLU model comparison | Local `tests/accuracy/generated/qwen35-*-4090-mmlu-full-20260924/` and `qwen36-4090-mmlu-full-fixed-20260923/` | Six complete 14,042-question RTX 4090 Vulkan runs, one/model, same binary/input/gold; each has 14,042 responses and zero evaluation failures. Summary below. |
| Full-MMLU CUDA comparison | Local `tests/accuracy/generated/qwen35-*-4090-cuda-mmlu-full-20260929/` and `qwen36-4090-cuda-mmlu-full-20260928/` | Six complete 14,042-question RTX 4090 CUDA runs, one/model, same binary/input/gold and run settings. The same-executable 570-question control below confirms backend-dependent answer changes for the three models tested. |
| Hybrid correctness and rejected batching | `docs/accuracy.md`, Hybrid recurrent batching correctness; `docs/performance.md`, Deferred independent-request experiment | Evidence for retaining serial recurrent continuations; not evidence of a universal native-backend bug. |
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
The historical 285-question preparation study is excluded from the manuscript.

The revised bibliography additionally cites the original Transformer paper,
Guo et al. on calibration, the EleutherAI evaluation harness, and llama.cpp.
Backend attribution was checked against `crates/jet-llama-sys/build.rs`, which
configures `GGML_CUDA`/`GGML_VULKAN` and links the upstream `ggml-cuda`/
`ggml-vulkan` libraries. JET's decision-specific orchestration is distinguished
from the upstream model runtime and GPU kernels. The report title is now
“JET: Justification Evaluation in Transformer”.

## Full-MMLU Arc A770 runs

The two selected local `tests/accuracy/generated/a770-*-mmlu-full-20260928-fixed/`
directories retain input, gold, provenance, responses, timing, and scoring
reports. Each contains 14,042 responses and zero failures. Both runs use
the same input and executable. The executable includes fixes for duplicate
choice texts and small positive Vulkan log-probabilities caused by rounding.
A partial Qwen3.6-35B-A3B Arc A770 attempt was stopped and
is excluded from the manuscript.

| Model | Correct / 14,042 | Accuracy | Wall (s) | Req./s |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B Q8_0 | 4,615 | 32.87% | 3,015.29 | 4.66 |
| Qwen3.5-2B Q8_0 | 6,807 | 48.48% | 4,219.81 | 3.33 |

The manuscript uses these complete-set measurements for Arc A770 model quality and throughput.

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
Those hashes are for the Windows run's CRLF JSONL. Linux LF output from the
same preparation command hashes differently; replacing LF with CRLF in the
Linux output reproduces both recorded hashes exactly. This is a line-ending
difference, not a different set of MMLU questions.

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

The RTX 4090 CUDA series consists of the September 28 Qwen3.6 run under
`tests/accuracy/generated/qwen36-4090-cuda-mmlu-full-20260928/` and five
September 29 Qwen3.5 runs under
`tests/accuracy/generated/qwen35-*-4090-cuda-mmlu-full-20260929/`.
The CUDA runs use the same request and gold files and evaluation settings as
the Vulkan runs. The CUDA invocation selects `--backend cuda` and writes to a
separate output path.
Each CUDA model ran once in a fresh process with full GPU placement.

| Model | Correct / 14,042 | Accuracy | Wall (s) | Req./s |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B Q8_0 | 4,600 | 32.76% | 991.75 | 14.16 |
| Qwen3.5-2B Q8_0 | 6,800 | 48.43% | 1,027.70 | 13.66 |
| Qwen3.5-4B Q8_0 | 9,332 | 66.46% | 1,434.30 | 9.79 |
| Qwen3.5-9B Q8_0 | 10,330 | 73.57% | 1,682.73 | 8.34 |
| Qwen3.5-27B Q4_K_M | 11,925 | 84.92% | 3,832.36 | 3.66 |
| Qwen3.6-35B-A3B Q4_K_M | 11,595 | 82.57% | 2,829.36 | 4.96 |

All six summaries record one completed run, 14,042 responses, zero evaluation
failures, and the CUDA backend. Their logs select `CUDA0` and offload every
model layer to the RTX 4090. The displayed rate is 14,042 divided by each
run's full-process wall time; there is no repeat-run variance estimate.

Qwen3.6 has 328 changed top-1 choices between the CUDA and Vulkan runs.
A same-executable control on 570 questions (ten per subject) changes 16, 12,
and 14 top-1 choices for Qwen3.5-0.8B, Qwen3.5-2B, and Qwen3.6-35B-A3B,
respectively, when switching between CUDA and Vulkan. On those rows, its
outputs reproduce the corresponding historical full-run projections exactly.
The control establishes backend-dependent answer changes on the measured
subset; the complete-run throughput values remain the observed rates of their
respective backend runs.

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
- `figures/mmlu-results.json` contains two Arc A770 and twelve RTX 4090 full-MMLU
  runs, plus historical Qwen3.5 prefix measurements.
- Every displayed JET point combines full-MMLU accuracy and throughput from
  the same run. Jev uses full-MMLU accuracy and a separately reported API rate.
- Figure 2 selects representative disabled-thinking configurations using
  `show_in_figure`; `figure_label` supplies reader-facing deployment names.
  Historical measurements remain in the data file.
- For each displayed JET point, the plotted rate is request count divided by
  complete-process wall time. Jev uses the separately supplied API rate.
- No throughput is imputed for external model-card results, invalid state-isolation
  experiments, or diagnostic slices. Those points are not part of the figure.

## Editorial revision

The editorial revision simplified prose and retained the original five displayed
equations and workflow diagram. This update adds full-MMLU model tables and
two Arc A770 and twelve RTX 4090 points to the accuracy--throughput figure. The
prototype 285-question preparation study is excluded from the manuscript and figure data.

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
