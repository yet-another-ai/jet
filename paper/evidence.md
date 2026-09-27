# Evidence map for the first report draft

Prepared against Jet `4ff0ab63018e79e5f4639aef97924f4e26bb0405`, with native
submodule `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. These identify the source
snapshot consulted, not every historical benchmark binary. No new model runs
were performed for the report. Figures are rounded from the recorded experiments.

| Report claim/table | Checked-in source | Conditions and interpretation |
| --- | --- | --- |
| Semantic prompt comparison | `docs/accuracy.md`, Accuracy baseline | Same 541 examples, Qwen3-0.6B Q8_0 CPU; prompt and candidate representation change together. |
| Small-model decision-quality results | `docs/accuracy.md`, Disabled-thinking model comparison | Arc A770, Q8_0, corrected hybrid execution before serial prefix reuse; fresh-process timing. |
| CPU/Vulkan and target gather | `docs/accuracy.md`, Vulkan backend comparison | CPU time approximate; CPU/Vulkan outputs differ. Target gather comparison is against full-row GPU softmax. |
| 2.18–2.23x prefix comparison | `docs/accuracy.md`, Vulkan serial-prefix reuse benchmark | One old run, two new runs/model; bundles prefix reuse and metadata resets. Local `vulkan-prefix-benchmark-20260922/summary.json` cross-checked. |
| 66.03% fewer prefill tokens | Same prefix section | 378,526 old tokens derived from schedule; 128,583 new tokens instrumented. Not a wall-time estimate. |
| 35B MoE placement/accuracy | `docs/accuracy.md`, Qwen3.6 Q4 CPU/Vulkan offload | 541 examples, 11 CPU expert layers, i9-13900K/64 GiB/16 GiB A770. |
| 7.36% worker improvement | `docs/performance.md`, Full 541-request validation | Single full run/version, byte-identical responses; separate 32-case probe supports direction. Local full-pool541 report cross-checked. |
| 4090 execution-ablation table | `docs/performance.md`, RTX 4090 preparation pipeline / Fixed 285-question MMLU comparison | 285 questions only, Vulkan full GPU, one final run/configuration, loading included; raw 4090 artifacts not present locally. |
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
The author subsequently recovered the RTX 4090 Qwen3.6-35B-A3B full-MMLU
result from another machine: 87.48%. The corresponding configuration is shown
at 3.693 req/s using its recorded optimized execution rate, as requested by the
author; this is not a newly supplied timing for the full-set run. No new inference
was run. Jev 1.13 full-MMLU accuracy is corrected from 89% to 89.06%.
The 0.271 seconds/request for 4090 is 77.183 / 285, an amortized
complete-process cost, not a measured latency percentile.

The revised bibliography additionally cites the original Transformer paper,
Guo et al. on calibration, the EleutherAI evaluation harness, and llama.cpp.
Backend attribution was checked against `crates/jet-llama-sys/build.rs`, which
configures `GGML_CUDA`/`GGML_VULKAN` and links the upstream `ggml-cuda`/
`ggml-vulkan` libraries. JET's decision-specific orchestration is distinguished
from the upstream model runtime and GPU kernels. The report title is now
“JET: Justification Evaluation in Transformer”.

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
- Full-MMLU JET versus Jev score gap: 89.06 - 87.48 = 1.58 percentage points.
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
- `figures/mmlu-results.json` contains 22 rows from the benchmark notes and the
  subsequently supplied full-MMLU result, including repeated
  prefix runs and the response-equivalent GPU full-row-softmax configuration.
- CPU/A770 throughput covers mixed BoolQ/MMLU; RTX 4090 throughput covers MMLU
  alone. JET includes both subset and full-MMLU accuracy; Jev uses full MMLU.
- Figure 2 selects the best-performing disabled-thinking configuration per model
  and deployment using `show_in_figure`; `figure_label` supplies reader-facing
  deployment names. Historical measurements remain in the data file. The RTX 4090
  entry uses full-MMLU accuracy and separately measured subset throughput.
- Missing rates remain null. Where wall time is known but no rate is published,
  rate = 541 / complete-process seconds. Approximate CPU rate is marked as such.
- No throughput is imputed for external model-card results, invalid state-isolation
  experiments, or diagnostic slices. Those points are not part of the figure.
- Thinking improves BoolQ and combined accuracy in the recorded run but reduces
  MMLU accuracy. The report describes a task-dependent opportunity to exchange
  throughput for accuracy, not a universal MMLU improvement.

## Editorial revision

The editorial revision simplifies prose and removes unrelated engineering
descriptions. It preserves the original five displayed equations, five experimental
tables, the complete workflow diagram (Figure 1), and all 22 configurations in the
accuracy--throughput comparison (Figure 2). Figure sources and measurements are
unchanged. Quantitative results and their experimental qualifications are retained;
source records above provide implementation details and artifact provenance.
The revised text explicitly separates the 87.48% full-MMLU accuracy from the
3.693 req/s rate measured on the 285-question execution study.

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
