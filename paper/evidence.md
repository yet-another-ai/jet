# Evidence map for the report

The manuscript now has one decision rule: softmax over each candidate's mean
conditional token log-probability. All active JET measurements come from
completed fresh runs of that implementation. Earlier benchmark outputs are retained as
historical artifacts and are not evidence for current accuracy, calibration,
throughput, prefix equivalence, or hardware comparisons.

The RTX 4090 campaign is **complete**: 13 full-MMLU runs and 18 backend
controls finished on September 29--30, 2026 (UTC). Arc A770 full-set and
shared-prefix measurements remain pending for the machine switch.

## Current result sources

- Campaign artifacts: `tests/accuracy/generated/4090-token-mean-20260930/`.
  Its `manifest.json`, `status.json`, and `results.json` record the campaign;
  each case attempt retains inputs, provenance, responses, reports, timings,
  and logs. A failed or partial attempt is never combined with a later run.
- Tracked aggregate evidence: `figures/rtx4090-evidence.json`.
- Manuscript numeric source: `figures/rtx4090-results.tex`, consumed through
  `\jetresult{case-id}{metric}`. `update_results.py` emits pending entries
  before completion and imports validated campaign results afterward.
- Accuracy/throughput figure data: `figures/mmlu-results.json`.
  Displayed JET points require complete mean-token full-MMLU measurements;
  pending Arc A770 and historical execution-study points are excluded.

The exporter validated the complete 31-case matrix before publishing this
campaign. No previous accuracy or runtime fills a missing case. Signed model
differences, CUDA/Vulkan changed-answer counts, LoRA transitions and subject
counts, and the Jev gap derive from the same new records.

## Full-MMLU RTX 4090 protocol

Every configuration evaluates 14,042 MMLU test questions from 57 subjects,
with reasoning disabled and full GPU placement. The six pretrained models are
Qwen3.5-0.8B, 2B, 4B, 9B, and 27B, and Qwen3.6-35B-A3B; each runs once with
CUDA and once with Vulkan. Models through 9B use Q8_0, and the two larger models
use Q4_K_M. The thirteenth run evaluates Qwen3.5-4B LoRA Q8_0 with CUDA.

Frozen Windows input files are
`tests/accuracy/generated-mmlu-full-ascii/requests.jsonl` and `gold.jsonl`.

- Input SHA-256: `5393a9f78039744f999e7c403a0bac0fd7b26931f513e3f9e8dc8ef34d69cf7c`.
- Gold SHA-256: `6f0e61960c7459f6738805a4c2e2f89b9287171d798760ea140f8b3bb57cf209`.
- Pinned MMLU archive SHA-256: `bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b`.
- MMLU repository revision: `hendrycks/test@4450500f923c49f1fb1dd3d99108a0bd9717b660`.

`prepare-accuracy-data.py --boolq-limit 0 --mmlu-all --ascii-json` regenerates
these question records. Windows CRLF and Linux LF JSONL differ in byte hashes;
record equality and order establish dataset identity across those serializations.

Each fresh-process benchmark uses eight requests per chunk, two sequence slots,
a microbatch size of 256, an output-row limit of 256, eight CPU threads, and
no memory mapping. The executable, input, gold, runtime settings, and evaluator
must match within the campaign. No visual projector is loaded. Successful
completion requires 14,042 responses and zero evaluation failures. Wall time
includes startup, model loading, processing, and shutdown; throughput is
14,042 divided by that time. There is no repeat-run variance estimate for
the full suite.

Exact invocation and source state are recorded in each case's provenance.
Qwen3.5 model revisions and hashes are retained in
`tests/accuracy/generated/qwen35-model-sources-20260924.json`; pinned downloads
for the smaller models and Qwen3.6 are defined in
`scripts/download-accuracy-models.sh`. The campaign records current model
identity and binary/source hashes rather than inheriting historical binary hashes.

## Same-executable backend controls

The frozen 570-question files are retained in
`target/backend-ablation-20260929/requests.jsonl` and `gold.jsonl`.
For each subject, selection takes the ten smallest SHA-256 hashes of `gold.id`,
then restores full-set order. Model responses play no role in selection.

- Control input SHA-256: `25a390ad5ebba58b68277b028a7c387f3c31b2d4a82beea949aa59e71fbbc678`.
- Control gold SHA-256: `9d3c49baa93050fcd662b0a1253aa06bb66b6e8a2ecfd6a81be98ad3f34d9884`.

The new control suite contains eighteen runs, all with mean-token scoring:

| Model | Cases |
| --- | --- |
| Qwen3.5-0.8B | CPU; CUDA and repeat; default Vulkan and repeat; the five Vulkan variants below |
| Qwen3.5-2B | CUDA; default Vulkan; disable COOPMAT2; disable both cooperative-matrix paths |
| Qwen3.6-35B-A3B | Same four cases as 2B |

The five Vulkan variants disable COOPMAT2; COOPMAT2 plus COOPMAT; both plus
F16; both plus INTEGER_DOT_PRODUCT; or all four. Environment variables use the
`GGML_VK_DISABLE_` prefix with value `1`. FP16 and integer-dot-product
effects are measured conditional on disabled cooperative-matrix paths,
not as independent changes from the default path. Logs record actual feature
selection and layer placement.

Top-1 label changes, semantic answer changes, decision-weight differences, and
repeated-run equality are computed from fresh paired responses. Distinct labels
with identical answer text count as the same semantic answer. No causal
attribution of cross-device differences to a particular Vulkan feature follows
from this single-device suite.

## Qwen3.5-4B LoRA provenance

The completed training run is `training/runs/qwen35-4b-lora/completed.json`.
It records Qwen/Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, one epoch, 1,425 steps,
22,791 retained records, and zero overlength drops. The log reports
32,464,896 trainable of 4,238,216,192 total parameters (0.7660%).
The mixture contains 9,427 BoolQ, 2,248 ARC-Easy, 1,116 ARC-Challenge,
and 10,000 SNLI examples.

`training/data/default/manifest.json` records source revisions and hashes,
10,708 validation records, split isolation, and exclusion of all 14,042 frozen
MMLU requests. A 512-example validation subset monitors loss. Exact normalized
overlap checks do not rule out paraphrases or pretraining contamination.
The objective is next-token cross-entropy over correct answer text,
not direct optimization of the candidate-set decision distribution.

Adapter SHA-256:
`2e148c2ef49af6a02e6c2cbcd69f3f30ef3386d6763e466243c44e4622fcf20f`.

The retained text-only export is
`training/exports/qwen35-4b-lora-text-Q8_0.gguf`, with provenance in
`training/exports/qwen35-4b-lora-merged/export.json` and SHA-256
`03dd562dabca7dedf49f2523b1806e58ba6e626ff3377c4f959e041fa70fa737`.
It uses `--no-nextn`, declares 32 blocks, and excludes MTP weights.
The adapter and export remain unchanged; inference is rerun with the new rule.

The comparison uses campaign cases `full-4b-cuda` and `full-4b-lora-cuda`.
Their accuracy, NLL, Brier score, throughput, paired transitions, and subject
counts are regenerated together. Historical paired-analysis files under
`training/runs/` are not imported as current results.

The reference GGUF includes MTP weights and comes from
`bartowski/Qwen_Qwen3.5-4B-GGUF` revision
`4168f45a16a1290d65a4ec0fa312ae917a4c15d6`. Exact upstream weight identity
with the pinned HF training checkpoint has not been verified. A zero-adapter
export of that checkpoint has not been evaluated. The comparison therefore
characterizes deployed artifacts under a matched protocol, not a LoRA-only
causal effect. One timing run per artifact does not establish a repeatable speedup.

## Arc A770 and execution studies: pending

Four Qwen3.5 Q8_0 full-MMLU reruns are required on the Arc A770: 0.8B, 2B,
4B, and 9B. Use the current mean-token binary, a fresh campaign tag, and
`scripts/run-a770-full-mmlu.sh`. No old A770 measurement supplies a current
table row or figure point. Hardware for the planned follow-up is an i9-13900K,
64 GiB RAM, and a 16 GiB Arc A770.

The shared-prefix study also requires a fresh comparison under the current
decision rule. Its workload is 256 label-balanced BoolQ validation examples
plus five MMLU test examples per subject, totaling 541 requests
(seed 20260921). Intended settings are eight requests per chunk, nine
sequence slots, 2,048 context tokens per sequence, token batches of 2,048,
a microbatch size of 512, and an output-row limit of 256. Both independent
execution and prefix reuse must use mean-token scoring. The planned matrix
has one independent-execution run and two prefix-reuse runs per model, with
model order reversed in the second round.

Speedup, prefix-token counts, exact-response equivalence, and state-isolation
diagnostics remain pending. Historical prototype preparation and rejected
batching experiments are not current-method evidence. The article makes no
numerical claim about their outputs or performance.

## External comparisons

- Jev 1.13: author-supplied manual full-MMLU accuracy of 89.06% and separately
  reported API throughput of 2.86 requests/s. Concurrency and detailed timing
  boundaries were not supplied. The interface is sampler-free, with no
  configurable reasoning or few-shot setting. These are the author's values,
  not a provider-published measurement.
- Llama 3.1 70B/405B Instruct: 83.6/87.3 from the official Meta model card,
  English MMLU, 5-shot, subject-macro accuracy.
- Qwen2.5-72B-Instruct / DeepSeek-V3: 85.3/88.5 in DeepSeek's published
  chat-model comparison, MMLU exact match. DeepSeek-V3 has 671B total /
  37B activated parameters in that source.
- The signed Jev-minus-best-JET gap is regenerated from the complete campaign.
  Shared benchmark coverage does not establish identical prompts, scoring,
  aggregation, or statistical equivalence. No throughput is imputed for
  published model-card results.

## Implementation boundary

`crates/jet-engine/src/llama.rs` supplies teacher-forced token log-probability
sums and actual candidate token counts. `crates/jet-engine/src/evaluator.rs`
forms candidate means before normalization in `crates/jet-core/src/math.rs`.
Counts include the scored serialized continuation, including quotations and
escapes, but exclude context, reasoning, and image tokens. No end-of-turn token
or temperature is added to candidate scoring. Optional reasoning is generated
once and becomes shared conditioning context.

`noul` returns the normalized true weight, `choice` the argmax label, and
`score` the expected zero-based level index. Choice and score retain weight
maps. These softmax values are decision weights, not candidate-restricted
sequence likelihoods or demonstrated calibrated confidence. The numerical
method example is illustrative rather than a benchmark result.

JET orchestrates decision inference; upstream llama.cpp supplies model execution
and GPU kernels. Native submodule revision is recorded with each campaign.
Multimodal interfaces remain implemented without a quantitative visual-quality
claim. Before external circulation, archive raw artifacts and source/binary
provenance, then add matched-system baselines, repeated timing measurements,
calibration studies, and visual evaluation.
