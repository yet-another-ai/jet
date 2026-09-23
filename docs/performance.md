# Qwen3.6 CPU/Vulkan profiling

This profile follows the [Qwen3.6 Q4 benchmark](accuracy.md#qwen36-q4-cpuvulkan-offload)
on the i9-13900K / 16 GiB Arc A770. The model, quantization, 11 CPU expert layers,
eight CPU threads, two sequence slots, 2,048-token context, and 256-token
microbatch/output-row limits are unchanged. Thinking is disabled and weights use
allocated memory (`--no-mmap`).

## Measurement method

The reproducible profile sample contains 32 requests: 16 randomly selected BoolQ
and 16 MMLU examples, with seed 20260922. It is a performance probe, not a new
accuracy benchmark. An additional eight-case mixed sample diagnoses individual
changes. Each variant runs alone in a fresh process with model files already
cached. Stage times exclude loading; complete process wall time includes it.

Production-path observations use i915 DRM fdinfo and `/proc` at 200 ms intervals.
The response file is created after model loading, so its existence marks the
inference sampling window. GPU busy is the change in the render-engine time
counter divided by elapsed time, with duplicate DRM clients removed. It measures
engine busy time, **not shader occupancy, memory-bandwidth saturation, or a
percentage of peak FLOPS**. Process CPU time includes user and system time and
can exceed elapsed time because workers execute in parallel.

A separate `perf record -e cpu-clock:u -F 99 --call-graph dwarf,4096` run samples
CPU execution. Separate `strace -f -c -e trace=clone,clone3,futex` runs count thread
creation. Traced wall times and summed syscall waiting time are not used as
performance results.

Local artifacts, binary snapshots, input selections, source patches, commands,
logs, counters, and reusable analysis scripts are in
`tests/accuracy/generated/qwen36-profile-20260922/`.

## CPU experts interrupt the GPU dependency chain

The original configuration reproduces approximately 62% GPU engine busy. Each
decode is split into 24 backend graphs: CPU expert islands alternate with GPU
attention/shared-expert work. The next layer cannot consume an expert result
before its CPU computation finishes. This is not equivalent to concurrent CPU
and GPU execution of independent requests; Jet's hybrid scorer currently processes
requests and candidate continuations serially for correctness.

Jet previously left the native CPU threadpool unset. With OpenMP disabled in this
build, ggml created and freed a disposable pool for every CPU graph. The 32-case
perf trace contains **10,472 short-lived workers**, exactly
`136 decodes × 11 CPU expert layers × 7 workers`. Median worker lifetime is
2.39 ms. The independent eight-case syscall probe counts 2,697 thread creations;
with a persistent pool the same probe counts only 9, including driver helpers.

The retained implementation gives each scorer one owned native pool and attaches
it to both scoring and thinking contexts. The contexts execute serially, and CPU
graph computation is synchronous. Both contexts are destroyed before the pool.

Recorded user-space CPU samples primarily fall in `ggml_vec_dot_q4_K_q8_K`
(78.15%), Vulkan fence active waiting (6.78%), CPU graph dispatch (3.54%), and CPU
barriers (3.47%). These percentages are not a wall-time decomposition. Only 2,568
of the 10,472 short-lived workers received a sample before exiting; the sample
distribution therefore understates some short tasks. There were 5,501 samples
and no lost events, which does not imply complete sampling of short threads.
This user-space capture covers loading and shutdown as well as inference and
includes five Python harness samples; it does not measure kernel CPU execution.

## Isolated comparisons

The following 32-case runs use no sampling profiler or syscall tracer. GPU/CPU
observations come only from the lightweight process-counter collector.

| Variant | Scorer seconds/request | GPU engine busy | Process CPU seconds/second |
| --- | ---: | ---: | ---: |
| Disposable threadpools | 1.198 | 62.18% | 2.455 |
| Persistent pool, sleeping workers (`poll=0`) | 1.103 | 67.14% | 2.079 |
| Same pool, process limited to eight physical P cores | 1.148 | 64.96% | 2.229 |
| Same pool, CPU expert weights repacked | 1.113 | 66.25% | 2.190 |
| Persistent pool, native busy-poll window (`poll=50`) | 1.089 | 67.85% | 6.336 |

Persistent sleeping workers reduced scorer time by 7.95% in this comparison;
all 32 response probabilities and usage values were unchanged. Restricting the
whole process to CPUs `0,2,4,6,8,10,12,14` was slower. Although many original
short-lived workers were observed on E cores, these observations alone do not
justify forcing P-core affinity; process affinity also constrains driver work.

The native busy-poll window was only 1.2% faster than sleeping workers in this
single-run probe, while consuming about three times as much process CPU time.
The retained pool uses `poll=0`; the small polling-time difference does not justify
making those CPU resources busy during GPU execution.

The repack experiment set native `no_host=true` for CPU expert weights. It selected
4,752 MiB of `CPU_REPACK` weights while preserving pinned activation buffers and
GPU placement. It did not establish a scoring speed benefit, increased loading
time, and changed some candidate probabilities (maximum absolute change 0.1203,
no top-1 changes on these 32 examples). It was not retained.

An eight-case comparison also moved the first 11 complete layers to CPU using
`--gpu-layers 30`, replacing the 11 expert-only overrides. This reduced graph
splits from 24 to 2, but increased CPU computation:

| Placement, persistent sleeping pool | Prefill seconds/request | Candidate seconds/request | Scorer seconds/request |
| --- | ---: | ---: | ---: |
| CPU experts only | 0.707 | 0.258 | 0.977 |
| Complete CPU layers | 0.939 | 0.222 | 1.173 |

The complete-layer placement reduced candidate time but made prefill worse;
it was about 20% slower overall and was not selected. This experiment changes
both backend transitions and computation placement, so their effects are not isolated.
The eight-case timings are not directly comparable with the separate 32-case set.

## Full 541-request validation

The final production binary was rebuilt with sleeping persistent workers and
without native profiling patches. The same full accuracy workload and placement
completed successfully:

| Measurement | Original workers | Persistent workers |
| --- | ---: | ---: |
| Complete process wall time | 647.607 s | 599.949 s |
| Amortized wall seconds/request | 1.1971 | 1.1090 |
| Scorer seconds/request, excluding loading | 1.1896 | 1.1014 |
| Total prefill time | 442.901 s | 400.351 s |
| Total candidate continuation time | 193.959 s | 188.811 s |
| Correct answers | 479/541 (88.54%) | 479/541 (88.54%) |

Wall time fell 7.36%; scorer time fell 7.42%. Prefill improved by 9.61% and
candidate evaluation by 2.65%. These are one full run per version, supported by
the separate controlled 32-case comparison, not latency percentiles or repeated
confidence intervals. The result still exceeds the 0.5-second/request budget.

The complete response files are byte-identical, including all probabilities and
usage values. BoolQ remains 230/256 (89.84%), MMLU 249/285 (87.37%), and mean gold
NLL 0.35531. Both versions process 128,583 prompt tokens and 9,795 candidate
tokens with 2,369 decodes. Response SHA-256 is
`d2a818c399c39fb09d913b5912376d87b26bb55ca05477c9fa66de74c6c94b96`.

The final run's 2,926 inference snapshots show 66.08% GPU engine busy and
2.128 process CPU seconds per elapsed second. Peak sampled device residency is
14.90 GiB; GPU frequency has a 2,400 MHz median and temperature peaks at 68 C.
The full workload differs from the 32-case probe, so the 62.18% to 67.14%
before/after busy comparison should use that matched probe.

CPU and Vulkan workspace tests and Clippy passed, as did model-backed CPU
reference/prefix recovery, Vulkan hybrid/reference equivalence, and a two-batch
mixed CPU/Vulkan thinking-context smoke test. The vendor tree is unchanged.
Full-run artifacts are in `full-pool541/`, comparison details in
`baseline-pool-full541-comparison.json`, and validation commands/results in
`verification-summary.json` under the profile artifact directory.

## GPU operation hotspots

A separate eight-request diagnostic uses the native Vulkan performance logger.
It records 5.647 seconds of timestamp intervals over 52,226 node/fusion calls:

| Operation family | Share of diagnostic GPU intervals |
| --- | ---: |
| Routed Q4_K expert matrix multiplication | 37.99% |
| Other dense matrix multiplication | 32.23% |
| Gated Delta Net | 14.41% |
| RMS normalization | 4.37% |
| Full attention | 1.90% |
| Output vocabulary projection | 1.43% |
| MoE routing/top-k/softmax fusion | 1.29% |
| All row gathers, including model-internal gathers | 0.40% |
| Score softmax and log | 0.09% |

These are **diagnostic interval shares, not production wall-time shares or
available speedups**. The logger inserts barriers and waits at GPU graph splits;
a temporary diagnostic-only patch also drains pending input copies before query
initialization to avoid a mixed-backend logger assertion. The intervals can
include internal conversions and queue gaps. This eight-case run has no explicit
warm-up, and the logger combines all Gated Delta Net shapes. The logger patch was
removed before rebuilding and testing the production binary.

The largest expert shape families are gate/up (`m=512, k=2048`, 25.13%) and
down (`m=2048, k=512`, 12.86%), both with 256 experts. Dense Q8_0 QKV projection
(`m=8192, k=2048`, 8.55%) and SSM output projection
(`m=2048, k=4096`, 6.17%) are also useful targets. The output vocabulary tensor
is Q6_K in this Q4_K_M model. The complete shape and fusion breakdown is in the
local `gpu-kernel-summary.md` and `.json` artifacts.

The most concrete dispatch lead is expert-down multiplication for short token
chunks. Native `ggml_vk_use_mul_mat_vec_id` uses the vector path through eight
tokens and the matrix path above eight. The diagnostic averages 2.120/2.222 ms
for 15/17 tokens, versus 1.617 ms for 158 tokens. The logger's `_VEC` label is
only applied to one-token entries, so it does not reliably identify this switch.

A follow-up standalone benchmark confirms a sharp crossover without the logger.
It uses Q4_K weights `[512,2048,256]` (144 MiB), eight unique routed experts per
token, and F32 activations `[512,8,n]` matching the expert-down layout. Each shape
has ten warm-up and thirty measured synchronous graph evaluations, with all
tensors already on the GPU. Two separate runs record these medians:

| Tokens | First run, ms | Repeat, ms |
| --- | ---: | ---: |
| 8 | 0.518 | 0.518 |
| 9 | 2.881 | 2.815 |
| 15 | 4.144 | 4.160 |
| 17 | 4.632 | 4.611 |
| 32 | 7.365 | 7.246 |
| 158 | 2.598 | 2.586 |
| 256 | 2.650 | 2.635 |

The 8-to-9-token transition costs 5.56x and 5.43x in these runs. Broadcast
activations `[512,1,n]` reproduce the discontinuity in two further runs. Source
inspection finds a dispatch heuristic rather than an eight-token shader capacity
limit: the vector path allocates descriptors dynamically and submits one dispatch
per token. Raising the threshold would also increase CPU submission work, and
the shared dispatch helper controls fusion eligibility. The non-monotonic costs
above eight tokens also warrant investigation of matrix tile selection; they
cannot all be attributed to the vector/matrix switch.

These standalone times include CPU submission, internal conversions, and GPU
completion synchronization. Repeated expert contents, fixed random routing, warm
caches, and isolated operation submission differ from full-model execution; the
5.5x discontinuity is **not a model speedup estimate**. The experiment checks
finite outputs but does not validate a new computation path. No dispatch threshold
or shader change is retained. Before selecting a threshold, compare vector and
matrix alternatives on the same shapes, verify their numerical results, then
measure complete requests. Source, build instructions, four raw result files, and
analysis are saved as the local `moe-dispatch-*` artifacts.

The A770's `matrix cores: none` initialization message is intentional in this
native revision: cooperative-matrix shader support is compiled in, but the device
support check disables it for Xe1/A770 because of reported performance
regressions. Enabling it needs its own correctness and performance experiment.

## Next optimization targets

The RTX 4090 preparation work described below leaves these GPU backend targets unchanged.

1. Tune the confirmed short-chunk Q4_K expert dispatch crossover, followed by the large Q8_0
   projections. These account for most recorded GPU operation time. Compare
   warmed standalone shapes first, then full requests with numerical checks.
2. Separate recurrent and chunked-prefill Gated Delta Net measurements. The
   current 14.41% aggregate cannot identify which shader variant needs work.
3. Investigate independent-request batching/overlap in the hybrid scorer, with
   explicit recurrent-state isolation and reference checks. `--batch-requests`
   alone does not make its serial hybrid execution concurrent. Increasing input
   microbatch capacity independently of output-row storage is another possible
   way to avoid short tail chunks, subject to the remaining VRAM budget.

GPU engine busy near 60% does not imply 40% spare throughput: CPU dependencies
cause idle gaps, while busy periods can still contain inefficient GPU work.
Even eliminating all idle gaps, with GPU work otherwise unchanged, would leave
about 0.74 seconds/request in the original 32-case probe. This is a conditional
estimate, not a hardware lower bound; reaching the 0.5-second budget also calls
for more efficient GPU execution or amortizing work across requests.

## RTX 4090 preparation pipeline (2026-09-23)

This change deliberately leaves Vulkan kernels and recurrent scoring order unchanged:

- A scorer-owned native renderer retains the parsed chat template. Each render returns its
  complete JSON once instead of separately rendering to query size and then fill a buffer.
- A retained token buffer avoids repeated sizing passes, and the already-tokenized prompt is
  reused. Each full prompt-plus-candidate is still tokenized to verify the assistant boundary;
  disabled/enabled template checks, context limits, and score validation remain intact.
- For disabled-thinking, batched Vulkan hybrid execution, a persistent CPU worker prepares
  upcoming questions behind a bounded result queue. The GPU still scores one question and one
  candidate fork at a time. `--no-preparation-pipeline` provides a synchronous comparison.
- `prepare_wait_ms` measures visible consumer stalls. Worker `prepare_ms` excludes queue
  backpressure and overlaps scoring, so phase totals are not additive.

### Fixed 285-question MMLU comparison

Qwen3.6-35B-A3B Q4_K_M ran on the RTX 4090 with Vulkan, all model layers on GPU, disabled thinking,
eight input requests per CLI chunk, two native sequence slots, microbatch/output rows 256,
eight CPU threads, and `--no-mmap`. These are sequential fresh-process runs, with startup and
roughly ten seconds of model loading included; no concurrent compilation or model tests ran
during this final comparison.

| Configuration | Wall time | Requests/s | Prepare time | Visible prepare wait | Prefill / candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved old binary | 111.460 s | 2.557 | 33.451 s | synchronous | 24.080 / 42.604 s |
| Cached preparation, synchronous | 80.420 s | 3.544 | 3.640 s | synchronous | 23.767 / 42.147 s |
| Cached preparation, pipeline | 77.183 s | 3.693 | 3.853 s | 0.464 s | 23.864 / 42.182 s |

The synchronous cache change reduced preparation time by 89.1%; pipeline overlap saved another
3.24 seconds of process wall time in this sequence. Overall wall time fell 30.8% and throughput
rose 44.4% versus the repeated baseline. These are observations, not confidence intervals or
an extrapolation to the full MMLU set. Earlier exploratory runs measured 142.028 / 80.931 /
86.910 seconds respectively, with compilation overlapping some new-binary runs and visibly
different GPU-stage times; they are retained but not used for the final speedup comparison.

Every variant returned 285 responses, 250 correct (87.7193%), mean gold NLL
`0.44754015556245685`, and zero failures. Output JSONL was byte-identical, SHA-256
`b4d1617c0e1b364a2fce24c50c81569590b844faf64177aea2cd62204f54155a`.
All three final runs performed 285 prefills, 60,680 prefill tokens, 9,283 continuation decode
tokens, and 1,472 decode calls. The optimization does not skip scoring work.

Final raw artifacts are `baseline-repeat/`, `final-cached-sync/`, and `final-pipeline/` beneath
`tests/accuracy/generated/qwen36-preparation-20260923/`. Each contains commands, provenance,
responses, stage timings, reports, and logs. New untracked module sources are additionally
snapshotted under `final-source/`, since the harness's Git patch excludes untracked files.

To reproduce, use `scripts/benchmark-model.py` with the fixed input/gold files in
`tests/accuracy/generated/qwen36-4090-mmlu-run/`, `--runs 1 --batch-requests 8`, then
`--extra-args --max-sequences 2 --micro-batch 256 --max-output-rows 256 --threads 8 --no-mmap`.
Append `--no-preparation-pipeline` for the cached synchronous comparison. Each configuration
must use a new output directory.

### Deferred independent-request experiment

An independent-request batching experiment was **not retained**, including its CLI switch.
On Qwen3.5-0.8B Q8_0, serial/reference scoring matched, but retaining multiple independent prefixes
while advancing just one candidate at a time changed a 101-token candidate's log-probability
from `-43.8409075778909` to `-43.91248824680224`. The `0.07158` difference exceeded the unchanged
`0.01011` test tolerance. A multi-sequence run also changed a three-token candidate by `0.00568`.
This does not isolate a specific native bug or shader: the state/cache layout and calculation
path still need investigation. The failed experiment and diagnostic logs are preserved under
`tests/accuracy/generated/qwen36-preparation-20260923/rejected-independent-batching/` and the
sibling `hybrid-diagnostic.log` / `hybrid-ablation.log`. No numerical guard was relaxed.

Validation for the retained path: release Clippy with warnings denied, 27 unit tests, five
focused model regressions on the pinned Qwen3/Qwen3.5 small fixtures, and an additional fresh-scorer
pipeline run that forces the device target-gather width to grow between questions. The CPU CLI
golden test passed. An earlier all-ignored run was stopped during the long legacy CPU reference
test; that test is not claimed as completed. The 14,042-example full MMLU run was not repeated
for this change; the performance comparison uses the fixed 285-question subset.
