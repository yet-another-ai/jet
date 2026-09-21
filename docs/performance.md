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
A warmed, uninstrumented shape benchmark is needed before changing thresholds.

The A770's `matrix cores: none` initialization message is intentional in this
native revision: cooperative-matrix shader support is compiled in, but the device
support check disables it for Xe1/A770 because of reported performance
regressions. Enabling it needs its own correctness and performance experiment.

## Next optimization targets

1. Validate and tune short-chunk Q4_K expert dispatch, followed by the large Q8_0
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
