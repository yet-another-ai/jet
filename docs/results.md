---
title: Model comparison
description: Full-MMLU accuracy and throughput for Jet configurations, copied from the technical report tables.
---

# Model comparison

These figures are **copied from the Jet technical report**, not recomputed for the docs site. Every row comes from the manuscript macros in [`paper/figures/rtx4090-results.tex`](https://github.com/yet-another-ai/jet/blob/main/paper/figures/rtx4090-results.tex) and [`paper/figures/a770-results.tex`](https://github.com/yet-another-ai/jet/blob/main/paper/figures/a770-results.tex), as published in [`paper/sections/evaluation.tex`](https://github.com/yet-another-ai/jet/blob/main/paper/sections/evaluation.tex).

Protocol shared by the tables below (see the report’s Experimental Evaluation section):

- 14,042 MMLU test questions across 57 subjects
- Mean-token candidate scoring with reasoning disabled
- Complete-process wall time includes initialization, model loading, evaluation, and shutdown
- One completed fresh-process run per configuration; no repeat-run variance estimate

The highest measured JET configuration in the manuscript is **Qwen3.5-27B (Vulkan)** at **85.01%** (`jetresult@best@` in `rtx4090-results.tex`).

## RTX 4090 · Vulkan

Source: paper Table `tab:full-mmlu`.

| Model | Quantization | Correct | MMLU (%) | Wall (s) | Req./s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | Q8_0 | 4,659 | 33.18 | 468.41 | 29.98 |
| Qwen3.5-2B | Q8_0 | 6,807 | 48.48 | 613.82 | 22.88 |
| Qwen3.5-4B | Q8_0 | 9,295 | 66.19 | 1,117.25 | 12.57 |
| Qwen3.5-9B | Q8_0 | 10,344 | 73.66 | 1,605.52 | 8.75 |
| Qwen3.5-27B | Q4_K_M | 11,937 | 85.01 | 4,805.68 | 2.92 |
| Qwen3.6-35B-A3B | Q4_K_M | 11,598 | 82.60 | 3,933.73 | 3.57 |

## RTX 4090 · CUDA

Source: paper Table `tab:full-mmlu-cuda`.

| Model | Quantization | Correct | MMLU (%) | Wall (s) | Req./s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | Q8_0 | 4,700 | 33.47 | 972.68 | 14.44 |
| Qwen3.5-2B | Q8_0 | 6,802 | 48.44 | 984.51 | 14.26 |
| Qwen3.5-4B | Q8_0 | 9,305 | 66.27 | 1,319.27 | 10.64 |
| Qwen3.5-4B + LoRA | Q8_0 | 10,452 | 74.43 | 1,351.77 | 10.39 |
| Qwen3.5-9B | Q8_0 | 10,337 | 73.61 | 1,573.78 | 8.92 |
| Qwen3.5-27B | Q4_K_M | 11,912 | 84.83 | 3,721.45 | 3.77 |
| Qwen3.6-35B-A3B | Q4_K_M | 11,574 | 82.42 | 2,774.94 | 5.06 |

The LoRA row is the task-aligned Qwen3.5-4B export evaluated under the same full-MMLU protocol (`full-4b-lora-cuda`).

## Arc A770 · Vulkan

Source: paper Table `tab:a770-full-mmlu`.

| Model | Quantization | Correct | MMLU (%) | Wall (s) | Req./s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | Q8_0 | 4,695 | 33.44 | 3,178.82 | 4.42 |
| Qwen3.5-2B | Q8_0 | 6,800 | 48.43 | 4,442.30 | 3.16 |
| Qwen3.5-4B | Q8_0 | 9,305 | 66.27 | 9,797.73 | 1.43 |
| Qwen3.5-9B | Q8_0 | 10,345 | 73.67 | 13,393.47 | 1.05 |

## How to read these numbers

The report’s zero-shot semantic scoring protocol differs from standard MMLU leaderboard procedures. Throughput is complete-process requests per second, not saturated serving throughput. Different host systems (RTX 4090 vs Arc A770) do not support a controlled hardware comparison. See [`paper/evidence.md`](https://github.com/yet-another-ai/jet/blob/main/paper/evidence.md) for artifact paths and provenance boundaries.
