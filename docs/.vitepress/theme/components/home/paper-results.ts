/**
 * Full-MMLU numbers shown on the docs site.
 *
 * Source of truth: the Jet technical report manuscript macros in
 * `paper/figures/rtx4090-results.tex` and `paper/figures/a770-results.tex`,
 * as rendered in `paper/sections/evaluation.tex`:
 * - Table `tab:full-mmlu` (RTX 4090, Vulkan)
 * - Table `tab:full-mmlu-cuda` (RTX 4090, CUDA)
 * - Table `tab:a770-full-mmlu` (Arc A770, Vulkan)
 *
 * Display rounding matches the manuscript macros (two decimal places for
 * accuracy and req./s). Do not invent or recompute values here.
 */

export type ResultsSeries = "vulkan" | "cuda" | "a770"

export interface PaperResultRow {
  model: string
  quantization: string
  correct: string
  accuracy: number
  wall: string
  rate: number
  /** Manuscript case id, e.g. full-27b-vulkan. */
  caseId: string
  note?: string
}

export interface ResultsSeriesCopy {
  id: ResultsSeries
  tab: string
  title: string
  citation: string
  protocol: string
  rows: PaperResultRow[]
}

export const paperResultsSeries: Record<ResultsSeries, ResultsSeriesCopy> = {
  vulkan: {
    id: "vulkan",
    tab: "RTX 4090 · Vulkan",
    title: "Full MMLU on RTX 4090 with Vulkan",
    citation: "paper Table tab:full-mmlu · macros in paper/figures/rtx4090-results.tex",
    protocol:
      "14,042 MMLU test questions, mean-token candidate scoring, reasoning disabled. Wall time includes model loading and shutdown. One completed fresh-process run per model.",
    rows: [
      {
        model: "Qwen3.5-0.8B",
        quantization: "Q8_0",
        correct: "4,659",
        accuracy: 33.18,
        wall: "468.41",
        rate: 29.98,
        caseId: "full-08b-vulkan"
      },
      {
        model: "Qwen3.5-2B",
        quantization: "Q8_0",
        correct: "6,807",
        accuracy: 48.48,
        wall: "613.82",
        rate: 22.88,
        caseId: "full-2b-vulkan"
      },
      {
        model: "Qwen3.5-4B",
        quantization: "Q8_0",
        correct: "9,295",
        accuracy: 66.19,
        wall: "1,117.25",
        rate: 12.57,
        caseId: "full-4b-vulkan"
      },
      {
        model: "Qwen3.5-9B",
        quantization: "Q8_0",
        correct: "10,344",
        accuracy: 73.66,
        wall: "1,605.52",
        rate: 8.75,
        caseId: "full-9b-vulkan"
      },
      {
        model: "Qwen3.5-27B",
        quantization: "Q4_K_M",
        correct: "11,937",
        accuracy: 85.01,
        wall: "4,805.68",
        rate: 2.92,
        caseId: "full-27b-vulkan"
      },
      {
        model: "Qwen3.6-35B-A3B",
        quantization: "Q4_K_M",
        correct: "11,598",
        accuracy: 82.6,
        wall: "3,933.73",
        rate: 3.57,
        caseId: "full-35b-vulkan"
      }
    ]
  },
  cuda: {
    id: "cuda",
    tab: "RTX 4090 · CUDA",
    title: "Full MMLU on RTX 4090 with CUDA",
    citation: "paper Table tab:full-mmlu-cuda · macros in paper/figures/rtx4090-results.tex",
    protocol:
      "Same 14,042-question protocol as the Vulkan series. One fresh-process run per model.",
    rows: [
      {
        model: "Qwen3.5-0.8B",
        quantization: "Q8_0",
        correct: "4,700",
        accuracy: 33.47,
        wall: "972.68",
        rate: 14.44,
        caseId: "full-08b-cuda"
      },
      {
        model: "Qwen3.5-2B",
        quantization: "Q8_0",
        correct: "6,802",
        accuracy: 48.44,
        wall: "984.51",
        rate: 14.26,
        caseId: "full-2b-cuda"
      },
      {
        model: "Qwen3.5-4B",
        quantization: "Q8_0",
        correct: "9,305",
        accuracy: 66.27,
        wall: "1,319.27",
        rate: 10.64,
        caseId: "full-4b-cuda"
      },
      {
        model: "Qwen3.5-4B + LoRA",
        quantization: "Q8_0",
        correct: "10,452",
        accuracy: 74.43,
        wall: "1,351.77",
        rate: 10.39,
        caseId: "full-4b-lora-cuda",
        note: "Task-aligned LoRA export evaluated under the same full-MMLU protocol."
      },
      {
        model: "Qwen3.5-9B",
        quantization: "Q8_0",
        correct: "10,337",
        accuracy: 73.61,
        wall: "1,573.78",
        rate: 8.92,
        caseId: "full-9b-cuda"
      },
      {
        model: "Qwen3.5-27B",
        quantization: "Q4_K_M",
        correct: "11,912",
        accuracy: 84.83,
        wall: "3,721.45",
        rate: 3.77,
        caseId: "full-27b-cuda"
      },
      {
        model: "Qwen3.6-35B-A3B",
        quantization: "Q4_K_M",
        correct: "11,574",
        accuracy: 82.42,
        wall: "2,774.94",
        rate: 5.06,
        caseId: "full-35b-cuda"
      }
    ]
  },
  a770: {
    id: "a770",
    tab: "Arc A770 · Vulkan",
    title: "Full MMLU on Arc A770 with Vulkan",
    citation: "paper Table tab:a770-full-mmlu · macros in paper/figures/a770-results.tex",
    protocol:
      "14,042 requests per model, Q8_0 weights, full GPU placement, Vulkan. One completed fresh-process run per model.",
    rows: [
      {
        model: "Qwen3.5-0.8B",
        quantization: "Q8_0",
        correct: "4,695",
        accuracy: 33.44,
        wall: "3,178.82",
        rate: 4.42,
        caseId: "a770-08b"
      },
      {
        model: "Qwen3.5-2B",
        quantization: "Q8_0",
        correct: "6,800",
        accuracy: 48.43,
        wall: "4,442.30",
        rate: 3.16,
        caseId: "a770-2b"
      },
      {
        model: "Qwen3.5-4B",
        quantization: "Q8_0",
        correct: "9,305",
        accuracy: 66.27,
        wall: "9,797.73",
        rate: 1.43,
        caseId: "a770-4b"
      },
      {
        model: "Qwen3.5-9B",
        quantization: "Q8_0",
        correct: "10,345",
        accuracy: 73.67,
        wall: "13,393.47",
        rate: 1.05,
        caseId: "a770-9b"
      }
    ]
  }
}

/** Highest measured JET configuration from manuscript macro jetresult@best@. */
export const paperBestResult = {
  model: "Qwen3.5-27B (Vulkan)",
  accuracy: 85.01,
  citation: "paper/figures/rtx4090-results.tex · jetresult@best@"
} as const
