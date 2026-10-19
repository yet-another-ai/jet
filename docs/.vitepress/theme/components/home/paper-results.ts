/**
 * RTX 4090 full-MMLU numbers used by the landing-page scatter plot.
 *
 * Source of truth: the Jet technical report manuscript macros in
 * `paper/figures/rtx4090-results.tex`,
 * as rendered in `paper/sections/evaluation.tex`:
 * - Table `tab:full-mmlu` (RTX 4090, Vulkan)
 * - Table `tab:full-mmlu-cuda` (RTX 4090, CUDA)
 *
 * Display rounding matches the manuscript macros (two decimal places for
 * accuracy and req./s). Do not invent or recompute values here.
 */

export type ResultsSeries = "vulkan" | "cuda"

export interface PaperResultRow {
  model: string
  accuracy: number
  rate: number
  /** Manuscript case id, e.g. full-27b-vulkan. */
  caseId: string
}

export interface ResultsSeriesCopy {
  rows: PaperResultRow[]
}

export const paperResultsSeries: Record<ResultsSeries, ResultsSeriesCopy> = {
  vulkan: {
    rows: [
      {
        model: "Qwen3.5-0.8B",
        accuracy: 33.18,
        rate: 29.98,
        caseId: "full-08b-vulkan"
      },
      {
        model: "Qwen3.5-2B",
        accuracy: 48.48,
        rate: 22.88,
        caseId: "full-2b-vulkan"
      },
      {
        model: "Qwen3.5-4B",
        accuracy: 66.19,
        rate: 12.57,
        caseId: "full-4b-vulkan"
      },
      {
        model: "Qwen3.5-9B",
        accuracy: 73.66,
        rate: 8.75,
        caseId: "full-9b-vulkan"
      },
      {
        model: "Qwen3.5-27B",
        accuracy: 85.01,
        rate: 2.92,
        caseId: "full-27b-vulkan"
      },
      {
        model: "Qwen3.6-35B-A3B",
        accuracy: 82.6,
        rate: 3.57,
        caseId: "full-35b-vulkan"
      }
    ]
  },
  cuda: {
    rows: [
      {
        model: "Qwen3.5-0.8B",
        accuracy: 33.47,
        rate: 14.44,
        caseId: "full-08b-cuda"
      },
      {
        model: "Qwen3.5-2B",
        accuracy: 48.44,
        rate: 14.26,
        caseId: "full-2b-cuda"
      },
      {
        model: "Qwen3.5-4B",
        accuracy: 66.27,
        rate: 10.64,
        caseId: "full-4b-cuda"
      },
      {
        model: "Qwen3.5-4B + LoRA",
        accuracy: 74.43,
        rate: 10.39,
        caseId: "full-4b-lora-cuda"
      },
      {
        model: "Qwen3.5-9B",
        accuracy: 73.61,
        rate: 8.92,
        caseId: "full-9b-cuda"
      },
      {
        model: "Qwen3.5-27B",
        accuracy: 84.83,
        rate: 3.77,
        caseId: "full-27b-cuda"
      },
      {
        model: "Qwen3.6-35B-A3B",
        accuracy: 82.42,
        rate: 5.06,
        caseId: "full-35b-cuda"
      }
    ]
  }
}

/** Highest measured JET configuration from manuscript macro jetresult@best@. */
export const paperBestResult = {
  model: "Qwen3.5-27B (Vulkan)",
  accuracy: 85.01
} as const
