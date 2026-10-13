<script setup lang="ts">
import { computed, ref } from "vue"
import { withBase } from "vitepress"
import { paperBestResult, paperResultsSeries } from "./paper-results"
import { useRevealOnce } from "./home-motion"

const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.2)

/** Model axis for the landing summary (best score across all measured series). */
const modelAxis = [
  "Qwen3.5-0.8B",
  "Qwen3.5-2B",
  "Qwen3.5-4B",
  "Qwen3.5-4B + LoRA",
  "Qwen3.5-9B",
  "Qwen3.5-27B",
  "Qwen3.6-35B-A3B"
] as const

interface Dot {
  model: string
  short: string
  accuracy: number
  x: number
  y: number
}

const width = 720
const height = 260
const pad = { top: 36, right: 24, bottom: 56, left: 48 }
const plotW = width - pad.left - pad.right
const plotH = height - pad.top - pad.bottom
const yMin = 25
const yMax = 90

function xFor(index: number): number {
  return pad.left + (index / (modelAxis.length - 1)) * plotW
}

function yFor(accuracy: number): number {
  return pad.top + (1 - (accuracy - yMin) / (yMax - yMin)) * plotH
}

function shortLabel(model: string): string {
  if (model.includes("LoRA")) return "4B LoRA"
  if (model.includes("35B")) return "35B-A3B"
  return model.replace("Qwen3.5-", "").replace("Qwen3.6-", "")
}

/** Best full-MMLU accuracy for each model across Vulkan / CUDA / A770 paper rows. */
const dots = computed((): Dot[] =>
  modelAxis.flatMap((model, index) => {
    let best: number | undefined
    for (const series of Object.values(paperResultsSeries)) {
      for (const row of series.rows) {
        if (row.model !== model) continue
        best = best === undefined ? row.accuracy : Math.max(best, row.accuracy)
      }
    }
    if (best === undefined) return []
    return [
      {
        model,
        short: shortLabel(model),
        accuracy: best,
        x: xFor(index),
        y: yFor(best)
      }
    ]
  })
)

const yTicks = [30, 45, 60, 75, 90]
const gridLines = yTicks.map((tick) => ({ tick, y: yFor(tick) }))
</script>

<template>
  <section id="results" ref="root" class="results reveal" :class="{ 'is-shown': shown }">
    <header class="results__intro">
      <p class="home-kicker reveal-line" style="--line: 0">Model comparison</p>
      <h2 class="home-heading reveal-line" style="--line: 1">
        Full MMLU on consumer GPUs.
      </h2>
      <p class="results__lead reveal-line" style="--line: 2">
        Best mean-token full-MMLU accuracy per model from the Jet technical report
        (14,042 questions). Peak result: {{ paperBestResult.model }} at
        {{ paperBestResult.accuracy.toFixed(2) }}%. Backend and hardware breakdowns are
        on the comparison page.
      </p>
    </header>

    <div class="results__panel reveal-line" style="--line: 3">
      <div class="results__meta">
        <h3>Best MMLU accuracy by model</h3>
        <p>
          One point per model—the highest score across the published RTX&nbsp;4090 and
          Arc&nbsp;A770 runs. Source:
          <code>paper/figures/rtx4090-results.tex</code>,
          <code>paper/figures/a770-results.tex</code>.
        </p>
      </div>

      <svg
        class="results__plot"
        :viewBox="`0 0 ${width} ${height}`"
        role="img"
        aria-label="Dot plot of each Jet model's best full-MMLU accuracy from the technical report"
      >
        <line
          v-for="line in gridLines"
          :key="line.tick"
          class="results__grid"
          :x1="pad.left"
          :x2="width - pad.right"
          :y1="line.y"
          :y2="line.y"
        />
        <text
          v-for="line in gridLines"
          :key="`y-${line.tick}`"
          class="results__tick"
          :x="pad.left - 10"
          :y="line.y + 4"
          text-anchor="end"
        >
          {{ line.tick }}%
        </text>

        <g v-for="dot in dots" :key="dot.model">
          <text class="results__xlabel" :x="dot.x" :y="height - 16" text-anchor="middle">
            {{ dot.short }}
          </text>
          <circle class="results__dot" :cx="dot.x" :cy="dot.y" r="8">
            <title>{{ dot.model }} · {{ dot.accuracy.toFixed(2) }}%</title>
          </circle>
          <text class="results__value" :x="dot.x" :y="dot.y - 14" text-anchor="middle">
            {{ dot.accuracy.toFixed(1) }}%
          </text>
        </g>
      </svg>
    </div>

    <p class="results__links reveal-line" style="--line: 4">
      <a :href="withBase('/results')">Full comparison tables <span aria-hidden="true">→</span></a>
      <a href="https://github.com/yet-another-ai/jet/tree/main/paper"
        >Technical report sources <span aria-hidden="true">→</span></a
      >
    </p>
  </section>
</template>

<style scoped>
.results {
  width: min(960px, calc(100% - 48px));
  margin: 0 auto;
  padding: clamp(96px, 12vw, 150px) 0 40px;
  scroll-margin-top: calc(var(--vp-nav-height) - 40px);
}

.results__intro {
  max-width: 760px;
  margin: 0 auto 36px;
  text-align: center;
}

.results__lead {
  margin: 14px 0 0;
  color: var(--home-ink-2);
  font-size: 18px;
  line-height: 1.55;
}

.results__panel {
  padding: 28px 24px 18px;
  border: 1px solid var(--home-edge);
  border-radius: 24px;
  background: var(--home-panel);
  backdrop-filter: blur(18px);
}

.results__meta h3 {
  margin: 0 0 8px;
  font-size: 20px;
  font-weight: 750;
  letter-spacing: -0.02em;
}

.results__meta p {
  margin: 0;
  color: var(--home-ink-2);
  font-size: 14.5px;
  line-height: 1.55;
}

.results__meta code {
  font-size: 0.9em;
}

.results__plot {
  display: block;
  width: 100%;
  height: auto;
  margin-top: 12px;
}

.results__grid {
  stroke: var(--home-edge);
  stroke-width: 1;
}

.results__tick,
.results__xlabel,
.results__value {
  fill: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-weight: 650;
}

.results__tick,
.results__xlabel {
  font-size: 11px;
}

.results__value {
  fill: var(--home-ink);
  font-size: 12px;
}

.results__dot {
  fill: var(--home-cyan);
  stroke: var(--home-window-solid);
  stroke-width: 2;
}

.results__links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px 24px;
  margin: 28px 0 0;
}

.results__links a {
  color: var(--home-ink);
  font-size: 15px;
  font-weight: 650;
  text-decoration: none;
}

.results__links a span {
  display: inline-block;
  color: var(--home-cyan-ink);
  transition: transform 220ms ease;
}

.results__links a:hover span {
  transform: translateX(3px);
}

@media (max-width: 720px) {
  .results {
    width: calc(100% - 32px);
  }

  .results__panel {
    padding: 22px 14px 14px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .results__links a span {
    transition: none;
  }
}
</style>
