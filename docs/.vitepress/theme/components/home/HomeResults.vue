<script setup lang="ts">
import { computed, ref } from "vue"
import { withBase } from "vitepress"
import {
  paperBestResult,
  paperResultsSeries,
  type ResultsSeries
} from "./paper-results"
import { useRevealOnce } from "./home-motion"

const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.2)

/** Shared model axis for the overall landing dot plot. */
const modelAxis = [
  "Qwen3.5-0.8B",
  "Qwen3.5-2B",
  "Qwen3.5-4B",
  "Qwen3.5-4B + LoRA",
  "Qwen3.5-9B",
  "Qwen3.5-27B",
  "Qwen3.6-35B-A3B"
] as const

const seriesStyle: Record<
  ResultsSeries,
  { label: string; color: string; short: string }
> = {
  vulkan: { label: "RTX 4090 · Vulkan", color: "var(--home-cyan)", short: "Vulkan" },
  cuda: { label: "RTX 4090 · CUDA", color: "var(--home-violet)", short: "CUDA" },
  a770: { label: "Arc A770 · Vulkan", color: "var(--home-amber)", short: "A770" }
}

interface Dot {
  series: ResultsSeries
  model: string
  accuracy: number
  x: number
  y: number
  caseId: string
}

const width = 720
const height = 280
const pad = { top: 28, right: 24, bottom: 72, left: 48 }
const plotW = width - pad.left - pad.right
const plotH = height - pad.top - pad.bottom
const yMin = 25
const yMax = 90

const seriesOffset: Record<ResultsSeries, number> = {
  vulkan: -10,
  cuda: 0,
  a770: 10
}

function xFor(index: number, series: ResultsSeries): number {
  return pad.left + (index / (modelAxis.length - 1)) * plotW + seriesOffset[series]
}

function yFor(accuracy: number): number {
  return pad.top + (1 - (accuracy - yMin) / (yMax - yMin)) * plotH
}

const dots = computed((): Dot[] => {
  const points: Dot[] = []
  for (const series of ["vulkan", "cuda", "a770"] as const) {
    for (const row of paperResultsSeries[series].rows) {
      const index = modelAxis.indexOf(row.model as (typeof modelAxis)[number])
      if (index < 0) continue
      points.push({
        series,
        model: row.model,
        accuracy: row.accuracy,
        x: xFor(index, series),
        y: yFor(row.accuracy),
        caseId: row.caseId
      })
    }
  }
  return points
})

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
        One compact view of mean-token full-MMLU accuracy from the Jet technical report.
        The best measured configuration is {{ paperBestResult.model }} at
        {{ paperBestResult.accuracy.toFixed(2) }}%. Detailed tables live on the comparison
        page.
      </p>
    </header>

    <div class="results__panel reveal-line" style="--line: 3">
      <div class="results__meta">
        <h3>MMLU accuracy by model</h3>
        <p>
          Same 14,042-question protocol across series. Source macros:
          <code>paper/figures/rtx4090-results.tex</code>,
          <code>paper/figures/a770-results.tex</code>
          (tables <code>tab:full-mmlu</code>, <code>tab:full-mmlu-cuda</code>,
          <code>tab:a770-full-mmlu</code>).
        </p>
      </div>

      <svg
        class="results__plot"
        :viewBox="`0 0 ${width} ${height}`"
        role="img"
        aria-label="Dot plot of full MMLU accuracy for Jet models on RTX 4090 Vulkan, RTX 4090 CUDA, and Arc A770 Vulkan"
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

        <g v-for="(model, index) in modelAxis" :key="model">
          <text
            class="results__xlabel"
            :x="xFor(index, 'cuda')"
            :y="height - 18"
            text-anchor="middle"
          >
            <template v-if="model.includes('LoRA')">4B LoRA</template>
            <template v-else-if="model.includes('35B')">35B-A3B</template>
            <template v-else>{{ model.replace('Qwen3.5-', '').replace('Qwen3.6-', '') }}</template>
          </text>
        </g>

        <circle
          v-for="dot in dots"
          :key="dot.caseId"
          class="results__dot"
          :cx="dot.x"
          :cy="dot.y"
          r="7"
          :fill="seriesStyle[dot.series].color"
        >
          <title>
            {{ seriesStyle[dot.series].label }} · {{ dot.model }} ·
            {{ dot.accuracy.toFixed(2) }}%
          </title>
        </circle>
      </svg>

      <ul class="results__legend">
        <li v-for="(style, key) in seriesStyle" :key="key">
          <i :style="{ background: style.color }" />
          {{ style.label }}
        </li>
      </ul>
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
  padding: 28px 24px 22px;
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
  margin-top: 18px;
}

.results__grid {
  stroke: var(--home-edge);
  stroke-width: 1;
}

.results__tick,
.results__xlabel {
  fill: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  font-weight: 650;
}

.results__dot {
  stroke: var(--home-window-solid);
  stroke-width: 2;
}

.results__legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
}

.results__legend li {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--home-ink-2);
  font-size: 13px;
  font-weight: 650;
}

.results__legend i {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--home-ink) 12%, transparent);
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
    padding: 22px 14px 18px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .results__links a span {
    transition: none;
  }
}
</style>
