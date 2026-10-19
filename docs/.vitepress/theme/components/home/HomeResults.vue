<script setup lang="ts">
import { computed, ref } from "vue"
import { withBase } from "vitepress"
import { paperBestResult, paperResultsSeries, type PaperResultRow } from "./paper-results"
import { useRevealOnce } from "./home-motion"

const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.2)
const active = ref<string | null>(null)

/**
 * Landing scatter points: for each model, the RTX 4090 full-MMLU row with the
 * highest accuracy (Vulkan or CUDA series in rtx4090-results.tex). Accuracy and
 * req./s come from that same row so the point is one real measured run.
 * Arc A770 is excluded here because this axis is RTX 4090 throughput.
 */
const rtx4090Rows: PaperResultRow[] = [
  ...paperResultsSeries.vulkan.rows,
  ...paperResultsSeries.cuda.rows
]

const modelOrder = [
  "Qwen3.5-0.8B",
  "Qwen3.5-2B",
  "Qwen3.5-4B",
  "Qwen3.5-4B + LoRA",
  "Qwen3.5-9B",
  "Qwen3.5-27B",
  "Qwen3.6-35B-A3B"
] as const

interface ScatterPoint {
  id: string
  model: string
  short: string
  accuracy: number
  rate: number
  x: number
  y: number
}

const width = 720
const height = 340
const pad = { top: 28, right: 28, bottom: 52, left: 64 }
const plotW = width - pad.left - pad.right
const plotH = height - pad.top - pad.bottom
const xMin = 0
const xMax = 32
const yMin = 25
const yMax = 90

function shortLabel(model: string): string {
  if (model.includes("LoRA")) return "4B LoRA"
  if (model.includes("35B")) return "35B-A3B"
  return model.replace("Qwen3.5-", "").replace("Qwen3.6-", "")
}

function xFor(rate: number): number {
  return pad.left + ((rate - xMin) / (xMax - xMin)) * plotW
}

function yFor(accuracy: number): number {
  return pad.top + (1 - (accuracy - yMin) / (yMax - yMin)) * plotH
}

const points = computed((): ScatterPoint[] => {
  const bestByModel = new Map<string, PaperResultRow>()
  for (const row of rtx4090Rows) {
    const prev = bestByModel.get(row.model)
    if (!prev || row.accuracy > prev.accuracy) bestByModel.set(row.model, row)
  }

  return modelOrder.flatMap((model) => {
    const row = bestByModel.get(model)
    if (!row) return []
    return [
      {
        id: row.caseId,
        model,
        short: shortLabel(model),
        accuracy: row.accuracy,
        rate: row.rate,
        x: xFor(row.rate),
        y: yFor(row.accuracy)
      }
    ]
  })
})

const xTicks = [0, 8, 16, 24, 32]
const yTicks = [30, 45, 60, 75, 90]

function activate(id: string) {
  active.value = id
}

function deactivate(id: string) {
  if (active.value === id) active.value = null
}
</script>

<template>
  <section id="results" ref="root" class="results reveal" :class="{ 'is-shown': shown }">
    <header class="results__intro">
      <p class="home-kicker reveal-line" style="--line: 0">Model comparison</p>
      <h2 class="home-heading reveal-line" style="--line: 1">
        Full MMLU on consumer GPUs.
      </h2>
      <p class="results__lead reveal-line" style="--line: 2">
        Accuracy versus complete-process throughput on RTX&nbsp;4090, from the Jet
        technical report. Hover or focus a point for exact values. Peak accuracy:
        {{ paperBestResult.model }} at {{ paperBestResult.accuracy.toFixed(2) }}%.
      </p>
    </header>

    <div class="results__panel reveal-line" style="--line: 3">
      <div class="results__meta">
        <h3>Accuracy vs throughput</h3>
        <p>
          Each point is one RTX&nbsp;4090 full-MMLU run (best accuracy per model across
          Vulkan and CUDA). Source:
          <code>paper/figures/rtx4090-results.tex</code>.
        </p>
      </div>

      <div class="results__plot">
        <svg
          :viewBox="`0 0 ${width} ${height}`"
          role="img"
          aria-label="Scatter plot of RTX 4090 full-MMLU accuracy versus requests per second"
        >
          <line
            v-for="tick in yTicks"
            :key="`yg-${tick}`"
            class="results__grid"
            :x1="pad.left"
            :x2="width - pad.right"
            :y1="yFor(tick)"
            :y2="yFor(tick)"
          />
          <line
            v-for="tick in xTicks"
            :key="`xg-${tick}`"
            class="results__grid results__grid--v"
            :x1="xFor(tick)"
            :x2="xFor(tick)"
            :y1="pad.top"
            :y2="height - pad.bottom"
          />

          <text
            v-for="tick in yTicks"
            :key="`yt-${tick}`"
            class="results__tick"
            :x="pad.left - 10"
            :y="yFor(tick) + 4"
            text-anchor="end"
          >
            {{ tick }}
          </text>
          <text
            v-for="tick in xTicks"
            :key="`xt-${tick}`"
            class="results__tick"
            :x="xFor(tick)"
            :y="height - pad.bottom + 22"
            text-anchor="middle"
          >
            {{ tick }}
          </text>

          <text
            class="results__axis"
            :x="18"
            :y="pad.top + plotH / 2"
            text-anchor="middle"
            :transform="`rotate(-90 18 ${pad.top + plotH / 2})`"
          >
            MMLU %
          </text>
          <text
            class="results__axis"
            :x="pad.left + plotW / 2"
            :y="height - 8"
            text-anchor="middle"
          >
            Req./s on RTX 4090
          </text>
        </svg>

        <button
          v-for="point in points"
          :key="point.id"
          type="button"
          class="results__point"
          :class="{ 'is-active': active === point.id }"
          :style="{ left: `${(point.x / width) * 100}%`, top: `${(point.y / height) * 100}%` }"
          :aria-label="`${point.model}: ${point.accuracy.toFixed(2)} percent MMLU, ${point.rate.toFixed(2)} requests per second`"
          @mouseenter="activate(point.id)"
          @mouseleave="deactivate(point.id)"
          @focus="activate(point.id)"
          @blur="deactivate(point.id)"
        >
          <i aria-hidden="true" />
          <span class="results__tip" aria-hidden="true">
            <strong>{{ point.short }}</strong>
            <em>{{ point.accuracy.toFixed(2) }}% · {{ point.rate.toFixed(2) }} req/s</em>
          </span>
        </button>
      </div>
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
  position: relative;
  width: 100%;
  margin-top: 8px;
  aspect-ratio: 720 / 340;
}

.results__plot svg {
  display: block;
  width: 100%;
  height: auto;
}

.results__grid {
  stroke: var(--home-edge);
  stroke-width: 1;
}

.results__grid--v {
  opacity: 0.55;
}

.results__tick,
.results__axis {
  fill: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  font-weight: 650;
}

.results__axis {
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.results__point {
  position: absolute;
  z-index: 1;
  width: 28px;
  height: 28px;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: transparent;
  cursor: pointer;
  transform: translate(-50%, -50%);
}

.results__point i {
  display: block;
  width: 12px;
  height: 12px;
  margin: 8px;
  border-radius: 50%;
  background: var(--home-cyan);
  box-shadow:
    0 0 0 2px var(--home-window-solid),
    0 6px 16px -8px var(--home-shadow);
  transition: transform 220ms var(--home-ease);
}

.results__point.is-active {
  z-index: 2;
}

.results__point.is-active i {
  transform: scale(1.18);
}

.results__point:focus-visible {
  outline: 2px solid var(--home-cyan);
  outline-offset: 2px;
}

.results__tip {
  position: absolute;
  left: 50%;
  bottom: calc(100% + 10px);
  display: grid;
  gap: 2px;
  min-width: max-content;
  padding: 8px 10px;
  border: 1px solid var(--home-edge);
  border-radius: 12px;
  background: var(--home-window-solid);
  box-shadow: 0 14px 30px -18px var(--home-shadow);
  opacity: 0;
  filter: blur(4px);
  pointer-events: none;
  transform: translate(-50%, 8px);
  transition:
    opacity 0.45s var(--home-ease),
    filter 0.45s var(--home-ease),
    transform 0.55s var(--home-ease);
}

.results__tip strong,
.results__tip em {
  display: block;
  font-style: normal;
  white-space: nowrap;
}

.results__tip strong {
  color: var(--home-ink);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.results__tip em {
  color: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  font-weight: 650;
}

.results__point.is-active .results__tip {
  opacity: 1;
  filter: none;
  transform: translate(-50%, 0);
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
  .results__point i,
  .results__tip,
  .results__links a span {
    transition: none;
  }
}
</style>
