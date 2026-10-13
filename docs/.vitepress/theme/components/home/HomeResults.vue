<script setup lang="ts">
import { computed, ref } from "vue"
import { withBase } from "vitepress"
import {
  paperBestResult,
  paperResultsSeries,
  type ResultsSeries
} from "./paper-results"
import { useRevealOnce } from "./home-motion"
import SegmentedControl from "./SegmentedControl.vue"

const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.15)
const series = ref<ResultsSeries>("vulkan")

const options = (["vulkan", "cuda", "a770"] as const).map((value) => ({
  value,
  label: paperResultsSeries[value].tab
}))
const active = computed(() => paperResultsSeries[series.value])
const maxAccuracy = computed(() =>
  Math.max(...active.value.rows.map((row) => row.accuracy))
)
</script>

<template>
  <section id="results" ref="root" class="results reveal" :class="{ 'is-shown': shown }">
    <header class="results__intro">
      <p class="home-kicker reveal-line" style="--line: 0">Model comparison</p>
      <h2 class="home-heading reveal-line" style="--line: 1">
        Full MMLU on consumer GPUs.
      </h2>
      <p class="results__lead reveal-line" style="--line: 2">
        Numbers are copied from the Jet technical report’s full-MMLU tables—mean-token
        scoring over 14,042 questions—not invented for this page. The best measured
        configuration is {{ paperBestResult.model }} at
        {{ paperBestResult.accuracy.toFixed(2) }}%.
      </p>
    </header>

    <div class="results__controls reveal-line" style="--line: 3">
      <SegmentedControl v-model="series" :options="options" label="Result series" />
    </div>

    <Transition name="results-swap" mode="out-in">
      <div :key="series" class="results__panel">
        <div class="results__meta">
          <h3>{{ active.title }}</h3>
          <p>{{ active.protocol }}</p>
          <p class="results__cite">Source: {{ active.citation }}</p>
        </div>

        <div class="results__chart" aria-hidden="true">
          <div
            v-for="row in active.rows"
            :key="row.caseId"
            class="results__bar"
            :style="{ '--w': `${(row.accuracy / maxAccuracy) * 100}%` }"
          >
            <span class="results__bar-label">{{ row.model }}</span>
            <i />
            <span class="results__bar-value">{{ row.accuracy.toFixed(2) }}%</span>
          </div>
        </div>

        <div class="results__table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Quant.</th>
                <th scope="col">Correct</th>
                <th scope="col">MMLU (%)</th>
                <th scope="col">Wall (s)</th>
                <th scope="col">Req./s</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in active.rows" :key="row.caseId">
                <th scope="row">
                  {{ row.model }}
                  <small v-if="row.note">{{ row.note }}</small>
                </th>
                <td>{{ row.quantization }}</td>
                <td>{{ row.correct }}</td>
                <td>{{ row.accuracy.toFixed(2) }}</td>
                <td>{{ row.wall }}</td>
                <td>{{ row.rate.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </Transition>

    <p class="results__links reveal-line" style="--line: 4">
      <a :href="withBase('/results')">Full comparison page <span aria-hidden="true">→</span></a>
      <a href="https://github.com/yet-another-ai/jet/tree/main/paper"
        >Technical report sources <span aria-hidden="true">→</span></a
      >
    </p>
  </section>
</template>

<style scoped>
.results {
  width: min(1120px, calc(100% - 48px));
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

.results__controls {
  display: flex;
  justify-content: center;
  margin-bottom: 18px;
}

.results__panel {
  padding: 28px 24px 24px;
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

.results__cite {
  margin-top: 8px !important;
  font-family: var(--vp-font-family-mono);
  font-size: 12px !important;
}

.results__chart {
  display: grid;
  gap: 10px;
  margin: 24px 0 8px;
}

.results__bar {
  display: grid;
  grid-template-columns: minmax(120px, 180px) 1fr auto;
  align-items: center;
  gap: 12px;
}

.results__bar-label {
  overflow: hidden;
  font-size: 13px;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.results__bar i {
  display: block;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--home-cyan), var(--home-violet));
  transform-origin: left center;
  width: var(--w);
}

.results__bar-value {
  min-width: 4.5em;
  color: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  font-weight: 650;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.results__table-wrap {
  overflow-x: auto;
  margin-top: 18px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
}

th,
td {
  padding: 10px 12px;
  border-top: 1px solid var(--home-edge);
  text-align: left;
  white-space: nowrap;
}

thead th {
  color: var(--home-ink-2);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border-top: none;
}

tbody th {
  font-weight: 650;
}

tbody th small {
  display: block;
  max-width: 280px;
  color: var(--home-ink-2);
  font-size: 11.5px;
  font-weight: 500;
  line-height: 1.4;
  white-space: normal;
}

td {
  font-family: var(--vp-font-family-mono);
  font-variant-numeric: tabular-nums;
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

.results-swap-enter-active,
.results-swap-leave-active {
  transition:
    opacity 0.35s ease,
    transform 0.45s var(--home-ease);
}

.results-swap-enter-from {
  opacity: 0;
  transform: translateY(8px);
}

.results-swap-leave-to {
  opacity: 0;
}

@media (max-width: 720px) {
  .results {
    width: calc(100% - 32px);
  }

  .results__bar {
    grid-template-columns: 1fr auto;
    grid-template-areas:
      "label value"
      "bar bar";
  }

  .results__bar-label {
    grid-area: label;
  }

  .results__bar-value {
    grid-area: value;
  }

  .results__bar i {
    grid-area: bar;
  }
}

@media (prefers-reduced-motion: reduce) {
  .results-swap-enter-active,
  .results-swap-leave-active,
  .results__links a span {
    transition: none;
  }
}
</style>
