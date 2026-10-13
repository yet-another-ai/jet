<script setup lang="ts">
import { computed } from "vue"
import AppWindow from "../AppWindow.vue"
import { animatedProbabilities, demoQuestion } from "../demo-scenario"

const props = defineProps<{ t: number }>()

const question = computed(() => demoQuestion(props.t))
const probabilities = computed(() => animatedProbabilities(props.t, question.value))
const winner = computed(() => {
  let best = 0
  for (let i = 1; i < probabilities.value.length; i++) {
    if (probabilities.value[i]! > probabilities.value[best]!) best = i
  }
  return best
})

function formatProbability(value: number): string {
  return `${(value * 100).toFixed(0)}%`
}
</script>

<template>
  <AppWindow title="candidate scores" :basis="40">
    <template #bar><span class="meta">{{ question.name }} · softmax</span></template>
    <div class="panel">
      <div
        v-for="(row, i) in question.candidates"
        :key="`${question.name}-${row.key}`"
        class="row"
        :class="{ 'is-top': i === winner }"
      >
        <div class="row__head">
          <span>{{ row.label }}</span>
          <span class="row__prob">{{ formatProbability(probabilities[i]!) }}</span>
        </div>
        <div class="bar">
          <i :style="{ width: `${probabilities[i]! * 100}%`, background: row.color }" />
        </div>
      </div>
    </div>
  </AppWindow>
</template>

<style scoped>
.meta {
  color: var(--home-ink-2);
  font-size: 0.78em;
  font-weight: 650;
}

.panel {
  display: grid;
  height: 100%;
  align-content: center;
  gap: 1em;
  padding: 1.1em;
}

.row__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.6em;
  margin-bottom: 0.35em;
  font-size: 0.9em;
  font-weight: 650;
}

.row__prob {
  color: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.82em;
  font-variant-numeric: tabular-nums;
}

.bar {
  overflow: hidden;
  height: 0.7em;
  border-radius: 999px;
  background: var(--home-fill);
}

.bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  transition: width 220ms linear;
}

.row.is-top .row__head span:first-child,
.row.is-top .row__prob {
  color: var(--home-cyan-ink);
}

@media (prefers-reduced-motion: reduce) {
  .bar i {
    transition: none;
  }
}
</style>
