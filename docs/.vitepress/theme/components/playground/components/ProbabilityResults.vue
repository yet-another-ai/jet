<script setup lang="ts">
import type { ResultGroup } from '../types';

defineProps<{ groups: ResultGroup[] }>();

function percent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}
</script>

<template>
  <div class="results">
    <article
      v-for="(group, groupIndex) in groups"
      :key="group.id"
      class="result card"
      :style="{ '--i': groupIndex }"
    >
      <header class="result__head">
        <span class="result__key key">question_{{ groupIndex + 1 }}</span>
        <h3>{{ group.question }}</h3>
        <p class="result__answer">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" /></svg>
          {{ group.options.find((option) => option.winner)?.label }}
        </p>
      </header>
      <div class="result__rows">
        <div
          v-for="option in group.options"
          :key="option.id"
          class="row"
          :class="{ 'is-winner': option.winner }"
        >
          <div class="row__label">
            <span>{{ option.label }}</span>
            <strong>{{ percent(option.probability) }}</strong>
          </div>
          <div
            class="row__track"
            role="meter"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-valuenow="Math.round(option.probability * 100)"
            :aria-label="`${option.label}: ${percent(option.probability)}`"
          >
            <i :style="{ '--p': Math.max(option.probability, 0.006) }" />
          </div>
        </div>
      </div>
    </article>
  </div>
</template>

<style scoped>
.results {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.result {
  padding: 16px 18px 18px;
  animation: rise 0.6s var(--ease) both;
  animation-delay: calc(var(--i) * 90ms);
}

.result__head {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 4px 12px;
  margin-bottom: 16px;
}

.result__key {
  grid-column: 1 / -1;
}

h3 {
  margin: 0;
  font-size: 15.5px;
  font-weight: 700;
  letter-spacing: -0.01em;
  line-height: 1.4;
}

.result__answer {
  display: inline-flex;
  max-width: 220px;
  align-items: center;
  gap: 5px;
  margin: 0;
  padding: 4px 10px 4px 8px;
  border-radius: 999px;
  color: var(--teal-ink);
  background: color-mix(in srgb, var(--teal) 14%, transparent);
  font-size: 12.5px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.result__answer svg {
  width: 13px;
  height: 13px;
  stroke-width: 2;
}

.result__rows {
  display: grid;
  gap: 12px;
}

.row__label {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 6px;
  color: var(--ink-2);
  font-size: 13.5px;
  font-weight: 600;
}

.row__label strong {
  font-family: var(--font-mono);
  font-size: 12.5px;
  font-variant-numeric: tabular-nums;
  font-weight: 650;
}

.row.is-winner .row__label {
  color: var(--ink);
}

.row.is-winner .row__label strong {
  color: var(--teal-ink);
}

.row__track {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--fill);
}

.row__track i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: color-mix(in srgb, var(--sky) 70%, var(--ink-2));
  transform: scaleX(var(--p));
  transform-origin: left;
  animation: grow 0.9s var(--ease) both;
  animation-delay: calc(var(--i) * 90ms + 120ms);
}

.row.is-winner .row__track i {
  background: linear-gradient(90deg, var(--teal), color-mix(in srgb, var(--teal) 60%, var(--sky)));
}

@keyframes rise {
  from {
    opacity: 0;
    filter: blur(3px);
    transform: translateY(10px);
  }
}

@keyframes grow {
  from {
    transform: scaleX(0);
  }
}

@media (max-width: 620px) {
  .result__head {
    grid-template-columns: 1fr;
  }

  .result__answer {
    justify-self: start;
  }
}
</style>
