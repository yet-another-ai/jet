<script setup lang="ts">
import { ref } from "vue"
import { homeCopy } from "./home-copy"
import { useRevealOnce } from "./home-motion"

const t = homeCopy
const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.2)

const accents = ["var(--home-cyan)", "var(--home-violet)", "var(--home-amber)", "var(--home-green)"]
const icons = [
  // Checklist / choice.
  "M9 11l3 3L22 4 M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11",
  // Local chip.
  "M12 2a7 7 0 0 1 7 7c0 5-7 13-7 13S5 14 5 9a7 7 0 0 1 7-7z M12 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z",
  // Streaming lines.
  "M4 6h16 M4 12h10 M4 18h13",
  // Nested braces for API.
  "M8 4c-2 1-3 3-3 8s1 7 3 8 M16 4c2 1 3 3 3 8s-1 7-3 8"
]
</script>

<template>
  <section ref="root" class="values reveal" :class="{ 'is-shown': shown }">
    <header class="values__intro">
      <p class="home-kicker reveal-line" style="--line: 0">{{ t.valuesKicker }}</p>
      <h2 class="values__headline reveal-line" style="--line: 1">{{ t.valuesHeadline }}</h2>
      <p class="values__subtitle reveal-line" style="--line: 2">{{ t.valuesTitle }}</p>
    </header>
    <ul class="values__grid">
      <li
        v-for="(value, i) in t.values"
        :key="value.title"
        class="values__card reveal-line"
        :style="{ '--line': i + 3, '--accent': accents[i] }"
      >
        <svg class="values__icon" viewBox="0 0 24 24" aria-hidden="true">
          <path :d="icons[i]" />
        </svg>
        <h3>{{ value.title }}</h3>
        <p>{{ value.body }}</p>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.values {
  width: min(1120px, calc(100% - 48px));
  margin: 0 auto;
  padding: clamp(96px, 12vw, 150px) 0 40px;
}

.values__intro {
  max-width: 1040px;
  margin: 0 auto 48px;
  text-align: center;
}

.values__headline {
  margin: 0 0 18px;
  color: var(--home-ink);
  font-size: clamp(40px, 6vw, 72px);
  font-weight: 780;
  letter-spacing: -0.04em;
  line-height: 1;
  text-wrap: balance;
}

.values__subtitle {
  margin: 0;
  color: var(--home-ink-2);
  font-size: clamp(19px, 2vw, 24px);
  font-weight: 600;
  letter-spacing: -0.01em;
  text-wrap: balance;
}

.values__grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.values__card {
  padding: 24px 22px 26px;
  border: 1px solid var(--home-edge);
  border-radius: 22px;
  background: var(--home-panel);
  backdrop-filter: blur(18px);
}

.values__icon {
  width: 40px;
  height: 40px;
  margin-bottom: 18px;
  padding: 9px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  fill: none;
  stroke: color-mix(in srgb, var(--accent) 75%, var(--home-ink));
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.8;
}

.values__card h3 {
  margin: 0 0 8px;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.values__card p {
  margin: 0;
  color: var(--home-ink-2);
  font-size: 14.5px;
  line-height: 1.55;
}

@media (max-width: 960px) {
  .values__grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 560px) {
  .values {
    width: calc(100% - 32px);
  }

  .values__grid {
    grid-template-columns: 1fr;
  }
}
</style>
