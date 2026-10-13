<script setup lang="ts">
import { computed } from "vue"
import AppWindow from "../AppWindow.vue"

const props = defineProps<{ t: number }>()

const rows = [
  { key: "app", label: "Application team", base: 0.82, color: "var(--home-cyan)" },
  { key: "infra", label: "Infrastructure team", base: 0.41, color: "var(--home-violet)" },
  { key: "support", label: "Support team", base: 0.18, color: "var(--home-amber)" }
]

const widths = computed(() =>
  rows.map((row, i) => {
    const wobble = 0.04 * Math.sin(props.t * 1.7 + i * 1.1)
    return Math.min(0.96, Math.max(0.08, row.base + wobble))
  })
)

const winner = computed(() => {
  let best = 0
  for (let i = 1; i < widths.value.length; i++) {
    if (widths.value[i]! > widths.value[best]!) best = i
  }
  return best
})
</script>

<template>
  <AppWindow title="candidate scores" :basis="40">
    <template #bar><span class="meta">softmax</span></template>
    <div class="panel">
      <div v-for="(row, i) in rows" :key="row.key" class="row" :class="{ 'is-top': i === winner }">
        <div class="row__head">
          <span>{{ row.label }}</span>
          <span class="row__key">{{ row.key }}</span>
        </div>
        <div class="bar">
          <i :style="{ width: `${widths[i]! * 100}%`, background: row.color }" />
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

.row__key {
  color: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.82em;
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
  transition: width 120ms linear;
}

.row.is-top .row__head span:first-child {
  color: var(--home-cyan-ink);
}

@media (prefers-reduced-motion: reduce) {
  .bar i {
    transition: none;
  }
}
</style>
