<script setup lang="ts">
import { computed } from "vue"
import AppWindow from "../AppWindow.vue"
import { demoPhase, demoQuestions } from "../demo-scenario"

const props = defineProps<{ t: number }>()

const active = computed(() => demoPhase(props.t))
</script>

<template>
  <AppWindow title="result.json" :basis="36">
    <template #bar><span class="meta">ordered</span></template>
    <div class="panel">
      <div class="label">answers</div>
      <ul>
        <li
          v-for="(answer, i) in demoQuestions"
          :key="answer.name"
          :class="{ 'is-active': i === active }"
        >
          <span class="name">{{ answer.name }}</span>
          <span class="value" :style="{ color: answer.tone }">{{ answer.answer }}</span>
        </li>
      </ul>
      <div class="usage">
        <span>prompt 184</span>
        <span>candidates {{ demoQuestions[active]!.candidates.length }}</span>
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
  display: flex;
  height: 100%;
  flex-direction: column;
  gap: 0.85em;
  padding: 1.05em 1.15em 1.15em;
}

.label {
  color: var(--home-ink-2);
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

ul {
  display: grid;
  margin: 0;
  padding: 0;
  list-style: none;
  gap: 0.55em;
}

li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8em;
  padding: 0.7em 0.85em;
  border: 1px solid var(--home-edge);
  border-radius: 0.75em;
  background: var(--home-fill);
  transition:
    border-color 320ms ease,
    background 320ms ease,
    transform 320ms ease;
}

li.is-active {
  border-color: color-mix(in srgb, var(--home-cyan) 40%, var(--home-edge));
  background: color-mix(in srgb, var(--home-cyan) 12%, var(--home-fill));
  transform: translateX(0.15em);
}

.name {
  font-family: var(--vp-font-family-mono);
  font-size: 0.9em;
  font-weight: 650;
}

.value {
  font-family: var(--vp-font-family-mono);
  font-size: 0.95em;
  font-weight: 700;
}

.usage {
  display: flex;
  gap: 0.9em;
  margin-top: auto;
  color: var(--home-ink-2);
  font-size: 0.78em;
  font-weight: 650;
}

@media (prefers-reduced-motion: reduce) {
  li {
    transition: none;
  }

  li.is-active {
    transform: none;
  }
}
</style>
