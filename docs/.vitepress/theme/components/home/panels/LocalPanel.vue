<script setup lang="ts">
import { computed } from "vue"
import AppWindow from "../AppWindow.vue"

const props = defineProps<{ t: number }>()

const backends = [
  { name: "CPU", ready: true },
  { name: "CUDA", ready: true },
  { name: "Metal", ready: true },
  { name: "Vulkan", ready: true }
]

const load = computed(() => 0.35 + 0.2 * Math.sin(props.t * 1.2))
</script>

<template>
  <AppWindow title="local runtime" :basis="40">
    <template #bar><span class="meta">GGUF</span></template>
    <div class="panel">
      <div class="model">
        <span class="model__name">Qwen3-0.6B-Q8_0.gguf</span>
        <span class="model__state">loaded</span>
      </div>
      <div class="meter">
        <span>offload</span>
        <i><b :style="{ width: `${load * 100}%` }" /></i>
      </div>
      <ul>
        <li v-for="backend in backends" :key="backend.name">
          <i />
          {{ backend.name }}
        </li>
      </ul>
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
  padding: 1.15em;
}

.model {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8em;
  padding: 0.85em 0.95em;
  border: 1px solid var(--home-edge);
  border-radius: 0.85em;
  background: var(--home-fill);
}

.model__name {
  overflow: hidden;
  font-family: var(--vp-font-family-mono);
  font-size: 0.86em;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model__state {
  color: var(--home-green);
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.meter {
  display: grid;
  gap: 0.4em;
  color: var(--home-ink-2);
  font-size: 0.82em;
  font-weight: 650;
}

.meter i {
  display: block;
  overflow: hidden;
  height: 0.65em;
  border-radius: 999px;
  background: var(--home-fill);
}

.meter b {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--home-cyan), var(--home-violet));
}

ul {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.55em;
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: flex;
  align-items: center;
  gap: 0.45em;
  padding: 0.55em 0.7em;
  border: 1px solid var(--home-edge);
  border-radius: 0.7em;
  font-size: 0.86em;
  font-weight: 650;
}

li i {
  width: 0.45em;
  height: 0.45em;
  border-radius: 50%;
  background: var(--home-green);
  box-shadow: 0 0 0 0.2em color-mix(in srgb, var(--home-green) 22%, transparent);
}
</style>
