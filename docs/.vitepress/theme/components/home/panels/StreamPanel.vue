<script setup lang="ts">
import { computed } from "vue"
import AppWindow from "../AppWindow.vue"

const props = defineProps<{ t: number }>()

const lines = [
  { kind: "in", text: '{"questions":{"urgent":{"type":"noul"}}}' },
  { kind: "out", text: '{"answers":{"urgent":true},"usage":{…}}' },
  { kind: "in", text: '{"questions":{"owner":{"type":"choice"}}}' },
  { kind: "out", text: '{"answers":{"owner":"app"},"usage":{…}}' }
]

const visible = computed(() => Math.min(lines.length, 1 + Math.floor(props.t / 2.1) % lines.length))
</script>

<template>
  <AppWindow title="jet judge" :basis="44">
    <template #bar><span class="meta">JSONL</span></template>
    <div class="panel">
      <div class="prompt" aria-hidden="true">$</div>
      <div class="stream">
        <div
          v-for="(line, i) in lines.slice(0, visible)"
          :key="`${line.kind}-${i}`"
          class="line"
          :class="`line--${line.kind}`"
        >
          <span>{{ line.kind === "in" ? "←" : "→" }}</span>
          <code>{{ line.text }}</code>
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
  grid-template-columns: auto 1fr;
  gap: 0.75em;
  padding: 1em 1.1em;
}

.prompt {
  color: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.95em;
  font-weight: 700;
  opacity: 0.55;
}

.stream {
  display: grid;
  align-content: start;
  gap: 0.55em;
  min-width: 0;
}

.line {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.55em;
  align-items: start;
  font-size: 0.84em;
}

.line span {
  color: var(--home-ink-2);
  font-weight: 700;
}

.line code {
  overflow: hidden;
  font-family: var(--vp-font-family-mono);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.line--in code {
  color: var(--home-violet-ink);
}

.line--out code {
  color: var(--home-cyan-ink);
}
</style>
