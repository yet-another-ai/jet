<script setup lang="ts">
import { computed } from "vue"
import AppWindow from "../AppWindow.vue"

const props = defineProps<{ t: number }>()

const pulse = computed(() => 0.5 + 0.5 * Math.sin(props.t * 0.75))
</script>

<template>
  <AppWindow title="request.jsonl" :basis="42">
    <template #bar><span class="meta">1 line</span></template>
    <div class="panel">
      <pre class="code"><code><span class="k">{</span>
  <span class="k">"state"</span>: {
    <span class="k">"ticket"</span>: <span class="s">"Login fails"</span>
  },
  <span class="k">"questions"</span>: {
    <span class="k">"urgent"</span>: { <span class="k">"type"</span>: <span class="s">"noul"</span> },
    <span class="k">"owner"</span>: {
      <span class="k">"type"</span>: <span class="s">"choice"</span>,
      <span class="k">"criteria"</span>: {
        <span class="k">"app"</span>: <span class="s">"…"</span>,
        <span class="k">"infra"</span>: <span class="s">"…"</span>
      }
    },
    <span class="k">"severity"</span>: {
      <span class="k">"type"</span>: <span class="s">"score"</span>,
      <span class="k">"criteria"</span>: [<span class="s">"low"</span>, <span class="s">"medium"</span>, <span class="s">"high"</span>]
    }
  }
<span class="k">}</span></code></pre>
      <div class="chip" :style="{ opacity: 0.55 + pulse * 0.45 }">schema locked</div>
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
  gap: 0.9em;
  padding: 1em 1.1em 1.1em;
}

.code {
  margin: 0;
  overflow: hidden;
  color: var(--home-ink-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.78em;
  line-height: 1.45;
  white-space: pre;
}

.k {
  color: var(--home-cyan-ink);
  font-weight: 650;
}

.s {
  color: var(--home-violet-ink);
}

.chip {
  align-self: flex-start;
  padding: 0.35em 0.7em;
  border: 1px solid var(--home-edge);
  border-radius: 999px;
  background: color-mix(in srgb, var(--home-cyan) 14%, transparent);
  color: var(--home-cyan-ink);
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
</style>
