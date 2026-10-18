<script setup vapor lang="ts">
import { computed, ref } from 'vue';

const props = defineProps<{ value: unknown }>();

const text = computed(() => JSON.stringify(props.value, null, 2));
const copied = ref(false);

const tokenPattern =
  /("(?:\\.|[^"\\])*")(\s*:)?|(\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

const html = computed(() =>
  text.value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(tokenPattern, (_, string, colon = '', value) =>
      string
        ? `<span class="${colon ? 'json-key' : 'json-string'}">${string}</span>${colon}`
        : `<span class="json-value">${value}</span>`,
    ),
);

let resetCopied = 0;
async function copy(): Promise<void> {
  await navigator.clipboard.writeText(text.value);
  copied.value = true;
  window.clearTimeout(resetCopied);
  resetCopied = window.setTimeout(() => (copied.value = false), 1600);
}
</script>

<template>
  <div class="json-view">
    <button class="json-view__copy" type="button" @click="copy">
      {{ copied ? 'Copied' : 'Copy' }}
    </button>
    <pre><code v-html="html" /></pre>
  </div>
</template>

<style scoped>
.json-view {
  position: relative;
  min-height: 0;
  flex: 1;
  overflow: auto;
}

pre {
  margin: 0;
  padding: 18px 20px 22px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.65;
  tab-size: 2;
}

.json-view__copy {
  position: sticky;
  z-index: 1;
  top: 12px;
  float: right;
  height: 28px;
  margin: 12px 12px 0 0;
  padding: 0 12px;
  border: 1px solid var(--edge);
  border-radius: 14px;
  color: var(--ink-2);
  background: var(--panel);
  font-size: 12px;
  font-weight: 650;
  backdrop-filter: blur(12px);
}

.json-view__copy:hover {
  color: var(--ink);
}

:deep(.json-key) {
  color: var(--sky-ink);
}

:deep(.json-string) {
  color: var(--teal-ink);
}

:deep(.json-value) {
  color: var(--amber-ink);
}
</style>
