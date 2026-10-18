<script setup vapor lang="ts" generic="T extends string">
defineProps<{ options: { value: T; label: string }[]; label: string }>();

const model = defineModel<T>({ required: true });
</script>

<template>
  <div
    class="segmented"
    role="tablist"
    :aria-label="label"
    :style="{
      '--count': options.length,
      '--index': options.findIndex((option) => option.value === model),
    }"
  >
    <span class="segmented__indicator" aria-hidden="true" />
    <button
      v-for="option in options"
      :key="option.value"
      type="button"
      role="tab"
      class="segmented__option"
      :aria-selected="model === option.value"
      @click="model = option.value"
    >
      {{ option.label }}
    </button>
  </div>
</template>

<style scoped>
.segmented {
  position: relative;
  display: grid;
  grid-template-columns: repeat(var(--count), minmax(0, 1fr));
  padding: 3px;
  border-radius: 11px;
  background: var(--fill);
}

.segmented__indicator {
  position: absolute;
  top: 3px;
  bottom: 3px;
  left: 3px;
  width: calc((100% - 6px) / var(--count));
  border-radius: 8px;
  background: var(--raised);
  box-shadow:
    inset 0 0 0 1px var(--edge),
    0 1px 3px var(--shadow-near);
  transform: translateX(calc(var(--index) * 100%));
  transition: transform 0.35s var(--ease);
}

.segmented__option {
  position: relative;
  z-index: 1;
  height: 28px;
  padding: 0 12px;
  border: 0;
  border-radius: 8px;
  color: var(--ink-2);
  background: transparent;
  font-size: 12.5px;
  font-weight: 650;
  white-space: nowrap;
  transition: color 0.2s;
}

.segmented__option[aria-selected='true'],
.segmented__option:hover {
  color: var(--ink);
}
</style>
