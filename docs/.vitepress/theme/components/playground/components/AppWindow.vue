<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <section class="app-window">
    <header class="app-window__bar">
      <span class="app-window__lights" aria-hidden="true"><i /><i /><i /></span>
      <h2 class="app-window__title">{{ title }}</h2>
      <span class="app-window__end"><slot name="bar" /></span>
    </header>
    <div class="app-window__body"><slot /></div>
    <footer v-if="$slots.footer" class="app-window__footer"><slot name="footer" /></footer>
  </section>
</template>

<style scoped>
.app-window {
  display: flex;
  overflow: hidden;
  min-width: 0;
  flex-direction: column;
  border: 1px solid var(--edge);
  border-radius: 18px;
  background: var(--window);
  box-shadow:
    0 28px 56px -28px var(--shadow),
    0 4px 14px -6px var(--shadow-near);
  backdrop-filter: blur(22px) saturate(1.4);
}

.app-window__bar {
  display: grid;
  min-height: 46px;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 12px;
  padding: 0 10px 0 16px;
  border-bottom: 1px solid var(--edge);
}

.app-window__lights {
  display: flex;
  gap: 7px;
}

.app-window__lights i {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: #ff5f57;
}

.app-window__lights i:nth-child(2) {
  background: #febc2e;
}

.app-window__lights i:nth-child(3) {
  background: #28c840;
}

.app-window__title {
  margin: 0;
  color: var(--ink-2);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.01em;
}

.app-window__end {
  display: flex;
  min-width: 0;
  justify-content: flex-end;
}

.app-window__body {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  overflow: auto;
}

.app-window__footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 16px;
  padding: 12px 18px;
  border-top: 1px solid var(--edge);
  color: var(--ink-2);
  font-family: var(--font-mono);
  font-size: 11.5px;
  font-variant-numeric: tabular-nums;
}

@media (max-width: 620px) {
  .app-window__bar {
    grid-template-columns: auto 1fr;
  }

  .app-window__lights {
    display: none;
  }

  .app-window__title {
    justify-self: start;
  }
}
</style>
