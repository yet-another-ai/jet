<script setup lang="ts">
import { computed, ref } from "vue"
import { withBase } from "vitepress"
import { homeCopy, type InstallBackend } from "./home-copy"
import { useRevealOnce } from "./home-motion"
import SegmentedControl from "./SegmentedControl.vue"

const t = homeCopy
const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.15)
const backend = ref<InstallBackend>("cpu")
const copied = ref(false)

const backendOptions = computed(() =>
  (["cpu", "cuda", "metal", "vulkan"] as const).map((value) => ({
    value,
    label: t.installBackends[value]
  }))
)
const target = computed(() => t.install[backend.value])
const commandLines = computed(() =>
  target.value.command.lines.map((line) => {
    const [program = "", ...args] = line.split(" ")
    return { line, program, args: args.join(" ") }
  })
)

let resetCopied = 0
async function copyCommand() {
  try {
    await navigator.clipboard.writeText(target.value.command.lines.join("\n"))
  } catch {
    return
  }
  copied.value = true
  window.clearTimeout(resetCopied)
  resetCopied = window.setTimeout(() => {
    copied.value = false
  }, 1600)
}
</script>

<template>
  <section id="get-jet" ref="root" class="install reveal" :class="{ 'is-shown': shown }">
    <div class="install__inner">
      <div class="install__mark reveal-line" style="--line: 0" aria-hidden="true">
        <span>J</span>
      </div>
      <p class="home-kicker reveal-line" style="--line: 1">{{ t.installKicker }}</p>
      <h2 class="home-heading reveal-line" style="--line: 2">{{ t.installTitle }}</h2>
      <p class="install__lead reveal-line" style="--line: 3">{{ t.installLead }}</p>

      <div class="install__controls reveal-line" style="--line: 4">
        <SegmentedControl
          v-model="backend"
          :options="backendOptions"
          :label="t.installKicker"
        />
      </div>

      <Transition name="install-swap" mode="out-in">
        <div :key="backend" class="install__target">
          <div class="install__block">
            <div class="install__head">
              <span>{{ target.command.label }}</span>
              <button type="button" :class="{ 'is-done': copied }" @click="copyCommand">
                {{ copied ? t.copied : t.copy }}
              </button>
            </div>
            <pre><code><span v-for="command in commandLines" :key="command.line" class="install__line"><span class="install__prompt" aria-hidden="true">$</span><span class="install__cmd">{{ command.program }}</span><template v-if="command.args"> {{ command.args }}</template></span></code></pre>
          </div>
          <p class="install__note">{{ target.note }}</p>
        </div>
      </Transition>

      <p class="install__links reveal-line" style="--line: 5">
        <a
          v-for="link in t.installLinks"
          :key="link.href"
          :href="link.href.startsWith('http') ? link.href : withBase(link.href)"
          >{{ link.label }} <span aria-hidden="true">→</span></a
        >
      </p>
    </div>
  </section>
</template>

<style scoped>
.install {
  padding: clamp(96px, 12vw, 150px) 24px clamp(80px, 10vw, 120px);
  scroll-margin-top: calc(var(--vp-nav-height) - 40px);
}

.install__inner {
  max-width: 780px;
  margin: 0 auto;
}

.install__mark {
  display: grid;
  width: 64px;
  height: 64px;
  place-items: center;
  margin-bottom: 32px;
  border-radius: 16px;
  color: var(--vp-c-bg);
  background:
    linear-gradient(145deg, var(--home-cyan), color-mix(in srgb, var(--home-violet) 55%, var(--home-cyan)));
  box-shadow: 0 14px 30px -12px var(--home-shadow);
  font-size: 28px;
  font-weight: 780;
  letter-spacing: -0.04em;
}

.install__lead {
  max-width: 580px;
  margin: 14px 0 32px;
  color: var(--home-ink-2);
  font-size: 18px;
  line-height: 1.55;
}

.install__controls {
  display: grid;
  justify-items: stretch;
  gap: 10px;
  margin-bottom: 14px;
}

.install__target {
  display: grid;
  gap: 12px;
}

.install__block {
  overflow: hidden;
  border-radius: 16px;
  background: var(--home-fill);
}

.install__head {
  display: flex;
  height: 40px;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px 0 18px;
  color: var(--home-ink-2);
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.02em;
}

.install__head button {
  height: 30px;
  padding: 0 12px;
  border-radius: 10px;
  color: var(--home-ink-2);
  font-size: 13px;
  font-weight: 650;
  cursor: pointer;
  transition:
    background 0.2s,
    color 0.2s;
}

.install__head button:hover {
  color: var(--home-ink);
  background: var(--home-fill);
}

.install__head button.is-done {
  color: var(--home-cyan-ink);
}

.install__block pre {
  margin: 0;
  padding: 2px 18px 16px;
  overflow-x: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  line-height: 1.8;
}

.install__line {
  display: block;
  white-space: pre;
}

.install__prompt {
  display: inline-block;
  width: 2ch;
  color: var(--home-ink-2);
  opacity: 0.6;
  user-select: none;
}

.install__cmd {
  color: var(--home-cyan-ink);
  font-weight: 650;
}

.install__note {
  margin: 0 2px;
  color: var(--home-ink-2);
  font-size: 14px;
  line-height: 1.55;
}

.install__links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
  margin: 28px 0 0;
}

.install__links a {
  color: var(--home-ink);
  font-size: 15px;
  font-weight: 650;
  text-decoration: none;
}

.install__links a span {
  display: inline-block;
  color: var(--home-cyan-ink);
  transition: transform 160ms ease;
}

.install__links a:hover span {
  transform: translateX(3px);
}

.install-swap-enter-active,
.install-swap-leave-active {
  transition:
    opacity 0.22s ease,
    transform 0.3s var(--home-ease);
}

.install-swap-enter-from {
  opacity: 0;
  transform: translateY(6px);
}

.install-swap-leave-to {
  opacity: 0;
}

@media (max-width: 560px) {
  .install {
    padding-inline: 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .install-swap-enter-active,
  .install-swap-leave-active {
    transition: none;
  }
}
</style>
