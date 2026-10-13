<script setup lang="ts">
import { computed, ref } from "vue"
import { withBase } from "vitepress"
import { homeCopy, type CapabilityKey } from "./home-copy"
import {
  smoothstep,
  useAnimationClock,
  usePrefersReducedMotion,
  useRevealOnce,
  useScrollProgress
} from "./home-motion"
import SegmentedControl from "./SegmentedControl.vue"
import LocalPanel from "./panels/LocalPanel.vue"
import RequestPanel from "./panels/RequestPanel.vue"
import ResultPanel from "./panels/ResultPanel.vue"
import ScoresPanel from "./panels/ScoresPanel.vue"
import StreamPanel from "./panels/StreamPanel.vue"

const REST_TIME = 4.8
const t = homeCopy
const intro = ref<HTMLElement>()
const scroller = ref<HTMLElement>()
const reduced = usePrefersReducedMotion()
const introShown = useRevealOnce(intro)
const time = useAnimationClock(scroller, reduced, REST_TIME)
const progress = useScrollProgress(scroller, "through")

const windows = {
  decide: ScoresPanel,
  local: LocalPanel,
  embed: StreamPanel
} as const

const companions = {
  decide: ResultPanel,
  local: RequestPanel,
  embed: ScoresPanel
} as const

const glows: Record<CapabilityKey, string> = {
  decide: "var(--home-cyan)",
  local: "var(--home-violet)",
  embed: "var(--home-amber)"
}

const count = computed(() => t.capabilities.length)

/** Extra scroll, in viewport heights, spent resting on the first and last product. */
const EDGE_REST = 0.35

/** Scroll position mapped to a capability index that rests on each item before moving on. */
const position = computed(() => {
  const span = count.value - 1
  const x = Math.min(span, Math.max(0, progress.value * (span + 2 * EDGE_REST) - EDGE_REST))
  const index = Math.min(Math.floor(x), span)
  return index + smoothstep(0.3, 0.7, x - index)
})
const active = computed(() => Math.round(position.value))
const activeKey = computed(() => t.capabilities[active.value]!.key)

function cardStyle(index: number) {
  const offset = index - position.value
  const distance = Math.min(Math.abs(offset), 2)
  return {
    zIndex: 10 - Math.round(distance * 3),
    opacity: Math.max(0, 1 - distance * 0.6),
    filter: distance > 0.01 ? `blur(${distance * 5}px)` : "none",
    transform: `translateX(${offset * 64}%) scale(${1 - distance * 0.24}) rotateY(${offset * -16}deg)`
  }
}

function show(key: CapabilityKey) {
  const element = scroller.value
  const index = t.capabilities.findIndex((capability) => capability.key === key)
  if (!element || index < 0) return
  const top = element.getBoundingClientRect().top + window.scrollY
  const distance = element.offsetHeight - window.innerHeight
  const span = count.value - 1
  window.scrollTo({
    top: top + (distance * (index + EDGE_REST)) / (span + 2 * EDGE_REST),
    behavior: reduced.value ? "auto" : "smooth"
  })
}
</script>

<template>
  <section id="capabilities" class="showcase" :aria-label="t.capabilitiesAria">
    <header ref="intro" class="showcase__intro reveal" :class="{ 'is-shown': introShown }">
      <p class="home-kicker reveal-line" style="--line: 0">{{ t.capabilitiesKicker }}</p>
      <h2 class="home-heading reveal-line" style="--line: 1">{{ t.capabilitiesTitle }}</h2>
    </header>

    <div
      ref="scroller"
      class="showcase__scroller"
      :style="{ height: `${(count + 2 * EDGE_REST) * 100}svh` }"
    >
      <div class="showcase__sticky">
        <div class="showcase__stage" aria-hidden="true">
          <span class="showcase__glow" :style="{ background: glows[activeKey] }" />
          <div
            v-for="(capability, i) in t.capabilities"
            :key="capability.key"
            class="showcase__card"
            :style="cardStyle(i)"
          >
            <div class="showcase__pair">
              <component :is="windows[capability.key]" :t="time" />
              <component :is="companions[capability.key]" :t="time" />
            </div>
          </div>
        </div>

        <div class="showcase__text">
          <article
            v-for="(capability, i) in t.capabilities"
            :key="capability.key"
            class="showcase__item reveal"
            :class="{ 'is-shown': i === active }"
            :inert="i !== active"
          >
            <p class="home-kicker reveal-line" style="--line: 0">{{ capability.kicker }}</p>
            <h3 class="showcase__title reveal-line" style="--line: 1">{{ capability.title }}</h3>
            <p class="showcase__body reveal-line" style="--line: 2">{{ capability.body }}</p>
            <p class="showcase__links reveal-line" style="--line: 3">
              <a
                v-for="link in capability.links"
                :key="link.href"
                :href="link.href.startsWith('http') ? link.href : withBase(link.href)"
                >{{ link.label }} <span aria-hidden="true">→</span></a
              >
            </p>
          </article>
        </div>

        <SegmentedControl
          class="showcase__island"
          size="small"
          :label="t.capabilitiesAria"
          :options="t.capabilities.map((capability) => ({ value: capability.key, label: capability.tab }))"
          :model-value="activeKey"
          @update:model-value="show"
        />
      </div>
    </div>
  </section>
</template>

<style scoped>
.showcase {
  position: relative;
}

.showcase__intro {
  max-width: 760px;
  margin: 0 auto;
  padding: clamp(96px, 14vw, 160px) 24px 0;
  text-align: center;
}

.showcase__sticky {
  position: sticky;
  top: 0;
  display: flex;
  height: 100svh;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: clamp(18px, 3vh, 32px);
  padding: calc(var(--vp-nav-height) + 8px) 16px 24px;
  overflow: hidden;
}

.showcase__stage {
  position: relative;
  width: min(880px, 88vw, (100svh - 380px) * 1.6);
  aspect-ratio: 16 / 10;
  flex: none;
  perspective: 1800px;
}

.showcase__glow {
  position: absolute;
  inset: 8% 14%;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.28;
  transition: background 1.2s ease;
}

.showcase__card {
  position: absolute;
  inset: 0;
  transform-origin: 50% 60%;
  will-change: transform, opacity;
}

.showcase__pair {
  display: grid;
  height: 100%;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 3%;
}

.showcase__text {
  display: grid;
  width: min(640px, 100%);
  text-align: center;
}

.showcase__item {
  grid-area: 1 / 1;
}

.showcase__item:not(.is-shown) {
  pointer-events: none;
}

.showcase__title {
  margin: 0 0 10px;
  font-size: clamp(26px, 3vw, 34px);
  font-weight: 750;
  letter-spacing: -0.025em;
  line-height: 1.1;
}

.showcase__body {
  margin: 0;
  color: var(--home-ink-2);
  font-size: 16px;
  line-height: 1.6;
}

.showcase__links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 6px 20px;
  margin: 16px 0 0;
}

.showcase__links a {
  color: var(--home-ink);
  font-size: 15px;
  font-weight: 650;
  text-decoration: none;
}

.showcase__links a span {
  display: inline-block;
  color: var(--home-cyan-ink);
  transition: transform 160ms ease;
}

.showcase__links a:hover span {
  transform: translateX(3px);
}

.showcase__island {
  flex: none;
  border: 1px solid var(--home-edge);
  border-radius: 999px;
  background: var(--home-panel);
  box-shadow: 0 10px 30px var(--home-shadow-near);
  backdrop-filter: blur(20px) saturate(1.4);
}

.showcase__island :deep(.segmented__indicator),
.showcase__island :deep(.segmented__option) {
  border-radius: 999px;
}

@media (max-width: 640px) {
  .showcase__stage {
    width: min(92vw, (100svh - 420px) * 1.6);
  }

  .showcase__pair {
    grid-template-columns: 1fr;
  }

  .showcase__pair > :last-child {
    display: none;
  }

  .showcase__body {
    font-size: 15px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .showcase__glow {
    transition: none;
  }
}
</style>
