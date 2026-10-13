<script setup lang="ts">
import { ref } from "vue"
import { withBase } from "vitepress"
import { useRevealOnce } from "./home-motion"

const root = ref<HTMLElement>()
const shown = useRevealOnce(root, 0.2)

const points = [
  {
    title: "Nine admissible controls",
    body: "Forward, backward, strafe, turn, fire, use door, and wait—scored as fixed candidates on every observation."
  },
  {
    title: "160×100 RGB screenshots",
    body: "The controller downsamples the live 640×400 view to an exact 160×100 PNG before Jet scores the image."
  },
  {
    title: "Visible state only",
    body: "HUD health and ammo, the previous action, and recent motion cues are included. Hidden enemy positions and map geometry are not."
  },
  {
    title: "Vision path, thinking off",
    body: "The demo uses Jet’s multimodal judge with a vision-language model and projector. The current implementation requires reasoning disabled."
  }
]
</script>

<template>
  <section id="doom" ref="root" class="doom reveal" :class="{ 'is-shown': shown }">
    <div class="doom__copy">
      <p class="home-kicker reveal-line" style="--line: 0">Multimodal · Doom demo</p>
      <h2 class="home-heading reveal-line" style="--line: 1">
        Score actions from what the agent can see.
      </h2>
      <p class="doom__lead reveal-line" style="--line: 2">
        The Doom controller in <code>demos/doom</code> shows Jet’s multimodal path:
        choose among nine controls from a screenshot plus visible status, while the
        game keeps running. It is a visual control demo, not a trained Doom policy,
        and visual decision accuracy has not been quantitatively evaluated.
      </p>
      <p class="doom__links reveal-line" style="--line: 3">
        <a href="https://github.com/yet-another-ai/jet/tree/main/demos/doom"
          >Doom demo README <span aria-hidden="true">→</span></a
        >
        <a :href="withBase('/advanced#multimodal-decisions')"
          >Multimodal decisions <span aria-hidden="true">→</span></a
        >
      </p>
    </div>

    <ul class="doom__grid">
      <li
        v-for="(point, i) in points"
        :key="point.title"
        class="doom__card reveal-line"
        :style="{ '--line': i + 4 }"
      >
        <h3>{{ point.title }}</h3>
        <p>{{ point.body }}</p>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.doom {
  width: min(1120px, calc(100% - 48px));
  margin: 0 auto;
  padding: clamp(72px, 10vw, 120px) 0 20px;
  scroll-margin-top: calc(var(--vp-nav-height) - 40px);
}

.doom__copy {
  max-width: 720px;
  margin-bottom: 36px;
}

.doom__lead {
  margin: 14px 0 0;
  color: var(--home-ink-2);
  font-size: 18px;
  line-height: 1.55;
}

.doom__lead code {
  font-size: 0.92em;
}

.doom__links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
  margin: 22px 0 0;
}

.doom__links a {
  color: var(--home-ink);
  font-size: 15px;
  font-weight: 650;
  text-decoration: none;
}

.doom__links a span {
  display: inline-block;
  color: var(--home-cyan-ink);
  transition: transform 220ms ease;
}

.doom__links a:hover span {
  transform: translateX(3px);
}

.doom__grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.doom__card {
  padding: 24px 22px 26px;
  border: 1px solid var(--home-edge);
  border-radius: 22px;
  background: var(--home-panel);
  backdrop-filter: blur(18px);
}

.doom__card h3 {
  margin: 0 0 8px;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.doom__card p {
  margin: 0;
  color: var(--home-ink-2);
  font-size: 14.5px;
  line-height: 1.55;
}

@media (max-width: 720px) {
  .doom {
    width: calc(100% - 32px);
  }

  .doom__grid {
    grid-template-columns: 1fr;
  }
}

@media (prefers-reduced-motion: reduce) {
  .doom__links a span {
    transition: none;
  }
}
</style>
