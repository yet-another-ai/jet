<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';

import AppWindow from './components/AppWindow.vue';
import JsonView from './components/JsonView.vue';
import ProbabilityResults from './components/ProbabilityResults.vue';
import QuestionEditor from './components/QuestionEditor.vue';
import SegmentedControl from './components/SegmentedControl.vue';
import { draftId, type DecisionResponse, type QuestionDraft, type ResultGroup } from './types'
import { isModelLoaded, loadModel, probeRuntime, runDecision, supportsWebGpu } from './runtime'
import { ensurePlaygroundCoi } from './ensure-coi'
import './playground.css'

const defaultModelUrl =
  'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf';
const modelName = 'Qwen3 0.6B · Q8_0';

const prompt = ref('Login fails for all users');

function draft(text: string, options: string[]): QuestionDraft {
  return {
    id: draftId(),
    text,
    options: options.map((option) => ({ id: draftId(), text: option })),
  };
}

const questions = ref([
  draft('Who should own this incident?', ['Application team', 'Infrastructure team']),
  draft('How severe is it?', ['Low', 'Medium', 'High']),
]);

const webGpu = supportsWebGpu();
const modelLoaded = ref(isModelLoaded(defaultModelUrl));
const loading = ref(false);
const running = ref(false);
const download = ref({ loaded: 0, total: 0 });
const errorText = ref('');
const results = ref<ResultGroup[]>([]);
const response = ref<DecisionResponse>();
const elapsedMs = ref(0);
const requestView = ref<'editor' | 'json'>('editor');
const resultView = ref<'scores' | 'json'>('scores');
const shown = ref(false)
const bootstrapped = ref(false)
const runtimeReady = ref(false)
const coiReady = ref(false)
const coiBlocked = ref(false)
const busy = computed(() => loading.value || running.value);
const controlsEnabled = computed(
  () => bootstrapped.value && runtimeReady.value && coiReady.value && webGpu && !coiBlocked.value,
);
const showRuntimeMissing = computed(
  () => bootstrapped.value && !runtimeReady.value && !coiBlocked.value,
);
const showCoiBlocked = computed(() => bootstrapped.value && coiBlocked.value);

const downloadPercent = computed(() =>
  download.value.total > 0 ? Math.round((download.value.loaded / download.value.total) * 100) : 0,
);

const modelStatus = computed(() => {
  if (!webGpu) return 'WebGPU unavailable';
  if (loading.value) {
    return download.value.total > 0
      ? `Downloading ${Math.round(download.value.loaded / 1e6)} / ${Math.round(download.value.total / 1e6)} MB`
      : 'Preparing model…';
  }
  if (running.value) return 'Scoring candidates…';
  if (modelLoaded.value) return `${modelName} · ready`;
  return `${modelName} · not loaded`;
});

const runLabel = computed(() => {
  if (loading.value) return `Loading ${downloadPercent.value}%`;
  if (running.value) return 'Running…';
  return modelLoaded.value ? 'Run Jet' : 'Load & run';
});

const requestMeta = computed(() => {
  const count = questions.value.length;
  const candidates = questions.value.flatMap((question) =>
    question.options.filter((option) => option.text.trim()),
  ).length;
  return `${count} ${count === 1 ? 'question' : 'questions'} · ${candidates} candidates`;
});

watch(
  [prompt, questions],
  () => {
    results.value = [];
    response.value = undefined;
    errorText.value = '';
  },
  { deep: true },
);

onMounted(() => {
  void (async () => {
    const coi = await ensurePlaygroundCoi()
    if (coi === 'reloading') return
    if (coi === 'unavailable') {
      coiBlocked.value = true
      coiReady.value = false
    } else {
      coiReady.value = true
    }
    runtimeReady.value = await probeRuntime()
    bootstrapped.value = true
  })()
  requestAnimationFrame(() => (shown.value = true))
})

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function ensureModel(): Promise<boolean> {
  if (modelLoaded.value) return true;
  loading.value = true;
  download.value = { loaded: 0, total: 0 };
  errorText.value = '';
  try {
    await loadModel(defaultModelUrl, (progress) => (download.value = progress));
    modelLoaded.value = true;
    return true;
  } catch (error) {
    errorText.value = errorMessage(error);
    return false;
  } finally {
    loading.value = false;
  }
}

function validateDraft(): void {
  if (!prompt.value.trim()) throw new Error('Enter a prompt before running Jet.');
  if (questions.value.length === 0) throw new Error('Add at least one question.');
  questions.value.forEach((question, questionIndex) => {
    if (!question.text.trim()) throw new Error(`Question ${questionIndex + 1} is empty.`);
    const options = question.options.map((option) => option.text.trim()).filter(Boolean);
    if (options.length < 2) {
      throw new Error(`Question ${questionIndex + 1} needs at least two options.`);
    }
  });
}

const request = computed(() => ({
    state: { prompt: prompt.value.trim() },
    questions: Object.fromEntries(
      questions.value.map((question, questionIndex) => [
        `question_${questionIndex + 1}`,
        {
          type: 'choice',
          instructions: question.text.trim(),
          criteria: Object.fromEntries(
            question.options
              .filter((option) => option.text.trim())
              .map((option, optionIndex) => [`option_${optionIndex + 1}`, option.text.trim()]),
          ),
        },
      ]),
    ),
  }));

function responseToResults(decision: DecisionResponse): ResultGroup[] {
  return questions.value.map((question, questionIndex) => {
    const answer = decision.answers[`question_${questionIndex + 1}`];
    if (!answer || answer.type !== 'choice') {
      throw new Error(`Jet did not return a choice result for question ${questionIndex + 1}.`);
    }
    return {
      id: question.id,
      question: question.text.trim(),
      options: question.options
        .filter((option) => option.text.trim())
        .map((option, optionIndex) => {
          const key = `option_${optionIndex + 1}`;
          return {
            id: option.id,
            label: option.text.trim(),
            probability: answer.probabilities[key] ?? 0,
            winner: answer.choice === key,
          };
        }),
    };
  });
}

async function run(): Promise<void> {
  running.value = true;
  errorText.value = '';
  results.value = [];
  response.value = undefined;
  try {
    validateDraft();
    if (!(await ensureModel())) return;
    const started = performance.now();
    const decision = (await runDecision(defaultModelUrl, request.value)) as DecisionResponse;
    elapsedMs.value = performance.now() - started;
    results.value = responseToResults(decision);
    response.value = decision;
    if (matchMedia('(max-width: 980px)').matches) {
      document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  } catch (error) {
    errorText.value = errorMessage(error);
  } finally {
    running.value = false;
  }
}

</script>

<template>
  <div class="playground-shell shell" :class="{ 'is-shown': shown }">
    <div v-if="showRuntimeMissing" class="runtime-missing" role="status">
      <h2>Playground Wasm not built</h2>
      <p>
        This docs build did not include the WebGPU Wasm module. Build it locally,
        then rebuild the docs:
      </p>
      <p><code>mise run playground:wllama && mise exec -- pnpm docs:build</code></p>
      <p>See the <a href="/guide/playground">playground guide</a> for requirements.</p>
    </div>
    <div v-else-if="showCoiBlocked" class="runtime-missing" role="status">
      <h2>Cross-origin isolation unavailable</h2>
      <p>
        This browser session could not enable
        <code>SharedArrayBuffer</code> (COOP/COEP) for the playground. Use a
        secure context with service workers, or open the page in a full browser
        tab (not a restricted iframe).
      </p>
    </div>
    <header class="hero">
      <p class="hero__kicker reveal" style="--line: 0">WebGPU playground</p>
      <h1 class="reveal" style="--line: 1">Jet, in your browser</h1>
      <p class="hero__lead reveal" style="--line: 2">
        The native Rust engine and llama.cpp scorer, compiled to WebAssembly and running on your
        GPU. Your prompt never leaves this tab.
      </p>
      <ul class="hero__facts reveal" style="--line: 3">
        <li><i style="background: var(--teal)" />jet-engine → Wasm</li>
        <li><i style="background: var(--sky)" />llama.cpp on WebGPU</li>
        <li><i style="background: var(--amber)" />Teacher-forced scoring</li>
      </ul>
    </header>

    <main class="workspace">
      <AppWindow class="workspace__request reveal" style="--line: 4" title="request.json">
        <template #bar>
          <SegmentedControl
            v-model="requestView"
            label="Request view"
            :options="[
              { value: 'editor', label: 'Editor' },
              { value: 'json', label: 'JSON' },
            ]"
          />
        </template>

        <div v-if="requestView === 'editor'" class="editor">
          <label class="editor__label key" for="prompt">
            <span>state.prompt</span>
            <span>{{ prompt.length }} / 4000</span>
          </label>
          <textarea
            id="prompt"
            v-model="prompt"
            class="editor__prompt field"
            maxlength="4000"
            rows="3"
            :disabled="busy"
            placeholder="Describe the situation Jet should evaluate"
          />

          <div class="editor__label editor__label--section key">
            <span>questions</span>
            <span>{{ requestMeta }}</span>
          </div>
          <div class="editor__questions">
            <QuestionEditor
              v-for="(question, questionIndex) in questions"
              :key="question.id"
              :question="question"
              :index="questionIndex"
              :disabled="busy"
              :can-remove="questions.length > 1"
              @remove="questions.splice(questionIndex, 1)"
            />
          </div>
          <button class="editor__add" type="button" :disabled="busy" @click="questions.push(draft('', ['', '']))">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v10M3 8h10" /></svg>
            Add question
          </button>
        </div>
        <JsonView v-else :value="request" />
      </AppWindow>

      <AppWindow
        id="results"
        class="workspace__result reveal"
        style="--line: 5"
        :title="resultView === 'json' ? 'response.json' : 'candidate scores'"
        aria-live="polite"
      >
        <template #bar>
          <SegmentedControl
            v-if="response"
            v-model="resultView"
            label="Result view"
            :options="[
              { value: 'scores', label: 'Scores' },
              { value: 'json', label: 'JSON' },
            ]"
          />
          <span v-else class="window-meta">softmax</span>
        </template>

        <p v-if="errorText" class="alert" role="alert">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 4.5v4M8 11v.5" /><circle cx="8" cy="8" r="6.5" /></svg>
          {{ errorText }}
        </p>

        <template v-if="response">
          <ProbabilityResults v-if="resultView === 'scores'" :groups="results" />
          <JsonView v-else :value="response" />
        </template>

        <div v-else-if="busy" class="pending">
          <div v-for="row in 2" :key="row" class="pending__card card">
            <span class="pending__line pending__line--title" />
            <span class="pending__line" />
            <span class="pending__line pending__line--short" />
          </div>
          <p>{{ loading ? `Loading ${modelName} into WebGPU…` : 'Scoring every candidate in one pass…' }}</p>
        </div>

        <div v-else class="empty">
          <div class="empty__bars" aria-hidden="true">
            <span style="width: 72%; background: color-mix(in srgb, var(--teal) 40%, transparent)" />
            <span style="width: 46%; background: color-mix(in srgb, var(--sky) 32%, transparent)" />
            <span style="width: 22%" />
          </div>
          <h3>Scores appear here</h3>
          <p>
            Jet compares every candidate under each question and returns a probability for each one,
            with no free-form generation.
          </p>
        </div>

        <template v-if="response" #footer>
          <span>model {{ response.model }}</span>
          <span>input {{ response.usage.input_tokens }} tok</span>
          <span>output {{ response.usage.output_tokens }} tok</span>
          <span>{{ Math.round(elapsedMs) }} ms</span>
        </template>
      </AppWindow>
    </main>

    <div class="island" role="group" aria-label="Model and run controls">
      <span
        class="island__status"
        :class="{ 'is-ready': modelLoaded, 'is-busy': busy, 'is-unavailable': !webGpu }"
      >
        <i class="island__dot" aria-hidden="true" />
        <span class="island__text">{{ modelStatus }}</span>
        <span
          v-if="loading"
          class="island__progress"
          :style="{ '--p': downloadPercent / 100 }"
          aria-hidden="true"
        />
      </span>
      <button
        v-if="!modelLoaded"
        class="island__button"
        type="button"
        :disabled="busy || !controlsEnabled"
        @click="ensureModel"
      >
        Load model
      </button>
      <button
        class="island__button island__button--solid"
        type="button"
        :disabled="busy || !controlsEnabled"
        @click="run"
      >
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.5v9l7.5-4.5z" /></svg>
        {{ runLabel }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.shell {
  min-height: 100vh;
  padding-bottom: 120px;
}

.runtime-missing {
  max-width: 40rem;
  margin: 24px auto 0;
  padding: 20px 22px;
  border: 1px solid var(--edge);
  border-radius: 16px;
  background: var(--panel);
  text-align: left;
}

.runtime-missing h2 {
  margin: 0 0 8px;
  font-size: 18px;
}

.runtime-missing p {
  margin: 0 0 8px;
  color: var(--ink-2);
  font-size: 14px;
  line-height: 1.5;
}

.runtime-missing a {
  color: var(--teal-ink);
}

.runtime-missing code {
  font-family: var(--font-mono);
  font-size: 0.92em;
}










.hero {
  max-width: 820px;
  margin: 0 auto;
  padding: 44px 24px 34px;
  text-align: center;
}

.hero__kicker {
  margin: 0 0 14px;
  color: var(--sky-ink);
  font-size: 13px;
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero h1 {
  margin: 0 0 14px;
  font-size: clamp(36px, 4.8vw, 56px);
  font-weight: 760;
  letter-spacing: -0.035em;
  line-height: 1.04;
  text-wrap: balance;
}

.hero__lead {
  max-width: 680px;
  margin: 0 auto;
  color: var(--ink-2);
  font-size: clamp(16px, 1.6vw, 19px);
  line-height: 1.55;
  text-wrap: pretty;
}

.hero__facts {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin: 26px 0 0;
  padding: 0;
  list-style: none;
}

.hero__facts li {
  display: inline-flex;
  height: 32px;
  align-items: center;
  gap: 8px;
  padding: 0 14px;
  border: 1px solid var(--edge);
  border-radius: 16px;
  color: var(--ink-2);
  background: var(--panel);
  font-size: 13px;
  font-weight: 600;
  backdrop-filter: blur(14px);
}

.hero__facts i {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}

.workspace {
  display: grid;
  width: min(1320px, calc(100% - 48px));
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
  align-items: start;
  gap: 24px;
  margin: 0 auto;
}

.workspace__result {
  position: sticky;
  top: 88px;
  min-height: 460px;
  max-height: calc(100vh - 188px);
  scroll-margin-top: 72px;
}

.window-meta {
  padding-right: 6px;
  color: var(--ink-2);
  font-size: 12px;
  font-weight: 600;
}

.editor {
  padding: 18px 16px 18px;
}

.editor__label {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin: 0 4px 8px;
}

.editor__label--section {
  margin-top: 22px;
}

.editor__prompt {
  display: block;
  width: 100%;
  min-height: 84px;
  padding: 13px 15px;
  border-radius: 12px;
  font-size: 15px;
  line-height: 1.55;
  resize: vertical;
}

.editor__questions {
  display: grid;
  gap: 10px;
}

.editor__add {
  display: flex;
  width: 100%;
  height: 44px;
  align-items: center;
  justify-content: center;
  gap: 7px;
  margin-top: 10px;
  border: 1px dashed color-mix(in srgb, var(--ink) 18%, transparent);
  border-radius: 14px;
  color: var(--ink-2);
  background: transparent;
  font-size: 13.5px;
  font-weight: 650;
  transition:
    color 0.16s,
    border-color 0.16s,
    background 0.16s;
}

.editor__add:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--teal) 55%, transparent);
  color: var(--teal-ink);
  background: color-mix(in srgb, var(--teal) 6%, transparent);
}

.alert {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  margin: 16px 16px 0;
  padding: 11px 14px;
  border-radius: 12px;
  color: var(--red-ink);
  background: color-mix(in srgb, var(--red) 11%, transparent);
  font-size: 13.5px;
  line-height: 1.5;
}

.alert svg {
  margin-top: 3px;
}

.empty {
  display: grid;
  flex: 1;
  align-content: center;
  justify-items: center;
  padding: 56px 32px;
  text-align: center;
}

.empty__bars {
  display: grid;
  width: 180px;
  gap: 9px;
  margin-bottom: 26px;
}

.empty__bars span {
  height: 8px;
  border-radius: 999px;
  background: var(--fill-strong);
}

.empty h3 {
  margin: 0 0 8px;
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.empty p,
.pending p {
  max-width: 340px;
  margin: 0;
  color: var(--ink-2);
  font-size: 14px;
  line-height: 1.55;
}

.pending {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.pending p {
  justify-self: center;
  margin-top: 8px;
  text-align: center;
}

.pending__card {
  display: grid;
  gap: 12px;
  padding: 18px;
}

.pending__line {
  height: 8px;
  border-radius: 999px;
  background: linear-gradient(
      90deg,
      transparent,
      color-mix(in srgb, var(--teal) 30%, transparent),
      transparent
    )
    0 0 / 50% 100% no-repeat,
    var(--fill-strong);
  animation: shimmer 1.4s ease-in-out infinite;
}

.pending__line--title {
  width: 55%;
  height: 12px;
}

.pending__line--short {
  width: 38%;
}

@keyframes shimmer {
  from {
    background-position: -100% 0, 0 0;
  }

  to {
    background-position: 200% 0, 0 0;
  }
}

.island {
  position: fixed;
  z-index: 30;
  bottom: 22px;
  left: 50%;
  display: flex;
  max-width: calc(100vw - 24px);
  align-items: center;
  gap: 6px;
  padding: 6px;
  border: 1px solid var(--edge);
  border-radius: 999px;
  background: var(--panel);
  box-shadow: 0 14px 36px var(--shadow-near);
  backdrop-filter: blur(20px) saturate(1.4);
  translate: -50% 0;
}

.island__status {
  position: relative;
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 9px;
  padding: 0 14px 0 12px;
  color: var(--ink-2);
  font-size: 13px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.island__text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.island__dot {
  width: 8px;
  height: 8px;
  flex: none;
  border-radius: 50%;
  background: color-mix(in srgb, var(--ink) 30%, transparent);
}

.is-ready .island__dot {
  background: var(--green);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--green) 18%, transparent);
}

.is-busy .island__dot {
  background: var(--sky);
  animation: pulse 1.1s ease-in-out infinite;
}

.is-unavailable .island__dot {
  background: var(--red);
}

.island__progress {
  position: absolute;
  right: 14px;
  bottom: -5px;
  left: 12px;
  height: 2px;
  border-radius: 1px;
  background: linear-gradient(var(--sky), var(--sky)) 0 0 / calc(var(--p) * 100%) 100% no-repeat,
    var(--fill-strong);
}

.island__button {
  display: inline-flex;
  height: 40px;
  flex: none;
  align-items: center;
  gap: 7px;
  padding: 0 18px;
  border: 1px solid var(--edge);
  border-radius: 20px;
  color: var(--ink);
  background: transparent;
  font-size: 14px;
  font-weight: 650;
  white-space: nowrap;
  transition:
    background 0.16s,
    transform 0.16s;
}

.island__button:hover:not(:disabled) {
  background: var(--fill-strong);
}

.island__button--solid {
  min-width: 128px;
  justify-content: center;
  border-color: transparent;
  color: var(--bg);
  background: var(--ink);
}

.island__button--solid:hover:not(:disabled) {
  background: color-mix(in srgb, var(--ink) 86%, var(--bg));
  transform: translateY(-1px);
}

.island__button svg {
  fill: currentColor;
  stroke: none;
}

@keyframes pulse {
  50% {
    opacity: 0.35;
  }
}

.reveal {
  opacity: 0;
  filter: blur(4px);
  transform: translateY(14px);
}

.is-shown .reveal {
  opacity: 1;
  filter: none;
  transform: none;
  transition:
    opacity 0.9s var(--ease) calc(var(--line, 0) * 90ms),
    filter 0.9s var(--ease) calc(var(--line, 0) * 90ms),
    transform 1.05s var(--ease) calc(var(--line, 0) * 90ms);
}

@media (max-width: 980px) {
  .workspace {
    width: min(760px, calc(100% - 32px));
    grid-template-columns: minmax(0, 1fr);
  }

  .workspace__result {
    position: static;
    min-height: 360px;
    max-height: none;
  }
}

@media (max-width: 620px) {
  .hero {
    padding: 32px 18px 28px;
  }

  .hero__facts li {
    height: 28px;
    padding: 0 11px;
    font-size: 12px;
  }

  .workspace {
    width: calc(100% - 20px);
    gap: 16px;
  }

  .editor {
    padding: 14px 10px;
  }

  .island {
    right: 10px;
    bottom: max(10px, env(safe-area-inset-bottom));
    left: 10px;
    translate: none;
  }

  .island__status {
    flex: 1;
    padding: 0 6px 0 10px;
    font-size: 12px;
  }

  .island__button {
    padding: 0 14px;
  }

  .island__button--solid {
    min-width: 0;
  }

  .island__button:not(.island__button--solid) {
    display: none;
  }
}
</style>
