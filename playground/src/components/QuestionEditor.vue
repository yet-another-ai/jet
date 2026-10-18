<script setup vapor lang="ts">
import { draftId, type QuestionDraft } from '../playground';

defineProps<{
  question: QuestionDraft;
  index: number;
  disabled: boolean;
  canRemove: boolean;
}>();

defineEmits<{ remove: [] }>();
</script>

<template>
  <article class="question card">
    <div class="question__head">
      <span class="key">question_{{ index + 1 }}</span>
      <button
        v-if="canRemove"
        class="icon-button"
        type="button"
        :disabled="disabled"
        :aria-label="`Remove question ${index + 1}`"
        @click="$emit('remove')"
      >
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
      </button>
    </div>
    <label class="visually-hidden" :for="question.id">Question {{ index + 1 }}</label>
    <input
      :id="question.id"
      v-model="question.text"
      class="question__input field"
      type="text"
      :disabled="disabled"
      placeholder="What should Jet decide?"
    />

    <ul class="question__options">
      <li v-for="(option, optionIndex) in question.options" :key="option.id" class="option">
        <span class="option__letter" aria-hidden="true">{{ String.fromCharCode(65 + optionIndex) }}</span>
        <label class="visually-hidden" :for="option.id">
          Option {{ optionIndex + 1 }} for question {{ index + 1 }}
        </label>
        <input
          :id="option.id"
          v-model="option.text"
          class="option__input field"
          type="text"
          :disabled="disabled"
          :placeholder="`Option ${optionIndex + 1}`"
        />
        <button
          class="icon-button"
          type="button"
          :disabled="disabled || question.options.length <= 2"
          :aria-label="`Remove option ${optionIndex + 1}`"
          @click="question.options.splice(optionIndex, 1)"
        >
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
        </button>
      </li>
    </ul>

    <button
      class="question__add"
      type="button"
      :disabled="disabled"
      @click="question.options.push({ id: draftId(), text: '' })"
    >
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v10M3 8h10" /></svg>
      Add option
    </button>
  </article>
</template>

<style scoped>
.question {
  padding: 12px 12px 10px;
  transition: border-color 0.2s;
}

.question:focus-within {
  border-color: color-mix(in srgb, var(--teal) 45%, var(--edge));
}

.question__head {
  display: flex;
  height: 26px;
  align-items: center;
  justify-content: space-between;
  margin: -2px 0 6px 4px;
}

.question__input {
  height: 40px;
  padding: 0 14px;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 650;
}

.question__options {
  display: grid;
  gap: 6px;
  margin: 8px 0 6px;
  padding: 0;
  list-style: none;
}

.option {
  display: grid;
  grid-template-columns: 26px minmax(0, 1fr) 28px;
  align-items: center;
  gap: 8px;
  padding-left: 4px;
}

.option__letter {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border-radius: 7px;
  color: var(--ink-2);
  background: var(--fill);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 650;
}

.option__input {
  height: 34px;
  padding: 0 12px;
  border-radius: 9px;
  font-size: 13.5px;
}

.icon-button {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  padding: 0;
  border: 0;
  border-radius: 8px;
  color: var(--ink-2);
  background: transparent;
}

.icon-button:hover:not(:disabled) {
  color: var(--red);
  background: color-mix(in srgb, var(--red) 10%, transparent);
}

.icon-button:disabled {
  opacity: 0.3;
}

.question__add {
  display: inline-flex;
  height: 30px;
  align-items: center;
  gap: 6px;
  margin-left: 38px;
  padding: 0 10px;
  border: 0;
  border-radius: 8px;
  color: var(--teal-ink);
  background: transparent;
  font-size: 12.5px;
  font-weight: 650;
}

.question__add:hover:not(:disabled) {
  background: color-mix(in srgb, var(--teal) 10%, transparent);
}
</style>
