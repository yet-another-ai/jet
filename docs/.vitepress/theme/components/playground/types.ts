export type OptionDraft = {
  id: string;
  text: string;
};

export type QuestionDraft = {
  id: string;
  text: string;
  options: OptionDraft[];
};

export type ChoiceAnswer = {
  type: 'choice';
  choice: string;
  probabilities: Record<string, number>;
};

export type DecisionResponse = {
  model: string;
  answers: Record<string, ChoiceAnswer>;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
};

export type ResultOption = {
  id: string;
  label: string;
  probability: number;
  winner: boolean;
};

export type ResultGroup = {
  id: string;
  question: string;
  options: ResultOption[];
};

let lastDraftId = 0;

export function draftId(): string {
  return `draft-${++lastDraftId}`;
}
