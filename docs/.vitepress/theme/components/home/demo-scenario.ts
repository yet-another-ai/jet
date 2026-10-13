/**
 * Shared hero/showcase demo state. Scores and the highlighted answer stay in
 * lockstep so the winning candidate always carries the highest probability.
 */

export interface DemoCandidate {
  key: string
  label: string
  /** Softmax probability in [0, 1]; candidates for a question sum to 1. */
  probability: number
  color: string
}

export interface DemoQuestion {
  name: string
  answer: string
  tone: string
  candidates: DemoCandidate[]
}

/** Seconds spent resting on each question before advancing. */
export const DEMO_QUESTION_SECONDS = 3.2

export const demoQuestions: DemoQuestion[] = [
  {
    name: "urgent",
    answer: "true",
    tone: "var(--home-amber)",
    candidates: [
      { key: "true", label: "true", probability: 0.87, color: "var(--home-amber)" },
      { key: "false", label: "false", probability: 0.13, color: "var(--home-ink-2)" }
    ]
  },
  {
    name: "owner",
    answer: '"app"',
    tone: "var(--home-cyan)",
    candidates: [
      {
        key: "app",
        label: "Application team",
        probability: 0.72,
        color: "var(--home-cyan)"
      },
      {
        key: "infra",
        label: "Infrastructure team",
        probability: 0.28,
        color: "var(--home-violet)"
      }
    ]
  },
  {
    name: "severity",
    answer: "2",
    tone: "var(--home-violet)",
    candidates: [
      { key: "0", label: "low", probability: 0.06, color: "var(--home-green)" },
      { key: "1", label: "medium", probability: 0.18, color: "var(--home-amber)" },
      { key: "2", label: "high", probability: 0.76, color: "var(--home-violet)" }
    ]
  }
]

export function demoPhase(time: number): number {
  return Math.floor(time / DEMO_QUESTION_SECONDS) % demoQuestions.length
}

export function demoQuestion(time: number): DemoQuestion {
  return demoQuestions[demoPhase(time)]!
}

/** Soft wobble that never lets a runner-up overtake the winner. */
export function animatedProbabilities(time: number, question: DemoQuestion): number[] {
  const winner = question.candidates.reduce(
    (best, candidate, index, list) =>
      candidate.probability > list[best]!.probability ? index : best,
    0
  )
  const raw = question.candidates.map((candidate, index) => {
    const wobble = 0.018 * Math.sin(time * 0.85 + index * 1.3)
    return Math.max(0.02, candidate.probability + wobble)
  })
  // Keep the chosen answer strictly ahead of every other candidate.
  const winnerFloor = Math.max(...raw.filter((_, index) => index !== winner)) + 0.04
  raw[winner] = Math.max(raw[winner]!, winnerFloor, question.candidates[winner]!.probability - 0.02)
  const total = raw.reduce((sum, value) => sum + value, 0)
  return raw.map((value) => value / total)
}
