use std::collections::BTreeMap;

use jet_core::{
    Answer, ChoiceAnswer, DecisionRequest, DecisionResponse, JetError, NoulAnswer, Result,
    ScoreAnswer, Usage, normalize_log_probabilities,
};

use crate::prompt::{AnswerKind, QuestionPlan, SYSTEM_INSTRUCTION, prepare_request};

#[derive(Debug, Clone)]
pub(crate) struct ScoreJob {
    pub system_content: &'static str,
    pub user_content: String,
    pub targets: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScoreResult {
    pub log_probabilities: Vec<f64>,
    pub prompt_tokens: usize,
    pub thinking_tokens: usize,
    pub target_token_counts: Vec<usize>,
    #[allow(dead_code)]
    pub prefill_count: usize,
}

pub(crate) trait SequenceScorer {
    fn score_batch(&mut self, jobs: &[ScoreJob]) -> Vec<Result<ScoreResult>>;
}

pub(crate) fn decide_batch<S: SequenceScorer>(
    scorer: &mut S,
    model_id: &str,
    requests: &[DecisionRequest],
) -> Vec<Result<DecisionResponse>> {
    let mut slots: Vec<Option<Result<DecisionResponse>>> =
        (0..requests.len()).map(|_| None).collect();
    let mut prepared = Vec::new();
    let mut jobs = Vec::new();

    for (request_index, request) in requests.iter().enumerate() {
        match prepare_request(request) {
            Ok(plans) => {
                let job_start = jobs.len();
                jobs.extend(plans.iter().map(plan_to_job));
                prepared.push((request_index, plans, job_start));
            }
            Err(error) => slots[request_index] = Some(Err(error)),
        }
    }

    if !jobs.is_empty() {
        let scored = scorer.score_batch(&jobs);
        if scored.len() == jobs.len() {
            let mut scored = scored.into_iter();
            for (request_index, plans, _job_start) in prepared {
                let request_scores = scored
                    .by_ref()
                    .take(plans.len())
                    .collect::<Result<Vec<_>>>();
                slots[request_index] = Some(
                    request_scores.and_then(|scores| build_response(model_id, &plans, &scores)),
                );
            }
        } else {
            let message = format!(
                "scorer returned {} results for {} jobs",
                scored.len(),
                jobs.len()
            );
            fill_scoring_errors(&mut slots, &message);
        }
    }

    slots
        .into_iter()
        .map(|slot| {
            slot.unwrap_or_else(|| {
                Err(JetError::NativeRuntime(
                    "request was not completed by the batch evaluator".to_owned(),
                ))
            })
        })
        .collect()
}

fn fill_scoring_errors(slots: &mut [Option<Result<DecisionResponse>>], message: &str) {
    for slot in slots.iter_mut().filter(|slot| slot.is_none()) {
        *slot = Some(Err(JetError::NativeRuntime(format!(
            "batch scoring failed: {message}"
        ))));
    }
}

pub(crate) fn plan_to_job(plan: &QuestionPlan) -> ScoreJob {
    ScoreJob {
        system_content: SYSTEM_INSTRUCTION,
        user_content: plan.user_content.clone(),
        targets: plan
            .candidates
            .iter()
            .map(|candidate| candidate.target.clone())
            .collect(),
    }
}

pub(crate) fn build_response(
    model_id: &str,
    plans: &[QuestionPlan],
    scores: &[ScoreResult],
) -> Result<DecisionResponse> {
    let mut answers = BTreeMap::new();
    let mut usage = Usage::default();

    for (plan, score) in plans.iter().zip(scores) {
        if score.log_probabilities.len() != plan.candidates.len()
            || score.target_token_counts.len() != plan.candidates.len()
        {
            return Err(JetError::NativeRuntime(format!(
                "question {:?} received a malformed scorer result",
                plan.question_id
            )));
        }
        let normalized = normalize_log_probabilities(&score.log_probabilities)?;
        let probabilities: BTreeMap<String, f64> = plan
            .candidates
            .iter()
            .zip(&normalized)
            .map(|(candidate, score)| (candidate.key.clone(), score.normalized_probability))
            .collect();

        let answer = match plan.kind {
            AnswerKind::Noul => Answer::Noul(NoulAnswer {
                noul: probabilities.get("true").copied().ok_or_else(|| {
                    JetError::NativeRuntime("noul result is missing the true candidate".to_owned())
                })?,
            }),
            AnswerKind::Choice => {
                let choice = probabilities
                    .iter()
                    .max_by(|left, right| {
                        left.1.total_cmp(right.1).then_with(|| right.0.cmp(left.0))
                    })
                    .map(|(key, _)| key.clone())
                    .ok_or_else(|| {
                        JetError::NativeRuntime("choice result has no candidates".to_owned())
                    })?;
                Answer::Choice(ChoiceAnswer {
                    choice,
                    probabilities,
                })
            }
            AnswerKind::Score => {
                let expected_score = probabilities.iter().try_fold(0.0, |sum, (key, value)| {
                    let index = key.parse::<usize>().map_err(|error| {
                        JetError::NativeRuntime(format!("invalid score index {key:?}: {error}"))
                    })?;
                    Ok::<_, JetError>(sum + index as f64 * value)
                })?;
                Answer::Score(ScoreAnswer {
                    score: expected_score,
                    probabilities,
                })
            }
        };

        usage.input_tokens = usage
            .input_tokens
            .saturating_add(u64::try_from(score.prompt_tokens).unwrap_or(u64::MAX));
        usage.output_tokens = usage.output_tokens.saturating_add(
            u64::try_from(score.thinking_tokens)
                .unwrap_or(u64::MAX)
                .saturating_add(
                    score
                        .target_token_counts
                        .iter()
                        .map(|count| u64::try_from(*count).unwrap_or(u64::MAX))
                        .sum(),
                ),
        );
        answers.insert(plan.question_id.clone(), answer);
    }

    Ok(DecisionResponse {
        model: model_id.to_owned(),
        answers,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockScorer;

    impl SequenceScorer for MockScorer {
        fn score_batch(&mut self, jobs: &[ScoreJob]) -> Vec<Result<ScoreResult>> {
            jobs.iter()
                .map(|job| {
                    Ok(ScoreResult {
                        log_probabilities: (0..job.targets.len())
                            .map(|index| -(index as f64))
                            .collect(),
                        prompt_tokens: 10,
                        thinking_tokens: 0,
                        target_token_counts: vec![1; job.targets.len()],
                        prefill_count: 1,
                    })
                })
                .collect()
        }
    }

    #[test]
    fn maps_all_decisions_shapes() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": {"ticket": "broken"},
            "questions": {
                "boolean": {"type": "noul", "instructions": "Is it broken?"},
                "choice": {
                    "type": "choice",
                    "instructions": "Pick a team",
                    "criteria": {"a": "Alpha", "b": "Beta"}
                },
                "score": {
                    "type": "score",
                    "instructions": "Rate it",
                    "criteria": ["low", "high"]
                }
            }
        }))?;
        let results = decide_batch(&mut MockScorer, "test-model", &[request]);
        let response = results
            .into_iter()
            .next()
            .ok_or_else(|| JetError::NativeRuntime("missing response".into()))??;
        assert_eq!(response.model, "test-model");
        assert_eq!(response.usage.input_tokens, 30);
        assert_eq!(response.usage.output_tokens, 6);
        match response.answers.get("score") {
            Some(Answer::Score(answer)) => {
                let expected = 1.0 / (1.0 + 1.0_f64.exp());
                assert!((answer.score - expected).abs() < 1e-12);
            }
            _ => return Err(JetError::NativeRuntime("missing score answer".into())),
        }
        Ok(())
    }

    #[test]
    fn ties_use_lexicographically_first_choice_key() -> Result<()> {
        struct TieScorer;
        impl SequenceScorer for TieScorer {
            fn score_batch(&mut self, jobs: &[ScoreJob]) -> Vec<Result<ScoreResult>> {
                jobs.iter()
                    .map(|job| {
                        Ok(ScoreResult {
                            log_probabilities: vec![0.0; job.targets.len()],
                            prompt_tokens: 1,
                            thinking_tokens: 0,
                            target_token_counts: vec![1; job.targets.len()],
                            prefill_count: 1,
                        })
                    })
                    .collect()
            }
        }
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"z": "z", "a": "a"}}}
        }))?;
        let response = decide_batch(&mut TieScorer, "m", &[request])
            .into_iter()
            .next()
            .ok_or_else(|| JetError::NativeRuntime("missing response".into()))??;
        match response.answers.get("q") {
            Some(Answer::Choice(answer)) => assert_eq!(answer.choice, "a"),
            _ => return Err(JetError::NativeRuntime("missing choice answer".into())),
        }
        Ok(())
    }

    #[test]
    fn a_failed_question_fails_its_request_without_hiding_other_requests() -> Result<()> {
        struct PartialScorer;
        impl SequenceScorer for PartialScorer {
            fn score_batch(&mut self, jobs: &[ScoreJob]) -> Vec<Result<ScoreResult>> {
                jobs.iter()
                    .enumerate()
                    .map(|(index, job)| {
                        if index == 0 {
                            Err(JetError::ContextExceeded {
                                required: 11,
                                limit: 10,
                            })
                        } else {
                            Ok(ScoreResult {
                                log_probabilities: vec![0.0; job.targets.len()],
                                prompt_tokens: 2,
                                thinking_tokens: 0,
                                target_token_counts: vec![1; job.targets.len()],
                                prefill_count: 1,
                            })
                        }
                    })
                    .collect()
            }
        }

        let first: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": "first",
            "questions": {"q": {"type": "noul", "instructions": "pick"}}
        }))?;
        let second: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": "second",
            "questions": {"q": {"type": "noul", "instructions": "pick"}}
        }))?;
        let responses = decide_batch(&mut PartialScorer, "m", &[first, second]);
        assert!(matches!(
            responses.first(),
            Some(Err(JetError::ContextExceeded { .. }))
        ));
        assert!(matches!(responses.get(1), Some(Ok(_))));
        Ok(())
    }
}
