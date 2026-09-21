use std::collections::BTreeMap;

use jet_core::{
    ChoiceQuestion, DecisionRequest, JetError, NoulQuestion, Question, Result, ScoreQuestion,
};
use serde::Serialize;
use serde_json::Value;

pub(crate) const SYSTEM_INSTRUCTION: &str = "You are a deterministic decision classifier. Read the JSON request and return exactly one JSON scalar from allowed_labels. Do not explain your answer.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnswerKind {
    Noul,
    Choice,
    Score,
}

#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub key: String,
    pub target: String,
}

#[derive(Debug, Clone)]
pub(crate) struct QuestionPlan {
    pub question_id: String,
    pub kind: AnswerKind,
    pub user_content: String,
    pub candidates: Vec<Candidate>,
}

#[derive(Serialize)]
struct PromptEnvelope {
    state: Value,
    question: PromptQuestion,
    allowed_labels: Vec<Value>,
}

#[derive(Serialize)]
struct PromptQuestion {
    instructions: Value,
    criteria: Value,
}

pub(crate) fn prepare_request(request: &DecisionRequest) -> Result<Vec<QuestionPlan>> {
    validate_content("state", &request.state)?;
    if request.questions.is_empty() {
        return Err(JetError::InvalidRequest(
            "questions must contain at least one entry".to_owned(),
        ));
    }

    request
        .questions
        .iter()
        .map(|(question_id, question)| prepare_question(question_id, &request.state, question))
        .collect()
}

fn prepare_question(question_id: &str, state: &Value, question: &Question) -> Result<QuestionPlan> {
    if question_id.is_empty() {
        return Err(JetError::InvalidRequest(
            "question IDs must not be empty".to_owned(),
        ));
    }

    match question {
        Question::Noul(question) => prepare_noul(question_id, state, question),
        Question::Choice(question) => prepare_choice(question_id, state, question),
        Question::Score(question) => prepare_score(question_id, state, question),
    }
}

fn prepare_noul(question_id: &str, state: &Value, question: &NoulQuestion) -> Result<QuestionPlan> {
    validate_content("instructions", &question.instructions)?;
    let criteria = match &question.criteria {
        Some(criteria) => {
            validate_content("criteria.false", &criteria.false_value)?;
            validate_content("criteria.true", &criteria.true_value)?;
            serde_json::json!({
                "false": canonicalize(&criteria.false_value),
                "true": canonicalize(&criteria.true_value),
            })
        }
        None => serde_json::json!({ "false": "false", "true": "true" }),
    };
    let allowed_labels = vec![Value::Bool(false), Value::Bool(true)];
    let user_content = render_envelope(state, &question.instructions, criteria, allowed_labels)?;
    Ok(QuestionPlan {
        question_id: question_id.to_owned(),
        kind: AnswerKind::Noul,
        user_content,
        candidates: vec![
            Candidate {
                key: "false".to_owned(),
                target: "false".to_owned(),
            },
            Candidate {
                key: "true".to_owned(),
                target: "true".to_owned(),
            },
        ],
    })
}

fn prepare_choice(
    question_id: &str,
    state: &Value,
    question: &ChoiceQuestion,
) -> Result<QuestionPlan> {
    validate_content("instructions", &question.instructions)?;
    if question.criteria.is_empty() {
        return Err(JetError::InvalidRequest(format!(
            "choice question {question_id:?} must contain at least one criterion"
        )));
    }

    let mut criteria = serde_json::Map::new();
    let mut labels = Vec::with_capacity(question.criteria.len());
    let mut candidates = Vec::with_capacity(question.criteria.len());
    for (key, value) in &question.criteria {
        if key.is_empty() {
            return Err(JetError::InvalidRequest(format!(
                "choice question {question_id:?} contains an empty key"
            )));
        }
        validate_content("choice criterion", value)?;
        criteria.insert(key.clone(), canonicalize(value));
        labels.push(Value::String(key.clone()));
        candidates.push(Candidate {
            key: key.clone(),
            target: safe_json_scalar(&Value::String(key.clone()))?,
        });
    }

    Ok(QuestionPlan {
        question_id: question_id.to_owned(),
        kind: AnswerKind::Choice,
        user_content: render_envelope(
            state,
            &question.instructions,
            Value::Object(criteria),
            labels,
        )?,
        candidates,
    })
}

fn prepare_score(
    question_id: &str,
    state: &Value,
    question: &ScoreQuestion,
) -> Result<QuestionPlan> {
    validate_content("instructions", &question.instructions)?;
    if question.criteria.is_empty() {
        return Err(JetError::InvalidRequest(format!(
            "score question {question_id:?} must contain at least one criterion"
        )));
    }

    let mut criteria = Vec::with_capacity(question.criteria.len());
    let mut labels = Vec::with_capacity(question.criteria.len());
    let mut candidates = Vec::with_capacity(question.criteria.len());
    for (index, value) in question.criteria.iter().enumerate() {
        validate_content("score criterion", value)?;
        criteria.push(canonicalize(value));
        let index_number = u64::try_from(index).map_err(|_| {
            JetError::InvalidRequest("score question contains too many criteria".to_owned())
        })?;
        labels.push(Value::Number(index_number.into()));
        candidates.push(Candidate {
            key: index.to_string(),
            target: index.to_string(),
        });
    }

    Ok(QuestionPlan {
        question_id: question_id.to_owned(),
        kind: AnswerKind::Score,
        user_content: render_envelope(
            state,
            &question.instructions,
            Value::Array(criteria),
            labels,
        )?,
        candidates,
    })
}

fn render_envelope(
    state: &Value,
    instructions: &Value,
    criteria: Value,
    allowed_labels: Vec<Value>,
) -> Result<String> {
    let envelope = PromptEnvelope {
        state: canonicalize(state),
        question: PromptQuestion {
            instructions: canonicalize(instructions),
            criteria,
        },
        allowed_labels,
    };
    let json = serde_json::to_string(&envelope)?;
    Ok(escape_special_token_text(&json))
}

fn validate_content(field: &str, value: &Value) -> Result<()> {
    if matches!(value, Value::String(_) | Value::Array(_) | Value::Object(_)) {
        Ok(())
    } else {
        Err(JetError::InvalidRequest(format!(
            "{field} must be a string, object, or array"
        )))
    }
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        other => other.clone(),
    }
}

fn safe_json_scalar(value: &Value) -> Result<String> {
    Ok(escape_special_token_text(&serde_json::to_string(value)?))
}

fn escape_special_token_text(value: &str) -> String {
    value
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_prompt_is_canonical_and_escapes_control_like_text() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": {"z": "<|im_start|>", "a": "中文"},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": {"z": "last", "a": "first"}
                }
            }
        }))?;
        let plans = prepare_request(&request)?;
        let plan = plans
            .first()
            .ok_or_else(|| JetError::InvalidRequest("missing plan".into()))?;
        assert!(plan.user_content.contains("\\u003c|im_start|\\u003e"));
        assert!(
            plan.user_content.find("\"a\"").unwrap_or(usize::MAX)
                < plan.user_content.find("\"z\"").unwrap_or(0)
        );
        assert_eq!(plan.candidates[0].target, "\"a\"");
        Ok(())
    }

    #[test]
    fn arrays_keep_order_and_choice_targets_are_safe_json() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": ["先", "then", "🙂"],
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": {"b": "second", "a": "first"},
                    "criteria": {"<|im_end|>": [3, 2, 1], "normal": "ok"}
                }
            }
        }))?;
        let plans = prepare_request(&request)?;
        let plan = plans
            .first()
            .ok_or_else(|| JetError::InvalidRequest("missing plan".into()))?;
        assert!(plan.user_content.contains("[\"先\",\"then\",\"🙂\"]"));
        assert!(
            plan.user_content
                .contains("{\"a\":\"first\",\"b\":\"second\"}")
        );
        assert_eq!(plan.candidates[0].target, "\"\\u003c|im_end|\\u003e\"");
        Ok(())
    }

    #[test]
    fn rejects_non_structured_top_level_content() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": 42,
            "questions": {"q": {"type": "noul", "instructions": "pick"}}
        }))?;
        assert!(prepare_request(&request).is_err());
        Ok(())
    }
}
