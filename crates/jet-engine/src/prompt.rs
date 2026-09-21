use std::collections::{BTreeMap, BTreeSet};

use jet_core::{
    ChoiceQuestion, DecisionRequest, JetError, NoulQuestion, Question, Result, ScoreQuestion,
};
use serde::Serialize;
use serde_json::Value;
use tera::{Context, Tera};

pub(crate) const SYSTEM_INSTRUCTION: &str = "You are a careful decision classifier. Choose by meaning, not by candidate position. Reply with exactly one quoted candidate as listed, with no explanation, whitespace, or extra text.";

const QUESTION_TEMPLATE: &str = r#"Context:
{{ state }}

Instruction:
{{ instructions }}

Candidate answers (choose one exactly):
{{ candidates | join(sep=", ") }}

Required output format:
One quoted candidate exactly as listed above; no explanation, whitespace, or extra text.
"#;

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
struct PromptContext {
    state: String,
    instructions: String,
    candidates: Vec<String>,
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
    let default_false = Value::String("No".to_owned());
    let default_true = Value::String("Yes".to_owned());
    let (false_value, true_value) = match &question.criteria {
        Some(criteria) => {
            validate_content("criteria.false", &criteria.false_value)?;
            validate_content("criteria.true", &criteria.true_value)?;
            (&criteria.false_value, &criteria.true_value)
        }
        None => (&default_false, &default_true),
    };
    let candidates = vec![
        semantic_candidate("false", false_value)?,
        semantic_candidate("true", true_value)?,
    ];
    finish_plan(
        question_id,
        AnswerKind::Noul,
        state,
        &question.instructions,
        candidates,
    )
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

    let mut candidates = Vec::with_capacity(question.criteria.len());
    for (key, value) in &question.criteria {
        if key.is_empty() {
            return Err(JetError::InvalidRequest(format!(
                "choice question {question_id:?} contains an empty key"
            )));
        }
        validate_content("choice criterion", value)?;
        candidates.push(semantic_candidate(key, value)?);
    }

    finish_plan(
        question_id,
        AnswerKind::Choice,
        state,
        &question.instructions,
        candidates,
    )
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

    let candidates = question
        .criteria
        .iter()
        .enumerate()
        .map(|(index, value)| {
            validate_content("score criterion", value)?;
            semantic_candidate(&index.to_string(), value)
        })
        .collect::<Result<Vec<_>>>()?;
    finish_plan(
        question_id,
        AnswerKind::Score,
        state,
        &question.instructions,
        candidates,
    )
}

fn semantic_candidate(key: &str, value: &Value) -> Result<Candidate> {
    let semantic_text = render_value(value);
    if semantic_text.trim().is_empty() {
        return Err(JetError::InvalidRequest(format!(
            "candidate {key:?} renders to empty text"
        )));
    }
    let target = escape_special_token_text(&serde_json::to_string(&semantic_text)?);
    Ok(Candidate {
        key: key.to_owned(),
        target,
    })
}

fn finish_plan(
    question_id: &str,
    kind: AnswerKind,
    state: &Value,
    instructions: &Value,
    candidates: Vec<Candidate>,
) -> Result<QuestionPlan> {
    let mut targets = BTreeSet::new();
    for candidate in &candidates {
        if !targets.insert(candidate.target.clone()) {
            return Err(JetError::InvalidRequest(format!(
                "question {question_id:?} contains duplicate rendered candidate {:?}",
                candidate.target
            )));
        }
    }
    let prompt = PromptContext {
        state: escape_special_token_text(&render_value(state)),
        instructions: escape_special_token_text(&render_value(instructions)),
        candidates: candidates
            .iter()
            .map(|candidate| candidate.target.clone())
            .collect(),
    };
    let context = Context::from_serialize(&prompt).map_err(template_error)?;
    let user_content = Tera::one_off(QUESTION_TEMPLATE, &context, false).map_err(template_error)?;
    Ok(QuestionPlan {
        question_id: question_id.to_owned(),
        kind,
        user_content,
        candidates,
    })
}

fn render_value(value: &Value) -> String {
    let mut output = String::new();
    render_value_at(value, 0, &mut output);
    output
}

fn render_value_at(value: &Value, indent: usize, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => output.push_str(value),
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push('\n');
                }
                output.push_str(&" ".repeat(indent));
                output.push_str("- ");
                if matches!(value, Value::Array(_) | Value::Object(_)) {
                    output.push('\n');
                    render_value_at(value, indent + 2, output);
                } else {
                    render_value_at(value, indent + 2, output);
                }
            }
        }
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values.iter().collect();
            for (index, (key, value)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    output.push('\n');
                }
                output.push_str(&" ".repeat(indent));
                output.push_str(key);
                output.push(':');
                if matches!(value, Value::Array(_) | Value::Object(_)) {
                    output.push('\n');
                    render_value_at(value, indent + 2, output);
                } else {
                    output.push(' ');
                    render_value_at(value, indent + 2, output);
                }
            }
        }
    }
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

fn escape_special_token_text(value: &str) -> String {
    value
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}

fn template_error(error: tera::Error) -> JetError {
    JetError::NativeRuntime(format!("prompt template rendering failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_prompt_is_human_readable_and_targets_semantics() -> Result<()> {
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
        assert!(plan.user_content.contains("Context:\na: 中文"));
        assert!(plan.user_content.contains("z: \\u003c|im_start|\\u003e"));
        assert!(!plan.user_content.contains("allowed_labels"));
        assert_eq!(plan.candidates[0].key, "a");
        assert_eq!(plan.candidates[0].target, "\"first\"");
        assert_eq!(plan.candidates[1].key, "z");
        assert_eq!(plan.candidates[1].target, "\"last\"");
        Ok(())
    }

    #[test]
    fn structured_values_keep_order_and_special_tokens_are_safe() -> Result<()> {
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
        assert!(plan.user_content.contains("- 先\n- then\n- 🙂"));
        assert!(plan.user_content.contains("a: first\nb: second"));
        assert_eq!(plan.candidates[0].key, "<|im_end|>");
        assert_eq!(plan.candidates[0].target, "\"- 3\\n- 2\\n- 1\"");
        Ok(())
    }

    #[test]
    fn noul_defaults_to_semantic_yes_and_no() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": "The sky is blue.",
            "questions": {"q": {"type": "noul", "instructions": "Is the sky blue?"}}
        }))?;
        let plan = prepare_request(&request)?
            .into_iter()
            .next()
            .ok_or_else(|| JetError::InvalidRequest("missing plan".into()))?;
        assert_eq!(plan.candidates[0].target, "\"No\"");
        assert_eq!(plan.candidates[1].target, "\"Yes\"");
        assert!(plan.user_content.contains("\"No\", \"Yes\""));
        assert!(plan.user_content.contains(
            "One quoted candidate exactly as listed above; no explanation, whitespace, or extra \
             text."
        ));
        assert!(!plan.user_content.contains("Answer:"));
        assert!(plan.user_content.ends_with("extra text.\n"));
        Ok(())
    }

    #[test]
    fn duplicate_semantic_candidates_are_rejected() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": "x",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {"a": "same", "b": "same"}
                }
            }
        }))?;
        assert!(prepare_request(&request).is_err());
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
