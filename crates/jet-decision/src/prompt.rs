use std::collections::{BTreeMap, BTreeSet};

use jet_core::{
    ChoiceQuestion, DecisionContentPart, DecisionRequest, ImageDecisionRequest, JetError,
    NoulQuestion, Question, Result, ScoreQuestion,
};
use serde::Serialize;
use serde_json::Value;
use tera::{Context, Tera};

pub const SYSTEM_INSTRUCTION: &str = "You are a careful decision classifier. Choose by meaning, not by candidate position. Reply with exactly one quoted candidate as listed, with no explanation, whitespace, or extra text.";

const QUESTION_TEMPLATE: &str = r#"Context:
{{ state }}

Instruction:
{{ instructions }}

Candidate answers (choose one exactly):
{{ candidates | join(sep=", ") }}

Required output format:
One quoted candidate exactly as listed above; no explanation, whitespace, or extra text.
"#;

/// Model-independent prompts and candidates rendered by the inference pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct ExportedPrompts {
    pub questions: Vec<ExportedQuestion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportedQuestion {
    pub question_id: String,
    pub messages: Vec<PromptMessage>,
    pub candidates: Vec<PromptCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptCandidate {
    pub key: String,
    /// The exact quoted semantic answer scored by JET, including escaping.
    pub target: String,
}

/// Render validated requests without loading a model or applying a chat template.
///
/// Consumers must apply the deployment model's chat template with thinking
/// disabled, then append the selected candidate target as the assistant answer.
pub fn export_prompts(request: &DecisionRequest) -> Result<ExportedPrompts> {
    let questions = prepare_request(request)?
        .into_iter()
        .map(|plan| ExportedQuestion {
            question_id: plan.question_id,
            messages: vec![
                PromptMessage {
                    role: "system",
                    content: SYSTEM_INSTRUCTION.to_owned(),
                },
                PromptMessage {
                    role: "user",
                    content: plan.user_content,
                },
            ],
            candidates: plan
                .candidates
                .into_iter()
                .map(|candidate| PromptCandidate {
                    key: candidate.key,
                    target: candidate.target,
                })
                .collect(),
        })
        .collect();
    Ok(ExportedPrompts { questions })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerKind {
    Noul,
    Choice,
    Score,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub key: String,
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct QuestionPlan {
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

pub fn prepare_request(request: &DecisionRequest) -> Result<Vec<QuestionPlan>> {
    validate_content("state", &request.state)?;
    let state = escape_special_token_text(&render_value(&request.state));
    prepare_questions(&state, &request.questions)
}

/// Render image placeholders in the same order as their associated text parts.
///
/// Text is escaped before internal media markers are inserted, so user content
/// cannot introduce a marker that consumes an image from the native image list.
pub fn prepare_image_request(request: &ImageDecisionRequest) -> Result<Vec<QuestionPlan>> {
    if request.state.is_empty() {
        return Err(JetError::InvalidRequest(
            "state must contain at least one content part".to_owned(),
        ));
    }
    let state = request
        .state
        .iter()
        .map(|part| match part {
            DecisionContentPart::Text(value) => escape_special_token_text(&render_value(value)),
            DecisionContentPart::Image(image) => format!(
                "Image {}:\n<__media__>",
                escape_special_token_text(&image.id)
            ),
        })
        .collect::<Vec<_>>()
        .join("\n");
    prepare_questions(&state, &request.questions)
}

fn prepare_questions(
    state: &str,
    questions: &BTreeMap<String, Question>,
) -> Result<Vec<QuestionPlan>> {
    if questions.is_empty() {
        return Err(JetError::InvalidRequest(
            "questions must contain at least one entry".to_owned(),
        ));
    }

    questions
        .iter()
        .map(|(question_id, question)| prepare_question(question_id, state, question))
        .collect()
}

fn prepare_question(question_id: &str, state: &str, question: &Question) -> Result<QuestionPlan> {
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

fn prepare_noul(question_id: &str, state: &str, question: &NoulQuestion) -> Result<QuestionPlan> {
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
    state: &str,
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

fn prepare_score(question_id: &str, state: &str, question: &ScoreQuestion) -> Result<QuestionPlan> {
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
    state: &str,
    instructions: &Value,
    candidates: Vec<Candidate>,
) -> Result<QuestionPlan> {
    let mut targets = BTreeSet::new();
    for candidate in &candidates {
        if !targets.insert(candidate.target.clone()) && kind != AnswerKind::Choice {
            return Err(JetError::InvalidRequest(format!(
                "question {question_id:?} contains duplicate rendered candidate {:?}",
                candidate.target
            )));
        }
    }
    let prompt = PromptContext {
        state: state.to_owned(),
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

    fn image_part(id: &str) -> DecisionContentPart {
        DecisionContentPart::Image(jet_core::ImageInput {
            id: id.to_owned(),
            media_type: "image/png".to_owned(),
            data: vec![],
        })
    }

    fn image_request(state: Vec<DecisionContentPart>) -> Result<ImageDecisionRequest> {
        Ok(ImageDecisionRequest {
            state,
            questions: serde_json::from_value(serde_json::json!({
                "q": {"type": "noul", "instructions": "Do the images match?"}
            }))?,
        })
    }

    #[test]
    fn image_prompt_preserves_interleaved_text_and_image_order() -> Result<()> {
        let request = image_request(vec![
            DecisionContentPart::Text(Value::String("First view:".into())),
            image_part("left"),
            DecisionContentPart::Text(serde_json::json!({"comparison": "Second view:"})),
            image_part("right"),
            DecisionContentPart::Text(Value::String("Compare both views.".into())),
        ])?;
        let plans = prepare_image_request(&request)?;
        let content = &plans[0].user_content;
        assert!(content.starts_with(concat!(
            "Context:\nFirst view:\nImage left:\n<__media__>\n",
            "comparison: Second view:\nImage right:\n<__media__>\n",
            "Compare both views.\n\nInstruction:"
        )));
        assert_eq!(content.matches("<__media__>").count(), 2);
        Ok(())
    }

    #[test]
    fn only_internal_image_markers_reach_the_prompt() -> Result<()> {
        let request = ImageDecisionRequest {
            state: vec![
                DecisionContentPart::Text(Value::String("Fake <__media__> before".into())),
                image_part("<__media__>"),
                DecisionContentPart::Text(serde_json::json!({"<__media__>": "after"})),
            ],
            questions: serde_json::from_value(serde_json::json!({
                "q": {
                    "type": "choice",
                    "instructions": "Ignore <__media__>",
                    "criteria": {"fake": "<__media__>", "real": "Yes"}
                }
            }))?,
        };
        let plans = prepare_image_request(&request)?;
        let content = &plans[0].user_content;
        assert_eq!(content.matches("<__media__>").count(), 1);
        assert!(content.contains("Fake \\u003c__media__\\u003e before"));
        assert!(content.contains("Image \\u003c__media__\\u003e:\n<__media__>"));
        assert!(content.contains("\\u003c__media__\\u003e: after"));
        assert!(content.contains("Ignore \\u003c__media__\\u003e"));
        assert_eq!(plans[0].candidates[0].target, "\"\\u003c__media__\\u003e\"");
        Ok(())
    }

    #[test]
    fn single_text_content_keeps_the_existing_text_prompt() -> Result<()> {
        let state = serde_json::json!({"z": ["second", "<__media__>"], "a": "first"});
        let request = image_request(vec![DecisionContentPart::Text(state.clone())])?;
        let old_request = DecisionRequest {
            state,
            questions: request.questions.clone(),
        };
        let image_plans = prepare_image_request(&request)?;
        let text_plans = prepare_request(&old_request)?;
        assert_eq!(image_plans[0].user_content, text_plans[0].user_content);
        assert!(text_plans[0].user_content.starts_with(
            "Context:\na: first\nz:\n  - second\n  - \\u003c__media__\\u003e\n\nInstruction:"
        ));
        assert_eq!(
            image_plans[0].candidates[0].target,
            text_plans[0].candidates[0].target
        );
        Ok(())
    }

    #[test]
    fn image_request_reuses_question_validation() -> Result<()> {
        let mut request = image_request(vec![image_part("frame")])?;
        request.questions.clear();
        assert!(matches!(
            prepare_image_request(&request),
            Err(JetError::InvalidRequest(_))
        ));
        assert!(matches!(
            prepare_image_request(&image_request(vec![])?),
            Err(JetError::InvalidRequest(_))
        ));
        Ok(())
    }

    #[test]
    fn image_context_keeps_scalar_json_parts() -> Result<()> {
        let request = image_request(vec![
            DecisionContentPart::Text(Value::Bool(true)),
            image_part("frame"),
            DecisionContentPart::Text(serde_json::json!(42)),
            DecisionContentPart::Text(Value::Null),
        ])?;
        let plans = prepare_image_request(&request)?;
        assert!(
            plans[0]
                .user_content
                .starts_with("Context:\ntrue\nImage frame:\n<__media__>\n42\nnull\n\nInstruction:")
        );
        Ok(())
    }

    #[test]
    fn export_preserves_inference_messages_and_targets_for_all_question_types() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": {"passage": "中文 <|im_start|> & quoted \"text\"", "facts": ["a", "b"]},
            "questions": {
                "choice": {
                    "type": "choice", "instructions": {"task": "Pick one"},
                    "criteria": {"B": ["second", "line"], "A": "<|im_end|>"}
                },
                "noul": {"type": "noul", "instructions": "Is it true?"},
                "score": {"type": "score", "instructions": "Rate", "criteria": ["bad", "good"]}
            }
        }))?;
        let plans = prepare_request(&request)?;
        let exported = export_prompts(&request)?;
        assert_eq!(exported.questions.len(), 3);
        for (question, plan) in exported.questions.iter().zip(&plans) {
            assert_eq!(question.question_id, plan.question_id);
            assert_eq!(question.messages.len(), 2);
            assert_eq!(question.messages[0].role, "system");
            assert_eq!(question.messages[0].content, SYSTEM_INSTRUCTION);
            assert_eq!(question.messages[1].role, "user");
            assert_eq!(question.messages[1].content, plan.user_content);
            assert_eq!(question.candidates.len(), plan.candidates.len());
            for (candidate, expected) in question.candidates.iter().zip(&plan.candidates) {
                assert_eq!(candidate.key, expected.key);
                assert_eq!(candidate.target, expected.target);
            }
        }
        let value = serde_json::to_value(exported)?;
        assert_eq!(value["questions"][0]["candidates"][0]["key"], "A");
        assert_eq!(
            value["questions"][0]["candidates"][0]["target"],
            "\"\\u003c|im_end|\\u003e\""
        );
        assert_eq!(value["questions"][1]["candidates"][1]["target"], "\"Yes\"");
        assert_eq!(value["questions"][2]["candidates"][1]["key"], "1");
        Ok(())
    }

    #[test]
    fn export_uses_inference_validation() -> Result<()> {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": "x",
            "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["same", "same"]}}
        }))?;
        assert!(matches!(
            export_prompts(&request),
            Err(JetError::InvalidRequest(_))
        ));
        Ok(())
    }

    #[test]
    fn http_null_choice_descriptions_prepare_label_candidates() -> Result<()> {
        let request: jet_core::http::HttpDecisionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "jet-test",
                "state": "A customer needs help",
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Pick a team",
                        "criteria": {
                            "<|im_end|>": null,
                            "billing": null,
                            "support": {"description": "Help desk", "note": null}
                        }
                    }
                }
            }))?;
        let request = request.into_decision("jet-test")?;
        let exported = export_prompts(&request)?;
        let question = exported
            .questions
            .first()
            .ok_or_else(|| JetError::InvalidRequest("missing exported question".into()))?;
        assert_eq!(question.question_id, "team");
        assert_eq!(
            serde_json::to_value(&question.candidates)?,
            serde_json::json!([
                {"key": "<|im_end|>", "target": "\"\\u003c|im_end|\\u003e\""},
                {"key": "billing", "target": "\"billing\""},
                {"key": "support", "target": "\"description: Help desk\\nnote: null\""}
            ])
        );
        Ok(())
    }

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
    fn choice_candidates_with_duplicate_meanings_remain_scoreable() -> Result<()> {
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
        let plan = prepare_request(&request)?
            .into_iter()
            .next()
            .ok_or_else(|| JetError::InvalidRequest("missing plan".into()))?;
        assert_eq!(plan.candidates.len(), 2);
        assert_eq!(plan.candidates[0].target, plan.candidates[1].target);
        assert!(plan.user_content.contains("\"same\", \"same\""));
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
