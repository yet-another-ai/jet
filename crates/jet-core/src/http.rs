//! HTTP transport types for the OpenRouter Decisions API question subset.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{DecisionRequest, DecisionResponse, JetError, Question, Result};

/// An HTTP request naming the model loaded by the server.
///
/// Observability metadata is accepted but never passed to the decision engine.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpDecisionRequest {
    pub model: String,
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
    pub provider: Option<Value>,
    pub session_id: Option<String>,
    pub trace: Option<Map<String, Value>>,
    pub user: Option<String>,
}

impl HttpDecisionRequest {
    /// Validate the HTTP contract, normalize choice descriptions, and remove transport-only fields.
    pub fn into_decision(mut self, model_id: &str) -> Result<DecisionRequest> {
        if self.model != model_id {
            return Err(JetError::InvalidRequest(format!(
                "model must match the loaded model {model_id:?}"
            )));
        }

        if let Some(provider) = &self.provider
            && !provider.is_null()
            && !matches!(provider, Value::Object(preferences) if preferences.is_empty())
        {
            return Err(JetError::InvalidRequest(
                "provider routing is unavailable; omit provider or use null or an empty object"
                    .to_owned(),
            ));
        }

        for (name, value) in [
            ("session_id", self.session_id.as_deref()),
            ("user", self.user.as_deref()),
        ] {
            if value.is_some_and(|value| value.chars().count() > 256) {
                return Err(JetError::InvalidRequest(format!(
                    "{name} must contain at most 256 characters"
                )));
            }
        }

        validate_text(&self.state, "state")?;
        for (name, question) in &mut self.questions {
            let path = format!("questions.{name}");
            match question {
                Question::Noul(question) => {
                    validate_text(&question.instructions, &format!("{path}.instructions"))?;
                    if let Some(criteria) = &question.criteria {
                        validate_text(&criteria.false_value, &format!("{path}.criteria.false"))?;
                        validate_text(&criteria.true_value, &format!("{path}.criteria.true"))?;
                    }
                }
                Question::Choice(question) => {
                    validate_text(&question.instructions, &format!("{path}.instructions"))?;
                    for (choice, description) in &mut question.criteria {
                        if description.is_null() {
                            *description = Value::String(choice.clone());
                        } else {
                            validate_text(description, &format!("{path}.criteria.{choice}"))?;
                        }
                    }
                }
                Question::Score(question) => {
                    validate_text(&question.instructions, &format!("{path}.instructions"))?;
                    if question.criteria.is_empty() {
                        return Err(JetError::InvalidRequest(format!(
                            "{path}.criteria must contain at least one item"
                        )));
                    }
                    for (index, description) in question.criteria.iter().enumerate() {
                        validate_text(description, &format!("{path}.criteria.{index}"))?;
                    }
                }
            }
        }

        Ok(DecisionRequest {
            state: self.state,
            questions: self.questions,
        })
    }
}

fn validate_text(value: &Value, field: &str) -> Result<()> {
    if matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_)) {
        Ok(())
    } else {
        Err(JetError::InvalidRequest(format!(
            "{field} must be a string, object, or array"
        )))
    }
}

/// A Decisions response with engine execution time, excluding queueing and HTTP I/O.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HttpDecisionResponse {
    #[serde(flatten)]
    pub decision: DecisionResponse,
    pub inference_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HttpErrorResponse {
    pub error: HttpErrorBody,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HttpErrorBody {
    pub code: u16,
    pub message: String,
}

impl HttpErrorResponse {
    pub fn new(code: u16, message: impl Into<String>) -> Self {
        Self {
            error: HttpErrorBody {
                code,
                message: message.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, Usage};

    fn request() -> Value {
        json!({
            "model": "jet-test",
            "state": {"ticket": "The checkout failed", "attempts": 2},
            "questions": {
                "bug": {"type": "noul", "instructions": "Is this a bug?"},
                "team": {
                    "type": "choice",
                    "instructions": ["Select a team", {"priority": true}],
                    "criteria": {"billing": null, "support": {"owns": [1, false, null]}}
                },
                "urgency": {
                    "type": "score",
                    "instructions": {"task": "Rate the urgency"},
                    "criteria": ["Can wait", ["Immediate", null]]
                }
            }
        })
    }

    fn convert(value: Value) -> Result<DecisionRequest> {
        serde_json::from_value::<HttpDecisionRequest>(value)?.into_decision("jet-test")
    }

    #[test]
    fn compatible_questions_are_normalized_without_transport_metadata() -> Result<()> {
        let mut input = request();
        input["provider"] = json!({});
        input["session_id"] = json!("界".repeat(256));
        input["trace"] = json!({"trace_id": "trace-123", "custom": {"attempt": 1}});
        input["user"] = json!("user-123");
        let mut expected = json!({"state": input["state"], "questions": input["questions"]});
        expected["questions"]["team"]["criteria"]["billing"] = json!("billing");

        assert_eq!(convert(input)?, serde_json::from_value(expected)?);
        Ok(())
    }

    #[test]
    fn required_model_and_unknown_fields_are_checked() {
        let mut missing_model = request();
        if let Value::Object(fields) = &mut missing_model {
            fields.remove("model");
        }
        assert!(convert(missing_model).is_err());

        let mut wrong_model = request();
        wrong_model["model"] = json!("another-model");
        assert!(convert(wrong_model).is_err());

        let mut unknown_field = request();
        unknown_field["temperature"] = json!(0);
        assert!(convert(unknown_field).is_err());
    }

    #[test]
    fn rejects_scalar_context_and_invalid_question_descriptions() {
        for scalar in [Value::Null, json!(true), json!(42)] {
            let mut input = request();
            input["state"] = scalar.clone();
            assert!(convert(input).is_err());

            let mut input = request();
            input["questions"]["bug"]["instructions"] = scalar.clone();
            assert!(convert(input).is_err());

            let mut input = request();
            input["questions"]["urgency"]["criteria"][0] = scalar;
            assert!(convert(input).is_err());
        }

        let mut incomplete_noul = request();
        incomplete_noul["questions"]["bug"]["criteria"] = json!({"true": "A defect"});
        assert!(convert(incomplete_noul).is_err());

        let mut invalid_choice = request();
        invalid_choice["questions"]["team"]["criteria"]["billing"] = json!(true);
        assert!(convert(invalid_choice).is_err());

        let mut empty_score = request();
        empty_score["questions"]["urgency"]["criteria"] = json!([]);
        assert!(convert(empty_score).is_err());
    }

    #[test]
    fn accepts_complete_noul_criteria_and_one_score_level() -> Result<()> {
        let mut input = request();
        input["questions"]["bug"]["criteria"] =
            json!({"false": "A question", "true": {"kind": "defect"}});
        input["questions"]["urgency"]["criteria"] = json!(["Only level"]);
        convert(input)?;
        Ok(())
    }

    #[test]
    fn routing_and_metadata_limits_are_explicit() -> Result<()> {
        for provider in [Value::Null, json!({})] {
            let mut input = request();
            input["provider"] = provider;
            convert(input)?;
        }
        for provider in [json!({"order": ["other"]}), json!([]), json!("local")] {
            let mut input = request();
            input["provider"] = provider;
            assert!(convert(input).is_err());
        }
        for field in ["session_id", "user"] {
            let mut input = request();
            input[field] = json!("界".repeat(257));
            assert!(convert(input).is_err());
        }
        Ok(())
    }

    #[test]
    fn response_keeps_decisions_fields_and_adds_inference_time() -> Result<()> {
        let response = HttpDecisionResponse {
            decision: DecisionResponse {
                model: "jet-test".to_owned(),
                answers: BTreeMap::from([
                    ("bug".to_owned(), Answer::Noul(NoulAnswer { noul: 0.9 })),
                    (
                        "team".to_owned(),
                        Answer::Choice(ChoiceAnswer {
                            choice: "billing".to_owned(),
                            probabilities: BTreeMap::from([("billing".to_owned(), 1.0)]),
                        }),
                    ),
                    (
                        "urgency".to_owned(),
                        Answer::Score(ScoreAnswer {
                            score: 0.0,
                            probabilities: BTreeMap::from([("0".to_owned(), 1.0)]),
                        }),
                    ),
                ]),
                usage: Usage {
                    input_tokens: 42,
                    output_tokens: 3,
                },
            },
            inference_ms: 12.25,
        };
        assert_eq!(
            serde_json::to_value(response)?,
            json!({
                "model": "jet-test",
                "answers": {
                    "bug": {"type": "noul", "noul": 0.9},
                    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0}},
                    "urgency": {"type": "score", "score": 0.0, "probabilities": {"0": 1.0}}
                },
                "usage": {"input_tokens": 42, "output_tokens": 3},
                "inference_ms": 12.25
            })
        );
        assert_eq!(
            serde_json::to_value(HttpErrorResponse::new(400, "Invalid request"))?,
            json!({"error": {"code": 400, "message": "Invalid request"}})
        );
        Ok(())
    }
}
