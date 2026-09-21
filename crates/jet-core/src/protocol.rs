use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Question {
    Noul(NoulQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulQuestion {
    pub instructions: Value,
    #[serde(default)]
    pub criteria: Option<NoulCriteria>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "false")]
    pub false_value: Value,
    #[serde(rename = "true")]
    pub true_value: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceQuestion {
    pub instructions: Value,
    pub criteria: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreQuestion {
    pub instructions: Value,
    pub criteria: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NoulAnswer {
    pub noul: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_golden_covers_all_question_types() -> serde_json::Result<()> {
        let json = serde_json::json!({
            "state": {"language": "中文", "items": [1, true, null]},
            "questions": {
                "approve": {
                    "type": "noul",
                    "instructions": "Approve?",
                    "criteria": {"false": "reject", "true": "accept"}
                },
                "route": {
                    "type": "choice",
                    "instructions": ["choose", "one"],
                    "criteria": {"alpha": {"team": "A"}, "beta": ["B"]}
                },
                "priority": {
                    "type": "score",
                    "instructions": {"task": "rate"},
                    "criteria": ["low", "medium", "high"]
                }
            }
        });
        let request: DecisionRequest = serde_json::from_value(json.clone())?;
        assert_eq!(serde_json::to_value(request)?, json);
        Ok(())
    }

    #[test]
    fn request_rejects_model_and_unknown_question_fields() {
        let with_model = serde_json::json!({
            "model": "must-not-be-accepted",
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x"}}
        });
        assert!(serde_json::from_value::<DecisionRequest>(with_model).is_err());

        let unknown = serde_json::json!({
            "state": "x",
            "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["a"], "extra": true}}
        });
        assert!(serde_json::from_value::<DecisionRequest>(unknown).is_err());
    }

    #[test]
    fn response_has_only_decisions_compatible_public_fields() -> serde_json::Result<()> {
        let response = DecisionResponse {
            model: "model-id".to_owned(),
            answers: BTreeMap::from([(
                "q".to_owned(),
                Answer::Choice(ChoiceAnswer {
                    choice: "a".to_owned(),
                    probabilities: BTreeMap::from([("a".to_owned(), 0.75), ("b".to_owned(), 0.25)]),
                }),
            )]),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
            },
        };
        assert_eq!(
            serde_json::to_value(response)?,
            serde_json::json!({
                "model": "model-id",
                "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.75, "b": 0.25}}},
                "usage": {"input_tokens": 10, "output_tokens": 2}
            })
        );
        Ok(())
    }
}
