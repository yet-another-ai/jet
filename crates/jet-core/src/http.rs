//! HTTP transport types for the OpenRouter Decisions API question subset.

use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    DecisionContentPart, DecisionRequest, DecisionResponse, ImageDecisionRequest, ImageInput,
    JetError, Question, Result,
};

/// A normalized HTTP decision, ready for the appropriate engine entry point.
#[derive(Debug, Clone, PartialEq)]
pub enum HttpDecisionInput {
    Text(DecisionRequest),
    Images(ImageDecisionRequest),
}

/// Limits on inline images, applied before admitting the request to the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpImageLimits {
    pub max_images: usize,
    /// Maximum decoded file bytes per image, before native image decoding.
    pub max_image_bytes: usize,
}

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
    /// Explicit media blocks require `into_input` with image support enabled.
    pub fn into_decision(self, model_id: &str) -> Result<DecisionRequest> {
        match self.into_input(model_id, None)? {
            HttpDecisionInput::Text(request) => Ok(request),
            HttpDecisionInput::Images(_) => Err(images_unavailable()),
        }
    }

    /// Normalize text and optional inline images while retaining their input order.
    /// Only media blocks directly inside a state array have multimodal semantics.
    pub fn into_input(
        self,
        model_id: &str,
        image_limits: Option<HttpImageLimits>,
    ) -> Result<HttpDecisionInput> {
        let request = self.normalize(model_id)?;
        let Value::Array(parts) = &request.state else {
            return Ok(HttpDecisionInput::Text(request));
        };
        let has_images = parts
            .iter()
            .any(|part| media_kind(part) == Some("image_url"));
        for part in parts {
            if matches!(
                media_kind(part),
                Some("input_image" | "input_audio" | "video_url")
            ) {
                return Err(JetError::InvalidRequest(
                    "unsupported media block; use image_url with a PNG or JPEG base64 data URL"
                        .to_owned(),
                ));
            }
        }
        if !has_images {
            return Ok(HttpDecisionInput::Text(request));
        }
        let limits = image_limits.ok_or_else(images_unavailable)?;
        let Value::Array(parts) = request.state else {
            return Err(JetError::InvalidRequest(
                "image context must be a state array".to_owned(),
            ));
        };
        let mut state = Vec::with_capacity(parts.len());
        let mut image_count = 0;
        for part in parts {
            if media_kind(&part) == Some("image_url") {
                image_count += 1;
                if image_count > limits.max_images {
                    return Err(JetError::InvalidRequest(format!(
                        "state must contain at most {} images",
                        limits.max_images
                    )));
                }
                state.push(DecisionContentPart::Image(decode_image(
                    part,
                    image_count,
                    limits.max_image_bytes,
                )?));
            } else {
                state.push(DecisionContentPart::Text(part));
            }
        }
        Ok(HttpDecisionInput::Images(ImageDecisionRequest {
            state,
            questions: request.questions,
        }))
    }

    fn normalize(mut self, model_id: &str) -> Result<DecisionRequest> {
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

fn media_kind(value: &Value) -> Option<&str> {
    value.as_object()?.get("type")?.as_str()
}

fn images_unavailable() -> JetError {
    JetError::InvalidRequest(
        "image input is unavailable; start a vision-enabled server with a matching mmproj"
            .to_owned(),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpImagePart {
    #[serde(rename = "type")]
    _kind: String,
    image_url: HttpImageUrl,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpImageUrl {
    url: String,
    /// Accepted for wire compatibility; resolution is controlled by the vision engine.
    #[serde(rename = "detail")]
    _detail: Option<ImageDetail>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ImageDetail {
    Auto,
    Low,
    High,
}

fn decode_image(value: Value, index: usize, max_bytes: usize) -> Result<ImageInput> {
    let part: HttpImagePart = serde_json::from_value(value).map_err(|_| {
        JetError::InvalidRequest(
            "image_url requires an object with url and optional detail (auto, low, or high)"
                .to_owned(),
        )
    })?;
    let (header, encoded) = part.image_url.url.split_once(',').ok_or_else(|| {
        JetError::InvalidRequest(
            "image_url.url must be a PNG or JPEG base64 data URL; remote URLs and file paths are unsupported"
                .to_owned(),
        )
    })?;
    let media_type = match header {
        "data:image/png;base64" => "image/png",
        "data:image/jpeg;base64" => "image/jpeg",
        _ => {
            return Err(JetError::InvalidRequest(
                "image_url.url must use data:image/png;base64 or data:image/jpeg;base64".to_owned(),
            ));
        }
    };
    // At most two bytes of padding slack, without allocating an oversized decoded buffer.
    if encoded.len() > max_bytes.div_ceil(3).saturating_mul(4) {
        return Err(image_too_large(max_bytes));
    }
    let data = STANDARD.decode(encoded).map_err(|_| {
        JetError::InvalidRequest("image_url.url contains invalid base64 image data".to_owned())
    })?;
    if data.len() > max_bytes {
        return Err(image_too_large(max_bytes));
    }
    let image = ImageInput {
        id: format!("image_{index}"),
        media_type: media_type.to_owned(),
        data,
    };
    image.validate_format()?;
    Ok(image)
}

fn image_too_large(max_bytes: usize) -> JetError {
    JetError::InvalidRequest(format!(
        "each image must contain at most {max_bytes} decoded file bytes"
    ))
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

/// A Decisions response with separately measured engine execution and queueing time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HttpDecisionResponse {
    #[serde(flatten)]
    pub decision: DecisionResponse,
    /// Monotonic elapsed time around the engine call, including native image decoding,
    /// visual encoding, and candidate scoring. Excludes queueing, base64 normalization,
    /// and HTTP I/O.
    pub inference_ms: f64,
    /// Monotonic elapsed time from accepted submission until the engine call begins.
    /// Includes scheduler handoff, but excludes parsing, normalization, and network I/O.
    /// May be slightly positive even when queueing is disabled.
    pub queue_ms: f64,
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

    fn convert_input(value: Value, limits: Option<HttpImageLimits>) -> Result<HttpDecisionInput> {
        serde_json::from_value::<HttpDecisionRequest>(value)?.into_input("jet-test", limits)
    }

    fn image_part(media_type: &str, data: &[u8]) -> Value {
        json!({
            "type": "image_url",
            "image_url": {"url": format!("data:{media_type};base64,{}", STANDARD.encode(data))}
        })
    }

    fn image_limits() -> HttpImageLimits {
        HttpImageLimits {
            max_images: 2,
            max_image_bytes: 1024,
        }
    }

    #[test]
    fn image_context_preserves_text_image_order_and_normalizes_questions() -> Result<()> {
        let red = include_bytes!("../../../tests/fixtures/vision/red.png");
        let blue = include_bytes!("../../../tests/fixtures/vision/blue.png");
        let mut input = request();
        input["state"] = json!([
            "Compare these images",
            image_part("image/png", red),
            {"note": "then this one", "weight": 2},
            image_part("image/png", blue),
            ["What changed?", null]
        ]);
        input["trace"] = json!({"trace_id": "not engine input"});
        let converted = convert_input(input, Some(image_limits()))?;
        let HttpDecisionInput::Images(converted) = converted else {
            return Err(JetError::InvalidRequest(
                "expected an image decision".to_owned(),
            ));
        };
        assert_eq!(
            converted.state,
            vec![
                DecisionContentPart::Text(json!("Compare these images")),
                DecisionContentPart::Image(ImageInput {
                    id: "image_1".to_owned(),
                    media_type: "image/png".to_owned(),
                    data: red.to_vec(),
                }),
                DecisionContentPart::Text(json!({"note": "then this one", "weight": 2})),
                DecisionContentPart::Image(ImageInput {
                    id: "image_2".to_owned(),
                    media_type: "image/png".to_owned(),
                    data: blue.to_vec(),
                }),
                DecisionContentPart::Text(json!(["What changed?", null])),
            ]
        );
        let Some(Question::Choice(team)) = converted.questions.get("team") else {
            return Err(JetError::InvalidRequest(
                "expected a choice question".to_owned(),
            ));
        };
        assert_eq!(team.criteria.get("billing"), Some(&json!("billing")));
        Ok(())
    }

    #[test]
    fn text_and_nested_json_keep_existing_semantics_with_or_without_vision() -> Result<()> {
        for state in [
            json!("plain text"),
            json!({"type": "image_url", "image_url": {"url": "ordinary object"}}),
            json!(["text", 42, true, null]),
            json!([{"data": {"type": "image_url", "image_url": {"url": "nested"}}}]),
            json!([[{"type": "input_audio", "input_audio": {"data": "nested"}}]]),
            json!([{"type": "text", "text": "ordinary JSON"}]),
        ] {
            let mut input = request();
            input["state"] = state;
            let expected = HttpDecisionInput::Text(convert(input.clone())?);
            assert_eq!(convert_input(input.clone(), None)?, expected);
            assert_eq!(convert_input(input, Some(image_limits()))?, expected);
        }
        Ok(())
    }

    #[test]
    fn image_count_and_decoded_byte_limits_are_inclusive() -> Result<()> {
        let png = include_bytes!("../../../tests/fixtures/vision/red.png");
        let mut input = request();
        input["state"] = json!([image_part("image/png", png), image_part("image/png", png)]);
        let limits = HttpImageLimits {
            max_images: 2,
            max_image_bytes: png.len(),
        };
        convert_input(input.clone(), Some(limits))?;
        for limits in [
            HttpImageLimits {
                max_images: 1,
                ..limits
            },
            HttpImageLimits {
                max_images: 0,
                ..limits
            },
            HttpImageLimits {
                max_image_bytes: png.len() - 1,
                ..limits
            },
            HttpImageLimits {
                max_image_bytes: 0,
                ..limits
            },
        ] {
            assert!(convert_input(input.clone(), Some(limits)).is_err());
        }
        // A one-byte size difference can have the same encoded length; check after decoding too.
        let jpeg = b"\xff\xd8\xff\x01\x02";
        input["state"] = json!([image_part("image/jpeg", jpeg)]);
        assert!(
            convert_input(
                input,
                Some(HttpImageLimits {
                    max_images: 1,
                    max_image_bytes: jpeg.len() - 1,
                })
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn jpeg_and_supported_detail_hints_are_accepted() -> Result<()> {
        // Full raster validity is checked by the native decoder, after this signature check.
        let jpeg = b"\xff\xd8\xff\x01\x02";
        for detail in [Value::Null, json!("auto"), json!("low"), json!("high")] {
            let mut image = image_part("image/jpeg", jpeg);
            image["image_url"]["detail"] = detail;
            let mut input = request();
            input["state"] = json!([image]);
            assert!(matches!(
                convert_input(input, Some(image_limits()))?,
                HttpDecisionInput::Images(_)
            ));
        }
        Ok(())
    }

    #[test]
    fn disabled_vision_and_other_explicit_media_are_not_scored_as_text() -> Result<()> {
        let mut input = request();
        input["state"] = json!([image_part("image/png", b"\x89PNG\r\n\x1a\n")]);
        let error = convert_input(input.clone(), None).err().ok_or_else(|| {
            JetError::InvalidRequest("expected an unsupported image error".to_owned())
        })?;
        assert!(error.to_string().contains("image input is unavailable"));
        assert!(convert(input).is_err());
        for kind in ["input_image", "input_audio", "video_url"] {
            let mut input = request();
            input["state"] = json!([{"type": kind}]);
            assert!(convert_input(input.clone(), None).is_err());
            assert!(convert_input(input, Some(image_limits())).is_err());
        }
        Ok(())
    }

    #[test]
    fn malformed_images_and_remote_sources_are_rejected_without_echoing_payloads() -> Result<()> {
        let png = b"\x89PNG\r\n\x1a\n";
        let payload = "PRIVATE_IMAGE_DATA".repeat(128);
        let bad_images = [
            json!({"type": "image_url"}),
            json!({"type": "image_url", "image_url": payload}),
            json!({"type": "image_url", "image_url": {"url": 123}}),
            json!({"type": "image_url", "image_url": {"url": "https://example.org/image.png"}}),
            json!({"type": "image_url", "image_url": {"url": "C:/images/frame.png"}}),
            json!({"type": "image_url", "image_url": {"url": "file:///images/frame.png"}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png,raw"}}),
            image_part("image/gif", b"GIF89a"),
            image_part("image/jpeg", png),
            image_part("image/png", b"\xff\xd8\xff"),
            image_part("image/png", b""),
            json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{payload}")}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%="}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "detail": payload}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "unexpected": true}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}, "unexpected": true}),
        ];
        for image in bad_images {
            let mut input = request();
            input["state"] = json!([image]);
            let error = convert_input(input, Some(image_limits()))
                .err()
                .ok_or_else(|| {
                    JetError::InvalidRequest("expected an invalid image error".to_owned())
                })?;
            let message = error.to_string();
            assert!(!message.contains("PRIVATE_IMAGE_DATA"));
            assert!(message.len() < 256);
        }
        Ok(())
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
    fn response_keeps_decisions_fields_and_adds_inference_and_queue_times() -> Result<()> {
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
            queue_ms: 0.5,
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
                "inference_ms": 12.25,
                "queue_ms": 0.5
            })
        );
        assert_eq!(
            serde_json::to_value(HttpErrorResponse::new(400, "Invalid request"))?,
            json!({"error": {"code": 400, "message": "Invalid request"}})
        );
        Ok(())
    }
}
