use std::collections::BTreeMap;

use jet_core::{Answer, DecisionRequest, JetError, Question, Result};
use serde::Serialize;
use serde_json::Value;

use crate::evaluator::{build_response, plan_to_job};
use crate::llama::LlamaScorer;
use crate::prompt::prepare_request;

#[derive(Debug, Clone)]
pub struct ImageInput {
    pub id: String,
    pub media_type: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct MultimodalDecisionRequest {
    pub request_id: Option<String>,
    pub state: Value,
    pub images: Vec<ImageInput>,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultimodalDecisionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: MultimodalUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MultimodalUsage {
    pub input_tokens: u64,
    pub input_text_tokens: u64,
    pub input_image_tokens: u64,
    pub output_tokens: u64,
}

pub(crate) fn decide_multimodal(
    scorer: &mut LlamaScorer,
    model_id: &str,
    request: MultimodalDecisionRequest,
) -> Result<MultimodalDecisionResponse> {
    scorer.validate_images(&request.images)?;
    let decision = DecisionRequest {
        state: request.state,
        questions: request.questions,
    };
    let plans = prepare_request(&decision)?;
    let mut scores = Vec::with_capacity(plans.len());
    let mut image_tokens = 0_u64;
    for plan in &plans {
        let job = plan_to_job(plan);
        let (score, count) = scorer.score_vision_job(&job, &request.images)?;
        image_tokens = image_tokens.saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
        scores.push(score);
    }
    let response = build_response(model_id, &plans, &scores)?;
    Ok(MultimodalDecisionResponse {
        request_id: request.request_id,
        model: response.model,
        answers: response.answers,
        usage: MultimodalUsage {
            input_tokens: response.usage.input_tokens,
            input_text_tokens: response.usage.input_tokens.saturating_sub(image_tokens),
            input_image_tokens: image_tokens,
            output_tokens: response.usage.output_tokens,
        },
    })
}

pub(crate) fn validate_image_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || matches!(value, b'_' | b'-'))
    {
        return Err(JetError::InvalidRequest(
            "image id must contain 1-64 ASCII letters, digits, '_' or '-'".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_image_format(image: &ImageInput) -> Result<()> {
    let png = image.data.starts_with(b"\x89PNG\r\n\x1a\n");
    let jpeg = image.data.starts_with(b"\xff\xd8\xff");
    let correct = matches!(image.media_type.as_str(), "image/png" if png)
        || matches!(image.media_type.as_str(), "image/jpeg" if jpeg);
    if !correct {
        return Err(JetError::InvalidRequest(format!(
            "image {:?} must contain PNG or JPEG bytes matching media_type",
            image.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_ids_and_bytes_are_checked_before_native_decode() {
        let fixture = include_bytes!("../../../tests/fixtures/vision/red.png");
        let image = ImageInput {
            id: "current_frame".to_owned(),
            media_type: "image/png".to_owned(),
            data: fixture.to_vec(),
        };
        assert!(validate_image_id(&image.id).is_ok());
        assert!(validate_image_format(&image).is_ok());
        assert!(validate_image_id("<__media__>").is_err());
        let mismatched = ImageInput {
            media_type: "image/jpeg".to_owned(),
            ..image.clone()
        };
        assert!(validate_image_format(&mismatched).is_err());
        let truncated = ImageInput {
            data: vec![0, 1, 2],
            ..image
        };
        assert!(validate_image_format(&truncated).is_err());
    }
}
