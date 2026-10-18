use std::collections::BTreeMap;

use serde_json::Value;

use crate::{JetError, Question, Result};

/// Encoded image bytes, decoded from the transport's base64 representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub id: String,
    pub media_type: String,
    pub data: Vec<u8>,
}

impl ImageInput {
    /// Check the supported media type and file signature before native decoding.
    /// The native image decoder remains responsible for validating the full image.
    pub fn validate_format(&self) -> Result<()> {
        let png = self.data.starts_with(b"\x89PNG\r\n\x1a\n");
        let jpeg = self.data.starts_with(b"\xff\xd8\xff");
        if matches!(self.media_type.as_str(), "image/png" if png)
            || matches!(self.media_type.as_str(), "image/jpeg" if jpeg)
        {
            Ok(())
        } else {
            Err(JetError::InvalidRequest(
                "image must contain PNG or JPEG bytes matching its media type".to_owned(),
            ))
        }
    }
}

/// An ordered part of a decision's context. JSON text retains its original value.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionContentPart {
    Text(Value),
    Image(ImageInput),
}

/// A decision containing one or more images interleaved with JSON text.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageDecisionRequest {
    pub state: Vec<DecisionContentPart>,
    pub questions: BTreeMap<String, Question>,
}
