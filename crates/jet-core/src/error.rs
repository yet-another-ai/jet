use serde::Serialize;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, JetError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    UnsupportedModel,
    ContextExceeded,
    NativeRuntime,
    NonFiniteScore,
    Io,
    Json,
}

#[derive(Debug, Error)]
pub enum JetError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("request needs {required} context tokens, but the configured limit is {limit}")]
    ContextExceeded { required: usize, limit: usize },
    #[error("native runtime error: {0}")]
    NativeRuntime(String),
    #[error("scoring produced a non-finite value: {0}")]
    NonFiniteScore(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl JetError {
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::InvalidRequest(_) => ErrorCode::InvalidRequest,
            Self::UnsupportedModel(_) => ErrorCode::UnsupportedModel,
            Self::ContextExceeded { .. } => ErrorCode::ContextExceeded,
            Self::NativeRuntime(_) => ErrorCode::NativeRuntime,
            Self::NonFiniteScore(_) => ErrorCode::NonFiniteScore,
            Self::Io(_) => ErrorCode::Io,
            Self::Json(_) => ErrorCode::Json,
        }
    }
}
