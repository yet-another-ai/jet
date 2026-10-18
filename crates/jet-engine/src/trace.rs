use serde::Serialize;

#[derive(Debug, Default, Clone, Serialize)]
pub struct EngineTrace {
    pub jobs: Vec<ScoringJobTrace>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoringJobTrace {
    pub rendered_prompt: String,
    pub prompt_token_ids: Vec<i32>,
    pub candidates: Vec<ScoringCandidateTrace>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoringCandidateTrace {
    pub target: String,
    pub target_suffix_token_ids: Vec<i32>,
    pub raw_log_probability: f64,
}
