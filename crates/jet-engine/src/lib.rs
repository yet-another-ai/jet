mod config;
mod evaluator;
mod llama;
mod prompt;
mod timing;

pub use config::{Backend, EngineConfig, ExecutionMode, ThinkingConfig, ThinkingMode};
pub use timing::EngineTimings;

use jet_core::{DecisionRequest, DecisionResponse, JetError, Result};

use evaluator::decide_batch;
use llama::LlamaScorer;

pub struct Engine {
    scorer: LlamaScorer,
    model_id: String,
}

impl Engine {
    pub fn load(config: EngineConfig) -> Result<Self> {
        let model_id = config.model_id.clone();
        let scorer = LlamaScorer::load(config)?;
        Ok(Self { scorer, model_id })
    }

    pub fn decide(&mut self, request: DecisionRequest) -> Result<DecisionResponse> {
        match self.decide_batch(&[request]).into_iter().next() {
            Some(result) => result,
            None => Err(JetError::NativeRuntime(
                "single-request evaluation returned no result".to_owned(),
            )),
        }
    }

    pub fn decide_batch(&mut self, requests: &[DecisionRequest]) -> Vec<Result<DecisionResponse>> {
        decide_batch(&mut self.scorer, &self.model_id, requests)
    }

    /// Return cumulative stage timings, or zeros if collection is disabled.
    pub fn timings(&self) -> &EngineTimings {
        self.scorer.timings()
    }
}
