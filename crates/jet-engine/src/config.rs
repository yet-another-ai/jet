use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    Reference,
    #[default]
    Batched,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub model_id: String,
    pub context_tokens_per_sequence: u32,
    pub token_batch: u32,
    pub micro_batch: u32,
    pub max_sequences: u32,
    pub max_output_rows: u32,
    pub threads: i32,
    pub execution_mode: ExecutionMode,
}

impl EngineConfig {
    pub fn qwen3_cpu(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_id: "qwen/qwen3-0.6b-q8_0".to_owned(),
            context_tokens_per_sequence: 2_048,
            token_batch: 2_048,
            micro_batch: 512,
            max_sequences: 9,
            max_output_rows: 2_048,
            threads: std::thread::available_parallelism()
                .map(|count| i32::try_from(count.get()).unwrap_or(i32::MAX))
                .unwrap_or(1),
            execution_mode: ExecutionMode::Batched,
        }
    }
}
