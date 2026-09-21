use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Cpu,
    Vulkan,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    Reference,
    #[default]
    Batched,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThinkingMode {
    #[default]
    Disabled,
    Auto,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThinkingConfig {
    pub mode: ThinkingMode,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            mode: ThinkingMode::Disabled,
            max_tokens: 256,
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub model_id: String,
    pub backend: Backend,
    pub context_tokens_per_sequence: u32,
    pub token_batch: u32,
    pub micro_batch: u32,
    pub max_sequences: u32,
    pub max_output_rows: u32,
    pub threads: i32,
    pub execution_mode: ExecutionMode,
    pub thinking: ThinkingConfig,
}

impl EngineConfig {
    pub fn cpu(model_path: impl Into<PathBuf>, model_id: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_id: model_id.into(),
            backend: Backend::Cpu,
            context_tokens_per_sequence: 2_048,
            token_batch: 2_048,
            micro_batch: 512,
            max_sequences: 9,
            max_output_rows: 2_048,
            threads: std::thread::available_parallelism()
                .map(|count| i32::try_from(count.get()).unwrap_or(i32::MAX))
                .unwrap_or(1),
            execution_mode: ExecutionMode::Batched,
            thinking: ThinkingConfig::default(),
        }
    }

    pub fn qwen3_cpu(model_path: impl Into<PathBuf>) -> Self {
        Self::cpu(model_path, "qwen/qwen3-0.6b-q8_0")
    }

    pub fn vulkan(model_path: impl Into<PathBuf>, model_id: impl Into<String>) -> Self {
        let mut config = Self::cpu(model_path, model_id);
        config.backend = Backend::Vulkan;
        config
    }

    pub fn qwen3_vulkan(model_path: impl Into<PathBuf>) -> Self {
        Self::vulkan(model_path, "qwen/qwen3-0.6b-q8_0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_select_the_requested_backend() {
        assert_eq!(EngineConfig::qwen3_cpu("model.gguf").backend, Backend::Cpu);
        assert_eq!(
            EngineConfig::qwen3_vulkan("model.gguf").backend,
            Backend::Vulkan
        );
    }
}
