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
    /// Transformer/output layers to offload. None selects all layers on Vulkan and none on CPU.
    /// Values above the model's layer count follow llama.cpp's all-layers behavior.
    /// Explicit placement disables opportunistic GPU execution of CPU weight operations.
    pub gpu_layers: Option<u32>,
    /// Keep routed expert tensors in the first N transformer layers on CPU.
    /// Requires Vulkan and a MoE model; attention and shared experts retain normal placement.
    /// Selected expert operations stay on CPU even during large prompt batches.
    pub cpu_moe_layers: u32,
    /// Allow llama.cpp to select memory mapping automatically; false disables mapping.
    pub use_mmap: bool,
    pub context_tokens_per_sequence: u32,
    pub token_batch: u32,
    pub micro_batch: u32,
    pub max_sequences: u32,
    /// Prepare the next Vulkan hybrid question on a CPU worker while the current question scores.
    pub preparation_pipeline: bool,
    pub max_output_rows: u32,
    pub threads: i32,
    pub execution_mode: ExecutionMode,
    pub thinking: ThinkingConfig,
    /// Collect cumulative stage timings and workload counters.
    pub collect_timings: bool,
    #[cfg(feature = "vision")]
    pub vision: Option<VisionConfig>,
}

#[cfg(feature = "vision")]
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub mmproj_path: PathBuf,
    pub max_images: usize,
    pub max_image_bytes: usize,
    pub max_image_pixels: usize,
    pub image_max_tokens: i32,
}

#[cfg(feature = "vision")]
impl VisionConfig {
    pub fn new(mmproj_path: impl Into<PathBuf>) -> Self {
        Self {
            mmproj_path: mmproj_path.into(),
            max_images: 4,
            max_image_bytes: 10 * 1024 * 1024,
            max_image_pixels: 16 * 1024 * 1024,
            image_max_tokens: 1024,
        }
    }
}

impl EngineConfig {
    pub fn cpu(model_path: impl Into<PathBuf>, model_id: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_id: model_id.into(),
            backend: Backend::Cpu,
            gpu_layers: None,
            cpu_moe_layers: 0,
            use_mmap: true,
            context_tokens_per_sequence: 2_048,
            token_batch: 2_048,
            micro_batch: 512,
            max_sequences: 9,
            preparation_pipeline: true,
            max_output_rows: 2_048,
            threads: std::thread::available_parallelism()
                .map(|count| i32::try_from(count.get()).unwrap_or(i32::MAX))
                .unwrap_or(1),
            execution_mode: ExecutionMode::Batched,
            thinking: ThinkingConfig::default(),
            collect_timings: false,
            #[cfg(feature = "vision")]
            vision: None,
        }
    }

    pub fn qwen3_cpu(model_path: impl Into<PathBuf>) -> Self {
        Self::cpu(model_path, "qwen/qwen3-0.6b-q8_0")
    }

    pub fn vulkan(model_path: impl Into<PathBuf>, model_id: impl Into<String>) -> Self {
        let mut config = Self::cpu(model_path, model_id);
        config.backend = Backend::Vulkan;
        config.max_output_rows = 256;
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
        assert_eq!(
            EngineConfig::qwen3_vulkan("model.gguf").max_output_rows,
            256
        );
        for config in [
            EngineConfig::qwen3_cpu("model.gguf"),
            EngineConfig::qwen3_vulkan("model.gguf"),
        ] {
            assert_eq!(config.gpu_layers, None);
            assert_eq!(config.cpu_moe_layers, 0);
            assert!(config.preparation_pipeline);
            assert!(config.use_mmap);
        }
    }
}
