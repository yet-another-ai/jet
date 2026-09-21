use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::slice;
use std::sync::Once;
use std::time::Instant;

use jet_core::{JetError, Result};
use jet_llama_sys as sys;
use serde::Deserialize;

use crate::EngineTimings;
use crate::config::{Backend, EngineConfig, ExecutionMode, ThinkingMode};
use crate::evaluator::{ScoreJob, ScoreResult, SequenceScorer};

static BACKEND_INIT: Once = Once::new();

struct TokenizedJob {
    prompt: Vec<sys::llama_token>,
    targets: Vec<Vec<sys::llama_token>>,
    logical_prompt_tokens: usize,
    thinking_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ChatPlan {
    prompt: String,
    generation_prompt: String,
    supports_thinking: bool,
    thinking_start_tag: String,
    thinking_end_tags: Vec<String>,
}

struct ThinkingSampler(NonNull<sys::jet_thinking_sampler>);

impl Drop for ThinkingSampler {
    fn drop(&mut self) {
        // SAFETY: the pointer was returned by jet_thinking_sampler_init and is owned here.
        unsafe { sys::jet_thinking_sampler_free(self.0.as_ptr()) };
    }
}

struct CpuThreadpool(NonNull<sys::ggml_threadpool>);

impl CpuThreadpool {
    fn new(threads: i32) -> Result<Self> {
        // SAFETY: configuration validation requires a positive thread count and the
        // bridge returns an owned threadpool or null after backend initialization.
        NonNull::new(unsafe { sys::jet_cpu_threadpool_new(threads) })
            .map(Self)
            .ok_or_else(|| JetError::NativeRuntime("failed to create CPU threadpool".to_owned()))
    }
}

impl Drop for CpuThreadpool {
    fn drop(&mut self) {
        // SAFETY: all contexts borrowing this pool have already been freed.
        unsafe { sys::jet_cpu_threadpool_free(self.0.as_ptr()) };
    }
}

#[derive(Clone, Copy)]
struct GroupSpec {
    job_index: usize,
    candidate_start: usize,
    candidate_end: usize,
}

struct ActiveGroup {
    spec: GroupSpec,
    prefix_sequence: sys::llama_seq_id,
    candidate_sequences: Vec<sys::llama_seq_id>,
}

enum OutputAction {
    None,
    Prefix(usize),
    Target {
        job_index: usize,
        candidate_index: usize,
        target: sys::llama_token,
    },
}

enum OutputDistribution<'a> {
    Logits(&'a [f32]),
    DeviceLogProbabilities(&'a [f32]),
}

impl OutputDistribution<'_> {
    fn target_log_probability(&self, target: sys::llama_token, target_index: usize) -> Result<f64> {
        match self {
            Self::Logits(values) => target_log_probability(values, target),
            Self::DeviceLogProbabilities(values) => {
                let value = f64::from(*values.get(target_index).ok_or_else(|| {
                    JetError::NativeRuntime(
                        "Vulkan score target index exceeds sampler output".to_owned(),
                    )
                })?);
                if value.is_nan() || value == f64::INFINITY || value > 0.0 {
                    return Err(JetError::NonFiniteScore(format!(
                        "Vulkan softmax returned invalid target log-probability {value}"
                    )));
                }
                Ok(value)
            }
        }
    }
}

struct DecodeItem {
    token: sys::llama_token,
    position: sys::llama_pos,
    sequence: sys::llama_seq_id,
    output: OutputAction,
}

#[derive(Default)]
struct TensorBufferOverrides {
    // Native model parameters retain these pointers, so both allocations outlive the model.
    _patterns: Vec<CString>,
    entries: Vec<sys::llama_model_tensor_buft_override>,
}

impl TensorBufferOverrides {
    fn cpu_experts(layers: u32) -> Result<Self> {
        if layers == 0 {
            return Ok(Self::default());
        }
        // SAFETY: backend initialization completed before constructing model overrides.
        let cpu_buffer = unsafe { sys::jet_cpu_buffer_type() };
        if cpu_buffer.is_null() {
            return Err(JetError::NativeRuntime(
                "CPU buffer type is unavailable for cpu_moe_layers".to_owned(),
            ));
        }
        // This follows llama.cpp's LLM_FFN_EXPS_REGEX, including fused gate/up and
        // chunked expert tensors, while excluding the router and shared experts.
        let patterns = (0..layers)
            .map(|layer| {
                CString::new(format!(
                    r"^blk\.{layer}\.ffn_(up|down|gate|gate_up)_(ch|)exps\."
                ))
                .map_err(|_| {
                    JetError::NativeRuntime(
                        "invalid generated CPU expert tensor pattern".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut entries: Vec<_> = patterns
            .iter()
            .map(|pattern| sys::llama_model_tensor_buft_override {
                pattern: pattern.as_ptr(),
                buft: cpu_buffer,
            })
            .collect();
        entries.push(sys::llama_model_tensor_buft_override::default());
        Ok(Self {
            _patterns: patterns,
            entries,
        })
    }

    fn as_ptr(&self) -> *const sys::llama_model_tensor_buft_override {
        if self.entries.is_empty() {
            ptr::null()
        } else {
            self.entries.as_ptr()
        }
    }
}

pub(crate) struct LlamaScorer {
    model: NonNull<sys::llama_model>,
    context: NonNull<sys::llama_context>,
    generation_context: Option<NonNull<sys::llama_context>>,
    vocab: NonNull<sys::llama_vocab>,
    config: EngineConfig,
    vocab_size: usize,
    score_samplers: Vec<NonNull<sys::llama_sampler>>,
    device_softmax: bool,
    device_target_width: usize,
    candidate_capacity: usize,
    isolate_candidate_groups: bool,
    timings: EngineTimings,
    // LlamaScorer::drop frees both contexts before this owned pool is dropped.
    _cpu_threadpool: CpuThreadpool,
    _tensor_buffer_overrides: TensorBufferOverrides,
    _single_threaded: PhantomData<Rc<()>>,
}

impl LlamaScorer {
    pub(crate) fn load(config: EngineConfig) -> Result<Self> {
        validate_config(&config)?;
        let started = config.collect_timings.then(Instant::now);
        BACKEND_INIT.call_once(|| {
            // SAFETY: llama.cpp requires one process-wide initialization before any model calls.
            unsafe { sys::llama_backend_init() };
        });
        if config.backend == Backend::Vulkan {
            if !cfg!(feature = "vulkan") {
                return Err(JetError::NativeRuntime(
                    "Vulkan was requested, but this build does not include the `vulkan` feature"
                        .to_owned(),
                ));
            }
            // SAFETY: llama.cpp's process-wide backend initialization completed above.
            if !unsafe { sys::llama_supports_gpu_offload() } {
                return Err(JetError::NativeRuntime(
                    "Vulkan was requested, but llama.cpp found no GPU device available for offload"
                        .to_owned(),
                ));
            }
        }

        let path = config
            .model_path
            .to_str()
            .ok_or_else(|| JetError::InvalidRequest("model_path must be valid UTF-8".to_owned()))?;
        let path = CString::new(path).map_err(|_| {
            JetError::InvalidRequest("model_path contains an interior NUL byte".to_owned())
        })?;

        // SAFETY: default parameter structs are returned by value and remain valid for this call.
        let mut model_params = unsafe { sys::llama_model_default_params() };
        model_params.n_gpu_layers = match config.backend {
            Backend::Cpu => 0,
            Backend::Vulkan => config.gpu_layers.map_or(-1, |layers| layers as i32),
        };
        if !config.use_mmap {
            model_params.load_mode = sys::llama_load_mode_LLAMA_LOAD_MODE_NONE;
        }
        if config.cpu_moe_layers > 0 {
            validate_cpu_moe_model(&path, model_params, config.cpu_moe_layers)?;
        }
        let tensor_buffer_overrides = TensorBufferOverrides::cpu_experts(config.cpu_moe_layers)?;
        model_params.tensor_buft_overrides = tensor_buffer_overrides.as_ptr();
        // SAFETY: path is a live NUL-terminated string and model_params came from llama.cpp.
        let model =
            NonNull::new(unsafe { sys::llama_model_load_from_file(path.as_ptr(), model_params) })
                .ok_or_else(|| {
                JetError::NativeRuntime(format!(
                    "failed to load GGUF model from {}",
                    config.model_path.display()
                ))
            })?;

        let mut load_result = Self::finish_load(model, config);
        if let Ok(scorer) = &mut load_result {
            scorer.timings.load_ms = elapsed_ms(started);
            scorer._tensor_buffer_overrides = tensor_buffer_overrides;
        }
        if load_result.is_err() {
            // SAFETY: model was returned by llama_model_load_from_file and no context owns it.
            unsafe { sys::llama_model_free(model.as_ptr()) };
        }
        load_result
    }

    fn finish_load(model: NonNull<sys::llama_model>, config: EngineConfig) -> Result<Self> {
        // Create the pool before native contexts. Every failure path below frees
        // its contexts before this local owner drops, and success transfers it to
        // the scorer. Without an attached pool each CPU graph split starts and
        // joins a fresh set of worker threads.
        let cpu_threadpool = CpuThreadpool::new(config.threads)?;
        let device_softmax =
            config.backend == Backend::Vulkan && config.execution_mode == ExecutionMode::Batched;
        let needs_generation_context =
            device_softmax && config.thinking.mode != ThinkingMode::Disabled;
        // llama.cpp can copy a hybrid/recurrent prefix to multiple sequence IDs, but advancing
        // multiple sequence groups in the same decode changes their recurrent results. Keep each
        // candidate fork serial. Pure KV-cache models retain shared-prefix fan-out and
        // cross-question batching.
        // SAFETY: model is live for the duration of the scorer.
        let isolates_candidate_forks = unsafe {
            sys::llama_model_is_recurrent(model.as_ptr())
                || sys::llama_model_is_hybrid(model.as_ptr())
        };
        let candidate_capacity = if isolates_candidate_forks {
            1
        } else {
            (config.max_sequences as usize).saturating_sub(1)
        };
        let device_target_width = candidate_capacity;
        let total_context = config
            .context_tokens_per_sequence
            .checked_mul(config.max_sequences)
            .ok_or_else(|| {
                JetError::InvalidRequest("context configuration overflows u32".to_owned())
            })?;
        // SAFETY: default parameter struct is initialized by llama.cpp.
        let mut context_params = unsafe { sys::llama_context_default_params() };
        context_params.n_ctx = total_context;
        context_params.n_batch = config.token_batch;
        context_params.n_ubatch = config.micro_batch;
        context_params.n_seq_max = config.max_sequences;
        context_params.n_outputs_max = config.max_output_rows;
        context_params.n_outputs_max_per_seq = config.max_output_rows;
        context_params.n_threads = config.threads;
        context_params.n_threads_batch = config.threads;
        context_params.embeddings = false;
        let offload = config.backend == Backend::Vulkan;
        // Native op_offload can upload CPU weights again for sufficiently large batches.
        // Explicit placement must keep their computation on CPU, including MoE prefill.
        let offload_cpu_ops = offload && config.gpu_layers.is_none() && config.cpu_moe_layers == 0;
        context_params.offload_kqv = offload;
        context_params.op_offload = offload_cpu_ops;
        context_params.kv_unified = true;

        let mut score_samplers: Vec<NonNull<sys::llama_sampler>> = Vec::new();
        let mut sampler_configs = Vec::new();
        if device_softmax {
            for sequence in 0..config.max_sequences {
                // SAFETY: the bridge returns a newly allocated llama sampler chain or null.
                let sampler =
                    NonNull::new(unsafe { sys::jet_score_sampler_init(device_target_width) })
                        .ok_or_else(|| {
                            for sampler in &score_samplers {
                                // SAFETY: every pointer came from jet_score_sampler_init and is owned here.
                                unsafe { sys::llama_sampler_free(sampler.as_ptr()) };
                            }
                            JetError::NativeRuntime(
                                "failed to create Vulkan score softmax sampler".to_owned(),
                            )
                        })?;
                score_samplers.push(sampler);
                sampler_configs.push(sys::llama_sampler_seq_config {
                    seq_id: sequence as i32,
                    sampler: sampler.as_ptr(),
                });
            }
            context_params.samplers = sampler_configs.as_mut_ptr();
            context_params.n_samplers = sampler_configs.len();
        }

        // SAFETY: model remains owned by the scorer and parameters are initialized.
        let context =
            NonNull::new(unsafe { sys::llama_init_from_model(model.as_ptr(), context_params) });
        let Some(context) = context else {
            for sampler in score_samplers {
                // SAFETY: context creation failed, so no native object retains these samplers.
                unsafe { sys::llama_sampler_free(sampler.as_ptr()) };
            }
            return Err(JetError::NativeRuntime(
                "failed to create llama context".to_owned(),
            ));
        };
        // SAFETY: context is live and the pool outlives it. Both normal and
        // batched CPU evaluation reuse the same workers.
        unsafe {
            sys::llama_attach_threadpool(
                context.as_ptr(),
                cpu_threadpool.0.as_ptr(),
                cpu_threadpool.0.as_ptr(),
            );
            sys::llama_set_causal_attn(context.as_ptr(), true);
        };

        let generation_context = if needs_generation_context {
            // Keep generation on the Vulkan model backend, but omit the device-side scoring
            // sampler so llama.cpp exposes the selected logits row to the CPU sampler.
            // This context shares model weights with the scoring context and owns only its
            // single-sequence runtime state (KV cache and compute buffers).
            // SAFETY: default parameter struct is initialized by llama.cpp.
            let mut generation_params = unsafe { sys::llama_context_default_params() };
            generation_params.n_ctx = config.context_tokens_per_sequence;
            generation_params.n_batch = config.token_batch;
            generation_params.n_ubatch = config.micro_batch;
            generation_params.n_seq_max = 1;
            generation_params.n_outputs_max = 1;
            generation_params.n_outputs_max_per_seq = 1;
            generation_params.n_threads = config.threads;
            generation_params.n_threads_batch = config.threads;
            generation_params.embeddings = false;
            generation_params.offload_kqv = offload;
            generation_params.op_offload = offload_cpu_ops;
            generation_params.kv_unified = true;

            // SAFETY: model remains live and generation_params contains no borrowed pointers.
            match NonNull::new(unsafe {
                sys::llama_init_from_model(model.as_ptr(), generation_params)
            }) {
                Some(generation_context) => {
                    // SAFETY: the pool outlives both contexts, and scoring and
                    // generation never execute concurrently.
                    unsafe {
                        sys::llama_attach_threadpool(
                            generation_context.as_ptr(),
                            cpu_threadpool.0.as_ptr(),
                            cpu_threadpool.0.as_ptr(),
                        );
                        sys::llama_set_causal_attn(generation_context.as_ptr(), true);
                    };
                    Some(generation_context)
                }
                None => {
                    // SAFETY: scoring context was created above and is still owned here.
                    unsafe { sys::llama_free(context.as_ptr()) };
                    for sampler in score_samplers {
                        // SAFETY: scoring context is gone, so these remain owned here.
                        unsafe { sys::llama_sampler_free(sampler.as_ptr()) };
                    }
                    return Err(JetError::NativeRuntime(
                        "failed to create llama generation context".to_owned(),
                    ));
                }
            }
        } else {
            None
        };

        let vocabulary = (|| {
            // SAFETY: the model is live; its vocab remains valid for the model lifetime.
            let vocab = NonNull::new(unsafe {
                sys::llama_model_get_vocab(model.as_ptr()) as *mut sys::llama_vocab
            })
            .ok_or_else(|| {
                JetError::NativeRuntime("model does not expose a vocabulary".to_owned())
            })?;
            // SAFETY: vocab is live.
            let vocab_size_i32 = unsafe { sys::llama_vocab_n_tokens(vocab.as_ptr()) };
            let vocab_size = usize::try_from(vocab_size_i32).map_err(|_| {
                JetError::NativeRuntime(format!("invalid vocabulary size {vocab_size_i32}"))
            })?;
            Ok::<_, JetError>((vocab, vocab_size))
        })();
        let (vocab, vocab_size) = match vocabulary {
            Ok(value) => value,
            Err(error) => {
                if let Some(generation_context) = generation_context {
                    // SAFETY: generation context was successfully created and is owned here.
                    unsafe { sys::llama_free(generation_context.as_ptr()) };
                }
                // SAFETY: context was successfully created and has not been transferred.
                unsafe { sys::llama_free(context.as_ptr()) };
                for sampler in score_samplers {
                    // SAFETY: the context is gone and the sampler remains owned by this function.
                    unsafe { sys::llama_sampler_free(sampler.as_ptr()) };
                }
                return Err(error);
            }
        };

        Ok(Self {
            model,
            context,
            generation_context,
            vocab,
            config,
            vocab_size,
            score_samplers,
            device_softmax,
            device_target_width,
            candidate_capacity,
            isolate_candidate_groups: isolates_candidate_forks,
            timings: EngineTimings::default(),
            _cpu_threadpool: cpu_threadpool,
            // load() transfers its still-live overrides here immediately on success.
            _tensor_buffer_overrides: TensorBufferOverrides::default(),
            _single_threaded: PhantomData,
        })
    }

    pub(crate) fn timings(&self) -> &EngineTimings {
        &self.timings
    }

    fn timing_start(&self) -> Option<Instant> {
        self.config.collect_timings.then(Instant::now)
    }

    fn tokenize_jobs(&mut self, jobs: &[ScoreJob]) -> Result<Vec<TokenizedJob>> {
        jobs.iter()
            .map(|job| {
                let (prompt_text, logical_prompt_tokens, thinking_tokens) =
                    self.prepare_prompt(job)?;
                let prompt = self.tokenize(&prompt_text)?;
                if prompt.is_empty() {
                    return Err(JetError::UnsupportedModel(
                        "chat template produced an empty prompt".to_owned(),
                    ));
                }
                let mut targets = Vec::with_capacity(job.targets.len());
                for target in &job.targets {
                    let combined = self.tokenize(&format!("{prompt_text}{target}"))?;
                    if !combined.starts_with(&prompt) {
                        return Err(JetError::UnsupportedModel(format!(
                            "tokenizer boundary is unstable for candidate {target:?}"
                        )));
                    }
                    let target_tokens = combined[prompt.len()..].to_vec();
                    if target_tokens.is_empty() {
                        return Err(JetError::InvalidRequest(format!(
                            "candidate {target:?} has no scoreable tokens"
                        )));
                    }
                    let required = prompt.len().saturating_add(target_tokens.len());
                    let limit = self.config.context_tokens_per_sequence as usize;
                    if required > limit {
                        return Err(JetError::ContextExceeded { required, limit });
                    }
                    targets.push(target_tokens);
                }
                Ok(TokenizedJob {
                    prompt,
                    targets,
                    logical_prompt_tokens,
                    thinking_tokens,
                })
            })
            .collect()
    }

    #[cfg(test)]
    fn render_prompt(&self, job: &ScoreJob) -> Result<String> {
        Ok(self.render_chat_plan(job, false, None)?.prompt)
    }

    fn prepare_prompt(&mut self, job: &ScoreJob) -> Result<(String, usize, usize)> {
        match self.config.thinking.mode {
            ThinkingMode::Disabled => {
                let disabled = self.render_chat_plan(job, false, None)?;
                let enabled = self.render_chat_plan(job, true, None)?;
                if thinking_is_open(&disabled)
                    || (enabled.supports_thinking
                        && enabled.prompt == disabled.prompt
                        && enabled.generation_prompt == disabled.generation_prompt)
                {
                    return Err(JetError::UnsupportedModel(
                        "the chat template does not expose a verified non-thinking path".to_owned(),
                    ));
                }
                let prompt_tokens = self.tokenize(&disabled.prompt)?.len();
                Ok((disabled.prompt, prompt_tokens, 0))
            }
            ThinkingMode::Auto | ThinkingMode::Required => {
                let plan = self.render_chat_plan(job, true, None)?;
                if !plan.supports_thinking || plan.thinking_end_tags.is_empty() {
                    if self.config.thinking.mode == ThinkingMode::Required {
                        return Err(JetError::UnsupportedModel(
                            "the chat template does not expose a supported thinking protocol"
                                .to_owned(),
                        ));
                    }
                    let disabled = self.render_chat_plan(job, false, None)?;
                    let prompt_tokens = self.tokenize(&disabled.prompt)?.len();
                    return Ok((disabled.prompt, prompt_tokens, 0));
                }

                let logical_prompt_tokens = self.tokenize(&plan.prompt)?.len();
                let started = self.timing_start();
                let cache_before = self.timings.cache_ms;
                let generated = self.generate_reasoning(&plan);
                if started.is_some() {
                    self.timings.thinking_ms +=
                        (elapsed_ms(started) - (self.timings.cache_ms - cache_before)).max(0.0);
                }
                let (reasoning, thinking_tokens) = generated?;
                let final_plan = self.render_chat_plan(job, true, Some(&reasoning))?;
                if final_plan.prompt.is_empty()
                    || final_plan.prompt == plan.prompt
                    || thinking_is_open(&final_plan)
                {
                    return Err(JetError::UnsupportedModel(
                        "chat template did not produce a verified final-answer continuation"
                            .to_owned(),
                    ));
                }
                Ok((final_plan.prompt, logical_prompt_tokens, thinking_tokens))
            }
        }
    }

    fn render_chat_plan(
        &self,
        job: &ScoreJob,
        enable_thinking: bool,
        reasoning: Option<&str>,
    ) -> Result<ChatPlan> {
        let system = CString::new(job.system_content).map_err(|_| {
            JetError::InvalidRequest("system content contains an interior NUL byte".to_owned())
        })?;
        let user = CString::new(job.user_content.as_str()).map_err(|_| {
            JetError::InvalidRequest("user content contains an interior NUL byte".to_owned())
        })?;
        let reasoning = reasoning.map(CString::new).transpose().map_err(|_| {
            JetError::NativeRuntime("generated reasoning contains an interior NUL byte".to_owned())
        })?;
        let reasoning_ptr = reasoning
            .as_ref()
            .map_or(ptr::null(), |value| value.as_ptr());
        let mut error = vec![0_i8; 1_024];
        // SAFETY: all pointers are live for the duration of the call; null output requests size.
        let required = unsafe {
            sys::jet_chat_render(
                self.model.as_ptr(),
                system.as_ptr(),
                user.as_ptr(),
                enable_thinking,
                reasoning_ptr,
                ptr::null_mut(),
                0,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if required < 0 {
            return Err(template_error(&error));
        }
        let required = usize::try_from(required)
            .map_err(|_| JetError::NativeRuntime("chat template length overflow".to_owned()))?;
        let mut output = vec![0_i8; required.saturating_add(1)];
        // SAFETY: output and error buffers are writable and model/messages remain live.
        let written = unsafe {
            sys::jet_chat_render(
                self.model.as_ptr(),
                system.as_ptr(),
                user.as_ptr(),
                enable_thinking,
                reasoning_ptr,
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if written < 0 {
            return Err(template_error(&error));
        }
        let written = usize::try_from(written)
            .map_err(|_| JetError::NativeRuntime("chat template length overflow".to_owned()))?;
        let bytes = output
            .get(..written)
            .ok_or_else(|| {
                JetError::NativeRuntime("chat template overflowed its buffer".to_owned())
            })?
            .iter()
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        let json = String::from_utf8(bytes).map_err(|error| {
            JetError::NativeRuntime(format!("chat template emitted invalid UTF-8: {error}"))
        })?;
        serde_json::from_str(&json).map_err(JetError::from)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<sys::llama_token>> {
        let text_len = i32::try_from(text.len()).map_err(|_| {
            JetError::InvalidRequest("text is too large for llama.cpp tokenization".to_owned())
        })?;
        // SAFETY: text bytes are valid for text_len; null output with zero capacity queries size.
        let count = unsafe {
            sys::llama_tokenize(
                self.vocab.as_ptr(),
                text.as_ptr().cast(),
                text_len,
                ptr::null_mut(),
                0,
                true,
                true,
            )
        };
        if count == i32::MIN {
            return Err(JetError::NativeRuntime(
                "tokenization length overflow".to_owned(),
            ));
        }
        let capacity = if count < 0 { -count } else { count };
        let capacity = usize::try_from(capacity)
            .map_err(|_| JetError::NativeRuntime("invalid tokenizer size response".to_owned()))?;
        let mut tokens = vec![0; capacity];
        // SAFETY: tokens has capacity elements and text/vocab remain valid.
        let written = unsafe {
            sys::llama_tokenize(
                self.vocab.as_ptr(),
                text.as_ptr().cast(),
                text_len,
                tokens.as_mut_ptr(),
                i32::try_from(tokens.len()).unwrap_or(i32::MAX),
                true,
                true,
            )
        };
        if written < 0 {
            return Err(JetError::NativeRuntime(format!(
                "tokenizer buffer was unexpectedly too small: needs {} tokens",
                -written
            )));
        }
        tokens.truncate(usize::try_from(written).unwrap_or(0));
        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[sys::llama_token]) -> Result<String> {
        let token_count = i32::try_from(tokens.len()).map_err(|_| {
            JetError::NativeRuntime("generated reasoning exceeds tokenizer limits".to_owned())
        })?;
        // SAFETY: the vocabulary and token slice are live; a null output queries the size.
        let required = unsafe {
            sys::llama_detokenize(
                self.vocab.as_ptr(),
                tokens.as_ptr(),
                token_count,
                ptr::null_mut(),
                0,
                false,
                true,
            )
        };
        if required == i32::MIN {
            return Err(JetError::NativeRuntime(
                "detokenization length overflow".to_owned(),
            ));
        }
        let capacity = if required < 0 { -required } else { required };
        let mut output = vec![
            0_i8;
            usize::try_from(capacity).map_err(|_| {
                JetError::NativeRuntime("invalid detokenizer size response".to_owned())
            })?
        ];
        // SAFETY: output has the queried capacity and all other pointers remain live.
        let written = unsafe {
            sys::llama_detokenize(
                self.vocab.as_ptr(),
                tokens.as_ptr(),
                token_count,
                output.as_mut_ptr(),
                i32::try_from(output.len()).unwrap_or(i32::MAX),
                false,
                true,
            )
        };
        if written < 0 {
            return Err(JetError::NativeRuntime(format!(
                "detokenizer buffer was unexpectedly too small: needs {} bytes",
                -written
            )));
        }
        output.truncate(usize::try_from(written).unwrap_or(0));
        let bytes = output.into_iter().map(|byte| byte as u8).collect();
        String::from_utf8(bytes).map_err(|error| {
            JetError::NativeRuntime(format!("generated reasoning is invalid UTF-8: {error}"))
        })
    }

    fn generate_reasoning(&mut self, plan: &ChatPlan) -> Result<(String, usize)> {
        let prompt = self.tokenize(&plan.prompt)?;
        if prompt.is_empty() {
            return Err(JetError::UnsupportedModel(
                "thinking template produced an empty prompt".to_owned(),
            ));
        }
        let limit = self.config.context_tokens_per_sequence as usize;
        let budget = self.config.thinking.max_tokens as usize;
        if prompt.len().saturating_add(budget) > limit {
            return Err(JetError::ContextExceeded {
                required: prompt.len().saturating_add(budget),
                limit,
            });
        }

        let end_tags_json = serde_json::to_string(&plan.thinking_end_tags)?;
        let end_tags_json = CString::new(end_tags_json).map_err(|_| {
            JetError::NativeRuntime("thinking end tags contain an interior NUL byte".to_owned())
        })?;
        let mut error = vec![0_i8; 1_024];
        // SAFETY: all arguments are live and the wrapper returns an owned sampler or null.
        let sampler = NonNull::new(unsafe {
            sys::jet_thinking_sampler_init(
                self.vocab.as_ptr(),
                end_tags_json.as_ptr(),
                i32::try_from(self.config.thinking.max_tokens).unwrap_or(i32::MAX),
                self.config.thinking.temperature,
                self.config.thinking.top_k,
                self.config.thinking.top_p,
                self.config.thinking.seed,
                error.as_mut_ptr(),
                error.len(),
            )
        })
        .ok_or_else(|| template_error(&error))?;
        let sampler = ThinkingSampler(sampler);

        let generation = (|| {
            let mut logits_index = self.decode_generation_input(&prompt, 0)?;
            let mut generated = Vec::with_capacity(budget.saturating_add(16));
            let hard_limit = budget
                .saturating_add(128)
                .min(limit.saturating_sub(prompt.len()));
            while generated.len() < hard_limit {
                // SAFETY: the sampler and context are live and logits_index was selected by the most recent decode.
                let mut token = unsafe {
                    sys::jet_thinking_sampler_sample(
                        sampler.0.as_ptr(),
                        self.generation_context().as_ptr(),
                        logits_index,
                    )
                };
                // Do not place EOG inside an unfinished reasoning block. Ask the budget sampler
                // to select its protocol-specific close sequence from the same logits instead.
                if unsafe { sys::llama_vocab_is_eog(self.vocab.as_ptr(), token) }
                    && !unsafe { sys::jet_thinking_sampler_done(sampler.0.as_ptr()) }
                {
                    // SAFETY: the sampler is live and currently owns the active budget state.
                    if !unsafe { sys::jet_thinking_sampler_force(sampler.0.as_ptr()) } {
                        return Err(JetError::UnsupportedModel(
                            "model ended before completing its thinking protocol".to_owned(),
                        ));
                    }
                    // SAFETY: forcing changes only the sampler state; the logits row is still live.
                    token = unsafe {
                        sys::jet_thinking_sampler_sample(
                            sampler.0.as_ptr(),
                            self.generation_context().as_ptr(),
                            logits_index,
                        )
                    };
                }

                // SAFETY: the sampler is live and accepts the sampled token exactly once.
                unsafe { sys::jet_thinking_sampler_accept(sampler.0.as_ptr(), token) };
                generated.push(token);
                // SAFETY: the sampler remains live for the duration of generation.
                if unsafe { sys::jet_thinking_sampler_done(sampler.0.as_ptr()) } {
                    return Ok(generated);
                }
                if prompt.len().saturating_add(generated.len()) >= limit {
                    return Err(JetError::ContextExceeded {
                        required: prompt
                            .len()
                            .saturating_add(generated.len())
                            .saturating_add(1),
                        limit,
                    });
                }
                logits_index = self.decode_generation_input(
                    std::slice::from_ref(&token),
                    prompt.len() + generated.len() - 1,
                )?;
            }
            Err(JetError::NativeRuntime(
                "thinking sampler did not complete its protocol within the bounded closure allowance"
                    .to_owned(),
            ))
        })();
        self.clear_generation_memory();
        let generated = generation?;
        let rendered = self.detokenize(&generated)?;
        let reasoning =
            strip_thinking_end(&rendered, &plan.thinking_end_tags).ok_or_else(|| {
                JetError::UnsupportedModel(
                    "generated reasoning did not end with a declared template marker".to_owned(),
                )
            })?;
        Ok((reasoning.to_owned(), generated.len()))
    }

    fn decode_generation_input(
        &mut self,
        tokens: &[sys::llama_token],
        start: usize,
    ) -> Result<i32> {
        let chunk_size = self.config.token_batch as usize;
        let mut logits_index = None;
        for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
            if chunk.is_empty() {
                continue;
            }
            let chunk_start = start + chunk_index * chunk_size;
            // SAFETY: llama_batch_init allocates storage for chunk.len() entries.
            let mut batch = unsafe {
                sys::llama_batch_init(i32::try_from(chunk.len()).unwrap_or(i32::MAX), 0, 1)
            };
            let result = (|| {
                if batch.token.is_null()
                    || batch.pos.is_null()
                    || batch.n_seq_id.is_null()
                    || batch.seq_id.is_null()
                    || batch.logits.is_null()
                {
                    return Err(JetError::NativeRuntime(
                        "llama_batch_init returned incomplete generation storage".to_owned(),
                    ));
                }
                batch.n_tokens = i32::try_from(chunk.len()).map_err(|_| {
                    JetError::NativeRuntime("native generation batch length overflow".to_owned())
                })?;
                for (index, token) in chunk.iter().copied().enumerate() {
                    // SAFETY: all arrays were allocated for chunk.len() elements.
                    unsafe {
                        *batch.token.add(index) = token;
                        *batch.pos.add(index) = position(chunk_start + index)?;
                        *batch.n_seq_id.add(index) = 1;
                        let ids = *batch.seq_id.add(index);
                        if ids.is_null() {
                            return Err(JetError::NativeRuntime(
                                "native generation sequence storage is null".to_owned(),
                            ));
                        }
                        *ids = 0;
                        *batch.logits.add(index) = i8::from(index + 1 == chunk.len());
                    }
                }
                // SAFETY: context and initialized batch remain live for the call.
                let status =
                    unsafe { sys::llama_decode(self.generation_context().as_ptr(), batch) };
                if self.config.collect_timings {
                    self.timings.decode_calls += 1;
                }
                if status != 0 {
                    return Err(JetError::NativeRuntime(format!(
                        "llama_decode failed during thinking with status {status}"
                    )));
                }
                Ok(())
            })();
            // SAFETY: batch was allocated above and is freed exactly once.
            unsafe { sys::llama_batch_free(batch) };
            result?;
            logits_index = Some(i32::try_from(chunk.len() - 1).map_err(|_| {
                JetError::NativeRuntime("generation logits index overflow".to_owned())
            })?);
        }
        logits_index.ok_or_else(|| {
            JetError::NativeRuntime("cannot decode an empty generation input".to_owned())
        })
    }

    fn score_batched(&mut self, jobs: &[TokenizedJob]) -> Result<Vec<ScoreResult>> {
        if self.isolate_candidate_groups {
            return self.score_serial_forks(jobs);
        }
        let candidate_capacity = self.candidate_capacity;
        let mut groups = Vec::new();
        for (job_index, job) in jobs.iter().enumerate() {
            for candidate_start in (0..job.targets.len()).step_by(candidate_capacity) {
                groups.push(GroupSpec {
                    job_index,
                    candidate_start,
                    candidate_end: (candidate_start + candidate_capacity).min(job.targets.len()),
                });
            }
        }

        let mut waves: Vec<Vec<GroupSpec>> = Vec::new();
        let mut current = Vec::new();
        let mut sequences_used = 0_usize;
        for group in groups {
            let needed = 1 + group.candidate_end - group.candidate_start;
            if sequences_used + needed > self.config.max_sequences as usize && !current.is_empty() {
                waves.push(std::mem::take(&mut current));
                sequences_used = 0;
            }
            current.push(group);
            sequences_used += needed;
        }
        if !current.is_empty() {
            waves.push(current);
        }

        let mut accumulated: Vec<Vec<f64>> = jobs
            .iter()
            .map(|job| vec![0.0; job.targets.len()])
            .collect();
        let mut prefill_counts = vec![0_usize; jobs.len()];
        for wave in waves {
            self.process_wave(jobs, &wave, &mut accumulated, &mut prefill_counts)?;
        }

        Ok(jobs
            .iter()
            .enumerate()
            .map(|(index, job)| ScoreResult {
                log_probabilities: accumulated[index].clone(),
                prompt_tokens: job.logical_prompt_tokens,
                thinking_tokens: job.thinking_tokens,
                target_token_counts: job.targets.iter().map(Vec::len).collect(),
                prefill_count: prefill_counts[index],
            })
            .collect())
    }

    fn ensure_target_width(&mut self, jobs: &[TokenizedJob]) -> Result<()> {
        if !self.device_softmax {
            return Ok(());
        }
        let width = jobs
            .iter()
            .map(|job| prefix_targets(&job.targets).len())
            .max()
            .unwrap_or(0);
        if width <= self.device_target_width {
            return Ok(());
        }
        // SAFETY: samplers are owned here. Finish any outstanding work before changing graph inputs.
        unsafe { sys::llama_synchronize(self.context.as_ptr()) };
        for (sequence, sampler) in self.score_samplers.iter_mut().enumerate() {
            // SAFETY: the bridge creates an owned, uninitialized chain. Existing chains cannot
            // be rebound because llama.cpp permits backend initialization only once per chain.
            let replacement = NonNull::new(unsafe { sys::jet_score_sampler_init(width) })
                .ok_or_else(|| {
                    JetError::NativeRuntime("failed to allocate Vulkan score targets".to_owned())
                })?;
            // SAFETY: context and replacement are live; the next decode rebuilds the sampling graph.
            if !unsafe {
                sys::llama_set_sampler(self.context.as_ptr(), sequence as i32, replacement.as_ptr())
            } {
                // SAFETY: the context did not retain this unsupported sampler.
                unsafe { sys::llama_sampler_free(replacement.as_ptr()) };
                return Err(JetError::NativeRuntime(
                    "failed to resize Vulkan score targets".to_owned(),
                ));
            }
            let previous = std::mem::replace(sampler, replacement);
            // SAFETY: pending work was synchronized and the context now uses replacement.
            unsafe { sys::llama_sampler_free(previous.as_ptr()) };
        }
        self.device_target_width = width;
        Ok(())
    }

    fn score_serial_forks(&mut self, jobs: &[TokenizedJob]) -> Result<Vec<ScoreResult>> {
        self.ensure_target_width(jobs)?;
        let mut accumulated: Vec<Vec<f64>> = jobs
            .iter()
            .map(|job| vec![0.0; job.targets.len()])
            .collect();
        let mut results = Vec::with_capacity(jobs.len());
        for (job_index, job) in jobs.iter().enumerate() {
            let result: Result<ScoreResult> = (|| {
                let groups = [ActiveGroup {
                    spec: GroupSpec {
                        job_index,
                        candidate_start: 0,
                        candidate_end: job.targets.len(),
                    },
                    prefix_sequence: 0,
                    candidate_sequences: Vec::new(),
                }];
                let prefix_items = job
                    .prompt
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, token)| {
                        Ok(DecodeItem {
                            token,
                            position: position(index)?,
                            sequence: 0,
                            output: if index + 1 == job.prompt.len() {
                                OutputAction::Prefix(0)
                            } else {
                                OutputAction::None
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.decode_prefill(&prefix_items, jobs, &groups, &mut accumulated)?;

                for (candidate_index, tokens) in job.targets.iter().enumerate() {
                    // A one-token candidate was completely scored by the shared prefix output.
                    if tokens.len() == 1 {
                        continue;
                    }
                    // Keep sequence 0 immutable. Advancing only sequence 1 lets recurrent memory
                    // copy its shared source state without mixing several candidate forks.
                    self.copy_prefix(0, 1)?;
                    let suffix_items = (0..tokens.len() - 1)
                        .map(|index| {
                            Ok(DecodeItem {
                                token: tokens[index],
                                position: position(job.prompt.len() + index)?,
                                sequence: 1,
                                output: OutputAction::Target {
                                    job_index,
                                    candidate_index,
                                    target: tokens[index + 1],
                                },
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.decode_candidates(&suffix_items, jobs, &groups, &mut accumulated)?;
                }
                Ok(ScoreResult {
                    log_probabilities: accumulated[job_index].clone(),
                    prompt_tokens: job.logical_prompt_tokens,
                    thinking_tokens: job.thinking_tokens,
                    target_token_counts: job.targets.iter().map(Vec::len).collect(),
                    prefill_count: 1,
                })
            })();
            self.clear_memory();
            results.push(result?);
        }
        Ok(results)
    }

    fn copy_prefix(
        &mut self,
        source: sys::llama_seq_id,
        destination: sys::llama_seq_id,
    ) -> Result<()> {
        let started = self.timing_start();
        // SAFETY: these sequence IDs are within the live context's configured capacity.
        let memory = unsafe { sys::llama_get_memory(self.context.as_ptr()) };
        // SAFETY: removing the entire destination is supported for both KV and recurrent state.
        let removed = unsafe { sys::llama_memory_seq_rm(memory, destination, -1, -1) };
        if removed {
            // SAFETY: source remains live; only the destination will be advanced by the caller.
            unsafe { sys::llama_memory_seq_cp(memory, source, destination, -1, -1) };
        }
        self.timings.cache_ms += elapsed_ms(started);
        if !removed {
            return Err(JetError::NativeRuntime(
                "failed to reset candidate sequence".to_owned(),
            ));
        }
        Ok(())
    }

    fn decode_prefill(
        &mut self,
        items: &[DecodeItem],
        jobs: &[TokenizedJob],
        groups: &[ActiveGroup],
        accumulated: &mut [Vec<f64>],
    ) -> Result<()> {
        let started = self.timing_start();
        if self.config.collect_timings {
            self.timings.prefill_tokens += items.len() as u64;
            self.timings.prefill_count += groups.len() as u64;
        }
        let result = self.decode_items(items, jobs, groups, accumulated);
        // Reading the selected final prompt output synchronizes the backend, including earlier
        // chunks without outputs. Do not insert a synchronization between those chunks.
        self.timings.prefill_ms += elapsed_ms(started);
        result
    }

    fn decode_candidates(
        &mut self,
        items: &[DecodeItem],
        jobs: &[TokenizedJob],
        groups: &[ActiveGroup],
        accumulated: &mut [Vec<f64>],
    ) -> Result<()> {
        let started = self.timing_start();
        if self.config.collect_timings {
            self.timings.candidate_tokens += items.len() as u64;
        }
        let result = self.decode_items(items, jobs, groups, accumulated);
        self.timings.candidate_ms += elapsed_ms(started);
        result
    }

    fn process_wave(
        &mut self,
        jobs: &[TokenizedJob],
        specs: &[GroupSpec],
        accumulated: &mut [Vec<f64>],
        prefill_counts: &mut [usize],
    ) -> Result<()> {
        let result = (|| {
            let mut next_sequence: sys::llama_seq_id = 0;
            let mut groups = Vec::with_capacity(specs.len());
            #[allow(clippy::explicit_counter_loop)]
            for spec in specs {
                let prefix_sequence = next_sequence;
                next_sequence += 1;
                let candidate_sequences = (spec.candidate_start..spec.candidate_end)
                    .map(|_| {
                        let sequence = next_sequence;
                        next_sequence += 1;
                        sequence
                    })
                    .collect();
                groups.push(ActiveGroup {
                    spec: *spec,
                    prefix_sequence,
                    candidate_sequences,
                });
                prefill_counts[spec.job_index] += 1;
            }

            let mut prefix_items = Vec::new();
            for (group_index, group) in groups.iter().enumerate() {
                let prompt = &jobs[group.spec.job_index].prompt;
                for (index, token) in prompt.iter().copied().enumerate() {
                    prefix_items.push(DecodeItem {
                        token,
                        position: position(index)?,
                        sequence: group.prefix_sequence,
                        output: if index + 1 == prompt.len() {
                            OutputAction::Prefix(group_index)
                        } else {
                            OutputAction::None
                        },
                    });
                }
            }
            self.decode_prefill(&prefix_items, jobs, &groups, accumulated)?;

            for group in &groups {
                for sequence in &group.candidate_sequences {
                    self.copy_prefix(group.prefix_sequence, *sequence)?;
                }
            }

            let mut suffix_items = Vec::new();
            for group in &groups {
                let job = &jobs[group.spec.job_index];
                for (offset, candidate_index) in
                    (group.spec.candidate_start..group.spec.candidate_end).enumerate()
                {
                    let tokens = &job.targets[candidate_index];
                    for target_index in 0..tokens.len().saturating_sub(1) {
                        suffix_items.push(DecodeItem {
                            token: tokens[target_index],
                            position: position(job.prompt.len() + target_index)?,
                            sequence: group.candidate_sequences[offset],
                            output: OutputAction::Target {
                                job_index: group.spec.job_index,
                                candidate_index,
                                target: tokens[target_index + 1],
                            },
                        });
                    }
                }
            }
            self.decode_candidates(&suffix_items, jobs, &groups, accumulated)
        })();
        self.clear_memory();
        result
    }

    fn decode_items(
        &mut self,
        items: &[DecodeItem],
        jobs: &[TokenizedJob],
        groups: &[ActiveGroup],
        accumulated: &mut [Vec<f64>],
    ) -> Result<()> {
        let chunk_size = self.config.token_batch.min(self.config.max_output_rows) as usize;
        for chunk in items.chunks(chunk_size) {
            if chunk.is_empty() {
                continue;
            }
            // SAFETY: llama_batch_init allocates arrays sized for chunk.len() entries.
            let mut batch = unsafe {
                sys::llama_batch_init(i32::try_from(chunk.len()).unwrap_or(i32::MAX), 0, 1)
            };
            let decode_result = (|| {
                if batch.token.is_null()
                    || batch.pos.is_null()
                    || batch.n_seq_id.is_null()
                    || batch.seq_id.is_null()
                    || batch.logits.is_null()
                {
                    return Err(JetError::NativeRuntime(
                        "llama_batch_init returned incomplete storage".to_owned(),
                    ));
                }
                batch.n_tokens = i32::try_from(chunk.len()).map_err(|_| {
                    JetError::NativeRuntime("native batch length overflow".to_owned())
                })?;
                for (index, item) in chunk.iter().enumerate() {
                    // SAFETY: every batch array was allocated for at least chunk.len() entries.
                    unsafe {
                        *batch.token.add(index) = item.token;
                        *batch.pos.add(index) = item.position;
                        *batch.n_seq_id.add(index) = 1;
                        let ids = *batch.seq_id.add(index);
                        if ids.is_null() {
                            return Err(JetError::NativeRuntime(
                                "native batch sequence storage is null".to_owned(),
                            ));
                        }
                        *ids = item.sequence;
                        *batch.logits.add(index) =
                            i8::from(!matches!(item.output, OutputAction::None));
                    }
                }

                if self.device_softmax {
                    let mut targets = vec![Vec::new(); self.score_samplers.len()];
                    for item in chunk {
                        match item.output {
                            OutputAction::None => {}
                            OutputAction::Prefix(group_index) => {
                                let group = groups.get(group_index).ok_or_else(|| {
                                    JetError::NativeRuntime(
                                        "missing active prefix group".to_owned(),
                                    )
                                })?;
                                let job = &jobs[group.spec.job_index];
                                let range = group.spec.candidate_start..group.spec.candidate_end;
                                let sequence = usize::try_from(item.sequence).map_err(|_| {
                                    JetError::NativeRuntime(
                                        "score sequence ID is negative".to_owned(),
                                    )
                                })?;
                                let sequence_targets =
                                    targets.get_mut(sequence).ok_or_else(|| {
                                        JetError::NativeRuntime(
                                            "score sequence ID exceeds sampler count".to_owned(),
                                        )
                                    })?;
                                sequence_targets.extend(prefix_targets(&job.targets[range]));
                                if sequence_targets.len() > self.device_target_width {
                                    return Err(JetError::NativeRuntime(
                                        "prefix targets exceed the device gather width".to_owned(),
                                    ));
                                }
                                let padding = sequence_targets.last().copied().unwrap_or_default();
                                sequence_targets.resize(self.device_target_width, padding);
                            }
                            OutputAction::Target { target, .. } => {
                                let sequence = usize::try_from(item.sequence).map_err(|_| {
                                    JetError::NativeRuntime(
                                        "score sequence ID is negative".to_owned(),
                                    )
                                })?;
                                let sequence_targets =
                                    targets.get_mut(sequence).ok_or_else(|| {
                                        JetError::NativeRuntime(
                                            "score sequence ID exceeds sampler count".to_owned(),
                                        )
                                    })?;
                                sequence_targets.push(target);
                                sequence_targets.resize(
                                    sequence_targets.len() + self.device_target_width - 1,
                                    target,
                                );
                            }
                        }
                    }
                    for (sampler, targets) in self.score_samplers.iter().zip(&targets) {
                        let target_ptr = if targets.is_empty() {
                            ptr::null()
                        } else {
                            targets.as_ptr()
                        };
                        // SAFETY: sampler is owned by the scorer and target_ptr is live for the call.
                        if !unsafe {
                            sys::jet_score_sampler_set_targets(
                                sampler.as_ptr(),
                                target_ptr,
                                targets.len(),
                            )
                        } {
                            return Err(JetError::NativeRuntime(
                                "failed to set Vulkan score targets".to_owned(),
                            ));
                        }
                    }
                }

                // SAFETY: context and batch are live; all positions and sequence IDs were initialized.
                let status = unsafe { sys::llama_decode(self.context.as_ptr(), batch) };
                if self.config.collect_timings {
                    self.timings.decode_calls += 1;
                }
                if status != 0 {
                    return Err(JetError::NativeRuntime(format!(
                        "llama_decode failed with status {status}"
                    )));
                }
                // SAFETY: llama_decode succeeded; selected output rows remain valid until the next decode.
                let logits = if self.device_softmax {
                    ptr::null_mut()
                } else {
                    unsafe { sys::llama_get_logits(self.context.as_ptr()) }
                };
                if !self.device_softmax
                    && logits.is_null()
                    && chunk
                        .iter()
                        .any(|item| !matches!(item.output, OutputAction::None))
                {
                    return Err(JetError::NativeRuntime(
                        "llama_decode did not expose requested logits".to_owned(),
                    ));
                }

                let mut output_row = 0_usize;
                let output_count = chunk
                    .iter()
                    .filter(|item| !matches!(item.output, OutputAction::None))
                    .count();
                for item in chunk {
                    match item.output {
                        OutputAction::None => {}
                        OutputAction::Prefix(group_index) => {
                            let distribution =
                                self.output_distribution(logits, output_row, output_count)?;
                            let group = groups.get(group_index).ok_or_else(|| {
                                JetError::NativeRuntime("missing active prefix group".to_owned())
                            })?;
                            let job = &jobs[group.spec.job_index];
                            let range = group.spec.candidate_start..group.spec.candidate_end;
                            let first_tokens = prefix_targets(&job.targets[range.clone()]);
                            for (target, total) in job.targets[range.clone()]
                                .iter()
                                .zip(&mut accumulated[group.spec.job_index][range])
                            {
                                let target_index =
                                    first_tokens.binary_search(&target[0]).map_err(|_| {
                                        JetError::NativeRuntime(
                                            "prefix target was not gathered".to_owned(),
                                        )
                                    })?;
                                *total +=
                                    distribution.target_log_probability(target[0], target_index)?;
                            }
                            output_row += 1;
                        }
                        OutputAction::Target {
                            job_index,
                            candidate_index,
                            target,
                        } => {
                            let distribution =
                                self.output_distribution(logits, output_row, output_count)?;
                            accumulated[job_index][candidate_index] +=
                                distribution.target_log_probability(target, 0)?;
                            output_row += 1;
                        }
                    }
                }
                Ok(())
            })();
            // SAFETY: batch was allocated by llama_batch_init and is freed exactly once.
            unsafe { sys::llama_batch_free(batch) };
            decode_result?;
        }
        Ok(())
    }

    fn logits_row(&self, logits: *mut f32, row: usize) -> &[f32] {
        // SAFETY: caller only requests rows selected in the most recent successful decode.
        unsafe { slice::from_raw_parts(logits.add(row * self.vocab_size), self.vocab_size) }
    }

    fn output_distribution(
        &self,
        logits: *mut f32,
        row: usize,
        output_count: usize,
    ) -> Result<OutputDistribution<'_>> {
        if !self.device_softmax {
            return Ok(OutputDistribution::Logits(self.logits_row(logits, row)));
        }
        let row = i32::try_from(row).map_err(|_| {
            JetError::NativeRuntime("Vulkan score output row exceeds i32".to_owned())
        })?;
        let output_count = i32::try_from(output_count).map_err(|_| {
            JetError::NativeRuntime("Vulkan score output count exceeds i32".to_owned())
        })?;
        let sampled_index = row - output_count;
        // SAFETY: row identifies an output selected in the most recent successful decode.
        let count =
            unsafe { sys::llama_get_sampled_probs_count_ith(self.context.as_ptr(), sampled_index) };
        if count as usize != self.device_target_width {
            return Err(JetError::NativeRuntime(format!(
                "Vulkan score softmax returned {count} values instead of {} target log-probabilities",
                self.device_target_width
            )));
        }
        // SAFETY: the count above describes this row, which remains live until the next decode.
        let probabilities =
            unsafe { sys::llama_get_sampled_probs_ith(self.context.as_ptr(), sampled_index) };
        if probabilities.is_null() {
            return Err(JetError::NativeRuntime(
                "Vulkan score softmax did not expose probabilities".to_owned(),
            ));
        }
        // SAFETY: llama.cpp reported device_target_width contiguous f32 values.
        Ok(OutputDistribution::DeviceLogProbabilities(unsafe {
            slice::from_raw_parts(probabilities, self.device_target_width)
        }))
    }

    fn score_reference(&mut self, jobs: &[TokenizedJob]) -> Result<Vec<ScoreResult>> {
        let mut results = Vec::with_capacity(jobs.len());
        for job in jobs {
            let mut log_probabilities = Vec::with_capacity(job.targets.len());
            for target in &job.targets {
                let result = (|| {
                    let mut prompt_items = Vec::with_capacity(job.prompt.len());
                    for (index, token) in job.prompt.iter().copied().enumerate() {
                        prompt_items.push(DecodeItem {
                            token,
                            position: position(index)?,
                            sequence: 0,
                            output: if index + 1 == job.prompt.len() {
                                OutputAction::Prefix(0)
                            } else {
                                OutputAction::None
                            },
                        });
                    }
                    let reference_job = TokenizedJob {
                        prompt: job.prompt.clone(),
                        targets: vec![target.clone()],
                        logical_prompt_tokens: job.logical_prompt_tokens,
                        thinking_tokens: job.thinking_tokens,
                    };
                    let groups = [ActiveGroup {
                        spec: GroupSpec {
                            job_index: 0,
                            candidate_start: 0,
                            candidate_end: 1,
                        },
                        prefix_sequence: 0,
                        candidate_sequences: Vec::new(),
                    }];
                    let mut accumulated = vec![vec![0.0]];
                    self.decode_prefill(
                        &prompt_items,
                        std::slice::from_ref(&reference_job),
                        &groups,
                        &mut accumulated,
                    )?;

                    let mut suffix_items = Vec::with_capacity(target.len().saturating_sub(1));
                    for target_index in 0..target.len().saturating_sub(1) {
                        suffix_items.push(DecodeItem {
                            token: target[target_index],
                            position: position(job.prompt.len() + target_index)?,
                            sequence: 0,
                            output: OutputAction::Target {
                                job_index: 0,
                                candidate_index: 0,
                                target: target[target_index + 1],
                            },
                        });
                    }
                    self.decode_candidates(
                        &suffix_items,
                        &[reference_job],
                        &groups,
                        &mut accumulated,
                    )?;
                    Ok::<_, JetError>(accumulated[0][0])
                })();
                self.clear_memory();
                log_probabilities.push(result?);
            }
            results.push(ScoreResult {
                log_probabilities,
                prompt_tokens: job.logical_prompt_tokens,
                thinking_tokens: job.thinking_tokens,
                target_token_counts: job.targets.iter().map(Vec::len).collect(),
                prefill_count: job.targets.len(),
            });
        }
        Ok(results)
    }

    fn clear_memory(&mut self) {
        let started = self.timing_start();
        // SAFETY: scoring has finished. Reset sequence metadata; new recurrent states are
        // initialized by llama.cpp's graph, and attention masks exclude the old KV entries.
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.context.as_ptr()), false) };
        self.timings.cache_ms += elapsed_ms(started);
    }

    fn generation_context(&self) -> NonNull<sys::llama_context> {
        self.generation_context.unwrap_or(self.context)
    }

    fn clear_generation_memory(&mut self) {
        let started = self.timing_start();
        let context = self.generation_context();
        // SAFETY: generation has finished; subsequent decoding initializes fresh sequence state.
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(context.as_ptr()), false) };
        self.timings.cache_ms += elapsed_ms(started);
    }
}

impl SequenceScorer for LlamaScorer {
    fn score_batch(&mut self, jobs: &[ScoreJob]) -> Vec<Result<ScoreResult>> {
        let batch_started = self.timing_start();
        let prepare_started = self.timing_start();
        let thinking_before = self.timings.thinking_ms;
        let cache_before = self.timings.cache_ms;
        if self.config.collect_timings {
            self.timings.batches += 1;
            self.timings.jobs += jobs.len() as u64;
        }
        let mut slots: Vec<Option<Result<ScoreResult>>> = (0..jobs.len()).map(|_| None).collect();
        let mut tokenized = Vec::new();
        let mut valid_indices = Vec::new();
        for (index, job) in jobs.iter().enumerate() {
            match self.tokenize_jobs(std::slice::from_ref(job)) {
                Ok(mut value) => {
                    if let Some(value) = value.pop() {
                        valid_indices.push(index);
                        tokenized.push(value);
                    } else {
                        slots[index] = Some(Err(JetError::NativeRuntime(
                            "tokenizer omitted a scoring job".to_owned(),
                        )));
                    }
                }
                Err(error) => slots[index] = Some(Err(error)),
            }
        }

        if prepare_started.is_some() {
            self.timings.prepare_ms += (elapsed_ms(prepare_started)
                - (self.timings.thinking_ms - thinking_before)
                - (self.timings.cache_ms - cache_before))
                .max(0.0);
        }
        let scored = match self.config.execution_mode {
            ExecutionMode::Reference => self.score_reference(&tokenized),
            ExecutionMode::Batched => self.score_batched(&tokenized),
        };
        match scored {
            Ok(results) if results.len() == valid_indices.len() => {
                for (index, result) in valid_indices.into_iter().zip(results) {
                    slots[index] = Some(Ok(result));
                }
            }
            Ok(results) => {
                let message = format!(
                    "native scorer returned {} results for {} jobs",
                    results.len(),
                    valid_indices.len()
                );
                for index in valid_indices {
                    slots[index] = Some(Err(JetError::NativeRuntime(message.clone())));
                }
            }
            Err(error) => {
                let message = error.to_string();
                for index in valid_indices {
                    slots[index] = Some(Err(JetError::NativeRuntime(message.clone())));
                }
            }
        }

        self.timings.batch_ms += elapsed_ms(batch_started);
        slots
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(JetError::NativeRuntime(
                        "scoring job was not completed".to_owned(),
                    ))
                })
            })
            .collect()
    }
}

impl Drop for LlamaScorer {
    fn drop(&mut self) {
        // SAFETY: contexts must be freed before the model and the shared CPU pool.
        // The pool remains live until automatic field destruction after this body.
        unsafe {
            if let Some(generation_context) = self.generation_context {
                sys::llama_free(generation_context.as_ptr());
            }
            sys::llama_free(self.context.as_ptr());
            for sampler in &self.score_samplers {
                sys::llama_sampler_free(sampler.as_ptr());
            }
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

fn validate_cpu_moe_model(
    path: &CStr,
    mut params: sys::llama_model_params,
    requested_layers: u32,
) -> Result<()> {
    // Inspect architecture and layer limits before allocating any model weights.
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    // SAFETY: path is NUL-terminated; parameters contain no tensor overrides yet.
    let model = NonNull::new(unsafe { sys::llama_model_load_from_file(path.as_ptr(), params) })
        .ok_or_else(|| {
            JetError::NativeRuntime("failed to read model metadata for cpu_moe_layers".to_owned())
        })?;
    let result = (|| {
        let architecture = model_metadata(model, c"general.architecture").ok_or_else(|| {
            JetError::UnsupportedModel(
                "cpu_moe_layers requires general.architecture model metadata".to_owned(),
            )
        })?;
        let metadata_number = |suffix: &str| -> Result<u32> {
            let key = CString::new(format!("{architecture}.{suffix}")).map_err(|_| {
                JetError::UnsupportedModel("model architecture contains NUL".to_owned())
            })?;
            model_metadata(model, &key).map_or(Ok(0), |value| {
                value.parse().map_err(|_| {
                    JetError::UnsupportedModel(format!(
                        "invalid model metadata {architecture}.{suffix}: {value}"
                    ))
                })
            })
        };
        let experts = metadata_number("expert_count")?;
        let leading_dense_layers = metadata_number("leading_dense_block_count")?;
        // vocab_only populates GGUF metadata but deliberately skips non-vocabulary
        // hparams, so llama_model_n_layer() is unavailable in this preflight.
        let total_layers = metadata_number("block_count")?;
        let mtp_layers = metadata_number("nextn_predict_layers")?;
        let layers = total_layers.checked_sub(mtp_layers).ok_or_else(|| {
            JetError::UnsupportedModel("model nextn_predict_layers exceeds block_count".to_owned())
        })?;
        validate_cpu_moe_layout(requested_layers, layers, experts, leading_dense_layers)
    })();
    // SAFETY: this vocabulary-only model is owned here and never created a context.
    unsafe { sys::llama_model_free(model.as_ptr()) };
    result
}

fn model_metadata(model: NonNull<sys::llama_model>, key: &CStr) -> Option<String> {
    // SAFETY: model and key are live; zero capacity allows a null output buffer.
    let length =
        unsafe { sys::llama_model_meta_val_str(model.as_ptr(), key.as_ptr(), ptr::null_mut(), 0) };
    if length < 0 {
        return None;
    }
    let mut value = vec![0_i8; length as usize + 1];
    // SAFETY: the mutable buffer has space for the advertised bytes plus terminating NUL.
    unsafe {
        sys::llama_model_meta_val_str(
            model.as_ptr(),
            key.as_ptr(),
            value.as_mut_ptr(),
            value.len(),
        );
        Some(
            CStr::from_ptr(value.as_ptr())
                .to_string_lossy()
                .into_owned(),
        )
    }
}

fn validate_cpu_moe_layout(
    requested_layers: u32,
    model_layers: u32,
    experts: u32,
    leading_dense_layers: u32,
) -> Result<()> {
    if experts == 0 {
        return Err(JetError::UnsupportedModel(
            "cpu_moe_layers requires a MoE model with routed experts".to_owned(),
        ));
    }
    if model_layers == 0 || requested_layers > model_layers {
        return Err(JetError::InvalidRequest(format!(
            "cpu_moe_layers ({requested_layers}) exceeds the model's {model_layers} transformer layers"
        )));
    }
    if requested_layers <= leading_dense_layers {
        return Err(JetError::InvalidRequest(format!(
            "cpu_moe_layers ({requested_layers}) selects only the model's leading dense layers and no routed experts"
        )));
    }
    Ok(())
}

fn validate_config(config: &EngineConfig) -> Result<()> {
    if config
        .gpu_layers
        .is_some_and(|layers| layers > i32::MAX as u32)
        || config.cpu_moe_layers > i32::MAX as u32
    {
        return Err(JetError::InvalidRequest(
            "gpu_layers and cpu_moe_layers must not exceed i32::MAX".to_owned(),
        ));
    }
    if config.backend == Backend::Cpu
        && (config.gpu_layers.is_some_and(|layers| layers > 0) || config.cpu_moe_layers > 0)
    {
        return Err(JetError::InvalidRequest(
            "gpu_layers > 0 and cpu_moe_layers > 0 require the Vulkan backend".to_owned(),
        ));
    }
    if config.model_id.is_empty() {
        return Err(JetError::InvalidRequest(
            "model_id must not be empty".to_owned(),
        ));
    }
    if config.context_tokens_per_sequence == 0
        || config.token_batch == 0
        || config.micro_batch == 0
        || config.max_output_rows == 0
    {
        return Err(JetError::InvalidRequest(
            "context and batch limits must be positive".to_owned(),
        ));
    }
    if config.max_sequences < 2 {
        return Err(JetError::InvalidRequest(
            "max_sequences must be at least 2".to_owned(),
        ));
    }
    if config.max_sequences > i32::MAX as u32 {
        return Err(JetError::InvalidRequest(
            "max_sequences exceeds llama.cpp limits".to_owned(),
        ));
    }
    if config.micro_batch > config.token_batch {
        return Err(JetError::InvalidRequest(
            "micro_batch must not exceed token_batch".to_owned(),
        ));
    }
    if config.threads <= 0 {
        return Err(JetError::InvalidRequest(
            "threads must be positive".to_owned(),
        ));
    }
    if config.thinking.mode != ThinkingMode::Disabled {
        if config.thinking.max_tokens == 0 {
            return Err(JetError::InvalidRequest(
                "thinking.max_tokens must be positive".to_owned(),
            ));
        }
        if !config.thinking.temperature.is_finite() || config.thinking.temperature <= 0.0 {
            return Err(JetError::InvalidRequest(
                "thinking.temperature must be finite and positive".to_owned(),
            ));
        }
        if config.thinking.top_k <= 0 {
            return Err(JetError::InvalidRequest(
                "thinking.top_k must be positive".to_owned(),
            ));
        }
        if !config.thinking.top_p.is_finite()
            || config.thinking.top_p <= 0.0
            || config.thinking.top_p > 1.0
        {
            return Err(JetError::InvalidRequest(
                "thinking.top_p must be in (0, 1]".to_owned(),
            ));
        }
    }
    Ok(())
}

fn thinking_is_open(plan: &ChatPlan) -> bool {
    if plan.thinking_start_tag.is_empty() {
        return false;
    }
    let Some(start) = plan.prompt.rfind(&plan.thinking_start_tag) else {
        return false;
    };
    let last_end = plan
        .thinking_end_tags
        .iter()
        .filter_map(|tag| plan.prompt.rfind(tag))
        .max();
    last_end.is_none_or(|end| end < start)
}

fn strip_thinking_end<'a>(rendered: &'a str, end_tags: &[String]) -> Option<&'a str> {
    end_tags
        .iter()
        .filter_map(|tag| rendered.strip_suffix(tag))
        .min_by_key(|prefix| prefix.len())
}

fn template_error(error: &[i8]) -> JetError {
    let message = if error.first().copied().unwrap_or_default() == 0 {
        "unknown chat template error".to_owned()
    } else {
        // SAFETY: wrapper always NUL-terminates the provided error buffer.
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    JetError::UnsupportedModel(message)
}

fn elapsed_ms(started: Option<Instant>) -> f64 {
    started.map_or(0.0, |start| start.elapsed().as_secs_f64() * 1_000.0)
}

fn prefix_targets(targets: &[Vec<sys::llama_token>]) -> Vec<sys::llama_token> {
    let mut tokens: Vec<_> = targets.iter().map(|target| target[0]).collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn position(value: usize) -> Result<sys::llama_pos> {
    i32::try_from(value)
        .map_err(|_| JetError::InvalidRequest("token position exceeds llama.cpp limits".to_owned()))
}

fn target_log_probability(logits: &[f32], target: sys::llama_token) -> Result<f64> {
    let target = usize::try_from(target)
        .map_err(|_| JetError::NativeRuntime(format!("target token ID {target} is negative")))?;
    let target_logit = logits.get(target).copied().ok_or_else(|| {
        JetError::NativeRuntime(format!("target token ID {target} exceeds the vocabulary"))
    })? as f64;
    let max = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() || !target_logit.is_finite() {
        return Err(JetError::NonFiniteScore(
            "logits contain a non-finite value".to_owned(),
        ));
    }
    let sum_exp: f64 = logits
        .iter()
        .map(|value| (f64::from(*value) - max).exp())
        .sum();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return Err(JetError::NonFiniteScore(
            "logsumexp normalization failed".to_owned(),
        ));
    }
    Ok(target_logit - (max + sum_exp.ln()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jet_core::normalize_log_probabilities;

    #[test]
    fn rejects_conflicting_and_overflowing_offload_configuration() -> Result<()> {
        let mut config = EngineConfig::qwen3_cpu("model.gguf");
        config.gpu_layers = Some(0);
        validate_config(&config)?;
        config.gpu_layers = Some(1);
        assert!(matches!(
            validate_config(&config),
            Err(JetError::InvalidRequest(_))
        ));
        config.gpu_layers = None;
        config.cpu_moe_layers = 1;
        assert!(matches!(
            validate_config(&config),
            Err(JetError::InvalidRequest(_))
        ));
        config.backend = Backend::Vulkan;
        validate_config(&config)?;
        config.gpu_layers = Some(u32::MAX);
        assert!(matches!(
            validate_config(&config),
            Err(JetError::InvalidRequest(_))
        ));
        config.gpu_layers = None;
        config.cpu_moe_layers = u32::MAX;
        assert!(matches!(
            validate_config(&config),
            Err(JetError::InvalidRequest(_))
        ));
        Ok(())
    }

    #[test]
    fn cpu_moe_configuration_must_select_existing_experts() -> Result<()> {
        validate_cpu_moe_layout(30, 40, 256, 0)?;
        validate_cpu_moe_layout(40, 40, 256, 0)?;
        assert!(matches!(
            validate_cpu_moe_layout(41, 40, 256, 0),
            Err(JetError::InvalidRequest(_))
        ));
        assert!(matches!(
            validate_cpu_moe_layout(1, 24, 0, 0),
            Err(JetError::UnsupportedModel(_))
        ));
        assert!(matches!(
            validate_cpu_moe_layout(2, 40, 256, 3),
            Err(JetError::InvalidRequest(_))
        ));
        validate_cpu_moe_layout(4, 40, 256, 3)?;
        Ok(())
    }

    #[test]
    fn computes_target_log_softmax() -> Result<()> {
        let value = target_log_probability(&[0.0, 1.0, 2.0], 2)?;
        let expected = 2.0 - (0.0_f64.exp() + 1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((value - expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn detects_open_tag_and_channel_reasoning_protocols() {
        let tag_plan = ChatPlan {
            prompt: "assistant:<think>work".to_owned(),
            generation_prompt: "<think>".to_owned(),
            supports_thinking: true,
            thinking_start_tag: "<think>".to_owned(),
            thinking_end_tags: vec!["</think>".to_owned()],
        };
        assert!(thinking_is_open(&tag_plan));

        let channel_plan = ChatPlan {
            prompt: "<|channel|>analysis<|message|>work<|end|>".to_owned(),
            generation_prompt: "<|channel|>analysis<|message|>".to_owned(),
            supports_thinking: true,
            thinking_start_tag: "<|channel|>analysis<|message|>".to_owned(),
            thinking_end_tags: vec!["<|end|>".to_owned()],
        };
        assert!(!thinking_is_open(&channel_plan));
        assert_eq!(
            strip_thinking_end("work<|end|>", &channel_plan.thinking_end_tags),
            Some("work")
        );
    }

    #[test]
    #[ignore = "requires JET_MODEL_PATH pointing to the pinned Qwen3 GGUF"]
    fn qwen_reference_batch_prefix_sharing_and_recovery() -> Result<()> {
        let model_path = std::env::var_os("JET_MODEL_PATH").ok_or_else(|| {
            JetError::InvalidRequest("JET_MODEL_PATH is required for model tests".to_owned())
        })?;
        let mut config = EngineConfig::qwen3_cpu(model_path);
        config.context_tokens_per_sequence = 512;
        config.token_batch = 512;
        config.micro_batch = 128;
        config.max_sequences = 4;
        config.max_output_rows = 512;
        config.threads = 2;
        config.execution_mode = ExecutionMode::Reference;
        let mut scorer = LlamaScorer::load(config)?;

        let compact = ScoreJob {
            system_content: crate::prompt::SYSTEM_INSTRUCTION,
            user_content: r#"{"allowed_labels":[false,true],"question":{"criteria":{"false":"no","true":"yes"},"instructions":"Choose."},"state":"English 中文 🙂"}"#.to_owned(),
            targets: vec!["false".to_owned(), "true".to_owned()],
        };
        let varied = ScoreJob {
            system_content: crate::prompt::SYSTEM_INSTRUCTION,
            user_content: r#"{"allowed_labels":[false,true,0,1,"short","a longer label"],"question":{"criteria":["a","b","c"],"instructions":"Pick one."},"state":{"marker":"\\u003c|im_start|\\u003e"}}"#.to_owned(),
            targets: vec![
                "false".to_owned(),
                "true".to_owned(),
                "0".to_owned(),
                "1".to_owned(),
                "\"short\"".to_owned(),
                "\"a longer label\"".to_owned(),
            ],
        };

        let rendered = scorer.render_prompt(&compact)?;
        assert!(!rendered.trim_end().ends_with("<think>"));
        if let Some(open_at) = rendered.find("<think>") {
            let content_at = open_at + "<think>".len();
            let close_at = rendered[content_at..]
                .find("</think>")
                .map(|offset| content_at + offset)
                .ok_or_else(|| {
                    JetError::UnsupportedModel("thinking marker was not closed".to_owned())
                })?;
            assert!(rendered[content_at..close_at].trim().is_empty());
        }
        scorer.tokenize_jobs(&[compact.clone(), varied.clone()])?;

        let reference = collect_scores(scorer.score_batch(&[compact.clone(), varied.clone()]))?;
        scorer.config.execution_mode = ExecutionMode::Batched;
        let batched = collect_scores(scorer.score_batch(&[compact.clone(), varied.clone()]))?;
        assert_eq!(batched[0].prefill_count, 1);
        assert_eq!(batched[1].prefill_count, 2);
        compare_scores(&reference, &batched)?;

        let mut permuted = varied.clone();
        permuted.targets.reverse();
        let permutation_result = collect_scores(scorer.score_batch(&[permuted]))?;
        for (left, right) in batched[1]
            .log_probabilities
            .iter()
            .zip(permutation_result[0].log_probabilities.iter().rev())
        {
            assert!((left - right).abs() <= 1e-5 + 1e-4 * 8.0);
        }

        scorer.config.thinking.mode = ThinkingMode::Required;
        scorer.config.thinking.max_tokens = 8;
        let thinking = collect_scores(scorer.score_batch(std::slice::from_ref(&compact)))?;
        assert!(thinking[0].thinking_tokens > 0);
        assert!(thinking[0].thinking_tokens <= 12);
        assert_eq!(thinking[0].prefill_count, 1);
        scorer.config.thinking.mode = ThinkingMode::Disabled;

        let long = ScoreJob {
            system_content: crate::prompt::SYSTEM_INSTRUCTION,
            user_content: format!(
                r#"{{"state":"{}","question":"long","allowed_labels":[true,false]}}"#,
                "word ".repeat(300)
            ),
            targets: compact.targets.clone(),
        };
        let compact_tokens = scorer.tokenize_jobs(std::slice::from_ref(&compact))?;
        let compact_limit = compact_tokens[0].prompt.len()
            + compact_tokens[0]
                .targets
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0);
        scorer.config.context_tokens_per_sequence = u32::try_from(compact_limit).map_err(|_| {
            JetError::InvalidRequest("compact fixture exceeds u32 context".to_owned())
        })?;
        let isolated = scorer.score_batch(&[long, compact.clone()]);
        assert!(matches!(
            isolated.first(),
            Some(Err(JetError::ContextExceeded { .. }))
        ));
        assert!(matches!(isolated.get(1), Some(Ok(_))));
        let reused = scorer.score_batch(&[compact]);
        assert!(matches!(reused.first(), Some(Ok(_))));
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires JET_MODEL_PATH and a working Vulkan device"]
    fn qwen_vulkan_smoke() -> Result<()> {
        let model_path = std::env::var_os("JET_MODEL_PATH").ok_or_else(|| {
            JetError::InvalidRequest("JET_MODEL_PATH is required for model tests".to_owned())
        })?;
        let mut config = EngineConfig::qwen3_vulkan(model_path);
        config.context_tokens_per_sequence = 512;
        config.token_batch = 512;
        config.micro_batch = 128;
        config.max_sequences = 3;
        config.max_output_rows = 512;
        config.threads = 2;
        let mut scorer = LlamaScorer::load(config)?;
        let job = ScoreJob {
            system_content: crate::prompt::SYSTEM_INSTRUCTION,
            user_content: r#"{"allowed_labels":[false,true],"question":{"criteria":{"false":"no","true":"yes"},"instructions":"Choose."},"state":"Vulkan smoke test"}"#.to_owned(),
            targets: vec!["false".to_owned(), "true".to_owned()],
        };

        let scores = collect_scores(scorer.score_batch(&[job]))?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].log_probabilities.len(), 2);
        assert!(
            scores[0]
                .log_probabilities
                .iter()
                .all(|value| value.is_finite())
        );
        Ok(())
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires JET_QWEN35_MODEL_PATH and a working Vulkan device"]
    fn qwen35_hybrid_batch_matches_reference() -> Result<()> {
        let model_path = std::env::var_os("JET_QWEN35_MODEL_PATH").ok_or_else(|| {
            JetError::InvalidRequest("JET_QWEN35_MODEL_PATH is required for model tests".to_owned())
        })?;
        let mut config = EngineConfig::vulkan(model_path, "qwen/qwen3.5");
        config.context_tokens_per_sequence = 1_024;
        config.token_batch = 512;
        config.micro_batch = 128;
        config.max_sequences = 2;
        config.max_output_rows = 64;
        config.threads = 2;
        config.collect_timings = true;
        let mut scorer = LlamaScorer::load(config)?;
        assert!(scorer.isolate_candidate_groups);
        assert!(scorer.device_softmax);
        assert_eq!(scorer.device_target_width, 1);

        let job = ScoreJob {
            system_content: crate::prompt::SYSTEM_INSTRUCTION,
            user_content: "State:\nFind the maximum possible order for an element of S_n for n = 10.\n\nTask:\nChoose the single correct answer.\n\nCandidates:\n\"6\", \"12\", \"30\", \"105\"\n\nResponse format:\nOne quoted candidate exactly as listed above; no explanation, whitespace, or extra text.".to_owned(),
            targets: vec!["\"6\"".to_owned(), "\"12\"".to_owned(), "\"30\"".to_owned(), "\"105\"".to_owned()],
        };
        let first = collect_scores(scorer.score_batch(std::slice::from_ref(&job)))?;
        assert_eq!(first[0].prefill_count, 1);

        // Grow the gather after a completed decode, exceed the physical sequence capacity,
        // and mix single-token answers, shared first tokens, and a multi-chunk suffix.
        let varied = ScoreJob {
            system_content: job.system_content,
            user_content: format!(
                "{}\n{}",
                job.user_content,
                "Additional context. ".repeat(80)
            ),
            targets: vec![
                "A".to_owned(),
                "B".to_owned(),
                "Yes".to_owned(),
                "No".to_owned(),
                "Maybe".to_owned(),
                "Alpha".to_owned(),
                "Beta".to_owned(),
                "Yes indeed".to_owned(),
                "No thanks".to_owned(),
                "word ".repeat(130),
            ],
        };
        let mut reversed = varied.clone();
        reversed.targets.reverse();
        let jobs = vec![varied, job.clone(), reversed];
        let tokens = scorer.tokenize_jobs(&jobs)?;
        assert!(tokens[0].targets.iter().any(|target| target.len() == 1));
        assert!(tokens[0].targets.iter().any(|target| target.len() > 64));
        let expected_width = tokens
            .iter()
            .map(|job| prefix_targets(&job.targets).len())
            .max()
            .unwrap_or(0);
        let before = scorer.timings.clone();
        let batched = collect_scores(scorer.score_batch(&jobs))?;
        assert!(batched.iter().all(|score| score.prefill_count == 1));
        assert!(scorer.device_target_width >= expected_width);
        assert!(scorer.device_target_width > 1);
        assert_eq!(
            scorer.timings.prefill_count - before.prefill_count,
            jobs.len() as u64
        );
        assert_eq!(
            scorer.timings.prefill_tokens - before.prefill_tokens,
            tokens
                .iter()
                .map(|job| job.prompt.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(
            scorer.timings.candidate_tokens - before.candidate_tokens,
            tokens
                .iter()
                .flat_map(|job| &job.targets)
                .map(|target| target.len().saturating_sub(1) as u64)
                .sum::<u64>()
        );
        assert!(scorer.timings.prefill_ms > before.prefill_ms);
        assert!(scorer.timings.candidate_ms > before.candidate_ms);

        // The same device sampler scores each candidate independently in reference mode.
        scorer.config.execution_mode = ExecutionMode::Reference;
        let reference = collect_scores(scorer.score_batch(&jobs))?;
        compare_scores(&reference, &batched)?;
        let reversed_scores: Vec<_> = batched[2].log_probabilities.iter().rev().copied().collect();
        assert_eq!(batched[0].log_probabilities, reversed_scores);

        scorer.config.execution_mode = ExecutionMode::Batched;
        let mut too_long = job.clone();
        too_long.user_content = "invalid oversized prompt ".repeat(1_024);
        let recovered = scorer.score_batch(&[too_long, job.clone()]);
        assert!(matches!(
            recovered.first(),
            Some(Err(JetError::ContextExceeded { .. }))
        ));
        let recovered = recovered.into_iter().skip(1).collect::<Result<Vec<_>>>()?;
        compare_scores(&first, &recovered)?;
        compare_scores(&first, &collect_scores(scorer.score_batch(&[job]))?)
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires JET_MODEL_PATH and a working Vulkan device"]
    fn qwen_vulkan_thinking_uses_split_contexts() -> Result<()> {
        let model_path = std::env::var_os("JET_MODEL_PATH").ok_or_else(|| {
            JetError::InvalidRequest("JET_MODEL_PATH is required for model tests".to_owned())
        })?;
        let mut config = EngineConfig::qwen3_vulkan(model_path);
        config.context_tokens_per_sequence = 512;
        config.token_batch = 512;
        config.micro_batch = 128;
        config.max_sequences = 3;
        config.max_output_rows = 128;
        config.threads = 2;
        config.thinking.mode = ThinkingMode::Required;
        config.thinking.max_tokens = 8;
        let mut scorer = LlamaScorer::load(config)?;
        assert!(scorer.device_softmax);
        assert!(scorer.generation_context.is_some());

        let job = ScoreJob {
            system_content: crate::prompt::SYSTEM_INSTRUCTION,
            user_content: r#"{"allowed_labels":[false,true],"question":{"criteria":{"false":"no","true":"yes"},"instructions":"Choose."},"state":"Vulkan thinking split-context smoke test"}"#.to_owned(),
            targets: vec!["false".to_owned(), "true".to_owned()],
        };
        let scores = collect_scores(scorer.score_batch(&[job]))?;
        assert_eq!(scores.len(), 1);
        assert!(scores[0].thinking_tokens > 0);
        assert!(scores[0].thinking_tokens <= 12);
        assert_eq!(scores[0].prefill_count, 1);
        assert!(
            scores[0]
                .log_probabilities
                .iter()
                .all(|value| value.is_finite())
        );
        Ok(())
    }

    fn collect_scores(results: Vec<Result<ScoreResult>>) -> Result<Vec<ScoreResult>> {
        results.into_iter().collect()
    }

    fn compare_scores(reference: &[ScoreResult], batched: &[ScoreResult]) -> Result<()> {
        if reference.len() != batched.len() {
            return Err(JetError::NativeRuntime(
                "reference and batched result lengths differ".to_owned(),
            ));
        }
        for (reference, batched) in reference.iter().zip(batched) {
            for ((reference_logp, batched_logp), token_count) in reference
                .log_probabilities
                .iter()
                .zip(&batched.log_probabilities)
                .zip(&reference.target_token_counts)
            {
                let tolerance = 1e-5 + 1e-4 * *token_count as f64;
                assert!(
                    (reference_logp - batched_logp).abs() <= tolerance,
                    "reference={reference_logp}, batched={batched_logp}, tokens={token_count}, tolerance={tolerance}"
                );
            }
            let reference_probabilities =
                normalize_log_probabilities(&reference.log_probabilities)?;
            let batched_probabilities = normalize_log_probabilities(&batched.log_probabilities)?;
            for (reference, batched) in reference_probabilities.iter().zip(batched_probabilities) {
                assert!(
                    (reference.normalized_probability - batched.normalized_probability).abs()
                        <= 1e-5
                );
            }
        }
        Ok(())
    }
}
