use std::cell::RefCell;
use std::ffi::{CStr, c_char};
use std::path::PathBuf;

use jet_core::{DecisionRequest, ErrorCode, JetError};
use jet_engine::{Engine, EngineConfig, ExecutionMode, ThinkingConfig, ThinkingMode};
use serde::{Deserialize, Serialize};

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WebEngineConfig {
    model_path: PathBuf,
    model_id: String,
    #[serde(default)]
    gpu_layers: Option<u32>,
    #[serde(default)]
    cpu_moe_layers: u32,
    // The wasm32 llama.cpp mmap loader does not complete with HeapFS model data.
    #[serde(default)]
    use_mmap: bool,
    #[serde(default = "default_context_tokens")]
    context_tokens_per_sequence: u32,
    #[serde(default = "default_token_batch")]
    token_batch: u32,
    #[serde(default = "default_micro_batch")]
    micro_batch: u32,
    #[serde(default = "default_max_sequences")]
    max_sequences: u32,
    #[serde(default = "default_true")]
    preparation_pipeline: bool,
    #[serde(default = "default_max_output_rows")]
    max_output_rows: u32,
    #[serde(default = "default_threads")]
    threads: i32,
    #[serde(default)]
    execution_mode: WebExecutionMode,
    #[serde(default)]
    thinking: WebThinkingConfig,
    #[serde(default = "default_true")]
    collect_timings: bool,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WebExecutionMode {
    Reference,
    #[default]
    Batched,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WebThinkingMode {
    #[default]
    Disabled,
    Auto,
    Required,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct WebThinkingConfig {
    mode: WebThinkingMode,
    max_tokens: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    seed: u32,
}

impl Default for WebThinkingConfig {
    fn default() -> Self {
        let config = ThinkingConfig::default();
        Self {
            mode: WebThinkingMode::Disabled,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            seed: config.seed,
        }
    }
}

#[derive(Serialize)]
struct Success {
    success: bool,
}

#[derive(Serialize)]
struct WebError {
    code: ErrorCode,
    message: String,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: WebError,
}

#[derive(Serialize)]
#[serde(untagged)]
enum BatchItem<T> {
    Success { response: T },
    Error { error: WebError },
}

const fn default_context_tokens() -> u32 {
    1_024
}

const fn default_token_batch() -> u32 {
    2_048
}

const fn default_micro_batch() -> u32 {
    512
}

const fn default_max_sequences() -> u32 {
    9
}

const fn default_max_output_rows() -> u32 {
    256
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|count| i32::try_from(count.get()).unwrap_or(i32::MAX))
        .unwrap_or(1)
}

const fn default_true() -> bool {
    true
}

impl From<WebEngineConfig> for EngineConfig {
    fn from(config: WebEngineConfig) -> Self {
        let mut engine = Self::webgpu(config.model_path, config.model_id);
        engine.gpu_layers = config.gpu_layers;
        engine.cpu_moe_layers = config.cpu_moe_layers;
        engine.use_mmap = config.use_mmap;
        engine.context_tokens_per_sequence = config.context_tokens_per_sequence;
        engine.token_batch = config.token_batch;
        engine.micro_batch = config.micro_batch;
        engine.max_sequences = config.max_sequences;
        engine.preparation_pipeline = config.preparation_pipeline;
        engine.max_output_rows = config.max_output_rows;
        engine.threads = config.threads;
        engine.execution_mode = match config.execution_mode {
            WebExecutionMode::Reference => ExecutionMode::Reference,
            WebExecutionMode::Batched => ExecutionMode::Batched,
        };
        engine.thinking = ThinkingConfig {
            mode: match config.thinking.mode {
                WebThinkingMode::Disabled => ThinkingMode::Disabled,
                WebThinkingMode::Auto => ThinkingMode::Auto,
                WebThinkingMode::Required => ThinkingMode::Required,
            },
            max_tokens: config.thinking.max_tokens,
            temperature: config.thinking.temperature,
            top_k: config.thinking.top_k,
            top_p: config.thinking.top_p,
            seed: config.thinking.seed,
        };
        engine.collect_timings = config.collect_timings;
        engine
    }
}

#[unsafe(no_mangle)]
/// Dispatch a JSON action from the browser worker.
///
/// # Safety
///
/// `action` and `request` must point to live NUL-terminated UTF-8 strings for the call.
pub unsafe extern "C" fn jet_web_action(
    action: *const c_char,
    request: *const c_char,
) -> *const u8 {
    let result = action_result(action, request).unwrap_or_else(error_json);
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        *output = result;
        output.as_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn jet_web_output_len() -> u32 {
    OUTPUT.with(|output| u32::try_from(output.borrow().len()).unwrap_or(u32::MAX))
}

fn action_result(action: *const c_char, request: *const c_char) -> Result<Vec<u8>, JetError> {
    if action.is_null() || request.is_null() {
        return Err(JetError::InvalidRequest(
            "action and request pointers must not be null".to_owned(),
        ));
    }
    // SAFETY: the browser bridge passes live, NUL-terminated UTF-8 strings for this call.
    let action = unsafe { CStr::from_ptr(action) }
        .to_str()
        .map_err(|error| JetError::InvalidRequest(error.to_string()))?;
    // SAFETY: the browser bridge passes live, NUL-terminated UTF-8 strings for this call.
    let request = unsafe { CStr::from_ptr(request) }
        .to_str()
        .map_err(|error| JetError::InvalidRequest(error.to_string()))?;

    match action {
        "load" => {
            let config: WebEngineConfig = serde_json::from_str(request)?;
            if config.use_mmap {
                return Err(JetError::InvalidRequest(
                    "use_mmap is not supported by the wasm32 HeapFS runtime".to_owned(),
                ));
            }
            let engine = Engine::load(config.into())?;
            ENGINE.with(|slot| *slot.borrow_mut() = Some(engine));
            serde_json::to_vec(&Success { success: true }).map_err(Into::into)
        }
        "unload" => {
            ENGINE.with(|slot| slot.borrow_mut().take());
            serde_json::to_vec(&Success { success: true }).map_err(Into::into)
        }
        "decide" => {
            let request: DecisionRequest = serde_json::from_str(request)?;
            with_engine(|engine| {
                let response = engine.decide(request)?;
                serde_json::to_vec(&response).map_err(Into::into)
            })
        }
        "decide_trace" => {
            let request: DecisionRequest = serde_json::from_str(request)?;
            with_engine(|engine| {
                let response = engine.decide_traced(request)?;
                serde_json::to_vec(&response).map_err(Into::into)
            })
        }
        "decide_batch" => {
            let requests: Vec<DecisionRequest> = serde_json::from_str(request)?;
            with_engine(|engine| {
                let responses: Vec<BatchItem<_>> = engine
                    .decide_batch(&requests)
                    .into_iter()
                    .map(|result| match result {
                        Ok(response) => BatchItem::Success { response },
                        Err(error) => BatchItem::Error {
                            error: web_error(error),
                        },
                    })
                    .collect();
                serde_json::to_vec(&responses).map_err(Into::into)
            })
        }
        "timings" => with_engine(|engine| serde_json::to_vec(engine.timings()).map_err(Into::into)),
        _ => Err(JetError::InvalidRequest(format!(
            "unknown Jet Web action `{action}`"
        ))),
    }
}

fn with_engine<T>(
    callback: impl FnOnce(&mut Engine) -> Result<T, JetError>,
) -> Result<T, JetError> {
    ENGINE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let engine = slot
            .as_mut()
            .ok_or_else(|| JetError::NativeRuntime("load a model before running Jet".to_owned()))?;
        callback(engine)
    })
}

fn error_json(error: JetError) -> Vec<u8> {
    serde_json::to_vec(&ErrorEnvelope {
        error: web_error(error),
    })
    .unwrap_or_else(|_| {
        br#"{"error":{"code":"native_runtime","message":"failed to serialize error"}}"#.to_vec()
    })
}

fn web_error(error: JetError) -> WebError {
    WebError {
        code: error.code(),
        message: error.to_string(),
    }
}
