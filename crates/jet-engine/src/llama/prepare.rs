use std::cell::RefCell;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use super::*;

pub(super) struct PreparedPrompt {
    pub text: String,
    pub tokens: Vec<sys::llama_token>,
    pub logical_tokens: usize,
    pub thinking_tokens: usize,
}

struct ChatRenderer(NonNull<sys::jet_chat_renderer>);

impl Drop for ChatRenderer {
    fn drop(&mut self) {
        // SAFETY: this handle has exclusive ownership and its model is still live.
        unsafe { sys::jet_chat_renderer_free(self.0.as_ptr()) };
    }
}

/// CPU-only state. It borrows an immutable vocabulary, never a decoding context.
pub(super) struct PromptPreparer {
    renderer: RefCell<ChatRenderer>,
    vocab: NonNull<sys::llama_vocab>,
    tokens: RefCell<Vec<sys::llama_token>>,
}

// SAFETY: moving transfers exclusive ownership of the renderer and scratch buffer.
// Vocabulary tables are immutable after loading; tokenizer scratch is call-local.
// The scorer joins its worker and drops all preparers before freeing the model.
unsafe impl Send for PromptPreparer {}

impl PromptPreparer {
    /// The caller must keep the model live until this preparer is dropped.
    pub(super) unsafe fn new(model: NonNull<sys::llama_model>) -> Result<Self> {
        let mut error = vec![0 as c_char; 1_024];
        // SAFETY: the caller guarantees the model lifetime.
        let renderer = NonNull::new(unsafe {
            sys::jet_chat_renderer_init(model.as_ptr(), error.as_mut_ptr(), error.len())
        })
        .map(ChatRenderer)
        .ok_or_else(|| template_error(&error))?;
        // SAFETY: vocabulary belongs to the live model.
        let vocab = NonNull::new(unsafe {
            sys::llama_model_get_vocab(model.as_ptr()) as *mut sys::llama_vocab
        })
        .ok_or_else(|| JetError::NativeRuntime("model has no vocabulary".to_owned()))?;
        Ok(Self {
            renderer: RefCell::new(renderer),
            vocab,
            tokens: RefCell::new(vec![0; 512]),
        })
    }

    pub(super) fn render(
        &self,
        job: &ScoreJob,
        thinking: bool,
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
        let mut error = vec![0 as c_char; 1_024];
        let renderer = self.renderer.borrow_mut();
        // SAFETY: renderer access is exclusive and all input C strings are live.
        let output = unsafe {
            sys::jet_chat_renderer_render(
                renderer.0.as_ptr(),
                system.as_ptr(),
                user.as_ptr(),
                thinking,
                reasoning.as_ref().map_or(ptr::null(), |text| text.as_ptr()),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if output.is_null() {
            return Err(template_error(&error));
        }
        // SAFETY: output remains valid until the next render; the borrow prevents one.
        let bytes = unsafe { CStr::from_ptr(output) }.to_bytes();
        serde_json::from_slice(bytes).map_err(JetError::from)
    }

    pub(super) fn tokenize(&self, text: &str) -> Result<Vec<sys::llama_token>> {
        let text_len = i32::try_from(text.len()).map_err(|_| {
            JetError::InvalidRequest("text is too large for llama.cpp tokenization".to_owned())
        })?;
        let mut scratch = self.tokens.borrow_mut();
        loop {
            let capacity = i32::try_from(scratch.len())
                .map_err(|_| JetError::NativeRuntime("tokenizer capacity overflow".to_owned()))?;
            // SAFETY: vocabulary is immutable; scratch is exclusively borrowed and sized.
            let written = unsafe {
                sys::llama_tokenize(
                    self.vocab.as_ptr(),
                    text.as_ptr().cast(),
                    text_len,
                    scratch.as_mut_ptr(),
                    capacity,
                    true,
                    true,
                )
            };
            if written == i32::MIN {
                return Err(JetError::NativeRuntime(
                    "tokenization length overflow".to_owned(),
                ));
            }
            if written >= 0 {
                return scratch
                    .get(..written as usize)
                    .map(<[_]>::to_vec)
                    .ok_or_else(|| {
                        JetError::NativeRuntime(
                            "tokenizer returned an invalid token count".to_owned(),
                        )
                    });
            }
            let required = (-written) as usize;
            if required <= scratch.len() {
                return Err(JetError::NativeRuntime(
                    "invalid tokenizer size response".to_owned(),
                ));
            }
            scratch.resize(required, 0);
        }
    }

    pub(super) fn disabled_prompt(&self, job: &ScoreJob) -> Result<PreparedPrompt> {
        let disabled = self.render(job, false, None)?;
        let enabled = self.render(job, true, None)?;
        if thinking_is_open(&disabled)
            || (enabled.supports_thinking
                && enabled.prompt == disabled.prompt
                && enabled.generation_prompt == disabled.generation_prompt)
        {
            return Err(JetError::UnsupportedModel(
                "the chat template does not expose a verified non-thinking path".to_owned(),
            ));
        }
        let tokens = self.tokenize(&disabled.prompt)?;
        Ok(PreparedPrompt {
            text: disabled.prompt,
            logical_tokens: tokens.len(),
            tokens,
            thinking_tokens: 0,
        })
    }

    pub(super) fn finish(
        &self,
        job: &ScoreJob,
        prompt: PreparedPrompt,
        limit: usize,
    ) -> Result<TokenizedJob> {
        if prompt.tokens.is_empty() {
            return Err(JetError::UnsupportedModel(
                "chat template produced an empty prompt".to_owned(),
            ));
        }
        let mut targets = Vec::with_capacity(job.targets.len());
        let mut combined_text = prompt.text;
        let prefix_len = combined_text.len();
        for target in &job.targets {
            combined_text.truncate(prefix_len);
            combined_text.push_str(target);
            let combined = self.tokenize(&combined_text)?;
            if !combined.starts_with(&prompt.tokens) {
                return Err(JetError::UnsupportedModel(format!(
                    "tokenizer boundary is unstable for candidate {target:?}"
                )));
            }
            let suffix = combined[prompt.tokens.len()..].to_vec();
            if suffix.is_empty() {
                return Err(JetError::InvalidRequest(format!(
                    "candidate {target:?} has no scoreable tokens"
                )));
            }
            let required = prompt.tokens.len().saturating_add(suffix.len());
            if required > limit {
                return Err(JetError::ContextExceeded { required, limit });
            }
            targets.push(suffix);
        }
        Ok(TokenizedJob {
            prompt: prompt.tokens,
            targets,
            logical_prompt_tokens: prompt.logical_tokens,
            thinking_tokens: prompt.thinking_tokens,
        })
    }
}

struct PreparationBatch {
    jobs: Vec<ScoreJob>,
    limit: usize,
    timings: bool,
    output: SyncSender<(Result<TokenizedJob>, f64)>,
}

pub(super) struct PreparationWorker {
    input: Option<SyncSender<PreparationBatch>>,
    thread: Option<JoinHandle<()>>,
}

impl PreparationWorker {
    pub(super) fn new(preparer: PromptPreparer) -> Result<Self> {
        let (input, batches) = mpsc::sync_channel::<PreparationBatch>(1);
        let thread = thread::Builder::new()
            .name("jet-prepare".to_owned())
            .spawn(move || {
                for batch in batches {
                    for job in batch.jobs {
                        let started = batch.timings.then(Instant::now);
                        let result = preparer
                            .disabled_prompt(&job)
                            .and_then(|prompt| preparer.finish(&job, prompt, batch.limit));
                        // Do not charge queue backpressure to preparation time.
                        let elapsed = elapsed_ms(started);
                        if batch.output.send((result, elapsed)).is_err() {
                            break;
                        }
                    }
                }
            })?;
        Ok(Self {
            input: Some(input),
            thread: Some(thread),
        })
    }

    pub(super) fn submit(
        &self,
        jobs: &[ScoreJob],
        limit: usize,
        timings: bool,
    ) -> Result<Receiver<(Result<TokenizedJob>, f64)>> {
        // Two queued results plus at most one locally prepared result awaiting send.
        let (output, results) = mpsc::sync_channel(2);
        self.input
            .as_ref()
            .ok_or_else(worker_stopped)?
            .send(PreparationBatch {
                jobs: jobs.to_vec(),
                limit,
                timings,
                output,
            })
            .map_err(|_| worker_stopped())?;
        Ok(results)
    }
}

impl Drop for PreparationWorker {
    fn drop(&mut self) {
        self.input.take();
        if let Some(thread) = self.thread.take() {
            // No result receiver can outlive a synchronous score_batch call.
            let _ = thread.join();
        }
    }
}

fn worker_stopped() -> JetError {
    JetError::NativeRuntime("prompt preparation worker stopped".to_owned())
}

impl LlamaScorer {
    pub(super) fn score_pipelined(&mut self, jobs: &[ScoreJob]) -> Vec<Result<ScoreResult>> {
        let setup = (|| {
            if self.preparation_worker.is_none() {
                // SAFETY: Drop joins the worker before releasing this scorer's model.
                let preparer = unsafe { PromptPreparer::new(self.model) }?;
                self.preparation_worker = Some(PreparationWorker::new(preparer)?);
            }
            self.preparation_worker
                .as_ref()
                .ok_or_else(worker_stopped)?
                .submit(
                    jobs,
                    self.config.context_tokens_per_sequence as usize,
                    self.config.collect_timings,
                )
        })();
        let receiver = match setup {
            Ok(receiver) => receiver,
            Err(error) => {
                return jobs
                    .iter()
                    .map(|_| Err(JetError::NativeRuntime(error.to_string())))
                    .collect();
            }
        };
        jobs.iter()
            .map(|_| {
                let waited = self.timing_start();
                let prepared = receiver.recv();
                self.timings.prepare_wait_ms += elapsed_ms(waited);
                let (prepared, elapsed) = prepared.map_err(|_| worker_stopped())?;
                self.timings.prepare_ms += elapsed;
                let job = prepared?;
                // Keep exactly the existing one-prefix/two-sequence GPU execution.
                let mut scores = self
                    .score_batched(std::slice::from_ref(&job))
                    .map_err(|error| JetError::NativeRuntime(error.to_string()))?;
                if scores.len() != 1 {
                    return Err(JetError::NativeRuntime(
                        "native scorer returned an unexpected number of results".to_owned(),
                    ));
                }
                scores
                    .pop()
                    .ok_or_else(|| JetError::NativeRuntime("scorer omitted a job".to_owned()))
            })
            .collect()
    }
}
