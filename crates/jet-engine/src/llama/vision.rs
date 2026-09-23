use std::collections::BTreeSet;
use std::ffi::{CStr, CString, c_char, c_void};

use super::*;
use crate::config::VisionConfig;
use crate::vision::{ImageInput, validate_image_format, validate_image_id};

unsafe extern "C" {
    fn jet_vision_init(
        model: *const sys::llama_model,
        mmproj: *const c_char,
        use_gpu: bool,
        image_max_tokens: i32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
    fn jet_vision_free(vision: *mut c_void);
    fn jet_vision_prepare(
        vision: *mut c_void,
        prompt: *const c_char,
        targets: *const *const c_char,
        target_count: usize,
        images: *const *const u8,
        image_lengths: *const usize,
        image_count: usize,
        max_pixels: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
    fn jet_vision_job_free(job: *mut c_void);
    fn jet_vision_job_expanded_tokens(job: *const c_void) -> usize;
    fn jet_vision_job_image_tokens(job: *const c_void) -> usize;
    fn jet_vision_job_tail_position(job: *const c_void) -> sys::llama_pos;
    fn jet_vision_job_next_position(job: *const c_void) -> sys::llama_pos;
    fn jet_vision_job_tail(job: *const c_void, count: *mut usize) -> *const sys::llama_token;
    fn jet_vision_job_suffix(
        job: *const c_void,
        index: usize,
        count: *mut usize,
    ) -> *const sys::llama_token;
    fn jet_vision_job_prefill_before_tail(
        vision: *mut c_void,
        context: *mut sys::llama_context,
        job: *const c_void,
        n_batch: i32,
        next_position: *mut sys::llama_pos,
    ) -> i32;
}

fn native_error(buffer: &[c_char]) -> String {
    // SAFETY: the vector is initialized to zeros, including its last byte.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub(super) struct VisionContext(NonNull<c_void>);

impl VisionContext {
    pub(super) fn new(
        model: NonNull<sys::llama_model>,
        backend: Backend,
        config: &VisionConfig,
    ) -> Result<Self> {
        if config.max_images == 0
            || config.max_image_bytes == 0
            || config.max_image_pixels == 0
            || config.image_max_tokens <= 0
        {
            return Err(JetError::InvalidRequest(
                "vision limits must be positive".to_owned(),
            ));
        }
        let path = config.mmproj_path.to_str().ok_or_else(|| {
            JetError::InvalidRequest("mmproj_path must be valid UTF-8".to_owned())
        })?;
        let path = CString::new(path).map_err(|_| {
            JetError::InvalidRequest("mmproj_path contains an interior NUL byte".to_owned())
        })?;
        let mut error = vec![0; 1024];
        // SAFETY: model is live, the path is NUL-terminated, and the returned handle is owned.
        let native = unsafe {
            jet_vision_init(
                model.as_ptr(),
                path.as_ptr(),
                backend == Backend::Vulkan,
                config.image_max_tokens,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        NonNull::new(native).map(Self).ok_or_else(|| {
            JetError::UnsupportedModel(format!("vision projector: {}", native_error(&error)))
        })
    }
}

impl Drop for VisionContext {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the native projector.
        unsafe { jet_vision_free(self.0.as_ptr()) };
    }
}

struct VisionJob(NonNull<c_void>);

impl VisionJob {
    fn tokens(&self, index: Option<usize>) -> Result<Vec<sys::llama_token>> {
        let mut count = 0;
        // SAFETY: the job owns the backing token arrays for this call.
        let pointer = unsafe {
            match index {
                Some(index) => jet_vision_job_suffix(self.0.as_ptr(), index, &mut count),
                None => jet_vision_job_tail(self.0.as_ptr(), &mut count),
            }
        };
        if pointer.is_null() || count == 0 {
            return Err(JetError::NativeRuntime(
                "vision tokenizer returned an empty token span".to_owned(),
            ));
        }
        // SAFETY: the native job owns exactly count contiguous tokens until it is freed.
        Ok(unsafe { slice::from_raw_parts(pointer, count) }.to_vec())
    }
}

impl Drop for VisionJob {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the native job.
        unsafe { jet_vision_job_free(self.0.as_ptr()) };
    }
}

impl LlamaScorer {
    pub(crate) fn validate_images(&self, images: &[ImageInput]) -> Result<()> {
        let config = self.config.vision.as_ref().ok_or_else(|| {
            JetError::UnsupportedModel("engine was not loaded with a vision projector".to_owned())
        })?;
        if images.is_empty() || images.len() > config.max_images {
            return Err(JetError::InvalidRequest(format!(
                "images must contain 1-{} entries",
                config.max_images
            )));
        }
        let mut ids = BTreeSet::new();
        for image in images {
            validate_image_id(&image.id)?;
            if !ids.insert(image.id.as_str()) {
                return Err(JetError::InvalidRequest(format!(
                    "duplicate image id {:?}",
                    image.id
                )));
            }
            if image.data.len() > config.max_image_bytes {
                return Err(JetError::InvalidRequest(format!(
                    "image {:?} exceeds byte limit",
                    image.id
                )));
            }
            validate_image_format(image)?;
        }
        if self.config.thinking.mode != ThinkingMode::Disabled {
            return Err(JetError::UnsupportedModel(
                "multimodal scoring currently requires disabled thinking".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn score_vision_job(
        &mut self,
        job: &ScoreJob,
        images: &[ImageInput],
    ) -> Result<(ScoreResult, usize)> {
        self.validate_images(images)?;
        let config = self.config.vision.as_ref().ok_or_else(|| {
            JetError::UnsupportedModel("vision projector is not configured".to_owned())
        })?;
        let max_pixels = config.max_image_pixels;
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| JetError::NativeRuntime("vision projector is unavailable".to_owned()))?;
        let vision_ptr = vision.0.as_ptr();
        let mut content = String::new();
        for image in images {
            content.push_str(&format!("Image {}:\n<__media__>\n", image.id));
        }
        content.push_str(&job.user_content);
        let multimodal_job = ScoreJob {
            system_content: job.system_content,
            user_content: content,
            targets: job.targets.clone(),
        };
        let rendered = self.preparer()?.disabled_prompt(&multimodal_job)?;
        let prompt = CString::new(rendered.text)
            .map_err(|_| JetError::InvalidRequest("rendered prompt contains NUL".to_owned()))?;
        let targets = job
            .targets
            .iter()
            .map(|target| CString::new(target.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| JetError::InvalidRequest("candidate contains NUL".to_owned()))?;
        let target_ptrs: Vec<_> = targets.iter().map(|target| target.as_ptr()).collect();
        let image_ptrs: Vec<_> = images.iter().map(|image| image.data.as_ptr()).collect();
        let image_lens: Vec<_> = images.iter().map(|image| image.data.len()).collect();
        let mut error = vec![0; 1024];
        // SAFETY: all argument buffers remain live during this native preparation call.
        let native = unsafe {
            jet_vision_prepare(
                vision_ptr,
                prompt.as_ptr(),
                target_ptrs.as_ptr(),
                target_ptrs.len(),
                image_ptrs.as_ptr(),
                image_lens.as_ptr(),
                image_ptrs.len(),
                max_pixels,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let prepared = VisionJob(NonNull::new(native).ok_or_else(|| {
            JetError::InvalidRequest(format!("vision preparation: {}", native_error(&error)))
        })?);
        // SAFETY: the job is live for all metadata reads.
        let expanded = unsafe { jet_vision_job_expanded_tokens(prepared.0.as_ptr()) };
        // SAFETY: the job is live for all metadata reads.
        let tail_position = unsafe { jet_vision_job_tail_position(prepared.0.as_ptr()) };
        // SAFETY: the job is live for all metadata reads.
        let next_position = unsafe { jet_vision_job_next_position(prepared.0.as_ptr()) };
        if tail_position < 0 || next_position < tail_position {
            return Err(JetError::NativeRuntime(
                "vision tokenizer returned invalid positions".to_owned(),
            ));
        }
        let tail = prepared.tokens(None)?;
        let suffixes = (0..job.targets.len())
            .map(|index| prepared.tokens(Some(index)))
            .collect::<Result<Vec<_>>>()?;
        let limit = self.config.context_tokens_per_sequence as usize;
        for suffix in &suffixes {
            let required = expanded.saturating_add(suffix.len());
            if required > limit {
                return Err(JetError::ContextExceeded { required, limit });
            }
        }
        // SAFETY: the prepared job remains live until scoring finishes.
        let image_tokens = unsafe { jet_vision_job_image_tokens(prepared.0.as_ptr()) };
        let tokenized = TokenizedJob {
            prompt: tail,
            targets: suffixes,
            logical_prompt_tokens: expanded,
            thinking_tokens: 0,
        };
        self.ensure_target_width(std::slice::from_ref(&tokenized))?;
        if self.config.execution_mode == ExecutionMode::Reference {
            return self
                .score_vision_reference(
                    vision_ptr,
                    &prepared,
                    &tokenized,
                    expanded,
                    tail_position,
                    next_position,
                )
                .map(|score| (score, image_tokens));
        }
        let result = (|| {
            let mut actual_position = 0;
            let started = self.timing_start();
            // SAFETY: the projector, scoring context, and prepared chunks remain live.
            let status = unsafe {
                jet_vision_job_prefill_before_tail(
                    vision_ptr,
                    self.context.as_ptr(),
                    prepared.0.as_ptr(),
                    i32::try_from(self.config.micro_batch).unwrap_or(i32::MAX),
                    &mut actual_position,
                )
            };
            self.timings.prefill_ms += elapsed_ms(started);
            if status != 0 || actual_position != tail_position {
                return Err(JetError::NativeRuntime(format!(
                    "vision prefill failed or returned inconsistent position: status={status}, position={actual_position}"
                )));
            }
            let groups = [ActiveGroup {
                spec: GroupSpec {
                    job_index: 0,
                    candidate_start: 0,
                    candidate_end: tokenized.targets.len(),
                },
                prefix_sequence: 0,
                candidate_sequences: Vec::new(),
            }];
            let mut accumulated = vec![vec![0.0; tokenized.targets.len()]];
            let tail_start = usize::try_from(tail_position)
                .map_err(|_| JetError::NativeRuntime("negative vision tail position".to_owned()))?;
            let prefix_items = tokenized
                .prompt
                .iter()
                .copied()
                .enumerate()
                .map(|(index, token)| {
                    Ok(DecodeItem {
                        token,
                        position: position(tail_start + index)?,
                        sequence: 0,
                        output: if index + 1 == tokenized.prompt.len() {
                            OutputAction::Prefix(0)
                        } else {
                            OutputAction::None
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            self.decode_prefill(
                &prefix_items,
                std::slice::from_ref(&tokenized),
                &groups,
                &mut accumulated,
            )?;
            let continuation_start = usize::try_from(next_position).map_err(|_| {
                JetError::NativeRuntime("negative vision continuation position".to_owned())
            })?;
            for (candidate_index, tokens) in tokenized.targets.iter().enumerate() {
                if tokens.len() == 1 {
                    continue;
                }
                self.copy_prefix(0, 1)?;
                let suffix_items = (0..tokens.len() - 1)
                    .map(|index| {
                        Ok(DecodeItem {
                            token: tokens[index],
                            position: position(continuation_start + index)?,
                            sequence: 1,
                            output: OutputAction::Target {
                                job_index: 0,
                                candidate_index,
                                target: tokens[index + 1],
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.decode_candidates(
                    &suffix_items,
                    std::slice::from_ref(&tokenized),
                    &groups,
                    &mut accumulated,
                )?;
            }
            Ok(ScoreResult {
                log_probabilities: accumulated.remove(0),
                prompt_tokens: expanded,
                thinking_tokens: 0,
                target_token_counts: tokenized.targets.iter().map(Vec::len).collect(),
                prefill_count: 1,
            })
        })();
        self.clear_memory();
        result.map(|score| (score, image_tokens))
    }

    fn score_vision_reference(
        &mut self,
        vision_ptr: *mut c_void,
        prepared: &VisionJob,
        job: &TokenizedJob,
        expanded: usize,
        tail_position: sys::llama_pos,
        next_position: sys::llama_pos,
    ) -> Result<ScoreResult> {
        let mut log_probabilities = Vec::with_capacity(job.targets.len());
        for target in &job.targets {
            let result = (|| {
                let mut actual_position = 0;
                // SAFETY: every candidate starts with cleared sequence memory and live chunks.
                let status = unsafe {
                    jet_vision_job_prefill_before_tail(
                        vision_ptr,
                        self.context.as_ptr(),
                        prepared.0.as_ptr(),
                        i32::try_from(self.config.micro_batch).unwrap_or(i32::MAX),
                        &mut actual_position,
                    )
                };
                if status != 0 || actual_position != tail_position {
                    return Err(JetError::NativeRuntime(format!(
                        "reference vision prefill failed: status={status}, position={actual_position}"
                    )));
                }
                let reference_job = TokenizedJob {
                    prompt: job.prompt.clone(),
                    targets: vec![target.clone()],
                    logical_prompt_tokens: expanded,
                    thinking_tokens: 0,
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
                let start = usize::try_from(tail_position).map_err(|_| {
                    JetError::NativeRuntime("negative vision tail position".to_owned())
                })?;
                let items = reference_job
                    .prompt
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, token)| {
                        Ok(DecodeItem {
                            token,
                            position: position(start + index)?,
                            sequence: 0,
                            output: if index + 1 == reference_job.prompt.len() {
                                OutputAction::Prefix(0)
                            } else {
                                OutputAction::None
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.decode_prefill(
                    &items,
                    std::slice::from_ref(&reference_job),
                    &groups,
                    &mut accumulated,
                )?;
                let start = usize::try_from(next_position).map_err(|_| {
                    JetError::NativeRuntime("negative vision continuation position".to_owned())
                })?;
                let suffix_items = (0..target.len().saturating_sub(1))
                    .map(|index| {
                        Ok(DecodeItem {
                            token: target[index],
                            position: position(start + index)?,
                            sequence: 0,
                            output: OutputAction::Target {
                                job_index: 0,
                                candidate_index: 0,
                                target: target[index + 1],
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.decode_candidates(
                    &suffix_items,
                    std::slice::from_ref(&reference_job),
                    &groups,
                    &mut accumulated,
                )?;
                Ok(accumulated[0][0])
            })();
            self.clear_memory();
            log_probabilities.push(result?);
        }
        Ok(ScoreResult {
            log_probabilities,
            prompt_tokens: expanded,
            thinking_tokens: 0,
            target_token_counts: job.targets.iter().map(Vec::len).collect(),
            prefill_count: job.targets.len(),
        })
    }
}
