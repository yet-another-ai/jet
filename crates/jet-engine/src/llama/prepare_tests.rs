use super::*;

struct VocabularyModel(NonNull<sys::llama_model>);

impl VocabularyModel {
    fn load() -> Result<Self> {
        let path = std::env::var("JET_QWEN35_MODEL_PATH")
            .or_else(|_| std::env::var("JET_MODEL_PATH"))
            .map_err(|_| JetError::InvalidRequest("a model test path is required".to_owned()))?;
        let path = CString::new(path)
            .map_err(|_| JetError::InvalidRequest("model path contains NUL".to_owned()))?;
        BACKEND_INIT.call_once(|| {
            // SAFETY: all tests share the engine's process-wide initialization guard.
            unsafe { sys::llama_backend_init() };
        });
        // SAFETY: the native defaults initialize the complete parameter struct.
        let mut params = unsafe { sys::llama_model_default_params() };
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        // SAFETY: the path remains live for loading; this owner frees the returned model.
        NonNull::new(unsafe { sys::llama_model_load_from_file(path.as_ptr(), params) })
            .map(Self)
            .ok_or_else(|| JetError::NativeRuntime("vocabulary-only model load failed".to_owned()))
    }
}

impl Drop for VocabularyModel {
    fn drop(&mut self) {
        // SAFETY: each test drops its preparers and joins its worker before this owner.
        unsafe { sys::llama_model_free(self.0.as_ptr()) };
    }
}

fn fixture() -> ScoreJob {
    ScoreJob {
        system_content: crate::prompt::SYSTEM_INSTRUCTION,
        user_content: "Choose a description: English 中文 🙂. Is the sky usually blue?".to_owned(),
        targets: vec!["\"yes\"".to_owned(), "\"no\"".to_owned()],
    }
}

fn plan_json(plan: ChatPlan) -> serde_json::Value {
    serde_json::json!({
        "prompt": plan.prompt,
        "generation_prompt": plan.generation_prompt,
        "supports_thinking": plan.supports_thinking,
        "thinking_start_tag": plan.thinking_start_tag,
        "thinking_end_tags": plan.thinking_end_tags,
    })
}

fn legacy_plan(
    model: NonNull<sys::llama_model>,
    job: &ScoreJob,
    thinking: bool,
    reasoning: Option<&str>,
) -> Result<serde_json::Value> {
    let cstring = |value| {
        CString::new(value).map_err(|_| JetError::InvalidRequest("fixture contains NUL".to_owned()))
    };
    let system = cstring(job.system_content)?;
    let user = cstring(job.user_content.as_str())?;
    let reasoning = reasoning.map(cstring).transpose()?;
    let mut error = vec![0 as c_char; 1_024];
    let mut render = |output, capacity| {
        // SAFETY: the model and C strings live throughout both bridge calls; output is sized.
        unsafe {
            sys::jet_chat_render(
                model.as_ptr(),
                system.as_ptr(),
                user.as_ptr(),
                thinking,
                reasoning.as_ref().map_or(ptr::null(), |text| text.as_ptr()),
                output,
                capacity,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    };
    let required = render(ptr::null_mut(), 0);
    if required < 0 {
        return Err(template_error(&error));
    }
    let mut output = vec![0 as c_char; required as usize + 1];
    assert_eq!(render(output.as_mut_ptr(), output.len()), required);
    // SAFETY: the bridge writes a NUL terminator into the extra output byte.
    let bytes = unsafe { CStr::from_ptr(output.as_ptr()) }.to_bytes();
    assert_eq!(bytes.len(), required as usize);
    serde_json::from_slice(bytes).map_err(JetError::from)
}

fn legacy_tokens(model: NonNull<sys::llama_model>, text: &str) -> Vec<sys::llama_token> {
    // SAFETY: the model owns this immutable vocabulary for the complete test.
    let vocab = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };
    assert!(text.len() < i32::MAX as usize);
    let tokenize = |output, capacity| {
        // SAFETY: text has the given byte length; output is null for probing or properly sized.
        unsafe {
            sys::llama_tokenize(
                vocab,
                text.as_ptr().cast(),
                text.len() as i32,
                output,
                capacity,
                true,
                true,
            )
        }
    };
    let required = tokenize(ptr::null_mut(), 0);
    assert!(required <= 0 && required != i32::MIN);
    let mut output = vec![0; (-required) as usize];
    assert_eq!(
        tokenize(output.as_mut_ptr(), output.len() as i32),
        -required
    );
    output
}

#[test]
#[ignore = "requires JET_QWEN35_MODEL_PATH or JET_MODEL_PATH; loads vocabulary only"]
fn cached_preparation_matches_legacy_and_keeps_validation() -> Result<()> {
    let model = VocabularyModel::load()?;
    // SAFETY: model is declared before and outlives this preparer.
    let preparer = unsafe { PromptPreparer::new(model.0) }?;
    let job = fixture();
    let mut changed = job.clone();
    changed.user_content = "A second conversation, without prior reasoning.".to_owned();
    for current in [&job, &changed, &job] {
        for (thinking, reasoning) in [
            (false, None),
            (true, None),
            (
                true,
                Some("The question is about ordinary daylight. 中文 🙂"),
            ),
        ] {
            let expected = legacy_plan(model.0, current, thinking, reasoning)?;
            assert_eq!(
                plan_json(preparer.render(current, thinking, reasoning)?),
                expected
            );
        }
    }
    let large = "Unicode 中文 🙂 with repeated but distinct words! ".repeat(600);
    for text in [
        "",
        "short text",
        large.as_str(),
        "<|im_start|>user\n<|im_end|><think></think>",
        "short again after growing the buffer",
        "embedded\0NUL is legal tokenizer input",
    ] {
        assert_eq!(preparer.tokenize(text)?, legacy_tokens(model.0, text));
    }

    let prompt = preparer.disabled_prompt(&job)?;
    let prompt_len = prompt.tokens.len();
    let tokenized = preparer.finish(&job, prompt, 4_096)?;
    let required = prompt_len + tokenized.targets.iter().map(Vec::len).max().unwrap_or(0);
    preparer.finish(&job, preparer.disabled_prompt(&job)?, required)?;
    assert!(matches!(
        preparer.finish(&job, preparer.disabled_prompt(&job)?, required - 1),
        Err(JetError::ContextExceeded { .. })
    ));
    let mut empty_target = job.clone();
    empty_target.targets = vec![String::new()];
    assert!(matches!(
        preparer.finish(
            &empty_target,
            preparer.disabled_prompt(&empty_target)?,
            4_096
        ),
        Err(JetError::InvalidRequest(_))
    ));
    let mut nul = job.clone();
    nul.user_content.push('\0');
    assert!(matches!(
        preparer.disabled_prompt(&nul),
        Err(JetError::InvalidRequest(_))
    ));
    assert!(matches!(
        preparer.render(&job, true, Some("bad\0reasoning")),
        Err(JetError::NativeRuntime(_))
    ));

    // Cancellation must release a worker blocked on its bounded output queue, and
    // a subsequent batch must not receive leftovers from the abandoned one.
    // SAFETY: this worker is explicitly dropped before model.
    let worker = PreparationWorker::new(unsafe { PromptPreparer::new(model.0) }?)?;
    let abandoned = worker.submit(&vec![job.clone(); 8], 4_096, false)?;
    drop(abandoned);
    let received = worker.submit(&[nul, job], 4_096, false)?;
    let outputs = received.into_iter().collect::<Vec<_>>();
    assert_eq!(outputs.len(), 2);
    assert!(matches!(outputs[0].0, Err(JetError::InvalidRequest(_))));
    assert!(outputs[1].0.is_ok());
    assert!(outputs.iter().all(|(_, elapsed)| *elapsed == 0.0));
    assert_eq!(worker.submit(&[], 4_096, true)?.into_iter().count(), 0);
    drop(worker);
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires JET_QWEN35_MODEL_PATH and a working Vulkan device"]
fn pipelined_hybrid_preparation_matches_sync_and_recovers() -> Result<()> {
    let model = std::env::var_os("JET_QWEN35_MODEL_PATH").ok_or_else(|| {
        JetError::InvalidRequest("JET_QWEN35_MODEL_PATH is required for model tests".to_owned())
    })?;
    let mut config = EngineConfig::vulkan(model, "qwen/hybrid");
    config.context_tokens_per_sequence = 512;
    config.token_batch = 256;
    config.micro_batch = 128;
    config.max_sequences = 4;
    config.max_output_rows = 64;
    config.threads = 2;
    config.collect_timings = true;
    config.preparation_pipeline = true;
    let mut scorer = LlamaScorer::load(config)?;
    assert!(scorer.isolate_candidate_groups);
    let compact = fixture();
    let mut varied = compact.clone();
    varied
        .user_content
        .push_str(" Please select one answer without any explanation.");
    varied.targets.push("\"perhaps sometimes\"".to_owned());
    // Distinct first tokens force sampler growth after the first pipelined job.
    varied.targets.push("A".to_owned());
    let mut permuted = varied.clone();
    permuted.targets.reverse();
    let mut too_long = compact.clone();
    too_long.user_content = "context too long ".repeat(512);
    let mut invalid = compact.clone();
    invalid.user_content.push('\0');
    let mut empty = compact.clone();
    empty.targets = vec![String::new()];
    let jobs = vec![compact, too_long, varied, invalid, permuted, empty];
    assert!(scorer.score_batch(&[]).is_empty());
    let initial_width = scorer.device_target_width;
    let first_pipeline = scorer.score_batch(&jobs);
    assert!(scorer.device_target_width > initial_width);
    scorer.config.preparation_pipeline = false;
    let reference = scorer.score_batch(&jobs);
    compare_results(&reference, &first_pipeline)?;
    assert!(reference[0].is_ok());
    assert!(matches!(
        reference[1],
        Err(JetError::ContextExceeded { .. })
    ));
    assert!(reference[2].is_ok());
    assert!(matches!(reference[3], Err(JetError::InvalidRequest(_))));
    assert!(reference[4].is_ok());
    assert!(matches!(reference[5], Err(JetError::InvalidRequest(_))));
    scorer.config.preparation_pipeline = true;
    for _ in 0..2 {
        let before = scorer.timings.clone();
        compare_results(&reference, &scorer.score_batch(&jobs))?;
        assert!(scorer.timings.prepare_ms > before.prepare_ms);
        assert!(scorer.timings.prepare_wait_ms >= before.prepare_wait_ms);
        let reordered = jobs.iter().rev().cloned().collect::<Vec<_>>();
        let mut actual = scorer.score_batch(&reordered);
        actual.reverse();
        compare_results(&reference, &actual)?;
        assert!(scorer.score_batch(&[]).is_empty());
    }
    scorer.config.collect_timings = false;
    let before = serde_json::to_value(&scorer.timings)?;
    compare_results(&reference, &scorer.score_batch(&jobs))?;
    assert_eq!(serde_json::to_value(&scorer.timings)?, before);
    // Dropping the scorer must join the persistent worker before freeing its model.
    drop(scorer);
    Ok(())
}

#[cfg(feature = "vulkan")]
fn compare_results(
    reference: &[Result<ScoreResult>],
    actual: &[Result<ScoreResult>],
) -> Result<()> {
    assert_eq!(reference.len(), actual.len());
    for (expected, actual) in reference.iter().zip(actual) {
        match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                assert_eq!(expected.prompt_tokens, actual.prompt_tokens);
                assert_eq!(expected.thinking_tokens, actual.thinking_tokens);
                assert_eq!(expected.target_token_counts, actual.target_token_counts);
                assert!(!expected.target_token_counts.contains(&0));
                assert_eq!(expected.prefill_count, actual.prefill_count);
                assert_eq!(
                    expected.log_probabilities.len(),
                    actual.log_probabilities.len()
                );
                for ((left, right), count) in expected
                    .log_probabilities
                    .iter()
                    .zip(&actual.log_probabilities)
                    .zip(&expected.target_token_counts)
                {
                    assert!((left - right).abs() <= 1e-5 + 1e-4 * *count as f64);
                }
                let means = |score: &ScoreResult| {
                    score
                        .log_probabilities
                        .iter()
                        .zip(&score.target_token_counts)
                        .map(|(logp, count)| logp / *count as f64)
                        .collect::<Vec<_>>()
                };
                let expected = jet_core::normalize_log_probabilities(&means(expected))?;
                let actual = jet_core::normalize_log_probabilities(&means(actual))?;
                for (left, right) in expected.iter().zip(actual) {
                    assert!(
                        (left.normalized_probability - right.normalized_probability).abs() <= 1e-5
                    );
                }
            }
            (Err(expected), Err(actual)) => assert_eq!(expected.to_string(), actual.to_string()),
            _ => {
                return Err(JetError::NativeRuntime(
                    "pipeline changed a result's success status".to_owned(),
                ));
            }
        }
    }
    Ok(())
}
