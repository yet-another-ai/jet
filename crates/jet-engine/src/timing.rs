use serde::Serialize;

/// Cumulative wall-clock timings for this engine, in milliseconds.
///
/// Collection is opt-in through [`crate::EngineConfig::collect_timings`]. These
/// measurements include host work and waits for the backend; they are not GPU
/// kernel timings. `batch_ms` includes preparation, thinking, cache management,
/// and scoring, but excludes model loading. `prepare_ms` excludes thinking.
/// Phase totals need not sum to `batch_ms`, which also includes scheduling and
/// other bookkeeping. Counters and durations are zero when collection is off.
#[derive(Debug, Default, Clone, Serialize)]
pub struct EngineTimings {
    /// Model, backend, and inference-context initialization.
    pub load_ms: f64,
    /// Native chat-template rendering, tokenization, and validation, excluding thinking.
    pub prepare_ms: f64,
    /// Optional reasoning generation.
    pub thinking_ms: f64,
    /// Decoding scoring prompts and reading their final-token scores.
    pub prefill_ms: f64,
    /// Decoding and scoring candidate continuations after the prompt.
    pub candidate_ms: f64,
    /// Explicit cache clearing, copying, and sequence management.
    pub cache_ms: f64,
    /// Total scoring-batch wall time, including preparation and thinking.
    pub batch_ms: f64,
    /// Number of scoring-batch calls.
    pub batches: u64,
    /// Number of scoring jobs (questions).
    pub jobs: u64,
    /// Number of prompt tokens submitted to the decoder for scoring.
    pub prefill_tokens: u64,
    /// Number of continuation tokens submitted to the decoder for scoring.
    pub candidate_tokens: u64,
    /// Number of decoder calls across scoring and thinking.
    pub decode_calls: u64,
    /// Number of prompt prefills, including repeated prefills.
    pub prefill_count: u64,
}
