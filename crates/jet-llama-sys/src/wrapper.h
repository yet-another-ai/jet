#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU buffer type used by llama_model_params.tensor_buft_overrides.
ggml_backend_buffer_type_t jet_cpu_buffer_type(void);

// Persistent CPU workers shared by a scorer's sequential llama contexts.
// The caller must free all attached contexts before freeing the threadpool.
ggml_threadpool_t jet_cpu_threadpool_new(int32_t n_threads);
void jet_cpu_threadpool_free(ggml_threadpool_t threadpool);

// Returns the required JSON byte length, excluding the terminating NUL.
// Returns -1 on failure and writes a diagnostic into error when provided.
int32_t jet_chat_render(
    const struct llama_model * model,
    const char * system_content,
    const char * user_content,
    bool enable_thinking,
    const char * reasoning_content,
    char * output,
    size_t output_capacity,
    char * error,
    size_t error_capacity);

// Creates a sampler chain that computes full-vocabulary softmax followed by
// target gather and log in llama.cpp's model-output backend. The caller owns
// the returned llama_sampler.
struct llama_sampler * jet_score_sampler_init(size_t target_width);
bool jet_score_sampler_set_targets(
    struct llama_sampler * sampler,
    const llama_token * targets,
    size_t target_count);

struct jet_thinking_sampler;

struct jet_thinking_sampler * jet_thinking_sampler_init(
    const struct llama_vocab * vocab,
    const char * end_tags_json,
    int32_t max_tokens,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t seed,
    char * error,
    size_t error_capacity);

llama_token jet_thinking_sampler_sample(
    struct jet_thinking_sampler * sampler,
    struct llama_context * context,
    int32_t logits_index);
void jet_thinking_sampler_accept(struct jet_thinking_sampler * sampler, llama_token token);
bool jet_thinking_sampler_done(const struct jet_thinking_sampler * sampler);
bool jet_thinking_sampler_force(struct jet_thinking_sampler * sampler);
void jet_thinking_sampler_free(struct jet_thinking_sampler * sampler);

#ifdef __cplusplus
}
#endif
