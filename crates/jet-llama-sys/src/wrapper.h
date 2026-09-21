#pragma once

#include <stddef.h>
#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the required UTF-8 byte length, excluding the terminating NUL.
// Returns -1 on failure and writes a diagnostic into error when provided.
int32_t jet_chat_apply_non_thinking(
    const struct llama_model * model,
    const char * system_content,
    const char * user_content,
    char * output,
    size_t output_capacity,
    char * error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
