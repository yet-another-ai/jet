#include "wrapper.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <string>

#include "chat.h"

namespace {
void copy_string(const std::string & value, char * output, size_t capacity) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    const size_t length = std::min(value.size(), capacity - 1);
    std::memcpy(output, value.data(), length);
    output[length] = '\0';
}

bool has_only_empty_closed_thinking(const std::string & prompt) {
    constexpr const char * open = "<think>";
    constexpr const char * close = "</think>";
    const size_t open_at = prompt.find(open);
    if (open_at == std::string::npos) {
        return true;
    }
    const size_t content_at = open_at + std::strlen(open);
    const size_t close_at = prompt.find(close, content_at);
    if (close_at == std::string::npos || prompt.find(open, content_at) != std::string::npos ||
        prompt.find(close, close_at + std::strlen(close)) != std::string::npos) {
        return false;
    }
    return std::all_of(
        prompt.begin() + static_cast<std::ptrdiff_t>(content_at),
        prompt.begin() + static_cast<std::ptrdiff_t>(close_at),
        [](unsigned char character) { return std::isspace(character) != 0; });
}
}  // namespace

extern "C" int32_t jet_chat_apply_non_thinking(
    const struct llama_model * model,
    const char * system_content,
    const char * user_content,
    char * output,
    size_t output_capacity,
    char * error,
    size_t error_capacity) {
    try {
        if (model == nullptr || system_content == nullptr || user_content == nullptr) {
            copy_string("model and message content must be non-null", error, error_capacity);
            return -1;
        }

        auto templates = common_chat_templates_init(model, "");
        if (!common_chat_templates_support_enable_thinking(templates.get())) {
            copy_string("the model chat template does not expose a reliable enable_thinking switch", error, error_capacity);
            return -1;
        }

        common_chat_msg system_message{};
        system_message.role = "system";
        system_message.content = system_content;

        common_chat_msg user_message{};
        user_message.role = "user";
        user_message.content = user_content;

        common_chat_templates_inputs inputs{};
        inputs.messages = {std::move(system_message), std::move(user_message)};
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        inputs.enable_thinking = false;

        const auto params = common_chat_templates_apply(templates.get(), inputs);
        if (!has_only_empty_closed_thinking(params.prompt)) {
            copy_string("non-thinking template emitted open or non-empty thinking content", error, error_capacity);
            return -1;
        }
        if (params.prompt.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            copy_string("rendered chat prompt exceeds the bridge length limit", error, error_capacity);
            return -1;
        }

        copy_string(params.prompt, output, output_capacity);
        return static_cast<int32_t>(params.prompt.size());
    } catch (const std::exception & exception) {
        copy_string(exception.what(), error, error_capacity);
        return -1;
    } catch (...) {
        copy_string("unknown chat template failure", error, error_capacity);
        return -1;
    }
}
