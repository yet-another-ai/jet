#include "wrapper.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "json.h"
#include "reasoning-budget.h"

using json = common_json;

namespace {
void copy_string(const std::string & value, char * output, size_t capacity) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    const size_t length = std::min(value.size(), capacity - 1);
    std::memcpy(output, value.data(), length);
    output[length] = '\0';
}

common_chat_msg message(const char * role, const char * content) {
    common_chat_msg result{};
    result.role = role;
    result.content = content;
    return result;
}
}  // namespace

extern "C" int32_t jet_chat_render(
    const struct llama_model * model,
    const char * system_content,
    const char * user_content,
    bool enable_thinking,
    const char * reasoning_content,
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
        common_chat_templates_inputs inputs{};
        inputs.messages = {message("system", system_content), message("user", user_content)};
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        inputs.enable_thinking = enable_thinking;

        if (reasoning_content != nullptr) {
            common_chat_msg assistant = message("assistant", "");
            assistant.reasoning_content = reasoning_content;
            inputs.messages.push_back(std::move(assistant));
            inputs.add_generation_prompt = false;
            inputs.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
        }

        const auto params = common_chat_templates_apply(templates.get(), inputs);
        const auto rendered = json{
            {"prompt", params.prompt},
            {"generation_prompt", params.generation_prompt},
            {"supports_thinking", params.supports_thinking},
            {"thinking_start_tag", params.thinking_start_tag},
            {"thinking_end_tags", params.thinking_end_tags},
        }.dump();
        if (rendered.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            copy_string("rendered chat plan exceeds the bridge length limit", error, error_capacity);
            return -1;
        }

        copy_string(rendered, output, output_capacity);
        return static_cast<int32_t>(rendered.size());
    } catch (const std::exception & exception) {
        copy_string(exception.what(), error, error_capacity);
        return -1;
    } catch (...) {
        copy_string("unknown chat template failure", error, error_capacity);
        return -1;
    }
}

struct jet_thinking_sampler {
    llama_sampler * chain;
    llama_sampler * budget;
};

extern "C" struct jet_thinking_sampler * jet_thinking_sampler_init(
    const struct llama_vocab * vocab,
    const char * end_tags_json,
    int32_t max_tokens,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t seed,
    char * error,
    size_t error_capacity) {
    try {
        if (vocab == nullptr || end_tags_json == nullptr) {
            throw std::invalid_argument("vocabulary and thinking end tags must be non-null");
        }
        if (max_tokens <= 0) {
            throw std::invalid_argument("thinking token budget must be positive");
        }

        const auto end_tags = json::parse(end_tags_json).get<std::vector<std::string>>();
        std::vector<llama_tokens> end_tokens;
        for (const auto & tag : end_tags) {
            auto tokens = common_tokenize(vocab, tag, false, true);
            if (!tokens.empty()) {
                end_tokens.push_back(std::move(tokens));
            }
        }
        if (end_tokens.empty()) {
            throw std::invalid_argument("thinking protocol has no tokenizable end tag");
        }

        auto * result = new jet_thinking_sampler{};
        result->chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        result->budget = common_reasoning_budget_init(
            vocab,
            {},
            end_tokens,
            end_tokens.front(),
            max_tokens,
            REASONING_BUDGET_COUNTING);
        if (result->chain == nullptr || result->budget == nullptr) {
            if (result->chain != nullptr) {
                llama_sampler_free(result->chain);
            } else if (result->budget != nullptr) {
                llama_sampler_free(result->budget);
            }
            delete result;
            throw std::runtime_error("failed to create thinking sampler");
        }

        llama_sampler_chain_add(result->chain, result->budget);
        llama_sampler_chain_add(result->chain, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(result->chain, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(result->chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(result->chain, llama_sampler_init_dist(seed));
        return result;
    } catch (const std::exception & exception) {
        copy_string(exception.what(), error, error_capacity);
        return nullptr;
    } catch (...) {
        copy_string("unknown thinking sampler failure", error, error_capacity);
        return nullptr;
    }
}

extern "C" llama_token jet_thinking_sampler_sample(
    struct jet_thinking_sampler * sampler,
    struct llama_context * context,
    int32_t logits_index) {
    return llama_sampler_sample(sampler->chain, context, logits_index);
}

extern "C" void jet_thinking_sampler_accept(struct jet_thinking_sampler * sampler, llama_token token) {
    llama_sampler_accept(sampler->chain, token);
}

extern "C" bool jet_thinking_sampler_done(const struct jet_thinking_sampler * sampler) {
    return common_reasoning_budget_get_state(sampler->budget) == REASONING_BUDGET_DONE;
}

extern "C" bool jet_thinking_sampler_force(struct jet_thinking_sampler * sampler) {
    return common_reasoning_budget_force(sampler->budget);
}

extern "C" void jet_thinking_sampler_free(struct jet_thinking_sampler * sampler) {
    if (sampler == nullptr) {
        return;
    }
    llama_sampler_free(sampler->chain);
    delete sampler;
}
