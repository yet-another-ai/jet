#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"
#include "wllama-fs.h"

// Keep the C++ half of the Emscripten link honest about the memory32 layouts
// asserted by bindings_emscripten.rs on the Rust side.
static_assert(sizeof(void *) == 4);
static_assert(sizeof(ggml_tensor) == 256);
static_assert(offsetof(ggml_tensor, buffer) == 4);
static_assert(sizeof(llama_model_params) == 52);
static_assert(sizeof(llama_context_params) == 132);
static_assert(sizeof(llama_batch) == 28);
static_assert(sizeof(llama_token_data_array) == 24);

extern "C" const std::uint8_t * jet_web_action(const char * action, const char * request);
extern "C" std::uint32_t jet_web_output_len();

static std::vector<char> input_buffer;

extern "C" const char * wllama_malloc(std::size_t size, std::uint32_t) {
    if (input_buffer.size() < size) {
        input_buffer.resize(size);
    }
    return input_buffer.data();
}

extern "C" const char * wllama_start() {
    wllama_fs::make_sure_ready();
    return "{\"success\":true}";
}

extern "C" const char * wllama_action(const char *, const char *) {
    return nullptr;
}

extern "C" const char * wllama_jet_action(const char * action, const char * request) {
    return reinterpret_cast<const char *>(jet_web_action(action, request));
}

extern "C" std::uint32_t wllama_jet_output_len() {
    return jet_web_output_len();
}

extern "C" const char * wllama_exit() {
    jet_web_action("unload", "{}");
    return "{\"success\":true}";
}

extern "C" const char * wllama_debug() {
    return "{}";
}

int main() {
    return 0;
}
