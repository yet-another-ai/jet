#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "stb/stb_image.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace {
void set_error(const std::string & message, char * output, size_t capacity) {
    if (output == nullptr || capacity == 0) return;
    const auto count = std::min(message.size(), capacity - 1);
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}

using chunks_ptr = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
using bitmap_ptr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;

chunks_ptr tokenize(mtmd_context * vision, const std::string & text,
                    const std::vector<bitmap_ptr> & bitmaps) {
    chunks_ptr chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    if (!chunks) return {nullptr, mtmd_input_chunks_free};
    std::vector<const mtmd_bitmap *> pointers;
    pointers.reserve(bitmaps.size());
    for (const auto & bitmap : bitmaps) pointers.push_back(bitmap.get());
    const mtmd_input_text input{text.data(), text.size(), true, true};
    if (mtmd_tokenize(vision, chunks.get(), &input, pointers.data(), pointers.size()) != 0) {
        return {nullptr, mtmd_input_chunks_free};
    }
    return chunks;
}

bool same_chunk(const mtmd_input_chunk * left, const mtmd_input_chunk * right) {
    if (mtmd_input_chunk_get_type(left) != mtmd_input_chunk_get_type(right) ||
        mtmd_input_chunk_get_n_tokens(left) != mtmd_input_chunk_get_n_tokens(right) ||
        mtmd_input_chunk_get_n_pos(left) != mtmd_input_chunk_get_n_pos(right)) return false;
    if (mtmd_input_chunk_get_type(left) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t left_count = 0, right_count = 0;
        const auto * left_tokens = mtmd_input_chunk_get_tokens_text(left, &left_count);
        const auto * right_tokens = mtmd_input_chunk_get_tokens_text(right, &right_count);
        if (left_count == 0 || right_count == 0) return left_count == right_count;
        return left_tokens != nullptr && right_tokens != nullptr && left_count == right_count &&
               std::equal(left_tokens, left_tokens + left_count, right_tokens);
    }
    const char * left_id = mtmd_input_chunk_get_id(left);
    const char * right_id = mtmd_input_chunk_get_id(right);
    return left_id != nullptr && right_id != nullptr && std::strcmp(left_id, right_id) == 0;
}
} // namespace

struct jet_vision {
    mtmd_context * native;
};

struct jet_vision_job {
    std::vector<bitmap_ptr> bitmaps;
    chunks_ptr chunks{nullptr, mtmd_input_chunks_free};
    std::vector<std::vector<llama_token>> suffixes;
    std::vector<llama_token> tail;
    size_t expanded_tokens = 0;
    size_t image_tokens = 0;
    llama_pos tail_position = 0;
    llama_pos next_position = 0;
};

extern "C" jet_vision * jet_vision_init(const llama_model * model, const char * mmproj,
                                         ggml_backend_dev_t device, int image_max_tokens,
                                         char * error, size_t error_capacity) {
    if (!model || !mmproj) {
        set_error("model and mmproj are required", error, error_capacity);
        return nullptr;
    }
    auto params = mtmd_context_params_default();
    params.use_gpu = device != nullptr;
    params.device = device;
    if (image_max_tokens > 0) params.image_max_tokens = image_max_tokens;
    auto * native = mtmd_init_from_file(mmproj, model, params);
    if (!native) {
        set_error("failed to load multimodal projector", error, error_capacity);
        return nullptr;
    }
    if (!mtmd_support_vision(native)) {
        mtmd_free(native);
        set_error("projector does not support images", error, error_capacity);
        return nullptr;
    }
    return new jet_vision{native};
}

extern "C" void jet_vision_free(jet_vision * vision) {
    if (!vision) return;
    mtmd_free(vision->native);
    delete vision;
}

extern "C" jet_vision_job * jet_vision_prepare(
    jet_vision * vision, const char * prompt,
    const char * const * targets, size_t target_count,
    const unsigned char * const * images, const size_t * image_lengths, size_t image_count,
    size_t max_pixels, char * error, size_t error_capacity) {
    try {
        if (!vision || !prompt || !targets || target_count == 0 || !images || !image_lengths || image_count == 0) {
            set_error("incomplete multimodal request", error, error_capacity);
            return nullptr;
        }
        auto job = std::make_unique<jet_vision_job>();
        job->bitmaps.reserve(image_count);
        const auto options = mtmd_helper_init_opt_default();
        for (size_t i = 0; i < image_count; ++i) {
            if (!images[i] || image_lengths[i] == 0) {
                set_error("empty image", error, error_capacity);
                return nullptr;
            }
            if (image_lengths[i] > static_cast<size_t>(std::numeric_limits<int>::max())) {
                set_error("image is too large to inspect", error, error_capacity);
                return nullptr;
            }
            int width = 0, height = 0, channels = 0;
            if (!stbi_info_from_memory(images[i], static_cast<int>(image_lengths[i]),
                                       &width, &height, &channels) ||
                width <= 0 || height <= 0 ||
                uint64_t(width) * uint64_t(height) > max_pixels) {
                set_error("invalid image or image exceeds configured pixel limit", error, error_capacity);
                return nullptr;
            }
            auto decoded = mtmd_helper_bitmap_init_from_buf(
                vision->native, images[i], image_lengths[i], false, options);
            bitmap_ptr bitmap(decoded.bitmap, mtmd_bitmap_free);
            if (decoded.video_ctx) mtmd_helper_video_free(decoded.video_ctx);
            if (!bitmap || mtmd_bitmap_is_audio(bitmap.get())) {
                set_error("invalid PNG or JPEG image", error, error_capacity);
                return nullptr;
            }
            const auto pixels = uint64_t(mtmd_bitmap_get_nx(bitmap.get())) *
                                uint64_t(mtmd_bitmap_get_ny(bitmap.get()));
            if (pixels == 0 || pixels > max_pixels) {
                set_error("image exceeds configured pixel limit", error, error_capacity);
                return nullptr;
            }
            job->bitmaps.push_back(std::move(bitmap));
        }
        const std::string prompt_text(prompt);
        job->chunks = tokenize(vision->native, prompt_text, job->bitmaps);
        if (!job->chunks) {
            set_error("multimodal prompt tokenization failed", error, error_capacity);
            return nullptr;
        }
        const auto chunk_count = mtmd_input_chunks_size(job->chunks.get());
        if (chunk_count == 0) {
            set_error("multimodal prompt is empty", error, error_capacity);
            return nullptr;
        }
        const auto * last = mtmd_input_chunks_get(job->chunks.get(), chunk_count - 1);
        if (mtmd_input_chunk_get_type(last) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            set_error("multimodal prompt has no final text chunk", error, error_capacity);
            return nullptr;
        }
        size_t tail_count = 0;
        const auto * tail_tokens = mtmd_input_chunk_get_tokens_text(last, &tail_count);
        if (!tail_tokens || tail_count == 0) {
            set_error("multimodal prompt has no final text tokens", error, error_capacity);
            return nullptr;
        }
        job->tail.assign(tail_tokens, tail_tokens + tail_count);
        job->expanded_tokens = mtmd_helper_get_n_tokens(job->chunks.get());
        for (size_t i = 0; i < chunk_count; ++i) {
            const auto * chunk = mtmd_input_chunks_get(job->chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
                job->image_tokens += mtmd_input_chunk_get_n_tokens(chunk);
            }
        }
        job->next_position = mtmd_helper_get_n_pos(job->chunks.get());
        job->tail_position = job->next_position - mtmd_input_chunk_get_n_pos(last);
        job->suffixes.reserve(target_count);
        for (size_t i = 0; i < target_count; ++i) {
            if (!targets[i]) {
                set_error("null candidate target", error, error_capacity);
                return nullptr;
            }
            auto combined = tokenize(vision->native, prompt_text + targets[i], job->bitmaps);
            if (!combined || mtmd_input_chunks_size(combined.get()) != chunk_count) {
                set_error("candidate changed multimodal chunk boundary", error, error_capacity);
                return nullptr;
            }
            for (size_t j = 0; j + 1 < chunk_count; ++j) {
                if (!same_chunk(mtmd_input_chunks_get(job->chunks.get(), j),
                                mtmd_input_chunks_get(combined.get(), j))) {
                    set_error("candidate changed multimodal prefix", error, error_capacity);
                    return nullptr;
                }
            }
            const auto * combined_tail = mtmd_input_chunks_get(combined.get(), chunk_count - 1);
            if (mtmd_input_chunk_get_type(combined_tail) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
                set_error("candidate changed final text chunk", error, error_capacity);
                return nullptr;
            }
            size_t combined_count = 0;
            const auto * combined_tokens = mtmd_input_chunk_get_tokens_text(combined_tail, &combined_count);
            if (!combined_tokens || combined_count <= tail_count ||
                !std::equal(job->tail.begin(), job->tail.end(), combined_tokens)) {
                set_error("candidate tokenizer boundary is unstable", error, error_capacity);
                return nullptr;
            }
            job->suffixes.emplace_back(combined_tokens + tail_count, combined_tokens + combined_count);
        }
        return job.release();
    } catch (const std::exception & exception) {
        set_error(exception.what(), error, error_capacity);
        return nullptr;
    }
}

extern "C" void jet_vision_job_free(jet_vision_job * job) { delete job; }
extern "C" size_t jet_vision_job_expanded_tokens(const jet_vision_job * job) { return job->expanded_tokens; }
extern "C" size_t jet_vision_job_image_tokens(const jet_vision_job * job) { return job->image_tokens; }
extern "C" llama_pos jet_vision_job_tail_position(const jet_vision_job * job) { return job->tail_position; }
extern "C" llama_pos jet_vision_job_next_position(const jet_vision_job * job) { return job->next_position; }
extern "C" const llama_token * jet_vision_job_tail(const jet_vision_job * job, size_t * count) {
    *count = job->tail.size();
    return job->tail.data();
}
extern "C" const llama_token * jet_vision_job_suffix(const jet_vision_job * job, size_t index, size_t * count) {
    if (index >= job->suffixes.size()) return nullptr;
    *count = job->suffixes[index].size();
    return job->suffixes[index].data();
}

extern "C" int32_t jet_vision_job_prefill_before_tail(
    jet_vision * vision, llama_context * context, const jet_vision_job * job,
    int32_t n_batch, llama_pos * next_position) {
    if (!vision || !context || !job || !next_position || n_batch <= 0) return -1;
    llama_pos current = 0;
    const auto count = mtmd_input_chunks_size(job->chunks.get());
    for (size_t i = 0; i + 1 < count; ++i) {
        const auto * chunk = mtmd_input_chunks_get(job->chunks.get(), i);
        llama_pos next = current;
        const int32_t status = mtmd_helper_eval_chunk_single(
            vision->native, context, chunk, current, 0, n_batch, false, &next);
        if (status != 0) return status;
        current = next;
    }
    *next_position = current;
    return 0;
}
