#pragma once

#include "ggml.h" // for ggml_log_level

#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

// LLAMA_KV_SLAB environment-variable controls opt-in KV-slab padding for
// CUDA-graph reuse. When set to a positive integer `S`, attention-related
// tensors that key off `n_kv` (KQ mask shape, KV-cache view `ne[2]`, and
// tree-gather idx width) are rounded UP to the next multiple of `S` so the
// ggml cgraph topology stays stable within a slab. Capped at `n_ctx`.
// Disabled (no-op) when the env var is unset or `S <= 1`. The padded K
// cells are masked out via -inf in the KQ mask (set_input_kq_mask_with_tree
// walks `args.n_kv` cells and writes -inf for empty cells beyond the
// logical n_kv), so attention numerics are bit-identical to the unpadded
// computation. See:
//   llama-graph.cpp: build_attn_inp_kq_mask, can_reuse_kq_mask, tree_gather_width
//   llama-kv-cache.cpp: get_k/get_v, set_input_kq_mask_with_tree
static inline int64_t llama_kv_slab_size() {
    static const int64_t slab = []() -> int64_t {
        const char * env = getenv("LLAMA_KV_SLAB");
        if (!env || !*env) return 0;
        char * end = nullptr;
        long long v = std::strtoll(env, &end, 10);
        if (end == env || v <= 1) return 0;
        return (int64_t) v;
    }();
    return slab;
}

// Round n_kv up to the next KV-slab boundary, capped at n_ctx. When
// LLAMA_KV_SLAB is unset, returns n_kv unchanged. The caller is responsible
// for passing n_ctx (the KV cache size / max context) so the result never
// exceeds the underlying KV buffer.
static inline int64_t llama_kv_slab_round_n_kv(int64_t n_kv, int64_t n_ctx) {
    const int64_t slab = llama_kv_slab_size();
    if (slab <= 1) return n_kv;
    if (n_kv < 0) return n_kv;
    int64_t padded = ((n_kv + slab - 1) / slab) * slab;
    if (padded > n_ctx) padded = n_ctx;
    return padded;
}

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

template <typename dst_t, typename src_t>
static inline dst_t llama_cast(src_t v) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        return v;
    } else if constexpr (std::is_same_v<src_t, ggml_fp16_t> && std::is_same_v<dst_t, float>) {
        return ggml_fp16_to_fp32(v);
    } else if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
        return ggml_fp32_to_fp16(v);
    } else {
        static_assert(std::is_same_v<dst_t, void>, "unsupported type combination");
    }
}

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to llama_format ?
LLAMA_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne);
std::string llama_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define LLAMA_TENSOR_NAME_FATTN   "__fattn__"
#define LLAMA_TENSOR_NAME_FGDN_AR "__fgdn_ar__"
#define LLAMA_TENSOR_NAME_FGDN_CH "__fgdn_ch__"
