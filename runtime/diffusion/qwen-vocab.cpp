#include "vocab.h"
#include "modelship_qwen_vocab.hpp"
#include "gguf.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
constexpr size_t merge_count = 151387;
constexpr size_t merge_bytes = 1671839;
constexpr size_t max_header_bytes = 8u << 20;
std::string qwen_merges;

std::string reject_embedded() {
    throw std::runtime_error("embedded vocabularies are not supported by the private Qwen worker");
}
}

bool modelship_load_qwen_vocab(const char *encoder_path) {
    if (!qwen_merges.empty()) return false;
    // The supervisor has checked the exact full-file pin. With a null tensor
    // context, gguf reads metadata and descriptors, not tensor payloads.
    const std::unique_ptr<gguf_context, decltype(&gguf_free)> header(
        gguf_init_from_file(encoder_path, {true, nullptr}), gguf_free);
    if (!header || gguf_get_version(header.get()) != 3 ||
        gguf_get_n_tensors(header.get()) != 399 || gguf_get_n_kv(header.get()) != 30 ||
        gguf_get_data_offset(header.get()) > max_header_bytes) return false;
    const auto architecture = gguf_find_key(header.get(), "general.architecture");
    if (architecture < 0 || gguf_get_kv_type(header.get(), architecture) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(header.get(), architecture), "qwen3vl") != 0) return false;
    const auto key = gguf_find_key(header.get(), "tokenizer.ggml.merges");
    if (key < 0 || gguf_get_kv_type(header.get(), key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(header.get(), key) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(header.get(), key) != merge_count) return false;

    // These merge lines, including the final LF, matched the previous
    // built-in byte-for-byte for the reviewed encoder revision. Keep the
    // upstream Qwen2Tokenizer code and special-token order unchanged.
    std::string bytes;
    bytes.reserve(merge_bytes);
    for (size_t i = 0; i < merge_count; ++i) {
        const std::string_view line(gguf_get_arr_str(header.get(), key, i));
        const auto space = line.find(' ');
        if (line.size() >= merge_bytes - bytes.size() ||
            space == std::string_view::npos || space == 0 || space+1 == line.size() ||
            line.find(' ', space+1) != std::string_view::npos || line.find_first_of("\r\n") != std::string_view::npos) return false;
        bytes.append(line);
        bytes.push_back('\n');
    }
    if (bytes.size() != merge_bytes) return false;
    qwen_merges = std::move(bytes);
    return true;
}

std::string load_qwen2_merges() {
    if (qwen_merges.empty()) return reject_embedded();
    return qwen_merges;
}

std::string load_clip_merges()        { return reject_embedded(); }
std::string load_mistral_merges()     { return reject_embedded(); }
std::string load_mistral_vocab_json() { return reject_embedded(); }
std::string load_t5_tokenizer_json()  { return reject_embedded(); }
std::string load_umt5_tokenizer_json(){ return reject_embedded(); }
std::string load_gemma_merges()       { return reject_embedded(); }
std::string load_gemma_vocab_json()   { return reject_embedded(); }
