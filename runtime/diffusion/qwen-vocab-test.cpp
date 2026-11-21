#include "ggml.h"
#include "gguf.h"
#include "modelship_qwen_vocab.hpp"
#include "vocab.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

int main() {
    struct gguf_context * metadata = gguf_init_empty();
    struct ggml_context * tensors = ggml_init({1u << 20, nullptr, true});
    if (!metadata || !tensors) return 1;
    for (int i = 0; i < 399; ++i) {
        auto *tensor = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 32);
        ggml_set_name(tensor, ("fixture_" + std::to_string(i)).c_str());
        gguf_add_tensor(metadata, tensor);
    }
    for (int i = 0; i < 28; ++i) gguf_set_val_bool(metadata, ("fixture." + std::to_string(i)).c_str(), true);
    gguf_set_val_str(metadata, "general.architecture", "qwen3vl");
    const char *key = "tokenizer.ggml.merges";
    gguf_set_val_str(metadata, key, "wrong type");
    char path[] = "/tmp/modelship-vocab-test.XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return 1;
    close(fd);
    struct Cleanup { const char *path; ~Cleanup() { unlink(path); } } cleanup{path};

    if (!gguf_write_to_file(metadata, path, true) || modelship_load_qwen_vocab(path)) return 1;
    try { load_qwen2_merges(); return 1; } catch (const std::runtime_error &) {}
    try { load_clip_merges(); return 1; } catch (const std::runtime_error &) {}

    std::vector<const char *> lines(151387, "abcd efghi");
    for (size_t i = 0; i < 6582; ++i) lines[i] = "abcd efghij";
    lines.back() = "abcd f ghi"; // Extra separator, same byte size.
    gguf_set_arr_str(metadata, key, lines.data(), lines.size());
    if (!gguf_write_to_file(metadata, path, true) || modelship_load_qwen_vocab(path)) return 1;
    lines.back() = "abcd efghi";
    gguf_set_arr_str(metadata, key, lines.data(), lines.size());
    if (!gguf_write_to_file(metadata, path, true) || !modelship_load_qwen_vocab(path) || modelship_load_qwen_vocab(path)) return 1;
    std::string expected;
    for (auto *line : lines) expected.append(line).push_back('\n');
    if (load_qwen2_merges() != expected) return 1;
    gguf_free(metadata);
    ggml_free(tensors);
    std::puts("External Qwen vocab accepts metadata-only GGUF and rejects invalid input without fallback");
}
