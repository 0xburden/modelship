#include "core/util.h"

#include <cstdio>

int main() {
    for (const std::string text : {"", "Café at dawn", "(word:2) [word] \\ BREAK", "<|image_pad|>"}) {
        const auto output = parse_prompt_attention(text);
        if (output.size() != 1 || output.front().first != text || output.front().second != 1.0f) return 1;
    }
    std::puts("Only literal unit-weight prompt segments are produced");
}
