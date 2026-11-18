#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace modelship::diffusion {
constexpr uint32_t side = 512;
constexpr uint32_t max_prompt = 1024;
constexpr unsigned char protocol = 1;

inline uint32_t read_u32(const unsigned char *p) {
    return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]);
}

inline bool valid_prompt(std::string_view prompt) {
    // The pinned conditioner interprets attention punctuation before tokenizing,
    // which can both synthesize role tags and multiply user weights. The first
    // profile accepts neither that syntax nor angle-bracket tags. Don't strip
    // or escape it: that could change the prompt or reach a different parser.
    if (prompt.empty() || prompt.size() > max_prompt || prompt.find_first_of("<>()[\\]") != prompt.npos ||
        prompt.find("BREAK") != prompt.npos) {
        return false;
    }
    bool nonblank = false;
    for (size_t i = 0; i < prompt.size();) {
        const auto first = static_cast<unsigned char>(prompt[i++]);
        uint32_t cp = first;
        if (first >= 0x80) {
            unsigned tail;
            uint32_t minimum;
            if (first >= 0xc2 && first <= 0xdf) {
                tail = 1; minimum = 0x80; cp = first & 0x1f;
            } else if (first >= 0xe0 && first <= 0xef) {
                tail = 2; minimum = 0x800; cp = first & 0x0f;
            } else if (first >= 0xf0 && first <= 0xf4) {
                tail = 3; minimum = 0x10000; cp = first & 0x07;
            } else {
                return false;
            }
            if (prompt.size() - i < tail) return false;
            for (; tail; --tail) {
                const auto next = static_cast<unsigned char>(prompt[i++]);
                if ((next & 0xc0) != 0x80) return false;
                cp = (cp << 6) | (next & 0x3f);
            }
            if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
        }
        if ((cp < 0x20 && cp != '\t' && cp != '\n') || (cp >= 0x7f && cp <= 0x9f)) return false;
        nonblank |= cp != ' ' && cp != '\t' && cp != '\n';
    }
    return nonblank;
}

inline bool write_frame(FILE *out, unsigned char tag, unsigned char field = 0,
                        const void *pixels = nullptr, uint32_t bytes = 0) {
    const uint32_t length = 1 + (tag == 'L' || tag == 'I' ? 1 : 0) + bytes;
    const unsigned char header[] = {static_cast<unsigned char>(length >> 24),
                                    static_cast<unsigned char>(length >> 16),
                                    static_cast<unsigned char>(length >> 8),
                                    static_cast<unsigned char>(length), tag, field};
    const size_t header_size = tag == 'L' || tag == 'I' ? 6 : 5;
    return std::fwrite(header, 1, header_size, out) == header_size &&
           (!bytes || std::fwrite(pixels, 1, bytes, out) == bytes) && std::fflush(out) == 0;
}

// Engine sends a bounded image only after successful inference and device drain.
// One request at a time. A bad frame or uncertain completion ends the process;
// a bounded, well-framed invalid prompt can be rejected without using the engine.
template <typename Generate>
int serve(FILE *in, FILE *out, Generate generate) {
    if (!write_frame(out, 'L', protocol)) return 1;
    std::array<unsigned char, 4 + max_prompt> payload{};
    for (;;) {
        unsigned char header[4];
        const size_t got = std::fread(header, 1, sizeof(header), in);
        if (got == 0 && std::feof(in) && !std::ferror(in)) return 0;
        if (got != sizeof(header)) return 1;
        const uint32_t length = read_u32(header);
        if (length < 5 || length > payload.size() ||
            std::fread(payload.data(), 1, length, in) != length) return 1;
        const std::string_view prompt(reinterpret_cast<const char *>(payload.data() + 4), length - 4);
        if (!valid_prompt(prompt)) {
            if (!write_frame(out, 'V')) return 1;
            continue;
        }
        if (!generate(read_u32(payload.data()), prompt, out)) {
            // Don't append another frame to an already broken output stream.
            if (!std::ferror(out)) write_frame(out, 'F');
            return 1;
        }
    }
}
} // namespace modelship::diffusion
