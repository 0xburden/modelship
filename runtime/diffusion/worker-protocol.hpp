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
// Edit candidate only: exactly one frame with a uint32 seed, uint16 prompt
// length, prompt bytes, and 512x512 interleaved RGB. No paths, dimensions,
// encoded image formats, or additional references cross this boundary.
constexpr unsigned char edit_protocol = 2;
constexpr uint32_t reference_bytes = 3 * side * side;

template <typename Edit>
int serve_edit(FILE *in, FILE *out, Edit edit) {
    if (!write_frame(out, 'L', edit_protocol)) return 1;
    unsigned char header[4];
    unsigned char fields[6];
    if (std::fread(header, 1, sizeof(header), in) != sizeof(header)) return 1;
    const uint32_t length = read_u32(header);
    if (length < sizeof(fields) + 1 + reference_bytes ||
        length > sizeof(fields) + max_prompt + reference_bytes ||
        std::fread(fields, 1, sizeof(fields), in) != sizeof(fields)) return 1;
    const uint32_t text_size = uint32_t(fields[4]) << 8 | uint32_t(fields[5]);
    if (text_size == 0 || text_size > max_prompt ||
        length != sizeof(fields) + text_size + reference_bytes) return 1;
    std::array<char, max_prompt> text{};
    if (std::fread(text.data(), 1, text_size, in) != text_size) return 1;
    const std::string_view prompt(text.data(), text_size);
    if (!valid_prompt(prompt)) {
        write_frame(out, 'V');
        return 1;
    }
    std::array<unsigned char, reference_bytes> pixels;
    if (std::fread(pixels.data(), 1, pixels.size(), in) != pixels.size() ||
        std::fgetc(in) != EOF || !std::feof(in) || std::ferror(in)) return 1;
    if (!edit(read_u32(fields), prompt, pixels.data(), out)) {
        if (!std::ferror(out)) write_frame(out, 'F');
        return 1;
    }
    return 0;
}

// Resident edit-only worker. Keep the one-shot v2 EOF contract separate. Here
// only frame boundaries may end cleanly; never scan for a new frame after a
// bad length, truncated payload, or engine failure.
constexpr unsigned char edit_service_protocol = 3;

template <typename Edit>
int serve_edit_stream(FILE *in, FILE *out, Edit edit) {
    if (!write_frame(out, 'L', edit_service_protocol)) return 1;
    std::array<unsigned char, reference_bytes> pixels;
    std::array<char, max_prompt> text;
    for (;;) {
        unsigned char header[4];
        const size_t got = std::fread(header, 1, sizeof(header), in);
        if (got == 0 && std::feof(in) && !std::ferror(in)) return 0;
        if (got != sizeof(header)) return 1;
        unsigned char fields[6];
        const uint32_t length = read_u32(header);
        if (length < sizeof(fields) + 1 + reference_bytes ||
            length > sizeof(fields) + max_prompt + reference_bytes ||
            std::fread(fields, 1, sizeof(fields), in) != sizeof(fields)) return 1;
        const uint32_t text_size = uint32_t(fields[4]) << 8 | uint32_t(fields[5]);
        if (text_size == 0 || text_size > max_prompt || length != sizeof(fields) + text_size + reference_bytes ||
            std::fread(text.data(), 1, text_size, in) != text_size ||
            std::fread(pixels.data(), 1, pixels.size(), in) != pixels.size()) return 1;
        const std::string_view prompt(text.data(), text_size);
        if (!valid_prompt(prompt)) {
            if (!write_frame(out, 'V')) return 1;
            continue;
        }
        if (!edit(read_u32(fields), prompt, pixels.data(), out)) {
            if (!std::ferror(out)) write_frame(out, 'F');
            return 1;
        }
    }
}
} // namespace modelship::diffusion
