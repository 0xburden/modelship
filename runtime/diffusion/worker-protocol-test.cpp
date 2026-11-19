#include "worker-protocol.hpp"

#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <string>

using namespace modelship::diffusion;
namespace {
void check(bool ok, const char *message) {
    if (ok) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

void check_session(const std::string &input, int wanted_status, int wanted_calls, bool fail,
                   std::initializer_list<unsigned char> wanted_tags, std::string_view wanted_prompt = "two cats") {
    const std::unique_ptr<FILE, decltype(&std::fclose)> in(std::tmpfile(), std::fclose);
    const std::unique_ptr<FILE, decltype(&std::fclose)> out(std::tmpfile(), std::fclose);
    check(in && out, "temporary file");
    check(std::fwrite(input.data(), 1, input.size(), in.get()) == input.size(), "write input");
    std::rewind(in.get());
    int calls = 0;
    const int status = serve(in.get(), out.get(), [&](uint32_t seed, std::string_view prompt, FILE *reply) {
        check(seed == UINT32_MAX && prompt == wanted_prompt, "request did not round trip");
        ++calls;
        if (fail) return false;
        static const std::array<unsigned char, 4 * side * side> pixels{};
        return write_frame(reply, 'I', 4, pixels.data(), pixels.size());
    });
    check(status == wanted_status && calls == wanted_calls, "wrong worker status or engine calls");
    std::rewind(out.get());
    for (const unsigned char tag : wanted_tags) {
        unsigned char header[5];
        check(std::fread(header, 1, sizeof(header), out.get()) == sizeof(header), "missing frame");
        check(header[4] == tag, "unexpected reply tag");
        if (tag == 'L') {
            check(read_u32(header) == 2 && std::fgetc(out.get()) == protocol, "invalid load signal");
        } else if (tag == 'I') {
            check(read_u32(header) == 2 + 4 * side * side && std::fgetc(out.get()) == 4, "invalid image shape");
            for (uint32_t i = 0; i < 4 * side * side; ++i) check(std::fgetc(out.get()) == 0, "invalid pixels");
        } else {
            check(read_u32(header) == 1, "invalid error frame");
        }
    }
    check(std::fgetc(out.get()) == EOF, "unexpected extra protocol bytes");
}

std::string request(const std::string &prompt) {
    std::string header(4, '\0');
    const auto length = static_cast<uint32_t>(4 + prompt.size());
    for (unsigned i = 0; i < 4; ++i) header[i] = static_cast<char>(length >> (24 - 8 * i));
    return header + std::string(4, static_cast<char>(0xff)) + prompt;
}
std::string edit_request(const std::string &prompt) {
    std::string frame(10, '\0');
    const auto length = static_cast<uint32_t>(6 + prompt.size() + reference_bytes);
    for (unsigned i = 0; i < 4; ++i) frame[i] = static_cast<char>(length >> (24 - 8 * i));
    frame[7] = 43;
    frame[8] = static_cast<char>(prompt.size() >> 8);
    frame[9] = static_cast<char>(prompt.size());
    return frame + prompt + std::string(reference_bytes - 1, 'x') + 'y';
}

void check_edit(const std::string &input, std::string_view wanted_prompt, bool success,
                int wanted_calls, std::string_view tail) {
    const std::unique_ptr<FILE, decltype(&std::fclose)> in(std::tmpfile(), std::fclose);
    const std::unique_ptr<FILE, decltype(&std::fclose)> out(std::tmpfile(), std::fclose);
    check(in && out, "temporary edit files");
    check(std::fwrite(input.data(), 1, input.size(), in.get()) == input.size(), "write edit fixture");
    std::rewind(in.get());
    int calls = 0;
    const int status = serve_edit(in.get(), out.get(), [&](uint32_t seed, std::string_view prompt,
                                                        const unsigned char *rgb, FILE *) {
        ++calls;
        check(seed == 43 && prompt == wanted_prompt, "edit seed or prompt did not round trip");
        check(rgb[0] == 'x' && rgb[reference_bytes - 1] == 'y', "edit RGB did not round trip");
        return success;
    });
    check(status == (success ? 0 : 1) && calls == wanted_calls, "wrong edit status or engine calls");
    std::rewind(out.get());
    for (const int byte : {0, 0, 0, 2, int('L'), int(edit_protocol)}) {
        check(std::fgetc(out.get()) == byte, "invalid edit handshake");
    }
    for (const char byte : tail) check(std::fgetc(out.get()) == byte, "invalid edit failure status");
    check(std::fgetc(out.get()) == EOF, "unexpected extra edit bytes");
}

void check_edit_stream(const std::string &input, int expected_status, int wanted_calls,
                       bool engine_ok, std::string_view replies) {
    const std::unique_ptr<FILE, decltype(&std::fclose)> in(std::tmpfile(), std::fclose);
    const std::unique_ptr<FILE, decltype(&std::fclose)> out(std::tmpfile(), std::fclose);
    check(in && out, "temporary resident edit files");
    check(std::fwrite(input.data(), 1, input.size(), in.get()) == input.size(), "write resident edit fixture");
    std::rewind(in.get());
    int calls = 0;
    const int status = serve_edit_stream(in.get(), out.get(), [&](uint32_t seed, std::string_view prompt,
                                                               const unsigned char *rgb, FILE *reply) {
        check(seed == 43 && rgb[reference_bytes - 1] == 'y', "resident edit fixture changed");
        check(calls == 0 ? prompt == "make it blue" && rgb[0] == 'x'
                         : prompt == "make it green" && rgb[0] == 'z', "resident edit reused prior input");
        ++calls;
        if (!engine_ok) return false;
        return write_frame(reply, 'I', 3, rgb, reference_bytes);
    });
    check(status == expected_status && calls == wanted_calls, "wrong resident edit status or engine calls");
    std::rewind(out.get());
    for (const int byte : {0, 0, 0, 2, int('L'), int(edit_service_protocol)}) {
        check(std::fgetc(out.get()) == byte, "invalid resident edit handshake");
    }
    int images = 0;
    for (const char tag : replies) {
        unsigned char header[5];
        check(std::fread(header, 1, sizeof(header), out.get()) == sizeof(header), "missing resident reply");
        check(header[4] == tag, "wrong resident reply tag");
        if (tag == 'I') {
            check(read_u32(header) == 2 + reference_bytes && std::fgetc(out.get()) == 3, "wrong resident image shape");
            check(std::fgetc(out.get()) == (images++ ? 'z' : 'x'), "wrong resident image first pixel byte");
            check(std::fseek(out.get(), reference_bytes - 2, SEEK_CUR) == 0 && std::fgetc(out.get()) == 'y', "wrong resident image tail");
        } else {
            check(read_u32(header) == 1, "wrong resident failure length");
        }
    }
    check(std::fgetc(out.get()) == EOF, "unexpected extra resident edit bytes");
}
} // namespace

int main() {
    check(valid_prompt("白い猫 🐈\ncaption"), "valid UTF-8 rejected");
    check(valid_prompt(std::string(max_prompt, 'a')), "prompt bound rejected");
    for (const auto *invalid : {"", " \t\n", "<image:source>", "<lora:style:1>", "<|vision_start|>",
                                "<tool_call>", "</think>", "<to()ol_call>", "((cat:999999999999999999999):999999999999999999999)",
                                "cat [blue]", R"(cat \ dog)", "cat BREAK dog", "\xc0\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x82", "\x1b"}) {
        check(!valid_prompt(invalid), "invalid prompt accepted");
    }
    check(!valid_prompt(std::string_view("abc\0xyz", 7)), "embedded NUL accepted");
    check(!valid_prompt(std::string(max_prompt + 1, 'a')), "oversized prompt accepted");
    const std::string valid = request("two cats");
    check_session("", 0, 0, false, {'L'});
    check_session(request("<image:source>") + valid + valid, 0, 2, false, {'L', 'V', 'I', 'I'});
    const std::string at_limit(max_prompt, 'a');
    check_session(request(at_limit), 0, 1, false, {'L', 'I'}, at_limit);
    check_session(request(at_limit + 'a') + valid, 1, 0, false, {'L'});
    check_session(valid + valid, 1, 1, true, {'L', 'F'});
    check_session(valid.substr(0, 3), 1, 0, false, {'L'});
    check_session(valid.substr(0, valid.size() - 1), 1, 0, false, {'L'});
    check_session(std::string(4, static_cast<char>(0xff)), 1, 0, false, {'L'});
    check_session(std::string(4, '\0'), 1, 0, false, {'L'});
    const auto edit = edit_request("make it blue");
    check_edit(edit, "make it blue", true, 1, {});
    check_edit(edit_request(at_limit), at_limit, true, 1, {});
    check_edit(edit, "make it blue", false, 1, std::string_view("\0\0\0\1F", 5));
    check_edit(edit_request("<image1>"), {}, false, 0, std::string_view("\0\0\0\1V", 5));
    check_edit(edit_request(at_limit + 'a'), {}, false, 0, {});
    check_edit(edit.substr(0, edit.size() - 1), {}, false, 0, {});
    check_edit(edit + edit, {}, false, 0, {});
    check_edit(valid, {}, false, 0, {});
    auto wrong_length = edit;
    wrong_length[9]++;
    check_edit(wrong_length, {}, false, 0, {});
    auto edit2 = edit_request("make it green");
    edit2[edit2.size() - reference_bytes] = 'z';
    check_edit_stream("", 0, 0, true, "");
    check_edit_stream(edit + edit_request("<image1>") + edit2, 0, 2, true, "IVI");
    check_edit_stream(edit + edit2, 1, 1, false, "F");
    check_edit_stream(edit + edit2.substr(0, edit2.size() - 1), 1, 1, true, "I");
    check_edit_stream(edit + edit2.substr(0, 3), 1, 1, true, "I");
    check_edit_stream(wrong_length, 1, 0, true, "");
    check_edit_stream(valid, 1, 0, true, "");
    check_edit_stream(edit_request(at_limit + 'a'), 1, 0, true, "");
    std::puts("MODELSHIP_SD_PROTOCOL_TEST_PASSED");
}
