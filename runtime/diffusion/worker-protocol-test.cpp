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
                   std::initializer_list<unsigned char> wanted_tags) {
    const std::unique_ptr<FILE, decltype(&std::fclose)> in(std::tmpfile(), std::fclose);
    const std::unique_ptr<FILE, decltype(&std::fclose)> out(std::tmpfile(), std::fclose);
    check(in && out, "temporary file");
    check(std::fwrite(input.data(), 1, input.size(), in.get()) == input.size(), "write input");
    std::rewind(in.get());
    int calls = 0;
    const int status = serve(in.get(), out.get(), [&](uint32_t seed, std::string_view prompt, FILE *reply) {
        check(seed == UINT32_MAX && prompt == "two cats", "request did not round trip");
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
    header[3] = static_cast<char>(4 + prompt.size());
    return header + std::string(4, static_cast<char>(0xff)) + prompt;
}
} // namespace

int main() {
    check(valid_prompt("白い猫 🐈\ncaption"), "valid UTF-8 rejected");
    check(valid_prompt(std::string(max_prompt, 'a')), "prompt bound rejected");
    for (const auto *invalid : {"", " \t\n", "<image:source>", "<lora:style:1>", "<|vision_start|>",
                                "\xc0\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x82", "\x1b"}) {
        check(!valid_prompt(invalid), "invalid prompt accepted");
    }
    check(!valid_prompt(std::string_view("abc\0xyz", 7)), "embedded NUL accepted");
    check(!valid_prompt(std::string(max_prompt + 1, 'a')), "oversized prompt accepted");
    const std::string valid = request("two cats");
    check_session("", 0, 0, false, {'L'});
    check_session(request("<image:source>") + valid + valid, 0, 2, false, {'L', 'V', 'I', 'I'});
    check_session(valid + valid, 1, 1, true, {'L', 'F'});
    check_session(valid.substr(0, 3), 1, 0, false, {'L'});
    check_session(valid.substr(0, valid.size() - 1), 1, 0, false, {'L'});
    check_session(std::string(4, static_cast<char>(0xff)), 1, 0, false, {'L'});
    check_session(std::string(4, '\0'), 1, 0, false, {'L'});
    std::puts("MODELSHIP_SD_PROTOCOL_TEST_PASSED");
}
