# Replace the upstream attention parser in the disposable source (all callers).
set(path "${SD_SOURCE}/src/core/util.cpp")
file(SHA256 "${path}" digest)
if(digest STREQUAL "59ad584a3bc20cf60268fe120f1b7a3578977004a0bdb3a95005679fa75b1beb")
    return()
endif()
if(NOT digest STREQUAL "cb7b7e300872da17833ea6f2c97f9469884f5b4668dcc8a27cc676f39354fe50")
    message(FATAL_ERROR "Unexpected source for literal prompt support")
endif()
file(READ "${path}" text)
string(FIND "${text}" "// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/prompt_parser.py#L345" start)
string(FIND "${text}" "static size_t get_utf8_char_len(char c) {" end)
if(start LESS 0 OR end LESS_EQUAL start)
    message(FATAL_ERROR "Cannot locate pinned attention parser")
endif()
string(SUBSTRING "${text}" 0 ${start} prefix)
string(SUBSTRING "${text}" ${end} -1 suffix)
set(literal [=[// The private Qwen worker accepts literal text only.
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    return {{text, 1.0f}};
}

]=])
set(text "${prefix}${literal}${suffix}")
string(SHA256 digest "${text}")
if(NOT digest STREQUAL "59ad584a3bc20cf60268fe120f1b7a3578977004a0bdb3a95005679fa75b1beb")
    message(FATAL_ERROR "Literal prompt patch did not match pinned source")
endif()
file(WRITE "${path}" "${text}")
