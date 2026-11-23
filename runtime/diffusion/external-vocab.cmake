# Replace only the pinned translation unit in the disposable tree, so none
# of the upstream model-derived tokenizer arrays are compiled into the ELFs.
set(vocab_source "${SD_SOURCE}/src/tokenizers/vocab/vocab.cpp")
file(SHA256 "${vocab_source}" actual)
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/qwen-vocab.cpp" replacement)
if(NOT actual STREQUAL replacement AND NOT actual STREQUAL "d52fff67f47e3eaf7123bc7f9b7800674001f1f7b7b47d8a3222b70eb19e7dad")
    message(FATAL_ERROR "Unexpected upstream vocabulary translation unit")
endif()
configure_file("${CMAKE_CURRENT_LIST_DIR}/qwen-vocab.cpp" "${vocab_source}" COPYONLY)
configure_file("${CMAKE_CURRENT_LIST_DIR}/qwen-vocab.hpp" "${SD_SOURCE}/src/tokenizers/vocab/modelship_qwen_vocab.hpp" COPYONLY)
