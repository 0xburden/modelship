# Candidate-specific, byte-pinned changes in the disposable stable-diffusion.cpp tree.
# Use the existing standard-library noise generator; do not include the port,
# even for the upstream model families this private Qwen worker cannot select.
function(modelship_rng_patch relative original patched)
    set(path "${SD_SOURCE}/${relative}")
    file(SHA256 "${path}" sha)
    if(sha STREQUAL patched)
        return()
    endif()
    if(NOT sha STREQUAL original)
        message(FATAL_ERROR "Unexpected source for standard RNG: ${relative}")
    endif()
    file(READ "${path}" text)
    math(EXPR last "${ARGC} - 2")
    foreach(index RANGE 3 ${last} 2)
        math(EXPR next "${index} + 1")
        string(REPLACE "${ARGV${index}}" "${ARGV${next}}" text "${text}")
    endforeach()
    string(SHA256 sha "${text}")
    if(NOT sha STREQUAL patched)
        message(FATAL_ERROR "Standard RNG patch did not match pinned source: ${relative}")
    endif()
    file(WRITE "${path}" "${text}")
endfunction()

modelship_rng_patch("src/pipeline/diffusion_engine.cpp"
    "5256a51a29a6ae0ad294612b2375bffe6d90656fe4e52f0767ece906ae8c41a5"
    "9b142f3e850a339cf73885d526667732c3b5fb80f99eb4c2f6cad29dc0e4faa7"
    [=[#include "core/rng_philox.hpp"
]=] [=[]=]
    [=[    } else {  // default: CUDA_RNG
        return std::make_shared<PhiloxRNG>();
    }]=] [=[    }
    throw std::invalid_argument("unsupported RNG");]=]
    [=[std::make_shared<PhiloxRNG>()]=] [=[std::make_shared<STDDefaultRNG>()]=]
    [=[#include "core/rng_mt19937.hpp"
]=] [=[#include "core/rng_mt19937.hpp"
#include <stdexcept>
]=])

modelship_rng_patch("src/runtime/denoiser.hpp"
    "21d30f30aa09ab915181dd7457ba80ddc7251c0a804a82c1aad78f16a5719cc2"
    "a0a7967468afcadb71389c070bb957a552186aa0d9615eeb1d0402dbc175935b"
    [=[#include "core/rng_philox.hpp"
]=] [=[]=]
    [=[PhiloxRNG]=] [=[STDDefaultRNG]=])

modelship_rng_patch("src/pipeline/seedvr2.cpp"
    "a7725f248fa2f277285575e6785e373e54ce1b842483af3484fd687ff38f37ab"
    "7f858f13f9a8ec7db03724fc3f6b62c34ef19101615cd179fb60a78b3b2caa5d"
    [=[#include "core/rng_philox.hpp"
]=] [=[#include "core/rng.hpp"
]=]
    [=[PhiloxRNG]=] [=[STDDefaultRNG]=])

modelship_rng_patch("src/pipeline/relay_vsr.cpp"
    "e1310ee71c060db9dcc48eef5c0c26d498674fe9cc699282a4c3b458d258c800"
    "b5511f048482897f0a8f6117d4f1e9815f2b91545a5eba0041d26aef9d41e1f3"
    [=[#include "core/rng_philox.hpp"
]=] [=[#include "core/rng.hpp"
]=]
    [=[PhiloxRNG]=] [=[STDDefaultRNG]=])

modelship_rng_patch("src/pipeline/video.cpp"
    "f57dfa169f60e78562d49c2468ebd21c2637de9d25b756953dad56dcfe793aed"
    "e7908ad52bddfb824fcc2d5f1a825c10a82a1323cb764017717f9497c2da68a2"
    [=[#include "core/rng_philox.hpp"
]=] [=[]=]
    [=[PhiloxRNG]=] [=[STDDefaultRNG]=])

modelship_rng_patch("src/config.cpp"
    "114d4b98204987b3d323021e7d7e83f4d6953e441816d90f31b90d553ff0f4a0"
    "050fd9b289377900a9a848d58af048021f6b5c794593a4ef2cabfd50887795ec"
    [=[rng_type                  = CUDA_RNG;]=] [=[rng_type                  = STD_DEFAULT_RNG;]=])

# Removing the exact file also makes any other include fail compilation.
set(port "${SD_SOURCE}/src/core/rng_philox.hpp")
if(EXISTS "${port}")
    file(SHA256 "${port}" sha)
    if(NOT sha STREQUAL "677fa3d5df0f368bb7a1c613e6cdecabafc0cf9a7676779fc9993a88c08ff8b1")
        message(FATAL_ERROR "Unexpected source for removed Philox port")
    endif()
    file(REMOVE "${port}")
endif()
