#include "stable-diffusion.h"

#ifdef MODELSHIP_SD_CUDA
#include "cuda-residency.hpp"
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int cuda_check() {
#ifdef MODELSHIP_SD_CUDA
    if (std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        std::fputs("Unset GGML_CUDA_ENABLE_UNIFIED_MEMORY for the CUDA probe.\n", stderr);
        return 1;
    }
    auto *dev = ggml_backend_dev_by_name("CUDA0");
    if (ggml_backend_cuda_get_device_count() == 1 && modelship::cuda0(dev)) {
        auto *backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            auto *type = ggml_backend_get_default_buffer_type(backend);
            // Do not accept a host allocation as a successful CUDA preflight.
            if (ggml_backend_is_cuda(backend) && modelship::resident_type(dev, type)) {
                auto *buffer = ggml_backend_alloc_buffer(backend, 4096);
                if (buffer != nullptr) {
                    const bool resident = modelship::resident_type(dev, ggml_backend_buffer_get_type(buffer));
                    ggml_backend_buffer_free(buffer);
                    if (resident) {
                        ggml_backend_free(backend);
                        std::puts("MODELSHIP_SD_CUDA0_ALLOCATED");
                        return 0;
                    }
                }
            }
            ggml_backend_free(backend);
        }
    }
#endif
    std::fputs("Exactly one usable visible CUDA GPU is required; no CPU fallback.\n", stderr);
    return 1;
}
}

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("stable-diffusion.cpp %s\n", sd_commit());
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--cuda-check") == 0) {
        return cuda_check();
    }
    std::fputs("usage: modelship-sd-probe --version | --cuda-check\n", stderr);
    return 2;
}
