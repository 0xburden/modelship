#include "stable-diffusion.h"

#ifdef MODELSHIP_SD_CUDA
#include "ggml-backend.h"
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
    if (ggml_backend_cuda_get_device_count() == 1 && dev != nullptr &&
        ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        auto *backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            auto *type = ggml_backend_get_default_buffer_type(backend);
            // Do not accept a host allocation as a successful CUDA preflight.
            if (ggml_backend_is_cuda(backend) && type != nullptr && !ggml_backend_buft_is_host(type)) {
                auto *buffer = ggml_backend_alloc_buffer(backend, 4096);
                if (buffer != nullptr) {
                    ggml_backend_buffer_free(buffer);
                    ggml_backend_free(backend);
                    std::puts("MODELSHIP_SD_CUDA0_ALLOCATED");
                    return 0;
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
