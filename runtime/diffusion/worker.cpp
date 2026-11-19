#include "stable-diffusion.h"
#include "worker-protocol.hpp"

#include <cstring>
#include <memory>
#include <string>

#ifdef MODELSHIP_SD_CUDA
#include "cuda-residency.hpp"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>

#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace modelship::diffusion;
using Options = std::unique_ptr<sd_opt_t, decltype(&sd_opt_free)>;
using Context = std::unique_ptr<sd_ctx_t, decltype(&sd_ctx_free)>;
using Result = std::unique_ptr<sd_result_t, decltype(&sd_result_free)>;
using Stream = std::unique_ptr<FILE, decltype(&std::fclose)>;

// The supervising process must check all component digests and keep the
// bundle private/unchanged. These checks also catch wrong wiring before load.
bool local_component(const std::string &path, uint64_t bytes) {
    struct stat st{};
    return lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 &&
           static_cast<uint64_t>(st.st_size) == bytes;
}

Context load(const std::string &dir) {
    const std::string model = dir + "/qwen_image_2.1-Q4_K.gguf";
    const std::string encoder = dir + "/Qwen3VL-8B-Instruct-Q4_K_M.gguf";
    const std::string vae = dir + "/qwen_image_2.1_vae_bf16.safetensors";
    Context context(nullptr, sd_ctx_free);
    if (!local_component(model, 4197494816) || !local_component(encoder, 5027784800) ||
        !local_component(vae, 675509688)) return context;

#ifdef MODELSHIP_SD_EDIT
    const std::string projector = dir + "/mmproj-Qwen3VL-8B-Instruct-F16.gguf";
    if (!local_component(projector, 1159029824)) return context;
#endif
    const Options opt(sd_ctx_opt_create(), sd_opt_free);
    if (!opt || sd_opt_set_string(opt.get(), "diffusion_model_path", model.c_str()) != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "llm_path", encoder.c_str()) != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "vae_path", vae.c_str()) != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "backend", "CUDA0") != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "params_backend", "CUDA0") != SD_STATUS_OK ||
        sd_opt_set_bool(opt.get(), "auto_fit", false) != SD_STATUS_OK ||
        sd_opt_set_bool(opt.get(), "enable_mmap", false) != SD_STATUS_OK ||
        sd_opt_set_bool(opt.get(), "eager_load", true) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "conditioning_cache_size", 0) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "n_threads", 2) != SD_STATUS_OK) return context;
#ifdef MODELSHIP_SD_EDIT
    if (sd_opt_set_string(opt.get(), "llm_vision_path", projector.c_str()) != SD_STATUS_OK) return context;
#endif
    sd_ctx_t *raw = nullptr;
    const auto status = sd_ctx_create(opt.get(), &raw);
    context.reset(raw);
    const char *version = context ? sd_get_model_version_name(context.get()) : nullptr;
    if (cudaDeviceSynchronize() != cudaSuccess || status != SD_STATUS_OK || !context ||
        !sd_ctx_supports_image_generation(context.get()) || !version ||
        std::strcmp(version, "Qwen Image 2.1") != 0) context.reset();
    return context;
}

bool generate(sd_ctx_t *ctx, uint32_t seed, std::string_view prompt, FILE *out
#ifdef MODELSHIP_SD_EDIT
              , const unsigned char *reference
#endif
) {
    const Options opt(sd_img_gen_opt_create(), sd_opt_free);
    const std::string text(prompt);
    if (!opt || sd_opt_set_string(opt.get(), "prompt", text.c_str()) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "seed", seed) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "width", side) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "height", side) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "batch_count", 1) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "sample.steps", 40) != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "sample.method", "euler") != SD_STATUS_OK ||
        sd_opt_set_float(opt.get(), "sample.guidance.txt_cfg", 6.0) != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "cache.mode", "disabled") != SD_STATUS_OK) return false;
#ifdef MODELSHIP_SD_EDIT
    const int64_t shape[] = {3, side, side};
    const std::unique_ptr<sd_tensor_t, decltype(&sd_tensor_free)>
        tensor_ref(sd_tensor_create(SD_TENSOR_U8, 3, shape, reference, reference_bytes), sd_tensor_free);
    const sd_tensor_t *refs[] = {tensor_ref.get()};
    if (!tensor_ref || sd_opt_set_tensors(opt.get(), "ref_images", refs, 1) != SD_STATUS_OK) return false;
#endif
    sd_result_t *raw = nullptr;
    const auto status = sd_generate_image(ctx, opt.get(), &raw);
    const Result result(raw, sd_result_free);
    // Attempt a whole-device drain even if the model returned an error.
    if (cudaDeviceSynchronize() != cudaSuccess || status != SD_STATUS_OK || !result ||
        sd_result_get_kind(result.get()) != SD_RESULT_IMAGE || sd_result_get_image_count(result.get()) != 1) return false;
    const auto *tensor = sd_result_get_image(result.get(), 0);
    if (!tensor || sd_tensor_rank(tensor) != 3 || sd_tensor_type(tensor) != SD_TENSOR_U8 ||
        sd_tensor_dim(tensor, 1) != side || sd_tensor_dim(tensor, 2) != side) return false;
    const auto channels = sd_tensor_dim(tensor, 0);
    if (channels != 3 && channels != 4) return false;
    const uint32_t bytes = static_cast<uint32_t>(channels) * side * side;
    const void *pixels = sd_tensor_data(tensor);
    return pixels && sd_tensor_byte_size(tensor) == bytes &&
           write_frame(out, 'I', static_cast<unsigned char>(channels), pixels, bytes);
}

int worker(const char *dir) {
    if (dir[0] != '/' || isatty(STDIN_FILENO) || isatty(STDOUT_FILENO) ||
        std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) return 2;
    // Reserve the protocol pipe. Suppress upstream stdout/stderr (including
    // diagnostics that may contain user prompts and file paths).
    const int fd = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    if (fd < 0) return 1;
    const Stream protocol_out(fdopen(fd, "wb"), std::fclose);
    if (!protocol_out) { close(fd); return 1; }
    const int discard = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (discard < 0) return 1;
    const bool redirected = dup2(discard, STDOUT_FILENO) >= 0 && dup2(discard, STDERR_FILENO) >= 0;
    close(discard);
    if (!redirected || std::signal(SIGPIPE, SIG_IGN) == SIG_ERR) return 1;

    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count == 1 &&
            cudaSetDevice(0) == cudaSuccess && ggml_backend_cuda_get_device_count() == 1 &&
            modelship::cuda0(ggml_backend_dev_by_name("CUDA0"))) {
            const Context ctx = load(dir);
            if (ctx) {
#ifdef MODELSHIP_SD_EDIT
                return serve_edit(stdin, protocol_out.get(), [&](uint32_t seed, std::string_view prompt,
                                                                const unsigned char *reference, FILE *out) {
                    return generate(ctx.get(), seed, prompt, out, reference);
                });
#else
                return serve(stdin, protocol_out.get(), [&](uint32_t seed, std::string_view prompt, FILE *out) {
                    return generate(ctx.get(), seed, prompt, out);
                });
#endif
            }
        }
        write_frame(protocol_out.get(), 'F');
        return 1;
    } catch (...) {
        // An unexpected exception must not be followed by another request or
        // reported as a clean drain. The parent must reap the child.
        return 1;
    }
}
} // namespace
#endif

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
#ifdef MODELSHIP_SD_EDIT
        std::printf("modelship-sd-edit-candidate protocol 2 stable-diffusion.cpp %s\n", sd_commit());
#else
        std::printf("modelship-sd-worker protocol 1 stable-diffusion.cpp %s\n", sd_commit());
#endif
        return 0;
    }
#ifdef MODELSHIP_SD_CUDA
#ifdef MODELSHIP_SD_EDIT
    if (argc == 3 && std::strcmp(argv[1], "--private-edit-stdio") == 0) return worker(argv[2]);
#else
    if (argc == 3 && std::strcmp(argv[1], "--private-stdio") == 0) return worker(argv[2]);
#endif
#else
    if (argc == 3 && std::strcmp(argv[1], "--private-stdio") == 0) {
        std::fputs("This worker requires CUDA; no CPU fallback.\n", stderr);
        return 1;
    }
#endif
    std::fputs("Private candidate only (no CLI deployment or HTTP server).\n", stderr);
    return 2;
}
