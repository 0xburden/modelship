#pragma once

#include "ggml-backend.h"
#include <cstring>

namespace modelship {
inline bool cuda0(ggml_backend_dev_t device) {
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const auto registry = ggml_backend_dev_backend_reg(device);
    return registry && std::strcmp(ggml_backend_reg_name(registry), "CUDA") == 0 &&
           std::strcmp(ggml_backend_dev_name(device), "CUDA0") == 0;
}

inline bool resident_type(ggml_backend_dev_t device, ggml_backend_buffer_type_t type) {
    // Demand the device's ordinary buffer, not split, host or staging storage.
    return cuda0(device) && type && ggml_backend_buft_get_device(type) == device &&
           type == ggml_backend_dev_buffer_type(device) && !ggml_backend_buft_is_host(type);
}

inline bool resident_tensor(ggml_backend_dev_t device, const ggml_tensor * tensor) {
    return tensor && tensor->data && tensor->buffer &&
           resident_type(device, ggml_backend_buffer_get_type(tensor->buffer));
}
} // namespace modelship
