# Apply only in the disposable, pinned stable-diffusion.cpp source tree.
set(file "${SD_SOURCE}/src/model_manager.cpp")
set(original "29227c977ba16504510711b2617c640c2264dd46e0a638b0d2472ec66b974880")
set(patched "d45f4b2a42a5115ab1356d10f296bb0024f670ab8d78a4108a10d36e3a3cd96e")
file(SHA256 "${file}" digest)
if(NOT digest STREQUAL patched)
    if(NOT digest STREQUAL original)
        message(FATAL_ERROR "Unexpected model manager source for CUDA residency guard")
    endif()
    file(READ "${file}" source)
    string(REPLACE [=[#include "model_manager.h"
]=] [=[#include "model_manager.h"
#include "modelship_cuda_residency.hpp"
#include <cstdlib>
]=] source "${source}")
    set(find_load [=[bool ModelManager::load_tensors_to_params_backend(const std::vector<TensorState*>& states) {
]=])
    set(guard_load [=[    if (std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        LOG_ERROR("Modelship: CUDA unified memory is not supported");
        return false;
    }
    const auto resident = [](const TensorState* state) {
        return modelship::resident_tensor(ggml_backend_get_device(state->params_backend), state->tensor);
    };
]=])
    string(REPLACE "${find_load}" "${find_load}${guard_load}" source "${source}")
    set(find_need_load [=[        if (!state->loaded_to_params_backend) {
            need_load.push_back(state);
        }
]=])
    set(guard_need_load [=[        if (state->residency_mode != ResidencyMode::ParamBackend || state->split_buffer_type ||
            !state->params_backend || state->params_backend != state->compute_backend ||
            !modelship::cuda0(ggml_backend_get_device(state->params_backend))) {
            LOG_ERROR("Modelship: model parameters require unsplit CUDA0 residency");
            return false;
        }
        if (state->loaded_to_params_backend) {
            if (!resident(state)) {
                LOG_ERROR("Modelship: previously loaded model parameters are not resident on CUDA0");
                return false;
            }
        } else {
            need_load.push_back(state);
        }
]=])
    string(REPLACE "${find_need_load}" "${guard_need_load}" source "${source}")
    set(find_mem_load [=[    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
        !load_tensors(need_load)) {
]=])
    set(guard_mem_load [=[    const bool allocated = alloc_params_buffers(need_alloc, created_storage_blocks);
    const bool all_resident = allocated && std::all_of(need_load.begin(), need_load.end(), resident);
    if (allocated && !all_resident) {
        LOG_ERROR("Modelship: allocated model parameters are not resident on CUDA0");
    }
    if (!all_resident || !load_tensors(need_load)) {
]=])
    string(REPLACE "${find_mem_load}" "${guard_mem_load}" source "${source}")
    set(host_buffer_helper [=[static bool backend_supports_host_buffer(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    if (sd_backend_is_cpu(backend)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.buffer_from_host_ptr;
}

]=])
    string(REPLACE "${host_buffer_helper}" "" source "${source}")
    set(find_mmap [=[    if (state.sources.size() > 1 || state.source_file != 0 || !enable_mmap_ || state.residency_mode != ResidencyMode::ParamBackend) {
        return false;
    }
    if (state.compute_backend == nullptr || state.params_backend == nullptr) {
        return false;
    }
    return sd_backend_is_cpu(state.compute_backend) ||
           sd_backend_is_cpu(state.params_backend) ||
           backend_supports_host_buffer(state.compute_backend);
]=])
    string(REPLACE "${find_mmap}" [=[    (void)state;
    return false; // No host mapping as persistent parameter storage.
]=] source "${source}")
    set(find_alloc [=[        if (params_buft == nullptr) {
            return false;
        }
        states_by_buffer_type]=])
    set(guard_alloc [=[        if (!modelship::resident_type(ggml_backend_get_device(state->params_backend), params_buft)) {
            LOG_ERROR("Modelship: nonresident parameter buffer refused before allocation");
            return false;
        }
        states_by_buffer_type]=])
    string(REPLACE "${find_alloc}" "${guard_alloc}" source "${source}")
    set(find_evict [=[    for (const auto& block : params_storage_blocks_) {
        // Loader-retained mappings cannot be reclaimed by freeing a storage block.
        if (block->mmap_tensor_stores.empty()) {
            append(block->buffer, block->states, true);
        }
    }
    return result;
]=])
    string(REPLACE "${find_evict}" [=[    // Only transient working buffers can be reclaimed. Do not demote model
    // parameters to disk under graph workspace pressure.
    return result;
]=] source "${source}")
    string(SHA256 digest "${source}")
    if(NOT digest STREQUAL patched)
        message(FATAL_ERROR "CUDA residency guard did not match pinned model manager: ${digest}")
    endif()
    file(WRITE "${file}" "${source}")
endif()
configure_file("${CMAKE_CURRENT_LIST_DIR}/cuda-residency.hpp" "${SD_SOURCE}/src/modelship_cuda_residency.hpp" COPYONLY)
