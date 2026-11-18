# Only for this pinned, disposable ggml source tree. A CMake reconfigure is
# idempotent; any other input bytes (including a partial patch) fail closed.
function(modelship_patch file original patched needle replacement)
    file(SHA256 "${file}" digest)
    if(digest STREQUAL patched)
        return()
    endif()
    if(NOT digest STREQUAL original)
        message(FATAL_ERROR "Unexpected ggml source for CUDA placement guard: ${file}")
    endif()
    file(READ "${file}" source)
    string(REPLACE "${needle}" "${replacement}" result "${source}")
    string(SHA256 digest "${result}")
    if(NOT digest STREQUAL patched)
        message(FATAL_ERROR "CUDA placement guard did not match pinned source: ${file}")
    endif()
    file(WRITE "${file}" "${result}")
endfunction()

set(backend_file "${MODELSHIP_GGML_SOURCE}/src/ggml-backend.cpp")
# Retain CPU registration (required by the upstream allocator), but admit only
# CUDA0 for actual compute. Check all scheduler splits before starting any.
set(backend_anchor [=[enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);
]=])
set(backend_guard [=[static bool modelship_compute_device_allowed(ggml_backend_t backend) {
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0 &&
           strcmp(ggml_backend_dev_name(dev), "CUDA0") == 0;
}

static enum ggml_status modelship_reject_compute_device() {
    GGML_LOG_ERROR("Modelship: non-CUDA0 graph dispatch refused\n");
    return GGML_STATUS_FAILED;
}

]=])
set(reject_device [=[    if (!modelship_compute_device_allowed(backend)) {
        return modelship_reject_compute_device();
    }
]=])
set(async_anchor [=[enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
]=])
set(split_anchor [=[static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;
]=])
set(split_guard [=[
    for (int i = 0; i < sched->n_splits; i++) {
        if (!modelship_compute_device_allowed(sched->backends[splits[i].backend_id])) {
            return modelship_reject_compute_device();
        }
    }
]=])

file(SHA256 "${backend_file}" backend_digest)
if(NOT backend_digest STREQUAL "b32719de83e933a99ee1de521cb040ec45723309c8db73be8c3385d4d5ce2101")
    if(NOT backend_digest STREQUAL "a9b8af5c31fc7873e96beb65804169d513fb3b582838df01c9c17e05f0b008fd")
        message(FATAL_ERROR "Unexpected ggml backend source for CUDA placement guard")
    endif()
    file(READ "${backend_file}" source)
    string(REPLACE "${backend_anchor}" "${backend_guard}${backend_anchor}${reject_device}" result "${source}")
    string(REPLACE "${async_anchor}" "${async_anchor}${reject_device}" result "${result}")
    string(REPLACE "${split_anchor}" "${split_anchor}${split_guard}" result "${result}")
    string(SHA256 backend_digest "${result}")
    if(NOT backend_digest STREQUAL "b32719de83e933a99ee1de521cb040ec45723309c8db73be8c3385d4d5ce2101")
        message(FATAL_ERROR "CUDA dispatch guard did not match pinned ggml source")
    endif()
    file(WRITE "${backend_file}" "${result}")
endif()

# Extension kernels can invoke the CPU primitive below the scheduler's public
# entry points. Deny that path as well, without touching host preprocessing.
set(cpu_anchor [=[enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
]=])
set(cpu_guard [=[    if (cgraph->n_nodes > 0) {
        GGML_LOG_ERROR("Modelship: CPU graph execution refused\n");
        return GGML_STATUS_FAILED;
    }
]=])
modelship_patch("${MODELSHIP_GGML_SOURCE}/src/ggml-cpu/ggml-cpu.c"
    "38525261982d4aa5923e2062baefdc3c6cc18ff617f2213510a3fe285d5456c4"
    "e5a3e18bdb1506c77e085f02d9b00c3fc52dc9b9be2719644b5d8bd6a62c75a5" "${cpu_anchor}" "${cpu_anchor}${cpu_guard}")
