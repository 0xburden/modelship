// Weight-free test for the exact patched scheduler and CPU primitive.
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "cuda-residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void require(bool test, const char *message) {
    if (!test) {
        std::fprintf(stderr, "Placement test failed: %s\n", message);
        std::exit(1);
    }
}

struct Diagnostics {
    int dispatch_refused = 0;
    int cpu_refused = 0;
    int evaluated = 0;
    int mock_dispatches = 0;
};

void log_message(enum ggml_log_level, const char *message, void *data) {
    auto &d = *static_cast<Diagnostics *>(data);
    d.dispatch_refused += std::strstr(message, "Modelship: non-CUDA0 graph dispatch refused") != nullptr;
    d.cpu_refused += std::strstr(message, "Modelship: CPU graph execution refused") != nullptr;
}

bool mark_eval(ggml_tensor *, bool, void *data) {
    static_cast<Diagnostics *>(data)->evaluated++;
    return false;
}

void check_residency(ggml_backend_t cpu, const ggml_tensor *source) {
    using modelship::resident_tensor;
    ggml_backend_reg reg{};
    reg.iface.get_name = [](ggml_backend_reg_t) { return "CUDA"; };
    auto device = *ggml_backend_get_device(cpu);
    device.reg = &reg;
    device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    device.iface.get_name = [](ggml_backend_dev_t) { return "CUDA0"; };

    auto type = *ggml_backend_buffer_get_type(source->buffer);
    type.device = &device;
    device.context = &type;
    device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
        return static_cast<ggml_backend_buffer_type_t>(dev->context);
    };
    require(!resident_tensor(&device, source), "CUDA name does not make host weights resident");
    auto buffer = *source->buffer;
    buffer.buft = &type;
    auto tensor = *source;
    tensor.buffer = &buffer;
    require(!resident_tensor(&device, &tensor), "device-owned host buffer is still host storage");
    type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
    require(resident_tensor(&device, &tensor), "synthetic ordinary CUDA0 buffer is admitted");
    auto alternate = type;
    buffer.buft = &alternate;
    require(!resident_tensor(&device, &tensor), "alternate/split CUDA buffer rejected");
    buffer.buft = &type;
    device.iface.get_name = [](ggml_backend_dev_t) { return "CUDA1"; };
    require(!resident_tensor(&device, &tensor), "CUDA1 cannot hold weights");
    device.iface.get_name = [](ggml_backend_dev_t) { return "CUDA0"; };
    tensor.data = nullptr;
    require(!resident_tensor(&device, &tensor), "unallocated tensor rejected");
}

// A host-backed fake that records dispatch, only linked into this test. Real
// CUDA execution and buffer residency are not inferred from this fixture.
void check_order(ggml_tensor *first, ggml_tensor *second,
                 ggml_cgraph *graph, ggml_backend_t cpu, Diagnostics &diag) {
    ggml_backend_reg reg{};
    reg.iface.get_name = [](ggml_backend_reg_t) { return "CUDA"; };
    auto device = *ggml_backend_get_device(cpu);
    device.reg = &reg;
    device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    device.iface.get_name = [](ggml_backend_dev_t) { return "CUDA0"; };
    device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor *node) {
        return node != dev->context;
    };
    auto fake = *cpu;
    fake.device = &device;
    fake.context = &diag;
    fake.iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph *) {
        static_cast<Diagnostics *>(backend->context)->mock_dispatches++;
        return GGML_STATUS_SUCCESS;
    };
    const auto run = [&](ggml_backend_t last, ggml_status expected) {
        // ggml may promote CPU nodes onto a device that also supports their
        // operation and buffers. Make the forbidden second split unavoidable.
        device.context = last == cpu ? second : nullptr;
        ggml_backend_t backends[] = {&fake, cpu};
        auto *sched = ggml_backend_sched_new(backends, nullptr, 2, 16, false, false);
        ggml_backend_sched_set_tensor_backend(sched, first, &fake);
        ggml_backend_sched_set_tensor_backend(sched, second, last);
        require(ggml_backend_sched_alloc_graph(sched, graph), "allocate mixed graph");
        require(ggml_backend_sched_get_tensor_backend(sched, first) == &fake &&
                    ggml_backend_sched_get_tensor_backend(sched, second) == last,
                "verify actual fixture backend assignments");
        require(ggml_backend_sched_graph_compute(sched, graph) == expected, "mixed graph status");
        ggml_backend_sched_free(sched);
    };
    run(cpu, GGML_STATUS_FAILED);
    require(diag.mock_dispatches == 0, "forbidden later split prevents earlier CUDA dispatch");
    run(&fake, GGML_STATUS_SUCCESS);
    require(diag.mock_dispatches == 1, "synthetic CUDA0 graph is admitted");
    device.iface.get_name = [](ggml_backend_dev_t) { return "CUDA1"; };
    run(&fake, GGML_STATUS_FAILED);
    require(diag.mock_dispatches == 1, "CUDA1 cannot substitute for CUDA0");
}
}

int main() {
    Diagnostics diag;
    ggml_log_set(log_message, &diag);
    auto *ctx = ggml_init({64 * 1024, nullptr, true});
    auto *cpu = ggml_backend_cpu_init();
    require(ctx && cpu, "initialize CPU allocator and context");
    auto *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto *out = ggml_add(ctx, a, b);
    auto *second = ggml_add(ctx, out, b);
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    auto *mixed = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(mixed, second);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    require(buffer, "host buffers remain available");
    const float left = 2.0f, right = 3.0f, sentinel = -17.0f;
    ggml_backend_tensor_set(a, &left, 0, sizeof(left));
    ggml_backend_tensor_set(b, &right, 0, sizeof(right));
    ggml_backend_tensor_set(out, &sentinel, 0, sizeof(sentinel));

    ggml_backend_t backends[] = {cpu};
    auto *sched = ggml_backend_sched_new(backends, nullptr, 1, 16, false, false);
    require(sched, "CPU fallback still registered for allocation");
    ggml_backend_sched_set_eval_callback(sched, mark_eval, &diag);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation remains available");
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_FAILED,
            "scheduler rejects the CPU split");
    require(diag.dispatch_refused == 1 && diag.cpu_refused == 0 && diag.evaluated == 0,
            "preflight rejects before any split callback or kernel");
    ggml_backend_sched_free(sched);

    require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_FAILED,
            "direct CPU graph dispatch fails");
    auto plan = ggml_backend_graph_plan_create(cpu, graph);
    require(plan, "create CPU graph plan");
    require(ggml_backend_graph_plan_compute(cpu, plan) == GGML_STATUS_FAILED,
            "direct planned CPU dispatch fails");
    ggml_backend_graph_plan_free(cpu, plan);
    require(diag.dispatch_refused == 3 && diag.cpu_refused == 0, "both public paths reject at dispatch");

    auto raw_plan = ggml_graph_plan(graph, 1, nullptr);
    require(raw_plan.work_size == 0, "scalar add has no scratch buffer");
    require(ggml_graph_compute(graph, &raw_plan) == GGML_STATUS_FAILED,
            "raw CPU kernel entry point fails");
    require(diag.cpu_refused == 1, "CPU primitive itself rejects");
    float value = 0;
    ggml_backend_tensor_get(out, &value, 0, sizeof(value));
    require(value == sentinel, "no forbidden path wrote its output");
    check_residency(cpu, a);
    check_order(out, second, mixed, cpu, diag);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    ggml_log_set(nullptr, nullptr);
    std::puts("MODELSHIP_SD_CPU_GRAPH_REFUSED");
}
