# Remove implementations unrelated to the private Qwen model; a missing
# model path at runtime alone does not remove unused code from the executables.
function(modelship_qwen_cut start end replacement)
    string(FIND "${text}" "${start}" begin)
    string(FIND "${text}" "${end}" stop)
    if(begin LESS 0 OR stop LESS_EQUAL begin)
        message(FATAL_ERROR "Cannot locate pinned Qwen-only exclusion: ${path}")
    endif()
    string(SUBSTRING "${text}" 0 ${begin} prefix)
    string(SUBSTRING "${text}" ${stop} -1 suffix)
    set(text "${prefix}${replacement}${suffix}" PARENT_SCOPE)
endfunction()

set(path "${SD_SOURCE}/src/conditioning/conditioner.hpp")
file(SHA256 "${path}" digest)
if(NOT digest STREQUAL "022c678a899d218877c162e397af97973232c1612b9cb21734829f9886edc274")
    if(NOT digest STREQUAL "353a23c43ba31fbde782ce13b074ca94285e14750144cd154908de878c34a31c")
        message(FATAL_ERROR "Unexpected source for the Qwen-only build: conditioning/conditioner.hpp")
    endif()
    file(READ "${path}" text)
    modelship_qwen_cut([=[static inline sd::Tensor<float> apply_token_weights(]=] [=[struct ConditionerParams {]=] [=[// With literal prompts there are no non-unit token weights.
static inline sd::Tensor<float> apply_token_weights(sd::Tensor<float> hidden_states,
                                                     const std::vector<float>&) {
    return hidden_states;
}

]=])
    modelship_qwen_cut([=[// ldm.modules.encoders.modules.FrozenCLIPEmbedder
]=] [=[struct FrozenCLIPVisionEmbedder : public GGMLRunner {]=] [=[]=])
    string(SHA256 digest "${text}")
    if(NOT digest STREQUAL "022c678a899d218877c162e397af97973232c1612b9cb21734829f9886edc274")
        message(FATAL_ERROR "Qwen-only patch did not match pinned source: conditioning/conditioner.hpp")
    endif()
    file(WRITE "${path}" "${text}")
endif()

set(path "${SD_SOURCE}/src/model/integrations/sd.cpp")
file(SHA256 "${path}" digest)
if(NOT digest STREQUAL "bed94a33e39fe3bab76655cc40e8785ca5cd4a87b4852d90b7c48a1c5af9a2b3")
    if(NOT digest STREQUAL "f856f0c8ba15ddfd7b37bc9256c7eed93d65ebcfe64eb17bfa771bdac79f441d")
        message(FATAL_ERROR "Unexpected source for the Qwen-only build: model/integrations/sd.cpp")
    endif()
    file(READ "${path}" text)
    modelship_qwen_cut([=[static bool build_sd(]=] [=[static prediction_t detect_sdxl_prediction(]=] [=[static bool build_sd(const sd::model_builders::Context&, const TokenizerConfig&, sd::model_builders::CoreRunners&) {
    return false;
}

]=])
    string(SHA256 digest "${text}")
    if(NOT digest STREQUAL "bed94a33e39fe3bab76655cc40e8785ca5cd4a87b4852d90b7c48a1c5af9a2b3")
        message(FATAL_ERROR "Qwen-only patch did not match pinned source: model/integrations/sd.cpp")
    endif()
    file(WRITE "${path}" "${text}")
endif()

set(path "${SD_SOURCE}/src/pipeline/model_builders.cpp")
file(SHA256 "${path}" digest)
if(NOT digest STREQUAL "73c7f2230d87b928b8186e9a6940bbed3547668182000e70767c007fc64e2f4f")
    if(NOT digest STREQUAL "46398a0247ae82d45193d325c1d82e4f7de298b49956e2e422080e9cc6fe259d")
        message(FATAL_ERROR "Unexpected source for the Qwen-only build: pipeline/model_builders.cpp")
    endif()
    file(READ "${path}" text)
    string(REPLACE [=[#include "model/diffusion/control.hpp"
]=] [=[]=] text "${text}")
    modelship_qwen_cut([=[    bool build_control_net_runner(]=] [=[    bool build_extension_runners(]=] [=[]=])
    string(REPLACE [=[{create_photomaker_extension(), create_pulid_extension()}]=] [=[{create_pulid_extension()}]=] text "${text}")
    string(SHA256 digest "${text}")
    if(NOT digest STREQUAL "73c7f2230d87b928b8186e9a6940bbed3547668182000e70767c007fc64e2f4f")
        message(FATAL_ERROR "Qwen-only patch did not match pinned source: pipeline/model_builders.cpp")
    endif()
    file(WRITE "${path}" "${text}")
endif()

set(path "${SD_SOURCE}/src/extensions/photomaker_extension.cpp")
file(SHA256 "${path}" digest)
if(NOT digest STREQUAL "9a55b6df145ff5fa67867f54dc109a17ef819cc274a227dd1d8f5ed8cd6db744")
    if(NOT digest STREQUAL "d1585f543d1f0a15242cf50b4af1129df0119288652b1c25b23f685bf95cf152")
        message(FATAL_ERROR "Unexpected source for the Qwen-only build: extensions/photomaker_extension.cpp")
    endif()
    file(READ "${path}" text)
    set(text [=[// PhotoMaker is not part of the private Qwen build.
]=])
    string(SHA256 digest "${text}")
    if(NOT digest STREQUAL "9a55b6df145ff5fa67867f54dc109a17ef819cc274a227dd1d8f5ed8cd6db744")
        message(FATAL_ERROR "Qwen-only patch did not match pinned source: extensions/photomaker_extension.cpp")
    endif()
    file(WRITE "${path}" "${text}")
endif()

set(path "${SD_SOURCE}/src/pipeline/diffusion_engine.cpp")
file(SHA256 "${path}" digest)
if(NOT digest STREQUAL "e863aa1b5a3a20a635555836ccb4457e4e870125fb8f80b8bf9cec5e155055e8")
    if(NOT digest STREQUAL "9b142f3e850a339cf73885d526667732c3b5fb80f99eb4c2f6cad29dc0e4faa7")
        message(FATAL_ERROR "Unexpected source for the Qwen-only build: pipeline/diffusion_engine.cpp")
    endif()
    file(READ "${path}" text)
    string(REPLACE [=[#include "model/diffusion/control.hpp"
]=] [=[]=] text "${text}")
    string(REPLACE [=[    if (control_net)
        control_net->runner_end();
]=] [=[]=] text "${text}")
    string(REPLACE [=[                control_net.reset();
]=] [=[]=] text "${text}")
    modelship_qwen_cut([=[bool StableDiffusionGGML::unload_control_net()]=] [=[bool StableDiffusionGGML::init_backend()]=] [=[bool StableDiffusionGGML::unload_control_net() { return false; }
bool StableDiffusionGGML::load_control_net_from_file(const std::string&) { return false; }

]=])
    modelship_qwen_cut([=[    if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0)]=] [=[    if (strlen(SAFE_STR(sd_ctx_params->ip_adapter_path)) > 0)]=] [=[    if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
        LOG_ERROR("ControlNet is not part of the private Qwen build");
        return false;
    }

]=])
    string(REPLACE [=[    if (version == VERSION_COUNT) {
        LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));]=] [=[    if (version != VERSION_QWEN_IMAGE_2_1) {
        LOG_ERROR("The private runtime supports only Qwen Image 2.1");]=] text "${text}")
    modelship_qwen_cut([=[bool StableDiffusionGGML::build_control_net_runner()]=] [=[bool StableDiffusionGGML::build_extension_runners()]=] [=[bool StableDiffusionGGML::build_control_net_runner() { return config_->control_net_file == 0; }

]=])
    string(REPLACE [=[    if (control_net) {
        control_net->set_flash_attention_enabled(flash_attention_for(SDBackendModule::CONTROL_NET));
    }
]=] [=[]=] text "${text}")
    string(REPLACE [=[    if (control_net) {
        set_conv_direct(SDBackendModule::CONTROL_NET, {control_net.get()});
    }
]=] [=[]=] text "${text}")
    modelship_qwen_cut([=[bool StableDiffusionGGML::set_control_type(]=] [=[sd::Tensor<float> StableDiffusionGGML::sample(]=] [=[bool StableDiffusionGGML::set_control_type(sd_control_type_t) { return false; }

void StableDiffusionGGML::compute_sample_controls(const sd::Tensor<float>&,
                                                  const sd::Tensor<float>&,
                                                  const sd::Tensor<float>&,
                                                  const SDCondition&,
                                                  std::vector<sd::Tensor<float>>* controls) {
    GGML_ASSERT(controls != nullptr);
    controls->clear();
}

]=])
    string(REPLACE [=[    RunnerEndOnExit sample_control_runner_end{!control_image.empty() && control_net != nullptr ? control_net.get() : nullptr};
]=] [=[]=] text "${text}")
    string(REPLACE [=[        if (control_net) {
            control_net->free_control_ctx();
        }
]=] [=[]=] text "${text}")
    string(REPLACE [=[    if (control_net) {
        control_net->free_control_ctx();
    }
]=] [=[]=] text "${text}")
    string(REPLACE [=[    if (this->control_net) {
        this->control_net->set_circular_axes(circular_x, circular_y);
    }
]=] [=[]=] text "${text}")
    string(SHA256 digest "${text}")
    if(NOT digest STREQUAL "e863aa1b5a3a20a635555836ccb4457e4e870125fb8f80b8bf9cec5e155055e8")
        message(FATAL_ERROR "Qwen-only patch did not match pinned source: pipeline/diffusion_engine.cpp")
    endif()
    file(WRITE "${path}" "${text}")
endif()
