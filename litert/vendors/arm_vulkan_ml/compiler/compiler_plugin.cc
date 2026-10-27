// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_arm_vulkan_ml_options.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_opaque_options_wrapper.h"
#include "litert/cc/internal/litert_options_wrapper.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"

namespace {

constexpr char kPluginManufacturer[] = "ArmVulkanML";
constexpr char kDefaultSocModel[] = "Generic";

}  // namespace

class LiteRtCompilerPluginT {
 public:
  LiteRtCompilerPluginT(const LiteRtCompilerContext* compiler_context,
                        LiteRtOptions options) {
    if (options == nullptr) {
      return;
    }

    litert_options_ = litert::internal::OptionsWrapper(
        litert::internal::ContextWrapper(compiler_context), options,
        litert::OwnHandle::kNo);
    if (!litert_options_) {
      return;
    }

    opaque_options_ = litert_options_->GetOpaqueOptions();
    if (!opaque_options_) {
      return;
    }

    auto target_options = opaque_options_->FindOpaqueOptions(
        LrtArmVulkanMLOptionsGetIdentifier());
    if (!target_options) {
      return;
    }

    const char* payload = static_cast<const char*>(target_options.Value());
    LrtArmVulkanMLOptions arm_vulkan_ml_options = nullptr;
    auto status =
        LrtCreateArmVulkanMLOptionsFromToml(payload, &arm_vulkan_ml_options);
    if (status == kLiteRtStatusOk) {
      arm_vulkan_ml_options_ =
          litert::arm_vulkan_ml::ArmVulkanMLOptions(arm_vulkan_ml_options);
    } else {
      LITERT_LOG(LITERT_ERROR, "Failed to parse ArmVulkanML options: %d",
                 status);
    }
  }

  bool IsJitRequested() const {
    if (!arm_vulkan_ml_options_) {
      return false;
    }
    litert::Expected<bool> enable_just_in_time =
        arm_vulkan_ml_options_->GetEnableJustInTime();
    return enable_just_in_time && *enable_just_in_time;
  }

 private:
  litert::Expected<litert::arm_vulkan_ml::ArmVulkanMLOptions>
      arm_vulkan_ml_options_ = litert::Error(kLiteRtStatusErrorNotFound,
                                             "ArmVulkanML options not found");
  litert::Expected<litert::internal::OptionsWrapper> litert_options_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::internal::OpaqueOptionsWrapper> opaque_options_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
};

struct LiteRtCompiledResultT {
  std::vector<std::vector<char>> byte_codes;
  std::vector<std::string> call_infos;
  std::vector<std::unique_ptr<litert::arm_vulkan_ml::ArmVulkanMLJitExecutable>>
      executables;
};

namespace {

LiteRtStatus EnsureJitMode(LiteRtCompilerPlugin compiler_plugin) {
  if (compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!compiler_plugin->IsJitRequested()) {
    LITERT_LOG(LITERT_ERROR,
               "Arm ML extensions for Vulkan compiler only supports the JIT "
               "flow. Set the ArmVulkanML enable_just_in_time option to use "
               "this plugin.");
    return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (compiler_plugin == nullptr || sdk_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version = "TFSC-2.0.0/TOSA.001000.1-direct-v1";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  (void)env;
  if (compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context, options);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (compiler_plugin == nullptr || supported_hardware == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr || soc_model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (soc_model_idx != 0) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *soc_model_name = kDefaultSocModel;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  (void)soc_model;
  if (subgraph == nullptr || selected_ops == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(EnsureJitMode(compiler_plugin));

  litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    std::string reason;
    if (litert::arm_vulkan_ml::AnalyzeOperation(op, reason)) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  (void)soc_model;
  if (partitions == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *compiled_result = nullptr;
  LITERT_RETURN_IF_ERROR(EnsureJitMode(compiler_plugin));

  auto result = std::make_unique<LiteRtCompiledResultT>();
  auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
  if (model.NumSubgraphs() == 0) return kLiteRtStatusErrorInvalidArgument;
  for (size_t i = 0; i < model.NumSubgraphs(); ++i) {
    auto graph = model.Subgraph(i);
    if (!graph) return graph.Error().Status();
    std::string reason;
    auto partition = litert::arm_vulkan_ml::CompilePartition(*graph, reason);
    if (!partition) {
      LITERT_LOG(LITERT_ERROR, "ArmVulkanML partition %zu: %s", i, reason.c_str());
      return kLiteRtStatusErrorUnsupported;
    }
    auto executable =
        std::make_unique<litert::arm_vulkan_ml::ArmVulkanMLJitExecutable>();
    executable->partition = std::move(partition);
    result->executables.push_back(std::move(executable));
    result->call_infos.push_back("arm_vulkan_ml_" + std::to_string(i));
    // Core dispatch metadata expects a module index even in the JIT flow.
    // The payload is the JIT handle, not a serialized VGF or SPIR-V bytecode.
    result->byte_codes.emplace_back();
  }
  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult result) { delete result; }

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (compiled_result == nullptr || byte_code == nullptr ||
      byte_code_size == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (byte_code_idx < 0 || static_cast<size_t>(byte_code_idx) >=
                               compiled_result->byte_codes.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *byte_code = compiled_result->byte_codes[byte_code_idx].data();
  *byte_code_size = compiled_result->byte_codes[byte_code_idx].size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (compiled_result == nullptr || num_byte_code == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->byte_codes.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultHandle(LiteRtCompiledResult compiled_result,
                                           LiteRtParamIndex call_idx,
                                           LiteRtJitExecutable* handle) {
  if (compiled_result == nullptr || handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx < 0 ||
      static_cast<size_t>(call_idx) >= compiled_result->call_infos.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *handle = reinterpret_cast<LiteRtJitExecutable>(
      compiled_result->executables[call_idx].get());
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (compiled_result == nullptr || call_info == nullptr ||
      call_info_size == nullptr || byte_code_idx == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx < 0 ||
      static_cast<size_t>(call_idx) >= compiled_result->call_infos.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *call_info = compiled_result->call_infos[call_idx].data();
  *call_info_size = compiled_result->call_infos[call_idx].size();
  *byte_code_idx = call_idx;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (compiled_result == nullptr || num_calls == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->call_infos.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (compiler_plugin == nullptr || transformations == nullptr ||
      num_patterns == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *transformations = nullptr;
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  (void)compiler_plugin;
  (void)env;
  (void)options;
  (void)soc_model_name;
  if (api_version.major != LITERT_API_VERSION_MAJOR) {
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  return kLiteRtStatusOk;
}

namespace {

static const LiteRtCompilerPluginInterface_V1 TheCompilerPluginInterface = {
    .abi_header =
        {
            .struct_size = sizeof(LiteRtCompilerPluginInterface_V1),
            .major_version = 1,
            .minor_version = 0,
            .reserved = 0,
        },
    .get_compiler_plugin_version = LiteRtGetCompilerPluginVersion,
    .get_compiler_plugin_soc_manufacturer =
        LiteRtGetCompilerPluginSocManufacturer,
    .create_compiler_plugin = LiteRtCreateCompilerPlugin,
    .destroy_compiler_plugin = LiteRtDestroyCompilerPlugin,
    .get_compiler_plugin_supported_hardware =
        LiteRtGetCompilerPluginSupportedHardware,
    .get_num_compiler_plugin_supported_models =
        LiteRtGetNumCompilerPluginSupportedSocModels,
    .get_compiler_plugin_supported_soc_model =
        LiteRtGetCompilerPluginSupportedSocModel,
    .compiler_plugin_partition = LiteRtCompilerPluginPartition,
    .compiler_plugin_compile = LiteRtCompilerPluginCompile,
    .destroy_compiled_result = LiteRtDestroyCompiledResult,
    .get_compiled_result_byte_code = LiteRtGetCompiledResultByteCode,
    .get_compiled_result_num_byte_code = LiteRtCompiledResultNumByteCodeModules,
    .get_compiled_result_call_info = LiteRtGetCompiledResultCallInfo,
    .get_num_compiled_result_calls = LiteRtGetNumCompiledResultCalls,
    .register_all_transformations =
        LiteRtCompilerPluginRegisterAllTransformations,
    .get_compiler_plugin_sdk_version = LiteRtGetCompilerPluginSDKVersion,
    .get_compiled_result_handle = LiteRtGetCompiledResultHandle,
    .check_compiler_compatibility =
        LiteRtCompilerPluginCheckCompilerCompatibility,
};

}  // namespace

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginQueryInterface(
    LiteRtCompilerPluginInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major >= 1) {
    switch (interface_id) {
      case kLiteRtCompilerPluginInterfaceBasic:
        *out_interface = &TheCompilerPluginInterface;
        return kLiteRtStatusOk;
      default:
        return kLiteRtStatusErrorUnsupported;
    }
  }
  return kLiteRtStatusErrorUnsupported;
}
