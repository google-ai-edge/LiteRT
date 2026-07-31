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
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_arm_options.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/internal/litert_opaque_options_wrapper.h"
#include "litert/cc/internal/litert_options_wrapper.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_arm_options.h"
#include "litert/vendors/c/litert_compiler_plugin.h"

namespace {

constexpr char kPluginManufacturer[] = "Arm";
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

    auto opaque_options = litert::OpaqueOptions::WrapCObject(
        opaque_options_->Get(), litert::OwnHandle::kNo);
    arm_options_ =
        litert::FindOpaqueOptions<litert::arm::ArmOptions>(opaque_options);
  }

  bool IsJitRequested() const {
    if (!arm_options_) {
      return false;
    }
    litert::Expected<bool> enable_just_in_time =
        arm_options_->GetEnableJustInTime();
    return enable_just_in_time && *enable_just_in_time;
  }

 private:
  litert::Expected<litert::arm::ArmOptions> arm_options_ =
      litert::Error(kLiteRtStatusErrorNotFound, "Arm options not found");
  litert::Expected<litert::internal::OptionsWrapper> litert_options_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::internal::OpaqueOptionsWrapper> opaque_options_ =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
};

struct LiteRtCompiledResultT {
  std::vector<std::vector<char>> byte_codes;
  std::vector<std::string> call_infos;
};

namespace {

LiteRtStatus EnsureJitMode(LiteRtCompilerPlugin compiler_plugin) {
  if (compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!compiler_plugin->IsJitRequested()) {
    LITERT_LOG(LITERT_ERROR,
               "Arm compiler only supports the JIT flow. Set the "
               "Arm enable_just_in_time option to use this plugin.");
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
  *sdk_version = "";
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
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  (void)soc_model;
  if (partitions == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(EnsureJitMode(compiler_plugin));

  auto result = std::make_unique<LiteRtCompiledResultT>();
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
  *handle = nullptr;
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
  *byte_code_idx = 0;
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
