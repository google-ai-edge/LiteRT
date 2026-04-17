// Copyright 2024 Google LLC.
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
// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_samsung_options.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/samsung/ai_litecore_manager.h"
#include "litert/vendors/samsung/compiler/compile_model.h"
#include "litert/vendors/samsung/compiler/create_model.h"
#include "litert/vendors/samsung/soc_model.h"

class LiteRtCompilerPluginT {
 public:
  using SamsungOptions = ::litert::samsung::SamsungOptions;

  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env, LiteRtOptions options) {
    if (options == nullptr) {
      return;
    }
    auto cc_options = litert::Options(options, litert::OwnHandle::kNo);
    auto opaques_status = cc_options.GetOpaqueOptions();
    if (!opaques_status) {
      return;
    }

    auto target_opq = litert::FindOpaqueOptions(
        *opaques_status, LrtSamsungOptionsGetIdentifier());
    if (!target_opq) {
      return;
    }
    auto payload_status = target_opq->GetData<const char>();
    if (!payload_status) {
      return;
    }
    LrtSamsungOptions samsung_options;
    auto status = LrtCreateSamsungOptionsFromToml(payload_status.Value(),
                                                  &samsung_options);
    if (status == kLiteRtStatusOk) {
      samsung_opts_ = SamsungOptions(samsung_options);
    } else {
      LITERT_LOG(LITERT_ERROR, "Failed to parse samsung options: %d", status);
    }
  }

  ::litert::Expected<SamsungOptions>& GetSamsungOptions() {
    return samsung_opts_;
  }

  ::litert::Expected<litert::OpaqueOptions>& GetOpaqueOptions() { return opq_; }

 private:
  litert::Expected<litert::OpaqueOptions> opq_ = litert::Error(
      litert::Status::kErrorInvalidArgument, "Null opaque options");
  litert::Expected<SamsungOptions> samsung_opts_ = litert::Error(
      litert::Status::kErrorInvalidArgument, "Null google tensor options");
};

constexpr char kPluginManufacturer[] = "Samsung";

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (!api_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = litert::samsung::kNumOfSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtParamIndex num_supported_soc_models = 0;
  // TODO: make soc check better
  LiteRtGetNumCompilerPluginSupportedSocModels(compiler_plugin,
                                               &num_supported_soc_models);
  if (soc_model_idx < 0 || soc_model_idx >= num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = litert::samsung::kSocModels[soc_model_idx].soc_name;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// Simple compiled result def holds byte code and per op data.
struct LiteRtCompiledResultT {
  std::vector<std::vector<char>> byte_code;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (byte_code_idx >= compiled_result->byte_code.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *byte_code = compiled_result->byte_code[byte_code_idx].data();
  *byte_code_size = compiled_result->byte_code[byte_code_idx].size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (call_idx >= compiled_result->per_op_data.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();
  *byte_code_idx = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->per_op_data.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
  LITERT_LOG(LITERT_INFO, "Destroy compiled result");
}

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  LiteRtPropagateMinLoggerSeverity(env);

  *compiler_plugin = new LiteRtCompilerPluginT(env, options);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  if (!compiler_plugin || !soc_model) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  ::litert::Subgraph graph(subgraph);

  for (const auto& op : graph.Ops()) {
    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->byte_code.resize(num_partitions);
  result->per_op_data.resize(num_partitions);
  LITERT_ASSIGN_OR_RETURN(
      auto ai_lite_core,
      litert::samsung::AiLiteCoreManager::Create(std::nullopt));
  for (auto i = 0; i < num_partitions; ++i) {
    // Get subgraph
    LITERT_ASSIGN_OR_RETURN(litert::Subgraph subgraph, model.Subgraph(i));
    // Create graph used in samsung backend
    LITERT_ASSIGN_OR_RETURN(
        auto graph_buffer,
        ::litert::samsung::CreateModel(ai_lite_core.get(), subgraph));

    // Compile graph and return binary
    LITERT_ASSIGN_OR_RETURN(auto soc_model_id,
                            litert::samsung::GetSocModelID(soc_model));
    LITERT_ASSIGN_OR_RETURN(
        auto compiled_binary,
        litert::samsung::Compile(ai_lite_core.get(), graph_buffer,
                                 soc_model_id));

    result->byte_code[i] = std::move(compiled_binary);
    LITERT_LOG(LITERT_INFO, "Compile output: %ld bytes",
               result->byte_code[i].size());
    result->per_op_data[i] = "UnknownName";
  }

  *compiled_result = result.release();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  return kLiteRtStatusOk;
}
