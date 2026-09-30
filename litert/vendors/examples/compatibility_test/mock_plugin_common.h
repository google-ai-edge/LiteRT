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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_COMPATIBILITY_TEST_MOCK_PLUGIN_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_COMPATIBILITY_TEST_MOCK_PLUGIN_COMMON_H_

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/examples/example_transformations.h"

// Define the handle underlying struct.
struct LiteRtCompilerPluginT {
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx)
      : compiler_context(ctx) {}

  std::vector<LiteRtTransformation> transformations;
  const LiteRtCompilerContext* compiler_context;
};

namespace litert::compatibility {

constexpr char kMockSocModel[] = "MockSocModel";

// Compiled result state holding real mock bytecode.
struct MockCompiledResultState {
  std::string global_byte_code = "MOCK_BYTECODE_V1";
  std::vector<std::string> call_infos = {"mock_call_0"};
};

inline LiteRtStatus MockGetCompilerPluginVersion(
    LiteRtApiVersion* api_version) {
  if (!api_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (!compiler_plugin) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context);
  return kLiteRtStatusOk;
}

inline void MockDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

inline LiteRtStatus MockGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = 1;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name || soc_model_idx != 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kMockSocModel;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockCompilerPluginPartition(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model_name,
    LiteRtSubgraph subgraph, LiteRtOpList selected_ops) {
  if (!compiler_plugin || !subgraph || !selected_ops) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* ctx = compiler_plugin->compiler_context;
  litert::compiler::Subgraph sg(ctx, subgraph);
  for (const auto& op : sg.Ops()) {
    // Select all Mul ops into partition 0.
    if (op.Code() == kLiteRtOpCodeTflMul) {
      if (ctx && ctx->push_op) {
        LITERT_RETURN_IF_ERROR(ctx->push_op(selected_ops, op.Get(), 0));
      }
    }
  }
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model_name,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (!compiler_plugin || !partitions || !compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = std::make_unique<MockCompiledResultState>();
  *compiled_result = reinterpret_cast<LiteRtCompiledResult>(result.release());
  return kLiteRtStatusOk;
}

inline void MockDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete reinterpret_cast<MockCompiledResultState*>(compiled_result);
}

inline LiteRtStatus MockGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  auto* res = reinterpret_cast<MockCompiledResultState*>(compiled_result);
  if (!res || byte_code_idx != 0 || !byte_code || !byte_code_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *byte_code = res->global_byte_code.data();
  *byte_code_size = res->global_byte_code.size();
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = 1;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  auto* res = reinterpret_cast<MockCompiledResultState*>(compiled_result);
  if (!res || call_idx >= res->call_infos.size() || !call_info ||
      !call_info_size || !byte_code_idx) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *call_info = res->call_infos[call_idx].data();
  *call_info_size = res->call_infos[call_idx].size();
  *byte_code_idx = 0;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  auto* res = reinterpret_cast<MockCompiledResultState*>(compiled_result);
  if (!res || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = res->call_infos.size();
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (!compiler_plugin || !transformations || !num_patterns) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  compiler_plugin->transformations.push_back(
      {&SqrtMeanSquareTransformation, "MockTransformation", 100});
  *num_patterns = compiler_plugin->transformations.size();
  *transformations = compiler_plugin->transformations.data();
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (!compiler_plugin || !sdk_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version = "1.0.0-mock";
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  return kLiteRtStatusOk;
}

typedef LiteRtStatus (*LiteRtCompilerPluginDummyT)(
    LiteRtCompilerPlugin compiler_plugin);

inline LiteRtStatus MockCompilerPluginDummy(
    LiteRtCompilerPlugin compiler_plugin) {
  return kLiteRtStatusOk;
}

// Extra function pointer signature for future V1.2 ABI extensions.
typedef LiteRtStatus (*MockPluginFutureExtensionT)();

inline LiteRtStatus MockPluginFutureExtension() {
  return kLiteRtStatusOk;
}

// V1.1 interface table (matching current LiteRT).
typedef struct {
  LiteRtAbiHeader abi_header;

  LiteRtGetCompilerPluginVersionT get_compiler_plugin_version;
  LiteRtGetCompilerPluginSocManufacturerT get_compiler_plugin_soc_manufacturer;
  LiteRtCreateCompilerPluginT create_compiler_plugin;
  LiteRtDestroyCompilerPluginT destroy_compiler_plugin;

  LiteRtGetCompilerPluginSupportedHardwareT
      get_compiler_plugin_supported_hardware;
  LiteRtGetNumCompilerPluginSupportedSocModelsT
      get_num_compiler_plugin_supported_models;
  LiteRtGetCompilerPluginSupportedSocModelT
      get_compiler_plugin_supported_soc_model;

  LiteRtCompilerPluginPartitionT compiler_plugin_partition;
  LiteRtCompilerPluginCompileT compiler_plugin_compile;

  LiteRtDestroyCompiledResultT destroy_compiled_result;
  LiteRtGetCompiledResultByteCodeT get_compiled_result_byte_code;
  LiteRtCompiledResultNumByteCodeModulesT get_compiled_result_num_byte_code;
  LiteRtGetCompiledResultCallInfoT get_compiled_result_call_info;
  LiteRtGetNumCompiledResultCallsT get_num_compiled_result_calls;
  LiteRtCompilerPluginRegisterAllTransformationsT register_all_transformations;

  LiteRtGetCompilerPluginSDKVersionT get_compiler_plugin_sdk_version;
  LiteRtGetCompiledResultHandleT get_compiled_result_handle;
  LiteRtCompilerPluginCheckCompilerCompatibilityT check_compiler_compatibility;

  LiteRtCompilerPluginDummyT dummy;
} LiteRtCompilerPluginInterface_V1_1;

// Future V1.2 interface table (extended beyond V1.1).
typedef struct {
  LiteRtCompilerPluginInterface_V1_1 v1_1;
  MockPluginFutureExtensionT future_extension;
} LiteRtCompilerPluginInterface_V1_2;

}  // namespace litert::compatibility

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_COMPATIBILITY_TEST_MOCK_PLUGIN_COMMON_H_
