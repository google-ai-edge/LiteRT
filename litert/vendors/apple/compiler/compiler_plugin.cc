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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_tfl_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/apple/bytecode.h"
#include "litert/vendors/c/litert_compiler_plugin.h"

namespace {

constexpr char kPluginManufacturer[] = "Apple";
constexpr char kPluginSocModel[] = "apple_mlx";

bool IsMlxOpSupported(const litert::compiler::Op& op) {
  if (op.Code() != kLiteRtOpCodeTflFullyConnected) {
    return false;
  }
  auto inputs = op.Inputs();
  auto outputs = op.Outputs();
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
    return false;
  }

  auto input_type = inputs[0].ElementType();
  auto weights_type = inputs[1].ElementType();
  auto output_type = outputs[0].ElementType();

  if (input_type != litert::ElementType::Float32 &&
      input_type != litert::ElementType::Float16) {
    return false;
  }
  if (weights_type != litert::ElementType::Float32 &&
      weights_type != litert::ElementType::Float16) {
    return false;
  }
  if (output_type != litert::ElementType::Float32 &&
      output_type != litert::ElementType::Float16) {
    return false;
  }

  if (!inputs[1].HasWeights()) {
    return false;
  }

  if (inputs.size() == 3) {
    auto bias_type = inputs[2].ElementType();
    if (bias_type != litert::ElementType::None) {
      if (bias_type != litert::ElementType::Float32 &&
          bias_type != litert::ElementType::Float16) {
        return false;
      }
      if (!inputs[2].HasWeights()) {
        return false;
      }
    }
  }

  uint32_t activation = litert::kActivationFunctionTypeNone;
  if (LiteRtGetFullyConnectedFusedActivationOption(op.Get(), &activation) !=
      kLiteRtStatusOk) {
    return false;
  }

  if (activation != litert::kActivationFunctionTypeNone &&
      activation != litert::kActivationFunctionTypeRelu &&
      activation != litert::kActivationFunctionTypeRelu6) {
    return false;
  }

  return true;
}

LiteRtElementType ConvertElementType(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
      return kLiteRtElementTypeFloat32;
    case litert::ElementType::Float16:
      return kLiteRtElementTypeFloat16;
    default:
      return kLiteRtElementTypeNone;
  }
}

}  // namespace

struct LiteRtCompiledResultT {
  std::vector<std::vector<uint8_t>> bytecodes;
  std::vector<std::string> call_infos;
};

struct LiteRtCompilerPluginT {
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx) : ctx(ctx) {}
  const LiteRtCompilerContext* ctx = nullptr;
};

extern "C" {

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

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // MLX runs on NPU (Neural Engine) or GPU or CPU. We can claim NPU/GPU
  // support.
  *supported_hardware = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorNpu);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name || soc_model_idx != 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModel;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (!compiler_plugin || !sdk_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version = "MLX C++ API";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->bytecodes.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size ||
      byte_code_idx >= compiled_result->bytecodes.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& bytecode = compiled_result->bytecodes[byte_code_idx];
  *byte_code = bytecode.data();
  *byte_code_size = bytecode.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size || !byte_code_idx) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx >= compiled_result->call_infos.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const auto& info = compiled_result->call_infos[call_idx];
  *call_info = info.data();
  *call_info_size = info.size();
  *byte_code_idx = call_idx;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->call_infos.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  if (!compiler_plugin || !subgraph || !selected_ops) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::compiler::Subgraph graph(compiler_plugin->ctx, subgraph);

  int partition_index = 0;
  for (const auto& op : graph.Ops()) {
    if (IsMlxOpSupported(op)) {
      // Put each supported op in its own partition for simplicity for now.
      // In practice, we would group contiguous supported ops.
      // But for a single linear layer, this is fine.
      LITERT_RETURN_IF_ERROR(compiler_plugin->ctx->push_op(
          selected_ops, op.Get(), partition_index++));
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (!compiler_plugin || !partitions || !compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::compiler::Model model(compiler_plugin->ctx, partitions);
  auto result = std::make_unique<LiteRtCompiledResultT>();
  const auto num_partitions = model.NumSubgraphs();
  result->bytecodes.reserve(num_partitions);
  result->call_infos.reserve(num_partitions);

  for (LiteRtParamIndex i = 0; i < num_partitions; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(i));

    // We expect each partition to contain exactly one FullyConnected op for
    // this prototype.
    auto ops = subgraph.Ops();
    if (ops.size() != 1) {
      return kLiteRtStatusErrorUnsupported;
    }
    const auto& op = ops[0];
    if (op.Code() != kLiteRtOpCodeTflFullyConnected) {
      return kLiteRtStatusErrorUnsupported;
    }

    litert::apple::MlxBytecode bytecode;

    auto inputs = op.Inputs();

    // 1. Weights (Input 1)
    const auto& weights_tensor = inputs[1];
    LITERT_ASSIGN_OR_RETURN(auto weights_type,
                            weights_tensor.RankedTensorType());
    bytecode.weights_type = ConvertElementType(weights_type.ElementType());
    for (auto dim : weights_type.Layout().Dimensions()) {
      bytecode.weights_dims.push_back(dim);
    }
    auto weights_span = weights_tensor.Weights().Bytes();
    bytecode.weights_data.assign(weights_span.begin(), weights_span.end());

    // 2. Bias (Input 2, optional)
    if (inputs.size() == 3) {
      bytecode.has_bias = true;
      const auto& bias_tensor = inputs[2];
      LITERT_ASSIGN_OR_RETURN(auto bias_type, bias_tensor.RankedTensorType());
      bytecode.bias_type = ConvertElementType(bias_type.ElementType());
      for (auto dim : bias_type.Layout().Dimensions()) {
        bytecode.bias_dims.push_back(dim);
      }
      auto bias_span = bias_tensor.Weights().Bytes();
      bytecode.bias_data.assign(bias_span.begin(), bias_span.end());
    }

    // 3. Activation
    uint32_t activation = litert::kActivationFunctionTypeNone;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetFullyConnectedFusedActivationOption(op.Get(), &activation));
    bytecode.activation = activation;

    // Pack
    LITERT_ASSIGN_OR_RETURN(auto packed,
                            litert::apple::PackMlxBytecode(bytecode));

    const std::string function_name = "mlx_partition_" + std::to_string(i);
    result->call_infos.push_back(function_name);
    result->bytecodes.push_back(std::move(packed));
  }

  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (!transformations || !num_patterns) {
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
  return kLiteRtStatusOk;
}

}  // extern "C"
