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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/examples/example_common.h"
#include "litert/vendors/examples/example_transformations.h"

// A simple compiler plugin example that implements everything directly.
// This plugin matches on mul ops, and emits "byte code" that is simply
// a string representative of the ops consumed.

constexpr char kExamplePluginVersion[] = "1";

// Plugins can hold state.
struct LiteRtCompilerPluginT {
  std::vector<LiteRtTransformation> transformations;
};

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  // Do not check when soc_model_name is not specified.
  if (!soc_model_name) {
    return kLiteRtStatusOk;
  }
  // Example plugin does not depend on any compiler library, so we can
  // return an error to test the error handling.
  if (absl::string_view(soc_model_name) ==
      litert::example::kIncompatiblePluginSocModel) {
    LITERT_LOG(LITERT_ERROR, "Incompatible compiler version.");
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  return kLiteRtStatusOk;
}

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
  return litert::example::kPluginManufacturer;
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
  if (!compiler_plugin || !soc_model_name) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (soc_model_idx != 0) {
    return kLiteRtStatusErrorUnsupported;
  }
  *soc_model_name = litert::example::kPluginSocModel;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// Simple compiled result def holds byte code and per op data.
struct LiteRtCompiledResultT {
  std::string global_byte_code;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *byte_code = compiled_result->global_byte_code.data();
  *byte_code_size = compiled_result->global_byte_code.size();
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
  if (!compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->per_op_data.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  *num_byte_code = compiled_result->per_op_data.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  *compiler_plugin = new LiteRtCompilerPluginT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph main_subgraph(subgraph);
  for (const auto& op : main_subgraph.Ops()) {
    bool only_f32 = true;
    for (const auto& input : op.Inputs()) {
      only_f32 &= input.ElementType() == ::litert::ElementType::Float32;
    }
    for (const auto& output : op.Outputs()) {
      only_f32 &= output.ElementType() == ::litert::ElementType::Float32;
    }
    if (!only_f32) {
      continue;
    }

    if (op.Code() == kLiteRtOpCodeTflMul) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    } else if (op.Code() == kLiteRtOpCodeTflSub) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 1));
    } else if (op.Code() == kLiteRtOpCodeShloComposite) {
      const auto opts =
          litert::GetOptionsAs<litert::CompositeOptions>(op.Get());
      if (!opts) {
        return opts.Error().Status();
      }
      if (opts->name == "odml.rms_norm") {
        LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
      }
    }
  }
  return kLiteRtStatusOk;
}

namespace litert::example {
namespace {

Expected<OpCode> ConvertOpCode(LiteRtOpCode code) {
  switch (code) {
    case kLiteRtOpCodeTflMul:
      return OpCode::kMul;
    case kLiteRtOpCodeTflSub:
      return OpCode::kSub;
    case kLiteRtOpCodeShloComposite:
      return OpCode::kRmsNorm;
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported op code: %d", code);
      return Error(kLiteRtStatusErrorUnsupported);
  }
}

LiteRtStatus CompileSinglePartition(LiteRtParamIndex partition_index,
                                    LiteRtSubgraph subgraph,
                                    ExampleGlobalGraph& global_graph) {
  const litert::Subgraph sg(subgraph);
  ExampleGraph example_graph;
  std::unordered_map<LiteRtTensor, int> tensor_map;  // NOLINT

  auto handle_constant = [&](const litert::Tensor& input,
                             int example_ind) -> LiteRtStatus {
    LiteRtWeights weights;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorWeights(input.Get(), &weights));
    int32_t buffer_id;
    LITERT_RETURN_IF_ERROR(LiteRtGetWeightsBufferId(weights, &buffer_id));
    const void* addr;
    size_t size;
    LITERT_RETURN_IF_ERROR(LiteRtGetWeightsBytes(weights, &addr, &size));

    if (buffer_id > 0) {
      if (global_graph.buffers_.find(buffer_id) ==
          global_graph.buffers_.end()) {
        ExampleTensor tensor(example_graph.Tensors()[example_ind].dims);
        tensor.data.resize(size / sizeof(float));
        std::memcpy(tensor.data.data(), addr, size);
        global_graph.buffers_[buffer_id] = std::move(tensor);
      }
      example_graph.AddConstMap(example_ind, buffer_id);
    }
    return kLiteRtStatusOk;
  };

  Inds example_graph_inputs;
  for (const auto& input : sg.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
    const auto litert_dims = input_type.Layout().Dimensions();
    const auto example_ind = example_graph.EmplaceTensor(
        Dims(litert_dims.cbegin(), litert_dims.cend()));
    tensor_map.emplace(input.Get(), example_ind);
    example_graph_inputs.push_back(example_ind);

    if (input.IsConstant()) {
      LITERT_RETURN_IF_ERROR(handle_constant(input, example_ind));
    }
  }

  for (const auto& op : sg.Ops()) {
    Inds example_inputs;
    for (const auto& input : op.Inputs()) {
      if (tensor_map.find(input.Get()) == tensor_map.end()) {
        if (!input.IsConstant()) {
          LITERT_LOG(LITERT_ERROR, "Unknown input tensor");
          return kLiteRtStatusErrorNotFound;
        }
        LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
        const auto litert_dims = input_type.Layout().Dimensions();
        const auto example_ind = example_graph.EmplaceTensor(
            Dims(litert_dims.cbegin(), litert_dims.cend()));
        tensor_map.emplace(input.Get(), example_ind);
        LITERT_RETURN_IF_ERROR(handle_constant(input, example_ind));
        if (input.IsConstant()) {
          example_graph_inputs.push_back(example_ind);
        }
      }
      example_inputs.push_back(tensor_map.at(input.Get()));
    }

    Inds example_outputs;
    for (const auto& output : op.Outputs()) {
      LITERT_ASSIGN_OR_RETURN(auto output_type, output.RankedTensorType());
      const auto litert_dims = output_type.Layout().Dimensions();
      const auto example_ind = example_graph.EmplaceTensor(
          Dims(litert_dims.cbegin(), litert_dims.cend()));
      tensor_map.emplace(output.Get(), example_ind);
      example_outputs.push_back(example_ind);
    }

    LITERT_ASSIGN_OR_RETURN(auto op_code, ConvertOpCode(op.Code()));
    example_graph.EmplaceOp(op_code, std::move(example_inputs),
                            std::move(example_outputs));
  }

  Inds example_graph_outputs;
  for (const auto& output : sg.Outputs()) {
    example_graph_outputs.push_back(tensor_map.at(output.Get()));
  }

  example_graph.SetInputs(std::move(example_graph_inputs));
  example_graph.SetOutputs(std::move(example_graph_outputs));
  example_graph.SetVersion(kExamplePluginVersion);

  global_graph.subgraphs_[absl::StrFormat("partition_%d", partition_index)] =
      std::move(example_graph);

  return kLiteRtStatusOk;
}

}  // namespace
}  // namespace litert::example

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->per_op_data.resize(num_partitions);

  ::litert::example::ExampleGlobalGraph global_graph;

  for (auto i = 0; i < num_partitions; ++i) {
    LITERT_ASSIGN_OR_RETURN(litert::Subgraph subgraph, model.Subgraph(i));
    LITERT_RETURN_IF_ERROR(::litert::example::CompileSinglePartition(
        i, subgraph.Get(), global_graph));
    result->per_op_data[i] = absl::StrFormat("partition_%d", i);
  }

  LITERT_ASSIGN_OR_RETURN(auto serialized, global_graph.Serialize());
  result->global_byte_code = std::string(serialized.StrView());

  // print the global graph
  LITERT_LOG(LITERT_INFO, "global_graph: %s", result->global_byte_code.c_str());

  *compiled_result = result.release();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  // Add SqrtMeanSquareTransformation.
  compiler_plugin->transformations.push_back(
      {&SqrtMeanSquareTransformation, "MyTransformation0", 100});
  // Add DummyTransformation.
  compiler_plugin->transformations.push_back(
      {&SqrtMeanSquareTransformation, "MyTransformation1"});
  *num_patterns = compiler_plugin->transformations.size();
  *transformations = compiler_plugin->transformations.data();

  return kLiteRtStatusOk;
}
