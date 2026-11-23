// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test-only exporter. Uses the production adapter/emitter and builtin CPU
// reference kernels. The Python test independently checks the accumulator.
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "TosaForSpirvCodegen.hpp"
#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d_adapter.h"
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"
#include "litert/vendors/arm_vulkan_ml/compiler/tosa_requirements.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register_ref.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml {
namespace {
template <typename V>
void JsonArray(std::ostream& out, const V& values) {
  out << "[";
  bool first = true;
  for (auto value : values) {
    if (!first) out << ",";
    out << +value;
    first = false;
  }
  out << "]";
}
bool Write(const std::filesystem::path& path, const void* data, size_t size) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), size);
  return out.good();
}

int RunConv2dFixture(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "Usage: conv2d_fixture MODEL OUTPUT_DIR OP_INDEX\n";
    return 1;
  }
  int op_index = 0;
  const char* end = argv[3] + std::strlen(argv[3]);
  std::from_chars_result parsed = std::from_chars(argv[3], end, op_index);
  if (parsed.ec != std::errc{} || parsed.ptr != end) return 1;
  litert::Expected<litert::ExtendedModel> model =
      litert::ExtendedModel::CreateFromFile(argv[1]);
  if (!model) {
    std::cerr << "Cannot load model\n";
    return 1;
  }
  litert::Expected<litert::Subgraph> subgraph = model->Subgraph(0);
  if (!subgraph || op_index < 0 || op_index >= subgraph->Ops().size()) return 1;
  std::string reason;
  std::optional<Conv2dPlan> plan =
      AnalyzeConv2d(subgraph->Ops()[op_index], reason);
  if (!plan) {
    std::cerr << reason << "\n";
    return 1;
  }
  const Conv2dSpec& spec = plan->spec;
  std::shared_ptr<tfsc::spirv::Module> module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, "main");
  tfsc::tosa::ResId input = graph.AddInput(
      tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t,
                         {spec.input_shape.begin(), spec.input_shape.end()}),
      0);
  GraphConstantPool constants;
  tfsc::tosa::ResId accumulator;
  tfsc::tosa::ResId output =
      EmitConv2d(graph, input, *plan, constants, &accumulator);
  if (tfsc::IsError(output)) {
    std::cerr << tfsc::GetErrorMsg(output);
    return 1;
  }
  graph.AddOutput(output, 1);
  graph.AddOutput(accumulator, 2);
  graph.FinalizeGraph();
  std::vector<uint32_t> spirv = tfsc::WriteToBinary(module);
  if (spirv.empty()) return 1;
  std::filesystem::path dir(argv[2]);
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  if (error) {
    std::cerr << error.message() << "\n";
    return 1;
  }
  if (!Write(dir / "graph.spv", spirv.data(), spirv.size() * 4)) return 1;
  if (!Write(dir / "weights.bin", spec.weights.data(), spec.weights.size()) ||
      !Write(dir / "bias.bin", spec.bias.data(),
             spec.bias.size() * sizeof(int32_t))) {
    return 1;
  }
  std::ofstream manifest(dir / "graph.json");
  manifest.precision(17);
  manifest << "{\"input_shape\":";
  JsonArray(manifest, spec.input_shape);
  manifest << ",\"output_shape\":";
  JsonArray(manifest, spec.output_shape);
  manifest << ",\"weight_shape\":";
  JsonArray(manifest, spec.weight_shape);
  manifest << ",\"padding\":";
  JsonArray(manifest, spec.padding);
  manifest << ",\"stride\":";
  JsonArray(manifest, spec.stride);
  manifest << ",\"dilation\":";
  JsonArray(manifest, spec.dilation);
  manifest << ",\"input_zero_point\":" << spec.input_zero_point
           << ",\"output_zero_point\":" << spec.output_zero_point
           << ",\"activation_min\":" << plan->activation_range.minimum
           << ",\"activation_max\":" << plan->activation_range.maximum;
  std::vector<int32_t> multipliers;
  std::vector<int32_t> shifts;
  for (Scale32 scale : plan->scales) {
    multipliers.push_back(scale.multiplier);
    shifts.push_back(scale.shift);
  }
  manifest << ",\"multipliers\":";
  JsonArray(manifest, multipliers);
  manifest << ",\"shifts\":";
  JsonArray(manifest, shifts);
  manifest << ",\"rounding_mode\":\"SINGLE_ROUND\"";
  manifest << ",\"depthwise\":" << (spec.depthwise ? "true" : "false");
  manifest << ",\"constants\":[";
  for (size_t i = 0; i < constants.values.size(); ++i) {
    const GraphConstant& constant = constants.values[i];
    if (i) manifest << ",";
    manifest << "{\"file\":\"constant_" << i << ".bin\",\"dtype\":\""
             << (constant.tensor.GetDataType() == tfsc::tosa::DataType::int8_t
                     ? "int8"
                     : "int32")
             << "\",\"shape\":";
    JsonArray(manifest, constant.tensor.GetTensorShape());
    manifest << "}";
    if (!Write(dir / ("constant_" + std::to_string(i) + ".bin"),
               constant.bytes.data(), constant.bytes.size())) {
      return 1;
    }
  }
  manifest << "],\"reference\":\"BuiltinRefOpResolver; delegates "
              "disabled\",\"single_rounding\":";
#if TFLITE_SINGLE_ROUNDING
  manifest << "true";
#else
  manifest << "false";
#endif
  manifest << "}\n";
  manifest.close();
  if (!manifest.good()) return 1;

  std::unique_ptr<VulkanInvocation> invocation;
  if (argc == 5) {
    if (std::string(argv[4]) != "--vulkan-emulation") return 1;
    auto artifact = std::make_shared<CompiledPartition>();
    artifact->spirv = spirv;
    if (!CollectTosaLevelRequirements(*module, artifact->level_requirements,
                                      reason)) {
      std::cerr << reason << "\n";
      return 1;
    }
    artifact->inputs.push_back(
        {TensorElement::kInt8,
         {spec.input_shape.begin(), spec.input_shape.end()},
         0});
    artifact->outputs.push_back(
        {TensorElement::kInt8,
         {spec.output_shape.begin(), spec.output_shape.end()},
         1});
    artifact->outputs.push_back(
        {TensorElement::kInt32,
         {spec.output_shape.begin(), spec.output_shape.end()},
         2});
    for (const GraphConstant& constant : constants.values) {
      TensorElement type =
          constant.tensor.GetDataType() == tfsc::tosa::DataType::int8_t
              ? TensorElement::kInt8
              : TensorElement::kInt32;
      artifact->constants.push_back(
          {{type, constant.tensor.GetTensorShape(),
            static_cast<uint32_t>(artifact->constants.size())},
           constant.bytes});
    }
    invocation = VulkanInvocation::Create(artifact, true, reason);
    if (!invocation) {
      std::cerr << reason << "\n";
      return 1;
    }
  }

  std::unique_ptr<tflite::FlatBufferModel> source =
      tflite::FlatBufferModel::BuildFromFile(argv[1]);
  if (!source) return 1;
  std::unique_ptr<tflite::ModelT> object(source->GetModel()->UnPack());
  tflite::SubGraphT& subgraph_t = *object->subgraphs[0];
  std::unique_ptr<tflite::OperatorT> selected =
      std::move(subgraph_t.operators[op_index]);
  subgraph_t.inputs = {selected->inputs[0]};
  subgraph_t.outputs = selected->outputs;
  subgraph_t.operators.clear();
  subgraph_t.operators.push_back(std::move(selected));
  object->signature_defs.clear();
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder,
                            tflite::Model::Pack(builder, object.get()));
  std::unique_ptr<tflite::FlatBufferModel> fixture =
      tflite::FlatBufferModel::BuildFromBuffer(
          reinterpret_cast<const char*>(builder.GetBufferPointer()),
          builder.GetSize());
  if (!fixture) return 1;
  tflite::ops::builtin::BuiltinRefOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*fixture, resolver)(&interpreter) !=
          kTfLiteOk ||
      interpreter->AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  TfLiteTensor* input_tensor = interpreter->input_tensor(0);
  // Four deterministic inputs cover saturation, zero points, negative values,
  // and layout. The emulator consumes exactly these bytes.
  for (int case_id = 0; case_id < 4; ++case_id) {
    for (size_t i = 0; i < input_tensor->bytes; ++i) {
      int value = static_cast<int>((i * 73 + i / 11 + 29) % 256) - 128;
      if (case_id == 1) value = spec.input_zero_point;
      if (case_id == 2) value = i % 2 ? 127 : -128;
      if (case_id == 3) value = -1 - (i * 17 % 128);
      input_tensor->data.int8[i] = value;
    }
    if (!Write(dir / ("input_" + std::to_string(case_id) + ".bin"),
               input_tensor->data.raw, input_tensor->bytes)) {
      return 1;
    }
    if (interpreter->Invoke() != kTfLiteOk) return 1;
    TfLiteTensor* result = interpreter->output_tensor(0);
    if (invocation) {
      std::vector<int8_t> actual(result->bytes);
      std::vector<int32_t> acc(result->bytes);
      if (!invocation->Run({input_tensor->data.raw},
                           {actual.data(), acc.data()}, reason)) {
        std::cerr << reason << "\n";
        return 1;
      }
      for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(static_cast<int>(actual[i]) -
                     static_cast<int>(result->data.int8[i])) > 1) {
          std::cerr << "Direct Vulkan output differs from CPU at " << i << "\n";
          return 1;
        }
      }
      if (!Write(dir / ("vulkan_" + std::to_string(case_id) + ".bin"),
                 actual.data(), actual.size()) ||
          !Write(dir / ("accumulator_" + std::to_string(case_id) + ".bin"),
                 acc.data(), acc.size() * 4)) {
        return 1;
      }
    }
    if (!Write(dir / ("reference_" + std::to_string(case_id) + ".bin"),
               result->data.raw, result->bytes)) {
      return 1;
    }
  }
  std::cout << "Compiled and CPU-referenced CONV_2D op " << op_index << "\n";
  return 0;
}
}  // namespace
}  // namespace litert::arm_vulkan_ml

int main(int argc, char** argv) {
  return litert::arm_vulkan_ml::RunConv2dFixture(argc, argv);
}
