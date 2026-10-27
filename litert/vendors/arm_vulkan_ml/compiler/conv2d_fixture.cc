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
// reference kernels. VGF packaging is done by the separate Python test harness.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "TosaForSpirvCodegen.hpp"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d_adapter.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register_ref.h"
#include "tflite/schema/schema_generated.h"

namespace {
template <typename V> void JsonArray(std::ostream& out, const V& values) {
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
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "Usage: conv2d_fixture MODEL OUTPUT_DIR OP_INDEX\n";
    return 1;
  }
  using namespace litert::arm_vulkan_ml;
  const int op_index = std::stoi(argv[3]);
  auto model = litert::ExtendedModel::CreateFromFile(argv[1]);
  if (!model) { std::cerr << "Cannot load model\n"; return 1; }
  auto subgraph = model->Subgraph(0);
  if (!subgraph || op_index < 0 || op_index >= subgraph->Ops().size()) return 1;
  std::string reason;
  auto plan = AnalyzeConv2d(subgraph->Ops()[op_index], reason);
  if (!plan) { std::cerr << reason << "\n"; return 1; }
  const auto& s = plan->spec;
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, "main");
  auto input = graph.AddInput(tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t,
      {s.input_shape.begin(), s.input_shape.end()}), 0);
  std::vector<GraphConstant> constants;
  tfsc::tosa::ResId accumulator;
  auto output = EmitConv2d(graph, input, *plan, constants, &accumulator);
  if (tfsc::IsError(output)) { std::cerr << tfsc::GetErrorMsg(output); return 1; }
  graph.AddOutput(output, 1);
  graph.AddOutput(accumulator, 2);
  graph.FinalizeGraph();
  auto spirv = tfsc::WriteToBinary(module);
  if (spirv.empty()) return 1;
  std::filesystem::path dir(argv[2]);
  std::filesystem::create_directories(dir);
  if (!Write(dir / "graph.spv", spirv.data(), spirv.size() * 4)) return 1;
  if(!Write(dir / "weights.bin",s.weights.data(),s.weights.size()) ||
     !Write(dir / "bias.bin",s.bias.data(),s.bias.size()*sizeof(int32_t)))return 1;
  std::ofstream manifest(dir / "graph.json");
  manifest.precision(17);
  manifest << "{\"input_shape\":";
  JsonArray(manifest, s.input_shape);
  manifest << ",\"output_shape\":";
  JsonArray(manifest, s.output_shape);
  manifest << ",\"weight_shape\":";
  JsonArray(manifest, s.weight_shape);
  manifest << ",\"padding\":";
  JsonArray(manifest, s.padding);
  manifest << ",\"stride\":";
  JsonArray(manifest, s.stride);
  manifest << ",\"dilation\":";
  JsonArray(manifest, s.dilation);
  manifest << ",\"input_zero_point\":" << s.input_zero_point
           << ",\"output_zero_point\":" << s.output_zero_point
           << ",\"activation_min\":" << plan->activation_range.minimum
           << ",\"activation_max\":" << plan->activation_range.maximum;
  std::vector<int32_t> multipliers, shifts;
  for (auto scale : plan->scales) { multipliers.push_back(scale.multiplier); shifts.push_back(scale.shift); }
  manifest << ",\"multipliers\":";
  JsonArray(manifest, multipliers);
  manifest << ",\"shifts\":";
  JsonArray(manifest, shifts);
  std::vector<int32_t> staged;
  for(size_t c=0;c<plan->scales.size();++c) {
    Conv2dPlan check{};check.scales={plan->scales[c]};check.accumulator_ranges={plan->accumulator_ranges[c]};
    staged.push_back(UsesStagedRounding(check));
  }
  manifest << ",\"staged_rounding_channels\":";
  JsonArray(manifest,staged);
  manifest << ",\"depthwise\":" << (s.depthwise ? "true" : "false");
  manifest << ",\"constants\":[";
  for (size_t i = 0; i < constants.size(); ++i) {
    const auto& c = constants[i];
    if (i) manifest << ",";
    manifest << "{\"file\":\"constant_" << i << ".bin\",\"dtype\":\""
             << (c.tensor.GetDataType() == tfsc::tosa::DataType::int8_t ? "int8" : "int32")
             << "\",\"shape\":";
    JsonArray(manifest, c.tensor.GetTensorShape());
    manifest << "}";
    if (!Write(dir / ("constant_" + std::to_string(i) + ".bin"), c.bytes.data(), c.bytes.size())) return 1;
  }
  manifest << "],\"reference\":\"BuiltinRefOpResolver; delegates disabled\",\"single_rounding\":";
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
    artifact->inputs.push_back({TensorElement::kInt8,
        {s.input_shape.begin(), s.input_shape.end()}, 0});
    artifact->outputs.push_back({TensorElement::kInt8,
        {s.output_shape.begin(), s.output_shape.end()}, 1});
    artifact->outputs.push_back({TensorElement::kInt32,
        {s.output_shape.begin(), s.output_shape.end()}, 2});
    for (const auto& c : constants) {
      auto type = c.tensor.GetDataType() == tfsc::tosa::DataType::int8_t
          ? TensorElement::kInt8 : TensorElement::kInt32;
      artifact->constants.push_back({{type, c.tensor.GetTensorShape(),
          static_cast<uint32_t>(artifact->constants.size())}, c.bytes});
    }
    invocation = VulkanInvocation::Create(artifact, true, reason);
    if (!invocation) { std::cerr << reason << "\n"; return 1; }
  }

  auto source = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  if (!source) return 1;
  std::unique_ptr<tflite::ModelT> object(source->GetModel()->UnPack());
  auto& sg = *object->subgraphs[0];
  auto selected = std::move(sg.operators[op_index]);
  sg.inputs = {selected->inputs[0]};
  sg.outputs = selected->outputs;
  sg.operators.clear();
  sg.operators.push_back(std::move(selected));
  object->signature_defs.clear();
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, object.get()));
  auto fixture = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
  if (!fixture) return 1;
  tflite::ops::builtin::BuiltinRefOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*fixture, resolver)(&interpreter) != kTfLiteOk ||
      interpreter->AllocateTensors() != kTfLiteOk) return 1;
  auto* input_tensor = interpreter->input_tensor(0);
  // Four deterministic inputs cover saturation, zero points, negative values,
  // and layout. The emulator consumes exactly these bytes.
  for (int case_id = 0; case_id < 4; ++case_id) {
    for (size_t i = 0; i < input_tensor->bytes; ++i) {
      int value = static_cast<int>((i * 73 + i / 11 + 29) % 256) - 128;
      if (case_id == 1) value = s.input_zero_point;
      if (case_id == 2) value = i % 2 ? 127 : -128;
      if (case_id == 3) value = -1 - (i * 17 % 128);
      input_tensor->data.int8[i] = value;
    }
    if (!Write(dir / ("input_" + std::to_string(case_id) + ".bin"),
               input_tensor->data.raw, input_tensor->bytes)) return 1;
    if (interpreter->Invoke() != kTfLiteOk) return 1;
    auto* result = interpreter->output_tensor(0);
    if (invocation) {
      std::vector<int8_t> actual(result->bytes);
      std::vector<int32_t> acc(result->bytes);
      if (!invocation->Run({input_tensor->data.raw}, {actual.data(), acc.data()}, reason)) {
        std::cerr << reason << "\n"; return 1;
      }
      for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(int(actual[i]) - int(result->data.int8[i])) > 1) {
          std::cerr << "Direct Vulkan output differs from CPU at " << i << "\n";
          return 1;
        }
      }
      if (!Write(dir / ("vulkan_" + std::to_string(case_id) + ".bin"), actual.data(), actual.size()) ||
          !Write(dir / ("accumulator_" + std::to_string(case_id) + ".bin"), acc.data(), acc.size() * 4)) return 1;
    }
    if (!Write(dir / ("reference_" + std::to_string(case_id) + ".bin"),
               result->data.raw, result->bytes)) return 1;
  }
  std::cout << "Compiled and CPU-referenced CONV_2D op " << op_index << "\n";
  return 0;
}
