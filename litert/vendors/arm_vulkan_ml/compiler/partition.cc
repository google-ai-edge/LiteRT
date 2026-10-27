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

#include "litert/vendors/arm_vulkan_ml/compiler/partition.h"
#include <unordered_map>
#include <utility>
#include "TosaForSpirvCodegen.hpp"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

namespace litert::arm_vulkan_ml {
std::shared_ptr<const CompiledPartition> CompilePartition(const Subgraph& subgraph,
                                                        std::string& reason) {
  auto artifact = std::make_shared<CompiledPartition>();
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, artifact->entry_point);
  std::unordered_map<LiteRtTensor, tfsc::tosa::ResId> values;
  std::vector<GraphConstant> constants;
  uint32_t binding = 0;
  auto descriptor = [&](const Tensor& tensor) -> std::optional<TensorDescriptor> {
    auto ranked = tensor.RankedTensorType();
    if (!ranked || (ranked->ElementType() != ElementType::Int8 && ranked->ElementType() != ElementType::Float32) ||
        ranked->Layout().HasStrides()) return std::nullopt;
    auto dims = ranked->Layout().Dimensions();
    if (dims.empty()) return std::nullopt;
    TensorDescriptor result{ranked->ElementType() == ElementType::Float32 ? TensorElement::kFloat32 : TensorElement::kInt8, {}, binding++};
    uint64_t count = 1;
    for (auto dim : dims) {
      if (dim <= 0 || count > uint64_t{INT32_MAX} / dim) return std::nullopt;
      count *= dim;
      result.shape.push_back(dim);
    }
    return result;
  };
  for (const auto& tensor : subgraph.Inputs()) {
    if (tensor.HasWeights()) continue;
    auto input = descriptor(tensor);
    if (!input) { reason = "Invalid partition input layout/type"; return nullptr; }
    values[tensor.Get()] = graph.AddInput(
        tfsc::tosa::Tensor(input->element == TensorElement::kFloat32 ? tfsc::tosa::DataType::float32_t : tfsc::tosa::DataType::int8_t, input->shape), input->binding);
    artifact->inputs.push_back(std::move(*input));
  }
  if (subgraph.Ops().empty()) { reason = "Empty partition"; return nullptr; }
  for (const auto& op : subgraph.Ops()) {
    // Partition selection and compilation use the exact same analysis.
    auto plan = AnalyzeOperation(op, reason);
    if (!plan) return nullptr;
    if ((plan->inputs[0].floating || plan->output.floating) && artifact->profiles.size()==1)
      artifact->profiles.push_back("PRO-FP");
    std::vector<tfsc::tosa::ResId> inputs;
    for (int index : plan->runtime_inputs) {
      auto tensor = op.Inputs()[index];
      auto value = values.find(tensor.Get());
      if (value == values.end() && tensor.HasWeights()) {
        auto bytes = tensor.Weights().Bytes();
        auto type = tensor.RankedTensorType();
        if (!type || tensor.ElementType() != ElementType::Int8) {
          reason = "Invalid constant activation"; return nullptr;
        }
        auto dims = type->Layout().Dimensions();
        tfsc::tosa::Tensor constant_type(tfsc::tosa::DataType::int8_t,
                                       {dims.begin(), dims.end()});
        if (bytes.size() != constant_type.GetNumElements()) {
          reason = "Invalid constant byte count"; return nullptr;
        }
        constants.push_back({constant_type,{bytes.begin(),bytes.end()}});
        value = values.emplace(tensor.Get(), graph.AddGraphConstant(constant_type)).first;
      }
      if (value == values.end()) {
        reason = "Partition is not topologically ordered or has an unbound input";
        return nullptr;
      }
      inputs.push_back(value->second);
    }
    auto output = EmitOperation(graph, inputs, *plan, constants);
    if (!output || tfsc::IsError(output)) { reason = tfsc::GetErrorMsg(output); return nullptr; }
    values[op.Outputs()[0].Get()] = output;
  }
  for (const auto& tensor : subgraph.Outputs()) {
    auto value = values.find(tensor.Get());
    auto output = descriptor(tensor);
    if (value == values.end() || !output) {
      reason = "Unbound or invalid partition output"; return nullptr;
    }
    graph.AddOutput(value->second, output->binding);
    artifact->outputs.push_back(std::move(*output));
  }
  if (artifact->outputs.empty()) { reason = "Partition has no outputs"; return nullptr; }
  graph.FinalizeGraph();
  artifact->spirv = tfsc::WriteToBinary(module);
  if (artifact->spirv.empty()) { reason = "TFSC failed to serialize partition"; return nullptr; }
  for (auto& constant : constants) {
    const auto element = constant.tensor.GetDataType() == tfsc::tosa::DataType::int8_t
        ? TensorElement::kInt8 : constant.tensor.GetDataType() == tfsc::tosa::DataType::float32_t
        ? TensorElement::kFloat32 : TensorElement::kInt32;
    artifact->constants.push_back({{element, constant.tensor.GetTensorShape(),
        static_cast<uint32_t>(artifact->constants.size())}, std::move(constant.bytes)});
  }
  return artifact;
}
}  // namespace litert::arm_vulkan_ml
