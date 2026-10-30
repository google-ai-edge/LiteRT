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

#include <chrono>
#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "TosaForSpirvCodegen.hpp"
#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"
#include "litert/vendors/arm_vulkan_ml/compiler/tosa_requirements.h"

namespace litert::arm_vulkan_ml {
std::shared_ptr<const CompiledPartition> CompilePartition(
    const Subgraph& subgraph, std::string& reason,
    std::vector<LoweringRecord>* diagnostics, OptimizationMode mode,
    OptimizationReport* optimization, SoftmaxLutType softmax_lut) {
  const auto start = std::chrono::steady_clock::now();
  // Import and optimize without allocating any TFSC result IDs. The operation
  // analysis is shared with partition selection; optimizations cannot add ops
  // to a partition that selection rejected.
  auto ir = ImportGraph(subgraph, reason, softmax_lut);
  if (!ir || !OptimizeGraph(*ir, mode, optimization, reason)) return nullptr;
  auto artifact = std::make_shared<CompiledPartition>();
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, artifact->entry_point);
  std::vector<tfsc::tosa::ResId> values(ir->values.size());
  GraphConstantPool constants;
  constants.deduplicate = mode != kLrtArmVulkanMLOptimizationOff;
  uint32_t binding = 0;
  auto descriptor = [&](const GraphValue& value) {
    TensorDescriptor result{
        value.storage_type == tfsc::tosa::DataType::float32_t ? TensorElement::kFloat32 :
        value.storage_type == tfsc::tosa::DataType::int32_t ? TensorElement::kInt32 : TensorElement::kInt8,
        value.shape, binding++};
    result.logical_element = value.logical_type;
    result.logical_scalar = value.logical_scalar;
    return result;
  };
  auto logical = [&](tfsc::tosa::ResId id, const GraphValue& value) {
    if (value.logical_type == kLiteRtElementTypeBool)
      return graph.AddCastOperator(id, tfsc::tosa::Tensor(tfsc::tosa::DataType::bool_t, value.shape));
    return id;
  };
  for (auto id : ir->inputs) {
    const auto& value = ir->values[id];
    auto input = descriptor(value);
    values[id] = logical(graph.AddInput(tfsc::tosa::Tensor(value.storage_type, value.shape), input.binding), value);
    artifact->inputs.push_back(std::move(input));
  }
  std::set<uint32_t> seen_ids;
  auto new_ids = [&]() {
    std::vector<uint32_t> ids;
    for (const auto& instruction : module->GetSpirvGraph())
      for (const auto& operand : instruction.m_Operands)
        if (operand.m_Type == tfsc::spirv::RES_ID && seen_ids.insert(operand.m_LiteralWord).second)
          ids.push_back(operand.m_LiteralWord);
    return ids;
  };
  if (diagnostics) {
    diagnostics->clear();
    for (size_t i = 0; i < ir->original_operator_count; ++i)
      diagnostics->push_back({i, {}, {}, "removed"});
    new_ids();
  }
  auto get_value = [&](ValueId id) {
    const auto& value = ir->values[id];
    if (!values[id] && value.is_constant)
      values[id] = logical(AddGraphConstant(graph, tfsc::tosa::Tensor(value.storage_type, value.shape), value.constant, constants), value);
    return values[id];
  };
  for (const auto& node : ir->nodes) {
    if (node.removed) continue;
    for (auto id : node.inputs) if (ir->values[id].storage_type == tfsc::tosa::DataType::float32_t && artifact->profiles.size() == 1)
      artifact->profiles.push_back("PRO-FP");
    for (auto id : node.outputs) if (ir->values[id].storage_type == tfsc::tosa::DataType::float32_t && artifact->profiles.size() == 1)
      artifact->profiles.push_back("PRO-FP");
    if (node.body.softmax && node.body.softmax->lut_type == kLrtArmVulkanMLSoftmaxLutInt16 &&
        std::find(artifact->extensions.begin(), artifact->extensions.end(), "EXT-INT16") == artifact->extensions.end())
      artifact->extensions.push_back("EXT-INT16");
    for (const auto& extension : node.body.required_extensions)
      if (std::find(artifact->extensions.begin(), artifact->extensions.end(), extension) == artifact->extensions.end())
        artifact->extensions.push_back(extension);
    const size_t reused_before = constants.reused;
    std::vector<tfsc::tosa::ResId> inputs, outputs;
    for (auto id : node.inputs) inputs.push_back(get_value(id));
    if (node.body.portable) outputs = EmitPortableOperation(graph, inputs, *node.body.portable);
    else if (node.transform_only) outputs = {inputs[0]};
    else outputs = {EmitOperation(graph, inputs, node.body, constants)};
    if (outputs.size() != node.outputs.size()) { reason = "IR expansion result count mismatch"; return nullptr; }
    for (const auto& transform : node.output_transforms) {
      if (!outputs[0] || tfsc::IsError(outputs[0])) { reason = tfsc::GetErrorMsg(outputs[0]); return nullptr; }
      outputs[0] = EmitOutputTransform(graph, outputs[0], transform, constants);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i] || tfsc::IsError(outputs[i])) { reason = tfsc::GetErrorMsg(outputs[i]); return nullptr; }
      values[node.outputs[i]] = outputs[i];
    }
    if (optimization && constants.reused != reused_before)
      optimization->decisions.push_back({"intern_constants", node.source_operators, true,
          std::to_string(constants.reused - reused_before) + " immutable constants reused with identical storage type/shape/bytes"});
    if (diagnostics) {
      // Many-to-one provenance, with newly created IDs attributed once only.
      auto& primary = (*diagnostics)[node.source_operators[0]];
      auto ids = new_ids();
      primary.result_ids.insert(primary.result_ids.end(), ids.begin(), ids.end());
      for (auto index : node.source_operators) {
        auto& record = (*diagnostics)[index];
        record.source_operators = node.source_operators;
        record.composite = CompositeKindName(node.kind);
      }
    }
  }
  for (auto id : ir->outputs) {
    const auto& value = ir->values[id];
    auto result = get_value(id);
    if (!result) { reason = "Unbound partition output"; return nullptr; }
    auto output = descriptor(value);
    if (value.logical_type == kLiteRtElementTypeBool)
      result = graph.AddCastOperator(result, tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t, output.shape));
    graph.AddOutput(result, output.binding);
    artifact->outputs.push_back(std::move(output));
  }
  graph.FinalizeGraph();
  // Internal decompositions can use FP32 even with integer graph interfaces
  // (for example BUCKETIZE). Derive profile requirements from emitted types.
  for (const auto& instruction : module->GetSpirvGraph()) {
    if (instruction.m_Opcode == spv::Op::OpTypeFloat && artifact->profiles.size() == 1)
      artifact->profiles.push_back("PRO-FP");
  }
  artifact->spirv = tfsc::WriteToBinary(module);
  if (artifact->spirv.empty()) { reason = "TFSC failed to serialize partition"; return nullptr; }
  if (!CollectTosaLevelRequirements(*module, artifact->level_requirements, reason)) return nullptr;
  if (optimization) {
    for (size_t i = 5; i < artifact->spirv.size();) {
      uint32_t count = artifact->spirv[i] >> 16;
      if (!count || count > artifact->spirv.size() - i) break;
      if ((artifact->spirv[i] & 0xffff) == 12) ++optimization->emitted_operators;
      i += count;
    }
  }
  for (auto& constant : constants.values) {
    const auto element = constant.tensor.GetDataType() == tfsc::tosa::DataType::int8_t
        ? TensorElement::kInt8 : constant.tensor.GetDataType() == tfsc::tosa::DataType::float32_t
        ? TensorElement::kFloat32 : TensorElement::kInt32;
    artifact->constants.push_back({{element, constant.tensor.GetTensorShape(),
        static_cast<uint32_t>(artifact->constants.size())}, std::move(constant.bytes)});
  }
  if (optimization) {
    optimization->constant_count = artifact->constants.size();
    for (const auto& constant : artifact->constants) optimization->constant_bytes += constant.bytes.size();
    optimization->compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  return artifact;
}
}  // namespace litert::arm_vulkan_ml
