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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_EMISSION_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_EMISSION_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "tosa-for-spirv-codegen/spirv/Instruction.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/spirv/Operand.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "spirv/unified1/spirv.hpp"  // from @arm_vulkan_ml_dep_spirv_headers
#include "tosa/Graph.hpp"

namespace litert::arm_vulkan_ml {
// Use a graph constant for ranked scalars. TFSC's inline uniform-constant
// reduction does not build a valid nested composite for a single element at
// rank > 1. This also avoids adding a TOSA RESHAPE for broadcasting constants.
template <typename T>
inline tfsc::tosa::ResId BroadcastScalar(tfsc::tosa::Graph& graph, T value,
                                         size_t rank, GraphConstantPool& pool) {
  std::vector<uint8_t> bytes(sizeof(T));
  std::memcpy(bytes.data(), &value, sizeof(T));
  if (rank > 4) {
    const std::vector<uint32_t> target(rank, 1);
    tfsc::tosa::ResId one = AddGraphConstant(
        graph, tfsc::tosa::Tensor(tfsc::tosa::GetDataType<T>(), {1}),
        std::move(bytes), pool);
    return graph.AddReshapeOperator(
        one, graph.AddTensorConstant(tfsc::tosa::Attribute(target)),
        tfsc::tosa::Tensor(tfsc::tosa::GetDataType<T>(), target));
  }
  return AddGraphConstant(graph,
                          tfsc::tosa::Tensor(tfsc::tosa::GetDataType<T>(),
                                             std::vector<uint32_t>(rank, 1)),
                          std::move(bytes), pool);
}
// TFSC uses instruction pointers for typed operands. Inspect only exact static
// shapes here; uncertainty preserves the requested operation and its checks.
inline const tfsc::spirv::Instruction* EmissionReference(
    const tfsc::spirv::Operand& operand) {
  return operand.m_Type == tfsc::spirv::INSTRUCTION_POINTER
             ? operand.m_InstructionPtr
             : nullptr;
}
// Extracts a 32-bit unsigned integer literal from an `OpConstant` operand.
inline bool EmissionInteger(const tfsc::spirv::Operand& operand,
                            uint32_t& value) {
  const tfsc::spirv::Instruction* constant_inst = EmissionReference(operand);
  if (!constant_inst || constant_inst->m_Opcode != spv::Op::OpConstant ||
      constant_inst->m_Operands.size() != 3 ||
      constant_inst->m_Operands[2].m_Type != tfsc::spirv::LITERAL_WORD) {
    return false;
  }
  value = constant_inst->m_Operands[2].m_LiteralWord;
  return true;
}
// Emits a TOSA RESHAPE to `output`, eliding the operator when `value` already
// has the exact static element type and shape of `output`.
inline tfsc::tosa::ResId ReshapeTensor(tfsc::tosa::Graph& graph,
                                       tfsc::tosa::ResId value,
                                       tfsc::tosa::ResId shape,
                                       const tfsc::tosa::Tensor& output) {
  const tfsc::spirv::Instruction* type =
      value && !value->m_Operands.empty()
          ? EmissionReference(value->m_Operands[0])
          : nullptr;
  if (type && type->m_Opcode == spv::Op::OpTypeTensorARM &&
      type->m_Operands.size() == 4) {
    uint32_t rank = 0;
    const tfsc::spirv::Instruction* dims =
        EmissionReference(type->m_Operands[3]);
    const tfsc::spirv::Instruction* element =
        EmissionReference(type->m_Operands[1]);
    const tfsc::tosa::DataType data_type = output.GetDataType();
    const uint32_t bits = data_type == tfsc::tosa::DataType::int8_t    ? 8
                          : data_type == tfsc::tosa::DataType::int16_t ? 16
                          : data_type == tfsc::tosa::DataType::int48_t ? 64
                                                                       : 32;
    bool same =
        element && element->m_Operands.size() >= 2 &&
        element->m_Operands[1].m_LiteralWord == bits &&
        (element->m_Opcode == (data_type == tfsc::tosa::DataType::float32_t
                                   ? spv::Op::OpTypeFloat
                                   : spv::Op::OpTypeInt));
    if (same && EmissionInteger(type->m_Operands[2], rank) &&
        rank == output.GetTensorShape().size() && dims) {
      if (dims->m_Opcode == spv::Op::OpConstantComposite &&
          dims->m_Operands.size() == rank + 2) {
        for (size_t i = 0; i < rank; ++i) {
          uint32_t dimension = 0;
          if (!EmissionInteger(dims->m_Operands[i + 2], dimension) ||
              dimension != output.GetTensorShape()[i]) {
            same = false;
          }
        }
      } else if (dims->m_Opcode == spv::Op::OpConstantCompositeReplicateEXT &&
                 dims->m_Operands.size() == 3) {
        uint32_t dimension = 0;
        same = EmissionInteger(dims->m_Operands[2], dimension);
        for (uint32_t dim : output.GetTensorShape()) {
          same &= dim == dimension;
        }
      } else {
        same = false;
      }
      if (same) return value;
    }
  }
  return graph.AddReshapeOperator(value, shape, output);
}
}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_EMISSION_H_
