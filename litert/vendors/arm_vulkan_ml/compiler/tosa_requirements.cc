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

#include "litert/vendors/arm_vulkan_ml/compiler/tosa_requirements.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "spirv/Module.hpp"
#include "tosa-for-spirv-codegen/spirv/Instruction.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/spirv/Operand.hpp"  // from @tosa_for_spirv_codegen
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "spirv/unified1/spirv.hpp"  // from @arm_vulkan_ml_dep_spirv_headers
#include "tosa/OperatorEnum.hpp"

namespace litert::arm_vulkan_ml {
namespace {
using ::tfsc::spirv::Instruction;
using ::tfsc::spirv::Operand;
using ::tfsc::tosa::OperatorEnum;

const Instruction* Reference(const Operand& operand) {
  return operand.m_Type == tfsc::spirv::INSTRUCTION_POINTER
             ? operand.m_InstructionPtr
             : nullptr;
}

bool Integer(const Operand& operand, uint32_t& value) {
  const Instruction* constant = Reference(operand);
  if (!constant || constant->m_Opcode != spv::Op::OpConstant ||
      constant->m_Operands.size() != 3 ||
      constant->m_Operands[2].m_Type != tfsc::spirv::LITERAL_WORD) {
    return false;
  }
  value = constant->m_Operands[2].m_LiteralWord;
  return true;
}

bool Components(const Instruction* constant, size_t count,
                std::vector<uint32_t>& values) {
  if (!constant) return false;
  values.clear();
  if (constant->m_Opcode == spv::Op::OpConstantNull &&
      constant->m_Operands.size() == 2) {
    values.assign(count, 0);
    return true;
  }
  if (constant->m_Opcode == spv::Op::OpConstantCompositeReplicateEXT &&
      constant->m_Operands.size() == 3) {
    uint32_t value;
    if (!Integer(constant->m_Operands[2], value)) return false;
    values.assign(count, value);
    return true;
  }
  if (constant->m_Opcode != spv::Op::OpConstantComposite ||
      constant->m_Operands.size() != count + 2) {
    return false;
  }
  for (size_t i = 2; i < constant->m_Operands.size(); ++i) {
    uint32_t value;
    if (!Integer(constant->m_Operands[i], value)) return false;
    values.push_back(value);
  }
  return true;
}

bool Shape(const Instruction* tensor, std::vector<uint32_t>& shape) {
  if (!tensor || tensor->m_Opcode != spv::Op::OpTypeTensorARM ||
      tensor->m_Operands.size() != 4) {
    return false;
  }
  uint32_t rank;
  if (!Integer(tensor->m_Operands[2], rank)) return false;
  const Instruction* dimensions = Reference(tensor->m_Operands[3]);
  return Components(dimensions, rank, shape) &&
         std::all_of(shape.begin(), shape.end(),
                     [](uint32_t dimension) { return dimension != 0; });
}

bool ValueShape(const Operand& operand, std::vector<uint32_t>& shape) {
  const Instruction* value = Reference(operand);
  return value && !value->m_Operands.empty() &&
         Shape(Reference(value->m_Operands[0]), shape);
}

bool Integers(const Operand& operand, size_t count,
              std::vector<int64_t>& values) {
  const Instruction* constant = Reference(operand);
  if (!constant) return false;
  values.clear();
  std::vector<uint32_t> shape, components;
  if (!ValueShape(operand, shape) ||
      shape != std::vector<uint32_t>{static_cast<uint32_t>(count)} ||
      !Components(constant, count, components)) {
    return false;
  }
  for (uint32_t value : components) {
    // TOSA uses signless SPIR-V integers; pad/stride/dilation are signed.
    values.push_back(value <= INT32_MAX
                         ? static_cast<int64_t>(value)
                         : static_cast<int64_t>(value) - (int64_t{1} << 32));
  }
  return true;
}

bool Maximum(const Operand& operand, size_t count, uint64_t& maximum,
             bool signed_padding = false) {
  std::vector<int64_t> values;
  if (!Integers(operand, count, values)) return false;
  for (int64_t value : values) {
    if (value < 0 && !signed_padding) return false;
    maximum =
        std::max(maximum, static_cast<uint64_t>(value < 0 ? -value : value));
  }
  return true;
}
}  // namespace

GraphMetrics CollectGraphMetrics(const tfsc::spirv::Module& module) {
  GraphMetrics metrics;
  for (const Instruction& instruction : module.GetSpirvGraph()) {
    const std::vector<Operand>& operands = instruction.m_Operands;
    if (instruction.m_Opcode != spv::Op::OpExtInst || operands.size() < 5) {
      continue;
    }
    ++metrics.operators[operands[3].m_LiteralWord];
    const Instruction* type = Reference(operands[0]);
    std::vector<uint32_t> shape;
    if (!Shape(type, shape)) continue;
    const Instruction* element = Reference(type->m_Operands[1]);
    if (!element || (element->m_Opcode != spv::Op::OpTypeBool &&
                     element->m_Operands.size() < 2)) {
      continue;
    }
    uint64_t bytes = element->m_Opcode == spv::Op::OpTypeBool
                         ? 1
                         : (element->m_Operands[1].m_LiteralWord + 7) / 8;
    for (uint32_t dim : shape) bytes *= dim;
    metrics.largest_result_bytes =
        std::max(metrics.largest_result_bytes, bytes);
    if (static_cast<OperatorEnum>(operands[3].m_LiteralWord) !=
        OperatorEnum::Reshape) {
      continue;
    }
    metrics.reshape_bytes += bytes;
    const Instruction* input = Reference(operands[4]);
    std::vector<uint32_t> input_shape;
    if (input && !input->m_Operands.empty() &&
        ValueShape(operands[4], input_shape) && shape == input_shape &&
        Reference(input->m_Operands[0]) == type) {
      ++metrics.identity_reshapes;
      metrics.identity_reshape_bytes += bytes;
    }
  }
  return metrics;
}

bool CollectTosaLevelRequirements(const tfsc::spirv::Module& module,
                                  TosaLevelRequirements& requirements,
                                  std::string& reason) {
  TosaLevelRequirements result;
  auto reject = [&]() {
    reason = "Cannot determine emitted TOSA level requirements";
    return false;
  };
  size_t graphs = 0;
  for (const Instruction& instruction : module.GetSpirvGraph()) {
    const std::vector<Operand>& operands = instruction.m_Operands;
    if (instruction.m_Opcode == spv::Op::OpGraphARM) ++graphs;
    if (instruction.m_Opcode != spv::Op::OpTypeTensorARM) continue;
    std::vector<uint32_t> shape;
    if (!Shape(&instruction, shape)) return reject();
    const Instruction* element = Reference(operands[1]);
    if (!element) return reject();
    uint64_t bytes = 1;
    if (element->m_Opcode != spv::Op::OpTypeBool) {
      if ((element->m_Opcode != spv::Op::OpTypeInt &&
           element->m_Opcode != spv::Op::OpTypeFloat) ||
          element->m_Operands.size() < 2 ||
          element->m_Operands[1].m_Type != tfsc::spirv::LITERAL_WORD) {
        return reject();
      }
      bytes =
          std::max(uint64_t{1},
                   (uint64_t{element->m_Operands[1].m_LiteralWord} + 7) / 8);
    }
    for (uint32_t dimension : shape) {
      if (bytes > std::numeric_limits<uint64_t>::max() / dimension) {
        return reject();
      }
      bytes *= dimension;
    }
    result.max_rank =
        std::max(result.max_rank, static_cast<uint64_t>(shape.size()));
    result.max_tensor_bytes = std::max(result.max_tensor_bytes, bytes);
  }
  // This compiler emits a single flat graph with no control-flow instructions.
  if (graphs != 1 || !result.max_tensor_bytes) return reject();
  result.max_nesting = 1;
  for (const Instruction& instruction : module.GetSpirvGraph()) {
    if (instruction.m_Opcode != spv::Op::OpExtInst) continue;
    const std::vector<Operand>& operands = instruction.m_Operands;
    if (operands.size() < 4 ||
        operands[3].m_Type != tfsc::spirv::LITERAL_WORD) {
      return reject();
    }
    const Instruction* set = Reference(operands[2]);
    if (!set || set->m_Opcode != spv::Op::OpExtInstImport ||
        set->m_Operands.size() != 2 ||
        set->m_Operands[1].m_Type != tfsc::spirv::LITERAL_STRING ||
        *set->m_Operands[1].m_LiteralStr != "TOSA.001000.1") {
      return reject();
    }
    switch (static_cast<OperatorEnum>(operands[3].m_LiteralWord)) {
      case OperatorEnum::Conv2d:
      case OperatorEnum::DepthwiseConv2d:
      case OperatorEnum::Conv3d: {
        const auto op = static_cast<OperatorEnum>(operands[3].m_LiteralWord);
        const size_t dimensions = op == OperatorEnum::Conv3d ? 3 : 2;
        std::vector<int64_t> dilation;
        std::vector<uint32_t> weight;
        if (operands.size() != 14 ||
            !Maximum(operands[4], 2 * dimensions, result.max_kernel) ||
            !Maximum(operands[5], dimensions, result.max_stride) ||
            !Integers(operands[6], dimensions, dilation) ||
            !ValueShape(operands[10], weight) ||
            weight.size() != dimensions + 2) {
          return reject();
        }
        for (size_t i = 0; i < dimensions; ++i) {
          if (dilation[i] <= 0) return reject();
          const uint32_t kernel =
              weight[i + (op == OperatorEnum::DepthwiseConv2d ? 0 : 1)];
          result.max_kernel = std::max(
              result.max_kernel, static_cast<uint64_t>(dilation[i]) * kernel);
        }
        break;
      }
      case OperatorEnum::TransposeConv2d: {
        std::vector<uint32_t> weight;
        if (operands.size() != 13 ||
            !Maximum(operands[4], 4, result.max_kernel, true) ||
            !Maximum(operands[5], 2, result.max_stride) ||
            !ValueShape(operands[9], weight) || weight.size() != 4) {
          return reject();
        }
        result.max_kernel = std::max(
            {result.max_kernel, uint64_t{weight[1]}, uint64_t{weight[2]}});
        break;
      }
      case OperatorEnum::AvgPool2d:
        if (operands.size() != 11 ||
            !Maximum(operands[4], 2, result.max_kernel) ||
            !Maximum(operands[5], 2, result.max_stride) ||
            !Maximum(operands[6], 4, result.max_kernel)) {
          return reject();
        }
        break;
      case OperatorEnum::MaxPool2d:
        if (operands.size() != 9 ||
            !Maximum(operands[4], 2, result.max_kernel) ||
            !Maximum(operands[5], 2, result.max_stride) ||
            !Maximum(operands[6], 4, result.max_kernel)) {
          return reject();
        }
        break;
      case OperatorEnum::Resize: {
        std::vector<int64_t> scale;
        if (operands.size() != 9 || !Integers(operands[6], 4, scale)) {
          return reject();
        }
        for (size_t i : {size_t{0}, size_t{2}}) {
          if (scale[i] <= 0 || scale[i + 1] <= 0) return reject();
          result.max_scale =
              std::max(result.max_scale,
                       static_cast<uint64_t>((scale[i] + scale[i + 1] - 1) /
                                             scale[i + 1]));
        }
        break;
      }
      case OperatorEnum::Concat:
        if (operands.size() <= 5) return reject();
        result.max_tensor_list_size =
            std::max(result.max_tensor_list_size,
                     static_cast<uint64_t>(operands.size() - 5));
        break;
      case OperatorEnum::Fft2d:
      case OperatorEnum::Rfft2d: {
        const bool real =
            static_cast<OperatorEnum>(operands[3].m_LiteralWord) ==
            OperatorEnum::Rfft2d;
        std::vector<uint32_t> shape;
        if (operands.size() != (real ? 6 : 8) ||
            !ValueShape(operands[real ? 5 : 6], shape) || shape.size() != 3) {
          return reject();
        }
        for (size_t axis : {size_t{1}, size_t{2}}) {
          if (!shape[axis] || (shape[axis] & (shape[axis] - 1))) {
            return reject();
          }
        }
        // RFFT2D/FFT2D have power-of-two shape rules, not a MAX_KERNEL
        // requirement. Their global rank/tensor byte limits were collected
        // above.
        break;
      }
      default:
        break;  // Remaining mapped operators have only the global tensor
                // limits.
    }
  }
  result.checked = true;
  requirements = result;
  return true;
}
}  // namespace litert::arm_vulkan_ml
