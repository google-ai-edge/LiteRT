// Copyright (c) 2025 MediaTek Inc.
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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_COMMON_OP_LEGALIZATION_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_COMMON_OP_LEGALIZATION_H_

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

bool VerifyCommonOp(const litert::Op& op, LiteRtOpCode op_code);

template <typename T>
inline Expected<NeuronOperationType> ResolveOpType(
    OperandMap& operand_map, std::vector<uint32_t>& input_indices,
    T type_or_name) {
  if constexpr (std::is_same_v<T, NeuronOperationType>) {
    return type_or_name;
  } else {
    NeuronOperationType nn_op_type;
    auto custom_operand_index =
        operand_map.AddOemExtensionOperand(type_or_name, &nn_op_type);

    if (!custom_operand_index) {
      return custom_operand_index.Error();
    }

    input_indices.push_back(*custom_operand_index);

    return nn_op_type;
  }
}

Expected<void> LegalizeCommonOp(const NeuronAdapterApi& neuron_adapter_api,
                                NeuronModel* model, OperandMap& operand_map,
                                const litert::Op& op,
                                NeuronOperationType mtk_operation_type);

template <typename OpTypeOrName, typename... AdditionalOperands>
Expected<void> LegalizeOp(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    OperandMap& operand_map, const litert::Op& op, OpTypeOrName op_type_or_name,
    std::tuple<AdditionalOperands...> additional_operands) {
  LITERT_LOG(LITERT_INFO, "Legalize Operation %d", op.Code());

  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  // Add additional operands to input_indices
  auto add_operand = [&](auto& operand) -> Expected<void> {
    auto additional_operand = operand(op, operand_map);
    if (!additional_operand) {
      return additional_operand.Error();
    }
    input_indices.push_back(*additional_operand);
    return {};
  };

  auto result = std::apply(
      [&](auto&... operands) -> Expected<void> {
        Expected<void> res = {};
        ((res = add_operand(operands), res) && ...);
        return res;
      },
      additional_operands);

  if (!result) {
    return result.Error();
  }

  auto resolved_type =
      ResolveOpType(operand_map, input_indices, op_type_or_name);
  if (!resolved_type) {
    return resolved_type.Error();
  }
  NeuronOperationType final_op_type = *resolved_type;

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, final_op_type, input_indices,
                        output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to add operation");
  }

  return {};
}

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_COMMON_OP_LEGALIZATION_H_
