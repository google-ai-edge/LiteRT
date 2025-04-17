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

#include "litert/vendors/mediatek/compiler/legalizations/gelu_op_legalization.h"

#include <cstdint>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

constexpr uint32_t kGeluApproximateTanh = 1;

Expected<void> LegalizeGeluOp(const NeuronAdapterApi& neuron_adapter_api,
                              NeuronModel* model, OperandMap& operand_map,
                              const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize Gelu");
  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  auto approximate_operand = operand_map.AddScalarUInt32(kGeluApproximateTanh);
  if (!approximate_operand) {
    return approximate_operand.Error();
  }

  input_indices.push_back(*approximate_operand);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, /*type=*/NEURON_GELU_V2,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add GELU operation");
  }

  return {};
}

}  // namespace litert::mediatek
