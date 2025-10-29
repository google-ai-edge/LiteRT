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

#include "litert/vendors/mediatek/compiler/legalizations/split_op_legalization.h"

#include <cstdint>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<void> LegalizeSplitOp(const NeuronAdapterApi& neuron_adapter_api,
                               NeuronModel* model, OperandMap& operand_map,
                               const litert::Op& op) {
  std::vector<uint32_t> input_indices;

  // op.Inputs()[1] is the input data
  LITERT_ASSIGN_OR_RETURN(auto input_tensor_id,
                          operand_map.GetOperandIndex(op.Inputs()[1]));
  input_indices.push_back(input_tensor_id);

  // op.Inputs()[0] is the axis
  LITERT_ASSIGN_OR_RETURN(auto axis_tensor_id,
                          operand_map.GetOperandIndex(op.Inputs()[0]));
  input_indices.push_back(axis_tensor_id);

  int32_t num_splits;
  if (auto status = LiteRtGetSplitNumSplitsOption(op.Get(), &num_splits);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get LiteRtGetSplitNumSplitsOption");
  }
  auto num_splits_operand_index = operand_map.AddScalarInt32(num_splits);
  if (!num_splits_operand_index) {
    return num_splits_operand_index.Error();
  }
  input_indices.push_back(*num_splits_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, NEURON_SPLIT, input_indices,
                        output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to add operation");
  }

  return {};
}

}  // namespace litert::mediatek
