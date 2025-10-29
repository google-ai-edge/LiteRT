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

#include "litert/vendors/mediatek/compiler/legalizations/strided_slice_op_legalization.h"

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

Expected<void> LegalizeStridedSliceOp(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    OperandMap& operand_map, const litert::Op& op) {
  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  int32_t begin_mask;
  if (auto status = LiteRtGetStridedSliceBeginMaskOption(op.Get(), &begin_mask);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get LiteRtGetStridedSliceBeginMaskOption");
  }
  auto begin_mask_operand_index = operand_map.AddScalarInt32(begin_mask);
  if (!begin_mask_operand_index) {
    return begin_mask_operand_index.Error();
  }
  input_indices.push_back(*begin_mask_operand_index);

  int32_t end_mask;
  if (auto status = LiteRtGetStridedSliceEndMaskOption(op.Get(), &end_mask);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get LiteRtGetStridedSliceEndMaskOption");
  }

  auto end_mask_operand_index = operand_map.AddScalarInt32(end_mask);
  if (!end_mask_operand_index) {
    return end_mask_operand_index.Error();
  }
  input_indices.push_back(*end_mask_operand_index);

  int32_t shrink_axis_mask;
  if (auto status = LiteRtGetStridedSliceShrinkAxisMaskOption(
          op.Get(), &shrink_axis_mask);
      status != kLiteRtStatusOk) {
    return Error(status,
                 "Failed to get LiteRtGetStridedSliceShrinkAxisMaskOption");
  }

  auto shrink_axis_mask_operand_index =
      operand_map.AddScalarInt32(shrink_axis_mask);
  if (!shrink_axis_mask_operand_index) {
    return shrink_axis_mask_operand_index.Error();
  }
  input_indices.push_back(*shrink_axis_mask_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, NEURON_STRIDED_SLICE,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to add operation");
  }

  return {};
}

}  // namespace litert::mediatek
