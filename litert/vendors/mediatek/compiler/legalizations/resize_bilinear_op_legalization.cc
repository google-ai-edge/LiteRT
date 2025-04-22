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

#include "litert/vendors/mediatek/compiler/legalizations/resize_bilinear_op_legalization.h"

#include <cstdint>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

#define GET_DIMENSION(op) ((op).RankedTensorType()->Layout().Dimensions())
#define CHECK_OP_IDX_AND_RETURN_ERROR(op_idx) \
  if (!(op_idx)) {                            \
    return (op_idx).Error();                  \
  }

namespace litert::mediatek {

Expected<void> LegalizeResizeBilinearOp(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    OperandMap& operand_map, const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize ResizeBilinear");

  // Only the first input tensor is added. The second one,
  // specifying the output height and width, is not added and
  // instead the height and width will be added individually as
  // scalars.
  std::vector<uint32_t> input_indices;
  auto input_tensor_id = operand_map.GetOperandIndex(op.Inputs()[0]);
  CHECK_OP_IDX_AND_RETURN_ERROR(input_tensor_id);
  input_indices.push_back(*input_tensor_id);

  auto output_dim = GET_DIMENSION(op.Outputs()[0]);
  const int32_t output_height = output_dim[1];
  auto output_height_operand_index = operand_map.AddScalarInt32(output_height);
  CHECK_OP_IDX_AND_RETURN_ERROR(output_height_operand_index);
  input_indices.push_back(*output_height_operand_index);

  const int32_t output_width = output_dim[2];
  auto output_width_operand_index = operand_map.AddScalarInt32(output_width);
  CHECK_OP_IDX_AND_RETURN_ERROR(output_width_operand_index);
  input_indices.push_back(*output_width_operand_index);

  // Use NHWC format
  auto format_operand_index = operand_map.AddScalarBool(false);
  CHECK_OP_IDX_AND_RETURN_ERROR(format_operand_index);
  input_indices.push_back(*format_operand_index);

  bool align_corners;
  if (auto status =
          LiteRtGetResizeBilinearAlignCornersOption(op.Get(), &align_corners);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get align corners");
  }
  auto align_corners_operand_index = operand_map.AddScalarBool(align_corners);
  CHECK_OP_IDX_AND_RETURN_ERROR(align_corners_operand_index);
  input_indices.push_back(*align_corners_operand_index);

  bool half_pixel_centers;
  if (auto status = LiteRtGetResizeBilinearHalfPixelCenterOption(
          op.Get(), &half_pixel_centers);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get align corners");
  }
  auto half_pixel_centers_operand_index =
      operand_map.AddScalarBool(half_pixel_centers);
  CHECK_OP_IDX_AND_RETURN_ERROR(half_pixel_centers_operand_index);
  input_indices.push_back(*half_pixel_centers_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    CHECK_OP_IDX_AND_RETURN_ERROR(id);
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model,
                        /*type=*/NEURON_RESIZE_BILINEAR, input_indices,
                        output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add NEURON_RESIZE_BILINEAR op");
  }

  return {};
}

}  // namespace litert::mediatek
