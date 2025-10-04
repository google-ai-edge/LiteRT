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

#include "litert/vendors/mediatek/compiler/legalizations/transpose_conv_op_legalization.h"

#include <cstdint>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/legalize_helper.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

#define GET_DIMENSION(op) ((op).RankedTensorType()->Layout().Dimensions())
#define CHECK_OP_IDX_AND_RETURN_ERROR(op_idx) \
  if (!(op_idx)) {                            \
    return (op_idx).Error();                  \
  }

namespace litert::mediatek {

Expected<void> LegalizeTransposeConvOp(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    OperandMap& operand_map, const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize TransposeConv");

  std::vector<uint32_t> input_indices;
  int32_t input_tensor_flags = 0;

  auto input_tensor_id = operand_map.GetOperandIndex(
      op.Inputs()[2],
      input_tensor_flags | NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED);
  CHECK_OP_IDX_AND_RETURN_ERROR(input_tensor_id);
  input_indices.push_back(*input_tensor_id);

  auto weight_tensor_id = operand_map.GetOperandIndex(op.Inputs()[1]);
  CHECK_OP_IDX_AND_RETURN_ERROR(weight_tensor_id);
  input_indices.push_back(*weight_tensor_id);

  // if there's no bias input, add a zero bias
  if (op.Inputs().size() < 4) {
    if (!op.Inputs()[0].HasWeights()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "The output_shape input of TransposeConv is not const.");
    }
    auto output_shape_data = op.Inputs()[0].WeightsData<int32_t>();
    if (!output_shape_data.HasValue()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Fail to get data of output shape tensor.");
    }
    const int32_t output_depth = output_shape_data.Value()[3];

    auto zero_bias_idx = AddZeroBiasForConvBase(op.Inputs()[2], op.Inputs()[1],
                                                output_depth, operand_map);
    if (!zero_bias_idx) {
      return zero_bias_idx.Error();
    }
    input_indices.push_back(*zero_bias_idx);
  } else {
    auto bias_tensor_id = operand_map.GetOperandIndex(op.Inputs()[3]);
    CHECK_OP_IDX_AND_RETURN_ERROR(bias_tensor_id);
    input_indices.push_back(*bias_tensor_id);
  }

  auto output_shape_tensor_id = operand_map.GetOperandIndex(op.Inputs()[0]);
  CHECK_OP_IDX_AND_RETURN_ERROR(output_shape_tensor_id);
  input_indices.push_back(*output_shape_tensor_id);

  uint32_t padding;
  if (auto status = LiteRtGetTransposeConvPaddingOption(op.Get(), &padding);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get padding");
  }
  NeuronAdapterPaddingCode neuron_padding = NEURON_PADDING_SAME;
  LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, neuron_padding))
      << "Fails to convert padding";
  auto padding_operand_index = operand_map.AddScalarInt32(neuron_padding);
  CHECK_OP_IDX_AND_RETURN_ERROR(padding_operand_index);
  input_indices.push_back(*padding_operand_index);

  int32_t stride_width;
  if (auto status =
          LiteRtGetTransposeConvStrideWOption(op.Get(), &stride_width);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get stride width");
  }
  auto stride_width_operand_index = operand_map.AddScalarInt32(stride_width);
  CHECK_OP_IDX_AND_RETURN_ERROR(stride_width_operand_index);
  input_indices.push_back(*stride_width_operand_index);

  int32_t stride_height;
  if (auto status =
          LiteRtGetTransposeConvStrideWOption(op.Get(), &stride_height);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get stride height");
  }
  auto stride_height_operand_index = operand_map.AddScalarInt32(stride_height);
  CHECK_OP_IDX_AND_RETURN_ERROR(stride_height_operand_index);
  input_indices.push_back(*stride_height_operand_index);

  auto fuse_code_operand_index =
      operand_map.AddScalarInt32(/*NEURON_FUSED_NONE*/ 0);
  CHECK_OP_IDX_AND_RETURN_ERROR(fuse_code_operand_index);
  input_indices.push_back(*fuse_code_operand_index);

  // Use NHWC format
  auto format_operand_index = operand_map.AddScalarBool(false);
  CHECK_OP_IDX_AND_RETURN_ERROR(format_operand_index);
  input_indices.push_back(*format_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    CHECK_OP_IDX_AND_RETURN_ERROR(id);
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model,
                        /*type=*/NEURON_TRANSPOSE_CONV_2D, input_indices,
                        output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add NEURON_TRANSPOSE_CONV_2D op");
  }

  return {};
}

}  // namespace litert::mediatek
