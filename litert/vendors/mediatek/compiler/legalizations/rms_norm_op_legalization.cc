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

#include "litert/vendors/mediatek/compiler/legalizations/rms_norm_op_legalization.h"

#include <cstdint>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

namespace {

absl::Span<const int32_t> GetDimensions(Tensor& op) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, op.RankedTensorType());
  return tensor_type.Layout().Dimensions();
}

inline ElementType GetElementType(const Tensor& tensor) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, tensor.RankedTensorType());
  return tensor_type.ElementType();
}

size_t GetRank(Tensor& op) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, op.RankedTensorType());
  return tensor_type.Layout().Rank();
}

}  // namespace

Expected<void> LegalizeRmsNormOp(const NeuronAdapterApi& neuron_adapter_api,
                                 NeuronModel* model, OperandMap& operand_map,
                                 const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize RMS Norm");
  std::vector<uint32_t> input_indices;

  // The first input is the input data
  LITERT_ASSIGN_OR_RETURN(auto input_tensor_id,
                          operand_map.GetOperandIndex(op.Inputs()[0]));
  input_indices.push_back(input_tensor_id);

  // Axis: The default axis of rms norm is the last dimension
  int32_t axis_value = GetRank(op.Inputs()[0]) - 1;
  std::vector<uint32_t> axis_shape = {1};
  LITERT_ASSIGN_OR_RETURN(auto axis_extra_data_idx,
                          operand_map.RegisterExtraData(sizeof(axis_value)));
  memcpy(operand_map.GetExtraData(axis_extra_data_idx), &axis_value,
         sizeof(axis_value));
  LITERT_ASSIGN_OR_RETURN(
      auto axis_tensor_id,
      operand_map.AddTensorByType(NEURON_TENSOR_INT32, axis_shape,
                                  operand_map.GetExtraData(axis_extra_data_idx),
                                  sizeof(axis_value)));
  input_indices.push_back(axis_tensor_id);

  // Gamma: The second input
  LITERT_ASSIGN_OR_RETURN(auto gamma_tensor_id,
                          operand_map.GetOperandIndex(op.Inputs()[1]));
  input_indices.push_back(gamma_tensor_id);

  // Beta: Set 0 as default beta
  std::vector<uint32_t> beta_shape = {
      static_cast<uint32_t>(GetDimensions(op.Inputs()[1])[0])};
  int32_t beta_bytes = sizeof(float) * beta_shape[0];
  LITERT_ASSIGN_OR_RETURN(auto beta_extra_data_idx,
                          operand_map.RegisterExtraData(beta_bytes));
  memset(operand_map.GetExtraData(beta_extra_data_idx), 0, beta_bytes);
  LITERT_ASSIGN_OR_RETURN(
      auto beta_tensor_id,
      operand_map.AddTensorByType(NEURON_TENSOR_FLOAT32, beta_shape,
                                  operand_map.GetExtraData(beta_extra_data_idx),
                                  beta_bytes));
  input_indices.push_back(beta_tensor_id);

  // Eplison
  float epsilon_value = std::numeric_limits<float>::epsilon();
  LITERT_ASSIGN_OR_RETURN(auto epsilon_tensor_id,
                          operand_map.AddScalarFloat32(epsilon_value));
  input_indices.push_back(epsilon_tensor_id);

  const char* custom_name = "MTKEXT_RMS_NORMALIZATION";
  NeuronOperationType nn_op_type;
  auto custom_name_operand_index =
      operand_map.AddOemExtensionOperand(custom_name, &nn_op_type);
  if (!custom_name_operand_index) {
    return custom_name_operand_index.Error();
  }
  input_indices.push_back(*custom_name_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, /*type=*/nn_op_type,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add MTKEXT_RMS_NORMALIZATION op");
  }

  return {};
}

}  // namespace litert::mediatek
