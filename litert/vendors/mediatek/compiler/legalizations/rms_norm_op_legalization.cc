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
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

namespace {

absl::Span<const int32_t> GetDimensions(const litert::compiler::Tensor& op) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, op.RankedTensorType());
  return tensor_type.Layout().Dimensions();
}

inline ElementType GetElementType(const litert::compiler::Tensor& tensor) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, tensor.RankedTensorType());
  return tensor_type.ElementType();
}

size_t GetRank(const litert::compiler::Tensor& op) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, op.RankedTensorType());
  return tensor_type.Layout().Rank();
}

}  // namespace

Expected<void> LegalizeRmsNormOp(const NeuronAdapterApi& neuron_adapter_api,
                                 NeuronModel* model, OperandMap& operand_map,
                                 const litert::compiler::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize RMS Norm");
  std::vector<uint32_t> input_indices;

  auto input_tensor = op.Inputs()[0];
  auto gamma_tensor = op.Inputs()[1];
  auto output_tensor = op.Outputs()[0];

  LITERT_ASSIGN_OR_RETURN(auto input_neuron_type,
                          GetNeuronTensorType(input_tensor));
  LITERT_ASSIGN_OR_RETURN(bool is_input_quantized,
                          IsQuantizedType(input_neuron_type));

  uint32_t input_norm_id;
  if (is_input_quantized) {
    LITERT_ASSIGN_OR_RETURN(auto input_quant_id,
                            operand_map.GetOperandIndex(input_tensor));
    std::vector<uint32_t> shape(GetDimensions(input_tensor).begin(),
                                GetDimensions(input_tensor).end());
    LITERT_ASSIGN_OR_RETURN(
        input_norm_id,
        operand_map.AddTensorByType(NEURON_TENSOR_FLOAT32, shape, nullptr, 0));
    if (ModelAddOperation(neuron_adapter_api, model, NEURON_DEQUANTIZE,
                          {input_quant_id}, {input_norm_id}) !=
        NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to add NEURON_DEQUANTIZE for input in RMS Norm");
    }
  } else {
    LITERT_ASSIGN_OR_RETURN(input_norm_id,
                            operand_map.GetOperandIndex(input_tensor));
  }
  input_indices.push_back(input_norm_id);

  // Axis: The default axis of rms norm is the last dimension
  int32_t axis_value = GetRank(input_tensor) - 1;
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
  LITERT_ASSIGN_OR_RETURN(auto gamma_neuron_type,
                          GetNeuronTensorType(gamma_tensor));
  LITERT_ASSIGN_OR_RETURN(bool is_gamma_quantized,
                          IsQuantizedType(gamma_neuron_type));

  uint32_t gamma_norm_id;
  if (is_gamma_quantized) {
    LITERT_ASSIGN_OR_RETURN(auto gamma_quant_id,
                            operand_map.GetOperandIndex(gamma_tensor));
    std::vector<uint32_t> gamma_shape(GetDimensions(gamma_tensor).begin(),
                                      GetDimensions(gamma_tensor).end());
    LITERT_ASSIGN_OR_RETURN(
        gamma_norm_id, operand_map.AddTensorByType(
                           NEURON_TENSOR_FLOAT32, gamma_shape, nullptr, 0));
    if (ModelAddOperation(neuron_adapter_api, model, NEURON_DEQUANTIZE,
                          {gamma_quant_id}, {gamma_norm_id}) !=
        NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to add NEURON_DEQUANTIZE for gamma in RMS Norm");
    }
  } else {
    LITERT_ASSIGN_OR_RETURN(gamma_norm_id,
                            operand_map.GetOperandIndex(gamma_tensor));
  }
  input_indices.push_back(gamma_norm_id);

  // Beta: Set 0 as default beta (Float32)
  std::vector<uint32_t> beta_shape = {
      static_cast<uint32_t>(GetDimensions(gamma_tensor)[0])};
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

  // Epsilon
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

  LITERT_ASSIGN_OR_RETURN(auto output_neuron_type,
                          GetNeuronTensorType(output_tensor));
  LITERT_ASSIGN_OR_RETURN(bool is_output_quantized,
                          IsQuantizedType(output_neuron_type));

  uint32_t output_norm_id;
  if (is_output_quantized) {
    std::vector<uint32_t> out_shape(GetDimensions(output_tensor).begin(),
                                    GetDimensions(output_tensor).end());
    LITERT_ASSIGN_OR_RETURN(
        output_norm_id, operand_map.AddTensorByType(
                            NEURON_TENSOR_FLOAT32, out_shape, nullptr, 0));
  } else {
    LITERT_ASSIGN_OR_RETURN(output_norm_id,
                            operand_map.GetOperandIndex(output_tensor));
  }

  std::vector<uint32_t> output_indices = {output_norm_id};

  if (ModelAddOperation(neuron_adapter_api, model, /*type=*/nn_op_type,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add MTKEXT_RMS_NORMALIZATION op");
  }

  if (is_output_quantized) {
    LITERT_ASSIGN_OR_RETURN(auto output_quant_id,
                            operand_map.GetOperandIndex(output_tensor));
    if (ModelAddOperation(neuron_adapter_api, model, NEURON_QUANTIZE,
                          {output_norm_id}, {output_quant_id}) !=
        NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to add NEURON_QUANTIZE for output in RMS Norm");
    }
  }

  return {};
}

}  // namespace litert::mediatek
