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

#include "litert/vendors/mediatek/compiler/legalizations/layer_norm_op_legalization.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

namespace {

absl::Span<const int32_t> GetDimensions(const litert::compiler::Tensor& op) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, op.RankedTensorType());
  return tensor_type.Layout().Dimensions();
}

size_t GetRank(const litert::compiler::Tensor& op) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, op.RankedTensorType());
  return tensor_type.Layout().Rank();
}

}  // namespace

Expected<void> LegalizeLayerNormOp(const NeuronAdapterApi& neuron_adapter_api,
                                   NeuronModel* model, OperandMap& operand_map,
                                   const litert::compiler::Op& op,
                                   float epsilon) {
  LITERT_LOG(LITERT_INFO, "Legalize Layer Norm");
  std::vector<uint32_t> input_indices;

  // The first input is the input data
  LITERT_ASSIGN_OR_RETURN(auto input_tensor_id,
                          operand_map.GetOperandIndex(op.Inputs()[0]));
  input_indices.push_back(input_tensor_id);

  // Axis: The default axis of layer norm is the last dimension
  int32_t axis_value = static_cast<int32_t>(GetRank(op.Inputs()[0])) - 1;
  auto info =
      litert::compiler::GetOptionsAs<litert::compiler::CompositeOptions>(
          op.ctx(), op.Get());
  if (info.HasValue() && info->attributes_map.has_value()) {
    auto attributes_map = info->attributes_map.value();
    if (!attributes_map["channel_axis"].IsNull()) {
      int64_t channel_axis = attributes_map["channel_axis"].AsInt64();
      int32_t rank = static_cast<int32_t>(GetRank(op.Inputs()[0]));
      axis_value = channel_axis < 0
                       ? (rank + static_cast<int32_t>(channel_axis))
                       : static_cast<int32_t>(channel_axis);
    }
    if (!attributes_map["epsilon"].IsNull()) {
      epsilon = attributes_map["epsilon"].AsFloat();
    }
  }

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

  // Beta: The third input if available, else 0
  if (op.Inputs().size() > 2) {
    LITERT_ASSIGN_OR_RETURN(auto beta_tensor_id,
                            operand_map.GetOperandIndex(op.Inputs()[2]));
    input_indices.push_back(beta_tensor_id);
  } else {
    std::vector<uint32_t> beta_shape = {
        static_cast<uint32_t>(GetDimensions(op.Inputs()[1])[0])};
    int32_t beta_bytes = sizeof(float) * beta_shape[0];
    LITERT_ASSIGN_OR_RETURN(auto beta_extra_data_idx,
                            operand_map.RegisterExtraData(beta_bytes));
    memset(operand_map.GetExtraData(beta_extra_data_idx), 0, beta_bytes);
    LITERT_ASSIGN_OR_RETURN(
        auto beta_tensor_id,
        operand_map.AddTensorByType(
            NEURON_TENSOR_FLOAT32, beta_shape,
            operand_map.GetExtraData(beta_extra_data_idx), beta_bytes));
    input_indices.push_back(beta_tensor_id);
  }

  // Epsilon
  LITERT_ASSIGN_OR_RETURN(auto epsilon_tensor_id,
                          operand_map.AddScalarFloat32(epsilon));
  input_indices.push_back(epsilon_tensor_id);

  const char* custom_name = "MTKEXT_LAYER_NORMALIZATION";
  int32_t raw_op_type = 0;
  auto custom_name_operand_index = operand_map.AddOemExtensionOperand(
      custom_name, reinterpret_cast<NeuronOperationType*>(&raw_op_type));
  if (!custom_name_operand_index) {
    return custom_name_operand_index.Error();
  }
  input_indices.push_back(*custom_name_operand_index);

  std::vector<uint32_t> output_indices;
  auto outputs = op.Outputs();
  for (auto& output : outputs) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  auto add_op_ret = neuron_adapter_api.api().model_add_operation(
      model, static_cast<NeuronOperationType>(raw_op_type),
      input_indices.size(), input_indices.data(), output_indices.size(),
      output_indices.data());
  if (add_op_ret != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add MTKEXT_LAYER_NORMALIZATION op");
  }

  return {};
}

}  // namespace litert::mediatek
