// Copyright 2024 Google LLC.
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

#include "litert/vendors/mediatek/compiler/legalizations/operand_map.h"

#include <cstdint>
#include <string>

#include "neuron/api/NeuronAdapter.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/compiler/legalizations/neuron_utils.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<uint32_t> OperandMap::Register(const NeuronOperandType& operand_type) {
  if (neuron_adapter_api_.api().model_add_operand(model_, &operand_type) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to register model operand");
  }
  return AllocateOperandIndex();
}

Expected<uint32_t> OperandMap::Register(const Tensor& t, int32_t tensor_flags) {
  auto operand_type = OperandType::Create(t, tensor_flags);
  if (!operand_type) {
    return operand_type.Error();
  }

  auto operand_index =
      Register(static_cast<const NeuronOperandType&>(*operand_type));
  if (!operand_index) {
    return operand_index.Error();
  }
  LITERT_LOG(LITERT_INFO, "\nOperandIndex: %d", operand_index.Value());
  operand_type->Info();

  if (t.HasWeights()) {
    auto weights = t.Weights().Bytes();
    if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
      LITERT_ASSIGN_OR_RETURN(auto quant_param,
                              operand_type->GetPerChannelQuantParams());
      if (neuron_adapter_api_.api().model_set_symm_per_channel_quant_params(
              model_, *operand_index, &quant_param) != NEURON_NO_ERROR) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Failed to set param of per channel quant params");
      }
    }

    LITERT_ASSIGN_OR_RETURN(auto tensor_type, t.RankedTensorType());
    if (tensor_type.ElementType() == ElementType::Int4 ||
        tensor_type.ElementType() == ElementType::Int64) {
      int num_element = static_cast<int>(operand_type->GetElementCount());
      int new_bytes = 0;
      int32_t extra_data_idx = -1;

      if (tensor_type.ElementType() == ElementType::Int4) {
        // Unpack Int4 into Int8
        new_bytes = num_element * sizeof(int8_t);
        LITERT_ASSIGN_OR_RETURN(extra_data_idx, RegisterExtraData(new_bytes));
        LITERT_LOG(LITERT_INFO, "\nUnpack Int4 into Int8, new bytes: %d",
                   new_bytes);
        LITERT_RETURN_IF_ERROR(UnpackDenseInt4IntoInt8(
            reinterpret_cast<const int8_t*>(weights.data()), num_element,
            reinterpret_cast<int8_t*>(GetExtraData(extra_data_idx))));
      } else if (tensor_type.ElementType() == ElementType::Int64) {
        // Cast Int64 into Int32
        new_bytes = num_element * sizeof(int32_t);
        LITERT_ASSIGN_OR_RETURN(extra_data_idx, RegisterExtraData(new_bytes));
        LITERT_LOG(LITERT_INFO, "\nCast Int64 into Int32, new bytes: %d",
                   new_bytes);
        LITERT_RETURN_IF_ERROR(CastInt64IntoInt32(
            reinterpret_cast<const int64_t*>(weights.data()), num_element,
            reinterpret_cast<int32_t*>(GetExtraData(extra_data_idx))));
      } else {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Failed to set value for some tensor type.");
      }

      if (neuron_adapter_api_.api().model_set_operand_value(
              model_, *operand_index, GetExtraData(extra_data_idx),
              new_bytes) != NEURON_NO_ERROR) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Failed to set value of tensor weights for special case: "
                     "int4/int64");
      }
    } else {
      if (neuron_adapter_api_.api().model_set_operand_value(
              model_, *operand_index, weights.data(), weights.size()) !=
          NEURON_NO_ERROR) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Failed to set value of tensor weights");
      }
    }
  }

  map_[t.Get()] = *operand_index;
  return *operand_index;
}

}  // namespace litert::mediatek
