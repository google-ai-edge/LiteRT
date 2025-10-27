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

#include "litert/vendors/mediatek/compiler/legalizations/neuron_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<NeuronTensorType> GetNeuronTensorType(const Tensor& t,
                                               int32_t tensor_flags) {
  auto element_type = t.ElementType();

  const bool use_int8_asymm_signed =
      tensor_flags & NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
  const bool use_invalid_tensor_type =
      tensor_flags & NN_TENSOR_FLAG_USE_INVALID_TENSOR_TYPE;

  int32_t mtk_type = -1;
  switch (element_type) {
    case ElementType::Float32:
      mtk_type = NEURON_TENSOR_FLOAT32;
      break;
    case ElementType::Float16:
      mtk_type = NEURON_TENSOR_FLOAT16;
      break;
    case ElementType::Int32:
      mtk_type = NEURON_TENSOR_INT32;
      break;
    case ElementType::Int16:
      if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
        mtk_type = NEURON_TENSOR_QUANT16_SYMM;
      } else {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Int16 is not supported.");
      }
      break;
    case ElementType::UInt8:
      mtk_type = NEURON_TENSOR_QUANT8_ASYMM;
      break;
    case ElementType::Int8:
      if (use_int8_asymm_signed) {
        mtk_type = NEURON_TENSOR_QUANT8_ASYMM_SIGNED;
      } else if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
        mtk_type = NEURON_TENSOR_QUANT8_SYMM;
      } else if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
        mtk_type = NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL;
      } else {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Int8 is not supported.");
      }
      break;
    case ElementType::Int4:
      if (t.QTypeId() == kLiteRtQuantizationPerTensor) {
        mtk_type = NEURON_EXT_TENSOR_QUANT4_SYMM;
      } else if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
        mtk_type = NEURON_EXT_TENSOR_QUANT4_SYMM_PER_CHANNEL;
      } else {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "Int4 is not supported.");
      }
      break;
    case ElementType::Bool:
      mtk_type = NEURON_TENSOR_BOOL8;
      break;
    case ElementType::Int64:
      if (t.HasWeights()) {
        if (t.QTypeId() == kLiteRtQuantizationPerTensor ||
            t.QTypeId() == kLiteRtQuantizationNone) {
          mtk_type = NEURON_TENSOR_INT32;
        } else if (t.QTypeId() == kLiteRtQuantizationPerChannel) {
          mtk_type = NEURON_EXT_TENSOR_INT32_SYMM_PER_CHANNEL;
        } else {
          return Error(kLiteRtStatusErrorRuntimeFailure,
                       "Int64 is not supported.");
        }
        LITERT_LOG(LITERT_WARNING,
                   "Currently force casting int64 to int32 on constant.");
      }
      break;
    default:
      break;
  }
  // Currently use TQ8AS as invalid tensor type
  if (mtk_type == -1) {
    if (use_invalid_tensor_type) {
      mtk_type = NEURON_TENSOR_QUANT8_ASYMM_SIGNED;
    } else {
      return Error(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat("Unsupported element type: %d", element_type));
    }
  }
  return mtk_type;
}

Expected<uint32_t> GetNeuronDataSize(NeuronTensorType type) {
  switch (type) {
    case NEURON_FLOAT32:
    case NEURON_TENSOR_FLOAT32:
    case NEURON_INT32:
    case NEURON_TENSOR_INT32:
      return 4;
    case NEURON_FLOAT16:
    case NEURON_TENSOR_FLOAT16:
    case NEURON_EXT_TENSOR_QUANT16_ASYMM_SIGNED:
      return 2;
    case NEURON_BOOL:
    case NEURON_TENSOR_BOOL8:
    case NEURON_TENSOR_QUANT8_ASYMM:
    case NEURON_TENSOR_QUANT8_ASYMM_SIGNED:
      return 1;
    default:
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Get Data Size fail for Neuron Type");
  }
  return Error(kLiteRtStatusErrorRuntimeFailure, "Unexpected neuron type");
}

Expected<bool> IsQuantizedType(NeuronTensorType type) {
  switch (type) {
    case NEURON_TENSOR_QUANT16_SYMM:
    case NEURON_TENSOR_QUANT16_ASYMM:
    case NEURON_TENSOR_QUANT8_ASYMM:
    case NEURON_TENSOR_QUANT8_ASYMM_SIGNED:
      return true;
  }
  return false;
}

NeuronReturnCode ModelAddOperation(const NeuronAdapterApi& api,
                                   NeuronModel* model, NeuronOperationType type,
                                   std::vector<uint32_t> input,
                                   std::vector<uint32_t> output) {
  return api.api().model_add_operation(model, type, input.size(), input.data(),
                                       output.size(), output.data());
};

/*
 * The format of data will be:
 *  -------------------------------------------------------------------------------
 *  | 1 byte typeLen  | N bytes type     | 4 bytes dataLen  | N bytes data |
 *  -------------------------------------------------------------------------------
 */
int EncodeOperandValue(OemOperandValue* operand, uint8_t* output) {
  size_t currPos = 0;

  // 1 byte for typeLen, 4 bytes for bufferLen
  if (output == nullptr) {
    return -1;
  }

  // Set length of type
  *output = operand->typeLen;
  currPos += sizeof(uint8_t);

  // Copy type to output
  memcpy(output + currPos, operand->type, operand->typeLen);
  currPos += operand->typeLen;

  // Set the length of buffer
  memcpy(output + currPos, &(operand->dataLen), sizeof(uint32_t));
  currPos += sizeof(uint32_t);

  // Copy operand value to output
  memcpy(&output[currPos], operand->data, operand->dataLen);

  return 0;
}

size_t PackOemScalarString(const char* str, uint8_t** out_buffer) {
  if (str == nullptr) {
    return 0;
  }
  size_t out_len = 0;
  uint8_t type[] = {'s', 't', 'r', 'i', 'n', 'g'};
  OemOperandValue operand_value;

  operand_value.typeLen = sizeof(type);
  operand_value.type = type;
  operand_value.dataLen = strlen(str);
  if (operand_value.dataLen > MAX_OEM_OP_STRING_LEN) {
    return 0;
  }
  operand_value.data =
      reinterpret_cast<uint8_t*>(malloc(operand_value.dataLen));
  if (operand_value.data == nullptr) {
    return 0;
  }
  memcpy(operand_value.data, str, operand_value.dataLen);

  out_len =
      operand_value.typeLen + operand_value.dataLen + (sizeof(size_t) * 2);
  *out_buffer = reinterpret_cast<uint8_t*>(calloc(out_len, sizeof(uint8_t)));
  if (*out_buffer == nullptr) {
    free(operand_value.data);
    return 0;
  }
  EncodeOperandValue(&operand_value, *out_buffer);
  free(operand_value.data);

  return out_len;
}

Expected<void> UnpackDenseInt4IntoInt8(const int8_t* src_buffer,
                                       int num_elements, int8_t* dst_buffer) {
  // num_elements means the number of elements regardless of packed or
  // For example, 3 elements means both
  //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
  //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
  //   2) Unpacked: 3 int8's = 3 bytes.
  //.     stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j =
  for (int i = 0; i < num_elements / 2; i++) {
    int8_t byte = src_buffer[i];
    // Shift left first so that sign is properly extended when shifted
    int8_t lower = static_cast<int8_t>(byte << 4) >> 4;
    int8_t higher = byte >> 4;
    dst_buffer[2 * i] = lower;
    dst_buffer[2 * i + 1] = higher;
  }

  // If the buffer size is odd, extract the final lower nibble.
  if (num_elements % 2 != 0) {
    dst_buffer[num_elements - 1] =
        static_cast<int8_t>(src_buffer[num_elements / 2] << 4) >> 4;
  }
  return {};
}

Expected<void> CastInt64IntoInt32(const int64_t* src_buffer, int num_elements,
                                  int32_t* dst_buffer) {
  for (int i = 0; i < num_elements; ++i) {
    int64_t value = src_buffer[i];
    // Check if the value exceeds the int32_t range
    if (value > std::numeric_limits<int32_t>::max() ||
        value < std::numeric_limits<int32_t>::min()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "CastInt64IntoInt32: value out of int32_t range.");
    } else {
      dst_buffer[i] = static_cast<int32_t>(value);
    }
  }
  return {};
}

}  // namespace litert::mediatek
