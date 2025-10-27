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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_NEURON_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_NEURON_UTILS_H_

#include <cstdint>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

#define MAX_OEM_OP_STRING_LEN 100

namespace litert::mediatek {

// Bit mask for tensor flags.
enum {
  NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED = 1U << 2,
  NN_TENSOR_FLAG_USE_INVALID_TENSOR_TYPE = 1U << 5,
};

using NeuronTensorType = int32_t;
using NeuronReturnCode = int32_t;

struct OemOperandValue {
  uint8_t* type;
  uint8_t typeLen;
  uint8_t* data;
  uint32_t dataLen;
};

Expected<NeuronTensorType> GetNeuronTensorType(const Tensor& t,
                                               int32_t tensor_flags = 0);

Expected<uint32_t> GetNeuronDataSize(NeuronTensorType type);

Expected<bool> IsQuantizedType(NeuronTensorType type);

NeuronReturnCode ModelAddOperation(const NeuronAdapterApi& api,
                                   NeuronModel* model, NeuronOperationType type,
                                   std::vector<uint32_t> input,
                                   std::vector<uint32_t> output);

size_t PackOemScalarString(const char* str, uint8_t** out_buffer);

// Unpack or inflate `src_buffer` by taking each element and splitting it as
// two elements into `dst_buffer`.
Expected<void> UnpackDenseInt4IntoInt8(const int8_t* src_buffer,
                                       int num_elements, int8_t* dst_buffer);

Expected<void> CastInt64IntoInt32(const int64_t* src_buffer, int num_elements,
                                  int32_t* dst_buffer);

}  // namespace litert::mediatek

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_NEURON_UTILS_H_
