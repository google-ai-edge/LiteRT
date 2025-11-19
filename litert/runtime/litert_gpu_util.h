// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_GPU_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_GPU_UTIL_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/shape.h"

namespace litert::internal {
using tflite::gpu::BHWC;
using tflite::gpu::DataType;

inline absl::Status ConvertLiteRtTensorTypeToGpuShape(
    const LiteRtRankedTensorType* tensor_type, BHWC* shape) {
  switch (tensor_type->layout.rank) {
    case 0:
      *shape = BHWC(1, 1, 1, 1);
      break;
    case 1:
      *shape = BHWC(tensor_type->layout.dimensions[0], 1, 1, 1);
      break;
    case 2:
      *shape = BHWC(tensor_type->layout.dimensions[0], 1, 1,
                    tensor_type->layout.dimensions[1]);
      break;
    case 3:
      *shape = BHWC(tensor_type->layout.dimensions[0], 1,
                    tensor_type->layout.dimensions[1],
                    tensor_type->layout.dimensions[2]);
      break;
    case 4:
      *shape = BHWC(
          tensor_type->layout.dimensions[0], tensor_type->layout.dimensions[1],
          tensor_type->layout.dimensions[2], tensor_type->layout.dimensions[3]);
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Rank ", tensor_type->layout.rank, " tensor is not supported."));
  }
  return absl::OkStatus();
}

inline bool IsFloat16BufferType(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeMetalBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16;
}

inline absl::Status ConvertLiteRtDataTypeToGpuDataType(
    const LiteRtRankedTensorType* tensor_type, DataType* data_type,
    const LiteRtTensorBufferType buffer_type) {
  switch (tensor_type->element_type) {
    case kLiteRtElementTypeFloat32:
      *data_type = IsFloat16BufferType(buffer_type) ? DataType::FLOAT16
                                                    : DataType::FLOAT32;
      break;
    case kLiteRtElementTypeBool:
      *data_type = DataType::BOOL;
      break;
    case kLiteRtElementTypeInt32:
      *data_type = DataType::INT32;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported element type: %d", tensor_type->element_type));
  }
  return absl::OkStatus();
}
}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_GPU_UTIL_H_
