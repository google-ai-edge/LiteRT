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

#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_utils.h"

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace litert::ml_drift {

absl::StatusOr<::ml_drift::TensorDescriptor> CreateTensorDescriptor(
    const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type) {
  ::ml_drift::BHWC shape;
  switch (tensor_type.layout.rank) {
    case 0:
      shape = ::ml_drift::BHWC(1, 1, 1, 1);
      break;
    case 1:
      shape = ::ml_drift::BHWC(tensor_type.layout.dimensions[0], 1, 1, 1);
      break;
    case 2:
      shape = ::ml_drift::BHWC(tensor_type.layout.dimensions[0], 1, 1,
                               tensor_type.layout.dimensions[1]);
      break;
    case 3:
      shape = ::ml_drift::BHWC(tensor_type.layout.dimensions[0], 1,
                               tensor_type.layout.dimensions[1],
                               tensor_type.layout.dimensions[2]);
      break;
    case 4:
      shape = ::ml_drift::BHWC(
          tensor_type.layout.dimensions[0], tensor_type.layout.dimensions[1],
          tensor_type.layout.dimensions[2], tensor_type.layout.dimensions[3]);
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Rank ", tensor_type.layout.rank, " tensor is not supported."));
  }

  ::ml_drift::DataType data_type;
  switch (tensor_type.element_type) {
    case kLiteRtElementTypeFloat32:
      data_type = IsGpuFloat16Memory(buffer_type)
                      ? ::ml_drift::DataType::FLOAT16
                      : ::ml_drift::DataType::FLOAT32;
      break;
    case kLiteRtElementTypeBool:
      data_type = ::ml_drift::DataType::BOOL;
      break;
    case kLiteRtElementTypeInt32:
      data_type = ::ml_drift::DataType::INT32;
      break;
    case kLiteRtElementTypeFloat16:
      data_type = ::ml_drift::DataType::FLOAT16;
      break;
    case kLiteRtElementTypeInt8:
      data_type = ::ml_drift::DataType::INT8;
      break;
    case kLiteRtElementTypeUInt8:
      data_type = ::ml_drift::DataType::UINT8;
      break;
    case kLiteRtElementTypeUInt32:
      data_type = ::ml_drift::DataType::UINT32;
      break;
    case kLiteRtElementTypeInt64:
      data_type = ::ml_drift::DataType::INT64;
      break;
    case kLiteRtElementTypeUInt64:
      data_type = ::ml_drift::DataType::UINT64;
      break;
    case kLiteRtElementTypeInt16:
      data_type = ::ml_drift::DataType::INT16;
      break;
    case kLiteRtElementTypeUInt16:
      data_type = ::ml_drift::DataType::UINT16;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported element type: ", tensor_type.element_type));
  }

  ::ml_drift::TensorStorageType storage_type;
  if (IsGpuBuffer(buffer_type)) {
    storage_type = ::ml_drift::TensorStorageType::BUFFER;
  } else if (IsGpuTexture(buffer_type)) {
    storage_type = ::ml_drift::TensorStorageType::TEXTURE_2D;
  } else if (IsGpuImageBuffer(buffer_type)) {
    storage_type = ::ml_drift::TensorStorageType::IMAGE_BUFFER;
  } else {
    return absl::InvalidArgumentError("Unsupported buffer type.");
  }

  if (shape.b == 1) {
    return ::ml_drift::CreateHwcTensorDescriptor(
        data_type, storage_type, ::ml_drift::HWC(shape.h, shape.w, shape.c));
  }
  return ::ml_drift::CreateBhwcTensorDescriptor(
      data_type, storage_type,
      ::ml_drift::BHWC(shape.b, shape.h, shape.w, shape.c));
}

void ConvertDataToDescriptor(void* host_memory,
                             ::ml_drift::TensorDescriptor& desc,
                             LiteRtElementType src_type) {
  if (src_type == kLiteRtElementTypeBool) {
    desc.UploadData(reinterpret_cast<bool*>(host_memory));
  } else if (src_type == kLiteRtElementTypeInt32) {
    desc.UploadData(reinterpret_cast<int32_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeFloat16) {
    desc.UploadData(reinterpret_cast<::ml_drift::half*>(host_memory));
  } else if (src_type == kLiteRtElementTypeInt8) {
    desc.UploadData(reinterpret_cast<int8_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeUInt8) {
    desc.UploadData(reinterpret_cast<uint8_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeUInt32) {
    desc.UploadData(reinterpret_cast<uint32_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeInt64) {
    desc.UploadData(reinterpret_cast<int64_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeUInt64) {
    desc.UploadData(reinterpret_cast<uint64_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeInt16) {
    desc.UploadData(reinterpret_cast<int16_t*>(host_memory));
  } else if (src_type == kLiteRtElementTypeUInt16) {
    desc.UploadData(reinterpret_cast<uint16_t*>(host_memory));
  } else {
    desc.UploadData(reinterpret_cast<float*>(host_memory));
  }
}

void ConvertDataFromDescriptor(const ::ml_drift::TensorDescriptor& desc,
                               void* host_memory, LiteRtElementType dst_type) {
  if (dst_type == kLiteRtElementTypeBool) {
    desc.DownloadData(reinterpret_cast<bool*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeInt32) {
    desc.DownloadData(reinterpret_cast<int32_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeFloat16) {
    desc.DownloadData(reinterpret_cast<::ml_drift::half*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeInt8) {
    desc.DownloadData(reinterpret_cast<int8_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeUInt8) {
    desc.DownloadData(reinterpret_cast<uint8_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeUInt32) {
    desc.DownloadData(reinterpret_cast<uint32_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeInt64) {
    desc.DownloadData(reinterpret_cast<int64_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeUInt64) {
    desc.DownloadData(reinterpret_cast<uint64_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeInt16) {
    desc.DownloadData(reinterpret_cast<int16_t*>(host_memory));
  } else if (dst_type == kLiteRtElementTypeUInt16) {
    desc.DownloadData(reinterpret_cast<uint16_t*>(host_memory));
  } else {
    desc.DownloadData(reinterpret_cast<float*>(host_memory));
  }
}

}  // namespace litert::ml_drift
