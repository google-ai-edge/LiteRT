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

#include "litert/runtime/open_cl_sync.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/gpu_environment.h"
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_memory.h"
#include "tflite/delegates/gpu/cl/tensor.h"
#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/shape.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/tensor.h"

using tflite::gpu::BHWC;
using tflite::gpu::CreateBhwcTensorDescriptor;
using tflite::gpu::CreateHwcTensorDescriptor;
using tflite::gpu::DataType;
using tflite::gpu::HWC;
using tflite::gpu::TensorDescriptor;
using tflite::gpu::TensorStorageType;
using TensorBool = tflite::gpu::Tensor<BHWC, DataType::BOOL>;
using TensorFloat32 = tflite::gpu::Tensor<BHWC, DataType::FLOAT32>;
using TensorInt32 = tflite::gpu::Tensor<BHWC, DataType::INT32>;

namespace litert::internal {

absl::StatusOr<TensorDescriptor> CreateTensorDescriptor(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) {
  BHWC shape;
  switch (tensor_type->layout.rank) {
    case 0:
      shape = BHWC(1, 1, 1, 1);
      break;
    case 1:
      shape = BHWC(tensor_type->layout.dimensions[0], 1, 1, 1);
      break;
    case 2:
      shape = BHWC(tensor_type->layout.dimensions[0], 1, 1,
                   tensor_type->layout.dimensions[1]);
      break;
    case 3:
      shape = BHWC(tensor_type->layout.dimensions[0], 1,
                   tensor_type->layout.dimensions[1],
                   tensor_type->layout.dimensions[2]);
      break;
    case 4:
      shape = BHWC(
          tensor_type->layout.dimensions[0], tensor_type->layout.dimensions[1],
          tensor_type->layout.dimensions[2], tensor_type->layout.dimensions[3]);
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Rank ", tensor_type->layout.rank, " tensor is not supported."));
  }

  DataType data_type;
  switch (tensor_type->element_type) {
    case kLiteRtElementTypeFloat32:
      data_type = (buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
                   buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16)
                      ? DataType::FLOAT16
                      : DataType::FLOAT32;
      break;
    case kLiteRtElementTypeBool:
      data_type = DataType::BOOL;
      break;
    case kLiteRtElementTypeInt32:
      data_type = DataType::INT32;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported element type: ", tensor_type->element_type));
  }

  TensorStorageType storage_type;
  switch (buffer_type) {
    case kLiteRtTensorBufferTypeOpenClBuffer:
    case kLiteRtTensorBufferTypeOpenClBufferFp16:
      storage_type = TensorStorageType::BUFFER;
      break;
    case kLiteRtTensorBufferTypeOpenClTexture:
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
      storage_type = TensorStorageType::TEXTURE_2D;
      break;
    default:
      return absl::InvalidArgumentError("Unsupported buffer type.");
  }

  if (shape.b == 1) {
    return CreateHwcTensorDescriptor(data_type, storage_type,
                                     HWC(shape.h, shape.w, shape.c));
  }
  return CreateBhwcTensorDescriptor(data_type, storage_type,
                                    BHWC(shape.b, shape.h, shape.w, shape.c));
}

LiteRtStatus LiteRtGpuMemoryCreate(const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, cl_mem* cl_memory) {
  auto tensor_desc = CreateTensorDescriptor(tensor_type, buffer_type);
  LITERT_RETURN_IF_ERROR(tensor_desc.status().ok(),
                         kLiteRtStatusErrorUnsupported);

  LITERT_ASSIGN_OR_RETURN(auto cl_env, GpuEnvironmentSingleton::GetInstance());

  tflite::gpu::cl::CLMemory tensor_memory;

  LITERT_RETURN_IF_ERROR(
      tflite::gpu::cl::AllocateTensorMemory(*cl_env->getContext(), *tensor_desc,
                                            &tensor_memory)
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  // TODO: Check bytes size.
  *cl_memory = tensor_memory.Release();
  return kLiteRtStatusOk;
}

template <typename TensorT, typename DataTypeT>
LiteRtStatus LiteRtGpuMemoryUploadImpl(tflite::gpu::cl::Tensor& cl_tensor,
                                       size_t bytes, const void* ptr,
                                       tflite::gpu::cl::CLCommandQueue* queue) {
  TensorT src_tensor;
  src_tensor.shape = BHWC(cl_tensor.Batch(), cl_tensor.Height(),
                          cl_tensor.Width(), cl_tensor.Channels());
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  if (src_tensor.data.size() * sizeof(DataTypeT) != bytes) {
    LITERT_LOG(LITERT_ERROR,
               "Upload buffer size mismatch: required: %zu vs given: %zu",
               src_tensor.data.size() * sizeof(DataTypeT), bytes);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  // TODO - b/413431454: Try to avoid the copy.
  std::memcpy(src_tensor.data.data(), ptr, bytes);

  TensorDescriptor descriptor_with_data = cl_tensor.GetDescriptor();
  descriptor_with_data.UploadData(src_tensor);
  LITERT_RETURN_IF_ERROR(
      cl_tensor.UploadDescriptorData(descriptor_with_data, queue).ok(),
      kLiteRtStatusErrorRuntimeFailure);
  return kLiteRtStatusOk;
};

LiteRtStatus LiteRtGpuMemoryUpload(const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, const void* ptr,
                                   cl_mem cl_memory) {
  auto tensor_desc = CreateTensorDescriptor(tensor_type, buffer_type);
  LITERT_RETURN_IF_ERROR(tensor_desc.status().ok(),
                         kLiteRtStatusErrorUnsupported);

  auto cl_tensor = std::make_unique<tflite::gpu::cl::Tensor>();
  LITERT_ASSIGN_OR_RETURN(auto cl_env, GpuEnvironmentSingleton::GetInstance());
  LITERT_RETURN_IF_ERROR(
      tflite::gpu::cl::CreateTensorShared(*cl_env->getContext(), cl_memory,
                                          *tensor_desc, cl_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::BOOL) {
    return LiteRtGpuMemoryUploadImpl<TensorBool, bool>(
        *cl_tensor, bytes, ptr, cl_env->getCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::INT32) {
    return LiteRtGpuMemoryUploadImpl<TensorInt32, int32_t>(
        *cl_tensor, bytes, ptr, cl_env->getCommandQueue());
  } else {
    return LiteRtGpuMemoryUploadImpl<TensorFloat32, float>(
        *cl_tensor, bytes, ptr, cl_env->getCommandQueue());
  }

  return kLiteRtStatusOk;
}

template <typename TensorT, typename DataTypeT>
LiteRtStatus LiteRtGpuMemoryDownloadImpl(
    tflite::gpu::cl::Tensor& cl_tensor, size_t bytes, void* ptr,
    tflite::gpu::cl::CLCommandQueue* queue) {
  TensorT dst_tensor;
  const BHWC shape = BHWC(cl_tensor.Batch(), cl_tensor.Height(),
                          cl_tensor.Width(), cl_tensor.Channels());
  dst_tensor.shape = shape;
  dst_tensor.data.resize(dst_tensor.shape.DimensionsProduct());
  TensorDescriptor desc;
  LITERT_RETURN_IF_ERROR(cl_tensor.ToDescriptor(&desc, queue).ok(),
                         kLiteRtStatusErrorRuntimeFailure);
  desc.DownloadData(&dst_tensor);
  if (dst_tensor.data.size() * sizeof(DataTypeT) != bytes) {
    LITERT_LOG(LITERT_ERROR,
               "Download buffer size mismatch: required: %zu vs given: %zu",
               dst_tensor.data.size() * sizeof(DataTypeT), bytes);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  std::memcpy(ptr, dst_tensor.data.data(), bytes);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGpuMemoryDownload(const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes, cl_mem cl_memory,
                                     void* ptr) {
  auto tensor_desc = CreateTensorDescriptor(tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create tensor descriptor: %s",
               tensor_desc.status().message().data());
    return kLiteRtStatusErrorUnsupported;
  }

  auto cl_tensor = std::make_unique<tflite::gpu::cl::Tensor>();
  LITERT_ASSIGN_OR_RETURN(auto cl_env, GpuEnvironmentSingleton::GetInstance());
  LITERT_RETURN_IF_ERROR(
      tflite::gpu::cl::CreateTensorShared(*cl_env->getContext(), cl_memory,
                                          *tensor_desc, cl_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::BOOL) {
    return LiteRtGpuMemoryDownloadImpl<TensorBool, bool>(
        *cl_tensor, bytes, ptr, cl_env->getCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::INT32) {
    return LiteRtGpuMemoryDownloadImpl<TensorInt32, int32_t>(
        *cl_tensor, bytes, ptr, cl_env->getCommandQueue());
  } else {
    return LiteRtGpuMemoryDownloadImpl<TensorFloat32, float>(
        *cl_tensor, bytes, ptr, cl_env->getCommandQueue());
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal
