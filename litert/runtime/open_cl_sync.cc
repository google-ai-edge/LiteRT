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
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/gpu_environment.h"
#include "litert/runtime/litert_gpu_util.h"
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_memory.h"
#include "tflite/delegates/gpu/cl/tensor.h"
#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/shape.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/tensor.h"
#include "tflite/delegates/gpu/common/types.h"

#if LITERT_HAS_OPENCL_SUPPORT

using tflite::gpu::BHWC;
using tflite::gpu::CreateBhwcTensorDescriptor;
using tflite::gpu::CreateHwcTensorDescriptor;
using tflite::gpu::DataType;
using tflite::gpu::HWC;
using tflite::gpu::TensorDescriptor;
using tflite::gpu::TensorStorageType;
using TensorBool = tflite::gpu::Tensor<BHWC, DataType::BOOL>;
using TensorFloat16 = tflite::gpu::Tensor<BHWC, DataType::FLOAT16>;
using TensorFloat32 = tflite::gpu::Tensor<BHWC, DataType::FLOAT32>;
using TensorInt32 = tflite::gpu::Tensor<BHWC, DataType::INT32>;

namespace litert::internal {
// TODO(b/431308296): Clean up the GPU memory sync logic to make it generic for
// all GPU backends.
absl::StatusOr<TensorDescriptor> CreateTensorDescriptor(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) {
  BHWC shape;
  LITERT_RETURN_IF_ERROR(
      ConvertLiteRtTensorTypeToGpuShape(tensor_type, &shape).ok());

  DataType data_type;
  LITERT_RETURN_IF_ERROR(
      ConvertLiteRtDataTypeToGpuDataType(tensor_type, &data_type, buffer_type)
          .ok());

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
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
      storage_type = TensorStorageType::IMAGE_BUFFER;
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

LiteRtStatus LiteRtGpuMemoryCreate(GpuEnvironment* gpu_env,
                                   const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, cl_mem* cl_memory) {
  auto tensor_desc = CreateTensorDescriptor(tensor_type, buffer_type);
  LITERT_RETURN_IF_ERROR(tensor_desc.status().ok(),
                         kLiteRtStatusErrorUnsupported);

  tflite::gpu::cl::CLMemory tensor_memory;

  LITERT_RETURN_IF_ERROR(
      tflite::gpu::cl::AllocateTensorMemory(*gpu_env->GetContext(),
                                            *tensor_desc, &tensor_memory)
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

LiteRtStatus LiteRtGpuMemoryUpload(GpuEnvironment* gpu_env,
                                   const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, const void* ptr,
                                   cl_mem cl_memory) {
  auto tensor_desc = CreateTensorDescriptor(tensor_type, buffer_type);
  LITERT_RETURN_IF_ERROR(tensor_desc.status().ok(),
                         kLiteRtStatusErrorUnsupported);

  auto cl_tensor = std::make_unique<tflite::gpu::cl::Tensor>();
  LITERT_RETURN_IF_ERROR(
      tflite::gpu::cl::CreateTensorShared(*gpu_env->GetContext(), cl_memory,
                                          *tensor_desc, cl_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::BOOL) {
    return LiteRtGpuMemoryUploadImpl<TensorBool, bool>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::INT32) {
    return LiteRtGpuMemoryUploadImpl<TensorInt32, int32_t>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::FLOAT16) {
    if (tensor_type->element_type == kLiteRtElementTypeFloat32) {
      return LiteRtGpuMemoryUploadImpl<TensorFloat32, float>(
          *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
    }
    return LiteRtGpuMemoryUploadImpl<TensorFloat16, tflite::gpu::half>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else {
    return LiteRtGpuMemoryUploadImpl<TensorFloat32, float>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
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

LiteRtStatus LiteRtGpuMemoryDownload(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType* tensor_type,
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
  LITERT_RETURN_IF_ERROR(
      tflite::gpu::cl::CreateTensorShared(*gpu_env->GetContext(), cl_memory,
                                          *tensor_desc, cl_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::BOOL) {
    return LiteRtGpuMemoryDownloadImpl<TensorBool, bool>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::INT32) {
    return LiteRtGpuMemoryDownloadImpl<TensorInt32, int32_t>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::FLOAT16) {
    if (tensor_type->element_type == kLiteRtElementTypeFloat32) {
      return LiteRtGpuMemoryDownloadImpl<TensorFloat32, float>(
          *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
    }
    return LiteRtGpuMemoryDownloadImpl<TensorFloat16, tflite::gpu::half>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else {
    return LiteRtGpuMemoryDownloadImpl<TensorFloat32, float>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // LITERT_HAS_OPENCL_SUPPORT
