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
#include <CL/cl.h>

#if LITERT_HAS_OPENCL_SUPPORT

#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_memory.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "litert/runtime/litert_gpu_util.h"

using ::ml_drift::BHWC;
using ::ml_drift::CreateBhwcTensorDescriptor;
using ::ml_drift::CreateHwcTensorDescriptor;
using ::ml_drift::DataType;
using ::ml_drift::HWC;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using TensorBool = ::ml_drift::Tensor<BHWC, DataType::kBool>;
using TensorFloat16 = ::ml_drift::Tensor<BHWC, DataType::kFloat16>;
using TensorFloat32 = ::ml_drift::Tensor<BHWC, DataType::kFloat32>;
using TensorInt32 = ::ml_drift::Tensor<BHWC, DataType::kInt32>;
using TensorInt8 = ::ml_drift::Tensor<BHWC, DataType::kInt8>;

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
      storage_type = TensorStorageType::kBuffer;
      break;
    case kLiteRtTensorBufferTypeOpenClTexture:
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
      storage_type = TensorStorageType::kTexture2D;
      break;
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
      storage_type = TensorStorageType::kImageBuffer;
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
  if (!tensor_desc.ok()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create tensor descriptor: %s",
               tensor_desc.status().message().data());
    return kLiteRtStatusErrorUnsupported;
  }

  ::ml_drift::cl::CLMemory tensor_memory;

  LITERT_RETURN_IF_ERROR(
      ::ml_drift::cl::AllocateTensorMemory(*gpu_env->GetContext(),
                                           *tensor_desc, &tensor_memory)
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  // TODO: Check bytes size.
  *cl_memory = tensor_memory.Release();
  return kLiteRtStatusOk;
}

template <typename TensorT, typename DataTypeT>
LiteRtStatus LiteRtGpuMemoryUploadImpl(::ml_drift::cl::Tensor& cl_tensor,
                                       size_t bytes, const void* ptr,
                                       ::ml_drift::cl::CLCommandQueue* queue) {
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
  if (!tensor_desc.ok()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create tensor descriptor: %s",
               tensor_desc.status().message().data());
    return kLiteRtStatusErrorUnsupported;
  }

  auto cl_tensor = std::make_unique<::ml_drift::cl::Tensor>();
  LITERT_RETURN_IF_ERROR(
      ::ml_drift::cl::CreateTensorShared(*gpu_env->GetContext(), cl_memory,
                                         *tensor_desc, cl_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::kBool) {
    return LiteRtGpuMemoryUploadImpl<TensorBool, bool>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::kInt32) {
    return LiteRtGpuMemoryUploadImpl<TensorInt32, int32_t>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::kFloat16) {
    if (tensor_type->element_type == kLiteRtElementTypeFloat32) {
      return LiteRtGpuMemoryUploadImpl<TensorFloat32, float>(
          *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
    }
    return LiteRtGpuMemoryUploadImpl<TensorFloat16, ::ml_drift::half>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_type->element_type == kLiteRtElementTypeInt8) {
    return LiteRtGpuMemoryUploadImpl<TensorInt8, int8_t>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else {
    return LiteRtGpuMemoryUploadImpl<TensorFloat32, float>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  }

  return kLiteRtStatusOk;
}

template <typename TensorT, typename DataTypeT>
LiteRtStatus LiteRtGpuMemoryDownloadImpl(
    ::ml_drift::cl::Tensor& cl_tensor, size_t bytes, void* ptr,
    ::ml_drift::cl::CLCommandQueue* queue) {
  TensorT dst_tensor;
  const BHWC shape = BHWC(cl_tensor.Batch(), cl_tensor.Height(),
                          cl_tensor.Width(), cl_tensor.Channels());
  dst_tensor.shape = shape;
  dst_tensor.data.resize(dst_tensor.shape.DimensionsProduct());
  TensorDescriptor desc;
  LITERT_RETURN_IF_ERROR(cl_tensor.ToDescriptor(&desc, queue).ok(),
                         kLiteRtStatusErrorRuntimeFailure);
  desc.DownloadData(dst_tensor.data.data());
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

  auto cl_tensor = std::make_unique<::ml_drift::cl::Tensor>();
  LITERT_RETURN_IF_ERROR(
      ::ml_drift::cl::CreateTensorShared(*gpu_env->GetContext(), cl_memory,
                                         *tensor_desc, cl_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::kBool) {
    return LiteRtGpuMemoryDownloadImpl<TensorBool, bool>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::kInt32) {
    return LiteRtGpuMemoryDownloadImpl<TensorInt32, int32_t>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_desc->GetDataType() == DataType::kFloat16) {
    if (tensor_type->element_type == kLiteRtElementTypeFloat32) {
      return LiteRtGpuMemoryDownloadImpl<TensorFloat32, float>(
          *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
    }
    return LiteRtGpuMemoryDownloadImpl<TensorFloat16, ::ml_drift::half>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else if (tensor_type->element_type == kLiteRtElementTypeInt8) {
    return LiteRtGpuMemoryDownloadImpl<TensorInt8, int8_t>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  } else {
    return LiteRtGpuMemoryDownloadImpl<TensorFloat32, float>(
        *cl_tensor, bytes, ptr, gpu_env->GetCommandQueue());
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#else

namespace litert::internal {

LiteRtStatus LiteRtGpuMemoryCreate(GpuEnvironment* gpu_env,
                                   const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, cl_mem* cl_memory) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGpuMemoryUpload(GpuEnvironment* gpu_env,
                                   const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, const void* ptr,
                                   cl_mem cl_memory) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGpuMemoryDownload(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes, cl_mem cl_memory,
                                     void* ptr) {
  return kLiteRtStatusErrorUnsupported;
}

}  // namespace litert::internal

#endif  // LITERT_HAS_OPENCL_SUPPORT
