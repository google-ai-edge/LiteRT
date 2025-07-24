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
#include "litert/runtime/metal_sync.h"

#import <Metal/Metal.h>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/litert_gpu_util.h"
#include "tflite/delegates/gpu/common/shape.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/tensor.h"
#include "tflite/delegates/gpu/metal/metal_device.h"
#include "tflite/delegates/gpu/metal/metal_spatial_tensor.h"

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

absl::StatusOr<TensorDescriptor> CreateMetalTensorDescriptor(
    const LiteRtRankedTensorType* tensor_type, LiteRtTensorBufferType buffer_type) {
  BHWC shape;
  LITERT_RETURN_IF_ERROR(ConvertLiteRtTensorTypeToGpuShape(tensor_type, &shape).ok());

  DataType data_type;
  LITERT_RETURN_IF_ERROR(
      ConvertLiteRtDataTypeToGpuDataType(tensor_type, &data_type, buffer_type).ok());

  TensorStorageType storage_type;
  switch (buffer_type) {
    case kLiteRtTensorBufferTypeMetalBuffer:
    case kLiteRtTensorBufferTypeMetalBufferFp16:
      storage_type = TensorStorageType::BUFFER;
      break;
    case kLiteRtTensorBufferTypeMetalTexture:
    case kLiteRtTensorBufferTypeMetalTextureFp16:
      storage_type = TensorStorageType::TEXTURE_2D;
      break;
    default:
      return absl::InvalidArgumentError("Unsupported buffer type.");
  }

  if (shape.b == 1) {
    return CreateHwcTensorDescriptor(data_type, storage_type, HWC(shape.h, shape.w, shape.c));
  }
  return CreateBhwcTensorDescriptor(data_type, storage_type,
                                    BHWC(shape.b, shape.h, shape.w, shape.c));
}

LiteRtStatus LiteRtMetalMemoryCreate(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type, size_t bytes,
                                     void** metal_memory) {
  auto tensor_desc = CreateMetalTensorDescriptor(tensor_type, buffer_type);
  LITERT_RETURN_IF_ERROR(tensor_desc.status().ok(), kLiteRtStatusErrorUnsupported);

  tflite::gpu::metal::MetalSpatialTensor tensor_memory;
  LITERT_RETURN_IF_ERROR(
      tflite::gpu::metal::CreateTensor((__bridge id<MTLDevice>)(gpu_env->getMetalDevice()),
                                       *tensor_desc, &tensor_memory)
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  *metal_memory = (__bridge void*)tensor_memory.GetBufferHandle();
  return kLiteRtStatusOk;
}

template <typename TensorT, typename DataTypeT>
LiteRtStatus LiteRtMetalGpuMemoryDownloadImpl(tflite::gpu::metal::MetalSpatialTensor& metal_tensor,
                                              size_t bytes, void* ptr, void* metal_device) {
  TensorT dst_tensor;
  const BHWC shape = BHWC(metal_tensor.Batch(), metal_tensor.Height(), metal_tensor.Width(),
                          metal_tensor.Channels());
  dst_tensor.shape = shape;
  dst_tensor.data.resize(dst_tensor.shape.DimensionsProduct());
  TensorDescriptor desc;
  LITERT_RETURN_IF_ERROR(
      metal_tensor.ToDescriptor(&desc, (__bridge id<MTLDevice>)(metal_device)).ok(),
      kLiteRtStatusErrorRuntimeFailure);
  desc.DownloadData(&dst_tensor);
  if (dst_tensor.data.size() * sizeof(DataTypeT) != bytes) {
    LITERT_LOG(LITERT_ERROR, "Download buffer size mismatch: required: %zu vs given: %zu",
               dst_tensor.data.size() * sizeof(DataTypeT), bytes);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  std::memcpy(ptr, dst_tensor.data.data(), bytes);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMetalMemoryDownload(GpuEnvironment* gpu_env, void* metal_memory,
                                       const LiteRtRankedTensorType* tensor_type,
                                       LiteRtTensorBufferType buffer_type, size_t bytes,
                                       void* data) {
  auto tensor_desc = CreateMetalTensorDescriptor(tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create tensor descriptor: %s",
               tensor_desc.status().message().data());
    return kLiteRtStatusErrorUnsupported;
  }
  auto metal_tensor = std::make_unique<tflite::gpu::metal::MetalSpatialTensor>();
  LITERT_RETURN_IF_ERROR(tflite::gpu::metal::CreateTensorSharedBuffer(
                             (__bridge id<MTLBuffer>)metal_memory, *tensor_desc, metal_tensor.get())
                             .ok(),
                         kLiteRtStatusErrorRuntimeFailure);
  if (tensor_desc->GetDataType() == DataType::BOOL) {
    return LiteRtMetalGpuMemoryDownloadImpl<TensorBool, bool>(*metal_tensor, bytes, data,
                                                              gpu_env->getMetalDevice());
  } else if (tensor_desc->GetDataType() == DataType::INT32) {
    return LiteRtMetalGpuMemoryDownloadImpl<TensorInt32, int32_t>(*metal_tensor, bytes, data,
                                                                  gpu_env->getMetalDevice());
  } else {
    return LiteRtMetalGpuMemoryDownloadImpl<TensorFloat32, float>(*metal_tensor, bytes, data,
                                                                  gpu_env->getMetalDevice());
  }
  return kLiteRtStatusOk;
}

template <typename TensorT, typename DataTypeT>
LiteRtStatus LiteRtMetalGpuMemoryUploadImpl(tflite::gpu::metal::MetalSpatialTensor& metal_tensor,
                                            size_t bytes, const void* ptr, void* metal_device) {
  TensorT src_tensor;
  src_tensor.shape = BHWC(metal_tensor.Batch(), metal_tensor.Height(), metal_tensor.Width(),
                          metal_tensor.Channels());
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  if (src_tensor.data.size() * sizeof(DataTypeT) != bytes) {
    LITERT_LOG(LITERT_ERROR, "Upload buffer size mismatch: required: %zu vs given: %zu",
               src_tensor.data.size() * sizeof(DataTypeT), bytes);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  // TODO - b/413431454: Try to avoid the copy.
  std::memcpy(src_tensor.data.data(), ptr, bytes);

  TensorDescriptor descriptor_with_data = metal_tensor.GetDescriptor();
  descriptor_with_data.UploadData(src_tensor);
  LITERT_RETURN_IF_ERROR(
      metal_tensor
          .UploadDescriptorData(descriptor_with_data, (__bridge id<MTLDevice>)(metal_device))
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtMetalMemoryUpload(GpuEnvironment* gpu_env, void* metal_memory,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type, size_t bytes,
                                     const void* data) {
  auto tensor_desc = CreateMetalTensorDescriptor(tensor_type, buffer_type);
  LITERT_RETURN_IF_ERROR(tensor_desc.status().ok(), kLiteRtStatusErrorUnsupported);

  auto metal_spatial_tensor = std::make_unique<tflite::gpu::metal::MetalSpatialTensor>();
  LITERT_RETURN_IF_ERROR(
      tflite::gpu::metal::CreateTensorSharedBuffer((__bridge id<MTLBuffer>)metal_memory,
                                                   *tensor_desc, metal_spatial_tensor.get())
          .ok(),
      kLiteRtStatusErrorRuntimeFailure);

  if (tensor_desc->GetDataType() == DataType::BOOL) {
    return LiteRtMetalGpuMemoryUploadImpl<TensorBool, bool>(*metal_spatial_tensor, bytes, data,
                                                            gpu_env->getMetalDevice());
  } else if (tensor_desc->GetDataType() == DataType::INT32) {
    return LiteRtMetalGpuMemoryUploadImpl<TensorInt32, int32_t>(*metal_spatial_tensor, bytes, data,
                                                                gpu_env->getMetalDevice());
  } else {
    return LiteRtMetalGpuMemoryUploadImpl<TensorFloat32, float>(*metal_spatial_tensor, bytes, data,
                                                                gpu_env->getMetalDevice());
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal
