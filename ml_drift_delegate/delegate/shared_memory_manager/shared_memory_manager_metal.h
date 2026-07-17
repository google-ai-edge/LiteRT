// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_METAL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_METAL_H_

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/util.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/c/common.h"

namespace ml_drift {

namespace internal {

inline absl::Status MaybeBindExternalWeightData(
    const LiteRtRuntimeContext* runtime_context, weight_loader::WeightLoader* weight_loader,
    const std::unordered_map<size_t, size_t>* external_buffer_id_map,
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor, TfLiteTensor& tensor) {
  if (weight_loader == nullptr || external_buffer_id_map == nullptr) {
    return absl::OkStatus();
  }
  auto it =
      external_buffer_id_map->find(static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
  if (it == external_buffer_id_map->end() || it->second == 0) {
    return absl::OkStatus();
  }
  if (runtime_context == nullptr) {
    return absl::InternalError("Runtime context not available.");
  }

  weight_loader::WeightAccessRequest request;
  request.cpu = true;
  absl::Status prepare_status =
      weight_loader->PrepareAccessForBuffer(static_cast<uint32_t>(it->second), request,
                                            /*env=*/nullptr);
  if (!prepare_status.ok()) {
    return prepare_status;
  }
  const auto* access = weight_loader->GetExternalWeightByBuffer(static_cast<uint32_t>(it->second));
  if (access == nullptr || access->GetHostBuffer() == nullptr) {
    return absl::NotFoundError("External weight not found.");
  }
  void* host_memory = nullptr;
  if (runtime_context->get_tensor_buffer_host_memory(access->GetHostBuffer(), &host_memory) !=
          kLiteRtStatusOk ||
      host_memory == nullptr) {
    return absl::InternalError("Failed to get host memory.");
  }
  size_t buffer_size = 0;
  if (runtime_context->get_tensor_buffer_size(access->GetHostBuffer(), &buffer_size) !=
      kLiteRtStatusOk) {
    return absl::InternalError("Failed to get buffer size.");
  }
  if (static_cast<size_t>(tensor.bytes) > buffer_size) {
    return absl::InternalError("Tensor size is larger than buffer size.");
  }
  size_t buffer_offset = 0;
  if (runtime_context->get_tensor_buffer_offset(access->GetHostBuffer(), &buffer_offset) !=
      kLiteRtStatusOk) {
    return absl::InternalError("Failed to get buffer offset.");
  }
  const char* raw_ptr = reinterpret_cast<const char*>(host_memory) + buffer_offset;
  tensor.data.raw_const = raw_ptr;
  tensor.data.raw = const_cast<char*>(raw_ptr);
  tensor.allocation_type = kTfLiteCustom;
  return absl::OkStatus();
}

inline absl::StatusOr<std::string> LookupExternalWeightPacking(
    weight_loader::WeightLoader* weight_loader, uint32_t global_id) {
  if (weight_loader == nullptr) {
    return absl::NotFoundError("Weight loader not available");
  }
  if (global_id == 0) {
    return absl::InvalidArgumentError("Global id is zero.");
  }
  const auto* info = weight_loader->FindWeightInfoByBuffer(global_id);
  if (info == nullptr || info->packing.empty()) {
    return absl::NotFoundError("Packing info not found.");
  }
  return std::string(info->packing);
}

inline absl::Status TryCreateMetalTensorFromDeviceBuffer(
    const LiteRtRuntimeContext* runtime_context, weight_loader::WeightLoader* weight_loader,
    const std::unordered_map<size_t, size_t>* external_buffer_id_map,
    const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
    const ::ml_drift::TensorDescriptor& tensor_desc,
    std::unique_ptr<::ml_drift::GpuSpatialTensor>& tensor) {
  if (runtime_context == nullptr || weight_loader == nullptr || external_buffer_id_map == nullptr) {
    return absl::NotFoundError("Metal external weight device buffer not available.");
  }
  auto it =
      external_buffer_id_map->find(static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
  if (it == external_buffer_id_map->end() || it->second == 0) {
    return absl::NotFoundError("Tensor lacks external buffer id.");
  }
  const auto* access = weight_loader->GetExternalWeightByBuffer(static_cast<uint32_t>(it->second));
  if (access == nullptr || access->GetDeviceBuffer() == nullptr) {
    return absl::NotFoundError("No external Metal device buffer.");
  }

  HwMemoryHandle hw_memory_handle = nullptr;
  if (runtime_context->get_tensor_buffer_custom_tensor_buffer_handle(
          access->GetDeviceBuffer(), &hw_memory_handle) != kLiteRtStatusOk ||
      hw_memory_handle == nullptr) {
    return absl::InternalError("Failed to get external Metal device buffer handle.");
  }

  const auto* source_tensor =
      reinterpret_cast<const ::ml_drift::metal::MetalSpatialTensor*>(hw_memory_handle);
  ::ml_drift::TensorDescriptor desc_without_data;
  tensor_desc.CopyWithoutData(&desc_without_data);

  auto metal_tensor = std::make_unique<::ml_drift::metal::MetalSpatialTensor>();
  if (source_tensor->GetBufferHandle() != nil) {
    auto status = ::ml_drift::metal::CreateTensorSharedBuffer(
        source_tensor->GetBufferHandle(), desc_without_data, metal_tensor.get());
    if (!status.ok()) {
      return status;
    }
  } else if (source_tensor->GetTextureHandle() != nil) {
    auto status = ::ml_drift::metal::CreateTensorSharedTexture(
        source_tensor->GetTextureHandle(), desc_without_data, metal_tensor.get());
    if (!status.ok()) {
      return status;
    }
  } else {
    return absl::InvalidArgumentError("External Metal tensor has no buffer or texture handle.");
  }
  tensor = std::move(metal_tensor);
  return absl::OkStatus();
}

}  // namespace internal

// Wrapper class for the SharedMemoryManager for Metal.
// Example Usages:
// auto shared_memory_manager_metal = MakeSharedMemoryManagerMetal(
//     device, create_info, graph, context, value_to_tensor_map,
//     quant_param_tensors, has_prepacked_external_tensors,
//     serialization_cache, madvise_original_tensors);
// shared_memory_manager_metal.RegisterExternalConstantTensors(
//     shared_tensor_id, shared_tflite_tensor, local_to_global_id_map);
inline std::unique_ptr<ml_drift::SharedMemoryManager> MakeSharedMemoryManagerMetal(
    metal::MetalDevice* device, const CreateGpuModelInfo& create_info, GraphFloat32& graph,
    TfLiteContext* context, ValueIdToSharedTensorMap& value_to_tensor_map,
    ValueIdToSharedTensorMap& quant_param_tensors, bool has_prepacked_external_tensors,
    SerializationWeightCache* serialization_cache, bool madvise_original_tensors,
    const LiteRtRuntimeContext* runtime_context = nullptr,
    weight_loader::WeightLoader* weight_loader = nullptr,
    const std::unordered_map<size_t, size_t>* external_buffer_id_map = nullptr) {
  SharedMemoryManager::MaybeBindTensorDataFunc maybe_bind_data =
      [runtime_context, weight_loader, external_buffer_id_map](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
          TfLiteTensor& tensor) -> absl::Status {
    return internal::MaybeBindExternalWeightData(
        runtime_context, weight_loader, external_buffer_id_map, shared_tflite_tensor, tensor);
  };

  SharedMemoryManager::PackingLookupFunc packing_lookup =
      [weight_loader](uint32_t global_id) -> absl::StatusOr<std::string> {
    return internal::LookupExternalWeightPacking(weight_loader, global_id);
  };

  SharedMemoryManager::DiscardTensorDataFunc discard_tensor_data =
      [weight_loader, external_buffer_id_map](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor) -> absl::Status {
    if (weight_loader == nullptr || external_buffer_id_map == nullptr) {
      return absl::OkStatus();
    }
    auto it =
        external_buffer_id_map->find(static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
    if (it == external_buffer_id_map->end() || it->second == 0) {
      return absl::OkStatus();
    }
    uint32_t external_buffer_id = static_cast<uint32_t>(it->second);
    return weight_loader->DiscardExternalWeightByBuffer(external_buffer_id);
  };

  SharedMemoryManager::CreateTensorFromDeviceBufferFunc device_buffer_import =
      [runtime_context, weight_loader, external_buffer_id_map](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
          const ::ml_drift::TensorDescriptor& tensor_desc,
          std::unique_ptr<::ml_drift::GpuSpatialTensor>& tensor) -> absl::Status {
    return internal::TryCreateMetalTensorFromDeviceBuffer(
        runtime_context, weight_loader, external_buffer_id_map, shared_tflite_tensor, tensor_desc,
        tensor);
  };

  return std::make_unique<SharedMemoryManager>(
      device->GetInfo(), create_info, graph,
      [device](ml_drift::TensorDescriptor& tensor_desc, size_t page_adjusted_offset,
               ml_drift_delegate::ReleaseDataCallback release_data_callback,
               std::unique_ptr<GpuSpatialTensor>& tensor) {
        if (tensor) {
          return absl::InternalError("Tensor is already initialized.");
        }
        if (device == nullptr) {
          return absl::InvalidArgumentError("Device is null.");
        }
        id<MTLDevice> mtl_device = device->device();
        if (mtl_device == nil) {
          return absl::InvalidArgumentError("Underlying id<MTLDevice> is nil.");
        }
        auto metal_tensor = std::make_unique<metal::MetalSpatialTensor>();
        // Create a shared buffer to avoid extra copy. The release callback is called when the
        // buffer is deallocated.
        if (release_data_callback && tensor_desc.GetStorageType() == TensorStorageType::BUFFER) {
          const size_t bytes_count = tensor_desc.GetData().size();
          void* data_ptr = const_cast<uint8_t*>(tensor_desc.GetData().data());
          auto* release_cb_ptr =
              new ml_drift_delegate::ReleaseDataCallback(std::move(release_data_callback));
          id<MTLBuffer> buffer =
              [mtl_device newBufferWithBytesNoCopy:data_ptr
                                            length:bytes_count
                                           options:MTLResourceStorageModeShared
                                       deallocator:^(void* pointer, NSUInteger length) {
                                         (**release_cb_ptr)();
                                         delete release_cb_ptr;
                                       }];
          if (buffer == nil) {
            (**release_cb_ptr)();
            delete release_cb_ptr;
            return absl::InternalError("Failed to create Metal buffer with no copy.");
          }
          absl::Status status =
              metal::CreateTensorSharedBuffer(buffer, tensor_desc, metal_tensor.get());
          if (!status.ok()) {
            return status;
          }
          tensor = std::move(metal_tensor);
          return absl::OkStatus();
        }

        // Fallback to default creation if no mmap or not buffer.
        absl::Status status = metal_tensor->CreateFromDescriptor(tensor_desc, mtl_device);
        if (!status.ok()) {
          return status;
        }
        if (release_data_callback) {
          (*release_data_callback)();
        }
        // Transfer ownership to the output tensor.
        tensor = std::move(metal_tensor);
        return absl::OkStatus();
      },
      context, value_to_tensor_map, quant_param_tensors, has_prepacked_external_tensors,
      serialization_cache, madvise_original_tensors, /*experimental_int4_unpacking=*/true,
      /*experimental_int2_unpacking=*/false, std::move(device_buffer_import),
      std::move(maybe_bind_data), std::move(packing_lookup),
      /*maybe_get_external_buffer_id_func=*/nullptr,
      std::move(discard_tensor_data));
}

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_METAL_H_
