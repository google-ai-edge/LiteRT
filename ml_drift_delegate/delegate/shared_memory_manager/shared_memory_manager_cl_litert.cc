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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_cl_litert.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "weight_loader/external_weight_loader_litert.h"
#include <CL/cl.h>
#include "tflite/c/common.h"

namespace ml_drift {

std::unique_ptr<::ml_drift::SharedMemoryManager>
MakeSharedMemoryManagerClLitert(
    const ::ml_drift::cl::Environment& env,
    const ::LiteRtRuntimeContext* runtime_context,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
    ::ml_drift::ValueIdToSharedTensorMap& value_to_tensor_map,
    ::ml_drift::ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_external_tensors,
    ::ml_drift::SerializationWeightCache* serialization_cache,
    bool madvise_original_tensors, weight_loader::WeightLoader* weight_loader,
    const TensorIndexToExternalBufferIdMap* external_buffer_id_map) {
  ::ml_drift::SharedMemoryManager::CreateTensorFromDeviceBufferFunc
      device_buffer_import =
          [&env, weight_loader, external_buffer_id_map, runtime_context](
              const ::litert::ml_drift::SharedTfliteTensor&
                  shared_tflite_tensor,
              const ::ml_drift::TensorDescriptor& tensor_desc,
              std::unique_ptr<::ml_drift::GpuSpatialTensor>& tensor)
      -> absl::Status {
    if (!weight_loader) {
      return absl::NotFoundError("Weight loader not available");
    }
    if (external_buffer_id_map == nullptr) {
      return absl::NotFoundError("External buffer map not available");
    }
    auto it = external_buffer_id_map->find(
        static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
    if (it == external_buffer_id_map->end() || it->second == 0) {
      return absl::NotFoundError("Tensor lacks external buffer id");
    }
    const auto* access = weight_loader->GetExternalWeightByBuffer(
        static_cast<uint32_t>(it->second));
    if (access == nullptr || access->GetDeviceBuffer() == nullptr) {
      return absl::NotFoundError("No external device buffer");
    }
    cl_mem cl_memory;
    if (runtime_context->get_tensor_buffer_opencl_memory(
            access->GetDeviceBuffer(), &cl_memory) != kLiteRtStatusOk) {
      return absl::InternalError(
          "Failed to get OpenCL memory from device buffer");
    }

    ::ml_drift::TensorDescriptor desc_copy;
    tensor_desc.CopyWithoutData(&desc_copy);

    auto cl_tensor = std::make_unique<::ml_drift::cl::Tensor>();
    ABSL_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
        env.context(), cl_memory, desc_copy, cl_tensor.get()));

    tensor = std::move(cl_tensor);
    return absl::OkStatus();
  };

  ::ml_drift::SharedMemoryManager::CreateTensorFunc create_tensor_func =
      [&env](ml_drift::TensorDescriptor& tensor_desc,
             size_t page_adjusted_offset,
             ::litert::ml_drift::ReleaseDataCallback release_data_callback,
             std::unique_ptr<GpuSpatialTensor>& tensor) {
        if (tensor) {
          return absl::InternalError("Tensor is already initialized.");
        }
        if (release_data_callback) {
          return absl::InvalidArgumentError(
              "Release data callback is not currently supported on OpenCL.");
        }
        tensor = std::make_unique<cl::Tensor>();
        return CreateTensor(env.context(), tensor_desc,
                            dynamic_cast<cl::Tensor*>(tensor.get()));
      };

  ::ml_drift::SharedMemoryManager::MaybeBindTensorDataFunc maybe_bind_data =
      [weight_loader, external_buffer_id_map, runtime_context](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
          TfLiteTensor& tensor) -> absl::Status {
    if (!weight_loader) {
      return absl::OkStatus();
    }
    if (external_buffer_id_map == nullptr) {
      return absl::OkStatus();
    }
    auto it = external_buffer_id_map->find(
        static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
    if (it == external_buffer_id_map->end() || it->second == 0) {
      // This is invoked for shared tensor broadly, not only external weight
      // tensors. For non-external tensors, "not in map / id is zero" is
      // expected.
      return absl::OkStatus();
    }
    const uint32_t external_buffer_id = static_cast<uint32_t>(it->second);
    weight_loader::WeightAccessRequest request;
    request.cpu = true;
    absl::Status prepare_status = weight_loader->PrepareAccessForBuffer(
        external_buffer_id, request, /*env=*/nullptr);
    if (!prepare_status.ok()) {
      return prepare_status;
    }
    const auto* access = weight_loader->GetExternalWeightByBuffer(
        external_buffer_id);
    if (access == nullptr || access->GetHostBuffer() == nullptr) {
      return absl::NotFoundError(
          "Prepared external OpenCL host weight not found.");
    }
    void* host_memory = nullptr;
    if (runtime_context->get_tensor_buffer_host_memory(
            access->GetHostBuffer(), &host_memory) != kLiteRtStatusOk ||
        host_memory == nullptr) {
      return absl::InternalError("Failed to get host memory.");
    }
    size_t buffer_size = 0;
    if (runtime_context->get_tensor_buffer_size(
            access->GetHostBuffer(), &buffer_size) != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get buffer size.");
    }
    if (static_cast<size_t>(tensor.bytes) > buffer_size) {
      return absl::InternalError("Tensor size is larger than buffer size.");
    }
    size_t buffer_offset = 0;
    if (runtime_context->get_tensor_buffer_offset(
            access->GetHostBuffer(), &buffer_offset) != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get buffer offset.");
    }
    const char* raw_ptr =
        reinterpret_cast<const char*>(host_memory) + buffer_offset;
    tensor.data.raw_const = raw_ptr;
    tensor.data.raw = const_cast<char*>(raw_ptr);
    // Mark as custom allocation so TFLite won't try to free this memory
    // during cleanup. The memory is owned by the weight_loader.
    tensor.allocation_type = kTfLiteCustom;
    return absl::OkStatus();
  };

  ::ml_drift::SharedMemoryManager::PackingLookupFunc packing_lookup =
      [weight_loader](uint32_t global_id) -> absl::StatusOr<std::string> {
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
  };

  return std::make_unique<::ml_drift::SharedMemoryManager>(
      env.GetDevicePtr()->GetInfo(), create_info, graph, create_tensor_func,
      context, value_to_tensor_map, quant_param_tensors,
      has_prepacked_external_tensors, serialization_cache,
      madvise_original_tensors, /*experimental_int4_unpacking=*/true,
      /*experimental_int2_unpacking=*/false,
      std::move(device_buffer_import), std::move(maybe_bind_data),
      std::move(packing_lookup));
}

}  // namespace ml_drift
