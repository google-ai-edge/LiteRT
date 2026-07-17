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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu_litert.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu_common.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/c/common.h"

namespace ml_drift {

std::unique_ptr<ml_drift::SharedMemoryManager>
MakeSharedMemoryManagerWebgpuLitert(
    const webgpu::ExecutionEnvironment& env,
    ::LiteRtRuntimeContext* runtime_context,
    const CreateGpuModelInfo& create_info, GraphFloat32& graph,
    TfLiteContext* context, ValueIdToSharedTensorMap& value_to_tensor_map,
    ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_tflite_tensors,
    SerializationWeightCache* serialization_cache,
    std::shared_ptr<Executor> upload_executor, bool madvise_original_tensors,
    weight_loader::WeightLoader* weight_loader,
    const TensorIndexToExternalBufferIdMap* external_buffer_id_map) {
  LITERT_LOG(LITERT_DEBUG,
             "MakeSharedMemoryManagerWebgpuLitert: weight_loader=%p, "
             "external_buffer_id_map=%p",
             weight_loader, external_buffer_id_map);
  // Callback to bind CPU memory from weight_loader to TFLite tensors.
  // This allows ML Drift to read the weight data and copy it to GPU.
  // Note: We use external_buffer_id_map to translate from tflite_tensor_id to
  // external_buffer_id, because
  // ::litert::ml_drift::SharedTfliteTensor.global_id contains the internal
  // buffer ID (tensor->buffer()), not the external buffer ID
  // (tensor->external_buffer()) that the weight_loader uses.
  ::ml_drift::SharedMemoryManager::MaybeBindTensorDataFunc maybe_bind_data =
      [weight_loader, external_buffer_id_map, runtime_context](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
          TfLiteTensor& tensor) -> absl::Status {
    LITERT_LOG(LITERT_DEBUG,
               "maybe_bind_data called: tflite_tensor_id=%d, global_id=%d",
               shared_tflite_tensor.tflite_tensor_id,
               shared_tflite_tensor.global_id);

    if (!weight_loader) {
      LITERT_LOG(LITERT_DEBUG,
                 "maybe_bind_data: weight_loader is null, skipping");
      return absl::OkStatus();
    }

    // Look up the external_buffer_id using the tflite_tensor_id.
    // The external_buffer_id_map maps tensor indices to external buffer IDs.
    // If the tensor is not in the map, it's not an external weight tensor,
    // so we return OkStatus() to let the normal tensor loading path handle it.
    uint32_t external_buffer_id = 0;
    if (external_buffer_id_map != nullptr) {
      auto it = external_buffer_id_map->find(
          static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
      if (it != external_buffer_id_map->end()) {
        external_buffer_id = static_cast<uint32_t>(it->second);
        LITERT_LOG(LITERT_DEBUG,
                   "maybe_bind_data: Found external_buffer_id=%u for tensor %d",
                   external_buffer_id, shared_tflite_tensor.tflite_tensor_id);
      } else {
        // Tensor is not an external weight - let normal loading path handle it.
        LITERT_LOG(LITERT_DEBUG,
                   "maybe_bind_data: tensor %d not in external_buffer_id_map, "
                   "skipping (not external weight)",
                   shared_tflite_tensor.tflite_tensor_id);
        return absl::OkStatus();
      }
    } else {
      // No external buffer ID map - this tensor is not an external weight.
      LITERT_LOG(LITERT_DEBUG,
                 "maybe_bind_data: external_buffer_id_map is null, skipping");
      return absl::OkStatus();
    }

    if (external_buffer_id == 0) {
      // External buffer ID of 0 means no external weight - let normal path
      // handle it.
      LITERT_LOG(LITERT_DEBUG,
                 "maybe_bind_data: external_buffer_id is 0, skipping");
      return absl::OkStatus();
    }

    LITERT_LOG(LITERT_DEBUG,
               "maybe_bind_data: looking up external_buffer_id=%u "
               "(tflite_tensor_id=%d, global_id=%d)",
               external_buffer_id, shared_tflite_tensor.tflite_tensor_id,
               shared_tflite_tensor.global_id);
    weight_loader::WeightAccessRequest request;
    request.cpu = true;
    absl::Status prepare_status = weight_loader->PrepareAccessForBuffer(
        external_buffer_id, request, /*env=*/nullptr);
#ifndef __EMSCRIPTEN__
    if (!prepare_status.ok()) {
      return prepare_status;
    }
#endif
    const auto* access =
        weight_loader->GetExternalWeightByBuffer(external_buffer_id);
    if (access == nullptr || access->GetHostBuffer() == nullptr) {
#ifdef __EMSCRIPTEN__
      LITERT_LOG(
          LITERT_WARNING,
          "maybe_bind_data: External weight not found for "
          "external_buffer_id=%u (continuing as weights may be streamed)",
          external_buffer_id);
      // Assign a dummy non-null pointer derived from external_buffer_id
      // so that ML Drift's GPU model builder doesn't deduplicate distinct
      // external weight tensors that have nullptr data pointers.
      tensor.data.raw =
          reinterpret_cast<char*>(static_cast<uintptr_t>(external_buffer_id));
      tensor.data.raw_const = tensor.data.raw;
      tensor.allocation_type = kTfLiteCustom;
      return absl::OkStatus();
#else
      LITERT_LOG(LITERT_ERROR,
                 "maybe_bind_data: External weight not found for "
                 "external_buffer_id=%u, access=%p",
                 external_buffer_id, access);
      return absl::NotFoundError("External weight not found.");
#endif
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
    LITERT_LOG(LITERT_DEBUG,
               "maybe_bind_data: Successfully bound external weight for "
               "tensor %d (external_buffer_id=%u, size=%zu bytes)",
               shared_tflite_tensor.tflite_tensor_id, external_buffer_id,
               buffer_size);
    return absl::OkStatus();
  };

  // Callback to look up packing information from weight_loader.
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

  ::ml_drift::SharedMemoryManager::MaybeGetExternalBufferIdFunc
      maybe_get_external_buffer_id =
          [external_buffer_id_map](
              int shared_tflite_tensor_id) -> absl::StatusOr<uint32_t> {
    if (external_buffer_id_map == nullptr) {
      return absl::NotFoundError("External buffer ID map is null.");
    }
    auto it = external_buffer_id_map->find(
        static_cast<size_t>(shared_tflite_tensor_id));
    if (it == external_buffer_id_map->end()) {
      return absl::NotFoundError("External buffer ID not found.");
    }
    return static_cast<uint32_t>(it->second);
  };

  ::ml_drift::SharedMemoryManager::DiscardTensorDataFunc discard_tensor_data =
      [weight_loader, external_buffer_id_map](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor)
      -> absl::Status {
    if (weight_loader == nullptr || external_buffer_id_map == nullptr) {
      return absl::OkStatus();
    }
    auto it = external_buffer_id_map->find(
        static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
    if (it == external_buffer_id_map->end() || it->second == 0) {
      return absl::OkStatus();
    }
    uint32_t external_buffer_id = static_cast<uint32_t>(it->second);
    return weight_loader->DiscardExternalWeightByBuffer(external_buffer_id);
  };

  // Create the SharedMemoryManager with the callbacks.
  // Note: We don't use device_buffer_import for WebGPU since we load weights
  // to CPU memory and let ML Drift copy to GPU.
  return std::make_unique<ml_drift::SharedMemoryManager>(
      env.GetInfo(), create_info, graph,
      [&env, has_prepacked_tflite_tensors, upload_executor](
          ml_drift::TensorDescriptor& tensor_desc, size_t page_adjusted_offset,
          ::litert::ml_drift::ReleaseDataCallback release_data_callback,
          std::unique_ptr<GpuSpatialTensor>& tensor) {
        return webgpu_internal::CreateSharedWebGpuTensor(
            env, tensor_desc, page_adjusted_offset,
            std::move(release_data_callback), has_prepacked_tflite_tensors,
            upload_executor.get(),
            webgpu_internal::UploadScheduling::kRequireExecutor, tensor);
      },
      context, value_to_tensor_map, quant_param_tensors,
      has_prepacked_tflite_tensors, serialization_cache,
      madvise_original_tensors, /*experimental_int4_unpacking=*/true,
      /*experimental_int2_unpacking=*/false,
      /*create_tensor_from_device_buffer_func=*/nullptr,
      std::move(maybe_bind_data), std::move(packing_lookup),
      std::move(maybe_get_external_buffer_id), std::move(discard_tensor_data));
}

}  // namespace ml_drift
