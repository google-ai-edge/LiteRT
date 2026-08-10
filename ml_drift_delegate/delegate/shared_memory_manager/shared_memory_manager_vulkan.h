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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_VULKAN_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_VULKAN_H_

#include <cstddef>
#include <memory>
#include <utility>

#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_spatial_tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/core/c/common.h"

namespace ml_drift {

using ::ml_drift::syrtis::VulkanSpatialTensor;

// Primary, graph-agnostic factory: builds the unified SharedMemoryManager over
// any graph via a GraphAdapter. Callers backed by an ir::IrModel construct an
// IrModelAdapter themselves and pass it here, keeping this backend layer free
// of any ir_model dependency.
inline std::unique_ptr<ml_drift::SharedMemoryManager>
MakeSharedMemoryManagerVulkan(::litert::ml_drift::SharedVulkanEnv* env,
                              const CreateGpuModelInfo& create_info,
                              std::unique_ptr<GraphAdapter> graph_adapter,
                              TfLiteContext* context,
                              ValueIdToSharedTensorMap& value_to_tensor_map,
                              ValueIdToSharedTensorMap& quant_param_tensors,
                              bool has_prepacked_external_tensors,
                              SerializationWeightCache* serialization_cache,
                              bool madvise_original_tensors) {
  return std::make_unique<SharedMemoryManager>(
      env->vulkan_env().GetInfo(), create_info, std::move(graph_adapter),
      [env](const TensorDescriptor& desc, size_t page_adjusted_offset,
            ::litert::ml_drift::ReleaseDataCallback release_data_callback,
            std::unique_ptr<GpuSpatialTensor>& tensor) -> absl::Status {
        if (tensor) {
          return absl::InternalError("Tensor is already initialized.");
        }
        if (release_data_callback) {
          return absl::InternalError(
              "Release data callback is not supported on Vulkan.");
        }
        VulkanSpatialTensor vk_tensor;
        ABSL_RETURN_IF_ERROR(::ml_drift::syrtis::CreateTensor(
            desc, &env->vulkan_env(), &vk_tensor));
        // TODO: b/403337563 - Use StagingBuffer to upload descriptor data.
        ABSL_RETURN_IF_ERROR(vk_tensor.UploadDescriptorData(
            desc, env->vulkan_env().GetVulkanDevice(),
            env->vulkan_env().GetQueue(), /*staging_buffer=*/nullptr,
            env->command_pool()));
        tensor = std::make_unique<VulkanSpatialTensor>(std::move(vk_tensor));
        return absl::OkStatus();
      },
      context, value_to_tensor_map, quant_param_tensors,
      has_prepacked_external_tensors, serialization_cache,
      madvise_original_tensors);
}

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_VULKAN_H_
