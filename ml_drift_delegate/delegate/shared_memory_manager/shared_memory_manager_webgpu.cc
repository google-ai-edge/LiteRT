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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu_common.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

namespace ml_drift {

std::unique_ptr<ml_drift::SharedMemoryManager> MakeSharedMemoryManagerWebgpu(
    const webgpu::ExecutionEnvironment& env,
    const CreateGpuModelInfo& create_info,
    std::unique_ptr<GraphAdapter> graph_adapter, TfLiteContext* context,
    ValueIdToSharedTensorMap& value_to_tensor_map,
    ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_tflite_tensors,
    SerializationWeightCache* serialization_cache,
    std::shared_ptr<Executor> upload_executor, bool madvise_original_tensors) {
  return std::make_unique<ml_drift::SharedMemoryManager>(
      env.GetInfo(), create_info, std::move(graph_adapter),
      [&env, has_prepacked_tflite_tensors, upload_executor](
          ml_drift::TensorDescriptor& tensor_desc, size_t page_adjusted_offset,
          ::litert::ml_drift::ReleaseDataCallback release_data_callback,
          std::unique_ptr<GpuSpatialTensor>& tensor) {
        return webgpu_internal::CreateSharedWebGpuTensor(
            env, tensor_desc, page_adjusted_offset,
            std::move(release_data_callback), has_prepacked_tflite_tensors,
            upload_executor.get(),
            webgpu_internal::UploadScheduling::kAllowInline, tensor);
      },
      context, value_to_tensor_map, quant_param_tensors,
      has_prepacked_tflite_tensors, serialization_cache,
      madvise_original_tensors, /*experimental_int4_unpacking=*/true,
      /*experimental_int2_unpacking=*/false);
}

}  // namespace ml_drift
