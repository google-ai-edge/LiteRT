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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_H_

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu_common.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/core/c/common.h"

namespace ml_drift {

inline absl::Status CopyBufferToBuffer(
    const webgpu::ExecutionEnvironment* env, const TensorDescriptor& desc,
    size_t page_adjusted_offset,
    ::litert::ml_drift::ReleaseDataCallback release_data_callback,
    webgpu::SpatialTensor* tensor) {
  return webgpu_internal::CopyBufferToBuffer(env, desc, page_adjusted_offset,
                                             std::move(release_data_callback),
                                             tensor);
}

// Primary, graph-agnostic factory: builds the unified SharedMemoryManager over
// any graph via a GraphAdapter. IR-backed callers construct an IrModelAdapter
// themselves and pass it here, keeping this backend layer free of any ir_model
// dependency.
std::unique_ptr<ml_drift::SharedMemoryManager> MakeSharedMemoryManagerWebgpu(
    const webgpu::ExecutionEnvironment& env,
    const CreateGpuModelInfo& create_info,
    std::unique_ptr<GraphAdapter> graph_adapter, TfLiteContext* context,
    ValueIdToSharedTensorMap& value_to_tensor_map,
    ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_tflite_tensors,
    SerializationWeightCache* serialization_cache,
    std::shared_ptr<Executor> upload_executor, bool madvise_original_tensors);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_H_
