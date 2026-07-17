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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_CL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_CL_H_

#include <cstddef>
#include <memory>

#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "tflite/core/c/common.h"

namespace ml_drift {

inline std::unique_ptr<ml_drift::SharedMemoryManager> MakeSharedMemoryManagerCl(
    const cl::Environment& env, const CreateGpuModelInfo& create_info,
    GraphFloat32& graph, TfLiteContext* context,
    ValueIdToSharedTensorMap& value_to_tensor_map,
    ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_external_tensors,
    SerializationWeightCache* serialization_cache,
    bool madvise_original_tensors) {
  return std::make_unique<ml_drift::SharedMemoryManager>(
      env.GetDevicePtr()->GetInfo(), create_info, graph,
      [&env](ml_drift::TensorDescriptor& tensor_desc,
             size_t page_adjusted_offset,
             ml_drift_delegate::ReleaseDataCallback release_data_callback,
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
      },
      context, value_to_tensor_map, quant_param_tensors,
      has_prepacked_external_tensors, serialization_cache,
      madvise_original_tensors, /*experimental_int4_unpacking=*/true,
      /*experimental_int2_unpacking=*/false);
}

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_CL_H_
