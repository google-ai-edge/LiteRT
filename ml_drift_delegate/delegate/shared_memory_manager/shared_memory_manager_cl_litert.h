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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_CL_LITERT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_CL_LITERT_H_

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "tflite/core/c/common.h"

struct LiteRtRuntimeContext;

namespace weight_loader {
class WeightLoader;
}  // namespace weight_loader

namespace ml_drift {
// Map from tensor index to external buffer ID.
using TensorIndexToExternalBufferIdMap = std::unordered_map<size_t, size_t>;

// Creates a SharedMemoryManager for the OpenCL backend. Primary, graph-agnostic
// overload: IR-backed callers construct an IrModelAdapter and pass it here,
// keeping this backend layer free of any ir_model dependency.
std::unique_ptr<::ml_drift::SharedMemoryManager>
MakeSharedMemoryManagerClLitert(
    const ::ml_drift::cl::Environment& env,
    const ::LiteRtRuntimeContext* runtime_context,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    std::unique_ptr<::ml_drift::GraphAdapter> graph_adapter,
    TfLiteContext* context,
    ::ml_drift::ValueIdToSharedTensorMap& value_to_tensor_map,
    ::ml_drift::ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_external_tensors,
    ::ml_drift::SerializationWeightCache* serialization_cache,
    bool madvise_original_tensors, weight_loader::WeightLoader* weight_loader,
    const TensorIndexToExternalBufferIdMap* external_buffer_id_map = nullptr);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_CL_LITERT_H_
