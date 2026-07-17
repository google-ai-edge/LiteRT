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

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_opencl_litert.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_cl_litert.h"
#include "tflite/c/common.h"
#include "tflite/core/subgraph.h"

namespace litert::ml_drift {

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendOpenClLitert::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
    MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  const ::ml_drift::TensorIndexToExternalBufferIdMap* external_buffer_id_map =
      &(reinterpret_cast<const tflite::Subgraph*>(context->impl_)
            ->GetExternalTensorBufferIdentifiers());
  return ::ml_drift::MakeSharedMemoryManagerClLitert(
      *cl_env(), delegate_data.options->runtime_context, create_info, graph,
      context, GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors,
      serialization_cache,
      delegate_data.options->madvise_original_shared_tensors,
      delegate_data.weight_loader, external_buffer_id_map);
}

}  // namespace litert::ml_drift
