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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_UTILS_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/shared_tensor_maps.h"
#include "tflite/core/c/common.h"

namespace litert::ml_drift {

bool IsAsyncExecutionMode(TfLiteContext* context,
                          const LiteRtRuntimeContext* runtime_context);

// Returns tensor maps from the delegate data. If the shared tensor maps are
// provided by the client, returns the shared tensor maps from the client.
inline ::ml_drift::ValueIdToSharedTensorMap& GetBufferIdToSpatialTensorMap(
    MlDriftDelegateData& delegate_data) {
  if (delegate_data.options->shared_tensor_maps_from_client) {
    auto* shared_tensor_maps = reinterpret_cast<SharedTensorMaps*>(
        delegate_data.options->shared_tensor_maps_from_client);
    return shared_tensor_maps->buffer_id_to_spatial_tensor;
  }
  return delegate_data.buffer_id_to_spatial_tensor;
}

inline ::ml_drift::ValueIdToSharedTensorMap& GetQuantParamIdToSpatialTensorMap(
    MlDriftDelegateData& delegate_data) {
  if (delegate_data.options->shared_tensor_maps_from_client) {
    auto* shared_tensor_maps = reinterpret_cast<SharedTensorMaps*>(
        delegate_data.options->shared_tensor_maps_from_client);
    return shared_tensor_maps->quant_param_id_to_spatial_tensor;
  }
  return delegate_data.quant_param_id_to_spatial_tensor;
}

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_UTILS_H_
