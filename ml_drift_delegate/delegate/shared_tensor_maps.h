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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_TENSOR_MAPS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_TENSOR_MAPS_H_

#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"

namespace litert::ml_drift {

// A container to share constant tensors, e.g. weights among multiple delegates
// and models.
struct SharedTensorMaps {
  // See MlDriftDelegateData::buffer_id_to_spatial_tensor for more details.
  ::ml_drift::ValueIdToSharedTensorMap buffer_id_to_spatial_tensor;
  // See MlDriftDelegateData::quant_param_id_to_spatial_tensor for more details.
  ::ml_drift::ValueIdToSharedTensorMap quant_param_id_to_spatial_tensor;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_TENSOR_MAPS_H_
