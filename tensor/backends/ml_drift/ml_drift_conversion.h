/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_CONVERSION_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert::tensor {

absl::StatusOr<
    absl::flat_hash_map<graph::Tensor, ml_drift::GpuModelBuilder::TensorHandle>>
AppendGpuModelBuilder(MlDriftBuildContext& ctx,
                      std::vector<TensorHandle> outputs);

template <class... Mixins>
absl::StatusOr<
    absl::flat_hash_map<graph::Tensor, ml_drift::GpuModelBuilder::TensorHandle>>
AppendGpuModelBuilder(MlDriftBuildContext& ctx,
                      absl::Span<const Tensor<Mixins...>> outputs) {
  std::vector<TensorHandle> erased_outputs(outputs.begin(), outputs.end());
  return AppendGpuModelBuilder(ctx, std::move(erased_outputs));
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_ML_DRIFT_CONVERSION_H_
