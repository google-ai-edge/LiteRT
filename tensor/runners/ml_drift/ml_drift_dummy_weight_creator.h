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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_DUMMY_WEIGHT_CREATOR_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_DUMMY_WEIGHT_CREATOR_H_

#include <cstdint>
#include <vector>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/tensor.h"

namespace litert::tensor {

class MlDriftDummyWeightCreator {
 public:
  explicit MlDriftDummyWeightCreator(ml_drift::GpuModelBuilder* model_builder);

  template <typename T>
  absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> CreateFcWeights(
      const Tensor<MlDriftMixinTag>& weight_tensor,
      absl::Span<const T> data = {},
      ml_drift::DataType mld_float_data_type = ml_drift::DataType::kFloat32);

  template <typename T>
  absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> CreateConstantTensor(
      std::vector<int32_t> shape, absl::Span<const T> data,
      ml_drift::DataType mld_float_data_type = ml_drift::DataType::kFloat32);

 private:
  ml_drift::GpuModelBuilder* model_builder_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_DUMMY_WEIGHT_CREATOR_H_
