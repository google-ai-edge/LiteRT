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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/tensor.h"

namespace litert::tensor {

// Builds the ML Drift `TensorDescriptor` for rearranged constant weights.
// For sub-byte (`kUint4`/`kInt4`/`kUint2`/`kInt2`) linear buffers, sets `c`
// to the packed byte count so `GetMemorySizeInBytes()` matches the packed
// payload instead of over-allocating 2x/4x (because `SizeOf` returns 1 for
// sub-byte types).
ml_drift::TensorDescriptor CreatePackedWeightsTensorDescriptor(
    const ml_drift::WeightsDescription& desc,
    const ml_drift::OHWI& weights_shape, size_t elements_count,
    bool is_embedding);

// Directly packs a 2-D `int8_t` weight matrix (`shape.h == 1 && shape.w == 1`)
// into ML Drift's `I4O4` `uint8_t` tile layout (`kOSpatialIOGroupI4O4` or
// `k2DYIsSpatialIOAndXIsOGroupI4O4`), flipping the sign bit (`^ 0x80u`) to map
// signed `[-128, 127]` into unsigned `[0, 255]` (padding out-of-bounds lanes
// with `0x80u`, i.e., signed `0`). Returns `false` if the shape, layout,
// `output_group_size`, or `dst` buffer size is unsupported so callers can fall
// back to `ml_drift::RearrangeWeightsInt8AsUint8`.
bool TryDirectRearrangeInt8AsUint8I4O4(const int8_t* src,
                                       const ml_drift::OHWI& shape,
                                       const ml_drift::WeightsDescription& desc,
                                       absl::Span<uint8_t> dst);

class MlDriftDummyWeightCreator {
 public:
  explicit MlDriftDummyWeightCreator(ml_drift::GpuModelBuilder* model_builder);

  template <typename T>
  absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> CreateFcWeights(
      const Tensor<MlDriftMixinTag>& weight_tensor,
      absl::Span<const T> data = {},
      ml_drift::DataType mld_float_data_type = ml_drift::DataType::kFloat32,
      ml_drift::TensorDescriptor* out_descriptor = nullptr);

  template <typename T>
  absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> CreateConstantTensor(
      std::vector<int32_t> shape, absl::Span<const T> data,
      ml_drift::DataType mld_float_data_type = ml_drift::DataType::kFloat32);

 private:
  ml_drift::GpuModelBuilder* model_builder_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_DUMMY_WEIGHT_CREATOR_H_
