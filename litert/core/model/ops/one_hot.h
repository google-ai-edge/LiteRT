/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CORE_MODEL_OPS_ONE_HOT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CORE_MODEL_OPS_ONE_HOT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {

inline LiteRtStatus InferOneHot(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  if (input_shapes.size() != 4) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& indices_shape = input_shapes[0];
  const int rank = indices_shape.size();

  // Read options to get axis.
  const auto& opts = GetTflOptions(op);
  if (opts.type != tflite::BuiltinOptions_OneHotOptions) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* options = opts.AsOneHotOptions();
  int positive_axis = options->axis;

  if (positive_axis < 0) {
    positive_axis += rank + 1;
  }
  if (positive_axis < 0 || positive_axis > rank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Read depth from input 1 (constant tensor).
  if (op.Inputs().size() < 2) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* depth_tensor = op.Inputs()[1];
  if (!depth_tensor || depth_tensor->Weights().Buffer().Size() == 0) {
    // Depth is not constant, we cannot infer shape statically.
    // In production models, depth might be dynamic. Here we handle constant
    // case.
    return kLiteRtStatusErrorInvalidArgument;
  }

  const int32_t* depth_data =
      reinterpret_cast<const int32_t*>(depth_tensor->Weights().Buffer().Data());
  int32_t depth = depth_data[0];

  std::vector<int32_t> output_shape(indices_shape.begin(), indices_shape.end());
  output_shape.insert(output_shape.begin() + positive_axis, depth);

  output_shapes.resize(1);
  output_shapes[0] = Dims(output_shape.begin(), output_shape.end());

  return kLiteRtStatusOk;
}

inline std::vector<int32_t> ComputeStrides(absl::Span<const int32_t> shape) {
  int rank = shape.size();
  std::vector<int32_t> strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

template <typename T>
void ReferenceOneHot(const int32_t* indices_data, size_t num_indices,
                     int32_t depth, T on_value, T off_value, int positive_axis,
                     int rank, const std::vector<int32_t>& indices_shape,
                     T* output_data) {
  std::vector<int32_t> indices_strides =
      ComputeStrides(absl::MakeSpan(indices_shape));

  std::vector<int32_t> output_shape = indices_shape;
  output_shape.insert(output_shape.begin() + positive_axis, depth);

  std::vector<int32_t> output_strides = ComputeStrides(output_shape);

  size_t total_output_elements = 1;
  for (auto dim : output_shape) {
    total_output_elements *= dim;
  }
  std::fill(output_data, output_data + total_output_elements, off_value);

  std::vector<int32_t> M(rank);
  for (int j = 0; j < rank; ++j) {
    if (j < positive_axis) {
      M[j] = output_strides[j];
    } else {
      M[j] = output_strides[j + 1];
    }
  }

  for (size_t i = 0; i < num_indices; ++i) {
    size_t remaining = i;
    size_t out_base = 0;
    for (size_t j = 0; j < rank; ++j) {
      int32_t coord = remaining / indices_strides[j];
      remaining %= indices_strides[j];
      out_base += coord * M[j];
    }

    int32_t target_index = indices_data[i];
    if (target_index >= 0 && target_index < depth) {
      size_t out_idx = out_base + target_index * output_strides[positive_axis];
      output_data[out_idx] = on_value;
    }
  }
}

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_MODEL_OPS_ONE_HOT_H_
