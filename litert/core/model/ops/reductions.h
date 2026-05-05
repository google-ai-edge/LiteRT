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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_REDUCTIONS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_REDUCTIONS_H_

#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/kernels/internal/reference/reduce.h"

namespace litert::internal {

// Helper for generic reduction.
inline LiteRtStatus InferGenericReduction(const LiteRtOpT& op,
                                          absl::Span<const Dims> input_shapes,
                                          std::vector<Dims>& output_shapes,
                                          bool keep_dims) {
  constexpr size_t kReductionMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kAxisArgIndex = 1;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  if (input_shapes.size() < kReductionMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& axis_tensor = op.Input(kAxisArgIndex);

  // Axis must be constant.
  if (axis_tensor.Weights().Buffer().Size() < kShapeTensorSize) {
    return kLiteRtStatusErrorUnsupported;
  }

  auto axis_buf = axis_tensor.Weights().Buffer();
  std::set<int> axes;
  size_t num_axes = 0;

  // Simple heuristic: check element size.
  if (axis_buf.Size() % sizeof(int32_t) == 0) {
    // Could be int32.
    const int32_t* axis_data =
        reinterpret_cast<const int32_t*>(axis_buf.Data());
    num_axes = axis_buf.Size() / sizeof(int32_t);
    for (size_t i = 0; i < num_axes; ++i) {
      int axis = axis_data[i];
      if (axis < 0) axis += input_shape.size();
      axes.insert(axis);
    }
  } else if (axis_buf.Size() % sizeof(int64_t) == 0) {
    // Could be int64.
    const int64_t* axis_data =
        reinterpret_cast<const int64_t*>(axis_buf.Data());
    num_axes = axis_buf.Size() / sizeof(int64_t);
    for (size_t i = 0; i < num_axes; ++i) {
      int axis = static_cast<int>(axis_data[i]);
      if (axis < 0) axis += input_shape.size();
      axes.insert(axis);
    }
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out;
  for (int i = 0; i < input_shape.size(); ++i) {
    if (axes.count(i)) {
      if (keep_dims) {
        out.push_back(1);
      }
    } else {
      out.push_back(input_shape[i]);
    }
  }
  output_shapes[0] = std::move(out);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferReduce(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  const auto& opts = GetTflOptions(op);
  const auto* reducer_opts = opts.AsReducerOptions();
  bool keep_dims = reducer_opts ? reducer_opts->keep_dims : false;
  return InferGenericReduction(op, input_shapes, output_shapes, keep_dims);
}

inline LiteRtStatus InferArgMinMax(const LiteRtOpT& op,
                                   absl::Span<const Dims> input_shapes,
                                   std::vector<Dims>& output_shapes) {
  return InferGenericReduction(op, input_shapes, output_shapes,
                               /*keep_dims=*/false);
}

template <typename T>
inline bool ReferenceReduction(const T* input_data, const int* input_dims,
                               int input_num_dims, T* output_data,
                               const int* output_dims, int output_num_dims,
                               const int* axis, int num_axis, bool keep_dims,
                               T init_value,
                               T reducer(const T current, const T in)) {
  std::vector<int> temp_index(input_num_dims);
  std::vector<int> resolved_axis(input_num_dims);

  return tflite::reference_ops::ReduceGeneric<T>(
      input_data, input_dims, input_num_dims, output_data, output_dims,
      output_num_dims, axis, num_axis, keep_dims, temp_index.data(),
      resolved_axis.data(), init_value, reducer);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_REDUCTIONS_H_
