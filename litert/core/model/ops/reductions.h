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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/kernels/internal/reference/reduce.h"
#include "tflite/kernels/internal/runtime_shape.h"

namespace litert::internal {

// Helper for generic reduction.
inline LiteRtStatus InferGenericReduction(const LiteRtOpT& op,
                                          absl::Span<const Dims> input_shapes,
                                          std::vector<Dims>& output_shapes,
                                          bool keep_dims,
                                          bool is_arg_min_max = false) {
  constexpr size_t kReductionMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kAxisArgIndex = 1;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  if (input_shapes.size() < kReductionMinArgs || output_shapes.size() != 1 ||
      op.Inputs().size() < kReductionMinArgs ||
      op.Inputs()[kAxisArgIndex] == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  if (input_shape.empty() &&
      (is_arg_min_max ||
       (op.Inputs()[kInputArgIndex] != nullptr &&
        op.Input(kInputArgIndex).Type().first != kLiteRtRankedTensorType))) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& axis_tensor = op.Input(kAxisArgIndex);

  // Axis must be constant.
  if (axis_tensor.Weights().Buffer().Size() < kShapeTensorSize) {
    return kLiteRtStatusErrorUnsupported;
  }

  auto axis_buf = axis_tensor.Weights().Buffer();
  const int64_t rank = static_cast<int64_t>(input_shape.size());
  std::set<int> axes;

  // Normalizes and validates one axis value like TFLite's
  // `ResolveAxisForShape` (kernels/reduce.cc) and
  // `kernel_utils::ReadAndNormalizeAxis` (kernels/arg_min_max.cc): negative
  // values are offset by the input rank and the result must lie in [0, rank);
  // a scalar input in a generic reducer ignores the axes and reduces to a
  // scalar.
  auto add_axis = [&](int64_t axis) -> bool {
    if (rank == 0) return true;
    if (axis < -rank || axis >= rank) {
      return false;
    }
    if (axis < 0) axis += rank;
    axes.insert(static_cast<int>(axis));
    return true;
  };

  // Decode the axis buffer by its declared element type. The byte size alone
  // cannot tell the widths apart: an INT64 buffer is always a multiple of
  // sizeof(int32_t), so `{1}` would otherwise be read as the two axes {1, 0}.
  // An untyped axis tensor is read as INT32, the TFLite default.
  LiteRtElementType axis_type = kLiteRtElementTypeNone;
  if (axis_tensor.Type().first == kLiteRtRankedTensorType) {
    axis_type = axis_tensor.Type().second.ranked_tensor_type.element_type;
  } else if (axis_tensor.Type().first == kLiteRtUnrankedTensorType) {
    axis_type = axis_tensor.Type().second.unranked_tensor_type.element_type;
  }

  const auto& axis_shape = input_shapes[kAxisArgIndex];
  const bool has_axis_shape =
      axis_tensor.Type().first == kLiteRtRankedTensorType ||
      !axis_shape.empty();
  size_t shape_num_axes = 1;
  bool has_dynamic_axis_dim = false;
  if (has_axis_shape) {
    for (int32_t dim : axis_shape) {
      if (dim < -1) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (dim == -1) {
        has_dynamic_axis_dim = true;
        continue;
      }
      const size_t d = static_cast<size_t>(dim);
      if (d > 0 && shape_num_axes > std::numeric_limits<size_t>::max() / d) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      shape_num_axes *= d;
    }
  }

  if (axis_type == kLiteRtElementTypeInt64) {
    if (axis_buf.Size() % sizeof(int64_t) != 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (has_dynamic_axis_dim) {
      return kLiteRtStatusErrorUnsupported;
    }
    const size_t buf_axes = axis_buf.Size() / sizeof(int64_t);
    const size_t num_axes = has_axis_shape ? shape_num_axes : buf_axes;
    if (buf_axes < num_axes || (is_arg_min_max && num_axes != 1)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const int64_t* axis_data =
        reinterpret_cast<const int64_t*>(axis_buf.Data());
    for (size_t i = 0; i < num_axes; ++i) {
      if (!add_axis(axis_data[i])) return kLiteRtStatusErrorInvalidArgument;
    }
  } else if (axis_type == kLiteRtElementTypeInt32 ||
             axis_type == kLiteRtElementTypeNone) {
    if (axis_buf.Size() % sizeof(int32_t) != 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (has_dynamic_axis_dim) {
      return kLiteRtStatusErrorUnsupported;
    }
    const size_t buf_axes = axis_buf.Size() / sizeof(int32_t);
    const size_t num_axes = has_axis_shape ? shape_num_axes : buf_axes;
    if (buf_axes < num_axes || (is_arg_min_max && num_axes != 1)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const int32_t* axis_data =
        reinterpret_cast<const int32_t*>(axis_buf.Data());
    for (size_t i = 0; i < num_axes; ++i) {
      if (!add_axis(axis_data[i])) return kLiteRtStatusErrorInvalidArgument;
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
                               /*keep_dims=*/false, /*is_arg_min_max=*/true);
}

template <typename T>
inline bool ReferenceReduction(const T* input_data, const int* input_dims,
                               int input_num_dims, T* output_data,
                               const int* output_dims, int output_num_dims,
                               const int* axis, int num_axis, bool keep_dims,
                               T init_value,
                               T reducer(const T current, const T in)) {
  constexpr int kMaxRank = tflite::RuntimeShape::kMaxSmallSize;
  assert(input_num_dims <= kMaxRank);

  int temp_index[kMaxRank];
  int resolved_axis[kMaxRank];

  return tflite::reference_ops::ReduceGeneric<T>(
      input_data, input_dims, input_num_dims, output_data, output_dims,
      output_num_dims, axis, num_axis, keep_dims, temp_index,
      resolved_axis, init_value, reducer);
}

inline bool ReferenceMean(const float* input_data, const int* input_dims,
                          int input_num_dims, float* output_data,
                          const int* output_dims, int output_num_dims,
                          const int* axis, int num_axis, bool keep_dims) {
  constexpr int kMaxRank = tflite::RuntimeShape::kMaxSmallSize;
  assert(input_num_dims <= kMaxRank);

  int temp_index[kMaxRank];
  int resolved_axis[kMaxRank];
  size_t num_outputs = 1;
  for (int i = 0; i < output_num_dims; ++i) {
    num_outputs *= output_dims[i];
  }
  std::vector<float> temp_sum(num_outputs);

  return tflite::reference_ops::Mean<float, float>(
      input_data, input_dims, input_num_dims, output_data, output_dims,
      output_num_dims, axis, num_axis, keep_dims, temp_index, resolved_axis,
      temp_sum.data());
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_REDUCTIONS_H_
