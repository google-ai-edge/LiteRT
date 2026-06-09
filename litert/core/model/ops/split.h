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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SPLIT_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SPLIT_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferSplit(const LiteRtOpT& op,
                               absl::Span<const Dims> input_shapes,
                               std::vector<Dims>& output_shapes) {
  constexpr int kSplitMinArgs = 2;
  constexpr int kAxisArgIndex = 0;
  constexpr int kInputArgIndex = 1;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  // Inputs: Axis, Input.
  if (input_shapes.size() < kSplitMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& axis_tensor = op.Input(kAxisArgIndex);
  const auto& input_shape = input_shapes[kInputArgIndex];

  const auto& opts = GetTflOptions(op);
  const auto* split_opts = opts.AsSplitOptions();
  int32_t num_splits = split_opts ? split_opts->num_splits : op.NumOutputs();

  if (axis_tensor.Weights().Buffer().Size() >= kShapeTensorSize) {
    auto buf = axis_tensor.Weights().Buffer();
    int32_t axis = *reinterpret_cast<const int32_t*>(buf.Data());
    if (axis < 0) axis += input_shape.size();

    Dims out_shape = input_shape;
    if (out_shape[axis] != -1) {
      out_shape[axis] /= num_splits;
    }

    output_shapes.resize(num_splits);
    for (size_t i = 0; i < output_shapes.size(); ++i)
      output_shapes[i] = out_shape;
    return kLiteRtStatusOk;
  }

  // Assume generic split if axis unknown.
  output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

template <typename T>
inline void ReferenceSplit(const T* input_data, const int32_t* input_dims,
                           int rank, int axis, int num_splits,
                           absl::Span<T* const> output_buffers) {
  if (rank <= 0 || num_splits <= 0 || output_buffers.size() != num_splits)
    return;
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return;

  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_dims[i];
  }

  int64_t split_dim = input_dims[axis] / num_splits;
  int64_t inner_size = 1;
  for (int i = axis + 1; i < rank; ++i) {
    inner_size *= input_dims[i];
  }

  int64_t copy_size = split_dim * inner_size;
  int64_t in_slice_stride = input_dims[axis] * inner_size;

  for (int64_t o = 0; o < outer_size; ++o) {
    for (int s = 0; s < num_splits; ++s) {
      const T* in_ptr = input_data + o * in_slice_stride + s * copy_size;
      T* out_ptr = output_buffers[s] + o * copy_size;
      std::memcpy(out_ptr, in_ptr, copy_size * sizeof(T));
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SPLIT_H_
