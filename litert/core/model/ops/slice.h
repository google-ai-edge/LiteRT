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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SLICE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SLICE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferSlice(const LiteRtOpT& op,
                               absl::Span<const Dims> input_shapes,
                               std::vector<Dims>& output_shapes) {
  constexpr int kSliceMinArgs = 3;
  constexpr int kInputArgIndex = 0;
  constexpr int kSizeArgIndex = 2;

  // Inputs: Input, Begin, Size.
  if (input_shapes.size() < kSliceMinArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& size_tensor = op.Input(kSizeArgIndex);

  // If size tensor is constant, use it.
  if (size_tensor.Weights().Buffer().Size() > 0) {
    auto buf = size_tensor.Weights().Buffer();
    // Expect int32 or int64.
    Dims out_shape;
    if (buf.Size() % sizeof(int32_t) == 0) {
      const int32_t* data = reinterpret_cast<const int32_t*>(buf.Data());
      int rank = buf.Size() / sizeof(int32_t);
      for (int i = 0; i < rank; ++i) {
        int32_t s = data[i];
        if (s == -1) {
          out_shape.push_back(-1);
        } else {
          out_shape.push_back(s);
        }
      }
    } else {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  // If size is dynamic, output is dynamic rank (if input rank is known) or just
  // matching rank.
  output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferStridedSlice(const LiteRtOpT& op,
                                      absl::Span<const Dims> input_shapes,
                                      std::vector<Dims>& output_shapes) {
  constexpr int kStridedSliceMinArgs = 4;
  constexpr int kInputArgIndex = 0;
  constexpr int kBeginArgIndex = 1;

  if (input_shapes.empty()) return kLiteRtStatusErrorShapeInferenceFailed;

  const auto& opts = GetTflOptions(op);
  const auto* ss_opts = opts.AsStridedSliceOptions();
  if (!ss_opts) return kLiteRtStatusErrorShapeInferenceFailed;

  int32_t shrink_axis_mask = ss_opts->shrink_axis_mask;
  int32_t new_axis_mask = ss_opts->new_axis_mask;

  const auto& input_shape = input_shapes[kInputArgIndex];
  Dims out_shape;

  if (input_shapes.size() < kStridedSliceMinArgs)
    return kLiteRtStatusErrorShapeInferenceFailed;
  const auto& begin_shape = input_shapes[kBeginArgIndex];  // Rank 1
  int processing_dims = begin_shape[0];

  int i_in = 0;
  for (int i = 0; i < processing_dims; ++i) {
    if (new_axis_mask & (1 << i)) {
      out_shape.push_back(1);
    } else if (shrink_axis_mask & (1 << i)) {
      i_in++;
    } else {
      out_shape.push_back(-1);
      i_in++;
    }
  }

  for (; i_in < input_shape.size(); ++i_in) {
    out_shape.push_back(input_shape[i_in]);
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferDynamicUpdateSlice(const LiteRtOpT& op,
                                            absl::Span<const Dims> input_shapes,
                                            std::vector<Dims>& output_shapes) {
  constexpr int kDynamicUpdateSliceNumArgs = 3;
  constexpr int kOperandArgIndex = 0;
  constexpr int kUpdateArgIndex = 1;
  constexpr int kStartIndicesArgIndex = 2;
  constexpr int kStartIndicesRank = 1;

  // Inputs: Operand, Update, StartIndices.
  if (input_shapes.size() != kDynamicUpdateSliceNumArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  const auto& operand_shape = input_shapes[kOperandArgIndex];
  const auto& update_shape = input_shapes[kUpdateArgIndex];
  const auto& start_indices_shape = input_shapes[kStartIndicesArgIndex];

  // Rank check: start_indices must be 1D.
  if (start_indices_shape.size() != kStartIndicesRank) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Dimension check: size of start_indices must match rank of operand.
  if (start_indices_shape[0] != operand_shape.size()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Rank check: update rank must match operand rank.
  if (update_shape.size() != operand_shape.size()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Boundary check: update dims must be <= operand dims.
  for (size_t i = 0; i < operand_shape.size(); ++i) {
    if (update_shape[i] > operand_shape[i]) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  // Output shape is same as input (operand) shape.
  output_shapes[0] = operand_shape;
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SLICE_H_
