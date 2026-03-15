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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferDepthToSpace(const LiteRtOpT& op,
                                      absl::Span<const Dims> input_shapes,
                                      std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kHeightDimIndex = 1;
  constexpr int kWidthDimIndex = 2;
  constexpr int kDepthDimIndex = 3;
  constexpr int kSpatialRank = 4;

  if (input_shapes.empty()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  if (input_shape.size() != kSpatialRank) {
    return kLiteRtStatusErrorShapeInferenceFailed;  // NHWC
  }

  const auto& opts = GetTflOptions(op);
  const auto* d2s_opts = opts.AsDepthToSpaceOptions();
  int32_t block_size = d2s_opts ? d2s_opts->block_size : 0;
  if (block_size <= 0) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = input_shape;
  int32_t height = input_shape[kHeightDimIndex];
  int32_t width = input_shape[kWidthDimIndex];
  int32_t depth = input_shape[kDepthDimIndex];

  if (depth != -1) {
    if (depth % (block_size * block_size) != 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    out_shape[kDepthDimIndex] = depth / (block_size * block_size);
  }
  if (height != -1) out_shape[kHeightDimIndex] = height * block_size;
  if (width != -1) out_shape[kWidthDimIndex] = width * block_size;

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferSpaceToDepth(const LiteRtOpT& op,
                                      absl::Span<const Dims> input_shapes,
                                      std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kHeightDimIndex = 1;
  constexpr int kWidthDimIndex = 2;
  constexpr int kDepthDimIndex = 3;
  constexpr int kSpatialRank = 4;

  if (input_shapes.empty()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  if (input_shape.size() != kSpatialRank) {
    return kLiteRtStatusErrorShapeInferenceFailed;  // NHWC
  }

  const auto& opts = GetTflOptions(op);
  const auto* s2d_opts = opts.AsSpaceToDepthOptions();
  int32_t block_size = s2d_opts ? s2d_opts->block_size : 0;
  if (block_size <= 0) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = input_shape;
  int32_t height = input_shape[kHeightDimIndex];
  int32_t width = input_shape[kWidthDimIndex];
  int32_t depth = input_shape[kDepthDimIndex];

  if (depth != -1) out_shape[kDepthDimIndex] = depth * block_size * block_size;
  if (height != -1) {
    if (height % block_size != 0) return kLiteRtStatusErrorShapeInferenceFailed;
    out_shape[kHeightDimIndex] = height / block_size;
  }
  if (width != -1) {
    if (width % block_size != 0)
      return kLiteRtStatusErrorShapeInferenceFailed;
    out_shape[kWidthDimIndex] = width / block_size;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferResizeOp(const LiteRtOpT& op,
                                  absl::Span<const Dims> input_shapes,
                                  std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kSizeArgIndex = 1;
  constexpr int kResizeMinArgs = 2;
  constexpr int kSpatialRank = 4;
  constexpr int kResizeSizeTensorSize = 2 * sizeof(int32_t);

  // Inputs: Input, Size.
  if (input_shapes.size() < kResizeMinArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& size_tensor = op.Input(kSizeArgIndex);

  if (input_shape.size() != kSpatialRank)
    return kLiteRtStatusErrorShapeInferenceFailed;  // Expect NHWC

  Dims out_shape = input_shape;

  if (size_tensor.Weights().Buffer().Size() > 0) {
    auto buf = size_tensor.Weights().Buffer();
    const int32_t* size_data = reinterpret_cast<const int32_t*>(buf.Data());
    if (buf.Size() != kResizeSizeTensorSize)
      return kLiteRtStatusErrorShapeInferenceFailed;

    out_shape[1] = size_data[0];  // New Height
    out_shape[2] = size_data[1];  // New Width
  } else {
    // Dynamic size
    out_shape[1] = -1;
    out_shape[2] = -1;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferResizeBilinear(const LiteRtOpT& op,
                                        absl::Span<const Dims> input_shapes,
                                        std::vector<Dims>& output_shapes) {
  return InferResizeOp(op, input_shapes, output_shapes);
}

inline LiteRtStatus InferResizeNearestNeighbor(
    const LiteRtOpT& op, absl::Span<const Dims> input_shapes,
    std::vector<Dims>& output_shapes) {
  return InferResizeOp(op, input_shapes, output_shapes);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_
