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
#include <cstring>
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

  if (size_tensor.Weights().Buffer().Size() == kResizeSizeTensorSize) {
    auto buf = size_tensor.Weights().Buffer();
    const int32_t* size_data = reinterpret_cast<const int32_t*>(buf.Data());

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

inline LiteRtStatus InferSpaceToBatchNd(const ShapeInferenceContext& ctx,
                                        InferenceResult& result) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kBlockShapeArgIndex = 1;
  constexpr size_t kPaddingsArgIndex = 2;

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(kInputArgIndex);
  Dims block_shape_shape = ctx.GetInputShape(kBlockShapeArgIndex);
  Dims paddings_shape = ctx.GetInputShape(kPaddingsArgIndex);

  if (input_shape.size() < 3) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const size_t spatial_dims_num = input_shape.size() - 2;

  if (!block_shape_shape.empty()) {
    if (block_shape_shape.size() != 1 ||
        (block_shape_shape[0] != -1 &&
         block_shape_shape[0] != static_cast<int32_t>(spatial_dims_num))) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }
  if (!paddings_shape.empty()) {
    if (paddings_shape.size() != 2 ||
        (paddings_shape[0] != -1 &&
         paddings_shape[0] != static_cast<int32_t>(spatial_dims_num)) ||
        (paddings_shape[1] != -1 && paddings_shape[1] != 2)) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  auto block_shape_buf = ctx.GetInputData(kBlockShapeArgIndex);
  auto paddings_buf = ctx.GetInputData(kPaddingsArgIndex);

  if (!block_shape_buf.empty() &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int32_t) &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!paddings_buf.empty() &&
      paddings_buf.size() != spatial_dims_num * 2 * sizeof(int32_t) &&
      paddings_buf.size() != spatial_dims_num * 2 * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (block_shape_buf.empty() || paddings_buf.empty()) {
    Dims out_shape = input_shape;
    out_shape[0] = -1;
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      out_shape[i + 1] = -1;
    }
    result.output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  std::vector<int64_t> block_shape_vals(spatial_dims_num);
  if (block_shape_buf.size() == spatial_dims_num * sizeof(int32_t)) {
    const uint8_t* data = block_shape_buf.data();
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      block_shape_vals[i] = val;
    }
  } else if (block_shape_buf.size() == spatial_dims_num * sizeof(int64_t)) {
    std::memcpy(block_shape_vals.data(), block_shape_buf.data(),
                spatial_dims_num * sizeof(int64_t));
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::vector<int64_t> paddings_vals(spatial_dims_num * 2);
  if (paddings_buf.size() == spatial_dims_num * 2 * sizeof(int32_t)) {
    const uint8_t* data = paddings_buf.data();
    for (size_t i = 0; i < spatial_dims_num * 2; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      paddings_vals[i] = val;
    }
  } else if (paddings_buf.size() == spatial_dims_num * 2 * sizeof(int64_t)) {
    std::memcpy(paddings_vals.data(), paddings_buf.data(),
                spatial_dims_num * 2 * sizeof(int64_t));
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape = input_shape;
  int64_t output_batch_size = input_shape[0];
  for (size_t dim = 0; dim < spatial_dims_num; ++dim) {
    int64_t bs = block_shape_vals[dim];
    if (bs <= 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (output_batch_size != -1) {
      output_batch_size *= bs;
    }
    int64_t pad_before = paddings_vals[dim * 2];
    int64_t pad_after = paddings_vals[dim * 2 + 1];
    if (pad_before < 0 || pad_after < 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (input_shape[dim + 1] != -1) {
      int64_t final_dim_size = input_shape[dim + 1] + pad_before + pad_after;
      if (final_dim_size % bs != 0) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      out_shape[dim + 1] = static_cast<int32_t>(final_dim_size / bs);
    } else {
      out_shape[dim + 1] = -1;
    }
  }
  if (output_batch_size != -1 && output_batch_size > INT32_MAX) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  out_shape[0] =
      (output_batch_size == -1) ? -1 : static_cast<int32_t>(output_batch_size);
  result.output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferBatchToSpaceNd(const ShapeInferenceContext& ctx,
                                        InferenceResult& result) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kBlockShapeArgIndex = 1;
  constexpr size_t kCropsArgIndex = 2;

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(kInputArgIndex);
  Dims block_shape_shape = ctx.GetInputShape(kBlockShapeArgIndex);
  Dims crops_shape = ctx.GetInputShape(kCropsArgIndex);

  if (input_shape.size() < 3) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const size_t spatial_dims_num = input_shape.size() - 2;

  if (!block_shape_shape.empty()) {
    if (block_shape_shape.size() != 1 ||
        (block_shape_shape[0] != -1 &&
         block_shape_shape[0] != static_cast<int32_t>(spatial_dims_num))) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }
  if (!crops_shape.empty()) {
    if (crops_shape.size() != 2 ||
        (crops_shape[0] != -1 &&
         crops_shape[0] != static_cast<int32_t>(spatial_dims_num)) ||
        (crops_shape[1] != -1 && crops_shape[1] != 2)) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  auto block_shape_buf = ctx.GetInputData(kBlockShapeArgIndex);
  auto crops_buf = ctx.GetInputData(kCropsArgIndex);

  if (!block_shape_buf.empty() &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int32_t) &&
      block_shape_buf.size() != spatial_dims_num * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!crops_buf.empty() &&
      crops_buf.size() != spatial_dims_num * 2 * sizeof(int32_t) &&
      crops_buf.size() != spatial_dims_num * 2 * sizeof(int64_t)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (block_shape_buf.empty() || crops_buf.empty()) {
    Dims out_shape = input_shape;
    out_shape[0] = -1;
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      out_shape[i + 1] = -1;
    }
    result.output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  std::vector<int64_t> block_shape_vals(spatial_dims_num);
  if (block_shape_buf.size() == spatial_dims_num * sizeof(int32_t)) {
    const uint8_t* data = block_shape_buf.data();
    for (size_t i = 0; i < spatial_dims_num; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      block_shape_vals[i] = val;
    }
  } else if (block_shape_buf.size() == spatial_dims_num * sizeof(int64_t)) {
    std::memcpy(block_shape_vals.data(), block_shape_buf.data(),
                spatial_dims_num * sizeof(int64_t));
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::vector<int64_t> crops_vals(spatial_dims_num * 2);
  if (crops_buf.size() == spatial_dims_num * 2 * sizeof(int32_t)) {
    const uint8_t* data = crops_buf.data();
    for (size_t i = 0; i < spatial_dims_num * 2; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      crops_vals[i] = val;
    }
  } else if (crops_buf.size() == spatial_dims_num * 2 * sizeof(int64_t)) {
    std::memcpy(crops_vals.data(), crops_buf.data(),
                spatial_dims_num * 2 * sizeof(int64_t));
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape = input_shape;
  int64_t output_batch_size = input_shape[0];
  for (size_t dim = 0; dim < spatial_dims_num; ++dim) {
    int64_t bs = block_shape_vals[dim];
    if (bs <= 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (output_batch_size != -1) {
      if (output_batch_size % bs != 0) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      output_batch_size /= bs;
    }
    int64_t crop_before = crops_vals[dim * 2];
    int64_t crop_after = crops_vals[dim * 2 + 1];
    if (crop_before < 0 || crop_after < 0) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    if (input_shape[dim + 1] != -1) {
      int64_t final_dim_size =
          input_shape[dim + 1] * bs - crop_before - crop_after;
      if (final_dim_size < 0) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      out_shape[dim + 1] = static_cast<int32_t>(final_dim_size);
    } else {
      out_shape[dim + 1] = -1;
    }
  }
  if (output_batch_size != -1 && output_batch_size > INT32_MAX) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  out_shape[0] =
      (output_batch_size == -1) ? -1 : static_cast<int32_t>(output_batch_size);
  result.output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SPATIAL_H_
