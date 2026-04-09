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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/op_shape_inference_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

inline LiteRtStatus InferConv2D(const LiteRtOpT& op,
                                absl::Span<Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kConv2DMinArgs = 2;
  constexpr size_t kConv2DRank = 4;

  if (input_shapes.size() < kConv2DMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& filter_shape = input_shapes[kFilterArgIndex];

  if (input_shape.size() != kConv2DRank || filter_shape.size() != kConv2DRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* conv_opts = opts.AsConv2DOptions();
  if (!conv_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_h = conv_opts->stride_h;
  int32_t stride_w = conv_opts->stride_w;
  int32_t dilation_h = conv_opts->dilation_h_factor;
  int32_t dilation_w = conv_opts->dilation_w_factor;
  tflite::Padding padding = conv_opts->padding;

  // Filter: [Out, H, W, In]
  int32_t filter_h = filter_shape[1];
  int32_t filter_w = filter_shape[2];
  if (input_shape[3] % filter_shape[3] != 0) {
    LITERT_LOG(LITERT_ERROR,
               "Conv2D input channels (%d) must be a multiple of filter input "
               "channels (%d)",
               input_shape[3], filter_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid stride: %dx%d", stride_h,
               stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h > input_shape[1] || stride_w > input_shape[2]) {
    LITERT_LOG(LITERT_ERROR,
               "Conv2D stride (%dx%d) cannot be larger than input size (%dx%d)",
               stride_h, stride_w, input_shape[1], input_shape[2]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (dilation_h <= 0 || dilation_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid dilation: %dx%d", dilation_h,
               dilation_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_channels = filter_shape[0];

  if (input_shapes.size() >= 3) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "Conv2D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_h = (filter_h - 1) * dilation_h + 1;
  int32_t effective_filter_w = (filter_w - 1) * dilation_w + 1;
  int32_t out_h = ComputeOutputSize(padding, input_shape[1], filter_h, stride_h,
                                    dilation_h);
  int32_t out_w = ComputeOutputSize(padding, input_shape[2], filter_w, stride_w,
                                    dilation_w);

  if (padding == tflite::Padding_VALID) {
    if (input_shape[1] < effective_filter_h ||
        input_shape[2] < effective_filter_w) {
      LITERT_LOG(LITERT_ERROR,
                 "Conv2D input size (%dx%d) too small for effective filter "
                 "size (%dx%d) with VALID padding",
                 input_shape[1], input_shape[2], effective_filter_h,
                 effective_filter_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  if (out_h <= 0 || out_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid output size: %dx%d", out_h, out_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = {input_shape[0], out_h, out_w, out_channels};
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferDepthwiseConv2D(const LiteRtOpT& op,
                                         absl::Span<Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kDepthwiseConv2DMinArgs = 2;
  constexpr size_t kDepthwiseConv2DRank = 4;

  if (input_shapes.size() < kDepthwiseConv2DMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& filter_shape = input_shapes[kFilterArgIndex];

  if (input_shape.size() != kDepthwiseConv2DRank ||
      filter_shape.size() != kDepthwiseConv2DRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* dw_opts = opts.AsDepthwiseConv2DOptions();
  if (!dw_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_h = dw_opts->stride_h;
  int32_t stride_w = dw_opts->stride_w;
  int32_t dilation_h = dw_opts->dilation_h_factor;
  int32_t dilation_w = dw_opts->dilation_w_factor;
  tflite::Padding padding = dw_opts->padding;

  // Filter: [1, H, W, OutChannels]
  int32_t filter_h = filter_shape[1];
  int32_t filter_w = filter_shape[2];
  if (filter_shape[0] != 1) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D filter dim 0 must be 1, got %d",
               filter_shape[0]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_channels = filter_shape[3];

  if (out_channels != input_shape[3] * dw_opts->depth_multiplier) {
    LITERT_LOG(LITERT_ERROR,
               "DepthwiseConv2D out_channels mismatch: %d vs %d*%d",
               out_channels, input_shape[3], dw_opts->depth_multiplier);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D invalid stride: %dx%d", stride_h,
               stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h > input_shape[1] || stride_w > input_shape[2]) {
    LITERT_LOG(
        LITERT_ERROR,
        "DepthwiseConv2D stride (%dx%d) cannot be larger than input size "
        "(%dx%d)",
        stride_h, stride_w, input_shape[1], input_shape[2]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (dilation_h <= 0 || dilation_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D invalid dilation: %dx%d",
               dilation_h, dilation_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shapes.size() >= 3) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_h = (filter_h - 1) * dilation_h + 1;
  int32_t effective_filter_w = (filter_w - 1) * dilation_w + 1;

  int32_t out_h = ComputeOutputSize(padding, input_shape[1], filter_h, stride_h,
                                    dilation_h);
  int32_t out_w = ComputeOutputSize(padding, input_shape[2], filter_w, stride_w,
                                    dilation_w);

  if (padding == tflite::Padding_VALID) {
    if (input_shape[1] < effective_filter_h ||
        input_shape[2] < effective_filter_w) {
      LITERT_LOG(LITERT_ERROR,
                 "DepthwiseConv2D input size (%dx%d) too small for effective "
                 "filter size (%dx%d) with VALID padding",
                 input_shape[1], input_shape[2], effective_filter_h,
                 effective_filter_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  if (out_h <= 0 || out_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D invalid output size: %dx%d",
               out_h, out_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = {input_shape[0], out_h, out_w, out_channels};
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferTransposeConv(const LiteRtOpT& op,
                                       absl::Span<Dims> input_shapes,
                                       std::vector<Dims>& output_shapes) {
  constexpr size_t kOutputShapeArgIndex = 0;
  constexpr size_t kWeightsArgIndex = 1;
  constexpr size_t kInputArgIndex = 2;
  constexpr size_t kTransposeConvMinArgs = 3;
  constexpr size_t kTransposeConvRank = 4;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  if (input_shapes.size() < kTransposeConvMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // If output_shape tensor is constant, use it.
  const auto& output_size_tensor = op.Input(kOutputShapeArgIndex);
  if (output_size_tensor.Weights().Buffer().Size() >= kShapeTensorSize) {
    auto buf = output_size_tensor.Weights().Buffer();
    const int32_t* dims = reinterpret_cast<const int32_t*>(buf.Data());
    int rank = buf.Size() / sizeof(int32_t);
    Dims out_shape;
    for (int i = 0; i < rank; ++i) out_shape.push_back(dims[i]);
    output_shapes[0] = out_shape;
    return kLiteRtStatusOk;
  }

  const auto& input_shape = input_shapes[kInputArgIndex];
  output_shapes[0] = Dims(input_shape.size(), -1);  // Preserve rank at least
  // Set batch dim
  output_shapes[0][0] = input_shape[0];
  // Set channels dim (from weights)
  // TFLite TransposeConv weights: [OutputDepth, H, W, InputDepth].
  const auto& weights_shape = input_shapes[kWeightsArgIndex];
  if (weights_shape.size() == kTransposeConvRank) {
    output_shapes[0][3] = weights_shape[0];  // Output channels
  }

  return kLiteRtStatusOk;
}

inline LiteRtStatus InferConv3D(const LiteRtOpT& op,
                                absl::Span<Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kConv3DMinArgs = 2;
  constexpr size_t kConv3DRank = 5;

  if (input_shapes.size() < kConv3DMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& filter_shape = input_shapes[kFilterArgIndex];

  if (input_shape.size() != kConv3DRank || filter_shape.size() != kConv3DRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* conv_opts = opts.AsConv3DOptions();
  if (!conv_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_d = conv_opts->stride_d;
  int32_t stride_h = conv_opts->stride_h;
  int32_t stride_w = conv_opts->stride_w;
  int32_t dilation_d = conv_opts->dilation_d_factor;
  int32_t dilation_h = conv_opts->dilation_h_factor;
  int32_t dilation_w = conv_opts->dilation_w_factor;
  tflite::Padding padding = conv_opts->padding;

  int32_t filter_d = filter_shape[0];
  int32_t filter_h = filter_shape[1];
  int32_t filter_w = filter_shape[2];
  if (input_shape[4] != filter_shape[3]) {
    LITERT_LOG(LITERT_ERROR, "Conv3D input channels mismatch: %d vs %d",
               input_shape[4], filter_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_d <= 0 || stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv3D invalid stride: %dx%dx%d", stride_d,
               stride_h, stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_d > input_shape[1] || stride_h > input_shape[2] ||
      stride_w > input_shape[3]) {
    LITERT_LOG(
        LITERT_ERROR,
        "Conv3D stride (%dx%dx%d) cannot be larger than input size (%dx%dx%d)",
        stride_d, stride_h, stride_w, input_shape[1], input_shape[2],
        input_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv3D invalid dilation: %dx%dx%d", dilation_d,
               dilation_h, dilation_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_channels = filter_shape[4];

  if (input_shapes.size() >= 3) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "Conv3D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_d = (filter_d - 1) * dilation_d + 1;
  int32_t effective_filter_h = (filter_h - 1) * dilation_h + 1;
  int32_t effective_filter_w = (filter_w - 1) * dilation_w + 1;

  int32_t out_d = ComputeOutputSize(padding, input_shape[1], filter_d, stride_d,
                                    dilation_d);
  int32_t out_h = ComputeOutputSize(padding, input_shape[2], filter_h, stride_h,
                                    dilation_h);
  int32_t out_w = ComputeOutputSize(padding, input_shape[3], filter_w, stride_w,
                                    dilation_w);

  if (padding == tflite::Padding_VALID) {
    if (input_shape[1] < effective_filter_d ||
        input_shape[2] < effective_filter_h ||
        input_shape[3] < effective_filter_w) {
      LITERT_LOG(LITERT_ERROR,
                 "Conv3D input size (%dx%dx%d) too small for effective filter "
                 "size (%dx%dx%d) with VALID padding",
                 input_shape[1], input_shape[2], input_shape[3],
                 effective_filter_d, effective_filter_h, effective_filter_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv3D invalid output size: %dx%dx%d", out_d,
               out_h, out_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = {input_shape[0], out_d, out_h, out_w, out_channels};
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferConv3DTranspose(const LiteRtOpT& op,
                                         absl::Span<Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  constexpr size_t kOutputShapeArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kInputArgIndex = 2;
  constexpr size_t kConv3DTransposeMinArgs = 3;
  constexpr size_t kConv3DTransposeRank = 5;

  if (input_shapes.size() < kConv3DTransposeMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& output_size_tensor = op.Input(kOutputShapeArgIndex);
  if (output_size_tensor.Weights().Buffer().Size() >= sizeof(int32_t)) {
    auto buf = output_size_tensor.Weights().Buffer();
    const int32_t* dims = reinterpret_cast<const int32_t*>(buf.Data());
    int rank = buf.Size() / sizeof(int32_t);
    Dims out_shape;
    for (int i = 0; i < rank; ++i) out_shape.push_back(dims[i]);
    output_shapes[0] = out_shape;
    return kLiteRtStatusOk;
  }

  const auto& input_shape = input_shapes[kInputArgIndex];
  output_shapes[0] = Dims(input_shape.size(), -1);
  if (!input_shape.empty()) {
    output_shapes[0][0] = input_shape[0];  // Batch
  }

  // Filter: [D, H, W, In, Out] -> Out is index 4.
  const auto& filter_shape = input_shapes[kFilterArgIndex];
  if (filter_shape.size() == kConv3DTransposeRank) {
    output_shapes[0][4] = filter_shape[4];
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_
