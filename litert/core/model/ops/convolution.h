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
#include <limits>
#include <utility>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
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
  int32_t remainder = 0;
  int32_t groups = 0;
  if (filter_shape[3] <= 0 ||
      !SafeModulus(input_shape[3], filter_shape[3], remainder) ||
      remainder != 0 || !SafeDivide(input_shape[3], filter_shape[3], groups)) {
    LITERT_LOG(LITERT_ERROR,
               "Conv2D input channels (%d) must be a multiple of filter input "
               "channels (%d)",
               input_shape[3], filter_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (groups <= 0 || !SafeModulus(filter_shape[0], groups, remainder) ||
      remainder != 0) {
    LITERT_LOG(LITERT_ERROR,
               "Conv2D output channels (%d) must be a multiple of groups (%d)",
               filter_shape[0], groups);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid stride: %dx%d", stride_h,
               stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  constexpr int32_t kMaxStrideOrDilation = std::numeric_limits<int16_t>::max();

  if (stride_h > kMaxStrideOrDilation || stride_w > kMaxStrideOrDilation ||
      dilation_h > kMaxStrideOrDilation || dilation_w > kMaxStrideOrDilation) {
    LITERT_LOG(LITERT_ERROR, "Conv2D stride or dilation exceeds limit");
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

  if (input_shapes.size() >= 3 && !input_shapes[2].empty()) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "Conv2D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_h = 0;
  int32_t effective_filter_w = 0;
  if (!ComputeEffectiveFilterSize(filter_h, dilation_h, effective_filter_h) ||
      !ComputeEffectiveFilterSize(filter_w, dilation_w, effective_filter_w)) {
    LITERT_LOG(LITERT_ERROR, "Conv2D effective filter size overflow");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
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

  Dims out_shape = {input_shape[0], out_h, out_w, out_channels};
  if (!CheckShapePositiveProductFitsInInt32(input_shape) ||
      !CheckShapePositiveProductFitsInInt32(filter_shape) ||
      !CheckShapePositiveProductFitsInInt32(out_shape)) {
    LITERT_LOG(LITERT_ERROR, "Conv2D tensor element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const bool need_im2col = stride_h != 1 || stride_w != 1 || dilation_h != 1 ||
                           dilation_w != 1 || filter_h != 1 || filter_w != 1;
  if (need_im2col && !CheckShapePositiveProductFitsInInt32(
                         {input_shape[0], out_h, out_w, filter_shape[3],
                          filter_h, filter_w})) {
    LITERT_LOG(LITERT_ERROR, "Conv2D im2col element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = std::move(out_shape);
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

  int32_t expected_out_channels = 0;
  if (!SafeMultiply(input_shape[3], dw_opts->depth_multiplier,
                    expected_out_channels) ||
      out_channels != expected_out_channels) {
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

  constexpr int32_t kMaxStrideOrDilation = std::numeric_limits<int16_t>::max();

  if (stride_h > kMaxStrideOrDilation || stride_w > kMaxStrideOrDilation ||
      dilation_h > kMaxStrideOrDilation || dilation_w > kMaxStrideOrDilation) {
    LITERT_LOG(LITERT_ERROR,
               "DepthwiseConv2D stride or dilation exceeds limit");
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

  if (input_shapes.size() >= 3 && !input_shapes[2].empty()) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_h = 0;
  int32_t effective_filter_w = 0;
  if (!ComputeEffectiveFilterSize(filter_h, dilation_h, effective_filter_h) ||
      !ComputeEffectiveFilterSize(filter_w, dilation_w, effective_filter_w)) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D effective filter size overflow");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

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

  Dims out_shape = {input_shape[0], out_h, out_w, out_channels};
  if (!CheckShapePositiveProductFitsInInt32(input_shape) ||
      !CheckShapePositiveProductFitsInInt32(filter_shape) ||
      !CheckShapePositiveProductFitsInInt32(out_shape)) {
    LITERT_LOG(LITERT_ERROR,
               "DepthwiseConv2D tensor element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferTransposeConv(const LiteRtOpT& op,
                                       absl::Span<Dims> input_shapes,
                                       std::vector<Dims>& output_shapes) {
  constexpr size_t kOutputShapeArgIndex = 0;
  constexpr size_t kWeightsArgIndex = 1;
  constexpr size_t kInputArgIndex = 2;
  constexpr size_t kBiasArgIndex = 3;
  constexpr size_t kTransposeConvMinArgs = 3;
  constexpr size_t kTransposeConvRank = 4;
  constexpr size_t kShapeTensorSize = kTransposeConvRank * sizeof(int32_t);
  constexpr int32_t kMaxStride = std::numeric_limits<int16_t>::max();

  if (input_shapes.size() < kTransposeConvMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& output_shape_arg = input_shapes[kOutputShapeArgIndex];
  const auto& weights_shape = input_shapes[kWeightsArgIndex];
  const auto& input_shape = input_shapes[kInputArgIndex];

  if (output_shape_arg.size() != 1 ||
      (output_shape_arg[0] != static_cast<int32_t>(kTransposeConvRank) &&
       output_shape_arg[0] != -1) ||
      weights_shape.size() != kTransposeConvRank ||
      input_shape.size() != kTransposeConvRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // TFLite TransposeConv weights: [OutputDepth, H, W, InputDepth].
  if (weights_shape[0] <= 0 || weights_shape[1] <= 0 || weights_shape[2] <= 0 ||
      weights_shape[3] <= 0 ||
      !CheckShapePositiveProductFitsInInt32(weights_shape)) {
    LITERT_LOG(LITERT_ERROR, "TransposeConv invalid weights shape");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shape[0] < -1 || (input_shape[1] <= 0 && input_shape[1] != -1) ||
      (input_shape[2] <= 0 && input_shape[2] != -1) ||
      (input_shape[3] <= 0 && input_shape[3] != -1) ||
      !CheckShapePositiveProductFitsInInt32(input_shape)) {
    LITERT_LOG(LITERT_ERROR, "TransposeConv invalid input shape");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shape[3] != -1 && input_shape[3] != weights_shape[3]) {
    LITERT_LOG(LITERT_ERROR, "TransposeConv input channels mismatch: %d vs %d",
               input_shape[3], weights_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Validate col2im rows (in_h * in_w), columns (out_c * filter_h * filter_w),
  // and total col2im element count fit in int32_t.
  if (!CheckShapePositiveProductFitsInInt32({input_shape[1], input_shape[2],
                                             weights_shape[0], weights_shape[1],
                                             weights_shape[2]})) {
    LITERT_LOG(LITERT_ERROR,
               "TransposeConv col2im element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shapes.size() > kBiasArgIndex &&
      !input_shapes[kBiasArgIndex].empty()) {
    const auto& bias_shape = input_shapes[kBiasArgIndex];
    if (bias_shape.size() != 1 ||
        (bias_shape[0] != weights_shape[0] && bias_shape[0] != -1)) {
      LITERT_LOG(LITERT_ERROR, "TransposeConv bias size mismatch: %d vs %d",
                 bias_shape[0], weights_shape[0]);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  const auto* tc_opts = GetTflOptions(op).AsTransposeConvOptions();
  if (tc_opts != nullptr) {
    if (tc_opts->stride_h <= 0 || tc_opts->stride_h > kMaxStride ||
        tc_opts->stride_w <= 0 || tc_opts->stride_w > kMaxStride) {
      LITERT_LOG(LITERT_ERROR, "TransposeConv invalid stride: %dx%d",
                 tc_opts->stride_h, tc_opts->stride_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  // If output_shape tensor is constant, validate and use it.
  if (op.Inputs().size() > kOutputShapeArgIndex &&
      op.Inputs()[kOutputShapeArgIndex] != nullptr) {
    const auto& output_size_tensor = op.Input(kOutputShapeArgIndex);
    auto buf = output_size_tensor.Weights().Buffer();
    if (buf.Size() > 0) {
      if (buf.Size() != kShapeTensorSize) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      const int32_t* dims = reinterpret_cast<const int32_t*>(buf.Data());
      Dims out_shape(dims, dims + kTransposeConvRank);
      if (out_shape[0] < 0 || out_shape[1] <= 0 || out_shape[2] <= 0 ||
          out_shape[3] != weights_shape[0] ||
          (input_shape[0] != -1 && out_shape[0] != input_shape[0]) ||
          !CheckShapePositiveProductFitsInInt32(out_shape)) {
        LITERT_LOG(LITERT_ERROR, "TransposeConv invalid constant output shape");
        return kLiteRtStatusErrorShapeInferenceFailed;
      }

      if (tc_opts != nullptr) {
        if (tc_opts->padding == tflite::Padding_VALID &&
            (out_shape[1] < weights_shape[1] ||
             out_shape[2] < weights_shape[2])) {
          LITERT_LOG(LITERT_ERROR,
                     "TransposeConv output spatial size smaller than filter "
                     "with VALID padding");
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        const int32_t expected_in_h =
            ComputeOutputSize(tc_opts->padding, out_shape[1], weights_shape[1],
                              tc_opts->stride_h);
        const int32_t expected_in_w =
            ComputeOutputSize(tc_opts->padding, out_shape[2], weights_shape[2],
                              tc_opts->stride_w);
        if (expected_in_h <= 0 || expected_in_w <= 0 ||
            (input_shape[1] > 0 && input_shape[1] != expected_in_h) ||
            (input_shape[2] > 0 && input_shape[2] != expected_in_w)) {
          LITERT_LOG(LITERT_ERROR,
                     "TransposeConv input spatial dimensions (%dx%d) do not "
                     "match expected (%dx%d)",
                     input_shape[1], input_shape[2], expected_in_h,
                     expected_in_w);
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
      }

      output_shapes[0] = std::move(out_shape);
      return kLiteRtStatusOk;
    }
  }

  Dims out_shape = {input_shape[0], -1, -1, weights_shape[0]};
  if (!CheckShapePositiveProductFitsInInt32(out_shape)) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  output_shapes[0] = std::move(out_shape);
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

  int32_t effective_filter_d = 0;
  int32_t effective_filter_h = 0;
  int32_t effective_filter_w = 0;
  if (!ComputeEffectiveFilterSize(filter_d, dilation_d, effective_filter_d) ||
      !ComputeEffectiveFilterSize(filter_h, dilation_h, effective_filter_h) ||
      !ComputeEffectiveFilterSize(filter_w, dilation_w, effective_filter_w)) {
    LITERT_LOG(LITERT_ERROR, "Conv3D effective filter size overflow");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

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

  Dims out_shape = {input_shape[0], out_d, out_h, out_w, out_channels};
  if (!CheckShapePositiveProductFitsInInt32(input_shape) ||
      !CheckShapePositiveProductFitsInInt32(filter_shape) ||
      !CheckShapePositiveProductFitsInInt32(out_shape)) {
    LITERT_LOG(LITERT_ERROR, "Conv3D tensor element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const bool need_im2col = stride_d != 1 || stride_h != 1 || stride_w != 1 ||
                           dilation_d != 1 || dilation_h != 1 ||
                           dilation_w != 1 || filter_d != 1 || filter_h != 1 ||
                           filter_w != 1;
  if (need_im2col && !CheckShapePositiveProductFitsInInt32(
                         {input_shape[0], out_d, out_h, out_w, filter_shape[3],
                          filter_d, filter_h, filter_w})) {
    LITERT_LOG(LITERT_ERROR, "Conv3D im2col element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = std::move(out_shape);
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

inline void ReferenceConv2D(const float* input_data, const float* filter_data,
                            const float* bias_data, float* output_data,
                            int batch, int in_h, int in_w, int in_c, int out_h,
                            int out_w, int out_c, int filter_h, int filter_w,
                            int stride_h, int stride_w, int dilation_h,
                            int dilation_w, int pad_t, int pad_l,
                            tflite::ActivationFunctionType faf) {
  for (int b = 0; b < batch; ++b) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int oc = 0; oc < out_c; ++oc) {
          float sum = 0;
          for (int kh = 0; kh < filter_h; ++kh) {
            for (int kw = 0; kw < filter_w; ++kw) {
              for (int ic = 0; ic < in_c; ++ic) {
                int ih = oh * stride_h + kh * dilation_h - pad_t;
                int iw = ow * stride_w + kw * dilation_w - pad_l;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                  int input_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                  int filter_idx =
                      ((oc * filter_h + kh) * filter_w + kw) * in_c + ic;
                  sum += input_data[input_idx] * filter_data[filter_idx];
                }
              }
            }
          }
          float output_val = sum + (bias_data ? bias_data[oc] : 0.0f);
          if (faf == tflite::ActivationFunctionType_RELU) {
            if (output_val < 0.0f) output_val = 0.0f;
          } else if (faf == tflite::ActivationFunctionType_RELU6) {
            if (output_val < 0.0f) output_val = 0.0f;
            if (output_val > 6.0f) output_val = 6.0f;
          }
          output_data[((b * out_h + oh) * out_w + ow) * out_c + oc] =
              output_val;
        }
      }
    }
  }
}

inline void ReferenceDepthwiseConv2D(
    const float* input_data, const float* filter_data, const float* bias_data,
    float* output_data, int batch, int in_h, int in_w, int in_c, int out_h,
    int out_w, int out_c, int filter_h, int filter_w, int stride_h,
    int stride_w, int dilation_h, int dilation_w, int pad_t, int pad_l,
    int depth_multiplier, tflite::ActivationFunctionType faf) {
  for (int b = 0; b < batch; ++b) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int oc = 0; oc < out_c; ++oc) {
          float sum = 0;
          for (int kh = 0; kh < filter_h; ++kh) {
            for (int kw = 0; kw < filter_w; ++kw) {
              int ih = oh * stride_h + kh * dilation_h - pad_t;
              int iw = ow * stride_w + kw * dilation_w - pad_l;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * in_h + ih) * in_w + iw) * in_c +
                                oc / depth_multiplier;
                int filter_index = (kh * filter_w + kw) * out_c + oc;
                sum += input_data[input_idx] * filter_data[filter_index];
              }
            }
          }
          float output_val = sum + (bias_data ? bias_data[oc] : 0.0f);
          if (faf == tflite::ActivationFunctionType_RELU) {
            if (output_val < 0.0f) output_val = 0.0f;
          } else if (faf == tflite::ActivationFunctionType_RELU6) {
            if (output_val < 0.0f) output_val = 0.0f;
            if (output_val > 6.0f) output_val = 6.0f;
          }
          output_data[((b * out_h + oh) * out_w + ow) * out_c + oc] =
              output_val;
        }
      }
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_
